import collections
from typing import Any, Iterator, Optional, Sequence, Tuple, Union

import gymnasium as gym
import jax
import numpy as np
from serl_launcher.data.dataset import Dataset, DatasetDict


def _init_replay_dict(
    obs_space: gym.Space, capacity: int
) -> Union[np.ndarray, DatasetDict]:
    if isinstance(obs_space, gym.spaces.Box):
        return np.empty((capacity, *obs_space.shape), dtype=obs_space.dtype)
    elif isinstance(obs_space, gym.spaces.Dict):
        data_dict = {}
        for k, v in obs_space.spaces.items():
            data_dict[k] = _init_replay_dict(v, capacity)
        return data_dict
    else:
        raise TypeError()


def _insert_recursively(
    dataset_dict: DatasetDict, data_dict: DatasetDict, insert_index: int
):
    if isinstance(dataset_dict, np.ndarray):
        dataset_dict[insert_index] = data_dict
    elif isinstance(dataset_dict, dict):
        for k in dataset_dict.keys():
            _insert_recursively(dataset_dict[k], data_dict[k], insert_index)
    else:
        raise TypeError()


class ReplayBuffer(Dataset):
    def __init__(
        self,
        observation_space: gym.Space,
        action_space: gym.Space,
        capacity: int,
        next_observation_space: Optional[gym.Space] = None,
        include_next_actions: Optional[bool] = False,
        include_label: Optional[bool] = False,
        include_grasp_penalty: Optional[bool] = False,
        include_octo_embeddings: Optional[bool] = False,
        include_mc_returns: Optional[bool] = False,
        include_alpha_correction: Optional[bool] = False,
        include_segment_ids: Optional[bool] = False,
    ):
        if next_observation_space is None:
            next_observation_space = observation_space

        observation_data = _init_replay_dict(observation_space, capacity)
        next_observation_data = _init_replay_dict(next_observation_space, capacity)
        dataset_dict = dict(
            observations=observation_data,
            next_observations=next_observation_data,
            actions=np.empty((capacity, *action_space.shape), dtype=action_space.dtype),
            rewards=np.empty((capacity,), dtype=np.float32),
            masks=np.empty((capacity,), dtype=np.float32),
            dones=np.empty((capacity,), dtype=bool),
        )
        
        if include_mc_returns:
            dataset_dict['mc_returns'] = np.empty((capacity,), dtype=np.float32)

        # [HIL-SERL Module 2] 位置感知衰减权重 alpha(t)
        # 语义：次优片段 [t_a, t_i] 内 alpha(t) = exp(-lam*(t_i - t))，其余步 alpha = 0.0
        # 使用 np.zeros 初始化（而非 np.empty）：确保未被 actor 赋值的 slot 默认为
        # "无修正"状态（0.0），避免 Critic 更新受到随机垃圾值干扰。
    # 真实值由 actor 端 compute_episode_alpha_and_segment_ids() 在 episode 结束时批量写入。
        if include_alpha_correction:
            dataset_dict['alpha_weight'] = np.zeros((capacity,), dtype=np.float32)

        # [HIL-SERL Module 2] 次优片段 ID（用于按段计算 A_cf）
        # 语义：
        #   -1 : 非次优片段（或无对应片段）
        #   >=0: 对应 actor 端识别出的 suboptimal segment 全局唯一 ID
        if include_segment_ids:
            dataset_dict['segment_ids'] = np.full((capacity,), -1, dtype=np.int32)

        if include_octo_embeddings:
            dataset_dict['embeddings'] = np.empty((capacity, 384), dtype=np.float32)
            dataset_dict['next_embeddings'] = np.empty((capacity, 384), dtype=np.float32)

        if include_next_actions:
            dataset_dict['next_actions'] = np.empty((capacity, *action_space.shape), dtype=action_space.dtype)
            dataset_dict['next_intvn'] = np.empty((capacity,), dtype=bool)

        if include_label:
            dataset_dict['labels'] = np.empty((capacity,), dtype=int)

        if include_grasp_penalty:
            dataset_dict['grasp_penalty'] = np.empty((capacity,), dtype=np.float32)

        super().__init__(dataset_dict)

        self._size = 0
        self._capacity = capacity
        self._insert_index = 0

    def __len__(self) -> int:
        return self._size

    def insert(self, data_dict: DatasetDict):
        _insert_recursively(self.dataset_dict, data_dict, self._insert_index)

        self._insert_index = (self._insert_index + 1) % self._capacity
        self._size = min(self._size + 1, self._capacity)

    def get_iterator(self, queue_size: int = 2, sample_args: dict = {}, device=None):
        # See https://flax.readthedocs.io/en/latest/_modules/flax/jax_utils.html#prefetch_to_device
        # queue_size = 2 should be ok for one GPU.
        queue = collections.deque()

        def enqueue(n):
            for _ in range(n):
                data = self.sample(**sample_args)
                queue.append(jax.device_put(data, device=device))

        enqueue(queue_size)
        while queue:
            yield queue.popleft()
            enqueue(1)

    def download(self, from_idx: int, to_idx: int):
        indices = np.arange(from_idx, to_idx)
        data_dict = self.sample(batch_size=len(indices), indx=indices)
        return to_idx, data_dict

    def get_download_iterator(self):
        last_idx = 0
        while True:
            if last_idx >= self._size:
                raise RuntimeError(
                    f"last_idx {last_idx} >= self._size {self._size}")
            last_idx, batch = self.download(last_idx, self._size)
            yield batch
