from collections import deque
from threading import Lock
from typing import Union, Iterable, Optional

import gymnasium as gym
import jax
import numpy as np
from flax.core import frozen_dict
from serl_launcher.data.replay_buffer import ReplayBuffer
from serl_launcher.data.memory_efficient_replay_buffer import (
    MemoryEfficientReplayBuffer,
)

from agentlace.data.data_store import DataStoreBase


class ReplayBufferDataStore(ReplayBuffer, DataStoreBase):
    def __init__(
        self,
        observation_space: gym.Space,
        action_space: gym.Space,
        capacity: int,
    ):
        ReplayBuffer.__init__(self, observation_space, action_space, capacity)
        DataStoreBase.__init__(self, capacity)
        self._lock = Lock()

    # ensure thread safety
    def insert(self, *args, **kwargs):
        with self._lock:
            super(ReplayBufferDataStore, self).insert(*args, **kwargs)

    # ensure thread safety
    def sample(self, *args, **kwargs):
        with self._lock:
            return super(ReplayBufferDataStore, self).sample(*args, **kwargs)

    # NOTE: method for DataStoreBase
    def latest_data_id(self):
        return self._insert_index

    # NOTE: method for DataStoreBase
    def get_latest_data(self, from_id: int):
        # Actor-side buffer: 只向 learner 推数据，不从 learner 拉，返回空列表即可
        return []


class MemoryEfficientReplayBufferDataStore(MemoryEfficientReplayBuffer, DataStoreBase):
    def __init__(
        self,
        observation_space: gym.Space,
        action_space: gym.Space,
        capacity: int,
        image_keys: Iterable[str] = ("image",),
        include_alpha_correction: bool = False,
        include_segment_ids: bool = False,
        **kwargs,
    ):
        MemoryEfficientReplayBuffer.__init__(
            self, observation_space, action_space, capacity,
            pixel_keys=image_keys,
            include_alpha_correction=include_alpha_correction,
            include_segment_ids=include_segment_ids,
            **kwargs
        )
        DataStoreBase.__init__(self, capacity)
        self._lock = Lock()

    # ensure thread safety
    def insert(self, *args, **kwargs):
        with self._lock:
            super(MemoryEfficientReplayBufferDataStore, self).insert(*args, **kwargs)

    # ensure thread safety
    def sample(self, *args, **kwargs):
        with self._lock:
            return super(MemoryEfficientReplayBufferDataStore, self).sample(
                *args, **kwargs
            )

    # NOTE: method for DataStoreBase
    def latest_data_id(self):
        return self._insert_index

    # NOTE: method for DataStoreBase
    def get_latest_data(self, from_id: int):
        # Actor-side buffer: 只向 learner 推数据，不从 learner 拉，返回空列表即可
        return []


def populate_data_store(
    data_store: DataStoreBase,
    demos_path: str,
):
    """
    Utility function to populate demonstrations data into data_store.
    :return data_store
    """
    import pickle as pkl
    import numpy as np
    from copy import deepcopy

    for demo_path in demos_path:
        with open(demo_path, "rb") as f:
            demo = pkl.load(f)
            for transition in demo:
                data_store.insert(transition)
        print(f"Loaded {len(data_store)} transitions.")
    return data_store


def populate_data_store_with_z_axis_only(
    data_store: DataStoreBase,
    demos_path: str,
):
    """
    Utility function to populate demonstrations data into data_store.
    This will remove the x and y cartesian coordinates from the state.
    :return data_store
    """
    import pickle as pkl
    import numpy as np
    from copy import deepcopy

    for demo_path in demos_path:
        with open(demo_path, "rb") as f:
            demo = pkl.load(f)
            for transition in demo:
                tmp = deepcopy(transition)
                tmp["observations"]["state"] = np.concatenate(
                    (
                        tmp["observations"]["state"][:, :4],
                        tmp["observations"]["state"][:, 6][None, ...],
                        tmp["observations"]["state"][:, 10:],
                    ),
                    axis=-1,
                )
                tmp["next_observations"]["state"] = np.concatenate(
                    (
                        tmp["next_observations"]["state"][:, :4],
                        tmp["next_observations"]["state"][:, 6][None, ...],
                        tmp["next_observations"]["state"][:, 10:],
                    ),
                    axis=-1,
                )
                data_store.insert(tmp)
        print(f"Loaded {len(data_store)} transitions.")
    return data_store


# =============================================================================
# [HIL-SERL Module 3] 偏好学习缓冲区
# =============================================================================

class PreferenceBufferDataStore(DataStoreBase):
    """
    轻量级干预偏好对缓冲区，存储 Module 3（偏好引导策略学习）所需数据。

    每条数据代表一个干预事件的起始时刻 t_i，包含：
      - observations:   s_{t_i}，干预发生时的状态（包含图像，stacked obs）
      - human_actions:  a^h，操作员给出的替代动作
            - segment_ids:    当前干预事件对应的次优片段 ID（全局唯一）

    Module 2 同时使用该缓冲区：通过 target network 前向推断计算反事实优势
        A_cf = max(0, mean_batch[Q_tgt(s, a^h) - Q_tgt(s, a^π)])

    线程安全：actor 写入（TrainerClient），learner 读取（TrainerServer），Lock 保护。
    """

    def __init__(self, capacity: int = 5000):
        super().__init__(capacity)
        self._capacity = capacity
        self._buffer: deque = deque(maxlen=capacity)
        self._lock = Lock()

    def insert(self, data: dict):
        """
        插入一条偏好数据。data 需含键：
                    "observations", "human_actions", "segment_ids"
        """
        with self._lock:
            self._buffer.append(data)

    def sample(self, batch_size: int) -> Optional[frozen_dict.FrozenDict]:
        """
        随机采样 batch_size 条偏好数据，返回 FrozenDict。
        若缓冲区数据不足 batch_size，返回 None。
        """
        with self._lock:
            if len(self._buffer) < batch_size:
                return None
            indices = np.random.choice(len(self._buffer), size=batch_size, replace=False)
            items = [self._buffer[i] for i in indices]

        # list-of-dict -> dict-of-array（格式与 MemoryEfficientReplayBuffer.sample 保持一致）
        # 顶层 key: observations（可能是嵌套字典）、human_actions、policy_actions
        # 对图像型 observations：每个 sub_key 对应一个摄像头视角，
        #   np.stack 后形状为 (B, H, W, C) 或 (B, T, H, W, C)（取决于帧堆叠数）
        # 对连续向量型 key（actions 等）：直接 np.stack 得到 (B, action_dim)
        batch = {}
        for key in items[0].keys():
            vals = [item[key] for item in items]
            if isinstance(vals[0], dict):
                # 嵌套字典（如 observations），逐 sub_key 展开
                batch[key] = {}
                for sub_key in vals[0].keys():
                    batch[key][sub_key] = np.stack([v[sub_key] for v in vals], axis=0)
            else:
                batch[key] = np.stack(vals, axis=0)
        return frozen_dict.freeze(batch)

    def __len__(self) -> int:
        with self._lock:
            return len(self._buffer)

    def latest_data_id(self) -> int:
        return len(self)

    def get_latest_data(self, from_id: int):
        # Actor-side buffer: 只向 learner 推数据，不从 learner 拉，返回空列表即可
        return []
