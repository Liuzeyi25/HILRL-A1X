"""
采样策略模块 — 为 RLPD Learner 提供三种经验回放采样/过滤策略。

策略 A: 空间范围截断过滤 (Workspace Filtering)
策略 B: 随机丢弃 (Random Drop)
策略 C: 优先经验重放 (Prioritized Experience Replay - PER)

每个策略通过统一接口 `apply(batch, ...)` 返回过滤/加权后的 batch。
使用工厂函数 `make_sampling_strategy(name, **kwargs)` 创建策略实例。
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import jax
import jax.numpy as jnp
import numpy as np


# ──────────────────────────────────────────────────────────────────────────────
# 类型别名
# ──────────────────────────────────────────────────────────────────────────────
Batch = Dict[str, Any]


# ──────────────────────────────────────────────────────────────────────────────
# 基类
# ──────────────────────────────────────────────────────────────────────────────
class SamplingStrategy(abc.ABC):
    """所有采样策略的基类。"""

    @abc.abstractmethod
    def apply(self, batch: Batch, rng: jax.Array, **kwargs) -> Batch:
        """对已采样 batch 进行过滤/重加权，返回处理后的 batch。

        Args:
            batch: 从 replay buffer 采到的 batch，包含 observations, actions,
                   rewards, next_observations, masks, dones 等键。
                   observations["state"] 形状为 (B, state_dim)，前三维为 (x,y,z)。
            rng:   JAX PRNGKey，用于随机操作。

        Returns:
            处理后的 batch（可能样本数 <= 原始 B）。
        """
        ...


# ──────────────────────────────────────────────────────────────────────────────
# 工具函数
# ──────────────────────────────────────────────────────────────────────────────
def _index_batch(batch: Batch, indices: jax.Array) -> Batch:
    """根据 indices 从 batch 中取子集，支持嵌套 dict。"""
    def _index(x: Any) -> Any:
        if isinstance(x, dict):
            return {k: _index(v) for k, v in x.items()}
        # jnp array or np array
        return x[indices]
    return {k: _index(v) for k, v in batch.items()}


def _batch_size(batch: Batch) -> int:
    """返回 batch 中第一个叶子数组的第 0 维大小。"""
    for v in batch.values():
        if isinstance(v, dict):
            return _batch_size(v)
        return v.shape[0]
    raise ValueError("Empty batch")


def _print_strategy_active_banner(name: str, details: list) -> None:
    """首次进入 apply() 时打印采样策略已生效的醒目横幅。"""
    bold   = "\033[1m"
    yellow = "\033[93m"
    reset  = "\033[00m"
    line   = "═" * 52
    print(f"\n{bold}{yellow}╔{line}╗")
    print(f"║{'  ✅  SAMPLING STRATEGY  IS  ACTIVE':^52}║")
    print(f"║{'  → ' + name + '.apply() entered':^52}║")
    print(f"╠{line}╣")
    for d in details:
        print(f"║  {d:<50}║")
    print(f"╚{line}╝{reset}\n")


# ──────────────────────────────────────────────────────────────────────────────
# 策略 A: 空间范围截断过滤 (Workspace Filtering)
# ──────────────────────────────────────────────────────────────────────────────
@dataclass
class WorkspaceFilteringStrategy(SamplingStrategy):
    """基于末端执行器 (x, y, z) 坐标的工作空间过滤。

    过滤逻辑:
      1. 判断每条样本的 state[:3] (x, y, z) 是否在给定的工作区间内。
      2. 若空间内样本占比 >= min_keep_ratio，直接保留空间内样本。
      3. 若空间内样本不足 min_keep_ratio，从空间外样本中随机抽回补足至
         ceil(B * min_keep_ratio)。

    Attributes:
        x_range: (min, max) — x 轴有效区间。
        y_range: (min, max) — y 轴有效区间。
        z_range: (min, max) — z 轴有效区间。
        min_keep_ratio: 最低保留比例，默认 0.85。
        state_key: observations 中状态向量的键名，默认 "state"。
        xyz_indices: state 向量中 x, y, z 对应的索引，默认 (0, 1, 2)。
    """

    x_range: Tuple[float, float] = (-1.0, 1.0)
    y_range: Tuple[float, float] = (-1.0, 1.0)
    z_range: Tuple[float, float] = (-1.0, 1.0)
    min_keep_ratio: float = 0.85
    state_key: str = "state"
    xyz_indices: Tuple[int, int, int] = (0, 1, 2)
    _first_call: bool = field(default=True, init=False, repr=False)

    def apply(self, batch: Batch, rng: jax.Array, **kwargs) -> Batch:
        """执行空间过滤。"""
        if self._first_call:
            _print_strategy_active_banner(
                "WorkspaceFilteringStrategy",
                [
                    f"x_range        = {self.x_range}",
                    f"y_range        = {self.y_range}",
                    f"z_range        = {self.z_range}",
                    f"min_keep_ratio = {self.min_keep_ratio}",
                ],
            )
            self._first_call = False
        B = _batch_size(batch)
        min_keep = int(np.ceil(B * self.min_keep_ratio))

        # 提取 (x, y, z)
        state = batch["observations"][self.state_key]  # (B, state_dim)
        ix, iy, iz = self.xyz_indices
        x, y, z = state[:, ix], state[:, iy], state[:, iz]

        # 布尔 mask: True = 在工作空间内
        in_ws = (
            (x >= self.x_range[0]) & (x <= self.x_range[1])
            & (y >= self.y_range[0]) & (y <= self.y_range[1])
            & (z >= self.z_range[0]) & (z <= self.z_range[1])
        )  # (B,)

        # 使用 numpy 进行索引操作（batch 从 CPU 来或已转为 numpy）
        in_ws_np = np.asarray(in_ws)
        in_indices = np.where(in_ws_np)[0]
        out_indices = np.where(~in_ws_np)[0]

        if len(in_indices) >= min_keep:
            # 空间内已足够，直接保留
            keep_indices = in_indices
        else:
            # 空间内不足，从空间外随机补足
            need = min_keep - len(in_indices)
            rng_np = int(jax.random.randint(rng, (), 0, jnp.int32(2**30)))
            local_rng = np.random.RandomState(rng_np)
            补回 = local_rng.choice(out_indices, size=min(need, len(out_indices)), replace=False)
            keep_indices = np.concatenate([in_indices, 补回])

        keep_indices = jnp.array(keep_indices)
        return _index_batch(batch, keep_indices)


# ──────────────────────────────────────────────────────────────────────────────
# 策略 B: 随机丢弃 (Random Drop)
# ──────────────────────────────────────────────────────────────────────────────
@dataclass
class RandomDropStrategy(SamplingStrategy):
    """随机丢弃 Baseline: 均匀随机地丢弃 drop_ratio 比例的样本。

    Attributes:
        drop_ratio: 丢弃比例，默认 0.15 (即保留 85%)。
    """

    drop_ratio: float = 0.15
    _first_call: bool = field(default=True, init=False, repr=False)

    def apply(self, batch: Batch, rng: jax.Array, **kwargs) -> Batch:
        """随机保留 (1 - drop_ratio) 比例的样本。"""
        if self._first_call:
            _print_strategy_active_banner(
                "RandomDropStrategy",
                [f"drop_ratio = {self.drop_ratio}"],
            )
            self._first_call = False
        B = _batch_size(batch)
        keep = int(np.ceil(B * (1.0 - self.drop_ratio)))

        # JAX 随机 permutation 取前 keep 个
        perm = jax.random.permutation(rng, B)
        keep_indices = jnp.sort(perm[:keep])  # 排序以保持时间顺序
        return _index_batch(batch, keep_indices)


# ──────────────────────────────────────────────────────────────────────────────
# 策略 C: 优先经验重放 (Prioritized Experience Replay — PER)
# ──────────────────────────────────────────────────────────────────────────────
@dataclass
class PERStrategy(SamplingStrategy):
    """基于 TD-error 的 Prioritized Experience Replay (PER) 采样策略。

    由于底层 replay buffer 不原生支持 PER，本实现采用"post-hoc"方式:
      1. 从 uniform-sampled batch 中，利用当前 critic 计算 TD-error。
      2. 基于 TD-error 计算采样概率，按概率 **有放回重采样** 得到新 batch。
      3. 计算 Importance-Sampling (IS) 权重，附加到 batch["is_weights"] 用于
         loss 加权。

    典型工作流:
      batch = next(iterator)
      batch = per_strategy.apply(batch, rng, agent=agent)
      # batch["is_weights"] 可在 loss 计算时加权
      agent, info = agent.update(batch)
      # info["predicted_qs"], info["target_qs"] 可用于更新优先级 (可选)

    Attributes:
        alpha: 优先级指数 (0 = uniform, 1 = 纯 proportional)。默认 0.6。
        beta: IS 权重指数 (0 = 无修正, 1 = 完全修正)。默认 0.4。
        epsilon: 防止 TD-error 为零时优先级为 0 的小常数。默认 1e-6。
        max_is_weight: IS 权重裁剪上限，防止梯度爆炸。默认 10.0。
    """

    alpha: float = 0.6
    beta: float = 0.4
    epsilon: float = 1e-6
    max_is_weight: float = 10.0
    _first_call: bool = field(default=True, init=False, repr=False)

    def compute_td_errors(
        self,
        batch: Batch,
        agent: Any,
        rng: jax.Array,
    ) -> jax.Array:
        """使用当前 critic 估算 batch 中每条样本的 |TD-error|。

        TD-error = |r + γ * min_j Q_target(s', a') - Q(s, a)|

        Args:
            batch: 含 observations, actions, rewards, next_observations, masks 的 batch。
            agent: SACAgent 实例，需要 forward_critic / sample_actions 等方法。
            rng: JAX PRNGKey。

        Returns:
            td_errors: (B,) 的非负数组。
        """
        rng1, rng2, rng3 = jax.random.split(rng, 3)

        # 当前 Q 值: (ensemble, B) → 取 mean over ensemble → (B,)
        current_qs = agent.forward_critic(
            batch["observations"], batch["actions"], rng1, train=False
        )  # (ensemble, B, 1) 或 (ensemble, B)
        current_q = jnp.mean(current_qs, axis=0).squeeze(-1) if current_qs.ndim == 3 else jnp.mean(current_qs, axis=0)

        # 下一步动作 (从策略采样)
        next_actions = agent.sample_actions(
            observations=batch["next_observations"], seed=rng2
        )

        # 目标 Q 值
        target_qs = agent.forward_critic(
            batch["next_observations"], next_actions, rng3, train=False
        )
        target_q = jnp.min(target_qs, axis=0).squeeze(-1) if target_qs.ndim == 3 else jnp.min(target_qs, axis=0)

        # TD target
        rewards = batch["rewards"].squeeze()
        masks = batch["masks"].squeeze()
        discount = 0.97  # 使用常见默认值；也可通过 kwargs 传入
        td_target = rewards + discount * masks * target_q
        td_errors = jnp.abs(td_target - current_q)
        return td_errors

    def apply(
        self,
        batch: Batch,
        rng: jax.Array,
        *,
        agent: Any = None,
        td_errors: Optional[jax.Array] = None,
        **kwargs,
    ) -> Batch:
        """基于 TD-error 的优先级重采样 + IS 权重计算。

        需要传入 `agent`（用于计算 TD-error）或直接传入 `td_errors`。

        Args:
            batch: 均匀采样的 batch。
            rng: JAX PRNGKey。
            agent: SACAgent (与 td_errors 二选一)。
            td_errors: 预计算的 (B,) TD-error 数组 (与 agent 二选一)。

        Returns:
            按优先级重采样后的 batch，附带 batch["is_weights"]。
        """
        if self._first_call:
            _print_strategy_active_banner(
                "PERStrategy",
                [
                    f"alpha         = {self.alpha}",
                    f"beta          = {self.beta}",
                    f"epsilon       = {self.epsilon}",
                    f"max_is_weight = {self.max_is_weight}",
                ],
            )
            self._first_call = False
        if td_errors is None:
            assert agent is not None, "PER requires `agent` or `td_errors`."
            rng, sub = jax.random.split(rng)
            td_errors = self.compute_td_errors(batch, agent, sub)

        # 优先级 = (|δ| + ε)^α
        priorities = jnp.power(td_errors + self.epsilon, self.alpha)
        probs = priorities / jnp.sum(priorities)  # (B,)

        B = _batch_size(batch)

        # 按概率有放回重采样
        rng, sub = jax.random.split(rng)
        indices = jax.random.choice(sub, B, shape=(B,), p=probs, replace=True)

        # Importance-Sampling 权重: w_i = (1 / (N * P(i)))^β, 归一化后 max=1
        is_weights = jnp.power(1.0 / (B * probs[indices] + 1e-10), self.beta)
        is_weights = is_weights / jnp.max(is_weights)  # 归一化
        is_weights = jnp.clip(is_weights, a_max=self.max_is_weight)

        new_batch = _index_batch(batch, indices)
        new_batch["is_weights"] = is_weights
        return new_batch


# ──────────────────────────────────────────────────────────────────────────────
# 无操作策略 (NoOp / Identity)
# ──────────────────────────────────────────────────────────────────────────────
@dataclass
class NoOpStrategy(SamplingStrategy):
    """不做任何处理，直接返回原始 batch。用作默认 / 对照。"""

    def apply(self, batch: Batch, rng: jax.Array, **kwargs) -> Batch:
        return batch


# ──────────────────────────────────────────────────────────────────────────────
# 工厂函数
# ──────────────────────────────────────────────────────────────────────────────
_STRATEGY_REGISTRY: Dict[str, type] = {
    "none": NoOpStrategy,
    "noop": NoOpStrategy,
    "workspace_filtering": WorkspaceFilteringStrategy,
    "random_drop": RandomDropStrategy,
    "per": PERStrategy,
}


def make_sampling_strategy(name: str, **kwargs) -> SamplingStrategy:
    """工厂函数: 根据名称创建采样策略实例。

    Args:
        name: 策略名称，可选:
              "none"  / "noop"               — 不做处理
              "workspace_filtering"          — 空间范围截断过滤 (策略 A)
              "random_drop"                  — 随机丢弃 (策略 B)
              "per"                          — 优先经验重放 (策略 C)
        **kwargs: 传递给对应策略 dataclass 的初始化参数。

    Returns:
        SamplingStrategy 子类实例。

    Raises:
        ValueError: 未知策略名称。

    Example:
        >>> strategy = make_sampling_strategy("workspace_filtering",
        ...     x_range=(0.2, 0.8), y_range=(-0.3, 0.3), z_range=(0.0, 0.5))
        >>> filtered_batch = strategy.apply(batch, rng)
    """
    name_lower = name.lower().strip()
    if name_lower not in _STRATEGY_REGISTRY:
        available = ", ".join(sorted(_STRATEGY_REGISTRY.keys()))
        raise ValueError(
            f"Unknown sampling strategy '{name}'. Available: {available}"
        )
    cls = _STRATEGY_REGISTRY[name_lower]
    return cls(**kwargs)
