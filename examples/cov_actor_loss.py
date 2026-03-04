"""
Covariance-based Entropy-Bounded Actor Loss — JAX/Flax 纯函数实现。

论文思路: 通过估算 Cov(logπ, π·A_soft) 来衡量策略熵的变化趋势，
并用分位数截断构造掩码 I(s)，在 Actor Loss 中只对"熵变化适度"的
样本求梯度，从而实现 Entropy-Bounded 的策略优化。

提供以下入口:
  1. `cov_actor_loss_fn(...)` — 独立纯函数，可替换 policy_loss_fn 的内部逻辑。
     info 字典中包含 "cov/cov_label{0,1,2}_mean_abs" / "_count" 等 per-label 统计。
  2. `compute_label_cov_stats(cov_per_state, labels)` — 按 label (0/1/2) 统计 |c(s)| 均值。
  3. `compute_label_cov_stats_from_batch(...)` — 独立入口，不依赖 cov-clipping 是否启用，
     可在标准 SAC 训练循环中额外调用仅用于诊断。
  4. `make_cov_policy_loss_fn(agent, **kwargs)` — 工厂函数，返回
     `(batch, params, rng) -> (loss, info)` 签名的函数，根据 `use_cov_loss`
     在标准 SAC 和 Cov-Actor Loss 之间切换。

label 约定: 0=离线演示, 1=在线策略采样, 2=人类干预。
"""

from __future__ import annotations

import collections.abc
from typing import Any, Dict, Optional, Tuple

import jax
import jax.numpy as jnp


# ──────────────────────────────────────────────────────────────────────────────
# 辅助纯函数
# ──────────────────────────────────────────────────────────────────────────────

def _covariance(
    x: jax.Array,
    y: jax.Array,
    axis: int = -1,
) -> jax.Array:
    """沿 `axis` 计算总体协方差 Cov(x, y)（均值中心化）。

    Args:
        x, y: 形状相同的数组。
        axis: 沿哪个轴做均值。

    Returns:
        与输入去掉 `axis` 后相同形状的数组。
    """
    x_mean = jnp.mean(x, axis=axis, keepdims=True)
    y_mean = jnp.mean(y, axis=axis, keepdims=True)
    return jnp.mean((x - x_mean) * (y - y_mean), axis=axis)


def _compute_cov_per_state(
    forward_policy_fn,
    forward_critic_fn,
    temperature: jax.Array,
    observations: Any,
    rng: jax.Array,
    K: int = 8,
    eps: float = 1e-8,
) -> jax.Array:
    """对一批 observations 计算每个 state 的 c(s) = -Cov(logπ, π·A_soft)。

    这是 cov_actor_loss_fn 中 K 采样估计协方差部分的独立提取，
    可在不使用 cov-clipping loss 的情况下单独调用。

    Args:
        forward_policy_fn: (obs, rng, grad_params=None) -> Distribution
        forward_critic_fn: (obs, actions, rng) -> (ensemble, B)
        temperature: 当前温度 α (标量)。
        observations: batch 中的 observations (dict 或 array)。
        rng: JAX PRNGKey。
        K: MC 采样数。
        eps: 数值稳定小常数。

    Returns:
        cov_per_state: shape (B,)，每个 state 的 -Cov(logπ, π·A_soft)。
    """
    rng, policy_rng, k_sample_rng, critic_k_rng = jax.random.split(rng, 4)

    # 采 K 个 action
    dist = forward_policy_fn(observations, rng=policy_rng, grad_params=None)
    a_k, logp_k = dist.sample_and_log_prob(
        seed=k_sample_rng, sample_shape=(K,)
    )
    # a_k: (K, B, D),  logp_k: (K, B)
    a_k = jnp.transpose(a_k, (1, 0, 2))    # (B, K, D)
    logp_k = jnp.transpose(logp_k, (1, 0))  # (B, K)

    B_sz, K_sz, D_sz = a_k.shape
    a_k_flat = a_k.reshape(B_sz * K_sz, D_sz)

    def _repeat_obs(obs):
        if isinstance(obs, collections.abc.Mapping):
            return {k: _repeat_obs(v) for k, v in obs.items()}
        reps_insert = list(obs.shape)
        reps_insert.insert(1, K_sz)
        expanded = jnp.broadcast_to(obs[:, None, ...], reps_insert)
        return expanded.reshape(B_sz * K_sz, *obs.shape[1:])

    obs_rep = _repeat_obs(observations)
    q_k_preds = forward_critic_fn(obs_rep, a_k_flat, rng=critic_k_rng)
    min_q_k = jnp.min(q_k_preds, axis=0).reshape(B_sz, K_sz)  # (B, K)

    g_k = min_q_k - temperature * logp_k                        # (B, K)
    V_s = jnp.mean(g_k, axis=1, keepdims=True)                  # (B, 1)
    Asoft_k = g_k - V_s                                          # (B, K)
    pi_k = jnp.clip(jnp.exp(logp_k), a_max=1e6)                # (B, K)
    y_k = pi_k * Asoft_k                                         # (B, K)

    cov_per_state = -_covariance(logp_k, y_k, axis=1)           # (B,)
    return cov_per_state


def compute_label_cov_stats(
    cov_per_state: jax.Array,
    labels: jax.Array,
    eps: float = 1e-8,
) -> Dict[str, jax.Array]:
    """按 label 分组统计 |c(s)| 均值。

    label 约定:
      0 = 离线演示 (offline demo)
      1 = 在线策略采样 (online policy)
      2 = 人类干预 (human intervention)

    Args:
        cov_per_state: shape (B,), 每个 state 的 -Cov(logπ, π·A_soft)。
        labels: shape (B,), 整数标签。
        eps: 防零除小常数。

    Returns:
        包含以下键的字典:
          "cov_label0_mean_abs" — 离线 demo 样本 |c| 均值
          "cov_label1_mean_abs" — 在线策略样本 |c| 均值
          "cov_label2_mean_abs" — 人类干预样本 |c| 均值
          "cov_label{i}_count"  — 各组样本数
    """
    c_abs = jnp.abs(cov_per_state)
    stats: Dict[str, jax.Array] = {}
    label_names = {0: "offline", 1: "online", 2: "intvn"}
    for lid in [0, 1, 2]:
        mask = (labels == lid).astype(c_abs.dtype)
        n = jnp.sum(mask)
        stats[f"cov_label{lid}_mean_abs"] = jnp.sum(c_abs * mask) / (n + eps)
        stats[f"cov_label{lid}_count"] = n
    return stats


def compute_label_cov_stats_from_batch(
    forward_policy_fn,
    forward_critic_fn,
    temperature: jax.Array,
    batch: Dict[str, Any],
    rng: jax.Array,
    K: int = 8,
    eps: float = 1e-8,
) -> Dict[str, jax.Array]:
    """独立入口：从 batch 计算各 label 组的 |c(s)| 均值统计。

    不依赖 cov-clipping actor loss 是否启用——可在标准 SAC actor 更新后
    额外调用，纯粹用于监控/诊断。

    Args:
        forward_policy_fn: agent.forward_policy
        forward_critic_fn: agent.forward_critic
        temperature: 当前温度 α。
        batch: 训练 batch，需含 "observations" 和 "labels" 字段。
        rng: JAX PRNGKey。
        K: MC 采样数。
        eps: 数值稳定小常数。

    Returns:
        per-label 统计字典（所有值均经 stop_gradient）。
    """
    cov_per_state = _compute_cov_per_state(
        forward_policy_fn=forward_policy_fn,
        forward_critic_fn=forward_critic_fn,
        temperature=temperature,
        observations=batch["observations"],
        rng=rng,
        K=K,
        eps=eps,
    )
    cov_per_state = jax.lax.stop_gradient(cov_per_state)
    labels = batch["labels"]
    return compute_label_cov_stats(cov_per_state, labels, eps=eps)


# ──────────────────────────────────────────────────────────────────────────────
# 核心: Covariance-bounded Actor Loss (纯函数)
# ──────────────────────────────────────────────────────────────────────────────

def cov_actor_loss_fn(
    # ---- agent 方法句柄 (由外层闭包提供) ----
    forward_policy_fn,   # (obs, rng, grad_params) -> distrax.Distribution
    forward_critic_fn,   # (obs, actions, rng) -> (ensemble, B)
    temperature: jax.Array,          # 标量 α
    # ---- 数据 ----
    batch: Dict[str, Any],
    params: Any,                     # Actor 参数 (grad 通过此传播)
    rng: jax.Array,
    # ---- 超参 ----
    K: int = 8,
    q_low: float = 0.05,
    q_high: float = 0.90,
    discount: float = 0.97,
    eps: float = 1e-8,
) -> Tuple[jax.Array, Dict[str, Any]]:
    """Covariance-based Entropy-Bounded Actor Loss (JAX 纯函数版)。

    与标准 SAC Actor Loss 的签名对齐: 返回 (scalar_loss, info_dict)。
    梯度仅对 `params` (Actor 参数) 计算。

    流程:
      1. 用当前策略采样 1 个 action → 计算 lt = Q - α logπ
      2. 对同一 obs 采 K 个 action，估算 V(s)、A_soft、π·A_soft
      3. 计算 c(s) = -Cov(logπ, π·A_soft)
      4. 分位数截断: I(s) = 1{ |c| ∈ [ℓ, u] }
      5. Actor loss = -Σ(I · lt) / (Σ I + ε)

    Args:
        forward_policy_fn: 前向策略函数。
        forward_critic_fn: 前向 critic 函数（不对 critic 参数求导）。
        temperature: 当前温度 α。
        batch: 训练 batch。
        params: Actor 网络参数。
        rng: JAX PRNGKey。
        K: 每个 state 采样的 MC 动作数。
        q_low: |c| 下分位数阈值。
        q_high: |c| 上分位数阈值。
        discount: 折扣因子（仅用于日志，此处不参与 backup 计算）。
        eps: 数值稳定小常数。

    Returns:
        (actor_loss, info_dict)
    """
    batch_size = batch["rewards"].shape[0]
    observations = batch["observations"]

    # ── Step 1: 采样 1 个 action（带梯度），计算 lt = Q(s,a) - α logπ(a|s) ──
    rng, policy_rng, sample_rng, critic_rng, cov_rng = jax.random.split(rng, 5)
    dist = forward_policy_fn(observations, rng=policy_rng, grad_params=params)
    actions_pi, log_probs = dist.sample_and_log_prob(seed=sample_rng)
    # actions_pi: (B, D),  log_probs: (B,)

    predicted_qs = forward_critic_fn(observations, actions_pi, rng=critic_rng)
    # predicted_qs: (ensemble, B)
    min_q = jnp.min(predicted_qs, axis=0)  # (B,)

    lt = min_q - temperature * log_probs   # (B,)  — "soft return" per sample

    # 策略熵（用于日志）
    policy_entropy = jax.lax.stop_gradient(-jnp.mean(log_probs))

    # ── Step 2: 调用辅助函数计算每个 state 的 c(s) (stop_gradient，只用于掩码) ──
    cov_per_state = _compute_cov_per_state(
        forward_policy_fn=forward_policy_fn,
        forward_critic_fn=forward_critic_fn,
        temperature=temperature,
        observations=observations,
        rng=cov_rng,
        K=K,
        eps=eps,
    )
    c = jax.lax.stop_gradient(cov_per_state)   # (B,)，已是 -Cov
    c_abs = jnp.abs(c)

    # ── Step 3: 分位数截断 → 掩码 I ──
    l_bound = jnp.percentile(c_abs, q_low * 100.0)
    u_bound = jnp.percentile(c_abs, q_high * 100.0)
    I_mask = ((c_abs >= l_bound) & (c_abs <= u_bound)).astype(lt.dtype)  # (B,)
    keep = jnp.sum(I_mask)
    keep_ratio = jax.lax.stop_gradient(keep / (batch_size + eps))

    # ── Step 4: Actor Loss = -Σ(I · lt) / (Σ I + ε) ──
    actor_loss = -(I_mask * lt).sum() / (keep + eps)

    # ── Step 5: 按 label 统计 |c(s)| 均值（不依赖截断是否开启）──
    label_cov_stats = compute_label_cov_stats(c, batch["labels"], eps=eps)

    info = {
        "actor_loss": actor_loss,
        "temperature": temperature,
        "entropy": policy_entropy,
        # Cov 整体诊断指标
        "cov_batch_mean": jax.lax.stop_gradient(jnp.mean(cov_per_state)),
        "c_abs_mean": jax.lax.stop_gradient(jnp.mean(c_abs)),
        "c_abs_l_bound": jax.lax.stop_gradient(l_bound),
        "c_abs_u_bound": jax.lax.stop_gradient(u_bound),
        "mask_keep_ratio": keep_ratio,
        # 按 label 分组的 |c| 均值 (label: 0=offline, 1=online, 2=intvn)
        **{f"cov/{k}": jax.lax.stop_gradient(v) for k, v in label_cov_stats.items()},
    }

    return actor_loss, info


def cov_actor_loss_fn_batch_global(
    forward_policy_fn,
    forward_critic_fn,
    temperature: jax.Array,
    batch: Dict[str, Any],
    params: Any,
    rng: jax.Array,
    # 与 PPO compute_policy_loss_clip_cov 对应的超参
    cov_lb: float = 0.0,     # 对应 clip_cov_lb（绝对值下界，设0则不过滤极小cov）
    cov_ub: float = float("inf"),  # 对应 clip_cov_ub
    clip_ratio: float = 0.02,  # 保留 top clip_ratio 比例的样本做 clip（置0则用分位数）
    q_low: float = 0.05,     # 分位数下界（clip_ratio=0 时生效）
    q_high: float = 0.90,    # 分位数上界
    eps: float = 1e-8,
) -> Tuple[jax.Array, Dict[str, Any]]:
    """Batch-level covariance actor loss — 与 PPO compute_policy_loss_clip_cov 结构对齐。

    不需要额外 MC 采样，协方差在整个 batch 的 transitions 上全局计算：
        cov_i = (logπ_i - mean_logπ) * (lt_i - mean_lt)
    其中 lt_i = Q(s_i, a_i) - α logπ(a_i|s_i)，即 SAC 的 soft return。
    """
    observations = batch["observations"]
    batch_size = batch["rewards"].shape[0]

    rng, policy_rng, sample_rng, critic_rng = jax.random.split(rng, 4)

    # ── Step 1: 采 1 个 action（带梯度），计算 lt ──────────────────────────────
    dist = forward_policy_fn(observations, rng=policy_rng, grad_params=params)
    actions_pi, log_probs = dist.sample_and_log_prob(seed=sample_rng)
    # log_probs: (B,)

    predicted_qs = forward_critic_fn(observations, actions_pi, rng=critic_rng)
    min_q = jnp.min(predicted_qs, axis=0)          # (B,)
    lt = min_q - temperature * log_probs            # (B,)  ← 类比 PPO 的 -pg_losses1

    policy_entropy = jax.lax.stop_gradient(-jnp.mean(log_probs))

    # ── Step 2: Batch-level 协方差，复用 Step 1 结果（无需额外采样）─────────────
    # 完全对应 PPO:
    #   cov_all = (advantages - mean_adv) * (log_prob - mean_logp)
    logp_sg = jax.lax.stop_gradient(log_probs)     # 协方差用 stop_gradient，只做掩码
    lt_sg   = jax.lax.stop_gradient(lt)

    cov_batch = (logp_sg - jnp.mean(logp_sg)) * (lt_sg - jnp.mean(lt_sg))  # (B,)
    c_abs = jnp.abs(cov_batch)

    # ── Step 3: 构造掩码 I（两种方式，择一）────────────────────────────────────
    #
    # 方式A：分位数截断（原 cov_actor_loss.py 风格）
    #   l_bound = percentile(c_abs, q_low*100)
    #   u_bound = percentile(c_abs, q_high*100)
    #   I_mask  = (c_abs >= l_bound) & (c_abs <= u_bound)
    #
    # 方式B：绝对阈值 + 固定比例（PPO clip_cov_lb/ub 风格，推荐）
    #   先过滤 [cov_lb, cov_ub] 内的样本，再随机保留 clip_ratio 比例
    l_bound = jnp.percentile(c_abs, q_low  * 100.0)
    u_bound = jnp.percentile(c_abs, q_high * 100.0)
    I_mask  = ((c_abs >= l_bound) & (c_abs <= u_bound)).astype(lt.dtype)   # (B,)

    keep = jnp.sum(I_mask)
    keep_ratio = jax.lax.stop_gradient(keep / (batch_size + eps))

    # ── Step 4: Actor Loss = -Σ(I · lt) / (Σ I + ε) ────────────────────────────
    actor_loss = -(I_mask * lt).sum() / (keep + eps)

    # ── Step 5: 按 label 统计（可选诊断）────────────────────────────────────────
    label_cov_stats = compute_label_cov_stats(cov_batch, batch["labels"], eps=eps)

    info = {
        "actor_loss":       actor_loss,
        "temperature":      temperature,
        "entropy":          policy_entropy,
        "cov_batch_mean":   jax.lax.stop_gradient(jnp.mean(cov_batch)),
        "c_abs_mean":       jax.lax.stop_gradient(jnp.mean(c_abs)),
        "c_abs_l_bound":    jax.lax.stop_gradient(l_bound),
        "c_abs_u_bound":    jax.lax.stop_gradient(u_bound),
        "mask_keep_ratio":  keep_ratio,
        **{f"cov/{k}": jax.lax.stop_gradient(v) for k, v in label_cov_stats.items()},
    }
    return actor_loss, info


# ──────────────────────────────────────────────────────────────────────────────
# 工厂: 生成可替换 policy_loss_fn 的闭包
# ──────────────────────────────────────────────────────────────────────────────

# 防止 JAX 多次 trace 时重复打印
_COV_FACTORY_BANNER_PRINTED: bool = False


def _print_cov_active_banner(K: int, q_low: float, q_high: float) -> None:
    """打印 COV ACTOR LOSS 已生效的醒目横幅（仅首次）。"""
    global _COV_FACTORY_BANNER_PRINTED
    if _COV_FACTORY_BANNER_PRINTED:
        return
    _COV_FACTORY_BANNER_PRINTED = True
    bold  = "\033[1m"
    cyan  = "\033[96m"
    reset = "\033[00m"
    line  = "═" * 52
    print(f"\n{bold}{cyan}╔{line}╗")
    print(f"║{'  ✅  COV ACTOR LOSS  IS  ACTIVE':^52}║")
    print(f"║{'  → make_cov_policy_loss_fn() entered by JAX':^52}║")
    print(f"╠{line}╣")
    print(f"║  MC samples  K        = {K:<27}║")
    print(f"║  Lower  percentile    = {q_low:<27}║")
    print(f"║  Upper  percentile    = {q_high:<27}║")
    print(f"╚{line}╝{reset}\n")


def make_cov_policy_loss_fn(
    agent: Any,
    *,
    K: int = 8,
    q_low: float = 0.05,
    q_high: float = 0.90,
    eps: float = 1e-8,
):
    """构造一个与 `SACAgent.policy_loss_fn` 签名完全一致的闭包。

    返回的函数签名: `(batch, params, rng) -> (loss, info)`
    可直接赋值给 agent 的 loss_fns 字典中的 "actor" 键。

    Args:
        agent: SACAgent (或其子类) 实例。
        K: 每个 obs 的 MC 采样数。
        q_low: 协方差下分位数。
        q_high: 协方差上分位数。
        eps: 数值稳定常数。

    Returns:
        policy_loss_fn 闭包。
    """
    _print_cov_active_banner(K, q_low, q_high)
    discount = agent.config.get("discount", 0.97)

    def _policy_loss_fn(batch, params, rng):
        temperature = agent.forward_temperature()
        return cov_actor_loss_fn(
            forward_policy_fn=agent.forward_policy,
            forward_critic_fn=agent.forward_critic,
            temperature=temperature,
            batch=batch,
            params=params,
            rng=rng,
            K=K,
            q_low=q_low,
            q_high=q_high,
            discount=discount,
            eps=eps,
        )

    return _policy_loss_fn
