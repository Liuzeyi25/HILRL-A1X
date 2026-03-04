"""
Diffusion-adapted Covariance-based Entropy-Bounded Actor Loss for ConRFT.

核心思想: ConRFT 的一致性策略无解析 log π(a|s)，但去噪重建 loss
    ℓ(a, s) = ||f_θ(a + σε, σ | s) - a||²
是 negative log-likelihood 的天然代理:
    高 ℓ → 策略对该 action 不熟悉 → "低 log π"
    低 ℓ → 策略很好地重建 → "高 log π"

将 h_k = -ℓ_k 作为 proxy log-prob 代入原始 Cov(logπ, π·A_soft) 公式:
    ĥ_k ≜ -||f_θ(a_k + σ_eval·ε, σ_eval | s) - a_k||²
    c(s) = -Cov_k(ĥ_k, exp(ĥ_k) · Â_k)
    其中 Â_k = Q(s, a_k) - V̂(s)（无温度项，ConRFT 没有可学习温度）

然后用分位数截断构造掩码 I(s)，在 Actor Loss 中只对 |c| 适中的样本求梯度。

提供入口:
  1. cov_actor_loss_diffusion_fn(...) — 核心纯函数
  2. make_cov_policy_loss_fn_diffusion(agent, ...) — 工厂函数，返回
     (batch, params, rng) -> (loss, info) 签名，可直接替换 loss_fns["actor"]
"""

from __future__ import annotations

import collections.abc
from typing import Any, Dict, Optional, Tuple

import jax
import jax.numpy as jnp

from serl_launcher.utils.jax_utils import append_dims, mean_flat
from serl_launcher.utils.train_utils import get_snr, get_weightings


# ──────────────────────────────────────────────────────────────────────────────
# 辅助纯函数
# ──────────────────────────────────────────────────────────────────────────────

def _covariance(x: jax.Array, y: jax.Array, axis: int = -1) -> jax.Array:
    """沿 axis 计算总体协方差 Cov(x, y)。"""
    x_mean = jnp.mean(x, axis=axis, keepdims=True)
    y_mean = jnp.mean(y, axis=axis, keepdims=True)
    return jnp.mean((x - x_mean) * (y - y_mean), axis=axis)


def _repeat_for_k(x, K: int, B: int):
    """将 (B, ...) 数组复制为 (B*K, ...)。支持嵌套 dict 和 None。"""
    if x is None:
        return None
    if isinstance(x, collections.abc.Mapping):
        return {k: _repeat_for_k(v, K, B) for k, v in x.items()}
    expanded = jnp.broadcast_to(x[:, None, ...], (B, K, *x.shape[1:]))
    return expanded.reshape(B * K, *x.shape[1:])


def compute_label_cov_stats(
    cov_per_state: jax.Array,
    labels: jax.Array,
    eps: float = 1e-8,
) -> Dict[str, jax.Array]:
    """按 label (0=offline, 1=online, 2=intvn) 分组统计 |c(s)| 均值。"""
    c_abs = jnp.abs(cov_per_state)
    stats: Dict[str, jax.Array] = {}
    for lid in [0, 1, 2]:
        mask = (labels == lid).astype(c_abs.dtype)
        n = jnp.sum(mask)
        stats[f"cov_label{lid}_mean_abs"] = jnp.sum(c_abs * mask) / (n + eps)
        stats[f"cov_label{lid}_count"] = n
    return stats


# ──────────────────────────────────────────────────────────────────────────────
# Step 1-4: 计算 per-state 协方差掩码 (stop_gradient)
# ──────────────────────────────────────────────────────────────────────────────

def _compute_cov_mask_diffusion(
    agent,
    batch: Dict[str, Any],
    rng: jax.Array,
    K: int,
    sigma_eval: float,
    q_low: float,
    q_high: float,
    eps: float = 1e-8,
) -> Tuple[jax.Array, Dict[str, jax.Array]]:
    """计算 per-state 协方差指标和分位数截断掩码。

    Returns:
        I_mask: (B,) 浮点掩码，0 或 1
        cov_info: 监控指标字典
    """
    batch_size = batch["rewards"].shape[0]

    # ── 采 K 个 action（从不同噪声一步去噪） ──
    rng, sample_rng = jax.random.split(rng)
    a_k = agent.forward_policy_and_sample(
        batch["tasks"], batch["observations"], batch["embeddings"],
        rng=sample_rng, repeat=K,
    )  # (B, K, D)
    a_k = jax.lax.stop_gradient(a_k)

    B, K_sz, D = a_k.shape

    # ── 计算 proxy log-prob: h_k = -||f_θ(a_k + σ·ε, σ | s) - a_k||² ──
    rng, noise_rng, denoise_rng = jax.random.split(rng, 3)
    noise_eval = jax.random.normal(noise_rng, shape=(B, K_sz, D))
    x_t_eval = a_k + sigma_eval * noise_eval         # (B, K, D)

    # 展为 batch 维度 (B*K, ...)
    x_t_flat = x_t_eval.reshape(B * K_sz, D)
    sigmas_flat = jnp.full((B * K_sz,), sigma_eval)
    obs_rep = _repeat_for_k(batch["observations"], K_sz, B)
    tasks_rep = _repeat_for_k(batch["tasks"], K_sz, B)
    emb_rep = _repeat_for_k(batch["embeddings"], K_sz, B)

    denoised_flat, _ = agent.forward_policy(
        tasks_rep, obs_rep, emb_rep,
        x_t=x_t_flat, sigmas=sigmas_flat,
        rng=denoise_rng, train=False,
    )  # (B*K, D)

    denoised = denoised_flat.reshape(B, K_sz, D)
    recon_err = jnp.sum((denoised - a_k) ** 2, axis=-1)  # (B, K)

    # proxy log-prob (shift by max for numerical stability of exp)
    h_k = -recon_err                                       # (B, K)
    h_k = jax.lax.stop_gradient(h_k)

    # ── Q 值 ──
    rng, critic_rng = jax.random.split(rng)
    a_k_flat = a_k.reshape(B * K_sz, D)
    q_preds = agent.forward_critic(
        obs_rep, emb_rep, a_k_flat, rng=critic_rng,
    )  # (ensemble, B*K)
    min_q = jnp.min(q_preds, axis=0).reshape(B, K_sz)     # (B, K)

    # ── advantage（无温度项） ──
    V_s = jnp.mean(min_q, axis=1, keepdims=True)          # (B, 1)
    A_k = min_q - V_s                                      # (B, K)

    # numerically stable exp(h_k)
    h_max = jnp.max(h_k, axis=1, keepdims=True)
    pi_proxy = jnp.clip(jnp.exp(h_k - h_max), a_max=1e6)  # (B, K)
    y_k = pi_proxy * A_k                                    # (B, K)

    # ── 协方差 ──
    c = -_covariance(h_k, y_k, axis=1)                     # (B,)
    c = jax.lax.stop_gradient(c)
    c_abs = jnp.abs(c)

    # ── 分位数截断 → 掩码 ──
    l_bound = jnp.percentile(c_abs, q_low * 100.0)
    u_bound = jnp.percentile(c_abs, q_high * 100.0)
    I_mask = ((c_abs >= l_bound) & (c_abs <= u_bound)).astype(jnp.float32)
    keep = jnp.sum(I_mask)
    keep_ratio = jax.lax.stop_gradient(keep / (batch_size + eps))

    # ── label 统计（如果 batch 含 labels 字段） ──
    label_stats = {}
    if "labels" in batch:
        label_stats = compute_label_cov_stats(c, batch["labels"], eps=eps)

    cov_info = {
        "cov_batch_mean": jax.lax.stop_gradient(jnp.mean(c)),
        "c_abs_mean": jax.lax.stop_gradient(jnp.mean(c_abs)),
        "c_abs_l_bound": jax.lax.stop_gradient(l_bound),
        "c_abs_u_bound": jax.lax.stop_gradient(u_bound),
        "mask_keep_ratio": keep_ratio,
        "proxy_h_mean": jax.lax.stop_gradient(jnp.mean(h_k)),
        "proxy_h_std": jax.lax.stop_gradient(jnp.std(h_k)),
        "sampled_q_mean": jax.lax.stop_gradient(jnp.mean(min_q)),
        **{f"cov/{k}": jax.lax.stop_gradient(v) for k, v in label_stats.items()},
    }

    return I_mask, cov_info


# ──────────────────────────────────────────────────────────────────────────────
# 核心: Diffusion-adapted Cov Actor Loss
# ──────────────────────────────────────────────────────────────────────────────

def cov_actor_loss_diffusion_fn(
    agent,
    batch: Dict[str, Any],
    params: Any,
    rng: jax.Array,
    *,
    K: int = 4,
    q_low: float = 0.05,
    q_high: float = 0.90,
    sigma_eval: float = 0.02,
    eps: float = 1e-8,
) -> Tuple[jax.Array, Dict[str, Any]]:
    """Diffusion-adapted Cov-bounded Actor Loss（JAX 纯函数版）。

    流程:
      1. 采 K 个 action (stop_gradient)
      2. 用去噪 loss 代理 proxy log-prob
      3. 估算 per-state c(s) = -Cov(ĥ, exp(ĥ)·Â)
      4. 分位数截断 → 掩码 I(s)
      5. 计算 per-sample ConRFT actor loss (recon + Q)
      6. 返回 masked loss = Σ(I·L) / (Σ I + ε)

    签名与 ConRFT agent.policy_loss_fn 完全一致:
        (batch, params, rng) -> (loss, info)
    """
    config = agent.config
    batch_size = batch["rewards"].shape[0]

    # ────────────────────────── 协方差掩码（无梯度） ──────────────────────────
    rng, cov_rng = jax.random.split(rng)
    I_mask, cov_info = _compute_cov_mask_diffusion(
        agent, batch, cov_rng,
        K=K, sigma_eval=sigma_eval,
        q_low=q_low, q_high=q_high, eps=eps,
    )
    keep = jnp.sum(I_mask)

    # ──────────────── per-sample ConRFT loss（带梯度，通过 params） ────────────
    rng, noise_rng, indice_rng, policy_rng1, policy_rng2, critic_rng = (
        jax.random.split(rng, 6)
    )

    # 1) 从纯噪声一步去噪生成 clean actions（带 grad_params）
    new_actions, _action_emb = agent.forward_policy(
        batch["tasks"], batch["observations"], batch["embeddings"],
        rng=policy_rng1, grad_params=params,
    )

    # 2) Consistency / Distillation loss（per-sample）
    actions = (
        batch["actions"][..., :-1] if config["fix_gripper"] else batch["actions"]
    )
    x_start = actions
    noise = jax.random.normal(noise_rng, shape=x_start.shape, dtype=x_start.dtype)
    dims = x_start.ndim

    indices = jax.random.randint(
        indice_rng, (batch_size,), 0, config["num_scales"] - 1
    )
    t = (
        config["sigma_max"] ** (1 / config["rho"])
        + indices / (config["num_scales"] - 1)
        * (
            config["sigma_min"] ** (1 / config["rho"])
            - config["sigma_max"] ** (1 / config["rho"])
        )
    )
    t = t ** config["rho"]

    x_t = x_start + noise * append_dims(t, dims)

    distiller, _ = agent.forward_policy(
        batch["tasks"], batch["observations"], batch["embeddings"],
        x_t, t,
        rng=policy_rng2, grad_params=params,
    )

    snrs = get_snr(t)
    weights = get_weightings("karras", snrs, config["sigma_data"])

    recon_diffs = (distiller - x_start) ** 2
    recon_per_sample = mean_flat(recon_diffs) * weights             # (B,)

    # 3) Q loss（per-sample，负 Q 值 → 最大化 Q）
    q_new_actions = agent.forward_critic(
        batch["observations"], batch["embeddings"],
        new_actions, rng=critic_rng,
    )                                                                # (ensemble, B)
    q_per_sample = -q_new_actions.mean(axis=0)                      # (B,)

    # 4) per-sample 总 loss
    loss_per_sample = (
        agent.state.bc_weight * recon_per_sample
        + agent.state.q_weight * q_per_sample
    )                                                                # (B,)

    # ──────────────────────────── 应用掩码 ────────────────────────────────────
    actor_loss = (I_mask * loss_per_sample).sum() / (keep + eps)

    # ──────────────────────────── 策略监控指标 ────────────────────────────────
    action_std_per_dim = jnp.std(new_actions, axis=0)
    action_mean_std = jnp.mean(action_std_per_dim)
    action_variance = jnp.var(new_actions, axis=0)
    estimated_entropy = jnp.sum(
        0.5 * jnp.log(2 * jnp.pi * jnp.e * (action_variance + 1e-8))
    )

    info = {
        "actor_loss": actor_loss,
        "q_weight": agent.state.q_weight,
        "bc_weight": agent.state.bc_weight,
        "q_loss": q_new_actions.mean(),
        "bc_loss": recon_per_sample.mean(),
        "mse": jnp.mean(jnp.sum((new_actions - actions) ** 2, axis=-1)),
        "policy_entropy": estimated_entropy,
        "action_std": action_mean_std,
        # ── cov 掩码诊断指标 ──
        **cov_info,
    }

    return actor_loss, info


# ──────────────────────────────────────────────────────────────────────────────
# 工厂: 生成可替换 policy_loss_fn 的闭包
# ──────────────────────────────────────────────────────────────────────────────

def make_cov_policy_loss_fn_diffusion(
    agent: Any,
    *,
    K: int = 4,
    q_low: float = 0.05,
    q_high: float = 0.90,
    sigma_eval: Optional[float] = None,
    eps: float = 1e-8,
):
    """构造与 ConRFT agent.policy_loss_fn 签名完全一致的闭包。

    返回的函数签名: (batch, params, rng) -> (loss, info)
    可直接赋值给 loss_fns 字典中的 "actor" 键。

    Args:
        agent: ConrftCPOctoAgentSingleArm 实例。
        K: 每个 obs 采样的 MC action 数（用于估算协方差）。
        q_low: |c| 下分位数阈值。
        q_high: |c| 上分位数阈值。
        sigma_eval: 用于评估 proxy log-prob 的噪声级别。
            默认使用 agent.config["sigma_min"]（最小噪声 → 最具区分力）。
        eps: 数值稳定常数。
    """
    if sigma_eval is None:
        sigma_eval = agent.config.get("sigma_min", 0.02)

    def _policy_loss_fn(batch, params, rng):
        return cov_actor_loss_diffusion_fn(
            agent=agent,
            batch=batch,
            params=params,
            rng=rng,
            K=K,
            q_low=q_low,
            q_high=q_high,
            sigma_eval=sigma_eval,
            eps=eps,
        )

    return _policy_loss_fn
