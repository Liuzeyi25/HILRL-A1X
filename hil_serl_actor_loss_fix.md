# HIL-SERL Actor Loss Fix — Code Modification Plan

## 修改思路

当前对比损失使用 `log π(a^h|s) - log π(a^π_old|s)` 作为偏好信号，其梯度含有
`1/σ²` 项，导致 entropy 下降时梯度爆炸，策略震荡崩溃。

根本原因：我们真正想优化的目标是**策略的确定性意图 μ_θ(s) 靠近人类动作**，
这是一个纯粹关于 μ 的目标，σ 不应该出现在梯度里。

**解决方案**：用 pre-tanh 空间的 MSE（行为克隆损失）替换对比损失：

```
L_BC = ||μ_θ(s) - atanh(a^h)||²
```

梯度为 `2(μ_θ(s) - atanh(a^h))`，不含 σ，天然有界，与 entropy 完全解耦。

---

## 修改一：`sac.py` — 删除 `contrastive_policy_loss_fn`，新增 `bc_policy_loss_fn`

### 1a. 新增 `bc_policy_loss_fn` 方法

在 `contrastive_policy_loss_fn` 的位置，**整体替换**为以下方法：

```python
def bc_policy_loss_fn(
    self,
    batch,
    preference_batch,
    params: Params,
    rng: PRNGKey,
):
    """
    Module 3: Pre-tanh 空间行为克隆损失。

    L_actor = L_RLPD + β · L_BC
    L_BC    = E[||μ_θ(s) - atanh(a^h)||²]

    只约束策略均值（意图）靠近人类动作，完全不涉及 σ，
    与 SAC temperature 机制正交，无 entropy 耦合。
    """
    batch_size = batch["rewards"].shape[0]

    # ── 标准 RLPD Actor 损失 ──────────────────────────────────────────
    temperature = self.forward_temperature()
    rng, policy_rng, sample_rng, critic_rng = jax.random.split(rng, 4)

    action_distributions = self.forward_policy(
        batch["observations"], rng=policy_rng, grad_params=params
    )
    actions, log_probs = action_distributions.sample_and_log_prob(seed=sample_rng)
    predicted_qs = self.forward_critic(batch["observations"], actions, rng=critic_rng)
    predicted_q = predicted_qs.mean(axis=0)
    chex.assert_shape(predicted_q, (batch_size,))
    chex.assert_shape(log_probs, (batch_size,))

    rlpd_loss = -jnp.mean(predicted_q - temperature * log_probs)

    # ── Module 3 核心：Pre-tanh 空间 BC 损失 ─────────────────────────
    rng, pref_rng = jax.random.split(rng)
    pref_distributions = self.forward_policy(
        preference_batch["observations"], rng=pref_rng, grad_params=params
    )

    # 取策略均值（确定性意图），不采样，不涉及 σ
    mu_theta = pref_distributions.distribution.loc   # (B, D)，pre-tanh 均值

    policy_dim = mu_theta.shape[-1]
    human_a = preference_batch["human_actions"][..., :policy_dim]   # (B, D)

    # 人类动作映射到 pre-tanh 空间
    # clip_eps=1e-2 确保 atanh 有界：atanh(0.99) ≈ 2.65
    clip_eps = self.config.get("bc_action_clip_eps", 1e-2)
    human_a_clipped = jnp.clip(human_a, -1.0 + clip_eps, 1.0 - clip_eps)
    u_human = jnp.arctanh(human_a_clipped)   # (B, D)，∈ [-2.65, 2.65]

    # Per-sample MSE，在 pre-tanh 空间计算策略意图与人类动作的距离
    bc_loss_per_sample = jnp.mean((mu_theta - u_human) ** 2, axis=-1)  # (B,)
    bc_loss = jnp.mean(bc_loss_per_sample)                              # scalar

    pref_batch_size = preference_batch["human_actions"].shape[0]
    chex.assert_shape(bc_loss_per_sample, (pref_batch_size,))

    bc_coef = self.config.get("contrastive_coef", 0.1)   # 复用现有 flag
    total_actor_loss = rlpd_loss + bc_coef * bc_loss

    # 监控：策略均值与人类动作在 tanh 空间的 L2 距离（有物理意义的指标）
    mu_tanh = jnp.tanh(mu_theta)   # (B, D)，策略意图在动作空间的值
    action_distance = jnp.mean(jnp.linalg.norm(mu_tanh - human_a, axis=-1))

    info = {
        "actor_loss":       total_actor_loss,
        "rlpd_actor_loss":  rlpd_loss,
        "bc_loss":          bc_loss,
        "temperature":      temperature,
        "entropy":          -log_probs.mean(),
        "action_distance":  action_distance,   # ↓ 趋近 0 说明策略在学习人类动作
    }
    return total_actor_loss, info
```

### 1b. 删除旧的 `contrastive_policy_loss_fn`

将原有的 `contrastive_policy_loss_fn` 方法**整体删除**。

---

## 修改二：`sac.py` — `update_with_correction` 中替换调用

### 2a. 将 `_contrastive_actor` 改为 `_bc_actor`

```python
# ❌ 删除
def _contrastive_actor(params, rng):
    if preference_batch_direct is None:
        return self.policy_loss_fn(batch, params, rng)
    return self.contrastive_policy_loss_fn(batch, preference_batch_direct, params, rng)

# ✅ 替换
def _bc_actor(params, rng):
    if preference_batch_direct is None:
        return self.policy_loss_fn(batch, params, rng)
    return self.bc_policy_loss_fn(batch, preference_batch_direct, params, rng)
```

### 2b. loss_fns 中同步替换 key

```python
# ❌ 删除
loss_fns = {
    "critic":      _corrected_critic,
    "actor":       _contrastive_actor,
    "temperature": partial(self.temperature_loss_fn, batch),
}

# ✅ 替换
loss_fns = {
    "critic":      _corrected_critic,
    "actor":       _bc_actor,
    "temperature": partial(self.temperature_loss_fn, batch),
}
```

---

## 修改三：`train_rlpd_hil.py` — wandb 日志 key 同步

将 learner 循环中 `actor_info` 的日志前缀 key 更新，
确保 `bc_loss` 和 `action_distance` 被正确上报：

```python
# 在 learner() 的日志分组部分，actor_info 的匹配前缀更新：
actor_info = {f"actor/{k}": v for k, v in update_info.items()
              if k.startswith((
                  "actor_loss", "rlpd_actor", "bc_loss",       # ← bc_loss 替换 contrastive_loss
                  "entropy", "temperature",
                  "action_distance",                            # ← 新增
              ))}
```

同步在 `known_prefixes` 元组中：
```python
# ❌ 删除
"contrastive_loss", "log_prob_", "pref_action_l2",
"contrastive_human", "contrastive_policy",

# ✅ 替换
"bc_loss", "action_distance",
```

---

## 修改四：`sac_hybrid_single.py` / `sac_hybrid_dual.py` 同步

如果这两个文件中也有 `contrastive_policy_loss_fn` 的重写或调用，
执行与修改一、二完全相同的替换。

---

## 不需要修改的部分

以下内容**保持不变**，无需任何改动：

- `data_store.py` 的 `PreferenceBufferDataStore.sample()` 方法
- `train_rlpd_hil.py` 的 learner 独立采样逻辑（`pref_batch_direct`）
- `update_with_correction` 中 Module 2（Q 值修正）的全部逻辑
- `contrastive_coef` flag（复用作为 BC loss 的系数 β）

---

## 验证指标

| 指标 | 期望行为 |
|------|---------|
| `actor/entropy` | 平稳下降至 target_entropy 附近，无大幅震荡 |
| `actor/bc_loss` | 单调下降 |
| `actor/action_distance` | 持续下降趋近 0，说明策略意图在学习人类动作 |
| `actor/rlpd_actor_loss` | 正常下降，不受 bc_loss 干扰 |

**建议验证顺序**：
1. 先设 `contrastive_coef=0` 运行 2000 步，确认 RLPD 基线的 entropy 平稳
2. 开启 BC loss（`contrastive_coef=0.1`），观察 `action_distance` 是否持续下降
3. 若 `rlpd_actor_loss` 受影响，适当调小 `contrastive_coef`（0.05）
