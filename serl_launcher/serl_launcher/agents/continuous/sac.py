from functools import partial
from typing import Iterable, Optional, Tuple, FrozenSet

import chex
import distrax
import flax
import flax.linen as nn
import jax
import jax.numpy as jnp

from serl_launcher.common.common import JaxRLTrainState, ModuleDict, nonpytree_field
from serl_launcher.common.encoding import EncodingWrapper
from serl_launcher.common.optimizers import make_optimizer
from serl_launcher.common.typing import Batch, Data, Params, PRNGKey
from serl_launcher.networks.actor_critic_nets import Critic, Policy, ensemblize
from serl_launcher.networks.lagrange import GeqLagrangeMultiplier
from serl_launcher.networks.mlp import MLP
from serl_launcher.utils.train_utils import _unpack


class SACAgent(flax.struct.PyTreeNode):
    """
    Online actor-critic supporting several different algorithms depending on configuration:
     - SAC (default)
     - TD3 (policy_kwargs={"std_parameterization": "fixed", "fixed_std": 0.1})
     - REDQ (critic_ensemble_size=10, critic_subsample_size=2)
     - SAC-ensemble (critic_ensemble_size>>1)
    """

    state: JaxRLTrainState
    config: dict = nonpytree_field()

    def forward_critic(
        self,
        observations: Data,
        actions: jax.Array,
        rng: PRNGKey,
        *,
        grad_params: Optional[Params] = None,
        train: bool = True,
    ) -> jax.Array:
        """
        Forward pass for critic network.
        Pass grad_params to use non-default parameters (e.g. for gradients).
        """
        if train:
            assert rng is not None, "Must specify rng when training"
        return self.state.apply_fn(
            {"params": grad_params or self.state.params},
            observations,
            actions,
            name="critic",
            rngs={"dropout": rng} if train else {},
            train=train,
        )

    def forward_target_critic(
        self,
        observations: Data,
        actions: jax.Array,
        rng: PRNGKey,
    ) -> jax.Array:
        """
        Forward pass for target critic network.
        Pass grad_params to use non-default parameters (e.g. for gradients).
        """
        return self.forward_critic(
            observations, actions, rng=rng, grad_params=self.state.target_params
        )

    @jax.jit
    def jitted_forward_target_critic(
        self,
        observations: Data,
        actions: jax.Array,
        rng: PRNGKey,
    ) -> jax.Array:
        """
        Forward pass for target critic network.
        Pass grad_params to use non-default parameters (e.g. for gradients).
        """
        return self.forward_critic(
            observations, actions, rng=rng, grad_params=self.state.target_params
        )

    def forward_policy(
        self,
        observations: Data,
        rng: Optional[PRNGKey] = None,
        *,
        grad_params: Optional[Params] = None,
        train: bool = True,
    ) -> distrax.Distribution:
        """
        Forward pass for policy network.
        Pass grad_params to use non-default parameters (e.g. for gradients).
        """
        if train:
            assert rng is not None, "Must specify rng when training"
        return self.state.apply_fn(
            {"params": grad_params or self.state.params},
            observations,
            name="actor",
            rngs={"dropout": rng} if train else {},
            train=train,
        )

    def forward_temperature(
        self, *, grad_params: Optional[Params] = None
    ) -> distrax.Distribution:
        """
        Forward pass for temperature Lagrange multiplier.
        Pass grad_params to use non-default parameters (e.g. for gradients).
        """
        return self.state.apply_fn(
            {"params": grad_params or self.state.params}, name="temperature"
        )

    def temperature_lagrange_penalty(
        self, entropy: jnp.ndarray, *, grad_params: Optional[Params] = None
    ) -> distrax.Distribution:
        """
        Forward pass for Lagrange penalty for temperature.
        Pass grad_params to use non-default parameters (e.g. for gradients).
        """
        return self.state.apply_fn(
            {"params": grad_params or self.state.params},
            lhs=entropy,
            rhs=self.config["target_entropy"],
            name="temperature",
        )

    def _compute_next_actions(self, batch, rng):
        """shared computation between loss functions"""
        batch_size = batch["rewards"].shape[0]

        next_action_distributions = self.forward_policy(
            batch["next_observations"], rng=rng
        )
        (
            next_actions,
            next_actions_log_probs,
        ) = next_action_distributions.sample_and_log_prob(seed=rng)
        
        actions = batch["actions"][..., :-1] if self.config["fix_gripper"] else batch["actions"]
        chex.assert_equal_shape([actions, next_actions])
        chex.assert_shape(next_actions_log_probs, (batch_size,))

        return next_actions, next_actions_log_probs

    def critic_loss_fn(self, batch, params: Params, rng: PRNGKey):
        """classes that inherit this class can change this function"""
        batch_size = batch["rewards"].shape[0]
        rng, next_action_sample_key = jax.random.split(rng)
        next_actions, next_actions_log_probs = self._compute_next_actions(
            batch, next_action_sample_key
        )

        # Evaluate next Qs for all ensemble members (cheap because we're only doing the forward pass)
        target_next_qs = self.forward_target_critic(
            batch["next_observations"],
            next_actions,
            rng=rng,
        )  # (critic_ensemble_size, batch_size)

        # Subsample if requested
        if self.config["critic_subsample_size"] is not None:
            rng, subsample_key = jax.random.split(rng)
            subsample_idcs = jax.random.randint(
                subsample_key,
                (self.config["critic_subsample_size"],),
                0,
                self.config["critic_ensemble_size"],
            )
            target_next_qs = target_next_qs[subsample_idcs]

        # Minimum Q across (subsampled) ensemble members
        target_next_min_q = target_next_qs.min(axis=0)
        chex.assert_shape(target_next_min_q, (batch_size,))

        target_q = (
            batch["rewards"]
            + self.config["discount"] * batch["masks"] * target_next_min_q
        )
        chex.assert_shape(target_q, (batch_size,))

        if self.config["backup_entropy"]:
            temperature = self.forward_temperature()
            target_q = target_q - temperature * next_actions_log_probs

        actions = batch["actions"][..., :-1] if self.config["fix_gripper"] else batch["actions"]
        predicted_qs = self.forward_critic(
            batch["observations"], actions, rng=rng, grad_params=params
        )

        chex.assert_shape(
            predicted_qs, (self.config["critic_ensemble_size"], batch_size)
        )
        target_qs = target_q[None].repeat(self.config["critic_ensemble_size"], axis=0)
        chex.assert_equal_shape([predicted_qs, target_qs])
        critic_loss = jnp.mean((predicted_qs - target_qs) ** 2)

        info = {
            "critic_loss": critic_loss,
            "predicted_qs": jnp.mean(predicted_qs),
            "target_qs": jnp.mean(target_qs),
            "rewards": batch["rewards"].mean(),
        }

        return critic_loss, info

    def policy_loss_fn(self, batch, params: Params, rng: PRNGKey):
        batch_size = batch["rewards"].shape[0]
        temperature = self.forward_temperature()

        rng, policy_rng, sample_rng, critic_rng = jax.random.split(rng, 4)
        action_distributions = self.forward_policy(
            batch["observations"], rng=policy_rng, grad_params=params
        )
        actions, log_probs = action_distributions.sample_and_log_prob(seed=sample_rng)

        predicted_qs = self.forward_critic(
            batch["observations"],
            actions,
            rng=critic_rng,
        )
        predicted_q = predicted_qs.mean(axis=0)
        chex.assert_shape(predicted_q, (batch_size,))
        chex.assert_shape(log_probs, (batch_size,))

        actor_objective = predicted_q - temperature * log_probs
        actor_loss = -jnp.mean(actor_objective)

        info = {
            "actor_loss": actor_loss,
            "temperature": temperature,
            "entropy": -log_probs.mean(),
        }

        return actor_loss, info

    def temperature_loss_fn(self, batch, params: Params, rng: PRNGKey):
        rng, next_action_sample_key = jax.random.split(rng)
        next_actions, next_actions_log_probs = self._compute_next_actions(
            batch, next_action_sample_key
        )

        entropy = -next_actions_log_probs.mean()
        temperature_loss = self.temperature_lagrange_penalty(
            entropy,
            grad_params=params,
        )
        return temperature_loss, {"temperature_loss": temperature_loss}

    # =========================================================================
    # [HIL-SERL Module 2] 偏好感知 Q 值修正 Critic 损失
    # =========================================================================

    def corrected_critic_loss_fn(
        self,
        batch,
        A_cf: jnp.ndarray,
        params: Params,
        rng: PRNGKey,
    ):
        """
        Module 2: 偏好感知 Q 值修正。

        修正后 Bellman 目标：
            ỹ_t = y_t - α(t) · A_cf(t)

        - α(t) = batch["alpha_weight"]，次优片段内 exp(-λ*(t_i-t))，其余 0
        - A_cf : (batch_size,) 逐样本修正量，由 update_with_correction 精确对齐后传入；
                 无对应偏好样本的位置为 0.0（不修正）

        兼容 fix_gripper=True（动作截断与 critic_loss_fn 保持一致）。
        """
        batch_size = batch["rewards"].shape[0]
        rng, next_action_sample_key = jax.random.split(rng)
        next_actions, next_actions_log_probs = self._compute_next_actions(
            batch, next_action_sample_key
        )

        # 标准 Bellman 目标
        target_next_qs = self.forward_target_critic(
            batch["next_observations"], next_actions, rng=rng,
        )
        if self.config["critic_subsample_size"] is not None:
            rng, subsample_key = jax.random.split(rng)
            subsample_idcs = jax.random.randint(
                subsample_key,
                (self.config["critic_subsample_size"],),
                0,
                self.config["critic_ensemble_size"],
            )
            target_next_qs = target_next_qs[subsample_idcs]

        target_next_min_q = target_next_qs.min(axis=0)
        chex.assert_shape(target_next_min_q, (batch_size,))

        target_q = (
            batch["rewards"]
            + self.config["discount"] * batch["masks"] * target_next_min_q
        )
        if self.config["backup_entropy"]:
            temperature = self.forward_temperature()
            target_q = target_q - temperature * next_actions_log_probs

        # Module 2 核心：逐样本位置感知衰减修正
        # ỹ_t = y_t - α(t) · A_cf(t)
        #   α(t)    : (B,) 次优片段内 = exp(-λ*(t_i-t))，其余 = 0
        #   A_cf(t) : (B,) 逐样本修正量；来自与该条 transition 同 segment_id 的偏好样本，
        #             无对应样本时为 0.0（不修正）
        alpha_weights_raw = batch["alpha_weight"]       # (B,) 原始衰减值
        chex.assert_shape(alpha_weights_raw, (batch_size,))
        chex.assert_shape(A_cf, (batch_size,))           # 调用方保证逐样本对齐

        # flat_alpha_correction=True：次优片段内 α 统一置 1（不做位置衰减），
        #   片段外 α=0 保持不变。效果是对整条次优片段施加等强度修正。
        # flat_alpha_correction=False（默认）：保留 exp(-λ*(t_i-t)) 位置衰减，
        #   t_i 处修正最强，t_a 处最弱。
        if self.config.get("flat_alpha_correction", False):
            alpha_weights = jnp.where(alpha_weights_raw > 0.0, 1.0, 0.0)
        else:
            alpha_weights = alpha_weights_raw

        corrected_target_q = target_q - alpha_weights * A_cf
        chex.assert_shape(corrected_target_q, (batch_size,))

        # fix_gripper 兼容（与 critic_loss_fn 一致）
        actions = batch["actions"][..., :-1] if self.config["fix_gripper"] else batch["actions"]
        predicted_qs = self.forward_critic(
            batch["observations"], actions, rng=rng, grad_params=params
        )
        chex.assert_shape(predicted_qs, (self.config["critic_ensemble_size"], batch_size))
        target_qs = corrected_target_q[None].repeat(self.config["critic_ensemble_size"], axis=0)
        chex.assert_equal_shape([predicted_qs, target_qs])
        critic_loss = jnp.mean((predicted_qs - target_qs) ** 2)

        # ── 监控指标 ─────────────────────────────────────────────────────────
        # [指标1] correction_hit_rate：α>0 且 A_cf>0 同时成立的 transition 占比
        #   健康值：初期可能低，随训练进行应 > 20%；若持续 <5% 说明匹配率太低
        correction_active = (A_cf > 0.0) & (alpha_weights > 0.0)
        correction_hit_rate = jnp.mean(correction_active.astype(jnp.float32))

        # [指标2] correction_magnitude：α(t)·A_cf(t) 的均值，即实际作用到 Q 目标的修正量
        #   健康值：应 > 0；量级应与 reward 的绝对值相当（如 reward 在 [-0.05, 1] 则修正量在 0.01~0.5）
        correction = alpha_weights * A_cf
        correction_magnitude = jnp.mean(correction)

        # [指标3] correction_ratio：修正量 / |未修正Q目标| 比值
        #   健康值：0.05~0.5；太小说明修正微弱；太大说明修正过猛（可调 alpha_lambda）
        correction_ratio = correction_magnitude / (jnp.mean(jnp.abs(target_q)) + 1e-8)

        # [指标4] A_cf_where_active：仅在有效修正位置上 A_cf 的均值（排除 0 的干扰）
        #   健康值：应 > 0；若此值 > 0 但 correction_hit_rate 仍低，说明问题在匹配率而非 Q 差距
        A_cf_where_active = jnp.where(correction_active, A_cf, 0.0).sum() / (
            correction_active.astype(jnp.float32).sum() + 1e-8
        )

        info = {
            "corrected_critic_loss": critic_loss,
            "predicted_qs": jnp.mean(predicted_qs),
            "corrected_target_qs": jnp.mean(corrected_target_q),
            "uncorrected_target_qs": jnp.mean(target_q),
            # ── 修正效果核心指标 ──
            "correction_hit_rate": correction_hit_rate,     # 有效修正占比 ↑ 越好
            "correction_magnitude": correction_magnitude,   # 修正量均值   ↑ 表示修正在起作用
            "correction_ratio": correction_ratio,           # 修正量/Q值比  健康值 0.05~0.5
            "A_cf_where_active": A_cf_where_active,         # 有效位置的 A_cf 均值
            # ── 辅助诊断 ──
            "mean_alpha_weight": jnp.mean(alpha_weights),
            "alpha_nonzero_ratio": jnp.mean(alpha_weights > 0.0),
            "rewards": batch["rewards"].mean(),
        }
        return critic_loss, info

    # =========================================================================
    # [HIL-SERL Module 3] 偏好引导策略学习损失
    # =========================================================================

    def contrastive_policy_loss_fn(
        self,
        batch,
        preference_batch,
        params: Params,
        rng: PRNGKey,
    ):
        """
        Module 3: ORPO 风格对比 Actor 损失。

        L_actor = L_RLPD + β · L_contrast
    L_contrast = -E[log σ(log π(a^h|s) - log π(a^π_cur|s))]

        兼容 fix_gripper=True（preference_batch 中的动作截断）。
        """
        batch_size = batch["rewards"].shape[0]

        # 标准 SAC/RLPD Actor 损失
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

        # Module 3 核心：对比项
        rng, pref_rng = jax.random.split(rng)
        pref_distributions = self.forward_policy(
            preference_batch["observations"], rng=pref_rng, grad_params=params
        )

        # 动作维度自动对齐：若 preference 动作比策略输出多（常见于带 gripper 的记录），
        # 自动按策略输出维度截断，避免维度不一致。
        policy_dim = pref_distributions.distribution.loc.shape[-1]
        human_a = preference_batch["human_actions"][..., :policy_dim]

        # ✅ 修复：使用 buffer 中记录的历史策略动作作为"被拒绝动作"，
        # 而非每步重采样——重采样会导致梯度方向每 step 随机变化，引起 log_prob 震荡。
        # 历史动作固定（干预时刻实际执行的 a^π），梯度方向稳定。
        policy_a_hist = preference_batch["policy_actions"][..., :policy_dim]
        policy_a_hist = jax.lax.stop_gradient(policy_a_hist)

        # 数值稳定性：Tanh-squashed 分布在 |a|=1 处 log_prob 可能出现极端值。
        # 在计算 log_prob 前做轻量裁剪，避免 atanh(±1) 触发数值爆炸。
        clip_eps = self.config.get("contrastive_action_clip_eps", 1e-6)
        human_a       = jnp.clip(human_a,       -1.0 + clip_eps, 1.0 - clip_eps)
        policy_a_hist = jnp.clip(policy_a_hist, -1.0 + clip_eps, 1.0 - clip_eps)

        pref_batch_size = preference_batch["human_actions"].shape[0]
        # ✅ 除以 policy_dim 归一化：防止高维动作空间下 log_prob 绝对值过大导致 sigmoid 饱和
        log_prob_human  = pref_distributions.log_prob(human_a)        / policy_dim  # (B,)
        log_prob_policy = pref_distributions.log_prob(policy_a_hist)  / policy_dim  # (B,)
        chex.assert_shape(log_prob_human,  (pref_batch_size,))
        chex.assert_shape(log_prob_policy, (pref_batch_size,))

        # Module 3 核心：BTL / logistic pairwise loss
        # preference_batch_direct 是独立采样的纯净 batch，每条均有效，直接取均值。
        log_prob_gap = log_prob_human - log_prob_policy                  # (B,)
        # ✅ 硬裁剪：防止 gap 极端值导致 log_sigmoid 数值溢出
        log_prob_gap = jnp.clip(log_prob_gap, -10.0, 10.0)
        contrastive_loss_per_sample = -jax.nn.log_sigmoid(log_prob_gap)  # (B,)
        contrastive_loss = jnp.mean(contrastive_loss_per_sample)

        contrastive_coef = self.config.get("contrastive_coef", 0.1)
        total_actor_loss = rlpd_loss + contrastive_coef * contrastive_loss
        pref_action_l2_mean = jnp.mean(
            jnp.linalg.norm(human_a - policy_a_hist, axis=-1)
        )

        info = {
            "actor_loss": total_actor_loss,
            "rlpd_actor_loss": rlpd_loss,
            "contrastive_loss": contrastive_loss,
            "temperature": temperature,
            "entropy": -log_probs.mean(),
            "log_prob_human":  jnp.mean(log_prob_human),
            "log_prob_policy": jnp.mean(log_prob_policy),
            "log_prob_gap":    jnp.mean(log_prob_gap),
            "pref_action_l2_mean": pref_action_l2_mean,
        }
        return total_actor_loss, info

    # =========================================================================
    # [HIL-SERL] 联合更新：Module 2 + Module 3
    # =========================================================================

    @partial(jax.jit, static_argnames=("pmap_axis", "networks_to_update"))
    def update_with_correction(
        self,
        batch,
        matched_pref,               # segment_id 对齐（仅用于 Module 2 A_cf 计算）
        preference_batch_direct,    # 独立采样的纯净 batch（用于 Module 3 对比损失）
        *,
        pmap_axis=None,
        networks_to_update=frozenset({"actor", "critic", "temperature"}),
    ):
        """
        同时应用 Module 2（Q 值修正）和 Module 3（对比策略学习）。

        1. target network 推断 matched_pref → 计算 A_cf（stop-gradient），Module 2
        2. Critic 用 corrected_critic_loss_fn（注入 A_cf 和 alpha_weight）
        3. Actor  用 contrastive_policy_loss_fn（注入 preference_batch_direct）
           - preference_batch_direct=None 时回退标准 RLPD actor loss（Critic-only 更新）
        4. Temperature 同标准 SAC

        兼容 fix_gripper=True（preference 动作在各 loss fn 内部截断）。
        """
        batch_size = batch["rewards"].shape[0]
        chex.assert_tree_shape_prefix(batch, (batch_size,))

        if self.config["image_keys"][0] not in batch["next_observations"]:
            batch = _unpack(batch)

        rng, aug_rng = jax.random.split(self.state.rng)
        if (
            "augmentation_function" in self.config.keys()
            and self.config["augmentation_function"] is not None
        ):
            batch = self.config["augmentation_function"](batch, aug_rng)

        batch = batch.copy(
            add_or_replace={"rewards": batch["rewards"] + self.config["reward_bias"]}
        )

        # ── Step 1: 逐样本计算 A_cf（精确对齐版）─────────────────────────────
        # preference_batch 已由 learner 端通过 get_by_segment_ids 对齐到 batch，
        # 即 preference_batch[i] 就是 batch[i] 对应干预事件的偏好数据。
        # valid_mask[i] = True 表示 batch[i] 在 preference buffer 中有匹配。
        #
        # 对比旧方案（随机 batch + segment 集合匹配）的优点：
        #   - 无随机性：同一条 transition 每次训练都用相同的 (s_{t_i}, a^h)
        #   - 无稀疏匹配问题：不再依赖两个随机 batch 碰巧采到同一 segment_id
        #   - 显存零增量：preference buffer 本身已有 obs，不需要扩大 replay buffer
        rng, cf_rng = jax.random.split(rng)

        # A_cf 用干预时刻记录的历史策略动作 a^π_old（非当前策略重采样），
        # 这样即使 Module 3 已让当前策略拟合人类动作，A_cf 仍能反映干预时的劣势。
        # fix_gripper 兼容：截掉最后一维（夹爪）
        if self.config["fix_gripper"]:
            h_actions = matched_pref["human_actions"][..., :-1]
            p_actions = matched_pref["policy_actions"][..., :-1]
        else:
            h_actions = matched_pref["human_actions"]
            p_actions = matched_pref["policy_actions"]

        # target critic 评估两种动作的 Q 值
        q_human_ensemble = self.forward_target_critic(
            matched_pref["observations"], h_actions, rng=cf_rng,
        )  # (ensemble_size, B)
        q_human_min = q_human_ensemble.min(axis=0)    # (B,)

        q_policy_ensemble = self.forward_target_critic(
            matched_pref["observations"], p_actions, rng=cf_rng,
        )  # (ensemble_size, B)
        q_policy_min = q_policy_ensemble.min(axis=0)  # (B,)

        # 逐样本 A_cf：仅在 valid_mask=True 且 Q_human > Q_policy 时才有修正量
        # stop_gradient：A_cf 是外部信号，不允许梯度通过它传回 Q 网络
        valid_mask = matched_pref["valid_mask"].astype(jnp.float32)  # (B,)
        q_gap_raw = q_human_min - q_policy_min                           # (B,)
        A_cf_per_sample = jax.lax.stop_gradient(
            valid_mask * jnp.maximum(0.0, q_gap_raw)                     # (B,)
        )

        # ── Step 2: 构造修正后的 loss_fns ─────────────────────────────────
        def _corrected_critic(params, rng):
            return self.corrected_critic_loss_fn(batch, A_cf_per_sample, params, rng)

        def _contrastive_actor(params, rng):
            if preference_batch_direct is None:
                # Critic-only 更新时传 None，回退标准 RLPD actor loss
                return self.policy_loss_fn(batch, params, rng)
            return self.contrastive_policy_loss_fn(batch, preference_batch_direct, params, rng)

        loss_fns = {
            "critic":      _corrected_critic,
            "actor":       _contrastive_actor,
            "temperature": partial(self.temperature_loss_fn, batch),
        }

        assert networks_to_update.issubset(loss_fns.keys()), (
            f"Invalid gradient steps: {networks_to_update}"
        )
        for key in loss_fns.keys() - networks_to_update:
            loss_fns[key] = lambda params, rng: (0.0, {})

        new_state, info = self.state.apply_loss_fns(
            loss_fns, pmap_axis=pmap_axis, has_aux=True
        )

        if "critic" in networks_to_update:
            new_state = new_state.target_update(self.config["soft_target_update_rate"])

        new_state = new_state.replace(rng=rng)

        for name, opt_state in new_state.opt_states.items():
            if (
                hasattr(opt_state, "hyperparams")
                and "learning_rate" in opt_state.hyperparams.keys()
            ):
                info[f"{name}_lr"] = opt_state.hyperparams["learning_rate"]

        # ── 全局监控指标（WandB 诊断修正是否在发挥作用）────────────────────
        # [指标5] pref_match_rate：replay batch 中命中偏好样本的比例
        #   健康值：初期可低，随 preference buffer 填充应逐渐上升；
        #           若持续 <10% 可增大 preference buffer 容量或增大 suboptimal_window
        pref_match_rate = jnp.mean(valid_mask)

        # [指标6] q_gap_where_valid：有效匹配位置的 Q_human - Q_policy 均值
        #   健康值：应 > 0，说明人类动作确实被 Q 值评估为更好；
        #           若接近 0 说明 Q 网络还未收敛到能区分好坏动作（正常，训练初期如此）
        q_gap_where_valid = jnp.sum(q_gap_raw * valid_mask) / (jnp.sum(valid_mask) + 1e-8)

        info["pref_match_rate"] = pref_match_rate
        info["q_human_mean"] = jnp.sum(q_human_min * valid_mask) / (jnp.sum(valid_mask) + 1e-8)
        info["q_policy_mean"] = jnp.sum(q_policy_min * valid_mask) / (jnp.sum(valid_mask) + 1e-8)
        info["q_gap_where_valid"] = q_gap_where_valid
        info["A_cf_mean"] = jnp.mean(A_cf_per_sample)

        return self.replace(state=new_state), info

    def loss_fns(self, batch):
        # 如果 config 中启用了 cov_actor_loss，使用协方差熵截断 Actor Loss
        # 工厂函数由 train_rlpd.py 注入至 config["_cov_fn_factory"]，避免在包内 import examples 目录
        if self.config.get("use_cov_actor_loss", False):
            _factory = self.config["_cov_fn_factory"]
            _cov_fn = _factory(
                self,
                K=self.config.get("cov_K", 8),
                q_low=self.config.get("cov_q_low", 0.05),
                q_high=self.config.get("cov_q_high", 0.90),
            )
            actor_loss_fn = partial(_cov_fn, batch)
        else:
            actor_loss_fn = partial(self.policy_loss_fn, batch)

        return {
            "critic": partial(self.critic_loss_fn, batch),
            "actor": actor_loss_fn,
            "temperature": partial(self.temperature_loss_fn, batch),
        }

    @partial(jax.jit, static_argnames=("pmap_axis", "networks_to_update"))
    def update(
        self,
        batch: Batch,
        *,
        pmap_axis: Optional[str] = None,
        networks_to_update: FrozenSet[str] = frozenset(
            {"actor", "critic", "temperature"}
        ),
        **kwargs
    ) -> Tuple["SACAgent", dict]:
        """
        Take one gradient step on all (or a subset) of the networks in the agent.

        Parameters:
            batch: Batch of data to use for the update. Should have keys:
                "observations", "actions", "next_observations", "rewards", "masks".
            pmap_axis: Axis to use for pmap (if None, no pmap is used).
            networks_to_update: Names of networks to update (default: all networks).
                For example, in high-UTD settings it's common to update the critic
                many times and only update the actor (and other networks) once.
        Returns:
            Tuple of (new agent, info dict).
        """
        batch_size = batch["rewards"].shape[0]
        chex.assert_tree_shape_prefix(batch, (batch_size,))

        if self.config["image_keys"][0] not in batch["next_observations"]:
            batch = _unpack(batch)
        rng, aug_rng = jax.random.split(self.state.rng)
        if "augmentation_function" in self.config.keys() and self.config["augmentation_function"] is not None:
            batch = self.config["augmentation_function"](batch, aug_rng)

        batch = batch.copy(
            add_or_replace={"rewards": batch["rewards"] + self.config["reward_bias"]}
        )

        # Compute gradients and update params
        loss_fns = self.loss_fns(batch, **kwargs)

        # Only compute gradients for specified steps
        assert networks_to_update.issubset(
            loss_fns.keys()
        ), f"Invalid gradient steps: {networks_to_update}"
        for key in loss_fns.keys() - networks_to_update:
            loss_fns[key] = lambda params, rng: (0.0, {})

        new_state, info = self.state.apply_loss_fns(
            loss_fns, pmap_axis=pmap_axis, has_aux=True
        )

        # Update target network (if requested)
        if "critic" in networks_to_update:
            new_state = new_state.target_update(self.config["soft_target_update_rate"])

        # Update RNG
        new_state = new_state.replace(rng=rng)

        # Log learning rates
        for name, opt_state in new_state.opt_states.items():
            if (
                hasattr(opt_state, "hyperparams")
                and "learning_rate" in opt_state.hyperparams.keys()
            ):
                info[f"{name}_lr"] = opt_state.hyperparams["learning_rate"]

        return self.replace(state=new_state), info

    @partial(jax.jit, static_argnames=("argmax",))
    def sample_actions(
        self,
        observations: Data,
        *,
        seed: Optional[PRNGKey] = None,
        argmax: bool = False,
        **kwargs,
    ) -> jnp.ndarray:
        """
        Sample actions from the policy network, **using an external RNG** (or approximating the argmax by the mode).
        The internal RNG will not be updated.
        """

        dist = self.forward_policy(observations, rng=seed, train=False)
        if argmax:
            actions = dist.mode()
        else:
            actions = dist.sample(seed=seed)
        
        if self.config["fix_gripper"]: # add gripper action, default to 0
            actions = jnp.concatenate([actions, jnp.array([0])])
            
        return actions

    @classmethod
    def create(
        cls,
        rng: PRNGKey,
        observations: Data,
        actions: jnp.ndarray,
        # Models
        actor_def: nn.Module,
        critic_def: nn.Module,
        temperature_def: nn.Module,
        # Optimizer
        actor_optimizer_kwargs={
            "learning_rate": 3e-4,
        },
        critic_optimizer_kwargs={
            "learning_rate": 3e-4,
        },
        temperature_optimizer_kwargs={
            "learning_rate": 3e-4,
        },
        # Algorithm config
        discount: float = 0.95,
        soft_target_update_rate: float = 0.005,
        target_entropy: Optional[float] = None,
        entropy_per_dim: bool = False,
        backup_entropy: bool = False,
        critic_ensemble_size: int = 2,
        critic_subsample_size: Optional[int] = None,
        image_keys: Iterable[str] = None,
        augmentation_function: Optional[callable] = None,
        reward_bias: float = 0.0,
        fix_gripper: bool = False,
        **kwargs,
    ):
        networks = {
            "actor": actor_def,
            "critic": critic_def,
            "temperature": temperature_def,
        }

        model_def = ModuleDict(networks)

        # Define optimizers
        txs = {
            "actor": make_optimizer(**actor_optimizer_kwargs),
            "critic": make_optimizer(**critic_optimizer_kwargs),
            "temperature": make_optimizer(**temperature_optimizer_kwargs),
        }

        rng, init_rng = jax.random.split(rng)
        params = model_def.init(
            init_rng,
            actor=[observations],
            critic=[observations, actions[:-1] if fix_gripper else actions],
            temperature=[],
        )["params"]

        rng, create_rng = jax.random.split(rng)
        state = JaxRLTrainState.create(
            apply_fn=model_def.apply,
            params=params,
            txs=txs,
            target_params=params,
            rng=create_rng,
        )

        # Config
        assert not entropy_per_dim, "Not implemented"
        action_dim = actions.shape[-1] - 1 if fix_gripper else actions.shape[-1]
        if target_entropy is None:
            target_entropy = -action_dim / 2

        return cls(
            state=state,
            config=dict(
                critic_ensemble_size=critic_ensemble_size,
                critic_subsample_size=critic_subsample_size,
                discount=discount,
                soft_target_update_rate=soft_target_update_rate,
                target_entropy=target_entropy,
                backup_entropy=backup_entropy,
                image_keys=image_keys,
                reward_bias=reward_bias,
                augmentation_function=augmentation_function,
                fix_gripper=fix_gripper,
                **kwargs,
            ),
        )

    @classmethod
    def create_pixels(
        cls,
        rng: PRNGKey,
        observations: Data,
        actions: jnp.ndarray,
        # Model architecture
        encoder_type: str = "resnet-pretrained",
        use_proprio: bool = False,
        critic_network_kwargs: dict = {
            "hidden_dims": [256, 256],
        },
        policy_network_kwargs: dict = {
            "hidden_dims": [256, 256],
        },
        policy_kwargs: dict = {
            "tanh_squash_distribution": True,
            "std_parameterization": "uniform",
        },
        critic_ensemble_size: int = 2,
        critic_subsample_size: Optional[int] = None,
        temperature_init: float = 1.0,
        image_keys: Iterable[str] = ("image",),
        augmentation_function: Optional[callable] = None,
        fix_gripper: bool = False,
        **kwargs,
    ):
        """
        Create a new pixel-based agent, with no encoders.
        """

        policy_network_kwargs["activate_final"] = True
        critic_network_kwargs["activate_final"] = True

        if encoder_type == "resnet":
            from serl_launcher.vision.resnet_v1 import resnetv1_configs

            encoders = {
                image_key: resnetv1_configs["resnetv1-10"](
                    pooling_method="spatial_learned_embeddings",
                    num_spatial_blocks=8,
                    bottleneck_dim=256,
                    name=f"encoder_{image_key}",
                )
                for image_key in image_keys
            }
        elif encoder_type == "resnet-pretrained":
            from serl_launcher.vision.resnet_v1 import (
                PreTrainedResNetEncoder,
                resnetv1_configs,
            )

            pretrained_encoder = resnetv1_configs["resnetv1-10-frozen"](
                pre_pooling=True,
                name="pretrained_encoder",
            )
            encoders = {
                image_key: PreTrainedResNetEncoder(
                    pooling_method="spatial_learned_embeddings",
                    num_spatial_blocks=8,
                    bottleneck_dim=256,
                    pretrained_encoder=pretrained_encoder,
                    name=f"encoder_{image_key}",
                )
                for image_key in image_keys
            }
        else:
            raise NotImplementedError(f"Unknown encoder type: {encoder_type}")

        encoder_def = EncodingWrapper(
            encoder=encoders,
            use_proprio=use_proprio,
            enable_stacking=True,
            image_keys=image_keys,
        )

        encoders = {
            "critic": encoder_def,
            "actor": encoder_def,
        }

        # Define networks
        critic_backbone = partial(MLP, **critic_network_kwargs)
        critic_backbone = ensemblize(critic_backbone, critic_ensemble_size)(
            name="critic_ensemble"
        )
        critic_def = partial(
            Critic, encoder=encoders["critic"], network=critic_backbone
        )(name="critic")

        policy_def = Policy(
            encoder=encoders["actor"],
            network=MLP(**policy_network_kwargs),
            action_dim=actions.shape[-1] - 1 if fix_gripper else actions.shape[-1],
            **policy_kwargs,
            name="actor",
        )

        temperature_def = GeqLagrangeMultiplier(
            init_value=temperature_init,
            constraint_shape=(),
            constraint_type="geq",
            name="temperature",
        )

        agent = cls.create(
            rng,
            observations,
            actions,
            actor_def=policy_def,
            critic_def=critic_def,
            temperature_def=temperature_def,
            critic_ensemble_size=critic_ensemble_size,
            critic_subsample_size=critic_subsample_size,
            image_keys=image_keys,
            augmentation_function=augmentation_function,
            fix_gripper=fix_gripper,
            **kwargs,
        )

        if "pretrained" in encoder_type:  # load pretrained weights for ResNet-10
            from serl_launcher.utils.train_utils import load_resnet10_params
            agent = load_resnet10_params(agent, image_keys)

        return agent
