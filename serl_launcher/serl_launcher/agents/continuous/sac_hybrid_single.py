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
from serl_launcher.networks.actor_critic_nets import Critic, Policy, GraspCritic, ensemblize
from serl_launcher.networks.lagrange import GeqLagrangeMultiplier
from serl_launcher.networks.mlp import MLP
from serl_launcher.utils.train_utils import _unpack


class SACAgentHybridSingleArm(flax.struct.PyTreeNode):
    """
    Online actor-critic supporting several different algorithms depending on configuration:
     - SAC (default)
     - TD3 (policy_kwargs={"std_parameterization": "fixed", "fixed_std": 0.1})
     - REDQ (critic_ensemble_size=10, critic_subsample_size=2)
     - SAC-ensemble (critic_ensemble_size>>1)
    
    Compared to SACAgent (in sac.py), this agent has a hybrid policy, with the gripper actions
    learned using DQN. Use this agent for single arm setups.
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
    
    def forward_grasp_critic(
        self,
        observations: Data,
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
            name="grasp_critic",
            rngs={"dropout": rng} if train else {},
            train=train,
        )

    def forward_target_grasp_critic(
        self,
        observations: Data, 
        rng: PRNGKey,
    ) -> jax.Array:
        """
        Forward pass for target critic network.
        Pass grad_params to use non-default parameters (e.g. for gradients).
        """
        return self.forward_grasp_critic(
            observations, rng=rng, grad_params=self.state.target_params
        )

    def forward_policy( # type: ignore              
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
        
        next_actions, next_actions_log_probs = next_action_distributions.sample_and_log_prob(seed=rng)
        chex.assert_shape(next_actions_log_probs, (batch_size,))

        return next_actions, next_actions_log_probs

    def critic_loss_fn(self, batch, params: Params, rng: PRNGKey):
        """classes that inherit this class can change this function"""
        batch_size = batch["rewards"].shape[0]
        # Extract continuous actions for critic
        actions = batch["actions"][..., :-1]

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
    

    def grasp_critic_loss_fn(self, batch, params: Params, rng: PRNGKey):
        """classes that inherit this class can change this function"""

        batch_size = batch["rewards"].shape[0]
        grasp_action = jnp.round(batch["actions"][..., -1]).astype(jnp.int16) + 1 # Cast env action from [-1, 1] to {0, 1, 2}

         # Evaluate next grasp Qs for all ensemble members (cheap because we're only doing the forward pass)
        target_next_grasp_qs = self.forward_target_grasp_critic(
            batch["next_observations"],
            rng=rng,
        )
        chex.assert_shape(target_next_grasp_qs, (batch_size, 3))

        # Select target next grasp Q based on the gripper action that maximizes the current grasp Q
        next_grasp_qs = self.forward_grasp_critic(
            batch["next_observations"],
            rng=rng,
        )
        # For DQN, select actions using online network, evaluate with target network
        best_next_grasp_action = next_grasp_qs.argmax(axis=-1) 
        chex.assert_shape(best_next_grasp_action, (batch_size,))
        
        target_next_grasp_q = target_next_grasp_qs[jnp.arange(batch_size), best_next_grasp_action]
        chex.assert_shape(target_next_grasp_q, (batch_size,))

        # Compute target Q-values
        grasp_rewards = batch["rewards"] + batch["grasp_penalty"]
        target_grasp_q = (
            grasp_rewards
            + self.config["discount"] * batch["masks"] * target_next_grasp_q
        )
        chex.assert_shape(target_grasp_q, (batch_size,))

        # Forward pass through the online grasp critic to get predicted Q-values
        predicted_grasp_qs = self.forward_grasp_critic(
            batch["observations"], 
            rng=rng, 
            grad_params=params
        )
        chex.assert_shape(predicted_grasp_qs, (batch_size, 3))
        
        # Select the predicted Q-values for the taken grasp actions in the batch
        predicted_grasp_q = predicted_grasp_qs[jnp.arange(batch_size), grasp_action]
        chex.assert_shape(predicted_grasp_q, (batch_size,))
        
        # Compute MSE loss between predicted and target Q-values
        chex.assert_equal_shape([predicted_grasp_q, target_grasp_q])
        grasp_critic_loss = jnp.mean((predicted_grasp_q - target_grasp_q) ** 2)

        info = {
            "grasp_critic_loss": grasp_critic_loss,
            "predicted_grasp_qs": jnp.mean(predicted_grasp_q),
            "target_grasp_qs": jnp.mean(target_grasp_q),
            "grasp_rewards": grasp_rewards.mean(),
        }

        return grasp_critic_loss, info


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
    # [HIL-SERL Module 2] 偏好感知 Q 值修正损失（Hybrid Single Arm 版本）
    # =========================================================================

    def corrected_critic_loss_fn(
        self,
        batch,
        A_cf: jnp.ndarray,
        params: Params,
        rng: PRNGKey,
    ):
        """
        Module 2: 偏好感知 Q 值修正（SACAgentHybridSingleArm 版本）。

        修正后 Bellman 目标：
            ỹ_t = y_t - α(t) · A_cf

        - α(t) = batch["alpha_weight"]，actor 端预计算
        - A_cf = max(0, mean[Q_tgt(s,a^h) - Q_tgt(s,a^π)])，外部 stop-gradient 传入

        该 agent 无 fix_gripper 配置，连续动作始终取 actions[..., :-1]。
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
        chex.assert_shape(target_q, (batch_size,))

        if self.config["backup_entropy"]:
            temperature = self.forward_temperature()
            target_q = target_q - temperature * next_actions_log_probs

        # Module 2 核心：位置感知衰减修正
        # 数学形式：y_tilde_t = y_t - alpha(t) * A_cf
        #   y_t       : 标准 Bellman 目标（温度修正后）
        #   alpha(t)  : 位置感知权重，次优片段 [t_a, t_i] 内 = exp(-lam*(t_i-t))，其余 = 0
        #   A_cf      : 反事实优势 = max(0, mean_batch[Q(s,a^h) - Q(s,a^pi)])
        # 作用：将次优片段内的 Q 目标值向下偏移，促使 Critic 学会"次优状态价值较低"
        alpha_weights = batch["alpha_weight"]   # (batch_size,)
        chex.assert_shape(alpha_weights, (batch_size,))
        A_cf_vec = jnp.broadcast_to(A_cf, alpha_weights.shape)
        corrected_target_q = target_q - alpha_weights * A_cf_vec
        chex.assert_shape(corrected_target_q, (batch_size,))

        # 该 agent 始终截断 [..:-1]（与 critic_loss_fn 保持一致，无 fix_gripper 开关）
        actions = batch["actions"][..., :-1]
        predicted_qs = self.forward_critic(
            batch["observations"], actions, rng=rng, grad_params=params
        )
        chex.assert_shape(predicted_qs, (self.config["critic_ensemble_size"], batch_size))
        target_qs = corrected_target_q[None].repeat(self.config["critic_ensemble_size"], axis=0)
        chex.assert_equal_shape([predicted_qs, target_qs])
        critic_loss = jnp.mean((predicted_qs - target_qs) ** 2)

        correction_magnitude = jnp.mean(alpha_weights * A_cf_vec)
        correction_ratio = correction_magnitude / (jnp.mean(jnp.abs(target_q)) + 1e-8)
        alpha_nonzero_ratio = jnp.mean(alpha_weights > 0.0)

        info = {
            "corrected_critic_loss": critic_loss,
            "predicted_qs": jnp.mean(predicted_qs),
            "corrected_target_qs": jnp.mean(corrected_target_q),
            "uncorrected_target_qs": jnp.mean(target_q),
            "A_cf": jnp.mean(A_cf_vec),
            "A_cf_nonzero_ratio": jnp.mean(A_cf_vec > 0.0),
            "mean_alpha_weight": jnp.mean(alpha_weights),
            "alpha_nonzero_ratio": alpha_nonzero_ratio,
            "correction_magnitude": correction_magnitude,
            "correction_ratio": correction_ratio,
            "rewards": batch["rewards"].mean(),
        }
        return critic_loss, info

    # =========================================================================
    # [HIL-SERL Module 3] 偏好引导策略学习损失（Hybrid Single Arm 版本）
    # =========================================================================

    def contrastive_policy_loss_fn(
        self,
        batch,
        preference_batch,
        params: Params,
        rng: PRNGKey,
    ):
        """
        Module 3: ORPO 风格对比 Actor 损失（SACAgentHybridSingleArm 版本）。

        L_actor = L_RLPD + β · L_contrast
    L_contrast = -E[log σ(log π(a^h|s) - log π(a^π_cur|s))]

    preference_batch 中仅需 human_actions（可带 gripper 维）；
    policy 动作在 learner 更新时由当前策略在线采样。
        """
        batch_size = batch["rewards"].shape[0]

        # 标准 SAC Actor 损失
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

        # 动作维度自动对齐：按策略输出维度截断 preference 动作。
        policy_dim = pref_distributions.distribution.loc.shape[-1]
        human_a = preference_batch["human_actions"][..., :policy_dim]

        # 不再使用历史 policy_actions：每次 learner 更新时在当前策略下重采样。
        rng, pref_sample_rng = jax.random.split(rng)
        sampled_policy_a = pref_distributions.sample(seed=pref_sample_rng)

        # 必须 stop_gradient，把策略采样的动作当作固定的“被拒绝样本”
        sampled_policy_a = jax.lax.stop_gradient(sampled_policy_a)

        # 数值稳定性：避免 |a|=1 导致 tanh-squashed log_prob 极端值。
        clip_eps = self.config.get("contrastive_action_clip_eps", 1e-6)
        human_edge_ratio = jnp.mean(jnp.abs(human_a) >= (1.0 - clip_eps))
        policy_edge_ratio = jnp.mean(jnp.abs(sampled_policy_a) >= (1.0 - clip_eps))
        human_a = jnp.clip(human_a, -1.0 + clip_eps, 1.0 - clip_eps)
        sampled_policy_a = jnp.clip(sampled_policy_a, -1.0 + clip_eps, 1.0 - clip_eps)

        log_prob_human  = pref_distributions.log_prob(human_a)   # (B,)
        log_prob_policy = pref_distributions.log_prob(sampled_policy_a)  # (B,)
        chex.assert_shape(log_prob_human,  (batch_size,))
        chex.assert_shape(log_prob_policy, (batch_size,))

        # Module 3 核心：BTL / logistic pairwise loss
        # 只在 valid_mask=True 的位置（有真实偏好样本）计算对比损失。
        valid_mask = preference_batch["valid_mask"].astype(jnp.float32)  # (B,)
        log_prob_gap = log_prob_human - log_prob_policy                  # (B,)
        contrastive_loss_per_sample = -jax.nn.log_sigmoid(log_prob_gap)  # (B,)
        valid_count = jnp.sum(valid_mask) + 1e-8
        contrastive_loss = jnp.sum(contrastive_loss_per_sample * valid_mask) / valid_count

        contrastive_coef = self.config.get("contrastive_coef", 0.1)
        total_actor_loss = rlpd_loss + contrastive_coef * contrastive_loss
        pref_action_l2_mean = jnp.sum(
            jnp.linalg.norm(human_a - sampled_policy_a, axis=-1) * valid_mask
        ) / valid_count

        info = {
            "actor_loss": total_actor_loss,
            "rlpd_actor_loss": rlpd_loss,
            "contrastive_loss": contrastive_loss,
            "temperature": temperature,
            "entropy": -log_probs.mean(),
            "log_prob_human": jnp.sum(log_prob_human * valid_mask) / valid_count,
            "log_prob_policy_hist": jnp.sum(log_prob_policy * valid_mask) / valid_count,
            "log_prob_gap": jnp.sum(log_prob_gap * valid_mask) / valid_count,
            "pref_action_l2_mean": pref_action_l2_mean,
            "contrastive_human_edge_ratio": human_edge_ratio,
            "contrastive_policy_edge_ratio": policy_edge_ratio,
        }
        return total_actor_loss, info

    # =========================================================================
    # [HIL-SERL] 联合更新：Module 2 + Module 3（含 grasp_critic）
    # =========================================================================

    @partial(jax.jit, static_argnames=("pmap_axis", "networks_to_update"))
    def update_with_correction(
        self,
        batch,
        preference_batch,
        *,
        pmap_axis=None,
        networks_to_update=frozenset({"actor", "critic", "grasp_critic", "temperature"}),
    ):
        """
        同时应用 Module 2（Q 值修正）和 Module 3（对比策略学习）。

        相比 SACAgent 版本：
          - grasp_critic 使用标准 grasp_critic_loss_fn（无连续动作修正）
          - 默认 networks_to_update 包含 grasp_critic
          - preference 动作始终截断 [..:-1]（无 fix_gripper 分支）
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

        # Step 1: 用 target network 逐样本计算 A_cf（stop-gradient）
        # preference_batch 已由 learner 端精确对齐到 batch（B 条，含 valid_mask）。
        # Hybrid agent 始终截断最后一维（gripper）：[..., :-1]
        rng, cf_rng = jax.random.split(rng)
        valid_mask_np = preference_batch["valid_mask"]          # (B,) bool
        # A_cf 用干预时刻记录的历史策略动作 a^π_old，
        # 这样即使 Module 3 已让当前策略拟合人类动作，A_cf 仍能反映干预时的劣势。
        h_actions = preference_batch["human_actions"][..., :-1] # (B, action_dim)
        p_actions = preference_batch["policy_actions"][..., :-1] # (B, action_dim)

        q_human_ensemble = self.forward_target_critic(
            preference_batch["observations"], h_actions, rng=cf_rng,
        )  # (ensemble_size, B)
        q_human_min = q_human_ensemble.min(axis=0)              # (B,)

        q_policy_ensemble = self.forward_target_critic(
            preference_batch["observations"], p_actions, rng=cf_rng,
        )
        q_policy_min = q_policy_ensemble.min(axis=0)            # (B,)

        # 逐样本 A_cf：只对 valid_mask=True 的位置计算，其余置 0
        q_gap_raw = q_human_min - q_policy_min                  # (B,)
        A_cf_batch_raw = jnp.where(
            valid_mask_np,
            jnp.maximum(0.0, q_gap_raw),
            0.0
        )                                                        # (B,)
        A_cf_batch = jax.lax.stop_gradient(A_cf_batch_raw)

        # 全局标量（用于监控）
        valid_mask_f = valid_mask_np.astype(jnp.float32)
        valid_count = jnp.sum(valid_mask_f) + 1e-8
        A_cf_raw = jnp.sum(q_gap_raw * valid_mask_f) / valid_count

        # Step 2: 构造修正后的 loss_fns（grasp_critic 使用标准损失）
        def _corrected_critic(params, rng):
            return self.corrected_critic_loss_fn(batch, A_cf_batch, params, rng)

        def _contrastive_actor(params, rng):
            return self.contrastive_policy_loss_fn(batch, preference_batch, params, rng)

        loss_fns = {
            "critic":       _corrected_critic,
            "grasp_critic": partial(self.grasp_critic_loss_fn, batch),
            "actor":        _contrastive_actor,
            "temperature":  partial(self.temperature_loss_fn, batch),
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

        info["q_human_mean"] = jnp.sum(q_human_min * valid_mask_f) / valid_count
        info["q_policy_mean"] = jnp.sum(q_policy_min * valid_mask_f) / valid_count
        info["q_gap_mean"] = jnp.sum(q_gap_raw * valid_mask_f) / valid_count
        info["q_gap_positive_ratio"] = jnp.sum((q_gap_raw > 0.0) * valid_mask_f) / valid_count
        info["A_cf_raw_mean"] = A_cf_raw
        info["A_cf_batch_mean"] = jnp.sum(A_cf_batch * valid_mask_f) / valid_count
        info["A_cf_batch_nonzero_ratio"] = jnp.sum((A_cf_batch > 0.0) * valid_mask_f) / valid_count
        info["pref_match_rate"] = jnp.mean(valid_mask_f)

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
            "grasp_critic": partial(self.grasp_critic_loss_fn, batch),
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
            {"actor", "critic", "grasp_critic", "temperature"}
        ),
        **kwargs
    ) -> Tuple["SACAgentHybridSingleArm", dict]:
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
        chex.assert_shape(batch["actions"], (batch_size, 7))

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

    @partial(jax.jit, static_argnames=("argmax"))
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
            ee_actions = dist.mode()
        else:
            ee_actions = dist.sample(seed=seed)
        
        seed, grasp_key = jax.random.split(seed, 2)
        grasp_q_values = self.forward_grasp_critic(observations, rng=grasp_key, train=False)
        
        # Select grasp actions based on the grasp Q-values
        grasp_action = grasp_q_values.argmax(axis=-1)
        grasp_action = grasp_action - 1 # Mapping back to {-1, 0, 1}

        return jnp.concatenate([ee_actions, grasp_action[..., None]], axis=-1)

    @classmethod
    def create(
        cls,
        rng: PRNGKey,
        observations: Data,
        actions: jnp.ndarray,
        # Models
        actor_def: nn.Module,
        critic_def: nn.Module,
        grasp_critic_def: nn.Module,
        temperature_def: nn.Module,
        # Optimizer
        actor_optimizer_kwargs={
            "learning_rate": 3e-4,
        },
        critic_optimizer_kwargs={
            "learning_rate": 3e-4,
        },
        grasp_critic_optimizer_kwargs={
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
        **kwargs,
    ):
        networks = {
            "actor": actor_def,
            "critic": critic_def,
            "grasp_critic": grasp_critic_def,
            "temperature": temperature_def,
        }

        model_def = ModuleDict(networks)

        # Define optimizers
        txs = {
            "actor": make_optimizer(**actor_optimizer_kwargs),
            "critic": make_optimizer(**critic_optimizer_kwargs),
            "grasp_critic": make_optimizer(**grasp_critic_optimizer_kwargs),
            "temperature": make_optimizer(**temperature_optimizer_kwargs),
        }

        rng, init_rng = jax.random.split(rng)

        params = model_def.init(
            init_rng,
            actor=[observations],
            critic=[observations, actions[..., :-1]],
            grasp_critic=[observations],
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
        if target_entropy is None:
            target_entropy = -actions.shape[-1] / 2

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
        grasp_critic_network_kwargs: dict = {
            "hidden_dims": [128, 128],
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
            "grasp_critic": encoder_def,
        }

        # Define networks
        critic_backbone = partial(MLP, **critic_network_kwargs)
        critic_backbone = ensemblize(critic_backbone, critic_ensemble_size)(
            name="critic_ensemble"
        )
        critic_def = partial(
            Critic, encoder=encoders["critic"], network=critic_backbone
        )(name="critic")
        
        grasp_critic_backbone = MLP(**grasp_critic_network_kwargs)
        grasp_critic_def = partial(
            GraspCritic, encoder=encoders["grasp_critic"], network=grasp_critic_backbone
        )(name="grasp_critic")
        
        policy_def = Policy(
            encoder=encoders["actor"],
            network=MLP(**policy_network_kwargs),
            action_dim=actions.shape[-1]-1,
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
            grasp_critic_def=grasp_critic_def,
            temperature_def=temperature_def,
            critic_ensemble_size=critic_ensemble_size,
            critic_subsample_size=critic_subsample_size,
            image_keys=image_keys,
            augmentation_function=augmentation_function,
            **kwargs,
        )

        if "pretrained" in encoder_type:  # load pretrained weights for ResNet-10
            from serl_launcher.utils.train_utils import load_resnet10_params
            agent = load_resnet10_params(agent, image_keys)

        return agent
