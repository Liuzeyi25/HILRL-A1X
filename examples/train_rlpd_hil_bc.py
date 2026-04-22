#!/usr/bin/env python3
"""
train_rlpd_hil_bc.py
====================
HIL-SERL BC 版训练脚本。

在 train_rlpd_hil.py 基础上，将 Module 3 从 ORPO 对比损失切换为
Pre-tanh MSE 行为克隆损失（bc_policy_loss_fn），彻底消除 1/σ² 梯度项
导致的 entropy 崩塌问题。

与 train_rlpd_hil.py 的唯一区别：
  - learner 循环中的 update_with_correction → update_with_correction_bc
  - wandb 日志 key：contrastive_loss / log_prob_* → bc_loss / action_distance
  - known_prefixes 对应更新

Module 1（Progress Model 次优段识别）、Module 2（Q 值修正）保持不变，
无需对 data_store.py / replay_buffer.py 做任何改动。
"""

# ── 从 train_rlpd_hil.py 复用所有 import 和工具函数 ─────────────────────────
import glob
import os
import time

import jax
import jax.numpy as jnp
import numpy as np
import tqdm

from functools import partial
from typing import Dict, List, Optional, Tuple

import flax.linen as nn
from absl import app, flags
from flax import config as flax_config
from flax.training import checkpoints
from gymnasium.wrappers.record_episode_statistics import RecordEpisodeStatistics
from natsort import natsorted

flax_config.update('flax_use_orbax_checkpointing', False)

from serl_launcher.agents.continuous.sac import SACAgent
from serl_launcher.agents.continuous.sac_hybrid_single import SACAgentHybridSingleArm
from serl_launcher.utils.timer_utils import Timer
from serl_launcher.utils.train_utils import concat_batches

from agentlace.trainer import TrainerServer, TrainerClient
from agentlace.data.data_store import QueuedDataStore

from serl_launcher.utils.launcher import (
    make_sac_pixel_agent,
    make_sac_pixel_agent_hybrid_single_arm,
    make_trainer_config,
    make_wandb_logger,
)
from serl_launcher.data.data_store import (
    MemoryEfficientReplayBufferDataStore,
    PreferenceBufferDataStore,
)

from experiments.mappings import CONFIG_MAPPING

# 复用 train_rlpd_hil.py 的所有辅助函数（不重复实现）
from examples.train_rlpd_hil import (
    detect_suboptimal_segments,
    compute_episode_alpha_and_segment_ids,
    actor,
    print_green,
    print_banner,
)

FLAGS = flags.FLAGS

devices = jax.local_devices()
num_devices = len(devices)
sharding = jax.sharding.PositionalSharding(devices)


# =============================================================================
# Learner 循环（BC 版）
# =============================================================================

def learner_bc(
    rng,
    agent,
    replay_buffer: MemoryEfficientReplayBufferDataStore,
    demo_buffer: MemoryEfficientReplayBufferDataStore,
    preference_buffer: PreferenceBufferDataStore,
    wandb_logger=None,
):
    """
    Learner 循环 BC 版。

    与 train_rlpd_hil.learner 的唯一区别：
      - update_with_correction    → update_with_correction_bc
      - wandb 日志 actor_info 的 key 改为 bc_loss / action_distance
    """
    start_step = (
        int(os.path.basename(
            checkpoints.latest_checkpoint(os.path.abspath(FLAGS.checkpoint_path))
        )[11:]) + 1
        if FLAGS.checkpoint_path and os.path.exists(FLAGS.checkpoint_path)
        else 0
    )
    step = start_step

    def stats_callback(type: str, payload: dict) -> dict:
        assert type == "send-stats", f"Invalid request type: {type}"
        if wandb_logger is not None:
            wandb_logger.log(payload, step=step)
        return {}

    server = TrainerServer(make_trainer_config(), request_callback=stats_callback)
    server.register_data_store("actor_env",       replay_buffer)
    server.register_data_store("actor_env_intvn", demo_buffer)
    server.register_data_store("actor_env_pref",  preference_buffer)
    server.start(threaded=True)

    pbar = tqdm.tqdm(
        total=config.training_starts,
        initial=len(replay_buffer),
        desc="Filling up replay buffer",
        position=0,
        leave=True,
    )
    while len(replay_buffer) < config.training_starts:
        pbar.update(len(replay_buffer) - pbar.n)
        time.sleep(1)
    pbar.update(len(replay_buffer) - pbar.n)
    pbar.close()

    server.publish_network(agent.state.params)
    print_green("sent initial network to actor")

    replay_iterator = replay_buffer.get_iterator(
        sample_args={"batch_size": config.batch_size // 2, "pack_obs": True},
        device=sharding.replicate(),
    )
    demo_iterator = demo_buffer.get_iterator(
        sample_args={"batch_size": config.batch_size // 2, "pack_obs": True},
        device=sharding.replicate(),
    )

    timer = Timer()

    if isinstance(agent, SACAgentHybridSingleArm):
        train_critic_networks_to_update = frozenset({"critic", "grasp_critic"})
        train_networks_to_update = frozenset({"critic", "grasp_critic", "actor", "temperature"})
    else:
        train_critic_networks_to_update = frozenset({"critic"})
        train_networks_to_update = frozenset({"critic", "actor", "temperature"})

    for step in tqdm.tqdm(
        range(start_step, config.max_steps), dynamic_ncols=True, desc="learner_bc"
    ):
        use_correction = len(preference_buffer) >= 1

        # ── 高 UTD：n-1 次 Critic-only 更新 ─────────────────────────────
        for _ in range(config.cta_ratio - 1):
            with timer.context("sample_replay_buffer"):
                batch      = next(replay_iterator)
                demo_batch = next(demo_iterator)
                batch      = concat_batches(batch, demo_batch, axis=0)

            with timer.context("train_critics"):
                if use_correction:
                    seg_ids_np = np.asarray(batch["segment_ids"])
                    matched_pref = preference_buffer.get_by_segment_ids(seg_ids_np)
                    if "observations" in matched_pref:
                        # BC 版：调用 update_with_correction_bc
                        agent, _ = agent.update_with_correction_bc(
                            batch, matched_pref,
                            preference_batch_direct=None,   # Critic-only，不需要 BC 项
                            networks_to_update=train_critic_networks_to_update,
                        )
                    else:
                        agent, _ = agent.update(
                            batch,
                            networks_to_update=train_critic_networks_to_update,
                        )
                else:
                    agent, _ = agent.update(
                        batch,
                        networks_to_update=train_critic_networks_to_update,
                    )

        # ── 1 次 Critic + Actor + Temperature 联合更新 ───────────────────
        with timer.context("train"):
            batch      = next(replay_iterator)
            demo_batch = next(demo_iterator)
            batch      = concat_batches(batch, demo_batch, axis=0)

            if use_correction:
                seg_ids_np = np.asarray(batch["segment_ids"])
                matched_pref = preference_buffer.get_by_segment_ids(seg_ids_np)
                pref_batch_direct = preference_buffer.sample(
                    min(FLAGS.preference_batch_size, len(preference_buffer))
                )
                if "observations" in matched_pref and pref_batch_direct is not None:
                    # BC 版：调用 update_with_correction_bc
                    agent, update_info = agent.update_with_correction_bc(
                        batch, matched_pref,
                        preference_batch_direct=pref_batch_direct,
                        networks_to_update=train_networks_to_update,
                    )
                else:
                    use_correction = False
                    agent, update_info = agent.update(
                        batch,
                        networks_to_update=train_networks_to_update,
                    )
            else:
                agent, update_info = agent.update(
                    batch,
                    networks_to_update=train_networks_to_update,
                )

        update_info["use_correction"] = float(use_correction)

        if step > 0 and step % config.steps_per_update == 0:
            agent = jax.block_until_ready(agent)
            server.publish_network(agent.state.params)

        # ── 日志（BC 版 key 更新）────────────────────────────────────────
        if step % config.log_period == 0 and wandb_logger:
            critic_info  = {f"critic/{k}":  v for k, v in update_info.items()
                            if k.startswith(("corrected_critic", "critic_loss",
                                             "predicted_qs", "target_qs", "uncorrected",
                                             "correction_", "A_cf", "mean_alpha",
                                             "alpha_nonzero", "rewards"))}
            module2_info = {f"module2/{k}": v for k, v in update_info.items()
                            if k.startswith(("q_human", "q_policy", "q_gap",
                                             "A_cf_raw", "A_cf_batch", "pref_match"))}
            # BC 版：bc_loss / action_distance 替换 contrastive_loss / log_prob_*
            actor_info   = {f"actor/{k}":   v for k, v in update_info.items()
                            if k.startswith(("actor_loss", "rlpd_actor",
                                             "bc_loss",           # ← BC 损失
                                             "entropy", "temperature",
                                             "action_distance",   # ← 新增监控指标
                                             ))}
            train_info   = {"train/use_correction": float(use_correction),
                            "train/preference_buffer_size": len(preference_buffer)}

            known_prefixes = (
                "corrected_critic", "critic_loss", "predicted_qs", "target_qs",
                "uncorrected", "correction_", "A_cf", "mean_alpha", "alpha_nonzero",
                "rewards", "q_human", "q_policy", "q_gap", "A_cf_raw", "A_cf_batch",
                "pref_match",
                # BC 版 actor 前缀（替换 contrastive_loss / log_prob_* / pref_action_l2）
                "actor_loss", "rlpd_actor", "bc_loss", "action_distance",
                "entropy", "temperature",
                "use_correction",
            )
            misc_info = {k: v for k, v in update_info.items()
                         if not any(k.startswith(p) for p in known_prefixes)}

            wandb_logger.log({**critic_info, **module2_info, **actor_info,
                              **train_info, **misc_info}, step=step)
            wandb_logger.log({"timer": timer.get_average_times()}, step=step)

        if (
            step > 0
            and config.checkpoint_period
            and step % config.checkpoint_period == 0
        ):
            checkpoints.save_checkpoint(
                os.path.abspath(FLAGS.checkpoint_path), agent.state, step=step, keep=100
            )


# =============================================================================
# Main
# =============================================================================

def main(_):
    global config
    config = CONFIG_MAPPING[FLAGS.exp_name]()

    assert config.batch_size % num_devices == 0

    rng = jax.random.PRNGKey(FLAGS.seed)
    rng, sampling_rng = jax.random.split(rng)

    assert FLAGS.exp_name in CONFIG_MAPPING, "Experiment folder not found."

    env = config.get_environment(
        fake_env=FLAGS.learner,
        save_video=FLAGS.save_video,
        classifier=True,
        stack_obs_num=2,
    )
    env = RecordEpisodeStatistics(env)

    rng, sampling_rng = jax.random.split(rng)

    # ── 初始化 Agent（与 train_rlpd_hil.py 完全相同）────────────────────
    if config.setup_mode in ("single-arm-fixed-gripper", "dual-arm-fixed-gripper"):
        agent: SACAgent = make_sac_pixel_agent(
            seed=FLAGS.seed,
            sample_obs=env.observation_space.sample(),
            sample_action=env.action_space.sample(),
            image_keys=config.image_keys,
            encoder_type=config.encoder_type,
            discount=config.discount,
        )
        include_grasp_penalty = False
    elif config.setup_mode == "single-arm-learned-gripper":
        agent: SACAgentHybridSingleArm = make_sac_pixel_agent_hybrid_single_arm(
            seed=FLAGS.seed,
            sample_obs=env.observation_space.sample(),
            sample_action=env.action_space.sample(),
            image_keys=config.image_keys,
            encoder_type=config.encoder_type,
            discount=config.discount,
        )
        include_grasp_penalty = True
    else:
        raise NotImplementedError(
            f"train_rlpd_hil_bc.py 支持 fixed-gripper 和 single-arm-learned-gripper，"
            f"当前 setup_mode={config.setup_mode}。"
        )

    # 注入 HIL 超参（与 train_rlpd_hil.py 相同，bc_action_clip_eps 可选覆盖）
    hil_config = dict(agent.config)
    hil_config["contrastive_coef"] = FLAGS.contrastive_coef
    hil_config["flat_alpha_correction"] = FLAGS.flat_alpha_correction
    # bc_action_clip_eps 默认 1e-2，可通过 agent.config 覆盖（此处不新增 flag）
    agent = agent.replace(config=hil_config)

    agent = jax.device_put(
        jax.tree_util.tree_map(jnp.array, agent), sharding.replicate()
    )

    if FLAGS.checkpoint_path is not None and os.path.exists(FLAGS.checkpoint_path):
        input("Checkpoint path already exists. Press Enter to resume training.")
        ckpt = checkpoints.restore_checkpoint(
            os.path.abspath(FLAGS.checkpoint_path), agent.state
        )
        agent = agent.replace(state=ckpt)
        ckpt_number = os.path.basename(
            checkpoints.latest_checkpoint(os.path.abspath(FLAGS.checkpoint_path))
        )[11:]
        print_green(f"Loaded previous checkpoint at step {ckpt_number}.")

    def create_replay_buffer_and_wandb_logger():
        replay_buffer = MemoryEfficientReplayBufferDataStore(
            env.observation_space,
            env.action_space,
            capacity=config.replay_buffer_capacity,
            image_keys=config.image_keys,
            include_alpha_correction=True,
            include_segment_ids=True,
        )
        _wandb_desc = (
            f"{FLAGS.exp_name}_bc_{FLAGS.run_tag}" if FLAGS.run_tag
            else f"{FLAGS.exp_name}_bc"
        )
        wandb_logger = make_wandb_logger(
            project="hil-serl",
            description=_wandb_desc,
            debug=FLAGS.debug,
        )
        return replay_buffer, wandb_logger

    # ── Learner 进程 ──────────────────────────────────────────────────────
    if FLAGS.learner:
        sampling_rng = jax.device_put(sampling_rng, device=sharding.replicate())
        replay_buffer, wandb_logger = create_replay_buffer_and_wandb_logger()

        demo_buffer = MemoryEfficientReplayBufferDataStore(
            env.observation_space,
            env.action_space,
            capacity=config.replay_buffer_capacity,
            image_keys=config.image_keys,
            include_alpha_correction=True,
            include_segment_ids=True,
        )

        preference_buffer = PreferenceBufferDataStore(capacity=10000)

        assert FLAGS.demo_path is not None
        for path in FLAGS.demo_path:
            with open(path, "rb") as f:
                import pickle
                demo_data = pickle.load(f)
                for transition in demo_data:
                    demo_buffer.insert(transition)
        print_green(f"demo buffer size: {len(demo_buffer)}")
        print_green(f"online buffer size: {len(replay_buffer)}")

        if FLAGS.checkpoint_path is not None and os.path.exists(
            os.path.join(FLAGS.checkpoint_path, "buffer")
        ):
            for file in glob.glob(os.path.join(FLAGS.checkpoint_path, "buffer/*.pkl")):
                with open(file, "rb") as f:
                    import pickle
                    for transition in pickle.load(f):
                        replay_buffer.insert(transition)
            print_green(f"Loaded previous buffer. Replay buffer size: {len(replay_buffer)}")

        if FLAGS.checkpoint_path is not None and os.path.exists(
            os.path.join(FLAGS.checkpoint_path, "demo_buffer")
        ):
            for file in glob.glob(
                os.path.join(FLAGS.checkpoint_path, "demo_buffer/*.pkl")
            ):
                with open(file, "rb") as f:
                    import pickle
                    for transition in pickle.load(f):
                        demo_buffer.insert(transition)
            print_green(f"Loaded previous demo buffer. Demo buffer size: {len(demo_buffer)}")

        print_banner(
            "[ACTIVE] HIL-SERL BC (pre-tanh MSE)",
            [
                f"alpha_lambda         = {FLAGS.alpha_lambda}",
                f"bc_coef (β)          = {FLAGS.contrastive_coef}  (复用 contrastive_coef flag)",
                f"preference_batch     = {FLAGS.preference_batch_size}",
                f"suboptimal_window    = {FLAGS.suboptimal_window}",
                f"progress_model       = {FLAGS.progress_model_path or 'fallback'}",
                f"Module 3             = bc_policy_loss_fn (pre-tanh MSE, no entropy coupling)",
            ],
            color="cyan",
        )

        print_green("starting learner_bc loop")
        learner_bc(
            sampling_rng,
            agent,
            replay_buffer,
            demo_buffer,
            preference_buffer=preference_buffer,
            wandb_logger=wandb_logger,
        )

    # ── Actor 进程（与 train_rlpd_hil.py 完全复用）────────────────────────
    elif FLAGS.actor:
        sampling_rng = jax.device_put(sampling_rng, sharding.replicate())
        data_store            = QueuedDataStore(20000)
        intvn_data_store      = QueuedDataStore(20000)
        preference_data_store = QueuedDataStore(10000)

        print_green("starting actor loop")
        actor(
            agent,
            data_store,
            intvn_data_store,
            preference_data_store,
            env,
            sampling_rng,
            include_grasp_penalty=include_grasp_penalty,
        )

    else:
        raise NotImplementedError("Must be either --learner or --actor")


if __name__ == "__main__":
    app.run(main)
