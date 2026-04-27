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
import examples.train_rlpd_hil as train_rlpd_hil_base

# 复用 train_rlpd_hil.py 的所有辅助函数（不重复实现）
from examples.train_rlpd_hil import (
    detect_suboptimal_segments,
    compute_episode_alpha_and_segment_ids,
    print_green,
    print_banner,
)

FLAGS = flags.FLAGS

# bc_post_steps: 人类干预结束后，附加施加 BC loss 的步数（权重逐步衰减）
# 为 0 时只对干预点本身（weight=1.0）施加 BC loss，与原来行为一致
flags.DEFINE_integer(
    "bc_post_steps", 0,
    "干预结束后附加 BC loss 的步数 x。\n"
    "权重公式：第 t 步（t=1..x）权重 = 1 - t/x，\n"
    "如 x=5 对干预后 5 步施加 weight=(0.8, 0.6, 0.4, 0.2, 0.0)。",
)

devices = jax.local_devices()
num_devices = len(devices)
sharding = jax.sharding.PositionalSharding(devices)


# =============================================================================
# Actor 循环（BC 版）：在原有 actor 基础上增加干预后 x 步 BC 引导
# =============================================================================

def actor_bc(agent, data_store, intvn_data_store, preference_data_store, env, sampling_rng,
             include_grasp_penalty: bool = False):
    """
    Actor 循环 BC 版。

    在原有 actor() 逻辑基础上新增：
      - 干预点本身的 preference_data_store.insert 添加 bc_weight=1.0
      - 干预结束后的 bc_post_steps 步，依次以衰减权重插入 preference buffer
    其余逻辑与 train_rlpd_hil.actor 完全相同。
    """
    import pickle as pkl

    # ── 验证模式 ────────────────────────────────────────────────
    if FLAGS.eval_checkpoint_step:
        success_counter = 0
        time_list = []
        ckpt = checkpoints.restore_checkpoint(
            os.path.abspath(FLAGS.checkpoint_path),
            agent.state,
            step=FLAGS.eval_checkpoint_step,
        )
        agent = agent.replace(state=ckpt)
        for episode in range(FLAGS.eval_n_trajs):
            obs, _ = env.reset()
            done = False
            start_time = time.time()
            while not done:
                sampling_rng, key = jax.random.split(sampling_rng)
                actions = agent.sample_actions(
                    observations=jax.device_put(obs), seed=key, argmax=True,
                )
                actions = np.asarray(jax.device_get(actions))
                obs, reward, done, truncated, info = env.step(actions)
                if done or truncated:
                    success_counter += int(info.get("succeed", reward > 0))
                    time_list.append(time.time() - start_time)
                done = done or truncated
        print(f"success rate: {success_counter / FLAGS.eval_n_trajs}")
        print(f"average time: {np.mean(time_list)}")
        return

    # ── 断点续训起始步 ────────────────────────────────────────
    start_step = (
        int(os.path.basename(
            natsorted(glob.glob(os.path.join(FLAGS.checkpoint_path, "buffer/*.pkl")))[-1]
        )[12:-4]) + 1
        if FLAGS.checkpoint_path and os.path.exists(FLAGS.checkpoint_path)
           and glob.glob(os.path.join(FLAGS.checkpoint_path, "buffer/*.pkl"))
        else 0
    )

    # ── 注册数据存储 ────────────────────────────────────────
    datastore_dict = {
        "actor_env":       data_store,
        "actor_env_intvn": intvn_data_store,
        "actor_env_pref":  preference_data_store,
    }
    client = TrainerClient(
        "actor_env",
        FLAGS.ip,
        make_trainer_config(),
        data_stores=datastore_dict,
        wait_for_server=True,
        timeout_ms=3000,
    )

    def update_params(params):
        nonlocal agent
        agent = agent.replace(state=agent.state.replace(params=params))

    client.recv_network_callback(update_params)

    # ── [Module 1] 加载 Progress Model Runner ────────────────────
    progress_runner = None
    if FLAGS.progress_model_path and FLAGS.state_stats_path:
        from progress_model_inference import ProgressModelRunner
        progress_runner = ProgressModelRunner(
            model_path=FLAGS.progress_model_path,
            stats_path=FLAGS.state_stats_path,
            side_key=FLAGS.progress_side_key,
            wrist_key=FLAGS.progress_wrist_key,
            hidden_dim=FLAGS.progress_hidden_dim,
            device=FLAGS.progress_device,
        )
        print_green(f"[Module 1] Progress Model 已加载: {FLAGS.progress_model_path}")
    else:
        print_green(
            f"[Module 1] 未提供 --progress_model_path，"
            f"使用 fallback（向前 {FLAGS.suboptimal_window} 步）"
        )

    # ── Episode 级缓存 ────────────────────────────────────────
    transitions:           List[Dict] = []
    demo_transitions:      List[Dict] = []
    episode_buffer:        List[Dict] = []
    intervention_markers:  List[Dict] = []
    next_segment_uid = 0

    # ── [BC post-steps] 状态跟踪 ───────────────────────────
    bc_post_counter     = 0       # 剩余干预后步数
    bc_post_last_human  = None    # 最近一次干预的最后人类动作
    bc_post_segment_id  = None    # 对应的 segment_id

    obs, _ = env.reset()
    done = False

    timer = Timer()
    running_return = 0.0
    already_intervened = False
    intervention_count = 0
    intervention_steps = 0

    pbar = tqdm.tqdm(range(start_step, config.max_steps), dynamic_ncols=True)
    for step in pbar:
        timer.tick("total")

        # ── 采样动作 ───────────────────────────────────────
        with timer.context("sample_actions"):
            if step < config.random_steps:
                actions = env.action_space.sample()
            else:
                sampling_rng, key = jax.random.split(sampling_rng)
                actions = agent.sample_actions(
                    observations=jax.device_put(obs),
                    seed=key,
                    argmax=False,
                )
                actions = np.asarray(jax.device_get(actions))

        # ── 环境交互 ─────────────────────────────────────────
        with timer.context("step_env"):
            policy_action_saved = actions.copy()
            next_obs, reward, done, truncated, info = env.step(actions)
            if "left" in info:
                info.pop("left")
            if "right" in info:
                info.pop("right")

            prev_was_intervened = already_intervened

            if "intervene_action_eef" in info:
                human_action = info.pop("intervene_action_eef")
                intervention_steps += 1

                if not already_intervened:
                    intervention_count += 1
                    t_i = len(episode_buffer)
                    intervention_markers.append({
                        "step_idx":      t_i,
                        "segment_id":    next_segment_uid,
                        "obs":           obs,
                        "human_action":  human_action,
                        "policy_action": policy_action_saved,
                    })
                    # 干预点 weight=1.0
                    preference_data_store.insert({
                        "observations":   obs,
                        "human_actions":  human_action,
                        "policy_actions": policy_action_saved,
                        "segment_ids":    np.int32(next_segment_uid),
                        "bc_weight":      np.float32(1.0),
                    })
                    next_segment_uid += 1

                already_intervened = True
                actions = human_action
                # [BC post-steps] 记录当前干预的最新人类动作
                bc_post_last_human = human_action
                bc_post_segment_id = next_segment_uid - 1
            else:
                # [BC post-steps] 干预刺刺结束→重置计数器
                if prev_was_intervened:
                    bc_post_counter = FLAGS.bc_post_steps

                already_intervened = False
                _scale = np.asarray(config.action_scale) if hasattr(config, "action_scale") else None
                if _scale is not None and _scale.ndim > 0:
                    _zero_mask = (_scale == 0)
                    if _zero_mask.any():
                        actions = actions.copy()
                        actions[:len(_zero_mask)][_zero_mask[:len(actions)]] = 0.0

                # [BC post-steps] 干预后 x 步，以逐步衰减权重将当前状态写入 preference buffer
                if bc_post_counter > 0 and bc_post_last_human is not None:
                    t_since = FLAGS.bc_post_steps - bc_post_counter + 1  # 1, 2, ..., x
                    weight = np.float32(1.0 - t_since / FLAGS.bc_post_steps)
                    if weight > 0:  # 最后一步 weight=0 无意义，跳过
                        preference_data_store.insert({
                            "observations":   obs,
                            "human_actions":  bc_post_last_human,
                            "policy_actions": policy_action_saved,
                            "segment_ids":    np.int32(bc_post_segment_id),
                            "bc_weight":      weight,
                        })
                    bc_post_counter -= 1

            running_return += reward

            transition = dict(
                observations=obs,
                actions=actions,
                next_observations=next_obs,
                rewards=reward,
                masks=1.0 - done,
                dones=done,
                label=2 if already_intervened else 1,
                alpha_weight=0.0,
                segment_ids=-1,
                _was_intervened=already_intervened,
            )
            if include_grasp_penalty and "grasp_penalty" in info:
                transition["grasp_penalty"] = info["grasp_penalty"]

            episode_buffer.append(transition)

            if already_intervened:
                intvn_tr = {k: v for k, v in transition.items()
                            if not k.startswith("_")}
                intvn_data_store.insert(intvn_tr)
                demo_transitions.append(intvn_tr)

            obs = next_obs

            if done or truncated:
                suboptimal_segs = detect_suboptimal_segments(
                    episode_buffer,
                    intervention_markers,
                    window=FLAGS.suboptimal_window,
                    progress_runner=progress_runner,
                    anomaly_window=FLAGS.anomaly_window,
                    delta_reg=FLAGS.delta_reg,
                    delta_stag=FLAGS.delta_stag,
                    detect_regression=FLAGS.detect_regression,
                    detect_stagnation=FLAGS.detect_stagnation,
                    recovery_k=FLAGS.recovery_k,
                )

                segment_uids = [m["segment_id"] for m in intervention_markers]
                alpha_weights, segment_ids = compute_episode_alpha_and_segment_ids(
                    len(episode_buffer), suboptimal_segs, segment_uids, FLAGS.alpha_lambda
                )

                for i, tr in enumerate(episode_buffer):
                    tr_to_insert = {k: v for k, v in tr.items()
                                    if not k.startswith("_")}
                    tr_to_insert["alpha_weight"] = float(alpha_weights[i])
                    tr_to_insert["segment_ids"] = int(segment_ids[i])
                    data_store.insert(tr_to_insert)
                    transitions.append(tr_to_insert)

                episode_len = len(episode_buffer)
                episode_return = running_return
                succeed = bool(info.get("succeed", reward > 0))

                ep_stats = {
                    "episode/return":            episode_return,
                    "episode/length":            episode_len,
                    "episode/success":           float(succeed),
                    "episode/intervention_rate": intervention_steps / max(episode_len, 1),
                    "episode/intervention_count": intervention_count,
                    "episode/n_suboptimal_segs":  len(suboptimal_segs),
                    "episode/suboptimal_ratio":   float(
                        np.sum(alpha_weights > 0) / max(episode_len, 1)
                    ),
                    "episode/mean_alpha":          float(alpha_weights[alpha_weights > 0].mean())
                                                   if np.any(alpha_weights > 0) else 0.0,
                    "episode/avg_intervention_len": intervention_steps / max(intervention_count, 1),
                }
                info["episode"].update(ep_stats)
                stats = {"environment": info}
                client.request("send-stats", stats)

                pbar.set_description(f"last return: {running_return:.2f}")
                running_return = 0.0
                intervention_count = 0
                intervention_steps = 0
                already_intervened = False
                # [BC post-steps] episode 结束时清空 post-step 状态
                bc_post_counter    = 0
                bc_post_last_human = None
                bc_post_segment_id = None
                episode_buffer = []
                intervention_markers = []

                client.update()

                if config.buffer_period > 0 and FLAGS.checkpoint_path:
                    buffer_path      = os.path.join(FLAGS.checkpoint_path, "buffer")
                    demo_buffer_path = os.path.join(FLAGS.checkpoint_path, "demo_buffer")
                    os.makedirs(buffer_path,      exist_ok=True)
                    os.makedirs(demo_buffer_path, exist_ok=True)
                    if transitions:
                        with open(os.path.join(buffer_path, f"transitions_{step}.pkl"), "wb") as f:
                            pkl.dump(transitions, f)
                        transitions = []
                    if demo_transitions:
                        with open(os.path.join(demo_buffer_path, f"transitions_{step}.pkl"), "wb") as f:
                            pkl.dump(demo_transitions, f)
                        demo_transitions = []

                obs, _ = env.reset()

        timer.tock("total")

        if step % config.log_period == 0:
            stats = {"timer": timer.get_average_times()}
            client.request("send-stats", stats)


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
    # actor() 复用自 examples.train_rlpd_hil，该函数内部读取其模块级全局 config。
    # 这里同步一次，避免 BC 脚本中出现 NameError: name 'config' is not defined。
    train_rlpd_hil_base.config = config

    assert config.batch_size % num_devices == 0

    rng = jax.random.PRNGKey(FLAGS.seed)
    rng, sampling_rng = jax.random.split(rng)

    assert FLAGS.exp_name in CONFIG_MAPPING, "Experiment folder not found."

    use_classifier = bool(getattr(config, "classifier_keys", []))
    if not use_classifier:
        print_green("[Env] classifier_keys 为空，已自动关闭 reward classifier wrapper。")

    env = config.get_environment(
        fake_env=FLAGS.learner,
        save_video=FLAGS.save_video,
        classifier=use_classifier,
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
                    if "infos" in transition and "grasp_penalty" in transition["infos"]:
                        transition["grasp_penalty"] = transition["infos"]["grasp_penalty"]
                    transition.setdefault("alpha_weight", 0.0)
                    transition.setdefault("segment_ids", -1)
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
                        transition.setdefault("alpha_weight", 0.0)
                        transition.setdefault("segment_ids", -1)
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
                        transition.setdefault("alpha_weight", 0.0)
                        transition.setdefault("segment_ids", -1)
                        demo_buffer.insert(transition)
            print_green(f"Loaded previous demo buffer. Demo buffer size: {len(demo_buffer)}")

        print_banner(
            "[ACTIVE] HIL-SERL BC (pre-tanh MSE)",
            [
                f"alpha_lambda         = {FLAGS.alpha_lambda}",
                f"bc_coef (β)          = {FLAGS.contrastive_coef}  (复用 contrastive_coef flag)",
                f"bc_post_steps        = {FLAGS.bc_post_steps}  (干预后附加 BC 引导的步数)",
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

        print_green("starting actor_bc loop")
        actor_bc(
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
