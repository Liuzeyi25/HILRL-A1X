#!/usr/bin/env python3
"""
train_rlpd_hil.py
=================
HIL-SERL 偏好引导强化学习训练脚本（Proposal 方法实现）
基于 haoyuan-HILRL/examples/train_rlpd.py 改写。

在原始 train_rlpd.py 基础上，新增三个协同模块：

  Module 1 - Progress Model 次优片段识别
             (progress_model_path 为 None 时退化为向前 suboptimal_window 步的 fallback)
  Module 2 - 偏好感知 Q 值修正（Preference-Aware Q-value Correction）
             alpha_weight 字段由 actor 预计算写入 replay buffer
  Module 3 - 偏好引导策略学习（ORPO 风格对比损失）
             PreferenceBufferDataStore 收集干预偏好对

依赖改动（相对原始 train_rlpd.py）
----------------------------------
  - serl_launcher/data/replay_buffer.py:          新增 include_alpha_correction 参数
  - serl_launcher/data/memory_efficient_replay_buffer.py: 透传 include_alpha_correction
  - serl_launcher/data/data_store.py:             透传 include_alpha_correction；
                                                   新增 PreferenceBufferDataStore
  - serl_launcher/agents/continuous/sac.py:            新增三个修正方法
  - serl_launcher/agents/continuous/sac_hybrid_single.py: 新增三个修正方法
"""

import glob
import json as _json
import os
import pickle as pkl
import time
from functools import partial
from typing import Dict, List, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import tqdm
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

FLAGS = flags.FLAGS

# ── 基础 flags（与 train_rlpd.py 对齐）────────────────────────────────────
flags.DEFINE_string("exp_name", None, "Name of experiment corresponding to folder.")
flags.DEFINE_integer("seed", 42, "Random seed.")
flags.DEFINE_boolean("learner", False, "Whether this is a learner.")
flags.DEFINE_boolean("actor", False, "Whether this is an actor.")
flags.DEFINE_string("ip", "localhost", "IP address of the learner.")
flags.DEFINE_multi_string("demo_path", None, "Path to the demo data.")
flags.DEFINE_string("checkpoint_path", None, "Path to save checkpoints.")
flags.DEFINE_integer("eval_checkpoint_step", 0, "Step to evaluate the checkpoint.")
flags.DEFINE_integer("eval_n_trajs", 0, "Number of trajectories to evaluate.")
flags.DEFINE_boolean("save_video", False, "Save video.")
flags.DEFINE_boolean("debug", False, "Debug mode.")
flags.DEFINE_string("run_tag", "", "Optional tag for wandb run name.")

# ── HIL-SERL Module 超参 ──────────────────────────────────────────────────
flags.DEFINE_float(
    "alpha_lambda", 3.0,
    "次优片段位置感知衰减系数 λ。α(t)=exp(-λ*(t_i-t))，"
    "λ=3 时 t_a 处≈0.05，t_i 处=1.0。",
)
flags.DEFINE_float(
    "contrastive_coef", 0.2,
    "对比损失系数 β。总 Actor 损失 = L_RLPD + β * L_contrast。",
)
flags.DEFINE_integer(
    "preference_batch_size", 4,
    "偏好学习批大小。低于此数量时跳过修正更新，使用标准 RLPD。",
)
flags.DEFINE_integer(
    "suboptimal_window", 5,
    "detect_suboptimal_segments 的 fallback 回溯窗口（步数）。",
)

# ── Progress Model 路径 ───────────────────────────────────────────────────
flags.DEFINE_string(
    "progress_model_path", None,
    "训练好的 Progress Model 权重路径（progress_model_best.pt）。"
    "为 None 时退化为 suboptimal_window fallback。",
)
flags.DEFINE_string(
    "state_stats_path", None,
    "Progress Model 状态归一化统计文件（state_stats.pt）。"
    "需与 progress_model_path 同时指定。",
)
flags.DEFINE_string("progress_side_key", "side_policy_256", "侧面摄像头 obs key。")
flags.DEFINE_string("progress_wrist_key", "wrist_1", "腕部摄像头 obs key。")
flags.DEFINE_integer("progress_hidden_dim", 128, "Progress Model MLP 隐层维度。")
flags.DEFINE_string("progress_device", "cpu", "Progress Model 推理设备。")

# ── 异常检测超参 ──────────────────────────────────────────────────────────
flags.DEFINE_integer("anomaly_window", 4, "异常检测滑动窗口大小（步数）。")
flags.DEFINE_float("delta_reg", 0.045, "回退检测阈值：ΔP < -delta_reg 判为回退。")
flags.DEFINE_float("delta_stag", 0.001, "停滞检测阈值：|ΔP| < delta_stag 判为停滞。")
flags.DEFINE_boolean("detect_regression", True, "是否启用进度回退异常检测。")
flags.DEFINE_boolean("detect_stagnation", False, "是否启用进度停滞异常检测。")
flags.DEFINE_integer("recovery_k", 3, "恢复确认窗口（连续 K 帧 >= P_anchor 才确认恢复）。")

devices = jax.local_devices()
num_devices = len(devices)
sharding = jax.sharding.PositionalSharding(devices)


def print_green(x):
    return print("\033[92m {}\033[00m".format(x))


def print_banner(title: str, lines: list, color: str = "yellow") -> None:
    _codes = {"yellow": "\033[93m", "green": "\033[92m",
               "cyan": "\033[96m", "red": "\033[91m"}
    bold  = "\033[1m"
    reset = "\033[00m"
    c     = _codes.get(color, _codes["yellow"])
    width = max(len(title), max((len(l) for l in lines), default=0)) + 4
    border = "═" * width
    print(f"\n{c}{bold}╔{border}╗")
    print(f"║  {title:^{width - 2}}  ║")
    print(f"╠{border}╣")
    for line in lines:
        print(f"║  {line:<{width - 2}}  ║")
    print(f"╚{border}╝{reset}\n")


# =============================================================================
# Module 1: 次优轨迹片段识别
# =============================================================================

def detect_suboptimal_segments(
    episode_buffer: List[Dict],
    intervention_markers: List[Dict],
    window: int = 10,
    progress_runner=None,
    anomaly_window: int = 4,
    delta_reg: float = 0.045,
    delta_stag: float = 0.001,
    detect_regression: bool = True,
    detect_stagnation: bool = False,
    recovery_k: int = 3,
) -> List[Tuple[int, int]]:
    """
    [Module 1] 识别 episode 中的次优轨迹片段，返回 [(t_a, t_i), ...] 列表。

    - progress_runner=None：fallback 模式，每个干预点向前回溯 window 步作为 t_a
    - progress_runner 已加载：调用 Progress Model 推理 + 滑窗异常检测确定 t_a
    """
    if progress_runner is None:
        # Fallback：无 Progress Model
        segments = []
        for marker in intervention_markers:
            t_i = marker["step_idx"]
            t_a = max(0, t_i - window)
            segments.append((t_a, t_i))
        return segments

    # 真实实现：Progress Model 推理 + 异常检测
    from progress_model_inference import detect_anomalies

    preds, labels = progress_runner.infer_episode(episode_buffer)
    if len(preds) == 0:
        return []

    anomaly_segs = detect_anomalies(
        preds, labels,
        window_size=anomaly_window,
        delta_reg=delta_reg,
        delta_stag=delta_stag,
        detect_regression=detect_regression,
        detect_stagnation=detect_stagnation,
        recovery_k=recovery_k,
    )

    segments = []
    for marker in intervention_markers:
        t_i = marker["step_idx"]
        candidates = [s for s in anomaly_segs if s["start"] <= t_i]
        if candidates:
            best = max(candidates, key=lambda s: s["start"])
            t_a = best["start"]
        else:
            t_a = max(0, t_i - window)
        segments.append((t_a, t_i))

    return segments


def compute_episode_alpha_weights(
    episode_len: int,
    suboptimal_segments: List[Tuple[int, int]],
    lam: float,
) -> np.ndarray:
    """
    为 episode 每条 transition 计算位置感知衰减权重 α(t)。

    规则：
      - t ∈ [t_a, t_i]（次优片段内）：alpha = exp(-λ*(t_i - t))
        t_i 处最强（α=1），t_a 处最弱（α≈0）
      - 其余 transition：alpha = 0.0
      - 多段重叠时取最大值
    """
    alpha_weights = np.zeros(episode_len, dtype=np.float32)
    for (t_a, t_i) in suboptimal_segments:
        for t in range(t_a, min(t_i + 1, episode_len)):
            w = float(np.exp(-lam * (t_i - t)))
            alpha_weights[t] = max(alpha_weights[t], w)
    return alpha_weights


# =============================================================================
# Actor 循环
# =============================================================================

def actor(agent, data_store, intvn_data_store, preference_data_store, env, sampling_rng,
          include_grasp_penalty: bool = False):
    """
    Actor 循环（--actor 模式）。

    相对 train_rlpd.py 的改动：
      1. 每步干预覆盖前保存策略动作 policy_action_saved
      2. 干预起始时刻 t_i 记录偏好对并写入 preference_data_store
      3. 逐 step 缓存 transition 到 episode_buffer（暂不写入 data_store）
      4. episode 结束时：
           a. detect_suboptimal_segments()  → 次优片段
           b. compute_episode_alpha_weights() → α(t)
           c. 带 alpha_weight 的 transition 批量写入 data_store

    include_grasp_penalty: True 时从 env info 中读取 grasp_penalty 并写入 transition
    """
    # ── 验证模式 ──────────────────────────────────────────────────────────
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

    # ── 断点续训起始步 ────────────────────────────────────────────────────
    start_step = (
        int(os.path.basename(
            natsorted(glob.glob(os.path.join(FLAGS.checkpoint_path, "buffer/*.pkl")))[-1]
        )[12:-4]) + 1
        if FLAGS.checkpoint_path and os.path.exists(FLAGS.checkpoint_path)
           and glob.glob(os.path.join(FLAGS.checkpoint_path, "buffer/*.pkl"))
        else 0
    )

    # ── 注册数据存储 ──────────────────────────────────────────────────────
    datastore_dict = {
        "actor_env":       data_store,
        "actor_env_intvn": intvn_data_store,
        "actor_env_pref":  preference_data_store,   # [Module 3]
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

    # ── [Module 1] 加载 Progress Model Runner ────────────────────────────
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
        print_green(
            f"[Module 1] Progress Model 已加载: {FLAGS.progress_model_path}"
        )
    else:
        print_green(
            f"[Module 1] 未提供 --progress_model_path，"
            f"使用 fallback（向前 {FLAGS.suboptimal_window} 步）"
        )

    # ── Episode 级缓存 ────────────────────────────────────────────────────
    transitions:           List[Dict] = []   # 全局，用于周期性落盘
    demo_transitions:      List[Dict] = []   # 干预步，用于 demo_buffer 落盘
    episode_buffer:        List[Dict] = []   # 当前 episode 缓存
    intervention_markers:  List[Dict] = []   # 当前 episode 干预起始点

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

        # ── 采样动作 ────────────────────────────────────────────────────
        with timer.context("sample_actions"):
            if step < config.random_steps:
                # 初始随机探索阶段：均匀随机动作，帮助 replay buffer 覆盖多样化状态空间
                actions = env.action_space.sample()
            else:
                # argmax=False：训练时使用随机策略（带熵），保留探索性
                # argmax=True 仅用于评估模式（当 --eval_checkpoint_step > 0 时）
                sampling_rng, key = jax.random.split(sampling_rng)
                actions = agent.sample_actions(
                    observations=jax.device_put(obs),
                    seed=key,
                    argmax=False,
                )
                actions = np.asarray(jax.device_get(actions))

        # ── 环境交互 ─────────────────────────────────────────────────────
        with timer.context("step_env"):
            # [Module 1/3] 必须在 env.step 之前保存策略动作 a^π(s_t)：
            # env.step 会将 SpaceMouse 输入写入 info["intervene_action_eef"]，
            # 此时 actions 尚未被覆盖，保存的是真实的策略输出。
            # 用途：① Module 3 构成偏好对 (a^h, a^π)；
            #       ② ProgressModelRunner 识别干预步标签（通过 _was_intervened 字段）
            policy_action_saved = actions.copy()

            next_obs, reward, done, truncated, info = env.step(actions)
            if "left" in info:
                info.pop("left")
            if "right" in info:
                info.pop("right")

            # 干预覆盖
            if "intervene_action_eef" in info:
                human_action = info.pop("intervene_action_eef")
                intervention_steps += 1

                if not already_intervened:
                    # ── 仅记录连续干预序列的第一步 t_i ──────────────────
                    # 连续干预期间（already_intervened=True）不重复记录，
                    # 避免 preference buffer 被同一干预事件的重复样本充塡而失去多样性
                    intervention_count += 1
                    t_i = len(episode_buffer)   # episode 内相对索引（非全局 step）
                    intervention_markers.append({
                        "step_idx":      t_i,   # 供 Module 1 detect_suboptimal_segments 使用
                        "obs":           obs,
                        "human_action":  human_action,
                        "policy_action": policy_action_saved,
                    })
                    # [Module 3] 偏好对 (s_{t_i}, a^h, a^π) 写入 PreferenceBufferDataStore
                    # learner 端使用此数据计算对比损失和 A_cf
                    preference_data_store.insert({
                        "observations":   obs,
                        "human_actions":  human_action,
                        "policy_actions": policy_action_saved,
                    })

                already_intervened = True
                actions = human_action
            else:
                already_intervened = False
                # 将 action_scale==0 的维度置 0（与环境执行一致）
                _scale = np.asarray(config.action_scale) if hasattr(config, "action_scale") else None
                if _scale is not None and _scale.ndim > 0:
                    _zero_mask = (_scale == 0)
                    if _zero_mask.any():
                        actions = actions.copy()
                        actions[:len(_zero_mask)][_zero_mask[:len(actions)]] = 0.0

            running_return += reward

            # ── 构造 transition ──────────────────────────────
            # alpha_weight=0.0 仅为占位符：必须等 episode 结束才能确定
            #   次优片段位置（t_a, t_i），所以不能逻步立即写入。
            # _was_intervened 是 episode 内部元数据，不属于 replay buffer 的 schema，
            #   以下划线开头确保写入 buffer 前被过滤掉。
            transition = dict(
                observations=obs,
                actions=actions,
                next_observations=next_obs,
                rewards=reward,
                masks=1.0 - done,
                dones=done,
                alpha_weight=0.0,          # 占位符，episode 结束后由 compute_episode_alpha_weights 填充
                _was_intervened=already_intervened,   # 内部字段，不写入 buffer
            )
            if include_grasp_penalty and "grasp_penalty" in info:
                transition["grasp_penalty"] = info["grasp_penalty"]

            # 缓存到当前 episode（不立即写入 data_store）
            episode_buffer.append(transition)

            # intvn_data_store：干预步立即写入，作为 RLPD 50/50 采样中的 demo 来源。
            # alpha_weight 保持 0.0（设计意图：干预步本身是高质量演示，不需要 Q 値修正）。
            # 立即写入而非等到 episode 结束：干预步的 alpha 印非次优，无需等待片段识别。
            if already_intervened:
                intvn_tr = {k: v for k, v in transition.items()
                            if not k.startswith("_")}
                intvn_data_store.insert(intvn_tr)
                demo_transitions.append(intvn_tr)

            obs = next_obs

            # ── Episode 结束后处理 ────────────────────────────────────
            if done or truncated:
                # [Module 1] 识别次优片段
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

                # [Module 2] 计算每步 alpha_weight
                alpha_weights = compute_episode_alpha_weights(
                    len(episode_buffer), suboptimal_segs, FLAGS.alpha_lambda
                )

                # 批量写入 data_store（含真实 alpha_weight）
                # 必须等 episode 结束才写入的原因：需要 t_i 位置才能计算 alpha_weight。
                # 过滤 "_" 开头的内部字段，用真实位置权重覆盖占位符 0.0。
                for i, tr in enumerate(episode_buffer):
                    tr_to_insert = {k: v for k, v in tr.items()
                                    if not k.startswith("_")}
                    tr_to_insert["alpha_weight"] = float(alpha_weights[i])
                    data_store.insert(tr_to_insert)
                    transitions.append(tr_to_insert)

                # Episode 统计
                info["episode"]["intervention_count"] = intervention_count
                info["episode"]["intervention_steps"] = intervention_steps
                info["episode"]["intervention_rate"] = (
                    intervention_steps / max(len(episode_buffer), 1)
                )
                info["episode"]["n_suboptimal_segs"] = len(suboptimal_segs)
                info["episode"]["mean_alpha"] = float(alpha_weights.mean())
                stats = {"environment": info}
                client.request("send-stats", stats)

                pbar.set_description(f"last return: {running_return:.2f}")
                running_return = 0.0
                intervention_count = 0
                intervention_steps = 0
                already_intervened = False
                episode_buffer = []
                intervention_markers = []

                client.update()

                # 周期性落盘
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
# Learner 循环
# =============================================================================

def learner(
    rng,
    agent,
    replay_buffer: MemoryEfficientReplayBufferDataStore,
    demo_buffer: MemoryEfficientReplayBufferDataStore,
    preference_buffer: PreferenceBufferDataStore,
    wandb_logger=None,
):
    """
    Learner 循环（--learner 模式）。

    相对 train_rlpd.py 的改动：
      1. 新增 preference_buffer，注册到 TrainerServer
      2. preference_buffer 积累足够数据后切换至 agent.update_with_correction()
         - Module 2: corrected_critic_loss_fn（alpha_weight + A_cf 修正）
         - Module 3: contrastive_policy_loss_fn（干预偏好对对比损失）
      3. 不足时退回标准 agent.update()
      4. 根据 agent 类型自动确定 networks_to_update（含/不含 grasp_critic）
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

    # 创建 TrainerServer
    server = TrainerServer(make_trainer_config(), request_callback=stats_callback)
    server.register_data_store("actor_env",       replay_buffer)
    server.register_data_store("actor_env_intvn", demo_buffer)
    server.register_data_store("actor_env_pref",  preference_buffer)   # [Module 3]
    server.start(threaded=True)

    # 等待 replay buffer 积累足够数据
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

    # 50/50 RLPD 采样迭代器（pack_obs=True 与 haoyuan-HILRL 保持一致）
    replay_iterator = replay_buffer.get_iterator(
        sample_args={"batch_size": config.batch_size // 2, "pack_obs": True},
        device=sharding.replicate(),
    )
    demo_iterator = demo_buffer.get_iterator(
        sample_args={"batch_size": config.batch_size // 2, "pack_obs": True},
        device=sharding.replicate(),
    )
    # preference_buffer 数量少，直接 numpy 采样（无需 JAX prefetch）

    timer = Timer()

    # 根据 agent 类型决定需要更新的网络集合（hybrid agent 含 grasp_critic）
    if isinstance(agent, SACAgentHybridSingleArm):
        train_critic_networks_to_update = frozenset({"critic", "grasp_critic"})
        train_networks_to_update = frozenset({"critic", "grasp_critic", "actor", "temperature"})
    else:
        # SACAgent（fixed-gripper，单臂或双臂）
        train_critic_networks_to_update = frozenset({"critic"})
        train_networks_to_update = frozenset({"critic", "actor", "temperature"})

    for step in tqdm.tqdm(
        range(start_step, config.max_steps), dynamic_ncols=True, desc="learner"
    ):
        # 修正更新的触发条件： preference buffer 积累足够的偏好对才切换到修正模式。
        # 训练初期 buffer 为空时退回标准 RLPD，避免无效的零样本计算开销。
        # preference buffer 由 actor 通过 TrainerClient 异步推送。
        use_correction = len(preference_buffer) >= FLAGS.preference_batch_size

        # ── 高 UTD（Update-to-Data）训练：n-1 次 Critic-only + 1 次全局更新 ──
        # 每个 actor step 对应 cta_ratio 次 learner 更新。
        # 前 cta_ratio-1 次只更新 Critic（计算量小，加速训练）；最后 1 次同时更新
        # Critic + Actor + Temperature（保证 Actor 质量）。
        # 若具备修正条件，所有更新均切换至 update_with_correction。
        for _ in range(config.cta_ratio - 1):
            with timer.context("sample_replay_buffer"):
                batch      = next(replay_iterator)
                demo_batch = next(demo_iterator)
                # RLPD 50/50：一半来自 online replay buffer，一半来自 demo buffer
                batch      = concat_batches(batch, demo_batch, axis=0)

            with timer.context("train_critics"):
                if use_correction:
                    pref_batch = preference_buffer.sample(FLAGS.preference_batch_size)
                    if pref_batch is not None:
                        agent, _ = agent.update_with_correction(
                            batch, pref_batch,
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

        # ── 1 次 Critic + Actor + Temperature 联合更新 ───────────────
        with timer.context("train"):
            batch      = next(replay_iterator)
            demo_batch = next(demo_iterator)
            batch      = concat_batches(batch, demo_batch, axis=0)

            if use_correction:
                pref_batch = preference_buffer.sample(FLAGS.preference_batch_size)
                if pref_batch is not None:
                    agent, update_info = agent.update_with_correction(
                        batch, pref_batch,
                        networks_to_update=train_networks_to_update,
                    )
                else:
                    use_correction = False   # 样本不足，回退
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

        # ── 发布网络参数 ──────────────────────────────────────────────
        if step > 0 and step % config.steps_per_update == 0:
            agent = jax.block_until_ready(agent)
            server.publish_network(agent.state.params)

        # ── 日志 ──────────────────────────────────────────────────────
        if step % config.log_period == 0 and wandb_logger:
            wandb_logger.log(update_info, step=step)
            wandb_logger.log({"timer": timer.get_average_times()}, step=step)
            wandb_logger.log({"preference_buffer_size": len(preference_buffer)}, step=step)

        # ── Checkpoint ────────────────────────────────────────────────
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

    # ── 初始化 Agent（支持 fixed-gripper 和 single-arm-learned-gripper）────
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
            f"train_rlpd_hil.py 目前支持 fixed-gripper 和 single-arm-learned-gripper，"
            f"当前 setup_mode={config.setup_mode}。"
        )

    # [Module 3] 注入对比损失系数到 agent.config
    hil_config = dict(agent.config)
    hil_config["contrastive_coef"] = FLAGS.contrastive_coef
    agent = agent.replace(config=hil_config)

    # 分发到所有设备
    agent = jax.device_put(
        jax.tree_util.tree_map(jnp.array, agent), sharding.replicate()
    )

    # ── 加载已有 checkpoint ───────────────────────────────────────────
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
            include_alpha_correction=True,    # [Module 2] 含 alpha_weight 字段
        )
        _wandb_desc = (
            f"{FLAGS.exp_name}__{FLAGS.run_tag}" if FLAGS.run_tag else FLAGS.exp_name
        )
        wandb_logger = make_wandb_logger(
            project="hil-serl",
            description=_wandb_desc,
            debug=FLAGS.debug,
        )
        return replay_buffer, wandb_logger

    # ── Learner 进程 ──────────────────────────────────────────────────
    if FLAGS.learner:
        sampling_rng = jax.device_put(sampling_rng, device=sharding.replicate())
        replay_buffer, wandb_logger = create_replay_buffer_and_wandb_logger()

        # Demo buffer（来自人类演示 pkl，alpha_weight 默认 0.0）
        demo_buffer = MemoryEfficientReplayBufferDataStore(
            env.observation_space,
            env.action_space,
            capacity=config.replay_buffer_capacity,
            image_keys=config.image_keys,
            include_alpha_correction=True,    # schema 保持一致
        )

        # [Module 3] 干预偏好对缓冲区
        preference_buffer = PreferenceBufferDataStore(capacity=10000)

        # 加载 demo 数据（alpha_weight=0.0 默认值）
        assert FLAGS.demo_path is not None
        for path in FLAGS.demo_path:
            with open(path, "rb") as f:
                demo_transitions_loaded = pkl.load(f)
                for transition in demo_transitions_loaded:
                    if "infos" in transition and "grasp_penalty" in transition["infos"]:
                        transition["grasp_penalty"] = transition["infos"]["grasp_penalty"]
                    transition["alpha_weight"] = 0.0   # 优质演示无需修正
                    demo_buffer.insert(transition)
        print_green(f"demo buffer size: {len(demo_buffer)}")
        print_green(f"online buffer size: {len(replay_buffer)}")

        # 恢复 checkpoint 时加载已有 buffer
        if FLAGS.checkpoint_path is not None and os.path.exists(
            os.path.join(FLAGS.checkpoint_path, "buffer")
        ):
            for file in glob.glob(os.path.join(FLAGS.checkpoint_path, "buffer/*.pkl")):
                with open(file, "rb") as f:
                    saved_transitions = pkl.load(f)
                    for transition in saved_transitions:
                        transition.setdefault("alpha_weight", 0.0)
                        replay_buffer.insert(transition)
            print_green(f"Loaded previous buffer. Replay buffer size: {len(replay_buffer)}")

        if FLAGS.checkpoint_path is not None and os.path.exists(
            os.path.join(FLAGS.checkpoint_path, "demo_buffer")
        ):
            for file in glob.glob(
                os.path.join(FLAGS.checkpoint_path, "demo_buffer/*.pkl")
            ):
                with open(file, "rb") as f:
                    saved_transitions = pkl.load(f)
                    for transition in saved_transitions:
                        transition.setdefault("alpha_weight", 0.0)
                        demo_buffer.insert(transition)
            print_green(f"Loaded previous demo buffer. Demo buffer size: {len(demo_buffer)}")

        print_banner(
            "[ACTIVE] HIL-SERL CORRECTION",
            [
                f"alpha_lambda         = {FLAGS.alpha_lambda}",
                f"contrastive_coef (β) = {FLAGS.contrastive_coef}",
                f"preference_batch     = {FLAGS.preference_batch_size}",
                f"suboptimal_window    = {FLAGS.suboptimal_window}",
                f"progress_model       = {FLAGS.progress_model_path or 'fallback'}",
            ],
            color="cyan",
        )

        print_green("starting learner loop")
        learner(
            sampling_rng,
            agent,
            replay_buffer,
            demo_buffer,
            preference_buffer=preference_buffer,
            wandb_logger=wandb_logger,
        )

    # ── Actor 进程 ────────────────────────────────────────────────────
    elif FLAGS.actor:
        sampling_rng = jax.device_put(sampling_rng, sharding.replicate())
        data_store            = QueuedDataStore(20000)
        intvn_data_store      = QueuedDataStore(20000)
        preference_data_store = PreferenceBufferDataStore(capacity=10000)

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
