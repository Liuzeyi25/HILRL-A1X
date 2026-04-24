#!/usr/bin/env python3
"""
evaluate_q_values.py
====================
评估多个 Critic 模型对轨迹的 Q 值估计，并生成可视化对比。

支持：
  1. 多个模型路径（列表）与多条轨迹路径（列表）的组合评估
  2. Q 值曲线可视化与统计对比
  3. 轨迹图像序列可视化
  4. 收益预测 vs 实际收益对比

使用方法
--------
# 单模型、单轨迹
python scripts/evaluate_q_values.py \
    --model_paths "examples/experiments/insert_block/experiments/insert_block/hilserl/0423_baseline_1/checkpoint_16000" \
    --trajectory_paths "path/to/trajectory.pkl" \
    --exp_name insert_block \
    --output_dir ./q_evaluation_results

# 多模型、多轨迹
python scripts/evaluate_q_values.py \
    --model_paths "model1/checkpoint_4000" "model1/checkpoint_8000" \
    --trajectory_paths "traj1.pkl" "traj2.pkl" \
    --exp_name insert_block \
    --output_dir ./q_evaluation_results \
    --save_video_samples

命令行参数
---------
--model_paths         : 一个或多个 checkpoint 路径（空格分隔），路径格式：
                        /path/to/checkpoint_STEP（内含 checkpoint/params.pkl）
--trajectory_paths    : 一个或多个轨迹 pkl 路径（空格分隔），每个 pkl 包含
                        transitions 列表或 {episodes: []}
--exp_name            : 实验名（与 train_rlpd_hil.py 一致）
--output_dir          : 输出目录
--n_frames_per_traj   : 每条轨迹最多可视化的帧数（默认 500）
--ensemble_agg        : Q 聚合方式 (min/mean)，默认 min
--use_target_critic   : 是否用 target critic
--save_video_samples  : 保存轨迹图像序列（PNG 帧）
--chunk_size          : 批推理大小
--seed                : 随机种子
"""

import argparse
import glob
import os
import pickle as pkl
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import imageio.v2 as imageio
import jax
import jax.numpy as jnp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
from flax.training import checkpoints

# ── 加入工程路径 ──────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "examples"))
sys.path.insert(0, str(REPO_ROOT / "serl_launcher"))

from experiments.mappings import CONFIG_MAPPING
from serl_launcher.utils.launcher import (
    make_sac_pixel_agent,
    make_sac_pixel_agent_hybrid_single_arm,
)


# =============================================================================
# 工具函数：轨迹加载与预处理
# =============================================================================

def load_trajectory_from_pkl(traj_path: str) -> List[Dict]:
    """
    从 pkl 加载轨迹。支持两种格式：
      1. 直接 list[dict]（transitions）
      2. {"episodes": [list[dict], ...]} 或 {"trajectories": [...]}

    返回：单条轨迹（list[dict]），按时间顺序排列。
    """
    with open(traj_path, "rb") as f:
        data = pkl.load(f)

    if isinstance(data, list):
        # 格式 1：直接是 transitions 列表
        trajectory = data
    elif isinstance(data, dict):
        # 格式 2：{"episodes": [...]} 或 {"trajectories": [...]}
        if "episodes" in data:
            episodes = data["episodes"]
            # 如果 episodes 是 list of episodes，取第一条
            if isinstance(episodes, list) and len(episodes) > 0:
                if isinstance(episodes[0], list):
                    trajectory = episodes[0]  # 第一个 episode
                else:
                    trajectory = episodes  # 本身就是一条轨迹
            else:
                trajectory = episodes
        elif "trajectories" in data:
            trajectories = data["trajectories"]
            trajectory = trajectories[0] if isinstance(trajectories, list) else trajectories
        else:
            # 默认当作 transitions 列表
            trajectory = list(data.values())[0] if data else []
    else:
        raise ValueError(f"未知的轨迹格式: {type(data)}")

    print(f"  已加载轨迹: {traj_path} (长度={len(trajectory)})")
    return trajectory


def load_agent_from_checkpoint(checkpoint_path: str, config, 
                               sample_obs, sample_action, 
                               seed: int = 42):
    """
    从 checkpoint 加载 agent。

    checkpoint_path: /path/to/checkpoint_STEP 或类似格式
    """
    setup_mode = config.setup_mode
    image_keys = list(config.image_keys)

    resolved_root, resolved_step = resolve_checkpoint_path(checkpoint_path)
    print(f"  Loading checkpoint from root={resolved_root}, step={resolved_step}")

    if setup_mode in ("single-arm-fixed-gripper", "dual-arm-fixed-gripper"):
        agent = make_sac_pixel_agent(
            seed=seed,
            sample_obs=sample_obs,
            sample_action=sample_action,
            image_keys=image_keys,
            encoder_type=config.encoder_type,
            discount=config.discount,
        )
    elif setup_mode == "single-arm-learned-gripper":
        agent = make_sac_pixel_agent_hybrid_single_arm(
            seed=seed,
            sample_obs=sample_obs,
            sample_action=sample_action,
            image_keys=image_keys,
            encoder_type=config.encoder_type,
            discount=config.discount,
        )
    else:
        raise NotImplementedError(f"setup_mode={setup_mode} 不支持")

    # 尝试加载 checkpoint（显式 step，避免路径歧义导致静默回退到随机初始化）
    try:
        ckpt_state = checkpoints.restore_checkpoint(resolved_root, agent.state, step=resolved_step)

        if not _is_checkpoint_state_loaded(agent.state, ckpt_state):
            raise RuntimeError(
                f"Checkpoint appears not loaded (state unchanged). root={resolved_root}, step={resolved_step}"
            )

        agent = agent.replace(state=ckpt_state)
        print(f"  ✓ Loaded checkpoint step={resolved_step}")
    except Exception as e:
        print(f"  ✗ Failed to load checkpoint: {e}")
        raise

    return agent


def resolve_checkpoint_path(checkpoint_path: str) -> Tuple[str, int]:
    """
    Resolve user checkpoint path into (checkpoint_root_dir, step).

    Supports:
      1) /path/to/run_dir                -> latest step in run_dir
      2) /path/to/run_dir/checkpoint_8000 -> root=/path/to/run_dir, step=8000
    """
    abs_path = os.path.abspath(checkpoint_path)
    base = os.path.basename(abs_path)
    m = re.fullmatch(r"checkpoint_(\d+)", base)

    if m is not None:
        step = int(m.group(1))
        root = os.path.dirname(abs_path)
        if not os.path.exists(os.path.join(root, base)):
            raise FileNotFoundError(f"Checkpoint path does not exist: {abs_path}")
        return root, step

    if not os.path.isdir(abs_path):
        raise FileNotFoundError(f"Checkpoint directory not found: {abs_path}")

    latest = checkpoints.latest_checkpoint(abs_path)
    if latest is None:
        raise FileNotFoundError(f"No checkpoint_* found under: {abs_path}")

    latest_base = os.path.basename(latest)
    m_latest = re.fullmatch(r"checkpoint_(\d+)", latest_base)
    if m_latest is None:
        raise RuntimeError(f"Unexpected latest checkpoint name: {latest}")

    return abs_path, int(m_latest.group(1))


def _is_checkpoint_state_loaded(init_state, loaded_state) -> bool:
    """
    Heuristic check: at least one actor parameter array differs.
    Prevents silent fallback where restore_checkpoint returns target unchanged.
    """
    try:
        init_actor = init_state.params["modules_actor"]
        load_actor = loaded_state.params["modules_actor"]
        init_leaves = jax.tree_util.tree_leaves(init_actor)
        load_leaves = jax.tree_util.tree_leaves(load_actor)
        if len(init_leaves) != len(load_leaves) or len(init_leaves) == 0:
            return False
        for a, b in zip(init_leaves, load_leaves):
            if not np.array_equal(np.asarray(a), np.asarray(b)):
                return True
        return False
    except Exception:
        return True


def build_unique_model_name(model_path: str, used_names: set) -> str:
    """
    Build a unique model display name from path to avoid collisions.

    Example:
      .../hilserl/0423_baseline_1/checkpoint_6000
      -> hilserl/0423_baseline_1/checkpoint_6000
    """
    path_obj = Path(model_path)
    base = path_obj.name
    parent = path_obj.parent.name
    grandparent = path_obj.parent.parent.name if path_obj.parent.parent else "root"
    candidate = f"{grandparent}/{parent}/{base}"

    if candidate not in used_names:
        used_names.add(candidate)
        return candidate

    i = 2
    while f"{candidate}#{i}" in used_names:
        i += 1
    unique_name = f"{candidate}#{i}"
    used_names.add(unique_name)
    return unique_name


def build_obs_batch(transitions: List[Dict], image_keys: List[str]) -> Dict:
    """
    将 transition list 转为批处理 obs dict，形状 (T, ...)。
    """
    obs_list = [tr["observations"] for tr in transitions]
    batch = {}
    if not obs_list:
        return batch

    first = obs_list[0]
    for key in first.keys():
        vals = [o[key] for o in obs_list]
        batch[key] = np.stack(vals, axis=0)  # (T, ...)

    return batch


def estimate_q_values(agent, obs_batch: Dict, actions: np.ndarray,
                      rng: jax.random.PRNGKey,
                      use_target: bool = False,
                      ensemble_agg: str = "min",
                      fix_gripper: bool = True,
                      chunk_size: int = 32) -> np.ndarray:
    """
    用 critic 逐批推理 Q 值。

    返回：(T,) 数组，每个时间步的 Q 值。
    """
    T = actions.shape[0]
    if T == 0:
        return np.array([])

    q_values = []

    for start in range(0, T, chunk_size):
        end = min(start + chunk_size, T)
        obs_chunk = jax.tree_util.tree_map(
            lambda x: jax.device_put(x[start:end]), obs_batch
        )
        act_chunk = jax.device_put(actions[start:end])

        if fix_gripper and act_chunk.shape[-1] > 0:
            # 假设最后一维是夹爪，设为 0
            act_chunk = act_chunk.at[..., -1].set(0.0)

        rng, key = jax.random.split(rng)
        if use_target:
            qs = agent.forward_target_critic(obs_chunk, act_chunk, rng=key)
        else:
            qs = agent.forward_critic(obs_chunk, act_chunk, rng=key, train=False)

        qs = np.asarray(qs)  # (ensemble_size, chunk_size)
        if ensemble_agg == "min":
            q_chunk = qs.min(axis=0)
        else:  # mean
            q_chunk = qs.mean(axis=0)

        q_values.append(q_chunk)

    return np.concatenate(q_values, axis=0)  # (T,)


def compute_discounted_returns(rewards: np.ndarray, 
                               dones: np.ndarray,
                               gamma: float = 0.99) -> np.ndarray:
    """
    逆向计算折现回报。

    rewards: (T,)
    dones: (T,) bool
    返回：(T,) 每个时刻的未来折现回报
    """
    T = len(rewards)
    returns = np.zeros(T)
    cumsum = 0.0

    for t in range(T - 1, -1, -1):
        cumsum = rewards[t] + (1 - dones[t]) * gamma * cumsum
        returns[t] = cumsum

    return returns


def _to_uint8_image(img: np.ndarray) -> np.ndarray:
    """Normalize image to uint8 HWC for display."""
    img = np.asarray(img)
    if img.ndim == 4 and img.shape[0] <= 10:
        img = img[-1]

    if img.ndim == 2:
        img = np.stack([img, img, img], axis=-1)

    if img.ndim == 3 and img.shape[-1] == 1:
        img = np.repeat(img, 3, axis=-1)

    if img.ndim != 3 or img.shape[-1] != 3:
        raise ValueError(f"Unsupported image shape: {img.shape}")

    if img.dtype != np.uint8:
        if img.max() <= 1.0:
            img = (img * 255.0).clip(0, 255).astype(np.uint8)
        else:
            img = img.clip(0, 255).astype(np.uint8)
    return img


def resolve_camera_keys(first_obs: Dict, config_image_keys: List[str]) -> Tuple[str, str]:
    """Pick wrist and third-person camera keys from observation dict."""
    keys = list(first_obs.keys())

    wrist_candidates = [k for k in keys if "wrist" in k.lower()]
    third_candidates = [k for k in keys if any(tag in k.lower() for tag in ["side", "third", "agent", "front", "policy"])]

    if not wrist_candidates:
        wrist_candidates = [k for k in config_image_keys if k in keys and "wrist" in k.lower()]
    if not third_candidates:
        third_candidates = [k for k in config_image_keys if k in keys and k not in wrist_candidates]

    if not wrist_candidates:
        wrist_candidates = [k for k in keys if np.asarray(first_obs[k]).ndim >= 3]
    if not third_candidates:
        third_candidates = [k for k in keys if np.asarray(first_obs[k]).ndim >= 3 and k not in wrist_candidates]

    if not wrist_candidates or not third_candidates:
        raise ValueError(f"Cannot resolve two camera keys from observation keys: {keys}")

    return wrist_candidates[0], third_candidates[0]


def extract_intervention_flags(trajectory: List[Dict]) -> np.ndarray:
    """Best-effort extraction of human intervention flags from trajectory."""
    flags = []
    for tr in trajectory:
        intervened = False

        if "labels" in tr:
            try:
                intervened = int(tr["labels"]) == 2
            except Exception:
                intervened = False

        for k in ["is_intervention", "intervened", "human_intervened", "gello_intervened"]:
            if k in tr:
                intervened = intervened or bool(tr[k])

        infos = tr.get("infos", {}) if isinstance(tr.get("infos", {}), dict) else {}
        for k in ["gello_intervened", "human_intervened"]:
            if k in infos:
                intervened = intervened or bool(infos[k])

        flags.append(intervened)

    return np.asarray(flags, dtype=bool)


def extract_suboptimal_flags(trajectory: List[Dict]) -> np.ndarray:
    """Use alpha_weight > 0 as suboptimal marker when available."""
    flags = []
    for tr in trajectory:
        alpha = tr.get("alpha_weight", 0.0)
        try:
            flags.append(float(alpha) > 1e-8)
        except Exception:
            flags.append(False)
    return np.asarray(flags, dtype=bool)


def make_overlay_video(
    trajectory: List[Dict],
    wrist_key: str,
    third_key: str,
    q_a: np.ndarray,
    q_b: np.ndarray,
    returns: np.ndarray,
    model_name_a: str,
    model_name_b: str,
    suboptimal_flags: np.ndarray,
    intervention_flags: np.ndarray,
    output_video_path: str,
    fps: int,
):
    """Render one mp4 with two cameras + status + Q/return plot."""
    T = len(trajectory)
    if T == 0:
        raise ValueError("Empty trajectory, cannot render video")

    fig = plt.figure(figsize=(14, 8), dpi=100)
    gs = fig.add_gridspec(2, 2, height_ratios=[2.2, 1.3])

    ax_wrist = fig.add_subplot(gs[0, 0])
    ax_third = fig.add_subplot(gs[0, 1])
    ax_curve = fig.add_subplot(gs[1, :])

    t_axis = np.arange(T)
    y_all = np.concatenate([q_a, q_b, returns], axis=0)
    y_min = float(np.nanmin(y_all))
    y_max = float(np.nanmax(y_all))
    if abs(y_max - y_min) < 1e-6:
        y_min -= 1.0
        y_max += 1.0

    with imageio.get_writer(output_video_path, fps=fps, codec="libx264", quality=8) as writer:
        for t in range(T):
            obs = trajectory[t]["observations"]
            wrist_img = _to_uint8_image(obs[wrist_key])
            third_img = _to_uint8_image(obs[third_key])

            ax_wrist.clear()
            ax_third.clear()
            ax_curve.clear()

            ax_wrist.imshow(wrist_img)
            ax_wrist.set_title(f"Wrist Camera ({wrist_key})")
            ax_wrist.axis("off")

            ax_third.imshow(third_img)
            ax_third.set_title(f"Third-person Camera ({third_key})")
            ax_third.axis("off")

            status_items = []
            if suboptimal_flags[t]:
                status_items.append("SUBOPTIMAL")
            if intervention_flags[t]:
                status_items.append("HUMAN_INTERVENTION")
            status_text = " | ".join(status_items) if status_items else "NORMAL"
            status_color = "red" if len(status_items) > 0 else "green"

            fig.suptitle(
                f"t={t}/{T-1} | {status_text}",
                color=status_color,
                fontsize=14,
                fontweight="bold",
            )

            ax_curve.plot(t_axis, returns, color="black", linewidth=2.0, alpha=0.9, label="Discounted Return")
            ax_curve.plot(t_axis, q_a, color=cm.tab10(0), linewidth=1.6, alpha=0.9, label=f"Q-A ({model_name_a})")
            ax_curve.plot(t_axis, q_b, color=cm.tab10(1), linewidth=1.6, alpha=0.9, label=f"Q-B ({model_name_b})")

            ax_curve.axvline(t, color="magenta", linestyle="--", linewidth=1.2, alpha=0.85)
            ax_curve.scatter([t], [returns[t]], color="black", s=30)
            ax_curve.scatter([t], [q_a[t]], color=cm.tab10(0), s=24)
            ax_curve.scatter([t], [q_b[t]], color=cm.tab10(1), s=24)

            if suboptimal_flags[t]:
                ax_curve.axvspan(max(0, t - 0.5), min(T - 1, t + 0.5), color="orange", alpha=0.18)
            if intervention_flags[t]:
                ax_curve.axvspan(max(0, t - 0.5), min(T - 1, t + 0.5), color="red", alpha=0.15)

            ax_curve.set_xlim(0, T - 1)
            ax_curve.set_ylim(y_min, y_max)
            ax_curve.set_xlabel("Timestep")
            ax_curve.set_ylabel("Value")
            ax_curve.set_title("Two-Critic Q Values + Discounted Return")
            ax_curve.grid(True, alpha=0.3)
            ax_curve.legend(loc="best", fontsize=9)

            fig.tight_layout(rect=[0, 0, 1, 0.95])
            fig.canvas.draw()

            frame = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
            frame = frame.reshape(fig.canvas.get_width_height()[1], fig.canvas.get_width_height()[0], 4)
            frame = frame[:, :, :3]
            writer.append_data(frame)

    plt.close(fig)


# =============================================================================
# 可视化函数
# =============================================================================

def plot_q_comparison(q_dict: Dict[str, np.ndarray],
                      returns: np.ndarray,
                      trajectory_idx: int,
                      output_dir: str,
                      gamma: float = 0.99):
    """
    绘制多个模型的 Q 值曲线对比。

    q_dict: {model_name -> (T,) Q 值数组}
    returns: (T,) 实际折现回报
    trajectory_idx: 轨迹序号（用于文件名）
    """
    fig, axes = plt.subplots(2, 1, figsize=(14, 8))

    # ── Q 值曲线 ──────────────────────────────────────────────
    ax = axes[0]
    t_steps = np.arange(len(returns))

    ax.plot(t_steps, returns, "k-", linewidth=2, label="Discounted Return", alpha=0.8)

    colors = cm.tab10(np.linspace(0, 1, len(q_dict)))
    for (model_name, q_vals), color in zip(q_dict.items(), colors):
        ax.plot(t_steps, q_vals, "-", linewidth=1.5, label=f"Q ({model_name})", 
                color=color, alpha=0.7)

    ax.set_xlabel("Timestep", fontsize=11)
    ax.set_ylabel("Q Value / Discounted Return", fontsize=11)
    ax.set_title(f"Trajectory #{trajectory_idx} - Q Value Comparison", fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=9)

    # ── Q 值误差 ─────────────────────────────────────────────
    ax = axes[1]
    for (model_name, q_vals), color in zip(q_dict.items(), colors):
        error = np.abs(q_vals - returns)
        ax.plot(t_steps, error, "-", linewidth=1.5, label=f"Error ({model_name})",
                color=color, alpha=0.7)

    ax.set_xlabel("Timestep", fontsize=11)
    ax.set_ylabel("|Q - Return|", fontsize=11)
    ax.set_title("Q Estimation Error", fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=9)
    ax.set_yscale("log")

    plt.tight_layout()
    out_file = os.path.join(output_dir, f"q_comparison_traj{trajectory_idx}.png")
    plt.savefig(out_file, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ 保存对比图: {out_file}")


def plot_q_statistics(all_results: Dict[str, Dict[int, Dict]],
                      output_dir: str):
    """
    绘制不同模型、不同轨迹的 Q 值统计信息。

    all_results: {model_name -> {traj_idx -> {metrics}}}
                 其中 metrics = {"mae": float, "rmse": float, ...}
    """
    model_names = list(all_results.keys())
    n_models = len(model_names)
    n_trajs = len(all_results[model_names[0]])

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # ── MAE (Mean Absolute Error) ──────────────────────────────
    ax = axes[0]
    x = np.arange(n_trajs)
    width = 0.8 / n_models

    for i, model_name in enumerate(model_names):
        maes = [all_results[model_name][j].get("mae", 0) for j in range(n_trajs)]
        ax.bar(x + i * width, maes, width, label=model_name, alpha=0.8)

    ax.set_xlabel("Trajectory Index", fontsize=11)
    ax.set_ylabel("MAE (Q - Return)", fontsize=11)
    ax.set_title("Q Estimation Error (MAE)", fontsize=12, fontweight="bold")
    ax.set_xticks(x + width * (n_models - 1) / 2)
    ax.set_xticklabels([f"#{j}" for j in range(n_trajs)])
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")

    # ── RMSE (Root Mean Square Error) ──────────────────────────
    ax = axes[1]
    for i, model_name in enumerate(model_names):
        rmses = [all_results[model_name][j].get("rmse", 0) for j in range(n_trajs)]
        ax.bar(x + i * width, rmses, width, label=model_name, alpha=0.8)

    ax.set_xlabel("Trajectory Index", fontsize=11)
    ax.set_ylabel("RMSE", fontsize=11)
    ax.set_title("Q Estimation Error (RMSE)", fontsize=12, fontweight="bold")
    ax.set_xticks(x + width * (n_models - 1) / 2)
    ax.set_xticklabels([f"#{j}" for j in range(n_trajs)])
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    out_file = os.path.join(output_dir, "q_statistics.png")
    plt.savefig(out_file, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ 保存统计图: {out_file}")


def plot_trajectory_images(trajectory: List[Dict],
                          trajectory_idx: int,
                          output_dir: str,
                          n_frames: int = 50,
                          image_key: str = "agentview_rgb"):
    """
    提取并保存轨迹的图像帧（采样）。

    trajectory: transitions list
    image_key: 图像键名（如 "agentview_rgb", "side_rgb"）
    n_frames: 最多保存多少帧
    """
    frame_dir = os.path.join(output_dir, f"trajectory_{trajectory_idx}_frames")
    os.makedirs(frame_dir, exist_ok=True)

    T = len(trajectory)
    sample_indices = np.linspace(0, T - 1, min(n_frames, T), dtype=int)

    for idx_in_sample, step_idx in enumerate(sample_indices):
        tr = trajectory[step_idx]
        obs = tr.get("observations", {})

        # 尝试提取图像
        img = None
        if image_key in obs:
            img = obs[image_key]
        else:
            # fallback：查找包含 "rgb" 的第一个键
            for k in obs.keys():
                if "rgb" in k.lower():
                    img = obs[k]
                    break

        if img is not None:
            img = np.asarray(img)
            # 处理可能的堆叠维度 (stack, H, W, C) -> (H, W, C)
            if img.ndim == 4 and img.shape[0] <= 10:  # stack 维度通常很小
                img = img[-1]  # 取最后一帧
            if img.ndim == 3 and img.shape[-1] == 3:
                img = (img * 255).astype(np.uint8) if img.max() <= 1 else img.astype(np.uint8)
                fig, ax = plt.subplots(figsize=(6, 6))
                ax.imshow(img)
                ax.set_title(f"Frame {step_idx}/{T}")
                ax.axis("off")
                plt.tight_layout()
                out_file = os.path.join(frame_dir, f"frame_{idx_in_sample:03d}.png")
                plt.savefig(out_file, dpi=80, bbox_inches="tight")
                plt.close()

    print(f"  ✓ 轨迹 #{trajectory_idx} 图像帧保存至: {frame_dir} ({len(sample_indices)} 帧)")


# =============================================================================
# 主程序
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Two-checkpoint single-trajectory Q video generator")
    parser.add_argument("--checkpoint_a", required=True, help="Path to checkpoint A")
    parser.add_argument("--checkpoint_b", required=True, help="Path to checkpoint B")
    parser.add_argument("--trajectory_path", required=True, help="Path to one trajectory pkl")
    parser.add_argument("--exp_name", required=True, help="Experiment name (CONFIG_MAPPING key)")
    parser.add_argument("--output_dir", default="./q_video_output", help="Output directory")
    parser.add_argument("--output_name", default="q_trajectory_video.mp4", help="Output video filename")
    parser.add_argument("--fps", type=int, default=10, help="Output video fps")
    parser.add_argument("--ensemble_agg", default="min", choices=["min", "mean"], help="Q ensemble aggregation")
    parser.add_argument("--use_target_critic", action="store_true", help="Use target critic")
    parser.add_argument("--chunk_size", type=int, default=16, help="Inference chunk size")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.exp_name not in CONFIG_MAPPING:
        raise ValueError(f"exp_name '{args.exp_name}' is not in CONFIG_MAPPING")

    checkpoint_a = os.path.abspath(args.checkpoint_a)
    checkpoint_b = os.path.abspath(args.checkpoint_b)
    trajectory_path = os.path.abspath(args.trajectory_path)
    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 80)
    print("Two-checkpoint Q video generation")
    print("=" * 80)
    print(f"[Input] checkpoint_a={checkpoint_a}")
    print(f"[Input] checkpoint_b={checkpoint_b}")
    print(f"[Input] trajectory={trajectory_path}")

    config = CONFIG_MAPPING[args.exp_name]()
    setup_mode = config.setup_mode
    image_keys = list(config.image_keys)
    fix_gripper = "fixed-gripper" in setup_mode

    print(f"[Config] exp_name={args.exp_name}")
    print(f"[Config] setup_mode={setup_mode}")
    print(f"[Config] image_keys={image_keys}")

    trajectory = load_trajectory_from_pkl(trajectory_path)
    if len(trajectory) == 0:
        raise ValueError("Trajectory is empty")

    print("[Env] Build fake environment for sample spaces...")
    env = config.get_environment(
        fake_env=True,
        save_video=False,
        classifier=False,
        stack_obs_num=2,
    )
    sample_obs = env.observation_space.sample()
    sample_action = env.action_space.sample()
    try:
        env.close()
    except Exception as e:
        print(f"[Warning] env.close() failed (non-fatal): {type(e).__name__}")

    print("[Model] Loading checkpoint A...")
    agent_a = load_agent_from_checkpoint(checkpoint_a, config, sample_obs, sample_action, args.seed)
    print("[Model] Loading checkpoint B...")
    agent_b = load_agent_from_checkpoint(checkpoint_b, config, sample_obs, sample_action, args.seed)

    model_name_a = Path(checkpoint_a).parent.name + "/" + Path(checkpoint_a).name
    model_name_b = Path(checkpoint_b).parent.name + "/" + Path(checkpoint_b).name

    obs_batch = build_obs_batch(trajectory, image_keys)
    actions = np.stack([tr["actions"] for tr in trajectory], axis=0)
    rewards = np.array([tr.get("rewards", 0.0) for tr in trajectory], dtype=np.float32)
    dones = np.array([tr.get("dones", False) for tr in trajectory], dtype=np.float32)

    rng = jax.random.PRNGKey(args.seed)
    rng, key_a = jax.random.split(rng)
    rng, key_b = jax.random.split(rng)

    print("[Inference] Estimating Q for checkpoint A...")
    q_a = estimate_q_values(
        agent_a, obs_batch, actions, key_a,
        use_target=args.use_target_critic,
        ensemble_agg=args.ensemble_agg,
        fix_gripper=fix_gripper,
        chunk_size=args.chunk_size,
    )
    print("[Inference] Estimating Q for checkpoint B...")
    q_b = estimate_q_values(
        agent_b, obs_batch, actions, key_b,
        use_target=args.use_target_critic,
        ensemble_agg=args.ensemble_agg,
        fix_gripper=fix_gripper,
        chunk_size=args.chunk_size,
    )

    returns = compute_discounted_returns(rewards, dones, gamma=args.gamma)

    suboptimal_flags = extract_suboptimal_flags(trajectory)
    intervention_flags = extract_intervention_flags(trajectory)

    print(f"[Marker] suboptimal frames: {int(suboptimal_flags.sum())}/{len(suboptimal_flags)}")
    print(f"[Marker] intervention frames: {int(intervention_flags.sum())}/{len(intervention_flags)}")

    first_obs = trajectory[0]["observations"]
    wrist_key, third_key = resolve_camera_keys(first_obs, image_keys)
    print(f"[Camera] wrist={wrist_key}, third={third_key}")

    output_video_path = os.path.join(args.output_dir, args.output_name)
    print(f"[Render] Writing video to {output_video_path} ...")
    make_overlay_video(
        trajectory=trajectory,
        wrist_key=wrist_key,
        third_key=third_key,
        q_a=q_a,
        q_b=q_b,
        returns=returns,
        model_name_a=model_name_a,
        model_name_b=model_name_b,
        suboptimal_flags=suboptimal_flags,
        intervention_flags=intervention_flags,
        output_video_path=output_video_path,
        fps=args.fps,
    )

    print("=" * 80)
    print(f"Done. Video saved to: {output_video_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()
