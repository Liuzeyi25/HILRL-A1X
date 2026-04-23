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
import sys
from pathlib import Path
from typing import Dict, List, Tuple

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

    # 查找 checkpoint 内部的参数文件
    # 通常 checkpoint 结构为：checkpoint_STEP/ 下有 checkpoint 文件夹或直接是参数
    ckpt_dir = os.path.abspath(checkpoint_path)

    print(f"  从 {ckpt_dir} 加载 checkpoint...")

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

    # 尝试加载 checkpoint
    try:
        ckpt_state = checkpoints.restore_checkpoint(ckpt_dir, agent.state)
        agent = agent.replace(state=ckpt_state)
        print(f"  ✓ 成功加载 checkpoint")
    except Exception as e:
        print(f"  ✗ 加载 checkpoint 失败: {e}")
        raise

    return agent


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

    ax.plot(t_steps, returns, "k-", linewidth=2, label="实际折现回报", alpha=0.8)

    colors = cm.tab10(np.linspace(0, 1, len(q_dict)))
    for (model_name, q_vals), color in zip(q_dict.items(), colors):
        ax.plot(t_steps, q_vals, "-", linewidth=1.5, label=f"Q值({model_name})", 
                color=color, alpha=0.7)

    ax.set_xlabel("时间步", fontsize=11)
    ax.set_ylabel("Q值 / 折现回报", fontsize=11)
    ax.set_title(f"轨迹 #{trajectory_idx} - Q 值曲线对比", fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=9)

    # ── Q 值误差 ─────────────────────────────────────────────
    ax = axes[1]
    for (model_name, q_vals), color in zip(q_dict.items(), colors):
        error = np.abs(q_vals - returns)
        ax.plot(t_steps, error, "-", linewidth=1.5, label=f"误差({model_name})",
                color=color, alpha=0.7)

    ax.set_xlabel("时间步", fontsize=11)
    ax.set_ylabel("|Q值 - 实际回报|", fontsize=11)
    ax.set_title("Q值估计误差", fontsize=12, fontweight="bold")
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

    ax.set_xlabel("轨迹索引", fontsize=11)
    ax.set_ylabel("MAE (Q值 - 回报)", fontsize=11)
    ax.set_title("Q值估计误差（MAE）", fontsize=12, fontweight="bold")
    ax.set_xticks(x + width * (n_models - 1) / 2)
    ax.set_xticklabels([f"#{j}" for j in range(n_trajs)])
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")

    # ── RMSE (Root Mean Square Error) ──────────────────────────
    ax = axes[1]
    for i, model_name in enumerate(model_names):
        rmses = [all_results[model_name][j].get("rmse", 0) for j in range(n_trajs)]
        ax.bar(x + i * width, rmses, width, label=model_name, alpha=0.8)

    ax.set_xlabel("轨迹索引", fontsize=11)
    ax.set_ylabel("RMSE", fontsize=11)
    ax.set_title("Q值估计误差（RMSE）", fontsize=12, fontweight="bold")
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
    parser = argparse.ArgumentParser(description="多模型 Q 值评估工具")

    parser.add_argument("--model_paths", nargs="+", required=True,
                        help="模型 checkpoint 路径列表（空格分隔）")
    parser.add_argument("--trajectory_paths", nargs="+", required=True,
                        help="轨迹 pkl 路径列表（空格分隔）")
    parser.add_argument("--exp_name", required=True,
                        help="实验名（CONFIG_MAPPING 键）")
    parser.add_argument("--output_dir", default="./q_evaluation_results",
                        help="输出目录")
    parser.add_argument("--n_frames_per_traj", type=int, default=50,
                        help="每条轨迹保存的图像帧数")
    parser.add_argument("--ensemble_agg", default="min", choices=["min", "mean"],
                        help="Q 值聚合方式")
    parser.add_argument("--use_target_critic", action="store_true",
                        help="使用 target critic")
    parser.add_argument("--save_video_samples", action="store_true",
                        help="保存轨迹图像帧")
    parser.add_argument("--chunk_size", type=int, default=16,
                        help="批推理大小")
    parser.add_argument("--gamma", type=float, default=0.99,
                        help="折现因子")
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    # ── 路径检查与标准化 ────────────────────────────────────────
    model_paths = [os.path.abspath(p) for p in args.model_paths]
    trajectory_paths = [os.path.abspath(p) for p in args.trajectory_paths]
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"多模型 Q 值评估工具")
    print(f"{'='*70}")
    print(f"[输入] 模型数={len(model_paths)}, 轨迹数={len(trajectory_paths)}")
    print(f"[输出] {args.output_dir}")
    print()

    # ── 加载配置与环境 ──────────────────────────────────────────
    assert args.exp_name in CONFIG_MAPPING, \
        f"exp_name '{args.exp_name}' 不在 CONFIG_MAPPING 中"
    config = CONFIG_MAPPING[args.exp_name]()
    setup_mode = config.setup_mode
    image_keys = list(config.image_keys)
    fix_gripper = "fixed-gripper" in setup_mode

    print(f"[配置] exp_name={args.exp_name}")
    print(f"       setup_mode={setup_mode}")
    print(f"       image_keys={image_keys}")
    print(f"       fix_gripper={fix_gripper}")
    print()

    # ── 加载轨迹数据 ────────────────────────────────────────────
    print(f"[轨迹] 正在加载 {len(trajectory_paths)} 条轨迹...")
    trajectories = []
    for traj_path in trajectory_paths:
        traj = load_trajectory_from_pkl(traj_path)
        trajectories.append(traj)
    print(f"  共加载 {len(trajectories)} 条轨迹")
    print()

    # ── 构建 dummy 环境 ──────────────────────────────────────────
    print("[环境] 初始化环境以获取 obs/action 形状...")
    env = config.get_environment(
        fake_env=True,
        save_video=False,
        classifier=False,
        stack_obs_num=2,
    )
    sample_obs = env.observation_space.sample()
    sample_action = env.action_space.sample()
    
    # 安全关闭环境（可能不完全初始化）
    try:
        env.close()
    except Exception as e:
        print(f"  [警告] 环境关闭失败（非致命）: {type(e).__name__}")
    
    print(f"  ✓ obs_space={sample_obs.keys()}, action_space={sample_action.shape}")
    print()

    # ── 逐模型加载 agent ────────────────────────────────────────
    print(f"[模型] 加载 {len(model_paths)} 个 checkpoint...")
    agents = {}
    rng = jax.random.PRNGKey(args.seed)

    for model_path in model_paths:
        try:
            agent = load_agent_from_checkpoint(
                model_path, config,
                sample_obs, sample_action,
                seed=args.seed,
            )
            model_name = os.path.basename(model_path)
            agents[model_name] = agent
        except Exception as e:
            print(f"  ✗ 加载 {model_path} 失败: {e}")
            continue

    print(f"  ✓ 成功加载 {len(agents)} 个模型")
    print()

    # ── 逐轨迹、逐模型评估 Q 值 ─────────────────────────────────
    print(f"[推理] 评估 {len(trajectories)} 条轨迹的 Q 值...")
    all_results = {model_name: {} for model_name in agents.keys()}

    for traj_idx, trajectory in enumerate(trajectories):
        print(f"\n  轨迹 #{traj_idx} (长度={len(trajectory)}):")

        # 提取 observations, actions, rewards, dones
        obs_batch = build_obs_batch(trajectory, image_keys)
        actions = np.stack([tr["actions"] for tr in trajectory], axis=0)
        rewards = np.array([tr.get("rewards", 0) for tr in trajectory])
        dones = np.array([tr.get("dones", False) for tr in trajectory])

        # 计算真实折现回报
        returns = compute_discounted_returns(rewards, dones, gamma=args.gamma)

        # 逐模型评估
        q_dict = {}
        for model_name, agent in agents.items():
            rng, key = jax.random.split(rng)
            q_vals = estimate_q_values(
                agent, obs_batch, actions, key,
                use_target=args.use_target_critic,
                ensemble_agg=args.ensemble_agg,
                fix_gripper=fix_gripper,
                chunk_size=args.chunk_size,
            )
            q_dict[model_name] = q_vals

            # 计算指标
            mae = np.mean(np.abs(q_vals - returns))
            rmse = np.sqrt(np.mean((q_vals - returns) ** 2))
            pearson_r = np.corrcoef(q_vals, returns)[0, 1]

            all_results[model_name][traj_idx] = {
                "mae": mae,
                "rmse": rmse,
                "pearson_r": pearson_r,
                "mean_q": q_vals.mean(),
                "mean_return": returns.mean(),
            }

            print(f"    {model_name:30s} | "
                  f"MAE={mae:7.4f} RMSE={rmse:7.4f} R={pearson_r:6.3f}")

        # ── 可视化：Q 值对比曲线 ────────────────────────────────
        plot_q_comparison(q_dict, returns, traj_idx, args.output_dir, 
                         gamma=args.gamma)

        # ── 可视化：轨迹图像帧 ──────────────────────────────────
        if args.save_video_samples:
            # 自动检测图像键
            img_key = None
            first_obs = trajectory[0].get("observations", {})
            for k in first_obs.keys():
                if "rgb" in k.lower():
                    img_key = k
                    break

            if img_key:
                plot_trajectory_images(trajectory, traj_idx, args.output_dir,
                                      n_frames=args.n_frames_per_traj,
                                      image_key=img_key)

    print(f"\n{'='*70}")

    # ── 汇总统计 ────────────────────────────────────────────────
    print("\n[汇总] 所有轨迹的平均指标:")
    for model_name in agents.keys():
        metrics_list = all_results[model_name].values()
        avg_mae = np.mean([m["mae"] for m in metrics_list])
        avg_rmse = np.mean([m["rmse"] for m in metrics_list])
        avg_r = np.mean([m["pearson_r"] for m in metrics_list])
        print(f"  {model_name:30s} | "
              f"avg_MAE={avg_mae:7.4f} avg_RMSE={avg_rmse:7.4f} avg_R={avg_r:6.3f}")

    # ── 统计可视化 ──────────────────────────────────────────────
    if len(agents) > 1 or len(trajectories) > 1:
        plot_q_statistics(all_results, args.output_dir)

    print(f"\n✓ 评估完成，所有结果已保存至: {args.output_dir}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
