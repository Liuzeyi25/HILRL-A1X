#!/usr/bin/env python3
"""
analyze_q_values.py
===================
用训练好的 Critic 对 buffer 中保存的轨迹逐步估计 Q 值，
比较不同训练步数（如 4000 / 6000 / 12000）的 Q 值曲线差异。

使用方法
--------
python scripts/analyze_q_values.py \
    --checkpoint_dir  /path/to/checkpoint \
    --buffer_dir      /path/to/checkpoint/buffer \
    --steps           4000 6000 12000 \
    --exp_name        insert_block \
    --n_trajs         5 \
    --output_dir      /path/to/output

命令行参数说明
--------------
--checkpoint_dir   : checkpoint 根目录，内含 checkpoint_XXXXXX 文件夹
--buffer_dir       : buffer pkl 所在目录（默认 checkpoint_dir/buffer）
--steps            : 要对比的训练步数列表，空格分隔
--exp_name         : 实验名称，与 train_rlpd_hil.py 的 --exp_name 保持一致
--n_trajs          : 展示前 N 条轨迹（按时间排序）
--traj_idx         : 仅分析指定索引的轨迹（可多选），不指定则用 n_trajs
--output_dir       : 图片输出目录（默认当前目录）
--use_target_critic: 是否使用 target critic（默认 False，使用 online critic）
--ensemble_agg     : ensemble Q 值聚合方式：min / mean（默认 min）
--seed             : 随机种子（默认 42）
--setup_mode       : fixed-gripper 或 learned-gripper（默认从 config 读取）
"""

import argparse
import os
import pickle as pkl
import sys
from pathlib import Path

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
# 工具函数
# =============================================================================

def load_trajectories_from_buffer(buffer_dir: str, n_trajs: int = None,
                                   traj_idx: list = None) -> list:
    """
    从 buffer 目录读取 pkl 文件，按文件名中的 step 数排序，
    每个 pkl 视为一批 transitions（一条 episode 或多条的合并）。

    返回: list of list[dict]，每条轨迹是一个 dict 列表（按时间顺序）。
    """
    pkl_files = sorted(
        Path(buffer_dir).glob("transitions_*.pkl"),
        key=lambda p: int(p.stem.split("_")[-1])
    )
    if not pkl_files:
        raise FileNotFoundError(f"在 {buffer_dir} 中未找到 transitions_*.pkl 文件")

    # 把每个 pkl 内的 transitions 按 episode 切分（遇到 done=True 切断）
    all_trajs = []
    for pkl_file in pkl_files:
        with open(pkl_file, "rb") as f:
            transitions = pkl.load(f)
        if not transitions:
            continue
        # 切分 episode
        ep = []
        for tr in transitions:
            ep.append(tr)
            done = bool(tr.get("dones", False)) or bool(tr.get("done", False))
            if done:
                if len(ep) > 1:
                    all_trajs.append(ep)
                ep = []
        if len(ep) > 1:  # 末尾未完成的 episode 也保留
            all_trajs.append(ep)

    if not all_trajs:
        # fallback：把整个 buffer 当成一条轨迹
        all_trs = []
        for pkl_file in pkl_files:
            with open(pkl_file, "rb") as f:
                all_trs.extend(pkl.load(f))
        all_trajs = [all_trs]

    if traj_idx is not None:
        all_trajs = [all_trajs[i] for i in traj_idx if i < len(all_trajs)]
    elif n_trajs is not None:
        all_trajs = all_trajs[:n_trajs]

    print(f"[数据] 共加载 {len(all_trajs)} 条轨迹，"
          f"步数范围: {min(len(t) for t in all_trajs)}~{max(len(t) for t in all_trajs)}")
    return all_trajs


def build_obs_batch(transitions: list, image_keys: list) -> dict:
    """
    将 transition list 转为 batch obs dict，形状 (T, ...)。
    自动处理图像 stacking（T, stack, H, W, C）和 proprio 向量。
    """
    obs_list = [tr["observations"] for tr in transitions]
    batch = {}
    first = obs_list[0]
    for key in first.keys():
        vals = [o[key] for o in obs_list]
        batch[key] = np.stack(vals, axis=0)  # (T, ...)
    return batch


def estimate_q_values(agent, obs_batch: dict, actions: np.ndarray,
                      rng: jax.random.PRNGKey,
                      use_target: bool = False,
                      ensemble_agg: str = "min",
                      fix_gripper: bool = True,
                      chunk_size: int = 32) -> np.ndarray:
    """
    用 agent 的 critic 逐批次估计 Q 值，返回 (T,) 数组。

    chunk_size: 每次推理的批大小，避免显存溢出（图像较大时需调小）。
    """
    T = actions.shape[0]
    q_values = []

    for start in range(0, T, chunk_size):
        end = min(start + chunk_size, T)
        obs_chunk = jax.tree_util.tree_map(
            lambda x: jax.device_put(x[start:end]), obs_batch
        )
        act_chunk = jax.device_put(actions[start:end])
        if fix_gripper:
            act_chunk = act_chunk[..., :-1]  # 截掉夹爪维度

        rng, key = jax.random.split(rng)
        if use_target:
            qs = agent.forward_target_critic(obs_chunk, act_chunk, rng=key)
        else:
            qs = agent.forward_critic(obs_chunk, act_chunk, rng=key, train=False)
        # qs: (ensemble_size, chunk)
        qs = np.asarray(qs)
        if ensemble_agg == "min":
            q_chunk = qs.min(axis=0)
        else:
            q_chunk = qs.mean(axis=0)
        q_values.append(q_chunk)

    return np.concatenate(q_values, axis=0)  # (T,)


# =============================================================================
# 主流程
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Q 值轨迹分析工具")
    parser.add_argument("--checkpoint_dir", required=True,
                        help="checkpoint 根目录")
    parser.add_argument("--buffer_dir", default=None,
                        help="buffer pkl 目录，默认 checkpoint_dir/buffer")
    parser.add_argument("--steps", nargs="+", type=int, required=True,
                        help="要对比的训练步数，如 4000 6000 12000")
    parser.add_argument("--exp_name", required=True,
                        help="实验名（CONFIG_MAPPING 中的键）")
    parser.add_argument("--n_trajs", type=int, default=5,
                        help="展示前 N 条轨迹")
    parser.add_argument("--traj_idx", nargs="+", type=int, default=None,
                        help="仅分析指定索引的轨迹")
    parser.add_argument("--output_dir", default=".",
                        help="图片输出目录")
    parser.add_argument("--use_target_critic", action="store_true",
                        help="使用 target critic 而非 online critic")
    parser.add_argument("--ensemble_agg", default="min", choices=["min", "mean"],
                        help="ensemble Q 值聚合方式")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--chunk_size", type=int, default=16,
                        help="每次推理的批大小（图像大时调小避免 OOM）")
    args = parser.parse_args()

    # ── 路径设置 ─────────────────────────────────────────────────────────
    ckpt_dir   = os.path.abspath(args.checkpoint_dir)
    buffer_dir = os.path.abspath(args.buffer_dir) if args.buffer_dir \
                 else os.path.join(ckpt_dir, "buffer")
    os.makedirs(args.output_dir, exist_ok=True)

    # ── 加载实验配置 ──────────────────────────────────────────────────────
    assert args.exp_name in CONFIG_MAPPING, \
        f"exp_name '{args.exp_name}' 不在 CONFIG_MAPPING 中"
    config = CONFIG_MAPPING[args.exp_name]()
    setup_mode  = config.setup_mode
    image_keys  = list(config.image_keys)
    fix_gripper = "fixed-gripper" in setup_mode
    print(f"[配置] exp_name={args.exp_name}, setup_mode={setup_mode}, "
          f"image_keys={image_keys}, fix_gripper={fix_gripper}")

    # ── 加载轨迹数据 ──────────────────────────────────────────────────────
    trajectories = load_trajectories_from_buffer(
        buffer_dir, n_trajs=args.n_trajs, traj_idx=args.traj_idx
    )

    # ── 构建 dummy 环境用于采样 obs/action 形状 ───────────────────────────
    print("[环境] 构建 fake_env 获取 obs/action space...")
    env = config.get_environment(
        fake_env=True,
        save_video=False,
        classifier=False,
        stack_obs_num=2,
    )
    sample_obs    = env.observation_space.sample()
    sample_action = env.action_space.sample()
    env.close()

    # ── 逐 checkpoint step 加载 agent 并推理 ─────────────────────────────
    rng = jax.random.PRNGKey(args.seed)
    # step -> traj_idx -> q_values (T,)
    all_q: dict = {step: [] for step in args.steps}

    for step in args.steps:
        print(f"\n[Checkpoint] 加载 step={step} ...")

        # 初始化同结构的 agent
        if setup_mode in ("single-arm-fixed-gripper", "dual-arm-fixed-gripper"):
            agent = make_sac_pixel_agent(
                seed=args.seed,
                sample_obs=sample_obs,
                sample_action=sample_action,
                image_keys=image_keys,
                encoder_type=config.encoder_type,
                discount=config.discount,
            )
        elif setup_mode == "single-arm-learned-gripper":
            agent = make_sac_pixel_agent_hybrid_single_arm(
                seed=args.seed,
                sample_obs=sample_obs,
                sample_action=sample_action,
                image_keys=image_keys,
                encoder_type=config.encoder_type,
                discount=config.discount,
            )
        else:
            raise NotImplementedError(f"不支持的 setup_mode: {setup_mode}")

        # 恢复 checkpoint
        restored_state = checkpoints.restore_checkpoint(
            os.path.abspath(ckpt_dir),
            agent.state,
            step=step,
        )
        agent = agent.replace(state=restored_state)
        print(f"  ✓ checkpoint 已加载")

        # 对每条轨迹推理 Q 值
        for traj_i, traj in enumerate(trajectories):
            obs_batch = build_obs_batch(traj, image_keys)
            actions   = np.stack([tr["actions"] for tr in traj], axis=0)  # (T, action_dim)

            rng, key = jax.random.split(rng)
            q_vals = estimate_q_values(
                agent, obs_batch, actions, key,
                use_target=args.use_target_critic,
                ensemble_agg=args.ensemble_agg,
                fix_gripper=fix_gripper,
                chunk_size=args.chunk_size,
            )
            all_q[step].append(q_vals)
            print(f"  轨迹 {traj_i:2d}: len={len(traj):4d}, "
                  f"Q mean={q_vals.mean():.4f}, "
                  f"Q min={q_vals.min():.4f}, "
                  f"Q max={q_vals.max():.4f}")

    # ── 绘图 ─────────────────────────────────────────────────────────────
    n_trajs = len(trajectories)
    colors  = [cm.tab10(i / max(len(args.steps), 1)) for i in range(len(args.steps))]

    # ── 图1：每条轨迹一个子图，各 step 的 Q 值曲线叠加 ──────────────────
    ncols = min(3, n_trajs)
    nrows = (n_trajs + ncols - 1) // ncols
    fig1, axes = plt.subplots(nrows, ncols,
                               figsize=(6 * ncols, 4 * nrows),
                               squeeze=False)
    fig1.suptitle("Q 值随轨迹步数的变化（不同 checkpoint）", fontsize=14)

    for traj_i in range(n_trajs):
        row, col = divmod(traj_i, ncols)
        ax = axes[row][col]
        traj_len = len(trajectories[traj_i])
        t_axis   = np.arange(traj_len)

        # 标注干预步（alpha_weight > 0）
        alpha_weights = np.array([tr.get("alpha_weight", 0.0)
                                   for tr in trajectories[traj_i]])
        subopt_mask = alpha_weights > 0
        if subopt_mask.any():
            ax.axvspan(
                np.where(subopt_mask)[0][0],
                np.where(subopt_mask)[0][-1],
                alpha=0.10, color="red", label="次优片段"
            )

        for step, color in zip(args.steps, colors):
            if traj_i < len(all_q[step]):
                q = all_q[step][traj_i]
                ax.plot(t_axis, q, color=color, linewidth=1.2,
                        label=f"step {step}")

        # 标注奖励获得步
        rewards = np.array([tr.get("rewards", 0.0) for tr in trajectories[traj_i]])
        reward_steps = np.where(rewards > 0)[0]
        for rs in reward_steps:
            ax.axvline(rs, color="green", linestyle="--", alpha=0.6, linewidth=0.8)

        ax.set_title(f"轨迹 {traj_i}  (len={traj_len})")
        ax.set_xlabel("Step")
        ax.set_ylabel("Q value")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    # 隐藏多余子图
    for traj_i in range(n_trajs, nrows * ncols):
        row, col = divmod(traj_i, ncols)
        axes[row][col].set_visible(False)

    fig1.tight_layout()
    out1 = os.path.join(args.output_dir, "q_curves_per_traj.png")
    fig1.savefig(out1, dpi=150)
    print(f"\n[输出] 每条轨迹的 Q 曲线图已保存: {out1}")

    # ── 图2：各 checkpoint 的平均 Q 值曲线（对所有轨迹归一化长度后均值）──
    fig2, ax2 = plt.subplots(figsize=(10, 5))
    norm_len = 100  # 归一化到 100 步

    for step, color in zip(args.steps, colors):
        q_list = all_q[step]
        if not q_list:
            continue
        # 插值到统一长度
        q_resampled = []
        for q in q_list:
            t_src = np.linspace(0, 1, len(q))
            t_dst = np.linspace(0, 1, norm_len)
            q_resampled.append(np.interp(t_dst, t_src, q))
        q_mean = np.mean(q_resampled, axis=0)
        q_std  = np.std(q_resampled, axis=0)
        t_axis = np.linspace(0, 1, norm_len)

        ax2.plot(t_axis, q_mean, color=color, linewidth=1.8, label=f"step {step}")
        ax2.fill_between(t_axis, q_mean - q_std, q_mean + q_std,
                         color=color, alpha=0.15)

    ax2.set_title("所有轨迹平均 Q 值（归一化到 100 步，±1σ 阴影）")
    ax2.set_xlabel("归一化轨迹进度")
    ax2.set_ylabel("Q value")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    fig2.tight_layout()
    out2 = os.path.join(args.output_dir, "q_mean_normalized.png")
    fig2.savefig(out2, dpi=150)
    print(f"[输出] 归一化均值 Q 曲线图已保存: {out2}")

    # ── 图3：各 checkpoint 的 Q 值分布（violin plot）────────────────────
    fig3, ax3 = plt.subplots(figsize=(8, 5))
    all_q_flat = []
    labels_flat = []
    for step in args.steps:
        flat = np.concatenate(all_q[step]) if all_q[step] else np.array([])
        all_q_flat.append(flat)
        labels_flat.append(f"step {step}")

    parts = ax3.violinplot(all_q_flat, showmedians=True, showextrema=True)
    for pc, color in zip(parts["bodies"], colors):
        pc.set_facecolor(color)
        pc.set_alpha(0.6)
    ax3.set_xticks(range(1, len(args.steps) + 1))
    ax3.set_xticklabels(labels_flat)
    ax3.set_title("Q 值全局分布（不同 checkpoint）")
    ax3.set_ylabel("Q value")
    ax3.grid(True, alpha=0.3, axis="y")
    fig3.tight_layout()
    out3 = os.path.join(args.output_dir, "q_distribution.png")
    fig3.savefig(out3, dpi=150)
    print(f"[输出] Q 值分布图已保存: {out3}")

    # ── 打印汇总表格 ──────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print(f"{'Step':>8}  {'Q mean':>10}  {'Q std':>10}  "
          f"{'Q min':>10}  {'Q max':>10}")
    print("-" * 60)
    for step in args.steps:
        flat = np.concatenate(all_q[step]) if all_q[step] else np.array([0.0])
        print(f"{step:>8}  {flat.mean():>10.4f}  {flat.std():>10.4f}  "
              f"{flat.min():>10.4f}  {flat.max():>10.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
