#!/usr/bin/env python3
"""
Demo vs Online Replay Batch 对比分析脚本
=========================================
对比 demo_batch.pkl 和 online_replay_batch.pkl 中的训练数据指标：
  1. actions 的分布统计（min/max/mean/std, 各维度）
  2. masks 分布
  3. mc_returns 分布
  4. 图像像素范围 / dtype / shape / 通道均值
  5. demo 数据中 action(delta) 与相邻 observation 的一致性验证

Usage:
  python analyze_demo_vs_online.py \
      --demo_batch /path/to/demo_batch.pkl \
      --online_batch /path/to/online_replay_batch.pkl \
      --out_dir analyze_output
"""

import os
import sys
import pickle as pkl
import argparse
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ─────────────────────────────────────── 终端颜色 ─────────────────────────
RESET  = "\033[0m"
GREEN  = "\033[92m"
YELLOW = "\033[93m"
RED    = "\033[91m"
CYAN   = "\033[96m"
BOLD   = "\033[1m"

def _ok(msg):   print(f"{GREEN}  [OK]  {msg}{RESET}")
def _warn(msg): print(f"{YELLOW}  [WARN] {msg}{RESET}")
def _err(msg):  print(f"{RED}  [ERR]  {msg}{RESET}")
def _info(msg): print(f"{CYAN}  [INFO] {msg}{RESET}")
def _head(msg): print(f"\n{BOLD}{'='*60}\n  {msg}\n{'='*60}{RESET}")
def _sub(msg):  print(f"\n{BOLD}--- {msg} ---{RESET}")


# ─────────────────────────────────────── 工具函数 ─────────────────────────
def _to_numpy(v):
    """递归地将 FrozenDict / jax arrays 转为普通 dict / np.ndarray。"""
    if hasattr(v, "unfreeze"):
        return {k: _to_numpy(v2) for k, v2 in v.unfreeze().items()}
    if isinstance(v, dict):
        return {k: _to_numpy(v2) for k, v2 in v.items()}
    try:
        import jax
        if isinstance(v, jax.Array):
            return np.asarray(v)
    except ImportError:
        pass
    if hasattr(v, "__array__"):
        return np.asarray(v)
    return v


def load_batch(path):
    """加载 pkl batch 并转 numpy。"""
    with open(path, "rb") as f:
        batch = pkl.load(f)
    return _to_numpy(batch)


def batch_size(batch):
    """推断 batch size。"""
    for v in batch.values():
        if isinstance(v, np.ndarray) and v.ndim >= 1:
            return v.shape[0]
        if isinstance(v, dict):
            n = batch_size(v)
            if n is not None:
                return n
    return None


def print_dict_structure(d, prefix="", depth=0):
    """递归打印 dict 的 key / shape / dtype。"""
    for k, v in sorted(d.items()):
        indent = "  " * depth
        if isinstance(v, dict):
            print(f"{indent}{prefix}{k}/  (dict, {len(v)} keys)")
            print_dict_structure(v, prefix="", depth=depth+1)
        elif isinstance(v, np.ndarray):
            print(f"{indent}{prefix}{k}: shape={v.shape}, dtype={v.dtype}, "
                  f"min={v.min():.6g}, max={v.max():.6g}")
        else:
            print(f"{indent}{prefix}{k}: type={type(v).__name__}, value={v}")


def stat_array(arr, name=""):
    """打印 array 的统计信息。"""
    if arr is None:
        _warn(f"{name}: 不存在")
        return
    arr = np.asarray(arr, dtype=np.float32)
    print(f"    {name:30s}  shape={str(arr.shape):20s}  "
          f"min={arr.min():10.4f}  max={arr.max():10.4f}  "
          f"mean={arr.mean():10.4f}  std={arr.std():10.4f}")


def per_dim_stats(arr, name=""):
    """对 (B, D) array 逐维打印统计。"""
    if arr is None or arr.ndim < 2:
        return
    arr = np.asarray(arr, dtype=np.float32)
    print(f"    {name} — 逐维统计 (D={arr.shape[-1]}):")
    # 如果是 (B, T, D) 则展平到 (B*T, D)
    if arr.ndim == 3:
        arr = arr.reshape(-1, arr.shape[-1])
    for d in range(arr.shape[-1]):
        col = arr[:, d]
        print(f"      dim[{d}]: min={col.min():10.4f}  max={col.max():10.4f}  "
              f"mean={col.mean():10.4f}  std={col.std():10.4f}")


def find_image_keys(batch):
    """找到 observations 中的图像 key（shape 含 H,W,C 或 T,H,W,C）。"""
    obs = batch.get("observations", {})
    if not isinstance(obs, dict):
        return []
    img_keys = []
    for k, v in obs.items():
        if isinstance(v, np.ndarray) and v.ndim >= 4:  # (B, H, W, C) or (B, T, H, W, C)
            img_keys.append(k)
    return img_keys


def analyze_images(batch, label, image_keys, out_dir=None):
    """分析图像像素范围、dtype、形状。"""
    obs = batch.get("observations", {})
    if not isinstance(obs, dict):
        _warn(f"[{label}] observations 不是 dict，跳过图像分析")
        return
    for k in image_keys:
        if k not in obs:
            _warn(f"[{label}] 缺少图像 key: {k}")
            continue
        imgs = obs[k]
        print(f"    {k:30s}  shape={str(imgs.shape):25s}  dtype={imgs.dtype}")
        # 取最后帧（如有时间维度）
        if imgs.ndim == 5:  # (B, T, H, W, C)
            imgs_flat = imgs[:, -1]  # 用最后一个时间步
        else:
            imgs_flat = imgs
        imgs_f = imgs_flat.astype(np.float32)
        ch_means = imgs_f.mean(axis=(0, 1, 2))  # (C,)
        print(f"      pixel range: [{imgs_flat.min()}, {imgs_flat.max()}]")
        print(f"      channel means: {np.round(ch_means, 2)}")
        print(f"      overall mean={imgs_f.mean():.2f}  std={imgs_f.std():.2f}")

        # 保存预览图
        if out_dir is not None:
            fig, axes = plt.subplots(1, min(6, len(imgs_flat)), figsize=(18, 3))
            if not hasattr(axes, '__len__'):
                axes = [axes]
            for i, ax in enumerate(axes):
                img = imgs_flat[i]
                if img.max() > 1.5:  # uint8-ish
                    img = img.astype(np.uint8)
                ax.imshow(img)
                ax.set_title(f"#{i}")
                ax.axis("off")
            plt.suptitle(f"{label} — {k}")
            plt.tight_layout()
            fname = os.path.join(out_dir, f"images_{label}_{k}.png")
            plt.savefig(fname, dpi=120)
            plt.close()
            _info(f"  保存图像预览 → {fname}")


# ─────────────────── action delta 与 observation 差异校验 ─────────────────
def check_action_obs_consistency(batch, label, out_dir=None):
    """
    验证 demo 中 action (delta action) 与相邻两个 observation 的 state 差是否对得上。
    
    假设：
      - observations 包含 state 向量（如 'state' 或 'tcp_pose' 等 key）
      - actions 是 delta action，即 next_obs_state - obs_state ≈ action[:state_dim]
      - next_observations 包含下一步的 state
    """
    obs = batch.get("observations", {})
    next_obs = batch.get("next_observations", {})
    actions = batch.get("actions")

    if actions is None:
        _warn(f"[{label}] 无 actions，跳过一致性检查")
        return
    if not isinstance(obs, dict) or not isinstance(next_obs, dict):
        _warn(f"[{label}] observations 不是 dict，跳过一致性检查")
        return

    # 找 state key（可能叫 state / tcp_pose / proprio 等）
    state_key = None
    for candidate in ["state", "tcp_pose", "proprio", "robot_state", "ee_pos", "joint_positions"]:
        if candidate in obs and candidate in next_obs:
            state_key = candidate
            break

    if state_key is None:
        # 如果 obs 和 next_obs 都有某些数值 key，推断用
        numeric_keys_obs = [k for k, v in obs.items()
                           if isinstance(v, np.ndarray) and v.ndim <= 2 and v.dtype in (np.float32, np.float64)]
        numeric_keys_next = [k for k, v in next_obs.items()
                             if isinstance(v, np.ndarray) and v.ndim <= 2 and v.dtype in (np.float32, np.float64)]
        common_keys = [k for k in numeric_keys_obs if k in numeric_keys_next]
        if common_keys:
            state_key = common_keys[0]
            _info(f"[{label}] 自动检测到 state key: '{state_key}'")
        else:
            _warn(f"[{label}] 找不到 state key，无法校验 action-obs 一致性")
            _info(f"  obs keys: {list(obs.keys())}")
            _info(f"  next_obs keys: {list(next_obs.keys())}")
            return

    obs_state = np.asarray(obs[state_key], dtype=np.float32)
    next_obs_state = np.asarray(next_obs[state_key], dtype=np.float32)
    actions_np = np.asarray(actions, dtype=np.float32)

    # 处理 window/time 维度
    if obs_state.ndim == 3:  # (B, T, D) — 取最后时间步
        obs_state = obs_state[:, -1]
    if next_obs_state.ndim == 3:
        next_obs_state = next_obs_state[:, -1]
    if actions_np.ndim == 3:  # (B, T, D) — 取第一个 action
        actions_np = actions_np[:, 0]

    print(f"    state_key='{state_key}':")
    print(f"      obs_state shape:      {obs_state.shape}")
    print(f"      next_obs_state shape: {next_obs_state.shape}")
    print(f"      actions shape:        {actions_np.shape}")

    state_dim = obs_state.shape[-1]
    action_dim = actions_np.shape[-1]

    delta_state = next_obs_state - obs_state
    # 比较维度：取较小的维度数
    compare_dim = min(state_dim, action_dim)

    # 对于带 gripper 的情况，通常前 6 或 7 维是位姿 delta，最后 1 维是 gripper
    # 我们比较前 compare_dim-1 维（排除 gripper 维度）
    if compare_dim > 1:
        pose_dim = compare_dim - 1  # 排除 gripper
    else:
        pose_dim = compare_dim

    delta_pose = delta_state[:, :pose_dim]
    action_pose = actions_np[:, :pose_dim]

    diff = delta_pose - action_pose
    abs_diff = np.abs(diff)

    print(f"\n    Delta action vs state diff 对比 (前 {pose_dim} 维，排除 gripper):")
    print(f"      {'dim':>5s}  {'|diff|_mean':>12s}  {'|diff|_max':>12s}  "
          f"{'|diff|_std':>12s}  {'δstate_mean':>12s}  {'action_mean':>12s}")
    per_dim_ok = []
    for d in range(pose_dim):
        d_mean = abs_diff[:, d].mean()
        d_max = abs_diff[:, d].max()
        d_std = abs_diff[:, d].std()
        ds_mean = delta_pose[:, d].mean()
        a_mean = action_pose[:, d].mean()
        is_ok = d_mean < 0.05  # 阈值
        per_dim_ok.append(is_ok)
        flag = f"{GREEN}✓{RESET}" if is_ok else f"{RED}✗{RESET}"
        print(f"    {flag} dim[{d}]  {d_mean:12.6f}  {d_max:12.6f}  "
              f"{d_std:12.6f}  {ds_mean:12.6f}  {a_mean:12.6f}")

    overall_mae = abs_diff.mean()
    overall_max = abs_diff.max()
    print(f"\n    Overall MAE = {overall_mae:.6f},  Overall Max|diff| = {overall_max:.6f}")
    if overall_mae < 0.01:
        _ok(f"[{label}] action 与 state delta 非常一致 (MAE={overall_mae:.6f})")
    elif overall_mae < 0.05:
        _warn(f"[{label}] action 与 state delta 有轻微偏差 (MAE={overall_mae:.6f})")
    else:
        _err(f"[{label}] action 与 state delta 差距较大！(MAE={overall_mae:.6f})")

    # 如果有 gripper 维度，也看看
    if compare_dim > 1 and state_dim >= compare_dim and action_dim >= compare_dim:
        gripper_idx = pose_dim  # 下一个维度
        if gripper_idx < min(state_dim, action_dim):
            gripper_delta = delta_state[:, gripper_idx]
            gripper_action = actions_np[:, gripper_idx]
            print(f"\n    Gripper 维 (dim[{gripper_idx}]):")
            print(f"      delta_state: mean={gripper_delta.mean():.4f}, "
                  f"min={gripper_delta.min():.4f}, max={gripper_delta.max():.4f}")
            print(f"      action:      mean={gripper_action.mean():.4f}, "
                  f"min={gripper_action.min():.4f}, max={gripper_action.max():.4f}")
            print(f"      注：gripper 通常是绝对值（非 delta），不一致是正常的")

    # 绘图
    if out_dir is not None:
        n_plot = min(pose_dim, 7)
        fig, axes = plt.subplots(n_plot, 1, figsize=(12, 3 * n_plot), sharex=True)
        if n_plot == 1:
            axes = [axes]
        for d in range(n_plot):
            ax = axes[d]
            n_samples = min(100, len(delta_pose))
            ax.plot(delta_pose[:n_samples, d], label="next_obs - obs (delta state)", alpha=0.7)
            ax.plot(action_pose[:n_samples, d], label="action", alpha=0.7, linestyle="--")
            ax.set_ylabel(f"dim[{d}]")
            ax.legend(fontsize=8)
            ax.set_title(f"dim[{d}] — MAE={abs_diff[:, d].mean():.6f}")
        axes[-1].set_xlabel("Sample index")
        plt.suptitle(f"{label}: Action vs State Delta", fontsize=14)
        plt.tight_layout()
        fname = os.path.join(out_dir, f"action_vs_delta_{label}.png")
        plt.savefig(fname, dpi=120)
        plt.close()
        _info(f"  保存对比图 → {fname}")


# ─────────────────── 对比两个 batch 的指标 ─────────────────────────────────
def compare_distributions(demo, online, key, out_dir=None):
    """柱状图对比两个 batch 中某个 key 的分布。"""
    d = demo.get(key)
    o = online.get(key)
    if d is None and o is None:
        _warn(f"两个 batch 都缺少 key: {key}")
        return
    if d is not None:
        d = np.asarray(d, dtype=np.float32).ravel()
    if o is not None:
        o = np.asarray(o, dtype=np.float32).ravel()

    fig, ax = plt.subplots(1, 1, figsize=(10, 4))
    bins = 50
    if d is not None:
        ax.hist(d, bins=bins, alpha=0.5, label=f"demo (n={len(d)})", density=True)
    if o is not None:
        ax.hist(o, bins=bins, alpha=0.5, label=f"online (n={len(o)})", density=True)
    ax.set_xlabel(key)
    ax.set_ylabel("Density")
    ax.set_title(f"Distribution of '{key}'")
    ax.legend()
    plt.tight_layout()
    if out_dir:
        fname = os.path.join(out_dir, f"dist_{key}.png")
        plt.savefig(fname, dpi=120)
        plt.close()
        _info(f"  保存分布图 → {fname}")
    else:
        plt.show()


def compare_action_per_dim(demo, online, out_dir=None):
    """对比两个 batch action 每维的分布（boxplot）。"""
    d = demo.get("actions")
    o = online.get("actions")
    if d is None or o is None:
        return
    d = np.asarray(d, dtype=np.float32)
    o = np.asarray(o, dtype=np.float32)
    # 展平时间维
    if d.ndim == 3:
        d = d.reshape(-1, d.shape[-1])
    if o.ndim == 3:
        o = o.reshape(-1, o.shape[-1])

    n_dim = d.shape[-1]
    fig, axes = plt.subplots(1, n_dim, figsize=(3 * n_dim, 5))
    if n_dim == 1:
        axes = [axes]
    for i in range(n_dim):
        ax = axes[i]
        data = [d[:, i], o[:, i]]
        bp = ax.boxplot(data, tick_labels=["demo", "online"], patch_artist=True)
        bp['boxes'][0].set_facecolor('#5b9bd5')
        bp['boxes'][1].set_facecolor('#ed7d31')
        ax.set_title(f"dim[{i}]")
        ax.grid(True, alpha=0.3)
    plt.suptitle("Actions per-dim comparison", fontsize=14)
    plt.tight_layout()
    if out_dir:
        fname = os.path.join(out_dir, "actions_per_dim_boxplot.png")
        plt.savefig(fname, dpi=120)
        plt.close()
        _info(f"  保存 action 逐维对比图 → {fname}")


# ═════════════════════════════════════════════════════════════════════════════
#  MAIN
# ═════════════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(description="Demo vs Online batch 对比分析")
    parser.add_argument("--demo_batch",
                        default="/home/dungeon_master/conrft/examples/experiments/insert_block/conrft/02241/diag_batches/demo_batch.pkl",
                        help="demo_batch.pkl 路径")
    parser.add_argument("--online_batch",
                        default="/home/dungeon_master/conrft/examples/experiments/insert_block/conrft/02241/diag_batches/online_replay_batch.pkl",
                        help="online_replay_batch.pkl 路径")
    parser.add_argument("--out_dir",
                        default="/home/dungeon_master/conrft/examples/experiments/insert_block/conrft/02241/diag_batches/analyze_output",
                        help="输出目录")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # ══════════════════════════ 加载数据 ══════════════════════════════════
    _head("加载数据")
    demo = load_batch(args.demo_batch)
    _ok(f"Demo batch loaded: {args.demo_batch}")
    demo_n = batch_size(demo)
    _info(f"  batch size = {demo_n}")

    online = load_batch(args.online_batch)
    _ok(f"Online batch loaded: {args.online_batch}")
    online_n = batch_size(online)
    _info(f"  batch size = {online_n}")

    # ══════════════════════════ 数据结构 ══════════════════════════════════
    _head("1. 数据结构对比")
    _sub("Demo batch 结构")
    print_dict_structure(demo)
    _sub("Online batch 结构")
    print_dict_structure(online)

    # key 差异
    demo_keys = set(demo.keys())
    online_keys = set(online.keys())
    if demo_keys - online_keys:
        _warn(f"Demo 独有 keys: {demo_keys - online_keys}")
    if online_keys - demo_keys:
        _warn(f"Online 独有 keys: {online_keys - demo_keys}")
    common_keys = demo_keys & online_keys
    _info(f"公共 keys: {common_keys}")

    # ══════════════════════════ Actions ══════════════════════════════════
    _head("2. Actions 分布对比")
    for source_name, batch in [("demo", demo), ("online", online)]:
        _sub(f"{source_name} actions")
        a = batch.get("actions")
        if a is not None:
            stat_array(a, f"{source_name}/actions")
            per_dim_stats(a, f"{source_name}/actions")
        else:
            _warn(f"{source_name} 无 actions 字段")

    compare_distributions(demo, online, "actions", args.out_dir)
    compare_action_per_dim(demo, online, args.out_dir)

    # ══════════════════════════ Masks ═════════════════════════════════════
    _head("3. Masks 分布对比")
    for source_name, batch in [("demo", demo), ("online", online)]:
        m = batch.get("masks")
        if m is not None:
            m_np = np.asarray(m, dtype=np.float32).ravel()
            stat_array(m_np, f"{source_name}/masks")
            n_zero = (m_np == 0).sum()
            n_one = (m_np == 1).sum()
            print(f"      masks=0 (done): {n_zero}/{len(m_np)} ({100*n_zero/len(m_np):.1f}%)")
            print(f"      masks=1 (ongoing): {n_one}/{len(m_np)} ({100*n_one/len(m_np):.1f}%)")
        else:
            _warn(f"{source_name} 无 masks 字段")

    compare_distributions(demo, online, "masks", args.out_dir)

    # ══════════════════════════ MC Returns ════════════════════════════════
    _head("4. MC Returns 分布对比")
    for source_name, batch in [("demo", demo), ("online", online)]:
        mc = batch.get("mc_returns")
        if mc is not None:
            stat_array(mc, f"{source_name}/mc_returns")
            mc_np = np.asarray(mc, dtype=np.float32).ravel()
            print(f"      mc > 0: {(mc_np > 0).sum()}/{len(mc_np)} ({100*(mc_np > 0).sum()/len(mc_np):.1f}%)")
            print(f"      mc < 0: {(mc_np < 0).sum()}/{len(mc_np)} ({100*(mc_np < 0).sum()/len(mc_np):.1f}%)")
            print(f"      mc = 0: {(mc_np == 0).sum()}/{len(mc_np)} ({100*(mc_np == 0).sum()/len(mc_np):.1f}%)")
        else:
            _warn(f"{source_name} 无 mc_returns 字段")

    compare_distributions(demo, online, "mc_returns", args.out_dir)

    # ══════════════════════════ Rewards ═══════════════════════════════════
    _head("5. Rewards 分布对比")
    for source_name, batch in [("demo", demo), ("online", online)]:
        r = batch.get("rewards")
        if r is not None:
            stat_array(r, f"{source_name}/rewards")
            r_np = np.asarray(r, dtype=np.float32).ravel()
            print(f"      reward > 0: {(r_np > 0).sum()}/{len(r_np)}")
            print(f"      reward = 0: {(r_np == 0).sum()}/{len(r_np)}")
            print(f"      reward < 0: {(r_np < 0).sum()}/{len(r_np)}")
        else:
            _warn(f"{source_name} 无 rewards 字段")

    compare_distributions(demo, online, "rewards", args.out_dir)

    # ══════════════════════════ 图像分析 ══════════════════════════════════
    _head("6. 图像像素范围对比")
    demo_img_keys = find_image_keys(demo)
    online_img_keys = find_image_keys(online)
    all_img_keys = list(set(demo_img_keys + online_img_keys))
    _info(f"Demo 图像 keys: {demo_img_keys}")
    _info(f"Online 图像 keys: {online_img_keys}")

    for source_name, batch in [("demo", demo), ("online", online)]:
        _sub(f"{source_name} 图像")
        analyze_images(batch, source_name, all_img_keys, args.out_dir)

    # 跨 batch 图像像素均值对比
    if all_img_keys:
        _sub("图像像素统计对比表")
        print(f"    {'key':>25s}  {'source':>8s}  {'dtype':>8s}  {'min':>6s}  {'max':>6s}  "
              f"{'mean':>8s}  {'std':>8s}  {'ch0':>8s}  {'ch1':>8s}  {'ch2':>8s}")
        for k in all_img_keys:
            for source_name, batch in [("demo", demo), ("online", online)]:
                obs = batch.get("observations", {})
                if k in obs:
                    imgs = obs[k]
                    if imgs.ndim == 5:
                        imgs = imgs[:, -1]
                    f = imgs.astype(np.float32)
                    ch = f.mean(axis=(0,1,2))
                    print(f"    {k:>25s}  {source_name:>8s}  {str(imgs.dtype):>8s}  "
                          f"{imgs.min():>6}  {imgs.max():>6}  "
                          f"{f.mean():8.2f}  {f.std():8.2f}  "
                          f"{ch[0]:8.2f}  {ch[1]:8.2f}  {ch[2]:8.2f}")

    # ══════════════════════════ Intervened 标记 ═══════════════════════════
    _head("7. 其他字段对比")
    for field in ["dones", "intervened", "grasp_penalty"]:
        has_any = False
        for source_name, batch in [("demo", demo), ("online", online)]:
            v = batch.get(field)
            if v is not None:
                has_any = True
                v_np = np.asarray(v, dtype=np.float32).ravel()
                stat_array(v_np, f"{source_name}/{field}")
                n_true = (v_np > 0.5).sum()
                print(f"      True: {n_true}/{len(v_np)} ({100*n_true/len(v_np):.1f}%)")
        if not has_any:
            _info(f"两个 batch 都无 '{field}' 字段")

    # ══════════════════════ Action-Obs 一致性验证 ═════════════════════════
    _head("8. Demo Action(delta) 与 Observation 差值一致性验证")
    check_action_obs_consistency(demo, "demo", args.out_dir)

    _sub("Online batch 一致性验证（作参考）")
    check_action_obs_consistency(online, "online", args.out_dir)

    # ══════════════════════════ 总结 ═════════════════════════════════════
    _head("分析完成")
    _info(f"输出目录: {args.out_dir}")
    _info(f"生成的文件:")
    for f in sorted(os.listdir(args.out_dir)):
        print(f"    {f}")

    # 保存文本摘要
    summary_path = os.path.join(args.out_dir, "summary.txt")
    with open(summary_path, "w") as f:
        f.write("Demo vs Online Batch Analysis Summary\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Demo batch: {args.demo_batch}\n")
        f.write(f"Online batch: {args.online_batch}\n")
        f.write(f"Demo batch size: {demo_n}\n")
        f.write(f"Online batch size: {online_n}\n\n")

        for source_name, batch in [("demo", demo), ("online", online)]:
            f.write(f"\n--- {source_name} ---\n")
            a = batch.get("actions")
            if a is not None:
                a = np.asarray(a, dtype=np.float32)
                f.write(f"  actions: shape={a.shape}, min={a.min():.4f}, max={a.max():.4f}, "
                        f"mean={a.mean():.4f}, std={a.std():.4f}\n")
            m = batch.get("masks")
            if m is not None:
                m = np.asarray(m, dtype=np.float32).ravel()
                f.write(f"  masks: done_rate={(m==0).sum()/len(m)*100:.1f}%\n")
            mc = batch.get("mc_returns")
            if mc is not None:
                mc = np.asarray(mc, dtype=np.float32)
                f.write(f"  mc_returns: min={mc.min():.4f}, max={mc.max():.4f}, "
                        f"mean={mc.mean():.4f}\n")
            r = batch.get("rewards")
            if r is not None:
                r = np.asarray(r, dtype=np.float32)
                f.write(f"  rewards: min={r.min():.4f}, max={r.max():.4f}, "
                        f"mean={r.mean():.4f}\n")

    _ok(f"摘要已保存 → {summary_path}")


if __name__ == "__main__":
    main()
