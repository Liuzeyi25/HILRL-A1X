#!/usr/bin/env python3
"""
Learner Sample Batch 全量分析脚本
==================================
分析 debug/ 下的所有 learner_sample_batch_step{step}.pkl 文件。

每个 pkl 文件结构：
  {
    '_step': int,
    '_critic_step': int,
    'online_batch': { actions, masks, mc_returns, rewards, observations, next_observations, ... },
    'demo_batch':   { 同上 },
    '_intervention_stats': { online_size, demo_size, *_intervened_count, ... },
  }

分析项目：
  1. 每个 step 的 actions 分布（逐维 min/max/mean/std），demo vs online 对比
  2. masks 分布
  3. mc_returns 分布
  4. rewards 分布
  5. 图像像素范围 / dtype / 通道均值 + 保存 PNG 预览
  6. intervention 统计随 step 变化
  7. demo action(delta) 与相邻 observation 差值一致性
  8. 各指标随 step 变化的趋势图

Usage:
  python analyze_learner_batch.py --debug_dir debug/ --out_dir analyze_learner_output
  python analyze_learner_batch.py  # 使用默认路径
"""

import os
import re
import glob
import pickle as pkl
import argparse
import numpy as np
import tqdm

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
def _head(msg): print(f"\n{BOLD}{'='*70}\n  {msg}\n{'='*70}{RESET}")
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


def normalize_format(data):
    """
    统一新旧两种格式为新格式：{ online_batch, demo_batch, _step, _critic_step, _intervention_stats }
    旧格式: 直接是 flat batch dict（actions, masks, observations, ... _step, _critic_step）
    新格式: { online_batch: {...}, demo_batch: {...}, _step, _critic_step, _intervention_stats }
    """
    if 'online_batch' in data and 'demo_batch' in data:
        return data  # 已经是新格式

    # 旧格式：将整个 batch 视为 "combined" 并复制到 online_batch
    meta_keys = {'_step', '_critic_step', '_file'}
    batch = {k: v for k, v in data.items() if k not in meta_keys}
    return {
        'online_batch': batch,
        'demo_batch': {},  # 旧格式没有 demo_batch 分离
        '_step': data.get('_step', 0),
        '_critic_step': data.get('_critic_step', 0),
        '_file': data.get('_file', ''),
        '_intervention_stats': {},
        '_format': 'legacy_flat',
    }


def list_batch_paths(debug_dir):
    """返回按 step 排序的 (step, path) 列表，不加载内容。"""
    pattern = os.path.join(debug_dir, "learner_sample_batch_step*.pkl")
    paths = glob.glob(pattern)
    result = []
    for p in paths:
        m = re.search(r'step(\d+)\.pkl$', p)
        step_num = int(m.group(1)) if m else 0
        result.append((step_num, p))
    result.sort(key=lambda x: x[0])
    return result


def load_one_batch(step_num, path):
    """加载单个 pkl 文件，返回 normalize 后的 dict（失败返回 None）。"""
    try:
        with open(path, "rb") as f:
            data = pkl.load(f)
        data = _to_numpy(data)
        data['_file'] = path
        data.setdefault('_step', step_num)
        return normalize_format(data)
    except Exception as e:
        _warn(f"加载失败 {path}: {e}")
        return None


def batch_size(batch):
    """推断 batch size。"""
    if batch is None:
        return 0
    for v in batch.values():
        if isinstance(v, np.ndarray) and v.ndim >= 1:
            return v.shape[0]
        if isinstance(v, dict):
            n = batch_size(v)
            if n is not None and n > 0:
                return n
    return 0


def print_dict_structure(d, prefix="", depth=0):
    """递归打印 dict 的 key / shape / dtype。"""
    for k, v in sorted(d.items()):
        indent = "  " * depth
        if isinstance(v, dict):
            print(f"{indent}{prefix}{k}/  (dict, {len(v)} keys)")
            print_dict_structure(v, prefix="", depth=depth + 1)
        elif isinstance(v, np.ndarray):
            print(f"{indent}{prefix}{k}: shape={v.shape}, dtype={v.dtype}, "
                  f"min={v.min():.6g}, max={v.max():.6g}")
        else:
            print(f"{indent}{prefix}{k}: type={type(v).__name__}, value={v}")


def find_image_keys(batch):
    """找到 observations 中的图像 key。"""
    obs = batch.get("observations", {})
    if not isinstance(obs, dict):
        return []
    return [k for k, v in obs.items()
            if isinstance(v, np.ndarray) and v.ndim >= 4]


# ────────────────────────── 单个 batch 的统计提取 ─────────────────────────
def extract_stats(batch, label=""):
    """从单个 online_batch 或 demo_batch 中提取统计量，返回 dict。"""
    stats = {'label': label}
    if batch is None:
        return stats

    # actions
    a = batch.get("actions")
    if a is not None:
        a = np.asarray(a, dtype=np.float32)
        if a.ndim == 3:
            a = a.reshape(-1, a.shape[-1])
        stats['action_min'] = a.min()
        stats['action_max'] = a.max()
        stats['action_mean'] = a.mean()
        stats['action_std'] = a.std()
        stats['action_per_dim_mean'] = a.mean(axis=0)
        stats['action_per_dim_std'] = a.std(axis=0)
        stats['action_per_dim_min'] = a.min(axis=0)
        stats['action_per_dim_max'] = a.max(axis=0)
        stats['action_dim'] = a.shape[-1]

    # masks
    m = batch.get("masks")
    if m is not None:
        m = np.asarray(m, dtype=np.float32).ravel()
        stats['masks_mean'] = m.mean()
        stats['masks_done_rate'] = (m == 0).sum() / max(len(m), 1)

    # mc_returns
    mc = batch.get("mc_returns")
    if mc is not None:
        mc = np.asarray(mc, dtype=np.float32).ravel()
        stats['mc_min'] = mc.min()
        stats['mc_max'] = mc.max()
        stats['mc_mean'] = mc.mean()
        stats['mc_std'] = mc.std()
        stats['mc_pos_rate'] = (mc > 0).sum() / max(len(mc), 1)

    # rewards
    r = batch.get("rewards")
    if r is not None:
        r = np.asarray(r, dtype=np.float32).ravel()
        stats['reward_min'] = r.min()
        stats['reward_max'] = r.max()
        stats['reward_mean'] = r.mean()
        stats['reward_pos_rate'] = (r > 0).sum() / max(len(r), 1)

    # images
    obs = batch.get("observations", {})
    if isinstance(obs, dict):
        for k, v in obs.items():
            if isinstance(v, np.ndarray) and v.ndim >= 4:
                imgs = v[:, -1] if v.ndim == 5 else v
                f = imgs.astype(np.float32)
                stats[f'img_{k}_min'] = int(imgs.min())
                stats[f'img_{k}_max'] = int(imgs.max())
                stats[f'img_{k}_mean'] = f.mean()
                stats[f'img_{k}_std'] = f.std()
                ch_means = f.mean(axis=(0, 1, 2))
                stats[f'img_{k}_ch_means'] = ch_means

    return stats


# ─────────────────── action delta 与 observation 差异校验 ─────────────────
def check_action_obs_consistency(batch, label):
    """验证 action(delta) 与 state 差是否对得上，返回 MAE 和逐维信息。"""
    obs = batch.get("observations", {})
    next_obs = batch.get("next_observations", {})
    actions = batch.get("actions")

    if actions is None or not isinstance(obs, dict) or not isinstance(next_obs, dict):
        return None

    # 找 state key
    state_key = None
    for candidate in ["state", "tcp_pose", "proprio", "robot_state", "ee_pos", "joint_positions"]:
        if candidate in obs and candidate in next_obs:
            state_key = candidate
            break
    if state_key is None:
        numeric_obs = [k for k, v in obs.items()
                       if isinstance(v, np.ndarray) and v.ndim <= 3 and v.dtype in (np.float32, np.float64)]
        numeric_next = [k for k, v in next_obs.items()
                        if isinstance(v, np.ndarray) and v.ndim <= 3 and v.dtype in (np.float32, np.float64)]
        common = [k for k in numeric_obs if k in numeric_next]
        if common:
            state_key = common[0]
        else:
            return None

    obs_state = np.asarray(obs[state_key], dtype=np.float32)
    next_obs_state = np.asarray(next_obs[state_key], dtype=np.float32)
    actions_np = np.asarray(actions, dtype=np.float32)

    if obs_state.ndim == 3:
        obs_state = obs_state[:, -1]
    if next_obs_state.ndim == 3:
        next_obs_state = next_obs_state[:, -1]
    if actions_np.ndim == 3:
        actions_np = actions_np[:, 0]

    state_dim = obs_state.shape[-1]
    action_dim = actions_np.shape[-1]
    compare_dim = min(state_dim, action_dim)
    pose_dim = max(compare_dim - 1, 1)  # 排除 gripper

    delta_state = next_obs_state - obs_state
    diff = np.abs(delta_state[:, :pose_dim] - actions_np[:, :pose_dim])
    overall_mae = diff.mean()
    per_dim_mae = diff.mean(axis=0)

    return {
        'state_key': state_key,
        'overall_mae': overall_mae,
        'overall_max': diff.max(),
        'per_dim_mae': per_dim_mae,
        'pose_dim': pose_dim,
    }


# ─────────────────── 保存图像 PNG ────────────────────────────────────────
def save_batch_images(batch, step, source_label, image_keys, out_dir, max_samples=6):
    """将 batch 中的图像保存为 PNG 文件。"""
    obs = batch.get("observations", {})
    if not isinstance(obs, dict):
        return

    for cam_key in image_keys:
        if cam_key not in obs:
            continue
        imgs = obs[cam_key]
        if imgs.ndim == 5:  # (B, T, H, W, C) → 取最后时间步
            imgs = imgs[:, -1]

        n = min(max_samples, len(imgs))

        # 保存单张图片
        img_sub_dir = os.path.join(out_dir, "images", f"step{step}_{source_label}")
        os.makedirs(img_sub_dir, exist_ok=True)
        for i in range(n):
            img = imgs[i]
            if img.max() > 1.5:
                img = img.astype(np.uint8)
            fname = os.path.join(img_sub_dir, f"{cam_key}_sample{i}.png")
            plt.imsave(fname, img)

        # 保存拼接预览
        fig, axes = plt.subplots(1, n, figsize=(3 * n, 3))
        if n == 1:
            axes = [axes]
        for i in range(n):
            img = imgs[i]
            if img.max() > 1.5:
                img = img.astype(np.uint8)
            axes[i].imshow(img)
            axes[i].set_title(f"#{i}", fontsize=8)
            axes[i].axis("off")
        plt.suptitle(f"Step {step} | {source_label} | {cam_key}", fontsize=11)
        plt.tight_layout()
        overview_fname = os.path.join(img_sub_dir, f"{cam_key}_overview.png")
        plt.savefig(overview_fname, dpi=100)
        plt.close()


# ═══════════════════════════════ 趋势图 ═══════════════════════════════════
def plot_trends(all_records, out_dir):
    """绘制各指标随 step 变化的趋势图。"""
    steps = [r['step'] for r in all_records]
    if len(steps) < 2:
        _info("步数不足 2，跳过趋势图")
        return

    def _get(records, key, source):
        return [r[source].get(key) for r in records]

    # ── 1. Action 统计趋势 ──
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    for source, color, ls in [('online', '#ed7d31', '-'), ('demo', '#5b9bd5', '--')]:
        vals = _get(all_records, 'action_mean', source)
        if any(v is not None for v in vals):
            axes[0, 0].plot(steps, vals, color=color, linestyle=ls, label=f"{source} mean", marker='.')
    axes[0, 0].set_title("Action Mean"); axes[0, 0].legend(); axes[0, 0].grid(True, alpha=0.3)

    for source, color, ls in [('online', '#ed7d31', '-'), ('demo', '#5b9bd5', '--')]:
        vals = _get(all_records, 'action_std', source)
        if any(v is not None for v in vals):
            axes[0, 1].plot(steps, vals, color=color, linestyle=ls, label=f"{source} std", marker='.')
    axes[0, 1].set_title("Action Std"); axes[0, 1].legend(); axes[0, 1].grid(True, alpha=0.3)

    for source, color, ls in [('online', '#ed7d31', '-'), ('demo', '#5b9bd5', '--')]:
        vals = _get(all_records, 'action_min', source)
        if any(v is not None for v in vals):
            axes[1, 0].plot(steps, vals, color=color, linestyle=ls, label=f"{source} min", marker='.')
        vals = _get(all_records, 'action_max', source)
        if any(v is not None for v in vals):
            axes[1, 0].plot(steps, vals, color=color, linestyle=':', label=f"{source} max", marker='x')
    axes[1, 0].set_title("Action Min / Max"); axes[1, 0].legend(); axes[1, 0].grid(True, alpha=0.3)

    # Action per-dim mean
    sample_rec = next((r for r in all_records if r['online'].get('action_dim')), None)
    if sample_rec:
        n_dim = sample_rec['online'].get('action_dim', 7)
        for d in range(min(n_dim, 7)):
            for source, color in [('online', '#ed7d31'), ('demo', '#5b9bd5')]:
                vals = []
                for r in all_records:
                    pdm = r[source].get('action_per_dim_mean')
                    vals.append(pdm[d] if pdm is not None and d < len(pdm) else None)
                if any(v is not None for v in vals):
                    axes[1, 1].plot(steps, vals, label=f"{source} dim[{d}]", alpha=0.7, marker='.', markersize=3)
    axes[1, 1].set_title("Action Per-dim Mean"); axes[1, 1].legend(fontsize=6, ncol=2); axes[1, 1].grid(True, alpha=0.3)

    for ax in axes.flat:
        ax.set_xlabel("Step")
    plt.suptitle("Action 统计趋势", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "trend_actions.png"), dpi=120)
    plt.close()
    _info("  保存 → trend_actions.png")

    # ── 2. MC Returns & Rewards 趋势 ──
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for source, color, ls in [('online', '#ed7d31', '-'), ('demo', '#5b9bd5', '--')]:
        vals = _get(all_records, 'mc_mean', source)
        if any(v is not None for v in vals):
            axes[0].plot(steps, vals, color=color, linestyle=ls, label=f"{source}", marker='.')
    axes[0].set_title("MC Returns Mean"); axes[0].legend(); axes[0].grid(True, alpha=0.3)

    for source, color, ls in [('online', '#ed7d31', '-'), ('demo', '#5b9bd5', '--')]:
        vals = _get(all_records, 'reward_mean', source)
        if any(v is not None for v in vals):
            axes[1].plot(steps, vals, color=color, linestyle=ls, label=f"{source}", marker='.')
    axes[1].set_title("Reward Mean"); axes[1].legend(); axes[1].grid(True, alpha=0.3)

    for source, color, ls in [('online', '#ed7d31', '-'), ('demo', '#5b9bd5', '--')]:
        vals = _get(all_records, 'masks_done_rate', source)
        if any(v is not None for v in vals):
            axes[2].plot(steps, vals, color=color, linestyle=ls, label=f"{source}", marker='.')
    axes[2].set_title("Done Rate (masks=0)"); axes[2].legend(); axes[2].grid(True, alpha=0.3)

    for ax in axes:
        ax.set_xlabel("Step")
    plt.suptitle("MC Returns / Rewards / Masks 趋势", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "trend_mc_rewards_masks.png"), dpi=120)
    plt.close()
    _info("  保存 → trend_mc_rewards_masks.png")

    # ── 3. Intervention 趋势 ──
    online_intvn = [r.get('online_intervened_count') for r in all_records]
    demo_intvn = [r.get('demo_intervened_count') for r in all_records]
    if any(v is not None for v in online_intvn + demo_intvn):
        fig, ax = plt.subplots(1, 1, figsize=(10, 4))
        if any(v is not None for v in online_intvn):
            ax.plot(steps, online_intvn, color='#ed7d31', label="online intervened", marker='.')
        if any(v is not None for v in demo_intvn):
            ax.plot(steps, demo_intvn, color='#5b9bd5', label="demo intervened", marker='.')
        ax.set_xlabel("Step"); ax.set_ylabel("Intervened Count")
        ax.set_title("Intervention Count per Batch"); ax.legend(); ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "trend_intervention.png"), dpi=120)
        plt.close()
        _info("  保存 → trend_intervention.png")

    # ── 4. Action-Obs Consistency 趋势 ──
    online_mae = [r.get('online_consistency_mae') for r in all_records]
    demo_mae = [r.get('demo_consistency_mae') for r in all_records]
    if any(v is not None for v in online_mae + demo_mae):
        fig, ax = plt.subplots(1, 1, figsize=(10, 4))
        if any(v is not None for v in demo_mae):
            ax.plot(steps, demo_mae, color='#5b9bd5', label="demo MAE", marker='.')
        if any(v is not None for v in online_mae):
            ax.plot(steps, online_mae, color='#ed7d31', label="online MAE", marker='.')
        ax.set_xlabel("Step"); ax.set_ylabel("MAE")
        ax.set_title("Action-State Delta Consistency (MAE)"); ax.legend(); ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "trend_action_obs_consistency.png"), dpi=120)
        plt.close()
        _info("  保存 → trend_action_obs_consistency.png")

    # ── 5. 图像像素均值趋势 ──
    img_key_set = set()
    for r in all_records:
        for src in ['online', 'demo']:
            for k in r[src]:
                if k.startswith('img_') and k.endswith('_mean'):
                    img_key_set.add(k.replace('img_', '').replace('_mean', ''))
    if img_key_set:
        fig, axes = plt.subplots(1, len(img_key_set), figsize=(7 * len(img_key_set), 4))
        if len(img_key_set) == 1:
            axes = [axes]
        for ax, cam in zip(axes, sorted(img_key_set)):
            for source, color, ls in [('online', '#ed7d31', '-'), ('demo', '#5b9bd5', '--')]:
                vals = [r[source].get(f'img_{cam}_mean') for r in all_records]
                if any(v is not None for v in vals):
                    ax.plot(steps, vals, color=color, linestyle=ls, label=source, marker='.')
            ax.set_title(f"Image Mean — {cam}"); ax.legend(); ax.grid(True, alpha=0.3)
            ax.set_xlabel("Step")
        plt.suptitle("图像像素均值趋势", fontsize=14)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "trend_image_pixel_mean.png"), dpi=120)
        plt.close()
        _info("  保存 → trend_image_pixel_mean.png")


# ═══════════════════════════ 单步详细对比图 ═══════════════════════════════
def plot_single_step_comparison(online_batch, demo_batch, step, out_dir):
    """对单个 step 对比 online vs demo 的 action/mc_returns 分布。"""
    step_dir = os.path.join(out_dir, f"step_{step}")
    os.makedirs(step_dir, exist_ok=True)

    # action boxplot
    oa = online_batch.get("actions")
    da = demo_batch.get("actions")
    if oa is not None and da is not None:
        oa = np.asarray(oa, dtype=np.float32)
        da = np.asarray(da, dtype=np.float32)
        if oa.ndim == 3:
            oa = oa.reshape(-1, oa.shape[-1])
        if da.ndim == 3:
            da = da.reshape(-1, da.shape[-1])
        n_dim = oa.shape[-1]

        fig, axes = plt.subplots(1, n_dim, figsize=(3 * n_dim, 5))
        if n_dim == 1:
            axes = [axes]
        for i in range(n_dim):
            ax = axes[i]
            bp = ax.boxplot([da[:, i], oa[:, i]], tick_labels=["demo", "online"], patch_artist=True)
            bp['boxes'][0].set_facecolor('#5b9bd5')
            bp['boxes'][1].set_facecolor('#ed7d31')
            ax.set_title(f"dim[{i}]", fontsize=9)
            ax.grid(True, alpha=0.3)
        plt.suptitle(f"Step {step}: Actions per-dim", fontsize=13)
        plt.tight_layout()
        plt.savefig(os.path.join(step_dir, "actions_boxplot.png"), dpi=100)
        plt.close()

        # histogram
        fig, axes = plt.subplots(1, n_dim, figsize=(3 * n_dim, 4))
        if n_dim == 1:
            axes = [axes]
        for i in range(n_dim):
            ax = axes[i]
            ax.hist(da[:, i], bins=30, alpha=0.5, label="demo", color='#5b9bd5', density=True)
            ax.hist(oa[:, i], bins=30, alpha=0.5, label="online", color='#ed7d31', density=True)
            ax.set_title(f"dim[{i}]", fontsize=9)
            ax.legend(fontsize=7)
        plt.suptitle(f"Step {step}: Action Distribution", fontsize=13)
        plt.tight_layout()
        plt.savefig(os.path.join(step_dir, "actions_hist.png"), dpi=100)
        plt.close()

    # mc_returns
    omc = online_batch.get("mc_returns")
    dmc = demo_batch.get("mc_returns")
    if omc is not None or dmc is not None:
        fig, ax = plt.subplots(figsize=(8, 4))
        if dmc is not None:
            ax.hist(np.asarray(dmc, dtype=np.float32).ravel(), bins=40, alpha=0.5,
                    label="demo", color='#5b9bd5', density=True)
        if omc is not None:
            ax.hist(np.asarray(omc, dtype=np.float32).ravel(), bins=40, alpha=0.5,
                    label="online", color='#ed7d31', density=True)
        ax.set_title(f"Step {step}: MC Returns Distribution")
        ax.legend(); ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(step_dir, "mc_returns_hist.png"), dpi=100)
        plt.close()

    # rewards
    orw = online_batch.get("rewards")
    drw = demo_batch.get("rewards")
    if orw is not None or drw is not None:
        fig, ax = plt.subplots(figsize=(8, 4))
        if drw is not None:
            ax.hist(np.asarray(drw, dtype=np.float32).ravel(), bins=40, alpha=0.5,
                    label="demo", color='#5b9bd5', density=True)
        if orw is not None:
            ax.hist(np.asarray(orw, dtype=np.float32).ravel(), bins=40, alpha=0.5,
                    label="online", color='#ed7d31', density=True)
        ax.set_title(f"Step {step}: Rewards Distribution")
        ax.legend(); ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(step_dir, "rewards_hist.png"), dpi=100)
        plt.close()


# ─────────── Action-Obs consistency 可视化 ────────────────────────────────
def plot_action_obs_consistency(batch, step, source_label, out_dir):
    """绘制 action vs state delta 的逐维对比图。"""
    result = check_action_obs_consistency(batch, source_label)
    if result is None:
        return

    obs = batch.get("observations", {})
    next_obs = batch.get("next_observations", {})
    actions_np = np.asarray(batch["actions"], dtype=np.float32)
    obs_state = np.asarray(obs[result['state_key']], dtype=np.float32)
    next_obs_state = np.asarray(next_obs[result['state_key']], dtype=np.float32)
    if obs_state.ndim == 3:
        obs_state = obs_state[:, -1]
    if next_obs_state.ndim == 3:
        next_obs_state = next_obs_state[:, -1]
    if actions_np.ndim == 3:
        actions_np = actions_np[:, 0]

    delta_state = next_obs_state - obs_state
    pose_dim = result['pose_dim']

    step_dir = os.path.join(out_dir, f"step_{step}")
    os.makedirs(step_dir, exist_ok=True)

    n_plot = min(pose_dim, 7)
    fig, axes = plt.subplots(n_plot, 1, figsize=(12, 2.5 * n_plot), sharex=True)
    if n_plot == 1:
        axes = [axes]
    n_samples = min(80, len(delta_state))
    for d in range(n_plot):
        ax = axes[d]
        ax.plot(delta_state[:n_samples, d], label="next_obs - obs", alpha=0.7)
        ax.plot(actions_np[:n_samples, d], label="action", alpha=0.7, linestyle="--")
        mae_d = result['per_dim_mae'][d]
        ax.set_ylabel(f"dim[{d}]")
        ax.set_title(f"dim[{d}] — MAE={mae_d:.6f}", fontsize=9)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel("Sample index")
    plt.suptitle(f"Step {step} | {source_label}: Action vs State Delta  (MAE={result['overall_mae']:.6f})", fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(step_dir, f"action_vs_delta_{source_label}.png"), dpi=100)
    plt.close()


# ═════════════════════════════════════════════════════════════════════════════
#  MAIN
# ═════════════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(description="分析所有 learner_sample_batch_step*.pkl")
    parser.add_argument("--debug_dir", default="debug/",
                        help="存放 learner_sample_batch_step*.pkl 的目录")
    parser.add_argument("--out_dir", default="analyze_learner_output",
                        help="分析结果输出目录")
    parser.add_argument("--save_images_every", type=int, default=1,
                        help="每隔多少个 pkl 文件保存一次图片 PNG（默认每个都保存）")
    parser.add_argument("--detail_steps", type=str, default="",
                        help="逗号分隔的 step 列表，对这些 step 生成详细对比图 (留空=全部)")
    parser.add_argument("--max_image_samples", type=int, default=6,
                        help="每个 batch 最多保存多少张图片 (默认 6)")
    parser.add_argument("--stride", type=int, default=1,
                        help="每隔多少个文件处理一次，用于降采样（默认=1 全部处理）")
    parser.add_argument("--max_files", type=int, default=0,
                        help="最多处理多少个文件（0=全部）")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # ══════════════════════════ 文件发现 ═════════════════════════════════
    _head("加载数据")
    all_paths = list_batch_paths(args.debug_dir)   # 仅获取路径，不读文件
    if not all_paths:
        _err(f"在 {args.debug_dir} 中未找到 learner_sample_batch_step*.pkl 文件")
        return
    _ok(f"发现 {len(all_paths)} 个 batch 文件")

    # stride / max_files 下采样
    sampled_paths = all_paths[::args.stride]
    if args.max_files > 0:
        sampled_paths = sampled_paths[:args.max_files]
    _info(f"实际处理: {len(sampled_paths)} 个文件  "
          f"(stride={args.stride}, max_files={args.max_files or 'all'})")
    steps_range = (sampled_paths[0][0], sampled_paths[-1][0])
    _info(f"Step 范围: {steps_range[0]} ~ {steps_range[1]}")

    # 解析 detail_steps
    if args.detail_steps:
        detail_steps = set(int(s.strip()) for s in args.detail_steps.split(",") if s.strip())
    else:
        detail_steps = None  # 按 save_images_every 控制

    # ══════════════════ 首个文件：打印数据结构 ════════════════════════════
    _head("1. 数据结构 (首个 batch)")
    first = load_one_batch(*sampled_paths[0])
    if first is None:
        _err("首个文件加载失败，退出")
        return
    _info(f"Step = {first['_step']}, File = {first.get('_file', 'N/A')}")
    if first.get('_format') == 'legacy_flat':
        _warn("检测到旧格式 (flat batch)，将整个 batch 视为 online_batch，demo_batch 为空")

    _sub("Top-level keys")
    for k, v in sorted(first.items()):
        if k in ('online_batch', 'demo_batch'):
            print(f"  {k}/ (dict, {len(v)} keys)")
        elif isinstance(v, dict):
            print(f"  {k}/ (dict, {len(v)} keys): {v}")
        elif isinstance(v, np.ndarray):
            print(f"  {k}: shape={v.shape}, dtype={v.dtype}")
        else:
            print(f"  {k}: {type(v).__name__} = {v}")
    if first.get('online_batch'):
        _sub("online_batch 结构")
        print_dict_structure(first['online_batch'])
    if first.get('demo_batch'):
        _sub("demo_batch 结构")
        print_dict_structure(first['demo_batch'])

    # 推断图像 key
    all_img_keys = set()
    for src_key in ['online_batch', 'demo_batch']:
        if src_key in first:
            all_img_keys.update(find_image_keys(first[src_key]))
    _info(f"图像 keys: {sorted(all_img_keys)}")

    # ══════════════════════════ 流式逐文件处理 ════════════════════════════
    _head("2. 逐步统计分析（流式处理）")
    all_records = []
    steps_list = []

    pbar = tqdm.tqdm(sampled_paths, desc="处理 batch", ncols=100, unit="file")
    for idx, (step_num, path) in enumerate(pbar):
        pbar.set_postfix(step=step_num)
        data = load_one_batch(step_num, path)
        if data is None:
            continue

        step = data['_step']
        steps_list.append(step)
        online = data.get('online_batch', {})
        demo   = data.get('demo_batch', {})
        intvn  = data.get('_intervention_stats', {})

        online_n = batch_size(online)
        demo_n   = batch_size(demo)

        # 提取统计
        online_stats = extract_stats(online, 'online')
        demo_stats   = extract_stats(demo, 'demo')

        # consistency
        online_cons = check_action_obs_consistency(online, 'online')
        demo_cons   = check_action_obs_consistency(demo, 'demo')

        record = {
            'step': step,
            'online': online_stats,
            'demo': demo_stats,
            'online_intervened_count': intvn.get('online_intervened_count'),
            'demo_intervened_count':   intvn.get('demo_intervened_count'),
            'online_consistency_mae':  online_cons['overall_mae'] if online_cons else None,
            'demo_consistency_mae':    demo_cons['overall_mae']   if demo_cons   else None,
        }
        all_records.append(record)

        # 逐步打印（不影响进度条）
        tqdm.tqdm.write(f"  Step {step:7d} | online={online_n} demo={demo_n}")
        for src_name, stats in [("online", online_stats), ("demo", demo_stats)]:
            parts = []
            if 'action_mean' in stats:
                parts.append(f"act=[{stats['action_min']:.4f},{stats['action_max']:.4f}] "
                             f"μ={stats['action_mean']:.4f} σ={stats['action_std']:.4f}")
            if 'mc_mean' in stats:
                parts.append(f"mc=[{stats['mc_min']:.4f},{stats['mc_max']:.4f}] "
                             f"μ={stats['mc_mean']:.4f}")
            if 'reward_mean' in stats:
                parts.append(f"rew_μ={stats['reward_mean']:.4f}")
            if 'masks_done_rate' in stats:
                parts.append(f"done={stats['masks_done_rate']:.1%}")
            tqdm.tqdm.write(f"    {src_name:8s}: {' | '.join(parts)}")

        # intervention
        o_istr = (f"intervened={intvn.get('online_intervened_count','?')}"
                  if intvn.get('online_has_intervened') else "no field")
        d_istr = (f"intervened={intvn.get('demo_intervened_count','?')}"
                  if intvn.get('demo_has_intervened') else "no field")
        tqdm.tqdm.write(f"    intervention: online({o_istr}) | demo({d_istr})")

        # consistency
        for cons, label in [(demo_cons, 'demo'), (online_cons, 'online')]:
            if cons:
                mae = cons['overall_mae']
                flag = (f"{GREEN}✓{RESET}" if mae < 0.01 else
                        f"{YELLOW}~{RESET}" if mae < 0.05 else f"{RED}✗{RESET}")
                tqdm.tqdm.write(f"    {flag} {label} action-obs MAE={mae:.6f}")

        # 保存图片 PNG
        should_save_images = (idx % args.save_images_every == 0)
        if should_save_images:
            for src_key, src_label in [('online_batch', 'online'), ('demo_batch', 'demo')]:
                if data.get(src_key):
                    save_batch_images(data[src_key], step, src_label, sorted(all_img_keys),
                                      args.out_dir, max_samples=args.max_image_samples)

        # 详细对比图（指定 step 或 detail_steps=None 时全部）
        should_detail = (detail_steps is not None and step in detail_steps)
        if should_detail:
            plot_single_step_comparison(online, demo, step, args.out_dir)
            plot_action_obs_consistency(demo,   step, "demo",   args.out_dir)
            plot_action_obs_consistency(online, step, "online", args.out_dir)

        # 处理完释放内存
        del data, online, demo

    # ══════════════════════════ 趋势图 ════════════════════════════════════
    _head("3. 趋势图")
    plot_trends(all_records, args.out_dir)
    _ok("趋势图已保存")

    # ══════════════════════════ 汇总表格 ══════════════════════════════════
    _head("4. 汇总统计表")
    print(f"  {'Step':>6s}  {'Src':>8s}  {'act_mean':>10s}  {'act_std':>10s}  "
          f"{'mc_mean':>10s}  {'rew_mean':>10s}  {'done%':>7s}  {'cons_MAE':>10s}  {'intvn':>6s}")
    print("  " + "-" * 95)
    for r in all_records:
        for src in ['online', 'demo']:
            s = r[src]
            cons_mae = r.get(f'{src}_consistency_mae')
            intvn_cnt = r.get(f'{src}_intervened_count')
            done_rate = s.get('masks_done_rate')
            done_str = f"{done_rate*100:6.1f}%" if done_rate is not None else "   N/A "
            print(f"  {r['step']:>6d}  {src:>8s}  "
                  f"{s.get('action_mean', float('nan')):>10.5f}  "
                  f"{s.get('action_std', float('nan')):>10.5f}  "
                  f"{s.get('mc_mean', float('nan')):>10.4f}  "
                  f"{s.get('reward_mean', float('nan')):>10.5f}  "
                  f"{done_str}  "
                  f"{cons_mae if cons_mae is not None else float('nan'):>10.6f}  "
                  f"{str(intvn_cnt) if intvn_cnt is not None else 'N/A':>6s}")

    # ══════════════════════════ 保存文本摘要 ══════════════════════════════
    summary_path = os.path.join(args.out_dir, "summary.txt")
    with open(summary_path, "w") as f:
        f.write("Learner Sample Batch Analysis Summary\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Debug dir: {args.debug_dir}\n")
        f.write(f"Total batches processed: {len(all_records)}\n")
        f.write(f"Total files found: {len(all_paths)}\n")
        f.write(f"stride={args.stride}, max_files={args.max_files or 'all'}\n")
        f.write(f"Step range: {min(steps_list) if steps_list else 'N/A'} ~ {max(steps_list) if steps_list else 'N/A'}\n\n")

        f.write(f"{'Step':>6s}  {'Src':>8s}  {'act_mean':>10s}  {'act_std':>10s}  "
                f"{'mc_mean':>10s}  {'rew_mean':>10s}  {'done%':>7s}  {'cons_MAE':>10s}  {'intvn':>6s}\n")
        f.write("-" * 100 + "\n")
        for r in all_records:
            for src in ['online', 'demo']:
                s = r[src]
                cons_mae = r.get(f'{src}_consistency_mae')
                intvn_cnt = r.get(f'{src}_intervened_count')
                done_rate = s.get('masks_done_rate')
                done_str = f"{done_rate*100:6.1f}%" if done_rate is not None else "   N/A "
                f.write(f"{r['step']:>6d}  {src:>8s}  "
                        f"{s.get('action_mean', float('nan')):>10.5f}  "
                        f"{s.get('action_std', float('nan')):>10.5f}  "
                        f"{s.get('mc_mean', float('nan')):>10.4f}  "
                        f"{s.get('reward_mean', float('nan')):>10.5f}  "
                        f"{done_str}  "
                        f"{cons_mae if cons_mae is not None else float('nan'):>10.6f}  "
                        f"{str(intvn_cnt) if intvn_cnt is not None else 'N/A':>6s}\n")

    _ok(f"摘要已保存 → {summary_path}")

    # ══════════════════════════ 输出文件列表 ══════════════════════════════
    _head("分析完成")
    _info(f"输出目录: {args.out_dir}")
    total_files = 0
    for root, dirs, files in os.walk(args.out_dir):
        total_files += len(files)
    _info(f"共生成 {total_files} 个文件")
    for item in sorted(os.listdir(args.out_dir)):
        full = os.path.join(args.out_dir, item)
        if os.path.isdir(full):
            n = sum(len(fs) for _, _, fs in os.walk(full))
            print(f"    {item}/  ({n} files)")
        else:
            print(f"    {item}")


if __name__ == "__main__":
    main()
