"""
数据对齐诊断脚本
================
对比三类样本：
  A. offline_demo        — demo_data 目录下的 traj_*.pkl
  B. online_policy       — learner 保存的 diag_batches/online_replay_batch.pkl
                           其中 intervened=False 的部分
  C. online_intervention — 同上，intervened=True 的部分
  (D. learner_demo_batch — learner 从 demo_buffer 采样的 batch，用于验证无损坏)

检查项：
  1. Observation 结构 / shape / dtype
  2. 图像颜色格式推断（RGB vs BGR）+ 可视化预览
  3. 动作各维 min/max/mean/std  +  跨来源对比图
  4. 奖励标注（reward>0 比例、succeed 比例）
  5. intervened 标记统计

运行方法：
  # 只看 offline demo
  python diagnose_data_alignment.py

  # online 训练后，加入 learner 保存的 batch
  python diagnose_data_alignment.py \
      --diag_batch_dir /path/to/checkpoint/diag_batches

  # 完整三路对比
  python diagnose_data_alignment.py \
      --demo_dir experiments/insert_block/demo_data/20260222 \
      --diag_batch_dir /path/to/checkpoint/diag_batches \
      --max_demo_trajs 10 \
      --out_dir diagnose_output
"""

import os
import sys
import glob
import pickle as pkl
import argparse
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESET  = "\033[0m"
GREEN  = "\033[92m"
YELLOW = "\033[93m"
RED    = "\033[91m"
CYAN   = "\033[96m"
BOLD   = "\033[1m"


def _ok(msg):   print(f"{GREEN}  [OK]  {msg}{RESET}")
def _warn(msg): print(f"{YELLOW}  [WARN] {msg}{RESET}")
def _err(msg):  print(f"{RED}  [ERR] {msg}{RESET}")
def _info(msg): print(f"{CYAN}  [INFO] {msg}{RESET}")
def _head(msg): print(f"\n{BOLD}{msg}{RESET}")


# ─────────────────────────────────────────────────────────────────────────────
# 数据加载
# ─────────────────────────────────────────────────────────────────────────────

def load_demo_trajs(demo_dir, glob_pat, max_trajs):
    """
    加载 traj_*.pkl 格式的 offline demo。
    每个文件是一条轨迹 list[transition]，每个 transition 是 dict。
    返回 (all_transitions, list_of_traj_lengths)。
    """
    paths = sorted(glob.glob(os.path.join(demo_dir, glob_pat)))[:max_trajs]
    if not paths:
        return [], []
    transitions = []
    lengths = []
    for p in paths:
        with open(p, "rb") as f:
            traj = pkl.load(f)
        traj = traj if isinstance(traj, list) else [traj]
        transitions.extend(traj)
        lengths.append(len(traj))
    return transitions, lengths


def _to_numpy(v):
    """
    递归把任意嵌套的 FrozenDict / dict / jax array 全转成
    普通 dict / np.ndarray，方便后续按整数索引切片。
    """
    # flax FrozenDict 或普通 dict → 递归展开
    if hasattr(v, "unfreeze"):          # flax.core.FrozenDict
        return {k2: _to_numpy(v2) for k2, v2 in v.unfreeze().items()}
    if isinstance(v, dict):
        return {k2: _to_numpy(v2) for k2, v2 in v.items()}
    # jax / numpy array → numpy
    try:
        import jax
        if isinstance(v, jax.Array):
            return np.asarray(v)
    except ImportError:
        pass
    if hasattr(v, "__array__"):
        return np.asarray(v)
    return v


def _batch_size(batch):
    """从 batch dict 中推断 batch size（取第一个有数值长度的 leaf）。"""
    for v in batch.values():
        v = _to_numpy(v)
        if isinstance(v, np.ndarray) and v.ndim >= 1:
            return v.shape[0]
        if isinstance(v, dict):
            # 递归找子 dict 的 leaf
            sub_n = _batch_size(v)
            if sub_n is not None:
                return sub_n
    return None


def load_batch_pkl(pkl_path):
    """
    加载 learner 保存的 batch pkl。
    格式是 dict{key: array(batch, ...)} 或含 FrozenDict 的嵌套结构。
    转换成 list[transition dict] 方便统一处理。
    """
    with open(pkl_path, "rb") as f:
        batch = pkl.load(f)

    # 把所有 FrozenDict / jax.Array 递归转为普通 dict / np.ndarray
    batch = _to_numpy(batch)

    n = _batch_size(batch)
    if n is None:
        raise ValueError(f"无法推断 batch size: {pkl_path}")

    def _index(v, i):
        """递归按第 0 维切出第 i 个样本。"""
        if isinstance(v, dict):
            return {k2: _index(v2, i) for k2, v2 in v.items()}
        if isinstance(v, np.ndarray):
            return v[i]
        return v

    transitions = [_index(batch, i) for i in range(n)]
    return transitions


# ─────────────────────────────────────────────────────────────────────────────
# 图像
# ─────────────────────────────────────────────────────────────────────────────

def _get_obs(transition):
    """兼容两种格式：transition['observations'] dict 或 flat batch dict。"""
    return transition.get("observations", transition)


def _collect_images(transitions, cam_key, max_n=50):
    """从 transitions 收集指定相机最新帧，返回 (N,H,W,C) 或 None。"""
    imgs = []
    for t in transitions[:max_n]:
        obs = _get_obs(t)
        if cam_key not in obs:
            return None
        img = np.asarray(obs[cam_key])
        if img.ndim == 4:   # (T, H, W, C)
            img = img[-1]
        imgs.append(img)
    return np.stack(imgs)


def detect_color_format(images):
    """
    粗略判断 (N,H,W,3) 图像是 RGB 还是 BGR。
    自然场景 RGB: ch0(R) 均值通常 >= ch2(B)；
    BGR: ch0(B) < ch2(R)，即 ch2 - ch0 > 6。
    返回 (label, ch0, ch1, ch2)。
    """
    flat = images.reshape(-1, images.shape[-1]).astype(np.float32)
    ch0, ch1, ch2 = flat[:, 0].mean(), flat[:, 1].mean(), flat[:, 2].mean()
    diff = float(ch2) - float(ch0)
    if diff > 6:
        label = "BGR"
    elif diff < -6:
        label = "RGB"
    else:
        label = "UNKNOWN(ch_similar)"
    return label, float(ch0), float(ch1), float(ch2)


def check_image_color(transitions, image_keys, label):
    for cam in image_keys:
        imgs = _collect_images(transitions, cam, max_n=50)
        if imgs is None:
            _warn(f"[{label}] 相机 '{cam}' 未找到")
            continue
        fmt, c0, c1, c2 = detect_color_format(imgs)
        color = GREEN if fmt == "RGB" else (RED if fmt == "BGR" else YELLOW)
        print(f"    {color}{cam:30s}  {fmt:<22}  "
              f"ch0={c0:5.1f}  ch1={c1:5.1f}  ch2={c2:5.1f}{RESET}")


def plot_sample_images(transitions, image_keys, label, save_path, n_samples=5):
    cams = [k for k in image_keys
            if _collect_images(transitions[:1], k) is not None]
    if not cams:
        return
    fig, axes = plt.subplots(n_samples, len(cams),
                             figsize=(4 * len(cams), 3 * n_samples),
                             squeeze=False)
    idxs = np.linspace(0, len(transitions) - 1, n_samples, dtype=int)
    for row, idx in enumerate(idxs):
        obs = _get_obs(transitions[idx])
        for col, cam in enumerate(cams):
            ax = axes[row][col]
            if cam not in obs:
                ax.axis("off")
                continue
            img = np.asarray(obs[cam])
            if img.ndim == 4:
                img = img[-1]
            ax.imshow(img)
            ax.set_title(f"{cam} | idx={idx}", fontsize=7)
            ax.axis("off")
    fig.suptitle(f"{label} — sample images (as RGB)", fontsize=9)
    plt.tight_layout()
    plt.savefig(save_path, dpi=80)
    plt.close()
    print(f"  [Saved] images -> {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
# 动作
# ─────────────────────────────────────────────────────────────────────────────

DIM_NAMES = ["dx", "dy", "dz", "drx", "dry", "drz", "d_gripper"]


def collect_actions(transitions):
    acts = []
    for t in transitions:
        a = np.asarray(t["actions"])
        if a.ndim == 2:     # (chunk, D)
            a = a[0]
        acts.append(a)
    return np.stack(acts)   # (N, D)


def summarize_actions(actions, label):
    D = actions.shape[-1]
    stats = {}
    print(f"    {'Dim':<4} {'name':>12}  {'min':>8}  {'max':>8}"
          f"  {'mean':>8}  {'std':>8}  {'%zero':>6}")
    print("    " + "-" * 63)
    for d in range(D):
        col = actions[:, d]
        pct_zero = 100.0 * (np.abs(col) < 1e-8).sum() / max(len(col), 1)
        name = DIM_NAMES[d] if d < len(DIM_NAMES) else f"dim{d}"
        s = dict(min=float(col.min()), max=float(col.max()),
                 mean=float(col.mean()), std=float(col.std()),
                 pct_zero=float(pct_zero))
        stats[d] = s
        flag = "  << all-zero!" if pct_zero > 99 else ""
        print(f"    {d:<4} {name:>12}  {s['min']:>8.4f}  {s['max']:>8.4f}"
              f"  {s['mean']:>8.4f}  {s['std']:>8.4f}  {pct_zero:>5.1f}%{flag}")
    return stats


# ─────────────────────────────────────────────────────────────────────────────
# 奖励
# ─────────────────────────────────────────────────────────────────────────────

def check_reward(transitions, label):
    rewards = np.array([float(t["rewards"]) for t in transitions])
    n = len(rewards)
    n_pos  = int((rewards > 0).sum())
    n_zero = int((rewards == 0).sum())
    n_neg  = int((rewards < 0).sum())
    n_succeed = 0
    for t in transitions:
        info = t.get("infos", {}) or {}
        if isinstance(info, dict):
            v = info.get("succeed", t.get("succeed", None))
            if v:
                n_succeed += 1
    print(f"    样本数               : {n}")
    print(f"    reward > 0           : {n_pos}  ({100*n_pos/n:.1f}%)")
    print(f"    reward = 0           : {n_zero}  ({100*n_zero/n:.1f}%)")
    print(f"    reward < 0           : {n_neg}  ({100*n_neg/n:.1f}%)")
    print(f"    succeed=True (infos) : {n_succeed}  ({100*n_succeed/n:.1f}%)")
    if n_pos == 0:
        _err(f"[{label}] 无 reward>0 样本，成功标注缺失！")
    else:
        _ok(f"[{label}] {n_pos} 个 reward>0 的样本。")


# ─────────────────────────────────────────────────────────────────────────────
# 干预标记
# ─────────────────────────────────────────────────────────────────────────────

def check_intervention_flag(transitions, label):
    flags = [bool(t.get("intervened", False)) for t in transitions]
    n_intvn = sum(flags)
    n = len(flags)
    if n == 0:
        _warn(f"[{label}] 无样本")
        return
    pct = 100 * n_intvn / n
    print(f"    intervened=True  : {n_intvn}  ({pct:.1f}%)")
    print(f"    intervened=False : {n - n_intvn}  ({100 - pct:.1f}%)")


# ─────────────────────────────────────────────────────────────────────────────
# 绘图
# ─────────────────────────────────────────────────────────────────────────────

def plot_action_comparison(all_stats, action_dim, save_path):
    labels = list(all_stats.keys())
    n_rows = action_dim
    fig, axes = plt.subplots(n_rows, 1,
                             figsize=(10, 2.8 * n_rows), squeeze=False)
    bar_w = 0.2
    for d in range(n_rows):
        ax = axes[d][0]
        valid = [(lbl, all_stats[lbl][d])
                 for lbl in labels if d in all_stats[lbl]]
        if not valid:
            ax.axis("off")
            continue
        x = np.arange(len(valid))
        maxs  = [s["max"] for _, s in valid]
        means = [s["mean"] for _, s in valid]
        mins  = [s["min"] for _, s in valid]
        stds  = [s["std"] for _, s in valid]
        vlbls = [lbl for lbl, _ in valid]
        ax.bar(x - bar_w, maxs,  width=bar_w, label="max",  color="#2196F3", alpha=0.75)
        ax.bar(x,         means, width=bar_w, label="mean", color="#4CAF50", alpha=0.75)
        ax.bar(x + bar_w, mins,  width=bar_w, label="min",  color="#F44336", alpha=0.75)
        ax.errorbar(x, means, yerr=stds, fmt="none", color="black", capsize=4)
        ax.axhline(0, color="gray", lw=0.8, ls="--")
        ax.set_xticks(x)
        ax.set_xticklabels(vlbls, fontsize=8)
        dim_name = DIM_NAMES[d] if d < len(DIM_NAMES) else f"dim{d}"
        ax.set_title(f"dim {d}  ({dim_name})", fontsize=10)
        ax.legend(fontsize=7, loc="upper right")
    plt.tight_layout()
    plt.savefig(save_path, dpi=100)
    plt.close()
    print(f"  [Saved] action comparison -> {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
# 单数据源完整分析
# ─────────────────────────────────────────────────────────────────────────────

def analyze_source(label, transitions, image_keys, out_dir):
    print(f"\n{'='*64}")
    print(f"{BOLD}  {label}  ({len(transitions)} samples){RESET}")
    print(f"{'='*64}")
    if not transitions:
        _warn("无样本，跳过。")
        return {}

    # 1. obs 结构
    _head("1. Observation 结构")
    obs0 = _get_obs(transitions[0])
    if isinstance(obs0, dict):
        for k in sorted(obs0.keys()):
            v = np.asarray(obs0[k])
            print(f"    {k:30s}  shape={str(v.shape):24s}  dtype={v.dtype}")
    else:
        _warn(f"obs type={type(obs0)}")

    # 2. 图像颜色
    _head("2. 图像颜色格式")
    check_image_color(transitions, image_keys, label)
    img_path = os.path.join(
        out_dir, f"images_{label.replace(' ', '_').replace('/', '_')}.png")
    plot_sample_images(transitions, image_keys, label, img_path)

    # 3. 动作
    _head("3. 动作维度与范围")
    actions = collect_actions(transitions)
    print(f"    shape={actions.shape}  dtype={actions.dtype}")
    if np.any(np.isnan(actions)):
        _err(f"[{label}] actions 含 NaN！")
    elif np.any(np.isinf(actions)):
        _err(f"[{label}] actions 含 Inf！")
    else:
        _ok(f"[{label}] actions 无 NaN/Inf")
    stats = summarize_actions(actions, label)

    # 4. 奖励
    _head("4. 奖励标注")
    check_reward(transitions, label)

    # 5. 干预标记
    _head("5. 干预标记 (intervened)")
    check_intervention_flag(transitions, label)

    # 6. state 范围
    _head("6. state (proprio) 范围")
    obs0 = _get_obs(transitions[0])
    if isinstance(obs0, dict) and "state" in obs0:
        states = np.stack([
            np.asarray(_get_obs(t)["state"]).reshape(-1)
            for t in transitions
        ])
        print(f"    shape=(N={len(states)}, D={states.shape[-1]})")
        for d in range(states.shape[-1]):
            col = states[:, d]
            print(f"    dim{d:>2d}  "
                  f"min={col.min():>8.4f}  max={col.max():>8.4f}  "
                  f"mean={col.mean():>8.4f}  std={col.std():>8.4f}")
    else:
        _warn("未找到 state key，跳过 proprio 范围检查")

    return stats


# ─────────────────────────────────────────────────────────────────────────────
# main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="检查 offline/online/intervention 三类样本对齐情况"
    )
    parser.add_argument(
        "--demo_dir",
        default="./experiments/a1x_pick_banana/demo_data/20260222",
    )
    parser.add_argument("--demo_glob", default="traj_*.pkl")
    parser.add_argument("--max_demo_trajs", type=int, default=5)
    parser.add_argument(
        "--diag_batch_dir", default=None,
        help=(
            "learner 保存的 diag_batches 目录，"
            "内含 online_replay_batch.pkl 和 demo_batch.pkl"
        ),
    )
    parser.add_argument(
        "--image_keys", nargs="+",
        default=["wrist_1", "side_policy_256"],
    )
    parser.add_argument("--out_dir", default="./diagnose_output")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # ── 加载数据源 ────────────────────────────────────────────────────────────
    sources = {}

    # A. offline demo
    demo_t, traj_lens = load_demo_trajs(
        args.demo_dir, args.demo_glob, args.max_demo_trajs)
    if demo_t:
        _ok(f"offline_demo: {len(demo_t)} transitions "
            f"({len(traj_lens)} trajs, lengths={traj_lens})")
        sources["offline_demo"] = demo_t
    else:
        _warn(f"未找到 demo: {os.path.join(args.demo_dir, args.demo_glob)}")

    # B. learner 保存的 online batch
    if args.diag_batch_dir and os.path.isdir(args.diag_batch_dir):
        replay_pkl = os.path.join(
            args.diag_batch_dir, "online_replay_batch.pkl")
        demo_batch_pkl = os.path.join(
            args.diag_batch_dir, "demo_batch.pkl")

        if os.path.exists(replay_pkl):
            online_all = load_batch_pkl(replay_pkl)
            _ok(f"online replay batch: {len(online_all)} samples")
            policy_t = [t for t in online_all
                        if not t.get("intervened", False)]
            intvn_t  = [t for t in online_all
                        if t.get("intervened", False)]
            if policy_t:
                sources["online_policy"] = policy_t
                _ok(f"  -> online_policy (intervened=False): {len(policy_t)}")
            if intvn_t:
                sources["online_intervention"] = intvn_t
                _ok(f"  -> online_intervention (intervened=True): {len(intvn_t)}")
            if not policy_t and not intvn_t:
                sources["online_replay(all)"] = online_all

        if os.path.exists(demo_batch_pkl):
            demo_batch_t = load_batch_pkl(demo_batch_pkl)
            _ok(f"learner demo_batch: {len(demo_batch_t)} samples")
            sources["learner_demo_batch"] = demo_batch_t
    else:
        _warn(
            "未提供 --diag_batch_dir，跳过 online 样本。\n"
            "  online 训练启动后 learner 会自动保存到:\n"
            "  <checkpoint_path>/diag_batches/"
        )

    if not sources:
        _err("没有任何可用数据源，退出。")
        sys.exit(1)

    # ── 分析每个来源 ──────────────────────────────────────────────────────────
    all_action_stats = {}
    for label, transitions in sources.items():
        stats = analyze_source(label, transitions, args.image_keys, args.out_dir)
        if stats:
            all_action_stats[label] = stats

    # ── 跨源对比 ──────────────────────────────────────────────────────────────
    if len(sources) >= 2:
        print(f"\n{'='*64}")
        print(f"{BOLD}  跨数据源对比{RESET}")
        print(f"{'='*64}")

        # action shape 一致性
        action_shapes = {}
        for lbl, ts in sources.items():
            a = np.asarray(ts[0]["actions"])
            if a.ndim == 2:
                a = a[0]
            action_shapes[lbl] = a.shape
        shapes_set = set(action_shapes.values())
        if len(shapes_set) == 1:
            _ok(f"所有来源 action shape 一致: {shapes_set.pop()}")
        else:
            _err("action shape 不一致！")
            for k, v in action_shapes.items():
                print(f"    {k}: {v}")

        # obs keys 一致性
        obs_keys_map = {}
        for lbl, ts in sources.items():
            obs0 = _get_obs(ts[0])
            if isinstance(obs0, dict):
                obs_keys_map[lbl] = frozenset(obs0.keys())
        if obs_keys_map:
            all_ksets = list(obs_keys_map.values())
            if all(s == all_ksets[0] for s in all_ksets):
                _ok(f"所有来源 obs keys 一致: {sorted(all_ksets[0])}")
            else:
                _warn("obs keys 不完全一致：")
                for k, v in obs_keys_map.items():
                    print(f"    {k}: {sorted(v)}")

        # 动作范围对比图
        if all_action_stats:
            first_stats = next(iter(all_action_stats.values()))
            action_dim = len(first_stats)
            plot_path = os.path.join(
                args.out_dir, "action_range_comparison.png")
            plot_action_comparison(all_action_stats, action_dim, plot_path)

    print(f"\n{'='*64}")
    print(f"{BOLD}  诊断完成。输出目录: {args.out_dir}{RESET}")
    print(f"{'='*64}\n")


if __name__ == "__main__":
    main()
