"""
检查 demo/online buffer 中 action 各维度的数量级和数值变化
"""
import pickle as pkl
import numpy as np
import glob


def sep(title):
    print(f"\n{'='*65}")
    print(f"  {title}")
    print('='*65)


def analyze_actions(all_actions, label=""):
    """all_actions: (T, action_dim) numpy array"""
    T, D = all_actions.shape
    print(f"\n  [{label}] 共 {T} 步，action_dim={D}")
    print(f"  {'dim':<5} {'min':>8} {'max':>8} {'mean':>8} {'std':>8}  说明")
    print(f"  {'-'*55}")
    dim_labels = ['dx', 'dy', 'dz', 'droll', 'dpitch', 'dyaw', 'gripper']
    for i in range(D):
        col = all_actions[:, i]
        label_str = dim_labels[i] if i < len(dim_labels) else f'dim{i}'
        print(f"  {i:<5} {col.min():>8.4f} {col.max():>8.4f}"
              f" {col.mean():>8.4f} {col.std():>8.4f}  {label_str}")


# ─── 1. 分析 Demo 数据 ────────────────────────────────────────────────────────
sep("1. Demo 数据 action 数值（实际执行的人工操作）")

demo_files = sorted(
    glob.glob("experiments/insert_block/demo_data/20260222/*.pkl")
)[:5]

if not demo_files:
    print("  ⚠️  未找到 demo pkl 文件，请检查路径")
else:
    all_demo_actions = []
    for fpath in demo_files:
        with open(fpath, "rb") as f:
            transitions = pkl.load(f)
        traj_actions = []
        for t in transitions:
            a = np.asarray(t['actions'])
            a_step = a[0] if a.ndim == 2 else a  # 取第一步
            traj_actions.append(a_step)
            all_demo_actions.append(a_step)

        traj_arr = np.array(traj_actions)
        fname = fpath.split('/')[-1]
        print(f"\n  文件: {fname}  ({len(traj_actions)} steps)")
        print(f"  前3步: ")
        for i, row in enumerate(traj_arr[:3]):
            print(f"    step {i}: {np.round(row, 4)}")
        print(f"  整体: min={traj_arr.min():.4f}  "
              f"max={traj_arr.max():.4f}  mean={traj_arr.mean():.4f}")

    analyze_actions(np.array(all_demo_actions), label="全部 Demo 汇总")


# ─── 2. 分析 Online Buffer 数据（replay buffer pkl）────────────────────────────
sep("2. Online Buffer 数据 action 数值（实际执行的动作）")

# 尝试多个可能的路径
buffer_patterns = [
    "experiments/insert_block/conrft/*/buffer/*.pkl",
    "experiments/*/buffer/*.pkl",
    "experiments/insert_block/*/buffer/*.pkl",
]
buffer_files = []
for pat in buffer_patterns:
    buffer_files = sorted(glob.glob(pat))
    if buffer_files:
        break

if not buffer_files:
    print("  ⚠️  未找到 online buffer pkl，跳过")
else:
    all_online_actions = []
    for fpath in buffer_files[:3]:
        with open(fpath, "rb") as f:
            transitions = pkl.load(f)
        traj_actions = []
        intervened_actions = []
        for t in transitions:
            a = np.asarray(t['actions'])
            a_step = a[0] if a.ndim == 2 else a
            traj_actions.append(a_step)
            all_online_actions.append(a_step)
            if t.get('intervened', False):
                intervened_actions.append(a_step)

        traj_arr = np.array(traj_actions)
        fname = fpath.split('/')[-1]
        print(f"\n  文件: {fname}  ({len(traj_actions)} steps, "
              f"intervened={len(intervened_actions)})")
        print(f"  前3步:")
        for i, row in enumerate(traj_arr[:3]):
            print(f"    step {i}: {np.round(row, 4)}")
        if intervened_actions:
            intvn_arr = np.array(intervened_actions)
            print(f"  干预步 actions ({len(intvn_arr)} 步):")
            for i, row in enumerate(intvn_arr[:3]):
                print(f"    intvn {i}: {np.round(row, 4)}")

    if all_online_actions:
        analyze_actions(np.array(all_online_actions), label="全部 Online 汇总")


# ─── 3. 对比：Demo vs Online 数值差异 ────────────────────────────────────────
sep("3. Demo vs Online 数量级对比（逐维度）")

if demo_files and buffer_files and all_demo_actions and all_online_actions:
    demo_arr = np.array(all_demo_actions)
    online_arr = np.array(all_online_actions)
    D = min(demo_arr.shape[1], online_arr.shape[1])
    dim_labels = ['dx', 'dy', 'dz', 'droll', 'dpitch', 'dyaw', 'gripper']
    print(f"\n  {'dim':<8} {'demo_mean':>10} {'demo_std':>9}"
          f" {'online_mean':>12} {'online_std':>10}  {'ratio(o/d)':>10}")
    print(f"  {'-'*65}")
    for i in range(D):
        dm, ds = demo_arr[:, i].mean(), demo_arr[:, i].std()
        om, os_ = online_arr[:, i].mean(), online_arr[:, i].std()
        ratio = om / dm if abs(dm) > 1e-6 else float('inf')
        lbl = dim_labels[i] if i < len(dim_labels) else f'dim{i}'
        print(f"  {lbl:<8} {dm:>10.4f} {ds:>9.4f}"
              f" {om:>12.4f} {os_:>10.4f}  {ratio:>10.3f}")
elif not buffer_files:
    print("  ⚠️  无 online buffer 数据，跳过对比")
