"""
分析 demo trajectory pkl 文件，验证状态变化与动作是否对齐。

用法:
  python examples/analyze_demo_traj.py \
      --traj_path examples/experiments/toast_bread/demo_data/20260227/traj_001_2026-02-27_15-06-41.pkl
"""

import pickle as pkl
import numpy as np
import argparse
import sys

# ⚠️ 重要：GelloIntervention.step() 中 info["intervene_action_eef"] = target_a1x_joints
#    即存储的是 Gello→A1X 映射后的「绝对关节目标位置 (rad)」，
#    并 NOT 是归一化的 EEF delta。因此与 Δstate(EEF位置差) 的直接比较没有物理意义。
#
# 下面的 ACTION_SCALE 保留供对照，但分析会同时打印原始动作值以供判断。
# [x, y, z, roll, pitch, yaw, gripper]
ACTION_SCALE = np.array([0.005, 0.005, 0.005, 0.0, 0.0, 0.0, 0.0])


# ─── 辅助 ─────────────────────────────────────────────────────────────────────

def _find_state_keys(obs: dict):
    """从 obs dict 中找到可用于对比的向量状态键（非图像）。"""
    state_keys = []
    for k, v in obs.items():
        v = np.asarray(v)
        if v.ndim <= 2 and v.size <= 64:          # 小向量，不是图像
            state_keys.append(k)
    return state_keys


def _get_ee_or_joints(obs: dict):
    """优先取 ee_pos / state，次选第一个小向量键。"""
    priority = ["state", "ee_pos", "tcp_pose", "joint_positions", "proprio"]
    for k in priority:
        if k in obs:
            return k, np.asarray(obs[k]).flatten()
    keys = _find_state_keys(obs)
    if keys:
        return keys[0], np.asarray(obs[keys[0]]).flatten()
    return None, None


# ─── 主分析 ───────────────────────────────────────────────────────────────────

def analyze(traj_path: str, n_show: int = 10, verbose_all: bool = False):
    with open(traj_path, "rb") as f:
        traj = pkl.load(f)

    print(f"\n{'='*65}")
    print(f"  文件: {traj_path}")
    print(f"  轨迹长度: {len(traj)} 个 transition")
    print(f"{'='*65}\n")

    # ── 1. 打印第一条 transition 的完整结构 ──
    t0 = traj[0]
    print("【Transition 结构】")
    print(f"  顶层 keys: {list(t0.keys())}")

    obs0 = t0["observations"]
    if isinstance(obs0, dict):
        print(f"  observations keys: {list(obs0.keys())}")
        for k, v in obs0.items():
            v = np.asarray(v)
            print(f"    {k}: shape={v.shape}, dtype={v.dtype}")
    else:
        obs0_arr = np.asarray(obs0)
        print(f"  observations: shape={obs0_arr.shape}, dtype={obs0_arr.dtype}")

    act0 = np.asarray(t0["actions"])
    print(f"  actions:      shape={act0.shape}, dtype={act0.dtype}")
    print(f"  actions[0]:   {np.round(act0, 4)}")
    print(f"  rewards:      {t0['rewards']}")
    print(f"  dones:        {t0['dones']}")
    print(f"  masks:        {t0['masks']}")

    # ── 1b. 动作语义快速诊断 ──
    actions_all = np.array([np.asarray(t["actions"]).flatten() for t in traj])
    print(f"\n  ⚠️  ACTION 语义诊断")
    print(f"  action 全集范围:  min={actions_all.min(axis=0)[:7].round(4)}")
    print(f"                    max={actions_all.max(axis=0)[:7].round(4)}")
    print(f"  如果 action 是 A1X 绝对关节目标 (rad)，范围应在 ~[-3, 3]")
    print(f"  如果 action 是归一化 EEF delta，范围应严格在 [-1, 1]")
    print(f"  → GelloIntervention.step() 中存的是 target_a1x_joints (绝对关节目标)，")
    print(f"    而非归一化 EEF delta！scaled_action = action × ACTION_SCALE 没有物理意义。")

    # ── 2. 时序对齐分析 ──
    print(f"\n{'─'*65}")
    print("【时序对齐验证】")
    print("  obs_t 是 2 帧堆叠: [frame_{t-1}, frame_t]")
    print("  action_t 执行于 frame_t → 产生 frame_{t+1}")
    print("  next_obs_t = [frame_t, frame_{t+1}]")
    print("  验证: obs_t 的最后一帧 == next_obs_t 的第一帧 (frame_t 共享)\n")

    # 找状态键
    obs_dict = traj[0]["observations"]
    if isinstance(obs_dict, dict):
        state_key, _ = _get_ee_or_joints(obs_dict)
    else:
        state_key = None

    if state_key:
        print(f"  使用状态键: '{state_key}' 进行对齐验证")

        mismatch_count = 0
        for i in range(min(len(traj), len(traj) - 1)):
            obs_now  = np.asarray(traj[i]["observations"][state_key])
            next_obs = np.asarray(traj[i]["next_observations"][state_key])

            obs_nxt  = np.asarray(traj[i + 1]["observations"][state_key]) if i + 1 < len(traj) else None

            # 堆叠维度: 第0维是帧维还是展平了?
            # 尝试两种情况: (2, D) 或 (2*D,)
            if obs_now.ndim == 2:
                last_frame_of_obs   = obs_now[-1]     # frame_t
                first_frame_of_next = next_obs[0]     # 应该也是 frame_t
            elif obs_now.ndim == 1 and obs_now.shape[0] % 2 == 0:
                D = obs_now.shape[0] // 2
                last_frame_of_obs   = obs_now[D:]
                first_frame_of_next = next_obs[:D]
            else:
                last_frame_of_obs   = obs_now
                first_frame_of_next = next_obs

            err = np.max(np.abs(last_frame_of_obs - first_frame_of_next))
            if err > 1e-6:
                mismatch_count += 1

        print(f"  overlap 一致性检查 ({len(traj)-1} 对): "
              f"{'✅ 全部一致' if mismatch_count == 0 else f'❌ {mismatch_count} 对不一致'}")
    else:
        print("  ⚠️  未找到小向量状态键，跳过 overlap 一致性检查")

    # ── 3. 逐步动作 vs 状态变化 ──
    print(f"\n{'─'*65}")
    print(f"【逐步 scaled_action ↔ Δstate（前 {n_show} 步）】")
    print(f"  scaled_action = raw_action × ACTION_SCALE {ACTION_SCALE.tolist()}\n")

    show_count = len(traj) if verbose_all else min(n_show, len(traj))

    header_printed = False
    step_errors = []   # 每步 scaled_action[:3] vs delta_state[:3] 的 L∞ 误差

    for i in range(show_count):
        t = traj[i]
        raw_act = np.asarray(t["actions"]).flatten()
        # 对齐长度后缩放
        scale = ACTION_SCALE[:len(raw_act)] if len(raw_act) <= len(ACTION_SCALE) \
                else np.concatenate([ACTION_SCALE, np.zeros(len(raw_act) - len(ACTION_SCALE))])
        scaled_act = raw_act * scale
        rew  = t["rewards"]
        done = t["dones"]

        if state_key:
            obs_arr  = np.asarray(t["observations"][state_key])
            nobs_arr = np.asarray(t["next_observations"][state_key])

            # 提取 frame_t 和 frame_{t+1}
            if obs_arr.ndim == 2:
                frame_t   = obs_arr[-1]
                frame_tp1 = nobs_arr[-1]
            elif obs_arr.ndim == 1 and obs_arr.shape[0] % 2 == 0:
                D = obs_arr.shape[0] // 2
                frame_t   = obs_arr[D:]
                frame_tp1 = nobs_arr[D:]
            else:
                frame_t   = obs_arr
                frame_tp1 = nobs_arr

            delta_state = frame_tp1 - frame_t

            # 仅比较 scale 非零的维度（前3维 xyz）
            nonzero_dims = np.where(scale != 0)[0]
            if len(nonzero_dims) > 0:
                err = np.max(np.abs(scaled_act[nonzero_dims] - delta_state[nonzero_dims]))
                step_errors.append(err)
            else:
                err = float('nan')

            if not header_printed:
                print(f"  {'步':>4}  {'scaled_act[:3]':>28}  {'Δstate[:3]':>28}  {'err(L∞)':>9}  {'rew':>5}")
                print(f"  {'─'*4}  {'─'*28}  {'─'*28}  {'─'*9}  {'─'*5}")
                header_printed = True

            sa_disp  = np.array2string(scaled_act[:3],   precision=5, separator=',', suppress_small=True)
            ds_disp  = np.array2string(delta_state[:3],  precision=5, separator=',', suppress_small=True)
            print(f"  {i:>4}  {sa_disp:>28}  {ds_disp:>28}  {err:>9.5f}  {rew:>5.2f}")
        else:
            sa_disp = np.array2string(scaled_act[:6], precision=5, separator=',', suppress_small=True)
            print(f"  步 {i:>3}  scaled_action={sa_disp}  rew={rew:.2f}  done={done}")

    # ── 4. 统计摘要 ──
    print(f"\n{'─'*65}")
    print("【统计摘要】")

    rewards = np.array([t["rewards"] for t in traj])
    dones   = np.array([t["dones"]   for t in traj])
    actions = np.array([np.asarray(t["actions"]).flatten() for t in traj])

    print(f"  总 transition 数:  {len(traj)}")
    print(f"  累计 reward:       {rewards.sum():.4f}")
    print(f"  非零 reward 步数:  {(rewards != 0).sum()}")
    print(f"  done=True 步数:    {dones.sum()}")
    print(f"  action 维度:       {actions.shape[1]}")
    print(f"  action 均值:       {actions.mean(axis=0)[:7]}")
    print(f"  action 标准差:     {actions.std(axis=0)[:7]}")
    print(f"  action 最大绝对值: {np.abs(actions).max(axis=0)[:7]}")

    if state_key:
        all_raw_acts = np.array([np.asarray(t["actions"]).flatten() for t in traj])
        scale_full   = ACTION_SCALE[:all_raw_acts.shape[1]] if all_raw_acts.shape[1] <= len(ACTION_SCALE) \
                       else np.concatenate([ACTION_SCALE, np.zeros(all_raw_acts.shape[1] - len(ACTION_SCALE))])
        all_scaled   = all_raw_acts * scale_full[None, :]

        def get_last_frame(obs_dict, key):
            arr = np.asarray(obs_dict[key])
            if arr.ndim == 2:
                return arr[-1]
            elif arr.ndim == 1 and arr.shape[0] % 2 == 0:
                return arr[arr.shape[0]//2:]
            return arr

        obs_frames  = np.array([get_last_frame(t["observations"],      state_key) for t in traj])
        nobs_frames = np.array([get_last_frame(t["next_observations"], state_key) for t in traj])
        all_deltas  = nobs_frames - obs_frames

        nonzero_dims = np.where(scale_full != 0)[0]
        if len(nonzero_dims) > 0:
            errs_all = np.max(np.abs(all_scaled[:, nonzero_dims] - all_deltas[:, nonzero_dims]), axis=1)
            print(f"\n  状态键 '{state_key}' 维度: {obs_frames.shape[1]}")
            print(f"  比较维度 (scale≠0): {nonzero_dims.tolist()}")
            print(f"\n  ⚠️  以下误差「SCALED action vs Δstate」因 action 存的是关节目标(非EEF delta)")
            print(f"     数值本身无物理意义，仅供参考：")
            print(f"  scaled_act vs Δstate  误差 均值: {errs_all.mean():.6f} m  (≈ {errs_all.mean()*1000:.2f} mm)")
            print(f"  scaled_act vs Δstate  误差 中位数: {np.median(errs_all):.6f} m  (≈ {np.median(errs_all)*1000:.2f} mm)")
            print(f"  scaled_act vs Δstate  误差 最大值: {errs_all.max():.6f} m  (≈ {errs_all.max()*1000:.2f} mm)")

            # 更有意义的分析：检查 raw action[:3] 和 Δstate[:3] 的方向一致性
            # (若 Gello 的 joint target 与 EEF 移动方向有线性关系，余弦相似度应为正)
            act_xyz = all_raw_acts[:, :3]
            ds_xyz  = all_deltas[:, :3]
            norms_a = np.linalg.norm(act_xyz, axis=1, keepdims=True) + 1e-9
            norms_s = np.linalg.norm(ds_xyz,  axis=1, keepdims=True) + 1e-9
            cos_sim = np.sum((act_xyz / norms_a) * (ds_xyz / norms_s), axis=1)
            print(f"\n  ─ 方向一致性分析 (raw_action[:3] vs Δstate[:3]) ─")
            print(f"  余弦相似度 均值:   {cos_sim.mean():.4f}  (1.0=完全同向, -1=完全反向)")
            print(f"  余弦相似度 中位数: {np.median(cos_sim):.4f}")
            print(f"  同向步数 (cos>0):  {(cos_sim > 0).sum()} / {len(cos_sim)}")
            print(f"  强同向步数 (cos>0.5): {(cos_sim > 0.5).sum()} / {len(cos_sim)}")
            print(f"  → 若强同向占比高，说明关节目标与EEF位移方向吻合，否则坐标系/控制模式有问题")

            print(f"\n  按维度分析 (scaled_act vs Δstate, 仅 scale≠0 维度):")
            for d in nonzero_dims:
                diff_d = np.abs(all_scaled[:, d] - all_deltas[:, d])
                print(f"    dim {d}: 误差均值={diff_d.mean():.6f}  最大={diff_d.max():.6f}  "
                      f"scaled_act 范围=[{all_scaled[:,d].min():.5f}, {all_scaled[:,d].max():.5f}]  "
                      f"Δstate 范围=[{all_deltas[:,d].min():.5f}, {all_deltas[:,d].max():.5f}]")
        else:
            print("  ⚠️  ACTION_SCALE 全为 0，无可比较的非零维度")

        state_deltas = np.diff(obs_frames, axis=0)
        print(f"\n  状态变化 均值绝对值:  {np.abs(state_deltas).mean(axis=0)[:7]}")
        print(f"  状态变化 最大绝对值:  {np.abs(state_deltas).max(axis=0)[:7]}")

    print(f"\n{'='*65}\n")


# ─── 入口 ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="分析 demo trajectory pkl")
    parser.add_argument("--traj_path", type=str,
                        default="examples/experiments/toast_bread/demo_data/20260227/traj_001_2026-02-27_15-06-41.pkl",
                        help="trajectory pkl 文件路径")
    parser.add_argument("--n_show", type=int, default=15,
                        help="打印前 N 步的对齐信息")
    parser.add_argument("--verbose_all", action="store_true",
                        help="打印所有步骤（忽略 n_show）")
    args = parser.parse_args()

    analyze(args.traj_path, n_show=args.n_show, verbose_all=args.verbose_all)
