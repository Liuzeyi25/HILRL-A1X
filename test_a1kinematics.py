#!/usr/bin/env python3
"""测试 IK 漂移问题：不准确种子 + 固定四元数 vs 实际FK四元数"""

import numpy as np
from a1_x_kenimetic_haoyuan import A1Kinematics
import torch
from scipy.spatial.transform import Rotation as R


# Reset后的实际关节反馈 (有偏差的"当前关节")
RESET_JOINTS = np.array([-0.01531915, 1.82553191, -1.13914894, 0.85808511, -0.05276596, -0.10148936])

# 之前硬编码的固定目标四元数 [x,y,z,w]
FIXED_TARGET_QUAT_XYZW = np.array([-0.04987142, 0.7015382, 0.01178707, 0.71078694])

# 之前 IK 得到的"漂移"解
OBSERVED_IK_RESULT = np.array([-0.01494499, 1.8474945, -1.1707656, 0.8778316, -0.05301241, -0.10262609])


def _set_seed(ik_solver, joints):
    """设置 IK 种子为指定关节角度"""
    ik_solver.prev_q = torch.as_tensor(
        joints, dtype=torch.float32, device=ik_solver.tensor_args.device
    ).unsqueeze(0)


def _solve_and_report(ik_solver, pos, quat_xyzw, seed_joints, label):
    """求解 IK 并报告与种子的偏差"""
    _set_seed(ik_solver, seed_joints)
    result = ik_solver.solve_ik(pos=pos, quat=quat_xyzw)

    if result is None or not result.success.cpu().numpy().any():
        print(f"  [{label}] IK 求解失败")
        return None

    solution = result.js_solution.position.cpu().numpy()[:6]
    diff = solution - seed_joints
    max_diff = np.abs(diff).max()
    max_idx = np.abs(diff).argmax()
    print(f"  [{label}] IK 解:  {np.array2string(solution, precision=8, separator=', ')}")
    print(f"  [{label}] 偏差:   {np.array2string(diff, precision=6, separator=', ')}")
    print(f"  [{label}] 最大偏差: 关节{max_idx+1} = {max_diff:.6f} rad ({np.rad2deg(max_diff):.4f} deg)")
    return solution


def main():
    # ----------------------------------------------------------------
    # 初始化
    # ----------------------------------------------------------------
    print("=" * 70)
    print("IK 漂移测试")
    print("=" * 70)

    ik_solver = A1Kinematics(
        urdf_file="/home/dungeon_master/A1_X/arm/install/mobiman/share/mobiman/urdf/A1X/urdf/a1x.urdf",
        base_link="base_link",
        ee_link="arm_link6",
    )
    print("IK solver 初始化成功\n")

    # ----------------------------------------------------------------
    # 1. 用 Reset 关节角做 FK，得到实际末端位姿
    # ----------------------------------------------------------------
    print("=" * 70)
    print("1. Reset 关节角 FK 分析")
    print("=" * 70)
    fk_pos, fk_quat_wxyz = ik_solver.forward_kinematics(RESET_JOINTS)
    fk_quat_xyzw = fk_quat_wxyz[[1, 2, 3, 0]]  # 转为 [x,y,z,w]
    fk_euler = R.from_quat(fk_quat_xyzw).as_euler("xyz", degrees=True)

    print(f"Reset 关节角: {RESET_JOINTS}")
    print(f"FK 位置:       {fk_pos}")
    print(f"FK 四元数(xyzw): {fk_quat_xyzw}")
    print(f"FK 欧拉角(deg):  {fk_euler}")

    # 对比固定四元数 vs FK四元数
    fixed_euler = R.from_quat(FIXED_TARGET_QUAT_XYZW).as_euler("xyz", degrees=True)
    quat_angle_diff = R.from_quat(fk_quat_xyzw).inv() * R.from_quat(FIXED_TARGET_QUAT_XYZW)
    angle_diff_deg = quat_angle_diff.magnitude() * 180 / np.pi

    print(f"\n固定四元数(xyzw): {FIXED_TARGET_QUAT_XYZW}")
    print(f"固定欧拉角(deg):  {fixed_euler}")
    print(f"FK vs 固定四元数 旋转偏差: {angle_diff_deg:.4f} deg")

    # ----------------------------------------------------------------
    # 2. 核心对比：零 EEF delta 下，固定四元数 vs 实际FK四元数
    # ----------------------------------------------------------------
    print("\n" + "=" * 70)
    print("2. 零 EEF delta IK 对比 (种子 = Reset 关节)")
    print("   目标位置 = FK位置 (模拟 eef_delta=0)")
    print("=" * 70)

    print("\n--- Case A: 使用固定四元数 (旧代码行为, 导致漂移) ---")
    sol_fixed = _solve_and_report(ik_solver, fk_pos, FIXED_TARGET_QUAT_XYZW, RESET_JOINTS, "固定quat")

    print("\n--- Case B: 使用实际 FK 四元数 (修复后行为) ---")
    sol_actual = _solve_and_report(ik_solver, fk_pos, fk_quat_xyzw, RESET_JOINTS, "实际quat")

    # ----------------------------------------------------------------
    # 3. 模拟漂移循环：用固定四元数连续 20 步
    # ----------------------------------------------------------------
    print("\n" + "=" * 70)
    print("3. 模拟漂移循环 (20步, 每步 eef_delta=0, 使用固定四元数)")
    print("   每步: FK(当前关节) -> IK(FK位置, 固定quat, seed=当前关节)")
    print("=" * 70)

    current = RESET_JOINTS.copy()
    cumulative_drift = np.zeros(6)

    for step in range(20):
        _set_seed(ik_solver, current)
        step_pos, step_quat_wxyz = ik_solver.forward_kinematics(current)
        result = ik_solver.solve_ik(pos=step_pos, quat=FIXED_TARGET_QUAT_XYZW)

        if result is None or not result.success.cpu().numpy().any():
            print(f"  Step {step+1:2d}: IK 失败, 停止")
            break

        new_joints = result.js_solution.position.cpu().numpy()[:6]
        step_diff = new_joints - current
        cumulative_drift += step_diff
        max_d = np.abs(step_diff).max()
        cum_max = np.abs(cumulative_drift).max()

        if step < 10 or step % 5 == 4:
            print(f"  Step {step+1:2d}: 单步最大偏差 {max_d:.6f} rad ({np.rad2deg(max_d):.3f} deg) | "
                  f"累计最大漂移 {cum_max:.6f} rad ({np.rad2deg(cum_max):.3f} deg)")

        current = new_joints

    total_drift = current - RESET_JOINTS
    print(f"\n  20步后总漂移: {np.array2string(total_drift, precision=6, separator=', ')}")
    print(f"  总漂移最大值: {np.abs(total_drift).max():.6f} rad ({np.rad2deg(np.abs(total_drift).max()):.3f} deg)")

    # ----------------------------------------------------------------
    # 4. 对照: 用实际 FK 四元数连续 20 步 (应无漂移)
    # ----------------------------------------------------------------
    print("\n" + "=" * 70)
    print("4. 对照组 (20步, 每步 eef_delta=0, 使用实际 FK 四元数)")
    print("   每步: FK(当前关节) -> IK(FK位置, FK四元数, seed=当前关节)")
    print("=" * 70)

    current = RESET_JOINTS.copy()
    cumulative_drift = np.zeros(6)

    for step in range(20):
        _set_seed(ik_solver, current)
        step_pos, step_quat_wxyz = ik_solver.forward_kinematics(current)
        step_quat_xyzw = step_quat_wxyz[[1, 2, 3, 0]]
        result = ik_solver.solve_ik(pos=step_pos, quat=step_quat_xyzw)

        if result is None or not result.success.cpu().numpy().any():
            print(f"  Step {step+1:2d}: IK 失败, 停止")
            break

        new_joints = result.js_solution.position.cpu().numpy()[:6]
        step_diff = new_joints - current
        cumulative_drift += step_diff
        max_d = np.abs(step_diff).max()
        cum_max = np.abs(cumulative_drift).max()

        if step < 10 or step % 5 == 4:
            print(f"  Step {step+1:2d}: 单步最大偏差 {max_d:.6f} rad ({np.rad2deg(max_d):.3f} deg) | "
                  f"累计最大漂移 {cum_max:.6f} rad ({np.rad2deg(cum_max):.3f} deg)")

        current = new_joints

    total_drift = current - RESET_JOINTS
    print(f"\n  20步后总漂移: {np.array2string(total_drift, precision=6, separator=', ')}")
    print(f"  总漂移最大值: {np.abs(total_drift).max():.6f} rad ({np.rad2deg(np.abs(total_drift).max()):.3f} deg)")

    # ----------------------------------------------------------------
    # 5. 一致性测试：100 次零 delta 求解 (实际 FK 四元数, 每次重置种子)
    # ----------------------------------------------------------------
    print("\n" + "=" * 70)
    print("5. 一致性测试 (100次, 每次重置种子到 Reset 关节)")
    print("=" * 70)

    solutions = []
    for i in range(100):
        _set_seed(ik_solver, RESET_JOINTS)
        result = ik_solver.solve_ik(pos=fk_pos, quat=fk_quat_xyzw)
        if result is not None and result.success.cpu().numpy().any():
            solutions.append(result.js_solution.position.cpu().numpy()[:6])

    print(f"成功: {len(solutions)}/100")

    if solutions:
        arr = np.array(solutions)
        std = np.std(arr, axis=0)
        print(f"标准差: {np.array2string(std, precision=8, separator=', ')}")
        print(f"最大标准差: {np.max(std):.8f} rad ({np.rad2deg(np.max(std)):.6f} deg)")

        mean_diff = np.mean(arr, axis=0) - RESET_JOINTS
        print(f"平均解 vs 种子偏差: {np.array2string(mean_diff, precision=6, separator=', ')}")
        print(f"平均偏差最大值: {np.abs(mean_diff).max():.6f} rad ({np.rad2deg(np.abs(mean_diff).max()):.4f} deg)")

        for j in range(6):
            rng = np.ptp(arr[:, j])
            print(f"  关节{j+1}: 范围 {rng:.8f} rad ({np.rad2deg(rng):.6f} deg)")

    print("\n" + "=" * 70)
    print("测试完成")
    print("=" * 70)


if __name__ == "__main__":
    main()
