#!/usr/bin/env python3
"""
分析 EEF Delta 控制精度

功能：
1. 读取轨迹数据
2. 对于每一步，计算：当前状态 + delta 动作 → 预测的下一状态
3. 与实际的下一状态比较
4. 分析位置、姿态、夹爪的误差

用法：
    python analyze_eef_control.py
    python analyze_eef_control.py --pkl path/to/data.pkl
    python analyze_eef_control.py --verbose  # 显示所有帧详情
"""

import pickle
import numpy as np
from scipy.spatial.transform import Rotation as R
import sys

# ==================== 配置 ====================
DEFAULT_PKL_PATH = "/home/dungeon_master/conrft/examples/experiments/a1x_pick_banana/demo_data/traj_001_manual_2026-02-06_19-39-14.pkl"


def euler_to_quat(euler):
    """欧拉角转四元数 [qx, qy, qz, qw]"""
    rot = R.from_euler('xyz', euler)
    return rot.as_quat()  # [qx, qy, qz, qw]


def quat_to_euler(quat):
    """四元数 [qx, qy, qz, qw] 转欧拉角"""
    rot = R.from_quat(quat)
    return rot.as_euler('xyz')


def apply_eef_delta(current_eef, delta_action):
    """
    应用 EEF delta 动作到当前状态
    
    Args:
        current_eef: [x, y, z, roll, pitch, yaw, gripper]
        delta_action: [dx, dy, dz, droll, dpitch, dyaw, dgripper]
    
    Returns:
        predicted_eef: [x, y, z, roll, pitch, yaw, gripper]
    """
    # 位置：直接相加
    predicted_pos = current_eef[:3] + delta_action[:3]
    
    # 姿态：旋转合成
    current_rot = R.from_euler('xyz', current_eef[3:6])
    delta_rot = R.from_euler('xyz', delta_action[3:6])
    predicted_rot = delta_rot * current_rot  # 应用增量
    predicted_euler = predicted_rot.as_euler('xyz')
    
    # 夹爪：直接相加
    predicted_gripper = current_eef[6] + delta_action[6]
    
    return np.concatenate([predicted_pos, predicted_euler, [predicted_gripper]])


def compute_rotation_error(euler1, euler2):
    """
    计算两个姿态之间的旋转误差（准确方法）
    
    Returns:
        error_angle: 旋转轴角误差（弧度）
    """
    rot1 = R.from_euler('xyz', euler1)
    rot2 = R.from_euler('xyz', euler2)
    
    # R_error = rot2 * rot1^(-1)
    rot_error = rot2 * rot1.inv()
    
    return rot_error.magnitude()


def main():
    # 解析命令行参数
    pkl_path = DEFAULT_PKL_PATH
    verbose = False
    
    for i, arg in enumerate(sys.argv[1:]):
        if arg == '--pkl' and i + 1 < len(sys.argv) - 1:
            pkl_path = sys.argv[i + 2]
        elif arg == '--verbose':
            verbose = True
    
    # 读取数据
    print("=" * 80)
    print("🔍 EEF Delta 控制精度分析")
    print("=" * 80)
    print(f"数据文件: {pkl_path}")
    
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    
    print(f"✓ 加载轨迹数据: {len(data)} 帧")
    
    # 提取状态和动作
    states = []
    delta_actions = []
    
    for i, frame in enumerate(data):
        if 'observations' in frame and 'state' in frame['observations']:
            state = frame['observations']['state']
            # state 可能是 dict (新格式) 或 array (旧格式)
            if isinstance(state, dict) and 'ee_pos_rot_gripper' in state:
                # 新格式：直接取 ee_pos_rot_gripper
                states.append(state['ee_pos_rot_gripper'])
            elif isinstance(state, np.ndarray):
                # 旧格式：state 是 (2, 7) 数组，第一个手臂
                # 但这个格式可能不包含 ee_pos_rot_gripper，跳过
                states.append(None)
            else:
                states.append(None)
        else:
            states.append(None)
        
        if 'infos' in frame and 'intervene_action_eef' in frame['infos']:
            delta_actions.append(frame['infos']['intervene_action_eef'])
        else:
            delta_actions.append(None)
    
    # 统计数据
    pos_errors = []
    rot_errors = []
    gripper_errors = []
    error_frames = []  # 记录帧号，用于后续分析
    
    valid_count = 0
    
    num_to_print = 10 if not verbose else len(data) - 1
    print(f"\n📊 逐帧分析 (前 {min(num_to_print, len(data)-1)} 帧):\n")
    
    for i in range(len(data) - 1):  # 最后一帧没有下一状态
        current_state = states[i]
        delta_action = delta_actions[i]
        next_state = states[i + 1]
        
        # 跳过缺失数据
        if current_state is None or delta_action is None or next_state is None:
            continue
        
        # 预测下一状态
        predicted_next = apply_eef_delta(current_state, delta_action)
        
        # 计算误差
        pos_error = np.linalg.norm(next_state[:3] - predicted_next[:3])
        rot_error = compute_rotation_error(next_state[3:6], predicted_next[3:6])
        gripper_error = abs(next_state[6] - predicted_next[6])
        
        pos_errors.append(pos_error)
        rot_errors.append(rot_error)
        gripper_errors.append(gripper_error)
        error_frames.append(i)
        
        valid_count += 1
        
        # 打印详细信息
        should_print = verbose or i < num_to_print
        if should_print:
            print(f"帧 {i} → {i+1}:")
            print(f"  Delta 动作:  pos=[{delta_action[0]:7.4f}, {delta_action[1]:7.4f}, {delta_action[2]:7.4f}] "
                  f"rot=[{delta_action[3]:7.4f}, {delta_action[4]:7.4f}, {delta_action[5]:7.4f}] "
                  f"gripper={delta_action[6]:6.3f}")
            print(f"  当前状态:    pos=[{current_state[0]:7.4f}, {current_state[1]:7.4f}, {current_state[2]:7.4f}] "
                  f"rot=[{current_state[3]:7.4f}, {current_state[4]:7.4f}, {current_state[5]:7.4f}] "
                  f"gripper={current_state[6]:6.3f}")
            print(f"  预测下一状态: pos=[{predicted_next[0]:7.4f}, {predicted_next[1]:7.4f}, {predicted_next[2]:7.4f}] "
                  f"rot=[{predicted_next[3]:7.4f}, {predicted_next[4]:7.4f}, {predicted_next[5]:7.4f}] "
                  f"gripper={predicted_next[6]:6.3f}")
            print(f"  实际下一状态: pos=[{next_state[0]:7.4f}, {next_state[1]:7.4f}, {next_state[2]:7.4f}] "
                  f"rot=[{next_state[3]:7.4f}, {next_state[4]:7.4f}, {next_state[5]:7.4f}] "
                  f"gripper={next_state[6]:6.3f}")
            print(f"  ❗ 误差: 位置={pos_error*1000:.2f}mm, 姿态={np.degrees(rot_error):.2f}°, 夹爪={gripper_error:.3f}")
            print()
    
    # 统计汇总
    if valid_count == 0:
        print("❌ 没有有效数据进行分析")
        return
    
    pos_errors = np.array(pos_errors)
    rot_errors = np.array(rot_errors)
    gripper_errors = np.array(gripper_errors)
    
    print("\n" + "=" * 80)
    print("📈 统计汇总 (全部 {} 个有效转移)".format(valid_count))
    print("=" * 80)
    
    print("\n位置误差 (米):")
    print(f"  平均值: {np.mean(pos_errors)*1000:.3f} mm")
    print(f"  中位数: {np.median(pos_errors)*1000:.3f} mm")
    print(f"  最大值: {np.max(pos_errors)*1000:.3f} mm")
    print(f"  标准差: {np.std(pos_errors)*1000:.3f} mm")
    print(f"  95分位: {np.percentile(pos_errors, 95)*1000:.3f} mm")
    
    print("\n姿态误差 (旋转角度):")
    print(f"  平均值: {np.degrees(np.mean(rot_errors)):.3f}°")
    print(f"  中位数: {np.degrees(np.median(rot_errors)):.3f}°")
    print(f"  最大值: {np.degrees(np.max(rot_errors)):.3f}°")
    print(f"  标准差: {np.degrees(np.std(rot_errors)):.3f}°")
    print(f"  95分位: {np.degrees(np.percentile(rot_errors, 95)):.3f}°")
    
    print("\n夹爪误差 (归一化):")
    print(f"  平均值: {np.mean(gripper_errors):.4f}")
    print(f"  中位数: {np.median(gripper_errors):.4f}")
    print(f"  最大值: {np.max(gripper_errors):.4f}")
    print(f"  标准差: {np.std(gripper_errors):.4f}")
    
    # 精度评估
    print("\n" + "=" * 80)
    print("🎯 精度评估")
    print("=" * 80)
    
    pos_under_1mm = np.sum(pos_errors < 0.001) / valid_count * 100
    pos_under_5mm = np.sum(pos_errors < 0.005) / valid_count * 100
    pos_under_1cm = np.sum(pos_errors < 0.01) / valid_count * 100
    
    rot_under_1deg = np.sum(rot_errors < np.radians(1.0)) / valid_count * 100
    rot_under_5deg = np.sum(rot_errors < np.radians(5.0)) / valid_count * 100
    rot_under_10deg = np.sum(rot_errors < np.radians(10.0)) / valid_count * 100
    
    print(f"\n位置精度:")
    print(f"  {pos_under_1mm:.1f}% 的步数误差 < 1mm")
    print(f"  {pos_under_5mm:.1f}% 的步数误差 < 5mm")
    print(f"  {pos_under_1cm:.1f}% 的步数误差 < 1cm")
    
    print(f"\n姿态精度:")
    print(f"  {rot_under_1deg:.1f}% 的步数误差 < 1°")
    print(f"  {rot_under_5deg:.1f}% 的步数误差 < 5°")
    print(f"  {rot_under_10deg:.1f}% 的步数误差 < 10°")
    
    # 找出误差最大的帧
    print("\n" + "=" * 80)
    print("⚠️  误差最大的 5 帧")
    print("=" * 80)
    
    # 位置误差最大
    pos_indices = np.argsort(pos_errors)[-5:][::-1]
    print("\n位置误差最大的帧:")
    for idx in pos_indices:
        frame_num = error_frames[idx]
        print(f"  帧 {frame_num} → {frame_num+1}: {pos_errors[idx]*1000:.2f} mm")
    
    # 姿态误差最大
    rot_indices = np.argsort(rot_errors)[-5:][::-1]
    print("\n姿态误差最大的帧:")
    for idx in rot_indices:
        frame_num = error_frames[idx]
        print(f"  帧 {frame_num} → {frame_num+1}: {np.degrees(rot_errors[idx]):.2f}°")
    
    # 结论
    print("\n" + "=" * 80)
    print("💡 结论")
    print("=" * 80)
    
    avg_pos_mm = np.mean(pos_errors) * 1000
    avg_rot_deg = np.degrees(np.mean(rot_errors))
    
    if avg_pos_mm < 5 and avg_rot_deg < 5:
        print("✅ 控制精度优秀！位置和姿态误差都很小。")
    elif avg_pos_mm < 10 and avg_rot_deg < 10:
        print("⚠️  控制精度良好，但有改进空间。")
    else:
        print("❌ 控制精度较差，需要检查：")
        if avg_pos_mm >= 10:
            print(f"   - 位置平均误差 {avg_pos_mm:.1f}mm 过大")
        if avg_rot_deg >= 10:
            print(f"   - 姿态平均误差 {avg_rot_deg:.1f}° 过大")
        print("   可能原因：")
        print("   1. 执行延迟：命令发出到实际执行有时间差")
        print("   2. 运动学误差：机器人硬件精度限制")
        print("   3. 动作计算错误：检查 intervene_action_eef 的生成逻辑")
    
    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
