#!/usr/bin/env python3
"""
从日志文件分析 EEF Delta 控制精度

解析格式：
    EEF delta: pos=[...], rot=[...], gripper: X -> Y (Zmm)
    Current EE Pos: [...], Rot (quat): [...]
    
用法：
    python analyze_log.py haoyuan.md
    python analyze_log.py --input haoyuan.md --verbose
"""

import re
import sys
import numpy as np
from scipy.spatial.transform import Rotation as R


def parse_log_file(filepath):
    """解析日志文件，提取 EEF delta 和状态信息"""
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 正则表达式模式
    delta_pattern = r'EEF delta: pos=\[([-\d\.\s]+)\], rot=\[([-\d\.\s]+)\], gripper: ([\d\.]+) -> ([\d\.]+)'
    state_pattern = r'Current EE Pos: \[([-\d\.\se]+)\], Rot \(quat\): \[([-\d\.\se]+)\]'
    
    # 查找所有匹配
    deltas = re.findall(delta_pattern, content)
    states = re.findall(state_pattern, content)
    
    # 解析数据
    parsed_deltas = []
    for pos_str, rot_str, gripper_from, gripper_to in deltas:
        pos = np.array([float(x) for x in pos_str.split()])
        rot = np.array([float(x) for x in rot_str.split()])
        gripper_delta = float(gripper_to) - float(gripper_from)
        
        # delta action: [dx, dy, dz, drx, dry, drz, dgripper_normalized]
        # gripper_to 是目标值（单位 mm），需要转换为归一化增量
        gripper_to_norm = float(gripper_to) / 100.0
        gripper_from_norm = float(gripper_from) / 100.0
        dgripper_norm = gripper_to_norm - gripper_from_norm
        
        delta = np.concatenate([pos, rot, [dgripper_norm]])
        parsed_deltas.append(delta)
    
    parsed_states = []
    for pos_str, quat_str in states:
        pos = np.array([float(x) for x in pos_str.split()])
        quat = np.array([float(x) for x in quat_str.split()])
        
        # 将四元数转换为欧拉角
        rot_obj = R.from_quat(quat)
        euler = rot_obj.as_euler('xyz')
        
        # state: [x, y, z, roll, pitch, yaw, gripper_normalized]
        # 从日志中无法直接获取夹爪位置，需要从下一个 delta 的 gripper_from 推断
        # 暂时设为 0，后续修正
        state = np.concatenate([pos, euler, [0.0]])
        parsed_states.append(state)
    
    # 修正夹爪值：使用 delta 中的 gripper_from
    for i in range(len(parsed_states)):
        if i < len(deltas):
            _, _, gripper_from, _ = deltas[i]
            parsed_states[i][6] = float(gripper_from) / 100.0  # 归一化
    
    return parsed_deltas, parsed_states


def apply_eef_delta(current_state, delta_action):
    """
    应用 EEF delta 动作到当前状态
    
    Args:
        current_state: [x, y, z, roll, pitch, yaw, gripper]
        delta_action: [dx, dy, dz, droll, dpitch, dyaw, dgripper]
    
    Returns:
        predicted_state: [x, y, z, roll, pitch, yaw, gripper]
    """
    # 位置：直接相加
    predicted_pos = current_state[:3] + delta_action[:3]
    
    # 姿态：旋转合成
    current_rot = R.from_euler('xyz', current_state[3:6])
    delta_rot = R.from_euler('xyz', delta_action[3:6])
    predicted_rot = delta_rot * current_rot
    predicted_euler = predicted_rot.as_euler('xyz')
    
    # 夹爪：直接相加
    predicted_gripper = current_state[6] + delta_action[6]
    
    return np.concatenate([predicted_pos, predicted_euler, [predicted_gripper]])


def compute_rotation_error(euler1, euler2):
    """计算两个姿态之间的旋转误差"""
    rot1 = R.from_euler('xyz', euler1)
    rot2 = R.from_euler('xyz', euler2)
    rot_error = rot2 * rot1.inv()
    return rot_error.magnitude()


def main():
    # 解析命令行参数
    log_file = "haoyuan.md"
    verbose = False
    
    for i, arg in enumerate(sys.argv[1:]):
        if arg == '--input' and i + 1 < len(sys.argv) - 1:
            log_file = sys.argv[i + 2]
        elif arg == '--verbose':
            verbose = True
        elif not arg.startswith('--'):
            log_file = arg
    
    print("=" * 80)
    print("🔍 从日志分析 EEF Delta 控制精度")
    print("=" * 80)
    print(f"日志文件: {log_file}")
    
    # 解析日志
    deltas, states = parse_log_file(log_file)
    
    print(f"✓ 解析完成")
    print(f"  - 找到 {len(deltas)} 个 delta action")
    print(f"  - 找到 {len(states)} 个状态")
    
    if len(deltas) == 0 or len(states) == 0:
        print("❌ 未找到有效数据")
        return
    
    # 统计误差
    pos_errors = []
    rot_errors = []
    gripper_errors = []
    error_frames = []
    
    num_to_print = 10 if not verbose else len(states) - 1
    print(f"\n📊 逐帧分析 (前 {min(num_to_print, len(states)-1)} 帧):\n")
    
    for i in range(min(len(deltas), len(states) - 1)):
        current_state = states[i]
        delta_action = deltas[i]
        next_state = states[i + 1]
        
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
    if len(pos_errors) == 0:
        print("❌ 没有有效数据进行分析")
        return
    
    pos_errors = np.array(pos_errors)
    rot_errors = np.array(rot_errors)
    gripper_errors = np.array(gripper_errors)
    
    print("\n" + "=" * 80)
    print(f"📈 统计汇总 (全部 {len(pos_errors)} 个有效转移)")
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
    
    valid_count = len(pos_errors)
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
    num_worst = min(5, len(pos_errors))
    pos_indices = np.argsort(pos_errors)[-num_worst:][::-1]
    print("\n位置误差最大的帧:")
    for idx in pos_indices:
        frame_num = error_frames[idx]
        print(f"  帧 {frame_num} → {frame_num+1}: {pos_errors[idx]*1000:.2f} mm")
    
    # 姿态误差最大
    rot_indices = np.argsort(rot_errors)[-num_worst:][::-1]
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
        print("   3. 模型预测不准：策略输出的动作可能不够精确")
    
    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
