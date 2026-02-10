#!/usr/bin/env python
"""
测试 a1_x_kenimetic_haoyuan.py 的IK实现
使用实际机械臂数据验证
"""

import sys
import numpy as np

sys.path.insert(0, '/home/dungeon_master/conrft')

from a1_x_kenimetic_haoyuan import A1Kinematics

# Import Pinocchio for FK comparison
try:
    import pinocchio as pin
    HAS_PINOCCHIO = True
except ImportError:
    HAS_PINOCCHIO = False
    print("Warning: Pinocchio not available for FK comparison")


def compute_fk_pinocchio(joint_positions, urdf_path=None):
    """Compute FK using Pinocchio (same as a1x_ros2_node.py).
    
    Args:
        joint_positions: 6D joint positions
        urdf_path: Path to URDF file
    
    Returns:
        dict with 'position' and 'orientation' (quaternion xyzw)
        or None if failed
    """
    if not HAS_PINOCCHIO:
        return None
    
    if urdf_path is None:
        urdf_path = '/home/dungeon_master/A1_X/arm/install/mobiman/share/mobiman/urdf/A1X/urdf/a1x.urdf'
    
    try:
        # Build model
        model = pin.buildModelFromUrdf(urdf_path)
        data = model.createData()
        
        # Find end-effector frame
        ee_frame_id = None
        for name in ['gripper_link', 'end_effector', 'ee_link', 'tool0']:
            if model.existFrame(name):
                ee_frame_id = model.getFrameId(name)
                break
        
        if ee_frame_id is None:
            ee_frame_id = model.nframes - 1
        
        # Prepare joint positions (pad with zeros if needed)
        q = np.zeros(model.nq)
        q[:min(len(joint_positions), model.nq)] = joint_positions[:min(
            len(joint_positions), model.nq
        )]
        
        # Compute FK
        pin.forwardKinematics(model, data, q)
        pin.updateFramePlacements(model, data)
        
        # Get end-effector pose
        ee_placement = data.oMf[ee_frame_id]
        position = ee_placement.translation
        quat = pin.Quaternion(ee_placement.rotation)
        
        return {
            'position': np.array([
                float(position[0]),
                float(position[1]),
                float(position[2])
            ]),
            'orientation': np.array([
                float(quat.x),
                float(quat.y),
                float(quat.z),
                float(quat.w)
            ])  # [x, y, z, w]
        }
    except Exception as e:
        print(f"Pinocchio FK computation failed: {e}")
        return None

def test_fk_and_ik():
    """测试FK和IK循环"""
    print("\n" + "="*60)
    print("测试: A1Kinematics FK/IK验证")
    print("="*60)
    
    # 初始化
    print("\n初始化A1Kinematics...")
    kin = A1Kinematics()
    print("✓ 初始化成功")
    
    # 两个关节状态（6DOF）
    joints_1 = np.array([
        -0.02254488,  1.9641477,  -0.86475754,  0.4416942,  -0.05064837, -0.10235895
    ])
    
    joints_2 = np.array([
        -0.01531915,  1.8106383,  -1.13723404,  0.86212766, -0.05276596, -0.10234043
    ])
    
    # 目标EEF位姿（用于IK求解）
    target_pos = np.array([0.26452743, -0.0071034, 0.18859598])   
    target_quat = np.array([-0.04645897, 0.69688486, 0.00940827, 0.7156148])  # [x,y,z,w]
    
    print(f"\n关节状态1 (6DOF): {np.array2string(joints_1, precision=6)}")
    print(f"关节状态2 (6DOF): {np.array2string(joints_2, precision=6)}")
    print(f"\n目标EEF位置: {np.array2string(target_pos, precision=6)}")
    print(f"目标EEF四元数(xyzw): {np.array2string(target_quat, precision=6)}")
    
    # 步骤1: 计算joints_1的FK (CuRobo)
    print("\n--- 步骤1a: 使用CuRobo计算joints_1的FK ---")
    fk_pos_1, fk_quat_wxyz_1 = kin.forward_kinematics(joints_1)
    # CuRobo返回 (w,x,y,z)，转换为 (x,y,z,w)
    fk_quat_1 = np.array([
        fk_quat_wxyz_1[1], fk_quat_wxyz_1[2],
        fk_quat_wxyz_1[3], fk_quat_wxyz_1[0]
    ])
    
    print(f"CuRobo FK位置: {np.array2string(fk_pos_1, precision=6)}")
    print(f"CuRobo FK四元数(wxyz): "
          f"{np.array2string(fk_quat_wxyz_1, precision=6)}")
    print(f"CuRobo FK四元数(xyzw): "
          f"{np.array2string(fk_quat_1, precision=6)}")
    
    # 步骤1b: 使用Pinocchio计算joints_1的FK
    print("\n--- 步骤1b: 使用Pinocchio计算joints_1的FK ---")
    pin_fk_1 = compute_fk_pinocchio(joints_1)
    if pin_fk_1:
        pin_pos_1 = pin_fk_1['position']
        pin_quat_1 = pin_fk_1['orientation']
        print(f"Pinocchio FK位置: "
              f"{np.array2string(pin_pos_1, precision=6)}")
        print(f"Pinocchio FK四元数(xyzw): "
              f"{np.array2string(pin_quat_1, precision=6)}")
        
        # 对比两种FK结果
        pos_diff_1 = fk_pos_1 - pin_pos_1
        pos_error_1 = np.linalg.norm(pos_diff_1)
        print("\n对比 joints_1 的两种FK:")
        print(f"  位置差异: "
              f"{np.array2string(pos_diff_1*1000, precision=3)} mm")
        print(f"  位置误差: {pos_error_1*1000:.3f} mm")
        
        # 四元数差异（通过点积计算角度差）
        quat_dot = np.abs(np.dot(fk_quat_1, pin_quat_1))
        quat_angle_diff = 2 * np.arccos(np.clip(quat_dot, -1, 1))
        print(f"  旋转差异: {np.rad2deg(quat_angle_diff):.3f}°")
    else:
        print("Pinocchio FK 不可用，跳过对比")
    
    # 步骤2: 计算joints_2的FK (CuRobo)
    print("\n--- 步骤2a: 使用CuRobo计算joints_2的FK ---")
    fk_pos_2, fk_quat_wxyz_2 = kin.forward_kinematics(joints_2)
    # CuRobo返回 (w,x,y,z)，转换为 (x,y,z,w)
    fk_quat_2 = np.array([
        fk_quat_wxyz_2[1], fk_quat_wxyz_2[2],
        fk_quat_wxyz_2[3], fk_quat_wxyz_2[0]
    ])
    
    print(f"CuRobo FK位置: {np.array2string(fk_pos_2, precision=6)}")
    print(f"CuRobo FK四元数(wxyz): "
          f"{np.array2string(fk_quat_wxyz_2, precision=6)}")
    print(f"CuRobo FK四元数(xyzw): "
          f"{np.array2string(fk_quat_2, precision=6)}")
    
    # 步骤2b: 使用Pinocchio计算joints_2的FK
    print("\n--- 步骤2b: 使用Pinocchio计算joints_2的FK ---")
    pin_fk_2 = compute_fk_pinocchio(joints_2)
    if pin_fk_2:
        pin_pos_2 = pin_fk_2['position']
        pin_quat_2 = pin_fk_2['orientation']
        print(f"Pinocchio FK位置: "
              f"{np.array2string(pin_pos_2, precision=6)}")
        print(f"Pinocchio FK四元数(xyzw): "
              f"{np.array2string(pin_quat_2, precision=6)}")
        
        # 对比两种FK结果
        pos_diff_2 = fk_pos_2 - pin_pos_2
        pos_error_2 = np.linalg.norm(pos_diff_2)
        print("\n对比 joints_2 的两种FK:")
        print(f"  位置差异: "
              f"{np.array2string(pos_diff_2*1000, precision=3)} mm")
        print(f"  位置误差: {pos_error_2*1000:.3f} mm")
        
        # 四元数差异
        quat_dot = np.abs(np.dot(fk_quat_2, pin_quat_2))
        quat_angle_diff = 2 * np.arccos(np.clip(quat_dot, -1, 1))
        print(f"  旋转差异: {np.rad2deg(quat_angle_diff):.3f}°")
    else:
        print("Pinocchio FK 不可用，跳过对比")
    
    # 步骤3: 对比两个FK结果
    print("\n--- 步骤3: 对比两个FK结果 ---")
    pos_diff = fk_pos_2 - fk_pos_1
    pos_dist = np.linalg.norm(pos_diff)
    print(f"位置差异: {np.array2string(pos_diff, precision=6)}")
    print(f"位置距离: {pos_dist*1000:.2f} mm")
    
    # 步骤4: 从joints_2开始，对target_pos进行IK求解
    print("\n--- 步骤4: 从joints_2出发，求解目标EEF位姿的IK ---")
    print(f"起始关节(seed): {np.array2string(joints_2, precision=6)}")
    print(f"目标EEF位置: {np.array2string(target_pos, precision=6)}")
    print(f"目标EEF四元数(xyzw): {np.array2string(target_quat, precision=6)}")
    
    # 设置prev_q作为seed
    import torch
    kin.prev_q = torch.tensor(
        joints_2,
        device=kin.tensor_args.device,
        dtype=torch.float32
    ).unsqueeze(0)
    
    # 调用IK
    result = kin.solve_ik(target_pos, target_quat)
    
    if result.success.any():
        ik_joints = result.js_solution.position.cpu().numpy()
        print("\n✓ IK求解成功")
        print(f"IK求解结果: {np.array2string(ik_joints, precision=6)}")
        print(f"起始关节(seed): {np.array2string(joints_2, precision=6)}")
        
        # 计算关节差异
        joint_diff = ik_joints - joints_2
        print("\n关节角度变化:")
        print(f"  差值(rad): {np.array2string(joint_diff, precision=6)}")
        print(f"  差值(度): "
              f"{np.array2string(np.rad2deg(joint_diff), precision=3)}")
        print(f"  最大变化: {np.rad2deg(np.max(np.abs(joint_diff))):.3f}°")
        
        # 用FK验证IK结果
        print("\n--- 步骤5: FK验证IK结果 ---")
        verify_pos, verify_quat_wxyz = kin.forward_kinematics(ik_joints)
        verify_quat = np.array([
            verify_quat_wxyz[1], verify_quat_wxyz[2],
            verify_quat_wxyz[3], verify_quat_wxyz[0]
        ])
        
        pos_error = np.linalg.norm(verify_pos - target_pos)
        print(f"IK→FK位置: {np.array2string(verify_pos, precision=6)}")
        print(f"目标位置:   {np.array2string(target_pos, precision=6)}")
        print(f"位置误差: {pos_error*1000:.2f} mm")
        
        if pos_error < 0.005:  # 5mm
            print("✓ FK验证通过: 位置误差 < 5mm")
        else:
            print(f"⚠️  位置误差较大: {pos_error*1000:.2f}mm")
    else:
        print("\n✗ IK求解失败")
        print(f"成功标志: {result.success.cpu().numpy()}")
        if hasattr(result, 'position_error'):
            pos_err = result.position_error.cpu().numpy()
            print(f"位置误差: {pos_err}")

def test_multiple_ik_calls():
    """测试连续多次IK调用（模拟tracking）"""
    print("\n" + "="*60)
    print("测试: 连续IK调用（Tracking模式）")
    print("="*60)
    
    kin = A1Kinematics()
    
    # 起始关节
    start_joints = np.array([-0.062553, 1.728085, -0.706170, 0.284681, -0.134468, -0.044681])
    kin.prev_q = None  # 第一次调用无seed
    
    # 第1次：使用起始关节的FK位姿
    print("\n第1次IK调用（无seed）:")
    pos1, quat1_wxyz = kin.forward_kinematics(start_joints)
    quat1 = np.array([quat1_wxyz[1], quat1_wxyz[2], quat1_wxyz[3], quat1_wxyz[0]])
    
    result1 = kin.solve_ik(pos1, quat1)
    if result1.success.any():
        joints1 = result1.js_solution.position.cpu().numpy()
        print(f"✓ 成功, 关节: {np.array2string(joints1, precision=3)}")
    else:
        print(f"✗ 失败")
        return
    
    # 第2次：稍微偏移位置（模拟delta控制）
    print("\n第2次IK调用（有seed，delta=+5cm x方向）:")
    pos2 = pos1 + np.array([0.05, 0.0, 0.0])  # x方向+5cm
    quat2 = quat1
    
    result2 = kin.solve_ik(pos2, quat2)
    if result2.success.any():
        joints2 = result2.js_solution.position.cpu().numpy()
        delta_joints = joints2 - joints1
        print(f"✓ 成功, 关节: {np.array2string(joints2, precision=3)}")
        print(f"关节变化: {np.array2string(delta_joints, precision=3)}")
        print(f"最大变化: {np.rad2deg(np.max(np.abs(delta_joints))):.1f}°")
    else:
        print(f"✗ 失败")
        return
    
    # 第3次：再次偏移
    print("\n第3次IK调用（有seed，delta=-3cm y方向）:")
    pos3 = pos2 + np.array([0.0, -0.03, 0.0])  # y方向-3cm
    quat3 = quat2
    
    result3 = kin.solve_ik(pos3, quat3)
    if result3.success.any():
        joints3 = result3.js_solution.position.cpu().numpy()
        delta_joints = joints3 - joints2
        print(f"✓ 成功, 关节: {np.array2string(joints3, precision=3)}")
        print(f"关节变化: {np.array2string(delta_joints, precision=3)}")
        print(f"最大变化: {np.rad2deg(np.max(np.abs(delta_joints))):.1f}°")
    else:
        print(f"✗ 失败")

def test_ik_timing():
    """测试连续100次IK求解的时间分布"""
    print("\n" + "="*60)
    print("测试: 连续100次IK求解时间分析")
    print("="*60)
    
    kin = A1Kinematics()
    
    # 使用实际关节作为起点
    actual_joints = np.array([-0.062553, 1.728085, -0.706170, 0.284681, -0.134468, -0.044681])
    
    # 获取FK位姿
    pos, quat_wxyz = kin.forward_kinematics(actual_joints)
    quat = np.array([quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]])
    
    # 设置seed
    import torch
    import time
    kin.prev_q = torch.tensor(actual_joints, device=kin.tensor_args.device, dtype=torch.float32)
    
    # 连续求解100次，记录时间
    times = []
    print("\n开始连续100次IK求解...")
    
    for i in range(100):
        # 稍微扰动目标位置（模拟真实场景）
        pos_perturbed = pos + np.random.uniform(-0.002, 0.002, 3)  # ±2mm随机扰动
        
        start = time.time()
        result = kin.solve_ik(pos_perturbed, quat)
        elapsed = (time.time() - start) * 1000  # ms
        times.append(elapsed)
        
        if not result.success.any():
            print(f"  第{i+1}次求解失败")
    
    times = np.array(times)
    
    # 统计分析
    print(f"\n✓ 完成100次IK求解")
    print(f"\n时间统计:")
    print(f"  第1次:  {times[0]:.2f} ms")
    print(f"  第2次:  {times[1]:.2f} ms")
    print(f"  第3次:  {times[2]:.2f} ms")
    print(f"  平均值: {np.mean(times):.2f} ms")
    print(f"  中位数: {np.median(times):.2f} ms")
    print(f"  最小值: {np.min(times):.2f} ms")
    print(f"  最大值: {np.max(times):.2f} ms")
    print(f"  标准差: {np.std(times):.2f} ms")
    
    # 前10次 vs 后90次对比
    first_10 = times[:10]
    last_90 = times[10:]
    print(f"\n前10次 vs 后90次对比:")
    print(f"  前10次平均: {np.mean(first_10):.2f} ms")
    print(f"  后90次平均: {np.mean(last_90):.2f} ms")
    print(f"  差异: {np.mean(first_10) - np.mean(last_90):.2f} ms")
    
    # 简单的ASCII图表显示前20次
    print(f"\n前20次求解时间趋势:")
    max_time = np.max(times[:20])
    for i in range(20):
        bar_len = int((times[i] / max_time) * 50)
        bar = '█' * bar_len
        print(f"  #{i+1:2d}: {bar} {times[i]:.2f}ms")
    
    return times

def main():
    print("\n" + "#"*60)
    print("#  测试 A1Kinematics (haoyuan版本)")
    print("#  验证FK/IK准确性和tracking能力")
    print("#"*60)
    
    # 测试1: FK/IK循环验证
    test_fk_and_ik()
    
    # 测试2: 连续IK调用
    test_multiple_ik_calls()
    
    # # 测试3: 100次IK求解时间分析
    # times = test_ik_timing()
    
    print("\n" + "#"*60)
    print("#  测试完成")
    print("#"*60 + "\n")

if __name__ == "__main__":
    main()
