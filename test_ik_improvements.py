#!/usr/bin/env python
"""
测试改进后的IK solver
验证：
1. 自动EE link检测
2. FK验证功能
3. 日志输出

无需ROS运行环境，可独立测试
"""

import sys
import numpy as np

# 添加当前目录到路径
sys.path.insert(0, '/home/dungeon_master/conrft')

from A1_x_controller import URDFInverseKinematics

def test_ik_initialization():
    """测试IK初始化和自动EE link检测"""
    print("\n" + "="*60)
    print("测试1: IK初始化和自动EE link检测")
    print("="*60)
    
    try:
        # 不指定ee_link，让系统自动检测
        ik_solver = URDFInverseKinematics()
        print(f"✓ IK solver初始化成功")
        print(f"  Base link: {ik_solver.base_link}")
        print(f"  EE link: {ik_solver.ee_link}")
        print(f"  FK verifier available: {ik_solver.fk_model is not None}")
        return ik_solver
    except Exception as e:
        print(f"✗ 初始化失败: {e}")
        return None

def test_ik_solve(ik_solver):
    """测试IK求解功能"""
    print("\n" + "="*60)
    print("测试2: IK求解（有/无current joints seed）")
    print("="*60)
    
    # 测试目标位置和姿态
    target_pos = [0.4, 0.0, 0.3]  # meters
    target_quat = [0.001, 0.664, -0.001, 0.748]  # [x, y, z, w]
    
    print(f"\n目标位置: {target_pos}")
    print(f"目标姿态 (quat): {target_quat}")
    
    # 测试1: 无seed（随机初始化）
    print("\n--- 测试2a: 无current joints (随机初始化) ---")
    result1 = ik_solver.solve_ik(target_pos, target_quat)
    if result1 is not None:
        joints1 = result1.solution.cpu().numpy()[0]
        print(f"✓ IK求解成功")
        print(f"  关节角度: {np.array2string(joints1, precision=3, suppress_small=True)}")
    else:
        print(f"✗ IK求解失败")
        return
    
    # 测试2: 使用上次的解作为seed
    print("\n--- 测试2b: 使用current joints作为seed ---")
    # 稍微扰动目标位置
    target_pos_perturbed = [0.2941, 0.0136, 0.2032]
    result2 = ik_solver.solve_ik(
        target_pos_perturbed, 
        target_quat, 
        current_joints=joints1  # 使用上次的解
    )
    if result2 is not None:
        joints2 = result2.solution.cpu().numpy()[0]
        print(f"✓ IK求解成功")
        print(f"  关节角度: {np.array2string(joints2, precision=3, suppress_small=True)}")
        
        # 计算关节角度变化
        joint_delta = joints2 - joints1
        print(f"  关节变化: {np.array2string(joint_delta, precision=3, suppress_small=True)}")
        print(f"  最大变化: {np.max(np.abs(joint_delta)):.3f} rad ({np.rad2deg(np.max(np.abs(joint_delta))):.1f}°)")
    else:
        print(f"✗ IK求解失败")

def test_reachable_pose(ik_solver):
    """测试3: 从FK计算可达位姿，确保IK有解"""
    print("\n" + "="*60)
    print("测试3: 从FK计算可达目标（确保在工作空间内）")
    print("="*60)
    
    if ik_solver.fk_model is None:
        print("⚠️  FK verifier不可用，跳过此测试")
        return
    
    # 使用一个已知的关节角度计算目标位姿
    import pinocchio as pin
    
    test_joints = np.array([0.0, -0.5, 0.0, -1.5, 0.0, 1.0])  # 6个关节
    q = np.zeros(ik_solver.fk_model.nq)
    q[:len(test_joints)] = test_joints
    
    # 计算FK得到目标位姿
    pin.forwardKinematics(ik_solver.fk_model, ik_solver.fk_data, q)
    pin.updateFramePlacements(ik_solver.fk_model, ik_solver.fk_data)
    
    ee_placement = ik_solver.fk_data.oMf[ik_solver.fk_ee_frame_id]
    target_pos = ee_placement.translation
    target_quat_pin = pin.Quaternion(ee_placement.rotation)
    target_quat = np.array([target_quat_pin.x, target_quat_pin.y, target_quat_pin.z, target_quat_pin.w])
    
    print(f"参考关节角度: {np.array2string(test_joints, precision=3)}")
    print(f"FK计算目标位置: {np.array2string(target_pos, precision=4)}")
    print(f"FK计算目标姿态: {np.array2string(target_quat, precision=4)}")
    
    # 用IK求解这个目标（应该能成功）
    print("\n使用IK求解此目标...")
    result = ik_solver.solve_ik(
        target_pos.tolist(), 
        target_quat.tolist(),
        current_joints=test_joints  # 用FK的关节角度作为seed
    )
    
    if result is not None:
        ik_joints = result.solution.cpu().numpy()[0]
        print(f"\n✓ IK求解成功")
        print(f"  IK关节角度: {np.array2string(ik_joints, precision=3)}")
        print(f"  参考关节角度: {np.array2string(test_joints, precision=3)}")
        print(f"  差异: {np.array2string(ik_joints - test_joints, precision=3)}")
    else:
        print(f"\n✗ IK求解失败（理论上应该成功，因为目标是FK计算的）")

def test_actual_robot_data(ik_solver):
    """测试4: 使用实际机械臂读取的数据"""
    print("\n" + "="*60)
    print("测试4: 实际机械臂数据验证")
    print("="*60)
    
    # 实际读取的关节角度（7维，最后一维是夹爪）
    actual_joints_full = np.array([
        -0.062553, 1.728085, -0.706170, 0.284681, 
        -0.134468, -0.044681, -2.739787
    ])
    actual_joints = actual_joints_full[:6]  # 前6个关节用于IK
    gripper = actual_joints_full[6]
    
    # 实际读取的末端位姿 (gripper_link)
    gripper_pos = np.array([0.249017, -0.029588, 0.145086])
    gripper_quat = np.array([-0.039498, 0.609009, -0.064511, 0.789548])  # [x,y,z,w]
    
    print(f"实际关节角度(6DOF): {np.array2string(actual_joints, precision=6)}")
    print(f"夹爪角度: {gripper:.6f}")
    print(f"实际gripper_link位置: {np.array2string(gripper_pos, precision=6)}")
    print(f"实际gripper_link四元数: {np.array2string(gripper_quat, precision=6)}")
    
    # 步骤1: 用FK验证关节角度产生的arm_link6位姿
    print("\n--- 步骤1: FK计算arm_link6位姿（关节→末端） ---")
    if ik_solver.fk_model is not None:
        import pinocchio as pin
        
        # 准备关节位置
        q = np.zeros(ik_solver.fk_model.nq)
        q[:6] = actual_joints
        
        # 计算FK
        pin.forwardKinematics(ik_solver.fk_model, ik_solver.fk_data, q)
        pin.updateFramePlacements(ik_solver.fk_model, ik_solver.fk_data)
        
        # 获取arm_link6的位姿
        ee_placement = ik_solver.fk_data.oMf[ik_solver.fk_ee_frame_id]
        arm_link6_pos = ee_placement.translation
        arm_link6_quat_pin = pin.Quaternion(ee_placement.rotation)
        arm_link6_quat = np.array([arm_link6_quat_pin.x, arm_link6_quat_pin.y, 
                                   arm_link6_quat_pin.z, arm_link6_quat_pin.w])
        
        print(f"FK计算arm_link6位置: {np.array2string(arm_link6_pos, precision=6)}")
        print(f"FK计算arm_link6四元数: {np.array2string(arm_link6_quat, precision=6)}")
        
        # 验证gripper_link位姿（gripper_link = arm_link6 + offset [0.08165, 0, 0]）
        gripper_frame_id = ik_solver.fk_model.getFrameId("gripper_link")
        gripper_placement = ik_solver.fk_data.oMf[gripper_frame_id]
        fk_gripper_pos = gripper_placement.translation
        
        gripper_pos_error = np.linalg.norm(fk_gripper_pos - gripper_pos)
        print(f"\nFK计算gripper_link位置: {np.array2string(fk_gripper_pos, precision=6)}")
        print(f"实际gripper_link位置:   {np.array2string(gripper_pos, precision=6)}")
        print(f"gripper位置误差: {gripper_pos_error*1000:.2f} mm")
        
        if gripper_pos_error < 0.001:  # 1mm
            print(f"✓ FK验证通过: 关节角度与末端位姿一致")
        else:
            print(f"⚠️  FK验证: gripper位置误差 {gripper_pos_error*1000:.2f}mm > 1mm")
    else:
        print("⚠️  FK verifier不可用")
        return
    
    # 步骤2: 用IK反求关节角度（使用arm_link6位姿）
    print("\n--- 步骤2: IK反求（arm_link6位姿→关节） ---")
    print(f"目标: 使用FK计算的arm_link6位姿反求关节角度")
    print(f"目标位置: {np.array2string(arm_link6_pos, precision=6)}")
    print(f"目标四元数: {np.array2string(arm_link6_quat, precision=6)}")
    
    result = ik_solver.solve_ik(
        arm_link6_pos.tolist(),
        arm_link6_quat.tolist(),
        current_joints=actual_joints  # 用实际关节作为seed
    )
    
    if result is not None:
        ik_joints = result.solution.cpu().numpy()[0]
        print(f"\n✓ IK求解成功")
        print(f"IK计算关节: {np.array2string(ik_joints, precision=6)}")
        print(f"实际关节角度: {np.array2string(actual_joints, precision=6)}")
        
        # 计算关节差异
        joint_diff = ik_joints - actual_joints
        print(f"\n关节角度差异:")
        print(f"  差值: {np.array2string(joint_diff, precision=6)}")
        print(f"  差值(度): {np.array2string(np.rad2deg(joint_diff), precision=3)}")
        print(f"  最大差异: {np.rad2deg(np.max(np.abs(joint_diff))):.3f}°")
        
        # 用Pinocchio FK验证IK结果
        q_ik = np.zeros(ik_solver.fk_model.nq)
        q_ik[:6] = ik_joints
        pin.forwardKinematics(ik_solver.fk_model, ik_solver.fk_data, q_ik)
        pin.updateFramePlacements(ik_solver.fk_model, ik_solver.fk_data)
        
        ik_ee_placement = ik_solver.fk_data.oMf[ik_solver.fk_ee_frame_id]
        ik_pos_verify = ik_ee_placement.translation
        ik_quat_verify_pin = pin.Quaternion(ik_ee_placement.rotation)
        ik_quat_verify = np.array([ik_quat_verify_pin.x, ik_quat_verify_pin.y,
                                   ik_quat_verify_pin.z, ik_quat_verify_pin.w])
        
        print(f"\n用Pinocchio验证IK结果:")
        print(f"  IK→FK位置: {np.array2string(ik_pos_verify, precision=6)}")
        print(f"  目标位置:   {np.array2string(arm_link6_pos, precision=6)}")
        print(f"  位置误差: {np.linalg.norm(ik_pos_verify - arm_link6_pos)*1000:.2f}mm")
        
        if np.max(np.abs(joint_diff)) < np.deg2rad(5):  # 5度
            print(f"✓ IK反求成功: 关节角度差异 < 5°")
        else:
            print(f"⚠️  关节角度差异较大 (>{np.rad2deg(np.max(np.abs(joint_diff))):.1f}°)")
    else:
        print(f"\n✗ IK求解失败")
        print(f"可能原因: 目标姿态的旋转部分难以达到，或存在关节限位约束")

def test_fk_verification(ik_solver):
    """测试5: FK验证功能"""
    print("\n" + "="*60)
    print("测试5: FK验证IK解的准确性")
    print("="*60)
    
    if ik_solver.fk_model is None:
        print("⚠️  FK verifier不可用，跳过此测试")
        return
    
    # 求解IK
    target_pos = [0.2841, 0.0036, 0.1932]
    target_quat = [0.001, 0.664, -0.001, 0.748]  # 90度绕Y轴
    
    print(f"目标位置: {target_pos}")
    print(f"目标姿态 (quat): {target_quat}")
    
    result = ik_solver.solve_ik(target_pos, target_quat)
    if result is None:
        print("✗ IK求解失败")
        return
    
    joints = result.solution.cpu().numpy()[0]
    print(f"\n✓ IK求解成功")
    print(f"  关节角度: {np.array2string(joints, precision=3, suppress_small=True)}")
    
    # 使用FK验证
    fk_result = ik_solver.verify_ik_solution(joints, target_pos, target_quat)
    if fk_result:
        print(f"\nFK验证结果:")
        print(f"  位置误差: {fk_result['position_error']*1000:.2f} mm")
        print(f"  姿态误差: {fk_result['orientation_error']:.4f}")
        print(f"  FK计算位置: {np.array2string(fk_result['fk_position'], precision=4)}")
        print(f"  FK计算四元数: {np.array2string(fk_result['fk_quaternion'], precision=4)}")
        
        if fk_result['position_error'] < 0.01:  # 10mm
            print(f"  ✓ 位置验证通过 (<10mm)")
        else:
            print(f"  ⚠️  位置误差较大 (>10mm)")
    else:
        print("✗ FK验证失败")

def main():
    print("\n" + "#"*60)
    print("#  A1_X IK Solver 改进验证测试")
    print("#  独立测试模式（无需ROS运行环境）")
    print("#"*60)
    
    # 测试1: 初始化
    ik_solver = test_ik_initialization()
    if not ik_solver:
        print("\n✗ 初始化失败，终止测试")
        return
    
    # 测试2: IK求解
    test_ik_solve(ik_solver)
    
    # 测试3: 从FK计算可达目标
    test_reachable_pose(ik_solver)
    
    # 测试4: 实际机械臂数据验证
    test_actual_robot_data(ik_solver)
    
    # 测试5: FK验证
    test_fk_verification(ik_solver)
    
    print("\n" + "#"*60)
    print("#  测试完成")
    print("#"*60 + "\n")

if __name__ == "__main__":
    main()
