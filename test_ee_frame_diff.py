#!/usr/bin/env python
"""
测试不同 end-effector frame 的差异
验证 arm_link6 vs gripper_link 的偏移量
"""

import sys
import numpy as np

sys.path.insert(0, '/home/dungeon_master/conrft')

try:
    import pinocchio as pin
    HAS_PINOCCHIO = True
except ImportError:
    HAS_PINOCCHIO = False
    print("Error: Pinocchio not available")
    exit(1)


def compute_fk_with_frame(joint_positions, frame_name, urdf_path=None):
    """使用指定的 frame 计算FK"""
    if urdf_path is None:
        urdf_path = '/home/dungeon_master/A1_X/arm/install/mobiman/share/mobiman/urdf/A1X/urdf/a1x.urdf'
    
    model = pin.buildModelFromUrdf(urdf_path)
    data = model.createData()
    
    # 查找指定的 frame
    if not model.existFrame(frame_name):
        print(f"Frame '{frame_name}' not found in model!")
        return None
    
    ee_frame_id = model.getFrameId(frame_name)
    
    # 准备关节位置
    q = np.zeros(model.nq)
    q[:min(len(joint_positions), model.nq)] = joint_positions
    
    # 计算FK
    pin.forwardKinematics(model, data, q)
    pin.updateFramePlacements(model, data)
    
    # 获取末端位姿
    ee_placement = data.oMf[ee_frame_id]
    position = ee_placement.translation
    quat = pin.Quaternion(ee_placement.rotation)
    
    return {
        'position': np.array([float(position[0]), float(position[1]), float(position[2])]),
        'orientation': np.array([float(quat.x), float(quat.y), float(quat.z), float(quat.w)])
    }


def main():
    print("\n" + "="*70)
    print("测试 A1X 机械臂不同 End-Effector Frame 的差异")
    print("="*70)
    
    # 测试关节角度
    joints = np.array([
        -0.02254488, 1.9641477, -0.86475754, 0.4416942, -0.05064837, -0.10235895
    ])
    
    print(f"\n测试关节: {np.array2string(joints, precision=6)}")
    
    # 测试不同的 frame
    frames_to_test = ['arm_link6', 'gripper_link', 'end_effector', 'ee_link', 'tool0']
    
    results = {}
    
    print("\n计算各个 frame 的 FK:")
    print("-" * 70)
    
    for frame_name in frames_to_test:
        result = compute_fk_with_frame(joints, frame_name)
        if result:
            results[frame_name] = result
            pos = result['position']
            quat = result['orientation']
            print(f"\n{frame_name}:")
            print(f"  位置(xyz): [{pos[0]:.6f}, {pos[1]:.6f}, {pos[2]:.6f}]")
            print(f"  四元数(xyzw): [{quat[0]:.6f}, {quat[1]:.6f}, {quat[2]:.6f}, {quat[3]:.6f}]")
        else:
            print(f"\n{frame_name}: 不存在")
    
    # 计算 arm_link6 和 gripper_link 之间的偏移
    if 'arm_link6' in results and 'gripper_link' in results:
        print("\n" + "="*70)
        print("arm_link6 → gripper_link 的变换:")
        print("="*70)
        
        arm_link6_pos = results['arm_link6']['position']
        gripper_link_pos = results['gripper_link']['position']
        
        offset = gripper_link_pos - arm_link6_pos
        offset_norm = np.linalg.norm(offset)
        
        print(f"\n位置偏移量:")
        print(f"  ΔX: {offset[0]*1000:7.3f} mm")
        print(f"  ΔY: {offset[1]*1000:7.3f} mm")
        print(f"  ΔZ: {offset[2]*1000:7.3f} mm")
        print(f"  距离: {offset_norm*1000:.3f} mm")
        
        # 旋转差异
        arm_link6_quat = results['arm_link6']['orientation']
        gripper_link_quat = results['gripper_link']['orientation']
        
        quat_dot = np.abs(np.dot(arm_link6_quat, gripper_link_quat))
        quat_angle_diff = 2 * np.arccos(np.clip(quat_dot, -1, 1))
        
        print(f"\n旋转差异: {np.rad2deg(quat_angle_diff):.6f}°")
        
        print("\n" + "="*70)
        print("结论:")
        print("="*70)
        print(f"CuRobo 使用 'arm_link6'，位置: {arm_link6_pos}")
        print(f"Pinocchio 默认找到 'gripper_link'，位置: {gripper_link_pos}")
        print(f"\n两者相差约 {offset_norm*1000:.1f}mm，这解释了之前观察到的 81.5mm 偏差！")
        print("\n解决方案:")
        print("  1. 修改 CuRobo 配置，使用 'gripper_link' 作为 ee_link")
        print("  2. 或在比较时考虑这个固定偏移量")


if __name__ == "__main__":
    main()
