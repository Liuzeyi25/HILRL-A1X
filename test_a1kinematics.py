#!/usr/bin/env python3
"""测试 A1Kinematics 是否正常工作"""

import numpy as np
from a1_x_kenimetic_haoyuan import A1Kinematics

def main():
    print("=" * 60)
    print("测试 A1Kinematics 初始化")
    print("=" * 60)
    
    try:
        ik_solver = A1Kinematics(
            urdf_file="/home/dungeon_master/A1_X/arm/install/mobiman/share/mobiman/urdf/A1X/urdf/a1x.urdf",
            base_link="base_link",
            ee_link="arm_link6"
        )
        print("✅ A1Kinematics 初始化成功")
    except Exception as e:
        print(f"❌ A1Kinematics 初始化失败: {e}")
        return
    
    print("\n" + "=" * 60)
    print("测试 IK 求解")
    print("=" * 60)
    
    # 测试位置和四元数 (注意：A1Kinematics 接受 [x,y,z,w] 格式)
    target_pos = np.array([0.3, 0.0, 0.2])
    target_quat = np.array([0.0, 1.0, 0.0, 0.0])  # [x, y, z, w]
    
    print(f"目标位置: {target_pos}")
    print(f"目标四元数: {target_quat}")
    
    try:
        result = ik_solver.solve_ik(pos=target_pos, quat=target_quat)
        
        if result and result.success.cpu().numpy().any():
            solution = result.js_solution.position.cpu().numpy()[:6]
            print(f"✅ IK 求解成功")
            print(f"关节解: {solution}")
            
            # 测试 FK 验证
            print("\n验证 FK:")
            fk_pos, fk_quat = ik_solver.forward_kinematics(solution)
            print(f"FK 位置: {fk_pos}")
            print(f"FK 四元数 (w,x,y,z): {fk_quat}")
            
            pos_error = np.linalg.norm(fk_pos - target_pos)
            print(f"位置误差: {pos_error * 1000:.2f} mm")
        else:
            print("❌ IK 求解失败")
    except Exception as e:
        print(f"❌ IK 求解出错: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 60)
    print("测试完成")
    print("=" * 60)

if __name__ == "__main__":
    main()
