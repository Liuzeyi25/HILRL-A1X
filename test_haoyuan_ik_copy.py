#!/usr/bin/env python
"""
测试 A1X 实机末端执行器控制
使用 CuRobo IK 求解 + 直接关节命令
"""

import sys
import time
import zmq
import numpy as np
from scipy.spatial.transform import Rotation as R

sys.path.insert(0, '/home/dungeon_master/conrft')
from a1_x_kenimetic_haoyuan import A1Kinematics


class A1XRealRobotClient:
    """A1X 实机客户端 - 通过 ZMQ 与 ROS2 节点通信"""
    
    def __init__(self, command_port=6100, state_port=6101, host="127.0.0.1"):
        self.ctx = zmq.Context()
        
        # Command socket - 发送控制命令
        self.command_socket = self.ctx.socket(zmq.REQ)
        self.command_socket.setsockopt(zmq.RCVTIMEO, 5000)  # 5s timeout
        self.command_socket.setsockopt(zmq.SNDTIMEO, 5000)
        self.command_socket.connect(f"tcp://{host}:{command_port}")
        
        # State socket - 查询状态
        self.state_socket = self.ctx.socket(zmq.REQ)
        self.state_socket.setsockopt(zmq.RCVTIMEO, 5000)
        self.state_socket.setsockopt(zmq.SNDTIMEO, 5000)
        self.state_socket.connect(f"tcp://{host}:{state_port}")
        
        # 初始化 IK 求解器
        print("初始化 CuRobo IK 求解器...")
        self.ik_solver = A1Kinematics()
        print("✓ IK 求解器初始化完成")
        
        print(f"✓ 已连接到 A1X ROS2 节点: {host}:{command_port}/{state_port}")
    
    def get_state(self):
        """获取当前机器人状态"""
        try:
            self.state_socket.send_json({"cmd": "get_state"})
            response = self.state_socket.recv_json()
            return response
        except zmq.Again:
            print("❌ 获取状态超时")
            return None
        except Exception as e:
            print(f"❌ 获取状态失败: {e}")
            return None
    
    def command_eef_pose(
        self, delta_pose, wait_for_completion=True, timeout=2.0
    ):
        """发送末端执行器位姿命令 - 使用 IK 求解
        
        Args:
            delta_pose: [dx, dy, dz, drx, dry, drz, gripper] (7D)
                - dx, dy, dz: 位置增量 (m)
                - drx, dry, drz: 旋转增量 (rad, euler angles)
                - gripper: 夹爪位置 (0-100mm)
            wait_for_completion: 是否等待执行到位
            timeout: 超时时间 (s)
        
        Returns:
            dict with status, reached, final_error, target_joints
        """
        try:
            # 1. 获取当前状态
            state = self.get_state()
            if not state:
                return {"status": "error", "error": "cannot get current state"}
            
            current_joints = np.array(state['positions'][:6])
            current_pos = np.array(state['ee_pos'])
            current_quat = np.array(state['ee_quat'])  # [x, y, z, w]
            
            # 2. 计算目标末端位姿
            delta_pos = np.array(delta_pose[:3])
            delta_rot_euler = np.array(delta_pose[3:6])
            gripper_position = delta_pose[6]
            
            # 目标位置
            target_pos = current_pos + delta_pos
            
            # 目标姿态（旋转组合）
            current_rotation = R.from_quat(current_quat)  # [x,y,z,w]
            delta_rotation = R.from_euler('xyz', delta_rot_euler)
            target_rotation = delta_rotation * current_rotation
            target_quat = target_rotation.as_quat()  # [x,y,z,w]
            
            print(f"\n当前末端: {np.array2string(current_pos, precision=4)}")
            print(f"目标末端: {np.array2string(target_pos, precision=4)}")
            print(f"位置增量: {np.array2string(delta_pos*100, precision=2)} cm")
            
            # 3. 使用 IK 求解目标关节
            print("求解 IK...")
            ik_result = self.ik_solver.solve_ik(
                target_pos,
                target_quat,
                current_joints=current_joints,
                max_joint_delta=0.2
            )
            
            if not ik_result.success.any():
                return {
                    "status": "error",
                    "error": "IK solve failed"
                }
            
            # 获取关节解（确保是1D数组，6个关节）
            target_joints = ik_result.js_solution.position.cpu().numpy()
            
            # 如果是2D数组，取第一个解
            if target_joints.ndim > 1:
                target_joints = target_joints.flatten()
            
            # 确保只有6个关节
            target_joints = target_joints[:6]
            
            print("✓ IK 求解成功")
            print(f"  目标关节 shape: {target_joints.shape}")
            print(f"  目标关节: {np.array2string(target_joints, precision=3)}")
            
            # 4. 发送关节命令（添加夹爪）
            target_with_gripper = np.append(target_joints, gripper_position)
            
            print("发送关节命令...")
            print(f"  完整命令 (6关节+夹爪): "
                  f"{np.array2string(target_with_gripper, precision=3)}")
            print(f"  命令数据类型: {type(target_with_gripper)}, "
                  f"shape: {target_with_gripper.shape}")
            
            self.command_socket.send_json({
                "cmd": "command_joint_state",
                "positions": target_with_gripper.tolist()
            })
            response = self.command_socket.recv_json()
            
            if response.get('status') != 'ok':
                return {
                    "status": "error",
                    "error": f"command failed: {response}"
                }
            
            # 5. 等待到达（如果需要）
            reached = False
            final_error = float('inf')
            
            if wait_for_completion:
                print("等待关节到达目标...")
                start_time = time.time()
                joint_tolerance = 0.01  # 0.01 rad
                
                while time.time() - start_time < timeout:
                    current_state = self.get_state()
                    if not current_state:
                        time.sleep(0.1)
                        continue
                    
                    current_joints_check = np.array(
                        current_state['positions'][:6]
                    )
                    joint_error = np.abs(
                        current_joints_check - target_joints
                    ).max()
                    final_error = joint_error
                    
                    if joint_error < joint_tolerance:
                        reached = True
                        print(f"✓ 到达目标 (误差: {joint_error:.4f} rad)")
                        break
                    
                    time.sleep(0.01)
                
                if not reached:
                    print(f"⚠️  超时未到达 (最终误差: {final_error:.4f} rad)")
            
            return {
                "status": "ok",
                "reached": reached,
                "final_error": float(final_error),
                "target_joints": target_joints.tolist(),
                "gripper": float(gripper_position)
            }
            
        except Exception as e:
            print(f"❌ 错误: {e}")
            import traceback
            traceback.print_exc()
            return {"status": "error", "error": str(e)}
    
    def close(self):
        """关闭连接"""
        self.command_socket.close()
        self.state_socket.close()
        self.ctx.term()


def test_real_robot_state():
    """测试1: 获取实机当前状态"""
    print("\n" + "="*60)
    print("测试1: 获取实机当前状态")
    print("="*60)
    
    client = A1XRealRobotClient()
    
    print("\n查询机器人状态...")
    state = client.get_state()
    
    if state:
        print("✓ 成功获取状态")
        print(f"关节位置 (6+1): {np.array2string(np.array(state['positions']), precision=4)}")  # noqa: E501
        print(f"关节速度: {np.array2string(np.array(state['velocities']), precision=4)}")  # noqa: E501
        print(f"末端位置 (xyz): {np.array2string(np.array(state['ee_pos']), precision=4)}")  # noqa: E501
        print(f"末端姿态 (xyzw): {np.array2string(np.array(state['ee_quat']), precision=4)}")  # noqa: E501
    else:
        print("✗ 获取状态失败")
    
    client.close()
    return state


def test_small_delta_movement():
    """测试2: 小幅度delta运动 (+2cm X方向)"""
    print("\n" + "="*60)
    print("测试2: 小幅度delta运动 (X方向 +2cm)")
    print("="*60)
    
    client = A1XRealRobotClient()
    
    # 获取初始状态
    print("\n获取初始状态...")
    state = client.get_state()
    if not state:
        print("✗ 无法获取状态")
        client.close()
        return
    
    initial_pos = np.array(state['ee_pos'])
    print(f"初始末端位置: {np.array2string(initial_pos, precision=4)}")
    
    # 构造delta命令: [dx, dy, dz, drx, dry, drz, gripper]
    delta_pose = np.array([
        0.02,   # +2cm X方向
        0.0,    # Y不变
        0.0,    # Z不变
        0.0,    # 旋转不变
        0.0,
        0.0,
        50.0    # 夹爪位置保持50mm
    ])
    
    print(f"\n发送delta命令: X+{delta_pose[0]*100:.1f}cm")
    print("等待执行到位（超时: 10秒）...")
    
    result = client.command_eef_pose(
        delta_pose,
        wait_for_completion=True,
        timeout=10.0  # 增加超时到10秒
    )
    
    if result.get('status') == 'ok':
        reached = result.get('reached', False)
        final_error_rad = result.get('final_error', float('inf'))
        
        if reached:
            print("✓ 命令执行成功，已到达目标")
        else:
            print("⚠️  命令执行完成，但未完全到达目标")
        
        print(f"  到达状态: {reached}")
        print(f"  关节误差: {final_error_rad:.4f} rad ({np.rad2deg(final_error_rad):.2f}°)")  # noqa: E501
        
        if 'target_joints' in result:
            target_joints = np.array(result['target_joints'])
            print(f"  目标关节: {np.array2string(target_joints, precision=3)}")
        
        # 查询最终状态
        print("\n查询最终状态...")
        time.sleep(0.3)
        final_state = client.get_state()
        if final_state:
            final_pos = np.array(final_state['ee_pos'])
            actual_delta = final_pos - initial_pos
            print(f"  初始位置: {np.array2string(initial_pos, precision=4)} m")
            print(f"  最终位置: {np.array2string(final_pos, precision=4)} m")
            print(f"  实际移动: {np.array2string(actual_delta*100, precision=2)} cm")  # noqa: E501
            print(f"  位置误差: {np.linalg.norm(actual_delta - delta_pose[:3])*1000:.1f} mm")  # noqa: E501
    else:
        print(f"✗ 命令失败: {result.get('error', 'unknown')}")
    
    client.close()


def test_circular_motion():
    """测试3: 圆形轨迹运动 (XY平面, 半径3cm, 8个点)"""
    print("\n" + "="*60)
    print("测试3: 圆形轨迹运动 (XY平面)")
    print("="*60)
    
    client = A1XRealRobotClient()
    
    # 圆形轨迹参数
    radius = 0.03  # 3cm半径
    num_points = 8
    angles = np.linspace(0, 2*np.pi, num_points, endpoint=False)
    
    print(f"\n生成圆形轨迹: 半径={radius*100:.1f}cm, {num_points}个点")
    
    success_count = 0
    errors = []
    
    for i, angle in enumerate(angles):
        # 计算圆形轨迹上的delta
        dx = radius * np.cos(angle) / num_points
        dy = radius * np.sin(angle) / num_points
        
        delta_pose = np.array([dx, dy, 0.0, 0.0, 0.0, 0.0, 50.0])
        
        print(f"\n步骤 {i+1}/{num_points}: dx={dx*100:.2f}cm, dy={dy*100:.2f}cm")
        
        result = client.command_eef_pose(
            delta_pose,
            wait_for_completion=True,
            timeout=8.0  # 增加超时到8秒
        )
        
        if result.get('status') == 'ok' and result.get('reached'):
            success_count += 1
            error_mm = result.get('final_error', 0) * 1000
            errors.append(error_mm)
            print(f"  ✓ 成功 (误差: {error_mm:.1f}mm)")
        else:
            print("  ✗ 失败或超时")
            errors.append(float('inf'))
        
        time.sleep(0.2)  # 短暂停顿
    
    print("\n轨迹执行完成:")
    print(f"  成功率: {success_count}/{num_points}")
    valid_errors = [e for e in errors if e != float('inf')]
    if valid_errors:
        print(f"  平均误差: {np.mean(valid_errors):.1f}mm")
        print(f"  最大误差: {np.max(valid_errors):.1f}mm")
    
    client.close()


def test_gripper_control():
    """测试4: 夹爪控制 (打开→关闭→打开)"""
    print("\n" + "="*60)
    print("测试4: 夹爪控制")
    print("="*60)
    
    client = A1XRealRobotClient()
    
    gripper_positions = [
        (100.0, "完全打开"),
        (0.0, "完全关闭"),
        (50.0, "中间位置")
    ]
    
    for gripper_pos, description in gripper_positions:
        print(f"\n设置夹爪: {description} ({gripper_pos}mm)")
        
        # 夹爪命令: 末端位置不变，只改变夹爪
        delta_pose = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, gripper_pos])
        
        result = client.command_eef_pose(
            delta_pose,
            wait_for_completion=True,
            timeout=2.0
        )
        
        if result.get('status') == 'ok':
            print("  ✓ 命令发送成功")
        else:
            print("  ✗ 命令失败")
        
        time.sleep(1.0)  # 等待夹爪运动
    
    client.close()

def main():
    print("\n" + "#"*60)
    print("#  A1X 实机末端控制测试")
    print("#  通过 ZMQ 连接到 ROS2 节点")
    print("#"*60)
    
    # 交互式选择测试
    print("\n请选择要运行的测试:")
    print("  1 - 查看机器人当前状态")
    print("  2 - 小幅度delta运动 (X+2cm)")
    print("  3 - 圆形轨迹运动 (半径3cm)")
    print("  4 - 夹爪控制测试")
    print("  5 - 运行所有测试")
    print("  0 - 退出")
    
    try:
        choice = input("\n输入选项 (0-5): ").strip()
    except KeyboardInterrupt:
        print("\n退出")
        return
    
    if choice == '1':
        test_real_robot_state()
    elif choice == '2':
        test_small_delta_movement()
    elif choice == '3':
        test_circular_motion()
    elif choice == '4':
        test_gripper_control()
    elif choice == '5':
        test_real_robot_state()
        input("\n按Enter继续下一个测试...")
        test_small_delta_movement()
        input("\n按Enter继续下一个测试...")
        test_circular_motion()
        input("\n按Enter继续下一个测试...")
        test_gripper_control()
    elif choice == '0':
        print("退出")
        return
    else:
        print("无效选项")
        return
    
    print("\n" + "#"*60)
    print("#  测试完成")
    print("#"*60 + "\n")


if __name__ == "__main__":
    main()

