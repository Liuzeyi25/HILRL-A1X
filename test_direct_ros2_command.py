#!/usr/bin/env python3
"""
直接测试 ROS2 发布命令 - 绕过 ZMQ
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
import numpy as np
import time

class TestPublisher(Node):
    def __init__(self):
        super().__init__('test_direct_command')
        self.publisher = self.create_publisher(
            JointState,
            '/motion_target/target_joint_state_arm',
            10
        )
        self.get_logger().info('✓ 测试发布者已启动')
    
    def publish_command(self, joint_positions):
        """发布关节命令"""
        msg = JointState()
        msg.name = ['joint_1', 'joint_2', 'joint_3', 'joint_4', 'joint_5', 'joint_6', 'gripper']
        msg.position = joint_positions.tolist()
        
        self.publisher.publish(msg)
        self.get_logger().info(f'发布命令: {np.array2string(joint_positions[:6], precision=3)}')

def main():
    print("="*60)
    print("测试: 直接发布 ROS2 关节命令")
    print("="*60)
    
    rclpy.init()
    node = TestPublisher()
    
    # 读取当前关节位置
    print("\n请先运行以下命令查看当前关节位置:")
    print("ros2 topic echo /hdas/feedback_arm --once")
    print("\n然后输入当前的6个关节位置（用空格分隔）:")
    
    try:
        user_input = input("关节位置: ")
        current_joints = np.array([float(x) for x in user_input.split()])
        
        if len(current_joints) != 6:
            print(f"错误: 需要6个关节，你输入了{len(current_joints)}个")
            return
        
        print(f"\n当前关节: {np.array2string(current_joints, precision=3)}")
        
        # 计算小幅度移动 - 第1个关节+0.05 rad
        target_joints = current_joints.copy()
        target_joints[0] += 0.05  # +0.05 rad ≈ 2.9°
        
        # 添加夹爪位置
        target_with_gripper = np.append(target_joints, 50.0)
        
        print(f"目标关节: {np.array2string(target_joints, precision=3)}")
        print(f"关节1变化: +0.05 rad (+2.9°)")
        
        print("\n按 Enter 发送命令...")
        input()
        
        # 连续发送命令 5 秒
        print("\n开始发送命令（5秒）...")
        start_time = time.time()
        count = 0
        
        while time.time() - start_time < 5.0:
            node.publish_command(target_with_gripper)
            count += 1
            time.sleep(0.01)  # 100Hz
            rclpy.spin_once(node, timeout_sec=0)
        
        print(f"\n✓ 共发送 {count} 条命令")
        print("\n检查机器人是否移动了...")
        
    except KeyboardInterrupt:
        print("\n中断")
    except Exception as e:
        print(f"\n错误: {e}")
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
