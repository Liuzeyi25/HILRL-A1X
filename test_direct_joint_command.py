#!/usr/bin/env python
"""
测试直接关节命令 - 绕过 EEF/IK
直接发送关节位置，就像正常工作的代码一样
"""

import time
import zmq
import numpy as np


def test_direct_joint_command():
    """直接发送关节命令，不通过IK"""
    print("\n" + "="*60)
    print("测试: 直接关节命令（绕过 IK）")
    print("="*60)
    
    ctx = zmq.Context()
    
    # 连接到命令端口
    command_socket = ctx.socket(zmq.REQ)
    command_socket.setsockopt(zmq.RCVTIMEO, 5000)
    command_socket.setsockopt(zmq.SNDTIMEO, 5000)
    command_socket.connect("tcp://127.0.0.1:6100")
    
    # 连接到状态端口
    state_socket = ctx.socket(zmq.REQ)
    state_socket.setsockopt(zmq.RCVTIMEO, 5000)
    state_socket.connect("tcp://127.0.0.1:6101")
    
    print("\n✓ 已连接到 ZMQ")
    
    # 1. 获取当前关节位置
    print("\n[1/3] 获取当前关节位置...")
    state_socket.send_json({"cmd": "get_state"})
    state = state_socket.recv_json()
    
    if not state or 'positions' not in state:
        print("❌ 无法获取状态")
        return
    
    current_joints = np.array(state['positions'][:7])  # 6关节 + 夹爪
    print(f"当前关节 (6+gripper): {np.array2string(current_joints, precision=3)}")
    print(f"末端位置: {state['ee_pos']}")
    
    # 2. 计算目标关节 - 第1个关节 +0.05 rad (约2.9°)
    print("\n[2/3] 计算目标关节位置...")
    target_joints = current_joints.copy()
    target_joints[0] += 0.05  # 第1个关节增加 0.05 rad
    
    print(f"目标关节: {np.array2string(target_joints, precision=3)}")
    print(f"变化: 关节1 +0.05 rad (+2.9°)")
    
    # 3. 发送关节命令（就像 A1XRobot.command_joint_positions 一样）
    print("\n[3/3] 发送关节命令...")
    print("按 Enter 发送命令...")
    input()
    
    try:
        command_socket.send_json({
            "cmd": "command_joint_state",
            "positions": target_joints.tolist()
        })
        response = command_socket.recv_json()
        
        if response.get('status') == 'ok':
            print("✓ 命令发送成功")
            print("观察机器人是否移动...")
            
            # 等待5秒观察
            time.sleep(5)
            
            # 检查最终位置
            print("\n检查最终位置...")
            state_socket.send_json({"cmd": "get_state"})
            final_state = state_socket.recv_json()
            
            if final_state:
                final_joints = np.array(final_state['positions'][:7])
                joint_diff = final_joints - current_joints
                
                print(f"最终关节: {np.array2string(final_joints, precision=3)}")
                print(f"实际变化: {np.array2string(joint_diff, precision=3)}")
                print(f"最大变化: {np.rad2deg(np.max(np.abs(joint_diff))):.2f}°")
                
                if np.max(np.abs(joint_diff)) > 0.01:
                    print("✓ 机器人移动了！")
                else:
                    print("❌ 机器人没有明显移动")
        else:
            print(f"❌ 命令失败: {response}")
    
    except zmq.Again:
        print("❌ 命令超时")
    except Exception as e:
        print(f"❌ 错误: {e}")
    
    command_socket.close()
    state_socket.close()
    ctx.term()


def test_publish_frequency():
    """测试命令发布频率 - 就像控制循环一样"""
    print("\n" + "="*60)
    print("测试: 高频关节命令（模拟控制循环）")
    print("="*60)
    
    ctx = zmq.Context()
    
    command_socket = ctx.socket(zmq.REQ)
    command_socket.setsockopt(zmq.RCVTIMEO, 100)
    command_socket.setsockopt(zmq.SNDTIMEO, 100)
    command_socket.connect("tcp://127.0.0.1:6100")
    
    state_socket = ctx.socket(zmq.REQ)
    state_socket.setsockopt(zmq.RCVTIMEO, 100)
    state_socket.connect("tcp://127.0.0.1:6101")
    
    # 获取当前位置
    state_socket.send_json({"cmd": "get_state"})
    state = state_socket.recv_json()
    
    if not state:
        print("❌ 无法获取状态")
        return
    
    current_joints = np.array(state['positions'][:7])
    print(f"初始关节: {np.array2string(current_joints, precision=3)}")
    
    # 目标：第1关节 +0.1 rad
    target_joints = current_joints.copy()
    target_joints[0] += 0.1
    
    print(f"目标关节: {np.array2string(target_joints, precision=3)}")
    print("\n按 Enter 开始高频发送（持续5秒）...")
    input()
    
    # 高频发送命令（100Hz）
    start_time = time.time()
    count = 0
    errors = 0
    
    print("\n开始发送...")
    while time.time() - start_time < 5.0:
        try:
            command_socket.send_json({
                "cmd": "command_joint_state",
                "positions": target_joints.tolist()
            })
            command_socket.recv_json()
            count += 1
        except zmq.Again:
            errors += 1
        
        time.sleep(0.01)  # 100Hz
    
    elapsed = time.time() - start_time
    print(f"\n✓ 发送完成")
    print(f"  发送次数: {count}")
    print(f"  失败次数: {errors}")
    print(f"  平均频率: {count/elapsed:.1f} Hz")
    
    # 检查最终位置
    time.sleep(0.5)
    state_socket.send_json({"cmd": "get_state"})
    final_state = state_socket.recv_json()
    
    if final_state:
        final_joints = np.array(final_state['positions'][:7])
        print(f"\n最终关节: {np.array2string(final_joints, precision=3)}")
        print(f"目标达成: {np.max(np.abs(final_joints - target_joints)) < 0.02}")
    
    command_socket.close()
    state_socket.close()
    ctx.term()


def main():
    print("\n" + "#"*60)
    print("#  直接关节命令测试")
    print("#  (绕过 EEF/IK，像正常代码一样)")
    print("#"*60)
    
    print("\n请选择测试:")
    print("  1 - 单次关节命令")
    print("  2 - 高频关节命令（5秒）")
    print("  0 - 退出")
    
    try:
        choice = input("\n输入选项: ").strip()
    except KeyboardInterrupt:
        print("\n退出")
        return
    
    if choice == '1':
        test_direct_joint_command()
    elif choice == '2':
        test_publish_frequency()
    elif choice == '0':
        print("退出")
    else:
        print("无效选项")
    
    print("\n" + "#"*60)
    print("#  测试完成")
    print("#"*60 + "\n")


if __name__ == "__main__":
    main()
