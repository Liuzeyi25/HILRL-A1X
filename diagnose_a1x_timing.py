#!/usr/bin/env python
"""
A1X 诊断脚本 - 测试关节到达时间和误差
"""

import time
import zmq
import numpy as np


def diagnose_movement_time():
    """诊断机器人从命令到到达目标的时间"""
    print("\n" + "="*60)
    print("诊断: 测量运动执行时间")
    print("="*60)
    
    ctx = zmq.Context()
    
    # 连接到状态端口
    state_socket = ctx.socket(zmq.REQ)
    state_socket.setsockopt(zmq.RCVTIMEO, 5000)
    state_socket.connect("tcp://127.0.0.1:6101")
    
    # 连接到命令端口
    cmd_socket = ctx.socket(zmq.REQ)
    cmd_socket.setsockopt(zmq.RCVTIMEO, 15000)  # 15秒超时
    cmd_socket.connect("tcp://127.0.0.1:6100")
    
    # 获取初始状态
    print("\n获取初始关节位置...")
    state_socket.send_json({"cmd": "get_state"})
    state = state_socket.recv_json()
    
    if not state or 'positions' not in state:
        print("✗ 无法获取状态")
        return
    
    initial_joints = np.array(state['positions'][:6])
    print(f"初始关节: {np.array2string(initial_joints, precision=3)}")
    
    # 发送一个小delta命令
    delta_pose = [0.01, 0.0, 0.0, 0.0, 0.0, 0.0, 50.0]  # 1cm X方向
    
    print(f"\n发送命令: X+1cm")
    print("监控关节变化...")
    
    start_time = time.time()
    cmd_socket.send_json({
        "cmd": "command_eef_pose",
        "pose": delta_pose,
        "wait_for_completion": False,  # 不等待，手动轮询
        "timeout": 10.0
    })
    
    # 接收命令确认
    response = cmd_socket.recv_json()
    if response.get('status') != 'ok':
        print(f"✗ 命令失败: {response.get('error')}")
        return
    
    target_joints = np.array(response.get('target_joints', []))
    print(f"目标关节: {np.array2string(target_joints, precision=3)}")
    print(f"关节变化: {np.array2string(target_joints - initial_joints, precision=3)}")
    
    # 轮询关节位置
    print("\n开始轮询关节位置...")
    print(f"{'时间(s)':<8} {'关节误差(rad)':<15} {'最大误差(rad)':<15} {'状态'}")
    print("-" * 60)
    
    last_state_time = start_time
    reached = False
    tolerance = 0.01  # 0.01 rad = 0.57°
    
    for i in range(200):  # 最多20秒
        try:
            # 查询当前状态
            state_socket.send_json({"cmd": "get_state"})
            state = state_socket.recv_json()
            
            if not state or 'positions' not in state:
                time.sleep(0.1)
                continue
            
            current_joints = np.array(state['positions'][:6])
            joint_errors = np.abs(current_joints - target_joints)
            max_error = joint_errors.max()
            
            elapsed = time.time() - start_time
            
            status = "运动中"
            if max_error < tolerance:
                status = "✓ 到达"
                reached = True
            
            # 每0.5秒打印一次
            if elapsed - last_state_time >= 0.5 or reached:
                print(f"{elapsed:<8.2f} {np.array2string(joint_errors, precision=4):<15} {max_error:<15.4f} {status}")
                last_state_time = elapsed
            
            if reached:
                print(f"\n✓ 到达目标，总耗时: {elapsed:.2f}秒")
                break
            
            time.sleep(0.1)
            
        except zmq.Again:
            print("⚠️  状态查询超时")
            time.sleep(0.1)
    
    if not reached:
        elapsed = time.time() - start_time
        print(f"\n⚠️  未到达目标，耗时: {elapsed:.2f}秒")
        print(f"最终关节误差: {max_error:.4f} rad ({np.rad2deg(max_error):.2f}°)")
        print(f"容忍度阈值: {tolerance:.4f} rad ({np.rad2deg(tolerance):.2f}°)")
    
    state_socket.close()
    cmd_socket.close()
    ctx.term()


def test_joint_tolerance():
    """测试不同的关节容忍度"""
    print("\n" + "="*60)
    print("测试: 关节容忍度对执行时间的影响")
    print("="*60)
    
    tolerances = [0.02, 0.015, 0.01, 0.005]  # rad
    
    for tol in tolerances:
        print(f"\n测试容忍度: {tol:.4f} rad ({np.rad2deg(tol):.2f}°)")
        
        ctx = zmq.Context()
        cmd_socket = ctx.socket(zmq.REQ)
        cmd_socket.setsockopt(zmq.RCVTIMEO, 15000)
        cmd_socket.connect("tcp://127.0.0.1:6100")
        
        delta_pose = [0.01, 0.0, 0.0, 0.0, 0.0, 0.0, 50.0]
        
        start_time = time.time()
        cmd_socket.send_json({
            "cmd": "command_eef_pose",
            "pose": delta_pose,
            "wait_for_completion": True,
            "timeout": 10.0,
            "joint_tolerance": tol  # 传递容忍度
        })
        
        try:
            response = cmd_socket.recv_json()
            elapsed = time.time() - start_time
            
            if response.get('status') == 'ok':
                reached = response.get('reached', False)
                error = response.get('final_error', float('inf'))
                
                status_str = "✓ 到达" if reached else "✗ 超时"
                print(f"  {status_str} | 耗时: {elapsed:.2f}s | 最终误差: {error:.4f} rad")
            else:
                print(f"  ✗ 失败: {response.get('error')}")
                
        except zmq.Again:
            print(f"  ✗ 超时")
        
        cmd_socket.close()
        ctx.term()
        
        time.sleep(1.0)  # 间隔1秒


def main():
    print("\n" + "#"*60)
    print("#  A1X 运动诊断工具")
    print("#"*60)
    
    print("\n请选择测试:")
    print("  1 - 诊断运动执行时间（实时监控）")
    print("  2 - 测试不同关节容忍度")
    print("  0 - 退出")
    
    try:
        choice = input("\n输入选项: ").strip()
    except KeyboardInterrupt:
        print("\n退出")
        return
    
    if choice == '1':
        diagnose_movement_time()
    elif choice == '2':
        test_joint_tolerance()
    elif choice == '0':
        print("退出")
    else:
        print("无效选项")
    
    print("\n" + "#"*60)
    print("#  诊断完成")
    print("#"*60 + "\n")


if __name__ == "__main__":
    main()
