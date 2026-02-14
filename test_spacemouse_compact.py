#!/usr/bin/env python3
"""
简单的 SpaceMouse Compact 测试脚本
显示 6 自由度的输入和按钮状态
"""

import sys
import time

# 确保导入本地的 pyspacemouse.py 文件
sys.path.insert(0, '/home/dungeon_master/conrft/serl_robot_infra/franka_env/spacemouse')
import pyspacemouse

print(f"导入的 pyspacemouse 路径: {pyspacemouse.__file__}")

def print_state_callback(state):
    """打印当前状态"""
    print(f"时间: {state.t:.2f}s | "
          f"X: {state.x:+.2f} Y: {state.y:+.2f} Z: {state.z:+.2f} | "
          f"Roll: {state.roll:+.2f} Pitch: {state.pitch:+.2f} Yaw: {state.yaw:+.2f} | "
          f"按钮: {list(state.buttons)}")

def button_callback(state, buttons):
    """按钮回调"""
    if buttons[0] == 1:
        print(">>> 左键按下!")
    if buttons[1] == 1:
        print(">>> 右键按下!")

def main():
    print("=" * 80)
    print("SpaceMouse Compact 测试程序")
    print("=" * 80)
    
    # 列出所有连接的设备
    devices = pyspacemouse.list_devices()
    print(f"\n检测到的设备: {devices}")
    
    if not devices:
        print("错误: 未检测到 SpaceMouse 设备!")
        print("请确保:")
        print("  1. SpaceMouse 已插入 USB 端口")
        print("  2. 已安装 easyhid: pip install easyhid")
        return
    
    # 打开 SpaceMouse Compact
    print("\n正在打开 SpaceMouse Compact...")
    try:
        dev = pyspacemouse.open(
            dof_callback=print_state_callback,
            button_callback=button_callback,
            device="SpaceMouse Compact"
        )
        
        if dev is None:
            print("错误: 无法打开设备!")
            return
        
        print(f"✓ 设备已连接: {dev.describe_connection()}")
        print("\n" + "=" * 80)
        print("开始读取数据 (按 Ctrl+C 退出)")
        print("=" * 80)
        print("\n请移动 SpaceMouse 或按下按钮...")
        print()
        
        # 持续读取数据
        while True:
            state = dev.read()
            time.sleep(0.01)  # 100Hz 读取频率
            
    except KeyboardInterrupt:
        print("\n\n正在退出...")
    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if dev:
            dev.close()
            print("设备已关闭")

if __name__ == "__main__":
    main()
