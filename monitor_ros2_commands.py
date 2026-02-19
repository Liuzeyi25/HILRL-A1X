#!/usr/bin/env python3
"""
测试 ROS2 话题是否收到命令
"""

import time
import subprocess
import sys

print("="*60)
print("测试: 监控 ROS2 命令话题")
print("="*60)

print("\n正在监听 /motion_target/target_joint_state_arm ...")
print("请在另一个终端运行测试脚本发送命令")
print("按 Ctrl+C 停止监听\n")

try:
    # 使用 ros2 topic echo 监听命令
    proc = subprocess.Popen(
        ["ros2", "topic", "echo", "/motion_target/target_joint_state_arm"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1
    )
    
    count = 0
    last_print_time = time.time()
    
    for line in proc.stdout:
        if "position:" in line:
            count += 1
            current_time = time.time()
            if current_time - last_print_time >= 1.0:  # 每秒打印一次
                print(f"[{time.strftime('%H:%M:%S')}] 收到命令 #{count}")
                last_print_time = current_time
        
        # 打印位置数据
        if line.strip() and not line.startswith("---"):
            sys.stdout.write(line)
    
except KeyboardInterrupt:
    print(f"\n\n总共收到 {count} 条命令")
    proc.terminate()
except Exception as e:
    print(f"错误: {e}")
