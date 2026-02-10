#!/usr/bin/env python3
"""
快速诊断 A1X 为什么不动
"""

import subprocess
import time
import sys

def run_command(cmd, description):
    """运行命令并显示结果"""
    print(f"\n{'='*60}")
    print(f"检查: {description}")
    print(f"{'='*60}")
    try:
        result = subprocess.run(
            cmd, 
            shell=True, 
            capture_output=True, 
            text=True, 
            timeout=5
        )
        print(result.stdout)
        if result.stderr and "WARNING" not in result.stderr:
            print(f"错误: {result.stderr}")
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        print("⚠️  命令超时")
        return False
    except Exception as e:
        print(f"❌ 错误: {e}")
        return False

def main():
    print("\n" + "#"*60)
    print("#  A1X 不动问题诊断")
    print("#"*60)
    
    # 1. 检查 ROS2 环境
    print("\n[1/6] 检查 ROS2 环境...")
    if not run_command("ros2 --version", "ROS2 版本"):
        print("❌ ROS2 未安装或环境未加载")
        print("解决: source /opt/ros/humble/setup.bash")
        return
    
    # 2. 检查关键话题
    print("\n[2/6] 检查关键话题...")
    run_command(
        "ros2 topic list | grep -E 'target_joint|target_pose|feedback'",
        "控制和反馈话题"
    )
    
    # 3. 检查话题发布者/订阅者
    print("\n[3/6] 检查关节命令话题...")
    run_command(
        "ros2 topic info /motion_target/target_joint_state_arm",
        "关节命令话题信息"
    )
    
    print("\n[4/6] 检查位姿命令话题...")
    run_command(
        "ros2 topic info /motion_target/target_pose_arm",
        "位姿命令话题信息"
    )
    
    # 4. 检查反馈数据
    print("\n[5/6] 检查机器人反馈...")
    result = subprocess.run(
        "timeout 2 ros2 topic echo /hdas/feedback_arm --once 2>&1 | head -20",
        shell=True,
        capture_output=True,
        text=True
    )
    if result.returncode == 0 and "position:" in result.stdout:
        print("✓ 机器人正在发送反馈数据")
        # 提取关节位置
        lines = result.stdout.split('\n')
        for i, line in enumerate(lines):
            if 'position:' in line:
                print(f"\n当前关节位置:")
                for j in range(i+1, min(i+8, len(lines))):
                    if lines[j].strip() and not lines[j].startswith('-'):
                        print(f"  {lines[j]}")
                break
    else:
        print("❌ 没有收到机器人反馈")
    
    # 5. 检查运动控制模式
    print("\n[6/6] 检查运动控制模式...")
    result = subprocess.run(
        "timeout 2 ros2 topic echo /hdas/feedback_status_arm --once 2>&1",
        shell=True,
        capture_output=True,
        text=True
    )
    if result.returncode == 0:
        print(result.stdout[:500])
    
    # 总结
    print("\n" + "="*60)
    print("诊断建议:")
    print("="*60)
    
    print("""
1. 如果话题存在但机器人不动，可能原因:
   - 机器人控制器未启用运动控制模式
   - 需要先"使能"机器人
   - 安全限位触发

2. 尝试手动发送简单命令:
   ros2 topic pub --once /motion_target/target_joint_state_arm sensor_msgs/msg/JointState "{name: ['joint_1'], position: [0.0]}"

3. 检查 A1X 控制器状态:
   - 查看 A1X 控制器界面
   - 确认是否处于"远程控制"模式
   - 检查是否有急停或报警

4. 如果使用位姿控制而非关节控制:
   - 机器人可能只响应 /motion_target/target_pose_arm
   - 而不是 /motion_target/target_joint_state_arm
    """)
    
    print("\n" + "#"*60)
    print("#  诊断完成")
    print("#"*60 + "\n")

if __name__ == "__main__":
    main()
