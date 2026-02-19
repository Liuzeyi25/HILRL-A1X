#!/usr/bin/env python3
"""
简单的 Gello 测试脚本 - 用于诊断 follower 模式问题
"""

import sys
import time
import numpy as np

# 添加路径
sys.path.insert(0, 'Gello/gello_software')

from gello.agents.gello_agent import GelloAgent, DynamixelRobotConfig

def test_gello_basic():
    """测试基本的 Gello 读取"""
    print("="*60)
    print("测试 1: 基本 Gello 读取")
    print("="*60)
    
    try:
        # 使用与 launch_yaml.py 相同的配置
        config = DynamixelRobotConfig(
            joint_ids=[1, 2, 3, 4, 5, 6],
            joint_offsets=[1.5708, 4.71239, 4.71239, 3.14159, 1.5708, 3.14159],
            joint_signs=[1.0, -1.0, -1.0, -1.0, 1.0, 1.0],
            gripper_config=[7, 139.66015625, 199.16015625]
        )
        
        agent = GelloAgent(
            port="/dev/serial/by-id/usb-FTDI_USB__-__Serial_Converter_FTA7NNNU-if00-port0",
            dynamixel_config=config
        )
        
        print("✅ Gello Agent 初始化成功")
        
        # 读取几次状态
        print("\n读取 Gello 状态（5次）：")
        for i in range(5):
            joints = agent.act({})
            print(f"  {i+1}. {joints}")
            time.sleep(0.2)
        
        print("\n✅ 基本读取测试通过")
        return agent
        
    except Exception as e:
        print(f"\n❌ 失败: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_gello_follower(agent):
    """测试 Gello follower 模式"""
    print("\n" + "="*60)
    print("测试 2: Gello Follower 模式")
    print("="*60)
    
    if agent is None:
        print("❌ 跳过（agent 未初始化）")
        return
    
    try:
        from gello.agents.gello_follower import GelloFollower
        
        robot = agent._robot
        follower = GelloFollower(robot)
        
        print("✅ GelloFollower 对象创建成功")
        
        # 获取当前位置
        current_pos = robot.get_joint_state()
        print(f"\n当前 Gello 位置: {current_pos}")
        
        # 尝试启动 follower 模式
        print("\n🔄 启动 follower 模式...")
        
        try:
            follower.start()
            print("✅ Follower 模式启动成功!")
            
            # 尝试命令几个位置
            print("\n测试命令 Gello 移动到当前位置...")
            for i in range(3):
                follower.command_follow(current_pos)
                print(f"  命令 {i+1} 发送成功")
                time.sleep(0.5)
            
            print("\n✅ Follower 命令测试通过")
            
            # 停止 follower 模式
            print("\n🛑 停止 follower 模式...")
            follower.stop()
            print("✅ 已返回自由模式")
            
        except Exception as e:
            print(f"\n❌ Follower 模式失败: {e}")
            import traceback
            traceback.print_exc()
            
            # 尝试详细诊断
            print("\n🔍 详细诊断:")
            driver = robot._driver
            
            # 检查 torque 状态
            print("  - 检查驱动器状态...")
            try:
                # 读取第一个关节的状态
                torque_enabled = driver.read_position()  # 如果能读取说明通信正常
                print(f"    ✓ 驱动器通信正常")
            except Exception as e2:
                print(f"    ✗ 驱动器通信异常: {e2}")
            
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()


def test_manual_control():
    """手动测试驱动器模式切换"""
    print("\n" + "="*60)
    print("测试 3: 手动驱动器模式切换")
    print("="*60)
    
    try:
        from gello.agents.gello_agent import GelloAgent, DynamixelRobotConfig
        from gello.dynamixel.driver import POSITION_CONTROL_MODE, CURRENT_CONTROL_MODE
        
        config = DynamixelRobotConfig(
            joint_ids=[1, 2, 3, 4, 5, 6],
            joint_offsets=[1.5708, 4.71239, 4.71239, 3.14159, 1.5708, 3.14159],
            joint_signs=[1.0, -1.0, -1.0, -1.0, 1.0, 1.0],
            gripper_config=[7, 139.66015625, 199.16015625]
        )
        
        agent = GelloAgent(
            port="/dev/serial/by-id/usb-FTDI_USB__-__Serial_Converter_FTA7NNNU-if00-port0",
            dynamixel_config=config
        )
        
        robot = agent._robot
        driver = robot._driver
        
        print("✅ 驱动器初始化成功")
        
        # 读取当前位置
        current_pos = robot.get_joint_state()
        print(f"\n当前位置: {current_pos}")
        
        # 测试模式切换
        print("\n🔄 测试模式切换...")
        
        print("  1. 禁用 torque...")
        driver.set_torque_mode(False)
        time.sleep(0.5)
        
        print("  2. 设置为位置控制模式...")
        driver.set_operating_mode(POSITION_CONTROL_MODE)
        time.sleep(0.5)
        
        print("  3. 启用 torque...")
        driver.set_torque_mode(True)
        time.sleep(1.0)  # 给更多时间稳定
        
        print("  4. 尝试写入目标位置...")
        try:
            # 使用 set_joints 而不是 write_desired_pos
            driver.set_joints(current_pos[:6].tolist())
            print("  ✅ 位置写入成功!")
        except Exception as e:
            print(f"  ❌ 位置写入失败: {e}")
        
        # 恢复到电流模式
        print("\n  5. 恢复到电流控制模式...")
        driver.set_torque_mode(False)
        time.sleep(0.3)
        driver.set_operating_mode(CURRENT_CONTROL_MODE)
        time.sleep(0.3)
        driver.set_torque_mode(True)
        time.sleep(0.3)
        
        print("\n✅ 模式切换测试完成")
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    print("\n" + "="*60)
    print("🔧 Gello Follower 诊断工具")
    print("="*60)
    
    agent = None
    
    try:
        # 测试 1: 基本读取
        agent = test_gello_basic()
        
        # 测试 2: Follower 模式（使用同一个 agent，不要关闭）
        if agent:
            test_gello_follower(agent)
        
        # 注意：测试3需要新连接，所以先关闭agent
        if agent and hasattr(agent, '_robot') and hasattr(agent._robot, '_driver'):
            print("\n🔌 关闭连接以准备测试3...")
            agent._robot._driver.close()
            time.sleep(1.0)
            agent = None
        
        # 测试 3: 手动模式切换（新的连接）
        test_manual_control()
        
    finally:
        # 确保清理
        if agent and hasattr(agent, '_robot') and hasattr(agent._robot, '_driver'):
            try:
                agent._robot._driver.close()
            except:
                pass
    
    print("\n" + "="*60)
    print("✅ 诊断完成")
    print("="*60)
