#!/usr/bin/env python3
"""
测试 Gello joint_state 维度的简单脚本
"""

import sys
import os
import time
import numpy as np

# 添加路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, 'Gello/gello_software')

def test_gello_dimensions():
    """测试 Gello agent 返回的动作维度"""
    
    print("=" * 60)
    print("Gello Dimensions Test")
    print("=" * 60)
    
    try:
        # 导入 Gello
        from gello.agents.gello_agent import GelloAgent, DynamixelRobotConfig
        
        # 配置 (A1_X 默认配置)
        config = DynamixelRobotConfig(
            joint_ids=[1, 2, 3, 4, 5, 6],
            joint_offsets=[1.5708, 4.71239, 4.71239, 3.14159, 1.5708, 3.14159],
            joint_signs=[1.0, -1.0, -1.0, -1.0, 1.0, 1.0],
            gripper_config=[7, 139.66015625, 199.16015625]
        )
        
        # 创建 agent
        port = "/dev/serial/by-id/usb-FTDI_USB__-__Serial_Converter_FTA7NNNU-if00-port0"
        print(f"\n📡 Connecting to Gello on port: {port}")
        agent = GelloAgent(port=port, dynamixel_config=config)
        print("✅ Connected successfully!")
        
        # 测试读取
        print("\n" + "-" * 60)
        print("Testing joint_state readings...")
        print("-" * 60)
        
        for i in range(5):
            print(f"\nReading #{i+1}:")
            
            # 调用 act()
            joint_state = agent.act({})
            
            # 打印详细信息
            print(f"  Type: {type(joint_state)}")
            print(f"  Shape: {joint_state.shape if hasattr(joint_state, 'shape') else 'N/A'}")
            print(f"  Length: {len(joint_state)}")
            print(f"  Data type: {joint_state.dtype if hasattr(joint_state, 'dtype') else type(joint_state[0])}")
            print(f"  Values: {joint_state}")
            
            if len(joint_state) >= 7:
                print(f"\n  📊 Breakdown:")
                print(f"     Arm joints [0:6]: {joint_state[:6]}")
                print(f"     Gripper [6]:      {joint_state[6]}")
            else:
                print(f"  ⚠️  WARNING: Expected 7 DOF, got {len(joint_state)}")
            
            time.sleep(0.5)
        
        print("\n" + "=" * 60)
        print("✅ Test completed successfully!")
        print("=" * 60)
        
        # 清理
        if hasattr(agent, 'close'):
            agent.close()
            
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit_code = test_gello_dimensions()
    sys.exit(exit_code)
