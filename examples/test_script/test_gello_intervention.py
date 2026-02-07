#!/usr/bin/env python3
"""
测试 Gello 干预模式下的动作映射
验证从 Gello 绝对位置到环境增量动作的转换是否正确
"""

import numpy as np
import time
from experiments.mappings import CONFIG_MAPPING

def test_gello_intervention():
    """测试 Gello 干预模式"""
    print("="*70)
    print("🧪 测试 Gello 干预模式 - 绝对位置到增量动作转换")
    print("="*70)
    
    # 创建环境 (包含 GelloIntervention wrapper)
    config = CONFIG_MAPPING["a1x_pick_banana"]()
    env = config.get_environment(
        fake_env=False, 
        save_video=False, 
        classifier=False, 
        stack_obs_num=1
    )
    
    print("\n✅ 环境创建成功")
    print(f"   动作空间: {env.action_space}")
    print(f"   观察空间: {env.observation_space.spaces.keys()}")
    
    # Reset 环境
    print("\n🔄 执行 Reset...")
    obs, info = env.reset()
    print("✅ Reset 完成")
    
    # 等待用户准备
    input("\n📍 请将 Gello 移动到 reset 位置附近，然后按 Enter 继续...")
    
    # 测试循环
    print("\n🎮 开始测试遥控模式:")
    print("   - 移动 Gello 观察机械臂响应")
    print("   - 按 Ctrl+C 退出")
    print()
    
    step_count = 0
    try:
        while True:
            # Step with zero action (will be replaced by Gello if moved)
            zero_action = np.zeros(env.action_space.shape)
            obs, reward, done, truncated, info = env.step(zero_action)
            
            step_count += 1
            
            # 检查是否有干预
            if "intervene_action" in info:
                intervene_action = info["intervene_action"]
                print(f"\n[Step {step_count}] 🎯 Gello 干预检测到!")
                print(f"   增量动作 (前3个关节): [{intervene_action[0]:7.4f}, {intervene_action[1]:7.4f}, {intervene_action[2]:7.4f}]")
                print(f"   当前位置 (前3个关节): {obs['state'][:3]}")
            else:
                # 每100步打印一次状态
                if step_count % 100 == 0:
                    print(f"[Step {step_count}] ⏸️  无干预 (Gello 静止)")
            
            # 检查 episode 结束
            if done:
                print(f"\n✅ Episode 完成! (step_count={step_count})")
                obs, info = env.reset()
                step_count = 0
            
            time.sleep(0.01)  # 100Hz
            
    except KeyboardInterrupt:
        print("\n\n⚠️  测试被用户中断")
    
    print("\n📊 测试总结:")
    print(f"   总步数: {step_count}")
    print("   如果机械臂能正确跟随 Gello 移动，说明修复成功! ✅")
    print()

if __name__ == "__main__":
    test_gello_intervention()
