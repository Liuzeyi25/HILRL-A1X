#!/usr/bin/env python3
"""
快速调试脚本:检查动作空间维度
"""

from experiments.mappings import CONFIG_MAPPING

def check_dimensions():
    """检查环境的动作空间维度"""
    print("="*70)
    print("🔍 检查环境动作空间维度")
    print("="*70)
    
    config = CONFIG_MAPPING["a1x_pick_banana"]()
    env = config.get_environment(
        fake_env=False, 
        save_video=False, 
        classifier=False, 
        stack_obs_num=1
    )
    
    print(f"\n✅ 环境创建成功")
    print(f"\n📐 动作空间信息:")
    print(f"   Shape: {env.action_space.shape}")
    print(f"   Low:   {env.action_space.low}")
    print(f"   High:  {env.action_space.high}")
    
    print(f"\n📐 观察空间信息:")
    if hasattr(env, 'observation_space'):
        print(f"   Keys: {list(env.observation_space.spaces.keys())}")
        if 'state' in env.observation_space.spaces:
            state_space = env.observation_space.spaces['state']
            if hasattr(state_space, 'spaces'):
                print(f"   State keys: {list(state_space.spaces.keys())}")
    
    # 检查 wrapper 配置
    print(f"\n🔧 Wrapper 配置:")
    current_env = env
    wrapper_count = 0
    while hasattr(current_env, 'env'):
        wrapper_name = current_env.__class__.__name__
        print(f"   [{wrapper_count}] {wrapper_name}")
        
        # 检查 GelloIntervention 的配置
        if wrapper_name == "GelloIntervention":
            print(f"       - gripper_enabled: {current_env.gripper_enabled}")
            print(f"       - action_space: {current_env.action_space.shape}")
        
        current_env = current_env.env
        wrapper_count += 1
    
    print(f"   [{wrapper_count}] {current_env.__class__.__name__} (base)")
    
    print("\n" + "="*70)

if __name__ == "__main__":
    check_dimensions()
