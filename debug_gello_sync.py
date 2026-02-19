#!/usr/bin/env python3
"""
🔍 Gello 同步调试脚本

用于诊断为什么第二次 reset 后 Gello 位置不对的问题

使用方法:
    python debug_gello_sync.py --exp_name a1x_pick_banana
"""

import sys
import os
import numpy as np
import time
from absl import app, flags

# 添加 examples 目录到 path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'examples'))

from experiments.mappings import CONFIG_MAPPING

FLAGS = flags.FLAGS
flags.DEFINE_string("exp_name", None, "Name of experiment")

def print_section(title):
    print("\n" + "="*70)
    print(f"  {title}")
    print("="*70)

def get_gello_position(env):
    """尝试多种方法获取 Gello 位置"""
    wrapper = env
    while wrapper is not None:
        if hasattr(wrapper, '_get_current_gello_joints'):
            return wrapper._get_current_gello_joints()
        wrapper = getattr(wrapper, 'env', None)
    return None

def get_a1x_position(env):
    """获取 A1X 位置"""
    wrapper = env
    while wrapper is not None:
        if hasattr(wrapper, '_get_current_a1x_joints'):
            return wrapper._get_current_a1x_joints()
        wrapper = getattr(wrapper, 'env', None)
    return None

def main(_):
    assert FLAGS.exp_name in CONFIG_MAPPING, 'Experiment not found.'
    config = CONFIG_MAPPING[FLAGS.exp_name]()
    
    # 临时禁用 action chunking
    original_chunk_size = config.action_chunk_size
    config.action_chunk_size = None
    
    print_section("初始化环境")
    env = config.get_environment(
        fake_env=False, save_video=False, classifier=False, stack_obs_num=2)
    
    print("✅ 环境初始化完成")
    
    # 恢复配置
    config.action_chunk_size = original_chunk_size
    
    try:
        # 第一次 Reset
        print_section("第 1 次 Reset")
        print("⏳ 执行 env.reset()...")
        obs1, info1 = env.reset()
        
        print("\n📊 Reset 后状态检查:")
        gello_pos_1 = get_gello_position(env)
        a1x_pos_1 = get_a1x_position(env)
        
        if gello_pos_1 is not None:
            print(f"  Gello 位置: [{', '.join(f'{v:7.3f}' for v in gello_pos_1)}]")
        else:
            print("  ⚠️  无法读取 Gello 位置")
        
        if a1x_pos_1 is not None:
            print(f"  A1X  位置: [{', '.join(f'{v:7.3f}' for v in a1x_pos_1)}]")
        else:
            print("  ⚠️  无法读取 A1X 位置")
        
        # 简单执行几个 steps
        print("\n⏳ 执行 10 个 steps...")
        for i in range(10):
            dummy_action = np.zeros(7)
            obs, rew, done, truncated, info = env.step(dummy_action)
            if done:
                break
        
        print(f"✅ 完成 {i+1} 个 steps")
        
        # 第二次 Reset
        print_section("第 2 次 Reset（关键测试）")
        
        print("📍 Reset 前检查:")
        gello_before_2 = get_gello_position(env)
        a1x_before_2 = get_a1x_position(env)
        
        if gello_before_2 is not None:
            print(f"  Gello 位置: [{', '.join(f'{v:7.3f}' for v in gello_before_2)}]")
        if a1x_before_2 is not None:
            print(f"  A1X  位置: [{', '.join(f'{v:7.3f}' for v in a1x_before_2)}]")
        
        print("\n⏳ 执行 env.reset()...")
        obs2, info2 = env.reset()
        
        print("\n📊 Reset 后状态检查:")
        gello_pos_2 = get_gello_position(env)
        a1x_pos_2 = get_a1x_position(env)
        
        if gello_pos_2 is not None:
            print(f"  Gello 位置: [{', '.join(f'{v:7.3f}' for v in gello_pos_2)}]")
        else:
            print("  ⚠️  无法读取 Gello 位置")
        
        if a1x_pos_2 is not None:
            print(f"  A1X  位置: [{', '.join(f'{v:7.3f}' for v in a1x_pos_2)}]")
        else:
            print("  ⚠️  无法读取 A1X 位置")
        
        # 对比分析
        print_section("对比分析")
        
        if gello_pos_1 is not None and gello_pos_2 is not None:
            diff = np.abs(gello_pos_2[:6] - gello_pos_1[:6])
            max_diff = np.max(diff)
            print(f"第 1 次 vs 第 2 次 Gello 位置差异:")
            print(f"  差异: [{', '.join(f'{v:7.3f}' for v in diff)}]")
            print(f"  最大差异: {max_diff:.4f} rad")
            
            if max_diff < 0.05:
                print("  ✅ 两次 reset 后 Gello 位置一致（很好！）")
            elif max_diff < 0.2:
                print("  ⚠️  存在小偏差，但可能在可接受范围内")
            else:
                print("  ❌ 位置差异过大！这就是问题所在！")
        
        if a1x_pos_1 is not None and a1x_pos_2 is not None:
            diff = np.abs(a1x_pos_2[:6] - a1x_pos_1[:6])
            max_diff = np.max(diff)
            print(f"\n第 1 次 vs 第 2 次 A1X 位置差异:")
            print(f"  差异: [{', '.join(f'{v:7.3f}' for v in diff)}]")
            print(f"  最大差异: {max_diff:.4f} rad")
            
            if max_diff < 0.05:
                print("  ✅ 两次 reset 后 A1X 位置一致")
            else:
                print("  ⚠️  A1X 位置有差异（可能正常，取决于环境设置）")
        
        # 第三次 Reset（进一步确认）
        print_section("第 3 次 Reset（进一步确认）")
        print("⏳ 执行 env.reset()...")
        obs3, info3 = env.reset()
        
        gello_pos_3 = get_gello_position(env)
        a1x_pos_3 = get_a1x_position(env)
        
        if gello_pos_3 is not None:
            print(f"  Gello 位置: [{', '.join(f'{v:7.3f}' for v in gello_pos_3)}]")
            
            if gello_pos_2 is not None:
                diff = np.abs(gello_pos_3[:6] - gello_pos_2[:6])
                max_diff = np.max(diff)
                print(f"  vs 第 2 次差异: 最大 {max_diff:.4f} rad")
        
        print_section("诊断总结")
        print("✅ 测试完成")
        print("\n请检查上述输出中:")
        print("1. 第 2 次 reset 后 Gello 位置是否与第 1 次一致")
        print("2. 如果不一致，差异是在哪个关节")
        print("3. 是否有任何错误或警告信息")
        
    except KeyboardInterrupt:
        print("\n⚠️  测试被中断")
    except Exception as e:
        print(f"\n❌ 发生错误: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("\n🧹 清理环境...")
        try:
            env.close()
        except:
            pass
        print("✅ 清理完成")

if __name__ == "__main__":
    app.run(main)
