"""
测试 Gello 跟随功能（双向控制）

验证新架构中恢复的双向控制功能：
1. Reset 时机器人移动到初始位置
2. Gello 自动跟随机器人移动
3. 同步完成后 Gello 回到远程操控模式

使用方法：
    python examples/test_script/test_gello_following.py
"""

import sys
from pathlib import Path

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent.parent.parent  # /home/dungeon_master/conrft
sys.path.insert(0, str(project_root))

# 添加 examples 目录到 Python 路径（用于导入 experiments 模块）
examples_dir = project_root / 'examples'
sys.path.insert(0, str(examples_dir))

# 添加 Gello 软件目录到 Python 路径
sys.path.insert(0, str(project_root / 'Gello' / 'gello_software'))

import numpy as np
import time

def test_bidirectional_control():
    """测试双向控制功能"""
    
    print("=" * 80)
    print("🔄 测试 Gello 双向控制（跟随功能）")
    print("=" * 80)
    
    # Import config
    exp_name = "a1x_pick_banana"
    config_module = __import__(
        f"experiments.{exp_name}.config", 
        fromlist=["TrainConfig"]
    )
    
    print("\n1️⃣ 创建环境（启用双向控制）...")
    train_config = config_module.TrainConfig()
    env = train_config.get_environment(
        fake_env=False,
        save_video=False,
        classifier=False,
    )
    
    # 检查是否启用了 sync_on_reset
    has_gello_wrapper = False
    temp_env = env
    while hasattr(temp_env, 'env'):
        if temp_env.__class__.__name__ == 'GelloIntervention':
            has_gello_wrapper = True
            if hasattr(temp_env, 'sync_on_reset'):
                print(f"   ✅ 找到 GelloIntervention wrapper")
                print(f"   🔄 sync_on_reset = {temp_env.sync_on_reset}")
                print(f"   ⏱️  reset_follow_duration = {temp_env.reset_follow_duration}s")
                break
        temp_env = temp_env.env
    
    if not has_gello_wrapper:
        print("   ⚠️  未找到 GelloIntervention wrapper")
        print("   💡 确保在 config.py 中创建了 GelloIntervention")
        return
    
    print("\n2️⃣ 开始测试循环...")
    print("   测试内容：")
    print("   1. Reset 环境")
    print("   2. 观察 Gello 是否跟随机器人到初始位置")
    print("   3. 验证同步完成后可以远程操控")
    print()
    
    for episode in range(3):
        print(f"\n{'='*80}")
        print(f"🧪 Episode {episode + 1}/3")
        print(f"{'='*80}")
        
        print("\n🔄 调用 env.reset()...")
        print("   预期行为：")
        print("   1. 机器人移动到初始位置")
        print("   2. Gello 自动跟随机器人")
        print("   3. 同步完成后 Gello 回到远程操控模式")
        print()
        
        obs, info = env.reset()
        
        print("\n✅ Reset 完成")
        print("   💡 检查上面的日志：")
        print("      - 是否看到 '🤖 Robot reset position'")
        print("      - 是否看到 '🎯 Gello 目标位置'")
        print("      - 是否看到 '⚡ 同步 Gello 到机器人位置'")
        print("      - 是否看到 '✅ Gello 已同步'")
        
        print("\n🎮 测试远程操控 (10步)...")
        print("   💡 移动 Gello 设备，验证机器人是否跟随")
        
        for step in range(10):
            action = np.zeros(env.action_space.shape)
            obs, rew, done, truncated, info = env.step(action)
            
            if step % 3 == 0:
                print(f"   Step {step}/10", end='\r')
            
            time.sleep(0.1)
        
        print(f"\n   ✅ 远程操控测试完成")
        
        if episode < 2:
            print(f"\n⏳ 等待 3 秒后进行下一次 Reset 测试...")
            time.sleep(3)
    
    print(f"\n{'='*80}")
    print(f"✅ 双向控制测试完成")
    print(f"{'='*80}")
    
    print("\n📊 验证清单：")
    print("   □ Reset 时看到 Gello 跟随日志")
    print("   □ Gello 物理移动到机器人位置")
    print("   □ 同步完成后可以远程操控")
    print("   □ 多次 Reset 都能正常同步")
    
    print("\n💡 如果 Gello 没有跟随：")
    print("   1. 检查 config.py 中是否设置 sync_on_reset=True")
    print("   2. 检查 Agent 是否支持 start_following/stop_following")
    print("   3. 检查逆映射函数是否正确")
    
    env.close()


def check_config():
    """检查配置是否正确"""
    
    print("\n" + "=" * 80)
    print("🔍 检查配置")
    print("=" * 80)
    
    # Check if config has sync_on_reset parameter
    exp_name = "a1x_pick_banana"
    config_module = __import__(
        f"experiments.{exp_name}.config", 
        fromlist=["TrainConfig"]
    )
    
    print("\n检查 TrainConfig...")
    train_config = config_module.TrainConfig()
    
    # Check if gello_config_path exists
    if hasattr(train_config, 'gello_config_path'):
        print(f"   ✅ gello_config_path: {train_config.gello_config_path}")
    else:
        print(f"   ❌ 未找到 gello_config_path")
    
    print("\n💡 推荐配置（在 config.py 中）：")
    print("""
    env = GelloIntervention(
        env,
        left_config_path=self.gello_config_path,
        always_intervene=True,      # 始终启用干预
        sync_on_reset=True,          # 🔧 启用双向控制
        reset_follow_duration=0.5,   # 跟随持续时间
    )
    """)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--check-config', action='store_true', help='只检查配置')
    args = parser.parse_args()
    
    if args.check_config:
        check_config()
    else:
        print("\n🎯 提示：")
        print("   1. 确保机器人已连接并上电")
        print("   2. 确保 Gello 设备已连接")
        print("   3. 准备观察 Gello 在 Reset 时的跟随动作")
        print()
        input("按 Enter 开始测试...")
        
        try:
            test_bidirectional_control()
        except KeyboardInterrupt:
            print("\n\n❌ 测试中断")
        except Exception as e:
            print(f"\n\n❌ 测试失败: {e}")
            import traceback
            traceback.print_exc()
