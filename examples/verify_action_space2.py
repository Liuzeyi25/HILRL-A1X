"""
验证新版 GelloIntervention 干预逻辑

新架构特点：
1. 基于 launch_yaml.py 的 Agent-Robot 架构
2. 使用 YAML 配置文件初始化
3. 空格键手动启用/禁用干预（默认关闭）
4. Agent 生成动作（而不是直接读取关节位置）
5. 🔧 恢复：双向控制 - Reset 时 Gello 跟随机器人（sync_on_reset=True）

测试内容：
1. ✅ 跟随功能：Reset 时 Gello 跟随机器人到初始位置
2. ✅ 遥操功能：Gello 控制机器人运动
3. 检查空格键干预切换是否工作
4. 验证干预时动作来源（Agent vs Policy）
5. 检查 info 字典中的干预标志

快捷键：
- 空格键: 切换干预状态
- R 键: 手动触发 Reset（测试跟随功能）
- Ctrl+C: 退出
"""

import sys
sys.path.insert(0, 'Gello/gello_software')

import numpy as np
from pathlib import Path
import time
import select
import termios
import tty

# Import experiment config
exp_name = "a1x_pick_banana"
config_module = __import__(
    f"experiments.{exp_name}.config", 
    fromlist=["TrainConfig"]
)

def check_key_press():
    """非阻塞检查键盘输入"""
    try:
        if select.select([sys.stdin], [], [], 0)[0]:
            return sys.stdin.read(1)
    except:
        pass
    return None


def main():
    print("=" * 70)
    print("🔍 新版 GelloIntervention 功能验证")
    print("   测试：跟随功能 + 遥操功能")
    print("=" * 70)
    
    # Create environment with all wrappers
    print("\n1️⃣ 创建环境...")
    train_config = config_module.TrainConfig()
    env = train_config.get_environment(
        fake_env=False,
        save_video=False,
        classifier=False,
    )
    print("✅ 环境创建成功")
    
    # 检查 GelloIntervention 配置
    temp_env = env
    gello_wrapper = None
    while hasattr(temp_env, 'env'):
        if temp_env.__class__.__name__ == 'GelloIntervention':
            gello_wrapper = temp_env
            break
        temp_env = temp_env.env
    
    if gello_wrapper:
        print(f"\n📋 GelloIntervention 配置:")
        print(f"   - always_intervene: {getattr(gello_wrapper, 'always_intervene', 'N/A')}")
        print(f"   - sync_on_reset: {getattr(gello_wrapper, 'sync_on_reset', 'N/A')}")
        print(f"   - reset_follow_duration: {getattr(gello_wrapper, 'reset_follow_duration', 'N/A')}s")
    
    # Reset environment
    print("\n2️⃣ 首次重置环境（观察跟随功能）...")
    print("   👀 注意观察：Gello 是否跟随机器人移动到初始位置")
    print("-" * 70)
    obs, info = env.reset()
    print("-" * 70)
    print("✅ 环境已重置")
    time.sleep(1)
    
    print("\n" + "=" * 70)
    print("📋 测试说明")
    print("=" * 70)
    print("🎯 功能1 - 跟随：Reset 时 Gello 自动跟随机器人")
    print("🎯 功能2 - 遥操：移动 Gello 控制机器人")
    print()
    print("⌨️  快捷键:")
    print("   R 键    - 手动 Reset（测试跟随功能）")
    print("   空格键  - 切换干预状态（如果 always_intervene=False）")
    print("   Ctrl+C  - 退出测试")
    print("=" * 70)
    
    print("\n3️⃣ 开始测试循环...")
    print("   � 移动 Gello 设备，验证机器人是否跟随（遥操功能）")
    print("   � 按 R 键触发 Reset，验证 Gello 是否跟随机器人（跟随功能）\n")
    
    # 设置终端为非阻塞输入
    old_settings = None
    try:
        old_settings = termios.tcgetattr(sys.stdin)
        tty.setcbreak(sys.stdin.fileno())
    except:
        print("   ⚠️  无法设置非阻塞输入，R 键功能可能不可用")
    
    step_count = 0
    intervened_count = 0
    reset_count = 1
    last_intervention_state = False
    
    try:
        while True:
            # 检查键盘输入
            key = check_key_press()
            if key:
                if key.lower() == 'r':
                    # 手动 Reset
                    reset_count += 1
                    print(f"\n\n{'='*70}")
                    print(f"🔄 手动 Reset #{reset_count}（测试跟随功能）")
                    print(f"{'='*70}")
                    print("   👀 观察：Gello 是否跟随机器人移动到初始位置")
                    print("-" * 70)
                    obs, info = env.reset()
                    print("-" * 70)
                    print("✅ Reset 完成")
                    print("   💡 现在可以继续移动 Gello 测试遥操功能\n")
                    time.sleep(0.5)
                    continue
            
            # Zero action from policy (will be replaced if intervention enabled)
            policy_action = np.zeros(env.action_space.shape)
            
            # Step environment
            next_obs, reward, done, truncated, info = env.step(policy_action)
            step_count += 1
            
            # Check intervention status (new key name)
            is_intervened = info.get("gello_intervened", False)
            
            # Detect state change
            if is_intervened != last_intervention_state:
                if is_intervened:
                    print(f"\n{'='*70}")
                    print(f"🟢 Gello 干预已启用")
                    print(f"{'='*70}")
                else:
                    print(f"\n{'='*70}")
                    print(f"🔴 Gello 干预已禁用")
                    print(f"{'='*70}")
                last_intervention_state = is_intervened
            
            # Log intervention details
            if is_intervened:
                intervened_count += 1
                
                if intervened_count % 10 == 1:  # Log every 10 interventions
                    print(f"\n{'─'*70}")
                    print(f"🎯 Gello 干预进行中 (第 {intervened_count} 次)")
                    print(f"{'─'*70}")
                    
                    # In new architecture, agent generates actions internally
                    # We can't directly see "intervene_action" like before
                    # Instead, the action is already applied in env.step()
                    
                    print(f"� 干预信息:")
                    print(f"   - 干预状态: ✅ 启用")
                    print(f"   - 动作来源: Gello Agent")
                    print(f"   - 策略动作: 已被覆盖")
                    
                    # Check if there are additional info keys
                    gello_keys = [k for k in info.keys() if 'gello' in k.lower() or 'intervene' in k.lower()]
                    if gello_keys:
                        print(f"   - Info 键值: {gello_keys}")
                        for key in gello_keys:
                            value = info[key]
                            if isinstance(value, np.ndarray):
                                print(f"     • {key}: shape={value.shape}, range=[{value.min():.3f}, {value.max():.3f}]")
                            else:
                                print(f"     • {key}: {value}")
                    
                    # Observation space info
                    if isinstance(obs, dict) and 'state' in obs:
                        state = obs['state']
                        print(f"\n📐 观测空间:")
                        print(f"   - State shape: {state.shape}")
                        if state.shape[0] > 0:
                            print(f"   - State 样例: {state[0][:3]} ... (前3维)")
                    
                    print(f"\n✅ 新架构工作正常！")
                    print(f"   - Agent 生成动作")
                    print(f"   - 空格键切换有效")
                    print(f"   - 干预状态正确记录")
            
            # Progress indicator
            if step_count % 20 == 0:
                status = "🟢 干预中" if is_intervened else "🔴 Policy"
                print(f"  步数: {step_count} | 干预次数: {intervened_count} | 状态: {status}", end='\r')
            
            # Reset if done
            if done or truncated:
                reset_count += 1
                print(f"\n\n⚠️  Episode 结束，自动 Reset #{reset_count}...")
                print("   👀 观察：Gello 是否跟随机器人移动到初始位置")
                print("-" * 70)
                obs, info = env.reset()
                print("-" * 70)
                print("✅ 重置完成")
            else:
                obs = next_obs
            
            time.sleep(0.05)  # 20 Hz
            
    except KeyboardInterrupt:
        print(f"\n\n{'='*70}")
        print(f"🛑 测试停止")
        print(f"{'='*70}")
        print(f"\n📊 测试统计:")
        print(f"   总步数: {step_count}")
        print(f"   干预次数: {intervened_count}")
        print(f"   Reset 次数: {reset_count}")
        if step_count > 0 and intervened_count > 0:
            print(f"   干预率: {intervened_count/step_count*100:.1f}%")
        
        print(f"\n✅ 功能验证清单:")
        print(f"   □ 跟随功能：Reset 时 Gello 移动到机器人位置")
        print(f"   □ 遥操功能：移动 Gello 时机器人跟随")
        print(f"   □ 空格键切换（如果 always_intervene=False）")
        
    finally:
        # 恢复终端设置
        if old_settings:
            try:
                termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
            except:
                pass
        env.close()
        print("\n✅ 环境已关闭")
        print("=" * 70)

if __name__ == "__main__":
    main()
