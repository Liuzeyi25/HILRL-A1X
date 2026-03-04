"""
验证Gello动作空间和双线程干预模式是否正确工作

这个脚本会：
1. 创建环境（包含GelloIntervention wrapper）
2. 移动Gello进行遥控
3. 检查info中是否有 intervene_action 和 threaded_mode
4. 验证观测完整性（包含图像）
5. 统计控制频率

控制方式：
- Gello 一直控制机器人
- 按 Enter 重置环境
- 按 Ctrl+C 退出
"""

import sys
sys.path.insert(0, 'Gello/gello_software')

import numpy as np
from pathlib import Path
import threading
import time

# 全局变量：用于检测 Enter 键
reset_requested = False
exit_requested = False

def input_listener():
    """监听键盘输入的线程"""
    global reset_requested, exit_requested
    while not exit_requested:
        try:
            user_input = input()
            if user_input.lower() == 'q':
                exit_requested = True
                print("\n🛑 收到退出请求...")
            else:
                reset_requested = True
                print("\n🔄 收到重置请求...")
        except EOFError:
            break

# Import experiment config
exp_name = "a1x_pick_banana"
config_module = __import__(
    f"experiments.{exp_name}.config", 
    fromlist=["TrainConfig"]
)

def main():
    global reset_requested, exit_requested
    
    print("=" * 60)
    print("🔍 Gello 干预模式验证测试")
    print("=" * 60)
    print("\n控制方式:")
    print("   🎮 Gello 一直控制机器人")
    print("   ⏎  按 Enter 重置环境")
    print("   q  输入 q 然后 Enter 退出")
    print("=" * 60)
    
    # Create environment with all wrappers
    print("\n1️⃣ 创建环境...")
    train_config = config_module.TrainConfig()
    env = train_config.get_environment(
        fake_env=False,
        save_video=False,
        classifier=False,
        
    )
    print("✅ 环境创建成功")
    
    # 启动输入监听线程
    input_thread = threading.Thread(target=input_listener, daemon=True)
    input_thread.start()
    
    # Reset environment
    print("\n2️⃣ 重置环境...")
    obs, info = env.reset()
    print("✅ 环境已重置")
    print("   等待Gello同步完成...")
    
    time.sleep(2)
    
    print("\n3️⃣ 开始测试遥控...")
    print("   请移动 Gello 以触发干预")
    print("   按 Enter 重置，输入 q 退出\n")
    
    step_count = 0
    intervened_count = 0
    episode_count = 1
    threaded_mode_detected = False
    
    # 🚀 新增：频率统计
    step_times = []
    last_step_time = time.time()
    
    try:
        while not exit_requested:
            # 检查是否需要重置
            if reset_requested:
                reset_requested = False
                print(f"\n{'='*60}")
                print(f"📊 Episode {episode_count} 结束")
                print(f"   步数: {step_count}, 干预次数: {intervened_count}")
                if len(step_times) > 0:
                    avg_hz = 1.0 / (sum(step_times) / len(step_times))
                    print(f"   平均 step 频率: {avg_hz:.1f} Hz")
                print(f"{'='*60}")
                
                print("\n🔄 重置环境...")
                obs, info = env.reset()
                print("✅ 重置成功，等待Gello同步...")
                time.sleep(1.5)
                print("   继续控制...\n")
                
                episode_count += 1
                step_times = []
                last_step_time = time.time()
                continue
                
            # Zero action (will be replaced by Gello if moved)
            action = np.zeros(env.action_space.shape)
            
            # Step environment
            step_start = time.time()
            next_obs, reward, done, truncated, info = env.step(action)
            step_end = time.time()
            
            step_count += 1
            step_times.append(step_end - step_start)
            
            # 🚀 新增：检查双线程模式
            if info.get("threaded_mode", False) and not threaded_mode_detected:
                threaded_mode_detected = True
                print(f"\n✅ 检测到双线程模式已启用!")
                print(f"   后台线程高频控制，主线程正常采集数据")
            
            # Check for intervention
            if "intervene_action_eef" in info:
                intervened_count += 1
                
                # 只打印前3次的详细信息
                if intervened_count <= 3:
                    print(f"\n{'='*60}")
                    print(f"🎯 检测到 Gello 干预 (第 {intervened_count} 次)")
                    print(f"{'='*60}")
                    
                    # Get action
                    intervene_action = info["intervene_action_eef"]
                    print(f"\n📊 干预动作信息:")
                    if intervene_action is not None:
                        print(f"   Shape: {intervene_action.shape}")
                        print(f"   范围: [{intervene_action.min():.3f}, {intervene_action.max():.3f}]")
                        print(f"   Gripper: {intervene_action[6]:.3f}")
                    else:
                        print(f"   ⚠️ intervene_action 为 None!")
                    
                    # 🚀 新增：检查观测完整性
                    print(f"\n📷 观测完整性检查:")
                    if "images" in next_obs:
                        for cam_name, img in next_obs["images"].items():
                            print(f"   {cam_name}: shape={img.shape}, dtype={img.dtype}")
                    else:
                        print(f"   ⚠️ 观测中没有 images 字段!")
                    
                    if "state" in next_obs:
                        print(f"   state keys: {list(next_obs['state'].keys())}")
                    
                    # 显示模式信息
                    print(f"\n🔧 模式信息:")
                    print(f"   threaded_mode: {info.get('threaded_mode', False)}")
                    print(f"   gello_intervened: {info.get('gello_intervened', False)}")
                    
                    if intervened_count == 3:
                        print(f"\n   (后续干预信息将静默)")
            
            # Progress indicator (每100步打印一次)
            if step_count % 100 == 0:
                avg_hz = 1.0 / (sum(step_times[-100:]) / len(step_times[-100:])) if step_times else 0
                print(f"  步数: {step_count}, 干预: {intervened_count}, step频率: {avg_hz:.1f} Hz", end='\r')
            
            if done or truncated:
                success = info.get("succeed", False)
                status = "✅ 成功!" if success else "⏱️ 达到最大步数"
                print(f"\n\n{status}")
                print(f"   Episode {episode_count} 结束，等待按 Enter 重置...")
            else:
                obs = next_obs
            
    except KeyboardInterrupt:
        pass
    
    exit_requested = True
    
    print(f"\n\n{'='*60}")
    print(f"🛑 测试停止")
    print(f"{'='*60}")
    print(f"总步数: {step_count}")
    print(f"总干预次数: {intervened_count}")
    print(f"总Episode数: {episode_count}")
    if step_count > 0:
        print(f"干预率: {intervened_count/step_count*100:.1f}%")
    if len(step_times) > 0:
        avg_hz = 1.0 / (sum(step_times) / len(step_times))
        print(f"平均 step 频率: {avg_hz:.1f} Hz")
    print(f"双线程模式: {'✅ 已启用' if threaded_mode_detected else '❌ 未检测到'}")
        
    env.close()
    print("\n✅ 环境已关闭")

if __name__ == "__main__":
    main()
