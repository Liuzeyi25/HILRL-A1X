"""
新版 GelloIntervention 快速测试

用于验证基于 launch_yaml.py 的新架构是否正常工作

控制方式：
- Gello 控制机器人（一直干预）
- 按 Enter 重置环境
- 按 Ctrl+C 退出
"""

import sys
sys.path.insert(0, 'Gello/gello_software')

import numpy as np
import time
import threading

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
            else:
                reset_requested = True
                print("\n🔄 收到重置请求...")
        except EOFError:
            break

def test_continuous_intervention():
    """持续干预测试 - 按 Enter 重置"""
    global reset_requested, exit_requested
    
    print("=" * 70)
    print("🧪 Gello 持续控制测试")
    print("=" * 70)
    print("\n控制方式:")
    print("   🎮 Gello 一直控制机器人")
    print("   ⏎  按 Enter 重置环境")
    print("   q  输入 q 然后 Enter 退出")
    print("=" * 70)
    
    # Import config
    exp_name = "a1x_pick_banana"
    config_module = __import__(
        f"experiments.{exp_name}.config", 
        fromlist=["TrainConfig"]
    )
    
    print("\n1. 创建环境...")
    train_config = config_module.TrainConfig()
    env = train_config.get_environment(
        fake_env=False,
        save_video=False,
        classifier=False,
    )
    print("   ✅ 环境创建成功")
    
    # 启动输入监听线程
    input_thread = threading.Thread(target=input_listener, daemon=True)
    input_thread.start()
    
    print("\n2. 重置环境...")
    obs, info = env.reset()
    print("   ✅ 重置成功")
    time.sleep(0.5)
    
    print("\n3. 开始控制循环...")
    print("   💡 现在可以用 Gello 控制机器人了！")
    print("   💡 按 Enter 重置，输入 q 退出\n")
    
    step_count = 0
    episode_count = 1
    episode_steps = 0
    
    try:
        while not exit_requested:
            # 检查是否需要重置
            if reset_requested:
                reset_requested = False
                print(f"\n{'='*50}")
                print(f"📊 Episode {episode_count} 结束，步数: {episode_steps}")
                print(f"{'='*50}")
                
                print("\n🔄 重置环境...")
                obs, info = env.reset()
                print("   ✅ 重置成功\n")
                time.sleep(0.5)
                
                episode_count += 1
                episode_steps = 0
                continue
            
            # 执行一步（空 action，干预时会被 Gello 覆盖）
            action = np.zeros(env.action_space.shape)
            obs, reward, done, truncated, info = env.step(action)
            
            step_count += 1
            episode_steps += 1
            
            # 每 50 步报告一次
            if step_count % 50 == 0:
                gello_joints = info.get("gello_joints", None)
                if gello_joints is not None:
                    print(f"   步数: {step_count:4d} | Episode: {episode_count} | "
                          f"Gello: [{gello_joints[0]:.2f}, {gello_joints[1]:.2f}, ...]")
                else:
                    print(f"   步数: {step_count:4d} | Episode: {episode_count}")
            
            # 如果环境自动结束（成功或超时）
            if done or truncated:
                success = info.get("succeed", False)
                status = "✅ 成功!" if success else "⏱️ 超时"
                print(f"\n{status} Episode {episode_count} 结束，步数: {episode_steps}")
                print("   等待 Enter 键重置...")
                
                # 等待用户按 Enter
                while not reset_requested and not exit_requested:
                    time.sleep(0.1)
            
    except KeyboardInterrupt:
        print("\n\n⚠️  收到 Ctrl+C，退出...")
    
    exit_requested = True
    
    print(f"\n{'='*70}")
    print("📊 测试结果")
    print(f"{'='*70}")
    print(f"总步数: {step_count}")
    print(f"总 Episode 数: {episode_count}")
    
    env.close()
    print(f"\n✅ 测试完成")


def test_basic_intervention():
    """测试基本干预功能"""
    print("=" * 70)
    print("🧪 新版 GelloIntervention 基础功能测试")
    print("=" * 70)
    
    # Import config
    exp_name = "a1x_pick_banana"
    config_module = __import__(
        f"experiments.{exp_name}.config", 
        fromlist=["TrainConfig"]
    )
    
    print("\n1. 创建环境...")
    train_config = config_module.TrainConfig()
    env = train_config.get_environment(
        fake_env=False,
        save_video=False,
        classifier=False,
    )
    print("   ✅ 环境创建成功")
    
    print("\n2. 重置环境...")
    obs, info = env.reset()
    print("   ✅ 重置成功")
    time.sleep(1)
    
    print("\n3. 测试干预切换...")
    print("   💡 提示：新架构默认干预是 ❌ 关闭的")
    print("   💡 按 空格键 启用干预，再按一次禁用")
    print("   💡 按 Ctrl+C 退出测试\n")
    
    step_count = 0
    intervention_history = []
    
    try:
        while step_count < 200:  # 测试 200 步
            action = np.zeros(env.action_space.shape)
            obs, reward, done, truncated, info = env.step(action)
            step_count += 1
            
            is_intervened = info.get("gello_intervened", False)
            intervention_history.append(is_intervened)
            
            # 每 20 步报告一次
            if step_count % 20 == 0:
                recent_intervention_rate = sum(intervention_history[-20:]) / 20 * 100
                status = "🟢 干预中" if is_intervened else "⚪ Policy"
                print(f"   步数: {step_count:3d} | {status} | 最近干预率: {recent_intervention_rate:5.1f}%")
            
            if done or truncated:
                obs, info = env.reset()
                time.sleep(1)
            
            time.sleep(0.05)
            
    except KeyboardInterrupt:
        pass
    
    print(f"\n{'='*70}")
    print("📊 测试结果")
    print(f"{'='*70}")
    print(f"总步数: {step_count}")
    print(f"干预次数: {sum(intervention_history)}")
    if step_count > 0:
        print(f"总干预率: {sum(intervention_history)/step_count*100:.1f}%")
    
    # 分析干预模式
    if sum(intervention_history) > 0:
        print(f"\n✅ 检测到干预！新架构工作正常")
        print(f"   - 空格键切换功能有效")
        print(f"   - 干预状态正确记录到 info['gello_intervened']")
    else:
        print(f"\n⚠️  未检测到干预")
        print(f"   - 请确保按了空格键启用干预")
        print(f"   - 请确保 Gello 设备已连接")
        print(f"   - 请确保 YAML 配置正确")
    
    env.close()
    print(f"\n✅ 测试完成")


def test_config_loading():
    """测试配置加载"""
    print("\n" + "=" * 70)
    print("🔧 配置加载测试")
    print("=" * 70)
    
    try:
        from omegaconf import OmegaConf
        import os
        
        # 检查常见配置路径
        config_paths = [
            "Gello/gello_software/configs/",
            "examples/experiments/a1x_pick_banana/",
        ]
        
        print("\n查找 YAML 配置文件...")
        for path in config_paths:
            if os.path.exists(path):
                yaml_files = [f for f in os.listdir(path) if f.endswith('.yaml')]
                if yaml_files:
                    print(f"\n✅ 找到配置目录: {path}")
                    print(f"   文件: {yaml_files[:5]}")  # 显示前5个
        
        print("\n💡 新架构需要:")
        print("   1. YAML 配置文件（包含 robot 和 agent 配置）")
        print("   2. 配置路径传递给 GelloIntervention")
        print("   3. 示例: left_config_path='path/to/config.yaml'")
        
    except Exception as e:
        print(f"⚠️  配置检查失败: {e}")


def print_architecture_summary():
    """打印架构摘要"""
    print("\n" + "=" * 70)
    print("📚 新架构 vs 旧架构")
    print("=" * 70)
    
    print("\n旧架构（已废弃）:")
    print("  • 直接串口通信 (port='/dev/ttyUSB0')")
    print("  • GelloExpert 直接读取关节位置")
    print("  • 自动检测移动触发干预")
    print("  • 手动 A1X ↔ Gello 坐标映射")
    
    print("\n新架构（当前）:")
    print("  • YAML 配置文件 (left_config_path='...')")
    print("  • Agent-Robot-ZMQ 通信架构")
    print("  • 空格键手动切换干预")
    print("  • Agent 内部处理坐标映射")
    
    print("\n关键变化:")
    print("  1. 初始化参数：port → left_config_path")
    print("  2. 干预触发：自动检测 → 空格键切换")
    print("  3. Info 键名：'intervene_action' → 'gello_intervened'")
    print("  4. 默认状态：启用 → 禁用")


if __name__ == "__main__":
    print_architecture_summary()
    print("\n" + "=" * 70)
    print("选择测试模式:")
    print("  1. 持续控制测试（推荐）- Gello 一直控制，Enter 重置")
    print("  2. 基础功能测试 - 空格键切换干预")
    print("  q. 退出")
    
    choice = input("\n请选择 (1/2/q): ").strip()
    
    if choice == '1':
        test_continuous_intervention()
    elif choice == '2':
        test_config_loading()
        print("\n" + "=" * 70)
        input("按 Enter 开始环境测试...")
        test_basic_intervention()
    elif choice.lower() == 'q':
        print("退出")
    else:
        print("默认选择 1: 持续控制测试")
        test_continuous_intervention()
