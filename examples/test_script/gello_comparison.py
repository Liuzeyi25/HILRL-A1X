"""
对比示例：独立控制 vs Wrapper 集成

展示 bidirectional_teleoperation.py (旧方式) 和 GelloIntervention (新方式) 的区别
"""

import numpy as np
import time


def old_way_example():
    """
    ❌ 旧方式：独立控制脚本
    
    问题：
    - 与环境分离
    - 需要单独管理控制循环
    - 无法与 wrapper 组合
    - 难以集成到训练流程
    """
    print("\n" + "="*60)
    print("❌ 旧方式：独立控制脚本")
    print("="*60)
    
    from gello.agents.gello_agent import GelloAgent
    from franka_env.robots.a1x_robot import A1XRobot
    
    # 需要单独初始化
    gello_agent = GelloAgent(port="/dev/ttyUSB0")
    a1x_robot = A1XRobot(port=6100)
    
    print("\n📝 代码结构：")
    print("""
    # 单独管理控制循环
    while True:
        gello_joints = gello_agent.act({})
        a1x_robot.update_command(gello_joints)
        time.sleep(0.02)
    
    # 问题：
    # 1. 无法与环境集成
    # 2. 无法录制演示数据
    # 3. 无法与 policy 混合使用
    # 4. 需要手动处理多线程
    """)
    
    print("\n⚠️  局限性：")
    print("  - 不能用于 record_demos.py")
    print("  - 不能与 RelativeFrame 等 wrapper 组合")
    print("  - 需要单独的脚本运行")
    print("  - 与 SpaceMouse 集成方式不一致")
    
    # 清理
    a1x_robot.close()


def new_way_example():
    """
    ✅ 新方式：Wrapper 集成
    
    优势：
    - 与环境无缝集成
    - 自动处理介入逻辑
    - 可与其他 wrapper 组合
    - 统一的接口
    """
    print("\n" + "="*60)
    print("✅ 新方式：Wrapper 集成")
    print("="*60)
    
    from franka_env.envs.franka_env import FrankaEnv, DefaultEnvConfig
    from franka_env.envs.wrappers import GelloIntervention
    from franka_env.envs.relative_env import RelativeFrame
    
    print("\n📝 代码结构：")
    print("""
    # 简单添加一个 wrapper
    env = FrankaEnv()
    env = GelloIntervention(env, port="/dev/ttyUSB0")
    env = RelativeFrame(env)  # 可以组合其他 wrapper
    
    # 正常使用
    obs, _ = env.reset()
    action = policy(obs)  # 或零动作
    obs, rew, done, _, info = env.step(action)
    
    # Gello 自动介入
    if "intervene_action_eef" in info:
        print("Gello 正在控制")
    """)
    
    print("\n✅ 优势：")
    print("  - 可直接用于 record_demos.py")
    print("  - 与 RelativeFrame 等 wrapper 无缝组合")
    print("  - 主循环代码无需修改")
    print("  - 与 SpaceMouse 集成方式完全一致")
    
    # 实际演示
    print("\n🎮 实际运行示例：")
    
    env = FrankaEnv(fake_env=True, config=DefaultEnvConfig())
    env = GelloIntervention(env, port="/dev/ttyUSB0")
    
    obs, _ = env.reset()
    
    for i in range(5):
        action = np.zeros(env.action_space.sample().shape)
        obs, rew, done, truncated, info = env.step(action)
        
        if "intervene_action_eef" in info:
            print(f"  Step {i}: ✋ Gello 介入，动作 = {info['intervene_action'][:3]}...")
        else:
            print(f"  Step {i}: 🤖 策略控制")
    
    env.close()


def comparison_table():
    """对比表格"""
    print("\n" + "="*80)
    print("📊 功能对比表")
    print("="*80)
    
    table = """
    | 功能                   | 旧方式 (独立脚本)      | 新方式 (Wrapper)      |
    |------------------------|------------------------|-----------------------|
    | 与环境集成             | ❌ 分离                | ✅ 无缝集成           |
    | 录制演示数据           | ❌ 需要单独实现        | ✅ 自动支持           |
    | 与其他 wrapper 组合    | ❌ 不支持              | ✅ 完全支持           |
    | 与 policy 混合         | ❌ 困难                | ✅ 自动处理           |
    | 代码复杂度             | ⚠️  高 (需要多线程)    | ✅ 低 (单行添加)      |
    | 与 SpaceMouse 一致性   | ❌ 完全不同            | ✅ 完全一致           |
    | 维护成本               | ⚠️  高                 | ✅ 低                 |
    | 学习曲线               | ⚠️  陡峭               | ✅ 平缓               |
    """
    print(table)


def code_comparison():
    """代码对比"""
    print("\n" + "="*80)
    print("💻 代码量对比")
    print("="*80)
    
    print("\n旧方式 (bidirectional_teleoperation.py - 约 200 行):")
    print("-" * 80)
    print("""
class BidirectionalTeleoperation:
    def __init__(self, gello_port, a1x_port, control_freq):
        self.gello_agent = GelloAgent(port=gello_port)
        self.a1x_robot = A1XRobot(port=a1x_port)
        self.control_freq = control_freq
        self.mode = TeleoperationMode.STOPPED
        self.control_thread = None
        # ... 更多初始化代码
    
    def start(self):
        self.is_running = True
        self.control_thread = threading.Thread(target=self._control_loop)
        self.control_thread.start()
    
    def _control_loop(self):
        while self.is_running:
            if self.mode == TeleoperationMode.NORMAL:
                self._normal_teleoperation()
            # ... 更多模式处理
            time.sleep(self.dt)
    
    # ... 约 200 行代码
    """)
    
    print("\n新方式 (使用 GelloIntervention - 约 3 行):")
    print("-" * 80)
    print("""
# 仅需 3 行！
env = FrankaEnv()
env = GelloIntervention(env, port="/dev/ttyUSB0")
# 完成！
    """)
    
    print("\n📈 代码减少：~98% (200行 → 3行)")


def migration_example():
    """迁移示例"""
    print("\n" + "="*80)
    print("🔄 迁移示例：从旧方式到新方式")
    print("="*80)
    
    print("\n步骤 1: 之前的配置文件")
    print("-" * 80)
    print("""
# experiments/task1_pick_banana/config.py

def get_environment(self):
    env = PickBananaEnv()
    # 无法添加 Gello，需要单独脚本
    env = SpacemouseIntervention(env)
    return env
    """)
    
    print("\n步骤 2: 迁移后的配置文件")
    print("-" * 80)
    print("""
# experiments/task1_pick_banana/config.py

from franka_env.envs.wrappers import GelloIntervention  # 新增导入

class TrainConfig:
    teleoperation_device = "gello"  # 新增配置
    
    def get_environment(self):
        env = PickBananaEnv()
        
        # 统一的集成方式
        if self.teleoperation_device == "gello":
            env = GelloIntervention(env, port="/dev/ttyUSB0")
        elif self.teleoperation_device == "spacemouse":
            env = SpacemouseIntervention(env)
        
        return env
    """)
    
    print("\n步骤 3: 使用方式")
    print("-" * 80)
    print("""
# 录制演示 (之前不支持，现在支持！)
python examples/record_demos_octo_manual.py --exp_name task1_pick_banana

# 训练 (自动支持 Gello 介入)
python examples/train_conrft_octo.py --exp_name task1_pick_banana

# 切换设备 (只需修改配置)
config.teleoperation_device = "spacemouse"  # 或 "gello" 或 None
    """)


def main():
    """运行所有对比示例"""
    print("\n" + "="*80)
    print("🎯 Gello 集成方式对比")
    print("="*80)
    
    # 对比表格
    comparison_table()
    
    # 代码量对比
    code_comparison()
    
    # 功能演示
    try:
        old_way_example()
    except Exception as e:
        print(f"\n⚠️  旧方式示例跳过（需要真实硬件）: {e}")
    
    try:
        new_way_example()
    except Exception as e:
        print(f"\n⚠️  新方式示例跳过（需要真实硬件）: {e}")
    
    # 迁移示例
    migration_example()
    
    # 总结
    print("\n" + "="*80)
    print("📝 总结")
    print("="*80)
    print("""
✅ 新方式优势：
   1. 代码量减少 98% (200行 → 3行)
   2. 与环境无缝集成
   3. 自动支持录制、训练等所有功能
   4. 与 SpaceMouse 集成方式完全一致
   5. 可与任意 wrapper 组合

❌ 旧方式问题：
   1. 代码复杂，需要手动管理线程
   2. 无法集成到现有流程
   3. 需要单独的脚本
   4. 维护成本高

💡 建议：
   - 新项目直接使用 GelloIntervention
   - 旧项目逐步迁移
   - bidirectional_teleoperation.py 可以归档或删除
    """)
    
    print("\n" + "="*80)
    print("✨ 迁移完成！Gello 现在与 SpaceMouse 地位平等！")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
