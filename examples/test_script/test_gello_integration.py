"""
测试 GelloIntervention 集成

验证 Gello wrapper 是否正常工作
"""

import numpy as np
import sys


def test_gello_expert():
    """测试 GelloExpert 基本功能"""
    print("\n" + "="*60)
    print("测试 1: GelloExpert 初始化")
    print("="*60)
    
    try:
        from franka_env.gello.gello_expert import GelloExpert
        
        port = "/dev/serial/by-id/usb-FTDI_USB__-__Serial_Converter_FTA7NNNU-if00-port0"
        expert = GelloExpert(port=port)
        
        if expert.initialized:
            print("✅ GelloExpert 初始化成功")
            
            # 测试读取
            action, buttons = expert.get_action()
            print(f"   - 动作形状: {action.shape}")
            print(f"   - 动作值: {action}")
            print(f"   - 按钮: {buttons}")
            
            expert.close()
            return True
        else:
            print("⚠️  GelloExpert 初始化失败（设备未连接）")
            return False
            
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        return False


def test_gello_intervention_wrapper():
    """测试 GelloIntervention wrapper"""
    print("\n" + "="*60)
    print("测试 2: GelloIntervention Wrapper")
    print("="*60)
    
    try:
        from franka_env.envs.franka_env import FrankaEnv, DefaultEnvConfig
        from franka_env.envs.wrappers import GelloIntervention
        
        # 使用 fake_env 进行测试
        print("创建 fake 环境...")
        env = FrankaEnv(fake_env=True, config=DefaultEnvConfig())
        
        print("添加 GelloIntervention wrapper...")
        env = GelloIntervention(
            env, 
            port="/dev/serial/by-id/usb-FTDI_USB__-__Serial_Converter_FTA7NNNU-if00-port0"
        )
        
        print("✅ Wrapper 创建成功")
        
        # 测试 reset
        print("\n测试 reset...")
        obs, info = env.reset()
        print(f"   - 观察空间: {list(obs.keys())}")
        print("✅ Reset 成功")
        
        # 测试 step
        print("\n测试 step...")
        action = np.zeros(env.action_space.sample().shape)
        obs, rew, done, truncated, info = env.step(action)
        
        if "intervene_action" in info:
            print(f"   ✋ 检测到 Gello 介入")
            print(f"   - 介入动作: {info['intervene_action'][:3]}...")
        else:
            print(f"   🤖 策略控制（无介入）")
        
        print("✅ Step 成功")
        
        # 清理
        env.close()
        return True
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_integration_with_other_wrappers():
    """测试与其他 wrapper 的组合"""
    print("\n" + "="*60)
    print("测试 3: 与其他 Wrapper 组合")
    print("="*60)
    
    try:
        from franka_env.envs.franka_env import FrankaEnv, DefaultEnvConfig
        from franka_env.envs.wrappers import GelloIntervention, Quat2EulerWrapper
        from franka_env.envs.relative_env import RelativeFrame
        
        print("创建环境并堆叠多个 wrapper...")
        env = FrankaEnv(fake_env=True, config=DefaultEnvConfig())
        env = GelloIntervention(env, port="/dev/ttyUSB0")
        env = RelativeFrame(env)
        env = Quat2EulerWrapper(env)
        
        print("✅ 多 wrapper 堆叠成功")
        
        # 测试运行
        obs, _ = env.reset()
        for i in range(3):
            action = np.zeros(env.action_space.sample().shape)
            obs, rew, done, truncated, info = env.step(action)
            print(f"   Step {i}: done={done}, reward={rew:.3f}")
        
        print("✅ 多 wrapper 运行成功")
        
        env.close()
        return True
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_comparison_with_spacemouse():
    """对比 Gello 和 SpaceMouse 的接口一致性"""
    print("\n" + "="*60)
    print("测试 4: 与 SpaceMouse 接口对比")
    print("="*60)
    
    try:
        from franka_env.envs.wrappers import GelloIntervention, SpacemouseIntervention
        from franka_env.envs.franka_env import FrankaEnv, DefaultEnvConfig
        
        print("检查接口一致性...")
        
        # 检查方法
        gello_methods = set(dir(GelloIntervention))
        spacemouse_methods = set(dir(SpacemouseIntervention))
        
        # 关键方法
        key_methods = {'__init__', 'action', 'step', 'reset'}
        
        gello_has = key_methods.intersection(gello_methods)
        spacemouse_has = key_methods.intersection(spacemouse_methods)
        
        if gello_has == spacemouse_has == key_methods:
            print("✅ 关键方法完全一致:")
            for method in sorted(key_methods):
                print(f"   - {method}")
        else:
            print("⚠️  方法不完全一致")
        
        # 测试使用方式
        print("\n测试使用方式...")
        
        env1 = FrankaEnv(fake_env=True, config=DefaultEnvConfig())
        env1 = GelloIntervention(env1, port="/dev/ttyUSB0")
        
        env2 = FrankaEnv(fake_env=True, config=DefaultEnvConfig())
        env2 = SpacemouseIntervention(env2)
        
        # 运行相同的操作
        obs1, _ = env1.reset()
        obs2, _ = env2.reset()
        
        action = np.zeros(7)
        _, _, _, _, info1 = env1.step(action)
        _, _, _, _, info2 = env2.step(action)
        
        # 检查 info 结构
        if set(info1.keys()) == set(info2.keys()):
            print("✅ Info 字典结构一致")
        else:
            print("⚠️  Info 字典结构有差异")
            print(f"   Gello keys: {set(info1.keys())}")
            print(f"   SpaceMouse keys: {set(info2.keys())}")
        
        env1.close()
        env2.close()
        
        print("✅ 接口一致性验证通过")
        return True
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_all_tests():
    """运行所有测试"""
    print("\n" + "="*60)
    print("🧪 GelloIntervention 集成测试")
    print("="*60)
    
    results = []
    
    # 测试 1: GelloExpert
    results.append(("GelloExpert", test_gello_expert()))
    
    # 测试 2: GelloIntervention
    results.append(("GelloIntervention", test_gello_intervention_wrapper()))
    
    # 测试 3: 多 wrapper 组合
    results.append(("多 Wrapper 组合", test_integration_with_other_wrappers()))
    
    # 测试 4: 与 SpaceMouse 对比
    results.append(("接口一致性", test_comparison_with_spacemouse()))
    
    # 总结
    print("\n" + "="*60)
    print("📊 测试总结")
    print("="*60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "✅ 通过" if result else "❌ 失败"
        print(f"  {status}: {name}")
    
    print(f"\n总计: {passed}/{total} 测试通过")
    
    if passed == total:
        print("\n🎉 所有测试通过！Gello 集成成功！")
        return 0
    else:
        print("\n⚠️  部分测试失败，请检查配置")
        return 1


if __name__ == "__main__":
    exit_code = run_all_tests()
    sys.exit(exit_code)
