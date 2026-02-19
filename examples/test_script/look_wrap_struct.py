"""
调试脚本：检查观测的实际结构
"""

import sys
from pathlib import Path

# 添加正确的路径
sys.path.insert(0, 'Gello/gello_software')
examples_dir = Path(__file__).parent.parent
sys.path.insert(0, str(examples_dir))

def main():
    print("=" * 60)
    print("🔍 观测结构调试")
    print("=" * 60)
    
    print("\n1️⃣ 直接创建A1XTaskEnv...")
    
    from experiments.a1x_pick_banana.wrapper import A1XTaskEnv
    
    # 创建基础环境（minimal配置）
    env = A1XTaskEnv(
        fake_env=False,
        save_video=False,
    )
    
    print("✅ 环境创建成功")
    
    # 重置并检查观测
    print("\n2️⃣ 重置环境并检查观测...")
    obs, info = env.reset()
    
    print("\n📊 观测结构分析:")
    print(f"顶层键: {list(obs.keys())}")
    
    # 检查state的结构
    if 'state' in obs:
        print(f"\n'state' 类型: {type(obs['state'])}")
        
        if isinstance(obs['state'], dict):
            print(f"\n✅ 'state' 是字典，包含的键:")
            for key in obs['state'].keys():
                value = obs['state'][key]
                if hasattr(value, 'shape'):
                    print(f"  ✓ {key}: shape={value.shape}, dtype={value.dtype}")
                    # 打印一些示例值
                    if value.size <= 10:
                        print(f"      值: {value}")
                else:
                    print(f"  ✓ {key}: type={type(value)}, value={value}")
        else:
            print(f"\n⚠️  'state' 是数组 (不是字典):")
            print(f"  Shape: {obs['state'].shape}")
            print(f"  Dtype: {obs['state'].dtype}")
            if obs['state'].size <= 20:
                print(f"  值: {obs['state']}")
            else:
                print(f"  前10个值: {obs['state'].flatten()[:10]}")
    else:
        print(f"\n❌ 观测中没有 'state' 键！")
    
    # 检查其他可能的图像键
    print("\n📷 其他观测键:")
    for key in obs.keys():
        if key != 'state':
            value = obs[key]
            if hasattr(value, 'shape'):
                print(f"  ✓ {key}: shape={value.shape}, dtype={value.dtype}")
            else:
                print(f"  ✓ {key}: type={type(value)}")
    
    env.close()
    
    print("\n" + "=" * 60)
    print("💡 解决方案:")
    print("=" * 60)
    
    if 'state' in obs:
        if isinstance(obs['state'], dict):
            state_keys = list(obs['state'].keys())
            print(f"\n✅ 你的环境使用字典式state")
            print(f"   当前包含的键: {state_keys}")
            
            # 检查是否有gripper相关的键
            gripper_keys = [k for k in state_keys if 'gripper' in k.lower()]
            if gripper_keys:
                print(f"\n   找到夹爪相关的键: {gripper_keys}")
            else:
                print(f"\n   ⚠️  没有找到包含'gripper'的键")
            
            print(f"\n🔧 修复步骤:")
            print(f"   在 config.py 中设置:")
            print(f"   proprio_keys = {state_keys}")
            
        else:
            print(f"\n⚠️  你的环境使用数组式state (不是字典)")
            print(f"   State shape: {obs['state'].shape}")
            print(f"\n🔧 修复选项:")
            print(f"\n   选项1: 修改A1XEnv，让_get_obs()返回字典式state")
            print(f"   在 serl_robot_infra/franka_env/envs/a1x_env.py 中:")
            print(f"   ```python")
            print(f"   def _get_obs(self):")
            print(f"       return {{")
            print(f"           'state': {{")
            print(f"               'joint_positions': self.curr_joint_positions[:6],")
            print(f"               'joint_velocities': self.curr_joint_velocities[:6],")
            print(f"               'gripper_position': self.curr_joint_positions[6:7],")
            print(f"           }},")
            print(f"           'images': {{...}}")
            print(f"       }}")
            print(f"   ```")
            print(f"\n   选项2: 添加包装器转换数组为字典")
            print(f"   在 wrapper.py 中添加 StateArrayToDictWrapper")
    else:
        print(f"\n❌ 观测中完全没有'state'键")
        print(f"   这不正常，请检查A1XEnv的_get_obs()实现")

if __name__ == "__main__":
    main()
    