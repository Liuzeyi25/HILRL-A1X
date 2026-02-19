#!/usr/bin/env python3
"""
最安全的测试脚本 - 在导入前就完全禁用 CUDA
"""
import os
import sys

# 在任何导入之前就设置最强制的环境变量
os.environ.update({
    "JAX_PLATFORM_NAME": "cpu",
    "JAX_PLATFORMS": "cpu",
    "CUDA_VISIBLE_DEVICES": "",
    "XLA_FLAGS": "--xla_force_host_platform_device_count=1",
    "JAX_DISABLE_MOST_OPTIMIZATIONS": "1"
})

# 添加项目路径
sys.path.append('/home/wenkai001/hil-serl')

print("🟢 最安全的导入测试")
print("=" * 40)

print("环境变量设置:")
for key in ["JAX_PLATFORM_NAME", "JAX_PLATFORMS", "CUDA_VISIBLE_DEVICES", "XLA_FLAGS"]:
    print(f"  {key}: {os.environ.get(key, '未设置')}")

# 不测试 JAX 计算，只测试导入
print("\n测试导入（不执行计算）:")

try:
    print("1. 测试项目导入...")
    from serl_launcher.agents.continuous.sac import SACAgent
    print("   ✅ SACAgent 导入成功")
    
    from serl_launcher.utils.launcher import make_sac_pixel_agent
    print("   ✅ make_sac_pixel_agent 导入成功")
    
    from serl_launcher.data.data_store import MemoryEfficientReplayBufferDataStore
    print("   ✅ MemoryEfficientReplayBufferDataStore 导入成功")
    
    from examples.mappingsexperiments.mappings import CONFIG_MAPPING
    print("   ✅ CONFIG_MAPPING 导入成功")
    
    print("\n🎉 所有项目导入都成功！")
    print("\n💡 这说明你的项目代码本身是正常的")
    print("   问题只是 JAX 的 CUDA 后端初始化")
    
except Exception as e:
    print(f"   ❌ 导入失败: {e}")
    import traceback
    traceback.print_exc()

print("\n建议:")
print("- 如果上面的导入都成功，说明问题仅在 JAX CUDA 后端")
print("- 可以考虑降级 JAX 到 0.4.20 版本")
print("- 或者在生产环境中设置 JAX_PLATFORM_NAME=cpu")
