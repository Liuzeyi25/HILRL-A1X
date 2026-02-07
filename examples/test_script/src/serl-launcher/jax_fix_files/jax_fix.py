#!/usr/bin/env python3
"""
检查并修复 JAX 版本问题
"""
import subprocess
import sys

def run_command(cmd):
    """运行命令并返回结果"""
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, check=True)
        return True, result.stdout.strip()
    except subprocess.CalledProcessError as e:
        return False, e.stderr.strip()

def check_jax_versions():
    """检查当前 JAX 相关包的版本"""
    print("🔍 检查当前 JAX 相关包的版本...")
    
    packages = ['jax', 'jaxlib', 'jax-cuda12-plugin', 'flax', 'optax']
    
    for pkg in packages:
        success, output = run_command(f"pip show {pkg}")
        if success:
            lines = output.split('\n')
            version_line = [line for line in lines if line.startswith('Version:')]
            if version_line:
                version = version_line[0].split(': ')[1]
                print(f"✅ {pkg}: {version}")
            else:
                print(f"✅ {pkg}: 已安装（版本未知）")
        else:
            print(f"❌ {pkg}: 未安装")

def check_cuda_environment():
    """检查 CUDA 环境"""
    print("\n🔍 检查 CUDA 环境...")
    
    # 检查 nvidia-smi
    success, output = run_command("nvidia-smi --query-gpu=name,driver_version,cuda_version --format=csv,noheader")
    if success:
        print("✅ NVIDIA GPU 信息:")
        for line in output.split('\n'):
            if line.strip():
                print(f"   {line}")
    else:
        print("❌ 无法获取 NVIDIA GPU 信息")
    
    # 检查 CUDA toolkit
    success, output = run_command("nvcc --version")
    if success:
        version_line = [line for line in output.split('\n') if 'release' in line.lower()]
        if version_line:
            print(f"✅ CUDA Toolkit: {version_line[0].strip()}")
    else:
        print("❌ CUDA Toolkit 未安装或不可用")

def suggest_fix():
    """建议修复方案"""
    print("\n🔧 基于诊断结果的修复建议:")
    print("=" * 50)
    
    print("问题分析:")
    print("- JAX 0.6.1 与 jax-cuda12-plugin 0.4.35 不兼容")
    print("- 即使强制 CPU 模式，JAX 在创建数组时仍会尝试初始化 CUDA 后端")
    print("- 这导致了段错误")
    
    print("\n解决方案 1: 降级到兼容的 JAX 版本 (推荐)")
    print("pip uninstall jax jaxlib jax-cuda12-plugin -y")
    print("pip install jax==0.4.20 jaxlib==0.4.20")
    
    print("\n解决方案 2: 升级到最新兼容版本")
    print("pip uninstall jax jaxlib jax-cuda12-plugin -y")
    print("pip install --upgrade jax[cuda12] -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html")
    
    print("\n解决方案 3: 纯 CPU 版本（最安全）")
    print("pip uninstall jax jaxlib jax-cuda12-plugin -y")
    print("pip install jax[cpu]")
    
    print("\n临时测试方案:")
    print("export JAX_PLATFORMS=cpu")
    print("export CUDA_VISIBLE_DEVICES=''")
    print("export XLA_FLAGS='--xla_force_host_platform_device_count=1'")

def create_safe_test():
    """创建一个最安全的测试脚本"""
    print("\n📝 创建最安全的测试脚本...")
    
    safe_script = '''#!/usr/bin/env python3
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
print("\\n测试导入（不执行计算）:")

try:
    print("1. 测试项目导入...")
    from serl_launcher.agents.continuous.sac import SACAgent
    print("   ✅ SACAgent 导入成功")
    
    from serl_launcher.utils.launcher import make_sac_pixel_agent
    print("   ✅ make_sac_pixel_agent 导入成功")
    
    from serl_launcher.data.data_store import MemoryEfficientReplayBufferDataStore
    print("   ✅ MemoryEfficientReplayBufferDataStore 导入成功")
    
    from experiments.mappings import CONFIG_MAPPING
    print("   ✅ CONFIG_MAPPING 导入成功")
    
    print("\\n🎉 所有项目导入都成功！")
    print("\\n💡 这说明你的项目代码本身是正常的")
    print("   问题只是 JAX 的 CUDA 后端初始化")
    
except Exception as e:
    print(f"   ❌ 导入失败: {e}")
    import traceback
    traceback.print_exc()

print("\\n建议:")
print("- 如果上面的导入都成功，说明问题仅在 JAX CUDA 后端")
print("- 可以考虑降级 JAX 到 0.4.20 版本")
print("- 或者在生产环境中设置 JAX_PLATFORM_NAME=cpu")
'''
    
    with open('/home/wenkai001/hil-serl/ultra_safe_test.py', 'w') as f:
        f.write(safe_script)
    
    print("✅ 超安全测试脚本已创建: /home/wenkai001/hil-serl/ultra_safe_test.py")

def main():
    """主函数"""
    print("🚀 JAX 版本诊断和修复工具")
    print("=" * 50)
    
    # 检查版本
    check_jax_versions()
    
    # 检查 CUDA 环境
    check_cuda_environment()
    
    # 建议修复方案
    suggest_fix()
    
    # 创建安全测试
    create_safe_test()
    
    print("\n🎯 下一步:")
    print("1. 先运行: python /home/wenkai001/hil-serl/ultra_safe_test.py")
    print("2. 如果项目导入成功，则选择一个 JAX 版本修复方案")
    print("3. 推荐使用解决方案 1（降级到 JAX 0.4.20）")

if __name__ == "__main__":
    main()