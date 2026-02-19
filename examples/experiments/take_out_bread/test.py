#!/usr/bin/env python3
"""诊断 cuDNN 和 JAX 配置问题"""

import os
import sys

print("=" * 70)
print("🔍 JAX & cuDNN 诊断工具")
print("=" * 70)

# 1. 检查环境变量
print("\n📋 环境变量:")
print(f"  LD_LIBRARY_PATH: {os.environ.get('LD_LIBRARY_PATH', 'Not set')}")
print(f"  XLA_FLAGS: {os.environ.get('XLA_FLAGS', 'Not set')}")
print(f"  XLA_PYTHON_CLIENT_MEM_FRACTION: {os.environ.get('XLA_PYTHON_CLIENT_MEM_FRACTION', 'Not set')}")

# 2. 检查 JAX 版本和设备
print("\n📦 JAX 配置:")
import jax
import jaxlib
print(f"  JAX version: {jax.__version__}")
print(f"  JAXlib version: {jaxlib.__version__}")
print(f"  Devices: {jax.devices()}")
print(f"  Default backend: {jax.default_backend()}")

# 3. 检查 cuDNN 库
print("\n📚 cuDNN 库:")
import nvidia.cudnn
cudnn_path = nvidia.cudnn.__path__[0]
print(f"  cuDNN package path: {cudnn_path}")

lib_path = os.path.join(cudnn_path, 'lib')
if os.path.exists(lib_path):
    libs = [f for f in os.listdir(lib_path) if f.startswith('libcudnn')]
    for lib in sorted(libs):
        lib_full_path = os.path.join(lib_path, lib)
        size_mb = os.path.getsize(lib_full_path) / (1024 * 1024)
        print(f"    ✓ {lib} ({size_mb:.1f} MB)")

# 4. 测试简单的 JAX 操作
print("\n🧪 测试简单 JAX 操作:")
import jax.numpy as jnp
try:
    x = jnp.ones((10, 10))
    result = x.sum()
    print(f"  ✓ 简单矩阵运算成功: {result}")
except Exception as e:
    print(f"  ✗ 简单矩阵运算失败: {e}")
    sys.exit(1)

# 5. 测试卷积 - 小尺寸
print("\n🧪 测试小尺寸卷积:")
from jax import random
import flax.linen as nn

class SmallConv(nn.Module):
    @nn.compact
    def __call__(self, x):
        return nn.Conv(features=64, kernel_size=(3, 3), strides=(1, 1))(x)

try:
    key = random.PRNGKey(0)
    x_small = random.normal(key, (1, 32, 32, 3))
    model = SmallConv()
    params = model.init(key, x_small)
    output = model.apply(params, x_small)
    print(f"  ✓ 小卷积成功: {output.shape}")
except Exception as e:
    print(f"  ✗ 小卷积失败: {e}")

# 6. 测试 Octo 类似的卷积配置
print("\n🧪 测试 Octo ViT 卷积配置:")

class OctoLikeConv(nn.Module):
    @nn.compact
    def __call__(self, x):
        # Octo 使用的配置: kernel_size=(14, 14), strides=(14, 14), features=768
        return nn.Conv(features=768, kernel_size=(14, 14), strides=(14, 14))(x)

try:
    key = random.PRNGKey(0)
    # 尝试不同的 batch sizes 和 timesteps
    for batch_size in [1, 2]:
        for timesteps in [1, 2]:
            shape = (batch_size, timesteps, 256, 256, 3)  # Octo 输入形状
            print(f"\n  测试形状: {shape}")
            x = random.normal(key, shape)
            
            # Reshape to (batch*timesteps, H, W, C)
            x_reshaped = x.reshape(-1, 256, 256, 3)
            print(f"  Reshaped: {x_reshaped.shape}")
            
            model = OctoLikeConv()
            params = model.init(key, x_reshaped)
            output = model.apply(params, x_reshaped)
            print(f"  ✓ Octo 卷积成功: {output.shape}")
            
except Exception as e:
    print(f"  ✗ Octo 卷积失败: {e}")
    import traceback
    traceback.print_exc()
    
    # 提供诊断建议
    print("\n" + "=" * 70)
    print("💡 诊断建议:")
    print("=" * 70)
    print("1. 尝试降级 cuDNN:")
    print("   pip install nvidia-cudnn-cu12==8.9.2.26")
    print("\n2. 添加更多 XLA 标志:")
    print("   export XLA_FLAGS=\"--xla_gpu_strict_conv_algorithm_picker=false --xla_gpu_enable_cudnn_frontend=false\"")
    print("\n3. 检查 GPU 驱动:")
    print("   nvidia-smi")
    print("\n4. 尝试清理 XLA 缓存:")
    print("   rm -rf ~/.cache/jax_cache/")
    sys.exit(1)

print("\n" + "=" * 70)
print("✅ 所有测试通过!")
print("=" * 70)