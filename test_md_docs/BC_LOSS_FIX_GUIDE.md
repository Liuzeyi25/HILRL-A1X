# BC Loss 不收敛问题 - 快速诊断指南

## 🎯 核心问题

预训练时 BC Loss 初始值很大且不收敛，主要原因是：

### 1️⃣ **动作空间尺度不平衡** (最可能的原因)

你的 A1X 机器人动作空间：
```python
# 关节角度范围: [-2.88, 2.88] 弧度 (范围 ~6)
# 夹爪位置范围: [2.0, 99.0] mm    (范围 ~97)
```

**问题:** 夹爪维度的数值远大于关节角度，导致：
- MSE loss 被夹爪误差主导
- 梯度优化倾向于只优化夹爪维度
- BC loss 数值很大 (因为夹爪误差的平方可能是数百)

### 2️⃣ **Diffusion 噪声参数不匹配**

当前默认参数：
```python
sigma_max = 80.0  # ❌ 对于未归一化的动作太大!
sigma_min = 0.02
sigma_data = 0.5
```

当 `t=80` 时，加噪过程 `x_t = x_start + noise * 80` 会完全淹没原始信号。

## 🔧 快速修复方案

### 方案 A: 降低 sigma_max (最快) ⭐⭐⭐⭐

```python
# 在 examples/experiments/a1x_pick_banana/config.py 或训练脚本中添加:

# 如果使用 make_conrft_octo_cp_pixel_agent_single_arm:
agent = make_conrft_octo_cp_pixel_agent_single_arm(
    # ... 其他参数 ...
    sigma_max=5.0,  # 从 80.0 降低到 5.0
    sigma_min=0.02,
    sigma_data=0.5,
)
```

或者修改 `serl_launcher/serl_launcher/utils/launcher.py`:
```python
def make_conrft_octo_cp_pixel_agent_single_arm(
    # ...
    sigma_max: float = 5.0,  # 改这里 (原来是 80.0)
    # ...
)
```

### 方案 B: 实现动作归一化 (最彻底) ⭐⭐⭐⭐⭐

在 `examples/experiments/a1x_pick_banana/wrapper.py` 中添加归一化 wrapper:

```python
import gymnasium as gym
import numpy as np

class ActionNormalizationWrapper(gym.Wrapper):
    """将 A1X 动作归一化到 [-1, 1]"""
    
    def __init__(self, env):
        super().__init__(env)
        
        # A1X 动作空间边界 (7维: 6关节 + 1夹爪)
        self.action_min = np.array([
            -2.88, -0.001, 0.0, 1.5, 1.521, -1.56, 2.0
        ])
        self.action_max = np.array([
            2.88, 3.14, -2.95, -1.55, -1.52, 1.56, 99.0
        ])
        
        # 更新 action_space
        self.action_space = gym.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(7,),
            dtype=np.float32
        )
    
    def step(self, action):
        # 反归一化: [-1, 1] -> 原始动作空间
        action_denorm = (action + 1.0) / 2.0 * (self.action_max - self.action_min) + self.action_min
        return self.env.step(action_denorm)
    
    def reset(self, **kwargs):
        return self.env.reset(**kwargs)
```

然后在 `config.py` 的 `get_environment` 中使用:

```python
def get_environment(self, fake_env=False, save_video=False, ...):
    env = A1XTaskEnv(...)
    
    # 添加归一化 wrapper
    env = ActionNormalizationWrapper(env)
    
    # 其他 wrappers...
    env = SERLObsWrapper(env, ...)
    
    return env
```

**同时调整 diffusion 参数:**
```python
sigma_max = 3.0  # 对于归一化到 [-1,1] 的动作
sigma_min = 0.002
sigma_data = 0.5
```

## 📊 诊断脚本使用

运行诊断脚本查看你的数据统计：

```bash
cd /home/dungeon_master/conrft/examples

python diagnose_bc_loss.py \
  --demo_path=/path/to/your/demo1.pkl \
  --demo_path=/path/to/your/demo2.pkl \
  --exp_name=a1x_pick_banana
```

关键输出：
- 动作各维度的范围 (应该相近)
- 是否包含 embeddings 和 mc_returns
- 初始 BC loss 估计值

## 🎓 添加训练监控

在预训练循环的第一步添加诊断打印：

```python
# 在 train_conrft_octo.py 的 learner() 函数中
if step < FLAGS.pretrain_steps:
    print_green("Pretraining the model with demo data")
    
    # 🔍 添加诊断: 采样一个 batch 查看数据
    sample_batch = next(demo_buffer.get_iterator(
        sample_args={"batch_size": config.batch_size, "pack_obs": True},
        device=sharding.replicate(),
    ))
    
    print(f"\n🔍 第一个 batch 的统计:")
    print(f"  Actions shape: {sample_batch['actions'].shape}")
    print(f"  Actions mean per dim: {jnp.mean(sample_batch['actions'], axis=0)}")
    print(f"  Actions std per dim:  {jnp.std(sample_batch['actions'], axis=0)}")
    print(f"  Actions min per dim:  {jnp.min(sample_batch['actions'], axis=0)}")
    print(f"  Actions max per dim:  {jnp.max(sample_batch['actions'], axis=0)}")
    print(f"  Has embeddings: {'embeddings' in sample_batch}")
    print(f"  Has mc_returns: {'mc_returns' in sample_batch}")
    print()
    
    # 继续正常训练...
    for step in tqdm.tqdm(range(start_step, FLAGS.pretrain_steps + 1), ...):
        # ...
```

## ✅ 验证修复

训练时观察这些指标：

1. **初始 BC loss** 应该 < 50 (理想 < 10)
2. **BC loss 下降**: 前 100 步应该明显下降
3. **Loss 稳定性**: 不应该出现 NaN 或突然爆炸
4. **Actor loss**: 应该和 BC loss 同步下降

## 📝 推荐步骤

1. **立即尝试:** 降低 `sigma_max` 到 5.0，重新训练
2. **运行诊断:** 使用 `diagnose_bc_loss.py` 检查数据
3. **彻底修复:** 实现动作归一化 wrapper
4. **监控训练:** 添加上述诊断代码，观察 loss 曲线

## 🆘 如果问题仍未解决

检查以下可能性：

1. **Demo 数据问题:**
   - 确认包含 `embeddings` 和 `next_embeddings`
   - 检查 `mc_returns` 是否正确计算
   - 验证动作数据没有 NaN 或异常值

2. **模型架构问题:**
   - Octo 模型是否正确加载
   - Critic 网络是否正常初始化

3. **训练配置:**
   - batch_size 是否合理 (推荐 256)
   - learning_rate 是否过大
   - bc_weight 和 q_weight 比例 (默认 1.0 和 0.1)

---

**快速测试命令:**

```bash
# 1. 运行诊断
python examples/diagnose_bc_loss.py \
  --demo_path=demo_data/your_demo.pkl \
  --exp_name=a1x_pick_banana

# 2. 修改 sigma_max 后重新训练
cd examples/experiments/a1x_pick_banana
python ../../train_conrft_octo.py \
  --learner \
  --exp_name=a1x_pick_banana \
  --demo_path=/path/to/demo.pkl \
  --checkpoint_path=/path/to/checkpoints \
  --pretrain_steps=2000
```
