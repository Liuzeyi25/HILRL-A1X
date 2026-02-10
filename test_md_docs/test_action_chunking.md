# Action Chunking 集成说明

## ✅ 修改完成

已成功将**滚动窗口 Action Chunking** 集成到 Gello 干预系统中。

---

## 🔧 实现原理

### 1. **滚动窗口 Chunk 策略**
由于 Gello 设备只能读取"当前时刻"的位置，无法预测未来动作，我们采用**历史动作滚动窗口**方案：

**采集阶段（Gello 干预）：**
- 维护最近 4 个动作的历史缓存 `deque(maxlen=4)`
- 每次返回 `[t-3, t-2, t-1, t]` 作为 action chunk
- 这样保持了数据格式一致性，同时符合滚动窗口特性

**训练/推理阶段（RL Agent）：**
- Agent 输出真正的未来预测：`[t, t+1, t+2, t+3]`
- 执行第一个动作 `t`
- 下一步输出 `[t+1, t+2, t+3, t+4]`（标准滚动窗口）

---

## 📝 修改清单

### 1. `wrappers.py` - GelloIntervention 类

#### a) 添加参数
```python
def __init__(
    self, 
    ...
    action_chunk_size: Optional[int] = None,  # 🚀 新增
):
```

#### b) 初始化 Action History
```python
# 🚀 Action Chunking 支持
self.action_chunk_size = action_chunk_size
if self.action_chunk_size:
    from collections import deque
    self.action_history = deque(maxlen=self.action_chunk_size)
```

#### c) 修改 step() 返回值
```python
# 构建滚动窗口 chunk [t-3, t-2, t-1, t]
if self.action_chunk_size and intervene_action_eef is not None:
    self.action_history.append(intervene_action_eef)
    
    if len(self.action_history) < self.action_chunk_size:
        # 初始阶段：重复当前动作填充
        intervene_action_chunk = np.array([intervene_action_eef] * self.action_chunk_size)
    else:
        # 正常阶段：使用历史动作
        intervene_action_chunk = np.array(list(self.action_history))  # [4, 7]

# 返回 chunk 或单个动作
info["intervene_action_eef"] = intervene_action_chunk if self.action_chunk_size else intervene_action_eef
```

#### d) 修改 reset() 方法
```python
def reset(self, **kwargs):
    # 🚀 清空动作历史
    if self.action_chunk_size:
        self.action_history.clear()
    ...
```

---

### 2. `config.py` - TrainConfig 类

#### a) 设置 chunk size
```python
class TrainConfig(DefaultTrainingConfig):
    # 🚀 Action Chunking 配置
    action_chunk_size = 4  # 滚动窗口大小
```

#### b) 传递参数到 GelloIntervention
```python
env = GelloIntervention(
    env, 
    left_config_path=self.gello_config_path,
    control_rate_hz=500,
    eval_mode=eval_mode,
    action_chunk_size=self.action_chunk_size,  # 🚀 传递 chunk size
)
```

#### c) ChunkingWrapper 配置
```python
env = ChunkingWrapper(
    env, 
    obs_horizon=stack_obs_num,
    act_exec_horizon=self.action_chunk_size  # 🚀 使用 chunk size
)
```

---

## 📊 数据格式

### Gello 干预时

**返回的 action chunk：**
```python
info["intervene_action_eef"]  # Shape: [4, 7]
# [
#   [t-3 的动作],  # 历史动作
#   [t-2 的动作],
#   [t-1 的动作],
#   [t   的动作],  # 当前动作
# ]
```

### Agent 推理时

**Agent 输出：**
```python
actions  # Shape: [4, 7]
# [
#   [t   的动作],  # 当前执行
#   [t+1 的动作],  # 未来预测
#   [t+2 的动作],
#   [t+3 的动作],
# ]
```

**ChunkingWrapper 行为：**
- 接收 `[4, 7]` 的动作序列
- 逐个执行 4 个动作（每步一个）
- 收集 4 个观测返回

---

## 🚀 使用示例

### 启动训练（启用 Gello + Chunking）

```bash
# Learner
python examples/train_conrft_octo.py \
    --exp_name a1x_pick_banana \
    --learner \
    --checkpoint_path ./checkpoints/a1x_banana \
    --demo_path ./demos/banana_*.pkl

# Actor (with Gello intervention)
python examples/train_conrft_octo.py \
    --exp_name a1x_pick_banana \
    --actor \
    --ip <learner_ip> \
    --checkpoint_path ./checkpoints/a1x_banana
```

**✅ 现在 Gello 干预时会自动返回 4 个动作的滚动窗口！**

---

## 🔍 验证方法

### 1. 检查动作形状
在 `train_conrft_octo.py` 中已有打印：
```python
if step % 100 == 0:
    print_green(f"[Actor] Step {step}: actions shape = {actions.shape}")
```

**期望输出：**
```
[Actor] Step 0: actions shape = (4, 7)    # Gello 干预时
[Actor] Step 100: actions shape = (4, 7)  # Agent 输出时
```

### 2. 检查 Gello 初始化信息
```
🚀 GelloIntervention: 启用 action chunking (size=4)
✅ GelloIntervention initialized
   - Agent: GelloAgent
   - Control rate: 500 Hz
   ...
```

### 3. 检查 ChunkingWrapper
```python
env = ChunkingWrapper(env, obs_horizon=2, act_exec_horizon=4)
print(env.action_space)  # 应该是 Box(shape=(4, 7), ...)
```

---

## ⚠️ 注意事项

### 1. 初始阶段填充
在 episode 开始时，action history 还不够 4 个，会用当前动作重复填充：
```python
# 初始几步：[current, current, current, current]
# 正常阶段：[t-3, t-2, t-1, t]
```

### 2. Reset 清空历史
每次 `env.reset()` 都会清空 action history，确保新 episode 的独立性。

### 3. 兼容性
- 当 `action_chunk_size = None` 时，退化为单步模式（与原版行为相同）
- 不影响其他 wrapper 的功能

---

## 🎯 预期效果

### 训练时
- **Demo 数据**：包含滚动窗口 action chunks（历史动作序列）
- **RL 数据**：Agent 学习从历史动作预测未来动作的模式
- **BC 损失**：能学到动作的时序平滑性

### 推理时
- Agent 输出平滑的动作序列
- 减少抖动和不连续性
- 提高操作的稳定性

---

## 📚 参考

- **Diffusion Policy**: 使用类似的滚动窗口 action chunk 策略
- **ACT (Action Chunking Transformer)**: chunk_size=100 的极端例子
- **本实现**: chunk_size=4 的温和版本，平衡了实时性和稳定性

---

## ✅ 完成状态

- [x] 修改 GelloIntervention 类
- [x] 修改 TrainConfig 配置
- [x] 更新 ChunkingWrapper 调用
- [x] 兼容单步模式（chunk_size=None）
- [x] Reset 时清空历史
- [x] 文档说明

**所有修改已完成，可以直接使用！** 🎉
