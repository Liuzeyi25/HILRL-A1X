# Action Chunking 功能说明

## 概述

已成功实现 **Action Chunking** 功能，类似 Diffusion Policy，策略网络一次预测多个连续的 action，然后逐步执行。

## 修改内容

### 1. 配置文件 ([config.py](examples/experiments/a1x_pick_banana/config.py))

**新增配置参数：**
```python
action_chunk_size = 4  # 一次输出4个连续的动作
```

**修改 ChunkingWrapper：**
```python
env = ChunkingWrapper(
    env, 
    obs_horizon=stack_obs_num,      # 观测历史（默认1）
    act_exec_horizon=self.action_chunk_size  # 🚀 动作chunk大小（4个动作）
)
```

### 2. Agent 代码 ([conrft_single_octo_cp.py](serl_launcher/serl_launcher/agents/continuous/conrft_single_octo_cp.py))

**修改点：**

#### (1) `create_pixels` 方法
- 自动检测 action space 的维度（1D vs 2D）
- 正确计算 `action_dim`：
  - 无 chunking: `action_dim = 7`
  - 有 chunking: `action_dim = chunk_size * action_dim_per_step = 4 * 7 = 28`

#### (2) `create` 方法
- 在 config 中保存 `action_chunk_size` 和 `action_dim_per_step`
- 用于后续的 reshape 操作

#### (3) `sample_actions` 方法
- 自动 reshape 输出：
  - 无 chunking: 输出 shape `(7,)`
  - 有 chunking: 输出 shape `(4, 7)`

#### (4) `forward_critic` 和 `forward_target_critic` 方法
- 自动 flatten chunked actions：`(batch, 4, 7)` → `(batch, 28)`
- Critic 期望 1D action，需要展平处理

#### (5) `update_ql` 和 `update_calql` 方法
- 修改 shape 检查，支持 chunked actions

## Action Space 变化

| 模式 | Agent 输出 | Environment 接收 | Critic 接收 |
|------|-----------|-----------------|-------------|
| **无 Chunking** | `(7,)` | `(7,)` | `(batch, 7)` |
| **有 Chunking** | `(28,)` → reshape → `(4, 7)` | `(4, 7)` | `(batch, 4, 7)` → flatten → `(batch, 28)` |

## 工作流程

### 训练时

1. **Agent 输出**：策略网络输出 28 维向量
2. **Reshape**：`sample_actions` 自动 reshape 为 `(4, 7)`
3. **传递给 Env**：ChunkingWrapper 接收 `(4, 7)`
4. **执行**：ChunkingWrapper 依次执行 4 个动作
5. **存储**：Replay buffer 保存完整的 chunked action `(4, 7)`
6. **训练**：Critic 接收 flatten 后的 action `(28,)`

### 采样时

```python
# Actor 循环
actions, action_embeddings = agent.sample_actions(
    observations=obs,
    tasks=tasks,
    seed=rng,
)
# actions shape: (4, 7) - 4个连续动作

# 环境 step（ChunkingWrapper 自动处理）
obs, reward, done, trunc, info = env.step(actions)
```

## 优势

1. **减少累积误差**：一次预测多步，减少误差积累
2. **提高稳定性**：动作序列更平滑
3. **兼容性好**：无需修改环境，wrapper 自动处理
4. **灵活配置**：通过 `action_chunk_size` 轻松调整

## 如何使用

### 启用 Action Chunking

在 `config.py` 中设置：
```python
action_chunk_size = 4  # 或其他值（2, 8, 16等）
```

### 禁用 Action Chunking

设置为 1（或保持之前的 `act_exec_horizon=None`）：
```python
action_chunk_size = 1
# 或
env = ChunkingWrapper(env, obs_horizon=1, act_exec_horizon=None)
```

## 注意事项

1. ⚠️ **Action chunk 增大会增加计算量**：  
   - Chunk size = 4 时，action_dim = 28
   - Policy network 的输出层大小相应增加

2. ⚠️ **Replay buffer 大小不变**：  
   - 虽然每个 transition 的 action 变大了，但 transition 数量不变
   - 因为 ChunkingWrapper 只返回最后一个观测

3. ⚠️ **Episode 长度会减少**：  
   - 每次 step 执行多个动作，所以总 step 数减少

## 实现细节

### ChunkingWrapper 的行为

```python
def step(self, action):
    # action shape: (4, 7)
    for i in range(4):
        obs, reward, done, trunc, info = self.env.step(action[i])
        # 只有最后一次的 obs, reward 等会被返回
    return obs, reward, done, trunc, info
```

### 关键代码位置

- **Config**: `examples/experiments/a1x_pick_banana/config.py`
- **Agent**: `serl_launcher/serl_launcher/agents/continuous/conrft_single_octo_cp.py`
- **Wrapper**: `serl_launcher/serl_launcher/wrappers/chunking.py` (无需修改)

## 测试建议

1. **先用小的 chunk size 测试**（如 2）
2. **检查 action shape**：在训练循环中打印 `batch['actions'].shape`
3. **比较训练曲线**：对比有无 chunking 的性能差异

---

**作者**: GitHub Copilot  
**日期**: 2026年2月8日
