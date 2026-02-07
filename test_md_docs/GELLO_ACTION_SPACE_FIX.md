# Gello动作空间修复说明

## 问题背景

在使用Gello进行遥控数据采集时，存在**动作空间不一致**的问题：

### 原始问题

```
Gello返回：绝对关节位置 [0.1, 1.2, 0.6, -0.5, 0.3, 1.5, 0.8]

↓ (GelloIntervention转换)

执行动作：delta关节位置 [2.0, 4.0, 2.0, -1.0, 0.6, 3.0, -0.5]

↓ (保存到demo)

保存动作：delta关节位置 [2.0, 4.0, 2.0, ...]  ❌ 错误！
```

**问题**：我们保存的是delta动作，但policy应该学习的是**绝对目标位置**！

### 为什么这是错误的？

1. **Delta动作依赖当前状态**：
   - Delta = [2.0, 4.0, ...] 只有在特定的当前位置才有意义
   - 如果机器人在不同位置，相同的delta会导致完全不同的结果

2. **Policy无法泛化**：
   - Policy学到的是"在位置A时，移动delta X"
   - 但不知道"目标位置是B"这个更本质的信息

3. **数学示例**：
   ```python
   # 场景1：当前在 [0.0, 1.0, 0.5, ...]
   delta = [2.0, 4.0, 2.0, ...]
   target = [0.0, 1.0, 0.5, ...] + [2.0, 4.0, 2.0, ...] * scale
         = [0.1, 1.2, 0.6, ...]  ✓ 正确目标
   
   # 场景2：当前在 [0.5, 1.5, 1.0, ...]（不同起始位置）
   delta = [2.0, 4.0, 2.0, ...]  # 相同的delta
   target = [0.5, 1.5, 1.0, ...] + [2.0, 4.0, 2.0, ...] * scale
         = [0.6, 1.7, 1.1, ...]  ✗ 完全不同的目标！
   ```

## 解决方案

### 核心思想

**分离执行动作和记录动作**：

- **执行时**：使用delta动作（A1XEnv需要）
- **记录时**：保存绝对目标位置（Policy学习）

### 实现细节

#### 1. GelloIntervention修改

```python
def action(self, action: np.ndarray) -> Tuple[np.ndarray, bool]:
    # 获取Gello绝对位置
    gello_absolute_pos, buttons = self.expert.get_action()
    
    if intervened:
        # 映射到A1X坐标系（绝对位置）
        target_a1x_pos = self._gello_to_a1x_mapping(gello_absolute_pos)
        
        # 🔑 保存绝对目标（用于记录）
        self.last_absolute_target = target_a1x_pos.copy()
        
        # 计算delta动作（用于执行）
        current_robot_pos = self._get_current_robot_position()
        delta_action = (target_a1x_pos - current_robot_pos) / action_scale
        
        return delta_action, True  # 返回delta用于执行
```

#### 2. Step方法增强

```python
def step(self, action):
    new_action, replaced = self.action(action)
    obs, rew, done, truncated, info = self.env.step(new_action)
    
    if replaced:
        # 存储delta动作（已执行的）
        info["intervene_action"] = new_action
        
        # 🔑 存储绝对目标（应该记录的）
        if hasattr(self, 'last_absolute_target') and self.last_absolute_target is not None:
            info["intervene_action_absolute"] = self.last_absolute_target
    
    return obs, rew, done, truncated, info
```

#### 3. 数据采集脚本修改

```python
# record_demos_octo_manual.py
actions = np.zeros(env.action_space.sample().shape)
next_obs, rew, done, truncated, info = env.step(actions)

# 🔑 优先使用绝对目标位置
if "intervene_action_absolute" in info:
    actions = info["intervene_action_absolute"]  # ✓ 绝对位置
elif "intervene_action" in info:
    actions = info["intervene_action"]  # 后备：delta

transition = dict(
    observations=obs,
    actions=actions,  # 保存的是绝对目标位置！
    ...
)
```

## 数据流对比

### 修复前（错误）

```
┌─────────────┐
│ Gello绝对值 │ [0.1, 1.2, 0.6, ...]
└──────┬──────┘
       │
       v
┌──────────────────────────┐
│ 转换为delta              │
└──────┬───────────────────┘
       │
       v
┌──────────────────┐
│ Delta执行        │ [2.0, 4.0, 2.0, ...]
└──────┬───────────┘
       │
       v
┌──────────────────┐
│ ❌ 保存delta     │ [2.0, 4.0, 2.0, ...]  ← 依赖当前状态
└──────────────────┘
```

### 修复后（正确）

```
┌─────────────┐
│ Gello绝对值 │ [0.1, 1.2, 0.6, ...]
└──────┬──────┘
       │
       ├────────────────────┐
       │                    │
       v                    v
┌──────────────────┐  ┌────────────────────┐
│ 转换为delta      │  │ 映射到A1X绝对值    │
│ (用于执行)       │  │ (用于记录)         │
└──────┬───────────┘  └────────┬───────────┘
       │                       │
       v                       │
┌──────────────────┐          │
│ Delta执行        │          │
│ [2.0, 4.0, ...]  │          │
└──────────────────┘          │
                              │
                              v
                        ┌────────────────────┐
                        │ ✅ 保存绝对位置    │
                        │ [0.1, 1.2, 0.6, ...]│ ← 状态无关
                        └────────────────────┘
```

## 验证方法

### 1. 打印检查

在`record_demos_octo_manual.py`中添加调试输出：

```python
if "intervene_action_absolute" in info:
    actions = info["intervene_action_absolute"]
    print(f"✓ 记录绝对动作: {actions[:3]}...")
elif "intervene_action" in info:
    actions = info["intervene_action"]
    print(f"⚠️  记录delta动作: {actions[:3]}...")
```

### 2. 数值范围检查

```python
# 绝对关节位置应该在合理范围内
# A1X joint limits approximately:
# J0: [-2.8973, 2.8973]
# J1: [0.499, 3.634]  
# J2: [-2.8973, 2.8973]
# ...
# Gripper: [-1, 1] (normalized)

# 如果看到动作值在 [-1, 1] 范围内，那是delta（错误）
# 如果看到动作值在上述关节范围内，那是绝对位置（正确）
```

### 3. 训练后测试

正确的绝对动作应该让policy：
- 学习到"到达目标位置X"而不是"移动距离Y"
- 对不同起始状态更鲁棒
- 泛化能力更强

## 重要提醒

### ⚠️ 数据兼容性

如果你之前已经收集了demos，它们保存的是**delta动作**。

**解决方案**：
1. **重新采集数据**（推荐）：使用修复后的代码采集新demos
2. **转换旧数据**：需要写脚本将delta转换为绝对位置（需要知道每步的当前状态）

### ✅ 使用场景

这个修复适用于：
- ✓ 关节空间控制（A1XEnv）
- ✓ Gello返回绝对关节位置
- ✓ 环境使用delta动作

如果你的环境已经使用绝对动作空间，则不需要这个修复。

## 技术总结

### 关键洞察

**遥控操作中的双重性**：
1. **控制空间**：机器人如何执行命令（delta or absolute）
2. **学习空间**：policy应该学什么（通常是目标，而非过程）

这两个可以不同！我们的修复就是正确分离它们。

### 数学原理

```python
# 人类遥控时：
human_intent = "我想让机器人到位置 X"  # 绝对目标

# 机器人执行时：
robot_command = "当前在 C，移动 delta = X - C"  # 相对指令

# Policy学习时：
policy_should_learn = human_intent  # 学习目标
                    ≠ robot_command  # 不是学习如何移动
```

### 代码清单

修改的文件：
1. `serl_robot_infra/franka_env/envs/wrappers.py`
   - `GelloIntervention.action()`: 添加`self.last_absolute_target`
   - `GelloIntervention.step()`: 添加`info["intervene_action_absolute"]`

2. `examples/record_demos_octo_manual.py`
   - 优先使用`intervene_action_absolute`而不是`intervene_action`

## 下一步

1. ✅ 验证修改是否生效（打印检查）
2. ⏭️ 采集新的demonstration数据
3. ⏭️ 训练policy并测试泛化能力
4. ⏭️ 对比新旧数据训练效果

---

**创建时间**: 2026-01-12  
**作者**: GitHub Copilot  
**状态**: ✅ 已实现
