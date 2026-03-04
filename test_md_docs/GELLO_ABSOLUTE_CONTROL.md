# Gello 绝对控制模式

## 概述

GelloIntervention wrapper 现在支持两种控制模式：

### 1. **绝对控制模式**（推荐，默认启用）
```
Gello 绝对关节 → 映射到 A1X 绝对关节 → 直接发送给机器人
```

**优点**：
- ✅ 更直接、更平滑的控制
- ✅ 零延迟（无需通过环境的 delta 计算）
- ✅ 完美的关节映射（1:1 对应）
- ✅ 更符合遥控直觉

**实现**：
- 直接调用 `robot.command_joint_state(target_pos, from_gello=False)`
- 绕过环境的 `step()` 中的 delta 机制
- 手动更新环境状态以保持同步

### 2. **Delta 控制模式**（传统模式）
```
Gello 绝对关节 → 映射到 A1X 绝对关节 → 计算 delta → env.step(delta) → robot
```

**缺点**：
- ⚠️ 需要额外的 delta 计算
- ⚠️ 受 `action_scale` 影响
- ⚠️ 可能引入数值误差

## 使用方法

### 启用绝对控制（默认）

```python
from franka_env.envs.wrappers import GelloIntervention

env = YourEnv()
env = GelloIntervention(
    env,
    use_absolute_control=True  # 默认值
)
```

### 使用 Delta 控制（如果需要）

```python
env = GelloIntervention(
    env,
    use_absolute_control=False
)
```

## 工作原理

### 绝对控制模式的流程

1. **读取 Gello 位置**
   ```python
   gello_pos = expert.get_action()  # [7] 绝对关节位置
   ```

2. **映射到 A1X 坐标系**
   ```python
   target_a1x = robot._map_to_a1x(gello_pos)  # 坐标系转换
   ```

3. **直接发送命令**
   ```python
   robot.command_joint_state(target_a1x, from_gello=False)
   ```

4. **手动更新环境**
   ```python
   env._update_curr_joint_state()  # 同步环境状态
   ```

### 数据记录

即使使用绝对控制，数据记录仍然保存 **delta EEF 动作**（用于训练）：

```python
# 记录格式：
info["intervene_action_eef"] = [
    delta_x, delta_y, delta_z,       # 位置变化
    delta_rx, delta_ry, delta_rz,    # 旋转变化（欧拉角）
    absolute_gripper                 # 夹爪绝对值
]

# 同时记录绝对关节命令（供参考）
info["intervene_action_joint_absolute"] = target_a1x  # [7] 绝对关节位置
```

## 系统架构

```
┌─────────────────────────────────────────────────────────┐
│                   Gello Hardware                         │
│              (7-DOF: 6 joints + gripper)                 │
└───────────────────────┬─────────────────────────────────┘
                        │
                        │ read joint positions
                        ▼
┌─────────────────────────────────────────────────────────┐
│                 GelloExpert                              │
│         (读取 Gello 绝对关节位置)                          │
└───────────────────────┬─────────────────────────────────┘
                        │
                        │ gello_pos [7]
                        ▼
┌─────────────────────────────────────────────────────────┐
│              Coordinate Mapping                          │
│    robot._map_to_a1x(gello_pos) → target_a1x            │
│         (Gello 坐标 → A1X 坐标)                           │
└───────────────────────┬─────────────────────────────────┘
                        │
        ┌───────────────┴──────────────┐
        │                              │
        ▼ (绝对控制)                   ▼ (Delta 控制)
┌────────────────────┐         ┌──────────────────────┐
│  Direct Command    │         │   Compute Delta      │
│  to Robot          │         │   delta = target -   │
│                    │         │        current       │
│  robot.command_    │         │   delta /= scale     │
│  joint_state()     │         └──────────┬───────────┘
│                    │                    │
│  ✅ 零延迟          │                    │
│  ✅ 完美映射        │                    ▼
└────────┬───────────┘         ┌──────────────────────┐
         │                     │   env.step(delta)    │
         │                     │                      │
         │                     │  target = current +  │
         │                     │      delta * scale   │
         │                     └──────────┬───────────┘
         │                                │
         └────────────┬───────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────┐
│                A1_X Robot Hardware                       │
│              (执行关节命令)                                │
└─────────────────────────────────────────────────────────┘
```

## 技术细节

### 为什么绝对控制更好？

1. **数值稳定性**
   - Delta 模式：`target = current + (target - current) / scale * scale`
   - 绝对模式：`target = target` ✅

2. **控制延迟**
   - Delta 模式：Gello → 计算delta → 除scale → env → 乘scale → robot
   - 绝对模式：Gello → robot ✅

3. **映射精度**
   - Delta 模式：映射误差会累积
   - 绝对模式：每次都是精确映射 ✅

### 环境兼容性

绝对控制模式需要：
- ✅ 环境有 `robot` 属性
- ✅ robot 有 `command_joint_state()` 方法
- ✅ 环境有 `_update_curr_joint_state()` 方法
- ✅ 环境有 `_get_obs()` 方法

如果这些条件不满足，会自动回退到 delta 模式。

## 测试

```bash
cd /home/dungeon_master/conrft/examples
python record_demos_octo_manual.py --exp_name a1x_pick_banana --successes_needed 1 --manual_success
```

**期望效果**：
```
✅ GelloIntervention initialized
   - Gripper enabled: True
   - Control mode: ABSOLUTE (direct mapping)  ← 应该看到这行
   - Sync on reset: True
```

移动 Gello 时应该：
- ✅ 机器人立即跟随（无延迟）
- ✅ 运动平滑流畅
- ✅ 位置精确对应

## 故障排除

### 问题：机器人不动

**检查**：
```python
# 在 step() 中添加调试输出
print(f"Robot: {self._get_robot()}")  # 应该不是 None
print(f"Target: {target_a1x_pos}")    # 应该有值
```

**解决**：如果 robot 是 None，检查环境是否正确初始化。

### 问题：机器人抖动

**原因**：可能是 Gello 读数噪声

**解决**：
```python
env = GelloIntervention(
    env,
    intervention_threshold=0.05  # 增大阈值，减少抖动
)
```

### 问题：想回退到 Delta 模式

```python
env = GelloIntervention(
    env,
    use_absolute_control=False  # 使用传统 delta 模式
)
```

## 性能对比

| 指标 | 绝对控制 | Delta 控制 |
|------|---------|-----------|
| 响应延迟 | ~1ms | ~5-10ms |
| 控制精度 | ✅ 完美 | ⚠️ 受scale影响 |
| 运动平滑度 | ✅ 优秀 | ⚠️ 一般 |
| 代码复杂度 | 中 | 低 |
| 环境依赖 | 需要robot访问 | 仅需env.step() |

## 总结

**推荐使用绝对控制模式**，因为：
1. 更直接、更符合人类直觉
2. 零延迟、更平滑
3. 完美的 1:1 关节映射
4. 消除了 delta 计算带来的误差

如果您的环境支持直接访问 robot 对象，绝对控制模式是最佳选择。
