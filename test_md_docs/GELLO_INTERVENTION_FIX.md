# 🐛 Gello 干预模式问题修复记录

## 📋 问题描述

**症状**:
- ✅ Reset 正常工作 (机械臂移动到位,Gello 跟随同步)
- ✅ 直接遥控测试正常 (`test_teleoperation_performance.py`)
- ❌ 数据采集时遥控失效 (机械臂持续下降,Gello 上抬无反应)

## 🔍 问题根因

### 核心问题: **坐标系不匹配**

1. **A1XEnv 期望增量动作 (delta action)**:
   ```python
   # a1x_env.py
   target_joints = self.curr_joint_positions + action * self.action_scale
   ```
   - 动作被解释为 **增量**: `Δθ = action * scale`
   - 目标位置 = 当前位置 + 增量

2. **GelloIntervention 返回绝对位置 (absolute position)**:
   ```python
   # wrappers.py (修复前)
   expert_a, buttons = self.expert.get_action()  # 返回 Gello 当前绝对位置
   return expert_a, True  # 直接传给环境
   ```
   - `GelloAgent.act()` → `robot.get_joint_state()` → **绝对位置**
   - 环境错误地将绝对位置当作增量处理!

3. **为什么测试脚本工作?**
   ```python
   # test_teleoperation_performance.py
   robot.command_joint_state(gello_pos, from_gello=True)  # 直接发送绝对位置
   ```
   - 测试脚本绕过了环境,直接调用机器人 API
   - 机器人 API 接受绝对位置,所以正常工作

### 数值示例

假设:
- 当前机械臂位置: `θ_current = [0.0, 1.0, 0.5, ...]`
- Gello 位置: `θ_gello = [0.1, 1.2, 0.6, ...]`
- 动作缩放: `scale = [0.05, 0.05, 0.05, ...]`

**修复前 (错误)**:
```python
action = [0.1, 1.2, 0.6, ...]  # Gello 绝对位置
target = [0.0, 1.0, 0.5, ...] + [0.1, 1.2, 0.6, ...] * [0.05, 0.05, 0.05, ...]
       = [0.0, 1.0, 0.5, ...] + [0.005, 0.06, 0.03, ...]
       = [0.005, 1.06, 0.53, ...]  # ❌ 完全错误!
```

**修复后 (正确)**:
```python
gello_absolute = [0.1, 1.2, 0.6, ...]
current = [0.0, 1.0, 0.5, ...]
delta = ([0.1, 1.2, 0.6, ...] - [0.0, 1.0, 0.5, ...]) / [0.05, 0.05, 0.05, ...]
      = [0.1, 0.2, 0.1, ...] / [0.05, 0.05, 0.05, ...]
      = [2.0, 4.0, 2.0, ...]  # 增量动作
target = [0.0, 1.0, 0.5, ...] + [2.0, 4.0, 2.0, ...] * [0.05, 0.05, 0.05, ...]
       = [0.0, 1.0, 0.5, ...] + [0.1, 0.2, 0.1, ...]
       = [0.1, 1.2, 0.6, ...]  # ✅ 正确!
```

## 🔧 修复方案

### 修改文件: `serl_robot_infra/franka_env/envs/wrappers.py`

#### 1. 修改 `GelloIntervention.action()` 方法

**修复内容**:
```python
def action(self, action: np.ndarray) -> Tuple[np.ndarray, bool]:
    # 1. 读取 Gello 绝对位置
    gello_absolute_pos, buttons = self.expert.get_action()
    
    # 2. 检测干预
    if intervened:
        # 3. 获取当前机械臂位置 (A1X 坐标)
        current_robot_pos = self._get_current_robot_position()
        
        # 4. 映射 Gello → A1X 坐标系
        target_a1x_pos = self._gello_to_a1x_mapping(gello_absolute_pos)
        
        # 5. 计算增量动作
        delta_action = (target_a1x_pos - current_robot_pos) / action_scale
        
        return delta_action, True
```

#### 2. 添加辅助方法

- `_get_current_robot_position()`: 从环境获取当前位置
- `_get_action_scale()`: 获取动作缩放参数
- `_gello_to_a1x_mapping()`: Gello 坐标 → A1X 坐标映射

## ✅ 验证步骤

### 方法 1: 单独测试干预模式

```bash
cd /home/dungeon_master/conrft/examples
python test_gello_intervention.py
```

**预期行为**:
1. 环境 reset,Gello 自动同步
2. 移动 Gello,机械臂应该跟随
3. 方向应该一致 (上抬→上抬,下压→下压)

### 方法 2: 运行数据采集

```bash
cd /home/dungeon_master/conrft/examples
python record_demos_octo_manual.py \
    --exp_name a1x_pick_banana \
    --successes_needed 1 \
    --manual_success
```

**预期行为**:
1. Reset 后 Gello 同步完成
2. 移动 Gello,机械臂精确跟随
3. 可以完成任务并手动标记成功

## 📊 调试信息

如果问题仍存在,检查以下输出:

1. **映射是否正确**:
   ```
   [A1X] Inverse mapping:
     A1X input:     [-0.120,  2.187, -1.147, ...]
     Gello output:  [-0.150,  1.234,  0.567, ...]
   ```

2. **当前位置获取**:
   - 应该打印 `Current A1X position: [...]`
   - 如果失败会打印警告

3. **干预检测**:
   ```
   [Step 123] 🎯 Gello 干预检测到!
      增量动作 (前3个关节): [ 0.1234,  0.5678, -0.2345]
   ```

## 🎯 技术要点

### 为什么需要逆映射?

Gello 和 A1X 的关节范围不同:

| 关节 | Gello 范围 | A1X 范围 |
|------|-----------|----------|
| Joint 1 | [-2.87, 2.87] | [-2.87, 2.89] |
| Joint 2 | [0.0, 3.14] | [0.499, 3.634] |
| Joint 3 | [0.0, 3.14] | [0.0, -2.95] |
| ... | ... | ... |
| Gripper | [0.103, 1.0] | [0, 100mm] |

映射公式:
```python
# 归一化 Gello 位置
t = (gello_pos - gello_min) / (gello_max - gello_min)

# 映射到 A1X
a1x_pos = a1x_min + t * (a1x_max - a1x_min)
```

### 为什么需要除以 action_scale?

因为环境会再乘回来:
```python
# 环境内部
target = current + action * scale

# 我们需要构造 action,使得:
# target = current + action * scale = desired_target
# 所以: action = (desired_target - current) / scale
```

## 📝 相关文件

- `serl_robot_infra/franka_env/envs/wrappers.py` - GelloIntervention wrapper
- `serl_robot_infra/franka_env/envs/a1x_env.py` - A1XEnv (增量动作处理)
- `serl_robot_infra/franka_env/robots/a1x_robot.py` - 坐标映射函数
- `serl_robot_infra/franka_env/gello/gello_expert.py` - Gello 接口

## 🔗 参考

- 测试脚本: `test_teleoperation_performance.py` (直接控制,绕过 wrapper)
- Gello 修复文档: `GELLO_FOLLOW_MODE_FIX.md` (跟随模式修复)
