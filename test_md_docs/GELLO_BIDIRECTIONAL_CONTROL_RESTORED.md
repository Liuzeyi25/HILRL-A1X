# Gello 双向控制功能恢复说明

## ✅ 已恢复功能

新架构现在支持与旧方法相同的**双向控制**功能：

### 🔄 双向控制流程

```
Reset 开始
   ↓
机器人移动到初始位置
   ↓
获取机器人关节状态 (A1X 坐标)
   ↓
逆映射：A1X → Gello 坐标
   ↓
Gello 进入跟随模式 (start_following)
   ↓
Gello 平滑移动到目标位置
   ↓
Gello 退出跟随模式 (stop_following)
   ↓
准备远程操控 ✅
```

---

## 🔧 新增参数

### `GelloIntervention.__init__()`

```python
env = GelloIntervention(
    env,
    left_config_path="path/to/config.yaml",
    
    # 🆕 双向控制参数
    sync_on_reset=True,           # 启用 Reset 时 Gello 跟随
    reset_follow_duration=0.5,    # 跟随持续时间（秒）
    
    # 其他参数
    always_intervene=False,       # 始终干预模式
    action_indices=None,          # 可控关节索引
)
```

### 参数说明

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `sync_on_reset` | bool | `True` | 是否在 Reset 时启用 Gello 跟随 |
| `reset_follow_duration` | float | `0.5` | 跟随持续时间（秒），自动根据距离调整 |

---

## 📝 新增方法

### 1. `_a1x_to_gello_mapping(a1x_joints)`

**功能**: 将 A1X 关节位置转换为 Gello 关节位置（逆映射）

**优先级**:
1. 使用 `agent.robot_to_leader_joints()` (推荐)
2. 使用 `robot_env._robot.robot_to_leader_joints()`
3. 使用简化的手动逆映射（fallback）

**返回**: `np.ndarray[7]` 或 `None`

---

### 2. `_start_following()` / `_stop_following()`

**功能**: 控制 Gello 跟随模式

**实现**:
- 调用 `agent.start_following()` / `agent.stop_following()`
- 或者调用 `agent._agent.start_following()` (BimanualAgent)

---

### 3. `_slow_follow_to_target(target_joints, duration)`

**功能**: 平滑移动 Gello 到目标位置

**特性**:
- 100 Hz 高频控制
- 自动根据距离调整持续时间
- 线性插值平滑运动

**距离自适应**:
```python
max_diff < 0.2  → 0.3s
max_diff < 0.5  → 0.5s
max_diff < 1.5  → 0.8s
max_diff >= 1.5 → 1.2s
```

---

### 4. `_get_robot_joint_state(obs, info)`

**功能**: 从观测或 info 中提取机器人关节状态

**尝试顺序**:
1. `info['joint_positions']`
2. `obs['state'][:7]`
3. `robot_env.get_joint_positions()`
4. `env.get_joint_positions()`
5. 递归查找包装的环境

---

### 5. `_get_current_gello_joints()` / `_set_gello_joints()`

**功能**: 获取/设置 Gello 关节位置

**用途**: 在跟随模式下控制 Gello 运动

---

## 🎯 使用示例

### 完整配置（推荐）

```python
from serl_robot_infra.franka_env.envs.wrappers import GelloIntervention

env = YourRobotEnv()

env = GelloIntervention(
    env,
    left_config_path="Gello/gello_software/configs/yam_A1_X.yaml",
    
    # 完整功能配置
    always_intervene=True,      # 始终启用干预（对齐旧方法）
    sync_on_reset=True,         # 🔧 启用双向控制
    reset_follow_duration=0.5,  # 跟随时长
    action_indices=None,        # 全关节控制
    control_rate_hz=30,         # 控制频率
)

# Reset 时 Gello 会自动跟随机器人
obs, info = env.reset()

# 预期输出：
# 🤖 Robot reset position (A1X): [0.1, 0.2, 0.3]...
# 🔄 计算 Gello 目标位置...
# 🎯 Gello 目标位置: [-0.1, 0.2, -0.3]...
# 🤖 Gello 进入跟随模式
# ⚡ 同步 Gello 到机器人位置 (用时 0.5s)...
# ✅ Gello 退出跟随模式（准备远程操控）
# ✅ Gello 已同步。准备远程操控。
```

---

## 🧪 测试方法

### 方法 1: 使用专用测试脚本

```bash
python examples/test_script/test_gello_following.py
```

**测试内容**:
- 多次 Reset 验证 Gello 跟随
- 检查日志输出
- 验证远程操控功能

---

### 方法 2: 使用 verify_action_space2.py

```bash
python examples/verify_action_space2.py
```

**观察要点**:
- Reset 时观察 Gello 物理运动
- 查看终端日志中的跟随信息
- 验证同步后的远程操控

---

## 📊 对比：旧方法 vs 新方法（恢复后）

| 功能 | 旧方法 | 新方法（恢复前） | 新方法（恢复后） |
|------|--------|----------------|----------------|
| **Reset 时 Gello 跟随** | ✅ | ❌ | ✅ |
| **逆映射** | ✅ | ❌ | ✅ |
| **跟随模式** | ✅ | ❌ | ✅ |
| **平滑运动** | ✅ | ❌ | ✅ |
| **自动距离调整** | ✅ | ❌ | ✅ |

---

## ⚠️ 注意事项

### 1. Agent 必须支持跟随模式

确保你的 Agent 实现了以下方法：
```python
agent.start_following()
agent.stop_following()
agent.get_joint_state()
agent.set_joint_positions(joints)
```

或者实现了逆映射方法：
```python
agent.robot_to_leader_joints(a1x_joints)
```

---

### 2. 逆映射准确性

**最佳实践**: 在 Agent 或 Robot 中实现精确的逆映射

**Fallback**: 当前使用简化的手动逆映射（可能不精确）：
```python
gello_joints[0] = -a1x_joints[0]  # 关节 0 取反
gello_joints[1] = a1x_joints[1]   # 关节 1 直接
# ... 等等
```

如果 Gello 跟随位置不准确，需要根据实际硬件调整逆映射。

---

### 3. 禁用双向控制

如果不需要 Gello 跟随功能：

```python
env = GelloIntervention(
    env,
    left_config_path="...",
    sync_on_reset=False,  # 禁用双向控制
)
```

此时 Reset 只会重置环境，Gello 保持原位置。

---

## 🐛 故障排查

### 问题 1: Reset 时 Gello 不动

**可能原因**:
- `sync_on_reset=False`
- Agent 不支持跟随模式
- 逆映射失败

**解决方案**:
```bash
# 检查配置
python examples/test_script/test_gello_following.py --check-config

# 查看 Reset 日志，寻找错误信息
```

---

### 问题 2: Gello 移动到错误位置

**可能原因**:
- 逆映射不准确
- 关节顺序不匹配

**解决方案**:
1. 检查 `_a1x_to_gello_mapping()` 中的映射关系
2. 在 Agent 中实现精确的逆映射方法
3. 对比实际位置和目标位置

---

### 问题 3: Gello 移动太快/太慢

**解决方案**:
调整 `reset_follow_duration` 参数：

```python
env = GelloIntervention(
    env,
    left_config_path="...",
    reset_follow_duration=1.0,  # 增加到 1 秒（更慢）
)
```

或修改 `_slow_follow_to_target()` 中的距离-时长映射。

---

## 📚 相关文件

- `serl_robot_infra/franka_env/envs/wrappers.py` - GelloIntervention 实现
- `examples/test_script/test_gello_following.py` - 双向控制测试脚本
- `examples/verify_action_space2.py` - 综合验证脚本
- `wrappers_20260125.py` - 旧方法参考实现

---

## ✅ 完整性检查清单

- [x] 添加 `sync_on_reset` 参数
- [x] 添加 `reset_follow_duration` 参数
- [x] 实现 `_a1x_to_gello_mapping()` 逆映射
- [x] 实现 `_start_following()` / `_stop_following()`
- [x] 实现 `_slow_follow_to_target()` 平滑运动
- [x] 实现 `_get_robot_joint_state()` 状态获取
- [x] 实现 `_get_current_gello_joints()` / `_set_gello_joints()`
- [x] 修改 `reset()` 方法添加跟随逻辑
- [x] 更新初始化提示信息
- [x] 创建测试脚本

---

## 🎉 总结

新架构现在已经**完全恢复**了旧方法的双向控制功能！

**主要改进**:
1. ✅ 保留 launch_yaml 架构的灵活性
2. ✅ 恢复 Reset 时的 Gello 跟随
3. ✅ 支持自动距离调整
4. ✅ 提供详细的调试日志
5. ✅ 可配置开关（`sync_on_reset`）

**下一步**: 运行测试验证功能！
