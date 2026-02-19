# 从旧版迁移到新版 GelloIntervention

## 概述

`GelloIntervention` 已从直接串口控制迁移到基于 `launch_yaml.py` 的 Agent-Robot 架构。

## 关键变化对比

| 方面 | 旧版本 | 新版本 |
|------|--------|--------|
| **初始化参数** | `port`, `sync_on_reset`, `use_absolute_control` | `left_config_path`, `control_rate_hz` |
| **干预触发** | 自动检测 Gello 移动 | 按空格键手动切换 |
| **默认状态** | 干预启用 | 干预禁用 |
| **Info 键** | `intervene_action`, `intervene_action_absolute` | `gello_intervened` (bool) |
| **坐标映射** | 手动实现 `_gello_to_a1x_mapping` | Agent 内部处理 |
| **架构** | GelloExpert → 串口 → Robot | Agent → ZMQ → Robot |

## 代码迁移示例

### 旧版代码
```python
from serl_robot_infra.franka_env.envs.wrappers import GelloIntervention

env = YourEnv()
env = GelloIntervention(
    env,
    port="/dev/ttyUSB0",
    sync_on_reset=True,
    use_absolute_control=True
)

# 旧版：自动检测干预
obs, info = env.reset()
for _ in range(1000):
    action = policy(obs)
    obs, reward, done, truncated, info = env.step(action)
    
    # 检查是否有干预
    if "intervene_action" in info:
        print("检测到干预")
        delta_action = info["intervene_action"]
        absolute_action = info.get("intervene_action_absolute")
```

### 新版代码
```python
from serl_robot_infra.franka_env.envs.wrappers import GelloIntervention

env = YourEnv()
env = GelloIntervention(
    env,
    left_config_path="Gello/gello_software/configs/a1x_config.yaml",
    control_rate_hz=30
)

# 新版：空格键手动启用干预
obs, info = env.reset()
print("按空格键启用/禁用 Gello 干预")

for _ in range(1000):
    action = policy(obs)
    obs, reward, done, truncated, info = env.step(action)
    
    # 检查干预状态
    if info.get("gello_intervened", False):
        print("Gello 干预中")
        # 注意：新版本中 Agent 内部生成动作
        # 无法直接访问 intervene_action
```

## 配置文件要求

新版本需要 YAML 配置文件，示例：

```yaml
# a1x_gello_config.yaml

robot:
  _target_: gello.robots.A1_X.A1XRobot
  port: /dev/ttyUSB0
  dynamixel_config: Gello/gello_software/configs/a1x_dynamixel.yaml

agent:
  _target_: gello.agents.agent.DynamixelRobotAgent
  robot_type: A1_X
  port: /dev/ttyUSB0
  start_joints: [0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785, 0.04]

hz: 30
start_joints: [0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785, 0.04]
```

## 测试脚本迁移

### verify_action_space.py 迁移要点

1. **移除对 `intervene_action` 的检查**
   ```python
   # 旧版
   if "intervene_action" in info:
       delta_action = info["intervene_action"]
       absolute_action = info["intervene_action_absolute"]
   
   # 新版
   if info.get("gello_intervened", False):
       print("干预中")
       # 动作已经被 Agent 处理，不需要额外获取
   ```

2. **添加空格键提示**
   ```python
   print("按空格键启用 Gello 干预")
   print("再按一次禁用")
   ```

3. **更新状态检测**
   ```python
   # 旧版：检测动作是否被替换
   if "intervene_action" in info:
       intervened = True
   
   # 新版：直接读取标志
   intervened = info.get("gello_intervened", False)
   ```

## 常见问题

### Q1: 为什么干预不工作？
**A:** 新版本默认干预是关闭的。请按空格键启用。

### Q2: 如何获取干预时的动作？
**A:** 新架构中，Agent 内部生成动作并直接传递给环境。如果需要记录动作，应该在数据采集时从环境的实际执行中获取，而不是从 info 字典。

### Q3: 旧版的 `intervene_action_absolute` 在哪里？
**A:** 新架构不在 info 中暴露这些细节。Agent 内部处理所有坐标转换。如果需要记录，建议在 Agent 或 RobotEnv 层面添加日志。

### Q4: 如何配置双臂操作？
**A:** 提供 `right_config_path` 参数：
```python
env = GelloIntervention(
    env,
    left_config_path="configs/left.yaml",
    right_config_path="configs/right.yaml"
)
```

### Q5: 旧版的 `sync_on_reset` 功能呢？
**A:** 在新架构中，同步逻辑由 Agent 内部处理。YAML 配置中的 `start_joints` 定义了重置位置。

## 优势

新架构的优势：

1. **更好的模块化** - Agent、Robot、Server 分离
2. **配置驱动** - 通过 YAML 调整，无需修改代码
3. **更易扩展** - 支持不同类型的 Agent 和 Robot
4. **更好的资源管理** - ZMQ 通信，自动清理
5. **明确的控制流** - 空格键切换，状态清晰

## 调试技巧

1. **检查配置加载**
   ```python
   from omegaconf import OmegaConf
   cfg = OmegaConf.load("config.yaml")
   print(OmegaConf.to_yaml(cfg))
   ```

2. **验证 Gello 连接**
   ```bash
   ls -l /dev/ttyUSB*
   ```

3. **测试 ZMQ 通信**
   ```python
   # 新架构会打印服务器启动信息
   # 检查 "Server ready!" 消息
   ```

4. **启用详细日志**
   ```python
   import logging
   logging.basicConfig(level=logging.DEBUG)
   ```

## 需要帮助？

参考文档：
- `GELLO_INTERVENTION_NEW.md` - 详细架构说明
- `GELLO_QUICK_REF.md` - 快速参考
- `test_new_gello_intervention.py` - 测试脚本
