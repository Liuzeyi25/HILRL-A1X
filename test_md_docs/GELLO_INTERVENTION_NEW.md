# GelloIntervention 新架构说明

## 概述

`GelloIntervention` 类已经被重构，采用了 `launch_yaml.py` 的架构设计。原有的直接串口控制逻辑已被替换为基于 YAML 配置的 Agent-Robot 架构。

## 主要变化

### 旧架构（已移除）
```python
# 旧的方式 - 直接使用串口和 GelloExpert
env = GelloIntervention(
    env,
    port="/dev/ttyUSB0",
    sync_on_reset=True,
    use_absolute_control=True
)
```

### 新架构（当前实现）
```python
# 新的方式 - 使用 YAML 配置文件
env = GelloIntervention(
    env,
    left_config_path="Gello/gello_software/configs/your_left_config.yaml",
    right_config_path="Gello/gello_software/configs/your_right_config.yaml",  # 可选，双臂操作
    control_rate_hz=30,
    use_save_interface=False
)
```

## 核心特性

### 1. **Agent-based 控制**
- 使用 Gello Agent（如 `DynamixelRobotAgent`）处理遥控逻辑
- Agent 从 YAML 配置文件实例化
- 支持单臂和双臂操作

### 2. **ZMQ 服务器/客户端架构**
- Robot 通过 ZMQ 服务器暴露接口
- Wrapper 使用 ZMQ 客户端与 Robot 通信
- 支持模拟器（MujocoRobotServer）和真实硬件

### 3. **RobotEnv 集成**
- 内部创建 `RobotEnv` 实例进行控制循环
- 自动处理频率控制和状态同步
- 支持起始位置自动移动

### 4. **空格键干预切换**
- 按空格键启用/禁用 Gello 干预
- 默认干预关闭，需要手动启用
- 实时显示干预状态

## 配置文件示例

创建一个 YAML 配置文件（例如 `a1x_gello_config.yaml`）：

```yaml
# Robot configuration
robot:
  _target_: gello.robots.A1_X.A1XRobot
  port: /dev/ttyUSB0
  dynamixel_config: Gello/gello_software/configs/a1x_dynamixel.yaml

# Agent configuration  
agent:
  _target_: gello.agents.agent.DynamixelRobotAgent
  robot_type: A1_X
  port: /dev/ttyUSB0
  start_joints: [0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785, 0.04]

# Control settings
hz: 30
start_joints: [0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785, 0.04]
```

## 使用方法

### 基本用法

```python
from serl_robot_infra.franka_env.envs.wrappers import GelloIntervention

# 创建基础环境
env = YourRobotEnv()

# 添加 Gello 干预
env = GelloIntervention(
    env,
    left_config_path="path/to/gello_config.yaml",
    control_rate_hz=30
)

# 运行环境
obs, info = env.reset()

for _ in range(1000):
    action = policy(obs)  # 策略动作
    
    # 如果按下空格键启用干预，action 会被 Gello 动作替代
    obs, reward, done, truncated, info = env.step(action)
    
    if info["gello_intervened"]:
        print("使用了 Gello 遥控")
    
    if done:
        obs, info = env.reset()

env.close()
```

### 双臂操作

```python
env = GelloIntervention(
    env,
    left_config_path="configs/left_arm.yaml",
    right_config_path="configs/right_arm.yaml",
    control_rate_hz=30
)
```

## 架构对比

### launch_yaml.py 的控制流程
```
YAML Config → Agent → ZMQ Server → Robot → ZMQ Client → RobotEnv → run_control_loop()
```

### GelloIntervention 的控制流程
```
YAML Config → Agent → ZMQ Server → Robot → ZMQ Client → RobotEnv
                                                          ↓
Base Env ← GelloIntervention.step() ← policy/gello_action
```

## 主要方法

### `__init__()`
- 加载 YAML 配置
- 实例化 Agent 和 Robot
- 启动 ZMQ 服务器
- 创建 RobotEnv 控制器
- 初始化键盘监听

### `action(action) -> (action, intervened)`
- 检查干预状态
- 如果启用，从 Agent 获取 Gello 动作
- 否则返回原始策略动作

### `step(action)`
- 调用 `action()` 处理干预
- 执行动作到基础环境
- 同步 robot_env 状态
- 返回观测和奖励

### `reset()`
- 重置基础环境
- 重置 robot_env 状态
- Agent 内部处理同步逻辑

### `close()`
- 停止键盘监听
- 清理 ZMQ 服务器和线程
- 关闭所有资源

## 优势

1. **模块化**: Agent、Robot、Server 分离，易于测试和维护
2. **可配置**: 通过 YAML 文件配置，无需修改代码
3. **可扩展**: 支持不同类型的 Agent 和 Robot
4. **双向控制**: Robot → Gello 同步由 Agent 内部处理
5. **资源管理**: 自动清理 ZMQ 连接和线程

## 迁移指南

如果你有使用旧版 `GelloIntervention` 的代码：

1. **创建 YAML 配置文件**（参考上面的示例）
2. **更新初始化代码**：
   ```python
   # 旧版
   env = GelloIntervention(env, port="/dev/ttyUSB0")
   
   # 新版
   env = GelloIntervention(env, left_config_path="config.yaml")
   ```
3. **移除对 `expert` 的直接访问**（Agent 内部处理）
4. **使用空格键控制干预**（而不是自动检测移动）

## 注意事项

- 确保 `Gello/gello_software` 在 Python 路径中
- YAML 配置文件必须包含 `robot` 和 `agent` 部分
- 串口设备需要有读写权限
- ZMQ 端口不能冲突（默认 5556 和 6001）
- 按空格键切换干预状态

## 调试技巧

如果遇到问题：

1. **检查 YAML 配置是否正确**
   ```bash
   python -c "from omegaconf import OmegaConf; print(OmegaConf.load('config.yaml'))"
   ```

2. **验证 Gello 设备连接**
   ```bash
   ls -l /dev/ttyUSB*
   ```

3. **检查 ZMQ 端口是否被占用**
   ```bash
   netstat -tuln | grep 5556
   ```

4. **启用详细日志**
   ```python
   import logging
   logging.basicConfig(level=logging.DEBUG)
   ```

## 相关文件

- `Gello/gello_software/experiments/launch_yaml.py` - 原始架构实现
- `Gello/gello_software/gello/agents/agent.py` - Agent 实现
- `Gello/gello_software/gello/robots/` - Robot 实现
- `Gello/gello_software/gello/env.py` - RobotEnv 实现
- `Gello/gello_software/gello/zmq_core/` - ZMQ 通信层
