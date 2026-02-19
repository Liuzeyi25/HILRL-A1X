# GelloIntervention 快速参考

## 一分钟快速开始

```python
from serl_robot_infra.franka_env.envs.wrappers import GelloIntervention

# 1. 创建环境
env = YourEnv()

# 2. 添加 Gello 干预
env = GelloIntervention(
    env,
    left_config_path="Gello/gello_software/configs/a1x_gello.yaml",
    control_rate_hz=30
)

# 3. 运行（按空格键启用/禁用干预）
obs, _ = env.reset()
action = policy(obs)
obs, reward, done, truncated, info = env.step(action)

print(f"Gello干预: {info['gello_intervened']}")
```

## 核心概念

| 组件 | 作用 |
|------|------|
| **Agent** | 处理 Gello 设备输入，生成动作 |
| **Robot** | 机器人硬件/模拟器接口 |
| **ZMQ Server** | 暴露 Robot 接口 |
| **ZMQ Client** | 连接到 Robot |
| **RobotEnv** | 控制循环管理 |

## 参数说明

```python
GelloIntervention(
    env,                          # 基础环境
    left_config_path: str,        # 左臂 YAML 配置（必需）
    right_config_path: str = None,# 右臂 YAML 配置（双臂）
    control_rate_hz: int = 30,    # 控制频率
    use_save_interface: bool = False,  # 数据保存界面
    action_indices = None         # 动作索引过滤
)
```

## YAML 配置模板

```yaml
robot:
  _target_: gello.robots.A1_X.A1XRobot
  port: /dev/ttyUSB0
  dynamixel_config: configs/dynamixel.yaml

agent:
  _target_: gello.agents.agent.DynamixelRobotAgent
  robot_type: A1_X
  port: /dev/ttyUSB0
  start_joints: [0, -0.785, 0, -2.356, 0, 1.571, 0.785, 0.04]

hz: 30
```

## 关键方法

| 方法 | 功能 |
|------|------|
| `reset()` | 重置环境和 Gello |
| `step(action)` | 执行动作（可能被 Gello 替换）|
| `close()` | 清理资源 |

## 控制流程

```
策略输出 action
     ↓
空格键启用干预？
     ↓
   是 → Agent 获取 Gello 动作 → 返回 Gello 动作
   否 → 返回原始 action
     ↓
env.step(action)
     ↓
返回 obs, reward, done, info
info["gello_intervened"] = True/False
```

## 常见问题

### Q: 如何启用 Gello 干预？
**A:** 按空格键切换，默认关闭

### Q: 干预时机器人不动？
**A:** 检查：
1. 空格键是否按下（看终端提示）
2. YAML 配置是否正确
3. `/dev/ttyUSB*` 权限

### Q: 如何查看干预状态？
**A:** 
```python
_, _, _, _, info = env.step(action)
if info["gello_intervened"]:
    print("🎮 使用 Gello")
```

### Q: 支持双臂吗？
**A:** 支持！提供 `right_config_path` 即可

### Q: 如何调试？
**A:** 
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## 与旧版对比

| 特性 | 旧版 | 新版 |
|------|------|------|
| 配置方式 | 代码参数 | YAML 文件 |
| 干预触发 | 自动检测移动 | 空格键切换 |
| 架构 | 直接串口 | Agent + ZMQ |
| 双臂支持 | ❌ | ✅ |
| 可扩展性 | 低 | 高 |

## 重要提示

⚠️ **原有的 GelloIntervention 逻辑已完全移除**
- 不再使用 `GelloExpert` 直接控制
- 不再有 `sync_on_reset` 参数
- 不再有 `use_absolute_control` 参数
- 所有控制通过 Agent 和 YAML 配置

✅ **新架构优势**
- 基于 `launch_yaml.py` 成熟设计
- 更好的模块化和可测试性
- 支持更多 Robot 类型
- 配置驱动，易于调整
