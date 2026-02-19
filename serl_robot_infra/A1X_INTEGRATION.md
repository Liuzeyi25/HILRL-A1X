# A1_X Robot Integration for SERL

本文档说明如何在 `serl_robot_infra` 框架中使用 A1_X 机械臂。

## 概述

A1_X 机械臂已集成到 serl_robot_infra 的 franka_env 中,提供与 Franka 机械臂类似的强化学习环境接口。

## 文件结构

```
serl_robot_infra/
├── franka_env/
│   ├── robots/
│   │   ├── __init__.py
│   │   ├── a1x_robot.py          # A1_X 机器人接口
│   │   └── a1x_ros2_node.py      # ROS2 通信节点
│   └── envs/
│       ├── a1x_env.py             # A1_X Gym 环境
│       └── a1x_config.py          # 配置示例
├── test_a1x_env.py                # 测试脚本
└── example_a1x_usage.py           # 使用示例
```

## 主要组件

### 1. A1XRobot 类 (`franka_env/robots/a1x_robot.py`)

通过 ZMQ 与 ROS2 节点通信的机器人接口:

- **初始化**: 自动启动 ROS2 节点进程
- **关节控制**: 发送关节位置指令 (6个关节 + 1个夹爪)
- **状态读取**: 获取当前关节位置和速度
- **范围映射**: 自动将 Gello 范围映射到 A1_X 范围

### 2. A1XEnv 类 (`franka_env/envs/a1x_env.py`)

Gymnasium 兼容的强化学习环境:

- **观察空间**: 关节位置、速度、夹爪状态、相机图像
- **动作空间**: 关节增量控制 (7维)
- **奖励**: 基于与目标关节状态的距离
- **重置**: 移动到预设的重置位置

### 3. 配置类 (`franka_env/envs/a1x_config.py`)

提供两种配置:

- `PickAndPlaceA1XConfig`: 完整配置,包含相机
- `MinimalA1XConfig`: 最小配置,用于测试

## 快速开始

### 1. 确保 ROS2 环境已安装

确保系统已安装 ROS2 Humble:

```bash
source /opt/ros/humble/setup.zsh
```

### 2. 安装依赖

```bash
pip install numpy zmq opencv-python gymnasium pynput
```

### 3. 运行测试

```bash
cd /home/dungeon_master/conrft/serl_robot_infra
python test_a1x_env.py
```

这将运行一系列测试验证环境设置正确。

### 4. 运行示例

#### 随机策略演示:

```bash
python example_a1x_usage.py
```

#### 手动控制演示:

```bash
python example_a1x_usage.py manual
```

## 使用示例

### 基础使用

```python
from franka_env.envs.a1x_env import A1XEnv
from franka_env.envs.a1x_config import MinimalA1XConfig

# 创建环境
config = MinimalA1XConfig()
env = A1XEnv(hz=10, config=config)

# 重置环境
obs, info = env.reset()

# 执行动作
action = env.action_space.sample()  # 随机动作
obs, reward, done, truncated, info = env.step(action)

# 清理
env.close()
```

### 自定义配置

```python
import numpy as np
from franka_env.envs.a1x_env import DefaultA1XEnvConfig

class MyTaskConfig(DefaultA1XEnvConfig):
    # ROS2 配置
    A1X_PORT = 6100
    A1X_PYTHON_PATH = "/usr/bin/python3"
    
    # 任务目标 (关节位置)
    TARGET_JOINT_STATE = np.array([0.0, -0.5, 0.0, -1.5, 0.0, 1.0, 50.0])
    
    # 重置位置
    RESET_JOINT_STATE = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 100.0])
    
    # 动作缩放 (每步最大变化)
    ACTION_SCALE = np.array([0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 5.0])
    
    # 相机配置
    REALSENSE_CAMERAS = {
        "wrist_1": {"serial_number": "YOUR_SERIAL", "dim": (640, 480)},
    }
```

## 与 Franka 环境的区别

### 相似之处

- 都继承 `gym.Env`
- 观察空间包含状态和图像
- 支持视频录制
- 支持相机集成

### 主要区别

| 特性 | Franka 环境 | A1_X 环境 |
|------|------------|-----------|
| 控制空间 | 笛卡尔坐标 (位置+姿态) | 关节空间 |
| 动作维度 | 7维 (x,y,z,rx,ry,rz,gripper) | 7维 (6个关节+夹爪) |
| 后端通信 | HTTP REST API | ROS2 + ZMQ |
| 夹爪控制 | 二值 (开/关) | 连续 (0-100mm) |
| 安全边界 | 笛卡尔空间限制 | 关节限制 (由映射处理) |

## 关节范围映射

A1_X 环境自动处理 Gello 关节范围到 A1_X 关节范围的映射:

```python
# Gello 范围 -> A1_X 范围 (自动映射)
Joint 1: [-2.87, 2.87]   -> [-2.870, 2.890]
Joint 2: [0.0, 3.14]     -> [0.499, 3.634]
Joint 3: [0.0, 3.14]     -> [0.0, -2.95]
Joint 4: [-1.57, 1.57]   -> [1.56, -1.55]
Joint 5: [-1.34, 1.34]   -> [1.521, -1.52]
Joint 6: [-2.0, 2.0]     -> [-1.56, 1.56]
Gripper: [0.103, 1.0]    -> [0mm, 100mm]
```

## 故障排除

### ROS2 节点启动失败

检查:
1. ROS2 是否正确安装: `ros2 --version`
2. Python 路径是否正确: 默认为 `/usr/bin/python3`
3. ZMQ 端口是否被占用: 默认 6100

### 关节状态超时

- 确保 ROS2 话题正常: `ros2 topic list`
- 检查是否发布 `/hdas/feedback_arm`
- 增加超时时间 (在 `a1x_robot.py` 中修改)

### 相机连接问题

- 检查相机序列号是否正确
- 运行 `rs-enumerate-devices` 查看连接的相机
- 尝试使用 `MinimalA1XConfig` 跳过相机

## 与 SERL 训练框架集成

A1_X 环境可以直接用于 SERL 训练:

```python
# 在训练脚本中
from franka_env.envs.a1x_env import A1XEnv
from franka_env.envs.a1x_config import PickAndPlaceA1XConfig

# 创建环境
env = A1XEnv(hz=10, config=PickAndPlaceA1XConfig())

# 使用 SERL 训练
# ... 你的 SERL 训练代码 ...
```

## 下一步

1. 根据你的任务调整 `TARGET_JOINT_STATE` 和 `RESET_JOINT_STATE`
2. 配置相机序列号
3. 调整 `ACTION_SCALE` 以控制动作幅度
4. 实现自定义奖励函数 (覆盖 `compute_reward` 方法)
5. 集成到你的强化学习训练流程

## 参考

- 原始 Gello 实现: `/home/dungeon_master/conrft/Gello/`
- Franka 环境实现: `/home/dungeon_master/conrft/serl_robot_infra/franka_env/envs/franka_env.py`
- SERL 框架: `/home/dungeon_master/conrft/serl_launcher/`
