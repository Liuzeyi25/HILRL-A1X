# A1_X 机械臂集成总结

## 已完成的工作

### 1. 文件创建

已将 Gello 中的 A1_X 机械臂成功集成到 `serl_robot_infra/franka_env` 中,创建了以下文件:

#### 机器人接口层
- `franka_env/robots/__init__.py` - 模块初始化
- `franka_env/robots/a1x_robot.py` - A1_X 机器人类,通过 ZMQ 与 ROS2 通信
- `franka_env/robots/a1x_ros2_node.py` - ROS2 节点,处理与机器人的实际通信

#### 环境层
- `franka_env/envs/a1x_env.py` - Gymnasium 兼容的 A1_X 强化学习环境
- `franka_env/envs/a1x_config.py` - 配置类示例 (包含完整和最小配置)

#### 测试和示例
- `test_a1x_env.py` - 完整的测试套件
- `example_a1x_usage.py` - 使用示例 (随机策略和手动控制)
- `A1X_INTEGRATION.md` - 详细的集成文档

## 架构设计

### 层次结构

```
应用层 (SERL训练/测试脚本)
    ↓
环境层 (A1XEnv - Gymnasium接口)
    ↓
机器人层 (A1XRobot - 关节控制)
    ↓
通信层 (ZMQ Bridge)
    ↓
ROS2层 (A1XRobotZMQNode)
    ↓
实际硬件 (A1_X 机械臂)
```

### 主要特性

1. **关节空间控制**: 直接控制 7 个自由度 (6个关节 + 1个夹爪)
2. **自动范围映射**: 从 Gello 范围自动映射到 A1_X 范围
3. **ZMQ通信**: 高效的进程间通信
4. **Gymnasium接口**: 标准强化学习环境接口
5. **相机集成**: 支持 RealSense 相机
6. **视频录制**: 支持保存演示视频

## 与原 Franka 环境的对比

| 特性 | Franka 环境 | A1_X 环境 |
|------|------------|-----------|
| 控制模式 | 笛卡尔空间 (xyz + rpy) | 关节空间 (7 个关节) |
| 通信方式 | HTTP REST API | ROS2 + ZMQ |
| 夹爪控制 | 二值 (开/关) | 连续 (0-100mm) |
| 动作空间 | 7D (位置+姿态+夹爪) | 7D (关节增量) |
| 状态反馈 | TCP位姿、力/力矩 | 关节位置、速度 |

## 使用方法

### 最小示例

```python
from franka_env.envs.a1x_env import A1XEnv
from franka_env.envs.a1x_config import MinimalA1XConfig

# 创建环境
env = A1XEnv(hz=10, config=MinimalA1XConfig())

# 使用环境
obs, _ = env.reset()
action = env.action_space.sample()
obs, reward, done, truncated, info = env.step(action)

# 清理
env.close()
```

### 运行测试

```bash
cd /home/dungeon_master/conrft/serl_robot_infra
python test_a1x_env.py
```

### 运行示例

```bash
# 随机策略
python example_a1x_usage.py

# 手动控制
python example_a1x_usage.py manual
```

## 配置要点

### 关键配置参数

1. **A1X_PORT**: ZMQ 通信端口 (默认 6100)
2. **A1X_PYTHON_PATH**: 系统 Python 路径 (默认 /usr/bin/python3)
3. **TARGET_JOINT_STATE**: 目标关节配置
4. **RESET_JOINT_STATE**: 重置关节配置
5. **ACTION_SCALE**: 动作缩放系数

### 自定义任务

继承 `DefaultA1XEnvConfig` 并修改:
- `TARGET_JOINT_STATE` - 设置任务目标
- `REWARD_THRESHOLD` - 调整成功判断阈值
- `ACTION_SCALE` - 控制动作幅度
- `REALSENSE_CAMERAS` - 配置相机

## 下一步工作

### 可选改进

1. **添加正向运动学**: 计算末端执行器位置 (当前为占位符)
2. **安全限制**: 添加关节限制检查
3. **平滑控制**: 实现轨迹平滑
4. **多相机支持**: 扩展相机配置
5. **碰撞检测**: 集成碰撞避免

### 集成到训练流程

参考 `serl_launcher` 中的训练脚本,将 `A1XEnv` 替换 `FrankaEnv`:

```python
# 在训练脚本中
from franka_env.envs.a1x_env import A1XEnv
from franka_env.envs.a1x_config import PickAndPlaceA1XConfig

env = A1XEnv(hz=10, config=PickAndPlaceA1XConfig())
# ... SERL 训练代码 ...
```

## 技术细节

### ZMQ 通信模式

使用 REQ-REP 模式:
- 客户端 (A1XRobot) 发送请求
- 服务器 (ROS2 节点) 回复响应
- 自动重连和错误恢复

### 关节映射

线性映射公式:
```
out = out_start + (in - in_start) * (out_end - out_start) / (in_end - in_start)
```

包含:
- 范围裁剪
- 反向范围处理
- 除零保护

### 夹爪平滑

使用指数移动平均 (EMA):
```
filtered = alpha * new + (1 - alpha) * old
```
- alpha = 0.01 (可调整)
- 减少夹爪抖动

## 故障排除

### 常见问题

1. **ROS2 节点启动失败**
   - 检查 ROS2 环境: `source /opt/ros/humble/setup.zsh`
   - 验证 Python 路径

2. **ZMQ 连接超时**
   - 检查端口占用: `lsof -i :6100`
   - 增加超时时间

3. **关节状态无数据**
   - 验证 ROS2 话题: `ros2 topic list`
   - 检查 `/hdas/feedback_arm` 是否发布

## 文档

详细文档请参阅:
- `A1X_INTEGRATION.md` - 完整集成指南
- `test_a1x_env.py` - 测试示例
- `example_a1x_usage.py` - 使用示例

## 依赖

- Python 3.8+
- ROS2 Humble
- numpy
- zmq (pyzmq)
- opencv-python
- gymnasium
- pynput (用于键盘监听)

## 许可

遵循原项目许可证。
