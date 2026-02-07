# Gello 反向遥控操作指南

## 📋 概述

本文档说明如何实现 Gello 跟随 A1_X 机械臂移动(反向遥控)。

---

## 🎯 应用场景

### 场景 1: 自主探索时的镜像
在强化学习训练中,当机械臂自主探索时,Gello 同步跟随机械臂动作:
- **好处**: 操作者可以感知机械臂的动作
- **好处**: 随时可以人工接管控制
- **好处**: 提供触觉反馈

### 场景 2: 演示回放
回放已保存的轨迹时,Gello 重现动作:
- **好处**: 可视化验证轨迹
- **好处**: 检查关节极限
- **好处**: 调试轨迹问题

### 场景 3: 双向遥控
在正常遥控和反向遥控之间切换:
- **正常模式**: 人操作 Gello → A1_X 跟随
- **反向模式**: A1_X 运动 → Gello 跟随
- **混合模式**: 两者协作

---

## 🔧 技术原理

### Dynamixel 电机的两种模式

#### 模式 1: Torque Disabled (力矩关闭) - 正常遥控
```python
# gello/robots/dynamixel.py, line 74
self._driver.set_torque_mode(False)  # 关闭力矩
```

**特点:**
- ✅ 电机可自由移动 (back-drivable)
- ✅ 人可轻松推动 Gello
- ✅ 只读取关节角度,不控制位置
- ✅ **这是 Gello 默认的遥控模式**

#### 模式 2: Position Control (位置控制) - 反向遥控
```python
# 切换到位置控制模式
self._driver.set_operating_mode(POSITION_CONTROL_MODE)  # Mode 3
self._driver.set_torque_mode(True)  # 开启力矩
self._robot.command_joint_state(target_joints)  # 命令位置
```

**特点:**
- ✅ 电机主动移动到目标位置
- ✅ Gello 跟随 A1_X 的关节角度
- ⚠️ 人无法轻易推动 Gello (电机锁定)
- ⚠️ 需要设置关节限位保证安全

#### 模式 3: Current Control (电流控制) - 高级反向遥控
```python
# 切换到电流控制模式
self._driver.set_operating_mode(CURRENT_CONTROL_MODE)  # Mode 0
self._driver.set_torque_mode(True)
# 使用虚拟弹簧力: F = -k * (x - x_target)
```

**特点:**
- ✅ 提供柔顺控制 (compliance)
- ✅ 人可以推动 Gello (感觉像弹簧辅助)
- ✅ 更自然的力反馈
- ⚠️ 需要调参 (弹簧系数、阻尼)

---

## 📦 实现代码

### 方案 A: 位置控制 (推荐新手)

**文件**: `Gello/gello_software/gello/agents/gello_follower.py`

#### 核心类: `GelloFollower`

```python
from gello.agents.gello_follower import GelloFollower

# 初始化
follower = GelloFollower(gello_robot)

# 开启跟随模式
follower.start()  # Gello 切换到位置控制

# 控制循环
for step in range(100):
    # 读取 A1_X 关节状态
    a1x_joints = a1x_robot.get_joint_state()
    
    # Gello 跟随
    follower.command_follow(a1x_joints)
    
    time.sleep(0.02)  # 50Hz

# 停止跟随
follower.stop()  # Gello 恢复自由模式
```

#### 关键函数

| 函数 | 功能 |
|------|------|
| `start()` | 切换到位置控制模式,开启力矩 |
| `command_follow(joints)` | 命令 Gello 跟随目标关节位置 |
| `stop()` | 返回自由模式,关闭力矩 |
| `get_current_position()` | 读取 Gello 当前位置 |

---

### 方案 B: 电流控制 (高级用户)

**文件**: `Gello/gello_software/gello/agents/gello_torque_follower.py`

#### 核心类: `GelloTorqueFollower`

```python
from gello.agents.gello_torque_follower import GelloTorqueFollower

# 初始化 (可调参数)
follower = GelloTorqueFollower(
    gello_robot,
    spring_stiffness=0.5,  # 弹簧系数 (0-1)
    damping=0.1,           # 阻尼系数 (0-1)
    max_current=100.0,     # 最大电流 (mA)
)

# 使用方法同方案 A
follower.start()
# ... 控制循环 ...
follower.stop()
```

#### 虚拟弹簧原理

```python
# 位置误差
error = current_position - target_position

# 弹簧力: 误差越大,拉力越强
spring_force = -spring_stiffness * error

# 阻尼力: 速度越快,阻力越大
damping_force = -damping * velocity

# 总力矩
torque = spring_force + damping_force
```

**效果**: Gello 像被"拉"向 A1_X 位置,但人仍可推动

---

## 🎮 集成示例

### 示例 1: 基础反向遥控

**文件**: `examples/bidirectional_teleoperation.py`

```python
from examples.bidirectional_teleoperation import BidirectionalTeleoperation, TeleoperationMode

# 初始化
teleop = BidirectionalTeleoperation()
teleop.start()

# 正常遥控: Gello → A1_X
teleop.set_mode(TeleoperationMode.NORMAL)

# 反向遥控: A1_X → Gello
teleop.set_mode(TeleoperationMode.REVERSE)

# 自主探索 + Gello 跟随
teleop.set_mode(TeleoperationMode.AUTONOMOUS)

# 停止
teleop.set_mode(TeleoperationMode.STOPPED)
teleop.stop()
```

### 示例 2: 集成到 SERL 训练

```python
from experiments.a1x_pick_banana.config import TrainConfig
from gello.agents.gello_agent import GelloAgent
from gello.agents.gello_follower import GelloFollower

# 初始化环境
config = TrainConfig()
env = config.get_environment(fake_env=False)

# 初始化 Gello 跟随器
gello = GelloAgent(port="/dev/serial/by-id/usb-FTDI_...")
follower = GelloFollower(gello._robot)

# 训练循环
follower.start()  # 开启跟随模式

for episode in range(100):
    obs, _ = env.reset()
    
    for step in range(100):
        # 策略选择动作
        action = policy(obs)
        
        # 执行动作
        obs, reward, done, _, info = env.step(action)
        
        # Gello 跟随机械臂
        robot_joints = env.unwrapped.curr_joint_positions
        follower.command_follow(robot_joints)
        
        if done:
            break

follower.stop()  # 恢复自由模式
env.close()
```

---

## 🚀 快速开始

### 步骤 1: 启动 A1_X ROS2 节点

```bash
cd /home/dungeon_master/conrft/serl_robot_infra/robot_servers
python3.10 a1x_ros2_node.py
```

### 步骤 2: 运行演示脚本

```bash
cd /home/dungeon_master/conrft/Gello/gello_software/scripts
chmod +x reverse_teleoperation_demo.sh
./reverse_teleoperation_demo.sh
```

### 步骤 3: 观察效果

1. **Phase 1 (5秒)**: 正常遥控
   - 手动移动 Gello
   - A1_X 跟随 Gello 动作

2. **Phase 2 (10秒)**: 反向遥控
   - Gello 自动跟随 A1_X
   - 观察 Gello 自动移动

---

## ⚠️ 安全注意事项

### 1. 关节限位
```python
# 在 gello_follower.py 中已设置
self.joint_limits_low = np.array([-π, -π, -π, -π, -π, -π, 0.0])
self.joint_limits_high = np.array([π, π, π, π, π, π, 1.0])

# 自动裁剪到安全范围
target_joints = np.clip(target_joints, limits_low, limits_high)
```

### 2. 紧急停止
- **手动推动 Gello**: 在位置控制模式下会有阻力,但仍可强制推动
- **软件停止**: 调用 `follower.stop()` 立即关闭力矩
- **硬件急停**: 按下急停按钮切断电源

### 3. 模式切换
- ✅ **正确**: 先调用 `stop()` 再切换模式
- ❌ **错误**: 直接切换可能导致电机突然启动

```python
# 正确的切换方式
follower.stop()           # 先停止当前模式
time.sleep(0.1)           # 等待电机稳定
follower.start()          # 启动新模式
```

### 4. 初始位置对齐
```python
# 开启跟随前,先对齐位置
gello_pos = gello.get_joint_state()
a1x_pos = a1x.get_joint_state()

# 检查初始误差
error = np.abs(gello_pos - a1x_pos)
if np.any(error > 0.5):  # 超过 0.5 rad (28°)
    print("Warning: Large initial error!")
    print("Please manually align Gello to A1_X position first.")
```

---

## 🐛 常见问题

### Q1: Gello 不跟随 A1_X?

**可能原因**:
1. 未调用 `follower.start()`
2. A1_X ROS2 节点未运行
3. ZMQ 通信端口冲突

**解决方法**:
```bash
# 检查 A1_X 节点
pgrep -f "a1x_ros2_node.py"

# 检查 ZMQ 端口
netstat -an | grep 6100
```

### Q2: Gello 动作很僵硬/抖动?

**原因**: 位置控制模式 PID 参数不合适

**解决方法**:
```python
# 调整 Dynamixel PID 增益 (需要修改 driver.py)
# P 增益: 控制响应速度
# I 增益: 消除稳态误差
# D 增益: 减少超调和抖动

# 或者使用电流控制模式 (更柔顺)
follower = GelloTorqueFollower(
    gello_robot,
    spring_stiffness=0.3,  # 降低刚度
    damping=0.2,           # 增加阻尼
)
```

### Q3: 如何检测人工干预?

**方案**: 监控位置误差

```python
# 在 gello_torque_follower.py 中已实现
human_intervening = follower.detect_human_intervention(threshold=0.1)

if human_intervening:
    print("Human is overriding! Switching to manual mode...")
    teleop.set_mode(TeleoperationMode.NORMAL)
```

### Q4: 能否同时控制多个 Gello?

**可以!** 创建多个 `GelloFollower` 实例:

```python
gello_left = GelloAgent(port="/dev/ttyUSB0")
gello_right = GelloAgent(port="/dev/ttyUSB1")

follower_left = GelloFollower(gello_left._robot)
follower_right = GelloFollower(gello_right._robot)

# 同时控制
follower_left.command_follow(a1x_left_joints)
follower_right.command_follow(a1x_right_joints)
```

---

## 📊 性能对比

| 特性 | 位置控制 | 电流控制 |
|------|---------|---------|
| **实现难度** | ⭐ 简单 | ⭐⭐⭐ 复杂 |
| **跟随精度** | ⭐⭐⭐⭐⭐ 高 | ⭐⭐⭐ 中 |
| **人工干预** | ⭐⭐ 困难 | ⭐⭐⭐⭐⭐ 容易 |
| **控制频率** | 50 Hz | 100 Hz |
| **力反馈** | ❌ 无 | ✅ 有 |
| **推荐场景** | 精确演示 | 协作控制 |

---

## 📚 技术细节

### Dynamixel 控制模式

| 模式编号 | 名称 | 用途 |
|---------|------|------|
| 0 | Current Control | 力矩控制,重力补偿 |
| 1 | Velocity Control | 速度控制 |
| 3 | Position Control | 位置控制 (用于反向遥控) |
| 4 | Extended Position | 多圈位置控制 |
| 5 | Current-based Position | 位置控制 + 电流限制 |

### 控制地址映射 (Dynamixel XM430)

| 地址 | 名称 | 读/写 | 说明 |
|------|------|-------|------|
| 11 | Operating Mode | R/W | 设置控制模式 |
| 64 | Torque Enable | R/W | 开启/关闭力矩 |
| 84 | Position P Gain | R/W | 位置 P 增益 |
| 102 | Goal Current | W | 目标电流 (电流模式) |
| 116 | Goal Position | W | 目标位置 (位置模式) |
| 132 | Present Position | R | 当前位置 |

---

## 🔗 相关文件

```
Gello/gello_software/
├── gello/agents/
│   ├── gello_agent.py              # 正常遥控
│   ├── gello_follower.py           # 反向遥控 (位置控制) ✨新增✨
│   └── gello_torque_follower.py    # 反向遥控 (电流控制) ✨新增✨
├── gello/dynamixel/
│   ├── driver.py                   # Dynamixel 驱动
│   └── protocol.py                 # 通信协议
├── gello/robots/
│   └── dynamixel.py                # DynamixelRobot 类
└── scripts/
    └── reverse_teleoperation_demo.sh  # 快速演示脚本 ✨新增✨

examples/
└── bidirectional_teleoperation.py  # 双向遥控示例 ✨新增✨

serl_robot_infra/franka_env/
└── robots/
    └── a1x_robot.py                # A1_X 机器人接口
```

---

## 📝 总结

### 回答你的问题

1. **Gello 遥控时的模式**: 
   - `torque_mode(False)` - 力矩关闭
   - 电机可自由移动,只读取角度

2. **反向遥控实现**:
   - ✅ 使用 `GelloFollower` 类 (位置控制)
   - ✅ 或使用 `GelloTorqueFollower` (电流控制)
   - ✅ 已提供完整代码和演示

### 推荐使用流程

```python
# 1. 正常遥控: 人示教
teleop.set_mode(TeleoperationMode.NORMAL)
collect_demonstrations()

# 2. 训练策略
train_policy()

# 3. 自主探索 + Gello 跟随
teleop.set_mode(TeleoperationMode.AUTONOMOUS)
autonomous_exploration()

# 4. 需要时人工干预
if human_wants_to_intervene:
    teleop.set_mode(TeleoperationMode.NORMAL)
```

---

**作者**: GitHub Copilot  
**创建日期**: 2024  
**适用系统**: A1_X + Gello + SERL
