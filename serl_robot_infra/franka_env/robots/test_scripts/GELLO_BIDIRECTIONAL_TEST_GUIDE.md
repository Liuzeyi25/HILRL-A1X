# Gello 双向控制测试指南

## 概述

`test_gello_bidirectional.py` 测试脚本实现了 Gello 和 A1_X 机器人之间的双向控制：

1. **Follow 模式**：Gello 跟随机器人移动到目标位置
2. **Teleoperation 模式**：机器人跟随 Gello 手动操作

## 功能特性

### 🤖 Follow 模式
- Gello 自动移动到机器人当前位置
- 使用 A1X → Gello 逆映射转换关节空间
- 平滑插值移动（3秒缓慢跟随）
- 力矩控制模式，Gello 电机主动移动

### 🎮 Teleoperation 模式
- 手动移动 Gello，机器人实时跟随
- 使用 Gello → A1X 正向映射转换
- 20Hz 控制频率，低延迟响应
- 自由拖动模式，Gello 电机关闭力矩

### ⌨️ 按键切换
- **[SPACE]**：在两种模式间切换
- **[Q]**：退出测试

## 使用方法

### 1. 准备工作

```bash
# 确保机器人和 Gello 都已连接
# A1_X 机器人 → ZMQ 端口 6100
# Gello 设备 → USB 串口
```

### 2. 运行测试

```bash
cd /home/dungeon_master/conrft/serl_robot_infra/franka_env/robots
python test_gello_bidirectional.py
```

### 3. 测试流程

```
启动测试
   ↓
确认安全 (y/n)
   ↓
初始化机器人和 Gello
   ↓
[Follow 模式] Gello 自动移动到机器人位置
   ↓
按 [SPACE] 切换
   ↓
[Teleoperation 模式] 手动移动 Gello 控制机器人
   ↓
按 [SPACE] 切换回 Follow 模式
   ↓
按 [Q] 退出
```

## 技术实现

### 关节空间映射

#### A1X → Gello (逆映射)
```python
def a1x_to_gello_mapping(self, a1x_joints: np.ndarray) -> np.ndarray:
    """使用 robot._map_from_a1x() 进行逆映射"""
    return self.robot._map_from_a1x(a1x_joints)
```

**映射规则**：
- A1_X 关节范围 → Gello 关节范围
- 使用线性逆变换
- 处理反向关节和偏移

#### Gello → A1X (正向映射)
```python
def gello_to_a1x_mapping(self, gello_joints: np.ndarray) -> np.ndarray:
    """使用 robot._map_to_a1x() 进行正向映射"""
    return self.robot._map_to_a1x(gello_joints)
```

### Follow 模式实现

```python
def slow_follow_to_target(self, target_gello_joints, duration=3.0):
    """
    平滑插值移动 Gello 到目标位置
    """
    # 1. 获取当前 Gello 位置
    current_pos = self.get_gello_joint_state()
    
    # 2. 启用力矩控制
    self.gello._robot.set_torque_mode(True)
    
    # 3. 平滑插值（ease-in-out cubic）
    for step in range(num_steps):
        t = step / num_steps
        t_smooth = 3 * t**2 - 2 * t**3  # 平滑曲线
        
        interpolated = current + t_smooth * (target - current)
        self.gello._robot.command_joint_state(interpolated)
        
        time.sleep(dt)
    
    # 4. 关闭力矩（准备手动操作）
    self.gello._robot.set_torque_mode(False)
```

**平滑函数**：
- Ease-in-out cubic: `t_smooth = 3t² - 2t³`
- 起始和结束时速度为0，中间加速
- 避免突然启动和停止

### Teleoperation 模式实现

```python
def test_teleoperation_mode(self):
    """
    机器人跟随 Gello 移动
    """
    # 关闭 Gello 力矩，允许手动拖动
    self.gello._robot.set_torque_mode(False)
    
    # 20Hz 控制循环
    rate = 20  # Hz
    while self.running and self.mode == "teleoperation":
        # 读取 Gello 位置
        gello_pos = self.get_gello_joint_state()
        
        # 转换到 A1X 空间
        a1x_target = self.gello_to_a1x_mapping(gello_pos)
        
        # 命令机器人（非阻塞）
        self.robot.command_joint_state(a1x_target, from_gello=False)
        
        time.sleep(1/rate)
```

**关键点**：
- `from_gello=False`：使用 A1X 关节空间（已映射）
- 非阻塞命令：立即返回，低延迟
- 20Hz 控制频率：平衡响应速度和计算负载

## 配置参数

### 默认配置

```python
# Robot
robot_port = 6100

# Gello
gello_port = "/dev/serial/by-id/usb-FTDI_USB__-__Serial_Converter_FT7WBG6A-if00-port0"

# Follow mode
follow_duration = 3.0  # 秒
follow_rate = 20       # Hz

# Teleoperation mode
teleop_rate = 20       # Hz
```

### 调整参数

#### 加快/减慢 Follow 速度
```python
# 更快（1秒）
self.slow_follow_to_target(target, duration=1.0)

# 更慢（5秒）
self.slow_follow_to_target(target, duration=5.0)
```

#### 调整 Teleoperation 频率
```python
# 更高频率（更平滑，更高 CPU）
rate = 50  # Hz

# 更低频率（更省资源）
rate = 10  # Hz
```

## 故障排查

### 问题 1：Gello 不移动（Follow 模式）

**症状**：执行 Follow 时 Gello 无反应

**可能原因**：
- 力矩未启用
- 串口连接问题
- 目标位置超出范围

**解决方案**：
```python
# 检查力矩状态
self.gello._robot.set_torque_mode(True)

# 检查串口
ls -l /dev/serial/by-id/usb-FTDI*

# 检查关节范围
print(f"Target: {target_gello_joints}")
print(f"Range: -π to π")
```

### 问题 2：机器人不跟随（Teleoperation 模式）

**症状**：移动 Gello 时机器人不响应

**可能原因**：
- 映射函数失败
- 机器人命令阻塞
- 关节范围映射错误

**解决方案**：
```python
# 测试映射
gello_pos = self.get_gello_joint_state()
print(f"Gello: {gello_pos}")

a1x_target = self.gello_to_a1x_mapping(gello_pos)
print(f"A1X: {a1x_target}")

# 确保使用非阻塞命令
self.robot.command_joint_state(a1x_target, from_gello=False)
```

### 问题 3：映射不准确

**症状**：Gello 和机器人位置不匹配

**可能原因**：
- 映射参数不正确
- 关节零点偏移
- Gello 校准问题

**解决方案**：
```python
# 检查 A1XRobot 中的映射参数
# 在 a1x_robot.py 中：
gello_range_start = [-2.87, 0.0, 0.0, -1.57, -1.34, -2.0, 0.103]
gello_range_end   = [2.87, 3.14, 3.14, 1.57, 1.34, 2.0, 1.0]

a1x_range_start = [-2.870, 0.499, 0.0, 1.56, 1.521, -1.56, 0.0]
a1x_range_end   = [2.890, 3.634, -2.95, -1.55, -1.52, 1.56, 100.0]

# 根据实际情况调整
```

### 问题 4：按键不响应

**症状**：按 [SPACE] 或 [Q] 无效

**可能原因**：
- 输入缓冲区问题
- 键盘线程未启动

**解决方案**：
```bash
# 确保在终端中按 Enter 后输入
[SPACE]
[Enter]

# 或直接输入字符
s
[Enter]

q
[Enter]
```

## 测试示例输出

```
======================================================================
Gello Bidirectional Control Test
======================================================================

🤖 Initializing A1_X robot...
Starting ROS2 node subprocess...
[INFO] A1XRobotZMQNode initialized on port 6100
✅ Robot initialized

🎮 Initializing Gello...
✅ Gello initialized

======================================================================
Interactive Bidirectional Control Test
======================================================================

🤖 Starting in FOLLOW mode

======================================================================
Test 1: Gello Follows Robot
======================================================================

📍 Current robot joint state (A1_X):
   [-0.123  0.456  0.789  1.234 -0.567  0.891  50.0]

🔄 Moving Gello to match robot position...

📍 Current Gello position:
   [0.234  1.567 -0.890  0.123  0.456 -0.789  0.456]
🎯 Target Gello position:
   [0.345  1.678 -0.901  0.234  0.567 -0.890  0.567]
⏱️  Moving in 60 steps over 3.0s...
  [██████████████████████████████] 100%
  ✅ Gello reached target position

✅ Test 1 complete: Gello now matches robot position

======================================================================
Keyboard Controls:
======================================================================
  [SPACE] - Switch between Follow and Teleoperation mode
  [Q]     - Quit test
======================================================================

💡 Press [SPACE] to switch to TELEOPERATION mode

[按 SPACE + Enter]

🎮 Switched to TELEOPERATION mode
   → Move Gello to control the robot

======================================================================
Test 2: Robot Follows Gello (Teleoperation)
======================================================================

🎮 Teleoperation mode active
   → Move Gello manually to control the robot
   → Press [SPACE] to switch back to Follow mode
   → Press [Q] to quit

[手动移动 Gello，机器人跟随...]

[按 Q + Enter]

⏹️  Quit requested
⏸️  Teleoperation mode stopped

🏁 Test completed

🧹 Cleaning up...
✅ Gello torque disabled
✅ Robot connection closed
✅ Cleanup complete

👋 Goodbye!
```

## 扩展功能

### 添加位置显示
```python
def test_teleoperation_mode(self):
    # ... 现有代码 ...
    
    last_print = time.time()
    while self.running and self.mode == "teleoperation":
        gello_pos = self.get_gello_joint_state()
        a1x_target = self.gello_to_a1x_mapping(gello_pos)
        self.robot.command_joint_state(a1x_target, from_gello=False)
        
        # 每秒打印一次位置
        if time.time() - last_print > 1.0:
            print(f"Gello: {gello_pos[:3]}")
            print(f"Robot: {a1x_target[:3]}")
            last_print = time.time()
        
        time.sleep(dt)
```

### 添加误差监控
```python
def monitor_tracking_error(self):
    """监控机器人跟随误差"""
    gello_pos = self.get_gello_joint_state()
    gello_in_a1x = self.gello_to_a1x_mapping(gello_pos)
    
    robot_pos = self.get_robot_joint_state()
    
    error = np.linalg.norm(robot_pos - gello_in_a1x)
    print(f"Tracking error: {error*1000:.2f}mm")
```

### 添加录制功能
```python
def record_trajectory(self, duration=10.0):
    """录制 Gello 操作轨迹"""
    trajectory = []
    rate = 20
    dt = 1.0 / rate
    
    start = time.time()
    while time.time() - start < duration:
        gello_pos = self.get_gello_joint_state()
        a1x_pos = self.gello_to_a1x_mapping(gello_pos)
        
        trajectory.append({
            "time": time.time() - start,
            "gello": gello_pos,
            "a1x": a1x_pos
        })
        
        time.sleep(dt)
    
    return trajectory
```

## 相关文档

- `a1x_robot.py` - A1_X 机器人接口和映射函数
- `wrappers.py` - GelloIntervention wrapper 实现
- `EEF_CHUNK_COMMAND_GUIDE.md` - EEF chunk 命令使用指南

---

**作者**: AI Assistant  
**最后更新**: 2026-01-11  
**状态**: 测试就绪 ✅
