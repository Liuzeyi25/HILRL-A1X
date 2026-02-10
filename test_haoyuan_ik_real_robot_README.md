# A1X 实机末端控制测试脚本

## 📝 脚本说明

`test_haoyuan_ik_copy.py` 是一个用于在 A1X 实机上测试末端执行器控制的交互式脚本。通过 ZMQ 与 ROS2 节点通信，发送实际的控制命令。

## 🚀 使用前准备

### 1. 启动 A1X ROS2 节点

确保 A1X 的 ROS2 控制节点已启动并监听 ZMQ 端口：

```bash
# 默认端口: 6100 (命令), 6101 (状态)
# 在 A1X 机器人端启动 ROS2 节点
```

### 2. 启动 CuRobo IK 服务（如果使用）

如果 ROS2 节点配置了外部 CuRobo IK，需要先启动服务：

```bash
cd /home/dungeon_master/conrft
python scripts/curobo_ik_service.py --bind tcp://0.0.0.0:6202 --pos-threshold 0.02 --rot-threshold 0.05
```

### 3. 运行测试脚本

```bash
cd /home/dungeon_master/conrft
python test_haoyuan_ik_copy.py
```

## 🧪 测试功能

### 测试 1: 查看机器人当前状态
- 查询当前关节位置（6个臂关节 + 1个夹爪）
- 查询末端执行器位置和姿态
- 查询关节速度和力矩

### 测试 2: 小幅度delta运动
- 在 X 方向移动 +2cm
- 验证位置控制精度
- 检查执行是否到达目标

### 测试 3: 圆形轨迹运动
- 在 XY 平面绘制圆形轨迹
- 半径: 3cm
- 8个路径点
- 评估轨迹跟踪性能和误差

### 测试 4: 夹爪控制
- 完全打开 (100mm)
- 完全关闭 (0mm)
- 中间位置 (50mm)
- 验证夹爪响应

### 测试 5: 运行所有测试
- 按顺序执行所有测试
- 中间有暂停确认

## 📊 Delta 控制格式

命令格式为 7D 数组：`[dx, dy, dz, drx, dry, drz, gripper]`

- `dx, dy, dz`: 位置增量 (单位: 米)
- `drx, dry, drz`: 旋转增量 (单位: 弧度, Euler angles XYZ)
- `gripper`: 夹爪绝对位置 (0-100mm)

**示例:**
```python
delta_pose = np.array([
    0.02,   # X方向 +2cm
    0.0,    # Y不变
    0.0,    # Z不变
    0.0,    # 绕X轴旋转不变
    0.0,    # 绕Y轴旋转不变
    0.0,    # 绕Z轴旋转不变
    50.0    # 夹爪保持50mm
])
```

## ⚙️ 配置选项

### 修改连接参数

在脚本中修改 `A1XRealRobotClient` 的初始化参数：

```python
client = A1XRealRobotClient(
    command_port=6100,  # 命令端口
    state_port=6101,    # 状态端口
    host="127.0.0.1"    # ROS2节点地址
)
```

### 修改运动参数

在各测试函数中可以修改：
- `timeout`: 等待超时时间
- `wait_for_completion`: 是否等待执行到位
- 运动幅度、速度等

## 🔒 安全注意事项

1. **首次运行**: 先用测试1查看当前状态，确保机器人在安全位置
2. **小幅度测试**: 先从小幅度运动开始（测试2），观察响应
3. **紧急停止**: 保持对机器人的监控，准备紧急停止
4. **工作空间**: 确保机器人周围没有障碍物
5. **Z轴限制**: ROS2节点有Z轴安全限制（>= 0.083m），避免碰撞桌面

## 📈 性能指标

根据实际测试，预期性能：
- **位置精度**: 5-8mm（CuRobo IK配置: 20mm阈值）
- **单步执行时间**: 2-3秒（4cm移动）
- **轨迹跟踪**: 圆形轨迹8点约16-24秒

## 🐛 故障排查

### 错误: 连接超时
```
❌ 获取状态超时
```
**解决方案:**
- 检查 ROS2 节点是否运行
- 确认端口号正确（6100/6101）
- 检查网络连接

### 错误: IK求解失败
```
❌ External CuRobo IK failed
```
**解决方案:**
- 检查 CuRobo IK 服务是否运行
- 确认目标位姿是否在工作空间内
- 检查关节限制

### 错误: 位置误差过大
```
⚠️ 未到达目标位置
```
**解决方案:**
- 增加 timeout 时间
- 降低 position_tolerance 阈值
- 检查机器人负载和摩擦力

## 💡 高级用法

### 自定义轨迹

可以修改测试函数创建自定义轨迹：

```python
def test_custom_trajectory():
    client = A1XRealRobotClient()
    
    # 定义路径点
    waypoints = [
        [0.02, 0.0, 0.0, 0.0, 0.0, 0.0, 50.0],   # 点1
        [0.0, 0.02, 0.0, 0.0, 0.0, 0.0, 50.0],   # 点2
        [-0.02, 0.0, 0.0, 0.0, 0.0, 0.0, 50.0],  # 点3
    ]
    
    for i, wp in enumerate(waypoints):
        print(f"执行路径点 {i+1}")
        result = client.command_eef_pose(np.array(wp))
        time.sleep(0.5)
    
    client.close()
```

### 实时轨迹跟踪

```python
# 持续delta控制模式
while True:
    # 获取目标delta（可以从遥操作设备读取）
    delta = get_target_delta()  # 用户实现
    
    result = client.command_eef_pose(
        delta,
        wait_for_completion=False,  # 不等待，实现实时控制
        timeout=0.1
    )
    
    time.sleep(0.01)  # 100Hz控制频率
```

## 📚 相关文件

- `a1_x_kenimetic_haoyuan.py` - CuRobo IK求解器
- `scripts/curobo_ik_service.py` - IK服务脚本
- `serl_robot_infra/franka_env/robots/a1x_ros2_node.py` - ROS2节点主文件
