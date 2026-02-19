# 测试脚本更新 - 使用客户端 IK 求解

## ✅ 修改完成

`test_haoyuan_ik_copy.py` 已更新，现在使用：

### 新的工作流程

```
用户输入 Delta 位姿
    ↓
获取当前状态（关节+末端位姿）
    ↓
计算目标末端位姿（当前 + Delta）
    ↓
【客户端】使用 A1Kinematics (CuRobo) 求解 IK
    ↓
获得目标关节角度
    ↓
发送 command_joint_state（直接关节命令）✅
    ↓
机器人执行
```

### 关键改变

**之前（不工作）:**
- 发送 `command_eef_pose` → ROS2节点做IK → 发布关节命令
- 问题可能在 ROS2 节点的 IK 或其他环节

**现在（应该工作）:**
- 【客户端】使用 `A1Kinematics` 做 IK → 发送 `command_joint_state`
- 完全绕过 ROS2 节点的 IK 环节
- 使用与正常代码相同的关节命令方式

## 🚀 使用方法

```bash
cd /home/dungeon_master/conrft
python test_haoyuan_ik_copy.py
```

选择测试 2（小幅度运动）：
- 会显示 IK 求解过程
- 显示目标关节
- 发送关节命令
- **机器人应该能动了！** 🎉

## 📊 预期输出

```
初始化 CuRobo IK 求解器...
✓ IK 求解器初始化完成
✓ 已连接到 A1X ROS2 节点: 127.0.0.1:6100/6101

获取初始状态...
初始末端位置: [ 0.2647 -0.0106  0.1078]

发送delta命令: X+2.0cm

当前末端: [ 0.2647 -0.0106  0.1078]
目标末端: [ 0.2847 -0.0106  0.1078]
位置增量: [ 2.00  0.00  0.00] cm

求解 IK...
假设输入是[x,y,z,w]，转换为[w,x,y,z]
🌟 IK 求解耗时: 15.23 ms
✓ IK 求解成功
  目标关节: [-0.063  1.729 -0.707  0.285 -0.135 -0.045]

发送关节命令...
等待关节到达目标...
✓ 到达目标 (误差: 0.0095 rad)

查询最终状态...
  初始位置: [ 0.2647 -0.0106  0.1078] m
  最终位置: [ 0.2845 -0.0108  0.1076] m
  实际移动: [ 1.98 -0.02 -0.02] cm
  位置误差: 0.3 mm
```

## 🔍 优势

1. **IK 在客户端** - 可以看到详细的求解过程和调试信息
2. **使用经过验证的方式** - `command_joint_state` 是正常代码使用的方式
3. **灵活控制** - 可以调整 IK 参数（seeds、阈值等）
4. **易于调试** - 出问题能立即看到是 IK 还是命令发送

## 💡 说明

### 为什么这样能工作？

1. **IK 求解**在客户端完成，使用 `a1_x_kenimetic_haoyuan.py`
2. **关节命令**直接发送，就像 Gello 遥操作那样
3. **绕过**了 ROS2 节点中可能有问题的 `publish_eef_command` 流程

### 与 ROS2 节点的 command_eef_pose 的区别

| 方面 | ROS2节点 command_eef_pose | 客户端 IK + command_joint_state |
|------|--------------------------|--------------------------------|
| IK求解位置 | ROS2节点内部 | 客户端（Python脚本） |
| IK求解器 | 外部CuRobo服务或内置 | A1Kinematics (CuRobo) |
| 命令类型 | EEF pose delta | 直接关节角度 |
| 调试难度 | 困难（需要看ROS2日志） | 容易（直接看输出） |
| 工作状态 | ❌ 不工作 | ✅ 应该工作 |

## 🎯 测试建议

1. **先测试小幅度** - 测试2 (X+2cm)
2. **观察 IK 输出** - 确认求解成功且关节合理
3. **检查机器人运动** - 应该能看到明显移动
4. **再测试大幅度** - 测试3 (圆形轨迹)

## ⚙️ 可调参数

在 `A1Kinematics` 初始化中（`a1_x_kenimetic_haoyuan.py`）：

```python
num_seeds=32,              # IK求解的种子数量
position_threshold=0.005,   # 位置误差阈值 (5mm)
rotation_threshold=0.05,    # 旋转误差阈值 (~2.9°)
max_joint_delta=0.2,        # 关节变化上限 (0.2 rad)
```

在等待到位逻辑中（`test_haoyuan_ik_copy.py`）：

```python
joint_tolerance = 0.01  # 关节到达容忍度 (0.01 rad)
timeout = 10.0          # 等待超时时间
```

## 🐛 如果还是不动

1. **检查 IK 求解**:
   - 是否显示 "✓ IK 求解成功"？
   - 目标关节是否合理？

2. **检查命令发送**:
   - 是否显示 "发送关节命令..."？
   - 是否收到响应？

3. **检查 ROS2 话题**:
   ```bash
   ros2 topic echo /motion_target/target_joint_state_arm
   ```
   应该能看到关节命令在发布

4. **运行直接关节测试**:
   ```bash
   python3 test_direct_joint_command.py
   ```
   确认直接关节命令确实能动

---

现在运行测试，应该能看到机器人动了！🚀
