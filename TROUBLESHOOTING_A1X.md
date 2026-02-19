# A1X 实机测试故障排查指南

## 问题: 获取状态超时

```
✓ 已连接到 A1X ROS2 节点: 127.0.0.1:6100/6101
查询机器人状态...
❌ 获取状态超时
```

### 原因分析

这个错误说明：
- ✅ ZMQ 客户端连接成功（能够 connect）
- ❌ ROS2 节点没有响应（没有服务在监听这些端口）

### 解决方案

需要启动 A1X ROS2 ZMQ 桥接节点。

## 🚀 启动步骤

### 步骤 1: 启动 CuRobo IK 服务

在**终端 1**运行：

```bash
cd /home/dungeon_master/conrft
conda activate conrft
python scripts/curobo_ik_service.py --bind tcp://0.0.0.0:6202 --pos-threshold 0.02 --rot-threshold 0.05
```

等待看到：
```
🚀 CuRobo IK Service started on tcp://0.0.0.0:6202
```

### 步骤 2: 启动 A1X ROS2 节点

在**终端 2**运行：

```bash
cd /home/dungeon_master/conrft
export USE_CUROBO_IK=true
export CUROBO_IK_SERVICE=tcp://127.0.0.1:6202
./start_a1x_ros2_node.sh
```

或者手动运行：

```bash
cd /home/dungeon_master/conrft
python3 serl_robot_infra/franka_env/robots/a1x_ros2_node.py \
    --port 6100 \
    --node-name a1x_serl_node \
    --use-curobo-ik \
    --curobo-ik-service tcp://127.0.0.1:6202
```

等待看到：
```
🚀 A1XRobotZMQNode initialized with dual sockets:
   Command port: 6100
   State port: 6101
🚀 Using external CuRobo IK service: tcp://127.0.0.1:6202
```

### 步骤 3: 运行测试脚本

在**终端 3**运行：

```bash
cd /home/dungeon_master/conrft
conda activate conrft
python test_haoyuan_ik_copy.py
```

## 🔍 诊断命令

### 检查端口是否在监听

```bash
# 检查 6100 和 6101 端口
netstat -tuln | grep -E "6100|6101"
# 或者
ss -tuln | grep -E "6100|6101"

# 应该看到：
# tcp  0  0  0.0.0.0:6100  0.0.0.0:*  LISTEN
# tcp  0  0  0.0.0.0:6101  0.0.0.0:*  LISTEN
```

### 检查 CuRobo IK 服务

```bash
netstat -tuln | grep 6202
# 应该看到：
# tcp  0  0  0.0.0.0:6202  0.0.0.0:*  LISTEN
```

### 测试 ZMQ 连接

```bash
python3 -c "
import zmq
ctx = zmq.Context()
sock = ctx.socket(zmq.REQ)
sock.setsockopt(zmq.RCVTIMEO, 2000)
sock.connect('tcp://127.0.0.1:6101')
sock.send_json({'cmd': 'get_state'})
resp = sock.recv_json()
print('✓ 连接成功:', resp.get('status', 'unknown'))
sock.close()
ctx.term()
"
```

## 🐛 常见问题

### 1. ROS2 环境未设置

**错误**: `ModuleNotFoundError: No module named 'rclpy'`

**解决**:
```bash
source /opt/ros/humble/setup.bash
# 或
source /opt/ros/foxy/setup.bash
```

### 2. A1X 机器人未连接

**错误**: ROS2 节点启动但没有收到机器人反馈

**解决**:
- 检查 A1X 机器人是否上电
- 检查 ROS2 话题是否发布：
  ```bash
  ros2 topic list
  ros2 topic echo /hdas/feedback_arm
  ```

### 3. CuRobo IK 服务超时

**错误**: `❌ External CuRobo IK failed: timeout`

**解决**:
- 确保 CuRobo IK 服务已启动
- 检查服务地址是否正确
- 重启 IK 服务

### 4. 张量维度不匹配（已修复）

**错误**: `The size of tensor a (32) must match the size of tensor b (64)`

**状态**: ✅ 已在 `a1_x_kenimetic_haoyuan.py` 中修复

## 📊 完整启动流程

```bash
# 终端 1: CuRobo IK 服务
cd /home/dungeon_master/conrft
conda activate conrft
python scripts/curobo_ik_service.py --bind tcp://0.0.0.0:6202

# 终端 2: ROS2 节点
cd /home/dungeon_master/conrft
source /opt/ros/humble/setup.bash  # 如果需要
export USE_CUROBO_IK=true
export CUROBO_IK_SERVICE=tcp://127.0.0.1:6202
./start_a1x_ros2_node.sh

# 终端 3: 测试脚本
cd /home/dungeon_master/conrft
conda activate conrft
python test_haoyuan_ik_copy.py
```

## ✅ 验证清单

启动后检查：

- [ ] CuRobo IK 服务在端口 6202 监听
- [ ] ROS2 节点在端口 6100/6101 监听
- [ ] ROS2 话题 `/hdas/feedback_arm` 有数据
- [ ] 测试脚本能获取机器人状态
- [ ] 测试命令能成功执行

## 🎯 快速测试

启动所有服务后，运行：

```bash
python test_haoyuan_ik_copy.py
# 选择 1 - 查看机器人当前状态
```

应该看到：
```
✓ 成功获取状态
关节位置 (6+1): [...]
末端位置 (xyz): [...]
```
