# Franka → A1_X 迁移指南

本文档说明如何将现有的 Franka 任务配置迁移到 A1_X 机械臂。

## 核心差异概览

| 方面 | Franka | A1_X |
|------|--------|------|
| **控制空间** | 笛卡尔空间 (xyz + 欧拉角) | 关节空间 (7个关节角度) |
| **动作定义** | `[Δx, Δy, Δz, Δrx, Δry, Δrz, gripper]` | `[Δq1, Δq2, Δq3, Δq4, Δq5, Δq6, Δgripper]` |
| **状态观察** | TCP pose, velocity, force, torque | Joint positions, velocities |
| **夹爪控制** | 二值 (±1 = 开/关) | 连续 (0-100mm) |
| **通信方式** | HTTP REST API (Flask) | ROS2 + ZMQ |
| **重置逻辑** | 笛卡尔插补 | 关节空间插补 |

## 代码迁移步骤

### 1. 环境配置类

#### Franka 配置:
```python
from franka_env.envs.franka_env import DefaultEnvConfig

class EnvConfig(DefaultEnvConfig):
    SERVER_URL: str = "http://127.0.0.2:5000/"
    
    # 笛卡尔空间
    TARGET_POSE = np.array([0.33, -0.15, 0.20, np.pi, 0, 0])  # xyz + rpy
    RESET_POSE = np.array([0.61, -0.17, 0.22, np.pi, 0, 0])
    ACTION_SCALE = np.array([0.08, 0.2, 1])  # [xyz_scale, rpy_scale, gripper]
    
    # 笛卡尔空间限制
    ABS_POSE_LIMIT_HIGH = TARGET_POSE + np.array([0.3, 0.03, 0.02, ...])
    ABS_POSE_LIMIT_LOW = TARGET_POSE - np.array([0.03, 0.05, 0.05, ...])
    
    # 阻抗参数
    COMPLIANCE_PARAM = {
        "translational_stiffness": 2000,
        "rotational_stiffness": 150,
        ...
    }
```

#### A1_X 配置:
```python
from franka_env.envs.a1x_env import DefaultA1XEnvConfig

class EnvConfig(DefaultA1XEnvConfig):
    # 移除 SERVER_URL - A1_X 不使用 Flask
    
    # A1_X 连接配置
    A1X_PORT = 6100
    A1X_NODE_NAME = "a1x_serl_node"
    
    # 关节空间
    TARGET_JOINT_STATE = np.array([0.0, -0.5, 0.0, -1.5, 0.0, 1.0, 20.0])
    RESET_JOINT_STATE = np.array([0.0, -0.2, 0.0, -1.0, 0.0, 0.5, 100.0])
    ACTION_SCALE = np.array([0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 10.0])
    
    # 关节限制
    JOINT_LIMIT_LOW = np.array([-2.87, 0.5, -2.95, -1.55, -1.52, -1.56, 0.0])
    JOINT_LIMIT_HIGH = np.array([2.89, 3.63, 0.0, 1.56, 1.52, 1.56, 100.0])
    
    # 移除 COMPLIANCE_PARAM - A1_X 不需要阻抗控制参数
```

**迁移要点:**
1. ✅ 用关节角度替代 TCP 位姿
2. ✅ 移除阻抗控制参数
3. ✅ 添加 A1_X 连接配置
4. ✅ 调整动作缩放 (关节角度的单位是弧度)

### 2. 训练配置类

#### Franka 配置:
```python
class TrainConfig(DefaultTrainingConfig):
    # Franka 使用笛卡尔空间状态
    proprio_keys = [
        "tcp_pose",      # 7D: xyz + quat
        "tcp_vel",       # 6D: linear + angular velocity
        "tcp_force",     # 3D: force
        "tcp_torque",    # 3D: torque
        "gripper_pose"   # 1D: gripper state
    ]
```

#### A1_X 配置:
```python
class TrainConfig(DefaultTrainingConfig):
    # A1_X 使用关节空间状态
    proprio_keys = [
        "joint_positions",   # 7D: 6 joints + gripper
        "joint_velocities",  # 7D: joint velocities
        "gripper_position"   # 1D: gripper position (0-100mm)
    ]
```

**迁移要点:**
1. ✅ 替换状态键名
2. ✅ 注意维度变化: TCP pose (7D) → Joint positions (7D) 相同
3. ✅ 移除 force/torque (A1_X 暂不支持)

### 3. 环境包装器

#### Franka Wrapper:
```python
from franka_env.envs.franka_env import FrankaEnv

class PickBananaEnv(FrankaEnv):
    def interpolate_move(self, goal: np.ndarray, timeout: float):
        """笛卡尔空间移动"""
        if goal.shape == (6,):
            goal = np.concatenate([goal[:3], euler_2_quat(goal[3:])])
        self._send_pos_command(goal)  # HTTP 请求
        time.sleep(timeout)
    
    def go_to_reset(self, joint_reset=False):
        """重置到笛卡尔位置"""
        if joint_reset:
            requests.post(self.url + "jointreset")
        
        reset_pose = self.resetpos.copy()
        self.interpolate_move(reset_pose, timeout=1)
        
        # 切换阻抗参数
        requests.post(self.url + "update_param", 
                     json=self.config.COMPLIANCE_PARAM)
```

#### A1_X Wrapper:
```python
from franka_env.envs.a1x_env import A1XEnv

class A1XTaskEnv(A1XEnv):
    def interpolate_move(self, goal_joints: np.ndarray, timeout: float):
        """关节空间移动"""
        assert len(goal_joints) == 7
        steps = int(timeout * self.hz)
        path = np.linspace(self.curr_joint_positions, goal_joints, steps)
        
        for joints in path:
            self.robot.command_joint_state(joints)  # 直接 ZMQ 通信
            time.sleep(1.0 / self.hz)
    
    def go_to_reset(self):
        """重置到关节位置"""
        reset_joints = self._RESET_JOINT_STATE.copy()
        
        if self.randomreset:
            # 添加关节噪声
            noise = np.random.uniform(-0.1, 0.1, size=(6,))
            reset_joints[:6] += noise
        
        self.interpolate_move(reset_joints, timeout=2.0)
        # 无需切换参数 - A1_X 直接控制
```

**迁移要点:**
1. ✅ 继承改为 `A1XEnv`
2. ✅ 移除所有 `requests.post()` 调用
3. ✅ 用 `robot.command_joint_state()` 替代 `_send_pos_command()`
4. ✅ 插补在关节空间进行,不需要四元数转换
5. ✅ 移除阻抗参数切换

### 4. 夹爪包装器

#### Franka Gripper:
```python
class GripperPenaltyWrapper(gym.Wrapper):
    def step(self, action):
        grasp_action = action[..., -1]
        
        # 二值化
        grasp_action = np.where(
            grasp_action > 0.5, 1,        # 开
            np.where(grasp_action < -0.5, -1, 0)  # 关
        )
        
        # 检测开关
        if (action[-1] < -0.5 and self.last_gripper_pos > 0.7):
            penalty = self.penalty  # 刚从开变关
```

#### A1_X Gripper:
```python
class A1XGripperPenaltyWrapper(gym.Wrapper):
    def __init__(self, env, penalty=-0.05, threshold=20.0):  # 新增阈值
        self.threshold = threshold  # mm
    
    def step(self, action):
        gripper_action = action[..., -1]  # 连续值 (0-100mm)
        
        # 检测大幅度变化
        if self.last_action_gripper is not None:
            delta = abs(gripper_action - self.last_action_gripper)
            if delta > self.threshold:
                penalty = self.penalty
```

**迁移要点:**
1. ✅ A1_X 夹爪是连续控制,不需要二值化
2. ✅ 检测变化量而非状态切换
3. ✅ 阈值单位是 mm

## 关键概念转换

### 1. 目标位置定义

#### Franka (笛卡尔):
```python
# 如何确定目标位置?
# 1. 手动移动机器人到目标
# 2. 读取 TCP pose
TARGET_POSE = [x, y, z, roll, pitch, yaw]
```

#### A1_X (关节):
```python
# 如何确定目标关节角?
# 1. 用 Gello 手动移动到目标
# 2. 读取关节角度
from franka_env.robots.a1x_robot import A1XRobot
r = A1XRobot()
TARGET_JOINT_STATE = r.get_joint_state()
print(TARGET_JOINT_STATE)
r.close()
```

### 2. 动作空间转换

#### Franka 动作:
```python
action = [
    0.01,   # Δx (m)
    0.0,    # Δy (m)
    0.005,  # Δz (m)
    0.0,    # Δroll (rad)
    0.0,    # Δpitch (rad)
    0.1,    # Δyaw (rad)
    1.0     # gripper: 1=open, -1=close
]
```

#### A1_X 动作:
```python
action = [
    0.05,   # Δq1 (rad)
    0.0,    # Δq2 (rad)
    0.0,    # Δq3 (rad)
    -0.02,  # Δq4 (rad)
    0.0,    # Δq5 (rad)
    0.01,   # Δq6 (rad)
    10.0    # Δgripper (mm): 正数=张开, 负数=闭合
]
```

### 3. 观察空间转换

#### Franka 观察:
```python
obs = {
    "images": {...},
    "state": {
        "tcp_pose": [x, y, z, qx, qy, qz, qw],  # 7D
        "tcp_vel": [vx, vy, vz, wx, wy, wz],    # 6D
        "tcp_force": [fx, fy, fz],              # 3D
        "tcp_torque": [tx, ty, tz],             # 3D
        "gripper_pose": [gripper_width]         # 1D
    }
}
```

#### A1_X 观察:
```python
obs = {
    "images": {...},
    "state": {
        "joint_positions": [q1, q2, q3, q4, q5, q6, gripper],  # 7D
        "joint_velocities": [dq1, ..., dq6, d_gripper],        # 7D
        "gripper_position": [gripper_mm]                        # 1D (0-100)
    }
}
```

## 迁移检查清单

使用此清单确保迁移完整:

### 配置文件 (`config.py`)
- [ ] 继承 `DefaultA1XEnvConfig` 而非 `DefaultEnvConfig`
- [ ] 添加 `A1X_PORT`, `A1X_NODE_NAME` 等配置
- [ ] 用 `TARGET_JOINT_STATE` 替换 `TARGET_POSE`
- [ ] 用 `RESET_JOINT_STATE` 替换 `RESET_POSE`
- [ ] 调整 `ACTION_SCALE` 为关节角度缩放
- [ ] 添加 `JOINT_LIMIT_LOW/HIGH` 替代 `ABS_POSE_LIMIT`
- [ ] 移除 `SERVER_URL`, `COMPLIANCE_PARAM`, `PRECISION_PARAM`
- [ ] 更新 `proprio_keys` 为关节空间

### 包装器 (`wrapper.py`)
- [ ] 继承 `A1XEnv` 而非 `FrankaEnv`
- [ ] 移除所有 `requests.post()` 调用
- [ ] 用 `robot.command_joint_state()` 替换 `_send_pos_command()`
- [ ] 更新 `interpolate_move()` 为关节空间插补
- [ ] 移除 `euler_2_quat()` 转换
- [ ] 更新夹爪包装器处理连续值
- [ ] 移除阻抗参数切换逻辑

### 启动脚本
- [ ] 更新 `--exp_name`
- [ ] 更新 `--checkpoint_path`
- [ ] 更新 `--demo_path`

### 数据收集
- [ ] 使用 Gello 而非 spacemouse (或更新 spacemouse 映射)
- [ ] 验证演示数据包含正确的关节空间观察

### 测试
- [ ] 测试环境创建: `env = TrainConfig().get_environment()`
- [ ] 测试重置: `obs, _ = env.reset()`
- [ ] 测试步进: `env.step(action)`
- [ ] 验证观察空间维度
- [ ] 验证动作空间维度

## 完整迁移示例

假设你有一个 Franka 任务 `task1_pick_banana`,这是迁移步骤:

```bash
# 1. 复制文件夹
cp -r experiments/task1_pick_banana experiments/task_a1x_pick_banana

cd experiments/task_a1x_pick_banana

# 2. 编辑 config.py
# - 替换 DefaultEnvConfig → DefaultA1XEnvConfig
# - 更新所有配置 (见上文)

# 3. 编辑 wrapper.py
# - 替换 FrankaEnv → A1XEnv
# - 移除 requests.post
# - 更新插补逻辑

# 4. 更新启动脚本
sed -i 's/task1_pick_banana/task_a1x_pick_banana/g' run_*.sh

# 5. 清空数据文件夹
rm -rf demo_data/* conrft/* classifier_ckpt/*

# 6. 测试
python -c "
from experiments.task_a1x_pick_banana.config import TrainConfig
env = TrainConfig().get_environment()
print('Migration successful!')
"
```

## 常见陷阱

### ❌ 陷阱 1: 混用笛卡尔和关节空间
```python
# 错误: 在 A1_X 中使用笛卡尔坐标
TARGET_POSE = [0.5, 0.3, 0.2, ...]  # ❌
```
**解决**: 始终使用关节角度

### ❌ 陷阱 2: 保留 Flask 相关代码
```python
# 错误: A1_X 不使用 HTTP
requests.post(self.url + "clearerr")  # ❌
```
**解决**: 移除所有 `requests` 调用

### ❌ 陷阱 3: 假设夹爪是二值的
```python
# 错误: A1_X 夹爪是连续的
if action[-1] > 0.5:  # ❌ 不是 0/1
    open_gripper()
```
**解决**: 使用连续值 (0-100mm)

### ❌ 陷阱 4: 忘记更新观察键
```python
# 错误: A1_X 没有 tcp_pose
proprio_keys = ["tcp_pose", ...]  # ❌
```
**解决**: 用 `joint_positions` 等替换

## 性能对比

| 指标 | Franka | A1_X |
|------|--------|------|
| 控制频率 | 10-20 Hz | 10-500 Hz |
| 通信延迟 | ~50ms (HTTP) | ~5ms (ZMQ) |
| 状态维度 | 20D (pose+vel+force+torque) | 15D (joints+vel) |
| 复杂度 | 需要逆运动学 | 直接关节控制 |

## 总结

从 Franka 迁移到 A1_X 的关键在于:

1. **控制空间转换**: 笛卡尔 → 关节
2. **通信方式替换**: HTTP → ZMQ
3. **观察空间更新**: TCP → Joints
4. **夹爪处理**: 二值 → 连续

按照本指南逐步迁移,大多数任务可以在 1-2 小时内完成移植。
