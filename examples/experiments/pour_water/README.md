# A1_X Robot Training Task

此文件夹包含 A1_X 机械臂的训练配置和脚本。

## 文件结构

```
a1x_pick_banana/
├── config.py                       # 训练和环境配置
├── wrapper.py                      # 自定义环境包装器
├── run_learner_conrft.sh          # 启动 learner (在线训练)
├── run_learner_conrft_pretrain.sh # 启动 learner (预训练)
├── run_actor_conrft.sh            # 启动 actor (执行策略)
├── demo_data/                     # 存放演示数据
├── conrft/                        # 存放训练检查点
└── classifier_ckpt/               # 存放奖励分类器检查点
```

## 配置说明

### EnvConfig (环境配置)

#### 1. A1_X 机器人配置
```python
A1X_NUM_DOFS = 7              # 自由度: 6个关节 + 1个夹爪
A1X_NODE_NAME = "a1x_serl_node"  # ROS2 节点名称
A1X_PORT = 6100                # ZMQ 通信端口
A1X_PYTHON_PATH = "/usr/bin/python3"  # ROS2 Python 路径
```

#### 2. 相机配置
修改为你的实际相机序列号:
```python
REALSENSE_CAMERAS = {
    "wrist_1": {
        "serial_number": "YOUR_SERIAL_NUMBER",  # 修改这里
        ...
    },
    ...
}
```

查看相机序列号: `rs-enumerate-devices`

#### 3. 关节空间配置

**目标关节状态** - 任务完成时的关节配置:
```python
TARGET_JOINT_STATE = np.array([0.0, -0.5, 0.0, -1.5, 0.0, 1.0, 20.0])
# [joint1, joint2, joint3, joint4, joint5, joint6, gripper(mm)]
```

**重置关节状态** - 每个 episode 开始的关节配置:
```python
RESET_JOINT_STATE = np.array([0.0, -0.2, 0.0, -1.0, 0.0, 0.5, 100.0])
# 夹爪 100mm = 完全张开
```

**动作缩放** - 控制每步的最大变化:
```python
ACTION_SCALE = np.array([0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 10.0])
# 前6个: 关节角度变化 (弧度/步)
# 最后1个: 夹爪变化 (mm/步)
```

**奖励阈值** - 判断成功的容差:
```python
REWARD_THRESHOLD = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 10.0])
# 当所有关节都在阈值内时,认为成功
```

#### 4. 安全限制
基于 A1_X 的实际关节限制:
```python
JOINT_LIMIT_LOW = np.array([-2.87, 0.5, -2.95, -1.55, -1.52, -1.56, 0.0])
JOINT_LIMIT_HIGH = np.array([2.89, 3.63, 0.0, 1.56, 1.52, 1.56, 100.0])
```

### TrainConfig (训练配置)

**观察键** - A1_X 使用关节空间,不是 TCP pose:
```python
proprio_keys = ["joint_positions", "joint_velocities", "gripper_position"]
```

**与 Franka 的区别:**
- Franka: `["tcp_pose", "tcp_vel", "tcp_force", "tcp_torque", "gripper_pose"]`
- A1_X: `["joint_positions", "joint_velocities", "gripper_position"]`

## 使用步骤

### 1. 修改配置

在 `config.py` 中修改:
- 相机序列号
- 目标关节配置 (根据你的任务)
- 重置关节配置
- 动作缩放和奖励阈值

### 2. 收集演示数据

```bash
# 在 examples/ 目录下运行
python record_demos_octo.py \
    --exp_name a1x_pick_banana \
    --demo_num 30 \
    --save_path ./experiments/a1x_pick_banana/demo_data/
```

这将:
- 使用 Gello 遥控 A1_X 机械臂
- 录制 30 个演示轨迹
- 保存到 `demo_data/a1x_pick_banana_30_demos.pkl`

### 3. 训练奖励分类器 (可选)

```bash
python train_reward_classifier.py \
    --exp_name a1x_pick_banana \
    --demo_path ./experiments/a1x_pick_banana/demo_data/
```

### 4. 启动训练

#### 方案 A: 预训练 + 在线微调 (推荐)

**步骤 1: 预训练**
```bash
cd examples/experiments/a1x_pick_banana
bash run_learner_conrft_pretrain.sh
```

**步骤 2: 在线训练**
```bash
# Terminal 1: Learner
bash run_learner_conrft.sh

# Terminal 2: Actor
bash run_actor_conrft.sh
```

#### 方案 B: 直接在线训练

```bash
# Terminal 1: Learner
bash run_learner_conrft.sh

# Terminal 2: Actor
bash run_actor_conrft.sh
```

## 关键差异: A1_X vs Franka

| 特性 | Franka | A1_X |
|------|--------|------|
| 控制空间 | 笛卡尔 (xyz + 姿态) | 关节空间 (7个关节) |
| 观察 | TCP pose, force, torque | 关节位置、速度 |
| 夹爪 | 二值 (开/关) | 连续 (0-100mm) |
| 动作空间 | [Δx, Δy, Δz, Δrx, Δry, Δrz, gripper] | [Δq1, ..., Δq6, Δgripper] |
| 后端 | ROS1 + Flask | ROS2 + ZMQ |

## 调试技巧

### 1. 测试环境
```python
from experiments.a1x_pick_banana.config import TrainConfig

config = TrainConfig()
env = config.get_environment(fake_env=False, save_video=True)

# 测试 reset
obs, info = env.reset()
print("Observation keys:", obs.keys())
print("Joint positions:", obs["state"][0])

# 测试 step
action = env.action_space.sample()
obs, reward, done, truncated, info = env.step(action)
```

### 2. 检查关节范围
```python
from franka_env.robots.a1x_robot import A1XRobot

robot = A1XRobot()
joints = robot.get_joint_state()
print("Current joints:", joints)
print("In limits:", 
      np.all(joints >= JOINT_LIMIT_LOW) and 
      np.all(joints <= JOINT_LIMIT_HIGH))
```

### 3. 可视化演示数据
```python
import pickle
with open('demo_data/a1x_pick_banana_30_demos.pkl', 'rb') as f:
    demos = pickle.load(f)
    
print(f"Number of demos: {len(demos)}")
print(f"Demo 0 length: {len(demos[0]['observations'])}")
print(f"Joint positions shape: {demos[0]['observations'][0]['joint_positions'].shape}")
```

## 常见问题

### Q: 如何调整任务目标?
A: 修改 `TARGET_JOINT_STATE`。可以先用 Gello 手动移动到目标位置,然后读取关节角度。

### Q: 训练不收敛怎么办?
A: 
1. 检查 `ACTION_SCALE` 是否合适 (太大会不稳定)
2. 增加演示数量 (推荐 30-50 个)
3. 调整 `REWARD_THRESHOLD` (放宽容差)
4. 检查关节限制是否正确

### Q: 如何使用预训练的 Octo 模型?
A: 在 `TrainConfig` 中设置:
```python
octo_path = "/path/to/octo-small"
```

### Q: 能否混用笛卡尔和关节控制?
A: 可以,但需要添加正/逆运动学。当前实现是纯关节空间控制。

## 下一步

1. **自定义奖励函数**: 在 `config.py` 的 `reward_func` 中修改
2. **添加任务特定逻辑**: 在 `wrapper.py` 的 `A1XTaskEnv` 中修改
3. **集成视觉**: 训练奖励分类器,使用视觉成功检测
4. **多任务训练**: 创建多个任务文件夹,共享检查点

## 参考

- SERL 论文: https://serl-robot.github.io/
- ConRFT: Conservative Reward Function Transfer
- Octo 模型: https://octo-models.github.io/
