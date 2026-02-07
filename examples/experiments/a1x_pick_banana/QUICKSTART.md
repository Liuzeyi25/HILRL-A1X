# A1_X 快速入门指南

## 前置条件检查

```bash
# 1. 检查 ROS2
source /opt/ros/humble/setup.zsh
ros2 --version

# 2. 检查 A1_X 机器人连接
ros2 topic list | grep hdas  # 应该看到 /hdas/feedback_arm

# 3. 检查相机
rs-enumerate-devices  # 记录序列号

# 4. 检查 Python 环境
python -c "import jax; import gymnasium; print('OK')"
```

## 5 分钟快速测试

### 1. 测试 A1_X 机器人接口

```bash
cd /home/dungeon_master/conrft/serl_robot_infra
python -c "
from franka_env.robots.a1x_robot import A1XRobot
import numpy as np

robot = A1XRobot(num_dofs=7, port=6100)
print('Robot initialized')
print('Joint state:', robot.get_joint_state())
robot.close()
"
```

### 2. 测试 A1_X 环境 (无相机)

```bash
cd /home/dungeon_master/conrft/serl_robot_infra
python -c "
from franka_env.envs.a1x_env import A1XEnv
from franka_env.envs.a1x_config import MinimalA1XConfig

env = A1XEnv(hz=5, config=MinimalA1XConfig())
obs, _ = env.reset()
print('Observation keys:', obs.keys())
print('Joint positions:', obs['state']['joint_positions'])

# 随机动作
import numpy as np
action = np.random.uniform(-0.1, 0.1, size=(7,))
obs, reward, done, _, info = env.step(action)
print('Step successful!')

env.close()
"
```

### 3. 测试训练环境 (完整)

```bash
cd /home/dungeon_master/conrft/examples/experiments/a1x_pick_banana

# 修改 config.py 中的相机序列号后运行:
python -c "
import sys
sys.path.append('../..')
from experiments.a1x_pick_banana.config import TrainConfig

config = TrainConfig()
env = config.get_environment(fake_env=False, save_video=False)
print('Training environment created successfully!')

obs, _ = env.reset()
print('Observation shape:', {k: v.shape for k, v in obs.items()})

env.close()
"
```

## 完整训练流程

### 步骤 1: 配置修改 (5分钟)

编辑 `config.py`:

```python
# 1. 修改相机序列号 (第 28-47 行)
REALSENSE_CAMERAS = {
    "wrist_1": {
        "serial_number": "YOUR_SERIAL_HERE",  # <-- 改这里
        ...
    },
    ...
}

# 2. 设置任务目标关节位置 (第 69 行)
TARGET_JOINT_STATE = np.array([0.0, -0.5, 0.0, -1.5, 0.0, 1.0, 20.0])
# 提示: 用 Gello 手动移到目标位置,运行下面命令获取关节角度:
# python -c "from franka_env.robots.a1x_robot import A1XRobot; \
#            r=A1XRobot(); print(r.get_joint_state()); r.close()"

# 3. 设置重置位置 (第 72 行)
RESET_JOINT_STATE = np.array([0.0, -0.2, 0.0, -1.0, 0.0, 0.5, 100.0])
```

### 步骤 2: 收集演示 (30分钟)

```bash
cd /home/dungeon_master/conrft/examples

# 方法 A: 使用 Gello 遥控录制
python record_demos_octo.py \
    --exp_name a1x_pick_banana \
    --demo_num 30 \
    --save_path ./experiments/a1x_pick_banana/demo_data/
    
 python /home/dungeon_master/conrft/examples/record_demos_octo_manual_new.py \\n    --exp_name a1x_pick_banana \\n    --successes_needed 10 \\n    --manual_success
# 方法 B: 使用 spacemouse (如果没有 Gello)
python record_demos_octo.py \
    --exp_name a1x_pick_banana \
    --demo_num 30 \
    --save_path ./experiments/a1x_pick_banana/demo_data/ \
    --use_spacemouse
```

录制技巧:
- 每个演示尽量平滑、一致
- 避免过快或过慢的动作
- 确保每个演示都成功完成任务

### 步骤 3: 预训练 (1-2小时)

```bash
cd /home/dungeon_master/conrft/examples/experiments/a1x_pick_banana

# 启动预训练
bash run_learner_conrft_pretrain.sh

# 监控训练 (在另一个终端)
tensorboard --logdir=./conrft/logs
# 打开浏览器访问 http://localhost:6006
```

预训练完成标志:
- BC loss < 0.1
- 训练步数达到 50000

### 步骤 4: 在线训练 (2-4小时)

```bash
cd /home/dungeon_master/conrft/examples/experiments/a1x_pick_banana

# Terminal 1: 启动 Learner
bash run_learner_conrft.sh

# Terminal 2: 启动 Actor (等 learner 初始化后)
bash run_actor_conrft.sh
```

监控指标:
- Success rate (目标 > 80%)
- Average reward (应该逐渐增加)
- Q-value (应该稳定)

### 步骤 5: 评估 (10分钟)

```bash
cd /home/dungeon_master/conrft/examples

python eval_policy.py \
    --exp_name a1x_pick_banana \
    --checkpoint_path ./experiments/a1x_pick_banana/conrft \
    --eval_episodes 10
```

## 常用命令速查

### 查看训练进度
```bash
# 检查检查点
ls -lh experiments/a1x_pick_banana/conrft/

# 查看最新日志
tail -f experiments/a1x_pick_banana/conrft/logs/train.log

# Tensorboard
tensorboard --logdir experiments/a1x_pick_banana/conrft/logs
```

### 恢复训练
```bash
# 从检查点恢复
bash run_learner_conrft.sh --resume_from_checkpoint=True
```

### 调试模式
```bash
# 单步调试环境
python -c "
from experiments.a1x_pick_banana.config import TrainConfig
env = TrainConfig().get_environment()
import pdb; pdb.set_trace()
"
```

## 故障排除

### 问题 1: ROS2 节点启动失败
```bash
# 检查 ROS2
source /opt/ros/humble/setup.zsh
ros2 topic list

# 检查端口占用
lsof -i :6100
kill -9 <PID>  # 如果被占用
```

### 问题 2: 关节角度超限
```bash
# 检查当前关节角度
python -c "
from franka_env.robots.a1x_robot import A1XRobot
r = A1XRobot()
joints = r.get_joint_state()
print('Current:', joints)
print('Low limit:', [-2.87, 0.5, -2.95, -1.55, -1.52, -1.56, 0.0])
print('High limit:', [2.89, 3.63, 0.0, 1.56, 1.52, 1.56, 100.0])
r.close()
"
```

### 问题 3: 演示数据加载失败
```bash
# 验证演示数据
python -c "
import pickle
with open('experiments/a1x_pick_banana/demo_data/a1x_pick_banana_30_demos.pkl', 'rb') as f:
    demos = pickle.load(f)
print(f'Demos: {len(demos)}')
print(f'Keys: {demos[0].keys()}')
print(f'Length: {len(demos[0][\"observations\"])}')
"
```

### 问题 4: 相机无法打开
```bash
# 检查相机连接
rs-enumerate-devices

# 测试相机读取
python -c "
from franka_env.camera.rs_capture import RSCapture
from franka_env.camera.video_capture import VideoCapture

cap = VideoCapture(RSCapture(
    name='test',
    serial_number='YOUR_SERIAL',
    dim=(640, 480)
))
img = cap.read()
print('Image shape:', img.shape)
cap.close()
"
```

## 性能优化

### GPU 内存优化
```bash
# 减少 GPU 内存使用
export XLA_PYTHON_CLIENT_MEM_FRACTION=.3  # 降低到 30%
```

### 控制频率调优
```python
# 在 config.py 中调整
env = A1XEnv(hz=5)  # 降低频率减少计算负担
```

### 批量大小调优
```python
# 在训练脚本中
--batch_size=128  # 减小批量大小
```

## 下一步学习

1. **理解关节空间控制**: 阅读 `franka_env/robots/a1x_robot.py`
2. **自定义任务**: 修改 `wrapper.py` 中的 `A1XTaskEnv`
3. **奖励函数设计**: 在 `config.py` 中实现自定义 `reward_func`
4. **多任务学习**: 创建多个任务文件夹,共享预训练模型

## 参考资料

- [SERL 论文](https://serl-robot.github.io/)
- [ConRFT 代码](https://github.com/rail-berkeley/serl)
- [Octo 模型](https://octo-models.github.io/)
- [A1_X 集成文档](../../../serl_robot_infra/A1X_INTEGRATION.md)
