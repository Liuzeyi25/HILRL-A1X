# HIL-SERL 实验操作指南

> 适用脚本：`examples/train_rlpd_hil.py`（HIL-SERL Proposal 方法）  
> 作者备注：请按顺序执行以下三个阶段，每阶段均有前提条件说明。

---

## ⚠️ 前提：Fixed Gripper 任务的动作维度统一

**问题**：`GripperCloseEnv` wrapper 会将 action space 从 7 维截断为 6 维，导致
不同任务的 demo 数据和 preference buffer 中动作 shape 不一致，跨任务复用时会报错。

**解决方案**：在任务 config 的 `get_environment()` 中，对 fixed gripper 任务统一
套用一个补齐 wrapper，对外暴露 7 维（最后一维恒为 0），内部仍由 `GripperCloseEnv`
截断传给底层：

```python
# 在 config.py 的 get_environment() 末尾添加：
if self.setup_mode in ("single-arm-fixed-gripper", "dual-arm-fixed-gripper"):
    from serl_robot_infra.franka_env.envs.wrappers import GripperCloseEnv
    env = GripperCloseEnv(env)          # action_space → 6 维（内部）
    # 再补一个 pad wrapper，对外暴露 7 维
    env = FixedGripperPadWrapper(env)   # action_space → 7 维（外部）
```

```python
# FixedGripperPadWrapper 参考实现（可放在 wrappers.py）：
class FixedGripperPadWrapper(gym.ActionWrapper):
    """统一 fixed gripper 任务动作空间为 7 维，最后一维置 0。"""
    def __init__(self, env):
        super().__init__(env)
        low  = np.append(env.action_space.low,  0.0)
        high = np.append(env.action_space.high, 0.0)
        self.action_space = gym.spaces.Box(low, high, dtype=np.float32)

    def action(self, action: np.ndarray) -> np.ndarray:
        return action[:6]   # 去掉第 7 维再传给底层

    def step(self, action):
        obs, rew, done, truncated, info = self.env.step(self.action(action))
        if "intervene_action_eef" in info:
            # 补齐到 7 维，使 train_rlpd_hil.py 存储格式一致
            info["intervene_action_eef"] = np.append(
                info["intervene_action_eef"], 0.0
            )
        return obs, rew, done, truncated, info
```

> **Progress Model 不受影响**：Progress Model 的 `STATE_DIM=7` 指的是机器人
> **关节状态**（joint positions），不是动作维度，与 gripper 模式无关。

---

## 阶段一：收集演示数据（Demo Collection）

### 1.1 环境准备

```bash
# 进入项目目录
cd /path/to/HILRL-A1X/examples

# 激活 conda 环境
conda activate conrft

# 确认机器人 ROS2 节点已启动（硬件端）
bash ../start_a1x_ros2_node.sh
```

### 1.2 运行录制脚本

```bash
python record_demos_octo_manual_new.py \
    --exp_name <YOUR_EXP_NAME> \
    --successes_needed 20 \
    --demo_data_subdir $(date +%Y%m%d)
```

**关键参数说明：**

| 参数 | 说明 | 示例 |
|------|------|------|
| `--exp_name` | 实验名，需在 `experiments/mappings.py` 中注册 | `a1x_pick_banana` |
| `--successes_needed` | 需要收集的成功 demo 数量 | `20`（简单任务）/ `30`（复杂任务）|
| `--demo_data_subdir` | demo 存储子目录名 | `20260417` |

### 1.3 验证数据

```python
import pickle
with open('experiments/<EXP_NAME>/demo_data/<DATE>/traj_XX.pkl', 'rb') as f:
    demo = pickle.load(f)
print(f"demo 条数: {len(demo)}")
print(f"动作维度: {demo[0]['actions'].shape}")   # 应为 (7,)
print(f"包含 alpha_weight: {'alpha_weight' in demo[0]}")  # 可选
```

### 1.4 注意事项

- 录制过程中通过 **Gello 或 SpaceMouse** 进行人类干预，脚本自动记录
- 每次成功后脚本会保存到 `experiments/<EXP_NAME>/demo_data/<DATE>/traj_<N>.pkl`
- 建议至少 **20 条成功 demo** 再进行后续训练
- Fixed gripper 任务使用了 `FixedGripperPadWrapper` 后，录制数据动作维度应为 7 维

---

## 阶段二：训练 Progress Model（可选但推荐）

> 若跳过此阶段，`train_rlpd_hil.py` 会自动退化为 fallback 模式
>（每个干预点向前回溯 `--suboptimal_window` 步作为次优片段起点），功能正常但精度较低。

### 2.1 当前状态

`train_progress_model.py` 脚本尚未集成到本仓库，需要单独运行。
输出文件为：
- `progress_model_best.pt` —— ProgressHead 权重
- `state_stats.pt` —— `{"mean": (7,), "std": (7,)}` 状态归一化统计

### 2.2 训练脚本（待集成）

```bash
# 占位命令，实际路径待确认
python train_progress_model.py \
    --demo_path experiments/<EXP_NAME>/demo_data/<DATE>/traj_*.pkl \
    --save_dir  experiments/<EXP_NAME>/progress_model/ \
    --hidden_dim 128 \
    --epochs 100
```

训练完成后，输出文件位于：
```
experiments/<EXP_NAME>/progress_model/
├── progress_model_best.pt
└── state_stats.pt
```

### 2.3 验证推理模块

```bash
# 快速验证 progress model 能正常加载并推理
python -c "
from progress_model_inference import ProgressModelRunner
runner = ProgressModelRunner(
    model_path='experiments/<EXP_NAME>/progress_model/progress_model_best.pt',
    stats_path='experiments/<EXP_NAME>/progress_model/state_stats.pt',
    side_key='side_policy_256',
    wrist_key='wrist_1',
    hidden_dim=128,
    device='cpu',
)
print('Progress Model 加载成功')
"
```

---

## 阶段三：HIL-SERL 在线强化学习训练

> `train_rlpd_hil.py` 采用 **Learner / Actor 分离架构**，需要在两台机器（或两个终端）上分别启动。
> - **Learner**：运行在服务器（GPU 机器），负责神经网络更新
> - **Actor**：运行在机器人端，负责与环境交互和收集数据

### 3.1 为新任务创建启动脚本

在 `examples/experiments/<EXP_NAME>/` 下创建两个脚本：

**`run_learner_hil.sh`**（服务器端）：
```bash
#!/usr/bin/env bash
RUN_TAG=${RUN_TAG:-$(date +%m%d-%H%M)}

export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=.5
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

# 有 Progress Model 时：
python ../../train_rlpd_hil.py \
    --exp_name=<EXP_NAME> \
    --checkpoint_path=experiments/<EXP_NAME>/checkpoints_hil/${RUN_TAG} \
    --demo_path=experiments/<EXP_NAME>/demo_data/<DATE>/traj_XX.pkl \
    --learner \
    --run_tag="${RUN_TAG}" \
    --alpha_lambda=3.0 \
    --contrastive_coef=0.2 \
    --preference_batch_size=4 \
    --suboptimal_window=5 \
    --progress_model_path=experiments/<EXP_NAME>/progress_model/progress_model_best.pt \
    --state_stats_path=experiments/<EXP_NAME>/progress_model/state_stats.pt

# 无 Progress Model 时（fallback 模式），删去最后两行 --progress_model_path / --state_stats_path
```

**`run_actor_hil.sh`**（机器人端）：
```bash
#!/bin/bash
RUN_TAG=${RUN_TAG:-"$(date +%m%d-%H%M)"}

export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=.2

python ../../train_rlpd_hil.py \
    --exp_name=<EXP_NAME> \
    --checkpoint_path=experiments/<EXP_NAME>/checkpoints_hil/${RUN_TAG} \
    --actor \
    --ip=<LEARNER_IP>        # 若同机则填 localhost
```

### 3.2 启动顺序

**Step 1：先启动 Learner（服务器）**

```bash
cd examples/experiments/<EXP_NAME>
conda activate conrft
xvfb-run -a bash run_learner_hil.sh
```

> `xvfb-run -a` 用于无显示器服务器，如果服务器有 GPU 显示则不需要。

**Step 2：等待 Learner 打印 "waiting for data"，再启动 Actor（机器人端）**

```bash
cd examples/experiments/<EXP_NAME>
conda activate conrft
bash run_actor_hil.sh
```

### 3.3 常用可调超参说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--alpha_lambda` | `3.0` | 次优片段位置衰减强度，越大越聚焦干预点附近 |
| `--contrastive_coef` | `0.2` | 对比损失权重 β，0 则退化为标准 RLPD |
| `--preference_batch_size` | `4` | 偏好 buffer 最小触发量，低于此数时不做修正更新 |
| `--suboptimal_window` | `5` | fallback 模式回溯步数（无 Progress Model 时生效）|
| `--suboptimal_window` | `5` | 建议简单任务 3~5，复杂任务 8~15 |
| `--alpha_lambda` | `3.0` | 可先用默认，根据 wandb 中 `mean_alpha` 曲线调整 |

### 3.4 训练监控

训练过程中 Learner 会将以下指标上报到 **WandB**（项目名 `hil-serl`）：

| 指标 | 说明 |
|------|------|
| `use_correction` | 1.0=修正模式，0.0=标准 RLPD（preference buffer 不足时） |
| `preference_buffer_size` | 累计偏好对数量 |
| `environment/episode/intervention_count` | 每个 episode 干预次数 |
| `environment/episode/mean_alpha` | 平均位置权重（反映次优片段识别质量） |
| `environment/episode/n_suboptimal_segs` | 每 episode 识别到的次优片段数 |

### 3.5 续训

```bash
# 续训时只需指定同一个 RUN_TAG
RUN_TAG=<已有TAG> bash run_learner_hil.sh
RUN_TAG=<已有TAG> bash run_actor_hil.sh
```

---

## 阶段四：评估

```bash
# 在机器人端运行，actor 会加载指定 step 的 checkpoint 并跑 eval_n_trajs 条轨迹
bash run_actor_hil.sh \
    --eval_checkpoint_step=<STEP> \
    --eval_n_trajs=20
```

---

## 附录：目录结构参考

```
examples/experiments/<EXP_NAME>/
├── config.py                          # 任务配置（setup_mode、image_keys 等）
├── demo_data/
│   └── <DATE>/
│       ├── traj_0.pkl
│       ├── traj_1.pkl
│       └── ...
├── progress_model/                    # 阶段二产出（可选）
│   ├── progress_model_best.pt
│   └── state_stats.pt
├── checkpoints_hil/
│   └── <RUN_TAG>/
│       ├── checkpoint_XXXXX           # Flax checkpoint
│       ├── buffer/                    # Online replay buffer 落盘
│       └── demo_buffer/               # Demo buffer 落盘
├── run_learner_hil.sh
└── run_actor_hil.sh
```

---

## 常见问题

**Q: Actor 启动后报 `Connection refused`**  
A: Learner 还未完成初始化，等待约 30 秒后 Learner 打印 `sent initial network to actor` 再启动 Actor。

**Q: `assert ub.shape == (7,)` 报错**  
A: 说明 `GripperCloseEnv` 被套了两次，检查 config.py 中 wrapper 堆叠顺序。

**Q: `use_correction` 始终为 0**  
A: preference buffer 积累不足 `--preference_batch_size`，需要更多人类干预操作后才会切换到修正模式。

**Q: Fixed gripper 任务动作 shape 不匹配**  
A: 检查是否在 config 中添加了 `FixedGripperPadWrapper`（见"前提"章节）。

**Q: Progress Model 推理报 `state shape mismatch`**  
A: Progress Model 的 `STATE_DIM=7` 指机器人关节状态维度，与 gripper 模式无关。
检查 obs dict 中 state key 的实际维度是否与 `STATE_DIM` 一致。
