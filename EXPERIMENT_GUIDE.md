# HIL-SERL 实验操作指南

> 适用脚本：`examples/train_rlpd_hil.py`（HIL-SERL Proposal 方法）  
> 作者备注：请按顺序执行以下三个阶段，每阶段均有前提条件说明。

---

## ⚠️ 前提：Fixed Gripper 任务的动作维度

**背景**：机器人动作向量默认是 **7 维**：

```
[x, y, z, rx, ry, rz,  gripper]
 ↑ 末端位置/姿态(6维)   ↑ 夹爪开合(1维)
```

Fixed Gripper 任务（夹爪固定不动）使用了 `GripperCloseEnv` wrapper，
它会把 action space **对外暴露为 6 维**（去掉最后的夹爪维度），
让策略不需要控制夹爪。

**潜在问题**：`train_rlpd_hil.py` 的训练脚本、replay buffer、
preference buffer 都以 `env.action_space` 的维度来分配存储空间。
如果某个任务是 6 维、另一个任务是 7 维，混用同一套代码时会报 shape 不匹配的错误。

**建议做法（二选一）**：

**方案 A（推荐）：不用 GripperCloseEnv，改用 `action_scale` 置零**

在任务 config 里把 `action_scale` 的第 7 维设为 0，让策略输出 7 维但夹爪维度
实际不生效。这样所有任务动作维度统一为 7 维，最简单。

```python
# config.py 示例：
action_scale = np.array([0.02, 0.02, 0.02, 0.1, 0.1, 0.1, 0.0])
#                                                              ↑ 夹爪维度置0，策略输出被忽略
```

`train_rlpd_hil.py` 的 actor 循环里已有对应处理（`_zero_mask`），
设置后夹爪维度会自动被置 0，无需其他修改。

**方案 B：使用 GripperCloseEnv + 补齐 Wrapper**

如果你的任务 config 里已经用了 `GripperCloseEnv`（6 维），
需要再套一个 `FixedGripperPadWrapper` 把对外维度补回 7 维：

```python
# serl_robot_infra/franka_env/envs/wrappers.py 中添加：
class FixedGripperPadWrapper(gym.ActionWrapper):
    """把 GripperCloseEnv 的 6 维 action space 对外补齐为 7 维（夹爪维恒为 0）。"""
    def __init__(self, env):
        super().__init__(env)
        # 在原 6 维基础上补一个 [0, 0] 的夹爪维
        low  = np.append(env.action_space.low,  0.0)
        high = np.append(env.action_space.high, 0.0)
        self.action_space = gym.spaces.Box(low, high, dtype=np.float32)

    def action(self, action: np.ndarray) -> np.ndarray:
        # 策略输出 7 维，传给底层时去掉第 7 维（底层只接受 6 维）
        return action[:6]

    def step(self, action):
        obs, rew, done, truncated, info = self.env.step(self.action(action))
        if "intervene_action_eef" in info:
            # Gello/SpaceMouse 干预动作也补齐到 7 维，保证存入 buffer 时格式一致
            info["intervene_action_eef"] = np.append(
                info["intervene_action_eef"], 0.0
            )
        return obs, rew, done, truncated, info
```

然后在 config.py 的 `get_environment()` 末尾加：
```python
env = GripperCloseEnv(env)        # 6 维（夹爪固定）
env = FixedGripperPadWrapper(env) # 补回 7 维（对外统一）
```

> **Progress Model 不受影响**：Progress Model 的输入是图像 + 关节**状态**（7 维），
> 与动作维度无关，fixed/learned gripper 切换不需要重新训练 Progress Model。

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
python /home/dungeon_master/liuzeyi/HILRL-A1X/examples/record_demos_octo_manual_new.py \
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

### 2.1 训练脚本

脚本位置：`examples/train_progress_model.py`

三阶段流程：
- **Phase 1**：用冻结的 ResNet-18 批量提取所有 demo 图像特征并缓存到磁盘
- **Phase 2**：仅训练 MLP head（Progress + MC + TD 三种损失）
- **Phase 3**：在 buffer 成功轨迹上评估并生成可视化图

### 2.2 运行命令

```bash
cd examples

python train_progress_model.py \
    --demo_dir  experiments/<EXP_NAME>/demo_data/<DATE> \
    --buffer_dir experiments/<EXP_NAME>/buffer \
    --output_dir experiments/<EXP_NAME>/progress_model \
    --hidden_dim 128 \
    --epochs 300 \
    --device cpu
```

**完整参数说明：**

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--demo_dir` | — | demo pkl 所在目录（含 `traj_*.pkl`） |
| `--buffer_dir` | — | online buffer 目录（用于 Phase 3 评估）|
| `--output_dir` | — | 输出根目录，脚本会自动在其下创建 `exp_NNN_YYYYMMDD_HHMM/` |
| `--cache_path` | 自动 | Phase 1 特征缓存路径；若文件已存在则跳过提取直接加载 |
| `--epochs` | `300` | 训练轮数 |
| `--batch_size` | `64` | 训练 batch size |
| `--lr` | `3e-4` | 学习率 |
| `--hidden_dim` | `128` | MLP 隐层维度（需与推理时一致）|
| `--lambda_prog` | `1.0` | 线性进度损失权重 |
| `--lambda_mc` | `0.1` | MC 回报损失权重 |
| `--lambda_td` | `0.1` | TD Bellman 一致性损失权重 |
| `--mc_last_n` | `5` | MC 损失仅作用于轨迹最后 N 步（0=全部）|
| `--n_eval_trajs` | `50` | Phase 3 评估的成功轨迹数量 |
| `--min_rl_ratio` | `0.1` | Phase 3 评估时过滤纯人工轨迹的最小 RL 占比 |
| `--device` | `cpu` | 推理设备（有 GPU 时可改为 `cuda`）|

### 2.3 训练输出

脚本会在 `--output_dir/exp_NNN_YYYYMMDD_HHMM/` 下生成：

```
experiments/<EXP_NAME>/progress_model/exp_001_20260417_HHMM/
├── progress_model_best.pt      # 验证 loss 最低的 MLP head 权重  ← 推理时使用
├── progress_model_final.pt     # 最后一个 epoch 的权重
├── state_stats.pt              # {"mean": (7,), "std": (7,)} 状态归一化统计  ← 推理时使用
├── training_curves.png         # 训练损失曲线
├── demo_progress_curves.png    # 每条 demo 的预测进度 vs 真实进度（训练集验证）
└── eval_visuals/
    ├── traj_001.png            # buffer 成功轨迹可视化（进度曲线 + 关键帧）
    ├── traj_002.png
    └── ...
```

> 关键：将 `progress_model_best.pt` 和 `state_stats.pt` 的路径记下来，
> 阶段三的 `--progress_model_path` 和 `--state_stats_path` 会用到。

### 2.4 验证推理模块

```bash
cd examples

python -c "
from progress_model_inference import ProgressModelRunner
runner = ProgressModelRunner(
    model_path='experiments/<EXP_NAME>/progress_model/exp_001_<TIMESTAMP>/progress_model_best.pt',
    stats_path='experiments/<EXP_NAME>/progress_model/exp_001_<TIMESTAMP>/state_stats.pt',
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
xvfb-run -a bash run_learner_zeyi.sh
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
