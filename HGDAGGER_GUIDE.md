# HG-DAgger 训练指南

> **适用算法**：HG-DAgger（Human-Guided DAgger，人类引导的交互式模仿学习）
> **训练脚本**：`examples/train_hgdagger.py`
> **适用平台**：A1X 单臂机器人
> **前置条件**：已完成环境安装（conda 环境 `conrft` 可用，ROS2 已启动，相机序列号已配置）

---

## 算法概述

HG-DAgger 是一种**纯模仿学习**方法，不使用强化学习信号（无 Q 函数、无 reward 优化）。

**核心思想**：
1. 用少量人类演示做行为克隆热启动（让策略有基本能力）
2. 部署策略让它自己跑，人类随时接管纠正错误动作
3. 只把人类干预的那部分轨迹拿来继续训练（BC 更新）
4. 策略随着干预数据不断积累而持续改进

**与 HIL-SERL 的区别**：

| | HIL-SERL | HG-DAgger |
|---|---|---|
| 训练目标 | 最大化累积 reward（RL） | 模仿人类动作（BC） |
| 需要 reward 信号 | ✅ 是 | ❌ 否 |
| Critic / Q 函数 | ✅ 有 | ❌ 无 |
| 人类干预数据用途 | 干预 buffer（类 demo 角色） | **唯一训练数据** |
| 策略架构 | SAC（Actor-Critic） | SAC 网络结构，只训练 Actor |
| Gripper 训练方式 | DQN（grasp_critic） | 交叉熵 BC（模仿 demo gripper 动作） |

---

## 整体流程

```
步骤 1  采集人类演示数据（record_demos_octo_manual_new.py）
           ↓
步骤 2  合并演示数据为单个 pkl 文件（merge_trajectories.py）
           ↓
步骤 3  Phase 1：离线 BC 预热（只跑 Learner，不跑 Actor）
           ↓
步骤 4  Phase 2：在线 DAgger（Learner + Actor 同时跑）
```

> **关键**：步骤 3 和步骤 4 都在**同一个脚本**中完成，通过 `--pretrain_steps` 参数控制两阶段的切换，无需手动拷贝 checkpoint。

---

## 步骤 1：采集人类演示数据

HG-DAgger 的 Phase 1 BC 预热需要一定数量的演示数据。
推荐采集 **20 条**成功轨迹（与 ConRFT 共用同一个脚本）。

```bash
cd /path/to/HILRL-A1X

python examples/record_demos_octo_manual_new.py \
    --exp_name <你的实验名，例如 a1x_pick_banana> \
    --successes_needed 20 \
    --demo_data_subdir $(date +%Y%m%d)
```

**操作说明**：
- 使用 Gello 遥控机器人完成任务
- 每次 episode 结束后自动判断成功
- 按 `s` 键手动标记成功，按 `f` 键丢弃当前轨迹重采
- 采集完成后脚本自动处理数据（耗时较长，请耐心等待）

**输出位置**：
```
examples/experiments/<exp_name>/demo_data/<日期>/traj_001_<时间戳>.pkl
examples/experiments/<exp_name>/demo_data/<日期>/traj_002_<时间戳>.pkl
...
```

---

## 步骤 2：合并演示数据

Learner 通过 `--demo_path` 传入**单个 pkl 文件**，需要将多条轨迹合并。

```bash
cd /path/to/HILRL-A1X/examples

python merge_trajectories.py \
    experiments/<exp_name>/demo_data/<日期> \
    experiments/<exp_name>/demo_data/<日期>/traj_merged.pkl
```

> ⚠️ **注意**：HG-DAgger 的 demo 数据**不需要** Octo embedding（`mc_returns`、`embeddings` 字段），
> 脚本合并时如有这些字段会保留，没有也不影响训练。
> 只要有 `observations`、`actions`、`next_observations`、`rewards`、`masks` 即可。

---

## 步骤 3 + 4：启动训练

HG-DAgger 的两个阶段在同一个脚本内自动衔接：Learner 先跑完 Phase 1 离线 BC，然后启动
agentlace 服务等待 Actor 连接，进入 Phase 2 在线 DAgger。

### 启动 Learner（终端 1）

```bash
cd examples/experiments/<你的实验名>

export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=.8

python ../../train_hgdagger.py \
    --exp_name=<你的实验名> \
    --learner \
    --demo_path=./demo_data/<日期>/traj_merged.pkl \
    --checkpoint_path=./hgdagger/<日期> \
    --pretrain_steps=5000 \
    --mix_demo_ratio=0.5 \
    --debug=False
```

**Learner 的启动日志**：

```
[Phase 1] 离线 BC 预热：0 → 5000 步，demo_buffer 大小=XXX
[Phase 1] pretrain BC: 100%|████████████| 5000/5000
[Phase 1] 完成，checkpoint 已保存 (step=5000)
[Phase 2] 在线 DAgger 启动，等待 Actor 推送干预数据...
[Phase 2] Filling intervention buffer: 0/100
```

**当 Learner 打印 `sent initial network to actor` 后，再启动 Actor。**

### 启动 Actor（终端 2）

```bash
cd examples/experiments/<你的实验名>

export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=.5

python ../../train_hgdagger.py \
    --exp_name=<你的实验名> \
    --actor \
    --checkpoint_path=./hgdagger/<日期>
    # 如果 Learner 在另一台机器，加：
    # --ip <Learner 机器的 IP>
```

Actor 启动后会自动连接 Learner，加载 Phase 1 训练好的初始策略并开始 rollout。

---

## 关键参数说明

### `--pretrain_steps`

Phase 1 离线 BC 的训练步数。

| 推荐值 | 场景 |
|--------|------|
| `0` | 跳过 Phase 1，直接进入在线 DAgger（不推荐，策略无初始能力） |
| `2000~5000` | demo 数据少（< 10 条），快速预热 |
| `5000~20000` | demo 数据充足（20 条），标准预热 |

如果 `--checkpoint_path` 目录中已有 step ≥ `pretrain_steps` 的 checkpoint，**自动跳过 Phase 1**，直接进入 Phase 2。这意味着续训时可以直接重新运行同样的命令，不会重复预训练。

### `--mix_demo_ratio`

Phase 2 在线训练时，每个训练 batch 中**离线 demo 数据的占比**。

| 值 | 含义 | 适用场景 |
|----|------|---------|
| `0.0` | 纯在线干预数据（标准 DAgger） | 干预数据充足时 |
| `0.5` | 50% demo + 50% 干预数据（推荐） | 默认，防止遗忘初始行为 |
| `0.8` | 80% demo + 20% 干预数据 | 干预数据很少时 |

**为什么混合有帮助**：DAgger 初期干预数据少，纯用干预数据会导致策略快速遗忘 Phase 1 学到的基础能力。混合 demo 数据是标准的 DAgger 数据聚合（aggregation）策略。

---

## 两阶段训练逻辑详解

### Phase 1：离线 BC 预热

```
demo_buffer（只读）
      ↓ 随机采样 batch_size 条
  bc_update_step
      ↓ 只更新 actor（EEF: pre-tanh MSE）
        可选更新 grasp_critic（gripper: 交叉熵）
      ↓ 每 checkpoint_period 步保存 checkpoint
  达到 pretrain_steps → 保存最终 checkpoint → 进入 Phase 2
```

### Phase 2：在线 DAgger

```
Actor 部署策略 rollout
      ↓ 人类随时接管（Gello 接管）
      ↓ 干预帧 → intvn_data_store → intvn_buffer（持续积累）
      ↓ 非干预帧 → data_store（仅存磁盘，不参与训练）

Learner 持续训练
      ↓ 从 intvn_buffer 采样 n_intvn 条
        + 从 demo_buffer 采样 n_demo 条（mix_demo_ratio > 0 时）
      ↓ bc_update_step（同 Phase 1）
      ↓ 每 steps_per_update 步同步网络到 Actor
```

### BC 损失函数

**EEF（连续，前 6 维）**：pre-tanh 空间 MSE

$$L_{\text{EEF}} = \mathbb{E}\left[\|\mu_\theta(s) - \text{atanh}(\text{clip}(a_{\text{demo}}^{:6}))\|^2\right]$$

- 在 tanh 变换前的"原始空间"计算均值偏差，避免 `log_prob` 损失中 $1/\sigma^2$ 带来的梯度不稳定

**Gripper（离散，最后 1 维，仅 `learned-gripper` 模式）**：交叉熵

$$L_{\text{gripper}} = \text{CrossEntropy}(Q_{\text{grasp}}(s),\ \text{round}(a_{\text{demo}}^{-1}) + 1)$$

- `grasp_critic` 输出 3 类 Q 值（对应 gripper 动作 `{-1, 0, 1}` → 类别 `{0, 1, 2}`）
- BC 目标：让 `grasp_critic` 在 demo 状态下给出 demo gripper 动作对应的最高 Q 值

---

## 磁盘目录结构

```
./hgdagger/<日期>/
├── checkpoint_5000/        # Phase 1 完成时保存（step = pretrain_steps）
├── checkpoint_10000/       # Phase 2 中按 checkpoint_period 保存
├── checkpoint_20000/
├── buffer/                 # Actor 所有 transition（含策略动作帧，仅存档）
│   ├── transitions_1000.pkl
│   ├── transitions_2000.pkl
│   └── ...
└── demo_buffer/            # Actor 人类干预 transition（用于 BC 训练）
    ├── transitions_1500.pkl
    └── ...
```

> `demo_buffer/` 下的文件是 Phase 2 的**核心训练数据**，续训时会自动恢复加载到 `intvn_buffer`。

---

## 监控训练进度

训练脚本默认启用 WandB（`--debug=False`）：

```bash
wandb login  # 首次运行需要登录
```

WandB 项目名为 `hg-dagger`，重点关注：

| 指标 | 前缀 | 健康状态 |
|------|------|---------|
| `eef_bc_loss` | `pretrain/` / `online/` | 应持续下降 |
| `eef_action_l2` | `pretrain/` / `online/` | 应持续减小（策略动作与人类动作的 L2 距离）|
| `gripper_bc_loss` | `pretrain/` / `online/` | 应下降（仅 learned-gripper 模式）|
| `gripper_accuracy` | `pretrain/` / `online/` | 应升高，健康值 > 0.8 |
| `intvn_buffer_size` | `online/` | 应随时间增长 |
| `episode/return` | `environment` | 应随干预数据积累而提升 |
| `intervention_rate` | `environment` | 应随训练进行而降低（策略越来越好，人类越来越少接管）|

调试阶段不想使用 WandB 时，加 `--debug=True`。

---

## 参数速查表

| 参数 | 推荐值 | 说明 |
|------|--------|------|
| `--pretrain_steps` | 5000~20000 | Phase 1 离线 BC 步数；已有 ckpt 时自动跳过 |
| `--mix_demo_ratio` | 0.5 | Phase 2 batch 中 demo 数据占比，`[0.0, 1.0)` |
| `--seed` | 42 | 随机种子 |
| `--debug` | False | True 时关闭 WandB |
| `XLA_MEM_FRACTION` | .8（Phase 1）/ .5（Phase 2 各自）| JAX 显存预分配比例 |

`config.py` 中对 HG-DAgger 影响较大的配置项：

| 配置项 | 推荐值 | 说明 |
|--------|--------|------|
| `batch_size` | 256 | 每步 BC 更新的 batch 大小 |
| `training_starts` | **20~100** | Phase 2 等待 `intvn_buffer` 积累的最小条数；建议设小，DAgger 干预数据比 RL 慢 |
| `steps_per_update` | 50 | 每隔多少步同步网络到 Actor |
| `checkpoint_period` | 5000 | 每隔多少步保存 checkpoint |
| `max_steps` | 1,000,000 | Phase 2 最大训练步数 |
| `replay_buffer_capacity` | 200,000 | `intvn_buffer` 容量上限（DAgger 聚合所有历史干预数据）|

---

## 常见问题

### Q: Phase 1 结束后，Actor 怎么加载训练好的权重？

Phase 1 结束时，Learner 自动将 checkpoint 保存到 `--checkpoint_path`，然后进入 Phase 2 并通过 agentlace 发送 `sent initial network to actor`。Actor 启动后连接 Learner，自动接收最新网络参数，无需手动加载 checkpoint。

### Q: Learner 停在 `Filling intervention buffer` 很久？

这是正常的——Phase 2 只有当人类**主动接管**策略时才会有干预数据写入 `intvn_buffer`。如果策略已经能较好地完成任务（Phase 1 预热效果好），人类接管频率会很低。

解决方法：
1. 降低 `config.training_starts`（如从默认 100 降到 20）
2. 在 Actor 侧多接管几次，积累初始干预数据
3. 如果策略完全不动（没有基础能力），先增加 `--pretrain_steps`

### Q: Learner 打印 `sent initial network to actor` 前，Actor 可以先启动吗？

不推荐。Actor 的 `wait_for_server=True` 会阻塞到 Learner 服务就绪，但第一次网络同步需要等到 Phase 1 结束。建议按顺序：**等 Learner 打印 `sent initial network to actor` 后再启动 Actor**。

### Q: 续训时（已有 checkpoint）如何操作？

直接重新运行相同的 Learner 和 Actor 命令，脚本会自动：
1. 检测到 `--checkpoint_path` 中已有 checkpoint，提示按 Enter 确认续训
2. 恢复 `intvn_buffer`（从 `demo_buffer/` 目录加载历史干预数据）
3. 检测 `start_step >= pretrain_steps`，自动跳过 Phase 1

### Q: 如何区分 Phase 1 和 Phase 2 保存的 checkpoint？

- Phase 1 结束时固定保存 `checkpoint_{pretrain_steps}/`（如 `checkpoint_5000/`）
- Phase 2 按 `config.checkpoint_period` 间隔保存，步数从 `pretrain_steps` 继续计数

可以通过 checkpoint 步数判断：步数 = `pretrain_steps` 的是 Phase 1 产物，更大步数的是 Phase 2 产物。

### Q: 为什么不需要 Octo embedding？

HG-DAgger 使用的是与 HIL-SERL 完全相同的 SAC 网络结构（图像 CNN encoder + MLP policy），直接从图像观测中学习策略，不依赖 Octo 提取的语言条件 embedding。
采集 demo 数据时用 `record_demos_octo_manual_new.py`，即使该脚本会提取并存储 embedding，HG-DAgger 训练时也会忽略这些字段，只使用 `observations`、`actions` 等基础字段。

### Q: 显存 OOM？

Phase 1 只跑 Learner，显存宽松，`XLA_MEM_FRACTION=.8` 没问题。
Phase 2 Learner 和 Actor 同卡运行时，各设 `.5`：
```bash
export XLA_PYTHON_CLIENT_MEM_FRACTION=.5
```
如果仍然 OOM，降低到 `.4` 或减小 `config.batch_size`。

### Q: `training_starts` 应该设多少？

DAgger 的干预数据只在人类接管时才有，填充速度远慢于 RL 的 replay buffer。
建议在 `config.py` 中为 HG-DAgger 专门设置较小的值：

```python
# config.py 中针对 HG-DAgger 可覆盖
training_starts = 50  # 而非默认的 100
```