# ConRFT 复现指南

> **适用算法**：ConRFT（Consistency Policy + Cal-QL）  
> **训练脚本**：`examples/train_conrft_octo.py`（不启用 Cov Actor Loss，不启用采样策略）  
> **适用平台**：A1X 单臂机器人  
> **前置条件**：已完成环境安装（conda 环境 `conrft` 可用，ROS2 已启动，相机序列号已配置）

---

## 整体流程

```
步骤 1  采集人类演示数据（record_demos_octo_manual_new.py）
           ↓
步骤 2  合并演示数据为单个 pkl 文件（merge_trajectories.py）
           ↓
步骤 3  新建实验配置（config.py）
           ↓
步骤 4  预训练（learner 只跑，actor 不跑）
           ↓
步骤 5  在线 RL 微调（learner + actor 同时跑）
```

**关于 Octo 模型**：脚本默认自动从 HuggingFace 下载 `rail-berkeley/octo-small-1.5`。
如果机器无法访问外网，需要提前下载到本地（见步骤 3 注意事项）。

---

## 步骤 1：采集人类演示数据

ConRFT 需要一定数量的人类演示来完成预训练阶段（Cal-QL 中的 MC return 下界）。
推荐采集 **20 条**成功轨迹。

```bash
cd /path/to/HILRL-A1X

python examples/record_demos_octo_manual_new.py \
    --exp_name <你的实验名，例如 a1x_pick_banana> \
    --successes_needed 20 \
    --demo_data_subdir $(date +%Y%m%d)
```

**操作说明**：
- 用 spacemouse 遥控机器人完成任务
- 按 `s` 键手动标记成功，按 `f` 键丢弃当前轨迹重采
- 采集完成后脚本自动提取 Octo embedding（耗时较长，请耐心等待）

**输出位置**：
```
examples/experiments/<exp_name>/demo_data/<日期>/traj_001_<时间戳>.pkl
examples/experiments/<exp_name>/demo_data/<日期>/traj_002_<时间戳>.pkl
...
```

---

## 步骤 2：合并演示数据

Learner 启动时通过 `--demo_path` 传入**单个 pkl 文件**，因此需要将多条轨迹合并。

```bash
cd /path/to/HILRL-A1X/examples

python merge_trajectories.py \
    experiments/<exp_name>/demo_data/<日期> \
    experiments/<exp_name>/demo_data/<日期>/traj_merged.pkl
```

**验证合并结果**：
脚本运行完会自动打印 transitions 数量和字段信息。
正常输出应包含 `embeddings`、`next_embeddings`、`mc_returns` 三个字段，缺少任何一个则预训练会失败。

---

## 步骤 3：准备 Octo 模型（无外网时）

如果服务器无法访问 HuggingFace，需要在有网络的机器上提前下载：

```bash
# 在有网络的机器上
python -c "
from octo.model.octo_model import OctoModel
model = OctoModel.load_pretrained('hf://rail-berkeley/octo-small-1.5')
model.save_pretrained('/path/to/save/octo-small-1.5')
"
```

然后将下载好的文件夹复制到训练服务器，并在 `config.py` 中修改 `octo_path`。

---

## 步骤 4：预训练（Cal-QL on Demo Data）

预训练阶段**只启动 Learner，不启动 Actor**。对应脚本：
`examples/experiments/a1x_pick_banana/run_learner_conrft_pretrain.sh`

**在移植到新任务时，修改脚本里的以下两处路径再运行**：

```bash
# run_learner_conrft_pretrain.sh（只需修改带箭头的行）
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH && \
export XLA_PYTHON_CLIENT_PREALLOCATE=false && \
export XLA_PYTHON_CLIENT_MEM_FRACTION=.8 && \
python ../../train_conrft_octo.py "$@" \
    --exp_name=a1x_pick_banana \
    --q_weight=0.1 \
    --bc_weight=1.0 \
    --checkpoint_path=./conrft/0225 \           # ← 修改为你的预训练保存目录（例如按日期命名）
    --demo_path=./demo_data/20260225/traj_20_0225.pkl \  # ← 修改为步骤 2 的合并 pkl 路径
    --pretrain_steps=100000 \
    --debug=False \
    --learner
```

修改完成后运行：

```bash
cd examples/experiments/<你的实验名>
bash run_learner_conrft_pretrain.sh
```

**关键参数说明**：

| 参数 | 预训练推荐值 | 说明 |
|------|-------------|------|
| `--pretrain_steps` | 100000 | 预训练总步数，步数太少 Q 函数学不好 |
| `--q_weight` | 0.1 | 预训练阶段以 BC 为主，Q 权重低 |
| `--bc_weight` | 1.0 | 预训练阶段 BC 重建损失权重高 |
| `--checkpoint_path` | 自定义（如 `./conrft/0225`）| 预训练 checkpoint 单独目录，**不要与在线训练共用** |
| `XLA_MEM_FRACTION` | .8 | 预训练只跑 Learner，显存可分配大一些 |

**预训练结束标志**：终端打印 `Pretraining done`，脚本自动退出。

---

## 步骤 5：在线 RL 微调

### 5.0 重要：为在线训练准备 checkpoint 目录

**预训练和在线训练使用两个独立的 checkpoint 目录**。在线训练启动时，脚本会检查目标目录内已有的 checkpoint 步数，如果 `已有步数 >= pretrain_steps`，则自动跳过预训练、直接加载权重进入在线 RL。

具体做法：**将预训练 checkpoint 目录复制一份，作为在线训练的起始点**：

```bash
cd examples/experiments/<你的实验名>

# 将预训练结果复制到新目录（在线训练目录）
cp -r ./conrft/0225 ./conrft/0226   # 名称任意，建议按日期
```

之后在 `run_learner_conrft.sh` 和 `run_actor_conrft.sh` 中将 `--checkpoint_path` 指向这个新目录（`./conrft/0226`），并将 `--pretrain_steps` 设为**小于预训练步数**的值（如 `96000 < 100000`），脚本就会跳过预训练直接进入在线 RL。

> 💡 **为什么要拷贝而不是直接用同一个目录**？  
> 在线训练会继续在该目录写入新的 checkpoint（step 100001, 110000, …），  
> 如果与预训练目录混用，后续无法区分哪些 checkpoint 来自预训练、哪些来自在线训练，  
> 也无法回退到纯预训练状态。

---

### 方式 A：使用 `launch_conrft.sh`（推荐，tmux 自动分屏）

`launch_conrft.sh` 只负责**在线 RL 阶段**（Learner + Actor 同时启动），不包含预训练。

修改脚本里的两个硬编码路径（约第 60 行）：

```bash
# launch_conrft.sh
CKPT_PATH="<你的工作区绝对路径>/examples/experiments/<你的实验名>/conrft/0226"
DEMO_PATH="<你的工作区绝对路径>/examples/experiments/<你的实验名>/demo_data/<日期>/traj_merged.pkl"
```

然后运行：

```bash
cd examples/experiments/<你的实验名>
bash launch_conrft.sh
```

脚本会自动生成带时间戳的 `RUN_TAG`，在 tmux session 中左右分屏：左侧 Learner，右侧 Actor。

tmux 操作提示：
- 切换 pane：`Ctrl-b ←/→`
- Detach（后台保持运行）：`Ctrl-b d`
- 恢复查看：`tmux attach-session -t cr_a1x_pick_banana`

---

### 方式 B：手动两个终端（对应 `run_learner_conrft.sh` + `run_actor_conrft.sh`）

修改两个脚本中的 `--checkpoint_path` 和 `--demo_path` 后，分别在两个终端运行：

**修改 `run_learner_conrft.sh`（在线训练 Learner）**：

```bash
# run_learner_conrft.sh（需修改带箭头的行）
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH && \
export XLA_PYTHON_CLIENT_PREALLOCATE=false && \
export XLA_PYTHON_CLIENT_MEM_FRACTION=.5 && \
python ../../train_conrft_octo.py "$@" \
    --exp_name=a1x_pick_banana \
    --checkpoint_path=./conrft/0226 \          # ← 指向拷贝后的新目录
    --q_weight=1.0 \                            # ← 在线阶段 Q 权重调高
    --bc_weight=0.1 \                           # ← BC 权重调低
    --demo_path=./demo_data/20260225/traj_20_0225.pkl \  # ← 与预训练相同的 demo 文件
    --pretrain_steps=96000 \                    # ← 小于预训练的 100000，触发自动跳过
    --debug=False \
    --learner
```

**修改 `run_actor_conrft.sh`（在线训练 Actor）**：

```bash
# run_actor_conrft.sh（需修改带箭头的行）
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH && \
export XLA_PYTHON_CLIENT_PREALLOCATE=false && \
export XLA_PYTHON_CLIENT_MEM_FRACTION=.5 && \
python ../../train_conrft_octo.py "$@" \
    --exp_name=a1x_pick_banana \
    --checkpoint_path=./conrft/0226 \          # ← 与 Learner 相同
    --actor
    # 如果 Learner 在另一台机器，取消注释并修改：
    # --ip <Learner 机器的 IP>
```

**启动顺序**：

```bash
# 终端 1：先启动 Learner
cd examples/experiments/<你的实验名>
bash run_learner_conrft.sh

# 终端 2：等 Learner 打印 "sent initial network to actor" 后再启动 Actor
cd examples/experiments/<你的实验名>
bash run_actor_conrft.sh
```

**Q/BC 权重的两阶段逻辑**：

| 阶段 | `q_weight` | `bc_weight` | 原因 |
|------|-----------|------------|------|
| 预训练 | 0.1 | 1.0 | 以 BC 为主，先让策略学会完成任务 |
| 在线 RL | 1.0 | 0.1 | 以 Q 值优化为主，通过 RL 提升成功率 |

---

## 步骤 6：监控训练进度

### 查看 WandB 日志

训练脚本默认启用 WandB（`--debug=False`）。首次运行会提示登录：

```bash
wandb login
```

在 WandB 项目 `conrft` 下可以看到训练曲线，重点关注：

| 指标 | 健康状态 |
|------|---------|
| `rewards` | 应随训练逐渐升高 |
| `bc_loss` | 预训练阶段应单调下降 |
| `critic_loss` | 应稳定下降，不应爆炸 |
| `predicted_qs` | 预训练后应接近 MC return 量级 |
| `cql_diff` | 应为正值（保守性约束生效） |
| `episode/return` | 在线阶段应持续提升 |

如果不想用 WandB（调试阶段），加 `--debug=True`。

### 查看 txt 日志

训练指标也会保存到本地文件：

```
./conrft/pretrain_run1/logs/training_metrics_<时间戳>.txt
```

### 查看 checkpoint

```
./conrft/pretrain_run1/
├── checkpoint_10000/    # Flax checkpoint
├── checkpoint_20000/
├── buffer/              # Actor 写入的在线 replay buffer（pkl）
├── demo_buffer/         # 干预数据
└── logs/                # txt 日志
```

---

## 评估训练好的策略

```bash
cd examples/experiments/<你的实验名>

python ../../train_conrft_octo.py \
    --exp_name=<你的实验名> \
    --checkpoint_path=./conrft/pretrain_run1 \
    --eval_checkpoint_step=200000 \
    --eval_n_trajs=20 \
    --actor
```

这会加载指定步数的 checkpoint，运行 20 条轨迹，打印成功率和平均 episode 时长。

---

## 常见问题

### Q: Learner 启动后一直等待 `"Filling up replay buffer"`，Actor 却没有数据进来？

检查：
1. Actor 是否成功连接到 Learner？Actor 终端应打印 `"Connected to Learner"`
2. `--ip` 参数是否正确？如果不在同一台机器，Actor 需要加 `--ip <Learner IP>`
3. 防火墙是否放行了通信端口？

### Q: 预训练很快结束（几百步就退出）？

检查 `--pretrain_steps` 是否小于当前 checkpoint 步数。如果 `checkpoint_path` 中已有步数更大的 checkpoint，脚本会认为预训练已完成，直接跳过。

解决方法：删除 checkpoint 目录或换一个新的 `checkpoint_path`。

### Q: Actor 端出现 `"EEF 动作全0"` 警告？

说明 Gello 没有发送有效动作。检查：
1. Gello 是否已连接并正常运行？
2. `config.py` 中 `gello_port` 或 `gello_config_path` 是否正确？

### Q: 预训练后在线微调成功率不升，甚至下降？

可能原因：
1. `q_weight` 太大，导致策略迅速偏离 demo 分布。尝试降低到 `0.5`
2. demo 数据质量差（有大量零动作帧）。重新采集更干净的 demo
3. `pretrain_steps` 不够。增大到 100000 以上

### Q: 显存 OOM？

降低 `XLA_PYTHON_CLIENT_MEM_FRACTION`，例如 `0.4`。Learner 和 Actor 分别占用一部分显存，如果在同一 GPU 上运行，总占用不能超过 GPU 容量。

### Q: 想在不同任务之间共享 Octo 权重，避免重复下载？

在 `config.py` 中将 `octo_path` 指向同一个本地路径即可，多个实验可以共用。

---

## 参数速查表

| 参数 | 预训练推荐值 | 在线训练推荐值 | 说明 |
|------|------------|--------------|------|
| `--pretrain_steps` | 50000~100000 | 同预训练（自动跳过） | 预训练总步数 |
| `--q_weight` | 0.1 | 1.0 | Actor Q-loss 权重 |
| `--bc_weight` | 1.0 | 0.1 | Actor BC 重建损失权重 |
| `--seed` | 42 | 42 | 随机种子 |
| `--debug` | False | False | True 时关闭 WandB |
| `--use_cov_actor_loss` | False | False | **默认 False**，保持原始 Cal-QL；True 时启用 Cov Actor Loss 变体 |
| `--sampling_strategy` | `none` | `none` | **默认 none**，保持原始 Cal-QL；可选 `workspace_filtering`、`random_drop`、`per` |
| `XLA_MEM_FRACTION` | 0.8（Learner）| 0.5（各自）| JAX 显存预分配比例 |

`config.py` 中的关键默认值（来自 `DefaultTrainingConfig`）：

| 配置项 | 默认值 | 说明 |
|--------|--------|------|
| `max_steps` | 1,000,000 | 总训练步数 |
| `batch_size` | 256 | 每步训练的 batch 大小 |
| `cta_ratio` | 2 | Critic 更新次数 / Actor 更新次数（高 UTD）|
| `training_starts` | 100 | Replay buffer 至少积累这么多条才开始训练 |
| `replay_buffer_capacity` | 200,000 | Replay buffer 容量上限 |
| `steps_per_update` | 50 | 每隔多少步向 Actor 同步一次网络参数 |
| `discount` | 0.98 | 在 `a1x_pick_banana/config.py` 中已覆盖 |
