# HIL-SERL-BC 算法移植实现指南

> **受众**：在新平台/新代码库上从零移植本算法的合作者。  
> **基线**：本算法以标准 HIL-SERL（Actor-Learner 异步框架 + SAC/RLPD）为基础，
> 增加三个独立模块。  
> **版本**：本文档描述 BC-loss 版（`train_rlpd_hil_bc.py`），
> Contrastive-loss 版（`train_rlpd_hil.py`）只在 Module 3 实现上不同，其余完全一致。

---

## 目录

0. [完整训练流程总览（三步）](#0-完整训练流程总览三步)
1. [算法概览](#1-算法概览)
2. [Module 1 — 次优片段识别](#2-module-1--次优片段识别)
3. [Module 2 — Q 值修正（偏好感知 Critic）](#3-module-2--q-值修正偏好感知-critic)
4. [Module 3 — 行为克隆策略损失（BC 版）](#4-module-3--行为克隆策略损失bc-版)
5. [数据存储结构与字段规范](#5-数据存储结构与字段规范)
6. [Progress Model 外部依赖规范](#6-progress-model-外部依赖规范)
7. [Actor 循环改动清单](#7-actor-循环改动清单)
8. [Learner 循环改动清单](#8-learner-循环改动清单)
9. [超参数参考表](#9-超参数参考表)
10. [监控指标清单](#10-监控指标清单)
11. [移植检查清单](#11-移植检查清单)

---

## 0. 完整训练流程总览（三步）

在开始策略训练之前，需要先完成以下两个前置步骤。三步之间存在严格的数据依赖关系：

```
步骤 1：采集人类演示数据
   record_demos_octo_manual_new.py
   → examples/experiments/<exp_name>/demo_data/<subdir>/traj_*.pkl
             │
             ▼
步骤 2：训练 Progress Model
   train_progress_model.py
   → progress_model_best.pt
   → state_stats.pt
             │
             ▼
步骤 3：训练 HIL-SERL-BC 策略（本文档主体）
   train_rlpd_hil_bc.py
   使用上面两步的输出进行次优片段识别（Module 1）
```

---

### 步骤 1：采集人类演示数据

**脚本**：`examples/record_demos_octo_manual_new.py`

**典型调用**：

```bash
cd <workspace_root>
python examples/record_demos_octo_manual_new.py \
    --exp_name <实验名，需存在于 CONFIG_MAPPING> \
    --successes_needed 20 \
    --demo_data_subdir 20260208
```

**关键参数**：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--exp_name` | **必填** | 实验名，对应 `experiments/mappings.py` 中的 `CONFIG_MAPPING` key |
| `--successes_needed` | 20 | 需要采集的成功轨迹条数 |
| `--demo_data_subdir` | `20260208` | 保存到 `demo_data/` 下的子目录名（建议用日期） |
| `--reward_scale` | 1.0 | 奖励缩放系数 |
| `--reward_bias` | 0.0 | 奖励偏置 |

**流程说明**：

1. 脚本启动后等待机器人准备就绪，按提示用 Gello 控制机器人完成任务
2. 每次 episode 结束自动判断成功（或按 `s` 键手动标记成功，按 `f` 键标记失败丢弃）
3. **采集阶段**：禁用 action chunking，记录单步 EEF 动作（`info["intervene_action_eef"]`）
4. **后处理阶段**（采集完成后自动执行，无需人工干预）：
   - 重组为 action chunks（若 `config.action_chunk_size` 非 None）
   - 批量提取 Octo 图像 embedding（`add_embeddings_to_trajectory`）
   - 计算 MC returns
5. 每条成功轨迹保存为独立 pkl 文件：`demo_data/<subdir>/traj_NNN_<timestamp>.pkl`

**输出文件格式**：每个 `traj_*.pkl` 是一个 `list[dict]`，每条 `dict` 为单步 transition，
包含 `observations`、`actions`（可能为 chunked）、`next_observations`、`rewards`、
`masks`、`dones`、`infos`，以及 Octo embedding 相关字段。

**移植注意事项**：

- 干预动作键名 `"intervene_action_eef"` 依赖平台 wrapper，移植时需确认对应 key
- 图像 key（`side_policy_256`、`wrist_1`）需与 `train_progress_model.py` 中的 `SIDE_KEY`、`WRIST_KEY` 常量保持一致
- 若平台无 Octo 模型，可去掉 `add_embeddings_to_trajectory` 步骤，但需同步修改策略训练脚本中对 embedding 字段的依赖

---

### 步骤 2：训练 Progress Model

**脚本**：`examples/train_progress_model.py`

**典型调用**：

```bash
python examples/train_progress_model.py \
    --demo_dir  examples/experiments/<exp_name>/demo_data/<subdir> \
    --buffer_dir <已有 replay buffer 路径，用于评估> \
    --output_dir VF_training/runs/progress_model \
    --epochs 300 \
    --hidden_dim 128 \
    --device cuda
```

**关键参数**：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--demo_dir` | — | 步骤 1 输出的 `traj_*.pkl` 所在目录 |
| `--buffer_dir` | — | 已有 replay buffer 目录（用于 Phase 3 评估可视化，可选）|
| `--output_dir` | `VF_training/runs/progress_model` | 模型和日志输出目录 |
| `--cache_path` | 自动在 `output_dir` 下生成 | Phase 1 ResNet 特征缓存路径；已存在则直接复用 |
| `--epochs` | 300 | 训练轮数 |
| `--hidden_dim` | 128 | `ProgressHead` fusion MLP 的隐层宽度（**必须与推理时一致**）|
| `--lr` | 3e-4 | 学习率 |
| `--lambda_prog` | 1.0 | 线性进度监督损失权重 |
| `--lambda_mc` | 0.1 | MC return 回归损失权重 |
| `--lambda_td` | 0.1 | TD 一致性损失权重 |
| `--mc_last_n` | 5 | MC 损失仅作用于每条轨迹最后 N 步（0=全部） |
| `--device` | `cpu` | 推荐 `cuda` 加速 Phase 1 特征提取 |
| `--anomaly_window` | 4 | 异常检测滑窗（同第 9 节超参）|
| `--delta_reg` | 0.045 | 回退检测阈值（同第 9 节超参）|
| `--recovery_k` | 3 | 恢复确认窗口（同第 9 节超参）|

**三阶段流程**：

| 阶段 | 内容 |
|------|------|
| **Phase 1**（特征提取） | 用冻结的 ResNet-18 批量提取所有 demo 帧的 side/wrist 图像特征，缓存到 `demo_features.pt`；**若缓存已存在则跳过**，节省重复运行时间 |
| **Phase 2**（MLP 训练） | 固定 ResNet-18，仅训练 `ProgressHead`；监督信号为线性进度标签 + MC return + TD 一致性三项联合损失 |
| **Phase 3**（评估）| 对 `buffer_dir` 中成功轨迹运行推理，生成进度曲线可视化图（`eval_visuals/`）|

**输出文件**（均保存在 `output_dir/exp_NNN_<timestamp>/` 下）：

| 文件 | 用途 |
|------|------|
| `progress_model_best.pt` | **策略训练必须**，`ProgressHead` 的最优权重 (`state_dict`) |
| `state_stats.pt` | **策略训练必须**，状态差值归一化统计 `{"mean": ..., "std": ...}` |
| `demo_features.pt` | Phase 1 特征缓存，可复用 |
| `training_curves.png` | 训练损失曲线 |
| `demo_progress_curves.png` | Demo 轨迹进度预测可视化 |
| `eval_visuals/` | Buffer 成功轨迹的进度预测 + 异常段标注图 |

**移植注意事项**：

- 脚本顶部的 `SIDE_KEY`、`WRIST_KEY` 常量（默认 `"side_policy_256"`、`"wrist_1"`）必须与平台 obs 字段名一致
- `STATE_DIM`（默认 7）若平台本体状态维度不同，需同步修改并确保 `state_stats.pt` 对应维度
- `hidden_dim` 训练时指定后须记录，推理时（第 6 节 `ProgressModelRunner`）需传入相同值
- Phase 3 评估需要已有 buffer（`buffer_dir`），若无可跳过（不影响模型产出）

---

### 步骤 3：策略训练

完成步骤 1–2 后，将 `progress_model_best.pt` 和 `state_stats.pt` 路径配置到训练脚本，
按本文档第 2–11 节进行移植和训练。

**步骤 2 输出文件 → 策略训练的对应配置项**：

```
progress_model_best.pt  →  FLAGS.progress_model_path
state_stats.pt          →  FLAGS.state_stats_path（或同目录自动加载）
hidden_dim（记录值）     →  FLAGS.progress_hidden_dim
```

---

## 1. 算法概览

### 1.1 与 HIL-SERL 基线的关系

```
HIL-SERL 基线
│  actor 与 learner 异步运行
│  SpaceMouse / teleoperation 提供干预动作
│  干预动作覆盖策略动作，同时写入 demo buffer（50/50 RLPD 采样）
│
└── 本算法在基线基础上增加三个模块：
    ├── Module 1：识别"次优片段"（不良动作的起始位置）
    ├── Module 2：对这些次优片段修正 Q 值目标（Critic 改动）
    └── Module 3：用人类干预动作引导策略（Actor 改动）
```

### 1.2 三个模块的数据流关系

```
Actor 端
─────────────────────────────────────────────────────────────────
step_env()
│
├─ 若发生干预（info["intervene_action_eef"] 存在）
│    └── 记录干预起始时刻 t_i，将 (s_{t_i}, a^h, a^π_old) 写入 preference_buffer
│                                                      ↑ Module 2 & 3 所需数据
│
├─ 每步 transition 先缓存到 episode_buffer（不立即写 replay buffer）
│       因为 alpha_weight 需等 episode 结束后才能确定
│
└─ episode 结束时：
     ├── [Module 1] 调用 detect_suboptimal_segments()
     │       输入：episode_buffer, intervention_markers
     │       输出：[(t_a, t_i), ...]  次优片段起止索引列表
     │
     ├── [Module 2] 调用 compute_episode_alpha_and_segment_ids()
     │       输入：episode_len, suboptimal_segs, segment_uids, λ
     │       输出：alpha_weights[T], segment_ids[T]  逐步衰减权重
     │
     └── 批量写入 replay buffer（含真实 alpha_weight 和 segment_ids）

Learner 端
─────────────────────────────────────────────────────────────────
每训练步：
├── 从 replay_buffer + demo_buffer 各采 batch_size/2 条 → concat → batch
│
├── [Module 2] get_by_segment_ids(batch["segment_ids"])
│       → matched_pref：与 batch 等长，逐样本对齐的偏好数据（无匹配位置补零）
│
├── [Module 2] 用 target network 计算 A_cf = max(0, Q(s,a^π_old) - Q(s,a^h))
│
├── [Module 2] corrected_critic_loss_fn：修正 Bellman 目标
│       ỹ_t = y_t - α(t) · A_cf(t)
│
├── [Module 3] preference_buffer.sample(N) → preference_batch_direct（独立采样）
│
└── [Module 3 BC] bc_policy_loss_fn：
        L_actor = L_RLPD + β · E[||μ_θ(s) - atanh(a^h)||²]
```

---

## 2. Module 1 — 次优片段识别

### 2.1 目标

给定一个 episode 内的若干干预事件，确定每次干预对应的**次优行为起始时刻** $t_a$，即策略开始出错的那一步（而不是被人类纠正的那一步 $t_i$）。

片段 $[t_a, t_i]$ 内的所有 transition 都将被标记为次优并施加 Q 值修正（Module 2）。

### 2.2 两种实现模式

#### 模式 A：Fallback（无 Progress Model）

```python
for marker in intervention_markers:
    t_i = marker["step_idx"]          # 干预起始步（episode 内索引）
    t_a = max(0, t_i - window)         # 向前回溯固定步数
    segments.append((t_a, t_i))
```

`window` 是一个超参数（默认 5），直接决定次优片段长度。

#### 模式 B：Progress Model + 滑窗异常检测

需要预训练的 Progress Model（见第 6 节）。算法流程：

1. 对整条 episode 的每一帧调用 `ProgressModelRunner.infer_episode(episode_buffer)` → 得到 `preds: (T+1,) float32`，表示每帧的任务进度估计 $P(t) \in [0,1]$
2. 对 `preds` 序列调用 `detect_anomalies()` → 得到异常段列表 `[{"start", "end", "type", ...}]`
3. 对每个干预点 $t_i$，找最近的已触发异常段起点作为 $t_a$；若无匹配则退回 fallback

### 2.3 `detect_anomalies` 算法细节

基于**锚点扩展 + 恢复确认**：

1. 在每个 RL 帧 $t$ 上检查初始窗口 $[t, t+W]$ 内的进度变化量 $\Delta P$
   - 回退：$\Delta P < -\delta_{reg}$（默认 0.045）
   - 停滞：$|\Delta P| < \delta_{stag}$（默认 0.001）
2. 若触发异常，以 $P[t]$ 为锚点向后扩展，直到**连续 $K$ 帧**（默认 3）均 $\geq P_{anchor}$ 才确认恢复
3. Human 帧（label=2）不参与检测，遇到立即截断异常段扩展

### 2.4 `intervention_markers` 数据结构

每个干预事件在干预**首次发生**时记录一条：

```python
{
    "step_idx":      int,         # 该干预起始步在当前 episode_buffer 中的索引
    "segment_id":    int,         # 全局唯一 segment UID（跨 episode 单调递增）
    "obs":           np.ndarray,  # 干预时刻的观测 s_{t_i}（原始格式，与 replay buffer 一致）
    "human_action":  np.ndarray,  # 操作员给出的替代动作 a^h，shape (action_dim,)
    "policy_action": np.ndarray,  # 干预发生前策略的实际采样动作 a^π_old，shape (action_dim,)
}
```

**重要**：连续干预期间（`already_intervened=True`）**不重复记录**，避免 preference buffer 被重复样本占满。只在 `already_intervened` 从 False 转为 True 时记录一次。

### 2.5 `compute_episode_alpha_and_segment_ids` 规范

```
输入：
  episode_len       : int            当前 episode 的总步数
  suboptimal_segs   : [(t_a, t_i)]   Module 1 输出的次优片段列表
  segment_uids      : [int]          每个片段对应的全局 segment UID
  lam (λ)           : float          位置感知衰减系数（默认 3.0）

输出：
  alpha_weights : np.ndarray (episode_len,) float32
      - t ∈ [t_a, t_i]  →  α(t) = exp(-λ * (t_i - t))
                            t_i 处 α=1.0，t_a 处 α≈0.05（λ=3）
      - 其他位置         →  α(t) = 0.0
      - 多片段重叠       →  取最大值

  segment_ids : np.ndarray (episode_len,) int32
      - 与最强 alpha_weight 对应的 segment UID
      - 无归属位置为 -1
```

**`flat_alpha_correction` 变体**（可选）：Learner 端在使用 `alpha_weights` 时可将所有 `alpha_weights > 0` 的位置统一置 1，相当于对整条次优片段施加等强度修正，不做位置衰减。这是一个运行时开关，不影响存储在 replay buffer 中的值。

---

## 3. Module 2 — Q 值修正（偏好感知 Critic）

### 3.1 核心公式

标准 Bellman 目标 $y_t = r_t + \gamma \min_k Q^{tgt}_k(s_{t+1}, a')$ 修正为：

$$\tilde{y}_t = y_t - \alpha(t) \cdot A_{cf}(t)$$

其中：

$$A_{cf}(t) = \text{stop\_gradient}\left( \text{valid\_mask}(t) \cdot \max\left(0,\ Q^{tgt}(s_{t_i}, a^{\pi}_{old}) - Q^{tgt}(s_{t_i}, a^h) \right) \right)$$

- $\alpha(t)$：来自 `replay_batch["alpha_weight"]`，次优片段内的位置感知衰减权重
- $A_{cf}(t)$：反事实优势，由 target network 在干预时刻状态上评估两种动作的 Q 差
- `valid_mask`：该 batch transition 是否在 preference buffer 中找到对应的 segment_id

### 3.2 逐样本对齐机制

**核心设计**：通过 `segment_id` 实现 replay batch 与 preference buffer 的**精确逐样本对齐**。

```
replay_buffer 中的 transition：
  batch["segment_ids"][i]  ──→  preference_buffer.get_by_segment_ids()
                                      │
                                      ↓
                              matched_pref[i] = {
                                  "observations":   s_{t_i},     # 干预时刻状态
                                  "human_actions":  a^h,
                                  "policy_actions": a^π_old,
                                  "valid_mask":     True/False
                              }
```

无匹配（segment_id = -1 或已被覆盖）的位置：`valid_mask[i] = False`，`A_cf[i] = 0.0`，不施加修正。

### 3.3 夹爪维度处理与算法覆盖边界（重要）

夹爪在不同平台/模式下的处理方式**本质不同**，直接影响 Module 2 和 Module 3 的覆盖范围。移植前必须明确所用平台属于哪种情况。

---

#### 情况 A：固定夹爪（`SACAgent` + `fix_gripper=True`）

适用于 `single-arm-fixed-gripper` / `dual-arm-fixed-gripper` 模式。

- 策略只输出 6 维 EEF 连续动作，夹爪维度固定（不被策略控制）
- `actions[..., -1]` 存储固定夹爪值，但 **Q 网络只接受 6 维连续动作**
- 计算 A_cf 时必须截掉最后一维，与 Q 网络输入维度对齐：

```python
# config["fix_gripper"] = True 时
h_actions = matched_pref["human_actions"][..., :-1]   # (B, 6)
p_actions = matched_pref["policy_actions"][..., :-1]   # (B, 6)
```

- Module 2 和 Module 3 均完整覆盖策略可控的所有维度

---

#### 情况 B：学习夹爪（`SACAgentHybridSingleArm`）

适用于 `single-arm-learned-gripper` 模式。

- 存在**两个独立的 Critic 网络**：
  - `critic`：对 6 维 EEF 连续动作进行 Q 值估计（SAC 风格）
  - `grasp_critic`：对夹爪进行 DQN 估计（3 个离散动作：-1/0/+1）
- 策略网络（`actor`）只输出 6 维连续 EEF 分布，夹爪由 `grasp_critic` 独立控制

**算法覆盖范围：**

| 网络 | Module 2 Q 值修正 | Module 3 BC/对比损失 |
|------|:-----------------:|:--------------------:|
| `critic`（6 维 EEF） | ✅ 修正 | ✅ 约束连续策略均值 |
| `grasp_critic`（夹爪 DQN） | ❌ **不修正** | ❌ **不约束** |

A_cf 始终截断动作到 6 维后再由 `critic` 计算：

```python
# HybridSingleArm 中无 fix_gripper 配置，始终截断
h_actions = matched_pref["human_actions"][..., :-1]   # (B, 6)
p_actions = matched_pref["policy_actions"][..., :-1]   # (B, 6)
```

`grasp_critic` 在联合更新中使用**标准 DQN 损失**（`grasp_critic_loss_fn`），仅由 reward + `grasp_penalty` 驱动，与 Module 2/3 并行独立更新。

---

#### 情况 C：无夹爪（纯连续动作空间）

若平台动作空间全部为连续维度，无需任何截断：

```python
h_actions = matched_pref["human_actions"]   # (B, action_dim)
p_actions = matched_pref["policy_actions"]   # (B, action_dim)
```

Module 2 和 Module 3 完整覆盖所有动作维度，无覆盖盲区。

### 3.4 `corrected_critic_loss_fn` 最小实现要求

```python
# 标准 Bellman 目标（SAC with entropy backup）
target_q = rewards + γ * masks * min_k(Q_target(s', a')) - temperature * log_π(a'|s')

# 逐样本修正
corrected_target_q = target_q - alpha_weights * A_cf   # (B,)

# Critic 回归损失
critic_loss = mean((Q(s, a) - corrected_target_q)²)
```

**必须保证**：
1. `A_cf` 通过 `stop_gradient` 包裹，不传播梯度到 Q 网络
2. `alpha_weights` 来自 `batch["alpha_weight"]`，是外部数据，不参与梯度计算
3. `A_cf` 只在 `Q(s, a^π) > Q(s, a^h)` 时才有值（`max(0, ...)` 保证修正方向）

---

## 4. Module 3 — 行为克隆策略损失（BC 版）

### 4.1 核心公式

$$L_{actor} = L_{RLPD} + \beta \cdot L_{BC}$$

$$L_{BC} = \mathbb{E}_{(s, a^h) \sim \text{pref\_buffer}}\left[ \left\| \mu_\theta(s) - \text{atanh}(a^h) \right\|^2 \right]$$

- $\mu_\theta(s)$：策略网络输出的 **pre-tanh 均值**（不采样，不涉及方差 $\sigma$）
- $\text{atanh}(a^h)$：将人类动作映射到 pre-tanh 空间（需 clip 防止数值溢出）
- $\beta$：`contrastive_coef` 超参数（BC 版复用此 flag 名称）

### 4.2 为什么在 pre-tanh 空间计算

策略输出的动作经过 $\tanh$ 压缩：$a = \tanh(u)$，$u \sim \mathcal{N}(\mu_\theta, \sigma_\theta)$。

在 **tanh 后**的动作空间计算 MSE 会引入 $\tanh$ 的非线性梯度。在 **pre-tanh 空间**直接约束均值 $\mu_\theta$，梯度更直接，且完全不涉及 $\sigma_\theta$，与 SAC 的 temperature/entropy 机制**正交**，不会导致 entropy 崩塌。

### 4.3 实现关键点

```python
# 1. 获取策略的 pre-tanh 均值（对应 distrax.Transformed(Normal) 的 .distribution.loc）
mu_theta = policy.forward(obs).distribution.loc   # (B, action_dim)

# 2. 人类动作 clip 后映射到 pre-tanh 空间
clip_eps = 1e-2   # 保证 atanh(±0.99) ≈ ±2.65，不出现 inf
human_a_clipped = np.clip(human_a, -1.0 + clip_eps, 1.0 - clip_eps)
u_human = atanh(human_a_clipped)                  # (B, action_dim)

# 3. Pre-tanh MSE
bc_loss_per_sample = mean((mu_theta - u_human)², axis=-1)   # (B,)

# 4. 加权（bc_weight 字段，BC post-steps 衰减权重）
bc_loss = mean(bc_loss_per_sample * batch["bc_weight"])      # scalar
```

### 4.4 动作维度对齐

preference buffer 中存储的 `human_actions` 可能比策略输出维度多（如带夹爪记录）。需要在计算前截断：

```python
policy_dim = mu_theta.shape[-1]
human_a = preference_batch["human_actions"][..., :policy_dim]
```

### 4.5 BC Post-Steps

干预结束后的若干步，以线性衰减权重继续施加 BC loss（鼓励策略在干预刚结束后维持人类风格，加速策略收敛）。

步数由外部参数 `bc_post_steps` 控制（**默认值 5**）：

```
干预结束 → bc_post_counter = bc_post_steps   # 默认 5
第 t 步（t = 1..bc_post_steps）：weight = 1.0 - t / bc_post_steps
weight > 0 时写入 preference buffer（附带对应的 bc_weight）
weight = 0（即第 bc_post_steps 步结束）时停止写入
```

Actor 端实现要点：

```python
# 干预状态转换时重置计数器
if not already_intervened and "intervene_action_eef" not in info:
    if bc_post_counter > 0:
        t = bc_post_steps - bc_post_counter + 1
        bc_weight = 1.0 - t / bc_post_steps
        if bc_weight > 0:
            preference_buffer.insert({
                "observations":   obs,
                "human_actions":  last_human_action,
                "policy_actions": policy_action_saved,
                "segment_ids":    np.int32(last_segment_uid),
                "bc_weight":      np.float32(bc_weight),
            })
        bc_post_counter -= 1
```

此机制**仅影响 preference buffer 的写入逻辑**，不改变 `bc_policy_loss_fn` 本身。Learner 端无需任何改动，`bc_weight` 字段已在 4.3 节的加权公式中体现。

---

## 5. 数据存储结构与字段规范

### 5.1 三个 Buffer 总览

| Buffer | 用途 | 容量量级 | 写入端 | 读取端 |
|--------|------|----------|--------|--------|
| `replay_buffer` | 在线交互数据（含次优片段标注） | 数十万 | Actor（episode 结束后批量） | Learner（50/50 采样） |
| `demo_buffer` | 干预步数据（RLPD 50/50 中的 demo 端） | 数万 | Actor（干预步立即写） | Learner（50/50 采样） |
| `preference_buffer` | 干预偏好对（供 Module 2 & 3 使用） | ~5000-10000 | Actor（干预首步写） | Learner（两种方式读取） |

### 5.2 `replay_buffer` Transition 字段

| 字段 | 类型/形状 | 说明 |
|------|-----------|------|
| `observations` | `dict` 或 `np.ndarray` | 当前时刻观测（图像为 `{key: (T, H, W, C)}`，状态为 `(state_dim,)`） |
| `actions` | `(action_dim,) float32` | 实际执行的动作（干预步为人类动作） |
| `next_observations` | 同 observations | 下一时刻观测 |
| `rewards` | `float32` | 环境奖励 |
| `masks` | `float32` | `1.0 - done`，用于 Bellman backup |
| `dones` | `bool` | Episode 是否终止 |
| `alpha_weight` | `float32` | **Module 2 核心字段**。次优片段内 `exp(-λ*(t_i-t))`，其余为 0.0 |
| `segment_ids` | `int32` | **Module 2 核心字段**。所属 segment UID，无归属为 -1 |
| `label` | `int` | 控制类型：1=RL 策略，2=Human 干预（可选，供 Progress Model 使用） |
| `grasp_penalty` | `float32` | 可选，仅在 `single-arm-learned-gripper` 模式下存在 |

**不写入 buffer 的内部字段**（以 `_` 前缀标识，episode 结束前过滤）：
- `_was_intervened`：该步是否为干预步，供 `ProgressModelRunner` 构造 label

### 5.3 `demo_buffer` Transition 字段

与 `replay_buffer` 字段完全相同，但：
- `alpha_weight` 固定为 0.0（干预步是高质量演示，不施加修正）
- `segment_ids` 固定为 -1

### 5.4 `preference_buffer` 字段

每条对应一个**干预事件**的起始时刻 $t_i$：

| 字段 | 类型/形状 | 说明 |
|------|-----------|------|
| `observations` | 同 replay_buffer | 干预时刻状态 $s_{t_i}$（含图像，与 replay buffer 格式一致） |
| `human_actions` | `(action_dim,) float32` | 操作员给出的替代动作 $a^h$ |
| `policy_actions` | `(action_dim,) float32` | 干预发生时策略的实际采样动作 $a^{\pi}_{old}$ |
| `segment_ids` | `int32` | 全局唯一 segment UID，用于 `get_by_segment_ids()` 精确查找 |
| `bc_weight` | `float32` | 可选，BC post-steps 功能的加权系数（默认 1.0） |

### 5.5 `preference_buffer` 的两种读取方式

**方式 1：精确对齐查找**（用于 Module 2 A_cf 计算）

```python
# 输入：replay batch 中每条 transition 的 segment_id
matched_pref = preference_buffer.get_by_segment_ids(batch["segment_ids"])
# 输出：与 batch 等长，无匹配位置用零填充，附带 valid_mask
# matched_pref = {
#   "observations":   (B, ...),   # 无匹配位置全零
#   "human_actions":  (B, D),
#   "policy_actions": (B, D),
#   "valid_mask":     (B,) bool
# }
```

**方式 2：随机独立采样**（用于 Module 3 BC/Contrastive 损失）

```python
pref_batch = preference_buffer.sample(preference_batch_size)
# 返回随机采样的 N 条偏好对，格式与 get_by_segment_ids 输出相同（无 valid_mask）
```

### 5.6 Actor 端缓存机制（关键设计约束）

```
为什么不能逐步写入 replay_buffer？
  → alpha_weight 需要 t_i（干预时刻）才能计算 exp(-λ*(t_i-t))
  → t_i 只有在干预发生时才知道，而 t_a 只有在 episode 结束后才确定
  → 因此整条 episode 必须先缓存到 episode_buffer，episode 结束后批量写入

例外：intvn_data_store（demo_buffer）可以立即写入，
  因为干预步的 alpha_weight 固定为 0.0，不需要等待 t_i 确定
```

---

## 6. Progress Model 外部依赖规范

> 若不使用 Progress Model，跳过本节，使用 Fallback 模式（固定回溯窗口）。

### 6.1 模型输入规范

Progress Model 在 **episode 结束后**对整条轨迹批量推理。

输入（均为**相对轨迹起点 $t=0$ 的差值**）：

| 输入 | 形状 | 说明 |
|------|------|------|
| `d_side = feat_side_t - feat_side_0` | `(T+1, 512)` | 侧面摄像头 ResNet-18 特征差 |
| `d_wrist = feat_wrist_t - feat_wrist_0` | `(T+1, 512)` | 腕部摄像头 ResNet-18 特征差 |
| `d_state = (state_t - state_0 - mean) / std` | `(T+1, STATE_DIM)` | 归一化后的本体状态差（STATE_DIM=7） |

特征提取：使用 **冻结的 ResNet-18**（截断到 AvgPool 前，输出 512 维）配合 ImageNet 归一化。

### 6.2 模型输出规范

```
preds  : np.ndarray, shape (T+1,), float32    每帧的任务进度估计 ∈ [0, 1]
labels : np.ndarray, shape (T+1,), int32      控制标签：1=RL步，2=Human干预步
```

索引 0 到 T-1 对应 `episode_buffer[i]["observations"]`，索引 T 对应最终的 `next_observations`。

### 6.3 模型结构（`ProgressHead`）

```
输入：d_side (B,512) + d_wrist (B,512) + d_state (B,STATE_DIM)
                             ↓
state_encoder：Linear(STATE_DIM, 64) → LayerNorm → ReLU   → (B, 64)
                             ↓
concat：[d_side, d_wrist, state_enc] → (B, 1024+64=1088)
                             ↓
fusion_net：Linear(1088, H) → LN → ReLU
            Linear(H, H//2) → LN → ReLU
            Linear(H//2, 1) → Sigmoid
                             ↓
输出：(B,)  float32,  ∈ [0, 1]
```

H（`hidden_dim`）为超参数，训练时指定，推理时需与训练保持一致（默认 128）。

### 6.4 所需文件

| 文件 | 内容 |
|------|------|
| `progress_model_best.pt` | PyTorch 格式，`ProgressHead` 的 `state_dict` |
| `state_stats.pt` | `{"mean": Tensor(STATE_DIM,), "std": Tensor(STATE_DIM,)}`，状态差值的归一化统计 |

### 6.5 obs 格式处理要求

- 图像 obs 可能是堆叠帧 `(obs_horizon, H, W, C)`，取**最后一帧**（最新帧）
- 状态 obs 可能是堆叠帧 `(obs_horizon, STATE_DIM)`，取**最后一帧**
- 对灰度图 `(H, W)` 需扩展为 `(H, W, 3)`

移植时需根据平台实际 obs 格式调整 `obs_img_to_pil` 和 `obs_state_to_vec` 工具函数。

### 6.6 Progress Model 在 Actor 端的调用时机

```
训练开始时：加载模型，初始化 ProgressModelRunner（仅一次）
每个 episode 结束后：调用 runner.infer_episode(episode_buffer)
    → 得到 preds, labels
    → 传入 detect_anomalies()
    → 得到 anomaly_segs
    → 在 detect_suboptimal_segments() 中与 intervention_markers 对齐
```

**不在 step 内调用**，因为模型需要完整的 episode 轨迹才能做相对差值计算。

---

## 7. Actor 循环改动清单

相对 HIL-SERL 基线，Actor 需要以下改动：

### 7.1 必须新增的状态变量

```python
# Episode 级
episode_buffer        = []   # 当前 episode 的所有 transition（带 _was_intervened）
intervention_markers  = []   # 当前 episode 的干预记录
next_segment_uid      = 0    # 全局单调递增，跨 episode 不重置

# 干预状态跟踪
already_intervened    = False   # 是否处于持续干预中
intervention_count    = 0       # 本 episode 干预次数
intervention_steps    = 0       # 本 episode 干预总步数
```

### 7.2 每步动作采样前

```python
policy_action_saved = actions.copy()   # 必须在 env.step() 前保存
```

### 7.3 干预检测（替换/增强 env 交互逻辑）

```python
if "intervene_action_eef" in info:              # 平台相关，键名可能不同
    human_action = info.pop("intervene_action_eef")
    intervention_steps += 1

    if not already_intervened:                   # 仅记录干预序列的第一步
        intervention_count += 1
        t_i = len(episode_buffer)
        intervention_markers.append({
            "step_idx":      t_i,
            "segment_id":    next_segment_uid,
            "obs":           obs,
            "human_action":  human_action,
            "policy_action": policy_action_saved,
        })
        preference_buffer.insert({
            "observations":   obs,
            "human_actions":  human_action,
            "policy_actions": policy_action_saved,
            "segment_ids":    np.int32(next_segment_uid),
        })
        next_segment_uid += 1

    already_intervened = True
    actions = human_action                       # 覆盖策略动作
else:
    already_intervened = False
```

### 7.4 Transition 构造与缓存

```python
transition = {
    "observations":      obs,
    "actions":           actions,          # 干预步为 human_action
    "next_observations": next_obs,
    "rewards":           reward,
    "masks":             1.0 - done,
    "dones":             done,
    "alpha_weight":      0.0,             # 占位符，episode 结束后填充
    "segment_ids":       -1,              # 占位符，episode 结束后填充
    "_was_intervened":   already_intervened,  # 内部字段，写入 buffer 前过滤
}
episode_buffer.append(transition)

if already_intervened:
    intvn_tr = {k: v for k, v in transition.items() if not k.startswith("_")}
    demo_buffer.insert(intvn_tr)           # 干预步立即写入
```

### 7.5 Episode 结束时的批量处理

```python
if done or truncated:
    # Module 1
    suboptimal_segs = detect_suboptimal_segments(
        episode_buffer, intervention_markers,
        window=suboptimal_window,
        progress_runner=progress_runner,     # None 则 fallback
        ...
    )

    # Module 2
    segment_uids = [m["segment_id"] for m in intervention_markers]
    alpha_weights, segment_ids = compute_episode_alpha_and_segment_ids(
        len(episode_buffer), suboptimal_segs, segment_uids, alpha_lambda
    )

    # 批量写入 replay_buffer
    for i, tr in enumerate(episode_buffer):
        tr_to_insert = {k: v for k, v in tr.items() if not k.startswith("_")}
        tr_to_insert["alpha_weight"] = float(alpha_weights[i])
        tr_to_insert["segment_ids"]  = int(segment_ids[i])
        replay_buffer.insert(tr_to_insert)

    # 清空 episode 状态
    episode_buffer       = []
    intervention_markers = []
    intervention_count   = 0
    intervention_steps   = 0
    already_intervened   = False
```

---

## 8. Learner 循环改动清单

相对 HIL-SERL 基线，Learner 需要以下改动：

### 8.1 新增 preference_buffer

```python
preference_buffer = PreferenceBufferDataStore(capacity=10000)
# 注册到 TrainerServer，使 Actor 能通过网络写入
server.register_data_store("actor_env_pref", preference_buffer)
```

### 8.2 高 UTD（Critic-only）更新中的 Module 2

```python
for _ in range(cta_ratio - 1):   # n-1 次 Critic-only 更新
    batch = sample_and_concat(replay_buffer, demo_buffer)
    seg_ids = np.asarray(batch["segment_ids"])
    matched_pref = preference_buffer.get_by_segment_ids(seg_ids)

    if "observations" in matched_pref:
        agent, _ = agent.update_with_correction_bc(
            batch,
            matched_pref,
            preference_batch_direct=None,          # Critic-only，不需要 BC 项
            networks_to_update={"critic"},
        )
    else:
        agent, _ = agent.update(batch, networks_to_update={"critic"})
```

### 8.3 联合更新（Critic + Actor + Temperature）

```python
batch = sample_and_concat(replay_buffer, demo_buffer)
matched_pref      = preference_buffer.get_by_segment_ids(batch["segment_ids"])
pref_batch_direct = preference_buffer.sample(preference_batch_size)

if "observations" in matched_pref and pref_batch_direct is not None:
    agent, update_info = agent.update_with_correction_bc(
        batch,
        matched_pref,
        preference_batch_direct=pref_batch_direct,
        networks_to_update={"critic", "actor", "temperature"},
    )
else:
    agent, update_info = agent.update(batch)   # fallback 标准 RLPD
```

### 8.4 RLPD 50/50 采样要求

```python
# replay_buffer 和 demo_buffer 各采 batch_size/2，再 concat
batch = concat(
    replay_buffer.sample(batch_size // 2),
    demo_buffer.sample(batch_size // 2)
)
```

Learner 不直接区分在线数据和演示数据，两者合并后统一处理。这与 RLPD 基线一致。

### 8.5 Learner 加载 demo 数据时的字段补全

从 pkl 文件加载演示数据时，如果是在新算法之前录制的演示（不含新字段），需要补全：

```python
for transition in demo_data:
    transition.setdefault("alpha_weight", 0.0)    # 演示数据不施加修正
    transition.setdefault("segment_ids", -1)
    demo_buffer.insert(transition)
```

---

## 9. 超参数参考表

| 超参数 | 默认值 | 说明 |
|--------|--------|------|
| `alpha_lambda` (λ) | 3.0 | 位置感知衰减系数。λ=3 时 t_a 处 α≈0.05，t_i 处 α=1.0 |
| `flat_alpha_correction` | False | True：次优片段内 α 统一置 1，不做位置衰减 |
| `contrastive_coef` (β) | 0.2 | BC 损失系数（BC 版和 Contrastive 版均用此 flag）|
| `preference_batch_size` | 16 | Module 3 每步采样的偏好对数量 |
| `suboptimal_window` | 5 | Fallback 模式下回溯的步数 |
| `bc_post_steps` | 5 | 干预结束后附加衰减 BC loss 的步数，干预结束后第 t 步权重为 `1 - t/bc_post_steps` |
| `anomaly_window` | 4 | Progress Model 异常检测滑动窗口大小 |
| `delta_reg` | 0.045 | 回退检测阈值（ΔP < -delta_reg 触发）|
| `delta_stag` | 0.001 | 停滞检测阈值（\|ΔP\| < delta_stag 触发）|
| `detect_regression` | True | 是否启用进度回退检测 |
| `detect_stagnation` | False | 是否启用进度停滞检测 |
| `recovery_k` | 3 | 恢复确认窗口（连续 K 帧 ≥ P_anchor 才确认恢复）|
| `bc_action_clip_eps` | 1e-2 | atanh 前的 clip 范围（`clip(a, -1+ε, 1-ε)`）|

---

## 10. 监控指标清单

### Critic 侧（Module 2 效果）

| 指标 | 健康值 | 说明 |
|------|--------|------|
| `correction_hit_rate` | 训练初期可低，后期 > 20% | α > 0 且 A_cf > 0 同时成立的 transition 占比 |
| `correction_magnitude` | > 0；量级应与 reward 相当 | α(t)·A_cf(t) 的均值，修正实际作用量 |
| `correction_ratio` | 0.05 ~ 0.5 | 修正量 / \|未修正 Q 目标\|；太小说明修正微弱，太大说明过猛 |
| `pref_match_rate` | 初期可低，后期 > 10% | replay batch 中命中 preference buffer 的比例 |
| `q_gap_where_valid` | > 0 | 有效匹配位置上 Q(s, a^π) - Q(s, a^h) 的均值；接近 0 说明 Q 网络未收敛 |

### Actor 侧（Module 3 效果）

| 指标 | 健康值 | 说明 |
|------|--------|------|
| `bc_loss` | 单调下降 | BC 损失绝对值 |
| `action_distance` | 随训练降低 | 策略 tanh(μ) 与 a^h 的 L2 距离（有物理意义）|
| `entropy` | **不应崩塌**（相比 Contrastive 版更稳定）| SAC entropy，BC 版此指标应保持稳定 |

### Episode 侧（Module 1 效果）

| 指标 | 说明 |
|------|------|
| `episode/n_suboptimal_segs` | 本 episode 识别出的次优片段数 |
| `episode/suboptimal_ratio` | 次优片段占 episode 总步数的比例 |
| `episode/mean_alpha` | 有效 alpha_weight 的均值（衡量修正强度） |

---

## 11. 移植检查清单

### 数据结构

- [ ] `replay_buffer` 支持 `alpha_weight` (float32) 字段
- [ ] `replay_buffer` 支持 `segment_ids` (int32) 字段
- [ ] `preference_buffer` 支持 `observations`（与 replay_buffer 格式相同）、`human_actions`、`policy_actions`、`segment_ids` 字段
- [ ] `preference_buffer` 实现 `get_by_segment_ids(segment_ids: np.ndarray) -> dict`（含 `valid_mask`）
- [ ] `preference_buffer` 实现线程安全的 `insert` 和 `sample`

### Actor 端

- [ ] 在 `env.step()` **之前**保存 `policy_action_saved`
- [ ] 干预检测：仅在 `already_intervened` 从 False→True 时写 preference_buffer（不在连续干预期间重复写）
- [ ] `episode_buffer` 缓存机制：episode 结束后批量写入 replay_buffer
- [ ] transition 内的 `_was_intervened` 字段在写入 buffer 前过滤
- [ ] `next_segment_uid` 跨 episode 不重置（全局单调递增）
- [ ] Progress Model（若使用）：episode 结束后调用，不在 step 内调用

### Learner 端

- [ ] `preference_buffer` 注册到通信框架，Actor 端可写入
- [ ] 高 UTD Critic-only 更新中调用 `update_with_correction_bc(..., preference_batch_direct=None)`
- [ ] 联合更新时同时传入 `matched_pref`（Module 2）和 `pref_batch_direct`（Module 3）
- [ ] `preference_buffer` 为空时（或无匹配）回退到标准 `agent.update()`
- [ ] Demo 数据加载时补全 `alpha_weight=0.0` 和 `segment_ids=-1`

### 损失函数

- [ ] `corrected_critic_loss_fn`：`A_cf` 使用 `stop_gradient` 包裹
- [ ] `corrected_critic_loss_fn`：修正方向为 `target_q - α * A_cf`（降低次优片段的 Q 目标）
- [ ] `bc_policy_loss_fn`：在 **pre-tanh 空间**计算 MSE（使用 `μ_θ`，不采样）
- [ ] `bc_policy_loss_fn`：`atanh(human_a)` 前先 clip 到 `[-1+ε, 1-ε]`

### Progress Model（若使用）

- [ ] 模型文件 `progress_model_best.pt` 和 `state_stats.pt` 与训练时参数（hidden_dim 等）匹配
- [ ] `infer_episode` 输入的图像 obs 格式处理（堆叠帧取最后一帧）
- [ ] `infer_episode` 的 `_was_intervened` 字段来源正确
- [ ] 异常检测超参（`delta_reg`、`delta_stag`、`recovery_k` 等）与任务时间尺度匹配
