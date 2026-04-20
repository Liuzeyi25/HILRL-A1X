# 训练崩溃分析与修复报告

**问题描述**：HIL-SERL 训练启动后，WandB 监控显示训练迅速崩溃。

---

## 一、崩溃现象（WandB 观测）

| 指标 | 正常值 | 崩溃时观测值 |
|------|--------|------------|
| `log_prob_human` | -10 ~ -50 | **~10^12** |
| `log_prob_policy_hist` | -10 ~ -50 | **~10^12** |
| `log_prob_gap` | 小量 | **极大** |
| `contrastive_loss` | 小量 | **~ -5×10^12** |
| `q_policy_mean` ≈ `q_human_mean` | 有差距 | **趋近于 0（A_cf → 0）** |

---

## 二、根因分析

### 2.1 直接原因：`atanh(±1.0) = ±∞`

SAC 策略网络的动作分布是 **Tanh 压缩的高斯分布**，其 `log_prob` 含有修正项：

$$\log \pi(a|s) = \log \mathcal{N}(u|s) - \sum_i \log(1 - a_i^2)$$

当某维动作 $a_i = \pm 1.0$ 时：

$$\log(1 - a_i^2) = \log(0) = -\infty \implies -\log(1-a_i^2) = +\infty$$

此时整个 `log_prob` 发散至 `+∞`，对比损失 `contrastive_loss = -mean(log_prob_human - log_prob_policy)` 变为 `-∞` 或 `+∞` 级别，导致梯度爆炸。

### 2.2 触发来源：SpaceMouse 夹爪按键

查看 `SpacemouseIntervention.action()`（`wrappers_20260129.py` 第 376 行）：

```python
if self.left:   # close gripper
    gripper_action = np.random.uniform(-1, -0.9, size=(1,))
elif self.right: # open gripper
    gripper_action = np.random.uniform(0.9, 1, size=(1,))
```

`np.random.uniform(low, high)` 的返回范围是 **`[low, high)`**，即 `low` 可以精确取到。因此：
- 左键按下时，夹爪维度可能精确为 `-1.0`
- 该值写入 `preference_data_store` → learner 计算 `log_prob` 时触发 `atanh(-1.0) = -∞`

### 2.3 问题链路

```
SpaceMouse 左键
  └─ gripper_action = np.random.uniform(-1, -0.9)   # 可能精确为 -1.0
       └─ info["intervene_action_eef"][6] = -1.0
            └─ preference_data_store.insert(human_actions=...)
                 └─ contrastive_policy_loss_fn
                      └─ pref_distributions.log_prob(human_a)
                           └─ atanh(-1.0) = -∞
                                └─ log_prob = +∞  (tanh 修正项 -log(1-a²) → +∞)
                                     └─ contrastive_loss ~ -5×10^12
                                          └─ 梯度爆炸 → 训练崩溃
```

### 2.4 次要现象：A_cf → 0

由于 actor 网络权重已被爆炸梯度破坏，`q_policy` 和 `q_human` 的估值趋于相同，
导致 `A_cf = max(0, mean(q_human - q_policy)) → 0`，Q 值修正（Module 2）完全失效。

---

## 三、修复方案

### 最终采用：在 `contrastive_policy_loss_fn` 中始终截掉最后一维

**文件**：`serl_launcher/serl_launcher/agents/continuous/sac.py`  
**函数**：`contrastive_policy_loss_fn`（约第 350 行）

#### 修改前
```python
# fix_gripper 兼容：policy 输出 (action_dim-1) 维，preference 动作需截断
human_a = (
    preference_batch["human_actions"][..., :-1]
    if self.config["fix_gripper"]
    else preference_batch["human_actions"]
)
policy_a = (
    preference_batch["policy_actions"][..., :-1]
    if self.config["fix_gripper"]
    else preference_batch["policy_actions"]
)
```

#### 修改后
```python
# 计算对比 log_prob 时始终截掉最后一维（夹爪）。
# 理由：当前任务固定夹爪，SpaceMouse 左键会生成精确 -1.0 的夹爪值，
# atanh(-1.0) = -∞ 导致 log_prob 爆炸（10^12 级别）。
# 即使将来切换到 learned-gripper 模式，夹爪 log_prob 对对比信号的
# 贡献也远小于连续关节维度，排除不影响训练语义。
# fix_gripper=True 时 policy 只输出前 (action_dim-1) 维，也兼容。
human_a  = preference_batch["human_actions"][..., :-1]
policy_a = preference_batch["policy_actions"][..., :-1]
```

### 优点

| 对比方案 | 缺点 |
|----------|------|
| clip 到 `(-1+ε, 1-ε)` | 夹爪 `-1.0` 被 clip 为 `-0.999999`，信息失真，仍在参与计算 |
| 修改 SpaceMouse 采样范围 | 改变了干预行为，且不根治（Gello 也可能产生边界值） |
| **截掉最后一维（采用）** | 从根本排除问题维度，不影响对比信号的正确性 |

---

## 四、验证方法

重新训练后，在 WandB 中观察以下指标应恢复正常：

| 指标 | 期望值 |
|------|--------|
| `log_prob_human` | -10 ~ -100 数量级 |
| `log_prob_policy_hist` | -10 ~ -100 数量级 |
| `contrastive_loss` | 0 ~ 1 数量级（正值） |
| `log_prob_gap` | 前期负值（策略还未学好），逐渐趋向正值 |
| `q_gap_mean` | 应有明显正值（人类动作比 policy 好） |

---

## 五、相关文件

| 文件 | 说明 |
|------|------|
| `serl_launcher/serl_launcher/agents/continuous/sac.py` | **已修改**：`contrastive_policy_loss_fn` 截掉夹爪维度 |
| `serl_launcher/serl_launcher/agents/continuous/sac_hybrid_single.py` | 无需修改（`hybrid` 版本原本就是 `[..., :-1]`） |
| `serl_robot_infra/franka_env/envs/wrappers_20260129.py` | 根因所在：`SpacemouseIntervention.action()` 的 `np.random.uniform(-1, -0.9)` |
| `examples/train_rlpd_hil.py` | preference buffer 写入逻辑，无需修改 |
