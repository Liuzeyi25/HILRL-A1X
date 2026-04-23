# Q 值评估工具 - 使用指南

> **创建时间**: 2026-04-23  
> **工具版本**: 1.0  
> **状态**: ✅ 完成

## 📋 概览

创建了完整的 **多模型 Q 值评估工具链**，支持对 HIL-SERL BC 版训练的 Critic 模型进行系统性评估。

### 🎯 核心功能

✅ **多模型对比评估**  
- 同时加载多个 Critic checkpoint（不同训练步数）
- 对同一轨迹进行 Q 值推理
- 定量对比 MAE、RMSE、相关系数

✅ **多轨迹批量分析**  
- 支持多条轨迹的统计性能评估
- 生成轨迹间的误差分布分析
- 柱状图对比展示

✅ **可视化与诊断**  
- Q 值曲线 vs 实际折现回报（上图）
- Q 值估计误差（下图，对数尺度）
- 轨迹图像帧序列导出（可用于制作演示视频）
- 统计汇总图表（多模型/多轨迹对比）

✅ **灵活的数据格式支持**  
- 轨迹 pkl：`list[dict]` 或 `{"episodes": [...]}` 格式
- 自动从 buffer 中提取 transitions
- 支持使用 online 或 target critic

---

## 📁 文件结构

```
scripts/
├── evaluate_q_values.py              # 核心评估脚本
├── quick_q_eval_example.py          # 快速开始示例
├── evaluate_q_values_demo.sh         # Bash 演示命令
├── EVALUATE_Q_VALUES_README.md       # 详细文档（中文）
└── analyze_q_values.py               # 原有 Q 值分析工具（保留兼容）
```

---

## 🚀 快速开始

### 方式 1: 使用快速示例脚本（推荐）

```bash
cd /home/dungeon_master/liuzeyi/HILRL-A1X

# 自动提取 buffer 轨迹并进行评估
python3 scripts/quick_q_eval_example.py
```

这会自动：
1. 检测可用的 checkpoint
2. 从最新 checkpoint 的 buffer 中提取轨迹
3. 执行单模型和多模型评估
4. 生成可视化输出

**预期输出**：
```
======================================================================
Q 值评估工具 - 快速示例
======================================================================

✓ 找到 checkpoint_4000
✓ 找到 checkpoint_8000
✓ 找到 checkpoint_12000
✓ 找到 checkpoint_16000

[检测] 共找到 4 个 checkpoint

[场景 1] 单模型评估（最新 checkpoint）
  ✓ 评估完成！结果已保存至: /path/to/q_evaluation_results
```

### 方式 2: 手动指定模型和轨迹

```bash
# 提取轨迹
python3 << 'EOF'
import pickle as pkl

checkpoint_path = "examples/experiments/insert_block/experiments/insert_block/hilserl/0423_baseline_1/checkpoint_16000"
buffer_file = f"{checkpoint_path}/buffer/transitions_XXXXXX.pkl"

with open(buffer_file, "rb") as f:
    traj = pkl.load(f)

with open("my_trajectory.pkl", "wb") as f:
    pkl.dump(traj, f)
print(f"✓ 提取了 {len(traj)} transitions")
EOF

# 评估
python3 scripts/evaluate_q_values.py \
    --model_paths "examples/experiments/insert_block/experiments/insert_block/hilserl/0423_baseline_1/checkpoint_16000" \
    --trajectory_paths "my_trajectory.pkl" \
    --exp_name insert_block \
    --output_dir ./q_eval_results \
    --save_video_samples
```

---

## 📊 输出解释

### 终端输出示例

```
[推理] 评估 1 条轨迹的 Q 值...

  轨迹 #0 (长度=150):
    checkpoint_16000               | MAE=0.3245 RMSE=0.4102 R=0.923
    
  ✓ 轨迹 #0 图像帧保存至: ./q_eval_results/trajectory_0_frames (50 帧)
  ✓ 保存对比图: ./q_eval_results/q_comparison_traj0.png

✓ 评估完成，所有结果已保存至: ./q_eval_results
```

**指标说明**：
- **MAE** (Mean Absolute Error)：$\text{MAE} = \frac{1}{T}\sum_{t=0}^{T-1}|Q_t - R_t|$
  - 越小越好，表示 Q 值估计越准确
  
- **RMSE** (Root Mean Square Error)：$\text{RMSE} = \sqrt{\frac{1}{T}\sum_{t=0}^{T-1}(Q_t - R_t)^2}$
  - 惩罚大的误差，对离群值敏感
  
- **R** (Pearson Correlation)：$r = \frac{\text{Cov}(Q, R)}{\sigma_Q \sigma_R}$
  - 范围 $[-1, 1]$，越接近 1 说明 Q 值趋势与真实回报一致

### 图像文件

#### 1. `q_comparison_traj#.png`

**上子图**：Q 值曲线对比
- 黑线：实际折现回报（ground truth）
- 彩色线：各模型的 Q 值估计

```
如果曲线紧贴黑线 ➜ Critic 学习得很好 ✓
如果曲线明显偏离 ➜ Critic 需要进一步训练
```

**下子图**：估计误差（对数尺度）
- 显示 $|Q_t - R_t|$ 随时间的变化
- 对数尺度便于观察小误差和大误差的分布

#### 2. `q_statistics.png`

多模型/多轨迹的统计对比：
- **左**：各轨迹的 MAE（柱状图）
- **右**：各轨迹的 RMSE（柱状图）

用于直观比较不同 checkpoint 的性能改进趋势。

#### 3. `trajectory_#_frames/`

轨迹的图像帧序列（可用于制作演示视频）：
```
trajectory_0_frames/
├── frame_000.png
├── frame_001.png
├── frame_002.png
└── ...
```

---

## 🔧 高级用法

### 场景 1：多 checkpoint 对比

```bash
python3 scripts/evaluate_q_values.py \
    --model_paths \
        "checkpoint_4000" \
        "checkpoint_8000" \
        "checkpoint_12000" \
        "checkpoint_16000" \
    --trajectory_paths "trajectory.pkl" \
    --exp_name insert_block \
    --output_dir ./q_eval_checkpoints \
    --ensemble_agg min
```

✅ 观察 Q 值估计质量随训练步数的改进趋势

### 场景 2：多轨迹性能评估

```bash
python3 scripts/evaluate_q_values.py \
    --model_paths "checkpoint_16000" \
    --trajectory_paths \
        "traj_success.pkl" \
        "traj_failure.pkl" \
        "traj_recovery.pkl" \
    --exp_name insert_block \
    --output_dir ./q_eval_trajectories \
    --save_video_samples
```

✅ 评估 Critic 在不同轨迹类型上的性能差异

### 场景 3：Target vs Online Critic

```bash
# Online Critic（默认）
python3 scripts/evaluate_q_values.py ... \
    --output_dir ./q_eval_online

# Target Critic
python3 scripts/evaluate_q_values.py ... \
    --use_target_critic \
    --output_dir ./q_eval_target
```

✅ 对比 target 和 online critic 的差异

---

## 💾 内存优化

如果遇到显存溢出 (OOM)：

```bash
# 原始（可能 OOM）
python3 scripts/evaluate_q_values.py ... --chunk_size 32

# 优化版（更小的批大小）
python3 scripts/evaluate_q_values.py ... --chunk_size 8

# 极端优化（如果 8 还不够）
python3 scripts/evaluate_q_values.py ... --chunk_size 2
```

### 其他优化策略：
- 移除 `--save_video_samples`（省显存）
- 减少轨迹数量或长度
- 使用 `--n_frames_per_traj 20`（默认 50）

---

## 📈 性能指标解释

### Q 值估计质量的典型范围

| 阶段 | MAE | RMSE | 含义 |
|------|-----|------|------|
| 早期训练（step < 5k） | > 1.0 | > 1.5 | Critic 尚未收敛 |
| 中期训练（5k - 10k） | 0.5 - 1.0 | 0.7 - 1.3 | Critic 学习中 |
| 后期训练（> 10k） | < 0.5 | < 0.8 | Critic 基本收敛 |

### 如何判断 Critic 质量？

✅ **好的 Critic**：
- Q 值曲线紧跟实际回报
- MAE < 0.4，RMSE < 0.6
- 相关系数 R > 0.85
- 不同轨迹间的性能一致

⚠️ **中等的 Critic**：
- Q 值曲线大致跟踪回报，但有偏差
- MAE 在 0.4-0.7 之间
- R 在 0.7-0.85 之间

❌ **差的 Critic**：
- Q 值曲线明显偏离回报（如完全反向）
- MAE > 1.0，RMSE > 1.5
- R < 0.5 或为负
- 这通常表示训练问题（如 entropy 崩溃）

---

## 🐛 常见问题与解决方案

### Q1: 如何找到 checkpoint 路径？

**A:**
```bash
# 列出所有 checkpoint
find examples/experiments -name "checkpoint_*" -type d

# 或按时间排序
ls -ltr examples/experiments/insert_block/experiments/insert_block/hilserl/*/checkpoint_*
```

### Q2: 轨迹 pkl 应该包含什么内容？

**A:**
```python
# 有效的轨迹格式示例
trajectory = [
    {
        "observations": {"agentview_rgb": img1, "wrist_rgb": img2, ...},
        "actions": action_vector,
        "rewards": reward_scalar,
        "dones": done_bool,
        ...
    },
    # 更多 transitions
]

# 保存
with open("trajectory.pkl", "wb") as f:
    pkl.dump(trajectory, f)
```

### Q3: "checkpoint not found" 错误？

**A:**
```bash
# 检查路径是否存在
ls -la "your_checkpoint_path/"

# 应该看到类似这样的结构：
# checkpoint_16000/
# ├── checkpoint/
# │   └── default/
# │       └── ...
```

### Q4: 显存溢出?

**A:**
```bash
# 关键参数：chunk_size（默认 16）
python3 scripts/evaluate_q_values.py ... --chunk_size 4
```

---

## 🎓 使用示例完整流程

### 从训练到评估的完整工作流

```bash
# 1️⃣ 开始训练（如未训练）
cd examples/experiments/insert_block
RUN_TAG=q_eval_demo bash run_learner_zeyi.sh &  # 后台运行
bash run_actor_zeyi.sh

# 等待训练完成...
# Checkpoint 会保存到：
# examples/experiments/insert_block/experiments/insert_block/hilserl/q_eval_demo/checkpoint_*

# 2️⃣ 返回根目录
cd /home/dungeon_master/liuzeyi/HILRL-A1X

# 3️⃣ 运行快速评估
python3 scripts/quick_q_eval_example.py

# 4️⃣ 查看结果
open ./q_evaluation_results  # macOS
# xdg-open ./q_evaluation_results  # Linux

# 结果文件：
# - q_evaluation_results/single_model/
#   ├── q_comparison_traj0.png
#   └── trajectory_0_frames/
# - q_evaluation_results/multi_model/
#   ├── q_comparison_traj0.png
#   ├── q_statistics.png
#   └── trajectory_0_frames/
```

---

## 📚 相关文档

| 文档 | 位置 | 说明 |
|------|------|------|
| **详细英文文档** | `scripts/EVALUATE_Q_VALUES_README.md` | 完整参数说明 |
| **快速开始脚本** | `scripts/quick_q_eval_example.py` | 自动化示例 |
| **演示命令** | `scripts/evaluate_q_values_demo.sh` | Bash 示例 |
| **训练脚本** | `examples/train_rlpd_hil_bc.py` | BC 版训练 |

---

## ✅ 检查清单

在使用前，确保：

- [ ] 环境已配置：`conda activate hilrl`
- [ ] 模型已训练或 checkpoint 已存在
- [ ] 轨迹数据可获取（pkl 格式）
- [ ] 输出目录有写权限
- [ ] 显存充足（或已调整 chunk_size）

---

## 🎯 下一步

1. **运行快速示例**
   ```bash
   python3 scripts/quick_q_eval_example.py
   ```

2. **根据需要自定义参数**
   ```bash
   python3 scripts/evaluate_q_values.py --help
   ```

3. **查看并分析结果**
   - 打开 PNG 图表
   - 对比 MAE/RMSE 指标
   - 检查是否有异常趋势

4. **根据诊断调整训练** (如需要)
   - 调整超参数
   - 增加训练步数
   - 修改数据配置

---

## 📞 支持

如遇到问题：

1. 检查错误信息中的路径是否正确
2. 查阅 `EVALUATE_Q_VALUES_README.md` 常见问题部分
3. 尝试 `--chunk_size 4` 等内存优化方案
4. 验证轨迹 pkl 的格式是否正确

---

**创建日期**: 2026-04-23  
**最后更新**: 2026-04-23  
**维护者**: 编程助手 (GitHub Copilot)  
**版本**: 1.0 (稳定版)
