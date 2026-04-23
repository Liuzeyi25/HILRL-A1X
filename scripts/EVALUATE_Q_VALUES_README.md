# Q 值评估工具 (evaluate_q_values.py)

## 概述

`evaluate_q_values.py` 是一个专为 HIL-SERL BC 版训练的 Q 值分析工具，支持：

✅ **多模型评估**：同时加载多个 Critic 模型（不同训练步数）  
✅ **多轨迹分析**：批量评估多条轨迹的 Q 值估计性能  
✅ **可视化对比**：生成 Q 值曲线、误差分析、图像帧序列  
✅ **定量指标**：MAE、RMSE、Pearson 相关系数等  

## 功能特性

### 1. 灵活的数据输入

**模型路径** - 支持多个 checkpoint：
```bash
--model_paths "checkpoint_4000" "checkpoint_8000" "checkpoint_16000"
```

**轨迹路径** - 支持多种格式的 pkl 文件：
- 直接的 transitions 列表（list[dict]）
- `{"episodes": [list[dict]]}` 格式
- `{"trajectories": [list[dict]]}` 格式

支持多条轨迹：
```bash
--trajectory_paths "traj1.pkl" "traj2.pkl" "traj3.pkl"
```

### 2. Q 值评估

对每条轨迹的每个时间步：
- 从观测和动作推断 Q 值（使用 Critic 网络）
- 计算实际的折现回报（discounted return）
- 对比评估误差：MAE、RMSE

**指标说明**：
- **MAE** (Mean Absolute Error)：平均绝对误差
- **RMSE** (Root Mean Square Error)：均方根误差
- **Pearson R**：相关系数（越接近 1 越好）

### 3. 可视化输出

输出目录将包含：

```
q_eval_results/
├── q_comparison_traj0.png      # 轨迹 0 的 Q 值曲线对比
├── q_comparison_traj1.png      # 轨迹 1 的 Q 值曲线对比
├── q_statistics.png             # 多模型/多轨迹统计对比
├── trajectory_0_frames/
│   ├── frame_000.png           # 轨迹 0 的图像帧 (采样)
│   ├── frame_001.png
│   └── ...
└── trajectory_1_frames/
    ├── frame_000.png
    └── ...
```

#### 图像说明

**q_comparison_traj#.png**（两个子图）：
- 上图：Q 值曲线 vs 实际折现回报
- 下图：Q 值估计误差（对数尺度）

**q_statistics.png**（两个子图）：
- 左图：不同轨迹的 MAE 对比（柱状图）
- 右图：不同轨迹的 RMSE 对比（柱状图）

## 安装与使用

### 前置要求

确保已在 conda 环境中安装依赖：
```bash
conda activate hilrl  # 或你的环境名
```

### 基本用法

#### 场景 1：单模型、单轨迹

```bash
python scripts/evaluate_q_values.py \
    --model_paths "examples/experiments/insert_block/experiments/insert_block/hilserl/0423_baseline_1/checkpoint_16000" \
    --trajectory_paths "path/to/trajectory.pkl" \
    --exp_name insert_block \
    --output_dir ./q_eval_results \
    --save_video_samples
```

#### 场景 2：多 checkpoint 对比（同一轨迹）

```bash
python scripts/evaluate_q_values.py \
    --model_paths \
        "examples/experiments/insert_block/experiments/insert_block/hilserl/0423_baseline_1/checkpoint_4000" \
        "examples/experiments/insert_block/experiments/insert_block/hilserl/0423_baseline_1/checkpoint_8000" \
        "examples/experiments/insert_block/experiments/insert_block/hilserl/0423_baseline_1/checkpoint_16000" \
    --trajectory_paths "path/to/trajectory.pkl" \
    --exp_name insert_block \
    --output_dir ./q_eval_multi_model \
    --ensemble_agg min
```

#### 场景 3：多轨迹评估

```bash
python scripts/evaluate_q_values.py \
    --model_paths "checkpoint_16000" \
    --trajectory_paths \
        "path/to/traj1.pkl" \
        "path/to/traj2.pkl" \
        "path/to/traj3.pkl" \
    --exp_name insert_block \
    --output_dir ./q_eval_multi_traj \
    --save_video_samples \
    --n_frames_per_traj 100
```

#### 场景 4：从 buffer 中提取轨迹

如果 checkpoint 包含 buffer（保存的 transitions）：

```bash
# 1. 从 buffer pkl 中提取（作为一条轨迹）
python << 'EOF'
import pickle as pkl

buffer_dir = "examples/experiments/insert_block/experiments/insert_block/hilserl/0423_baseline_1/checkpoint_16000/buffer"
with open(f"{buffer_dir}/transitions_XXXXXX.pkl", "rb") as f:
    transitions = pkl.load(f)

# 保存为单独的轨迹文件
with open("trajectory_from_buffer.pkl", "wb") as f:
    pkl.dump(transitions, f)
print(f"提取了 {len(transitions)} 条 transition")
EOF

# 2. 评估
python scripts/evaluate_q_values.py \
    --model_paths "checkpoint_16000" \
    --trajectory_paths "trajectory_from_buffer.pkl" \
    --exp_name insert_block \
    --output_dir ./q_eval_buffer
```

### 完整参数列表

```bash
python scripts/evaluate_q_values.py --help
```

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--model_paths` | str (多个) | 必需 | 一个或多个模型 checkpoint 路径 |
| `--trajectory_paths` | str (多个) | 必需 | 一个或多个轨迹 pkl 路径 |
| `--exp_name` | str | 必需 | 实验名（CONFIG_MAPPING 中的键） |
| `--output_dir` | str | `./q_evaluation_results` | 输出目录 |
| `--n_frames_per_traj` | int | 50 | 每条轨迹保存的图像帧数 |
| `--ensemble_agg` | str | `min` | Q 聚合方式：`min` 或 `mean` |
| `--use_target_critic` | flag | False | 是否使用 target critic（无参数） |
| `--save_video_samples` | flag | False | 保存轨迹图像帧（无参数） |
| `--chunk_size` | int | 16 | 批推理大小（图像大时调小） |
| `--gamma` | float | 0.99 | 折现因子 |
| `--seed` | int | 42 | 随机种子 |

## 工作流示例

### 完整工作流：从训练到评估

```bash
# 1. 训练 BC 模型（if not already trained）
cd examples/experiments/insert_block
bash run_learner_zeyi.sh  # learner process
bash run_actor_zeyi.sh    # actor process in another terminal

# 等待训练完成，checkpoint 会保存到：
# examples/experiments/insert_block/experiments/insert_block/hilserl/RUN_TAG/checkpoint_STEP

# 2. 回到根目录
cd /home/dungeon_master/liuzeyi/HILRL-A1X

# 3. 从 buffer 中提取一条轨迹进行评估
python << 'EOF'
import pickle as pkl
import glob
from pathlib import Path

# 找到最新的 checkpoint
exp_dir = "examples/experiments/insert_block/experiments/insert_block/hilserl"
latest_ckpt = sorted(Path(exp_dir).glob("*/checkpoint_*"), 
                    key=lambda p: int(p.name.split("_")[1]))[-1]

# 读取 buffer 中的一个 pkl 作为轨迹
buffer_files = glob.glob(f"{latest_ckpt}/buffer/transitions_*.pkl")
if buffer_files:
    with open(buffer_files[0], "rb") as f:
        traj = pkl.load(f)
    with open("eval_trajectory.pkl", "wb") as f:
        pkl.dump(traj, f)
    print(f"已提取 {len(traj)} transitions")
else:
    print("未找到 buffer 文件")
EOF

# 4. 评估 Q 值
python scripts/evaluate_q_values.py \
    --model_paths "examples/experiments/insert_block/experiments/insert_block/hilserl/YOUR_RUN_TAG/checkpoint_16000" \
    --trajectory_paths "eval_trajectory.pkl" \
    --exp_name insert_block \
    --output_dir ./q_eval_final \
    --save_video_samples

# 5. 查看结果
ls -lh ./q_eval_final/
# 打开 PNG 文件查看
open ./q_eval_final/*.png  # macOS
# 或 xdg-open ./q_eval_final/*.png  # Linux
```

## 输出解释

### 终端输出

```
======================================================================
多模型 Q 值评估工具
======================================================================
[输入] 模型数=2, 轨迹数=3
[输出] ./q_eval_results

[配置] exp_name=insert_block
       setup_mode=single-arm-fixed-gripper
       image_keys=['agentview_rgb', 'wrist_rgb']
       fix_gripper=True

[轨迹] 正在加载 3 条轨迹...
  已加载轨迹: path/to/traj1.pkl (长度=150)
  已加载轨迹: path/to/traj2.pkl (长度=200)
  已加载轨迹: path/to/traj3.pkl (长度=180)
  共加载 3 条轨迹

[推理] 评估 3 条轨迹的 Q 值...

  轨迹 #0 (长度=150):
    checkpoint_4000                | MAE=0.4532 RMSE=0.5821 R=0.891
    checkpoint_8000                | MAE=0.3245 RMSE=0.4102 R=0.923

✓ 评估完成，所有结果已保存至: ./q_eval_results
```

### 汇总统计

```
[汇总] 所有轨迹的平均指标:
  checkpoint_4000                | avg_MAE=0.4201 avg_RMSE=0.5123 avg_R=0.885
  checkpoint_8000                | avg_MAE=0.3012 avg_RMSE=0.3945 avg_R=0.912
```

## 常见问题

### Q: 如何处理 "checkpoint not found" 错误？

**A:** 确保 checkpoint 路径正确，包含 checkpoint 参数文件：
```bash
# 正确的路径结构
examples/experiments/insert_block/experiments/insert_block/hilserl/0423_baseline_1/checkpoint_16000/
├── checkpoint  (Flax checkpoint 目录)
└── ...

# 验证
ls -lh "your_checkpoint_path/checkpoint/"
```

### Q: 可视化的图像帧为空或损坏？

**A:** 
1. 检查轨迹中是否包含图像观测：
```python
import pickle as pkl
with open("your_trajectory.pkl", "rb") as f:
    traj = pkl.load(f)
print(traj[0]['observations'].keys())  # 查看包含的键
```

2. 如果没有图像，移除 `--save_video_samples` 参数

### Q: 内存溢出 (OOM)？

**A:** 调小 `--chunk_size` 参数：
```bash
python scripts/evaluate_q_values.py \
    ... \
    --chunk_size 8  # 默认 16，改为 8 或 4
```

### Q: Q 值估计误差很大？

**A:** 这通常表示：
1. 模型训练不足（早期 checkpoint）
2. 轨迹数据分布与训练数据不匹配
3. Critic 网络容量不足

可以对比不同 checkpoint 观察 Q 值估计质量随训练的改进趋势。

## 扩展开发

### 自定义指标

在 `main()` 函数中修改指标计算：

```python
# 在 evaluate 循环中添加
custom_metric = compute_custom_metric(q_vals, returns)
all_results[model_name][traj_idx]["custom"] = custom_metric
```

### 自定义可视化

编写新的 `plot_*` 函数并在 `main()` 中调用：

```python
def plot_custom_analysis(all_results, output_dir):
    # 你的可视化代码
    pass

# 在 main() 末尾
plot_custom_analysis(all_results, args.output_dir)
```

## 性能建议

- **快速评估**：`--chunk_size 32`，移除 `--save_video_samples`
- **高质量可视化**：`--n_frames_per_traj 100+`
- **内存受限**：`--chunk_size 8` + 减少轨迹数

## 相关文件

- 训练脚本：`examples/train_rlpd_hil_bc.py`
- Agent 代码：`serl_launcher/agents/continuous/sac.py`
- 轨迹格式：参考 `examples/train_rlpd_hil.py` 中的 buffer 保存

## 更新日志

### v1.0 (2026-04-23)
- ✅ 多模型支持
- ✅ 多轨迹支持
- ✅ Q 值可视化
- ✅ 图像帧导出
- ✅ 统计指标

---

**如有问题或建议，请联系开发者！** 🚀
