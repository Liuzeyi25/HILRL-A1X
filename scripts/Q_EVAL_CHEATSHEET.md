# Q 值评估工具 - 快速参考卡

## 🎯 一分钟快速开始

```bash
# 最简单的方式
cd /home/dungeon_master/liuzeyi/HILRL-A1X
python3 scripts/quick_q_eval_example.py
```

结果将保存到：`./q_evaluation_results/`

---

## 📋 命令模板

### 模板 1：单模型评估
```bash
python3 scripts/evaluate_q_values.py \
    --model_paths "checkpoint_16000" \
    --trajectory_paths "traj.pkl" \
    --exp_name insert_block \
    --output_dir ./results \
    --save_video_samples
```

### 模板 2：多模型对比
```bash
python3 scripts/evaluate_q_values.py \
    --model_paths "ckpt_4k" "ckpt_8k" "ckpt_16k" \
    --trajectory_paths "eval_traj.pkl" \
    --exp_name insert_block \
    --output_dir ./multi_model_results
```

### 模板 3：多轨迹分析
```bash
python3 scripts/evaluate_q_values.py \
    --model_paths "checkpoint_final" \
    --trajectory_paths "traj1.pkl" "traj2.pkl" "traj3.pkl" \
    --exp_name insert_block \
    --output_dir ./trajectory_analysis \
    --save_video_samples
```

### 模板 4：内存优化（显存不足）
```bash
python3 scripts/evaluate_q_values.py \
    --model_paths "checkpoint" \
    --trajectory_paths "traj.pkl" \
    --exp_name insert_block \
    --output_dir ./results \
    --chunk_size 4  # 默认 16
```

---

## 📊 输出文件说明

| 文件 | 内容 | 用途 |
|------|------|------|
| `q_comparison_traj#.png` | 2 个子图 | **Q值对比**：看Critic学得好不好 |
| `q_statistics.png` | 柱状图 | **统计对比**：多模型性能差异 |
| `trajectory_#_frames/` | 50个PNG | **图像序列**：验证轨迹数据 |

### 怎样从图表判断好坏？

✅ **好的迹象**
- Q值曲线紧贴黑线（实际回报）
- MAE < 0.4，RMSE < 0.6
- 相关系数 R > 0.85

⚠️ **中等迹象**
- Q值曲线大致跟踪，有些偏差
- MAE 在 0.4-0.7
- R 在 0.7-0.85

❌ **差的迹象**
- Q值曲线明显偏离或反向
- MAE > 1.0，RMSE > 1.5
- R < 0.5

---

## 🔧 常用参数

| 参数 | 值 | 说明 |
|------|-----|------|
| `--ensemble_agg` | `min` | **推荐**：取 ensemble 最小值（更保守） |
| `--ensemble_agg` | `mean` | 取平均值（更平滑） |
| `--gamma` | `0.99` | 折现因子（通常不改） |
| `--chunk_size` | `16` | **推荐**：显存充足时用 |
| `--chunk_size` | `8` | 显存有限时用 |
| `--chunk_size` | `4` | 显存严重不足时用 |
| `--save_video_samples` | - | 保存图像帧（可选，省空间就不要） |
| `--n_frames_per_traj` | `50` | 每轨迹采样帧数（默认 50） |

---

## 📁 文件路径示例

### 模型 checkpoint 路径
```
examples/experiments/insert_block/experiments/insert_block/hilserl/
└── 0423_baseline_1/
    └── checkpoint_16000/     ← 这是你要的路径格式
```

### 轨迹 pkl 路径
```
1. 从 buffer 提取（推荐）：
   checkpoint_16000/buffer/transitions_XXXXXX.pkl
   
2. 或直接指定已有的轨迹：
   /path/to/your/trajectory.pkl
```

---

## 🚨 常见错误与解决

| 错误 | 原因 | 解决方案 |
|------|------|---------|
| `FileNotFoundError: checkpoint not found` | 路径不存在 | 检查路径，用 `ls` 验证 |
| `CUDA out of memory` | 显存不足 | 调小 `--chunk_size`（16→8→4） |
| `checkpoint 内无参数文件` | 路径错误 | 确保路径指向 `checkpoint_STEP` 目录 |
| `无法加载轨迹 pkl` | 格式错误 | 检查 pkl 内容：`pkl.load(open(path, 'rb'))` |

---

## 📞 快速问题排查

**Q: 脚本运行很慢？**  
A: 可能是 `--chunk_size` 太大或没用 GPU。检查：
```bash
python3 -c "import jax; print(jax.devices())"
```

**Q: 看不到图表？**  
A: 检查输出目录是否有 PNG 文件：
```bash
ls ./q_evaluation_results/*.png
```

**Q: 如何从 buffer 快速提取轨迹？**  
A: 使用这个 Python 脚本：
```python
import pickle as pkl
import glob

# 找 buffer 中的第一个 pkl
pkl_file = glob.glob("checkpoint_16000/buffer/transitions_*.pkl")[0]
with open(pkl_file, "rb") as f:
    traj = pkl.load(f)
with open("trajectory.pkl", "wb") as f:
    pkl.dump(traj, f)
print(f"✓ 提取了 {len(traj)} transitions")
```

---

## 💾 输出目录结构

```
your_output_dir/
├── q_comparison_traj0.png        (必有)
├── q_comparison_traj1.png        (如果多于1条轨迹)
├── q_statistics.png              (如果多于1个模型)
├── trajectory_0_frames/          (如果启用 --save_video_samples)
│   ├── frame_000.png
│   ├── frame_001.png
│   └── ...
└── trajectory_1_frames/
    └── ...
```

---

## 🎯 评估工作流

```
第 1 步：收集数据
  ↓
  准备 checkpoint 路径列表
  准备轨迹 pkl 路径列表
  
第 2 步：运行评估
  ↓
  python3 scripts/evaluate_q_values.py --model_paths ... --trajectory_paths ...
  
第 3 步：查看结果
  ↓
  打开 q_comparison_traj#.png（查看Q值曲线）
  打开 q_statistics.png（查看统计对比）
  查看 MAE/RMSE/R 指标
  
第 4 步：诊断与改进
  ↓
  如果 MAE 大 → 需要继续训练
  如果 R 低 → Critic 学得不好
  如果有问题 → 调整超参或检查数据
```

---

## 🎨 参数对结果的影响

| 参数 | 改变 | 效果 |
|------|------|------|
| `--chunk_size` | 16 → 8 | 更稳定但可能慢一点 |
| `--ensemble_agg` | min → mean | Q值更平滑但可能不够保守 |
| `--gamma` | 0.99 → 0.95 | 更注重近期回报 |
| `--use_target_critic` | False → True | 比较 online vs target 差异 |

---

## 📖 文档导航

```
想快速开始？
  → 运行 python3 scripts/quick_q_eval_example.py

想了解详细用法？
  → 读 scripts/Q_EVAL_QUICKSTART.md

想深入了解技术？
  → 读 scripts/EVALUATE_Q_VALUES_README.md

想看代码？
  → 看 scripts/evaluate_q_values.py
```

---

## ⚙️ 系统要求

| 要求 | 最低配置 | 推荐配置 |
|------|---------|---------|
| Python | 3.10+ | 3.10+ |
| GPU | - | NVIDIA GPU (4GB+) |
| RAM | 16GB | 32GB |
| Disk | 2GB | 10GB |

---

## 🚀 开始使用

### 最简单的方式（推荐新手）
```bash
python3 scripts/quick_q_eval_example.py
```

### 标准方式
```bash
python3 scripts/evaluate_q_values.py \
    --model_paths YOUR_MODEL_PATH \
    --trajectory_paths YOUR_TRAJ_PATH \
    --exp_name insert_block \
    --output_dir ./results \
    --save_video_samples
```

### 查看帮助
```bash
python3 scripts/evaluate_q_values.py --help
```

---

**版本**: 1.0 | **创建**: 2026-04-23 | **状态**: ✅ 完成
