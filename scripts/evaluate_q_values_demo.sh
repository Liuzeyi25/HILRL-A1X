#!/bin/bash
# evaluate_q_values_demo.sh
# 演示脚本：评估多个 Q 值模型对轨迹的估计性能

set -e

REPO_ROOT="/home/dungeon_master/liuzeyi/HILRL-A1X"
cd "$REPO_ROOT"

# ============================================================================
# 场景 1：单模型、单轨迹（快速测试）
# ============================================================================
echo "=================================================="
echo "场景 1: 单模型、单轨迹快速测试"
echo "=================================================="

# 示例：使用 baseline checkpoint 16000
# MODEL_CHECKPOINT="examples/experiments/insert_block/experiments/insert_block/hilserl/0423_baseline_1/checkpoint_16000"

# 示例轨迹路径（需要你指定实际路径）
# TRAJ_PATH="path/to/your/trajectory.pkl"

# 如果有 buffer 中的轨迹，可以这样调用：
# python scripts/evaluate_q_values.py \
#     --model_paths "$MODEL_CHECKPOINT" \
#     --trajectory_paths "$TRAJ_PATH" \
#     --exp_name insert_block \
#     --output_dir ./q_eval_single \
#     --save_video_samples

echo "[提示] 请指定实际的轨迹 pkl 路径，然后执行上面的命令"
echo ""

# ============================================================================
# 场景 2：多模型对比（同一轨迹）
# ============================================================================
echo "=================================================="
echo "场景 2: 多 checkpoint 对比（需要多个 checkpoint）"
echo "=================================================="

# 示例：如果你有多个 checkpoint
MODEL_1="./examples/experiments/insert_block/experiments/insert_block/checkpoints_hil/0422_4_bc/checkpoint_2000"
MODEL_2="./examples/experiments/insert_block/experiments/insert_block/checkpoints_hil/0422_4_bc/checkpoint_6000"
MODEL_3="./examples/experiments/insert_block/experiments/insert_block/checkpoints_hil/0422_4_bc/checkpoint_12000"

TRAJ_PATH="./examples/experiments/insert_block/experiments/insert_block/checkpoints_hil/0422_4_bc/buffer/transitions_5955.pkl"
python ./scripts/evaluate_q_values.py \
    --model_paths "$MODEL_1" "$MODEL_2" "$MODEL_3" \
    --trajectory_paths "$TRAJ_PATH" \
    --exp_name insert_block \
    --output_dir ./examples/experiments/insert_block/q_eval_multi_model \
    --save_video_samples \
    --ensemble_agg min

echo "[提示] 如有多个 checkpoint，取消注释上面的代码并执行"
echo ""

# ============================================================================
# 场景 3：多轨迹评估
# ============================================================================
echo "=================================================="
echo "场景 3: 多轨迹评估（需要多条轨迹）"
echo "=================================================="

# python scripts/evaluate_q_values.py \
#     --model_paths "$MODEL_CHECKPOINT" \
#     --trajectory_paths "$TRAJ_1" "$TRAJ_2" "$TRAJ_3" \
#     --exp_name insert_block \
#     --output_dir ./q_eval_multi_traj \
#     --save_video_samples \
#     --n_frames_per_traj 100

echo "[提示] 如有多条轨迹，取消注释上面的代码并执行"
echo ""

# ============================================================================
# 完整示例：从 buffer 中提取轨迹并评估
# ============================================================================
echo "=================================================="
echo "完整示例：从 buffer 中提取轨迹"
echo "=================================================="

cat << 'EOF'

如果想从已有的 buffer 中提取轨迹进行评估，可以：

1. 查看 checkpoint 的 buffer 结构：
   ls -lh examples/experiments/insert_block/experiments/insert_block/hilserl/0423_baseline_1/checkpoint_16000/

2. 从 buffer pkl 中提取轨迹（可选预处理）：
   python << 'PYTHON_EOF'
   import pickle as pkl
   
   # 读取一个 buffer pkl
   with open("path/to/buffer/transitions_*.pkl", "rb") as f:
       transitions = pkl.load(f)
   
   # 将其保存为单独的轨迹文件
   with open("trajectory_sample.pkl", "wb") as f:
       pkl.dump(transitions, f)
   PYTHON_EOF

3. 然后调用评估脚本：
   python scripts/evaluate_q_values.py \
       --model_paths "examples/experiments/insert_block/experiments/insert_block/hilserl/0423_baseline_1/checkpoint_16000" \
       --trajectory_paths "trajectory_sample.pkl" \
       --exp_name insert_block \
       --output_dir ./q_eval_results \
       --save_video_samples

EOF

echo ""
echo "=================================================="
echo "参数说明"
echo "=================================================="
python scripts/evaluate_q_values.py --help
