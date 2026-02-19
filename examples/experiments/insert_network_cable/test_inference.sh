#!/bin/bash
# 测试模型推理的快速启动脚本

# 设置环境变量
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=.5

# 运行测试
python /home/dungeon_master/conrft/examples/test_model_inference.py \
    --exp_name=a1x_pick_banana \
    --checkpoint_path=/home/dungeon_master/conrft/examples/experiments/a1x_pick_banana/conrft \
    --checkpoint_step=30000 \
    --demo_path='/home/dungeon_master/conrft/examples/experiments/a1x_pick_banana/demo_data/traj_*.pkl' \
    --num_samples=100 \
    --visualize=True
