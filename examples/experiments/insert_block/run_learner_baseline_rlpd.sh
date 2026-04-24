#!/usr/bin/env bash

# 纯 baseline RLPD learner（不走 HIL/BC/ProgressModel 逻辑）
# 用法：
#   bash run_learner_baseline_rlpd.sh
#   RUN_TAG=0423_baseline_1 bash run_learner_baseline_rlpd.sh

RUN_TAG=${RUN_TAG:-"0424_baseline_1"}

export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=.5
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

# wandb 配置
export WANDB_MODE=online
export WANDB_ENTITY=liuzeyicsu-central-south-university

python ../../train_rlpd.py \
    --exp_name=insert_block \
    --checkpoint_path=experiments/insert_block/hilserl/${RUN_TAG} \
    --demo_path=/home/dungeon_master/liuzeyi/HILRL-A1X/examples/experiments/insert_block/demo_data/20260421/all_demos_merged.pkl \
    --learner \
    --run_tag="${RUN_TAG}"
