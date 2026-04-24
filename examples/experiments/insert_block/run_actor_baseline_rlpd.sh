#!/usr/bin/env bash

# 纯 baseline RLPD actor（与 run_learner_baseline_rlpd.sh 配套）
# 用法：
#   bash run_actor_baseline_rlpd.sh
#   RUN_TAG=0423_baseline_1 bash run_actor_baseline_rlpd.sh

RUN_TAG=${RUN_TAG:-"0424_baseline_1"}

export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=.2

python ../../train_rlpd.py "$@" \
    --exp_name=insert_block \
    --checkpoint_path=experiments/insert_block/hilserl/${RUN_TAG} \
    --actor \
    --run_tag="${RUN_TAG}"
