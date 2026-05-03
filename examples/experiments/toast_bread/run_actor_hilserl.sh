#!/bin/bash
# Launch script for A1_X ConRFT actor
# 用法:
#   bash run_actor_hilserl.sh                  # 使用默认路径
#   RUN_TAG=<tag> bash run_actor_hilserl.sh    # 指定续训路径

RUN_TAG=${RUN_TAG:-"hilserl-0503-1"}

export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH && \
export XLA_PYTHON_CLIENT_PREALLOCATE=false && \
export XLA_PYTHON_CLIENT_MEM_FRACTION=.4 && \
python ../../train_rlpd.py "$@" \
    --exp_name=toast_bread \
    --checkpoint_path=hilserl/${RUN_TAG} \
    --actor \
    # --eval_checkpoint_step=26000 \