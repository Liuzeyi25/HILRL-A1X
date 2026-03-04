#!/bin/bash
# Launch script for A1_X ConRFT learner
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH && \
export XLA_PYTHON_CLIENT_PREALLOCATE=false && \
export XLA_PYTHON_CLIENT_MEM_FRACTION=.5 && \
python ../../train_conrft_octo.py "$@" \
    --exp_name=a1x_pick_banana \
    --checkpoint_path=/home/dungeon_master/conrft/examples/experiments/a1x_pick_banana/conrft/0208 \
    --q_weight=1.0 \
    --bc_weight=0.1 \
    --demo_path=./demo_data/20260210/a1x_pick_banana_21_demos.pkl\
    --pretrain_steps=40000 \
    --debug=False \
    --learner
