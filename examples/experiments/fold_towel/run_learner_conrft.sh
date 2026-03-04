#!/bin/bash
# Launch script for A1_X ConRFT learner
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH && \
export XLA_PYTHON_CLIENT_PREALLOCATE=false && \
export XLA_PYTHON_CLIENT_MEM_FRACTION=.5 && \
python ../../train_conrft_octo.py "$@" \
    --exp_name=fold_towel \
    --checkpoint_path=/home/dungeon_master/conrft/examples/experiments/fold_towel/conrft/0222\
    --q_weight=1.0 \
    --bc_weight=0.1 \
    --demo_path=./demo_data/20260222/traj_20.pkl\
    --pretrain_steps=100000 \
    --debug=False \
    --learner
