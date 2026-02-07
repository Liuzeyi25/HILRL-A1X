#!/bin/bash
# Launch script for A1_X ConRFT learner with pretraining
# --checkpoint_path=/home/dungeon_master/conrft/octo_model/octo-small-1.5/300000/default  \

export XLA_PYTHON_CLIENT_PREALLOCATE=false && \
export XLA_PYTHON_CLIENT_MEM_FRACTION=.8 && \
python ../../train_conrft_octo.py "$@" \
    --exp_name=a1x_pick_banana \
    --q_weight=0.1 \
    --bc_weight=1.0 \
    --checkpoint_path=/home/dungeon_master/conrft/examples/experiments/a1x_pick_banana/conrft/0207 \
    --demo_path=./demo_data/20260207/a1x_pick_banana_15_demos.pkl \
    --pretrain_steps=40000 \
    --debug=False \
    --learner 

