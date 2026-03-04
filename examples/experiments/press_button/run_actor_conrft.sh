#!/bin/bash
# Launch script for A1_X ConRFT actor
#    --eval_checkpoint_step=100000 \
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH && \
export XLA_PYTHON_CLIENT_PREALLOCATE=false && \
export XLA_PYTHON_CLIENT_MEM_FRACTION=.5 && \
python ../../train_conrft_octo.py "$@" \
    --exp_name=press_button \
    --checkpoint_path=/home/dungeon_master/conrft/examples/experiments/press_button/conrft/20260224 \
    --actor 
    
    
