export XLA_PYTHON_CLIENT_PREALLOCATE=false && \
export XLA_PYTHON_CLIENT_MEM_FRACTION=.2 && \
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH && \
python ../../train_rlpd.py "$@" \
    --exp_name=insert_block \
    --checkpoint_path=/home/dungeon_master/conrft/examples/experiments/insert_block/hilserl/0229method3 \
    --eval \
    --eval_steps 26000 \
    --eval_episodes 10 
    # --eval_checkpoint_step=26000 \