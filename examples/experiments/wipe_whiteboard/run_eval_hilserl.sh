export XLA_PYTHON_CLIENT_PREALLOCATE=false && \
export XLA_PYTHON_CLIENT_MEM_FRACTION=.2 && \
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH && \
python ../../train_rlpd.py "$@" \
    --exp_name=wipe_whiteboard \
    --checkpoint_path=/home/dungeon_master/conrft/examples/experiments/wipe_whiteboard/hilserl/0226 \
    --eval \
    --eval_steps 44000 \
    --eval_episodes 10 
    # --eval_checkpoint_step=26000 \