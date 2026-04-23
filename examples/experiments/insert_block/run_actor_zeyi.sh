RUN_TAG=${RUN_TAG:-"0423_1_multi-bc"}

export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=.2

python ../../train_rlpd_hil_bc.py \
    --exp_name=insert_block \
    --checkpoint_path=experiments/insert_block/checkpoints_hil/${RUN_TAG} \
    --actor\
    --progress_model_path=/home/dungeon_master/liuzeyi/HILRL-A1X/examples/experiments/insert_block/progress_model/exp_003_20260421_1242//progress_model_best.pt \
    --state_stats_path=/home/dungeon_master/liuzeyi/HILRL-A1X/examples/experiments/insert_block/progress_model/exp_003_20260421_1242/state_stats.pt \
    --bc_post_steps=5