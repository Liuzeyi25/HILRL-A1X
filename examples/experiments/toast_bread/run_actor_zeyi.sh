RUN_TAG=${RUN_TAG:-"0502_2_multi-bc"}

export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=.2

python ../../train_rlpd_hil_bc.py \
    --exp_name=toast_bread \
    --checkpoint_path=/home/dungeon_master/liuzeyi/HILRL-A1X/examples/experiments/toast_bread/checkpoints_hil/${RUN_TAG} \
    --actor\
    --progress_model_path=/home/dungeon_master/liuzeyi/HILRL-A1X/examples/experiments/toast_bread/progress_model/exp_001_20260430_1102//progress_model_best.pt \
    --state_stats_path=/home/dungeon_master/liuzeyi/HILRL-A1X/examples/experiments/toast_bread/progress_model/exp_001_20260430_1102/state_stats.pt \
    --bc_post_steps=5