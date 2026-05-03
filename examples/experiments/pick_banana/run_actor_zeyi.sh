RUN_TAG=${RUN_TAG:-"0503_2_hil_bc"}

export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=.2

python ../../train_rlpd_hil_bc.py \
    --exp_name=pick_banana \
    --checkpoint_path=checkpoints_hil/${RUN_TAG} \
    --actor\
    --progress_model_path=/home/dungeon_master/liuzeyi/HILRL-A1X/examples/experiments/pick_banana/progress_model/exp_001_20260503_1112/progress_model_best.pt \
    --state_stats_path=/home/dungeon_master/liuzeyi/HILRL-A1X/examples/experiments/pick_banana/progress_model/exp_001_20260503_1112/state_stats.pt \
    --bc_post_steps=5