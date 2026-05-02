RUN_TAG=${RUN_TAG:-"0502_1_multi-bc"}

export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=.2

python ../../train_rlpd_hil_bc.py \
    --exp_name=a1x_pick_banana \
    --checkpoint_path=/home/dungeon_master/liuzeyi/HILRL-A1X/examples/experiments/a1x_pick_banana/checkpoints_hil/${RUN_TAG} \
    --actor\
    --progress_model_path=/home/dungeon_master/liuzeyi/HILRL-A1X/examples/experiments/a1x_pick_banana/progress_model/exp_006_20260429_1046//progress_model_best.pt \
    --state_stats_path=/home/dungeon_master/liuzeyi/HILRL-A1X/examples/experiments/a1x_pick_banana/progress_model/exp_006_20260429_1046/state_stats.pt \
    --bc_post_steps=5