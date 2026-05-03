RUN_TAG=${RUN_TAG:-"0503_2_hil_bc"}

export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=.5
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

# wandb 配置：强制 online 上传，entity 填你的 wandb 用户名
export WANDB_MODE=online
export WANDB_ENTITY=liuzeyicsu-central-south-university

# 有 Progress Model 时：
python ../../train_rlpd_hil_bc.py \
    --exp_name=pick_banana \
    --checkpoint_path=checkpoints_hil/${RUN_TAG} \
    --demo_path=/home/dungeon_master/liuzeyi/HILRL-A1X/examples/experiments/pick_banana/demo_data/20260503/traj_20.pkl \
    --learner \
    --run_tag="${RUN_TAG}" \
    --alpha_lambda=3.0 \
    --contrastive_coef=1 \
    --preference_batch_size=4 \
    --suboptimal_window=5 \
    --progress_model_path=/home/dungeon_master/liuzeyi/HILRL-A1X/examples/experiments/pick_banana/progress_model/exp_001_20260503_1112/progress_model_best.pt \
    --state_stats_path=/home/dungeon_master/liuzeyi/HILRL-A1X/examples/experiments/pick_banana/progress_model/exp_001_20260503_1112/state_stats.pt \
    --bc_post_steps=5