RUN_TAG=${RUN_TAG:-"0422_3"}

export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=.5
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

# wandb 配置：强制 online 上传，entity 填你的 wandb 用户名
export WANDB_MODE=online
export WANDB_ENTITY=liuzeyicsu-central-south-university

# 有 Progress Model 时：
python ../../train_rlpd_hil.py \
    --exp_name=insert_block \
    --checkpoint_path=experiments/insert_block/checkpoints_hil/${RUN_TAG} \
    --demo_path=/home/dungeon_master/liuzeyi/HILRL-A1X/examples/experiments/insert_block/demo_data/20260418/merged_demo_data.pkl \
    --learner \
    --run_tag="${RUN_TAG}" \
    --alpha_lambda=3.0 \
    --contrastive_coef=0.2 \
    --preference_batch_size=3 \
    --suboptimal_window=5 \
    --progress_model_path=/home/dungeon_master/liuzeyi/HILRL-A1X/examples/experiments/insert_block/progress_model/exp_003_20260421_1242//progress_model_best.pt \
    --state_stats_path=/home/dungeon_master/liuzeyi/HILRL-A1X/examples/experiments/insert_block/progress_model/exp_003_20260421_1242/state_stats.pt