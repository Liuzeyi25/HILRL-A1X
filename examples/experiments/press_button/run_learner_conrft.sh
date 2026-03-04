#!/usr/bin/env bash
# Launch script for A1_X ConRFT learner
#
# 采样策略选项:
#   none                 — 不做任何处理（默认）
#   workspace_filtering  — 空间范围截断过滤（策略 A）
#   random_drop          — 随机丢弃 15%（策略 B）
#   per                  — 优先经验重放 PER（策略 C）
#
# Diffusion-adapted Cov Actor Loss:
#   使用去噪重建误差作为 proxy log-prob 的协方差掩码 actor loss。
#   设 USE_COV_ACTOR_LOSS=true 启用。
#
# 用法示例:
#   bash run_learner_conrft.sh                                     # 默认
#   SAMPLING_STRATEGY=random_drop bash run_learner_conrft.sh
#   SAMPLING_STRATEGY=workspace_filtering \
#     SAMPLING_KWARGS='{"x_range":[0.2,0.8]}' \
#     bash run_learner_conrft.sh
#   USE_COV_ACTOR_LOSS=true COV_K=8 bash run_learner_conrft.sh

SAMPLING_STRATEGY=${SAMPLING_STRATEGY:-none}
SAMPLING_KWARGS=${SAMPLING_KWARGS:-""}

# Cov Actor Loss params (diffusion-adapted)
USE_COV_ACTOR_LOSS=${USE_COV_ACTOR_LOSS:-false}
COV_K=${COV_K:-4}
COV_Q_LOW=${COV_Q_LOW:-0.05}
COV_Q_HIGH=${COV_Q_HIGH:-0.90}

export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=.5

EXTRA_ARGS=()
if [ -n "$SAMPLING_KWARGS" ]; then
    EXTRA_ARGS+=("--sampling_strategy_kwargs=${SAMPLING_KWARGS}")
fi
if [ "$USE_COV_ACTOR_LOSS" = "true" ]; then
    EXTRA_ARGS+=("--use_cov_actor_loss=True")
    EXTRA_ARGS+=("--cov_K=${COV_K}")
    EXTRA_ARGS+=("--cov_q_low=${COV_Q_LOW}")
    EXTRA_ARGS+=("--cov_q_high=${COV_Q_HIGH}")
fi

python ../../train_conrft_octo.py "$@" \
    --exp_name=press_button \
    --checkpoint_path=/home/dungeon_master/conrft/examples/experiments/press_button/conrft/20260224 \
    --q_weight=1.0 \
    --bc_weight=0.1 \
    --demo_path=./demo_data/20260222/traj_20.pkl \
    --pretrain_steps=10000 \
    --debug=False \
    --sampling_strategy="${SAMPLING_STRATEGY}" \
    "${EXTRA_ARGS[@]}" \
    --learner
