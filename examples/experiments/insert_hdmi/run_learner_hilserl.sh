#!/usr/bin/env bash
# 采样策略选项:
#   none                 — 不做任何处理（默认）
#   workspace_filtering  — 空间范围截断过滤（策略 A）
#   random_drop          — 随机丢弃 15%（策略 B）
#   per                  — 优先经验重放 PER（策略 C）
#
# Cov Actor Loss (协方差熵截断):
#   USE_COV_ACTOR_LOSS=true  — 启用 cov actor loss
#   COV_K=8                  — MC 采样数 (默认 8)
#   COV_Q_LOW=0.05           — 下分位数 (默认 0.05)
#   COV_Q_HIGH=0.90          — 上分位数 (默认 0.90)
#
# 用法示例:
#   bash run_learner_hilserl.sh
#   SAMPLING_STRATEGY=random_drop bash run_learner_hilserl.sh
#   USE_COV_ACTOR_LOSS=true bash run_learner_hilserl.sh
#   USE_COV_ACTOR_LOSS=true COV_K=16 SAMPLING_STRATEGY=workspace_filtering bash run_learner_hilserl.sh

SAMPLING_STRATEGY=${SAMPLING_STRATEGY:-none}
SAMPLING_KWARGS=${SAMPLING_KWARGS:-""}
USE_COV_ACTOR_LOSS=${USE_COV_ACTOR_LOSS:-false}
COV_K=${COV_K:-8} ##1,4,8,16
COV_Q_LOW=${COV_Q_LOW:-0.05}
COV_Q_HIGH=${COV_Q_HIGH:-0.90}

export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=.3
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

EXTRA_ARGS=()
if [ -n "$SAMPLING_KWARGS" ]; then
    EXTRA_ARGS+=("--sampling_strategy_kwargs=${SAMPLING_KWARGS}")
fi
if [ "$USE_COV_ACTOR_LOSS" = "true" ]; then
    EXTRA_ARGS+=("--use_cov_actor_loss" "--cov_K=${COV_K}" "--cov_q_low=${COV_Q_LOW}" "--cov_q_high=${COV_Q_HIGH}")
fi

# ---- 自动构建 run tag: <sampling>__<cov_or_nocov>__<timestamp> ----
# 若外部已传入 RUN_TAG（用于续训），则跳过自动生成
if [ -z "${RUN_TAG}" ]; then
    TIMESTAMP=$(date +%m%d_%H%M%S)
    if [ "$USE_COV_ACTOR_LOSS" = "true" ]; then
        COV_LABEL="cov_K${COV_K}"
    else
        COV_LABEL="nocov"
    fi
    RUN_TAG="${SAMPLING_STRATEGY}__${COV_LABEL}__${TIMESTAMP}"
fi

python ../../train_rlpd.py "$@" \
    --exp_name=insert_hdmi \
    --checkpoint_path=/home/dungeon_master/conrft/examples/experiments/insert_hdmi/hilserl/${RUN_TAG} \
    --demo_path=./demo_data/traj.pkl \
    --learner \
    --run_tag="${RUN_TAG}" \
    --sampling_strategy="${SAMPLING_STRATEGY}" \
    "${EXTRA_ARGS[@]}"
