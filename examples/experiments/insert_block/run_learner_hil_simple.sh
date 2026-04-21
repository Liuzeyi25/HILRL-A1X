#!/usr/bin/env bash
# run_learner_hil_simple.sh
# ─────────────────────────────────────────────────────────────────────────────
# 原始 HIL-SERL Learner 启动脚本（无 Q 值修正、无对比损失）
# 对应训练脚本: examples/train_rlpd_hil_simple.py
#
# 可调环境变量（均有默认值，不传则用默认）:
#   SAMPLING_STRATEGY   — none / workspace_filtering / random_drop / per (默认 none)
#   SAMPLING_KWARGS     — JSON 字符串，传给采样策略，例如 '{"drop_ratio":0.15}'
#   USE_COV_ACTOR_LOSS  — true / false (默认 false)
#   COV_K               — MC 采样数 (默认 8)
#   COV_Q_LOW           — 下分位数 (默认 0.05)
#   COV_Q_HIGH          — 上分位数 (默认 0.90)
#   RUN_TAG             — 续训时手动指定已有的 tag，留空则自动生成
#
# 用法示例:
#   bash run_learner_hil_simple.sh
#   SAMPLING_STRATEGY=random_drop bash run_learner_hil_simple.sh
#   USE_COV_ACTOR_LOSS=true COV_K=16 bash run_learner_hil_simple.sh
#   RUN_TAG=none__nocov__0421_120000 bash run_learner_hil_simple.sh   # 续训
# ─────────────────────────────────────────────────────────────────────────────

SAMPLING_STRATEGY=${SAMPLING_STRATEGY:-none}
SAMPLING_KWARGS=${SAMPLING_KWARGS:-""}
USE_COV_ACTOR_LOSS=${USE_COV_ACTOR_LOSS:-false}
COV_K=${COV_K:-8}
COV_Q_LOW=${COV_Q_LOW:-0.05}
COV_Q_HIGH=${COV_Q_HIGH:-0.90}

# ── 自动生成 RUN_TAG（续训时外部传入 RUN_TAG 即可跳过）─────────────────────
if [ -z "${RUN_TAG:-}" ]; then
    TIMESTAMP=$(date +%m%d_%H%M%S)
    if [ "$USE_COV_ACTOR_LOSS" = "true" ]; then
        COV_LABEL="cov_K${COV_K}"
    else
        COV_LABEL="nocov"
    fi
    RUN_TAG="${SAMPLING_STRATEGY}__${COV_LABEL}__${TIMESTAMP}"
fi

# ── 构建可选参数数组 ─────────────────────────────────────────────────────────
EXTRA_ARGS=()
if [ -n "$SAMPLING_KWARGS" ]; then
    EXTRA_ARGS+=("--sampling_strategy_kwargs=${SAMPLING_KWARGS}")
fi
if [ "$USE_COV_ACTOR_LOSS" = "true" ]; then
    EXTRA_ARGS+=(
        "--use_cov_actor_loss"
        "--cov_K=${COV_K}"
        "--cov_q_low=${COV_Q_LOW}"
        "--cov_q_high=${COV_Q_HIGH}"
    )
fi

# ── 路径配置（按需修改）─────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CKPT_PATH="${SCRIPT_DIR}/checkpoints_hil_simple/${RUN_TAG}"
DEMO_PATH="${SCRIPT_DIR}/demo_data/traj.pkl"

echo "========================================"
echo " [HIL-SIMPLE] Learner"
echo " RUN_TAG: ${RUN_TAG}"
echo " CKPT:    ${CKPT_PATH}"
echo " DEMO:    ${DEMO_PATH}"
echo "========================================"

export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=.3
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

python "${SCRIPT_DIR}/../../train_rlpd_hil_simple.py" "$@" \
    --exp_name=pour_water \
    --checkpoint_path="${CKPT_PATH}" \
    --demo_path="${DEMO_PATH}" \
    --learner \
    --run_tag="${RUN_TAG}" \
    --sampling_strategy="${SAMPLING_STRATEGY}" \
    "${EXTRA_ARGS[@]}"
