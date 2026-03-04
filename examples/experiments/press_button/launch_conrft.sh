#!/usr/bin/env bash
# 统一启动脚本 (ConRFT): 在同一个 tmux session 中左右分屏启动 learner 和 actor
# RUN_TAG 在此处统一生成/接收，两个 pane 使用完全相同的路径。
#
# ---- ConRFT 特有参数 ----
#   Q_WEIGHT=1.0          — Q-loss 权重 (默认 1.0)
#   BC_WEIGHT=0.1         — BC-loss 权重 (默认 0.1)
#   PRETRAIN_STEPS=96000  — 预训练步数 (默认 10000)
#
# ---- 采样策略 / Cov Actor Loss (与 hilserl 共享) ----
#   SAMPLING_STRATEGY=none|workspace_filtering|random_drop|per
#   USE_COV_ACTOR_LOSS=true  COV_K=4  COV_Q_LOW=0.05  COV_Q_HIGH=0.90
#
# 用法示例:
#   bash launch_conrft.sh                                          # 新训练
#   Q_WEIGHT=2.0 BC_WEIGHT=0.05 bash launch_conrft.sh             # 自定义权重
#   USE_COV_ACTOR_LOSS=true bash launch_conrft.sh                  # 启用 cov loss
#   RUN_TAG=none__nocov__0302_150000 bash launch_conrft.sh         # 续训
#
# tmux 操作:
#   切换 pane:    Ctrl-b ← / →
#   detach:       Ctrl-b d
#   kill session: tmux kill-session -t cr_press_button

set -euo pipefail

# ---- ConRFT 特有参数 ----
Q_WEIGHT=${Q_WEIGHT:-1.0}
BC_WEIGHT=${BC_WEIGHT:-0.1}
PRETRAIN_STEPS=${PRETRAIN_STEPS:-10000}
DEBUG=${DEBUG:-False}

# ---- 采样策略 ----
SAMPLING_STRATEGY=${SAMPLING_STRATEGY:-none}
SAMPLING_KWARGS=${SAMPLING_KWARGS:-""}

# ---- Cov Actor Loss ----
USE_COV_ACTOR_LOSS=${USE_COV_ACTOR_LOSS:-false}
COV_K=${COV_K:-4}
COV_Q_LOW=${COV_Q_LOW:-0.05}
COV_Q_HIGH=${COV_Q_HIGH:-0.90}

# ---- 统一生成 RUN_TAG（若外部已传入则复用，用于续训）----
if [ -z "${RUN_TAG:-}" ]; then
    TIMESTAMP=$(date +%m%d_%H%M%S)
    if [ "$USE_COV_ACTOR_LOSS" = "true" ]; then
        COV_LABEL="cov_K${COV_K}"
    else
        COV_LABEL="nocov"
    fi
    RUN_TAG="${SAMPLING_STRATEGY}__${COV_LABEL}__${TIMESTAMP}"
fi

# ---- 构建额外参数 ----
EXTRA_ARGS=""
if [ -n "$SAMPLING_KWARGS" ]; then
    EXTRA_ARGS+=" --sampling_strategy_kwargs=${SAMPLING_KWARGS}"
fi
if [ "$USE_COV_ACTOR_LOSS" = "true" ]; then
    EXTRA_ARGS+=" --use_cov_actor_loss=True --cov_K=${COV_K} --cov_q_low=${COV_Q_LOW} --cov_q_high=${COV_Q_HIGH}"
fi

# ---- 路径配置 ----
SESSION="cr_press_button"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CKPT_PATH="/home/dungeon_master/conrft/examples/experiments/press_button/conrft/${RUN_TAG}"
DEMO_PATH="${SCRIPT_DIR}/demo_data/20260222/traj_20.pkl"

# ---- 首次打印所有配置 ----
echo ""
echo "╔════════════════════════════════════════════════════════════╗"
echo "║              ConRFT Launch — press_button                  ║"
echo "╠════════════════════════════════════════════════════════════╣"
echo "║  RUN_TAG:          ${RUN_TAG}"
echo "║  CKPT_PATH:        ${CKPT_PATH}"
echo "║  DEMO_PATH:        ${DEMO_PATH}"
echo "║────────────────────────────────────────────────────────────"
echo "║  Q_WEIGHT:         ${Q_WEIGHT}"
echo "║  BC_WEIGHT:        ${BC_WEIGHT}"
echo "║  PRETRAIN_STEPS:   ${PRETRAIN_STEPS}"
echo "║  DEBUG:            ${DEBUG}"
echo "║────────────────────────────────────────────────────────────"
echo "║  SAMPLING_STRATEGY:  ${SAMPLING_STRATEGY}"
echo "║  SAMPLING_KWARGS:    ${SAMPLING_KWARGS:-<empty>}"
echo "║────────────────────────────────────────────────────────────"
echo "║  USE_COV_ACTOR_LOSS: ${USE_COV_ACTOR_LOSS}"
if [ "$USE_COV_ACTOR_LOSS" = "true" ]; then
echo "║    COV_K:            ${COV_K}"
echo "║    COV_Q_LOW:        ${COV_Q_LOW}"
echo "║    COV_Q_HIGH:       ${COV_Q_HIGH}"
fi
echo "╚════════════════════════════════════════════════════════════╝"
echo ""

# ---- 若 session 已存在则询问是否复用 ----
if tmux has-session -t "$SESSION" 2>/dev/null; then
    echo "[WARNING] tmux session '${SESSION}' already exists."
    read -rp "Kill and restart? [y/N] " ans
    if [[ "$ans" =~ ^[Yy]$ ]]; then
        tmux kill-session -t "$SESSION"
    else
        echo "Attaching to existing session..."
        tmux attach-session -t "$SESSION"
        exit 0
    fi
fi

# ---- Learner 命令 ----
LEARNER_CMD="cd '${SCRIPT_DIR}' && \
conda activate conrft && \
export XLA_PYTHON_CLIENT_PREALLOCATE=false && \
export XLA_PYTHON_CLIENT_MEM_FRACTION=.5 && \
export LD_LIBRARY_PATH=\$CONDA_PREFIX/lib:\$LD_LIBRARY_PATH && \
python ../../train_conrft_octo.py \
  --exp_name=press_button \
  --checkpoint_path='${CKPT_PATH}' \
  --demo_path='${DEMO_PATH}' \
  --q_weight=${Q_WEIGHT} \
  --bc_weight=${BC_WEIGHT} \
  --pretrain_steps=${PRETRAIN_STEPS} \
  --debug=${DEBUG} \
  --sampling_strategy='${SAMPLING_STRATEGY}' \
  --learner${EXTRA_ARGS}"

# ---- Actor 命令 ----
ACTOR_CMD="cd '${SCRIPT_DIR}' && \
conda activate conrft && \
export XLA_PYTHON_CLIENT_PREALLOCATE=false && \
export XLA_PYTHON_CLIENT_MEM_FRACTION=.5 && \
export LD_LIBRARY_PATH=\$CONDA_PREFIX/lib:\$LD_LIBRARY_PATH && \
python ../../train_conrft_octo.py \
  --exp_name=press_button \
  --checkpoint_path='${CKPT_PATH}' \
  --actor"

# ---- 创建 tmux session（单 window，左右分屏）----
tmux new-session -d -s "$SESSION" -n "main" -c "$SCRIPT_DIR"
tmux send-keys -t "${SESSION}:main.0" "$LEARNER_CMD" Enter

tmux split-window -t "${SESSION}:main" -h -c "$SCRIPT_DIR"
tmux send-keys -t "${SESSION}:main.1" "$ACTOR_CMD" Enter

tmux select-pane -t "${SESSION}:main.0"

echo "Started. Attaching to session '${SESSION}'..."
echo "  左侧 pane [learner]: ${CKPT_PATH}"
echo "  右侧 pane [actor]:   ${CKPT_PATH}"
echo ""
echo "  tmux 操作提示:"
echo "    切换 pane:   Ctrl-b ← / →"
echo "    detach:      Ctrl-b d"
echo "    kill session: tmux kill-session -t ${SESSION}"
tmux attach-session -t "$SESSION"
