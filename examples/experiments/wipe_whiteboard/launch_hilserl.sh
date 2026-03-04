#!/usr/bin/env bash
# 统一启动脚本：在同一个 tmux session 中同时启动 learner (window 0) 和 actor (window 1)
# RUN_TAG 在此处统一生成/接收，两个 window 使用完全相同的路径。
#
# 用法示例:
#   bash launch_hilserl.sh
#   USE_COV_ACTOR_LOSS=true bash launch_hilserl.sh
#   SAMPLING_STRATEGY=random_drop USE_COV_ACTOR_LOSS=true COV_K=16 bash launch_hilserl.sh
#   RUN_TAG=<existing_tag> bash launch_hilserl.sh   # 续训已有 checkpoint
#
# tmux 操作:
#   切换 window:  Ctrl-b n / Ctrl-b p  或  Ctrl-b 0/1
#   detach:       Ctrl-b d
#   kill session: tmux kill-session -t hl_wipe_whiteboard

set -euo pipefail

# ---- 可调参数 ----
SAMPLING_STRATEGY=${SAMPLING_STRATEGY:-none}
SAMPLING_KWARGS=${SAMPLING_KWARGS:-""}
USE_COV_ACTOR_LOSS=${USE_COV_ACTOR_LOSS:-false}
COV_K=${COV_K:-8}
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

# ---- 构建传给 learner 的额外参数字符串 ----
EXTRA_ARGS=""
if [ -n "$SAMPLING_KWARGS" ]; then
    EXTRA_ARGS+=" --sampling_strategy_kwargs=${SAMPLING_KWARGS}"
fi
if [ "$USE_COV_ACTOR_LOSS" = "true" ]; then
    EXTRA_ARGS+=" --use_cov_actor_loss --cov_K=${COV_K} --cov_q_low=${COV_Q_LOW} --cov_q_high=${COV_Q_HIGH}"
fi

# ---- tmux 配置 ----
SESSION="hl_wipe_whiteboard"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CKPT_PATH="/home/dungeon_master/conrft/examples/experiments/wipe_whiteboard/hilserl/${RUN_TAG}"
DEMO_PATH="/home/dungeon_master/conrft/examples/experiments/wipe_whiteboard/demo_data/20260229/traj_20.pkl"

echo "========================================"
echo " Task:    wipe_whiteboard"
echo " RUN_TAG: ${RUN_TAG}"
echo " CKPT:    ${CKPT_PATH}"
echo " Session: ${SESSION}"
echo "========================================"

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
export XLA_PYTHON_CLIENT_PREALLOCATE=false && \
export XLA_PYTHON_CLIENT_MEM_FRACTION=.3 && \
export LD_LIBRARY_PATH=\$CONDA_PREFIX/lib:\$LD_LIBRARY_PATH && \
python ../../train_rlpd.py \
  --exp_name=wipe_whiteboard \
  --checkpoint_path='${CKPT_PATH}' \
  --demo_path='${DEMO_PATH}' \
  --learner \
  --run_tag='${RUN_TAG}' \
  --sampling_strategy='${SAMPLING_STRATEGY}'${EXTRA_ARGS}"

# ---- Actor 命令 ----
ACTOR_CMD="cd '${SCRIPT_DIR}' && \
export XLA_PYTHON_CLIENT_PREALLOCATE=false && \
export XLA_PYTHON_CLIENT_MEM_FRACTION=.2 && \
export LD_LIBRARY_PATH=\$CONDA_PREFIX/lib:\$LD_LIBRARY_PATH && \
python ../../train_rlpd.py \
  --exp_name=wipe_whiteboard \
  --checkpoint_path='${CKPT_PATH}' \
  --actor"

# ---- 创建 tmux session ----
# window 0: learner
tmux new-session -d -s "$SESSION" -n "learner" -c "$SCRIPT_DIR"
tmux send-keys -t "${SESSION}:learner" "$LEARNER_CMD" Enter

# window 1: actor
tmux new-window -t "$SESSION" -n "actor" -c "$SCRIPT_DIR"
tmux send-keys -t "${SESSION}:actor" "$ACTOR_CMD" Enter

# 默认聚焦到 learner 窗口
tmux select-window -t "${SESSION}:learner"

echo "Started. Attaching to session '${SESSION}'..."
echo "  window 0 [learner]: ${CKPT_PATH}"
echo "  window 1 [actor]:   ${CKPT_PATH}"
tmux attach-session -t "$SESSION"
