#!/usr/bin/env bash
# launch_hil_simple.sh
# ─────────────────────────────────────────────────────────────────────────────
# 原始 HIL-SERL 统一启动脚本（无 Q 值修正、无对比损失）
# 在同一个 tmux session 中同时启动 Learner (window 0) 和 Actor (window 1)。
# RUN_TAG 在此统一生成，两个 window 使用完全相同的 checkpoint 路径。
#
# 前置要求: tmux 已安装（brew install tmux 或 apt install tmux）
#
# 用法示例:
#   bash launch_hil_simple.sh
#   USE_COV_ACTOR_LOSS=true bash launch_hil_simple.sh
#   SAMPLING_STRATEGY=random_drop bash launch_hil_simple.sh
#   RUN_TAG=none__nocov__0421_120000 bash launch_hil_simple.sh   # 续训
#
# tmux 快捷键:
#   切换 window:  Ctrl-b n / Ctrl-b p  或  Ctrl-b 0 / Ctrl-b 1
#   detach:       Ctrl-b d
#   kill session: tmux kill-session -t hl_simple_pour_water
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

# ── 可调参数 ─────────────────────────────────────────────────────────────────
SAMPLING_STRATEGY=${SAMPLING_STRATEGY:-none}
SAMPLING_KWARGS=${SAMPLING_KWARGS:-""}
USE_COV_ACTOR_LOSS=${USE_COV_ACTOR_LOSS:-false}
COV_K=${COV_K:-8}
COV_Q_LOW=${COV_Q_LOW:-0.05}
COV_Q_HIGH=${COV_Q_HIGH:-0.90}
IP=${IP:-"localhost"}

# ── 统一生成 RUN_TAG ──────────────────────────────────────────────────────────
if [ -z "${RUN_TAG:-}" ]; then
    TIMESTAMP=$(date +%m%d_%H%M%S)
    if [ "$USE_COV_ACTOR_LOSS" = "true" ]; then
        COV_LABEL="cov_K${COV_K}"
    else
        COV_LABEL="nocov"
    fi
    RUN_TAG="${SAMPLING_STRATEGY}__${COV_LABEL}__${TIMESTAMP}"
fi

# ── 路径配置 ──────────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CKPT_PATH="${SCRIPT_DIR}/checkpoints_hil_simple/${RUN_TAG}"
DEMO_PATH="${SCRIPT_DIR}/demo_data/traj.pkl"
TRAIN_SCRIPT="${SCRIPT_DIR}/../../train_rlpd_hil_simple.py"

# ── 构建 Learner 额外参数字符串 ────────────────────────────────────────────────
EXTRA_ARGS=""
if [ -n "$SAMPLING_KWARGS" ]; then
    EXTRA_ARGS+=" --sampling_strategy_kwargs=${SAMPLING_KWARGS}"
fi
if [ "$USE_COV_ACTOR_LOSS" = "true" ]; then
    EXTRA_ARGS+=" --use_cov_actor_loss --cov_K=${COV_K} --cov_q_low=${COV_Q_LOW} --cov_q_high=${COV_Q_HIGH}"
fi

# ── 打印配置摘要 ───────────────────────────────────────────────────────────────
echo "========================================"
echo " Task:              pour_water"
echo " Mode:              HIL-SIMPLE (no correction)"
echo " RUN_TAG:           ${RUN_TAG}"
echo " CKPT:              ${CKPT_PATH}"
echo " DEMO:              ${DEMO_PATH}"
echo " sampling_strategy: ${SAMPLING_STRATEGY}"
echo " use_cov_actor_loss:${USE_COV_ACTOR_LOSS}"
echo " Learner IP:        ${IP}"
echo "========================================"

# ── tmux session 处理 ─────────────────────────────────────────────────────────
SESSION="hl_simple_pour_water"
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

# ── Learner 命令 ───────────────────────────────────────────────────────────────
LEARNER_CMD="cd '${SCRIPT_DIR}' && \
export XLA_PYTHON_CLIENT_PREALLOCATE=false && \
export XLA_PYTHON_CLIENT_MEM_FRACTION=.3 && \
export LD_LIBRARY_PATH=\$CONDA_PREFIX/lib:\$LD_LIBRARY_PATH && \
python '${TRAIN_SCRIPT}' \
  --exp_name=pour_water \
  --checkpoint_path='${CKPT_PATH}' \
  --demo_path='${DEMO_PATH}' \
  --learner \
  --run_tag='${RUN_TAG}' \
  --sampling_strategy='${SAMPLING_STRATEGY}'${EXTRA_ARGS}"

# ── Actor 命令 ─────────────────────────────────────────────────────────────────
ACTOR_CMD="cd '${SCRIPT_DIR}' && \
export XLA_PYTHON_CLIENT_PREALLOCATE=false && \
export XLA_PYTHON_CLIENT_MEM_FRACTION=.2 && \
export LD_LIBRARY_PATH=\$CONDA_PREFIX/lib:\$LD_LIBRARY_PATH && \
python '${TRAIN_SCRIPT}' \
  --exp_name=pour_water \
  --checkpoint_path='${CKPT_PATH}' \
  --ip='${IP}' \
  --actor"

# ── 创建 tmux session ──────────────────────────────────────────────────────────
# window 0: learner
tmux new-session -d -s "$SESSION" -n "learner" -c "$SCRIPT_DIR"
tmux send-keys -t "${SESSION}:learner" "$LEARNER_CMD" Enter

# window 1: actor
tmux new-window -t "$SESSION" -n "actor" -c "$SCRIPT_DIR"
tmux send-keys -t "${SESSION}:actor" "$ACTOR_CMD" Enter

# 默认聚焦 learner 窗口
tmux select-window -t "${SESSION}:learner"

echo ""
echo "✅ 已启动 tmux session '${SESSION}'"
echo "   window 0 [learner]: ${CKPT_PATH}"
echo "   window 1 [actor]:   ${CKPT_PATH}"
echo ""
echo "Attaching..."
tmux attach-session -t "$SESSION"
