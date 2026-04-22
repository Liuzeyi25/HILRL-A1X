#!/usr/bin/env bash
# run_actor_hil_simple.sh
# ─────────────────────────────────────────────────────────────────────────────
# 原始 HIL-SERL Actor 启动脚本（无 Q 值修正、无对比损失）
# 对应训练脚本: examples/train_rlpd_hil_simple.py
#
# 用法示例:
#   bash run_actor_hil_simple.sh
#   RUN_TAG=none__nocov__0421_120000 bash run_actor_hil_simple.sh   # 续训
#   IP=192.168.1.10 bash run_actor_hil_simple.sh                    # 远程 Learner
# ─────────────────────────────────────────────────────────────────────────────

RUN_TAG=${RUN_TAG:-"hilserl"}
IP=${IP:-"localhost"}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CKPT_PATH="${SCRIPT_DIR}/checkpoints_hil_simple/${RUN_TAG}"

echo "========================================"
echo " [HIL-SIMPLE] Actor"
echo " RUN_TAG: ${RUN_TAG}"
echo " CKPT:    ${CKPT_PATH}"
echo " Learner: ${IP}"
echo "========================================"

export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=.2
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

python "${SCRIPT_DIR}/../../train_rlpd_hil_simple.py" "$@" \
    --exp_name=pour_water \
    --checkpoint_path="${CKPT_PATH}" \
    --ip="${IP}" \
    --actor

# ── 评估模式（取消注释并修改步数）──────────────────────────────────────────
#   --eval_checkpoint_step=20000 \
#   --eval_n_trajs=20 \
