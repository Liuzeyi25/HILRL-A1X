#!/usr/bin/env bash
# =============================================================================
# push_to_github.sh
# 将当前工程提交并推送到 GitHub (origin/main)
# 用法：
#   bash push_to_github.sh                     # 自动生成提交信息（含时间戳）
#   bash push_to_github.sh "你的提交信息"      # 指定提交信息
# =============================================================================

set -e   # 任何命令失败立即退出

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_DIR"

BRANCH="main"
REMOTE="origin"

# ── 提交信息 ─────────────────────────────────────────────────────────────────
if [ -n "$1" ]; then
    COMMIT_MSG="$1"
else
    TIMESTAMP=$(date "+%Y-%m-%d %H:%M")
    COMMIT_MSG="update ${TIMESTAMP}"
fi

echo ""
echo "================================================="
echo "  仓库路径 : $REPO_DIR"
echo "  远端     : $REMOTE → $(git remote get-url $REMOTE)"
echo "  分支     : $BRANCH"
echo "  提交信息 : $COMMIT_MSG"
echo "================================================="
echo ""

# ── 检查是否有变更 ────────────────────────────────────────────────────────────
if git diff --quiet && git diff --cached --quiet && \
   [ -z "$(git ls-files --others --exclude-standard)" ]; then
    echo "✅ 没有需要提交的变更，工作区已是最新状态。"
    exit 0
fi

# ── 显示将要提交的文件 ────────────────────────────────────────────────────────
echo "📋 变更文件列表："
git status --short
echo ""

# ── 暂存所有变更（新文件 + 修改 + 删除）────────────────────────────────────────
git add -A

# ── 提交 ─────────────────────────────────────────────────────────────────────
git commit -m "$COMMIT_MSG"

# ── 推送 ─────────────────────────────────────────────────────────────────────
echo ""
echo "⬆️  正在推送到 $REMOTE/$BRANCH ..."
git push "$REMOTE" "$BRANCH"

echo ""
echo "✅ 推送成功！"
echo "   链接：$(git remote get-url $REMOTE | sed 's/\.git$//')/commits/$BRANCH"
