#!/usr/bin/env bash
# =============================================================================
# pull_from_github.sh
# 从 GitHub (origin/main) 拉取最新更新
# 用法：
#   bash pull_from_github.sh          # 直接拉取，本地有未提交变更时自动 stash
# =============================================================================

set -e

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_DIR"

BRANCH="main"
REMOTE="origin"

echo ""
echo "================================================="
echo "  仓库路径 : $REPO_DIR"
echo "  远端     : $REMOTE → $(git remote get-url $REMOTE)"
echo "  分支     : $BRANCH"
echo "================================================="
echo ""

# ── 检查本地是否有未提交的变更 ────────────────────────────────────────────────
STASHED=false
if ! git diff --quiet || ! git diff --cached --quiet; then
    echo "⚠️  检测到本地未提交的变更，自动 stash 保存..."
    git stash push -m "auto-stash before pull $(date '+%Y-%m-%d %H:%M')"
    STASHED=true
fi

# ── 切换到目标分支 ────────────────────────────────────────────────────────────
CURRENT_BRANCH=$(git rev-parse --abbrev-ref HEAD)
if [ "$CURRENT_BRANCH" != "$BRANCH" ]; then
    echo "⚠️  当前分支为 $CURRENT_BRANCH，切换到 $BRANCH ..."
    git checkout "$BRANCH"
fi

# ── 拉取远端最新 ──────────────────────────────────────────────────────────────
echo "⬇️  正在从 $REMOTE/$BRANCH 拉取..."
git pull "$REMOTE" "$BRANCH"

# ── 恢复 stash ────────────────────────────────────────────────────────────────
if [ "$STASHED" = true ]; then
    echo ""
    echo "♻️  恢复之前 stash 的本地变更..."
    if git stash pop; then
        echo "✅ 本地变更已恢复。"
    else
        echo "⚠️  stash 恢复时出现冲突，请手动解决后运行："
        echo "    git stash pop"
    fi
fi

echo ""
echo "✅ 拉取完成！当前最新提交："
git log --oneline -3
