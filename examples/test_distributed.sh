#!/bin/bash

# 分布式训练快速测试脚本
# 用于在单机上同时启动 learner 和 actor 进行通信测试

set -e

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 配置
EXP_NAME="a1x_pick_banana"
DEMO_PATH="experiments/a1x_pick_banana/demo_data/traj_001_manual_2026-02-04_20-24-03.pkl"
CHECKPOINT_PATH="checkpoints/test_distributed"
PRETRAIN_STEPS=100
IP="localhost"

echo -e "${BLUE}=====================================================================${NC}"
echo -e "${BLUE}🌐 分布式训练通信测试${NC}"
echo -e "${BLUE}=====================================================================${NC}"
echo ""

# 检查是否在正确的目录
if [ ! -f "train_conrft_octo.py" ]; then
    echo -e "${RED}❌ 错误: 请在 examples 目录下运行此脚本${NC}"
    echo -e "${YELLOW}提示: cd /home/dungeon_master/conrft/examples${NC}"
    exit 1
fi

# 检查 demo 文件是否存在
if [ ! -f "$DEMO_PATH" ]; then
    echo -e "${RED}❌ 错误: Demo 文件不存在: $DEMO_PATH${NC}"
    echo -e "${YELLOW}提示: 请先录制 demo 数据${NC}"
    exit 1
fi

echo -e "${GREEN}✅ 环境检查通过${NC}"
echo ""

# 清理旧的 checkpoint
if [ -d "$CHECKPOINT_PATH" ]; then
    echo -e "${YELLOW}⚠️  发现旧的 checkpoint 目录，是否删除? [y/N]${NC}"
    read -r response
    if [[ "$response" =~ ^([yY][eE][sS]|[yY])$ ]]; then
        rm -rf "$CHECKPOINT_PATH"
        echo -e "${GREEN}✅ 已删除旧 checkpoint${NC}"
    fi
fi

echo ""
echo -e "${BLUE}=====================================================================${NC}"
echo -e "${BLUE}📋 测试配置${NC}"
echo -e "${BLUE}=====================================================================${NC}"
echo -e "实验名称:       ${YELLOW}$EXP_NAME${NC}"
echo -e "Demo 路径:      ${YELLOW}$DEMO_PATH${NC}"
echo -e "Checkpoint:     ${YELLOW}$CHECKPOINT_PATH${NC}"
echo -e "预训练步数:     ${YELLOW}$PRETRAIN_STEPS${NC}"
echo -e "IP 地址:        ${YELLOW}$IP${NC}"
echo -e "${BLUE}=====================================================================${NC}"
echo ""

# 创建日志目录
LOG_DIR="logs/test_distributed_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"
echo -e "${GREEN}✅ 日志目录: $LOG_DIR${NC}"
echo ""

# 步骤 1: 启动 Learner
echo -e "${BLUE}=====================================================================${NC}"
echo -e "${BLUE}🎓 步骤 1/3: 启动 Learner${NC}"
echo -e "${BLUE}=====================================================================${NC}"

LEARNER_LOG="$LOG_DIR/learner.log"

python train_conrft_octo.py \
  --exp_name="$EXP_NAME" \
  --learner \
  --ip="$IP" \
  --demo_path="$DEMO_PATH" \
  --checkpoint_path="$CHECKPOINT_PATH" \
  --pretrain_steps=$PRETRAIN_STEPS \
  --debug \
  > "$LEARNER_LOG" 2>&1 &

LEARNER_PID=$!
echo -e "${GREEN}✅ Learner 启动 (PID: $LEARNER_PID)${NC}"
echo -e "${BLUE}ℹ️  日志: $LEARNER_LOG${NC}"
echo ""

# 等待 learner 完成预训练
echo -e "${YELLOW}⏳ 等待 Learner 完成预训练...${NC}"
echo -e "${BLUE}ℹ️  可以运行以下命令查看实时日志:${NC}"
echo -e "   ${YELLOW}tail -f $LEARNER_LOG${NC}"
echo ""

# 等待 "sent initial network to actor" 出现在日志中
MAX_WAIT=300  # 最多等待 5 分钟
WAIT_COUNT=0
while [ $WAIT_COUNT -lt $MAX_WAIT ]; do
    if grep -q "sent initial network to actor" "$LEARNER_LOG"; then
        echo -e "${GREEN}✅ Learner 预训练完成并准备就绪!${NC}"
        break
    fi
    
    if ! kill -0 $LEARNER_PID 2>/dev/null; then
        echo -e "${RED}❌ Learner 进程异常退出${NC}"
        echo -e "${RED}最后 20 行日志:${NC}"
        tail -n 20 "$LEARNER_LOG"
        exit 1
    fi
    
    sleep 1
    WAIT_COUNT=$((WAIT_COUNT + 1))
    
    # 每 10 秒显示一次进度
    if [ $((WAIT_COUNT % 10)) -eq 0 ]; then
        echo -e "${YELLOW}  等待中... ${WAIT_COUNT}/${MAX_WAIT} 秒${NC}"
    fi
done

if [ $WAIT_COUNT -ge $MAX_WAIT ]; then
    echo -e "${RED}❌ 等待超时${NC}"
    echo -e "${RED}最后 50 行日志:${NC}"
    tail -n 50 "$LEARNER_LOG"
    kill $LEARNER_PID 2>/dev/null || true
    exit 1
fi

echo ""

# 步骤 2: 启动 Actor
echo -e "${BLUE}=====================================================================${NC}"
echo -e "${BLUE}🎮 步骤 2/3: 启动 Actor${NC}"
echo -e "${BLUE}=====================================================================${NC}"

ACTOR_LOG="$LOG_DIR/actor.log"

python train_conrft_octo.py \
  --exp_name="$EXP_NAME" \
  --actor \
  --ip="$IP" \
  --demo_path="$DEMO_PATH" \
  --checkpoint_path="$CHECKPOINT_PATH" \
  --debug \
  > "$ACTOR_LOG" 2>&1 &

ACTOR_PID=$!
echo -e "${GREEN}✅ Actor 启动 (PID: $ACTOR_PID)${NC}"
echo -e "${BLUE}ℹ️  日志: $ACTOR_LOG${NC}"
echo ""

# 等待 actor 开始运行
echo -e "${YELLOW}⏳ 等待 Actor 连接到 Learner...${NC}"
sleep 5

if ! kill -0 $ACTOR_PID 2>/dev/null; then
    echo -e "${RED}❌ Actor 进程异常退出${NC}"
    echo -e "${RED}最后 20 行日志:${NC}"
    tail -n 20 "$ACTOR_LOG"
    kill $LEARNER_PID 2>/dev/null || true
    exit 1
fi

if grep -q "starting actor loop" "$ACTOR_LOG"; then
    echo -e "${GREEN}✅ Actor 成功连接到 Learner!${NC}"
else
    echo -e "${YELLOW}⚠️  Actor 可能还在初始化中${NC}"
fi

echo ""

# 步骤 3: 监控运行
echo -e "${BLUE}=====================================================================${NC}"
echo -e "${BLUE}📊 步骤 3/3: 监控训练${NC}"
echo -e "${BLUE}=====================================================================${NC}"
echo -e "${GREEN}✅ 分布式训练已启动!${NC}"
echo ""
echo -e "${BLUE}进程信息:${NC}"
echo -e "  Learner PID: ${YELLOW}$LEARNER_PID${NC}"
echo -e "  Actor PID:   ${YELLOW}$ACTOR_PID${NC}"
echo ""
echo -e "${BLUE}日志文件:${NC}"
echo -e "  Learner: ${YELLOW}$LEARNER_LOG${NC}"
echo -e "  Actor:   ${YELLOW}$ACTOR_LOG${NC}"
echo ""
echo -e "${BLUE}实时查看日志:${NC}"
echo -e "  Learner: ${YELLOW}tail -f $LEARNER_LOG${NC}"
echo -e "  Actor:   ${YELLOW}tail -f $ACTOR_LOG${NC}"
echo ""
echo -e "${BLUE}停止训练:${NC}"
echo -e "  ${YELLOW}kill $LEARNER_PID $ACTOR_PID${NC}"
echo -e "${BLUE}=====================================================================${NC}"
echo ""

# 监控 30 秒
echo -e "${YELLOW}监控运行状态 (30秒)...${NC}"
for i in {1..30}; do
    if ! kill -0 $LEARNER_PID 2>/dev/null; then
        echo -e "${RED}❌ Learner 进程已退出${NC}"
        break
    fi
    
    if ! kill -0 $ACTOR_PID 2>/dev/null; then
        echo -e "${RED}❌ Actor 进程已退出${NC}"
        break
    fi
    
    echo -ne "\r${GREEN}✅ 运行中... ${i}/30 秒${NC}"
    sleep 1
done

echo ""
echo ""

# 显示最新日志
echo -e "${BLUE}=====================================================================${NC}"
echo -e "${BLUE}📋 最新日志摘要${NC}"
echo -e "${BLUE}=====================================================================${NC}"
echo ""
echo -e "${YELLOW}Learner 最后 10 行:${NC}"
tail -n 10 "$LEARNER_LOG"
echo ""
echo -e "${YELLOW}Actor 最后 10 行:${NC}"
tail -n 10 "$ACTOR_LOG"
echo ""

# 询问是否继续运行
echo -e "${BLUE}=====================================================================${NC}"
echo -e "${YELLOW}是否继续运行? [y/N]${NC}"
read -r -t 10 response || response="n"

if [[ "$response" =~ ^([yY][eE][sS]|[yY])$ ]]; then
    echo -e "${GREEN}✅ 继续运行...${NC}"
    echo -e "${YELLOW}按 Ctrl+C 停止训练${NC}"
    
    # 设置陷阱以在退出时清理
    trap "echo ''; echo -e '${YELLOW}正在停止进程...${NC}'; kill $LEARNER_PID $ACTOR_PID 2>/dev/null || true; echo -e '${GREEN}✅ 已停止${NC}'; exit 0" SIGINT SIGTERM
    
    # 等待进程结束
    wait $LEARNER_PID $ACTOR_PID
else
    echo -e "${YELLOW}正在停止进程...${NC}"
    kill $LEARNER_PID $ACTOR_PID 2>/dev/null || true
    sleep 2
    echo -e "${GREEN}✅ 测试完成${NC}"
fi

echo ""
echo -e "${BLUE}=====================================================================${NC}"
echo -e "${GREEN}🎉 分布式训练通信测试完成!${NC}"
echo -e "${BLUE}=====================================================================${NC}"
echo -e "${BLUE}日志保存在: ${YELLOW}$LOG_DIR${NC}"
echo -e "${BLUE}=====================================================================${NC}"
