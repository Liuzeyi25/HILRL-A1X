#!/bin/bash
# 启动 A1X ROS2 ZMQ 桥接节点

set -e

echo "========================================"
echo "  启动 A1X ROS2 ZMQ 桥接节点"
echo "========================================"
echo ""

# 检查 ROS2 环境
if [ -z "$ROS_DISTRO" ]; then
    echo "⚠️  ROS2 环境未设置，尝试加载..."
    if [ -f "/opt/ros/humble/setup.bash" ]; then
        source /opt/ros/humble/setup.bash
        echo "✓ 已加载 ROS2 Humble"
    elif [ -f "/opt/ros/foxy/setup.bash" ]; then
        source /opt/ros/foxy/setup.bash
        echo "✓ 已加载 ROS2 Foxy"
    else
        echo "❌ 未找到 ROS2 安装"
        exit 1
    fi
else
    echo "✓ ROS2 环境: $ROS_DISTRO"
fi

# 设置参数
PORT=${1:-6100}
NODE_NAME=${2:-"a1x_serl_node"}
USE_CUROBO_IK=${USE_CUROBO_IK:-"true"}
CUROBO_IK_SERVICE=${CUROBO_IK_SERVICE:-"tcp://127.0.0.1:6202"}

echo ""
echo "配置参数:"
echo "  ZMQ 端口: $PORT (命令), $((PORT+1)) (状态)"
echo "  节点名称: $NODE_NAME"
echo "  使用 CuRobo IK: $USE_CUROBO_IK"
if [ "$USE_CUROBO_IK" = "true" ]; then
    echo "  CuRobo IK 服务: $CUROBO_IK_SERVICE"
fi
echo ""

# 检查 CuRobo IK 服务是否运行（如果启用）
if [ "$USE_CUROBO_IK" = "true" ]; then
    echo "检查 CuRobo IK 服务..."
    timeout 2 python3 -c "
import zmq
ctx = zmq.Context()
sock = ctx.socket(zmq.REQ)
sock.setsockopt(zmq.RCVTIMEO, 1000)
sock.setsockopt(zmq.SNDTIMEO, 1000)
try:
    sock.connect('$CUROBO_IK_SERVICE')
    sock.send_json({'cmd': 'ping'})
    sock.recv_json()
    print('✓ CuRobo IK 服务已连接')
except Exception as e:
    print('⚠️  无法连接到 CuRobo IK 服务')
    print('   请先启动: python scripts/curobo_ik_service.py --bind tcp://0.0.0.0:6202')
finally:
    sock.close()
    ctx.term()
" 2>/dev/null || echo "⚠️  CuRobo IK 服务未响应"
    echo ""
fi

# 构建命令
CMD="python3 serl_robot_infra/franka_env/robots/a1x_ros2_node.py --port $PORT --node-name $NODE_NAME"

if [ "$USE_CUROBO_IK" = "true" ]; then
    CMD="$CMD --use-curobo-ik --curobo-ik-service $CUROBO_IK_SERVICE"
fi

echo "========================================"
echo "  启动节点..."
echo "========================================"
echo ""
echo "命令: $CMD"
echo ""
echo "提示: 按 Ctrl+C 停止节点"
echo ""

# 启动节点
exec $CMD
