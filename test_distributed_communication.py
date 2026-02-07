#!/usr/bin/env python3
"""
分布式训练通信测试脚本
用于测试 learner 和 actor 之间的网络通信

使用方法:
1. 测试端口连通性:
   python test_distributed_communication.py --mode=test_ports --learner_ip=192.168.1.100

2. 启动测试 learner:
   python test_distributed_communication.py --mode=test_learner

3. 启动测试 actor (在另一台机器或另一个终端):
   python test_distributed_communication.py --mode=test_actor --learner_ip=192.168.1.100
"""

import socket
import time
import sys
from absl import app, flags
import numpy as np

FLAGS = flags.FLAGS

flags.DEFINE_enum("mode", "test_ports", 
                  ["test_ports", "test_learner", "test_actor"],
                  "测试模式")
flags.DEFINE_string("learner_ip", "localhost", "Learner 的 IP 地址")
flags.DEFINE_integer("port_number", 3333, "数据传输端口")
flags.DEFINE_integer("broadcast_port", 3334, "广播端口")


def print_green(msg):
    print(f"\033[92m✅ {msg}\033[00m")


def print_red(msg):
    print(f"\033[91m❌ {msg}\033[00m")


def print_yellow(msg):
    print(f"\033[93m⚠️  {msg}\033[00m")


def print_blue(msg):
    print(f"\033[94mℹ️  {msg}\033[00m")


def test_port_connectivity(ip, port, timeout=5):
    """测试端口连通性"""
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(timeout)
        result = sock.connect_ex((ip, port))
        sock.close()
        return result == 0
    except socket.error as e:
        return False


def get_local_ip():
    """获取本机 IP 地址"""
    try:
        # 创建一个 UDP socket
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        # 不需要真正连接，只是为了获取本地 IP
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"


def test_ports_mode():
    """测试端口连通性模式"""
    print("\n" + "=" * 70)
    print("🔍 端口连通性测试")
    print("=" * 70)
    print(f"目标 IP: {FLAGS.learner_ip}")
    print(f"数据传输端口: {FLAGS.port_number}")
    print(f"广播端口: {FLAGS.broadcast_port}")
    print("=" * 70)
    print()

    # 测试数据传输端口
    print(f"测试端口 {FLAGS.port_number}...", end=" ", flush=True)
    if test_port_connectivity(FLAGS.learner_ip, FLAGS.port_number):
        print_green(f"端口 {FLAGS.port_number} 可连接")
    else:
        print_red(f"端口 {FLAGS.port_number} 无法连接")
        print_yellow("可能原因:")
        print("  1. Learner 未启动")
        print("  2. 防火墙阻止了端口")
        print("  3. IP 地址错误")

    # 测试广播端口
    print(f"测试端口 {FLAGS.broadcast_port}...", end=" ", flush=True)
    if test_port_connectivity(FLAGS.learner_ip, FLAGS.broadcast_port):
        print_green(f"端口 {FLAGS.broadcast_port} 可连接")
    else:
        print_red(f"端口 {FLAGS.broadcast_port} 无法连接")

    print()
    print("=" * 70)
    print("💡 提示:")
    print("=" * 70)
    print("如果端口无法连接，请在 Learner 机器上检查:")
    print(f"  sudo netstat -tulpn | grep -E '{FLAGS.port_number}|{FLAGS.broadcast_port}'")
    print()
    print("或开放防火墙端口:")
    print(f"  sudo ufw allow {FLAGS.port_number}/tcp")
    print(f"  sudo ufw allow {FLAGS.broadcast_port}/tcp")
    print("=" * 70)


def test_learner_mode():
    """测试 Learner 服务器模式"""
    print("\n" + "=" * 70)
    print("🎓 Learner 测试服务器")
    print("=" * 70)
    
    local_ip = get_local_ip()
    print(f"本机 IP: {local_ip}")
    print(f"监听端口: {FLAGS.port_number}, {FLAGS.broadcast_port}")
    print("=" * 70)
    print()

    try:
        from agentlace.trainer import TrainerServer, TrainerConfig
        from agentlace.data.data_store import QueuedDataStore
        
        print_blue("创建数据存储...")
        data_store = QueuedDataStore(1000)
        
        print_blue("创建训练配置...")
        config = TrainerConfig(
            port_number=FLAGS.port_number,
            broadcast_port=FLAGS.broadcast_port,
            request_types=["send-stats", "test-message"],
        )
        
        print_blue("创建 TrainerServer...")
        
        stats_received = []
        
        def stats_callback(msg_type: str, payload: dict) -> dict:
            """处理来自 actor 的消息"""
            print_green(f"收到消息: type={msg_type}, payload={payload}")
            stats_received.append((msg_type, payload))
            return {"status": "received"}
        
        server = TrainerServer(config, request_callback=stats_callback)
        server.register_data_store("test_data", data_store)
        
        print_blue("启动服务器...")
        server.start(threaded=True)
        
        print_green("服务器启动成功!")
        print()
        print("=" * 70)
        print("📡 等待 Actor 连接...")
        print("=" * 70)
        print(f"在 Actor 机器上运行:")
        print(f"  python test_distributed_communication.py \\")
        print(f"    --mode=test_actor \\")
        print(f"    --learner_ip={local_ip} \\")
        print(f"    --port_number={FLAGS.port_number} \\")
        print(f"    --broadcast_port={FLAGS.broadcast_port}")
        print("=" * 70)
        print()
        
        # 模拟发送网络参数
        print_blue("准备发送测试网络参数...")
        time.sleep(2)
        
        test_params = {
            "test_layer_1": np.random.randn(10, 10).tolist(),
            "test_layer_2": np.random.randn(5, 5).tolist(),
        }
        
        print_blue("广播网络参数...")
        server.publish_network(test_params)
        print_green("网络参数已广播!")
        
        # 保持运行
        print()
        print_yellow("服务器运行中... (按 Ctrl+C 停止)")
        
        try:
            while True:
                time.sleep(1)
                print(f"\r数据存储大小: {len(data_store)} | 收到消息数: {len(stats_received)}", 
                      end="", flush=True)
        except KeyboardInterrupt:
            print()
            print_yellow("收到停止信号")
            
    except ImportError as e:
        print_red(f"导入错误: {e}")
        print_yellow("请确保安装了 agentlace:")
        print("  pip install agentlace")
    except Exception as e:
        print_red(f"错误: {e}")
        import traceback
        traceback.print_exc()


def test_actor_mode():
    """测试 Actor 客户端模式"""
    print("\n" + "=" * 70)
    print("🎮 Actor 测试客户端")
    print("=" * 70)
    print(f"连接到 Learner: {FLAGS.learner_ip}")
    print(f"端口: {FLAGS.port_number}, {FLAGS.broadcast_port}")
    print("=" * 70)
    print()

    # 先测试连通性
    print_blue("测试连通性...")
    if not test_port_connectivity(FLAGS.learner_ip, FLAGS.port_number):
        print_red(f"无法连接到 {FLAGS.learner_ip}:{FLAGS.port_number}")
        print_yellow("请确保 Learner 已启动!")
        return
    print_green("连通性测试通过!")
    print()

    try:
        from agentlace.trainer import TrainerClient, TrainerConfig
        from agentlace.data.data_store import QueuedDataStore
        
        print_blue("创建数据存储...")
        data_store = QueuedDataStore(1000)
        
        print_blue("创建训练配置...")
        config = TrainerConfig(
            port_number=FLAGS.port_number,
            broadcast_port=FLAGS.broadcast_port,
            request_types=["send-stats", "test-message"],
        )
        
        print_blue("创建 TrainerClient...")
        client = TrainerClient(
            "test_actor",
            FLAGS.learner_ip,
            config,
            data_stores={"test_data": data_store},
            wait_for_server=True,
            timeout_ms=10000,  # 10秒超时
        )
        
        print_green("成功连接到 Learner!")
        print()
        
        # 注册网络参数回调
        params_received = []
        
        def update_params(params):
            print_green(f"收到网络参数更新!")
            print(f"  参数键: {list(params.keys())}")
            params_received.append(params)
        
        print_blue("注册网络参数回调...")
        client.recv_network_callback(update_params)
        print_green("回调注册成功!")
        print()
        
        # 发送测试数据
        print_blue("发送测试数据到 Learner...")
        for i in range(5):
            test_transition = {
                "observation": np.random.randn(84, 84, 3).tolist(),
                "action": np.random.randn(7).tolist(),
                "reward": float(np.random.rand()),
                "done": bool(i == 4),
            }
            data_store.insert(test_transition)
            print(f"  发送数据 {i+1}/5")
            time.sleep(0.5)
        
        print_green("测试数据发送完成!")
        print()
        
        # 发送统计信息
        print_blue("发送统计信息...")
        stats = {
            "episode_return": 100.5,
            "episode_length": 250,
            "success_rate": 0.85,
        }
        response = client.request("send-stats", stats)
        print_green(f"收到响应: {response}")
        print()
        
        # 触发同步
        print_blue("触发数据同步...")
        client.update()
        print_green("同步完成!")
        print()
        
        # 等待接收网络参数
        print_yellow("等待接收网络参数... (10秒)")
        for i in range(10):
            time.sleep(1)
            if params_received:
                print_green(f"已收到 {len(params_received)} 次网络参数更新!")
                break
            print(f"\r等待中... {i+1}/10", end="", flush=True)
        
        print()
        if not params_received:
            print_yellow("未收到网络参数更新（Learner 可能还未广播）")
        
        print()
        print("=" * 70)
        print_green("测试完成!")
        print("=" * 70)
        
    except ImportError as e:
        print_red(f"导入错误: {e}")
        print_yellow("请确保安装了 agentlace:")
        print("  pip install agentlace")
    except Exception as e:
        print_red(f"错误: {e}")
        import traceback
        traceback.print_exc()


def main(_):
    print("\n" + "=" * 70)
    print("🌐 分布式训练通信测试工具")
    print("=" * 70)
    print(f"模式: {FLAGS.mode}")
    print("=" * 70)
    
    if FLAGS.mode == "test_ports":
        test_ports_mode()
    elif FLAGS.mode == "test_learner":
        test_learner_mode()
    elif FLAGS.mode == "test_actor":
        test_actor_mode()
    else:
        print_red(f"未知模式: {FLAGS.mode}")
        sys.exit(1)


if __name__ == "__main__":
    app.run(main)
