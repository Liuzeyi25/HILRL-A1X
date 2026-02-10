# 🌐 分布式训练通信测试指南

## 📋 目录
1. [通信原理](#通信原理)
2. [单机测试（推荐先做）](#单机测试)
3. [多机分布式测试](#多机分布式测试)
4. [通信诊断工具](#通信诊断工具)
5. [常见问题排查](#常见问题排查)

---

## 🔧 通信原理

### 端口配置
在 `serl_launcher/serl_launcher/utils/launcher.py` 中：
```python
def make_trainer_config(port_number: int = 3333, broadcast_port: int = 3334):
    return TrainerConfig(
        port_number=3333,        # 用于数据传输和请求
        broadcast_port=3334,     # 用于广播网络参数
        request_types=["send-stats"],
    )
```

### 通信流程
```
Learner (Server)                    Actor (Client)
    |                                    |
    | 1. 启动服务器监听                   |
    |    port 3333, 3334                |
    |                                    |
    |<------------ 2. 连接请求 ----------|
    |                                    |
    | 3. 发送初始网络参数 --------------->|
    |    (via broadcast_port 3334)      |
    |                                    |
    |<-------- 4. 发送经验数据 ----------|
    |    (via port 3333)                |
    |                                    |
    | 5. 训练更新模型                     |
    |                                    |
    | 6. 广播新参数 -------------------->|
    |    (每 steps_per_update 步)        |
    |                                    |
    |<-------- 7. 发送统计信息 ----------|
    |    client.request("send-stats")   |
```

---

## 🖥️ 单机测试（推荐先做）

### 步骤 1: 准备 Demo 数据
```bash
# 确保你有 demo 数据
ls examples/experiments/a1x_pick_banana/demo_data/
```

### 步骤 2: 启动 Learner (终端 1)
```bash
cd /home/dungeon_master/conrft/examples

# 启动 learner，使用 localhost
python train_conrft_octo.py \
  --exp_name=a1x_pick_banana \
  --learner \
  --ip=localhost \
  --demo_path=experiments/a1x_pick_banana/demo_data/traj_001_manual_2026-02-04_20-24-03.pkl \
  --checkpoint_path=checkpoints/test_distributed \
  --pretrain_steps=100 \
  --debug
```

**预期输出:**
```
======================================================================
🎓 Learner 模式启动
======================================================================
使用设备: [cuda(id=0)]
设备数量: 1
默认后端: gpu
======================================================================

 Pretraining the model with demo data
📊 预训练配置:
   - 起始步数: 0
   - 目标步数: 100
   ...
 sent initial network to actor  ← 🎯 关键：等待这条消息
```

### 步骤 3: 启动 Actor (终端 2)
```bash
cd /home/dungeon_master/conrft/examples

# 等待 learner 完成预训练并显示 "sent initial network to actor"
python train_conrft_octo.py \
  --exp_name=a1x_pick_banana \
  --actor \
  --ip=localhost \
  --demo_path=experiments/a1x_pick_banana/demo_data/traj_001_manual_2026-02-04_20-24-03.pkl \
  --checkpoint_path=checkpoints/test_distributed \
  --debug
```

**预期输出:**
```
 starting actor loop  ← 成功连接到 learner
```

### 步骤 4: 观察通信
在 **Learner 终端** 你应该看到：
```
Filling up replay buffer: 100%|████████| 100/100
 sent initial network to actor
learner: 100%|████████| ...
```

在 **Actor 终端** 你应该看到：
```
actor loop running...
last return: XX.XX
```

---

## 🌍 多机分布式测试

### 前提条件
1. **两台机器在同一局域网**
2. **防火墙开放端口 3333 和 3334**
3. **知道 Learner 机器的 IP 地址**

### 获取 Learner IP
在 **Learner 机器** 上运行：
```bash
# 方法 1: 使用 hostname
hostname -I | awk '{print $1}'

# 方法 2: 使用 ip 命令
ip addr show | grep "inet " | grep -v 127.0.0.1 | awk '{print $2}' | cut -d/ -f1

# 方法 3: 使用 ifconfig (如果安装了)
ifconfig | grep "inet " | grep -v 127.0.0.1 | awk '{print $2}'
```

假设得到的 IP 是: `192.168.1.100`

### 测试端口连通性

#### 在 Learner 机器上:
```bash
# 安装 netcat (如果没有)
sudo apt-get install netcat

# 测试端口监听
nc -l 3333  # 在一个终端
nc -l 3334  # 在另一个终端
```

#### 在 Actor 机器上:
```bash
# 测试能否连接到 learner
nc -zv 10.87.117.249 3333
nc -zv 10.87.117.249 3334

# 预期输出：
# Connection to 192.168.1.100 3333 port [tcp/*] succeeded!
# Connection to 192.168.1.100 3334 port [tcp/*] succeeded!
```

### 启动分布式训练

#### Learner 机器:
```bash
cd /home/dungeon_master/conrft/examples

python train_conrft_octo.py \
  --exp_name=a1x_pick_banana \
  --learner \
  --ip=0.0.0.0 \
  --demo_path=experiments/a1x_pick_banana/demo_data/traj_001_manual_2026-02-04_20-24-03.pkl \
  --checkpoint_path=checkpoints/distributed_test \
  --pretrain_steps=2000 \
  --debug
```

**注意:** 使用 `--ip=0.0.0.0` 让服务器监听所有网络接口

#### Actor 机器:
```bash
cd /home/dungeon_master/conrft/examples

python train_conrft_octo.py \
  --exp_name=a1x_pick_banana \
  --actor \
  --ip=192.168.1.100 \
  --demo_path=experiments/a1x_pick_banana/demo_data/traj_001_manual_2026-02-04_20-24-03.pkl \
  --checkpoint_path=checkpoints/distributed_test \
  --debug
```

**注意:** 使用 learner 的实际 IP 地址

---

## 🔍 通信诊断工具

我已经为你创建了一个诊断脚本，请看下一个文件。

---

## ❗ 常见问题排查

### 1. Actor 连接超时
```
TimeoutError: Waiting for server timed out after 3000ms
```

**原因:**
- Learner 未启动
- IP 地址错误
- 防火墙阻止端口

**解决:**
```bash
# 在 learner 机器检查端口是否监听
sudo netstat -tulpn | grep -E '3333|3334'

# 或使用 ss
sudo ss -tulpn | grep -E '3333|3334'

# 应该看到:
# tcp   LISTEN   0   128   0.0.0.0:3333   0.0.0.0:*   users:(("python",pid=XXX,fd=X))
# tcp   LISTEN   0   128   0.0.0.0:3334   0.0.0.0:*   users:(("python",pid=XXX,fd=X))
```

### 2. 防火墙配置

#### Ubuntu/Debian:
```bash
# 检查防火墙状态
sudo ufw status

# 开放端口
sudo ufw allow 3333/tcp
sudo ufw allow 3334/tcp

# 重新加载
sudo ufw reload
```

#### CentOS/RHEL:
```bash
# 检查防火墙
sudo firewall-cmd --list-ports

# 开放端口
sudo firewall-cmd --permanent --add-port=3333/tcp
sudo firewall-cmd --permanent --add-port=3334/tcp
sudo firewall-cmd --reload
```

### 3. 网络参数未更新
**症状:** Actor 一直使用旧的策略

**检查:**
- Learner 是否调用 `server.publish_network()`
- Actor 的 `update_params` 回调是否被调用

**解决:** 在代码中添加调试打印：
```python
# 在 learner 中
def learner(...):
    ...
    if step % config.steps_per_update == 0:
        print(f"🔄 [Learner] Publishing network at step {step}")
        server.publish_network(agent.state.params)

# 在 actor 中
def update_params(params):
    print(f"✅ [Actor] Received new params!")
    nonlocal agent
    agent = agent.replace(state=agent.state.replace(params=params))
```

### 4. 数据未传输到 Learner
**症状:** Replay buffer 一直不增长

**检查:**
```python
# 在 learner 循环中添加
while len(replay_buffer) < config.training_starts:
    print(f"Buffer size: {len(replay_buffer)}/{config.training_starts}")
    time.sleep(1)
```

### 5. 自定义端口
如果端口冲突，可以修改：

```python
# 在 train_conrft_octo.py 中添加 flags
flags.DEFINE_integer("port_number", 3333, "Port for data transfer")
flags.DEFINE_integer("broadcast_port", 3334, "Port for broadcasting")

# 修改 trainer config
client = TrainerClient(
    "actor_env",
    FLAGS.ip,
    make_trainer_config(
        port_number=FLAGS.port_number,
        broadcast_port=FLAGS.broadcast_port
    ),
    ...
)

server = TrainerServer(
    make_trainer_config(
        port_number=FLAGS.port_number,
        broadcast_port=FLAGS.broadcast_port
    ),
    ...
)
```

---

## 📊 监控通信状态

### 实时监控网络流量
```bash
# 监控特定端口的流量
sudo tcpdump -i any port 3333 or port 3334 -n

# 或使用 iftop (需要安装)
sudo iftop -i eth0 -f "port 3333 or port 3334"
```

### 查看连接状态
```bash
# 查看建立的连接
watch -n 1 "netstat -tn | grep -E '3333|3334'"
```

---

## 🎯 验证成功的标志

### Learner 端:
✅ 显示 "sent initial network to actor"
✅ Replay buffer 逐渐填充
✅ 定期打印训练损失
✅ 定期发布网络参数

### Actor 端:
✅ 成功连接到 learner
✅ 接收到初始网络参数
✅ 能够与环境交互
✅ 数据成功插入 data_store
✅ 定期接收更新的网络参数

### 网络层面:
✅ 端口 3333, 3334 处于 ESTABLISHED 状态
✅ 双向数据流动
✅ 无连接重置或超时

---

## 🚀 快速测试脚本

见 `test_distributed_communication.py` 文件
