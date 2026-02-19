# GelloIntervention 配置对比

## 原始命令行方式 vs 环境 Wrapper 方式

### 方式 1: 命令行运行（launch_yaml.py）

```bash
# 原始的独立运行方式
python experiments/launch_yaml.py --left-config-path configs/yam_A1_X.yaml
```

**特点：**
- 独立运行，不在训练环境中
- 直接控制机器人
- 用于测试 Gello 连接和映射

---

### 方式 2: 作为环境 Wrapper（GelloIntervention）

```python
# 集成到训练环境中
env = GelloIntervention(
    env,
    left_config_path="Gello/gello_software/configs/yam_A1_X.yaml",
    control_rate_hz=30
)
```

**特点：**
- 作为环境的一部分
- 与策略训练集成
- 用于人工示教和数据采集

---

## 是的，YAML 配置已经写进去了！

### 在配置文件中的体现

`examples/experiments/a1x_pick_banana/config.py`:

```python
class TrainConfig(DefaultTrainingConfig):
    teleoperation_device = "gello"
    
    # 🆕 新版配置
    gello_config_path = "Gello/gello_software/configs/yam_A1_X.yaml"
    
    def get_environment(self, ...):
        if self.teleoperation_device == "gello":
            env = GelloIntervention(
                env,
                left_config_path=self.gello_config_path,  # ← 使用配置文件
                control_rate_hz=30
            )
```

---

## YAML 配置文件内容

`Gello/gello_software/configs/yam_A1_X.yaml`:

```yaml
# Robot 配置（A1_X 机器人）
robot:
  _target_: gello.robots.A1_X.A1XRobot
  num_dofs: 7
  node_name: "a1x_gello_node"
  port: 6100

# Agent 配置（Gello 遥控设备）
agent:
  _target_: gello.agents.gello_agent.GelloAgent
  port: /dev/serial/by-id/usb-FTDI_USB__-__Serial_Converter_FTA7NNNU-if00-port0
  dynamixel_config:
    joint_ids: [1, 2, 3, 4, 5, 6]
    joint_offsets: [1.5708, 4.71239, 4.71239, 3.14159, 1.5708, 3.14159]
    joint_signs: [1.0, -1.0, -1.0, -1.0, 1.0, 1.0]
    gripper_config: [7, 139.66015625, 199.16015625]

# 控制频率
hz: 500
```

---

## 工作流程

```
启动训练脚本
    ↓
config.py 加载
    ↓
TrainConfig.get_environment()
    ↓
创建 GelloIntervention
    ↓
读取 yam_A1_X.yaml
    ↓
初始化 Agent (GelloAgent)
    ↓
初始化 Robot (A1XRobot)
    ↓
启动 ZMQ 服务器
    ↓
等待空格键启用干预
    ↓
Agent.act() 获取 Gello 位置
    ↓
机器人执行动作
```

---

## 对比总结

| 方面 | 命令行方式 | Wrapper 方式 |
|------|-----------|-------------|
| **运行命令** | `python launch_yaml.py --left-config-path ...` | 自动集成在训练中 |
| **配置方式** | 命令行参数 | 在 config.py 中设置 |
| **使用场景** | 测试 Gello | 训练+示教 |
| **YAML 文件** | 必需 | 必需 |
| **Agent 创建** | launch_yaml.py | GelloIntervention |
| **控制方式** | 连续运行 | 按空格键切换 |

---

## 关键点

### ✅ 是的，YAML 已经写进去了

在 `config.py` 中：
```python
gello_config_path = "Gello/gello_software/configs/yam_A1_X.yaml"
```

### ✅ 使用相同的 YAML 配置

- 命令行：`--left-config-path configs/yam_A1_X.yaml`
- Wrapper：`left_config_path="Gello/gello_software/configs/yam_A1_X.yaml"`

**它们使用同一个配置文件！**

### ✅ 工作方式

1. **命令行方式**：专门用于测试 Gello 控制
   ```bash
   cd Gello/gello_software
   python experiments/launch_yaml.py --left-config-path configs/yam_A1_X.yaml
   ```

2. **Wrapper 方式**：集成到训练环境中
   ```bash
   cd /home/dungeon_master/conrft
   python examples/train_conrft_octo.py  # 自动加载配置
   ```

---

## 实际使用

### 测试 Gello 连接
```bash
# 使用命令行方式测试
cd Gello/gello_software
python experiments/launch_yaml.py --left-config-path configs/yam_A1_X.yaml
```

### 训练时使用 Gello
```bash
# 配置已经写在 config.py 中，直接运行训练
python examples/train_conrft_octo.py

# 运行时按空格键启用 Gello 干预
```

### 验证配置
```bash
# 检查 YAML 配置是否正确
cat Gello/gello_software/configs/yam_A1_X.yaml

# 运行测试脚本
python examples/verify_action_space2.py
```

---

## 总结

**简短回答：** 是的，YAML 配置路径已经写进 `config.py` 了：

```python
gello_config_path = "Gello/gello_software/configs/yam_A1_X.yaml"
```

**实现方式：** GelloIntervention 内部使用与 `launch_yaml.py` 完全相同的逻辑来加载和使用这个 YAML 文件，包括：
- 加载 YAML 配置
- 创建 Agent（GelloAgent）
- 创建 Robot（A1XRobot）
- 启动 ZMQ 服务器
- 运行控制循环

**区别：** Wrapper 方式把这些逻辑集成到环境中，而不是独立运行。
