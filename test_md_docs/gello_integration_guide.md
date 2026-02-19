# Gello 集成迁移指南

## 📋 概述

本指南说明如何将 Gello 控制从独立脚本迁移到环境包装器（Wrapper）模式，实现与 SpaceMouse 一致的集成方式。

---

## ✨ 优势对比

### 之前的方式 (bidirectional_teleoperation.py)

```python
# ❌ 独立控制脚本，与环境分离
teleop = BidirectionalTeleoperation()
teleop.start()
teleop.set_mode(TeleoperationMode.NORMAL)

while True:
    gello_joints = gello_agent.act({})
    a1x_robot.update_command(gello_joints)
```

**问题**：
- 🔴 与环境代码分离，难以集成到训练/测试流程
- 🔴 无法与其他 wrapper 组合使用
- 🔴 需要单独管理控制循环
- 🔴 与 SpaceMouse 的集成方式不一致

### 现在的方式 (GelloIntervention Wrapper)

```python
# ✅ 通过 wrapper 优雅集成
env = PickBananaEnv()
env = GelloIntervention(env, port="/dev/ttyUSB0")  # 添加一行即可！
env = RelativeFrame(env)
env = SERLObsWrapper(env)

# 正常使用环境
obs, info = env.reset()
action = policy(obs)  # 策略动作
obs, rew, done, _, info = env.step(action)

# Gello 自动介入
if "intervene_action" in info:
    print("Gello 正在控制机器人")
```

**优势**：
- ✅ 统一的集成方式（与 SpaceMouse 一致）
- ✅ 可与其他 wrapper 自由组合
- ✅ 无需修改主循环代码
- ✅ 自动处理介入逻辑
- ✅ 支持录制演示数据

---

## 🏗️ 架构设计

### 1. GelloExpert (底层硬件接口)

位置: `serl_robot_infra/franka_env/gello/gello_expert.py`

```python
class GelloExpert:
    """与 SpaceMouseExpert 平行的 Gello 接口"""
    
    def __init__(self, port: str):
        self.gello_agent = GelloAgent(port=port)
    
    def get_action(self) -> Tuple[np.ndarray, list]:
        """返回 (关节位置, 按钮状态)"""
        joint_state = self.gello_agent.act({})
        arm_action = joint_state[:7]
        buttons = self._extract_gripper_state(joint_state)
        return arm_action, buttons
```

### 2. GelloIntervention (环境包装器)

位置: `serl_robot_infra/franka_env/envs/wrappers.py`

```python
class GelloIntervention(gym.ActionWrapper):
    """Gello 介入包装器（支持双向控制）"""
    
    def __init__(self, env, port, intervention_threshold=0.01, sync_on_reset=True):
        self.expert = GelloExpert(port)
        self.intervention_threshold = intervention_threshold
        self.sync_on_reset = sync_on_reset
    
    def action(self, action: np.ndarray):
        """检测 Gello 是否移动，决定是否介入"""
        expert_a, buttons = self.expert.get_action()
        
        # 检测移动
        if self._is_moved(expert_a):
            return expert_a, True  # 介入
        return action, False  # 不介入
    
    def step(self, action):
        new_action, intervened = self.action(action)
        obs, rew, done, truncated, info = self.env.step(new_action)
        
        if intervened:
            info["intervene_action"] = new_action
        
        return obs, rew, done, truncated, info
    
    def reset(self, **kwargs):
        """
        Reset 时支持双向控制：
        1. 环境 reset 到初始位置
        2. Gello 自动跟随机械臂到 reset 位置
        3. 同步完成后，Gello 恢复自由模式供人类控制
        """
        obs, info = self.env.reset(**kwargs)
        
        if self.sync_on_reset:
            # 获取机械臂的 reset 关节位置
            robot_joints = self._get_robot_joint_state(obs, info)
            
            # Gello 跟随机械臂
            self.expert.start_following(robot_joints)
            time.sleep(2.0)  # 等待同步完成
            self.expert.stop_following()  # 恢复遥控模式
        
        return obs, info
```

---

## 🔧 使用方法

### 方法 1: 在配置文件中启用

```python
# examples/experiments/task1_pick_banana/config.py

from franka_env.envs.wrappers import GelloIntervention

class TrainConfig:
    teleoperation_device = "gello"  # 或 "spacemouse"
    gello_port = "/dev/ttyUSB0"
    
    def get_environment(self, fake_env=False):
        env = PickBananaEnv()
        
        if not fake_env:
            if self.teleoperation_device == "gello":
                env = GelloIntervention(env, port=self.gello_port)
            elif self.teleoperation_device == "spacemouse":
                env = SpacemouseIntervention(env)
        
        # 其他 wrappers...
        env = RelativeFrame(env)
        env = SERLObsWrapper(env)
        return env
```

### 方法 2: 直接在代码中使用

```python
from franka_env.envs.wrappers import GelloIntervention

env = YourEnv()
env = GelloIntervention(
    env, 
    port="/dev/ttyUSB0",
    intervention_threshold=0.01,  # 移动灵敏度（弧度）
    sync_on_reset=True,           # Reset时同步Gello
    reset_follow_duration=2.0     # 同步持续时间（秒）
)

# 正常使用
obs, _ = env.reset()  # Gello会自动跟随机械臂reset
action = np.zeros(7)  # 或策略动作
obs, rew, done, _, info = env.step(action)
```

---

## 📝 录制演示数据

使用 `record_demos_octo_manual.py` 时，Gello 介入会自动被记录：

```bash
# 使用 Gello 录制
python examples/record_demos_octo_manual.py \
    --exp_name task1_pick_banana \
    --successes_needed 20

# 配置文件中设置
# teleoperation_device = "gello"
```

记录的数据会自动包含 Gello 的介入动作，与 SpaceMouse 录制的数据格式完全一致。

---

## 🔄 数据流对比

### SpaceMouse 数据流
```
SpaceMouse 硬件
    ↓ (读取 6DOF)
SpaceMouseExpert
    ↓ (task-space action)
SpacemouseIntervention
    ↓ (检测输入)
info["intervene_action"]
```

### Gello 数据流
```
Gello 硬件
    ↓ (读取关节角度)
GelloExpert
    ↓ (joint-space action)
GelloIntervention
    ↓ (检测移动)
info["intervene_action"]
```

**关键区别**：
- SpaceMouse: 任务空间控制 (x, y, z, roll, pitch, yaw)
- Gello: 关节空间控制 (q1, q2, ..., q7)

---

## ⚙️ 配置参数

### GelloIntervention 参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `port` | str | 必填 | Gello 设备串口 |
| `action_indices` | list | None | 限制控制的关节（如 [0,1,2] 只控制前3个关节）|
| `intervention_threshold` | float | 0.01 | 移动阈值（弧度），低于此值不触发介入 |
| `sync_on_reset` | bool | True | 是否在 reset 时同步 Gello 到机械臂位置 |
| `reset_follow_duration` | float | 2.0 | Reset 同步持续时间（秒） |

### 常用端口

```python
# xArm Gello
"/dev/serial/by-id/usb-FTDI_USB__-__Serial_Converter_FT3M9NVB-if00-port0"

# YAM Gello
"/dev/serial/by-id/usb-FTDI_USB__-__Serial_Converter_FTA2U4GA-if00-port0"

# UR Gello
"/dev/serial/by-id/usb-FTDI_USB__-__Serial_Converter_FTA7NNNU-if00-port0"
```

---

## 🎯 迁移步骤

### 1. 更新配置文件

```python
# 之前
from franka_env.envs.wrappers import SpacemouseIntervention

# 之后
from franka_env.envs.wrappers import (
    SpacemouseIntervention,
    GelloIntervention,  # 新增
)
```

### 2. 修改环境创建逻辑

```python
# 之前
def get_environment(self):
    env = BaseEnv()
    env = SpacemouseIntervention(env)
    return env

# 之后
def get_environment(self, teleoperation_device="gello"):
    env = BaseEnv()
    
    if teleoperation_device == "gello":
        env = GelloIntervention(env, port=self.gello_port)
    elif teleoperation_device == "spacemouse":
        env = SpacemouseIntervention(env)
    
    return env
```

### 3. 删除独立控制脚本

```bash
# 可以删除或归档
# examples/bidirectional_teleoperation.py （如果不再需要）
```

---

## 🧪 测试示例

### 基础测试

```python
# examples/gello_example_config.py
python examples/gello_example_config.py mixed
```

### 录制演示

```python
python examples/gello_example_config.py record
```

### 与 SpaceMouse 对比

```python
# 使用 Gello
config.teleoperation_device = "gello"
env = config.get_environment()

# 使用 SpaceMouse
config.teleoperation_device = "spacemouse"
env = config.get_environment()

# 两者使用方式完全一致！
```

---

## 🎨 高级用法

### 1. 部分关节控制

```python
# 只控制前 3 个关节（位置）
env = GelloIntervention(
    env, 
    port="/dev/ttyUSB0",
    action_indices=[0, 1, 2]
)
```

### 2. 调整灵敏度

```python
# 更高灵敏度（更容易触发介入）
env = GelloIntervention(env, intervention_threshold=0.005)

# 更低灵敏度（需要更大移动才介入）
env = GelloIntervention(env, intervention_threshold=0.05)
```

### 3. 组合多个 Wrapper

```python
env = BaseEnv()
env = GelloIntervention(env)        # 遥控
env = RelativeFrame(env)             # 坐标系转换
env = Quat2EulerWrapper(env)         # 姿态表示
env = SERLObsWrapper(env)            # 观察处理
env = ChunkingWrapper(env)           # 动作分块
```

---

## 🐛 常见问题

### Q1: Gello 不响应怎么办？

```python
# 检查端口
ls /dev/serial/by-id/

# 检查权限
sudo chmod 666 /dev/ttyUSB0

# 测试连接
python -c "from franka_env.gello.gello_expert import GelloExpert; \
           expert = GelloExpert('/dev/ttyUSB0'); \
           print(expert.get_action())"
```

### Q2: 如何切换设备？

```python
# 在配置文件中
class Config:
    teleoperation_device = "gello"  # 改为 "spacemouse" 或 None
```

### Q3: 介入阈值如何设置？

```python
# 敏感任务（精细控制）
intervention_threshold = 0.005  # 0.3 度

# 普通任务
intervention_threshold = 0.01   # 0.6 度

# 粗糙任务（避免误触）
intervention_threshold = 0.05   # 3 度
```

---

## 📚 相关文件

```
serl_robot_infra/franka_env/
├── gello/
│   ├── __init__.py
│   └── gello_expert.py          # Gello 硬件接口
├── envs/
│   └── wrappers.py               # GelloIntervention 类
└── spacemouse/
    └── spacemouse_expert.py      # SpaceMouse 接口（对比参考）

examples/
├── gello_example_config.py       # Gello 使用示例
├── record_demos_octo_manual.py   # 录制脚本（支持 Gello）
└── experiments/
    └── task1_pick_banana/
        └── config.py             # 更新后的配置
```

---

## ✅ 迁移检查清单

- [ ] 创建 `GelloExpert` 类
- [ ] 创建 `GelloIntervention` wrapper
- [ ] 更新配置文件导入
- [ ] 修改环境创建逻辑
- [ ] 测试 Gello 连接
- [ ] 录制测试演示数据
- [ ] 验证数据格式与 SpaceMouse 一致
- [ ] 更新文档

---

## 🎉 总结

通过 Wrapper 模式集成 Gello：

1. **统一性**: 与 SpaceMouse 相同的集成方式
2. **模块化**: 轻松添加/移除
3. **可组合**: 与其他 wrapper 自由组合
4. **透明性**: 对主循环代码无侵入
5. **可维护**: 代码结构清晰

现在 Gello 和 SpaceMouse 在系统中的地位完全平等，可以随时切换使用！
