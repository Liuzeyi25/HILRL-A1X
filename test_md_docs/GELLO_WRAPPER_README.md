# Gello 环境包装器集成

## 📌 概述

本次更新将 Gello 控制通过**环境包装器（Wrapper）模式**集成到系统中，实现与 SpaceMouse 完全一致的使用方式。

---

## 🎯 为什么要重构？

### 之前的问题 (bidirectional_teleoperation.py)

```python
# ❌ 独立脚本，与环境分离
teleop = BidirectionalTeleoperation()
teleop.start()
# 需要 200+ 行代码管理控制循环
```

**局限性**：
- 🔴 无法集成到训练/测试流程
- 🔴 不能录制演示数据
- 🔴 与 SpaceMouse 集成方式不一致
- 🔴 维护成本高

### 现在的方式 (GelloIntervention)

```python
# ✅ 一行代码完成集成
env = GelloIntervention(env, port="/dev/ttyUSB0")
```

**优势**：
- ✅ 与 SpaceMouse 完全一致
- ✅ 自动支持所有功能
- ✅ 代码量减少 98%
- ✅ 维护简单

---

## 📂 新增文件

```
serl_robot_infra/franka_env/
├── gello/
│   ├── __init__.py                    # 新增：Gello 模块
│   └── gello_expert.py                # 新增：Gello 硬件接口
└── envs/
    └── wrappers.py                     # 更新：新增 GelloIntervention 类

examples/
├── gello_example_config.py            # 新增：使用示例
├── gello_comparison.py                # 新增：新旧方式对比
└── test_gello_integration.py          # 新增：集成测试

docs/
└── gello_integration_guide.md         # 新增：详细文档
```

---

## 🚀 快速开始

### 1. 在配置文件中启用

```python
# examples/experiments/your_task/config.py

from franka_env.envs.wrappers import GelloIntervention

class YourConfig:
    gello_port = "/dev/serial/by-id/usb-FTDI_USB__-__Serial_Converter_FT3M9NVB-if00-port0"
    
    def get_environment(self):
        env = YourEnv()
        env = GelloIntervention(env, port=self.gello_port)
        # ... 其他 wrappers
        return env
```

### 2. 录制演示数据

```bash
# 使用 Gello 录制（自动支持！）
python examples/record_demos_octo_manual.py \
    --exp_name your_task \
    --successes_needed 20
```

### 3. 切换控制设备

```python
# 在配置中切换
teleoperation_device = "gello"      # 使用 Gello
teleoperation_device = "spacemouse"  # 使用 SpaceMouse  
teleoperation_device = None          # 仅策略控制
```

---

## 🔧 核心组件

### 1. GelloExpert (硬件接口层)

```python
from franka_env.gello.gello_expert import GelloExpert

expert = GelloExpert(port="/dev/ttyUSB0")

# 遥控模式：读取 Gello 动作
action, buttons = expert.get_action()
# action: (7,) 关节角度
# buttons: [left_button, right_button] 夹爪状态

# 跟随模式：让 Gello 跟随机械臂
expert.start_following(initial_position=robot_joints)
expert.command_follow(target_joints)
expert.stop_following()  # 恢复遥控模式
```

### 2. GelloIntervention (包装器层) - 支持双向控制

```python
from franka_env.envs.wrappers import GelloIntervention

env = YourEnv()
env = GelloIntervention(
    env,
    port="/dev/ttyUSB0",
    intervention_threshold=0.01,  # 移动灵敏度
    sync_on_reset=True,           # Reset时同步Gello ⭐
    reset_follow_duration=2.0     # 同步持续时间
)

# Reset: Robot → Gello (自动同步)
obs, _ = env.reset()  # Gello会自动跟随机械臂到reset位置

# Step: Gello → Robot (遥控)
obs, rew, done, _, info = env.step(action)
if "intervene_action" in info:
    print("Gello 正在控制机器人")
```

**双向控制流程**：
1. **Reset 阶段**: 机械臂移动到初始位置 → Gello 自动跟随 (2秒) → 恢复遥控模式
2. **Step 阶段**: 人类移动 Gello → 机械臂跟随执行
3. **下次 Reset**: 自动重复同步过程
| 录制演示 | ❌ | ✅ |
| 与 wrapper 组合 | ❌ | ✅ |
| 与 policy 混合 | ❌ | ✅ |
| 代码量 | ~200 行 | ~3 行 |
| 与 SpaceMouse 一致 | ❌ | ✅ |

---

## 🧪 测试

### 运行集成测试

```bash
python examples/test_gello_integration.py
```

测试内容：
- ✅ GelloExpert 初始化
- ✅ GelloIntervention wrapper
- ✅ 多 wrapper 组合
- ✅ 与 SpaceMouse 接口一致性

### 查看对比示例

```bash
python examples/gello_comparison.py
```

展示新旧方式的详细对比。

### 运行示例

```bash
# 录制演示（自动支持Reset同步）
python examples/gello_example_config.py record

# 测试混合控制
python examples/gello_example_config.py mixed

# 测试双向控制
python examples/test_gello_bidirectional.py
```

---

## 📚 文档

详细文档请查看：
- [Gello 集成指南](docs/gello_integration_guide.md)
- [使用示例](examples/gello_example_config.py)
- [对比分析](examples/gello_comparison.py)

---

## 🔄 迁移指南

### 步骤 1: 更新导入

```python
# 添加到配置文件
from franka_env.envs.wrappers import GelloIntervention
```

### 步骤 2: 修改环境创建

```python
# 之前
env = YourEnv()
# 无法添加 Gello

# 之后
env = YourEnv()
env = GelloIntervention(env, port="/dev/ttyUSB0")
```

### 步骤 3: 删除旧代码（可选）

```python
# bidirectional_teleoperation.py 可以归档或删除
```

---

## 🎨 高级用法

### 1. 部分关节控制

```python
# 只控制前 3 个关节
env = GelloIntervention(env, action_indices=[0, 1, 2])
```

### 2. 调整灵敏度

```python
# 更敏感（小动作就触发）
env = GelloIntervention(env, intervention_threshold=0.005)

# 更不敏感（需要大动作）
env = GelloIntervention(env, intervention_threshold=0.05)
```

### 3. 与多个 Wrapper 组合

```python
env = BaseEnv()
env = GelloIntervention(env)        # Gello 控制
env = RelativeFrame(env)             # 坐标转换
env = Quat2EulerWrapper(env)         # 姿态转换
env = SERLObsWrapper(env)            # 观察处理
env = ChunkingWrapper(env)           # 动作分块
```

---

## 🐛 故障排除

### 问题 1: 找不到 Gello 设备

```bash
# 检查设备
ls /dev/serial/by-id/

# 检查权限
sudo chmod 666 /dev/ttyUSB0
```

### 问题 2: 导入错误

```bash
# 确保 Gello 软件已安装
cd Gello/gello_software
pip install -e .
```

### 问题 3: Gello 不响应

```python
# 测试连接
from franka_env.gello.gello_expert import GelloExpert
expert = GelloExpert("/dev/ttyUSB0")
print(expert.get_action())
```

---

## ✅ 验证清单

在使用前确认：

- [ ] Gello 硬件已连接
- [ ] `franka_env.gello` 模块可导入
- [ ] `GelloIntervention` 可从 `wrappers` 导入
- [ ] 配置文件已更新
- [ ] 测试脚本运行成功
- [ ] 能够录制演示数据

---

## 🎉 总结

通过 Wrapper 模式集成 Gello：

1. **统一性**: 与 SpaceMouse 完全一致的使用方式
2. **简洁性**: 代码量从 200 行减少到 3 行
3. **灵活性**: 可与任意 wrapper 组合
4. **透明性**: 对现有代码无侵入
5. **可维护**: 结构清晰，易于维护

**现在 Gello 和 SpaceMouse 在系统中的地位完全平等！** 🎊

---

## 📞 联系

如有问题或建议，请：
- 查看详细文档: `docs/gello_integration_guide.md`
- 运行测试脚本: `examples/test_gello_integration.py`
- 查看示例代码: `examples/gello_example_config.py`
