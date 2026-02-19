# A1X IK选择指南 🔧

## 📌 概述

A1X系统现在支持两种逆运动学(IK)求解器：

1. **RelaxedIK** (默认) - A1X自带的Cartesian控制器
2. **CuRobo IK** (可选) - NVIDIA GPU加速的IK求解器

---

## 🚀 快速开始

### 方案1: RelaxedIK (默认，推荐)

**无需修改任何配置，开箱即用**

```bash
# 直接运行训练/推理脚本
cd examples/experiments/a1x_pick_banana
bash run_learner_conrft_pretrain.sh
```

### 方案2: CuRobo IK (可选，GPU加速)

**修改配置文件启用CuRobo IK：**

```python
# 编辑 examples/experiments/a1x_pick_banana/config.py
class EnvConfig(DefaultA1XEnvConfig):
    # ...
    USE_CUROBO_IK = True  # 改为 True
    # ...
```

然后运行训练/推理：

```bash
cd examples/experiments/a1x_pick_banana
bash run_learner_conrft_pretrain.sh
```

---

## 🔍 详细说明

### RelaxedIK (默认方案)

**工作原理：**
```
策略输出 (6D EEF delta)
  ↓
计算目标EEF pose
  ↓
发布到 /motion_target/target_pose_arm
  ↓
RelaxedIK节点 (ROS2)
  ↓ 内部IK求解
发布到 /motion_target/target_joint_state_arm
  ↓
机械臂执行
```

**优点：**
- ✅ 已集成在A1X系统中
- ✅ 稳定可靠
- ✅ 无需额外配置
- ✅ 平滑轨迹
- ✅ 自动处理关节限制和奇异点

**适用场景：**
- 生产环境
- 需要稳定性
- 不关心IK求解速度

---

### CuRobo IK (可选方案)

**工作原理：**
```
策略输出 (6D EEF delta)
  ↓
计算目标EEF pose
  ↓
调用CuRobo IK (GPU加速)
  ↓
输出关节角度
  ↓
直接发布到 /motion_target/target_joint_state_arm
  ↓
机械臂执行
```

**优点：**
- ✅ GPU加速，更快 (~10ms)
- ✅ 支持FK验证
- ✅ 可自定义IK参数
- ✅ 适合高频控制

**缺点：**
- ⚠️ 需要CUDA/GPU
- ⚠️ 需要额外依赖 (CuRobo, torch)
- ⚠️ 需要URDF文件

**适用场景：**
- 研究/实验
- 需要高频控制 (>10Hz)
- 有GPU资源
- 需要自定义IK行为

---

## 📊 性能对比

| 特性 | RelaxedIK | CuRobo IK |
|------|-----------|-----------|
| 求解速度 | ~20-50ms | ~5-10ms |
| 稳定性 | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| 配置难度 | ⭐ (开箱即用) | ⭐⭐⭐ (需要配置) |
| GPU依赖 | ❌ 不需要 | ✅ 需要 |
| 轨迹平滑 | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ |

---

## ⚙️ 配置参数

### 在 config.py 中配置

```python
class EnvConfig(DefaultA1XEnvConfig):
    # IK选择 (True=CuRobo, False=RelaxedIK)
    USE_CUROBO_IK = False  # 默认False
```

### CuRobo IK 参数 (高级)

如果需要调整CuRobo IK参数，修改 `A1_x_controller.py`:

```python
class URDFInverseKinematics:
    def _init_ik_solver(self):
        self.ik_config = IKSolverConfig.load_from_robot_config(
            self.robot_cfg,
            None,
            rotation_threshold=0.02,     # 旋转容差 (rad)
            position_threshold=0.01,     # 位置容差 (m)
            num_seeds=1,                 # IK种子数
            self_collision_check=False,  # 自碰撞检测
            use_cuda_graph=True,         # GPU加速
        )
```

---

## 🐛 故障排除

### RelaxedIK不工作

**症状：** 机械臂不响应EEF命令

**检查：**
```bash
# 检查RelaxedIK节点是否运行
ros2 node list | grep relaxed_ik

# 检查话题订阅
ros2 topic info /motion_target/target_pose_arm
# 应该显示 Subscription count: 1
```

**解决：**
```bash
# 启动RelaxedIK节点 (通常在A1X启动脚本中)
ros2 run relaxed_ik relaxed_ik_node
```

---

### CuRobo IK不工作

**症状：** 报错 "CuRobo IK not available"

**检查依赖：**
```bash
python -c "import torch; print(torch.cuda.is_available())"  # 应该返回True
python -c "from curobo.wrap.reacher.ik_solver import IKSolver"  # 不应该报错
```

**解决：**
```bash
# 安装CuRobo (参考CuRobo官方文档)
pip install curobo-cuda11  # 根据你的CUDA版本选择
```

---

### IK求解失败

**症状：** 日志显示 "IK failed to converge"

**原因：**
- 目标pose超出机械臂工作空间
- 目标姿态存在奇异点
- IK容差设置过严格

**解决：**
1. **检查目标pose是否合理**
   ```python
   # 在代码中添加调试输出
   print(f"Target pos: {target_pos}")
   print(f"Target quat: {target_quat}")
   ```

2. **放宽IK容差** (CuRobo IK)
   ```python
   # 在 A1_x_controller.py 中
   position_threshold=0.02,  # 从0.01增加到0.02
   rotation_threshold=0.05,  # 从0.02增加到0.05
   ```

3. **使用RelaxedIK** (更鲁棒)
   ```python
   # 在 config.py 中
   USE_CUROBO_IK = False
   ```

---

## 📝 日志输出

### RelaxedIK模式

```bash
🎯 Using RelaxedIK (Cartesian control) for end-effector control
Published EEF command: pos=[0.3234, 0.1245, 0.2567], quat=[0.123, 0.456, 0.789, 0.234]
```

### CuRobo IK模式

```bash
🚀 Using CuRobo IK solver for end-effector control
✅ CuRobo IK solver initialized successfully
✅ CuRobo IK solved: pos=[0.3234, 0.1245, 0.2567]
[IK] ✓ Converged! Pos=3.2mm, Rot=0.015
```

---

## 🔄 切换方法总结

| 步骤 | RelaxedIK | CuRobo IK |
|------|-----------|-----------|
| 1 | `USE_CUROBO_IK = False` | `USE_CUROBO_IK = True` |
| 2 | 运行脚本 | 运行脚本 |
| 3 | ✅ 完成 | ✅ 完成 |

---

## 💡 推荐使用场景

### 使用 RelaxedIK (默认)
- ✅ 生产环境
- ✅ 演示/展示
- ✅ 新手用户
- ✅ 稳定性优先

### 使用 CuRobo IK (可选)
- ✅ 研究实验
- ✅ 高频控制 (>10Hz)
- ✅ 需要自定义IK
- ✅ 有GPU资源
- ✅ 性能优先

---

## 📚 参考资源

- **RelaxedIK**: https://github.com/uwgraphics/relaxed_ik_core
- **CuRobo**: https://github.com/NVlabs/curobo
- **A1X文档**: `/home/dungeon_master/A1_X/`

---

## ✅ 验证配置

运行以下命令验证当前使用的IK方案：

```bash
# 查看日志输出
tail -f /tmp/a1x_ros2_node.log | grep -E "RelaxedIK|CuRobo"

# 或直接运行训练脚本，观察启动日志
cd examples/experiments/a1x_pick_banana
bash run_learner_conrft_pretrain.sh | head -20
```

你应该看到：
- RelaxedIK: `🎯 Using RelaxedIK (Cartesian control)`
- CuRobo IK: `🚀 Using CuRobo IK solver`

---

**创建时间**: 2026-02-09
**版本**: 1.0
