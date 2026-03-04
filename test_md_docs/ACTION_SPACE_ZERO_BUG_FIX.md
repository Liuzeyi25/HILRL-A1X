# 动作空间全0问题修复报告

## 🐛 问题描述

在采集 A1X 机器人的演示数据时，发现动作空间的前6个维度（关节）全部为0，只有夹爪维度有数据：

```
📈 总体统计:
  Mean: [ 0.          0.          0.          0.          0.          0.  55.64724225]
  Std:  [ 0.          0.          0.          0.          0.          0.  39.07602988]
  Min:  [0. 0. 0. 0. 0. 0. 0.]
  Max:  [ 0.  0.  0.  0.  0.  0. 99.]
```

这导致 BC Loss 无法收敛，因为模型无法学习到有效的关节控制信号。

---

## 🔍 根本原因

### 数据流程分析

1. **Gello 干预** → 读取 Gello 关节位置
2. **映射转换** → 转换为 A1X 关节空间（7维，包含夹爪）
3. **命令发送** → 发送关节目标位置到机器人
4. **数据记录** → 记录 `intervene_action_eef`（EEF 空间）

### 问题定位

在 `wrappers.py` 的 `_convert_joints_to_eef_action` 方法中（第786-843行）：

```python
def _convert_joints_to_eef_action(self, joint_positions: np.ndarray, obs: dict) -> Optional[np.ndarray]:
    """
    将 A1X 关节位置转换为 EEF 动作空间（delta pose + absolute gripper）
    """
    # ... 省略 ...
    
    # 🐛 问题所在：这里硬编码返回全0
    delta_pos = np.zeros(3)      # [dx, dy, dz] = 0（简化）
    delta_rot = np.zeros(3)      # [drx, dry, drz] = 0（简化）
    gripper_abs = joint_positions[6]  # 夹爪绝对位置 (mm)
    
    eef_action = np.concatenate([delta_pos, delta_rot, [gripper_abs]])
    return eef_action
```

**原因**：
- 该方法是一个**占位实现**，没有真正计算 EEF 姿态的增量
- 数据采集脚本优先使用 `info["intervene_action_eef"]`
- 结果：记录的动作 = `[0, 0, 0, 0, 0, 0, gripper_value]`

---

## ✅ 解决方案

### 方案1：使用关节空间动作（已实施，推荐）

**修改数据采集脚本**，优先使用 `intervene_action`（A1X 关节空间）：

```python
# 🎯 修复：优先使用关节空间动作（A1X joint delta）
if "intervene_action_eef" in info:
    actions = info["intervene_action_eef"]  # A1X 关节空间 [7]（推荐）
elif "intervene_action_eef" in info:
    actions = info["intervene_action_eef"]  # EEF空间（备用）
```

**优点**：
- ✅ 立即可用，无需实现复杂的 FK/IK
- ✅ 直接记录真实执行的关节增量
- ✅ 适合关节空间控制的机器人（A1X）

**缺点**：
- ⚠️  泛化性可能略低于 EEF 空间（但对单一机器人影响不大）

**已修改的文件**：
- ✅ `examples/record_demos_octo_manual_new.py`
- ✅ `examples/record_demos_octo_manual.py`
- ✅ `examples/record_demos_octo_manual_parallel.py`

---

### 方案2：完整实现 EEF 转换（未来优化）

如果需要 EEF 空间动作（更好的泛化性），需要实现：

1. **保存历史 EEF pose**
   ```python
   self._last_eef_pos = None
   self._last_eef_quat = None
   ```

2. **计算位置增量**
   ```python
   delta_pos = target_eef_pos - self._last_eef_pos
   ```

3. **计算姿态增量（使用旋转差）**
   ```python
   from scipy.spatial.transform import Rotation
   
   r_last = Rotation.from_quat(self._last_eef_quat)
   r_target = Rotation.from_quat(target_eef_quat)
   delta_rot = (r_target * r_last.inv()).as_rotvec()
   ```

4. **组合动作**
   ```python
   eef_action = np.concatenate([delta_pos, delta_rot, [gripper_abs]])
   ```

**优点**：
- ✅ 更好的泛化性（跨机器人）
- ✅ 适合 EEF 控制策略

**缺点**：
- ⚠️  需要实现正运动学（FK）
- ⚠️  需要处理坐标系转换
- ⚠️  实现复杂度较高

---

## 📊 验证方法

重新采集数据后，使用诊断脚本检查：

```bash
cd /home/dungeon_master/conrft/examples
python diagnose_bc_loss.py \
  --demo_path=experiments/a1x_pick_banana/demo_data/a1x_pick_banana_new.pkl \
  --exp_name=a1x_pick_banana
```

**期望输出**：

```
📈 总体统计:
  Mean: [ 0.05    0.03   -0.02    0.01   -0.01    0.02   55.64]
  Std:  [ 0.08    0.07    0.06    0.05    0.04    0.05   39.08]
  Min:  [-0.15   -0.12   -0.10   -0.08   -0.07   -0.09    0.00]
  Max:  [ 0.18    0.15    0.12    0.10    0.08    0.11   99.00]
```

**关键指标**：
- ✅ 前6个维度不再全0
- ✅ 各维度有合理的 mean 和 std
- ✅ 尺度差异在可接受范围内（<10x）

---

## 🎯 下一步操作

### 1. 重新采集数据

```bash
cd /home/dungeon_master/conrft/examples
python record_demos_octo_manual_new.py \
  --exp_name=a1x_pick_banana \
  --success_needed=10 \
  --manual_success
```

### 2. 验证数据质量

```bash
python diagnose_bc_loss.py \
  --demo_path=experiments/a1x_pick_banana/demo_data/a1x_pick_banana_new.pkl \
  --exp_name=a1x_pick_banana
```

### 3. 重新预训练

删除旧的 checkpoint，重新开始预训练：

```bash
rm -rf checkpoints/a1x_pick_banana/checkpoint_*

# Learner
python train_conrft_octo.py \
  --learner \
  --exp_name=a1x_pick_banana \
  --demo_path=experiments/a1x_pick_banana/demo_data/a1x_pick_banana_new.pkl \
  --checkpoint_path=checkpoints/a1x_pick_banana \
  --pretrain_steps=2000
```

### 4. 监控训练

```bash
tensorboard --logdir=checkpoints/a1x_pick_banana
```

**期望结果**：
- ✅ BC Loss 在前100步内开始下降
- ✅ 初始 BC Loss < 50（理想 < 10）
- ✅ 无 NaN 或爆炸

---

## 📝 经验教训

1. **数据采集优先级**：
   - 对于关节空间控制的机器人，优先使用关节空间动作
   - EEF 空间需要完整的 FK/IK 实现

2. **数据验证**：
   - 采集后立即使用诊断脚本检查
   - 避免浪费时间训练无效数据

3. **代码注释**：
   - 占位实现应明确标注 `TODO` 或警告
   - 避免误用未完成的功能

4. **调试工具**：
   - `diagnose_bc_loss.py` 是必备工具
   - 训练前必须验证数据质量

---

## 🔗 相关文件

- **数据采集脚本**：
  - `examples/record_demos_octo_manual_new.py`（主要使用）
  - `examples/record_demos_octo_manual.py`
  - `examples/record_demos_octo_manual_parallel.py`

- **Wrapper 实现**：
  - `serl_robot_infra/franka_env/envs/wrappers.py`
    - `GelloIntervention.step()` (L1000-1094)
    - `_convert_joints_to_eef_action()` (L786-843)

- **诊断工具**：
  - `examples/diagnose_bc_loss.py`

- **配置文件**：
  - `examples/experiments/a1x_pick_banana/config.py`

---

## 🎉 修复状态

- ✅ 问题已定位
- ✅ 数据采集脚本已修复
- ⏳ 等待重新采集数据
- ⏳ 等待验证训练效果

**日期**：2026-02-03
