# Gello 双向控制使用指南

## ✨ 新功能：Reset 时自动同步

在环境 `reset()` 时，Gello 会自动跟随机械臂移动到初始位置，确保人机同步。

---

## 🎯 工作流程

```
Episode 开始
    ↓
1. env.reset()
    ├─ 机械臂移动到 reset 位置
    ├─ Gello 自动跟随 (2秒) ← 🆕 双向控制
    └─ Gello 恢复遥控模式
    ↓
2. env.step()
    ├─ 人类移动 Gello
    ├─ 机械臂跟随执行 ← 原有功能
    └─ 记录演示数据
    ↓
3. Episode 结束 → 重复步骤1
```

---

## 📝 使用方法

### 默认配置（推荐）

```python
env = GelloIntervention(
    env,
    port="/dev/ttyUSB0",
    sync_on_reset=True  # 默认开启
)

# Reset 时 Gello 自动同步
obs, _ = env.reset()  # ← Gello 会跟随机械臂
```

### 禁用 Reset 同步

```python
env = GelloIntervention(
    env,
    sync_on_reset=False  # 禁用同步
)

# Reset 时 Gello 保持不动
obs, _ = env.reset()  # ← Gello 位置不变
```

### 调整同步参数

```python
env = GelloIntervention(
    env,
    sync_on_reset=True,
    reset_follow_duration=3.0  # 延长同步时间到3秒
)
```

---

## 🎮 录制演示数据

现在录制演示时，每个 episode 开始前 Gello 会自动回到正确位置：

```bash
python examples/record_demos_octo_manual.py \
  --exp_name a1x_pick_banana \
  --successes_needed 10 \
  --manual_success=True
```

**好处**：
- ✅ 不需要手动调整 Gello 到起始位置
- ✅ 每个 episode 开始时人机自动对齐
- ✅ 提高演示数据质量和一致性

---

## ⚙️ 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `sync_on_reset` | `True` | 是否在 reset 时同步 Gello |
| `reset_follow_duration` | `2.0` | Gello 跟随机械臂的时间（秒）|
| `intervention_threshold` | `0.01` | 检测 Gello 移动的阈值（弧度）|

---

## 🔍 观察 Gello 同步

运行时会看到如下输出：

```
🔄 Syncing Gello to robot reset position...
   [进度显示: Gello 跟随中...]
✅ Gello synced. Ready for teleoperation.
```

同步完成后，你可以立即开始用 Gello 控制机械臂。

---

## 🐛 故障排除

### Gello 不跟随怎么办？

```bash
# 检查 Gello 是否正确初始化
# 应该看到这些消息：
# ✅ Gello Expert initialized
# ✅ Follower mode: Available for reset synchronization
```

### 同步时间太短？

```python
# 增加 reset_follow_duration
env = GelloIntervention(
    env,
    reset_follow_duration=5.0  # 延长到5秒
)
```

### 想禁用同步？

```python
# 设置 sync_on_reset=False
env = GelloIntervention(
    env,
    sync_on_reset=False
)
```

---

## 🧪 测试双向控制

```bash
# 运行双向控制测试
python examples/test_gello_bidirectional.py

# 测试内容：
# ✅ 基本遥控 (Gello → Robot)
# ✅ Reset同步 (Robot → Gello)
# ✅ 完整工作流
# ✅ 模式切换
```

---

## 💡 设计原理

参考 `/home/dungeon_master/conrft/examples/bidirectional_teleoperation.py` 中的双向遥控设计：

- **Normal mode**: `Gello → Robot` (人类遥控)
- **Reverse mode**: `Robot → Gello` (机械臂带动Gello)

在环境 Wrapper 中，我们在 `reset()` 时临时启用 Reverse mode，让 Gello 跟随机械臂到初始位置，然后自动切换回 Normal mode 供人类控制。

---

## ✅ 总结

通过双向控制：
1. **提高效率**: 不需要每次手动调整 Gello 位置
2. **提高质量**: 确保每个 episode 开始时人机对齐
3. **用户友好**: 全自动，无需额外操作
4. **向后兼容**: 可通过参数禁用

开始录制演示吧！🎉
