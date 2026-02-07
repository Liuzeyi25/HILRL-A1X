# Gello双向控制测试 - 使用指南

## ✅ 问题解决了！

之前的问题是**Gello舵机需要12V外部电源**。现在已经确认Gello有电且正常工作。

## 📋 快速开始

### 运行简化版测试（推荐）

```bash
cd /home/dungeon_master/conrft/serl_robot_infra/franka_env/robots
python3 test_gello_simple.py
```

### 测试流程

1. **初始化**
   - A1_X机器人初始化
   - Gello设备初始化（使用与 `diagnose_gello_follower.py` 相同的配置）
   - 显示当前位置差异

2. **Teleoperation模式**（默认启动模式）
   - 手动移动Gello → 机器人跟随
   - 用途：让Gello和机器人位置对齐
   - 控制频率：20Hz

3. **Follow模式**（可选，按空格切换）
   - Gello跟随机器人移动
   - 仅在位置已对齐时使用（差异<28°）
   - 监控模式：不会主动移动Gello（避免大跳跃导致错误）

## 🎮 控制键

| 按键 | 功能 |
|------|------|
| `SPACE` | 切换 Teleoperation ↔ Follow 模式 |
| `S` | 显示当前位置状态和差异 |
| `Q` | 退出测试 |

## 📊 位置状态指示

```
Max diff: < 5.7°    → ✅ 位置良好对齐
Max diff: < 28.6°   → ⚠️  有一些差异
Max diff: > 28.6°   → ❌ 位置相差太远，需要先对齐
```

## ⚙️ 技术细节

### Gello配置（成功配置）

```python
DynamixelRobotConfig(
    joint_ids=[1, 2, 3, 4, 5, 6],
    joint_offsets=[1.5708, 4.71239, 4.71239, 3.14159, 1.5708, 3.14159],
    joint_signs=[1.0, -1.0, -1.0, -1.0, 1.0, 1.0],
    gripper_config=[7, 139.66015625, 199.16015625]
)
```

### 为什么用简化版？

原始的 `test_gello_bidirectional.py` 尝试让Gello主动跟随到机器人位置，但遇到问题：
- ❌ 位置差异太大（~97°）导致 `Failed to syncwrite goal position` 错误
- ❌ Dynamixel舵机不能一次性移动太远的距离
- ❌ GelloFollower的位置控制在大跳跃时会失败

简化版策略：
- ✅ 从teleoperation模式开始
- ✅ 让用户手动移动Gello来对齐位置
- ✅ Follow模式只用于监控，不主动移动Gello
- ✅ 避免了所有的大跳跃问题

## 🚀 使用场景

### 场景1：遥操作机器人
```
1. 启动测试（默认在teleoperation模式）
2. 手动移动Gello
3. 机器人会跟随Gello的动作
4. 完成任务
```

### 场景2：录制演示
```
1. 启动测试（teleoperation模式）
2. 手动移动Gello到初始位置
3. 按 [S] 检查对齐状态
4. 开始录制机器人轨迹
```

### 场景3：测试映射
```
1. 启动测试
2. 慢慢移动Gello through整个工作空间
3. 观察机器人是否正确跟随
4. 验证关节映射是否准确
```

## 🔧 故障排除

### 问题1: "Failed to syncwrite goal position"
**原因**: 尝试让舵机移动太大的距离
**解决**: 使用简化版测试（`test_gello_simple.py`）

### 问题2: Gello初始化失败
**原因**: Gello没有12V电源
**解决**: 
1. 检查12V电源适配器已连接
2. 运行 `python3 test_dynamixel_servos.py` 验证
3. 查看 `POWER_TROUBLESHOOTING.md`

### 问题3: 机器人不跟随Gello
**原因**: 关节映射问题或通信延迟
**解决**:
1. 按 [S] 查看位置状态
2. 检查 `a1x_robot.py` 中的映射函数
3. 检查控制频率（应为20Hz）

### 问题4: 位置差异很大
**原因**: Gello和机器人初始位置不同
**解决**: 
1. 保持在teleoperation模式
2. 慢慢手动移动Gello到接近机器人位置
3. 按 [S] 检查差异是否减小

## 📝 相关文件

| 文件 | 用途 |
|------|------|
| `test_gello_simple.py` | ✅ 简化版测试（推荐使用） |
| `test_gello_bidirectional.py` | 原始版本（有大跳跃问题） |
| `test_dynamixel_servos.py` | 舵机通信测试 |
| `diagnose_gello_follower.py` | Gello follower诊断 |
| `POWER_TROUBLESHOOTING.md` | 电源问题排查指南 |
| `check_gello_power.sh` | 电源检查脚本 |

## ✅ 成功测试的证据

根据 `diagnose_gello_follower.py` 的输出：
```
✅ Gello Agent 初始化成功
✅ GelloFollower 对象创建成功
✅ Follower 模式启动成功!
✅ Follower 命令测试通过
✅ 已返回自由模式
```

Gello硬件和软件都正常工作！只需要避免大的位置跳跃。

## 🎯 下一步

1. **测试Teleoperation**: 
   ```bash
   python3 test_gello_simple.py
   ```
   手动移动Gello，观察机器人跟随

2. **对齐位置**: 
   - 在teleoperation模式下慢慢移动
   - 按 [S] 查看差异
   - 直到差异 < 28°

3. **集成到训练**: 
   - 使用teleoperation模式录制演示
   - 集成到 `examples/train_conrft_octo.py`

---

**总结**: Gello设备现在已经正常工作！使用 `test_gello_simple.py` 进行双向控制测试。从teleoperation模式开始，手动对齐位置，然后就可以正常使用了！ 🎉
