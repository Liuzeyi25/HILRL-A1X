# A1_X 训练任务创建完成

## 📁 已创建文件

### 核心配置和代码
```
examples/experiments/a1x_pick_banana/
├── config.py                       # 环境和训练配置
├── wrapper.py                      # 自定义环境包装器
├── run_learner_conrft.sh          # Learner 启动脚本
├── run_learner_conrft_pretrain.sh # 预训练脚本
├── run_actor_conrft.sh            # Actor 启动脚本
├── demo_data/                     # 演示数据存储目录
├── conrft/                        # 训练检查点目录
└── classifier_ckpt/               # 奖励分类器目录
```

### 文档
```
├── README.md          # 完整使用文档
├── QUICKSTART.md      # 快速入门指南
└── MIGRATION_GUIDE.md # Franka 迁移指南
```

## 🎯 快速开始

### 1. 修改配置 (必需)

编辑 `config.py`:

```python
# Line 28-47: 修改相机序列号
REALSENSE_CAMERAS = {
    "wrist_1": {
        "serial_number": "YOUR_SERIAL_HERE",  # <-- 改这里
        ...
    },
}

# Line 64: 设置目标关节位置
TARGET_JOINT_STATE = np.array([0.0, -0.5, 0.0, -1.5, 0.0, 1.0, 20.0])

# Line 67: 设置重置位置
RESET_JOINT_STATE = np.array([0.0, -0.2, 0.0, -1.0, 0.0, 0.5, 100.0])
```

### 2. 收集演示

```bash
cd /home/dungeon_master/conrft/examples

python record_demos_octo.py \
    --exp_name a1x_pick_banana \
    --demo_num 30 \
    --save_path ./experiments/a1x_pick_banana/demo_data/
```

### 3. 启动训练

```bash
cd /home/dungeon_master/conrft/examples/experiments/a1x_pick_banana

# Terminal 1: Learner
bash run_learner_conrft.sh

# Terminal 2: Actor
bash run_actor_conrft.sh
```

## 📚 文档指南

根据你的需求选择文档:

- **新手?** → 阅读 `QUICKSTART.md`
- **详细配置?** → 阅读 `README.md`
- **从 Franka 迁移?** → 阅读 `MIGRATION_GUIDE.md`

## 🔧 配置说明

### 关键配置参数

| 参数 | 说明 | 示例值 |
|------|------|--------|
| `TARGET_JOINT_STATE` | 任务目标的关节角度 | `[0.0, -0.5, ..., 20.0]` |
| `RESET_JOINT_STATE` | 每集开始的关节角度 | `[0.0, -0.2, ..., 100.0]` |
| `ACTION_SCALE` | 动作步长 (弧度/步) | `[0.05, 0.05, ..., 10.0]` |
| `REWARD_THRESHOLD` | 成功判定阈值 | `[0.1, 0.1, ..., 10.0]` |

### 与 Franka 的主要区别

1. **控制空间**: 关节角度 (不是 xyz 坐标)
2. **观察键**: `joint_positions` (不是 `tcp_pose`)
3. **夹爪**: 0-100mm 连续值 (不是 ±1 二值)
4. **无需 Flask**: 直接 ROS2 + ZMQ 通信

## ✅ 使用前检查清单

- [ ] 修改 `config.py` 中的相机序列号
- [ ] 设置合适的 `TARGET_JOINT_STATE` (用 Gello 测试)
- [ ] 设置合适的 `RESET_JOINT_STATE`
- [ ] 确认 A1_X 机器人已连接 (`ros2 topic list`)
- [ ] 确认 ROS2 环境已 source (`source /opt/ros/humble/setup.zsh`)

## 🎓 学习路径

### 初级 (1-2天)
1. 阅读 `QUICKSTART.md`
2. 运行快速测试 (5分钟测试)
3. 收集 10 个演示
4. 运行预训练

### 中级 (3-5天)
1. 阅读 `README.md`
2. 调整超参数 (ACTION_SCALE, REWARD_THRESHOLD)
3. 自定义奖励函数
4. 训练奖励分类器

### 高级 (1-2周)
1. 阅读 `MIGRATION_GUIDE.md`
2. 修改 `wrapper.py` 实现自定义逻辑
3. 多任务训练
4. 集成正向运动学

## 🔍 故障排除

### 常见问题

**Q: ROS2 节点无法启动?**
```bash
source /opt/ros/humble/setup.zsh
ros2 topic list
```

**Q: 相机打开失败?**
```bash
rs-enumerate-devices  # 查看序列号
```

**Q: 关节角度超限?**
```python
# 检查当前关节角度是否在限制内
from franka_env.robots.a1x_robot import A1XRobot
r = A1XRobot()
print("Joints:", r.get_joint_state())
r.close()
```

**Q: 演示数据格式错误?**
```python
# 验证演示数据
import pickle
with open('demo_data/a1x_pick_banana_30_demos.pkl', 'rb') as f:
    demos = pickle.load(f)
print("Keys:", demos[0].keys())
```

## 📊 性能优化

### GPU 内存
```bash
export XLA_PYTHON_CLIENT_MEM_FRACTION=.3  # 降低到 30%
```

### 控制频率
```python
# 在 config.py 中
env = A1XEnv(hz=5)  # 降低到 5Hz
```

### 批量大小
```bash
# 在启动脚本中添加
--batch_size=128
```

## 🚀 下一步

1. **测试环境**: 运行 QUICKSTART.md 中的快速测试
2. **收集数据**: 至少 30 个高质量演示
3. **开始训练**: 预训练 → 在线训练
4. **迭代优化**: 根据训练结果调整参数

## 📝 重要提示

### ⚠️ 注意事项

1. **关节限制**: A1_X 的关节限制与 Gello 不同,已在代码中映射
2. **夹爪单位**: A1_X 夹爪是 0-100mm,不是 ±1
3. **控制频率**: 建议从 5-10Hz 开始,稳定后可提高
4. **演示质量**: 演示的平滑度和一致性直接影响训练效果

### 💡 最佳实践

1. **先用少量演示测试**: 5-10 个演示验证配置正确
2. **渐进式训练**: 预训练 → 在线微调
3. **保存检查点**: 定期备份 `conrft/` 目录
4. **监控训练**: 使用 Tensorboard 实时查看指标

## 📞 获取帮助

- 详细文档: `README.md`
- 快速入门: `QUICKSTART.md`
- 迁移指南: `MIGRATION_GUIDE.md`
- A1_X 集成: `/home/dungeon_master/conrft/serl_robot_infra/A1X_INTEGRATION.md`

## ✨ 总结

你现在拥有:

✅ 完整的 A1_X 训练配置  
✅ 自定义环境包装器  
✅ 启动脚本 (预训练/在线训练)  
✅ 详细文档 (3 个 markdown 文件)  
✅ 目录结构 (演示/检查点/分类器)  

**立即开始**: `cd /home/dungeon_master/conrft/examples/experiments/a1x_pick_banana && cat QUICKSTART.md`

祝训练顺利! 🎉
