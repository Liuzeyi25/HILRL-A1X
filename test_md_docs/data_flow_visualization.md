# 🔄 A1X 数据采集完整流程

## 📊 数据流向图（从底层到最终保存）

```
┌─────────────────────────────────────────────────────────────────────┐
│  🤖 物理硬件层                                                        │
│  ├─ A1X 机器人 (关节位置、力矩传感器)                                │
│  ├─ RealSense 相机 (wrist_1, side_policy_256)                       │
│  └─ Gello 遥操作设备 (关节读数)                                      │
└─────────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────────┐
│  📦 Layer 1: A1XTaskEnv (wrapper.py)                                │
│  继承自: A1XEnv → FrankaEnv                                          │
│                                                                       │
│  输出数据:                                                            │
│  ├─ obs["state"]: shape (2, 7) - [左臂7维, 右臂7维]                 │
│  │   └─ [x, y, z, qx, qy, qz, gripper] 每个手臂                    │
│  ├─ obs["images"]: 字典 {"wrist_1": ..., "side_policy_256": ...}   │
│  ├─ reward: 基础奖励                                                 │
│  └─ info: {"succeed": False, ...}                                   │
└─────────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────────┐
│  🕹️ Layer 2: GelloIntervention (wrappers.py)                        │
│  功能: 读取 Gello 设备，添加干预信息                                │
│                                                                       │
│  修改/添加数据:                                                       │
│  ├─ info["intervene_action_eef"]: shape (7,) - 关节空间动作             │
│  │   └─ 从 Gello 读取并映射到 A1X 关节空间                          │
│  ├─ info["intervene_action_eef"]: shape (7,) - ⭐ 关键数据          │
│  │   └─ [dx, dy, dz, drx, dry, drz, dgripper_norm]                 │
│  │   └─ 通过 IK 差分转换得到                                        │
│  ├─ info["gello_intervened"]: bool - 是否有干预                     │
│  └─ info["threaded_mode"]: bool - 双线程控制模式                    │
└─────────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────────┐
│  🔍 Layer 3: SERLObsWrapper                                          │
│  功能: 标准化观测空间，扁平化 state                                  │
│                                                                       │
│  修改数据:                                                            │
│  ├─ obs["state"]: shape (14,) - 扁平化为一维数组                    │
│  │   └─ [左臂7维 + 右臂7维] 展平                                    │
│  └─ obs["images"]: 保持不变                                          │
└─────────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────────┐
│  📚 Layer 4: ChunkingWrapper                                         │
│  功能: 时序堆叠（obs_horizon 帧观测）                               │
│                                                                       │
│  修改数据:                                                            │
│  ├─ obs["state"]: shape (obs_horizon, 14) - 堆叠多帧                │
│  └─ obs["images"]: 堆叠多帧图像                                      │
└─────────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────────┐
│  🎯 Layer 5: MultiCameraBinaryRewardClassifierWrapper (wrappers.py) │
│  功能: 使用奖励分类器计算最终奖励                                    │
│                                                                       │
│  修改数据:                                                            │
│  ├─ reward: 覆盖为分类器输出 (二值奖励 0/1)                         │
│  ├─ done: 如果 reward > 0.5 则设为 True                            │
│  └─ info["succeed"]: 如果 reward > 0.5 则为 True                    │
└─────────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────────┐
│  ⚖️ Layer 6: A1XGripperPenaltyWrapper (wrapper.py)                  │
│  功能: 添加夹爪频繁开关的惩罚                                        │
│                                                                       │
│  修改/添加数据:                                                       │
│  ├─ reward: reward += gripper_penalty (penalty < 0)                │
│  ├─ info["gripper_penalty"]: 夹爪惩罚值 (例如 -0.2)                 │
│  └─ info["grasp_penalty"]: 同上（兼容性字段）                        │
└─────────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────────┐
│  💾 Layer 7: record_demos_octo_manual_new.py                        │
│  功能: 数据采集脚本，保存为 .pkl 文件                               │
│                                                                       │
│  构建 transition:                                                    │
│  {                                                                   │
│    "observations": obs,          # ← 来自 Layer 4 (ChunkingWrapper)│
│    "actions": actions,            # ← ⭐ 来自 Layer 2 的            │
│    │                               #    info["intervene_action_eef"]│
│    "next_observations": next_obs, # ← 来自 Layer 4                 │
│    "rewards": rew,                # ← 来自 Layer 5 + Layer 6       │
│    "masks": 1.0 - done,           # ← 来自 Layer 5                 │
│    "dones": done,                 # ← 来自 Layer 5                 │
│    "infos": info,                 # ← 包含所有层的信息              │
│  }                                                                   │
│                                                                       │
│  post-processing:                                                    │
│  ├─ add_mc_returns_to_trajectory() - 计算 MC returns               │
│  ├─ add_embeddings_to_trajectory() - 添加 Octo embeddings          │
│  └─ add_next_embeddings_to_trajectory() - 添加 next embeddings     │
└─────────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────────┐
│  📁 最终保存的 .pkl 文件                                             │
│  路径: conrft/examples/experiments/a1x_pick_banana/demo_data/       │
│                                                                       │
│  文件名: traj_001_manual_2026-02-06_19-39-14.pkl                    │
│                                                                       │
│  内容: List[Dict], 每个元素为一个 transition:                        │
│  [                                                                   │
│    {                                                                 │
│      "observations": {...},      # 观测数据                         │
│      "actions": [7,],             # ⭐ EEF 动作空间                 │
│      "next_observations": {...}, # 下一帧观测                       │
│      "rewards": float,            # 奖励 (含分类器+惩罚)            │
│      "masks": float,              # 1.0 - done                      │
│      "dones": bool,               # 是否结束                        │
│      "infos": {                   # 元信息                          │
│        "intervene_action_eef": [7,],      # 关节空间动作                │
│        "intervene_action_eef": [7,],   # EEF 空间动作 ⭐           │
│        "gello_intervened": bool,      # 是否干预                   │
│        "succeed": bool,                # 是否成功                   │
│        "data_valid": bool,             # 数据是否有效               │
│        "grasp_penalty": float,         # 夹爪惩罚                   │
│      },                                                              │
│      "mc_returns": float,         # MC return (后处理添加)          │
│      "embeddings": array,         # Octo embeddings (后处理添加)    │
│      "next_embeddings": array,    # Next embeddings (后处理添加)    │
│    },                                                                │
│    ...  # 重复 N 帧                                                 │
│  ]                                                                   │
└─────────────────────────────────────────────────────────────────────┘
```

## 🎯 关键数据来源总结

### 1. **observations** (`obs`)
- **最初来源**: Layer 1 (A1XTaskEnv) - 机器人硬件
- **经过处理**: 
  - Layer 3 (SERLObsWrapper) - 扁平化
  - Layer 4 (ChunkingWrapper) - 时序堆叠
- **最终形式**: 
  - `obs["state"]`: shape (obs_horizon, 14)
  - `obs["images"]`: 堆叠的多帧图像

### 2. **actions** (训练时使用的动作) ⭐ 最关键
- **来源**: Layer 2 (GelloIntervention)
- **字段**: `info["intervene_action_eef"]`
- **形式**: shape (7,) - `[dx, dy, dz, drx, dry, drz, dgripper]`
- **说明**: 
  - 这是**策略学习的目标**
  - 从 Gello 关节位置通过 IK 差分转换得到
  - 代表末端执行器的增量动作

### 3. **rewards**
- **基础来源**: Layer 5 (MultiCameraBinaryRewardClassifierWrapper)
  - 奖励分类器通过相机图像判断任务成功/失败
  - 二值奖励: 0 (失败) 或 1 (成功)
- **附加修改**: Layer 6 (A1XGripperPenaltyWrapper)
  - 添加夹爪惩罚 (例如 -0.2)
  - `final_reward = classifier_reward + gripper_penalty`

### 4. **infos** (元信息)
- **累积来源**: 所有层
- **关键字段**:
  - `intervene_action_eef`: Layer 2 添加 ⭐
  - `intervene_action`: Layer 2 添加
  - `gello_intervened`: Layer 2 添加
  - `succeed`: Layer 5 添加/修改
  - `grasp_penalty`: Layer 6 添加

## 📈 验证结果回顾

根据之前的分析 (analyze_traj.py):
- ✅ **Delta action 累加验证通过**
  - 位置误差: 5.26 mm (< 1cm)
  - 说明 EEF 动作转换准确
- ✅ **数据一致性良好**
  - 证明数据流各层协同工作正常

## 🔧 关键设计理由

1. **为什么使用 EEF 动作空间？**
   - 策略泛化能力更强（笛卡尔空间比关节空间更直观）
   - 与人类遥操作行为更一致
   - 便于迁移到不同臂长/构型的机器人

2. **为什么需要多层 wrapper？**
   - **模块化设计**: 每层负责特定功能
   - **可复用性**: GelloIntervention 可用于多个任务
   - **灵活组合**: 可根据需求启用/禁用特定层

3. **为什么 GripperPenalty 在最外层？**
   - 在所有其他 wrapper 之后添加惩罚
   - 不影响底层环境的奖励计算逻辑
   - 便于调试和禁用
