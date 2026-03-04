# Gello末端位姿(EEF)动作空间实现

## 🎯 正确的需求理解

### 遥控时（执行）
- **Gello绝对关节位置** → 映射为**A1X绝对关节位置** → 发送给机器人
- 这部分保持不变，关节空间直接映射

### 保存demo时（记录）
- **不是**保存关节位置（无论delta还是absolute）
- **而是**保存：**Delta 末端位姿(x,y,z,rx,ry,rz) + 绝对Gripper值**
- Policy学习任务空间控制，而不是关节空间控制

## 📊 数据流对比

### 之前的错误理解（已废弃）

```
遥控: Gello关节 → A1X关节 → Delta关节 → 执行
保存: 保存A1X绝对关节位置  ❌ Policy学不到任务目标
```

### 正确的实现

```
遥控执行:
┌─────────────┐
│ Gello绝对关节│ [0.1, 1.2, 0.6, ...]
└──────┬──────┘
       │
       v
┌──────────────────────┐
│ 映射到A1X绝对关节     │ [0.057, 1.607, -0.498, ...]
└──────┬───────────────┘
       │
       v
┌──────────────────────┐
│ 转换为delta关节      │ [2.0, 4.0, 2.0, ...] (for env.step)
└──────┬───────────────┘
       │
       v
┌──────────────────────┐
│ 机器人执行           │
└──────────────────────┘

数据记录:
┌──────────────────────┐
│ 执行前: 读取EEF位姿  │ [x1, y1, z1, qx1, qy1, qz1, qw1]
└──────┬───────────────┘
       │
       v
┌──────────────────────┐
│ Gello遥控机器人移动  │
└──────┬───────────────┘
       │
       v
┌──────────────────────┐
│ 执行后: 读取EEF位姿  │ [x2, y2, z2, qx2, qy2, qz2, qw2]
└──────┬───────────────┘
       │
       v
┌──────────────────────────────┐
│ 计算Delta EEF位姿            │
│ delta_pos = [x2-x1, y2-y1, z2-z1]
│ delta_rot = curr_quat * prev_quat^{-1} → euler
└──────┬───────────────────────┘
       │
       v
┌──────────────────────────────┐
│ 添加绝对Gripper值            │ gripper_abs (from obs)
└──────┬───────────────────────┘
       │
       v
┌──────────────────────────────┐
│ 保存到Demo                   │
│ [dx, dy, dz, drx, dry, drz, gripper] ✅
└──────────────────────────────┘
```

## 🔧 实现细节

### 1. A1X Robot 增强

#### 添加了`get_eef_pose()`方法

```python
def get_eef_pose(self) -> tuple:
    """Get current end-effector pose.
    
    Returns:
        tuple: (position, quaternion) where:
            - position: np.ndarray [x, y, z] in meters
            - quaternion: np.ndarray [x, y, z, w]
    """
    state = self._bridge.get_joint_state()
    if state is None:
        return np.zeros(3), np.array([0, 0, 0, 1])
    
    ee_pos = np.array(state.get("ee_pos", [0, 0, 0]))
    ee_quat = np.array(state.get("ee_quat", [0, 0, 0, 1]))
    
    return ee_pos, ee_quat
```

#### 修改了`get_observations()`

不再返回全零的`ee_pos_quat`，而是从ROS2获取真实末端位姿：

```python
# Get end-effector pose from ROS2
ee_pos, ee_quat = self.get_eef_pose()
ee_pos_quat = np.concatenate([ee_pos, ee_quat])  # [x, y, z, qx, qy, qz, qw]
```

### 2. ROS2 Node 增强

修改了ZMQ的`get_state`响应，包含末端位姿：

```python
response = {
    "positions": self._current_joint_positions,
    "velocities": self._current_joint_velocities,
    "joint_names": self._joint_names,
    "ee_pos": self._current_pos.tolist(),      # ✅ 新增
    "ee_quat": self._current_rot.tolist(),     # ✅ 新增
}
```

ROS2 node已经在订阅`/hdas/pose_ee_arm`话题，现在通过ZMQ暴露出来。

### 3. GelloIntervention Wrapper 核心改动

#### 在`__init__`中添加EEF跟踪

```python
# Track previous EEF pose for delta calculation (for demo recording)
self.prev_eef_pose = None
```

#### 修改`step()`方法

```python
def step(self, action):
    # 1. 执行前：读取当前EEF位姿
    current_eef_pose = self._get_current_eef_pose()  # [x,y,z,qx,qy,qz,qw]
    
    new_action, replaced = self.action(action)
    
    # 2. 执行动作
    obs, rew, done, truncated, info = self.env.step(new_action)
    
    if replaced:
        # 3. 执行后：读取新的EEF位姿
        new_eef_pose = self._get_current_eef_pose()
        
        # 4. 计算delta EEF
        if current_eef_pose is not None and new_eef_pose is not None:
            delta_eef = self._compute_delta_eef(current_eef_pose, new_eef_pose)
            # delta_eef = [dx, dy, dz, drx, dry, drz]
            
            # 5. 添加绝对gripper值
            gripper_val = obs['state']['gripper_position'][0]
            delta_eef_action = np.concatenate([delta_eef, [gripper_val]])
            
            # 6. 保存到info
            info["intervene_action_eef"] = delta_eef_action
    
    return obs, rew, done, truncated, info
```

#### 新增辅助方法

**`_get_current_eef_pose()`**
```python
def _get_current_eef_pose(self) -> Optional[np.ndarray]:
    """从robot.get_eef_pose()获取当前末端位姿"""
    env = self.env
    while hasattr(env, 'env'):
        env = env.env
    
    if hasattr(env, 'robot') and hasattr(env.robot, 'get_eef_pose'):
        pos, quat = env.robot.get_eef_pose()
        return np.concatenate([pos, quat])
    
    return None
```

**`_compute_delta_eef()`**
```python
def _compute_delta_eef(self, prev_pose: np.ndarray, curr_pose: np.ndarray) -> np.ndarray:
    """计算两个位姿之间的delta"""
    from scipy.spatial.transform import Rotation as R
    
    # 位置delta (直接相减)
    delta_pos = curr_pose[:3] - prev_pose[:3]
    
    # 旋转delta (四元数相对旋转)
    prev_rot = R.from_quat(prev_pose[3:])
    curr_rot = R.from_quat(curr_pose[3:])
    delta_rot = curr_rot * prev_rot.inv()
    delta_euler = delta_rot.as_euler('xyz')
    
    return np.concatenate([delta_pos, delta_euler])
```

### 4. 数据采集脚本修改

```python
# record_demos_octo_manual.py
actions = np.zeros(env.action_space.sample().shape)
next_obs, rew, done, truncated, info = env.step(actions)

# 优先使用delta EEF动作
if "intervene_action_eef" in info:
    actions = info["intervene_action_eef"]  # ✅ [dx,dy,dz,drx,dry,drz,gripper_abs]
elif "intervene_action_eef" in info:
    actions = info["intervene_action_eef"]      # Fallback: delta joint

transition = dict(
    observations=obs,
    actions=actions,  # 保存的是delta EEF + absolute gripper!
    ...
)
```

## 📐 动作空间定义

### 执行空间（Environment）
- **类型**: Delta关节位置
- **维度**: 7-DOF (6 arm joints + 1 gripper)
- **范围**: 归一化 [-1, 1]，通过`action_scale`映射到实际关节增量

### 记录空间（Demo/Policy）
- **类型**: Delta末端位姿 + 绝对Gripper
- **维度**: 7-DOF (3 pos + 3 rot + 1 gripper)
- **格式**: 
  ```
  [dx, dy, dz, drx, dry, drz, gripper_abs]
  
  - dx, dy, dz: 位置增量 (meters)
  - drx, dry, drz: 旋转增量，欧拉角 (radians, XYZ顺序)
  - gripper_abs: 绝对gripper位置 (0-100mm或归一化值)
  ```

## ✅ 优势分析

### 为什么保存Delta EEF而不是Delta Joint？

1. **任务相关性**
   - EEF: "抓香蕉 = 末端移动到香蕉位置"（任务空间）
   - Joint: "关节2移动0.5弧度"（配置空间，与任务无关）

2. **泛化能力**
   - Delta EEF: 不同起始姿态都能到达相同目标位置
   - Delta Joint: 相同关节增量在不同姿态导致完全不同的末端位置

3. **可解释性**
   - EEF: "向前10cm，向下5cm"（人类理解）
   - Joint: "关节[2.0, -3.5, 1.2, ...]"（机器人配置）

4. **与Octo模型兼容**
   - Octo预训练模型期望的是任务空间动作
   - 大多数机器人数据集使用EEF表示

### 数值示例

```python
# 场景：抓取香蕉
# 起始姿态1: 机器人在左侧
delta_eef = [0.1, 0.0, -0.05, 0, 0, 0, 50mm]  # 向右10cm，向下5cm，gripper打开50mm
→ Policy学到: "向右移动接近香蕉"

# 起始姿态2: 机器人在右侧  
delta_eef = [-0.1, 0.0, -0.05, 0, 0, 0, 50mm]  # 向左10cm，向下5cm
→ Policy学到: "向左移动接近香蕉"

# 关键: 虽然delta不同，但都表达了"接近香蕉"这个任务目标
# 如果用delta joint，两个姿态的关节增量完全不同，policy无法泛化
```

## 🧪 验证方法

### 1. 检查保存的动作

```python
# 在record_demos_octo_manual.py中添加打印
if "intervene_action_eef" in info:
    actions = info["intervene_action_eef"]
    print(f"记录EEF动作: pos={actions[:3]}, rot={actions[3:6]}, gripper={actions[6]}")
```

**期望输出**：
```
记录EEF动作: pos=[0.012, -0.003, 0.008], rot=[0.001, -0.002, 0.0], gripper=0.45
```

### 2. 检查数值范围

- **位置delta**: 应该在厘米级别 (例如 0.01 ~ 0.05 meters)
- **旋转delta**: 应该在度级别 (例如 0.01 ~ 0.1 radians ≈ 0.5~6°)
- **Gripper**: 应该在0-100范围或0-1归一化范围

### 3. 对比关节vs EEF

采集同一个demo，打印两种表示：

```python
print(f"Delta Joint: {info['intervene_action']}")
print(f"Delta EEF:   {info['intervene_action_eef']}")
```

应该看到：
- Joint: 数值较大，依赖`action_scale`（例如 [-10, 5, -8, ...]）
- EEF: 数值较小，物理意义明确（例如 [0.01, -0.005, 0.02, ...]）

## 📝 修改文件清单

1. **`serl_robot_infra/franka_env/robots/a1x_ros2_node.py`**
   - 修改`get_state`响应，包含`ee_pos`和`ee_quat`

2. **`serl_robot_infra/franka_env/robots/a1x_robot.py`**
   - 添加`get_eef_pose()`方法
   - 修改`get_observations()`使用真实EEF位姿

3. **`serl_robot_infra/franka_env/envs/wrappers.py`** (GelloIntervention)
   - 添加`self.prev_eef_pose`跟踪
   - 修改`step()`计算delta EEF
   - 添加`_get_current_eef_pose()`辅助方法
   - 添加`_compute_delta_eef()`计算delta位姿

4. **`examples/record_demos_octo_manual.py`**
   - 优先使用`info["intervene_action_eef"]`而不是`intervene_action`

## ⚠️ 注意事项

### Gripper值的处理

- **记录的是绝对值**，不是delta
- 原因：Gripper通常是"打开/关闭"的二值状态，delta没有意义
- 从observation中读取当前gripper位置作为绝对值

### 四元数 vs 欧拉角

- **存储时**：使用四元数（从ROS2获取）
- **计算delta时**：先用四元数计算相对旋转，再转为欧拉角
- **原因**：欧拉角的delta更直观，且避免万向锁问题

### 坐标系

- 确保EEF位姿使用的是**世界坐标系**或**机器人基座标系**
- A1X的`/hdas/pose_ee_arm`话题应该提供的是基座标系下的位姿
- 如果有多个机器人，注意坐标系对齐

## 🎯 下一步

1. ✅ 代码实现完成
2. ⏭️ 运行遥控测试，验证EEF delta计算正确
3. ⏭️ 采集新的demonstration数据（使用EEF表示）
4. ⏭️ 训练policy，观察任务空间控制的泛化能力

---

**创建时间**: 2026-01-12  
**状态**: ✅ 实现完成，待测试
