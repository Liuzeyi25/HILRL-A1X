import pickle
import numpy as np
from scipy.spatial.transform import Rotation as R

# 读取文件
pkl_path = "/home/luka/Haoyuan/Safevla_RL/examples/experiments/a1x_pick_banana/demo_data/traj_001_manual_2026-02-05_19-45-31.pkl"
with open(pkl_path, 'rb') as f:
    data = pickle.load(f)

print("=" * 70)
print("� Z轴最小值分析及 Delta Action 累加验证")
print("=" * 70)

if not isinstance(data, list) or len(data) == 0:
    print("❌ 数据格式错误或为空")
    exit(1)

print(f"✓ 数据包含 {len(data)} 帧")

# 提取所有帧的 state 和 delta action
z_values = []
states = []
delta_actions = []

for i, frame in enumerate(data):
    if 'observations' in frame and 'state' in frame['observations']:
        state = frame['observations']['state']
        states.append(state)
        # state shape: (2, 7) - 第一个手臂的 z 轴是 state[0, 2]
        z_values.append(state[0, 2])
    else:
        z_values.append(None)
        states.append(None)
    
    if 'infos' in frame and 'intervene_action_eef' in frame['infos']:
        delta_actions.append(frame['infos']['intervene_action_eef'])
    else:
        delta_actions.append(None)

# 找到 z 轴最小值所在帧
valid_z_values = [(i, z) for i, z in enumerate(z_values) if z is not None]
if not valid_z_values:
    print("❌ 未找到有效的 z 轴数据")
    exit(1)

min_frame, min_z = min(valid_z_values, key=lambda x: x[1])

print(f"\n🎯 Z轴最小值信息:")
print(f"   - 最小值所在帧: 第 {min_frame} 帧")
print(f"   - Z轴最小值: {min_z:.6f}")
print(f"   - 第0帧的Z值: {z_values[0]:.6f}")
print(f"   - Z轴下降量: {z_values[0] - min_z:.6f}")

# 累加从第0帧到最小值帧的 delta action
print(f"\n🔄 累加 Delta Action (第0帧到第{min_frame}帧):")

# 位置和夹爪可以线性累加
cumulative_pos = np.zeros(3)  # [dx, dy, dz]
cumulative_gripper = 0.0

# 姿态使用四元数累乘（正确方法）
cumulative_rotation = R.identity()

for i in range(min_frame + 1):
    if delta_actions[i] is not None:
        # 累加位置
        cumulative_pos += delta_actions[i][:3]
        
        # 累乘旋转（四元数）
        delta_rot_euler = delta_actions[i][3:6]
        delta_rotation = R.from_euler('xyz', delta_rot_euler)
        cumulative_rotation = delta_rotation * cumulative_rotation
        
        # 累加夹爪
        cumulative_gripper += delta_actions[i][6]

# 将累积的旋转转换回欧拉角
cumulative_euler = cumulative_rotation.as_euler('xyz')

print(f"   累加的 delta action (位置 - 线性累加):")
print(f"   dx  = {cumulative_pos[0]:.6f}")
print(f"   dy  = {cumulative_pos[1]:.6f}")
print(f"   dz  = {cumulative_pos[2]:.6f}")
print(f"\n   累加的 delta action (姿态 - 四元数累乘转欧拉角):")
print(f"   dqx = {cumulative_euler[0]:.6f}")
print(f"   dqy = {cumulative_euler[1]:.6f}")
print(f"   dqz = {cumulative_euler[2]:.6f}")
print(f"\n   累加的 delta action (夹爪):")
print(f"   gripper = {cumulative_gripper:.6f}")

# 计算实际的状态变化
if states[0] is not None and states[min_frame] is not None:
    state_0 = states[0][0]  # 第一个手臂的状态
    state_min = states[min_frame][0]
    
    # 位置变化：可以直接相减
    actual_pos_change = state_min[:3] - state_0[:3]
    
    # 姿态变化：需要用旋转矩阵计算相对旋转
    # R_relative = R_min * R_0^(-1)
    R_0 = R.from_euler('xyz', state_0[3:6])
    R_min = R.from_euler('xyz', state_min[3:6])
    R_relative = R_min * R_0.inv()
    actual_rot_change_euler = R_relative.as_euler('xyz')
    
    # 夹爪变化：可以直接相减
    actual_gripper_change = state_min[6] - state_0[6]
    
    print(f"\n📈 实际状态变化 (第{min_frame}帧相对第0帧):")
    print(f"   位置变化 (线性):")
    print(f"     Δx  = {actual_pos_change[0]:.6f}")
    print(f"     Δy  = {actual_pos_change[1]:.6f}")
    print(f"     Δz  = {actual_pos_change[2]:.6f}")
    print(f"   姿态变化 (旋转合成后转欧拉角):")
    print(f"     Δqx = {actual_rot_change_euler[0]:.6f}")
    print(f"     Δqy = {actual_rot_change_euler[1]:.6f}")
    print(f"     Δqz = {actual_rot_change_euler[2]:.6f}")
    print(f"   夹爪变化:")
    print(f"     Δgripper = {actual_gripper_change:.6f}")
    
    # 计算位置差异
    pos_difference = actual_pos_change - cumulative_pos
    
    # 计算姿态差异（比较两个旋转矩阵的差异）
    # 方法1：比较欧拉角（可能有万向锁问题）
    euler_difference = actual_rot_change_euler - cumulative_euler
    
    # 方法2：计算旋转误差（更准确）
    # R_error = R_actual * R_predicted^(-1)
    R_error = R_relative * cumulative_rotation.inv()
    rotation_error_angle = R_error.magnitude()  # 旋转轴角表示的角度
    
    # 夹爪差异
    gripper_difference = actual_gripper_change - cumulative_gripper
    
    print(f"\n⚖️  差异分析 (实际变化 - 累加delta action):")
    print(f"   位置差异:")
    print(f"     Δx误差:  {pos_difference[0]:.6f} m")
    print(f"     Δy误差:  {pos_difference[1]:.6f} m")
    print(f"     Δz误差:  {pos_difference[2]:.6f} m")
    print(f"     位置误差范数: {np.linalg.norm(pos_difference):.6f} m")
    print(f"\n   姿态差异 (欧拉角对比 - 可能不准确):")
    print(f"     Δqx误差: {euler_difference[0]:.6f} rad = {np.degrees(euler_difference[0]):.3f}°")
    print(f"     Δqy误差: {euler_difference[1]:.6f} rad = {np.degrees(euler_difference[1]):.3f}°")
    print(f"     Δqz误差: {euler_difference[2]:.6f} rad = {np.degrees(euler_difference[2]):.3f}°")
    print(f"\n   姿态差异 (旋转轴角误差 - 准确方法):")
    print(f"     旋转误差角度: {rotation_error_angle:.6f} rad = {np.degrees(rotation_error_angle):.3f}°")
    print(f"\n   夹爪差异:")
    print(f"     Δgripper误差: {gripper_difference:.6f}")
    
    # 判断一致性
    pos_error = np.linalg.norm(pos_difference)
    rot_error_deg = np.degrees(rotation_error_angle)
    
    print(f"\n📊 误差总结:")
    print(f"   位置误差: {pos_error*1000:.3f} mm")
    print(f"   姿态误差: {rotation_error_angle:.6f} rad = {rot_error_deg:.3f}°")
    
    if pos_error < 0.001:  # 1mm 阈值
        pos_status = "✅ 高度一致 (< 1mm)"
    elif pos_error < 0.01:  # 1cm 阈值
        pos_status = "⚠️  基本一致 (< 1cm)"
    else:
        pos_status = "❌ 存在显著差异 (>= 1cm)"
    
    if rot_error_deg < 1.0:  # 1度阈值
        rot_status = "✅ 高度一致 (< 1°)"
    elif rot_error_deg < 5.0:  # 5度阈值
        rot_status = "⚠️  基本一致 (< 5°)"
    else:
        rot_status = "❌ 存在显著差异 (>= 5°)"
    
    print(f"\n🎯 结论:")
    print(f"   位置: {pos_status}")
    print(f"   姿态: {rot_status}")

print("\n" + "=" * 70)
