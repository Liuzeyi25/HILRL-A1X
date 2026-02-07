#!/usr/bin/env python3
"""
诊断 BC Loss 不收敛问题的脚本
用于检查 demo 数据的统计信息和训练配置

用法:
python /home/dungeon_master/conrft/examples/diagnose_bc_loss.py \
  --demo_path=/home/dungeon_master/conrft/examples/experiments/a1x_pick_banana/demo_data/20260207/traj_010_2026-02-07_16-51-07.pkl \
  --exp_name=a1x_pick_banana \
  --detailed  # 可选: 显示详细的递归结构分析
"""

import pickle as pkl
import numpy as np
import jax.numpy as jnp
from pathlib import Path
import sys
from absl import app, flags

FLAGS = flags.FLAGS
flags.DEFINE_multi_string("demo_path", None, "Path to the demo data.")
flags.DEFINE_string("exp_name", None, "Name of experiment.")
flags.DEFINE_bool("detailed", False, "Show detailed recursive structure analysis.")

def print_section(title):
    """打印分隔线和标题"""
    print("\n" + "=" * 70)
    print(f"📊 {title}")
    print("=" * 70)


def analyze_value(value, name, indent=0, max_depth=5):
    """分析单个值"""
    prefix = "  " * indent
    
    if isinstance(value, dict):
        analyze_dict_recursive(value, name, indent, max_depth)
    elif isinstance(value, np.ndarray):
        print(f"{prefix}🔢 numpy.ndarray")
        print(f"{prefix}   shape: {value.shape}")
        print(f"{prefix}   dtype: {value.dtype}")
        print(f"{prefix}   范围: [{value.min():.4f}, {value.max():.4f}]")
        if value.size <= 10:
            print(f"{prefix}   值: {value}")
    elif isinstance(value, list):
        print(f"{prefix}📋 list - {len(value)} 个元素")
        if len(value) > 0 and not isinstance(
            value[0], (dict, list, np.ndarray)
        ):
            print(f"{prefix}   示例: {value[:3]}...")
        elif len(value) > 0:
            print(f"{prefix}   首元素类型: {type(value[0])}")
    elif isinstance(value, (int, float, bool, np.integer, np.floating)):
        print(f"{prefix}🔢 {type(value).__name__}: {value}")
    elif isinstance(value, str):
        print(f"{prefix}📝 str: '{value}'")
    elif value is None:
        print(f"{prefix}❌ None")
    else:
        print(f"{prefix}❓ {type(value).__name__}: {value}")


def analyze_dict_recursive(obj, name="root", indent=0, max_depth=5):
    """递归分析字典结构"""
    prefix = "  " * indent
    
    if indent > max_depth:
        print(f"{prefix}⚠️ 达到最大深度 {max_depth}")
        return
    
    if isinstance(obj, dict):
        print(f"{prefix}📦 {name} (dict) - {len(obj)} 个键")
        for key, value in obj.items():
            print(f"{prefix}  🔑 '{key}':")
            analyze_value(value, f"{name}['{key}']", indent + 2, max_depth)
    elif isinstance(obj, list):
        print(f"{prefix}📋 {name} (list) - {len(obj)} 个元素")
        if len(obj) > 0:
            print(f"{prefix}  示例元素 [0]:")
            analyze_value(obj[0], f"{name}[0]", indent + 2, max_depth)
            if len(obj) > 1:
                print(f"{prefix}  示例元素 [-1] (最后):")
                analyze_value(obj[-1], f"{name}[-1]", indent + 2, max_depth)
    else:
        analyze_value(obj, name, indent, max_depth)


def analyze_trajectory_structure(demo_paths):
    """完整分析轨迹文件结构（来自 analyze_all_keys.py）"""
    print_section("详细轨迹结构分析")
    
    for traj_file in demo_paths:
        print(f"\n{'='*80}")
        print(f"分析文件: {traj_file}")
        print(f"{'='*80}")
        
        with open(traj_file, "rb") as f:
            traj = pkl.load(f)
        
        print(f"\n📊 顶层结构:")
        print(f"   类型: {type(traj)}")
        print(f"   长度: {len(traj)} 帧")
        
        if not isinstance(traj, list) or len(traj) == 0:
            print("⚠️ 不是列表或为空")
            continue
        
        # 分析第一帧
        print("\n" + "="*80)
        print("🎯 第一帧 (索引 0) 详细结构")
        print("="*80)
        first_frame = traj[0]
        analyze_dict_recursive(first_frame, "frame[0]", indent=0, max_depth=6)
        
        # 分析最后一帧
        print("\n" + "="*80)
        print("🎯 最后一帧 (索引 -1) 详细结构")
        print("="*80)
        last_frame = traj[-1]
        analyze_dict_recursive(last_frame, "frame[-1]", indent=0, max_depth=6)
        
        # 统计分析
        print("\n" + "="*80)
        print("📈 键的完整统计分析")
        print("="*80)
        
        # 收集所有键
        all_keys = set()
        for frame in traj:
            if isinstance(frame, dict):
                all_keys.update(frame.keys())
        
        print(f"\n所有出现过的顶层键: {sorted(all_keys)}")
        
        # 检查键的一致性
        print(f"\n🔍 键的一致性检查:")
        for key in sorted(all_keys):
            count = sum(
                1 for frame in traj
                if isinstance(frame, dict) and key in frame
            )
            print(f"   '{key}': {count}/{len(traj)} 帧")
        
        # observations 键的统计
        print(f"\n👁️  Observations 键统计:")
        obs_keys = set()
        for frame in traj:
            if isinstance(frame, dict) and "observations" in frame:
                obs = frame["observations"]
                if isinstance(obs, dict):
                    obs_keys.update(obs.keys())
        
        print(f"   所有 observation 键: {sorted(obs_keys)}")
        
        # 检查每个 observation 键的形状一致性
        for obs_key in sorted(obs_keys):
            shapes = []
            for frame in traj:
                if isinstance(frame, dict) and "observations" in frame:
                    obs = frame["observations"]
                    if isinstance(obs, dict) and obs_key in obs:
                        val = obs[obs_key]
                        if hasattr(val, 'shape'):
                            shapes.append(val.shape)
            
            unique_shapes = set(shapes)
            if len(unique_shapes) == 1:
                shape_str = f"形状一致 {list(unique_shapes)[0]}"
                print(f"   ✅ '{obs_key}': {shape_str}")
            else:
                print(f"   ⚠️ '{obs_key}': 形状不一致 {unique_shapes}")
        
        # next_observations 键的统计
        print(f"\n👁️  Next Observations 键统计:")
        next_obs_keys = set()
        for frame in traj:
            if isinstance(frame, dict) and "next_observations" in frame:
                next_obs = frame["next_observations"]
                if isinstance(next_obs, dict):
                    next_obs_keys.update(next_obs.keys())
        
        print(f"   所有 next_observation 键: {sorted(next_obs_keys)}")
        
        # infos 键的统计
        print(f"\n📝 Infos 键统计:")
        info_keys = set()
        for frame in traj:
            if isinstance(frame, dict) and "infos" in frame:
                info = frame["infos"]
                if isinstance(info, dict):
                    info_keys.update(info.keys())
        
        print(f"   所有 info 键: {sorted(info_keys)}")
        
        # 分析每个 info 键
        for info_key in sorted(info_keys):
            values = []
            for frame in traj:
                if isinstance(frame, dict) and "infos" in frame:
                    info = frame["infos"]
                    if isinstance(info, dict) and info_key in info:
                        values.append(info[info_key])
            
            # 统计值的类型和分布
            value_types = set(type(v).__name__ for v in values)
            print(f"\n   🔑 '{info_key}':")
            print(f"      出现次数: {len(values)}/{len(traj)}")
            print(f"      值类型: {value_types}")
            
            # 如果是布尔值，统计分布
            if all(isinstance(v, (bool, np.bool_)) or v is None
                   for v in values):
                true_count = sum(1 for v in values if v is True)
                false_count = sum(1 for v in values if v is False)
                none_count = sum(1 for v in values if v is None)
                print(f"      True: {true_count}, "
                      f"False: {false_count}, None: {none_count}")
            
            # 如果是数值，显示范围
            elif all(isinstance(v, (int, float, np.number))
                     for v in values if v is not None):
                numeric_values = [v for v in values if v is not None]
                if numeric_values:
                    print(f"      范围: "
                          f"[{min(numeric_values)}, {max(numeric_values)}]")
            
            # 如果是数组，显示形状
            elif all(hasattr(v, 'shape') for v in values if v is not None):
                shapes = set(v.shape for v in values if v is not None)
                print(f"      形状: {shapes}")
        
        # actions 统计
        print(f"\n🎮 Actions 统计:")
        action_shapes = []
        for frame in traj:
            if isinstance(frame, dict) and "actions" in frame:
                action = frame["actions"]
                if hasattr(action, 'shape'):
                    action_shapes.append(action.shape)
        
        unique_shapes = set(action_shapes)
        print(f"   形状: {unique_shapes}")
        
        # 查看第一个和最后一个 action
        if len(traj) > 0 and "actions" in traj[0]:
            print(f"   第一个 action: {traj[0]['actions']}")
        if len(traj) > 0 and "actions" in traj[-1]:
            print(f"   最后一个 action: {traj[-1]['actions']}")
        
        # rewards 统计
        print(f"\n💰 Rewards 统计:")
        rewards = []
        for frame in traj:
            if isinstance(frame, dict) and "rewards" in frame:
                rewards.append(frame["rewards"])
        
        if rewards:
            print(f"   总数: {len(rewards)}")
            print(f"   唯一值: {set(rewards)}")
            print(f"   总和: {sum(rewards)}")
        
        # masks 和 dones 统计
        print(f"\n🎭 Masks 统计:")
        masks = []
        for frame in traj:
            if isinstance(frame, dict) and "masks" in frame:
                masks.append(frame["masks"])
        
        if masks:
            print(f"   唯一值: {set(masks)}")
            print(f"   1.0 的数量: {sum(1 for m in masks if m == 1.0)}")
            print(f"   0.0 的数量: {sum(1 for m in masks if m == 0.0)}")
        
        print(f"\n✅ Dones 统计:")
        dones = []
        for frame in traj:
            if isinstance(frame, dict) and "dones" in frame:
                dones.append(frame["dones"])
        
        if dones:
            print(f"   唯一值: {set(dones)}")
            print(f"   True 的数量: {sum(1 for d in dones if d)}")
            print(f"   False 的数量: {sum(1 for d in dones if not d)}")
        
        # 检查额外的顶层键 (embeddings, mc_returns 等)
        print(f"\n🔍 额外字段检查:")
        extra_keys = [
            "embeddings", "next_embeddings",
            "mc_returns", "language_embedding"
        ]
        for extra_key in extra_keys:
            count = sum(
                1 for frame in traj
                if isinstance(frame, dict) and extra_key in frame
            )
            if count > 0:
                # 查看第一个出现的值
                for frame in traj:
                    if isinstance(frame, dict) and extra_key in frame:
                        val = frame[extra_key]
                        if hasattr(val, 'shape'):
                            print(f"   ✅ '{extra_key}': {count}/{len(traj)} "
                                  f"帧, shape={val.shape}, dtype={val.dtype}")
                        else:
                            print(f"   ✅ '{extra_key}': {count}/{len(traj)} "
                                  f"帧, type={type(val)}, value={val}")
                        break
            else:
                print(f"   ❌ '{extra_key}': 不存在")

def analyze_demo_data(demo_paths):
    """分析 demo 数据的统计信息"""
    print_section("Demo 数据分析")
    
    all_transitions = []
    
    # 加载所有 demo 文件
    for path in demo_paths:
        print(f"\n📁 加载文件: {path}")
        with open(path, "rb") as f:
            transitions = pkl.load(f)
            all_transitions.extend(transitions)
            print(f"   - 加载了 {len(transitions)} 个 transitions")
    
    print(f"\n✅ 总共加载了 {len(all_transitions)} 个 transitions")
    
    if len(all_transitions) == 0:
        print("❌ 错误: 没有加载到任何数据!")
        return
    
    # 检查第一个 transition 的键
    print_section("Transition 数据结构")
    sample = all_transitions[0]
    print(f"可用的键: {list(sample.keys())}")
    print()
    
    # 检查必要的键
    required_keys = ['observations', 'actions', 'rewards', 'masks', 'dones']
    optional_keys = ['embeddings', 'next_embeddings', 'mc_returns', 'intervened', 'grasp_penalty']
    
    for key in required_keys:
        status = "✅" if key in sample else "❌"
        print(f"{status} {key}: {'存在' if key in sample else '缺失'}")
    
    print()
    for key in optional_keys:
        status = "✅" if key in sample else "⚠️"
        print(f"{status} {key}: {'存在' if key in sample else '缺失'}")
    
    # 🔍 分析 Observations 中的 state
    print_section("Observations 中的 State 分析")
    
    if 'observations' in sample and isinstance(sample['observations'], dict):
        obs_sample = sample['observations']
        print(f"Observations 包含的键: {list(obs_sample.keys())}")
        print()
        
        if 'state' in obs_sample:
            # 收集所有 state 数据
            states = []
            for t in all_transitions:
                if 'observations' in t and 'state' in t['observations']:
                    states.append(t['observations']['state'])
            
            states = np.array(states)
            print(f"📊 State 数据统计:")
            print(f"   形状: {states.shape}")
            print(f"   维度: {states.shape[-1]}")
            print()
            
            # 显示第一帧的 state
            print("🔍 第一帧 state:")
            first_state = states[0]
            print(f"   值: {first_state}")
            print()
            
            # 显示最后一帧的 state
            print("🔍 最后一帧 state:")
            last_state = states[-1]
            print(f"   值: {last_state}")
            print()
            
            # 总体统计
            print("📈 State 总体统计:")
            print(f"   Mean: {np.mean(states, axis=0)}")
            print(f"   Std:  {np.std(states, axis=0)}")
            print(f"   Min:  {np.min(states, axis=0)}")
            print(f"   Max:  {np.max(states, axis=0)}")
            print()
            
            # 每个维度的详细统计
            print("📊 State 各维度详细统计:")
            # A1X state 通常是: [joint1-7, gripper, eef_pos(3), eef_quat(4)]
            # 总共 7 + 1 + 3 + 4 = 15 维
            state_dim_names = [
                                  
                "EEF_X", "EEF_Y", "EEF_Z",
                "EEF_Quat_X", "EEF_Quat_Y", "EEF_Quat_Z", "Gripper"
            ]
            
            # for dim_idx in range(states.shape[-1]):
            #     dim_data = states[:, dim_idx]
            #     if dim_idx < len(state_dim_names):
            #         name = state_dim_names[dim_idx]
            #     else:
            #         name = f"Dim{dim_idx}"
                
            #     print(f"   {name:12s}: "
            #           f"mean={np.mean(dim_data):8.4f}, "
            #           f"std={np.std(dim_data):8.4f}, "
            #           f"min={np.min(dim_data):8.4f}, "
            #           f"max={np.max(dim_data):8.4f}")
            print()
        else:
            print("⚠️  Observations 中没有 'state' 键")
            print(f"   可用的键: {list(obs_sample.keys())}")
            print()
    else:
        print("⚠️  Sample 中没有 'observations' 或不是字典类型")
        print()
    
    # 分析动作空间
    print_section("动作空间统计")
    
    actions = np.array([t['actions'] for t in all_transitions])
    print(f"动作数据形状: {actions.shape}")
    print(f"动作维度: {actions.shape[-1]}")
    print()
    
    # 🎯 检查第一帧动作（应该为0）
    print("🔍 第一帧动作检查:")
    first_action = actions[0]
    print(f"  第一帧 actions: {first_action}")
    first_action_norm = np.linalg.norm(first_action)
    print(f"  第一帧 L2 范数: {first_action_norm:.6f}")
    
    if first_action_norm < 1e-6:
        print("  ✅ 第一帧动作为零向量 (正确)")
    elif first_action_norm < 0.01:
        print(f"  ⚠️  第一帧动作接近零但不为零 (范数={first_action_norm:.6f})")
    else:
        print(f"  ❌ 警告: 第一帧动作不为零! (范数={first_action_norm:.6f})")
        print("     这可能表示数据录制有问题")
    print()
    
    # 总体统计
    print("📈 总体统计:")
    print(f"  Mean: {np.mean(actions, axis=0)}")
    print(f"  Std:  {np.std(actions, axis=0)}")
    print(f"  Min:  {np.min(actions, axis=0)}")
    print(f"  Max:  {np.max(actions, axis=0)}")
    print()
    
    # 每个维度的详细统计
    print("📊 各维度详细统计:")
    dim_names = ["Joint1", "Joint2", "Joint3", "Joint4", "Joint5", "Joint6", "Gripper"]
    for dim_idx in range(actions.shape[-1]):
        dim_data = actions[:, dim_idx]
        name = dim_names[dim_idx] if dim_idx < len(dim_names) else f"Dim{dim_idx}"
        print(f"  {name:8s}: mean={np.mean(dim_data):7.3f}, "
              f"std={np.std(dim_data):7.3f}, "
              f"min={np.min(dim_data):7.3f}, "
              f"max={np.max(dim_data):7.3f}, "
              f"range={np.max(dim_data) - np.min(dim_data):7.3f}")
    
    # 检查尺度不平衡
    print()
    print("⚠️  尺度不平衡检查:")
    ranges = np.max(actions, axis=0) - np.min(actions, axis=0)
    max_range = np.max(ranges)
    min_range = np.min(ranges)
    print(f"  最大范围: {max_range:.3f}")
    print(f"  最小范围: {min_range:.3f}")
    print(f"  范围比值: {max_range / min_range:.2f}x")
    
    if max_range / min_range > 10:
        print("  ❌ 警告: 不同维度的尺度差异过大 (>10x)!")
        print("  建议: 实现动作归一化")
    else:
        print("  ✅ 尺度差异在可接受范围内")
    
    # 分析奖励
    print_section("奖励统计")
    
    rewards = np.array([t['rewards'] for t in all_transitions])
    print(f"奖励数据形状: {rewards.shape}")
    print(f"  Mean:   {np.mean(rewards):.4f}")
    print(f"  Std:    {np.std(rewards):.4f}")
    print(f"  Min:    {np.min(rewards):.4f}")
    print(f"  Max:    {np.max(rewards):.4f}")
    print(f"  Median: {np.median(rewards):.4f}")
    
    # 检查奖励分布
    positive_rewards = np.sum(rewards > 0)
    zero_rewards = np.sum(rewards == 0)
    negative_rewards = np.sum(rewards < 0)
    
    print(f"\n奖励分布:")
    print(f"  正奖励: {positive_rewards} ({positive_rewards/len(rewards)*100:.1f}%)")
    print(f"  零奖励: {zero_rewards} ({zero_rewards/len(rewards)*100:.1f}%)")
    print(f"  负奖励: {negative_rewards} ({negative_rewards/len(rewards)*100:.1f}%)")
    
    # 分析 MC returns (如果存在)
    if 'mc_returns' in sample:
        print_section("MC Returns 统计")
        mc_returns = np.array([t['mc_returns'] for t in all_transitions])
        print(f"MC Returns 数据形状: {mc_returns.shape}")
        print(f"  Mean:   {np.mean(mc_returns):.4f}")
        print(f"  Std:    {np.std(mc_returns):.4f}")
        print(f"  Min:    {np.min(mc_returns):.4f}")
        print(f"  Max:    {np.max(mc_returns):.4f}")
    else:
        print_section("MC Returns 统计")
        print("⚠️  警告: mc_returns 不存在!")
        print("   这可能导致预训练时使用错误的返回值")
    
    # 分析 embeddings
    if 'embeddings' in sample:
        print_section("Embeddings 统计")
        embeddings = np.array([t['embeddings'] for t in all_transitions])
        print(f"Embeddings 形状: {embeddings.shape}")
        print(f"  Mean norm: {np.mean(np.linalg.norm(embeddings, axis=-1)):.4f}")
        print(f"  Std norm:  {np.std(np.linalg.norm(embeddings, axis=-1)):.4f}")
    else:
        print_section("Embeddings 统计")
        print("❌ 错误: embeddings 不存在!")
        print("   这会导致预训练失败!")
    
    if 'next_embeddings' in sample:
        next_embeddings = np.array([t['next_embeddings'] for t in all_transitions])
        print(f"Next Embeddings 形状: {next_embeddings.shape}")
    else:
        print("❌ 错误: next_embeddings 不存在!")
    
    # 分析轨迹
    print_section("轨迹统计")
    
    # 找出轨迹边界
    dones = np.array([t['dones'] for t in all_transitions])
    episode_ends = np.where(dones)[0]
    
    if len(episode_ends) > 0:
        # 计算每个 episode 的长度
        episode_lengths = []
        start = 0
        for end in episode_ends:
            episode_lengths.append(end - start + 1)
            start = end + 1
        
        print(f"总 episode 数: {len(episode_lengths)}")
        print(f"Episode 长度统计:")
        print(f"  Mean:   {np.mean(episode_lengths):.1f}")
        print(f"  Std:    {np.std(episode_lengths):.1f}")
        print(f"  Min:    {np.min(episode_lengths)}")
        print(f"  Max:    {np.max(episode_lengths)}")
        print(f"  Median: {np.median(episode_lengths):.1f}")
    else:
        print("⚠️  警告: 未检测到完整的 episode (所有 dones=False)")
    
    # 分析干预数据
    if 'intervened' in sample:
        print_section("干预统计")
        intervened = np.array([t['intervened'] for t in all_transitions])
        intervention_count = np.sum(intervened)
        print(f"干预步数: {intervention_count} ({intervention_count/len(intervened)*100:.1f}%)")
    
    return actions, rewards


def estimate_bc_loss_scale(actions, sigma_max=80.0, sigma_data=0.5):
    """估计初始 BC Loss 的尺度"""
    print_section("BC Loss 尺度估计")
    
    print(f"使用参数:")
    print(f"  sigma_max = {sigma_max}")
    print(f"  sigma_data = {sigma_data}")
    print()
    
    # 模拟噪声过程
    batch_size = min(256, len(actions))
    batch_actions = actions[:batch_size]
    
    # 采样最大噪声 (worst case)
    t = sigma_max
    noise = np.random.randn(*batch_actions.shape)
    x_t = batch_actions + noise * t
    
    # 假设模型初始预测为 0 (随机初始化)
    distiller = np.zeros_like(batch_actions)
    
    # 计算 MSE
    recon_diffs = (distiller - batch_actions) ** 2
    mean_recon_diffs = np.mean(recon_diffs, axis=-1)
    
    # 计算权重 (Karras weighting)
    snr = 1.0 / (t ** 2)
    weight = snr + 1.0 / (sigma_data ** 2)
    
    # BC Loss
    bc_loss = np.mean(mean_recon_diffs * weight)
    
    print(f"初始 BC Loss 估计 (worst case, t=sigma_max):")
    print(f"  重建误差 (MSE): {np.mean(recon_diffs):.4f}")
    print(f"  SNR: {snr:.6f}")
    print(f"  Weight: {weight:.4f}")
    print(f"  Weighted BC Loss: {bc_loss:.4f}")
    print()
    
    # 不同 t 值的估计
    print("不同噪声水平下的 BC Loss 估计:")
    t_values = [0.02, 0.5, 1.0, 5.0, 10.0, 20.0, 80.0]
    for t in t_values:
        snr = 1.0 / (t ** 2)
        weight = snr + 1.0 / (sigma_data ** 2)
        # 假设模型完全随机,预测误差约等于动作方差
        estimated_mse = np.var(batch_actions, axis=0).mean()
        estimated_bc_loss = estimated_mse * weight
        print(f"  t={t:6.2f}: SNR={snr:10.4f}, weight={weight:10.4f}, est_loss={estimated_bc_loss:10.4f}")
    
    print()
    print("💡 建议:")
    if bc_loss > 100:
        print("  ❌ 初始 BC Loss 过大 (>100)")
        print("  建议 1: 降低 sigma_max (例如从 80.0 到 5.0)")
        print("  建议 2: 实现动作归一化到 [-1, 1]")
    elif bc_loss > 10:
        print("  ⚠️  初始 BC Loss 较大 (>10)")
        print("  建议: 考虑降低 sigma_max 或归一化动作")
    else:
        print("  ✅ 初始 BC Loss 在合理范围内")


def analyze_config(exp_name):
    """分析实验配置"""
    print_section("实验配置分析")
    
    # 动态导入配置
    try:
        sys.path.insert(0, str(Path(__file__).parent / "experiments"))
        from mappings import CONFIG_MAPPING
        
        if exp_name not in CONFIG_MAPPING:
            print(f"❌ 错误: 实验 '{exp_name}' 不在 CONFIG_MAPPING 中")
            print(f"可用实验: {list(CONFIG_MAPPING.keys())}")
            return
        
        config = CONFIG_MAPPING[exp_name]()
        
        print(f"实验名称: {exp_name}")
        print(f"Setup mode: {config.setup_mode}")
        print(f"Batch size: {config.batch_size}")
        print(f"CTA ratio: {config.cta_ratio}")
        print(f"Discount: {config.discount}")
        print(f"Image keys: {config.image_keys}")
        print(f"Proprio keys: {config.proprio_keys}")
        
        if hasattr(config, 'octo_path'):
            print(f"Octo model: {config.octo_path}")
        
        if hasattr(config, 'task_desc'):
            print(f"Task description: {config.task_desc}")
        
    except Exception as e:
        print(f"⚠️  无法加载配置: {e}")


def main(_):
    if FLAGS.demo_path is None:
        print("❌ 错误: 必须提供 --demo_path")
        print("用法: python diagnose_bc_loss.py "
              "--demo_path=/path/to/demo.pkl --exp_name=a1x_pick_banana")
        return
    
    print("\n" + "🔬 " * 35)
    print("Demo 数据完整诊断工具")
    print("🔬 " * 35)
    
    # 如果启用了详细模式，先进行递归结构分析
    if FLAGS.detailed:
        analyze_trajectory_structure(FLAGS.demo_path)
    
    # 分析 demo 数据
    actions, rewards = analyze_demo_data(FLAGS.demo_path)
    
    # 估计 BC Loss 尺度
    if actions is not None and len(actions) > 0:
        estimate_bc_loss_scale(actions, sigma_max=80.0, sigma_data=0.5)
        
        print()
        print_section("尝试不同的 sigma_max 值")
        for sigma_max in [5.0, 10.0, 20.0]:
            print(f"\n--- sigma_max = {sigma_max} ---")
            estimate_bc_loss_scale(actions, sigma_max=sigma_max, sigma_data=0.5)
    
    # 分析配置
    if FLAGS.exp_name:
        analyze_config(FLAGS.exp_name)
    
    # 总结建议
    print_section("总结与建议")
    print("""
1. 检查动作空间尺度:
   - 如果不同维度范围差异 >10x，需要归一化
   - 推荐归一化到 [-1, 1]

2. 调整 diffusion 参数:
   - 如果动作未归一化: sigma_max = 5-10
   - 如果动作已归一化到 [-1,1]: sigma_max = 1-3

3. 检查数据完整性:
   - 必须包含: embeddings, next_embeddings
   - 推荐包含: mc_returns

4. 监控训练:
   - 初始 BC loss 应该 <50 (理想 <10)
   - BC loss 应该在前 100 步内开始下降
   - 如果 loss 爆炸或 NaN，立即检查数据归一化

5. 下一步:
   - 实现动作归一化 wrapper
   - 或降低 sigma_max 参数
   - 添加训练监控脚本

6. 使用 --detailed 标志查看完整的数据结构分析
    """)

if __name__ == "__main__":
    app.run(main)
