#!/usr/bin/env python3
"""
诊断 BC Loss 不收敛问题的脚本
用于检查 demo 数据的统计信息和训练配置

用法:
python /home/dungeon_master/conrft/examples/diagnose_bc_loss.py \
  --demo_path=/home/dungeon_master/conrft/examples/experiments/press_button/demo_data/20260229/traj_003_2026-02-28_17-14-48.pkl\
  --show_first_frame \
  --action_scale=0.005,0.005,0.005,0,0,0,0 \
  --detailed  # 可选: 显示详细的递归结构分析
"""

import pickle as pkl
import numpy as np

from pathlib import Path
import sys
import cv2
import os
import matplotlib.pyplot as plt
from absl import app, flags

# 配置 matplotlib 中文字体支持
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial Unicode MS', 'SimHei', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

FLAGS = flags.FLAGS
flags.DEFINE_multi_string("demo_path", None, "Path to the demo data.")
flags.DEFINE_string("exp_name", None, "Name of experiment.")
flags.DEFINE_bool("detailed", False, "Show detailed recursive structure analysis.")
flags.DEFINE_bool("save_first_frame", False, "Save first frame images to disk.")
flags.DEFINE_bool("show_first_frame", False, "Display first frame images.")
flags.DEFINE_string(
    "action_scale", None,
    "Action scale applied in the environment: single float (e.g. '0.005') or "
    "comma-separated per-dim values (e.g. '0.005,0.005,0.005,0,0,0,0'). "
    "Used to convert raw action → expected Δstate for alignment analysis."
)

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

def show_first_frame_images(demo_paths):
    """直接显示第一帧和第二帧图像"""
    print_section("显示第一帧和第二帧图像")
    
    for path in demo_paths:
        print(f"\n📁 处理文件: {path}")
        with open(path, "rb") as f:
            transitions = pkl.load(f)
        
        if len(transitions) == 0:
            print("   ⚠️ 文件为空")
            continue
        
        if len(transitions) < 2:
            print(f"   ⚠️ 只有 {len(transitions)} 帧，无法显示第二帧")
            frames_to_show = [0]
        else:
            frames_to_show = [0, 1]
        
        # 查找图像键（从第一帧）
        if 'observations' not in transitions[0]:
            print("   ⚠️ 第一帧没有 observations")
            continue
        
        obs_first = transitions[0]['observations']
        
        # 🔍 调试：打印 observations 的所有键和类型
        print(f"   🔍 第一帧 observations 的键: {list(obs_first.keys())}")
        for key, value in obs_first.items():
            if isinstance(value, np.ndarray):
                print(f"      - {key}: ndarray, shape={value.shape}, dtype={value.dtype}")
            else:
                print(f"      - {key}: {type(value).__name__}")
        
        image_keys = []
        for key in obs_first.keys():
            value = obs_first[key]
            if isinstance(value, np.ndarray) and value.ndim == 3:
                image_keys.append(key)
            elif isinstance(value, np.ndarray) and value.ndim == 4:
                # 可能是 [batch, H, W, C] 格式，取第一个
                print(f"      ⚠️ {key} 是4维数组 {value.shape}，尝试取第一个元素")
                image_keys.append(key)
        
        if len(image_keys) == 0:
            print("   ⚠️ 第一帧没有图像数据（没有3维或4维的 ndarray）")
            continue
        
        # 显示每个相机的图像
        file_basename = Path(path).stem
        num_images = len(image_keys)
        num_frames = len(frames_to_show)
        
        # 创建子图: 行=帧数, 列=相机数
        fig, axes = plt.subplots(num_frames, num_images, 
                                 figsize=(6*num_images, 6*num_frames))
        
        # 确保 axes 是 2D 数组
        if num_frames == 1 and num_images == 1:
            axes = np.array([[axes]])
        elif num_frames == 1:
            axes = axes.reshape(1, -1)
        elif num_images == 1:
            axes = axes.reshape(-1, 1)
        
        for frame_idx, frame_num in enumerate(frames_to_show):
            frame = transitions[frame_num]
            obs = frame['observations']
            
            print(f"\n   📸 第 {frame_num+1} 帧:")
            
            for img_idx, img_key in enumerate(image_keys):
                if img_key not in obs:
                    print(f"      ⚠️ {img_key} 不存在于第 {frame_num+1} 帧")
                    continue
                
                img = obs[img_key]
                
                # 处理4维数组 (batch维度)
                if img.ndim == 4:
                    print(f"      📷 {img_key} (4D, 取第一个):")
                    print(f"         原始形状: {img.shape}")
                    img = img[0]  # 取第一个batch
                    print(f"         处理后形状: {img.shape}")
                else:
                    print(f"      📷 {img_key}:")
                    print(f"         形状: {img.shape}, dtype: {img.dtype}")
                
                print(f"         范围: [{img.min()}, {img.max()}]")
                
                # 显示图像 (假设是 RGB 格式)
                axes[frame_idx, img_idx].imshow(img)
                axes[frame_idx, img_idx].set_title(
                    f"Frame {frame_num+1} - {img_key}\n{img.shape}"
                )
                axes[frame_idx, img_idx].axis('off')
        
        plt.suptitle(f"Frame 1 & 2 Comparison - {file_basename}", fontsize=14, y=0.98)
        plt.tight_layout()
        plt.show()
        
        print(f"\n   ✅ 已显示 {num_frames} 帧 × {num_images} 相机 = {num_frames*num_images} 张图像")


def save_first_frame_images(demo_paths, output_dir="./first_frame_images"):
    """保存第一帧图像到磁盘"""
    print_section("保存第一帧图像")
    
    os.makedirs(output_dir, exist_ok=True)
    
    for path in demo_paths:
        print(f"\n📁 处理文件: {path}")
        with open(path, "rb") as f:
            transitions = pkl.load(f)
        
        if len(transitions) == 0:
            print("   ⚠️ 文件为空")
            continue
        
        first_frame = transitions[0]
        
        if 'observations' not in first_frame:
            print("   ⚠️ 第一帧没有 observations")
            continue
        
        obs = first_frame['observations']
        
        # 🔍 调试：打印 observations 的所有键和类型
        print(f"   🔍 observations 的键: {list(obs.keys())}")
        for key, value in obs.items():
            if isinstance(value, np.ndarray):
                print(f"      - {key}: ndarray, shape={value.shape}, dtype={value.dtype}")
            else:
                print(f"      - {key}: {type(value).__name__}")
        
        # 查找图像键
        image_keys = []
        for key in obs.keys():
            value = obs[key]
            if isinstance(value, np.ndarray) and value.ndim == 3:
                image_keys.append(key)
            elif isinstance(value, np.ndarray) and value.ndim == 4:
                # 可能是 [batch, H, W, C] 格式
                print(f"      ⚠️ {key} 是4维数组 {value.shape}，尝试取第一个元素")
                image_keys.append(key)
        
        if len(image_keys) == 0:
            print("   ⚠️ 第一帧没有图像数据（没有3维或4维的 ndarray）")
            continue
        
        # 保存每个相机的图像
        file_basename = Path(path).stem
        for img_key in image_keys:
            img = obs[img_key]
            
            # 处理4维数组 (batch维度)
            if img.ndim == 4:
                print(f"   📷 {img_key} (4D数组):")
                print(f"      原始形状: {img.shape}")
                img = img[0]  # 取第一个batch
                print(f"      处理后形状: {img.shape}")
            
            # 转换 RGB to BGR (OpenCV 格式)
            if img.shape[-1] == 3:
                img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            else:
                img_bgr = img
            
            # 保存图像
            output_path = os.path.join(
                output_dir, f"{file_basename}_{img_key}.png"
            )
            cv2.imwrite(output_path, img_bgr)
            
            print(f"   ✅ 保存 {img_key}: {output_path}")
            print(f"      形状: {img.shape}, dtype: {img.dtype}")
            print(f"      范围: [{img.min()}, {img.max()}]")
    
    print(f"\n✅ 所有图像已保存到: {output_dir}")


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
    
    # 🎯 检测是否是 chunked actions
    is_chunked = actions.ndim == 3
    
    if is_chunked:
        print(f"✅ 检测到 Action Chunking: [N={actions.shape[0]}, chunk_size={actions.shape[1]}, action_dim={actions.shape[2]}]")
        print(f"  - 每个样本包含 {actions.shape[1]} 个连续动作")
        print(f"  - 每个动作的维度: {actions.shape[2]}")
        print()
        
        # 展平 chunk 维度用于统计分析
        # [N, chunk_size, action_dim] -> [N*chunk_size, action_dim]
        actions_flat = actions.reshape(-1, actions.shape[-1])
        print(f"展平后用于统计: {actions_flat.shape}")
        print()
        
        # 🔍 检查第一帧的第一个动作（应该为0）
        print("🔍 第一帧第一个动作检查:")
        first_action = actions[0, 0]  # [chunk_size, action_dim] -> [action_dim]
        print(f"  第一帧第一个动作: {first_action}")
        first_action_norm = np.linalg.norm(first_action)
        print(f"  第一帧第一个动作 L2 范数: {first_action_norm:.6f}")
        
        if first_action_norm < 1e-6:
            print("  ✅ 第一帧第一个动作为零向量 (正确)")
        elif first_action_norm < 0.01:
            print(f"  ⚠️  第一帧第一个动作接近零但不为零 (范数={first_action_norm:.6f})")
        else:
            print(f"  ❌ 警告: 第一帧第一个动作不为零! (范数={first_action_norm:.6f})")
            print("     这可能表示数据录制有问题")
        print()
        
        # 使用展平后的数据进行统计
        actions_for_stats = actions_flat
    else:
        print(f"动作维度: {actions.shape[-1]} (单步动作)")
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
        
        actions_for_stats = actions
    
    # 总体统计
    print("📈 总体统计:")
    print(f"  Mean: {np.mean(actions_for_stats, axis=0)}")
    print(f"  Std:  {np.std(actions_for_stats, axis=0)}")
    print(f"  Min:  {np.min(actions_for_stats, axis=0)}")
    print(f"  Max:  {np.max(actions_for_stats, axis=0)}")
    print()
    
    # 每个维度的详细统计
    print("📊 各维度详细统计:")
    dim_names = ["Joint1", "Joint2", "Joint3", "Joint4", "Joint5", "Joint6", "Gripper"]
    action_dim = actions_for_stats.shape[-1]
    for dim_idx in range(action_dim):
        dim_data = actions_for_stats[:, dim_idx]
        name = dim_names[dim_idx] if dim_idx < len(dim_names) else f"Dim{dim_idx}"
        print(f"  {name:8s}: mean={np.mean(dim_data):7.3f}, "
              f"std={np.std(dim_data):7.3f}, "
              f"min={np.min(dim_data):7.3f}, "
              f"max={np.max(dim_data):7.3f}, "
              f"range={np.max(dim_data) - np.min(dim_data):7.3f}")
    
    # 检查尺度不平衡
    print()
    print("⚠️  尺度不平衡检查:")
    ranges = np.max(actions_for_stats, axis=0) - np.min(actions_for_stats, axis=0)
    max_range = np.max(ranges)
    min_range = np.min(ranges[ranges > 0])  # 避免除以0
    print(f"  最大范围: {max_range:.3f}")
    print(f"  最小范围: {min_range:.3f}")
    if min_range > 0:
        print(f"  范围比值: {max_range / min_range:.2f}x")
        
        if max_range / min_range > 10:
            print("  ❌ 警告: 不同维度的尺度差异过大 (>10x)!")
            print("  建议: 实现动作归一化")
        else:
            print("  ✅ 尺度差异在可接受范围内")
    else:
        print("  ⚠️  某些维度范围为0，无法计算比值")
    
    # 🔍 显示最后四帧的动作（用于检查 episode 结束时的动作）
    print()
    print("🔍 最后四帧动作详情:")
    num_frames = len(all_transitions)
    for i in range(max(0, num_frames - 4), num_frames):
        action = all_transitions[i]['actions']
        done = all_transitions[i]['dones']
        reward = all_transitions[i]['rewards']
        
        done_str = "✅ DONE" if done else "      "
        reward_str = f"rew={reward:+.2f}"
        
        print(f"\n  Frame {i:3d} [{done_str}] {reward_str}:")
        
        if is_chunked and action.ndim == 2:
            # Chunked action: 显示完整的chunk
            print(f"    Shape: {action.shape} (chunk_size={action.shape[0]}, action_dim={action.shape[1]})")
            for j in range(action.shape[0]):
                action_vec = action[j]
                action_norm = np.linalg.norm(action_vec)
                print(f"    [{j}]: {action_vec}  (norm={action_norm:.4f})")
        else:
            # Single-step action: 显示单个动作
            action_norm = np.linalg.norm(action)
            print(f"    Shape: {action.shape}")
            print(f"    Value: {action}  (norm={action_norm:.4f})")
    
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


def analyze_state_action_alignment(demo_paths, action_scale=None):
    """分析 state 与 action 的对齐情况
    
    核心逻辑:
      - action[t] 经过 ACTION_SCALE 缩放后才是实际施加的增量
      - 对于 delta 控制: action[t] * action_scale ≈ next_state[t] - state[t]
      - 对于 absolute 控制: action[t] * action_scale ≈ next_state[t]
      - 通过逐帧对比及相关系数判断数据是否对齐
    
    Args:
        demo_paths: demo 文件路径列表
        action_scale: None / float / np.ndarray, 对应环境中的 ACTION_SCALE
    """
    print_section("State-Action 对齐分析")

    for path in demo_paths:
        print(f"\n{'='*70}")
        print(f"文件: {path}")
        print(f"{'='*70}")

        with open(path, "rb") as f:
            traj = pkl.load(f)

        if len(traj) < 2:
            print("   ⚠️ 帧数不足，跳过")
            continue

        # ---- 打印第 45 帧（索引 44）的 observations / actions / next_observations 具体数值 ----
        target_frame_idx = 44
        print(f"\n{'='*70}")
        print(f"🔍 第 {target_frame_idx + 1} 帧（索引 {target_frame_idx}）具体数值")
        print(f"{'='*70}")
        if target_frame_idx < len(traj):
            f44 = traj[target_frame_idx]

            def _print_obs_values(obs_dict, label):
                print(f"\n  [{label}]")
                if not isinstance(obs_dict, dict):
                    print(f"    {obs_dict}")
                    return
                for k, v in obs_dict.items():
                    v_arr = np.asarray(v) if hasattr(v, '__iter__') and not isinstance(v, str) else v
                    if isinstance(v_arr, np.ndarray) and v_arr.ndim >= 3:
                        # 跳过图像数据（3维及以上的数组）
                        print(f"    '{k}' shape={v_arr.shape} dtype={v_arr.dtype}  [图像，已跳过]")
                        continue
                    if isinstance(v_arr, np.ndarray):
                        print(f"    '{k}' shape={v_arr.shape} dtype={v_arr.dtype}")
                        print(f"         值: {v_arr}")
                    else:
                        print(f"    '{k}': {v_arr}")

            # observations
            _print_obs_values(f44.get("observations", {}), "observations")

            # actions
            act = f44.get("actions", None)
            print(f"\n  [actions]")
            if act is not None:
                act_arr = np.asarray(act)
                print(f"    shape={act_arr.shape} dtype={act_arr.dtype}")
                print(f"    值: {act_arr}")
            else:
                print("    ❌ 不存在")

            # next_observations
            _print_obs_values(f44.get("next_observations", {}), "next_observations")
        else:
            print(f"⚠️  轨迹只有 {len(traj)} 帧，不存在第 {target_frame_idx + 1} 帧")
        print(f"\n{'='*70}\n")

        # ---- 提取 state 和 action ----
        # next_observations["state"] 形状为 [2, state_dim]：
        #   [0] = t 时刻的 state
        #   [1] = t+1 时刻的 state
        states, next_states, actions = [], [], []
        for i, frame in enumerate(traj):
            nobs = frame.get("next_observations", {})
            act = frame.get("actions", None)

            ns_state = nobs.get("state", None) if isinstance(nobs, dict) else None

            if ns_state is None or act is None:
                continue

            ns_arr_raw = np.asarray(ns_state, dtype=np.float64)
            if ns_arr_raw.ndim == 2 and ns_arr_raw.shape[0] == 2:
                # [2, state_dim]: index 0 = s_t, index 1 = s_{t+1}
                s_t  = ns_arr_raw[0]
                s_t1 = ns_arr_raw[1]
            else:
                # 兜底：无法拆分，跳过
                continue

            act_arr = np.asarray(act, dtype=np.float64)
            act_vec = act_arr.flatten() if act_arr.ndim == 1 else act_arr[0].flatten()  # chunked 取第一步

            states.append(s_t)
            next_states.append(s_t1)
            actions.append(act_vec)

        if len(states) == 0:
            print("   ⚠️ 未找到 next_obs state 或 action 数据，跳过")
            continue

        states_arr = np.array(states)       # [N, state_dim]
        actions_arr = np.array(actions)     # [N, action_dim]

        state_dim = states_arr.shape[-1]
        action_dim = actions_arr.shape[-1]

        print(f"\n📐 维度信息:")
        print(f"   state_dim  = {state_dim}")
        print(f"   action_dim = {action_dim}")

        # ---- 构建 action_scale 向量 ----
        if action_scale is None:
            scale_vec = np.ones(action_dim)
            scale_label = "无缩放 (raw action)"
        else:
            scale_arr = np.asarray(action_scale, dtype=np.float64).flatten()
            if scale_arr.size == 1:
                scale_vec = np.full(action_dim, scale_arr[0])
            elif scale_arr.size == action_dim:
                scale_vec = scale_arr
            else:
                # 长度不匹配: 截断或补1
                scale_vec = np.ones(action_dim)
                n = min(scale_arr.size, action_dim)
                scale_vec[:n] = scale_arr[:n]
                print(f"   ⚠️  action_scale 长度({scale_arr.size}) ≠ action_dim({action_dim})，"
                      f"已对齐前 {n} 维，其余置 1")
            scale_label = f"已缩放 (×{scale_vec})"

        print(f"   action_scale: {scale_label}")

        # ---- 计算 state diff（next_states 全部有效，直接使用）----
        ns_arr = np.array(next_states)   # [M, state_dim]
        s_arr  = states_arr              # [M, state_dim]
        a_arr  = actions_arr             # [M, action_dim]

        # 缩放后的 action（与环境实际增量量纲一致）
        a_scaled = a_arr * scale_vec[np.newaxis, :]  # [M, action_dim]

        state_diff = ns_arr - s_arr   # [M, state_dim]

        # ---- 逐帧对比（前10帧） ----
        print()
        compare_dim = min(state_dim, action_dim)
        has_scale = not np.allclose(scale_vec, 1.0)
        print(f"🔍 逐帧对比 (前 10 帧, {'scaled action' if has_scale else 'raw action'} vs Δstate):")

        col_w = 13
        header_sd = "  ".join([f"Δstate[{d}]".rjust(col_w) for d in range(compare_dim)])
        header_ac = "  ".join([(f"a*s[{d}]" if has_scale else f"action[{d}]").rjust(col_w)
                                for d in range(compare_dim)])
        sep = "-" * (7 + compare_dim * (col_w + 2))
        print(f"  {'Frame':>5}  {header_sd}")
        print(f"  {'':>5}  {header_ac}")
        print("  " + sep)

        num_show = len(state_diff)
        for i in range(num_show):
            sd = state_diff[i, :compare_dim]
            ac = a_scaled[i, :compare_dim]
            row_sd = "  ".join([f"{v:>{col_w}.5f}" for v in sd])
            row_ac = "  ".join([f"{v:>{col_w}.5f}" for v in ac])
            print(f"  {i:>5}  {row_sd}   ← Δstate")
            print(f"  {'':>5}  {row_ac}   ← {'action×scale' if has_scale else 'action'}")
            print()

        # ---- 相关系数分析（仅当维度匹配时，跳过 scale=0 的维度）----
        if state_dim == action_dim:
            print("📊 各维度 Pearson 相关系数 (scaled_action vs Δstate):")
            corrs = []
            for d in range(state_dim):
                sd_col = state_diff[:, d]
                ac_col = a_scaled[:, d]
                scale_d = scale_vec[d] if d < len(scale_vec) else 1.0
                if scale_d == 0.0:
                    corrs.append(float('nan'))
                    print(f"   dim {d:2d}: ⏭️  scale=0，该维度被环境忽略，跳过")
                elif np.std(sd_col) < 1e-9 or np.std(ac_col) < 1e-9:
                    corrs.append(float('nan'))
                    print(f"   dim {d:2d}: ⚠️  方差近零，无法计算（scale={scale_d}）")
                else:
                    corr = np.corrcoef(sd_col, ac_col)[0, 1]
                    corrs.append(corr)
                    flag = "✅" if abs(corr) > 0.7 else ("⚠️ " if abs(corr) > 0.3 else "❌")
                    print(f"   dim {d:2d}: {flag} corr = {corr:+.4f}  (scale={scale_d})")

            valid_corrs = [c for c in corrs if not np.isnan(c)]
            if valid_corrs:
                mean_corr = np.mean(np.abs(valid_corrs))
                print(f"\n   平均 |corr| (忽略 scale=0 维度): {mean_corr:.4f}")
                if mean_corr > 0.7:
                    print("   ✅ 整体对齐良好 → scaled_action 与 Δstate 强相关")
                elif mean_corr > 0.3:
                    print("   ⚠️  对齐一般 → 可能是 absolute 控制，或存在时序偏移")
                else:
                    print("   ❌ 对齐差 → scaled_action 与 Δstate 几乎不相关，")
                    print("      请检查: 1) 时序偏移  2) absolute vs delta  3) 坐标系差异")
            else:
                print("   ⚠️  所有维度均被跳过（全部 scale=0 或方差为零）")
        else:
            print(f"   ⚠️  state_dim({state_dim}) ≠ action_dim({action_dim})，")
            print("       跳过相关系数分析（维度不一致，可能是关节空间 action + 末端 state）")

        # ---- action magnitude vs state diff magnitude ----
        print()
        act_label = "‖a×scale‖" if has_scale else "‖action‖"
        print(f"📏 幅值对比 (全部帧):")
        print(f"  {'Frame':>5}  {'‖Δstate‖':>12}  {act_label:>12}  {'ratio':>8}")
        print("  " + "-" * 44)
        for i in range(len(state_diff)):
            norm_diff = np.linalg.norm(state_diff[i])
            norm_act  = np.linalg.norm(a_scaled[i])
            ratio = norm_act / norm_diff if norm_diff > 1e-9 else float('inf')
            print(f"  {i:>5}  {norm_diff:>12.6f}  {norm_act:>12.6f}  {ratio:>8.3f}")

        # ---- 全局幅值统计 ----
        print()
        norm_diffs = np.linalg.norm(state_diff, axis=-1)  # [M]
        norm_acts  = np.linalg.norm(a_scaled, axis=-1)     # [M]
        print("📈 全局幅值统计:")
        print(f"   ‖Δstate‖    mean={np.mean(norm_diffs):.6f}, std={np.std(norm_diffs):.6f}, "
              f"max={np.max(norm_diffs):.6f}")
        print(f"   {act_label:<12} mean={np.mean(norm_acts):.6f}, std={np.std(norm_acts):.6f}, "
              f"max={np.max(norm_acts):.6f}")


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
    
    # 🖼️ 如果启用了显示第一帧图像
    if FLAGS.show_first_frame:
        show_first_frame_images(FLAGS.demo_path)
        print()
    
    # 🖼️ 如果启用了保存第一帧图像
    if FLAGS.save_first_frame:
        save_first_frame_images(FLAGS.demo_path)
        print()
    
    # 如果启用了详细模式，先进行递归结构分析
    if FLAGS.detailed:
        analyze_trajectory_structure(FLAGS.demo_path)
    
    # 分析 demo 数据
    actions, rewards = analyze_demo_data(FLAGS.demo_path)

    # 解析 action_scale
    action_scale = None
    if FLAGS.action_scale is not None:
        try:
            parts = [float(x.strip()) for x in FLAGS.action_scale.split(",")]
            action_scale = np.array(parts) if len(parts) > 1 else parts[0]
            print(f"\n📏 使用 action_scale = {action_scale}")
        except ValueError:
            print(f"\n⚠️  无法解析 --action_scale='{FLAGS.action_scale}'，使用无缩放")

    # State-Action 对齐分析
    analyze_state_action_alignment(FLAGS.demo_path, action_scale=action_scale)

    # 总结建议
    print_section("总结与建议")
    print("""
1. State-Action 对齐判断:
   - 若各维度 corr > 0.7 → action 是 delta 控制，与 Δstate 强对应
   - 若 corr 较低但幅值比 ratio ≈ 1 → 可能是 absolute 控制
   - 若 corr 极低且 ratio 差异大 → 检查时序偏移或坐标系问题

3. 检查数据完整性:
   - 必须包含: embeddings, next_embeddings
   - 推荐包含: mc_returns

4. 使用 --detailed 标志查看完整的数据结构分析
    """)

if __name__ == "__main__":
    app.run(main)
