#!/usr/bin/env python3
"""
合并多个轨迹文件为单个预训练数据文件

使用方法：
   cd /home/dungeon_master/conrft/examples/
   python merge_trajectories.py \
       /home/dungeon_master/conrft/examples/experiments/wipe_whiteboard/demo_data/20260222 \
       /home/dungeon_master/conrft/examples/experiments/wipe_whiteboard/demo_data/20260222/traj_20.pkl 

"""
import pickle as pkl
import glob
import sys
import os
from natsort import natsorted


def merge_trajectories(input_path, output_file):
    """
    合并多个轨迹pkl文件
    
    Args:
        input_path: 文件匹配模式 或 目录路径
                   如 "traj_*_manual_*.pkl" 或 "/path/to/demo_data"
        output_file: 输出文件名，如 "a1x_pick_banana_30_demos.pkl"
    """
    # 判断是目录还是模式
    if os.path.isdir(input_path):
        # 如果是目录，匹配该目录下所有 pkl 文件
        pattern = os.path.join(input_path, "*.pkl")
        print(f"📁 扫描目录: {input_path}")
    else:
        # 否则作为模式使用
        pattern = input_path
    
    # 获取所有匹配的文件并排序
    files = natsorted(glob.glob(pattern))
    
    if not files:
        print(f"❌ 没有找到匹配的文件: {pattern}")
        return
    
    print(f"📁 找到 {len(files)} 个轨迹文件")
    
    # 过滤掉输出文件本身（如果存在）
    output_file_abs = os.path.abspath(output_file)
    files = [f for f in files if os.path.abspath(f) != output_file_abs]
    
    if not files:
        print(f"❌ 过滤后没有可用文件")
        return
    
    print(f"📊 将合并 {len(files)} 个文件")
    
    # 收集所有 transitions
    all_transitions = []
    
    for i, file_path in enumerate(files, 1):
        print(f"🔄 处理 {i}/{len(files)}: {file_path}")
        
        with open(file_path, "rb") as f:
            trajectory = pkl.load(f)
        
        # 验证数据格式
        if not isinstance(trajectory, list):
            print(f"   ⚠️ 警告: 不是列表格式，跳过")
            continue
        
        if len(trajectory) == 0:
            print(f"   ⚠️ 警告: 空轨迹，跳过")
            continue
        
        # 检查第一帧是否有必需字段
        first_frame = trajectory[0]
        required_keys = {"observations", "actions", "next_observations", 
                        "rewards", "masks", "dones"}
        
        if not required_keys.issubset(first_frame.keys()):
            missing = required_keys - first_frame.keys()
            print(f"   ⚠️ 警告: 缺少必需字段 {missing}，跳过")
            continue
        
        # 检查是否有 embeddings
        has_embeddings = "embeddings" in first_frame
        has_next_embeddings = "next_embeddings" in first_frame
        has_mc_returns = "mc_returns" in first_frame
        
        print(f"   ✅ 长度: {len(trajectory)} 帧")
        print(f"      embeddings: {has_embeddings}")
        print(f"      next_embeddings: {has_next_embeddings}")
        print(f"      mc_returns: {has_mc_returns}")
        
        if not (has_embeddings and has_next_embeddings and has_mc_returns):
            print("   ⚠️ 警告: 缺少训练所需字段，可能无法用于预训练")
        
        # 添加所有 transitions
        all_transitions.extend(trajectory)
    
    # 保存合并后的数据
    print(f"\n💾 保存合并后的数据...")
    print(f"   总transitions数: {len(all_transitions)}")
    
    # 转换为绝对路径
    output_file_abs = os.path.abspath(output_file)
    print(f"   输出文件: {output_file_abs}")
    
    with open(output_file_abs, "wb") as f:
        pkl.dump(all_transitions, f)
    
    print(f"\n🎉 完成！成功合并 {len(files)} 个轨迹文件")
    print(f"   共 {len(all_transitions)} 个 transitions")
    
    # 验证保存的文件
    print(f"\n🔍 验证保存的文件...")
    with open(output_file_abs, "rb") as f:
        loaded = pkl.load(f)
    
    print(f"   ✅ 文件可正常加载")
    print(f"   ✅ Transitions数: {len(loaded)}")
    if len(loaded) > 0:
        print(f"   ✅ 第一帧键: {list(loaded[0].keys())}")
        if "embeddings" in loaded[0]:
            print(f"   ✅ Embeddings shape: {loaded[0]['embeddings'].shape}")
        if "actions" in loaded[0]:
            print(f"   ✅ Actions shape: {loaded[0]['actions'].shape}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法:")
        print("  python merge_trajectories.py <input> [output_file]")
        print()
        print("参数:")
        print("  input: 目录路径 或 文件匹配模式")
        print("  output_file: 输出文件名 (可选，默认为 merged_demos.pkl)")
        print()
        print("示例:")
        print("  # 合并目录下所有 pkl 文件")
        print("  python merge_trajectories.py /path/to/demo_data")
        print("  python merge_trajectories.py /path/to/demo_data output.pkl")
        print()
        print("  # 使用文件模式")
        print("  python merge_trajectories.py 'traj_*_manual_*.pkl' output.pkl")
        print("  python merge_trajectories.py 'traj_*.pkl'")
        sys.exit(1)
    
    input_path = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) >= 3 else "merged_demos.pkl"
    
    merge_trajectories(input_path, output_file)
