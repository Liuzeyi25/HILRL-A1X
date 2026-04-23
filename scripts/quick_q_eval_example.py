#!/usr/bin/env python3
"""
quick_q_eval_example.py
=======================
快速示例：演示如何使用 evaluate_q_values.py

这个脚本展示了完整的工作流：
1. 从 checkpoint buffer 中提取轨迹
2. 调用评估脚本
3. 查看结果
"""

import os
import pickle as pkl
import subprocess
import sys
import glob
from pathlib import Path

def extract_trajectory_from_buffer(checkpoint_path: str, buffer_index: int = 0) -> str:
    """
    从 checkpoint 的 buffer 目录提取一条轨迹。
    
    返回：保存的轨迹文件路径
    """
    buffer_dir = os.path.join(checkpoint_path, "buffer")
    
    if not os.path.exists(buffer_dir):
        print(f"✗ 未找到 buffer 目录: {buffer_dir}")
        return None
    
    # 查找 buffer pkl 文件
    buffer_files = sorted(glob.glob(os.path.join(buffer_dir, "transitions_*.pkl")))
    
    if not buffer_files:
        print(f"✗ 未找到 transitions_*.pkl 文件")
        return None
    
    # 默认使用第一个文件
    pkl_file = buffer_files[buffer_index]
    print(f"[提取] 从 {os.path.basename(pkl_file)} 中提取轨迹...")
    
    with open(pkl_file, "rb") as f:
        transitions = pkl.load(f)
    
    print(f"  ✓ 加载了 {len(transitions)} 条 transitions")
    
    # 保存为单独的文件
    output_path = "extracted_trajectory.pkl"
    with open(output_path, "wb") as f:
        pkl.dump(transitions, f)
    
    print(f"  ✓ 已保存为: {output_path}")
    return output_path


def run_q_evaluation(model_paths, trajectory_path, exp_name, output_dir):
    """
    调用 evaluate_q_values.py 进行评估。
    """
    cmd = [
        "python3",
        "scripts/evaluate_q_values.py",
        "--exp_name", exp_name,
        "--trajectory_paths", trajectory_path,
        "--output_dir", output_dir,
        "--save_video_samples",
        "--ensemble_agg", "min",
        "--gamma", "0.99",
    ]
    
    # 添加模型路径
    cmd.extend(["--model_paths"] + model_paths)
    
    print(f"\n[命令] 执行 Q 值评估:")
    print("  " + " ".join(cmd))
    print()
    
    try:
        result = subprocess.run(cmd, check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ 命令执行失败: {e}")
        return False


def main():
    print("=" * 70)
    print("Q 值评估工具 - 快速示例")
    print("=" * 70)
    print()
    
    # ── 配置 ──────────────────────────────────────────────────────
    # 修改这些路径以匹配你的实际情况
    REPO_ROOT = "/home/dungeon_master/liuzeyi/HILRL-A1X"
    CHECKPOINT_ROOT = os.path.join(
        REPO_ROOT,
        "examples/experiments/insert_block/experiments/insert_block/hilserl/0423_baseline_1"
    )
    
    # 要评估的 checkpoint 步数（需要存在）
    CHECKPOINT_STEPS = [4000, 8000, 12000, 16000]
    
    # 检查哪些 checkpoint 存在
    available_checkpoints = []
    for step in CHECKPOINT_STEPS:
        ckpt_path = os.path.join(CHECKPOINT_ROOT, f"checkpoint_{step}")
        if os.path.exists(ckpt_path):
            available_checkpoints.append(ckpt_path)
            print(f"✓ 找到 checkpoint_{step}")
        else:
            print(f"✗ 未找到 checkpoint_{step}")
    
    if not available_checkpoints:
        print("\n✗ 未找到任何 checkpoint 文件")
        print(f"  预期路径: {CHECKPOINT_ROOT}")
        return
    
    print(f"\n[检测] 共找到 {len(available_checkpoints)} 个 checkpoint")
    print()
    
    # ── 提取轨迹 ──────────────────────────────────────────────────
    print("=" * 70)
    print("第 1 步：从 buffer 中提取轨迹")
    print("=" * 70)
    
    # 优先使用最新的 checkpoint 的 buffer
    latest_ckpt = available_checkpoints[-1]
    trajectory_path = extract_trajectory_from_buffer(latest_ckpt)
    
    if not trajectory_path:
        print("\n✗ 未能提取轨迹")
        return
    
    # ── 执行评估 ──────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("第 2 步：执行 Q 值评估")
    print("=" * 70)
    
    output_dir = "./q_evaluation_results"
    os.makedirs(output_dir, exist_ok=True)
    
    # 情况 1：单模型评估（最新 checkpoint）
    print("\n[场景 1] 单模型评估（最新 checkpoint）")
    success = run_q_evaluation(
        [latest_ckpt],
        trajectory_path,
        "insert_block",
        os.path.join(output_dir, "single_model")
    )
    
    if not success:
        print("\n✗ 评估失败")
        return
    
    # 情况 2：多模型对比（所有可用 checkpoint）
    if len(available_checkpoints) > 1:
        print("\n[场景 2] 多模型对比（所有可用 checkpoint）")
        success = run_q_evaluation(
            available_checkpoints,
            trajectory_path,
            "insert_block",
            os.path.join(output_dir, "multi_model")
        )
    
    # ── 显示结果 ──────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("第 3 步：查看结果")
    print("=" * 70)
    print()
    print(f"✓ 评估完成！结果已保存至: {os.path.abspath(output_dir)}")
    print()
    print("  输出文件列表:")
    for root, dirs, files in os.walk(output_dir):
        level = root.replace(output_dir, "").count(os.sep)
        indent = " " * 4 * (level + 1)
        print(f"{indent}{os.path.basename(root)}/")
        subindent = " " * 4 * (level + 2)
        for file in files[:10]:  # 最多显示 10 个文件
            print(f"{subindent}{file}")
        if len(files) > 10:
            print(f"{subindent}... 还有 {len(files) - 10} 个文件")
    
    print("\n  主要输出：")
    print(f"    - q_comparison_traj0.png      (Q 值曲线对比)")
    print(f"    - q_statistics.png            (统计对比)")
    print(f"    - trajectory_0_frames/        (图像帧序列)")
    
    print("\n  查看结果：")
    if sys.platform == "darwin":
        print(f"    open {os.path.abspath(output_dir)}")
    else:
        print(f"    xdg-open {os.path.abspath(output_dir)}")
    
    print("\n" + "=" * 70)
    print("✓ 示例完成！")
    print("=" * 70)


if __name__ == "__main__":
    os.chdir("/home/dungeon_master/liuzeyi/HILRL-A1X")
    main()
