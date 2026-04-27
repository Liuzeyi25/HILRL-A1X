#!/usr/bin/env python3
"""从 MP4 视频中抽取指定帧并保存为图片"""

import cv2
import os
import sys

# ========== 配置 ==========
VIDEO_PATH = "/Users/liuzeyi/Desktop/q_trajectory_video_zeyi_1.mp4"          # 输入视频路径，可通过命令行参数覆盖
OUTPUT_DIR = "extracted_frames"   # 输出目录
FRAMES_TO_EXTRACT = [5, 30, 55, 85, 110, 150]  # 要抽取的帧编号（从0开始计数）
# ==========================


def extract_frames(video_path: str, frame_indices: list, output_dir: str):
    if not os.path.exists(video_path):
        print(f"[错误] 视频文件不存在: {video_path}")
        sys.exit(1)

    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[错误] 无法打开视频: {video_path}")
        sys.exit(1)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"[信息] 视频总帧数: {total_frames}, FPS: {fps:.2f}")

    saved = []
    for idx in sorted(frame_indices):
        if idx >= total_frames:
            print(f"[警告] 帧 {idx} 超出视频总帧数 ({total_frames})，跳过")
            continue

        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            print(f"[警告] 读取第 {idx} 帧失败，跳过")
            continue

        out_path = os.path.join(output_dir, f"frame_{idx:06d}.png")
        cv2.imwrite(out_path, frame)
        print(f"[保存] 第 {idx:>4d} 帧 -> {out_path}")
        saved.append(out_path)

    cap.release()
    print(f"\n完成！共保存 {len(saved)} 张图片到目录: {output_dir}")


if __name__ == "__main__":
    video = sys.argv[1] if len(sys.argv) > 1 else VIDEO_PATH
    extract_frames(video, FRAMES_TO_EXTRACT, OUTPUT_DIR)
