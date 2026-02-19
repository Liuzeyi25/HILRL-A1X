#!/usr/bin/env python3
"""
测试 RealSense 相机连接和显示
识别所有连接的相机并实时显示图像
"""

import sys
import cv2
import numpy as np

# 添加路径以导入 franka_env
sys.path.insert(0, 'serl_robot_infra')

from franka_env.camera.rs_capture import RSCapture
from franka_env.camera.video_capture import VideoCapture


def test_all_cameras():
    """检测并显示所有连接的相机"""
    
    print("=" * 60)
    print("🎥 RealSense 相机检测工具")
    print("=" * 60)
    
    # 获取所有可用的相机序列号
    # 需要创建一个临时实例来调用方法
    try:
        import pyrealsense2 as rs
        ctx = rs.context()
        devices = ctx.devices
        available_serials = [d.get_info(rs.camera_info.serial_number) for d in devices]
    except Exception as e:
        print(f"❌ 获取相机列表失败: {e}")
        available_serials = []
    
    if not available_serials:
        print("❌ 未检测到任何 RealSense 相机!")
        print("请检查:")
        print("  1. 相机是否正确连接到 USB 端口")
        print("  2. USB 线缆是否工作正常")
        print("  3. RealSense SDK 是否正确安装")
        return
    
    print(f"✅ 检测到 {len(available_serials)} 个相机:")
    for i, serial in enumerate(available_serials, 1):
        print(f"   {i}. 序列号: {serial}")
    print()
    
    # 为每个相机创建捕获对象
    cameras = {}
    for serial in available_serials:
        try:
            print(f"初始化相机 {serial}...")
            cap = VideoCapture(
                RSCapture(
                    name=f"camera_{serial}",
                    serial_number=serial,
                    dim=(1280, 720),
                    exposure=10000
                )
            )
            cameras[serial] = cap
            print(f"  ✓ 相机 {serial} 初始化成功")
        except Exception as e:
            print(f"  ✗ 相机 {serial} 初始化失败: {e}")
    
    if not cameras:
        print("❌ 没有成功初始化任何相机!")
        return
    
    print()
    print("=" * 60)
    print("📺 开始显示相机画面")
    print("=" * 60)
    print("操作说明:")
    print("  - 按 'q' 键退出")
    print("  - 按 's' 键保存当前所有相机的截图")
    print("  - 按 'i' 键打印相机信息")
    print()
    
    frame_count = 0
    
    try:
        while True:
            frame_count += 1
            
            # 读取所有相机的图像
            images = {}
            for serial, cap in cameras.items():
                try:
                    img = cap.read()
                    if img is not None:
                        images[serial] = img
                except Exception as e:
                    print(f"读取相机 {serial} 失败: {e}")
            
            if not images:
                print("❌ 无法读取任何相机图像!")
                break
            
            # 显示每个相机的图像
            for serial, img in images.items():
                # 添加文字标签
                display_img = img.copy()
                
                # 添加相机信息
                text = f"Camera: {serial}"
                cv2.putText(display_img, text, (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                
                # 添加分辨率信息
                res_text = f"Resolution: {img.shape[1]}x{img.shape[0]}"
                cv2.putText(display_img, res_text, (10, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
                # 添加帧计数
                frame_text = f"Frame: {frame_count}"
                cv2.putText(display_img, frame_text, (10, 90),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
                # 显示窗口
                window_name = f"Camera {serial}"
                cv2.imshow(window_name, display_img)
            
            # 如果有多个相机，创建一个组合视图
            if len(images) > 1:
                # 将所有图像缩小并排列显示
                resized_images = []
                for serial, img in images.items():
                    # 缩小到 640x360
                    resized = cv2.resize(img, (640, 360))
                    # 添加标签
                    label = f"Camera: {serial}"
                    cv2.putText(resized, label, (10, 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    resized_images.append(resized)
                
                # 水平拼接
                if len(resized_images) == 2:
                    combined = np.hstack(resized_images)
                else:
                    # 如果有多个，尝试网格布局
                    combined = np.hstack(resized_images)
                
                cv2.imshow("All Cameras", combined)
            
            # 检查按键
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                print("\n用户退出")
                break
            elif key == ord('s'):
                # 保存截图
                import datetime
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                for serial, img in images.items():
                    filename = f"camera_{serial}_{timestamp}.png"
                    cv2.imwrite(filename, img)
                    print(f"💾 保存截图: {filename}")
            elif key == ord('i'):
                # 打印相机信息
                print("\n" + "=" * 60)
                print("📷 相机信息:")
                for serial, img in images.items():
                    print(f"  相机 {serial}:")
                    print(f"    - 分辨率: {img.shape[1]}x{img.shape[0]}")
                    print(f"    - 通道数: {img.shape[2]}")
                    print(f"    - 数据类型: {img.dtype}")
                print("=" * 60 + "\n")
    
    except KeyboardInterrupt:
        print("\n\n⚠️ 程序被中断")
    
    finally:
        # 关闭所有窗口
        cv2.destroyAllWindows()
        
        # 关闭所有相机
        for serial, cap in cameras.items():
            try:
                cap.close()
                print(f"关闭相机 {serial}")
            except:
                pass


def test_specific_camera(serial_number):
    """测试指定序列号的相机"""
    
    print(f"测试相机: {serial_number}")
    
    try:
        cap = VideoCapture(
            RSCapture(
                name="test_camera",
                serial_number=serial_number,
                dim=(1280, 720),
                exposure=10000
            )
        )
        
        print("✓ 相机初始化成功")
        print("按 'q' 键退出")
        
        while True:
            img = cap.read()
            
            if img is not None:
                # 添加信息
                display_img = img.copy()
                text = f"Camera: {serial_number}"
                cv2.putText(display_img, text, (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                cv2.imshow(f"Camera {serial_number}", display_img)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.close()
        cv2.destroyAllWindows()
        
    except Exception as e:
        print(f"❌ 错误: {e}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="RealSense 相机测试工具")
    parser.add_argument(
        "--serial", 
        type=str, 
        default=None,
        help="指定相机序列号（可选，不指定则显示所有相机）"
    )
    
    args = parser.parse_args()
    
    if args.serial:
        test_specific_camera(args.serial)
    else:
        test_all_cameras()
