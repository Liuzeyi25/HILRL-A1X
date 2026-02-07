#!/usr/bin/env python3
"""
检查 RealSense 相机是否正确连接和初始化

使用方法:
    python check_cameras.py
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from franka_env.camera.rs_capture import RSCapture
from franka_env.camera.video_capture import VideoCapture
from collections import OrderedDict
import cv2
import numpy as np


def check_realsense_devices():
    """使用 pyrealsense2 检查连接的设备"""
    print("=" * 70)
    print("🔍 检查 RealSense 设备")
    print("=" * 70)
    
    try:
        import pyrealsense2 as rs
        
        ctx = rs.context()
        devices = ctx.query_devices()
        
        if len(devices) == 0:
            print("❌ 没有检测到 RealSense 设备!")
            return []
        
        print(f"✅ 检测到 {len(devices)} 个 RealSense 设备:\n")
        
        device_info = []
        for i, dev in enumerate(devices):
            info = {
                'name': dev.get_info(rs.camera_info.name),
                'serial': dev.get_info(rs.camera_info.serial_number),
                'firmware': dev.get_info(rs.camera_info.firmware_version),
                'usb_type': dev.get_info(rs.camera_info.usb_type_descriptor),
            }
            device_info.append(info)
            
            print(f"设备 {i+1}:")
            print(f"  📷 名称: {info['name']}")
            print(f"  🔢 序列号: {info['serial']}")
            print(f"  📦 固件版本: {info['firmware']}")
            print(f"  🔌 USB类型: {info['usb_type']}")
            print()
        
        return device_info
        
    except ImportError:
        print("❌ pyrealsense2 未安装!")
        print("   安装命令: pip install pyrealsense2")
        return []
    except Exception as e:
        print(f"❌ 检查设备时出错: {e}")
        return []


def test_camera_init(camera_config):
    """测试相机初始化"""
    print("=" * 70)
    print("🎥 测试相机初始化")
    print("=" * 70)
    
    results = {}
    
    for cam_name, kwargs in camera_config.items():
        print(f"\n测试相机: {cam_name}")
        print(f"  序列号: {kwargs['serial_number']}")
        print(f"  分辨率: {kwargs['dim']}")
        print(f"  曝光: {kwargs['exposure']}")
        
        try:
            # 尝试初始化相机
            cap = VideoCapture(RSCapture(name=cam_name, **kwargs))
            
            # 尝试读取一帧
            frame = cap.read()
            
            if frame is not None and frame.size > 0:
                print(f"  ✅ 初始化成功!")
                print(f"  📐 图像尺寸: {frame.shape}")
                results[cam_name] = {
                    'status': 'success',
                    'shape': frame.shape,
                    'cap': cap
                }
            else:
                print(f"  ❌ 无法读取图像")
                results[cam_name] = {
                    'status': 'failed',
                    'error': 'Cannot read frame'
                }
                
        except Exception as e:
            print(f"  ❌ 初始化失败: {e}")
            results[cam_name] = {
                'status': 'failed',
                'error': str(e)
            }
    
    return results


def display_test_images(results):
    """显示测试图像"""
    print("\n" + "=" * 70)
    print("📺 显示测试图像 (按 'q' 退出)")
    print("=" * 70)
    
    successful_cams = {name: res for name, res in results.items() 
                      if res['status'] == 'success'}
    
    if not successful_cams:
        print("❌ 没有成功初始化的相机")
        return
    
    print(f"\n按任意键切换相机, 按 'q' 退出\n")
    
    try:
        while True:
            for cam_name, res in successful_cams.items():
                cap = res['cap']
                frame = cap.read()
                
                if frame is not None:
                    # 添加文本标签
                    labeled_frame = frame.copy()
                    cv2.putText(labeled_frame, f"Camera: {cam_name}", 
                               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                               1, (0, 255, 0), 2)
                    cv2.putText(labeled_frame, f"Shape: {frame.shape}", 
                               (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 
                               0.7, (0, 255, 0), 2)
                    
                    # 显示图像
                    cv2.imshow(f'RealSense Test - {cam_name}', labeled_frame)
                    
                    # 等待按键
                    key = cv2.waitKey(30)
                    if key == ord('q'):
                        print("\n退出显示")
                        cv2.destroyAllWindows()
                        return
                    elif key != -1:  # 任意按键切换到下一个相机
                        cv2.destroyAllWindows()
                        break
                        
    except KeyboardInterrupt:
        print("\n\n用户中断")
    finally:
        cv2.destroyAllWindows()
        # 关闭相机
        for res in successful_cams.values():
            if 'cap' in res:
                res['cap'].close()


def main():
    """主函数"""
    print("\n" + "🎯" * 35)
    print("RealSense 相机诊断工具")
    print("🎯" * 35 + "\n")
    
    # 步骤 1: 检查设备
    devices = check_realsense_devices()
    
    if not devices:
        print("\n❌ 无法继续,没有检测到设备")
        return
    
    # 步骤 2: 定义相机配置 (从 config.py 复制)
    camera_config = {
        "wrist_1": {
            "serial_number": "044322073334",
            "dim": (1280, 720),
            "exposure": 10500,
        },
        "side_policy_256": {
            "serial_number": "243222075799",
            "dim": (1280, 720),
            "exposure": 10500,
        },
    }
    
    # 检查配置的相机是否在检测到的设备中
    print("\n" + "=" * 70)
    print("✅ 配置验证")
    print("=" * 70)
    
    detected_serials = {dev['serial'] for dev in devices}
    
    for cam_name, config in camera_config.items():
        serial = config['serial_number']
        if serial in detected_serials:
            print(f"✅ {cam_name}: 序列号 {serial} 已检测到")
        else:
            print(f"❌ {cam_name}: 序列号 {serial} 未检测到!")
    
    # 步骤 3: 测试初始化
    input("\n按 Enter 继续测试相机初始化...")
    results = test_camera_init(camera_config)
    
    # 步骤 4: 显示摘要
    print("\n" + "=" * 70)
    print("📊 测试摘要")
    print("=" * 70)
    
    success_count = sum(1 for res in results.values() if res['status'] == 'success')
    total_count = len(results)
    
    print(f"\n✅ 成功: {success_count}/{total_count}")
    
    for cam_name, res in results.items():
        status_icon = "✅" if res['status'] == 'success' else "❌"
        print(f"{status_icon} {cam_name}: {res['status']}")
        if res['status'] == 'failed':
            print(f"   错误: {res.get('error', 'Unknown')}")
    
    # 步骤 5: 显示测试图像
    if success_count > 0:
        choice = input("\n是否显示测试图像? (y/n): ").strip().lower()
        if choice == 'y':
            display_test_images(results)
        else:
            # 清理资源
            for res in results.values():
                if 'cap' in res and res['status'] == 'success':
                    res['cap'].close()
    
    print("\n" + "🎯" * 35)
    print("测试完成!")
    print("🎯" * 35 + "\n")


if __name__ == "__main__":
    main()
