#!/usr/bin/env python3
"""
相机图像裁剪可视化工具
用于调试和调整 IMAGE_CROP 参数

使用方法:
    python visualize_camera_crop.py

键盘控制:
    q - 退出
    s - 保存当前裁剪参数到文件
    1-9 - 切换不同的预设裁剪方案
"""

import cv2
import numpy as np
from franka_env.camera.rs_capture import RSCapture
from franka_env.camera.video_capture import VideoCapture


class CropVisualizer:
    def __init__(self):
        # 从配置文件读取相机配置
        self.cameras = {
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
        
        # 定义多个裁剪预设方案
        self.crop_presets = {
            0: {
                "name": "原始图像 (无裁剪)",
                "wrist_1": lambda img: img,
                "side_policy_256": lambda img: img,
            },
            1: {
                "name": "中心裁剪 640x640",
                "wrist_1": lambda img: img[40:-40, 320:-320],
                "side_policy_256": lambda img: img[40:-40, 320:-320],
            },
            2: {
                "name": "上半部裁剪",
                "wrist_1": lambda img: img[0:360, :],
                "side_policy_256": lambda img: img[0:360, :],
            },
            3: {
                "name": "下半部裁剪",
                "wrist_1": lambda img: img[360:720, :],
                "side_policy_256": lambda img: img[360:720, :],
            },
            4: {
                "name": "左半部裁剪",
                "wrist_1": lambda img: img[:, 0:640],
                "side_policy_256": lambda img: img[:, 0:640],
            },
            5: {
                "name": "右半部裁剪",
                "wrist_1": lambda img: img[:, 640:1280],
                "side_policy_256": lambda img: img[:, 640:1280],
            },
            6: {
                "name": "自定义裁剪 1",
                "wrist_1": lambda img: img[100:-100, 200:-200],
                "side_policy_256": lambda img: img[250:-150, 400:-500],
            },
            7: {
                "name": "自定义裁剪 2 (侧面分类器)",
                "wrist_1": lambda img: img,
                "side_policy_256": lambda img: img[390:-150, 420:-700],
            },
            8: {
                "name": "中心 800x600",
                "wrist_1": lambda img: img[60:-60, 240:-240],
                "side_policy_256": lambda img: img[60:-60, 240:-240],
            },
        }
        
        self.current_preset = 0
        self.caps = {}
        
    def init_cameras(self):
        """初始化相机"""
        print("正在初始化相机...")
        for cam_name, kwargs in self.cameras.items():
            try:
                self.caps[cam_name] = VideoCapture(
                    RSCapture(name=cam_name, **kwargs)
                )
                print(f"✓ {cam_name} 初始化成功")
            except Exception as e:
                print(f"✗ {cam_name} 初始化失败: {e}")
        
        if not self.caps:
            raise RuntimeError("没有成功初始化任何相机")
    
    def apply_crop(self, image, camera_name):
        """应用当前选中的裁剪"""
        preset = self.crop_presets[self.current_preset]
        try:
            cropped = preset[camera_name](image)
            return cropped
        except Exception as e:
            print(f"裁剪失败 ({camera_name}): {e}")
            return image
    
    def add_info_overlay(self, image, camera_name, crop_info):
        """在图像上添加信息覆盖层"""
        overlay = image.copy()
        height, width = overlay.shape[:2]
        
        # 半透明背景
        cv2.rectangle(overlay, (0, 0), (width, 80), (0, 0, 0), -1)
        image_with_overlay = cv2.addWeighted(overlay, 0.6, image, 0.4, 0)
        
        # 添加文字信息
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(image_with_overlay, f"Camera: {camera_name}", 
                   (10, 20), font, 0.6, (255, 255, 255), 2)
        cv2.putText(image_with_overlay, f"Size: {width}x{height}", 
                   (10, 45), font, 0.6, (255, 255, 255), 2)
        cv2.putText(image_with_overlay, crop_info, 
                   (10, 70), font, 0.5, (0, 255, 255), 2)
        
        return image_with_overlay
    
    def save_crop_params(self):
        """保存当前裁剪参数到文件"""
        preset = self.crop_presets[self.current_preset]
        filename = f"crop_params_{self.current_preset}.txt"
        
        with open(filename, 'w') as f:
            f.write(f"# 裁剪方案: {preset['name']}\n")
            f.write(f"# 预设编号: {self.current_preset}\n\n")
            f.write("IMAGE_CROP = {\n")
            
            for cam_name in self.cameras.keys():
                # 尝试获取lambda的源码（仅作参考）
                f.write(f'    "{cam_name}": lambda img: img,  # 请根据实际效果手动填写\n')
            
            f.write("}\n")
        
        print(f"✓ 裁剪参数已保存到: {filename}")
    
    def run(self):
        """运行可视化循环"""
        self.init_cameras()
        
        print("\n" + "="*60)
        print("相机图像裁剪可视化工具")
        print("="*60)
        print("\n键盘控制:")
        print("  1-9: 切换裁剪预设方案")
        print("  0: 显示原始图像")
        print("  s: 保存当前裁剪参数")
        print("  q: 退出")
        print("="*60 + "\n")
        
        while True:
            frames = {}
            cropped_frames = {}
            
            # 读取并裁剪所有相机图像
            for cam_name, cap in self.caps.items():
                try:
                    rgb = cap.read()
                    if rgb is None:
                        continue
                    
                    # 应用裁剪
                    cropped = self.apply_crop(rgb, cam_name)
                    
                    # 添加信息覆盖层
                    preset = self.crop_presets[self.current_preset]
                    crop_info = f"Preset {self.current_preset}: {preset['name']}"
                    
                    frames[cam_name] = self.add_info_overlay(rgb, cam_name, "Original")
                    cropped_frames[cam_name] = self.add_info_overlay(
                        cropped, cam_name, crop_info
                    )
                    
                except Exception as e:
                    print(f"读取失败 ({cam_name}): {e}")
                    continue
            
            if not frames:
                print("无法读取任何相机图像")
                break
            
            # 显示图像 - 并排显示原始和裁剪后的图像
            for cam_name in frames.keys():
                # 调整显示尺寸
                original = frames[cam_name]
                cropped = cropped_frames[cam_name]
                
                # 统一高度以便并排显示
                target_height = 480
                h_orig, w_orig = original.shape[:2]
                h_crop, w_crop = cropped.shape[:2]
                
                # 缩放原始图像
                scale_orig = target_height / h_orig
                display_orig = cv2.resize(original, 
                    (int(w_orig * scale_orig), target_height))
                
                # 缩放裁剪图像
                scale_crop = target_height / h_crop
                display_crop = cv2.resize(cropped, 
                    (int(w_crop * scale_crop), target_height))
                
                # 并排显示
                combined = np.hstack([display_orig, display_crop])
                
                window_name = f"{cam_name} - Original vs Cropped"
                cv2.imshow(window_name, combined)
            
            # 处理键盘输入
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                print("退出...")
                break
            elif key == ord('s'):
                self.save_crop_params()
            elif ord('0') <= key <= ord('9'):
                preset_num = key - ord('0')
                if preset_num in self.crop_presets:
                    self.current_preset = preset_num
                    preset = self.crop_presets[preset_num]
                    print(f"切换到预设 {preset_num}: {preset['name']}")
                else:
                    print(f"预设 {preset_num} 不存在")
        
        # 清理
        for cap in self.caps.values():
            cap.close()
        cv2.destroyAllWindows()
        print("已退出")


def main():
    try:
        visualizer = CropVisualizer()
        visualizer.run()
    except KeyboardInterrupt:
        print("\n用户中断")
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
