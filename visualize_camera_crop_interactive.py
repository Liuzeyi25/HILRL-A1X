#!/usr/bin/env python3
"""
交互式相机图像裁剪调试工具
支持实时调整裁剪参数

使用方法:
    python visualize_camera_crop_interactive.py

功能:
    - 使用滑块实时调整裁剪区域
    - 显示原始图像和裁剪后的图像
    - 自动生成 IMAGE_CROP 代码
"""

import cv2
import numpy as np
from franka_env.camera.rs_capture import RSCapture
from franka_env.camera.video_capture import VideoCapture


class InteractiveCropTool:
    def __init__(self, camera_config):
        self.camera_config = camera_config
        self.caps = {}
        self.crop_params = {}
        self.current_camera = None
        
        # 初始化裁剪参数
        for cam_name in camera_config.keys():
            # [top, bottom, left, right] (负数表示从末尾算起)
            self.crop_params[cam_name] = {
                'top': 0,
                'bottom': 0,
                'left': 0,
                'right': 0,
            }
    
    def init_cameras(self):
        """初始化相机"""
        print("正在初始化相机...")
        for cam_name, kwargs in self.camera_config.items():
            try:
                self.caps[cam_name] = VideoCapture(
                    RSCapture(name=cam_name, **kwargs)
                )
                print(f"✓ {cam_name} 初始化成功")
                if self.current_camera is None:
                    self.current_camera = cam_name
            except Exception as e:
                print(f"✗ {cam_name} 初始化失败: {e}")
        
        if not self.caps:
            raise RuntimeError("没有成功初始化任何相机")
    
    def create_trackbars(self, window_name):
        """创建滑块控制"""
        # 获取图像尺寸
        rgb = self.caps[self.current_camera].read()
        if rgb is None:
            return
        
        h, w = rgb.shape[:2]
        
        # 创建滑块
        cv2.createTrackbar('Top', window_name, 0, h//2, self.on_trackbar)
        cv2.createTrackbar('Bottom', window_name, 0, h//2, self.on_trackbar)
        cv2.createTrackbar('Left', window_name, 0, w//2, self.on_trackbar)
        cv2.createTrackbar('Right', window_name, 0, w//2, self.on_trackbar)
        
        print(f"\n调整 {self.current_camera} 的裁剪参数:")
        print(f"  原始尺寸: {w}x{h}")
    
    def on_trackbar(self, val):
        """滑块回调函数"""
        pass
    
    def get_current_crop_params(self, window_name):
        """从滑块获取当前裁剪参数"""
        top = cv2.getTrackbarPos('Top', window_name)
        bottom = cv2.getTrackbarPos('Bottom', window_name)
        left = cv2.getTrackbarPos('Left', window_name)
        right = cv2.getTrackbarPos('Right', window_name)
        
        self.crop_params[self.current_camera] = {
            'top': top,
            'bottom': bottom,
            'left': left,
            'right': right,
        }
        
        return top, bottom, left, right
    
    def apply_crop(self, image, top, bottom, left, right):
        """应用裁剪"""
        h, w = image.shape[:2]
        
        # 处理边界情况
        if top >= h - bottom:
            bottom = 0
        if left >= w - right:
            right = 0
        
        # 计算裁剪区域
        y1 = top
        y2 = h - bottom if bottom > 0 else h
        x1 = left
        x2 = w - right if right > 0 else w
        
        # 在原图上绘制裁剪框
        preview = image.copy()
        cv2.rectangle(preview, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # 裁剪图像
        cropped = image[y1:y2, x1:x2]
        
        return preview, cropped
    
    def generate_code(self):
        """生成 IMAGE_CROP 代码"""
        print("\n" + "="*60)
        print("生成的 IMAGE_CROP 配置代码:")
        print("="*60)
        print("\nIMAGE_CROP = {")
        
        for cam_name, params in self.crop_params.items():
            top = params['top']
            bottom = params['bottom']
            left = params['left']
            right = params['right']
            
            # 生成lambda表达式
            if top == 0 and bottom == 0 and left == 0 and right == 0:
                code = "lambda img: img"
            else:
                # 构建切片字符串
                y_slice = ""
                if top > 0:
                    y_slice = f"{top}:"
                elif bottom > 0:
                    y_slice = f":-{bottom}"
                else:
                    y_slice = ":"
                
                if bottom > 0 and top > 0:
                    y_slice = f"{top}:-{bottom}"
                
                x_slice = ""
                if left > 0:
                    x_slice = f"{left}:"
                elif right > 0:
                    x_slice = f":-{right}"
                else:
                    x_slice = ":"
                
                if right > 0 and left > 0:
                    x_slice = f"{left}:-{right}"
                
                code = f"lambda img: img[{y_slice}, {x_slice}]"
            
            print(f'    "{cam_name}": {code},')
        
        print("}")
        print("="*60)
    
    def save_code_to_file(self):
        """保存代码到文件"""
        filename = "image_crop_config.py"
        
        with open(filename, 'w') as f:
            f.write("# AUTO-GENERATED IMAGE_CROP Configuration\n")
            f.write("# Generated by visualize_camera_crop_interactive.py\n\n")
            f.write("IMAGE_CROP = {\n")
            
            for cam_name, params in self.crop_params.items():
                top = params['top']
                bottom = params['bottom']
                left = params['left']
                right = params['right']
                
                if top == 0 and bottom == 0 and left == 0 and right == 0:
                    code = "lambda img: img"
                else:
                    y_slice = ""
                    if top > 0 and bottom > 0:
                        y_slice = f"{top}:-{bottom}"
                    elif top > 0:
                        y_slice = f"{top}:"
                    elif bottom > 0:
                        y_slice = f":-{bottom}"
                    else:
                        y_slice = ":"
                    
                    x_slice = ""
                    if left > 0 and right > 0:
                        x_slice = f"{left}:-{right}"
                    elif left > 0:
                        x_slice = f"{left}:"
                    elif right > 0:
                        x_slice = f":-{right}"
                    else:
                        x_slice = ":"
                    
                    code = f"lambda img: img[{y_slice}, {x_slice}]"
                
                f.write(f'    "{cam_name}": {code},\n')
            
            f.write("}\n")
        
        print(f"\n✓ 代码已保存到: {filename}")
    
    def run(self):
        """运行交互式工具"""
        self.init_cameras()
        
        window_name = "Interactive Crop Tool"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 1400, 700)
        
        self.create_trackbars(window_name)
        
        print("\n" + "="*60)
        print("交互式相机图像裁剪工具")
        print("="*60)
        print("\n键盘控制:")
        print("  滑块: 实时调整裁剪区域")
        print("    Top: 从顶部裁剪的像素数")
        print("    Bottom: 从底部裁剪的像素数")
        print("    Left: 从左侧裁剪的像素数")
        print("    Right: 从右侧裁剪的像素数")
        print("  g: 生成并打印 IMAGE_CROP 代码")
        print("  s: 保存代码到文件")
        print("  c: 切换相机")
        print("  r: 重置当前相机的裁剪参数")
        print("  q: 退出")
        print("="*60 + "\n")
        
        while True:
            # 读取当前相机图像
            rgb = self.caps[self.current_camera].read()
            if rgb is None:
                continue
            
            # 获取裁剪参数
            top, bottom, left, right = self.get_current_crop_params(window_name)
            
            # 应用裁剪
            preview, cropped = self.apply_crop(rgb, top, bottom, left, right)
            
            # 添加信息文字
            h, w = rgb.shape[:2]
            ch, cw = cropped.shape[:2]
            
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(preview, f"Camera: {self.current_camera}", 
                       (10, 30), font, 1, (0, 255, 0), 2)
            cv2.putText(preview, f"Original: {w}x{h}", 
                       (10, 60), font, 0.7, (255, 255, 255), 2)
            cv2.putText(preview, f"Cropped: {cw}x{ch}", 
                       (10, 90), font, 0.7, (0, 255, 255), 2)
            cv2.putText(preview, f"T:{top} B:{bottom} L:{left} R:{right}", 
                       (10, 120), font, 0.6, (255, 255, 0), 2)
            
            # 调整显示尺寸
            target_height = 600
            scale_preview = target_height / h
            display_preview = cv2.resize(preview, 
                (int(w * scale_preview), target_height))
            
            if cropped.size > 0:
                ch, cw = cropped.shape[:2]
                scale_crop = target_height / ch if ch > 0 else 1
                display_crop = cv2.resize(cropped, 
                    (int(cw * scale_crop), target_height))
                
                # 并排显示
                combined = np.hstack([display_preview, display_crop])
            else:
                combined = display_preview
            
            cv2.imshow(window_name, combined)
            
            # 处理键盘输入
            key = cv2.waitKey(30) & 0xFF
            
            if key == ord('q'):
                print("退出...")
                break
            elif key == ord('g'):
                self.generate_code()
            elif key == ord('s'):
                self.generate_code()
                self.save_code_to_file()
            elif key == ord('c'):
                # 切换相机
                cam_names = list(self.caps.keys())
                idx = cam_names.index(self.current_camera)
                idx = (idx + 1) % len(cam_names)
                self.current_camera = cam_names[idx]
                
                # 重新创建滑块
                cv2.destroyWindow(window_name)
                cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
                cv2.resizeWindow(window_name, 1400, 700)
                self.create_trackbars(window_name)
                
                # 恢复保存的参数
                params = self.crop_params[self.current_camera]
                cv2.setTrackbarPos('Top', window_name, params['top'])
                cv2.setTrackbarPos('Bottom', window_name, params['bottom'])
                cv2.setTrackbarPos('Left', window_name, params['left'])
                cv2.setTrackbarPos('Right', window_name, params['right'])
                
                print(f"切换到相机: {self.current_camera}")
            elif key == ord('r'):
                # 重置参数
                cv2.setTrackbarPos('Top', window_name, 0)
                cv2.setTrackbarPos('Bottom', window_name, 0)
                cv2.setTrackbarPos('Left', window_name, 0)
                cv2.setTrackbarPos('Right', window_name, 0)
                print(f"已重置 {self.current_camera} 的裁剪参数")
        
        # 清理
        for cap in self.caps.values():
            cap.close()
        cv2.destroyAllWindows()
        
        # 最后生成一次代码
        print("\n最终配置:")
        self.generate_code()


def main():
    # 相机配置 (从 pour_water config.py 读取)
    # 只配置 side_policy_256 以便直接调整
    camera_config = {
        "side_policy_256": {
            "serial_number": "243222075799",
            "dim": (1280, 720),
            "exposure": 10500,
        },
    }
    
    try:
        tool = InteractiveCropTool(camera_config)
        tool.run()
    except KeyboardInterrupt:
        print("\n用户中断")
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
