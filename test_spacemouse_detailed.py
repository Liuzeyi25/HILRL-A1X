#!/usr/bin/env python3
"""
详细的空间鼠标测试脚本
测试 SpaceMouse 的所有功能，包括:
- 6自由度运动 (x, y, z, roll, pitch, yaw)
- 按钮状态
- 数据刷新率
- 数值范围统计
"""

import time
import numpy as np
import sys
from collections import deque

# 添加路径以导入模块
sys.path.insert(0, '/home/dungeon_master/conrft/serl_robot_infra')
from franka_env.spacemouse.spacemouse_expert import SpaceMouseExpert


class SpaceMouseTester:
    """空间鼠标测试类"""
    
    def __init__(self):
        print("=" * 60)
        print("空间鼠标测试程序")
        print("=" * 60)
        print("初始化空间鼠标...")
        
        try:
            self.spacemouse = SpaceMouseExpert()
            print("✓ 空间鼠标初始化成功!")
        except Exception as e:
            print(f"✗ 空间鼠标初始化失败: {e}")
            sys.exit(1)
        
        # 统计数据
        self.action_history = deque(maxlen=100)
        self.button_history = deque(maxlen=100)
        self.last_time = time.time()
        self.frame_count = 0
        
        # 最大最小值跟踪
        self.max_values = np.zeros(6)
        self.min_values = np.zeros(6)
        
    def print_header(self):
        """打印表头"""
        print("\n" + "-" * 60)
        print(f"{'轴':<8} {'当前值':>10} {'最小值':>10} {'最大值':>10}")
        print("-" * 60)
        
    def print_action_details(self, action):
        """打印详细的动作信息"""
        labels = ['X', 'Y', 'Z', 'Roll', 'Pitch', 'Yaw']
        
        for i, label in enumerate(labels[:len(action)]):
            val = action[i]
            # 更新最大最小值
            self.max_values[i] = max(self.max_values[i], val)
            self.min_values[i] = min(self.min_values[i], val)
            
            print(f"{label:<8} {val:>10.4f} {self.min_values[i]:>10.4f} {self.max_values[i]:>10.4f}")
    
    def test_basic(self, duration=None):
        """
        基础测试: 实时显示空间鼠标的数据
        
        Args:
            duration: 测试持续时间(秒), None表示持续运行直到按Ctrl+C
        """
        print("\n" + "=" * 60)
        print("基础测试模式")
        print("=" * 60)
        print("提示:")
        print("  - 移动空间鼠标查看6自由度数值变化")
        print("  - 按下空间鼠标的按钮查看按钮状态")
        print("  - 按 Ctrl+C 停止测试")
        if duration:
            print(f"  - 测试将运行 {duration} 秒")
        print()
        
        start_time = time.time()
        
        try:
            with np.printoptions(precision=4, suppress=True):
                while True:
                    # 检查是否超时
                    if duration and (time.time() - start_time > duration):
                        break
                    
                    action, buttons = self.spacemouse.get_action()
                    self.frame_count += 1
                    
                    # 清屏(可选)
                    # print("\033[2J\033[H", end="")
                    
                    # 计算FPS
                    current_time = time.time()
                    elapsed = current_time - self.last_time
                    if elapsed >= 1.0:
                        fps = self.frame_count / elapsed
                        self.frame_count = 0
                        self.last_time = current_time
                        print(f"\n刷新率: {fps:.1f} Hz")
                    
                    # 显示数据
                    self.print_header()
                    self.print_action_details(action)
                    print("-" * 60)
                    print(f"按钮状态: {buttons}")
                    print(f"动作向量: {action}")
                    
                    # 检测活动状态
                    if np.any(np.abs(action) > 0.01):
                        print("状态: 🟢 检测到运动")
                    else:
                        print("状态: 🔵 静止")
                    
                    if any(buttons):
                        print(f"按钮: 🔴 按钮 {buttons} 被按下")
                    
                    time.sleep(0.1)  # 10Hz更新
                    
        except KeyboardInterrupt:
            print("\n\n测试被用户中断")
        
        print("\n测试结束")
    
    def test_calibration(self, samples=100):
        """
        校准测试: 检测空间鼠标的零点和噪声水平
        
        Args:
            samples: 采样数量
        """
        print("\n" + "=" * 60)
        print("校准测试模式")
        print("=" * 60)
        print(f"提示: 请不要触碰空间鼠标，将采集 {samples} 个样本...")
        print()
        
        data = []
        for i in range(samples):
            action, _ = self.spacemouse.get_action()
            data.append(action)
            if (i + 1) % 10 == 0:
                print(f"进度: {i + 1}/{samples}")
            time.sleep(0.05)
        
        data = np.array(data)
        
        print("\n校准结果:")
        print("-" * 60)
        labels = ['X', 'Y', 'Z', 'Roll', 'Pitch', 'Yaw']
        for i, label in enumerate(labels[:data.shape[1]]):
            mean = np.mean(data[:, i])
            std = np.std(data[:, i])
            max_val = np.max(np.abs(data[:, i]))
            print(f"{label:<8} - 均值: {mean:>8.5f}, 标准差: {std:>8.5f}, 最大偏移: {max_val:>8.5f}")
        
        print("\n建议:")
        max_noise = np.max(np.std(data, axis=0))
        if max_noise < 0.01:
            print("✓ 噪声水平低，零点稳定")
        elif max_noise < 0.05:
            print("⚠ 噪声水平中等，可能需要死区滤波")
        else:
            print("✗ 噪声水平高，建议检查设备连接")
    
    def test_range(self, duration=30):
        """
        量程测试: 测量空间鼠标各轴的最大量程
        
        Args:
            duration: 测试持续时间(秒)
        """
        print("\n" + "=" * 60)
        print("量程测试模式")
        print("=" * 60)
        print(f"提示: 请在 {duration} 秒内尽可能地移动和旋转空间鼠标到各个极限位置")
        print("      包括: 推/拉, 左/右, 上/下, 以及各个方向的旋转")
        print()
        
        input("按 Enter 开始...")
        
        start_time = time.time()
        max_actions = np.zeros(6)
        min_actions = np.zeros(6)
        
        print("测试进行中...")
        try:
            while time.time() - start_time < duration:
                action, _ = self.spacemouse.get_action()
                max_actions = np.maximum(max_actions, action[:6])
                min_actions = np.minimum(min_actions, action[:6])
                
                remaining = duration - (time.time() - start_time)
                print(f"\r剩余时间: {remaining:.1f}s | 当前值: {action[:6]}", end="")
                time.sleep(0.05)
        except KeyboardInterrupt:
            print("\n测试被用户中断")
        
        print("\n\n量程测试结果:")
        print("-" * 60)
        labels = ['X', 'Y', 'Z', 'Roll', 'Pitch', 'Yaw']
        for i, label in enumerate(labels):
            range_val = max_actions[i] - min_actions[i]
            print(f"{label:<8} - 最小: {min_actions[i]:>8.4f}, 最大: {max_actions[i]:>8.4f}, 量程: {range_val:>8.4f}")
    
    def test_buttons(self, duration=None):
        """
        按钮测试: 测试所有按钮
        
        Args:
            duration: 测试持续时间(秒), None表示持续运行直到按Ctrl+C
        """
        print("\n" + "=" * 60)
        print("按钮测试模式")
        print("=" * 60)
        print("提示: 请依次按下空间鼠标上的所有按钮")
        print("      按 Ctrl+C 停止测试")
        if duration:
            print(f"      测试将运行 {duration} 秒")
        print()
        
        pressed_buttons = set()
        start_time = time.time()
        
        try:
            while True:
                if duration and (time.time() - start_time > duration):
                    break
                
                _, buttons = self.spacemouse.get_action()
                
                for i, btn in enumerate(buttons):
                    if btn == 1:
                        if i not in pressed_buttons:
                            pressed_buttons.add(i)
                            print(f"✓ 按钮 {i} 被按下 (共检测到 {len(pressed_buttons)} 个按钮)")
                
                time.sleep(0.05)
        except KeyboardInterrupt:
            print("\n测试被用户中断")
        
        print(f"\n按钮测试结果: 检测到 {len(pressed_buttons)} 个按钮")
        if pressed_buttons:
            print(f"按钮索引: {sorted(pressed_buttons)}")
    
    def run_interactive(self):
        """交互式测试菜单"""
        while True:
            print("\n" + "=" * 60)
            print("空间鼠标测试菜单")
            print("=" * 60)
            print("1. 基础测试 - 实时显示数据")
            print("2. 校准测试 - 检测零点和噪声")
            print("3. 量程测试 - 测量最大量程")
            print("4. 按钮测试 - 测试所有按钮")
            print("5. 快速测试 - 运行所有测试")
            print("0. 退出")
            print("=" * 60)
            
            try:
                choice = input("请选择测试 (0-5): ").strip()
                
                if choice == '0':
                    print("退出程序...")
                    break
                elif choice == '1':
                    self.test_basic()
                elif choice == '2':
                    self.test_calibration()
                elif choice == '3':
                    self.test_range()
                elif choice == '4':
                    self.test_buttons()
                elif choice == '5':
                    print("\n运行快速测试套件...")
                    self.test_calibration(samples=50)
                    self.test_buttons(duration=10)
                    self.test_range(duration=15)
                else:
                    print("无效选择，请重试")
            except KeyboardInterrupt:
                print("\n\n测试被中断")
                break
    
    def cleanup(self):
        """清理资源"""
        print("\n清理资源...")
        try:
            self.spacemouse.close()
            print("✓ 空间鼠标已关闭")
        except:
            pass


def main():
    """主函数"""
    tester = SpaceMouseTester()
    
    try:
        # 如果提供了命令行参数，直接运行对应测试
        if len(sys.argv) > 1:
            mode = sys.argv[1]
            if mode == 'basic':
                tester.test_basic()
            elif mode == 'calibration':
                tester.test_calibration()
            elif mode == 'range':
                tester.test_range()
            elif mode == 'buttons':
                tester.test_buttons()
            else:
                print(f"未知模式: {mode}")
                print("可用模式: basic, calibration, range, buttons")
        else:
            # 交互式模式
            tester.run_interactive()
    finally:
        tester.cleanup()


if __name__ == "__main__":
    main()
