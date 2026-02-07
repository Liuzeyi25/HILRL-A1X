#!/usr/bin/env python3
"""
测试 A1X 到 Gello 的映射函数

验证 _manual_a1x_to_gello_mapping 和 A1XRobot._map_from_a1x 是否一致
"""

import sys
from pathlib import Path

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / 'Gello' / 'gello_software'))

import numpy as np

def manual_a1x_to_gello_mapping(a1x_joints):
    """手动实现的映射函数（从 wrappers.py 复制）"""
    # Gello 关节范围
    gello_range_start = np.array([-2.87, 0.0, 0.0, -1.57, -1.34, -2.0, 0.103], dtype=float)
    gello_range_end = np.array([2.87, 3.14, 3.14, 1.57, 1.34, 2.0, 1.0], dtype=float)
    
    # A1X 关节范围
    a1x_range_start = np.array([-2.880, -0.001, 0.0, 1.55, 1.521, -1.56, 4.0], dtype=float)
    a1x_range_end = np.array([2.880, 3.14, 2.95, -1.55, -1.52, 1.56, 99.0], dtype=float)
    
    a1x_joints = np.array(a1x_joints, dtype=float)
    
    if len(a1x_joints) < 7:
        a1x_full = np.zeros(7)
        a1x_full[:len(a1x_joints)] = a1x_joints
        a1x_joints = a1x_full
    
    # Clip
    clipped = a1x_joints.copy()
    for i in range(7):
        lo = min(a1x_range_start[i], a1x_range_end[i])
        hi = max(a1x_range_start[i], a1x_range_end[i])
        clipped[i] = np.clip(a1x_joints[i], lo, hi)
    
    result = np.zeros(7, dtype=float)
    
    for i in range(7):
        out_start = a1x_range_start[i]
        out_end = a1x_range_end[i]
        in_start = gello_range_start[i]
        in_end = gello_range_end[i]
        
        out_range = out_end - out_start
        if abs(out_range) < 1e-9:
            result[i] = in_start
        else:
            t = (clipped[i] - out_start) / out_range
            result[i] = in_start + t * (in_end - in_start)
    
    return result


def test_mapping():
    """测试映射函数"""
    print("=" * 80)
    print("🔬 测试 A1X -> Gello 映射")
    print("=" * 80)
    
    # 尝试导入 A1XRobot
    try:
        from gello.robots.A1_X import A1XRobot
        has_a1x_robot = True
        print("✅ 成功导入 A1XRobot")
    except Exception as e:
        print(f"⚠️  无法导入 A1XRobot: {e}")
        has_a1x_robot = False
    
    # 测试用例
    test_cases = [
        # 标签, A1X 关节位置
        ("零位置", [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 50.0]),
        ("关节1 正向", [1.5, 0.0, 0.0, 0.0, 0.0, 0.0, 50.0]),
        ("关节1 负向", [-1.5, 0.0, 0.0, 0.0, 0.0, 0.0, 50.0]),
        ("关节4 正向 (A1X 范围内)", [0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 50.0]),
        ("关节4 负向 (A1X 范围内)", [0.0, 0.0, 0.0, -0.5, 0.0, 0.0, 50.0]),
        ("混合位置", [0.5, 1.0, 0.8, 0.3, -0.2, 0.5, 30.0]),
    ]
    
    print("\n" + "-" * 80)
    for name, a1x in test_cases:
        print(f"\n🧪 测试: {name}")
        print(f"   A1X 输入:      [{', '.join(f'{v:7.3f}' for v in a1x)}]")
        
        # 手动映射
        gello_manual = manual_a1x_to_gello_mapping(a1x)
        print(f"   手动映射结果:  [{', '.join(f'{v:7.3f}' for v in gello_manual)}]")
        
        # A1XRobot 映射（如果可用）
        if has_a1x_robot:
            try:
                temp_robot = A1XRobot.__new__(A1XRobot)
                gello_a1x = temp_robot._map_from_a1x(np.array(a1x))
                print(f"   A1XRobot 结果: [{', '.join(f'{v:7.3f}' for v in gello_a1x)}]")
                
                # 比较差异
                diff = np.abs(gello_manual - gello_a1x)
                max_diff = np.max(diff)
                if max_diff < 1e-6:
                    print(f"   ✅ 两种方法结果一致")
                else:
                    print(f"   ⚠️  差异: [{', '.join(f'{v:7.6f}' for v in diff)}] (最大: {max_diff:.6f})")
            except Exception as e:
                print(f"   ⚠️  A1XRobot 映射失败: {e}")
    
    print("\n" + "=" * 80)
    print("📝 关节范围参考:")
    print("=" * 80)
    gello_range_start = np.array([-2.87, 0.0, 0.0, -1.57, -1.34, -2.0, 0.103])
    gello_range_end = np.array([2.87, 3.14, 3.14, 1.57, 1.34, 2.0, 1.0])
    a1x_range_start = np.array([-2.880, -0.001, 0.0, 1.55, 1.521, -1.56, 4.0])
    a1x_range_end = np.array([2.880, 3.14, 2.95, -1.55, -1.52, 1.56, 99.0])
    
    print("\n关节  |    Gello 范围    |     A1X 范围")
    print("-" * 50)
    for i in range(7):
        print(f"  {i}   | [{gello_range_start[i]:6.3f}, {gello_range_end[i]:6.3f}] | [{a1x_range_start[i]:6.3f}, {a1x_range_end[i]:6.3f}]")
    
    print("\n💡 注意：关节4、5 的 A1X 范围是反向的 (start > end)")


if __name__ == "__main__":
    test_mapping()
