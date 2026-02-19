#!/usr/bin/env python3
"""
示例：切换使用CuRobo IK或RelaxedIK

使用方法:
    # 使用RelaxedIK (默认)
    python switch_ik_example.py
    
    # 使用CuRobo IK
    python switch_ik_example.py --use-curobo-ik
"""

import sys
import os

# 添加项目路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

def modify_config(use_curobo_ik: bool):
    """修改config.py中的IK配置"""
    config_path = os.path.join(
        project_root,
        "examples/experiments/a1x_pick_banana/config.py"
    )
    
    with open(config_path, 'r') as f:
        content = f.read()
    
    # 替换USE_CUROBO_IK的值
    if use_curobo_ik:
        new_content = content.replace(
            "USE_CUROBO_IK = False",
            "USE_CUROBO_IK = True"
        )
        print("✅ 配置已修改: USE_CUROBO_IK = True")
        print("🚀 将使用 CuRobo IK (GPU加速)")
    else:
        new_content = content.replace(
            "USE_CUROBO_IK = True",
            "USE_CUROBO_IK = False"
        )
        print("✅ 配置已修改: USE_CUROBO_IK = False")
        print("🎯 将使用 RelaxedIK (默认)")
    
    with open(config_path, 'w') as f:
        f.write(new_content)
    
    print(f"\n配置文件: {config_path}")
    print("\n现在可以运行训练/推理脚本了!")

def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="切换A1X IK方案"
    )
    parser.add_argument(
        "--use-curobo-ik",
        action="store_true",
        help="使用CuRobo IK (默认使用RelaxedIK)"
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("A1X IK 配置切换工具")
    print("=" * 60)
    
    modify_config(args.use_curobo_ik)
    
    print("\n" + "=" * 60)
    print("📚 更多信息请查看: IK_SELECTION_GUIDE.md")
    print("=" * 60)

if __name__ == "__main__":
    main()
