#!/usr/bin/env python3
"""
Simple reset script for A1_X robot.
Can be run from anywhere in the project.
"""

import sys
import os

# Add project paths
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'serl_robot_infra'))

import time
import numpy as np
from franka_env.robots.a1x_robot import A1XRobot


# Zero position configuration
ZERO_POSITION = np.array([
    0.09702127659574468,   # Joint 1
    0.4621276595744681,    # Joint 2
    0.006595744680851064,  # Joint 3
    -0.08085106382978724,  # Joint 4
    -0.11,                 # Joint 5
    -0.017659574468085106, # Joint 6
    100.0                  # Gripper (100mm = fully open)
])

RESET_POSITION = np.array([-0.12, 2.187, -1.1474, 0.927, -0.009, -0.125, 100.0])

SELECT_POSITION = {
    "reset": RESET_POSITION,
    "zero": ZERO_POSITION
}


def main():
    print("=" * 60)
    print("A1_X Robot Reset Script")
    print("=" * 60)
    
    # Let user select position
    print("\n📍 Available reset positions:")
    print("  1. ZERO position   - Safe home position")
    print("  2. RESET position  - Task starting position")
    
    while True:
        choice = input("\nSelect position (1 or 2): ").strip()
        if choice == "1":
            selected_name = "ZERO"
            target_position = ZERO_POSITION
            break
        elif choice == "2":
            selected_name = "RESET"
            target_position = RESET_POSITION
            break
        else:
            print("❌ Invalid choice. Please enter 1 or 2.")
    
    print(f"\n✅ Selected: {selected_name} position")
    print(f"\nTarget joint positions:")
    for i, pos in enumerate(target_position[:6]):
        print(f"  Joint {i+1}: {pos:8.5f} rad")
    print(f"  Gripper: {target_position[6]:8.1f} mm")
    
    print("\n⚠️  Ensure the robot path is clear!")
    response = input("Continue? (y/n): ").strip().lower()
    if response != 'y':
        print("❌ Aborted by user.")
        return
    
    # Initialize robot
    print("\n🤖 Connecting to A1_X robot...")
    robot = A1XRobot(num_dofs=7, port=6100)
    
    # Get current position
    current = robot.get_joint_state()
    print(f"\n📍 Current position: {current}")

    
    # Calculate movement
    delta = np.abs(target_position - current)
    print(f"📏 Maximum joint movement: {np.max(delta[:6]):.3f} rad")
    
    # Smooth movement with 10 Hz control
    duration = 3.0
    steps = int(duration * 10)
    
    print(f"\n🔄 Moving to {selected_name} position ({duration}s)...")
    
    for i in range(steps + 1):
        t = i / steps
        # Ease-in-out interpolation
        t_smooth = 3 * t**2 - 2 * t**3
        
        position = current + t_smooth * (target_position - current)
        position[6] = target_position[6]  # Gripper directly to target
        robot.command_joint_state(position, from_gello=False)
        
        
        if i % 5 == 0:
            print(f"  {int(t*100):3d}%", end='\r')
        
        time.sleep(0.1)
    
    # Verify
    time.sleep(0.5)
    final = robot.get_joint_state()
    error = np.abs(final - target_position)
    
    print("  100% ✓")
    print(f"\n✅ Movement complete!")
    print(f"\n📊 Final joint positions:")
    for i in range(6):
        print(f"  Joint {i+1}: {final[i]:8.5f} rad (error: {error[i]:.5f})")
    print(f"  Gripper: {final[6]:8.1f} mm (target: {target_position[6]:.1f})")
    print(f"\n📏 Maximum position error: {np.max(error[:6]):.5f} rad")
    
    if np.max(error[:6]) > 0.05:
        print("⚠️  Warning: Position error is large!")
    else:
        print("✅ Position accuracy is good!")
    
    # Cleanup
    print("\n🧹 Closing connection...")
    robot.close()
    time.sleep(0.5)  # Wait for clean shutdown
    print("✅ Done!")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
