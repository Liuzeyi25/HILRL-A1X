#!/usr/bin/env python3
"""
Reset A1_X robot to zero position.

This script moves the A1_X robot to a predefined home/zero position.
Useful for initialization or emergency reset.

Usage:
    python reset_a1x.py
"""

import sys
import time
import numpy as np

# Add parent directory to path
sys.path.insert(0, '.')

from a1x_robot import A1XRobot


# Define the zero/home position for A1_X
ZERO_POSITION = np.array([
    0.09702127659574468,   # Joint 1
    0.4621276595744681,    # Joint 2
    0.006595744680851064,  # Joint 3
    -0.08085106382978724,  # Joint 4
    -0.11,                 # Joint 5
    -0.017659574468085106, # Joint 6
    100.0                  # Gripper (fully open, 100mm)
])


def smooth_move_to_position(robot: A1XRobot, target_position: np.ndarray, duration: float = 3.0):
    """
    Smoothly move robot to target position with linear interpolation.
    
    Args:
        robot: A1XRobot instance
        target_position: Target joint positions [7]
        duration: Time to complete the movement (seconds)
    """
    # Get current position
    current_position = robot.get_joint_state()
    
    # Calculate number of steps (10 Hz control rate)
    control_hz = 10
    num_steps = int(duration * control_hz)
    
    print(f"📍 Current position: {current_position}")
    print(f"🎯 Target position:  {target_position}")
    print(f"⏱️  Moving in {num_steps} steps over {duration}s...")
    
    # Generate smooth trajectory
    for i in range(num_steps + 1):
        t = i / num_steps  # 0.0 to 1.0
        
        # Smooth interpolation (ease-in-out cubic)
        t_smooth = 3 * t**2 - 2 * t**3
        
        # Interpolate position
        interpolated_position = current_position + t_smooth * (target_position - current_position)
        
        # Command robot (from_gello=False because these are native A1X positions)
        robot.command_joint_state(interpolated_position, from_gello=False)
        
        # Progress indicator
        if i % max(1, (num_steps // 10)) == 0:
            progress = int(t * 100)
            print(f"  Progress: {progress}%", end='\r')
        
        time.sleep(1.0 / control_hz)
    
    print(f"  Progress: 100% ✓")
    
    # Wait for robot to settle
    time.sleep(0.5)
    
    # Verify final position
    final_position = robot.get_joint_state()
    position_error = np.abs(final_position - target_position)
    
    print(f"\n📊 Final position: {final_position}")
    print(f"📏 Position error: {position_error}")
    print(f"   Max error: {np.max(position_error):.4f} rad")
    
    if np.max(position_error) > 0.1:
        print(f"⚠️  Warning: Large position error detected!")
    else:
        print(f"✅ Successfully reached target position!")


def main():
    print("=" * 70)
    print("A1_X Robot Reset Script")
    print("=" * 70)
    print("\nThis script will move the A1_X robot to the zero/home position:")
    print(f"  Joint positions: {ZERO_POSITION[:6]}")
    print(f"  Gripper: {ZERO_POSITION[6]} mm (open)")
    print()
    
    # Confirm before proceeding
    response = input("⚠️  Make sure the path is clear. Continue? (y/n): ")
    if response.lower() != 'y':
        print("❌ Aborted by user.")
        return
    
    print("\n🤖 Initializing A1_X robot...")
    try:
        robot = A1XRobot(
            num_dofs=7,
            node_name="a1x_reset_node",
            port=6100,
            python_path="/usr/bin/python3"
        )
    except Exception as e:
        print(f"❌ Failed to initialize robot: {e}")
        return
    
    print("✅ Robot initialized successfully")
    
    try:
        print("\n🔄 Moving to zero position...")
        smooth_move_to_position(robot, ZERO_POSITION, duration=3.0)
        
        print("\n✅ Reset complete!")
        
    except KeyboardInterrupt:
        print("\n⚠️  Interrupted by user (Ctrl+C)")
    except Exception as e:
        print(f"\n❌ Error during reset: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("\n🧹 Closing robot connection...")
        robot.close()
        print("✅ Done!")


if __name__ == "__main__":
    main()
