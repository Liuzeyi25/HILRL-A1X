#!/usr/bin/env python3
"""
Test Gello command_joint_state to debug the follow mode failure.
"""

import sys
import time
import numpy as np

sys.path.insert(0, '.')
sys.path.insert(0, 'Gello/gello_software')

from gello.agents.gello_agent import GelloAgent, DynamixelRobotConfig
from gello.agents.gello_follower import GelloFollower


def main():
    print("=" * 70)
    print("Gello Command Test")
    print("=" * 70)
    
    # Initialize Gello
    print("\n🎮 Initializing Gello...")
    gello_config = DynamixelRobotConfig(
        joint_ids=[1, 2, 3, 4, 5, 6],
        joint_offsets=[1.5708, 4.71239, 4.71239, 3.14159, 1.5708, 3.14159],
        joint_signs=[1.0, -1.0, -1.0, -1.0, 1.0, 1.0],
        gripper_config=[7, 139.66015625, 199.16015625]
    )
    
    gello = GelloAgent(
        port="/dev/serial/by-id/usb-FTDI_USB__-__Serial_Converter_FTA7NNNU-if00-port0",
        dynamixel_config=gello_config
    )
    print("✅ Gello initialized")
    
    # Read current position
    print("\n📊 Reading current Gello position...")
    current_pos = gello.act({})
    print(f"   Current: {current_pos}")
    print(f"   Joints (rad): {current_pos[:6]}")
    print(f"   Gripper (normalized): {current_pos[-1]}")
    
    # Test round-trip: read position, then command same position
    print("\n🔄 Test 1: Command same position (should not move)")
    print("   Enabling GelloFollower...")
    
    follower = GelloFollower(gello._robot)
    follower.start()
    
    time.sleep(1.0)
    
    print(f"   Commanding position: {current_pos}")
    try:
        follower.command_follow(current_pos)
        print("   ✅ Command successful!")
        
        # Read position again
        time.sleep(0.5)
        new_pos = gello.act({})
        diff = np.abs(new_pos[:6] - current_pos[:6])
        max_diff = np.max(diff)
        print(f"   Position after command: {new_pos}")
        print(f"   Max movement: {np.rad2deg(max_diff):.2f}°")
        
        if max_diff < 0.05:  # Less than 2.9 degrees
            print("   ✅ Position stable (good!)")
        else:
            print(f"   ⚠️  Position changed by {np.rad2deg(max_diff):.1f}°")
            
    except Exception as e:
        print(f"   ❌ Command failed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        follower.stop()
    
    print("\n👋 Test complete")


if __name__ == "__main__":
    main()
