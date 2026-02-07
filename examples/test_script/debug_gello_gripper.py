#!/usr/bin/env python3
"""
Debug script to check what Gello actually returns.
"""

import sys
import time
import numpy as np

# Add Gello to path
sys.path.insert(0, '../Gello/gello_software')

from gello.agents.gello_agent import GelloAgent, DynamixelRobotConfig

def main():
    print("="*60)
    print("Gello Gripper Debug Script")
    print("="*60)
    
    # Initialize Gello
    port = "/dev/serial/by-id/usb-FTDI_USB__-__Serial_Converter_FTA7NNNU-if00-port0"
    
    config = DynamixelRobotConfig(
        joint_ids=[1, 2, 3, 4, 5, 6],
        joint_offsets=[1.5708, 4.71239, 4.71239, 3.14159, 1.5708, 3.14159],
        joint_signs=[1.0, -1.0, -1.0, -1.0, 1.0, 1.0],
        gripper_config=[7, 139.66015625, 199.16015625]  # ID=7, open pos, close pos
    )
    
    print(f"\n📡 Connecting to Gello on {port}...")
    agent = GelloAgent(port=port, dynamixel_config=config)
    robot = agent._robot
    
    print("✅ Gello connected!")
    print(f"   Number of DOFs: {robot.num_dofs()}")
    
    print("\n" + "="*60)
    print("Reading Gello State (press Ctrl+C to stop)")
    print("="*60)
    print("\nTry pressing the gripper buttons and moving Gello...")
    print()
    
    try:
        while True:
            # Get joint state from robot directly
            joint_state = robot.get_joint_state()
            
            # Get action from agent
            action = agent.act({})
            
            print(f"\r🤖 Joint state length: {len(joint_state)}, values: {joint_state}", end='')
            print(f"  | Action length: {len(action)}", end='', flush=True)
            
            time.sleep(0.1)
            
    except KeyboardInterrupt:
        print("\n\n✅ Stopping...")
    finally:
        agent.close()
        print("🔌 Gello closed.")

if __name__ == "__main__":
    main()
