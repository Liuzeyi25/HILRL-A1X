#!/usr/bin/env python3
"""
Test script for Gello reset synchronization with A1X inverse mapping.

This script tests:
1. Environment reset with A1X robot
2. Inverse mapping from A1X joints to Gello joints
3. Smooth Gello movement to reset position
"""

import sys
import numpy as np
import time

# Add paths
sys.path.insert(0, 'Gello/gello_software')

from examples.experiments.a1x_pick_banana.config20260127 import get_env
from franka_env.envs.wrappers import GelloIntervention


def test_reset_sync():
    """Test reset synchronization with inverse mapping."""
    
    print("=" * 60)
    print("Testing Gello Reset Synchronization with A1X")
    print("=" * 60)
    
    # Create environment with Gello intervention
    print("\n📦 Creating environment...")
    env = get_env()
    
    # Wrap with Gello intervention
    gello_port = "/dev/serial/by-id/usb-FTDI_USB__-__Serial_Converter_FTA7NNNU-if00-port0"
    
    print(f"🎮 Initializing Gello on port: {gello_port}")
    env = GelloIntervention(
        env=env,
        gello_port=gello_port,
        sync_on_reset=True,  # Enable reset synchronization
        reset_follow_duration=3.0,  # 3 seconds for smooth movement
    )
    
    print("\n✅ Environment created successfully")
    
    # Test reset multiple times
    num_tests = 3
    
    for i in range(num_tests):
        print(f"\n{'=' * 60}")
        print(f"Reset Test {i+1}/{num_tests}")
        print(f"{'=' * 60}")
        
        print("\n🔄 Calling env.reset()...")
        obs, info = env.reset()
        
        print("\n✅ Reset completed!")
        print(f"Observation keys: {obs.keys() if isinstance(obs, dict) else 'array'}")
        
        if i < num_tests - 1:
            print("\n⏸️  Waiting 5 seconds before next reset...")
            time.sleep(5)
    
    print("\n" + "=" * 60)
    print("✅ All reset tests completed!")
    print("=" * 60)
    
    # Close environment
    print("\n🧹 Closing environment...")
    env.close()
    print("✅ Done!")


if __name__ == "__main__":
    test_reset_sync()
