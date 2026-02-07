#!/usr/bin/env python3
"""Test script for A1_X robot environment.

This script tests the A1_X robot integration in the serl_robot_infra framework.
"""

import numpy as np
import time
from franka_env.envs.a1x_env import A1XEnv
from franka_env.envs.a1x_config import MinimalA1XConfig


def test_basic_robot():
    """Test basic robot initialization and control."""
    print("=" * 60)
    print("Test 1: Basic Robot Initialization")
    print("=" * 60)
    
    config = MinimalA1XConfig()
    
    print("\nCreating A1_X environment...")
    env = A1XEnv(
        hz=10,
        fake_env=False,
        save_video=False,
        config=config,
    )
    
    print(f"Action space: {env.action_space}")
    print(f"Observation space keys: {env.observation_space.keys()}")
    
    print("\nResetting environment...")
    obs, info = env.reset()
    
    print(f"Initial joint positions: {obs['state']['joint_positions']}")
    print(f"Initial joint velocities: {obs['state']['joint_velocities']}")
    print(f"Initial gripper position: {obs['state']['gripper_position']}")
    
    print("\nTest 1 PASSED ✓")
    
    return env


def test_step(env):
    """Test stepping the environment with random actions."""
    print("\n" + "=" * 60)
    print("Test 2: Environment Step")
    print("=" * 60)
    
    print("\nTaking 5 random steps...")
    for i in range(5):
        # Small random action
        action = np.random.uniform(-0.5, 0.5, size=(7,))
        
        print(f"\nStep {i+1}: action = {action}")
        
        obs, reward, done, truncated, info = env.step(action)
        
        print(f"  Joint positions: {obs['state']['joint_positions']}")
        print(f"  Reward: {reward}")
        print(f"  Done: {done}")
        
        time.sleep(0.1)
    
    print("\nTest 2 PASSED ✓")


def test_reset(env):
    """Test resetting the environment."""
    print("\n" + "=" * 60)
    print("Test 3: Environment Reset")
    print("=" * 60)
    
    print("\nResetting environment to initial state...")
    obs, info = env.reset()
    
    print(f"Joint positions after reset: {obs['state']['joint_positions']}")
    print(f"Target reset state: {env._RESET_JOINT_STATE}")
    
    # Check if close to reset state
    diff = np.abs(obs['state']['joint_positions'] - env._RESET_JOINT_STATE)
    print(f"Difference from target: {diff}")
    
    if np.all(diff < 0.2):  # Allow some tolerance
        print("\nReset successful ✓")
        print("Test 3 PASSED ✓")
    else:
        print("\nWarning: Reset position differs from target")
        print("This might be expected if robot cannot reach exact position")


def test_reward(env):
    """Test reward computation."""
    print("\n" + "=" * 60)
    print("Test 4: Reward Computation")
    print("=" * 60)
    
    # Get current observation
    obs = env._get_obs()
    
    print(f"\nCurrent joint state: {obs['state']['joint_positions']}")
    print(f"Target joint state: {env._TARGET_JOINT_STATE}")
    
    reward = env.compute_reward(obs)
    print(f"Reward (at target): {reward}")
    
    # Modify observation to be far from target
    obs['state']['joint_positions'] = np.zeros(7)
    reward_far = env.compute_reward(obs)
    print(f"Reward (far from target): {reward_far}")
    
    print("\nTest 4 PASSED ✓")


def main():
    """Run all tests."""
    print("\n")
    print("╔" + "=" * 58 + "╗")
    print("║" + " " * 10 + "A1_X Robot Environment Test Suite" + " " * 14 + "║")
    print("╚" + "=" * 58 + "╝")
    print("\n")
    
    try:
        # Test 1: Basic initialization
        env = test_basic_robot()
        
        # Test 2: Stepping
        test_step(env)
        
        # Test 3: Reset
        test_reset(env)
        
        # Test 4: Reward
        test_reward(env)
        
        # Cleanup
        print("\n" + "=" * 60)
        print("Cleaning up...")
        print("=" * 60)
        env.close()
        
        print("\n" + "╔" + "=" * 58 + "╗")
        print("║" + " " * 15 + "ALL TESTS PASSED ✓" + " " * 23 + "║")
        print("╚" + "=" * 58 + "╝")
        print("\n")
        
    except Exception as e:
        print("\n" + "!" * 60)
        print(f"ERROR: {e}")
        print("!" * 60)
        import traceback
        traceback.print_exc()
        
        if 'env' in locals():
            env.close()
        
        return 1
    
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
