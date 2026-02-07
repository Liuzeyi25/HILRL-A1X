"""
Example configuration demonstrating GelloIntervention usage.

This shows how to integrate Gello teleoperation into your environment
using the Wrapper pattern, just like SpacemouseIntervention.
"""

from franka_env.envs.wrappers import (
    Quat2EulerWrapper,
    GelloIntervention,  # NEW: Gello wrapper
    SpacemouseIntervention,
)
from franka_env.envs.relative_env import RelativeFrame
from franka_env.envs.franka_env import DefaultEnvConfig
from serl_launcher.wrappers.serl_obs_wrappers import SERLObsWrapper
from serl_launcher.wrappers.chunking import ChunkingWrapper


class GelloExampleConfig:
    """Example configuration using GelloIntervention."""
    
    # Gello device configuration
    gello_port = "/dev/serial/by-id/usb-FTDI_USB__-__Serial_Converter_FTA7NNNU-if00-port0"
    
    # Task parameters
    task_desc = "Pick up the banana"
    discount = 0.99
    reward_neg = False
    
    # Observation keys
    proprio_keys = ["tcp_pose", "tcp_vel", "q"]
    
    def get_environment(
        self, 
        fake_env=False, 
        save_video=False, 
        teleoperation_device="gello"  # NEW: Choose device
    ):
        """
        Create environment with optional teleoperation device.
        
        Args:
            fake_env: Whether to use fake environment
            save_video: Whether to save video recordings
            teleoperation_device: "gello", "spacemouse", or None
        """
        from franka_env.envs.franka_env import FrankaEnv
        
        # Base environment
        env = FrankaEnv(
            fake_env=fake_env, 
            save_video=save_video, 
            config=DefaultEnvConfig()
        )
        
        # Add teleoperation device
        if not fake_env:
            if teleoperation_device == "gello":
                print("🎮 Using Gello for teleoperation")
                env = GelloIntervention(
                    env, 
                    port=self.gello_port,
                    intervention_threshold=0.01  # Movement sensitivity
                )
            elif teleoperation_device == "spacemouse":
                print("🎮 Using SpaceMouse for teleoperation")
                env = SpacemouseIntervention(env)
            else:
                print("🤖 No teleoperation device (pure policy)")
        
        # Standard wrappers
        env = RelativeFrame(env)
        env = Quat2EulerWrapper(env)
        env = SERLObsWrapper(env, proprio_keys=self.proprio_keys)
        env = ChunkingWrapper(env, obs_horizon=2, act_exec_horizon=None)
        
        return env


# ============================================================================
# Usage Examples
# ============================================================================

def example_record_gello_demos():
    """Example: Record demonstrations using Gello."""
    import pickle as pkl
    import numpy as np
    import copy
    from tqdm import tqdm
    
    config = GelloExampleConfig()
    env = config.get_environment(
        fake_env=False, 
        teleoperation_device="gello"
    )
    
    print("\n" + "="*60)
    print("Recording Gello Demonstrations")
    print("="*60)
    print("📖 Instructions:")
    print("   - Move Gello to control the robot")
    print("   - Press gripper buttons to open/close")
    print("   - Robot will follow your movements")
    print("="*60 + "\n")
    
    transitions = []
    success_count = 0
    success_needed = 10
    
    obs, info = env.reset()
    trajectory = []
    
    pbar = tqdm(total=success_needed, desc="Successful demos")
    
    while success_count < success_needed:
        # Zero action (will be overridden by Gello if moved)
        actions = np.zeros(env.action_space.sample().shape)
        
        next_obs, rew, done, truncated, info = env.step(actions)
        
        # Check if Gello intervened
        if "intervene_action" in info:
            actions = info["intervene_action"]
            # print(f"✋ Gello intervention detected")
        
        transition = copy.deepcopy(dict(
            observations=obs,
            actions=actions,
            next_observations=next_obs,
            rewards=rew,
            masks=1.0 - done,
            dones=done,
        ))
        trajectory.append(transition)
        
        obs = next_obs
        
        if done:
            if info.get("succeed", False):
                print(f"\n✅ Episode succeeded!")
                for t in trajectory:
                    transitions.append(copy.deepcopy(t))
                success_count += 1
                pbar.update(1)
            else:
                print(f"\n❌ Episode failed")
            
            trajectory = []
            obs, info = env.reset()
    
    pbar.close()
    
    # Save demonstrations
    with open("gello_demos.pkl", "wb") as f:
        pkl.dump(transitions, f)
    print(f"\n🎉 Saved {len(transitions)} transitions to gello_demos.pkl")
    
    env.close()


def example_mixed_control():
    """Example: Switch between Gello and SpaceMouse."""
    import time
    
    config = GelloExampleConfig()
    
    print("\n" + "="*60)
    print("Mixed Control Demo")
    print("="*60)
    
    # Test Gello
    print("\n[1/2] Testing Gello control...")
    env_gello = config.get_environment(teleoperation_device="gello")
    obs, _ = env_gello.reset()
    
    for _ in range(50):
        action = env_gello.action_space.sample() * 0  # Zero action
        obs, rew, done, truncated, info = env_gello.step(action)
        if "intervene_action" in info:
            print("  ✋ Gello is controlling the robot")
        time.sleep(0.02)
    env_gello.close()
    
    # Test SpaceMouse
    print("\n[2/2] Testing SpaceMouse control...")
    env_spacemouse = config.get_environment(teleoperation_device="spacemouse")
    obs, _ = env_spacemouse.reset()
    
    for _ in range(50):
        action = env_spacemouse.action_space.sample() * 0
        obs, rew, done, truncated, info = env_spacemouse.step(action)
        if "intervene_action" in info:
            print("  ✋ SpaceMouse is controlling the robot")
        time.sleep(0.02)
    env_spacemouse.close()
    
    print("\n✅ Both devices work correctly!")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "record":
        example_record_gello_demos()
    elif len(sys.argv) > 1 and sys.argv[1] == "mixed":
        example_mixed_control()
    else:
        print("Usage:")
        print("  python gello_example_config.py record  # Record demos with Gello")
        print("  python gello_example_config.py mixed   # Test both devices")
