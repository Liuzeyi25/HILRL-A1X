"""
Reverse teleoperation: Make Gello follow A1_X robot arm.

This allows the Gello leader arm to mirror the A1_X robot's movements,
useful for demonstrations or haptic feedback during autonomous exploration.
"""

import numpy as np
import time
from typing import Optional

from gello.robots.dynamixel import DynamixelRobot
from gello.dynamixel.driver import DynamixelDriver, POSITION_CONTROL_MODE


class GelloFollower:
    """
    Makes Gello arm follow another robot (reverse teleoperation).
    
    Usage:
        follower = GelloFollower(gello_robot)
        follower.start()  # Enable position control
        
        # In your control loop:
        a1x_joints = a1x_robot.get_joint_state()
        follower.command_follow(a1x_joints)
        
        follower.stop()  # Return to free-wheeling mode
    """
    
    def __init__(
        self,
        gello_robot: DynamixelRobot,
        compliance_margin: float = 0.0,
        compliance_slope: float = 32.0,
    ):
        """
        Args:
            gello_robot: The Gello DynamixelRobot instance
            compliance_margin: Compliance margin (0-254, lower = stiffer)
            compliance_slope: Compliance slope (1-254, lower = stiffer)
        """
        self.gello = gello_robot
        self.driver: DynamixelDriver = gello_robot._driver
        self.is_following = False
        
        # Compliance parameters for smooth following
        self.compliance_margin = compliance_margin
        self.compliance_slope = compliance_slope
        
        # Joint limits for safety
        self.joint_limits_low = np.array([-np.pi, -np.pi, -np.pi, 
                                          -np.pi, -np.pi, -np.pi, 0.0])
        self.joint_limits_high = np.array([np.pi, np.pi, np.pi, 
                                           np.pi, np.pi, np.pi, 1.0])
    
    def start(self):
        """Enable position control mode for following."""
        print("[GelloFollower] Switching to position control mode...")
        
        # Read current position before switching
        current_pos = self.gello.get_joint_state()
        print(f"[GelloFollower] Current Gello position: {current_pos}")
        
        # Check if already in position control mode
        try:
            current_mode = self.driver.verify_operating_mode(POSITION_CONTROL_MODE)
            print("[GelloFollower] Already in position control mode, skipping mode change")
            # Just ensure torque is on
            self.driver.set_torque_mode(True)
            time.sleep(0.5)
        except:
            # Need to switch modes
            print("[GelloFollower] Switching from current control to position control...")
            
            # Switch to position control mode
            self.driver.set_torque_mode(False)  # Must disable torque first
            time.sleep(0.2)
            
            self.driver.set_operating_mode(POSITION_CONTROL_MODE)  # Mode 3
            time.sleep(0.3)
            
            # Enable torque
            self.driver.set_torque_mode(True)
            time.sleep(0.5)
        
        self.is_following = True
        print("[GelloFollower] Position control enabled. Gello will follow commands.")
    
    def stop(self):
        """Return to free-wheeling mode (normal teleoperation)."""
        print("[GelloFollower] Returning to free-wheeling mode...")
        
        self.driver.set_torque_mode(False)
        self.is_following = False
        
        print("[GelloFollower] Free-wheeling mode enabled. You can manually move Gello.")
    
    def command_follow(self, target_joints: np.ndarray):
        """
        Command Gello to follow target joint positions.
        
        Args:
            target_joints: Target joint positions [7] (6 joints + gripper)
        """
        if not self.is_following:
            print("[GelloFollower] Warning: Not in following mode. Call start() first.")
            return
        
        # Safety: Clip to joint limits
        target_joints = np.clip(target_joints, 
                                self.joint_limits_low, 
                                self.joint_limits_high)
        
        # Command Gello to move
        self.gello.command_joint_state(target_joints)
    
    def get_current_position(self) -> np.ndarray:
        """Get Gello's current joint positions."""
        return self.gello.get_joint_state()
    
    def _set_compliance(self, margin: float, slope: float):
        """Set compliance parameters (model-specific)."""
        # This is for older Dynamixel models with compliance settings
        # XC330/XM430 use PID gains instead
        # You may need to adjust PID gains for smoother following:
        # ADDR_POSITION_P_GAIN = 84
        # ADDR_POSITION_I_GAIN = 82
        # ADDR_POSITION_D_GAIN = 80
        pass


# ============================================================================
# Integration with A1_X Training
# ============================================================================

def example_usage_with_a1x():
    """Example: Make Gello follow A1_X during autonomous exploration."""
    from franka_env.robots.a1x_robot import A1XRobot
    from gello.agents.gello_agent import GelloAgent
    
    # Initialize robots
    print("Initializing A1_X robot...")
    a1x = A1XRobot(num_dofs=7, port=6100)
    
    print("Initializing Gello...")
    gello_agent = GelloAgent(
        port="/dev/serial/by-id/usb-FTDI_USB__-__Serial_Converter_FTA7NNNU-if00-port0"
    )
    gello_robot = gello_agent._robot
    
    # Create follower
    follower = GelloFollower(gello_robot)
    
    # Enable following mode
    follower.start()
    
    try:
        # Simulation: A1_X autonomous exploration
        print("\\nA1_X is exploring autonomously. Gello will follow...")
        
        for step in range(100):
            # Get A1_X current joint state
            a1x_joints = a1x.get_joint_state()
            
            # Make Gello follow
            follower.command_follow(a1x_joints)
            
            # Control loop
            time.sleep(0.02)  # 50Hz
            
            if step % 25 == 0:
                print(f"Step {step}: A1_X joints = {a1x_joints[:3]}...")
        
        print("\\nExploration complete!")
        
    finally:
        # Always return to free-wheeling mode
        follower.stop()
        a1x.close()
        print("\\nGello returned to teleoperation mode.")


def example_usage_with_serl_training():
    """
    Example: Integrate Gello follower into SERL training loop.
    
    During autonomous exploration, Gello mirrors the robot,
    allowing human to feel the robot's movements or take over if needed.
    """
    from experiments.a1x_pick_banana.config import TrainConfig
    
    # Setup
    config = TrainConfig()
    env = config.get_environment(fake_env=False)
    
    # Initialize Gello follower
    from gello.agents.gello_agent import GelloAgent
    gello = GelloAgent(port="/dev/serial/by-id/usb-FTDI_USB__-__Serial_Converter_FTA7NNNU-if00-port0")
    follower = GelloFollower(gello._robot)
    
    # Enable following during autonomous episodes
    follower.start()
    
    try:
        obs, _ = env.reset()
        
        for step in range(100):
            # Policy selects action
            action = env.action_space.sample()  # Replace with policy
            
            # Execute action
            obs, reward, done, truncated, info = env.step(action)
            
            # Get robot's current joint state
            robot_joints = env.unwrapped.curr_joint_positions
            
            # Make Gello follow (with human override capability)
            if not info.get("human_intervention", False):
                follower.command_follow(robot_joints)
            
            if done:
                break
        
    finally:
        follower.stop()
        env.close()


if __name__ == "__main__":
    print("Gello Follower - Reverse Teleoperation")
    print("=" * 60)
    print("\\nThis module allows Gello to follow A1_X robot movements.")
    print("\\nUsage:")
    print("  1. Normal teleoperation: Gello in free-wheeling mode")
    print("  2. Autonomous exploration: Gello follows robot")
    print("  3. Switch back: Return to free-wheeling for manual control")
    print("\\n" + "=" * 60)
    
    # Run example
    example_usage_with_a1x()
