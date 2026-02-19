"""
Torque-based reverse teleoperation for Gello.

This approach uses current (torque) control to create a "virtual spring"
that pulls Gello toward the robot's position, providing natural haptic feedback.
"""

import numpy as np
import time
from typing import Optional

from gello.robots.dynamixel import DynamixelRobot
from gello.dynamixel.driver import DynamixelDriver, CURRENT_CONTROL_MODE


class GelloTorqueFollower:
    """
    Makes Gello follow robot using torque control (more natural feel).
    
    Instead of rigidly commanding positions, this uses "virtual springs"
    that gently pull Gello toward the robot's position.
    
    Benefits:
    - Human can still move Gello (feels like assistance, not force)
    - Smoother, more natural motion
    - Can detect human intervention (force feedback)
    """
    
    def __init__(
        self,
        gello_robot: DynamixelRobot,
        spring_stiffness: float = 0.5,
        damping: float = 0.1,
        max_current: float = 100.0,
    ):
        """
        Args:
            gello_robot: The Gello DynamixelRobot instance
            spring_stiffness: Virtual spring constant (0-1, higher = stiffer)
            damping: Damping coefficient (0-1, higher = more damped)
            max_current: Maximum current in mA (safety limit)
        """
        self.gello = gello_robot
        self.driver: DynamixelDriver = gello_robot._driver
        self.is_following = False
        
        # Control parameters
        self.spring_stiffness = spring_stiffness
        self.damping = damping
        self.max_current = max_current
        
        # State tracking
        self.prev_position = None
        self.prev_velocity = None
        self.target_position = None
    
    def start(self):
        """Enable torque control mode for following."""
        print("[GelloTorqueFollower] Switching to current control mode...")
        
        # Read current state
        self.prev_position = self.gello.get_joint_state()
        self.prev_velocity = np.zeros_like(self.prev_position)
        self.target_position = self.prev_position.copy()
        
        # Switch to current control mode
        self.driver.set_torque_mode(False)
        time.sleep(0.1)
        
        self.driver.set_operating_mode(CURRENT_CONTROL_MODE)  # Mode 0
        time.sleep(0.1)
        
        # Enable torque
        self.driver.set_torque_mode(True)
        time.sleep(0.1)
        
        self.is_following = True
        print("[GelloTorqueFollower] Torque control enabled.")
    
    def stop(self):
        """Return to free-wheeling mode."""
        print("[GelloTorqueFollower] Returning to free-wheeling mode...")
        self.driver.set_torque_mode(False)
        self.is_following = False
    
    def command_follow(
        self, 
        target_joints: np.ndarray,
        dt: float = 0.02,
    ):
        """
        Command Gello to follow using virtual springs.
        
        Args:
            target_joints: Target joint positions [7]
            dt: Time step for velocity estimation
        """
        if not self.is_following:
            print("[GelloTorqueFollower] Warning: Not in following mode.")
            return
        
        # Get current state
        current_position = self.gello.get_joint_state()
        
        # Estimate velocity
        current_velocity = (current_position - self.prev_position) / dt
        
        # Virtual spring force: F = -k * (x - x_target) - d * v
        position_error = current_position - target_joints
        spring_force = -self.spring_stiffness * position_error
        damping_force = -self.damping * current_velocity
        
        # Total torque command
        torque_command = spring_force + damping_force
        
        # Convert to current (simplified, model-specific)
        # For Dynamixel XM430: current = torque / torque_constant
        # Torque constant ≈ 0.0012 Nm/mA
        current_command = torque_command / 0.0012
        
        # Safety limit
        current_command = np.clip(current_command, -self.max_current, self.max_current)
        
        # Command current
        # Note: This requires implementing set_current() in DynamixelDriver
        # self.driver.set_current(current_command)
        
        # For now, we can use goal_current (address 102)
        # This is model-specific and may need adjustment
        
        # Update state
        self.prev_position = current_position
        self.prev_velocity = current_velocity
        self.target_position = target_joints
    
    def detect_human_intervention(self, threshold: float = 0.1) -> bool:
        """
        Detect if human is actively moving Gello (override).
        
        Args:
            threshold: Position error threshold (radians)
        
        Returns:
            True if human is likely intervening
        """
        if self.target_position is None:
            return False
        
        current_position = self.gello.get_joint_state()
        error = np.abs(current_position - self.target_position)
        
        # If error is large despite torque commands, human is intervening
        return np.any(error > threshold)


# ============================================================================
# Comparison: Position Control vs Torque Control
# ============================================================================

"""
| Feature                | Position Control (Recommended) | Torque Control (Advanced)    |
|------------------------|--------------------------------|------------------------------|
| Implementation         | Simple, robust                 | Complex, needs tuning        |
| Following accuracy     | High (rigid)                   | Medium (compliant)           |
| Human override         | Difficult (fights back)        | Easy (feels like assistance) |
| Safety                 | Need careful limits            | Inherently safer             |
| Haptic feedback        | No                             | Yes (feel forces)            |
| Use case               | Precise demonstrations         | Exploration with override    |

Recommendation:
- Start with Position Control (gello_follower.py) for simplicity
- Use Torque Control if you need human override during autonomous exploration
"""
