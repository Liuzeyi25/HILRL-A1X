"""
Gello Expert Module for Robot Teleoperation.

Provides a unified interface to read Gello state, similar to SpaceMouseExpert.
Supports bidirectional control: read for teleoperation, write for following.
"""

import numpy as np
from typing import Tuple, Optional


class GelloExpert:
    """
    This class provides an interface to the Gello robot.
    It reads the Gello joint state and provides a "get_action" method
    to get the latest action and button state (gripper commands).
    
    Also supports reverse control: making Gello follow robot positions.
    """

    def __init__(self, 
                 port: str = "/dev/serial/by-id/usb-FTDI_USB__-__Serial_Converter_FTA7NNNU-if00-port0",
                 dynamixel_config: Optional[dict] = None):
        """
        Initialize Gello expert.
        
        Args:
            port: Serial port for Gello device
            dynamixel_config: Optional configuration for Dynamixel motors
        """
        # Import here to avoid dependency issues if Gello is not installed
        try:
            from gello.agents.gello_agent import GelloAgent, DynamixelRobotConfig
            from gello.agents.gello_follower import GelloFollower
            
            # Use default A1_X configuration if not provided
            if dynamixel_config is None:
                dynamixel_config = DynamixelRobotConfig(
                    joint_ids=[1, 2, 3, 4, 5, 6],
                    joint_offsets=[1.5708, 4.71239, 4.71239, 3.14159, 1.5708, 3.14159],
                    joint_signs=[1.0, -1.0, -1.0, -1.0, 1.0, 1.0],
                    gripper_config=[7, 139.66015625, 199.16015625]
                )
            
            self.gello_agent = GelloAgent(port=port, dynamixel_config=dynamixel_config)
            self._robot = self.gello_agent._robot
            
            # Initialize follower for reverse control (Gello follows robot)
            self.gello_follower = GelloFollower(self._robot)
            
            self.initialized = True
            self._following_mode = False
            
            print(f"✅ Gello Expert initialized on port: {port}")
            print(f"   - Teleoperation mode: Ready to read Gello movements")
            print(f"   - Follower mode: Available for reset synchronization")
            
        except Exception as e:
            print(f"❌ Failed to initialize Gello: {e}")
            self.initialized = False
            self.gello_agent = None
            self._robot = None
            self.gello_follower = None
            self._following_mode = False
            self.gello_follower = None
            self._following_mode = False

    def get_action(self) -> np.ndarray:
        """
        Returns the latest action from Gello.
        
        Returns:
            action: Joint positions (7-DOF: 6 arm joints + 1 gripper value)
        """
        if not self.initialized:
            # Return zero action if not initialized
            return np.zeros(7)
        
        try:
            # Read Gello joint state (includes arm + gripper)
            joint_state = self.gello_agent.act({})
            
            # Return 7 DOF: 6 arm joints + gripper value
            return joint_state[:7]
            
        except Exception as e:
            print(f"Warning: Failed to read Gello state: {e}")
            return np.zeros(7)
    
    def get_joint_state(self) -> np.ndarray:
        """Get current joint positions directly from robot."""
        if not self.initialized or self._robot is None:
            return np.zeros(7)
        
        try:
            return self._robot.get_joint_state()
        except Exception as e:
            print(f"Warning: Failed to get joint state: {e}")
            return np.zeros(7)
    
    def start_following(self, initial_position: Optional[np.ndarray] = None):
        """
        Enable follower mode: Gello will follow commanded positions.
        
        Args:
            initial_position: Optional initial position to move to
        """
        if not self.initialized or self.gello_follower is None:
            print("⚠️  Gello follower not available")
            return
        
        if self._following_mode:
            print("ℹ️  Already in following mode")
            return
        
        try:
            self.gello_follower.start()
            self._following_mode = True
            
            if initial_position is not None:
                self.command_follow(initial_position)
            
            print("🔄 Gello follower mode enabled")
            
        except Exception as e:
            print(f"❌ Failed to start follower mode: {e}")
            self._following_mode = False
    
    def stop_following(self):
        """
        Disable follower mode: Return Gello to free-wheeling (normal teleoperation).
        """
        if not self.initialized or self.gello_follower is None:
            return
        
        if not self._following_mode:
            return
        
        try:
            self.gello_follower.stop()
            self._following_mode = False
            print("🎮 Gello returned to teleoperation mode")
            
        except Exception as e:
            print(f"Warning: Error stopping follower mode: {e}")
    
    def command_follow(self, target_joints: np.ndarray):
        """
        Command Gello to move to target joint positions (follower mode).
        
        Args:
            target_joints: Target joint positions [7] (arm joints + gripper)
        """
        if not self.initialized or self.gello_follower is None:
            return
        
        if not self._following_mode:
            print("⚠️  Not in following mode. Call start_following() first.")
            return
        
        try:
            self.gello_follower.command_follow(target_joints)
        except Exception as e:
            print(f"Warning: Failed to command follow: {e}")
    
    def is_following(self) -> bool:
        """Check if currently in follower mode."""
        return self._following_mode
    
    def close(self):
        """Cleanup resources."""
        if self.initialized:
            try:
                # Stop following mode if active
                if self._following_mode:
                    self.stop_following()
                
                # Close the underlying robot connection
                if self._robot is not None and hasattr(self._robot, 'close'):
                    self._robot.close()
                
                # Close the gello agent
                if self.gello_agent is not None and hasattr(self.gello_agent, 'close'):
                    self.gello_agent.close()
                
                print("Gello Expert closed.")
            except Exception as e:
                print(f"Warning during Gello cleanup: {e}")
