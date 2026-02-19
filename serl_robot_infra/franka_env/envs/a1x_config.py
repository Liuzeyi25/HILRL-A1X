"""Example configuration for A1_X robot environment."""
import numpy as np
from franka_env.envs.a1x_env import DefaultA1XEnvConfig


class PickAndPlaceA1XConfig(DefaultA1XEnvConfig):
    """Example configuration for pick and place task with A1_X robot."""
    
    # A1_X Robot Configuration
    A1X_NUM_DOFS = 7
    A1X_NODE_NAME = "a1x_serl_node"
    A1X_PORT = 6100
    A1X_PYTHON_PATH = "/usr/bin/python3"
    
    # Camera Configuration
    REALSENSE_CAMERAS = {
        "wrist_1": {"serial_number": "130322274175", "dim": (640, 480)},
        "wrist_2": {"serial_number": "127122270572", "dim": (640, 480)},
    }
    
    # Task Configuration - Joint Space
    # Target joint configuration (example values, adjust for your task)
    TARGET_JOINT_STATE = np.array([0.0, -0.5, 0.0, -1.5, 0.0, 1.0, 50.0])  # Last value is gripper (0-100mm)
    
    # Reset joint configuration (neutral/home position)
    RESET_JOINT_STATE = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 100.0])  # Gripper open
    
    # Reward threshold (tolerance for each joint in radians, gripper in mm)
    REWARD_THRESHOLD = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 5.0])
    
    # Control Configuration
    # Scaling factor for actions (how much delta per action step)
    ACTION_SCALE = np.array([0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 5.0])  # radians and mm
    
    # Display Configuration
    DISPLAY_IMAGE = True
    MAX_EPISODE_LENGTH = 100
    
    # Random reset
    RANDOM_RESET = False


# For minimal testing without cameras
class MinimalA1XConfig(DefaultA1XEnvConfig):
    """Minimal configuration for testing A1_X robot without cameras."""
    
    A1X_NUM_DOFS = 7
    A1X_NODE_NAME = "a1x_test_node"
    A1X_PORT = 6100
    A1X_PYTHON_PATH = "/usr/bin/python3"
    
    REALSENSE_CAMERAS = {}  # No cameras
    
    TARGET_JOINT_STATE = np.array([0.0, -0.5, 0.0, -1.5, 0.0, 1.0, 50.0])
    RESET_JOINT_STATE = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 100.0])
    REWARD_THRESHOLD = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 5.0])
    ACTION_SCALE = np.array([0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 5.0])
    
    DISPLAY_IMAGE = False
    MAX_EPISODE_LENGTH = 50
    RANDOM_RESET = False
