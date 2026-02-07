from __future__ import annotations

"""

Gym Interface for A1 robot in fake environment.
Notice this script only for test reward classifier training, not for real robot control.
To simply, this script removes camera, robot communication, and video functionality.
Author: Wenkai+Claude 
 
"""

import copy
from typing import Dict, Tuple

import gymnasium as gym
import numpy as np


class FakeDefaultEnvConfig:
    """Simplified configuration for fake environment"""

    # Basic parameters needed for gym interface
    TARGET_POSE: np.ndarray = np.array([0.42442, 0.04643, 0.09824, 0.70595, 0.70604181, 0.04329291,  0.03557814]) # np.zeros((6,))         
    GRASP_POSE: np.ndarray = np.zeros((6,))          
    RESET_Q_POSE: np.ndarray = np.array([              
        0.38, 1.60, -0.92, -1.66, 0.65, 1.96
    ])
    RESET_POSE: np.ndarray = np.zeros((6,))
    REWARD_THRESHOLD: np.ndarray = np.array([0.01, 0.01, 0.01, 0.1, 0.1, 0.1])    
    ACTION_SCALE: np.ndarray = np.array([0.2, 1, 1])                      
    GRIPPER_ACTION_SCALE: float = 1.0

    # Environment settings
    MAX_EPISODE_LENGTH: int = 100


class FakeEnv(gym.Env):
    """Simplified joint-space environment for reward classifier training."""

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        hz: int = 10,
        config: FakeDefaultEnvConfig | None = None,
        fake_env: bool = True,
        save_video: bool = False,
        classifier: bool = False,
    ) -> None:
        super().__init__()
        self.config = copy.deepcopy(config) if config else FakeDefaultEnvConfig()
        self.hz = hz
        self.action_scale = config.ACTION_SCALE if config else np.array([0.2, 1, 1])
        self.resetqpos = self.config.RESET_Q_POSE.copy()
        self.resetpos = self.config.RESET_POSE.copy()
        self._TARGET_POSE = self.config.TARGET_POSE
        self._REWARD_THRESHOLD = self.config.REWARD_THRESHOLD
        self.max_episode_length = self.config.MAX_EPISODE_LENGTH

        # Action & observation spaces
        # 6-dim joint increments + 1-dim gripper control
        
        # date: 6-23, modify to 7-dim cause data collected in XYZ+wxyz
        self.action_space = gym.spaces.Box(
            low=0, high=0.8, shape=(7,), dtype=np.float32
        )
        
        # Observation space compatible with SERLObsWrapper expectations
        self.observation_space = gym.spaces.Dict(
            {
                "state": gym.spaces.Box(-0.2, 0.2, shape=(1, 7), dtype=np.float32),
                # "images": gym.spaces.Dict({}),
                "wrist_1": gym.spaces.Box(0, 255, shape=(1, 128, 128, 3), dtype=np.uint8),
                "wrist_2": gym.spaces.Box(0, 255, shape=(1, 128, 128, 3), dtype=np.uint8),
            }
        )

        # Internal state variables - match the expected format
        self.curr_state_vector = np.zeros(7) 
        self.curr_path_length = 0

    def reset(self, *, seed: int | None = None, options=None):
        super().reset(seed=seed)
        # Reset to initial state
        self.curr_state_vector = np.zeros(7)  # Reset to zero state
        self.curr_path_length = 0
        return self._get_obs(), {}

    def step(self, action: np.ndarray, **kwargs) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict]:
        # Simple dummy step - just update internal state randomly
        action = np.clip(action, -0.15, 0.15)
        
        # Update state with small random changes
        self.curr_state_vector += np.random.normal(0, 0.01, 6)
        self.curr_path_length += 1

        obs = self._get_obs()
        
        # Simple reward computation
        reward = self.compute_reward(obs)
        done = self.curr_path_length >= self.max_episode_length
        
        return obs, reward, done, False, {}
    
    def compute_reward(self, obs) -> bool:
        # state现在是(1,9)的数组
        state_flat = obs["state"].flatten()  # 转为(9,)
        
        # Use first 3 elements as position
        target_position = self._TARGET_POSE[:3] if len(self._TARGET_POSE) >= 3 else np.zeros(3)
        
        if len(state_flat) >= 3:
            position_diff = np.abs(state_flat[:3] - target_position)
            threshold = self._REWARD_THRESHOLD[:3] if len(self._REWARD_THRESHOLD) >= 3 else np.array([0.1, 0.1, 0.1])
            
            if np.all(position_diff < threshold):
                return True
        
        return False

    def _get_obs(self):
    # 返回扁平的state数组，而不是嵌套字典
        state_flat = np.concatenate([
            self.curr_state_vector,  # 8维
            [0.0]  # 1维gripper
        ]).astype(np.float32).reshape(1, -1)  # (1, 9)
        
        return {
            "state": state_flat,  # 直接返回(1,9)数组
            "images": {},
            "wrist_1": np.zeros((1, 128, 128, 3), dtype=np.uint8),
            "wrist_2": np.zeros((1, 128, 128, 3), dtype=np.uint8),
        }

    def close(self):
        # No cleanup needed for fake environment
        pass