"""

New wrapper for CubeEnv that supports both real and fake environments.
Author: Wenkai

"""

import copy
import time
import numpy as np

import sys
import os

# 强制添加serl_robot_infra到Python路径
sys.path.insert(0, '/home/pine/hil-serl/serl_robot_infra')

# For real environment
try:
    from franka_env.envs.fake_env import FakeEnv
    print("✅ FakeEnv loaded successfully")
    FAKE_ENV_AVAILABLE = True
except ImportError as e:
    print(f"❌ FakeEnv import failed: {e}")
    FAKE_ENV_AVAILABLE = False
    FakeEnv = None

try:
    from franka_env.utils.rotations import euler_2_quat
    from scipy.spatial.transform import Rotation as R
    import requests
    from pynput import keyboard
    from franka_env.envs.a1_env import A1Env
    print("✅ Real environment dependencies loaded successfully")
    REAL_ENV_AVAILABLE = True
except ImportError as e:
    print(f"❌ Real environment import failed: {e}")
    REAL_ENV_AVAILABLE = False

# For fake environment /home/e230112/Hil-serl/serl_robot_infra/franka_env/envs/fake_env.py
# from serl_robot_infra.franka_env.envs.fake_env import FakeEnv


class CubeEnv:
    """Unified CubeEnv that works with both real and fake environments"""
    
    def __init__(self, fake_env=False, **kwargs):
        self.fake_env = fake_env
        
        if fake_env:
            # Use simplified fake environment
            self.env = FakeEnv(**kwargs)
            self._init_fake_env()
        else:
            # Use real environment (original implementation)
            if not REAL_ENV_AVAILABLE:
                raise ImportError("Real environment dependencies not available. Use fake_env=True")
            self.env = A1Env(**kwargs)
            self._init_real_env()
    
    # def __init__(self, fake_env=False, **kwargs):
    #     if fake_env:
    #         raise NotImplementedError("Fake environment not available. Use real environment.")
        
    #     if not REAL_ENV_AVAILABLE:
    #         raise ImportError("Real environment dependencies not available.")
    #     self.env = A1Env(**kwargs)
    #     self._init_real_env()
    
    
    def _init_fake_env(self):
        """Initialize fake environment specific attributes"""
        self.should_regrasp = False
        self.action_scale = self.env.config.ACTION_SCALE
        
        # Simple reset position for fake env
        pos = self.env.config.RESET_POSE[:3]
        quat = [0, 0, 0, 1]  # Simple quaternion for fake env
        self.resetpos = np.concatenate([pos, quat])
        
    def _init_real_env(self):
        """Initialize real environment specific attributes (original code)"""
        self.should_regrasp = False

        def on_press(key):
            if str(key) == "Key.f1":
                self.should_regrasp = True

        listener = keyboard.Listener(on_press=on_press)
        listener.start()
        
        quat = euler_2_quat(self.env.config.RESET_POSE[3:])
        pos = self.env.config.RESET_POSE[:3]
        self.action_scale = self.env.config.ACTION_SCALE
        self.resetpos = np.concatenate([pos, quat])

    def go_to_reset(self, joint_reset=False):
        """Move to reset position"""
        # if self.fake_env:
        #     # Simple fake reset - just update internal state
        #     time.sleep(0.1)  # Simulate reset time
        #     return
        
        # Real environment reset (original implementation)
        self.env._update_currpos()
        self.env._send_joint_command(self.env.curr_q)
        time.sleep(0.3)
        
        requests.post(self.env.server + "update_param", json=self.env.config.PRECISION_PARAM)

        # pull up
        self.env._update_currpos()
        self.env._send_joint_command(self.env.resetqpos)
        print("Resetting to joint position:", self.env.resetqpos)

        # perform joint reset if needed
        if joint_reset:
            print("JOINT RESET")
            requests.post(self.env.server + "jointreset")
            time.sleep(0.5)

        # perform Cartesian reset
        if self.env.randomreset:
            print("RANDOM RESET", self.env.randomreset)
            reset_pose = self.resetpos.copy()
            reset_pose[:2] += np.random.uniform(
                -self.env.random_xy_range, self.env.random_xy_range, (2,)
            )
            euler_random = self.env._RESET_POSE[3:].copy()
            euler_random[-1] += np.random.uniform(
                -self.env.random_rz_range, self.env.random_rz_range
            )
            reset_pose[3:] = euler_2_quat(euler_random)
            self.env._send_pos_command(reset_pose)
        else:
            reset_pose = self.resetpos.copy()
            print("Resetting to position:", reset_pose)
            self.env._send_pos_command(reset_pose)
        time.sleep(0.5)

        # Change to compliance mode
        requests.post(self.env.server + "update_param", json=self.env.config.COMPLIANCE_PARAM)

    def reset(self, joint_reset=False, **kwargs):
        """Reset environment"""
        # if self.fake_env:
        #     # Simple fake reset
        #     return self.env.reset(**kwargs)
        
        # Real environment reset (original implementation)
        self.env.last_gripper_act = time.time()
        if self.env.save_video:
            self.env.save_video_recording()

        self.env._recover()
        self.go_to_reset(joint_reset=False)
        self.env._recover()
        self.env.curr_path_length = 0

        self.env._update_currpos()
        obs = self.env._get_obs()
        requests.post(self.env.server + "update_param", json=self.env.config.COMPLIANCE_PARAM)
        self.env.terminate = False
        return obs, {}

    def step(self, action, **kwargs):
        """Step environment"""
        return self.env.step(action, **kwargs)

    def close(self):
        """Close environment"""
        return self.env.close()

    # Delegate all other attributes to the underlying environment
    def __getattr__(self, name):
        return getattr(self.env, name)