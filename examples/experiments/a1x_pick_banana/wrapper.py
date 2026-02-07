"""
Custom wrappers for A1_X robot training.
Adapted from PickBananaEnv for joint-space control.
"""
from typing import OrderedDict
from franka_env.camera.rs_capture import RSCapture
from franka_env.camera.video_capture import VideoCapture
import numpy as np
import copy
import gymnasium as gym
import time
from franka_env.envs.a1x_env import A1XEnv


class A1XTaskEnv(A1XEnv):
    """A1_X environment with task-specific reset and behavior."""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.success = False
    
    def init_cameras(self, name_serial_dict=None):
        """Initialize cameras with shared camera support."""
        if self.cap is not None:
            self.close_cameras()

        self.cap = OrderedDict()
        # Only create VideoCapture entries for cameras actually present in
        # `name_serial_dict`. Do not assume `side_policy_256` or
        # `side_classifier` exist on machines with a single camera.
        for cam_name, kwargs in name_serial_dict.items():
            try:
                cap = VideoCapture(RSCapture(name=cam_name, **kwargs))
                self.cap[cam_name] = cap
            except Exception:
                # If a particular camera cannot be initialized, skip it and
                # continue. This avoids KeyError/Assertion when only one
                # wrist camera is available.
                # (Previously code assumed side_policy_256 existed and
                # aliased side_classifier/demo to it.)
                continue

    def reset(self, **kwargs):
        """Task-specific reset procedure."""
        # 更新当前状态
        self._update_curr_joint_state()

        # 调用父类 reset
        obs, info = super().reset(**kwargs)
        
        # # 张开夹爪准备抓取
        # open_gripper_joints = self.curr_joint_positions.copy()
        # open_gripper_joints[6] = 100.0  # 完全张开 (100mm)
        # self.robot.command_joint_state(open_gripper_joints)
        # time.sleep(1.0)
        self.success = False
        self._update_curr_joint_state()
        obs = self._get_obs()
        
        return obs, info

    def step(self, action: np.ndarray):
        """Override step to add task-specific logic if needed."""
        obs, reward, done, truncated, info = super().step(action)
        
        # 添加任务特定的成功检测
        if reward > 5.0:  # 高奖励表示成功
            self.success = True
            info["success"] = True
        
        return obs, reward, done, truncated, info


class A1XGripperPenaltyWrapper(gym.Wrapper):
    """
    为 A1_X 添加夹爪惩罚,避免频繁开关夹爪.
    
    A1_X 的夹爪是连续控制 (0-100mm),我们需要检测大幅度变化.
    
    注意：这个wrapper应该在 SERLObsWrapper 之后使用，
    因此它不能依赖 obs["state"] 的结构（已被扁平化）。
    我们直接从 env.unwrapped 获取机器人状态。
    """
    
    def __init__(self, env, penalty=-0.05, threshold=30.0):
        """
        Args:
            env: A1_X 环境（可能已经包装过）
            penalty: 频繁操作夹爪的惩罚
            threshold: 触发惩罚的夹爪变化阈值 (mm)
        """
        super().__init__(env)
        
        # 获取底层 A1XEnv
        base_env = env
        while hasattr(base_env, 'env'):
            base_env = base_env.env
        
        if not hasattr(base_env, 'curr_joint_positions'):
            raise RuntimeError(
                "❌ A1XGripperPenaltyWrapper 需要底层环境有 curr_joint_positions 属性！\n"
                "   请确保基础环境是 A1XEnv 或其子类。"
            )
        
        self.penalty = penalty
        self.threshold = threshold
        self.last_gripper_pos = None
        self.last_action_gripper = None

    def reset(self, **kwargs):
        """Reset wrapper state."""
        obs, info = self.env.reset(**kwargs)
        
        # 直接从底层环境获取初始夹爪位置
        base_env = self.env
        while hasattr(base_env, 'env'):
            base_env = base_env.env
        
        # A1X的夹爪位置在 curr_joint_positions 的最后一个元素（索引6）
        self.last_gripper_pos = base_env.curr_joint_positions[6]
        self.last_action_gripper = None
        
        return obs, info

    def step(self, action):
        """添加夹爪变化惩罚."""
        action = copy.deepcopy(action)
        gripper_action = action[..., -1]  # 最后一个动作是夹爪
        
        # 执行动作
        observation, reward, terminated, truncated, info = self.env.step(action)
        
        # 检查是否有人工干预
        if "intervene_action" in info:
            action = info["intervene_action"]
            gripper_action = action[..., -1]
        
        # 计算夹爪变化惩罚
        gripper_penalty = 0.0
        
        if self.last_action_gripper is not None:
            # 检测快速开关: 连续两个动作的夹爪方向相反且幅度大
            gripper_delta = gripper_action - self.last_action_gripper
            
            if abs(gripper_delta) > self.threshold:
                gripper_penalty = self.penalty
                info["gripper_penalty"] = gripper_penalty
        
        # 更新上次夹爪动作
        self.last_action_gripper = gripper_action
        
        # 直接从底层环境更新当前夹爪位置
        base_env = self.env
        while hasattr(base_env, 'env'):
            base_env = base_env.env
        
        self.last_gripper_pos = base_env.curr_joint_positions[6]
        
        # 添加惩罚信息
        if "grasp_penalty" not in info:
            info["grasp_penalty"] = gripper_penalty
        
        return observation, reward, terminated, truncated, info
