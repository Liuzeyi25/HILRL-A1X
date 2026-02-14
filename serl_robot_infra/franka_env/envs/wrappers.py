import time
from gymnasium import Env, spaces
import gymnasium as gym
import numpy as np
from gymnasium.spaces import Box, Dict
import copy
from franka_env.spacemouse.spacemouse_expert import SpaceMouseExpert
from franka_env.gello.gello_expert import GelloExpert
import requests
from scipy.spatial.transform import Rotation as R
from franka_env.envs.franka_env import FrankaEnv
from typing import List, Tuple, Optional
import sys
import traceback
from pynput import keyboard
from gello.agents.gello_follower import GelloFollower
sigmoid = lambda x: 1 / (1 + np.exp(-x))


# class ManualRewardWrapper(gym.Wrapper):
#     """
#     独立的手动奖励 wrapper，支持成功/失败标记。
    
#     功能：
#     - 按 's' 键：标记成功 (reward=1.0, done=True)
#     - 按 'f' 键：标记失败 (reward=0.0, done=True)
#     - 不依赖任何干预状态，可以随时使用
#     - 适用于在线训练和数据采集
    
#     Usage:
#         env = YourEnv(...)
#         env = ManualRewardWrapper(env)  # 在最外层添加
#     """
    
#     def __init__(self, env, success_reward: float = 1.0):
#         super().__init__(env)
#         self.success_reward = success_reward
#         self.manual_success_flag = False
#         self.manual_failure_flag = False
        
#         # 启动键盘监听器
#         self.keyboard_listener = keyboard.Listener(
#             on_press=self._on_key_press)
#         self.keyboard_listener.start()
#         print("=" * 60)
#         print("🎹 ManualRewardWrapper 已启用")
#         print("=" * 60)
#         print("   - 按 's' 键：标记当前 episode 为成功")
#         print("   - 按 'f' 键：标记当前 episode 为失败")
#         print("=" * 60)
    
#     def _on_key_press(self, key):
#         """监听 's' 和 'f' 键"""
#         try:
#             if hasattr(key, 'char') and key.char is not None:
#                 if key.char == 's' or key.char == 'S':
#                     self.manual_success_flag = True
#                     print("\n" + "=" * 60)
#                     print("✅ [ManualReward] 手动标记: 成功!")
#                     print("=" * 60)
#                 elif key.char == 'f' or key.char == 'F':
#                     self.manual_failure_flag = True
#                     print("\n" + "=" * 60)
#                     print("❌ [ManualReward] 手动标记: 失败!")
#                     print("=" * 60)
#         except:
#             pass
    
#     def step(self, action):
#         obs, rew, done, truncated, info = self.env.step(action)
        
#         # 检查手动成功标志
#         if self.manual_success_flag:
#             rew = self.success_reward
#             done = True
#             info['succeed'] = True
#             info['manual_reward'] = True
#             self.manual_success_flag = False
#             print(f"✅ [ManualReward] 已应用: reward={rew}, succeed=True")
        
#         # 检查手动失败标志
#         elif self.manual_failure_flag:
#             rew = 0.0
#             done = True
#             info['succeed'] = False
#             info['manual_failure'] = True
#             self.manual_failure_flag = False
#             print(f"❌ [ManualReward] 已应用: reward={rew}, succeed=False")
        
#         return obs, rew, done, truncated, info
    
#     def reset(self, **kwargs):
#         self.manual_success_flag = False
#         self.manual_failure_flag = False
#         return self.env.reset(**kwargs)
    
#     def close(self):
#         if hasattr(self, 'keyboard_listener'):
#             self.keyboard_listener.stop()
#         return self.env.close()


class HumanClassifierWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
    
    def step(self, action):
        obs, rew, done, truncated, info = self.env.step(action)
        if done:
            while True:
                try:
                    rew = int(input("Success? (1/0)"))
                    assert rew == 0 or rew == 1
                    break
                except:
                    continue
        info['succeed'] = rew
        return obs, rew, done, truncated, info
    
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return obs, info
    
class FWBWFrontCameraBinaryRewardClassifierWrapper(gym.Wrapper):
    """
    This wrapper uses the front camera images to compute the reward,
    which is not part of the RL policy's observation space. This is used for the
    forward backward reset-free bin picking task, where there are two classifiers,
    one for classifying success + failure for the forward and one for the
    backward task. Here we also use these two classifiers to decide which
    task to transition into next at the end of the episode to maximize the
    learning efficiency.
    """

    def __init__(self, env: Env, fw_reward_classifier_func, bw_reward_classifier_func):
        # check if env.task_id exists
        assert hasattr(env, "task_id"), "fwbw env must have task_idx attribute"
        assert hasattr(env, "task_graph"), "fwbw env must have a task_graph method"

        super().__init__(env)
        self.reward_classifier_funcs = [
            fw_reward_classifier_func,
            bw_reward_classifier_func,
        ]

    def task_graph(self, obs):
        """
        predict the next task to transition into based on the current observation
        if the current task is not successful, stay in the current task
        else transition to the next task
        """
        success = self.compute_reward(obs)
        if success:
            return (self.task_id + 1) % 2
        return self.task_id

    def compute_reward(self, obs):
        reward = self.reward_classifier_funcs[self.task_id](obs).item()
        return (sigmoid(reward) >= 0.5) * 1

    def step(self, action):
        obs, rew, done, truncated, info = self.env.step(action)
        rew = self.compute_reward(self.env.get_front_cam_obs())
        done = done or rew
        return obs, rew, done, truncated, info


class MultiCameraBinaryRewardClassifierWrapper(gym.Wrapper):
    """
    This wrapper uses the camera images to compute the reward,
    which is not part of the observation space
    """

    def __init__(self, env: Env, reward_classifier_func, target_hz = None):
        super().__init__(env)
        self.reward_classifier_func = reward_classifier_func
        self.target_hz = target_hz

    def compute_reward(self, obs):
        if self.reward_classifier_func is not None:
            return self.reward_classifier_func(obs)
        return 0

    def step(self, action):
        start_time = time.time()
        obs, rew, done, truncated, info = self.env.step(action)
        rew = self.compute_reward(obs)
        done = done or (rew > 0.5)
        info['succeed'] = bool(rew > 0.5)
        if self.target_hz is not None:
            time.sleep(max(0, 1/self.target_hz - (time.time() - start_time)))
            
        return obs, rew, done, truncated, info

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        info['succeed'] = False
        return obs, info
    
    
class MultiStageBinaryRewardClassifierWrapper(gym.Wrapper):
    def __init__(self, env: Env, reward_classifier_func: List[callable]):
        super().__init__(env)
        self.reward_classifier_func = reward_classifier_func
        self.received = [False] * len(reward_classifier_func)
    
    def compute_reward(self, obs):
        rewards = [0] * len(self.reward_classifier_func)
        for i, classifier_func in enumerate(self.reward_classifier_func):
            if self.received[i]:
                continue

            logit = classifier_func(obs).item()
            if sigmoid(logit) >= 0.75:
                self.received[i] = True
                rewards[i] = 1

        reward = sum(rewards)
        return reward

    def step(self, action):
        obs, rew, done, truncated, info = self.env.step(action)
        rew = self.compute_reward(obs)
        done = (done or all(self.received)) # either environment done or all rewards satisfied
        info['succeed'] = all(self.received)
        return obs, rew, done, truncated, info

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.received = [False] * len(self.reward_classifier_func)
        info['succeed'] = False
        return obs, info

class FrontCameraBinaryRewardClassifierWrapperNew(gym.Wrapper):
    """
    This wrapper uses the front camera images to compute the reward,
    which is not part of the observation space
    """

    def __init__(self, env: Env, reward_classifier_func):
        super().__init__(env)
        self.reward_classifier_func = reward_classifier_func

    def compute_reward(self, img):
        import pdb

        pdb.set_trace()
        obs = {
            "state": np.zeros((1, 38)),
            "side": img,
            "left/wrist_1": np.zeros((1, 128, 128, 3)),
            "left/wrist_2": np.zeros((1, 128, 128, 3)),
            "right/wrist_1": np.zeros((1, 128, 128, 3)),
            "right/wrist_2": np.zeros((1, 128, 128, 3)),
        }
        if self.reward_classifier_func is not None:
            logit = self.reward_classifier_func(obs).item()
            return (sigmoid(logit) >= 0.5) * 1
        return 0

    def step(self, action):
        obs, rew, done, truncated, info = self.env.step(action)
        rew = self.compute_reward(self.env.get_front_cam_obs())
        done = done or rew
        return obs, rew, done, truncated, info


class FrontCameraBinaryRewardClassifierWrapper(gym.Wrapper):
    """
    This wrapper uses the front camera images to compute the reward,
    which is not part of the observation space
    """

    def __init__(self, env: Env, reward_classifier_func):
        super().__init__(env)
        self.reward_classifier_func = reward_classifier_func

    def compute_reward(self, obs):
        if self.reward_classifier_func is not None:
            logit = self.reward_classifier_func(obs).item()
            return (sigmoid(logit) >= 0.5) * 1
        return 0

    def step(self, action):
        obs, rew, done, truncated, info = self.env.step(action)
        rew = self.compute_reward(self.env.get_front_cam_obs())
        done = done or rew
        return obs, rew, done, truncated, info


class BinaryRewardClassifierWrapper(gym.Wrapper):
    """
    Compute reward with custom binary reward classifier fn
    """

    def __init__(self, env: Env, reward_classifier_func):
        super().__init__(env)
        self.reward_classifier_func = reward_classifier_func

    def compute_reward(self, obs):
        if self.reward_classifier_func is not None:
            logit = self.reward_classifier_func(obs).item()
            return (sigmoid(logit) >= 0.5) * 1
        return 0

    def step(self, action):
        obs, rew, done, truncated, info = self.env.step(action)
        rew = self.compute_reward(obs)
        done = done or rew
        return obs, rew, done, truncated, info


class ZOnlyWrapper(gym.ObservationWrapper):
    """
    Removal of X and Y coordinates
    """

    def __init__(self, env: Env):
        super().__init__(env)
        self.observation_space["state"] = spaces.Box(-np.inf, np.inf, shape=(14,))

    def observation(self, observation):
        observation["state"] = np.concatenate(
            (
                observation["state"][:4],
                np.array(observation["state"][6])[..., None],
                observation["state"][10:],
            ),
            axis=-1,
        )
        return observation


class ZOnlyNoFTWrapper(gym.ObservationWrapper):
    """
    Removal of X and Y coordinates and force torque sensor readings
    """

    def __init__(self, env: Env):
        super().__init__(env)
        self.observation_space["state"] = spaces.Box(-np.inf, np.inf, shape=(9,))

    def observation(self, observation):
        observation["state"] = np.concatenate(
            (
                np.array(observation["state"][0])[..., None],  # gripper
                np.array(observation["state"][6])[..., None],  # z
                np.array(observation["state"][9])[..., None],  # rz
                observation["state"][-6:],  # vel
            ),
            axis=-1,
        )
        return observation


class Quat2EulerWrapper(gym.ObservationWrapper):
    """
    Convert the quaternion representation of the tcp pose to euler angles
    """

    def __init__(self, env: Env):
        super().__init__(env)
        assert env.observation_space["state"]["tcp_pose"].shape == (7,)
        # from xyz + quat to xyz + euler
        self.observation_space["state"]["tcp_pose"] = spaces.Box(
            -np.inf, np.inf, shape=(6,)
        )

    def observation(self, observation):
        # convert tcp pose from quat to euler
        tcp_pose = observation["state"]["tcp_pose"]
        observation["state"]["tcp_pose"] = np.concatenate(
            (tcp_pose[:3], R.from_quat(tcp_pose[3:]).as_euler("xyz"))
        )
        return observation


class Quat2R2Wrapper(gym.ObservationWrapper):
    """
    Convert the quaternion representation of the tcp pose to rotation matrix
    """

    def __init__(self, env: Env):
        super().__init__(env)
        assert env.observation_space["state"]["tcp_pose"].shape == (7,)
        # from xyz + quat to xyz + euler
        self.observation_space["state"]["tcp_pose"] = spaces.Box(
            -np.inf, np.inf, shape=(9,)
        )

    def observation(self, observation):
        tcp_pose = observation["state"]["tcp_pose"]
        r = R.from_quat(tcp_pose[3:]).as_matrix()
        observation["state"]["tcp_pose"] = np.concatenate(
            (tcp_pose[:3], r[..., :2].flatten())
        )
        return observation


class DualQuat2EulerWrapper(gym.ObservationWrapper):
    """
    Convert the quaternion representation of the tcp pose to euler angles
    """

    def __init__(self, env: Env):
        super().__init__(env)
        assert env.observation_space["state"]["left/tcp_pose"].shape == (7,)
        assert env.observation_space["state"]["right/tcp_pose"].shape == (7,)
        # from xyz + quat to xyz + euler
        self.observation_space["state"]["left/tcp_pose"] = spaces.Box(
            -np.inf, np.inf, shape=(6,)
        )
        self.observation_space["state"]["right/tcp_pose"] = spaces.Box(
            -np.inf, np.inf, shape=(6,)
        )

    def observation(self, observation):
        # convert tcp pose from quat to euler
        tcp_pose = observation["state"]["left/tcp_pose"]
        observation["state"]["left/tcp_pose"] = np.concatenate(
            (tcp_pose[:3], R.from_quat(tcp_pose[3:]).as_euler("xyz"))
        )
        tcp_pose = observation["state"]["right/tcp_pose"]
        observation["state"]["right/tcp_pose"] = np.concatenate(
            (tcp_pose[:3], R.from_quat(tcp_pose[3:]).as_euler("xyz"))
        )
        return observation
    
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return self.observation(obs), info

class GripperCloseEnv(gym.ActionWrapper):
    """
    Use this wrapper to task that requires the gripper to be closed
    """

    def __init__(self, env):
        super().__init__(env)
        ub = self.env.action_space
        assert ub.shape == (7,)
        self.action_space = Box(ub.low[:6], ub.high[:6])

    def action(self, action: np.ndarray) -> np.ndarray:
        new_action = np.zeros((7,), dtype=np.float32)
        new_action[:6] = action.copy()
        return new_action

    def step(self, action):
        new_action = self.action(action)
        obs, rew, done, truncated, info = self.env.step(new_action)
        if "intervene_action" in info:
            info["intervene_action"] = info["intervene_action"][:6]
        return obs, rew, done, truncated, info
    
    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

    
class SpacemouseIntervention(gym.ActionWrapper):
    def __init__(self, env, action_indices=None):
        super().__init__(env)

        self.gripper_enabled = True
        if self.action_space.shape == (6,):
            self.gripper_enabled = False

        self.action_space = gym.spaces.Box(
            low=np.array([
                -0.02, -0.02, -0.02,   # 前三维
                -0.01, -0.01, -0.1,  # 第4-6维
                -0.2                # 最后一维
            ], dtype=np.float32),
            high=np.array([
                0.02,  0.02,  0.02,
                0.01, 0.01, 0.1,
                0.2
            ], dtype=np.float32),
            dtype=np.float32
        )
        
        self.action_scale = env.action_scale


        self.expert = SpaceMouseExpert()
        self.left, self.right = False, False
        self.action_indices = action_indices
        
        # 🎯 手动奖励设置标志
        self.manual_success_flag = False  # 是否手动标记成功
        self.manual_failure_flag = False  # 是否手动标记失败
        
        # 启动键盘监听器
        self.keyboard_listener = keyboard.Listener(on_press=self._on_key_press)
        self.keyboard_listener.start()
        print("=" * 60)
        print("🎹 SpaceMouse 键盘监听已启用")
        print("=" * 60)
        print("   - 按 's' 键：标记当前 episode 为成功")
        print("   - 按 'f' 键：标记当前 episode 为失败")
        print("=" * 60)
    
    def _on_key_press(self, key):
        """监听 's' 和 'f' 键"""
        try:
            if hasattr(key, 'char') and key.char is not None:
                if key.char == 's' or key.char == 'S':
                    self.manual_success_flag = True
                    print("\n" + "=" * 60)
                    print("✅ [SpaceMouse] 手动标记: 成功!")
                    print("=" * 60)
                elif key.char == 'f' or key.char == 'F':
                    self.manual_failure_flag = True
                    print("\n" + "=" * 60)
                    print("❌ [SpaceMouse] 手动标记: 失败!")
                    print("=" * 60)
        except:
            pass

    def action(self, action: np.ndarray) -> np.ndarray:
        """
        Input:
        - action: policy action
        Output:
        - action: spacemouse action if nonezero; else, policy action
        """
        expert_a, buttons = self.expert.get_action()
        # self.left, self.right = tuple(buttons)
        self.left, self.right = buttons[0], buttons[-1]
        intervened = False
        
        if np.linalg.norm(expert_a) > 0.001:
            intervened = True

        if self.gripper_enabled:
            if self.left:  # close gripper
                gripper_action = np.random.uniform(-0.2, -0.15, size=(1,))
                intervened = True
            elif self.right:  # open gripper
                gripper_action = np.random.uniform(0.15, 0.2, size=(1,))
                intervened = True
            else:
                gripper_action = np.zeros((1,))
            expert_a = np.concatenate((expert_a, gripper_action), axis=0)
            # expert_a[:6] += np.random.uniform(-0.5, 0.5, size=6)

        if self.action_indices is not None:
            filtered_expert_a = np.zeros_like(expert_a)
            filtered_expert_a[self.action_indices] = expert_a[self.action_indices]
            expert_a = filtered_expert_a

        if intervened:
            # 🔧 裁剪到动作空间范围内
            expert_a = np.clip(expert_a, self.action_space.low, self.action_space.high)
            expert_a = expert_a * self.action_scale
            # print("[SpaceMouse] Intervention detected: using expert action,", expert_a)
            return expert_a, True

        return action, False

    def step(self, action):

        new_action, replaced = self.action(action)

        obs, rew, done, truncated, info = self.env.step(new_action)
        if replaced:
            info["intervene_action_eef"] = new_action
        info["left"] = self.left
        info["right"] = self.right
        
        # 🎯 检查手动成功标志
        if self.manual_success_flag:
            rew = 1.0
            done = True
            info['succeed'] = True
            info['manual_reward'] = True
            self.manual_success_flag = False
            print(f"✅ [SpaceMouse] 已应用: reward={rew}, succeed=True")
        
        # 🎯 检查手动失败标志
        elif self.manual_failure_flag:
            rew = -1.0
            done = True
            info['succeed'] = False
            info['manual_failure'] = True
            self.manual_failure_flag = False
            print(f"❌ [SpaceMouse] 已应用: reward={rew}, succeed=False")
        
        return obs, rew, done, truncated, info
    
    def reset(self, **kwargs):
        """Reset 时清除手动标记"""
        self.manual_success_flag = False
        self.manual_failure_flag = False
        return self.env.reset(**kwargs)
    
    def close(self):
        """关闭键盘监听器和 SpaceMouse"""
        if hasattr(self, 'keyboard_listener'):
            self.keyboard_listener.stop()
        if hasattr(self, 'expert'):
            self.expert.close()
        return self.env.close()


# [新增] Rate 类：用于精确控制频率
# [旧代码] 原本没有频率控制类，直接用 time.sleep(1.0 / hz) 不够精确
class Rate:
    """
    精确控制频率的辅助类（与 gello/env.py 中的 Rate 相同）
    
    为什么需要这个类：
    - 简单的 time.sleep(1/hz) 不考虑代码执行时间，会导致实际频率低于目标
    - Rate 类会计算从上次 sleep 到现在的时间差，精确等待剩余时间
    """
    def __init__(self, rate: float):
        self.last = time.time()
        self.rate = rate

    def sleep(self) -> None:
        """精确等待到下一个控制周期"""
        while self.last + 1.0 / self.rate > time.time():
            time.sleep(0.0001)  # 高精度等待
        self.last = time.time()


class GelloIntervention(gym.ActionWrapper):
    """
    Gello teleoperation intervention wrapper.
    
    🔧 重要设计说明：
    这个 wrapper 只负责读取 Gello 设备的位置，不创建机器人连接！
    底层环境（如 A1XTaskEnv）已经建立了与机器人的连接。
    
    功能：
    - 读取 Gello 设备的关节位置作为动作输入
    - 支持干预模式切换（空格键）
    - 支持双向控制（Reset 时 Gello 跟随机器人）
    - 🚀 双线程模式：后台高频控制，主线程正常采集数据
    
    Usage:
        env = A1XTaskEnv(...)  # 底层环境已连接机器人
        env = GelloIntervention(
            env,
            left_config_path="path/to/config.yaml",
            threaded_control=True,  # 启用双线程模式
        )
        
        # step() 正常返回完整数据，后台线程高频控制机器人
        obs, rew, done, trunc, info = env.step(action)
    """
    
    def __init__(
        self, 
        env, 
        left_config_path: str,
        right_config_path: Optional[str] = None,
        control_rate_hz: int = 500,
        use_save_interface: bool = False,
        action_indices=None,
        always_intervene: bool = False,
        sync_on_reset: bool = False,
        reset_follow_duration: float = 0.5,
        fast_intervention_mode: bool = True,
        threaded_control: bool = True,  # 🚀 新增：双线程控制模式
        eval_mode: bool = False,  # 🎯 新增：评估模式（禁用干预）
        sync_max_retries: int = 3,  # 🆕 同步最大重试次数
        sync_error_threshold: float = 0.15,  # 🆕 同步误差阈值（弧度）
        sync_on_intervention: bool = True,  # 🆕 默认在线训练模式（按空格时同步）
        enable_follower: bool = True,  # 🆕 是否初始化 GelloFollower（用于同步功能）
       
    ):
        """
        Args:
            env: Base environment (已连接机器人，不需要再创建连接)
            left_config_path: Path to YAML configuration (只使用 agent 部分)
            right_config_path: Path to right arm config (for bimanual, optional)
            control_rate_hz: Control loop frequency for background thread
            use_save_interface: Enable keyboard interface for saving data
            action_indices: Optional indices to filter which actions can be controlled
            always_intervene: If True, intervention is always enabled
            sync_on_reset: If True, Gello follows robot to reset position
            reset_follow_duration: Duration for Gello to follow robot during reset
            fast_intervention_mode: If True, use fast direct control during intervention
            threaded_control: If True, use background thread for high-freq control
            eval_mode: If True, disable intervention (evaluation mode)
            sync_on_intervention: If True, sync when intervention enabled
            enable_follower: If True, initialize GelloFollower for sync
                - 在线训练：enable_follower=True, sync_on_reset=False
                  (初始化follower但reset不同步，只在按空格时同步)
                - 数据采集：enable_follower=True, sync_on_reset=True
                  (初始化follower并在reset时同步)
        """
        super().__init__(env)
        
        import atexit
        import signal
        import threading
        from pathlib import Path
        from omegaconf import OmegaConf
        
        # Add gello_software to path
        gello_path = 'Gello/gello_software'
        if gello_path not in sys.path:
            sys.path.insert(0, gello_path)
        
        from gello.utils.launch_utils import instantiate_from_dict
        
        self.action_indices = action_indices
        self.control_rate_hz = control_rate_hz
        self.bimanual = right_config_path is not None
        self.always_intervene = always_intervene and not eval_mode
        self.eval_mode = eval_mode  # 🎯 新增：保存评估模式标志
        self.sync_on_reset = sync_on_reset
        self.sync_on_intervention = sync_on_intervention  # 🆕 保存干预同步标志
        self.enable_follower = enable_follower  # 🆕 保存follower初始化标志
        self.reset_follow_duration = reset_follow_duration
        self.fast_intervention_mode = fast_intervention_mode
        self.threaded_control = threaded_control  # 🚀 新增
        
        # 🆕 同步验证参数
        self.sync_max_retries = sync_max_retries
        self.sync_error_threshold = sync_error_threshold
        
        
        # �🔧 精确控制频率
        self._rate = Rate(control_rate_hz)
        self._fast_rate = Rate(control_rate_hz)
        
        # 缓存 robot 和 base_env
        self._cached_robot = None
        self._cached_base_env = None
        
        # 🚀 双线程控制状态
        self._control_thread = None
        self._control_thread_running = False
        self._thread_lock = threading.Lock()
        self._resetting = False               # Reset 进行中标志
        
        # 🚀 共享状态（线程间通信）
        self._latest_gello_joints = None      # 最新 Gello 读数
        self._latest_a1x_command = None       # 最新发送的 A1X 命令
        self._control_step_count = 0          # 后台控制步数
        
        # 🆕 干预同步状态
        self._intervention_just_enabled = False  # 刚刚启用干预，需要同步
        self._syncing = False                    # 🔧 同步进行中标志（阻止其他控制）
        
        # 快速干预状态
        self._stop_fast_loop = False
        
        # Cleanup tracking
        self.cleanup_in_progress = False
        
        # Register cleanup handlers
        atexit.register(self._cleanup)
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        # Load configs (只使用 agent 部分)
        left_cfg = OmegaConf.to_container(
            OmegaConf.load(left_config_path), resolve=True
        )
        if self.bimanual:
            right_cfg = OmegaConf.to_container(
                OmegaConf.load(right_config_path), resolve=True
            )
        
        # 🔧 关键设计：不创建 robot！底层环境已经连接了机器人
        # 只创建 Gello Agent（用于读取 Gello 设备位置）
        
        print("📟 Initializing Gello Agent (device reader only)...")
        
        # Create agent with retry mechanism
        max_agent_retries = 3
        agent_created = False
        
        for attempt in range(max_agent_retries):
            try:
                if self.bimanual:
                    from gello.agents.agent import BimanualAgent
                    self.agent = BimanualAgent(
                        agent_left=instantiate_from_dict(left_cfg["agent"]),
                        agent_right=instantiate_from_dict(right_cfg["agent"]),
                    )
                else:
                    self.agent = instantiate_from_dict(left_cfg["agent"])
                
                agent_created = True
                print(f"   ✅ Gello Agent created successfully")
                break
                
            except RuntimeError as e:
                if "Failed to set torque mode" in str(e) and attempt < max_agent_retries - 1:
                    print(f"   ⚠️  Agent creation failed (attempt {attempt + 1}/{max_agent_retries}): {e}")
                    print(f"      Retrying in 1 second...")
                    time.sleep(1)
                else:
                    raise
        
        if not agent_created:
            raise RuntimeError("Failed to create Gello Agent after multiple attempts")
        
        # 🔧 不需要 robot server/client - 底层环境已经处理了
        # 🔧 不需要 RobotEnv - 直接使用底层环境
        # 🔧 不需要 move_to_start_position - 底层环境负责 reset
        
        # Initialize save interface if requested
        self.save_interface = None
        if use_save_interface:
            from gello.utils.control_utils import SaveInterface
            self.save_interface = SaveInterface(
                data_dir=Path(left_config_path).parents[1] / "data",
                agent_name=self.agent.__class__.__name__,
                expand_user=True,
            )
        
        # Control loop state
        self.intervention_enabled = always_intervene
        self.keyboard_listener = keyboard.Listener(on_press=self._on_key_press)
        self.keyboard_listener.start()
        print("🎹 键盘监听器已启动 (监听 's'/'f' 键和空格键)")
        
        # 初始化观测缓存
        self.last_obs = None
        
        # 🎯 手动奖励设置标志
        self.manual_success_flag = False  # 是否手动标记成功
        self.manual_failure_flag = False  # 是否手动标记失败
        
        # 🔧 恢复：初始化 GelloFollower（用于双向控制）
        self.gello_follower = None
        if self.enable_follower:
            try:
                
                
                # 获取底层 DynamixelRobot
                dynamixel_robot = None
                if hasattr(self.agent, '_robot'):
                    dynamixel_robot = self.agent._robot
                elif hasattr(self.agent, '_agent') and hasattr(self.agent._agent, '_robot'):
                    dynamixel_robot = self.agent._agent._robot
                
                if dynamixel_robot is not None:
                    self.gello_follower = GelloFollower(dynamixel_robot)
                    print("   ✅ GelloFollower initialized for bidirectional control")
                else:
                    print("   ⚠️  Warning: Could not initialize GelloFollower (sync disabled)")
            except Exception as e:
                print(f"   ⚠️  Warning: Failed to initialize GelloFollower: {e}")
        
        print(f"✅ GelloIntervention initialized")
        print(f"   - Agent: {self.agent.__class__.__name__}")
        print(f"   - Control rate: {control_rate_hz} Hz")
        print(f"   - Bimanual: {self.bimanual}")
        print(f"   - 🚀 双线程控制: {'启用' if threaded_control else '禁用'}")
        print(f"   - 快速干预模式: {'启用' if fast_intervention_mode else '禁用'}")
        if self.enable_follower:
            print(f"   - 🔄 GelloFollower: 已初始化 (支持同步功能)")
        if sync_on_reset:
            print(f"   - 🔄 同步验证: 最大重试={sync_max_retries}, 误差阈值={sync_error_threshold:.3f} rad")
        if eval_mode:
            print(f"   🎯 评估模式: 干预已禁用")
            print(f"   ⚠️  Gello 设备将被忽略，只使用 Agent 策略")
        elif always_intervene:
            print(f"   🎮 始终干预模式")
        else:
            print(f"   🎮 按空格键切换Gello干预 (当前: {'启用' if self.intervention_enabled else '禁用'})")
        
        # 🆕 打印同步策略
        if sync_on_reset and sync_on_intervention:
            print(f"   🔄 同步模式: Reset时同步 + 启用干预时同步（数据采集）")
        elif sync_on_reset:
            print(f"   🔄 同步模式: 仅 Reset时同步（数据采集模式）")
        elif sync_on_intervention:
            print(f"   🔄 同步模式: 仅启用干预时同步（在线训练模式）")
        else:
            print(f"   ⚪ 同步已禁用")
        
        # 🚀 如果启用双线程控制且不是评估模式，启动后台控制线程
        if self.threaded_control and not eval_mode:
            self._start_control_thread()
    
    # ==================== 🚀 双线程控制相关方法 ====================
    
    def _start_control_thread(self):
        """启动后台高频控制线程"""
        import threading
        
        if self._control_thread is not None and self._control_thread.is_alive():
            print("⚠️  控制线程已在运行")
            return
        
        self._control_thread_running = True
        self._control_thread = threading.Thread(
            target=self._control_loop,
            name="GelloControlThread",
            daemon=True,  # 主线程退出时自动结束
        )
        self._control_thread.start()
        print(f"🚀 后台控制线程已启动 (目标频率: {self.control_rate_hz} Hz)")
    
    def _stop_control_thread(self):
        """停止后台控制线程"""
        self._control_thread_running = False
        if self._control_thread is not None:
            self._control_thread.join(timeout=1.0)
            print("⏹️  后台控制线程已停止")
    
    def _control_loop(self):
        """
        🚀 后台高频控制循环
        
        职责：
        - 高频读取 Gello 位置（500Hz）
        - 立即发送命令到机器人
        - 更新共享状态供主线程使用
        
        不做：
        - 获取观测
        - 计算奖励
        - Episode 管理
        """
        # 🎯 评估模式下不应该启动这个线程
        if self.eval_mode:
            print("⚠️  评估模式下后台控制线程不应启动")
            return
        
        rate = Rate(self.control_rate_hz)
        last_print_time = time.time()
        
        # 预先获取 robot 引用（避免循环中重复查找）
        robot = self._get_cached_robot()
        
        while self._control_thread_running:
            try:
                # 🔧 Reset 期间完全暂停控制（让底层环境接管）
                if self._resetting:
                    rate.sleep()
                    continue
                
                # 🆕 检查是否需要同步 Gello（刚启用干预时）
                need_sync = False
                with self._thread_lock:
                    if self._intervention_just_enabled:
                        need_sync = True
                        self._intervention_just_enabled = False  # 重置标志
                        self._syncing = True  # 🔧 设置同步中标志，阻止其他控制
                
                # 🎯 只有启用 sync_on_intervention 时才执行同步
                if need_sync and self.sync_on_intervention:
                    print("\n" + "="*60)
                    print("🔄 检测到干预启用，开始同步 Gello...")
                    print("="*60)
                    try:
                        # 获取当前机器人位置
                        robot = self._get_cached_robot()
                        if robot is not None:
                            current_a1x_joints = robot.get_joint_state()
                            print(f"🤖 当前机器人位置: [{', '.join(f'{v:.3f}' for v in current_a1x_joints[:6])}]")
                            
                            # 执行同步
                            self._iterative_sync_to_robot(current_a1x_joints)
                            
                            # 验证同步结果
                            time.sleep(0.2)
                            final_gello = self._get_current_gello_joints()
                            if final_gello is not None:
                                gello_target = self._a1x_to_gello_mapping(current_a1x_joints)
                                if gello_target is not None:
                                    final_error = np.max(np.abs(gello_target[:6] - final_gello[:6]))
                                    if final_error < 0.15:
                                        print(f"✅ 同步完成！误差: {final_error:.4f} rad")
                                    else:
                                        print(f"⚠️  同步完成，但误差较大: {final_error:.4f} rad")
                            
                            # 更新缓存位置
                            with self._thread_lock:
                                self._latest_a1x_command = current_a1x_joints.copy()
                            
                            print("🎮 现在可以开始人工控制了！")
                        else:
                            print("⚠️  无法获取机器人引用，跳过同步")
                    except Exception as e:
                        print(f"⚠️  同步失败: {e}")
                        import traceback
                        traceback.print_exc()
                    finally:
                        # 🔧 无论同步成功或失败，都要清除同步中标志
                        with self._thread_lock:
                            self._syncing = False
                    print("="*60 + "\n")
                
                # 🔧 同步期间跳过控制（等待同步完成）
                if self._syncing:
                    rate.sleep()
                    continue
                
                # 🔧 干预未启用时，维持当前位置（避免漂移）
                if not self.intervention_enabled:
                   # print("⏸️  控制循环：干预未启用!")
                    # # 如果有缓存的位置，持续发送维持命令
                    # with self._thread_lock:
                    #     if (self._latest_a1x_command is not None and
                    #             robot is not None):
                    #         robot.command_joint_state(
                    #             self._latest_a1x_command,
                    #             from_gello=False
                    #         )
                    rate.sleep()
                    continue
                
                # 🔧 计时诊断（找出瓶颈）
             #   t0 = time.time()
                
                # 1. 读取 Gello 位置
                gello_joints = self.agent.act(None)
             #   t1 = time.time()
                
                # 2. 映射到 A1X
                target_a1x = self._gello_to_a1x_mapping(gello_joints)
           #     t2 = time.time()
                
                if target_a1x is not None:
                    # 3. 发送命令到机器人
                    if robot is not None:
                        robot.command_joint_state(target_a1x, from_gello=False)
                #    t3 = time.time()
                    
                    # 4. 更新共享状态（线程安全）
                    with self._thread_lock:
                        self._latest_gello_joints = gello_joints.copy()
                        self._latest_a1x_command = target_a1x.copy()
                        self._control_step_count += 1
                
                # 5. 定期打印诊断信息
                # now = time.time()
                # if now - last_print_time >= 2.0:  # 每2秒打印一次
                #     actual_hz = self._control_step_count / 2.0 if self._control_step_count > 0 else 0
                #     gello_time_ms = (t1 - t0) * 1000
                #     map_time_ms = (t2 - t1) * 1000
                #     cmd_time_ms = (t3 - t2) * 1000 if target_a1x is not None else 0
                #     total_ms = (t3 - t0) * 1000 if target_a1x is not None else (t2 - t0) * 1000
                #     print(f"   🔄 [诊断] 频率: ~{actual_hz:.0f} Hz | Gello读取: {gello_time_ms:.1f}ms | 映射: {map_time_ms:.1f}ms | 发送: {cmd_time_ms:.1f}ms | 总计: {total_ms:.1f}ms")
                #     self._control_step_count = 0
                #     last_print_time = now
                
                # 6. 精确等待
                rate.sleep()
                
            except Exception as e:
                print(f"⚠️  [后台线程] 控制错误: {e}")
                time.sleep(0.01)  # 防止错误循环过快
    
    def get_latest_intervention_action(self) -> Optional[np.ndarray]:
        """
        获取最新的干预动作（线程安全）
        
        用于 step() 中记录实际执行的动作
        """
        with self._thread_lock:
            return self._latest_a1x_command.copy() if self._latest_a1x_command is not None else None
    
    def _convert_joints_to_eef_action(self, prev_eef_pose: Optional[np.ndarray], 
                                      curr_eef_pose: Optional[np.ndarray],
                                      prev_gripper: float,
                                      curr_gripper: float) -> Optional[np.ndarray]:
        """
        将 EEF pose 变化转换为 EEF 动作空间（delta pose + delta gripper）
        
        用于数据录制：策略学习 EEF 空间动作更容易泛化
        
        Args:
            prev_eef_pose: 执行前的 EEF pose [x, y, z, qx, qy, qz, qw]
            curr_eef_pose: 执行后的 EEF pose [x, y, z, qx, qy, qz, qw]
            prev_gripper: 执行前的夹爪位置 (mm)
            curr_gripper: 执行后的夹爪位置 (mm)
            
        Returns:
            EEF 动作 [7]: [dx, dy, dz, drx, dry, drz, dgripper_norm]
        """
        if prev_eef_pose is None or curr_eef_pose is None:
            return None
        
        try:
            # 计算 EEF delta
            delta_eef = self._compute_delta_eef(prev_eef_pose, curr_eef_pose)
            
            # 🎯 夹爪归一化后计算增量
            # A1X 夹爪范围: [2mm, 99mm]
            GRIPPER_MIN = 2.0
            GRIPPER_MAX = 99.0
            
            # 归一化到 [0, 1]
            gripper_range = GRIPPER_MAX - GRIPPER_MIN
            prev_gripper_norm = (prev_gripper - GRIPPER_MIN) / gripper_range
            curr_gripper_norm = (curr_gripper - GRIPPER_MIN) / gripper_range
            
            # 计算归一化后的增量
            delta_gripper_norm = curr_gripper_norm - prev_gripper_norm
            
            # 拼接 delta EEF + delta 夹爪（归一化）
            eef_action = np.concatenate([delta_eef, [delta_gripper_norm]])
            return eef_action
            
        except Exception as e:
            print(f"⚠️  EEF 转换失败: {e}")
            return None
    
    def _get_current_eef_pose(self) -> Optional[np.ndarray]:
        """
        获取当前末端执行器位姿
        
        Returns:
            np.ndarray: [x, y, z, qx, qy, qz, qw] 或 None
        """
        try:
            robot = self._get_cached_robot()
            if robot is not None and hasattr(robot, 'get_eef_pose'):
                pos, quat = robot.get_eef_pose()
                return np.concatenate([pos, quat])  # [x, y, z, qx, qy, qz, qw]
            return None
        except Exception as e:
            print(f"⚠️  获取 EEF pose 失败: {e}")
            return None
    
    def _get_current_gripper_position(self) -> float:
        """
        获取当前夹爪位置
        
        Returns:
            float: 夹爪位置 (mm)，如果无法获取则返回 0.0
        """
        try:
            # 方法1: 从 robot 的 curr_joint_positions 获取
            robot = self._get_cached_robot()
            if robot is not None and hasattr(robot, 'curr_joint_positions'):
                joint_pos = robot.curr_joint_positions
                if joint_pos is not None and len(joint_pos) >= 7:
                    return float(joint_pos[6])
            
            # 方法2: 从 base_env 获取
            env = self._get_cached_base_env()
            if env is not None and hasattr(env, 'curr_joint_positions'):
                joint_pos = env.curr_joint_positions
                if joint_pos is not None and len(joint_pos) >= 7:
                    return float(joint_pos[6])
            
            # 方法3: 从 last_obs 获取
            if self.last_obs is not None and isinstance(self.last_obs, dict):
                if 'state' in self.last_obs:
                    state = self.last_obs['state']
                    if isinstance(state, np.ndarray) and state.shape[-1] >= 7:
                        return float(state[6])
            
            return 0.0
        except Exception as e:
            print(f"⚠️  获取夹爪位置失败: {e}")
            return 0.0
    
    def _compute_delta_eef(self, prev_pose: np.ndarray, curr_pose: np.ndarray) -> np.ndarray:
        """
        计算两个 EEF pose 之间的增量
        
        Args:
            prev_pose: 前一个 EEF pose [x, y, z, qx, qy, qz, qw]
            curr_pose: 当前 EEF pose [x, y, z, qx, qy, qz, qw]
            
        Returns:
            np.ndarray: Delta pose [dx, dy, dz, drx, dry, drz] (旋转用欧拉角表示)
        """
        from scipy.spatial.transform import Rotation as R
        
        # 位置增量（直接相减）
        delta_pos = curr_pose[:3] - prev_pose[:3]
        
        # 旋转增量（四元数运算）
        # delta_rot = curr_rot * prev_rot^{-1}
        prev_rot = R.from_quat(prev_pose[3:])  # [qx, qy, qz, qw]
        curr_rot = R.from_quat(curr_pose[3:])
        
        # 计算相对旋转
        delta_rot = curr_rot * prev_rot.inv()
        
        # 转换为欧拉角 (XYZ 约定)
        delta_euler = delta_rot.as_euler('xyz')
        
        return np.concatenate([delta_pos, delta_euler])
    
    def _validate_trajectory_data(self, joint_action: Optional[np.ndarray], 
                                  eef_action: Optional[np.ndarray], 
                                  obs: dict) -> bool:
        """
        验证轨迹数据的完整性和正确性
        
        检查项：
        1. 动作维度正确
        2. 观测包含必要字段（图像、状态）
        3. 动作值在合理范围内
        4. 夹爪值有效
        
        Returns:
            bool: 数据是否有效
        """
        issues = []
        
        # 1. 检查关节动作
        if joint_action is None:
            issues.append("❌ joint_action 为 None")
        elif len(joint_action) != 7:
            issues.append(f"❌ joint_action 维度错误: {len(joint_action)} (应为7)")
        elif np.any(np.isnan(joint_action)):
            issues.append("❌ joint_action 包含 NaN")
        else:
            # 检查夹爪范围 [2mm, 99mm]
            gripper = joint_action[6]
            if gripper < 0 or gripper > 100:
                issues.append(f"⚠️  夹爪值异常: {gripper:.2f} mm")
        
        # 2. 检查 EEF 动作
        if eef_action is not None:
            if len(eef_action) != 7:
                issues.append(f"❌ eef_action 维度错误: {len(eef_action)} (应为7)")
            elif np.any(np.isnan(eef_action)):
                issues.append("❌ eef_action 包含 NaN")
        
        # 3. 检查观测完整性
        if "images" not in obs:
            issues.append("❌ 观测中缺少 images")
        elif len(obs["images"]) == 0:
            issues.append("❌ images 为空")
        
        if "state" not in obs:
            issues.append("❌ 观测中缺少 state")
        
        # 打印问题（仅在有问题时）
        if issues:
            if not hasattr(self, '_validation_error_count'):
                self._validation_error_count = 0
            self._validation_error_count += 1
            
            # 每100次错误打印一次
            if self._validation_error_count % 100 == 1:
                print(f"\n🔍 [数据验证] 发现 {len(issues)} 个问题:")
                for issue in issues:
                    print(f"   {issue}")
        
        return len(issues) == 0
    
    # ==================== 键盘和清理 ====================
    
    def _on_key_press(self, key):
        """
        空格键按下 -> 切换干预状态（仅在非始终干预模式下生效）
        's' 键按下 -> 手动标记成功 (reward=1.0)
        'f' 键按下 -> 手动标记失败 (reward=0.0)
        ESC键 -> 停止快速干预循环
        """
        # 🔍 调试：打印所有按键
        print(f"\n🔍 [键盘监听器] 检测到按键: {key}, type={type(key)}")
        
        # 🔍 先处理字符键（如 's', 'f'）
        try:
            if hasattr(key, 'char') and key.char is not None:
                print(f"🔍 [键盘监听器] 字符键: '{key.char}'")
                
                if key.char in ['s', 'S']:
                    # 🎯 's' 键：手动标记成功
                    self.manual_success_flag = True
                    print("\n" + "="*60)
                    print("✅ 手动标记: 成功 (reward=1.0)")
                    print("="*60)
                    return  # 直接返回，不继续处理
                
                elif key.char in ['f', 'F']:
                    # 🎯 'f' 键：手动标记失败
                    self.manual_failure_flag = True
                    print("\n" + "="*60)
                    print("❌ 手动标记: 失败 (reward=0.0)")
                    print("="*60)
                    return  # 直接返回，不继续处理
                    
        except Exception as e:
            print(f"⚠️ [键盘监听器] 处理字符键时出错: {e}")
            import traceback
            traceback.print_exc()
        
        # 处理特殊键（空格、ESC等）
        try:
            if key == keyboard.Key.space:
                # 🎯 评估模式下禁用干预切换
                if self.eval_mode:
                    print("⚠️  评估模式下干预已禁用")
                    return
                
                if self.always_intervene:
                    print("⚠️  始终干预模式已启用，无法通过空格键切换")
                    return
                
                # 🆕 检测干预状态变化
                was_disabled = not self.intervention_enabled
                
                self.intervention_enabled = not self.intervention_enabled
                status = "🟢 启用" if self.intervention_enabled else "🔴 禁用"
                print(f"\n🎮 Gello干预已{status}")
                
                # 🆕 如果从禁用变为启用，设置同步标志
                if was_disabled and self.intervention_enabled:
                    with self._thread_lock:
                        self._intervention_just_enabled = True
                    print("🔄 将在下次控制循环中同步 Gello 到机器人位置...")
                
                # 如果禁用干预，停止快速干预循环
                if not self.intervention_enabled:
                    self._stop_fast_loop = True
                    
            elif key == keyboard.Key.esc:
                # ESC 键停止快速干预循环
                self._stop_fast_loop = True
                print("\n⏹️  ESC: 停止快速干预循环")
                
        except AttributeError:
            # 如果不是特殊键，就忽略
            pass
        except Exception as e:
            print(f"⚠️ [键盘监听器] 处理特殊键时出错: {e}")
            import traceback
            traceback.print_exc()
    
    def _cleanup(self):
        """Clean up Gello agent resources."""
        if self.cleanup_in_progress:
            return
        self.cleanup_in_progress = True
        
        print("Cleaning up Gello resources...")
        
        # 🚀 停止后台控制线程
        if hasattr(self, '_control_thread_running'):
            self._stop_control_thread()
        
        # Close Gello agent
        if hasattr(self, 'agent') and self.agent is not None:
            try:
                if hasattr(self.agent, 'close'):
                    self.agent.close()
            except Exception as e:
                print(f"Error closing agent: {e}")
        
        # Close GelloFollower
        if hasattr(self, 'gello_follower') and self.gello_follower is not None:
            try:
                if hasattr(self.gello_follower, 'stop_following'):
                    self.gello_follower.stop_following()
            except Exception as e:
                print(f"Error closing follower: {e}")
        
        print("Cleanup completed.")
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully."""
        self._cleanup()
        import os
        os._exit(0)
    
    def action(self, action: np.ndarray) -> Tuple[np.ndarray, bool]:
        """
        Process action through Gello agent if intervention is enabled.
        
        Input:
        - action: policy action (will be ignored if intervention is enabled)
        
        Output:
        - action: gello action if intervention enabled; else, policy action
        - intervened: whether intervention occurred
        """
        # Check if intervention is enabled
        if not self.intervention_enabled:
            return action, False
        
        # Get action from Gello agent
        try:
            # GelloAgent.act() returns absolute Gello joint positions
            gello_joints = self.agent.act(None)
            
            # Apply action filtering if specified
            if self.action_indices is not None:
                filtered = np.zeros_like(gello_joints)
                filtered[self.action_indices] = gello_joints[self.action_indices]
                gello_joints = filtered
            
            return gello_joints, True
        except Exception as e:
            print(f"⚠️  Error getting Gello action: {e}")
            import traceback
            traceback.print_exc()
            return action, False
    
    def step(self, action):
        """
        Execute one step in the environment.
        
        � 双线程模式 (threaded_control=True):
        后台线程高频控制机器人，主线程正常获取观测和计算奖励。
        step() 返回完整数据，适合数据采集。
        
        🔧 非线程模式：
        当干预时，直接将 Gello 位置映射到 A1X 并发送绝对位置命令。
        """
        # ========== 🔧 同步期间阻塞等待 ==========
        # 循环检查是否正在同步，如果是则阻塞等待同步完成
        sync_wait_start = time.time()
        while True:
            with self._thread_lock:
                syncing = self._syncing
            
            if not syncing:
                # 同步完成，继续执行
                break
            
            # 同步进行中，打印提示并等待
            if not hasattr(self, '_sync_wait_printed'):
                print("\n" + "="*60)
                print("⏸️  [step] 检测到同步进行中，阻塞等待同步完成...")
                print("="*60)
                self._sync_wait_printed = True
            
            time.sleep(0.1)  # 每100ms检查一次
            
            # 超时保护（最多等待10秒）
            if time.time() - sync_wait_start > 15.0:
                print("⚠️  [step] 同步等待超时（>15s），强制继续")
                break
        
        # 清除打印标志
        if hasattr(self, '_sync_wait_printed'):
            delattr(self, '_sync_wait_printed')
            print("✅ [step] 同步完成，恢复执行")
        
        # ========== 🎯 评估模式：跳过干预 ==========
        if self.eval_mode:
            obs, rew, done, truncated, info = self.env.step(action)
            self.last_obs = obs
            info["gello_intervened"] = False
            info["eval_mode"] = True
            return obs, rew, done, truncated, info
        
        # ==================== 🚀 双线程模式 ====================
        if self.threaded_control and self.intervention_enabled:
            return self._threaded_step(action)
        
        # ==================== 快速干预模式（不推荐用于数据采集）====================
        # if self.fast_intervention_mode and self.intervention_enabled and not self.threaded_control:
        #     return self._fast_intervention_step(action)
        
        # ==================== 原始干预模式 ====================
        # Process action through intervention logic
        gello_joints, intervened = self.action(action)     
        # if intervened:
        #     # 🎯 获取执行前的 EEF pose 和夹爪位置（从上一次观测）
        #     prev_eef_pose_gripper = self.last_obs['state']['ee_pos_rot_gripper'] if self.last_obs is not None else None
            
        #     # 绝对位置控制：直接发送目标位置到机器人
        #     try:
        #         target_a1x_joints = self._gello_to_a1x_mapping(gello_joints)
                
        #         if target_a1x_joints is not None:
        #             # DEBUG: 打印夹爪信息（每100步打印一次）
        #             if not hasattr(self, '_debug_step_count'):
        #                 self._debug_step_count = 0
        #             self._debug_step_count += 1
        #             if self._debug_step_count % 100 == 1:
        #                 print(f"🔍 [DEBUG] Gello gripper: {gello_joints[6]:.4f} -> A1X gripper: {target_a1x_joints[6]:.2f} mm")
                    
        #             # 发送命令
        #             robot = self._get_cached_robot()
        #             if robot is not None:
        #                 robot.command_joint_state(target_a1x_joints, from_gello=False)
                    
        #             # 获取观测
        #             env = self._get_cached_base_env()
        #             if env is not None and hasattr(env, '_update_curr_joint_state'):
        #                 env._update_curr_joint_state()
        #             if env is not None and hasattr(env, '_get_obs'):
        #                 obs = env._get_obs()
        #             else:
        #                 obs = self.last_obs
                    
        #             # 🎯 获取执行后的 EEF pose 和夹爪位置（从当前观测）
        #             curr_eef_pose_gripper = obs['state']['ee_pos_rot_gripper']
                    
        #             # 计算奖励和终止条件
        #             rew = 0
        #             done = False
        #             truncated = False
        #             if env is not None and hasattr(env, 'compute_reward'):
        #                 rew = env.compute_reward(obs)
        #                 done = rew > 0
        #             if env is not None and hasattr(env, 'curr_path_length'):
        #                 env.curr_path_length += 1
        #                 if hasattr(env, 'max_episode_length'):
        #                     done = done or (env.curr_path_length >= env.max_episode_length)
                    
        #             # 🆕 计算 EEF 动作空间（正确的旋转计算）
        #             intervene_action_eef = None
        #             if prev_eef_pose_gripper is not None:
        #                 # ee_pos_rot_gripper = [x, y, z, roll, pitch, yaw, gripper]
        #                 # 位置：可以直接相减
        #                 delta_pos = curr_eef_pose_gripper[:3] - prev_eef_pose_gripper[:3]
                        
        #                 # 姿态：用旋转矩阵计算（欧拉角不能直接相减！）
        #                 prev_rot = R.from_euler('xyz', prev_eef_pose_gripper[3:6])
        #                 curr_rot = R.from_euler('xyz', curr_eef_pose_gripper[3:6])
        #                 delta_rot = curr_rot * prev_rot.inv()  # 相对旋转
        #                 delta_euler = delta_rot.as_euler('xyz')  # 转回欧拉角
                        
        #                 # 夹爪：可以直接相减
        #                 delta_gripper = curr_eef_pose_gripper[6] - prev_eef_pose_gripper[6]
                        
        #                 # 组合：[dx, dy, dz, droll, dpitch, dyaw, dgripper]
        #                 intervene_action_eef = np.concatenate([delta_pos, delta_euler, [delta_gripper]])
                    
        #             info = {
        #                 "intervene_action": target_a1x_joints,
        #                 "intervene_action_eef": intervene_action_eef,  # EEF 动作
        #                 "gello_intervened": True,
        #                 "gello_joints": gello_joints,
        #                 "succeed": rew > 0
        #             }
                    
        #             self.last_obs = obs
        #             return obs, int(rew), done, truncated, info
                    
        #     except Exception as e:
        #         print(f"⚠️  绝对位置控制失败，回退到 delta 模式: {e}")
        #         import traceback
        #         traceback.print_exc()
        
        # 非干预模式或绝对控制失败
        obs, rew, done, truncated, info = self.env.step(action)
        self.last_obs = obs
        info["gello_intervened"] = False
        manual_succeed = None  # 用于后续设置 info
        if self.manual_success_flag:
            rew = 1.0
            self.manual_success_flag = False  # 重置标志
            done = True  # 手动标记成功后结束 episode
            manual_succeed = True
            print(f"✅ 手动奖励已应用: reward={rew}, succeed=True")
        elif self.manual_failure_flag:
            rew = -1.0
            self.manual_failure_flag = False  # 重置标志
            done = True  # 手动标记失败后结束 episode
            manual_succeed = False
            print(f"❌ 手动奖励已应用: reward={rew}, succeed=False")
        base_env = self._get_cached_base_env()
        if base_env is not None and hasattr(base_env, 'curr_path_length'):
            base_env.curr_path_length += 1  # ✅ 明确使用底层环境
            if hasattr(base_env, 'max_episode_length'):
                done = done or (base_env.curr_path_length >= base_env.max_episode_length)
        info = {
            "succeed": rew > 0 if manual_succeed is None else manual_succeed,  # 🎯 手动标记优先
        }
        return obs, rew, done, truncated, info
    
    def _threaded_step(self, action):
        """
        🚀 双线程模式的 step：后台线程已经在高频控制，主线程只负责数据采集   
        特点：
        - 后台线程：500Hz 读取 Gello + 发送命令
        - 主线程：正常频率获取观测、计算奖励、返回完整数据
        
        适合 record_demos_octo.py 等数据采集场景
        """
        # 🎯 获取执行前的 EEF pose 和夹爪位置（从上一次观测）
        prev_eef_pose_gripper = self.last_obs['state']['ee_pos_rot_gripper'] if self.last_obs is not None else None
        
        # 1. 获取后台线程记录的最新干预动作（A1X 关节空间）
        intervene_action = self.get_latest_intervention_action()
        
        # 🔧 如果后台线程还没有数据（刚启动），使用传入的 action
        if intervene_action is None:
            print("⚠️  后台线程还没有数据，使用策略动作")
            # 使用非干预模式执行一步
            obs, rew, done, truncated, info = self.env.step(action)
            self.last_obs = obs
            info["gello_intervened"] = False
            info["threaded_mode"] = True
            info["succeed"] = rew > 0
            return obs, rew, done, truncated, info
        
        # 2. 正常获取观测（不发送命令，后台线程已经在发了）
        env = self._get_cached_base_env()
        if env is not None and hasattr(env, '_update_curr_joint_state'):
            env._update_curr_joint_state()
        if env is not None and hasattr(env, '_get_obs'):
            obs = env._get_obs()
        else:
            obs = self.last_obs if self.last_obs is not None else {}
        
        # 🎯 获取执行后的 EEF pose 和夹爪位置（从当前观测）
        curr_eef_pose_gripper = obs['state']['ee_pos_rot_gripper'] if 'state' in obs else None
        
        # 3. 计算奖励和终止条件
        rew = 0
        done = False
        truncated = False
        
        # 🎯 检查手动奖励标志
        manual_succeed = None  # 用于后续设置 info
        if self.manual_success_flag:
            rew = 1.0
            self.manual_success_flag = False  # 重置标志
            done = True  # 手动标记成功后结束 episode
            manual_succeed = True
            print(f"✅ 手动奖励已应用: reward={rew}, succeed=True")
        elif self.manual_failure_flag:
            rew = -1.0
            self.manual_failure_flag = False  # 重置标志
            done = True  # 手动标记失败后结束 episode
            manual_succeed = False
            print(f"❌ 手动奖励已应用: reward={rew}, succeed=False")
        # elif env is not None and hasattr(env, 'compute_reward'):
        #     rew = env.compute_reward(obs)
        #     done = rew > 0
        if env is not None and hasattr(env, 'curr_path_length'):
            env.curr_path_length += 1
            if hasattr(env, 'max_episode_length'):
                done = done or (env.curr_path_length >= env.max_episode_length)
        
        # 4. 🆕 计算 EEF 动作空间（正确的旋转计算）
        intervene_action_eef = None
        if prev_eef_pose_gripper is not None and curr_eef_pose_gripper is not None:
            # ee_pos_rot_gripper = [x, y, z, roll, pitch, yaw, gripper]
            # 位置：可以直接相减
            delta_pos = curr_eef_pose_gripper[:3] - prev_eef_pose_gripper[:3]
            
            # 姿态：用旋转矩阵计算（欧拉角不能直接相减！）
            prev_rot = R.from_euler('xyz', prev_eef_pose_gripper[3:6])
            curr_rot = R.from_euler('xyz', curr_eef_pose_gripper[3:6])
            delta_rot = curr_rot * prev_rot.inv()  # 相对旋转
            delta_euler = delta_rot.as_euler('xyz')  # 转回欧拉角
            
            # 夹爪：可以直接相减
            delta_gripper = curr_eef_pose_gripper[6] - prev_eef_pose_gripper[6]
            
            # 组合：[dx, dy, dz, droll, dpitch, dyaw, dgripper]
            intervene_action_eef = np.concatenate([delta_pos, delta_euler, [delta_gripper]])
        
        # 5. 🆕 数据验证
        data_valid = self._validate_trajectory_data(
            intervene_action, intervene_action_eef, obs
        )
        
        # � 方案B：在线采集单步数据，离线重组为chunks
        # 这里只返回单步动作，chunking在保存时处理
        
        # 6. 构建完整 info（兼容 record_demos_octo.py）
        info = {
            "intervene_action": intervene_action,       # A1X 关节空间 [7]
            "intervene_action_eef": intervene_action_eef,  # 单步 EEF delta [7]
            "gello_intervened": True,
            "threaded_mode": True,
            "succeed": rew > 0 if manual_succeed is None else manual_succeed,  # 🎯 手动标记优先
            "data_valid": data_valid,                   # 数据有效性标志
        }
        
        # 7. 缓存观测
        self.last_obs = obs
        
        return obs, int(rew), done, truncated, info
    
    def intervention_step_only(self):
        """
        🚀 最精简的单步干预：只发送命令，不返回任何东西
        
        适用于自己管理循环的场景，最大化性能。
        
        Returns:
            bool: 是否成功执行
        """
        if not self.intervention_enabled:
            return False
        
        try:
            gello_joints = self.agent.act(None)
            target_a1x_joints = self._gello_to_a1x_mapping(gello_joints)
            
            if target_a1x_joints is not None:
                robot = self._get_cached_robot()
                if robot is not None:
                    robot.command_joint_state(target_a1x_joints, from_gello=False)
                    return True
            return False
        except:
            return False
    
    # [新增] 缓存方法：避免每次 step 都遍历 wrapper 链
    # [旧代码] 直接调用 _get_robot() 和 _get_base_env()，每次都要遍历
    def _get_cached_robot(self):
        """获取底层机器人实例（带缓存，避免每次遍历）"""
        if self._cached_robot is not None:
            return self._cached_robot
        self._cached_robot = self._get_robot()
        return self._cached_robot
    
    def _get_cached_base_env(self):
        """获取底层环境实例（带缓存，避免每次遍历）"""
        if self._cached_base_env is not None:
            return self._cached_base_env
        self._cached_base_env = self._get_base_env()
        return self._cached_base_env
    
    def _get_robot(self):
        """获取底层机器人实例"""
        env = self.env
        while env is not None:
            if hasattr(env, 'robot'):
                return env.robot
            if hasattr(env, 'env'):
                env = env.env
            else:
                break
        return None
    
    def _get_base_env(self):
        """获取最底层的环境实例（A1XEnv）"""
        env = self.env
        while env is not None:
            if hasattr(env, '_update_curr_joint_state'):
                return env
            if hasattr(env, 'env'):
                env = env.env
            else:
                break
        return None
    
    def _gello_to_a1x_mapping(self, gello_joints: np.ndarray) -> Optional[np.ndarray]:
        """
        将 Gello 关节位置映射到 A1X 关节位置。
        
        使用与 A1XRobot._map_to_a1x() 相同的线性范围映射。
        """
        try:
            # 方法1: 尝试使用 A1XRobot 的映射方法
            robot = self._get_robot()
            if robot is not None and hasattr(robot, '_map_to_a1x'):
                return robot._map_to_a1x(gello_joints)
            
            # 方法2: 手动实现映射
            return self._manual_gello_to_a1x_mapping(gello_joints)
            
        except Exception as e:
            print(f"❌ Gello→A1X 映射失败: {e}")
            return None
    
    def _manual_gello_to_a1x_mapping(self, gello_joints: np.ndarray) -> np.ndarray:
        """
        手动实现 Gello 到 A1X 的范围映射（与 A1XRobot._map_to_a1x 一致）。
        """
        # Gello 关节范围
        gello_range_start = np.array([-2.87, 0.0, 0.0, -1.57, -1.34, -2.0, 0.103], dtype=float)
        gello_range_end = np.array([2.87, 3.14, 3.14, 1.57, 1.34, 2.0, 1.0], dtype=float)
        
        # A1X 关节范围
        a1x_range_start = np.array([-2.880, 0.0, 0.0, 1.55, 1.521, -1.56, 2.0], dtype=float)
        a1x_range_end = np.array([2.880, 3.14, -2.95, -1.55, -1.52, 1.56, 99.0], dtype=float)
        
        gello_joints = np.array(gello_joints, dtype=float)
        
        # Clip 到 Gello 范围
        clipped = gello_joints.copy()
        for i in range(7):
            lo = min(gello_range_start[i], gello_range_end[i])
            hi = max(gello_range_start[i], gello_range_end[i])
            clipped[i] = np.clip(gello_joints[i], lo, hi)
        
        # 线性映射
        result = np.zeros(7, dtype=float)
        for i in range(7):
            in_start = gello_range_start[i]
            in_end = gello_range_end[i]
            out_start = a1x_range_start[i]
            out_end = a1x_range_end[i]
            
            in_range = in_end - in_start
            if abs(in_range) < 1e-9:
                result[i] = out_start
            else:
                t = (clipped[i] - in_start) / in_range
                result[i] = out_start + t * (out_end - out_start)
        
        return result
    
    def reset(self, **kwargs):
        """
        Reset environment and optionally sync Gello to robot's reset position.
        
        🔧 恢复双向控制：
        1. Robot resets to initial position
        2. Gello follows robot to match the reset position (if sync_on_reset=True)
        3. After sync, Gello returns to free-wheeling mode for teleoperation
        4. 自动关闭干预（需要手动按空格重新启用）
        """
        import time
        self._start_following()
        # � 方案B不需要在线buffer管理
        
        # 🔧 标记 reset 开始（暂停后台线程的控制）
        self._resetting = True
        
        # 🔧 自动关闭干预（除非是 always_intervene 模式）
        if not self.always_intervene and self.intervention_enabled:
            self.intervention_enabled = False
            print("🔴 Episode 结束，自动关闭 Gello 干预（按空格重新启用）")
        
        # 🔧 关键修复：在 reset 之前，确保 Gello 处于自由模式
        # 这样可以避免读取到缓存的错误位置
        print("\n" + "="*60)
        print("🔄 开始 Reset - 第 {} 次".format(
            getattr(self, '_reset_count', 0) + 1))
        print("="*60)
        
        # 强制停止任何可能残留的 following 模式
        # try:
        #     if self.gello_follower is not None:
        #         self.gello_follower.stop()
        #         print("🛑 强制停止 Gello following 模式（清理残留状态）")
        #         time.sleep(0.3)  # 等待模式切换完成
        # except Exception as e:
        #     print(f"⚠️  停止 following 模式时出错（忽略）: {e}")
        
        # 🔍 读取 Gello 当前位置（reset 前）
        gello_before_reset = self._get_current_gello_joints()
        if gello_before_reset is not None:
            print(f"📍 Reset 前 Gello 位置: "
                  f"[{', '.join(f'{v:.3f}' for v in gello_before_reset[:6])}]")
        
        # Reset base environment
        obs, info = self.env.reset(**kwargs)
        time.sleep(1.5)  # Wait for stability
        
        # 记录 reset 次数
        if not hasattr(self, '_reset_count'):
            self._reset_count = 0
        self._reset_count += 1
        
        # 🔧 缓存观测
        self.last_obs = obs
        
        # 🎯 评估模式：跳过 Gello 同步
        if self.eval_mode:
            print("🎯 评估模式: 跳过 Gello 同步")
            return obs, info
        
        # 🔧 恢复：Sync Gello to robot's reset position
        if self.sync_on_reset:
            try:
                # Get robot's reset joint state
                robot_joint_state = self._get_robot_joint_state(obs, info)
                
                if robot_joint_state is not None:
                    print(f"🤖 Robot reset position (A1X): "
                          f"[{', '.join(f'{v:.3f}' for v in robot_joint_state)}]")
                    
                    # 🆕 迭代同步：重复同步直到误差足够小
                    self._iterative_sync_to_robot(robot_joint_state)
                    
                    # 🔧 关键验证：确认 Gello 真的移动到了正确位置
                    time.sleep(0.3)
                    final_gello = self._get_current_gello_joints()
                    if final_gello is not None:
                        gello_target = self._a1x_to_gello_mapping(
                            robot_joint_state)
                        if gello_target is not None:
                            final_error = np.max(np.abs(
                                gello_target[:6] - final_gello[:6]))
                            print(f"🔍 最终验证: Gello 误差 = {final_error:.4f} rad")
                            
                            if final_error > 0.2:
                                print("⚠️  警告：Gello 位置误差过大，"
                                      "可能存在硬件问题！")
                                print(f"   目标: [{', '.join(f'{v:.3f}' for v in gello_target[:6])}]")
                                print(f"   实际: [{', '.join(f'{v:.3f}' for v in final_gello[:6])}]")
                    
                    # 🔧 同步后读取 A1X 实际位置（避免跳变抖动）
                    time.sleep(0.1)  # 等待位置稳定
                    actual_a1x_pos = self._get_current_a1x_joints()
                    if actual_a1x_pos is not None:
                        with self._thread_lock:
                            self._latest_a1x_command = actual_a1x_pos.copy()
                        print(f"💾 已缓存 A1X 实际位置: "
                              f"[{', '.join(f'{v:.3f}' for v in actual_a1x_pos[:6])}]")
                    else:
                        # 回退：使用目标位置
                        with self._thread_lock:
                            self._latest_a1x_command = robot_joint_state.copy()
                        print("💾 已缓存 reset 目标位置（无法读取实际位置）")
                    
                else:
                    print("⚠️  无法获取机器人关节状态（跳过同步）")
                    
            except Exception as e:
                print(f"⚠️  Gello 同步失败: {e}")
                import traceback
                traceback.print_exc()
                # Ensure we're back in teleoperation mode
                try:
                    self._stop_following()
                except Exception:
                    pass
        
        # 🔧 重要：等待 Gello 模式切换完成，避免读取到不稳定的位置
        # 这可以防止 reset 后 A1X 抖动
        time.sleep(0.5)  # 等待 Gello 完全进入自由模式
        
        # 🔧 标记 reset 结束（恢复后台线程的控制）
        self._resetting = False
        
        print("✅ Gello 同步完成，准备开始新的 episode")
        print("💡 提示：按空格键启用 Gello 干预")
        
        return obs, info
    
    def _iterative_sync_to_robot(self, robot_joint_state: np.ndarray):
        """
        🆕 迭代同步：重复同步直到误差足够小
        
        流程：
        1. 将 A1X 关节映射到 Gello 目标
        2. 同步 Gello 到目标位置
        3. 验证误差
        4. 如果误差过大且未超过最大重试次数，重复 1-3
        
        Args:
            robot_joint_state: A1X 机器人的目标关节位置
        """
        for attempt in range(self.sync_max_retries):
            print(f"\n{'='*60}")
            print(f"🔄 同步尝试 {attempt + 1}/{self.sync_max_retries}")
            print(f"{'='*60}")
            
            # 1. 计算 Gello 目标位置
            gello_target = self._a1x_to_gello_mapping(robot_joint_state)
            
            if gello_target is None:
                print("❌ 无法计算 Gello 目标位置，终止同步")
                break
            
            print(f"🎯 Gello 目标位置: [{', '.join(f'{v:.3f}' for v in gello_target)}]")
            
            # 2. 获取同步前的 Gello 位置
            before_gello = self._get_current_gello_joints()
            if before_gello is not None:
                before_diff = np.abs(gello_target - before_gello)
                before_max_error = np.max(before_diff[:6])  # 只看前6个关节，忽略夹爪
                print(f"📍 同步前 Gello: [{', '.join(f'{v:.3f}' for v in before_gello)}]")
                print(f"📏 同步前误差: max={before_max_error:.4f} rad, [{', '.join(f'{v:.3f}' for v in before_diff)}]")
            
            # 3. 执行同步
            self._start_following()
           # print(f"⚡ 同步中 (用时 {self.reset_follow_duration}s)...")
            self._slow_follow_to_target(gello_target, duration=self.reset_follow_duration)
            
            # 🔧 关键修复：在停止 following 之前，持续发送目标位置更长时间
            # 确保电机真正到位并稳定
            print("⏳ 持续保持目标位置 0.5 秒...")
            if self.gello_follower is not None:
                for _ in range(10):  # 30次 * 0.05秒 = 1.5秒
                    self.gello_follower.command_follow(gello_target)
                    time.sleep(0.05)
            
            self._stop_following()
            
            # # 4. 等待稳定（停止后等待更长时间）
            # print("⏳ 等待电机模式切换...")
            # time.sleep(0.5)
            
            # 5. 验证同步后的误差
            after_gello = self._get_current_gello_joints()
            
            if after_gello is None:
                print("⚠️  无法读取同步后的 Gello 位置")
                break
            
            # 计算误差（只看前6个关节，忽略夹爪）
            after_diff = np.abs(gello_target - after_gello)
            max_error = np.max(after_diff[:6])
            mean_error = np.mean(after_diff[:6])
            
            print(f"📍 同步后 Gello: [{', '.join(f'{v:.3f}' for v in after_gello)}]")
            print(f"📏 同步后误差: max={max_error:.4f} rad, mean={mean_error:.4f} rad")
            print(f"   各关节误差: [{', '.join(f'{v:.3f}' for v in after_diff)}]")
            
            # 🆕 额外验证：将 Gello 映射回 A1X 看看误差
            mapped_a1x = self._gello_to_a1x_mapping(after_gello)
            if mapped_a1x is not None:
                a1x_diff = np.abs(robot_joint_state - mapped_a1x)
                a1x_max_error = np.max(a1x_diff[:6])
                print(f"🔄 反向验证 (Gello→A1X):")
                print(f"   目标 A1X: [{', '.join(f'{v:.3f}' for v in robot_joint_state[:6])}]")
                print(f"   映射 A1X: [{', '.join(f'{v:.3f}' for v in mapped_a1x[:6])}]")
                print(f"   A1X 误差: max={a1x_max_error:.4f}, [{', '.join(f'{v:.3f}' for v in a1x_diff[:6])}]")
            
            # 6. 检查是否满足精度要求
            if max_error < self.sync_error_threshold:
                print(f"✅ 同步成功！最大误差 {max_error:.4f} < 阈值 {self.sync_error_threshold:.4f}")
                print(f"{'='*60}\n")
                return
            else:
                print(f"⚠️  误差仍然过大：{max_error:.4f} >= {self.sync_error_threshold:.4f}")
                
                if attempt < self.sync_max_retries - 1:
                    print(f"🔁 准备第 {attempt + 2} 次同步...")
                else:
                    print(f"❌ 已达到最大重试次数 {self.sync_max_retries}，终止同步")
                    print(f"⚠️  最终误差: {max_error:.4f} rad (阈值: {self.sync_error_threshold:.4f})")
                    print(f"💡 建议：")
                    print(f"   1. 检查 Gello 硬件是否正常")
                    print(f"   2. 检查映射参数是否正确")
                    print(f"   3. 适当增大 sync_error_threshold")
                    print(f"{'='*60}\n")
    
    def _a1x_to_gello_mapping(self, a1x_joints: np.ndarray) -> Optional[np.ndarray]:
        """
        Convert A1_X joint positions to Gello joint positions using inverse mapping.
        
        使用 A1XRobot._map_from_a1x() 方法进行范围线性映射。
        这是从旧代码 wrappers_20260125.py 移植的正确映射方法。
        
        Args:
            a1x_joints: A1_X joint positions [7] (6 arm joints + 1 gripper)
            
        Returns:
            Gello joint positions [7], or None if mapping fails
        """
        try:
            # 直接使用 wrapper 内置的映射实现（范围已修正）
            print("📐 使用内置范围映射")
            return self._manual_a1x_to_gello_mapping(a1x_joints)
                
        except Exception as e:
            print(f"❌ 映射失败: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _manual_a1x_to_gello_mapping(self, a1x_joints: np.ndarray) -> np.ndarray:
        """
        手动实现 A1X 到 Gello 的范围映射（从 A1_X.py._map_from_a1x 移植）。
        
        这是一个线性范围映射，将 A1X 关节空间映射到 Gello 关节空间。
        公式: t = (a1x - a1x_start) / (a1x_end - a1x_start)
              gello = gello_start + t * (gello_end - gello_start)
        """
        # Gello 关节范围（来自 A1_X.py）
        gello_range_start = np.array([-2.87, 0.0, 0.0, -1.57, -1.34, -2.0, 0.103], dtype=float)
        gello_range_end = np.array([2.87, 3.14, 3.14, 1.57, 1.34, 2.0, 1.0], dtype=float)
        
        # A1X 关节范围（必须与 _manual_gello_to_a1x_mapping 完全一致！）
        a1x_range_start = np.array([-2.880, 0.0, 0.0, 1.55, 1.521, -1.56, 2.0], dtype=float)
        a1x_range_end = np.array([2.880, 3.14, -2.95, -1.55, -1.52, 1.56, 99.0], dtype=float)
        
        # 确保输入是 numpy 数组
        a1x_joints = np.array(a1x_joints, dtype=float)
        
        # 确保有 7 个关节
        if len(a1x_joints) < 7:
            a1x_full = np.zeros(7)
            a1x_full[:len(a1x_joints)] = a1x_joints
            a1x_joints = a1x_full
        
        # Clip 输入到 A1X 范围（和原版代码一致）
        clipped = a1x_joints.copy()
        for i in range(7):
            lo = min(a1x_range_start[i], a1x_range_end[i])
            hi = max(a1x_range_start[i], a1x_range_end[i])
            clipped[i] = np.clip(a1x_joints[i], lo, hi)
        
        result = np.zeros(7, dtype=float)
        
        for i in range(7):
            out_start = a1x_range_start[i]
            out_end = a1x_range_end[i]
            in_start = gello_range_start[i]
            in_end = gello_range_end[i]
            
            # 计算归一化位置 t
            out_range = out_end - out_start
            if abs(out_range) < 1e-9:
                result[i] = in_start
            else:
                # 在 [out_start, out_end] 范围内归一化
                t = (clipped[i] - out_start) / out_range
                result[i] = in_start + t * (in_end - in_start)
        
        print(f"📐 [Manual] A1X -> Gello 映射:")
        print(f"   A1X 输入:  [{', '.join(f'{v:7.3f}' for v in a1x_joints)}]")
        print(f"   Gello 输出: [{', '.join(f'{v:7.3f}' for v in result)}]")
        
        return result
    
    def _start_following(self):
        """🔧 恢复：Enable Gello follower mode (robot controls Gello)"""
        try:
            if self.gello_follower is not None:
                self.gello_follower.start()
                print("🤖 Gello 进入跟随模式（GelloFollower）")
            else:
                print("⚠️  GelloFollower 未初始化（跳过跟随模式）")
        except Exception as e:
            print(f"⚠️  启动跟随模式失败: {e}")
            import traceback
            traceback.print_exc()
    
    def _stop_following(self):
        """🔧 恢复：Disable Gello follower mode (return to teleoperation)"""
        try:
            if self.gello_follower is not None:
                self.gello_follower.stop()
                print("✅ Gello 退出跟随模式（准备远程操控）")
            else:
                print("⚠️  GelloFollower 未初始化（跳过停止跟随）")
        except Exception as e:
            print(f"⚠️  停止跟随模式失败: {e}")
            import traceback
            traceback.print_exc()
    
    def _slow_follow_to_target(self, target_gello_joints: np.ndarray, duration: float = 0.5):
        """
        🔧 恢复：Move Gello to target position with smooth interpolation.
        
        Args:
            target_gello_joints: Target Gello joint positions [7]
            duration: Time to reach target (seconds)
        """
        try:
            # Get current Gello position
            current_pos = self._get_current_gello_joints()
            
            if current_pos is None:
                print("⚠️  无法获取当前 Gello 位置")
                return
            
            # Calculate movement parameters
            max_diff = np.max(np.abs(target_gello_joints[:6] - current_pos[:6]))
            
            # Adjust duration based on distance
            if max_diff < 0.2:
                duration = 0.1
            elif max_diff < 0.5:
                duration = 0.1
            elif max_diff < 1.5:
                duration = 0.5
            else:
                duration = 0.6
            
            # High control rate for smooth motion
            control_rate = 30 # Hz
            num_steps = max(int(duration * control_rate), 3)
            dt = duration / num_steps
            
            # Pre-compute delta
            delta = target_gello_joints - current_pos
            
            # Smooth motion
            for step in range(num_steps + 1):
                alpha = step / num_steps
                intermediate_pos = current_pos + alpha * delta
                # Send position command to Gello
                self._set_gello_joints(intermediate_pos)
                # Debug: 打印当前和目标位置
                actual_pos = self._get_current_gello_joints()
                print(f"[DEBUG] step {step:3d}/{num_steps:3d} | 当前: [{', '.join(f'{v:.3f}' for v in (actual_pos if actual_pos is not None else np.zeros_like(intermediate_pos)))}] | 目标: [{', '.join(f'{v:.3f}' for v in target_gello_joints)}]")
               # time.sleep(0.00001)
            #self._set_gello_joints(target_gello_joints, duration=0.2)###发送持续最终位置0.5s
            
        except Exception as e:
            print(f"⚠️  平滑移动失败: {e}")
            import traceback
            traceback.print_exc()
    # def _slow_follow_to_target(self, target_gello_joints: np.ndarray, duration: float = 0.5):
    #     """
    #     🚀 快速版：利用 Dynamixel 的 Profile Velocity，让电机自己平滑移动
    #     """
    #     try:
    #         current_pos = self._get_current_gello_joints()
    #         if current_pos is None:
    #             print("⚠️  无法获取当前 Gello 位置")
    #             return
            
    #         max_diff = np.max(np.abs(target_gello_joints[:6] - current_pos[:6]))
    #         print(f"   📏 距离: {max_diff:.3f} rad")
            
    #         # 直接发送目标位置（电机的 Profile Velocity 会自动平滑移动）
    #         self._set_gello_joints(target_gello_joints)
            
    #         # 根据距离等待电机到位
    #         # Profile Velocity = 100 大约是 0.229 * 100 = 22.9 RPM
    #         # 约 2.4 rad/s，所以 1 rad 需要约 0.4s
    #         if max_diff < 0.2:
    #             wait_time = 0.15
    #         elif max_diff < 0.5:
    #             wait_time = 0.25
    #         elif max_diff < 1.0:
    #             wait_time = 0.4
    #         else:
    #             wait_time = 0.5
            
    #         time.sleep(wait_time)
            
    #         # 验证是否到位
    #         final_pos = self._get_current_gello_joints()
    #         if final_pos is not None:
    #             final_diff = np.max(np.abs(target_gello_joints[:6] - final_pos[:6]))
    #             if final_diff > 0.1:
    #                 # 没到位，再等一下
    #                 time.sleep(0.2)
    #             print(f"   ✅ 完成，最终差异: {final_diff:.3f} rad")
            
    #     except Exception as e:
    #         print(f"⚠️  移动失败: {e}")
    
    def _get_current_gello_joints(self) -> Optional[np.ndarray]:
        """🔧 恢复：Get current Gello joint positions"""
        try:
            result = None
            method_used = "未知"
            
            # 方法1: 使用 GelloFollower 的 get_current_position
            if self.gello_follower is not None and hasattr(
                self.gello_follower, 'get_current_position'):
                result = self.gello_follower.get_current_position()
                method_used = "GelloFollower.get_current_position()"
            
            # 方法2: Access the underlying DynamixelRobot through GelloAgent
            if result is None:
                robot = None
                if hasattr(self.agent, '_robot'):
                    robot = self.agent._robot
                elif hasattr(self.agent, '_agent') and hasattr(
                    self.agent._agent, '_robot'):
                    robot = self.agent._agent._robot
                
                if robot is not None and hasattr(robot, 'get_joint_state'):
                    result = robot.get_joint_state()
                    method_used = "DynamixelRobot.get_joint_state()"
            
            # 方法3: try agent's act method (which reads joint state)
            if result is None and hasattr(self.agent, 'act'):
                result = self.agent.act(None)
                method_used = "GelloAgent.act()"
            
            if result is None:
                print("⚠️  无法获取 Gello 关节位置")
                return None
            
            # 🔍 调试：打印读取的位置和使用的方法
            if not hasattr(self, '_gello_read_count'):
                self._gello_read_count = 0
            self._gello_read_count += 1
            
            # 只在关键时刻打印（reset 时）
            if self._resetting or self._gello_read_count % 100 == 1:
                pos_str = ', '.join(f'{v:.3f}' for v in result[:6])
                print(f"🔍 [读取 #{self._gello_read_count}] "
                      f"方法: {method_used}, "
                      f"位置: [{pos_str}]")
            
            return result
            
        except Exception as e:
            print(f"⚠️  获取 Gello 关节位置失败: {e}")
            return None
    
    def _set_gello_joints(self, joint_positions: np.ndarray,
                          duration: float = 0.0, rate_hz: float = 20.0):
        """
        🔧 恢复：Set Gello joint positions (follower mode)
        
        Args:
            joint_positions: 目标关节位置
            duration: 持续发送时长（秒）。如果为 0，只发送一次
            rate_hz: 发送频率（Hz），默认 50Hz
        """
        try:
            if self.gello_follower is None:
                return
            
            if duration <= 0:
                # 单次发送（原有行为）
                self.gello_follower.command_follow(joint_positions)
            else:
                # 持续发送指定时长
                import time
                num_steps = int(duration * rate_hz)
                dt = 1.0 / rate_hz
                
                for _ in range(num_steps):
                    self.gello_follower.command_follow(joint_positions)
                    time.sleep(dt)
                
        except Exception:
            # Silent failure during smooth motion
            pass
    
    def _get_robot_joint_state(self, obs, info) -> Optional[np.ndarray]:
        """
        🔧 恢复：Extract robot's joint state from observation or info.
        
        Args:
            obs: Observation from environment
            info: Info dict from environment
            
        Returns:
            Robot joint positions [7], or None if unavailable
        """
        try:
            # Method 1: From info dict
            if 'joint_positions' in info:
                joint_pos = np.array(info['joint_positions'])
                print(f"   [调试] 从 info 获取关节位置: {joint_pos[:3]}...")
                return joint_pos
            
            # Method 2: From observation dict (direct key)
            if isinstance(obs, dict) and 'joint_positions' in obs:
                joint_pos = np.array(obs['joint_positions'])
                print(f"   [调试] 从 obs['joint_positions'] 获取: {joint_pos[:3]}...")
                return joint_pos
            
            # Method 3: From observation state
            if isinstance(obs, dict) and 'state' in obs:
                state = obs['state']
                if isinstance(state, np.ndarray):
                    # Assuming first 7 dimensions are joint positions
                    if state.shape[-1] >= 7:
                        joint_pos = state[..., :7].flatten()
                        print(f"   [调试] 从 obs['state'][:7] 获取: {joint_pos[:3]}...")
                        return joint_pos
            
            # Method 4: 递归查找所有包装的环境，检查 curr_joint_positions 属性
            env = self.env
            depth = 0
            while env is not None and depth < 10:
                if hasattr(env, 'curr_joint_positions'):
                    joint_pos = np.array(env.curr_joint_positions)
                    print(f"   [调试] 从 env.curr_joint_positions 获取 (深度 {depth}): {joint_pos[:3]}...")
                    return joint_pos
                
                # 尝试获取下一层
                if hasattr(env, 'env'):
                    env = env.env
                elif hasattr(env, 'unwrapped'):
                    env = env.unwrapped
                else:
                    break
                depth += 1
            
            # Method 5: From base env method
            if hasattr(self.env, 'get_joint_positions'):
                joint_pos = self.env.get_joint_positions()
                print(f"   [调试] 从 env.get_joint_positions() 获取: {joint_pos[:3]}...")
                return joint_pos
            
            print("   [调试] 无法从任何来源获取机器人关节状态")
            print(f"   [调试] obs keys: {obs.keys() if isinstance(obs, dict) else type(obs)}")
            print(f"   [调试] info keys: {info.keys() if isinstance(info, dict) else type(info)}")
            return None
            
        except Exception as e:
            print(f"⚠️  获取机器人关节状态失败: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _get_current_a1x_joints(self) -> Optional[np.ndarray]:
        """
        获取 A1X 机器人的当前实际关节位置
        
        Returns:
            当前关节位置 [7]，如果无法获取则返回 None
        """
        try:
            # 方法1: 从缓存的 robot 对象获取
            robot = self._get_cached_robot()
            if robot is not None and hasattr(robot, 'curr_joint_positions'):
                return robot.curr_joint_positions.copy()
            
            # 方法2: 从底层环境获取
            env = self._get_cached_base_env()
            if env is not None:
                # 更新关节状态
                if hasattr(env, '_update_curr_joint_state'):
                    env._update_curr_joint_state()
                
                # 获取位置
                if hasattr(env, 'curr_joint_positions'):
                    return env.curr_joint_positions.copy()
            
            # 方法3: 从最后观测获取
            if self.last_obs is not None and isinstance(self.last_obs, dict):
                if 'joint_positions' in self.last_obs:
                    return np.array(self.last_obs['joint_positions'])
            
            print("⚠️  无法获取 A1X 当前关节位置")
            return None
            
        except Exception as e:
            print(f"⚠️  获取 A1X 关节位置失败: {e}")
            return None
    
    def close(self):
        """Clean up resources."""
        # Stop keyboard listener
        if hasattr(self, 'keyboard_listener'):
            self.keyboard_listener.stop()
        
        # Cleanup Gello agent
        self._cleanup()
        
        # Close base environment
        super().close()


class DualSpacemouseIntervention(gym.ActionWrapper):
    def __init__(self, env, action_indices=None, gripper_enabled=True):
        super().__init__(env)

        self.gripper_enabled = gripper_enabled

        self.expert = SpaceMouseExpert()
        self.left1, self.left2, self.right1, self.right2 = False, False, False, False
        self.action_indices = action_indices

    def action(self, action: np.ndarray) -> np.ndarray:
        """
        Input:
        - action: policy action
        Output:
        - action: spacemouse action if nonezero; else, policy action
        """
        intervened = False
        expert_a, buttons = self.expert.get_action()
        self.left1, self.left2, self.right1, self.right2 = tuple(buttons)


        if self.gripper_enabled:
            if self.left1:  # close gripper
                left_gripper_action = np.random.uniform(-1, -0.9, size=(1,))
                intervened = True
            elif self.left2:  # open gripper
                left_gripper_action = np.random.uniform(0.9, 1, size=(1,))
                intervened = True
            else:
                left_gripper_action = np.zeros((1,))

            if self.right1:  # close gripper
                right_gripper_action = np.random.uniform(-1, -0.9, size=(1,))
                intervened = True
            elif self.right2:  # open gripper
                right_gripper_action = np.random.uniform(0.9, 1, size=(1,))
                intervened = True
            else:
                right_gripper_action = np.zeros((1,))
            expert_a = np.concatenate(
                (expert_a[:6], left_gripper_action, expert_a[6:], right_gripper_action),
                axis=0,
            )

        if self.action_indices is not None:
            filtered_expert_a = np.zeros_like(expert_a)
            filtered_expert_a[self.action_indices] = expert_a[self.action_indices]
            expert_a = filtered_expert_a

        if np.linalg.norm(expert_a) > 0.001:
            intervened = True

        if intervened:
            return expert_a, True
        return action, False

    def step(self, action):

        new_action, replaced = self.action(action)

        obs, rew, done, truncated, info = self.env.step(new_action)
        if replaced:
            info["intervene_action"] = new_action
        info["left1"] = self.left1
        info["left2"] = self.left2
        info["right1"] = self.right1
        info["right2"] = self.right2
        return obs, rew, done, truncated, info
    
    def reset(self, **kwargs):
        return self.env.reset(**kwargs)


class GripperPenaltyWrapper(gym.RewardWrapper):
    def __init__(self, env, penalty=0.1):
        super().__init__(env)
        assert env.action_space.shape == (7,)
        self.penalty = penalty
        self.last_gripper_pos = None

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.last_gripper_pos = obs["state"][0, 0]
        return obs, info

    def reward(self, reward: float, action) -> float:
        if (action[6] < -0.5 and self.last_gripper_pos > 0.95) or (
            action[6] > 0.5 and self.last_gripper_pos < 0.95
        ):
            return reward - self.penalty
        else:
            return reward

    def step(self, action):
        """Modifies the :attr:`env` :meth:`step` reward using :meth:`self.reward`."""
        observation, reward, terminated, truncated, info = self.env.step(action)
        if "intervene_action" in info:
            action = info["intervene_action"]
        reward = self.reward(reward, action)
        self.last_gripper_pos = observation["state"][0, 0]
        return observation, reward, terminated, truncated, info

class DualGripperPenaltyWrapper(gym.RewardWrapper):
    def __init__(self, env, penalty=0.1):
        super().__init__(env)
        assert env.action_space.shape == (14,)
        self.penalty = penalty
        self.last_gripper_pos_left = 0 #TODO: this assume gripper starts opened
        self.last_gripper_pos_right = 0 #TODO: this assume gripper starts opened
    
    def reward(self, reward: float, action) -> float:
        if (action[6] < -0.5 and self.last_gripper_pos_left==0):
            reward -= self.penalty
            self.last_gripper_pos_left = 1
        elif (action[6] > 0.5 and self.last_gripper_pos_left==1):
            reward -= self.penalty
            self.last_gripper_pos_left = 0
        if (action[13] < -0.5 and self.last_gripper_pos_right==0):
            reward -= self.penalty
            self.last_gripper_pos_right = 1
        elif (action[13] > 0.5 and self.last_gripper_pos_right==1):
            reward -= self.penalty
            self.last_gripper_pos_right = 0
        return reward
    
    def step(self, action):
        """Modifies the :attr:`env` :meth:`step` reward using :meth:`self.reward`."""
        observation, reward, terminated, truncated, info = self.env.step(action)
        if "intervene_action" in info:
            action = info["intervene_action"]
        reward = self.reward(reward, action)
        return observation, reward, terminated, truncated, info


class WaitWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.wait = False

    def reset(self, **kwargs):
        if self.wait:
            input("Press Enter to continue...")
        obs, info = self.env.reset(**kwargs)
        self.wait = False
        return obs, info
    
    def step(self, action):
        obs, rew, done, truncated, info = self.env.step(action)
        if rew:
            self.wait = True
        return obs, rew, done, truncated, info
    
class USBResetWrapper(gym.Wrapper, FrankaEnv):
    def __init__(self, env):
        super().__init__(env)
        self.success = False

    def reset(self, **kwargs):
        if self.success:
            requests.post(self.url + "update_param", json=self.config.PRECISION_PARAM)
            self._send_gripper_command(1.0)
            
            # Move above the target pose
            target = copy.deepcopy(self.config.TARGET_POSE)
            target[2] += 0.03
            self.interpolate_move(target, timeout=0.7)
            self.interpolate_move(self.config.TARGET_POSE, timeout=0.5)
            self._send_gripper_command(-1.0)

            self._update_currpos()
            reset_pose = copy.deepcopy(self.config.TARGET_POSE)
            reset_pose[1] += 0.04
            self.interpolate_move(reset_pose, timeout=0.5)
            # reset_pose[:2] += np.random.uniform(-0.01, 0.03, size=2)
            # self.interpolate_move(reset_pose, timeout=0.5)


        obs, info = self.env.reset(**kwargs)
        self.success = False
        return obs, info
    
    def step(self, action):
        obs, rew, done, truncated, info = self.env.step(action)
        self.success = info["succeed"]
        return obs, rew, done, truncated, info
    
    
class StackObsWrapper(gym.Wrapper):
    def __init__(self, env, num_stack=1):
        """
        A wrapper to stack observations over multiple time steps.

        Args:
            env: The environment to wrap.
            num_stack: Number of observations to stack.
        """
        super().__init__(env)
        self.num_stack = num_stack
        
        self.observation_space = self._stack_observation_space(env.observation_space)
        self._frames = {key: None for key in self.observation_space.spaces.keys()}
        
    def _stack_observation_space(self, obs_space):
        """Modify the observation space to support stacked frames."""
        stacked_spaces = {}
        for key, space in obs_space.spaces.items():
            if isinstance(space, Box):
                low = np.repeat(space.low, self.num_stack, axis=0)
                high = np.repeat(space.high, self.num_stack, axis=0)
                stacked_spaces[key] = Box(low=low, high=high, dtype=space.dtype)
            else:
                raise NotImplementedError(f"Stacking not implemented for {type(space)}")
        return Dict(stacked_spaces)
    
    def _get_stacked_obs(self):
        """Constructs the stacked observation."""
        return {key: np.stack(self._frames[key], axis=0) for key in self._frames.keys()}
    
    def reset(self, **kwargs):
        """Resets the environment and initializes the stacked frames."""
        obs, info = self.env.reset(**kwargs)
        self._frames = {key: [obs[key].squeeze(0)] * self.num_stack for key in self._frames.keys()}
        return self._get_stacked_obs(), info
    
    def step(self, action):
        """Steps through the environment and updates the stacked frames."""
        next_obs, reward, done, truncated, info = self.env.step(action)
        for key in self._frames.keys():
            self._frames[key].pop(0)  # Remove the oldest frame
            self._frames[key].append(next_obs[key].squeeze(0))  # Add the new frame
        return self._get_stacked_obs(), reward, done, truncated, info
        
        
