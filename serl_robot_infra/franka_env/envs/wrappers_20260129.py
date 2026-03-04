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

sigmoid = lambda x: 1 / (1 + np.exp(-x))

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
        if "intervene_action_eef" in info:
            info["intervene_action_eef"] = info["intervene_action_eef"][:6]
        return obs, rew, done, truncated, info
    
    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

    
class SpacemouseIntervention(gym.ActionWrapper):
    def __init__(self, env, action_indices=None):
        super().__init__(env)

        self.gripper_enabled = True
        if self.action_space.shape == (6,):
            self.gripper_enabled = False

        self.expert = SpaceMouseExpert()
        self.left, self.right = False, False
        self.action_indices = action_indices

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
                gripper_action = np.random.uniform(-1, -0.9, size=(1,))
                intervened = True
            elif self.right:  # open gripper
                gripper_action = np.random.uniform(0.9, 1, size=(1,))
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
            return expert_a, True

        return action, False

    def step(self, action):

        new_action, replaced = self.action(action)

        obs, rew, done, truncated, info = self.env.step(new_action)
        if replaced:
            info["intervene_action_eef"] = new_action
        info["left"] = self.left
        info["right"] = self.right
        return obs, rew, done, truncated, info


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
    
    Usage:
        env = A1XTaskEnv(...)  # 底层环境已连接机器人
        env = GelloIntervention(
            env,
            left_config_path="path/to/config.yaml",  # 只需要 agent 配置
        )
    """
    
    def __init__(
        self, 
        env, 
        left_config_path: str,
        right_config_path: Optional[str] = None,
        control_rate_hz: int =500,
        use_save_interface: bool = False,
        action_indices=None,
        always_intervene: bool = False,
        sync_on_reset: bool = True,
        reset_follow_duration: float = 0.5,
    ):
        """
        Args:
            env: Base environment (已连接机器人，不需要再创建连接)
            left_config_path: Path to YAML configuration (只使用 agent 部分)
            right_config_path: Path to right arm config (for bimanual, optional)
            control_rate_hz: Control loop frequency
            use_save_interface: Enable keyboard interface for saving data
            action_indices: Optional indices to filter which actions can be controlled
            always_intervene: If True, intervention is always enabled
            sync_on_reset: If True, Gello follows robot to reset position
            reset_follow_duration: Duration for Gello to follow robot during reset
        """
        super().__init__(env)
        
        import atexit
        import signal
        from pathlib import Path
        from omegaconf import OmegaConf
        import sys
        
        # Add gello_software to path
        gello_path = 'Gello/gello_software'
        if gello_path not in sys.path:
            sys.path.insert(0, gello_path)
        
        from gello.utils.launch_utils import instantiate_from_dict
        
        self.action_indices = action_indices
        self.control_rate_hz = control_rate_hz
        self.bimanual = right_config_path is not None
        self.always_intervene = always_intervene
        self.sync_on_reset = sync_on_reset
        self.reset_follow_duration = reset_follow_duration
        
        # 🔧 [新增] 添加精确控制频率（与 gello/env.py 的 Rate 类相同）
        # [旧代码] 无频率控制，导致控制不丝滑
        self._rate = Rate(control_rate_hz)
        
        # [新增] 缓存 robot 和 base_env，避免每次都遍历 wrapper 链
        # [旧代码] 每次 step 都调用 _get_robot()/_get_base_env() 遍历 wrapper 链
        self._cached_robot = None
        self._cached_base_env = None
        
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
        
        # 初始化观测缓存
        self.last_obs = None
        
        # 🔧 恢复：初始化 GelloFollower（用于双向控制）
        self.gello_follower = None
        if sync_on_reset:
            try:
                from gello.agents.gello_follower import GelloFollower
                
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
        if always_intervene:
            print(f"   🎮 始终干预模式")
        else:
            print(f"   🎮 按空格键切换Gello干预 (当前: {'启用' if self.intervention_enabled else '禁用'})")
        if sync_on_reset:
            print(f"   🔄 双向控制已启用：Reset 时 Gello 跟随机器人")
        else:
            print(f"   ⚪ 双向控制已禁用")
    
    def _on_key_press(self, key):
        """空格键按下 -> 切换干预状态（仅在非始终干预模式下生效）"""
        try:
            if key == keyboard.Key.space:
                if self.always_intervene:
                    print("⚠️  始终干预模式已启用，无法通过空格键切换")
                    return
                
                self.intervention_enabled = not self.intervention_enabled
                status = "🟢 启用" if self.intervention_enabled else "🔴 禁用"
                print(f"\n🎮 Gello干预已{status}")
        except AttributeError:
            pass
    
    def _cleanup(self):
        """Clean up Gello agent resources."""
        if self.cleanup_in_progress:
            return
        self.cleanup_in_progress = True
        
        print("Cleaning up Gello resources...")
        
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
        
        🔧 绝对位置控制模式：
        当干预时，直接将 Gello 位置映射到 A1X 并发送绝对位置命令，
        绕过环境的 delta action 机制，实现高效的实时跟随。
        """
        # Process action through intervention logic
        gello_joints, intervened = self.action(action)
        
        if intervened:
            # 🚀 绝对位置控制：直接发送目标位置到机器人
            try:
                # 1. 将 Gello 关节位置映射到 A1X 空间
                target_a1x_joints = self._gello_to_a1x_mapping(gello_joints)
                
                if target_a1x_joints is not None:
                    # 🔍 DEBUG: 打印夹爪信息（每100步打印一次）
                    if not hasattr(self, '_debug_step_count'):
                        self._debug_step_count = 0
                    self._debug_step_count += 1
                    if self._debug_step_count % 100 == 1:
                        print(f"🔍 [DEBUG] Gello gripper: {gello_joints[6]:.4f} -> A1X gripper: {target_a1x_joints[6]:.2f} mm")
                    
                    # 2. 直接发送绝对位置命令到机器人（绕过 env.step 的 delta 机制）
                    robot = self._get_robot()  # 每次遍历 wrapper 链，性能差
                    #robot = self._get_cached_robot()  # [新代码] 使用缓存，避免重复遍历
                    if robot is not None:
                        robot.command_joint_state(target_a1x_joints, from_gello=False)
                    
                    # 3. 等待控制周期（精确频率控制）
                    # [旧代码] time.sleep(1.0 / self.control_rate_hz)  # 简单 sleep，不精确
                    #self._rate.sleep()  # [新代码] 使用 Rate 类精确控制频率
                    
                    # 4. 获取观测（不通过 env.step）
                    env = self._get_base_env()  # 每次遍历 wrapper 链，性能差
                    #env = self._get_cached_base_env()  # [新代码] 使用缓存，避免重复遍历
                    if env is not None and hasattr(env, '_update_curr_joint_state'):
                        env._update_curr_joint_state()
                    if env is not None and hasattr(env, '_get_obs'):
                        obs = env._get_obs()
                    else:
                        obs = self.last_obs
                    
                    # 5. 计算奖励和终止条件
                    rew = 0
                    done = False
                    truncated = False
                    if env is not None and hasattr(env, 'compute_reward'):
                        rew = env.compute_reward(obs)
                        done = rew > 0
                    if env is not None and hasattr(env, 'curr_path_length'):
                        env.curr_path_length += 1
                        if hasattr(env, 'max_episode_length'):
                            done = done or (env.curr_path_length >= env.max_episode_length)
                    
                    info = {
                        "intervene_action_eef": target_a1x_joints,
                        "gello_intervened": True,
                        "gello_joints": gello_joints,
                        "succeed": rew > 0
                    }
                    
                    self.last_obs = obs
                    return obs, int(rew), done, truncated, info
                    
            except Exception as e:
                print(f"⚠️  绝对位置控制失败，回退到 delta 模式: {e}")
                import traceback
                traceback.print_exc()
                # Fall through to delta control
        
        # 非干预模式或绝对控制失败：使用原始的 delta 控制
        obs, rew, done, truncated, info = self.env.step(action)
        
        # 缓存观测
        self.last_obs = obs
        
        # 提供接口兼容信息
        info["gello_intervened"] = False
        
        return obs, rew, done, truncated, info
    
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
        """
        import time
        
        # Reset base environment
        obs, info = self.env.reset(**kwargs)
        time.sleep(1.5)  # Wait for stability
        
        # 🔧 缓存观测
        self.last_obs = obs
        
        # 🔧 恢复：Sync Gello to robot's reset position
        if self.sync_on_reset:
            try:
                # Get robot's reset joint state
                robot_joint_state = self._get_robot_joint_state(obs, info)
                
                if robot_joint_state is not None:
                    print(f"🤖 Robot reset position (A1X): [{', '.join(f'{v:.3f}' for v in robot_joint_state)}]")
                    print("🔄 计算 Gello 目标位置...")
                    
                    # Convert A1_X joints to Gello joints using inverse mapping
                    gello_target = self._a1x_to_gello_mapping(robot_joint_state)
                    
                    if gello_target is not None:
                        print(f"🎯 Gello 目标位置: [{', '.join(f'{v:.3f}' for v in gello_target)}]")
                        
                        # 获取当前 Gello 位置进行比较
                        current_gello = self._get_current_gello_joints()
                        if current_gello is not None:
                            print(f"📍 当前 Gello 位置: [{', '.join(f'{v:.3f}' for v in current_gello)}]")
                            diff = np.abs(gello_target - current_gello)
                            print(f"📏 位置差异: [{', '.join(f'{v:.3f}' for v in diff)}]")
                        
                        # Enable follower mode
                        self._start_following()
                        
                        # Move Gello to target position
                        print(f"⚡ 同步 Gello 到机器人位置 (用时 {self.reset_follow_duration}s)...")
                        self._slow_follow_to_target(gello_target, duration=self.reset_follow_duration)
                        
                        # Return to teleoperation mode
                        self._stop_following()
                        
                        # 验证最终位置
                        final_gello = self._get_current_gello_joints()
                        if final_gello is not None:
                            print(f"📍 最终 Gello 位置: [{', '.join(f'{v:.3f}' for v in final_gello)}]")
                            final_diff = np.abs(gello_target - final_gello)
                            print(f"📏 最终位置差异: [{', '.join(f'{v:.3f}' for v in final_diff)}]")
                        
                        print("✅ Gello 已同步。准备远程操控。")
                    else:
                        print("⚠️  无法计算 Gello 目标位置（跳过同步）")
                else:
                    print("⚠️  无法获取机器人关节状态（跳过同步）")
                    
            except Exception as e:
                print(f"⚠️  Gello 同步失败: {e}")
                import traceback
                traceback.print_exc()
                # Ensure we're back in teleoperation mode
                try:
                    self._stop_following()
                except:
                    pass
        
        return obs, info
    
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
                duration = 0.3
            elif max_diff < 1.5:
                duration = 0.5
            else:
                duration = 0.8
            
            # High control rate for smooth motion
            control_rate = 30 # Hz
            num_steps = max(int(duration * control_rate), 10)
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
            print(f"✅ Gello 已移动到目标位置")
            
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
            # 方法1: 使用 GelloFollower 的 get_current_position
            if self.gello_follower is not None and hasattr(self.gello_follower, 'get_current_position'):
                return self.gello_follower.get_current_position()
            
            # 方法2: Access the underlying DynamixelRobot through GelloAgent
            robot = None
            if hasattr(self.agent, '_robot'):
                robot = self.agent._robot
            elif hasattr(self.agent, '_agent') and hasattr(self.agent._agent, '_robot'):
                robot = self.agent._agent._robot
            
            if robot is not None and hasattr(robot, 'get_joint_state'):
                return robot.get_joint_state()
            
            # 方法3: try agent's act method (which reads joint state)
            if hasattr(self.agent, 'act'):
                return self.agent.act(None)
            
            print("⚠️  无法获取 Gello 关节位置")
            return None
        except Exception as e:
            print(f"⚠️  获取 Gello 关节位置失败: {e}")
            return None
    
    def _set_gello_joints(self, joint_positions: np.ndarray):
        """🔧 恢复：Set Gello joint positions (follower mode)"""
        try:
            if self.gello_follower is not None:
                self.gello_follower.command_follow(joint_positions)
            else:
                # Silent failure during smooth motion
                pass
        except Exception as e:
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
            info["intervene_action_eef"] = new_action
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
        if "intervene_action_eef" in info:
            action = info["intervene_action_eef"]
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
        if "intervene_action_eef" in info:
            action = info["intervene_action_eef"]
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
        
        
