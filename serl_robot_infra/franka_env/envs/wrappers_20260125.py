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


class GelloIntervention(gym.ActionWrapper):
    """
    Gello teleoperation intervention wrapper with bidirectional control.
    
    Features:
    - Normal mode: Human moves Gello → Robot follows (teleoperation)
    - Reset mode: Robot resets → Gello follows (synchronization)
    
    Allows human to control the robot via Gello device. When Gello is moved,
    its joint positions override the policy action. During reset, Gello
    automatically moves to match the robot's reset position.
    
    Usage:
        env = YourEnv()
        env = GelloIntervention(
            env, 
            port="/dev/ttyUSB0",
            sync_on_reset=True  # Enable Gello following during reset
        )
    """
    
    def __init__(
        self, 
        env, 
        port: str = "/dev/serial/by-id/usb-FTDI_USB__-__Serial_Converter_FTA7NNNU-if00-port0",
        action_indices=None,
        intervention_threshold: float = 0.01,
        sync_on_reset: bool = True,
        reset_follow_duration: float = 0.5,
        expert: Optional[GelloExpert] = None,
        use_absolute_control: bool = True,  # 新参数：使用绝对控制
    ):
        """
        Args:
            env: Base environment
            port: Serial port for Gello device (ignored if expert is provided)
            action_indices: Optional indices to filter which actions can be controlled（用于指定索引关节，none表示七关节全控制，）
            # 例如：只允许控制关节 0, 1, 2 和夹爪 (索引 6)
            env = GelloIntervention(
             env, 
            port=port,
            action_indices=[0, 1, 2, 6]  # 👈 传入允许控制的索引
            )
            # 结果：
            # - 关节 0, 1, 2 可以控制
            # - 关节 3, 4, 5 被强制为 0 (不动)
            # - 夹爪 (索引 6) 可以控制
            intervention_threshold: Minimum change to trigger intervention (rad)
            sync_on_reset: Whether to make Gello follow robot during reset
            reset_follow_duration: How long to keep Gello in follow mode during reset (seconds)
            expert: Optional pre-initialized GelloExpert instance (to avoid port conflicts)
            use_absolute_control: If True, directly send absolute joint commands to robot (bypassing env's delta mechanism)
        """
        super().__init__(env)
        
        # Check if gripper is part of action space
        # Action space: 6 DOF (arm only) or 7 DOF (arm + gripper)
        self.gripper_enabled = (self.action_space.shape[0] == 7)
        
        # Use provided expert or create new one
        if expert is not None:
            self.expert = expert
            self._owns_expert = False  # Don't close it in our close()
        else:
            self.expert = GelloExpert(port=port)
            self._owns_expert = True
        self.action_indices = action_indices
        self.intervention_threshold = intervention_threshold
        self.sync_on_reset = sync_on_reset
        self.reset_follow_duration = reset_follow_duration
        self.use_absolute_control = use_absolute_control
        
        # Track previous state to detect movement
        self.prev_joint_state = None
        
        # Track previous EEF pose for delta calculation (for demo recording)
        self.prev_eef_pose = None
        
        print(f"✅ GelloIntervention initialized")
        print(f"   - Gripper enabled: {self.gripper_enabled}")
        print(f"   - Control mode: {'ABSOLUTE (direct mapping)' if use_absolute_control else 'DELTA (via env)'}")
        print(f"   - Sync on reset: {sync_on_reset}")
        if sync_on_reset:
            print(f"   - Reset follow duration: {reset_follow_duration}s")
    
    def action(self, action: np.ndarray) -> Tuple[np.ndarray, bool]:
        """
        Input:
        - action: policy action (delta joint positions)
        Output:
        - action: gello delta action if moved; else, policy action
        - intervened: whether intervention occurred
        
        Note on data recording:
        - Execution: Gello absolute joints → A1X absolute joints → delta for env
        - Recording: Current EEF pose → Gello causes movement → New EEF pose → delta EEF + absolute gripper
        
        This allows policy to learn in task space (delta EEF) rather than joint space.
        """
        # Get Gello absolute position (in Gello coordinate space)
        gello_absolute_pos = self.expert.get_action()
        
        # Always convert Gello position to action (continuous teleoperation)
        # This ensures robot maintains position even when Gello is stationary
        intervened = True  # Always in teleoperation mode when wrapper is active
        
        if intervened:
            # Get current robot position (A1X coordinates)
            current_robot_pos = self._get_current_robot_position()
            
            if current_robot_pos is not None:
                # Map Gello position to A1X coordinates (absolute target)
                target_a1x_pos = self._gello_to_a1x_mapping(gello_absolute_pos)
                
                # Compute delta joint action for execution (what A1XEnv expects)
                delta_action = target_a1x_pos - current_robot_pos
                
                # Get action scale from environment (if available)
                action_scale = self._get_action_scale()
                if action_scale is not None:
                    delta_action = delta_action / action_scale
                
                expert_a = delta_action[:6]  # First 6 joints (arm only)
                
                # Handle gripper if enabled - use mapped absolute value directly
                if self.gripper_enabled and len(target_a1x_pos) >= 7:
                    # Use the 7th dimension from mapping as absolute gripper value
                    gripper_absolute = target_a1x_pos[6]
                    expert_a = np.concatenate((expert_a, [gripper_absolute]), axis=0)
            else:
                # Fallback: use raw delta (less accurate but better than nothing)
                if self.prev_joint_state is not None:
                    expert_a = (gello_absolute_pos - self.prev_joint_state)[:6]
                else:
                    expert_a = np.zeros(6)
                
                # Fallback gripper handling
                if self.gripper_enabled:
                    gripper_val = gello_absolute_pos[6] if len(gello_absolute_pos) >= 7 else 0.0
                    expert_a = np.concatenate((expert_a, [gripper_val]), axis=0)
        
        # Ensure correct dimensions match action space
        expected_dim = self.action_space.shape[0]
        if len(expert_a) != expected_dim:
            # Pad or truncate to correct size
            if len(expert_a) < expected_dim:
                expert_a = np.concatenate([expert_a, np.zeros(expected_dim - len(expert_a))])
            else:
                expert_a = expert_a[:expected_dim]
        
        # Apply action filtering if specified
        if self.action_indices is not None:
            filtered_expert_a = np.zeros_like(expert_a)
            filtered_expert_a[self.action_indices] = expert_a[self.action_indices]
            expert_a = filtered_expert_a
        
        # Always return Gello action (continuous teleoperation mode)
        return expert_a, True
    
    def step(self, action):
        # Get current EEF pose BEFORE action execution
        current_eef_pose = self._get_current_eef_pose()
        
        # Get Gello target position
        gello_absolute_pos = self.expert.get_action()
        target_a1x_pos = self._gello_to_a1x_mapping(gello_absolute_pos)
        
        # Choose control mode
        if self.use_absolute_control:
            # 🚀 ABSOLUTE CONTROL MODE: 直接发送绝对位置命令到机器人
            # 绕过环境的 delta 机制，直接控制机器人
            robot = self._get_robot()
            if robot is not None:
                # 直接命令机器人移动到目标位置
                robot.command_joint_state(target_a1x_pos, from_gello=False)
                
                # 手动更新环境状态（因为我们绕过了 env.step）
                self._manual_update_env_state()
                
                # 获取观测
                obs = self._get_obs()
                rew = self._compute_reward(obs)
                done = self._check_done(obs, rew)
                info = {"succeed": rew}
                
                # 记录遥控动作（用于数据采集）
                replaced = True
            else:
                # Fallback: 如果获取不到 robot，使用 delta 模式
                print("⚠️  Cannot access robot directly, falling back to delta mode")
                new_action, replaced = self.action(action)
                obs, rew, done, truncated, info = self.env.step(new_action)
        else:
            # DELTA CONTROL MODE: 使用环境的 delta 机制
            new_action, replaced = self.action(action)
            obs, rew, done, truncated, info = self.env.step(new_action)
        
        # 记录 EEF delta 动作（用于训练数据）
        if replaced:
            # Store absolute joint command for reference
            info["intervene_action_joint_absolute"] = target_a1x_pos
            
            # Get NEW EEF pose AFTER action execution
            new_eef_pose = self._get_current_eef_pose()
            
            # Calculate delta EEF action (what should be saved for demos)
            if current_eef_pose is not None and new_eef_pose is not None:
                delta_eef = self._compute_delta_eef(current_eef_pose, new_eef_pose)
                
                # Add absolute gripper value
                if self.gripper_enabled and len(target_a1x_pos) >= 7:
                    gripper_val = target_a1x_pos[6]  # 绝对夹爪值
                    delta_eef_action = np.concatenate([delta_eef, [gripper_val]])
                else:
                    delta_eef_action = delta_eef
                
                info["intervene_action_eef"] = delta_eef_action
        
        return obs, rew, done, False, info
    
    def reset(self, **kwargs):
        """
        Reset environment and optionally sync Gello to robot's reset position.
        
        This implements bidirectional control:
        1. Robot resets to initial position
        2. Gello follows robot to match the reset position
        3. After sync, Gello returns to free-wheeling mode for teleoperation
        """
        import time
        
        # Reset tracking state
        self.prev_joint_state = None
        
        # Call base environment reset
        obs, info = self.env.reset(**kwargs)
        time.sleep(1.5)  # Wait a moment for stability
        
        # Sync Gello to robot's reset position using inverse mapping
        if self.sync_on_reset and self.expert.initialized:
            try:
                # Get robot's reset joint state (A1_X joints)
                robot_joint_state = self._get_robot_joint_state(obs, info)
                print(f"🤖 Robot reset joint state (A1X): {robot_joint_state}")
                
                if robot_joint_state is not None:
                    print("🔄 Computing Gello target from A1X position...")
                    
                    # Convert A1_X joints to Gello joints using inverse mapping
                    gello_target = self._a1x_to_gello_mapping(robot_joint_state)
                    
                    if gello_target is not None:
                        print(f"🎯 Target Gello position: {gello_target}")
                        
                        # Enable follower mode
                        self.expert.start_following()
                        print("🤖 Gello started following mode")
                        
                        # Move Gello to target position (fast sync)
                        print("⚡ Syncing Gello to robot position...")
                        self._slow_follow_to_target(gello_target, duration=self.reset_follow_duration)
                        
                        # Return to teleoperation mode
                        self.expert.stop_following()
                        
                        print("✅ Gello synced. Ready for teleoperation.")
                        
                        # Update initial joint state for intervention detection
                        self.prev_joint_state = self.expert.get_joint_state()
                    else:
                        print("⚠️  Failed to compute Gello target position")
                
            except Exception as e:
                print(f"⚠️  Failed to sync Gello during reset: {e}")
                import traceback
                traceback.print_exc()
                # Ensure we're back in teleoperation mode
                if self.expert.is_following():
                    self.expert.stop_following()
        
        return obs, info
    
    def _a1x_to_gello_mapping(self, a1x_joints: np.ndarray) -> Optional[np.ndarray]:
        """
        Convert A1_X joint positions to Gello joint positions using inverse mapping.
        
        Args:
            a1x_joints: A1_X joint positions [7]
            
        Returns:
            Gello joint positions [7], or None if mapping fails
        """
        try:
            # Try to get the A1XRobot instance from the environment
            # Navigate through wrappers to find the base environment
            env = self.env
            while hasattr(env, 'env'):
                env = env.env
            
            # Check if environment has robot with inverse mapping capability
            if hasattr(env, 'robot'):
                robot = env.robot
                if hasattr(robot, '_map_from_a1x'):
                    return robot._map_from_a1x(a1x_joints)
            
            # Fallback: try importing A1XRobot and using its mapping
            try:
                import sys
                gello_path = 'Gello/gello_software'
                if gello_path not in sys.path:
                    sys.path.insert(0, gello_path)
                from gello.robots.A1_X import A1XRobot
                
                # Create temporary instance just for mapping
                temp_robot = A1XRobot.__new__(A1XRobot)
                return temp_robot._map_from_a1x(a1x_joints)
            except Exception as e:
                print(f"⚠️  Fallback mapping failed: {e}")
                return None
                
        except Exception as e:
            print(f"⚠️  Error in A1X to Gello mapping: {e}")
            return None
    
    def _slow_follow_to_target(self, target_gello_joints: np.ndarray, duration: float = 0.5):
        """
        Move Gello to target position with smooth interpolation.
        
        Args:
            target_gello_joints: Target Gello joint positions [7]
            duration: Time to reach target (seconds) - default 0.5s for fast response
        """
        import time
        
        # Get current Gello position
        current_pos = self.expert.get_joint_state()
        
        # Calculate maximum joint difference
        max_diff = np.max(np.abs(target_gello_joints[:6] - current_pos[:6]))
        
        # Much faster: use fixed duration regardless of distance
        # Small movements: 0.3s, medium: 0.5s, large: 0.8s max
        if max_diff < 0.2:  # Very small movement (~11 degrees)
            duration = 0.3
        elif max_diff < 0.5:  # Small to medium (~29 degrees)
            duration = 0.5
        elif max_diff < 1.5:  # Medium to large (~86 degrees)
            duration = 0.7
        else:  # Very large movements
            duration = 1.0  # Still much faster than before (was 2.0+)
        
        # High control rate for smooth, fast motion (100 Hz)
        control_rate = 100
        num_steps = max(int(duration * control_rate), 10)  # At least 10 steps
        dt = duration / num_steps
        
        # Pre-compute delta for efficiency
        delta = target_gello_joints - current_pos
        
        # Fast motion without verbose logging
        for step in range(num_steps + 1):
            t = step / num_steps
            
            # Smooth interpolation (ease-in-out cubic)
            t_smooth = 3 * t**2 - 2 * t**3
            
            interpolated_pos = current_pos + t_smooth * delta
            
            # Command Gello to follow (silent errors)
            try:
                self.expert.command_follow(interpolated_pos)
            except:
                pass  # Ignore errors for speed
            
            if step < num_steps:  # Skip sleep on last iteration
                time.sleep(dt)

    def _get_current_robot_position(self) -> Optional[np.ndarray]:
        """
        Get current robot position (A1X coordinates) from the environment.
        
        Returns:
            Current A1X joint positions [7], or None if unavailable
        """
        try:
            # Navigate to base environment
            env = self.env
            while hasattr(env, 'env'):
                env = env.env
            
            # Try to get current position from A1XEnv
            if hasattr(env, 'curr_joint_positions'):
                return np.array(env.curr_joint_positions[:7])
            
            # Fallback: try from robot directly
            if hasattr(env, 'robot'):
                robot = env.robot
                if hasattr(robot, 'get_joint_state'):
                    return robot.get_joint_state()[:7]
            
            return None
        except Exception as e:
            print(f"⚠️  Failed to get current robot position: {e}")
            return None
    
    def _get_robot(self):
        """Get direct access to robot instance for absolute control."""
        try:
            env = self.env
            while hasattr(env, 'env'):
                env = env.env
            
            if hasattr(env, 'robot'):
                return env.robot
            return None
        except Exception as e:
            print(f"⚠️  Failed to get robot: {e}")
            return None
    
    def _manual_update_env_state(self):
        """手动更新环境状态（当直接控制机器人时需要）"""
        try:
            env = self.env
            while hasattr(env, 'env'):
                env = env.env
            
            # 更新 A1XEnv 的内部状态
            if hasattr(env, '_update_curr_joint_state'):
                env._update_curr_joint_state()
            
            # 更新路径长度
            if hasattr(env, 'curr_path_length'):
                env.curr_path_length += 1
                
        except Exception as e:
            print(f"⚠️  Failed to update env state: {e}")
    
    def _get_obs(self):
        """获取当前观测"""
        try:
            env = self.env
            while hasattr(env, 'env'):
                env = env.env
            
            if hasattr(env, '_get_obs'):
                return env._get_obs()
            return {}
        except Exception as e:
            print(f"⚠️  Failed to get obs: {e}")
            return {}
    
    def _compute_reward(self, obs):
        """计算奖励"""
        try:
            env = self.env
            while hasattr(env, 'env'):
                env = env.env
            
            if hasattr(env, 'compute_reward'):
                return env.compute_reward(obs)
            return 0
        except:
            return 0
    
    def _check_done(self, obs, reward):
        """检查是否结束"""
        try:
            env = self.env
            while hasattr(env, 'env'):
                env = env.env
            
            if hasattr(env, 'curr_path_length') and hasattr(env, 'max_episode_length'):
                return env.curr_path_length >= env.max_episode_length or reward
            return False
        except:
            return False
    
    def _get_action_scale(self) -> Optional[np.ndarray]:
        """Get action scale from environment."""
        try:
            env = self.env
            while hasattr(env, 'env'):
                env = env.env
            
            if hasattr(env, 'action_scale'):
                return np.array(env.action_scale)
            
            return np.ones(7)  # Default scale
        except:
            return np.ones(7)
    
    def _gello_to_a1x_mapping(self, gello_joints: np.ndarray) -> np.ndarray:
        """
        Convert Gello joint positions to A1_X joint positions.
        
        Uses the robot's forward mapping if available.
        
        Args:
            gello_joints: Gello joint positions [7]
            
        Returns:
            A1_X joint positions [7]
        """
        try:
            # Navigate to base environment
            env = self.env
            while hasattr(env, 'env'):
                env = env.env
            
            # Try to use robot's mapping method
            if hasattr(env, 'robot'):
                robot = env.robot
                if hasattr(robot, '_map_to_a1x'):
                    return robot._map_to_a1x(gello_joints)
            
            # Fallback: direct copy (assume same coordinate system)
            print("⚠️  No Gello->A1X mapping found, using direct copy")
            return gello_joints
            
        except Exception as e:
            print(f"⚠️  Error in Gello->A1X mapping: {e}")
            return gello_joints
    
    def _get_current_eef_pose(self) -> Optional[np.ndarray]:
        """
        Get current end-effector pose from robot.
        
        Returns:
            np.ndarray: [x, y, z, qx, qy, qz, qw] or None if unavailable
        """
        try:
            # Navigate to base environment
            env = self.env
            while hasattr(env, 'env'):
                env = env.env
            
            # Try to get from robot
            if hasattr(env, 'robot'):
                robot = env.robot
                if hasattr(robot, 'get_eef_pose'):
                    pos, quat = robot.get_eef_pose()
                    return np.concatenate([pos, quat])  # [x, y, z, qx, qy, qz, qw]
            
            return None
        except Exception as e:
            print(f"⚠️  Failed to get EEF pose: {e}")
            return None
    
    def _compute_delta_eef(self, prev_pose: np.ndarray, curr_pose: np.ndarray) -> np.ndarray:
        """
        Compute delta end-effector pose between two poses.
        
        Args:
            prev_pose: Previous EEF pose [x, y, z, qx, qy, qz, qw]
            curr_pose: Current EEF pose [x, y, z, qx, qy, qz, qw]
            
        Returns:
            np.ndarray: Delta pose [dx, dy, dz, drx, dry, drz] where rotation is euler angles
        """
        from scipy.spatial.transform import Rotation as R
        
        # Position delta (straightforward)
        delta_pos = curr_pose[:3] - prev_pose[:3]
        
        # Rotation delta (requires quaternion math)
        # delta_rot = curr_rot * prev_rot^{-1}
        prev_rot = R.from_quat(prev_pose[3:])  # [qx, qy, qz, qw]
        curr_rot = R.from_quat(curr_pose[3:])
        
        # Compute relative rotation
        delta_rot = curr_rot * prev_rot.inv()
        
        # Convert to euler angles (XYZ convention)
        delta_euler = delta_rot.as_euler('xyz')
        
        return np.concatenate([delta_pos, delta_euler])
    
    def _get_robot_joint_state(self, obs, info) -> Optional[np.ndarray]:
        """
        Extract robot's joint state from observation or info.
        
        This is environment-specific. Override if needed.
        """
        try:
            # Try to get from observation (common key names)
            if isinstance(obs, dict):
                for key in ['joint_positions', 'q', 'qpos', 'joint_state']:
                    if key in obs:
                        joint_state = np.array(obs[key])
                        # Ensure it's 7-DOF (6 joints + gripper)
                        if len(joint_state) >= 7:
                            return joint_state[:7]
            
            # Try to get from info
            if 'joint_positions' in info:
                joint_state = np.array(info['joint_positions'])
                if len(joint_state) >= 7:
                    return joint_state[:7]
            
            # Try to access from unwrapped environment
            if hasattr(self.env, 'unwrapped'):
                unwrapped = self.env.unwrapped
                
                # For A1X environment
                if hasattr(unwrapped, 'curr_joint_positions'):
                    return np.array(unwrapped.curr_joint_positions)
                
                # For Franka environment (convert from other representations if needed)
                if hasattr(unwrapped, 'robot'):
                    if hasattr(unwrapped.robot, 'get_joint_state'):
                        return unwrapped.robot.get_joint_state()
            
            print("⚠️  Could not extract robot joint state")
            return None
            
        except Exception as e:
            print(f"⚠️  Error extracting robot joint state: {e}")
            return None
    
    def close(self):
        # Only close expert if we created it ourselves
        if hasattr(self, 'expert') and hasattr(self, '_owns_expert') and self._owns_expert:
            self.expert.close()
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
        
        
