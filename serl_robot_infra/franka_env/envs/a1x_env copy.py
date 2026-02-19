"""Gym Interface for A1_X Robot"""
import os
import numpy as np
import gymnasium as gym
import cv2
import copy
import time
import queue
import threading
from datetime import datetime
from collections import OrderedDict
from typing import Dict

from franka_env.camera.video_capture import VideoCapture
from franka_env.camera.rs_capture import RSCapture
from franka_env.robots.a1x_robot import A1XRobot


class ImageDisplayer(threading.Thread):
    def __init__(self, queue, name):
        threading.Thread.__init__(self)
        self.queue = queue
        self.daemon = True
        self.name = name

    def run(self):
        while True:
            img_array = self.queue.get()
            if img_array is None:
                break

            frame = np.concatenate(
                [cv2.resize(v, (128, 128)) for k, v in img_array.items() if "full" not in k], axis=1
            )

            cv2.imshow(self.name, frame)
            cv2.waitKey(1)


##############################################################################


class DefaultA1XEnvConfig:
    """Default configuration for A1XEnv. Fill in the values below."""

    # A1_X Robot Configuration
    A1X_NUM_DOFS: int = 7
    A1X_NODE_NAME: str = "a1x_serl_node"
    A1X_PORT: int = 6100
    A1X_PYTHON_PATH: str = "/usr/bin/python3"
    # Optional external CuRobo IK service (e.g. tcp://127.0.0.1:6202)
    A1X_CUROBO_IK_SERVICE: str | None = None
    
    # Camera Configuration
    REALSENSE_CAMERAS: Dict = {
        "wrist_1": "130322274175",
        "wrist_2": "127122270572",
    }
    IMAGE_CROP: dict[str, callable] = {}
    
    # Task Configuration
    TARGET_JOINT_STATE: np.ndarray = np.zeros((7,))  # Target joint positions
    RESET_JOINT_STATE: np.ndarray = np.zeros((7,))  # Reset joint positions
    REWARD_THRESHOLD: np.ndarray = np.ones((7,)) * 0.1  # Joint position tolerance
    
    # Control Configuration
    ACTION_SCALE: np.ndarray = np.ones((7,))  # Scaling for joint actions
    
    # Display Configuration
    DISPLAY_IMAGE: bool = True
    MAX_EPISODE_LENGTH: int = 100
    
    # Random reset
    RANDOM_RESET: bool = False


##############################################################################


class A1XEnv(gym.Env):
    """Gymnasium environment for A1_X robot with joint space control."""
    
    def __init__(
        self,
        hz=10,
        fake_env=False,
        save_video=True,
        config: DefaultA1XEnvConfig = None,
    ):
        self.config = config if config is not None else DefaultA1XEnvConfig()
        self.action_scale = self.config.ACTION_SCALE
        self._TARGET_JOINT_STATE = self.config.TARGET_JOINT_STATE
        self._RESET_JOINT_STATE = self.config.RESET_JOINT_STATE
        self._REWARD_THRESHOLD = self.config.REWARD_THRESHOLD
        self.max_episode_length = self.config.MAX_EPISODE_LENGTH
        self.display_image = self.config.DISPLAY_IMAGE
        self.randomreset = self.config.RANDOM_RESET
        self.use_gripper = self.config.USE_GRIPPER if hasattr(self.config, 'USE_GRIPPER') else True
        self.hz = hz
        

        if save_video:
            print("Saving videos!")
        self.save_video = save_video
        self.recording_frames = []

        # Action/Observation Space
        # Action: delta joint positions (7 DOF)
        # self.action_space = gym.spaces.Box(
        #     np.ones((7,), dtype=np.float32) * -0.01,
        #     np.ones((7,), dtype=np.float32) * 0.01,
        # )
        
        self.action_space = gym.spaces.Box(
            low=np.array([
                -0.005, -0.005, -0.005,   # 前三维
                -0.1, -0.1, -0.1,  # 第4-6维
                -0.2                # 最后一维
            ], dtype=np.float32),
            high=np.array([
                0.005,  0.005,  0.005,
                0.1, 0.1, 0.1,
                0.2
            ], dtype=np.float32),
            dtype=np.float32
        )

        self.observation_space = gym.spaces.Dict(
            {
                "state": gym.spaces.Dict(
                    {
                        "joint_positions": gym.spaces.Box(-np.inf, np.inf, shape=(7,)),
                        "joint_velocities": gym.spaces.Box(-np.inf, np.inf, shape=(7,)),
                        "ee_pos_rot_gripper": gym.spaces.Box(-np.inf, np.inf, shape=(7,)),
                        "gripper_position": gym.spaces.Box(-100, 100, shape=(1,)),
                    }
                ),
                "images": gym.spaces.Dict(
                    {key: gym.spaces.Box(0, 255, shape=(256, 256, 3), dtype=np.uint8) if '256' in key
                        else gym.spaces.Box(0, 255, shape=(128, 128, 3), dtype=np.uint8)
                        for key in self.config.REALSENSE_CAMERAS}
                ),
            }
        )

        self.curr_path_length = 0
        
        # Initialize state variables
        self.curr_joint_positions = None
        self.curr_joint_velocities = None
        self.curr_ee_pos_rot_gripper = None
        self.curr_gripper_position = None
        
        # 保存reset时的旋转值（rx, ry），用于在step中固定这两个维度
        self.reset_ee_rotation = None

        if fake_env:
            return

        # Initialize robot
        print("Initializing A1_X robot...")
        # 🔧 从配置中读取IK选项
        use_curobo_ik = getattr(self.config, 'USE_CUROBO_IK', False)
        self.robot = A1XRobot(
            num_dofs=self.config.A1X_NUM_DOFS,
            node_name=self.config.A1X_NODE_NAME,
            port=self.config.A1X_PORT,
            python_path=self.config.A1X_PYTHON_PATH,
            use_curobo_ik=use_curobo_ik,
            curobo_ik_service=getattr(self.config, "A1X_CUROBO_IK_SERVICE", None),
            reset_joint_state=self._RESET_JOINT_STATE
        )

        # Initialize cameras
        self.cap = None
        self.init_cameras(self.config.REALSENSE_CAMERAS)
        if self.display_image:
            self.img_queue = queue.Queue()
            self.displayer = ImageDisplayer(self.img_queue, "A1_X Camera")
            self.displayer.start()

        # Initialize keyboard listener for emergency stop
        if not fake_env:
            from pynput import keyboard
            self.terminate = False

            def on_press(key):
                if key == keyboard.Key.esc:
                    self.terminate = True
            self.listener = keyboard.Listener(on_press=on_press)
            self.listener.start()

        # Get current joint state
        self._update_curr_joint_state()

        print("Initialized A1_X Environment")

    def step(self, action: np.ndarray) -> tuple:
        """Standard gym step function with EEF delta control.
        
        Args:
            action: 7D array [delta_x, delta_y, delta_z, delta_rx, delta_ry, delta_rz, delta_gripper]
                   - First 3: delta position (m)
                   - Next 3: delta rotation (euler angles in radians)
                   - Last 1: delta gripper position (normalized, will be scaled)
        """
        start_time = time.time()
        # haoyuan debug
        action = np.clip(action, self.action_space.low, self.action_space.high)        
        # Scale action
        scaled_action = action * self.action_scale
        
        # 🛡️ 零动作检测：如果动作完全为零，跳过机器人控制
        # if np.allclose(scaled_action, 0.0, atol=1e-6):
        #     # 动作为零，不发送任何命令，直接返回当前观测
        #     obs = self._get_obs()
        #     self.curr_path_length += 1
        #     done = self.curr_path_length >= self.max_episode_length
        #     reward = 0.0
        #     info = {"intervene_action_eef": action}
        #     return obs, reward, done, False, info
        
        # print(f"Raw action: {action}, Scaled action: {scaled_action}")
        # For gripper: convert normalized delta to absolute position (mm)
        # self.curr_ee_pos_rot_gripper[6] is current gripper in [0, 1]
        current_gripper_normalized = self.curr_ee_pos_rot_gripper[6]
        delta_gripper_normalized = scaled_action[6]
        new_gripper_normalized = np.clip(current_gripper_normalized + delta_gripper_normalized, 0.0, 1.0)
        new_gripper_mm = new_gripper_normalized * 100.0  # Convert to [0, 100] mm
        
        # 🔒 固定夹爪位置：始终设置为 1.5mm，忽略任何夹爪命令
        if not self.use_gripper:
            new_gripper_mm = 1.5
        
        # 🔓 允许旋转控制：SpaceMouse 可以控制完整的6自由度
        # 如果需要固定旋转，取消下面两行注释：
        scaled_action = scaled_action.copy()
        scaled_action[3:6] = 0.0  # 强制 drx, dry, drz = 0
        
        # Construct EEF delta command: [dx, dy, dz, drx, dry, drz, gripper_absolute_mm]
        eef_command = np.concatenate([scaled_action[:6], [new_gripper_mm]])
        
        # Debug print
        # print(f"EEF delta: pos={scaled_action[:3]}, rot={scaled_action[3:6]}, gripper: {current_gripper_normalized:.3f} -> {new_gripper_normalized:.3f} ({new_gripper_mm:.1f}mm)")
        
        # Send EEF command to robot with intelligent waiting
        cmd_send_time = time.time()
        result = self.robot.command_eef_pose(
            eef_command, 
            wait_for_completion=True,  # 智能等待执行到位
            timeout=2.0  # 2秒超时
        )
        
        self.curr_path_length += 1
        
        # 智能等待已包含在command_eef_pose中，不需要额外延迟
        dt = time.time() - start_time
        remaining_time = (1.0 / self.hz) - dt
        if remaining_time > 0:
            time.sleep(remaining_time)
        
        # Update state (此时机器人应该已经到位)
        state_read_time = time.time()
        self._update_curr_joint_state()
        ob = self._get_obs()
        
        # 诊断输出
        if result:
            reached_str = "✓" if result.get('reached', False) else "✗"
            error_mm = result.get('final_error', 0) * 1000
            # print(f"⏱️  {reached_str} 执行耗时={(state_read_time - cmd_send_time)*1000:.0f}ms, 误差={error_mm:.1f}mm")
        else:
            print(f"⏱️  命令→状态读取 = {(state_read_time - cmd_send_time)*1000:.1f}ms")
        
        reward = self.compute_reward(ob)
        done = self.curr_path_length >= self.max_episode_length or reward or self.terminate
        print(f"Step done: {done}, reward: {reward}, path length: {self.curr_path_length}, terminate: {self.terminate}")
        
        return ob, int(reward), done, False, {"succeed": reward}

    # def compute_reward(self, obs) -> bool:
    #     """Compute reward based on distance to target joint state."""
    #     current_joints = obs["state"]["joint_positions"]
    #     delta = np.abs(current_joints - self._TARGET_JOINT_STATE)
        
    #     if np.all(delta < self._REWARD_THRESHOLD):
    #         return True
    #     else:
    #         return False
    
    def compute_reward(self, obs) -> bool:
        return False    

    def get_im(self) -> Dict[str, np.ndarray]:
        """Get images from the realsense cameras."""
        images = {}
        display_images = {}
        full_res_images = {}
        
        for key, cap in self.cap.items():
            try:
                rgb = cap.read()
                cropped_rgb = self.config.IMAGE_CROP[key](rgb) if key in self.config.IMAGE_CROP else rgb
                resized = cv2.resize(
                    cropped_rgb, self.observation_space["images"][key].shape[:2][::-1]
                )
                images[key] = resized[..., ::-1]
                display_images[key] = resized
                display_images[key + "_full"] = cropped_rgb
                full_res_images[key] = copy.deepcopy(cropped_rgb)
            except queue.Empty:
                input(f"{key} camera frozen. Check connection, then press enter to relaunch...")
                cap.close()
                self.init_cameras(self.config.REALSENSE_CAMERAS)
                return self.get_im()

        # Store full resolution cropped images separately
        self.recording_frames.append(full_res_images)

        if self.display_image:
            self.img_queue.put(display_images)
        
        return images

    def interpolate_move(self, goal_joints: np.ndarray, timeout: float):
        """Move the robot to the goal joint positions with linear interpolation."""
        steps = int(timeout * self.hz)  # Use timeout to calculate steps
        self._update_curr_joint_state()
        path = np.linspace(self.curr_joint_positions, goal_joints, steps)
        print("CURRENT JOINTS:", self.curr_joint_positions)
        print(f"Interpolating move TO {goal_joints} over {steps} steps ({timeout}s).")
        
        for joint_positions in path:
            # These are A1X native joint positions, no Gello mapping needed
            self.robot.command_joint_state(joint_positions, from_gello=False)
            time.sleep(1 / self.hz)
        
        # Wait for robot to settle at final position
        time.sleep(5.0)
        
       # 🔧 Reset 后关闭夹爪到 1.5mm（为 Gello 控制做准备）
       # 无论 use_gripper 配置如何，都执行此操作
        if not self.use_gripper:
            if len(goal_joints) >= 7 and goal_joints[-1] > 10:  # 如果目标夹爪位置 > 10mm（说明是张开状态）
                print("� Reset 完成，现在关闭夹爪到 1.5mm（为 Gello 控制做准备）")
                
                # 获取当前关节位置
                self._update_curr_joint_state()
                close_gripper_joints = self.curr_joint_positions.copy()
                close_gripper_joints[-1] = 1.5  # 设置夹爪为 1.5mm
                
                print(f"   📍 当前夹爪位置: {self.curr_joint_positions[-1]:.2f} mm")
                print(f"   🎯 目标夹爪位置: 1.5 mm")
                
                # 多次发送命令以克服滤波器效应
                for i in range(8):
                    self.robot.command_joint_state(close_gripper_joints, from_gello=False)
                    time.sleep(0.2)
                
                # 验证夹爪位置
                time.sleep(0.5)
                self._update_curr_joint_state()
                final_gripper = self.curr_joint_positions[-1]
                print(f"   ✅ 最终夹爪位置: {final_gripper:.2f} mm")
                
                if abs(final_gripper - 1.5) > 5.0:
                    print(f"   ⚠️ 警告：夹爪未到达目标位置（差距 {abs(final_gripper - 1.5):.2f} mm）")
        
        self._update_curr_joint_state()
        #self._RESET_JOINT_STATE[-1] = 1.5
        print("FINAL JOINTS:", self.curr_joint_positions)
        
        # Check if we reached the goal (with tolerance)
        position_error = np.abs(self.curr_joint_positions - goal_joints)
        if np.any(position_error > 0.1):  # 0.1 rad tolerance (~5.7 degrees)
            print(f"⚠️  Warning: Large position error detected!")
            print(f"   Position error: {position_error}")
            print(f"   Max error: {np.max(position_error):.4f} rad")

    def go_to_reset(self):
        """Move robot to reset position."""
        self._update_curr_joint_state()
        
        if self.randomreset:
            # Add random noise to reset position
            reset_joints = self._RESET_JOINT_STATE.copy()
            # Add random noise to first 6 joints (not gripper)
            reset_joints[:6] += np.random.uniform(-0.1, 0.1, size=(6,))
            self.interpolate_move(reset_joints, timeout=2.0)
        else:
            self.interpolate_move(self._RESET_JOINT_STATE, timeout=2.0)

    def reset(self, **kwargs):
        """Reset the environment."""
        if self.save_video:
            self.save_video_recording()

        self.go_to_reset()
        self.curr_path_length = 0

        self._update_curr_joint_state()
        
        # 保存reset时的旋转值（rx, ry），用于在step中固定这两个维度
   #     self.reset_ee_rotation = self.curr_ee_pos_rot_gripper[3:5].copy()
       # print(f"🔒 Reset时保存初始旋转 (rx, ry): {self.reset_ee_rotation}, 后续step将固定这两个维度")
        
        # # Update IK solver seed to current joint state after reset
        # if hasattr(self.robot, 'update_ik_seed'):
        #     self.robot.update_ik_seed()
        
        obs = self._get_obs()
        self.terminate = False
        
        return obs, {"succeed": False}

    def save_video_recording(self):
        """Save recorded video frames."""
        try:
            if len(self.recording_frames):
                if not os.path.exists('./videos'):
                    os.makedirs('./videos')

                timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

                for camera_key in self.recording_frames[0].keys():
                    video_path = f'./videos/a1x_{camera_key}_{timestamp}.mp4'

                    # Get the shape of the first frame for this camera
                    first_frame = self.recording_frames[0][camera_key]
                    height, width = first_frame.shape[:2]

                    video_writer = cv2.VideoWriter(
                        video_path,
                        cv2.VideoWriter_fourcc(*"mp4v"),
                        10,
                        (width, height),
                    )

                    for frame_dict in self.recording_frames:
                        video_writer.write(frame_dict[camera_key])

                    video_writer.release()
                    print(f"Saved video for camera {camera_key} at {video_path}")

            self.recording_frames.clear()
        except Exception as e:
            print(f"Failed to save video: {e}")

    def init_cameras(self, name_serial_dict=None):
        """Initialize realsense cameras."""
        if self.cap is not None:
            self.close_cameras()

        self.cap = OrderedDict()
        for cam_name, kwargs in name_serial_dict.items():
            cap = VideoCapture(RSCapture(name=cam_name, **kwargs))
            self.cap[cam_name] = cap

    def close_cameras(self):
        """Close all cameras."""
        try:
            for cap in self.cap.values():
                cap.close()
        except Exception as e:
            print(f"Failed to close cameras: {e}")

    def _update_curr_joint_state(self):
        """Update current joint state from robot."""
        obs = self.robot.get_observations()
        self.curr_joint_positions = obs["joint_positions"]
        self.curr_joint_velocities = obs["joint_velocities"]
        self.curr_ee_pos_rot_gripper = obs["ee_pos_rot_gripper"]
        #self.curr_gripper_position = obs["gripper_position"]#2025/01/26
        # haoyuan print
       # print("!!obs ee_pos_rot_gripper:", self.curr_ee_pos_rot_gripper)
      #  print("!!joint_positions:", self.curr_joint_positions)
        
    def _get_obs(self) -> dict:
        """Get current observation."""
        images = self.get_im()
        state_observation = {
            "joint_positions": self.curr_joint_positions,
            "joint_velocities": self.curr_joint_velocities,
            "ee_pos_rot_gripper": self.curr_ee_pos_rot_gripper,#2025/01/26
            #"gripper_position": self.curr_gripper_position,
        }
        return copy.deepcopy(dict(images=images, state=state_observation))

    def close(self):
        """Clean up resources."""
        if hasattr(self, 'listener'):
            self.listener.stop()
        if hasattr(self, 'robot'):
            self.robot.close()
        self.close_cameras()
        if self.display_image:
            self.img_queue.put(None)
            cv2.destroyAllWindows()
            self.displayer.join()
