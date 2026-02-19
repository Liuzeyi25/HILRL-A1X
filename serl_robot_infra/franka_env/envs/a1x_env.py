"""Gym Interface for A1_X Robot with EEF delta control."""

import copy
import os
import queue
import threading
import time
from collections import OrderedDict
from datetime import datetime
from typing import Dict

import cv2
import gymnasium as gym
import numpy as np

from franka_env.camera.rs_capture import RSCapture
from franka_env.camera.video_capture import VideoCapture
from franka_env.robots.a1x_robot import A1XRobot


class ImageDisplayer(threading.Thread):
    """Background thread for displaying camera images."""

    def __init__(self, queue, name):
        super().__init__(daemon=True)
        self.queue = queue
        self.name = name

    def run(self):
        while True:
            img_array = self.queue.get()
            if img_array is None:
                break
            frame = np.concatenate(
                [cv2.resize(v, (128, 128)) for k, v in img_array.items() if "full" not in k],
                axis=1,
            )
            cv2.imshow(self.name, frame)
            cv2.waitKey(1)


class DefaultA1XEnvConfig:
    """Default configuration for A1XEnv."""

    # Robot
    A1X_NUM_DOFS: int = 7
    A1X_NODE_NAME: str = "a1x_serl_node"
    A1X_PORT: int = 6100
    A1X_PYTHON_PATH: str = "/usr/bin/python3"
    A1X_CUROBO_IK_SERVICE: str | None = None

    # Camera
    REALSENSE_CAMERAS: Dict = {
        "wrist_1": "130322274175",
        "wrist_2": "127122270572",
    }
    IMAGE_CROP: dict[str, callable] = {}

    # Task
    TARGET_JOINT_STATE: np.ndarray = np.zeros((7,))
    RESET_JOINT_STATE: np.ndarray = np.zeros((7,))
    REWARD_THRESHOLD: np.ndarray = np.ones((7,)) * 0.1

    # Control
    ACTION_SCALE: np.ndarray = np.ones((7,))
    USE_GRIPPER: bool = True

    # Misc
    DISPLAY_IMAGE: bool = True
    MAX_EPISODE_LENGTH: int = 100
    RANDOM_RESET: bool = False


class A1XEnv(gym.Env):
    """Gymnasium environment for A1_X robot with EEF delta control.

    Action space: 7D [dx, dy, dz, drx, dry, drz, d_gripper]
        - [0:3]  delta position (m)
        - [3:6]  delta rotation (rad, currently zeroed)
        - [6]    delta gripper (normalized)
    """

    def __init__(
        self,
        hz: int = 10,
        fake_env: bool = False,
        save_video: bool = True,
        config: DefaultA1XEnvConfig = None,
    ):
        self.config = config or DefaultA1XEnvConfig()
        self.action_scale = self.config.ACTION_SCALE
        self._TARGET_JOINT_STATE = self.config.TARGET_JOINT_STATE
        self._RESET_JOINT_STATE = self.config.RESET_JOINT_STATE
        self._REWARD_THRESHOLD = self.config.REWARD_THRESHOLD
        self.max_episode_length = self.config.MAX_EPISODE_LENGTH
        self.display_image = self.config.DISPLAY_IMAGE
        self.randomreset = self.config.RANDOM_RESET
        self.use_gripper = getattr(self.config, "USE_GRIPPER", True)
        self.hz = hz
        self.save_video = save_video
        self.recording_frames = []

        # Action: [dx, dy, dz, drx, dry, drz, d_gripper]
        self.action_space = gym.spaces.Box(
            low=np.array([-0.005, -0.005, -0.005, -0.01, -0.01, -0.01, -0.2], dtype=np.float32),
            high=np.array([0.005, 0.005, 0.005, 0.01, 0.01, 0.01, 0.2], dtype=np.float32),
        )

        self.observation_space = gym.spaces.Dict({
            "state": gym.spaces.Dict({
                "joint_positions": gym.spaces.Box(-np.inf, np.inf, shape=(7,)),
                "joint_velocities": gym.spaces.Box(-np.inf, np.inf, shape=(7,)),
                "ee_pos_rot_gripper": gym.spaces.Box(-np.inf, np.inf, shape=(7,)),
                "gripper_position": gym.spaces.Box(-100, 100, shape=(1,)),
            }),
            "images": gym.spaces.Dict({
                key: gym.spaces.Box(
                    0, 255,
                    shape=(256, 256, 3) if "256" in key else (128, 128, 3),
                    dtype=np.uint8,
                )
                for key in self.config.REALSENSE_CAMERAS
            }),
        })

        self.curr_path_length = 0
        self.curr_joint_positions = None
        self.curr_joint_velocities = None
        self.curr_ee_pos_rot_gripper = None
        self.terminate = False

        if fake_env:
            return

        # Robot
        print("Initializing A1_X robot...")
        self.robot = A1XRobot(
            num_dofs=self.config.A1X_NUM_DOFS,
            node_name=self.config.A1X_NODE_NAME,
            port=self.config.A1X_PORT,
            python_path=self.config.A1X_PYTHON_PATH,
            use_curobo_ik=getattr(self.config, "USE_CUROBO_IK", False),
            curobo_ik_service=getattr(self.config, "A1X_CUROBO_IK_SERVICE", None),
            reset_joint_state=self._RESET_JOINT_STATE,
        )

        # Cameras
        self.cap = None
        self.init_cameras(self.config.REALSENSE_CAMERAS)
        if self.display_image:
            self.img_queue = queue.Queue()
            self.displayer = ImageDisplayer(self.img_queue, "A1_X Camera")
            self.displayer.start()

        # Emergency stop (ESC key)
        from pynput import keyboard
        self.listener = keyboard.Listener(
            on_press=lambda key: setattr(self, "terminate", True)
            if key == keyboard.Key.esc else None
        )
        self.listener.start()

        self._update_curr_joint_state()
        print("Initialized A1_X Environment")

    # ------------------------------------------------------------------
    # Step
    # ------------------------------------------------------------------

    def step(self, action: np.ndarray) -> tuple:
        start_time = time.time()
        action = np.clip(action, self.action_space.low, self.action_space.high)
        scaled_action = action * self.action_scale

        # Gripper: normalized delta -> absolute mm
        current_gripper = self.curr_ee_pos_rot_gripper[6]
        new_gripper_mm = np.clip(current_gripper + scaled_action[6], 0.0, 1.0) * 100.0
        if not self.use_gripper:
            new_gripper_mm = 1.5

        # (旋转控制已启用，不再清零 rotation deltas)
        # scaled_action[3:5] = 0.0   # 禁用xy旋转
        scaled_action[3] = 0.0  # 禁用 drx (x轴旋转)
        scaled_action[4] = 0.0  # 禁用 dry (y轴旋转)
        
        # 🔍 调试输出：检查旋转动作值
        if abs(scaled_action[4]) > 0.0001:  # dry (y轴旋转)
            print(f"🔄 Rotation action - dry: {scaled_action[4]:.6f}")
        
        eef_command = np.concatenate([scaled_action[:6], [new_gripper_mm]])
        result = self.robot.command_eef_pose(eef_command, wait_for_completion=True, timeout=2.0)

        self.curr_path_length += 1

        # Maintain control frequency
        remaining = (1.0 / self.hz) - (time.time() - start_time)
        if remaining > 0:
            time.sleep(remaining)

        self._update_curr_joint_state()
        ob = self._get_obs()

        if not result:
            print(f"EEF command returned no result")

        reward = self.compute_reward(ob)
        done = self.curr_path_length >= self.max_episode_length or reward or self.terminate
        return ob, int(reward), done, False, {"succeed": reward}

    def compute_reward(self, obs) -> bool:
        return False

    # ------------------------------------------------------------------
    # Camera
    # ------------------------------------------------------------------

    def get_im(self) -> Dict[str, np.ndarray]:
        images = {}
        display_images = {}
        full_res_images = {}

        for key, cap in self.cap.items():
            try:
                rgb = cap.read()
                cropped = self.config.IMAGE_CROP[key](rgb) if key in self.config.IMAGE_CROP else rgb
                target_size = self.observation_space["images"][key].shape[:2][::-1]
                resized = cv2.resize(cropped, target_size)
                images[key] = resized[..., ::-1]  # BGR -> RGB
                display_images[key] = resized
                display_images[key + "_full"] = cropped
                full_res_images[key] = cropped.copy()
            except queue.Empty:
                input(f"{key} camera frozen. Check connection, then press enter to relaunch...")
                cap.close()
                self.init_cameras(self.config.REALSENSE_CAMERAS)
                return self.get_im()

        self.recording_frames.append(full_res_images)
        if self.display_image:
            self.img_queue.put(display_images)

        return images

    def init_cameras(self, name_serial_dict: dict):
        if self.cap is not None:
            self.close_cameras()
        self.cap = OrderedDict()
        for cam_name, kwargs in name_serial_dict.items():
            self.cap[cam_name] = VideoCapture(RSCapture(name=cam_name, **kwargs))

    def close_cameras(self):
        try:
            for cap in self.cap.values():
                cap.close()
        except Exception as e:
            print(f"Failed to close cameras: {e}")

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def interpolate_move(self, goal_joints: np.ndarray, timeout: float):
        """Linear interpolation move to goal joint positions."""
        steps = int(timeout * self.hz)
        self._update_curr_joint_state()
        path = np.linspace(self.curr_joint_positions, goal_joints, steps)
        print(f"Interpolating from {self.curr_joint_positions} to {goal_joints} ({steps} steps)")

        for joint_positions in path:
            self.robot.command_joint_state(joint_positions, from_gello=False)
            time.sleep(1.0 / self.hz)

        time.sleep(5.0)  # settle

        # Close gripper to 1.5mm after reset if gripper is disabled
        if not self.use_gripper:
            self._close_gripper_after_reset()

        self._update_curr_joint_state()
        print(f"FINAL JOINTS: {self.curr_joint_positions}")

        position_error = np.abs(self.curr_joint_positions - goal_joints)
        if np.any(position_error > 0.1):
            print(f"Warning: max position error = {np.max(position_error):.4f} rad")

    def _close_gripper_after_reset(self):
        """Close gripper to 1.5mm for Gello control preparation."""
        self._update_curr_joint_state()
        close_joints = self.curr_joint_positions.copy()
        close_joints[-1] = 1.5

        for _ in range(8):
            self.robot.command_joint_state(close_joints, from_gello=False)
            time.sleep(0.2)

        time.sleep(0.5)
        self._update_curr_joint_state()
        final_gripper = self.curr_joint_positions[-1]
        if abs(final_gripper - 1.5) > 5.0:
            print(f"Warning: gripper at {final_gripper:.2f}mm, expected 1.5mm")

    def go_to_reset(self):
        reset_joints = self._RESET_JOINT_STATE.copy()
        if self.randomreset:
            reset_joints[:6] += np.random.uniform(-0.1, 0.1, size=(6,))
        self.interpolate_move(reset_joints, timeout=2.0)

    def reset(self, **kwargs):
        if self.save_video:
            self.save_video_recording()

        self.go_to_reset()
        self.curr_path_length = 0
        self._update_curr_joint_state()

        # Lock EE target to prevent servo drift during episode
        self.robot.lock_ee_target()

        obs = self._get_obs()
        self.terminate = False
        return obs, {"succeed": False}

    # ------------------------------------------------------------------
    # Video recording
    # ------------------------------------------------------------------

    def save_video_recording(self):
        if not self.recording_frames:
            return
        try:
            os.makedirs("./videos", exist_ok=True)
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

            for camera_key in self.recording_frames[0]:
                video_path = f"./videos/a1x_{camera_key}_{timestamp}.mp4"
                h, w = self.recording_frames[0][camera_key].shape[:2]
                writer = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*"mp4v"), 10, (w, h))
                for frame_dict in self.recording_frames:
                    writer.write(frame_dict[camera_key])
                writer.release()
                print(f"Saved video: {video_path}")
        except Exception as e:
            print(f"Failed to save video: {e}")
        finally:
            self.recording_frames.clear()

    # ------------------------------------------------------------------
    # State helpers
    # ------------------------------------------------------------------

    def _update_curr_joint_state(self):
        obs = self.robot.get_observations()
        self.curr_joint_positions = obs["joint_positions"]
        self.curr_joint_velocities = obs["joint_velocities"]
        self.curr_ee_pos_rot_gripper = obs["ee_pos_rot_gripper"]

    def _get_obs(self) -> dict:
        images = self.get_im()
        state = {
            "joint_positions": self.curr_joint_positions,
            "joint_velocities": self.curr_joint_velocities,
            "ee_pos_rot_gripper": self.curr_ee_pos_rot_gripper,
        }
        return copy.deepcopy({"images": images, "state": state})

    def close(self):
        if hasattr(self, "listener"):
            self.listener.stop()
        if hasattr(self, "robot"):
            self.robot.close()
        self.close_cameras()
        if self.display_image:
            self.img_queue.put(None)
            cv2.destroyAllWindows()
            self.displayer.join()
