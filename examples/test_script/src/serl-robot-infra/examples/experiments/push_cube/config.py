"""

New config for CubeEnv that supports both real and fake environments.
Author: Wenkai

"""
import os
import jax
import jax.numpy as jnp
import numpy as np

from franka_env.envs.wrappers import (
    Quat2EulerWrapper,
    JoystickIntervention,
    MultiCameraBinaryRewardClassifierWrapper,
    GripperCloseEnv
)
from franka_env.envs.relative_env import RelativeFrame
# from franka_env.envs.a1_env import DefaultEnvConfig
# Import our simplified fake environment
from franka_env.envs.fake_env import FakeDefaultEnvConfig
from serl_launcher.wrappers.serl_obs_wrappers import SERLObsWrapper
from serl_launcher.wrappers.chunking import ChunkingWrapper
from serl_launcher.networks.reward_classifier import load_classifier_func

from experiments.config import DefaultTrainingConfig
# Import our unified CubeEnv
from experiments.push_cube.wrapper import CubeEnv


class EnvConfig(FakeDefaultEnvConfig):
    SERVER_URL = "http://127.0.0.1:5000/"
    REALSENSE_CAMERAS = {
        "wrist_1": {
            "serial_number": "f0265239",#"f0210138", #f0190751
            "dim": (1280, 720),
            # "exposure": 40000,
        },
        "wrist_2": {
            "serial_number": "332522070934",
            "dim": (1280, 720),
            "exposure": 40000,
        },
    }
    IMAGE_CROP = {
        "wrist_1": lambda img: img[150:720, 400:1000],
        "wrist_2": lambda img: img[50:700, 200:1100],
    }
    TARGET_POSE = np.array([0.34374736, 0.11237716, 0.10896096, np.pi, 0, 0.5*np.pi])
    GRASP_POSE = np.array([0.34374736, 0.11237716, 0.10896096, np.pi, 0, 0.5*np.pi])
    RESET_Q_POSE = np.array([
        0.38, 1.60, -0.92, -1.66, 0.65, 1.96
    ])
    RESET_POSE = np.array([0.30374736, 0.11237716, 0.22896096, np.pi, 0, 0.5*np.pi])
    REWARD_THRESHOLD = np.array([0.01, 0.01, 0.01, 0.1, 0.1, 0.1])  # Position and orientation thresholds
    ABS_POSE_LIMIT_LOW = TARGET_POSE - np.array([0.03, 0.02, 0.01, 0.01, 0.1, 0.4])
    ABS_POSE_LIMIT_HIGH = TARGET_POSE + np.array([0.03, 0.02, 0.05, 0.01, 0.1, 0.4])
    RANDOM_RESET = False
    RANDOM_XY_RANGE = 0.02
    RANDOM_RZ_RANGE = 0.05
    ACTION_SCALE = np.array([0.2, 1, 1])  # Ensure it's numpy array
    DISPLAY_IMAGE = True
    MAX_EPISODE_LENGTH = 10000
    COMPLIANCE_PARAM = {
        "translational_stiffness": 2000,
        "translational_damping": 89,
        "rotational_stiffness": 150,
        "rotational_damping": 7,
        "translational_Ki": 0,
        "translational_clip_x": 0.0075,
        "translational_clip_y": 0.0016,
        "translational_clip_z": 0.0055,
        "translational_clip_neg_x": 0.002,
        "translational_clip_neg_y": 0.0016,
        "translational_clip_neg_z": 0.005,
        "rotational_clip_x": 0.01,
        "rotational_clip_y": 0.025,
        "rotational_clip_z": 0.005,
        "rotational_clip_neg_x": 0.01,
        "rotational_clip_neg_y": 0.025,
        "rotational_clip_neg_z": 0.005,
        "rotational_Ki": 0,
    }
    PRECISION_PARAM = {
        "translational_stiffness": 2000,
        "translational_damping": 89,
        "rotational_stiffness": 250,
        "rotational_damping": 9,
        "translational_Ki": 0.0,
        "translational_clip_x": 0.1,
        "translational_clip_y": 0.1,
        "translational_clip_z": 0.1,
        "translational_clip_neg_x": 0.1,
        "translational_clip_neg_y": 0.1,
        "translational_clip_neg_z": 0.1,
        "rotational_clip_x": 0.5,
        "rotational_clip_y": 0.5,
        "rotational_clip_z": 0.5,
        "rotational_clip_neg_x": 0.5,
        "rotational_clip_neg_y": 0.5,
        "rotational_clip_neg_z": 0.5,
        "rotational_Ki": 0.0,
    }


class FakeEnvConfig(FakeDefaultEnvConfig):
    """Configuration specifically for fake environment training"""
    TARGET_POSE = np.array([0.34374736, 0.11237716, 0.10896096, np.pi, 0, 0.5*np.pi])
    GRASP_POSE = np.array([0.34374736, 0.11237716, 0.10896096, np.pi, 0, 0.5*np.pi])
    RESET_Q_POSE = np.array([
        0.38, 1.60, -0.92, -1.66, 0.65, 1.96
    ])
    RESET_POSE = np.array([0.30374736, 0.11237716, 0.22896096, np.pi, 0, 0.5*np.pi])
    REWARD_THRESHOLD = np.array([0.01, 0.01, 0.01, 0.1, 0.1, 0.1])
    ACTION_SCALE = np.array([0.2, 1, 1])
    MAX_EPISODE_LENGTH = 100  # Shorter episodes for training


class TrainConfig(DefaultTrainingConfig):
    batch_size = 16  
    image_keys = ["wrist_1", "wrist_2"]
    classifier_keys = []  # Empty for fake environment without images
    
    # training_starts: int
    training_starts: int = 0
    proprio_keys = ["tcp_pose", "gripper_pose"]
    buffer_period = 1000
    checkpoint_period = 5000
    steps_per_update = 50
    encoder_type = "resnet-pretrained"
    setup_mode = "single-arm-fixed-gripper"
    
    # other training parameters

    def get_environment(self, fake_env=False, save_video=False, classifier=False):
        if fake_env:
            # Use simplified fake environment for classifier training
            env = CubeEnv(
                fake_env=fake_env,
                save_video=False,
                config=FakeEnvConfig(),
            )
            # Apply minimal wrappers for fake environment
            # env = SERLObsWrapper(env, proprio_keys=self.proprio_keys)
            from serl_launcher.wrappers.serl_obs_wrappers_fake import FakeSERLObsWrapper
            env = FakeSERLObsWrapper(env, proprio_keys=self.proprio_keys)
            env = ChunkingWrapper(env, obs_horizon=1, act_exec_horizon=None)
            
            if classifier:
                try:
                    classifier_func = load_classifier_func(
                        key=jax.random.PRNGKey(0),
                        sample=env.observation_space.sample(),
                        image_keys=self.classifier_keys,
                        checkpoint_path=os.path.abspath("/home/e230112/Hil-serl/classifier_ckpt/push_cube_15000_epoch")# ("classifier_ckpt/"), # /home/e230112/Hil-serl/classifier_ckpt/push_cube_15000_epoch/model_state.pkl
                    )

                    def reward_func(obs):
                        sigmoid = lambda x: 1 / (1 + jnp.exp(-x))
                        # Simple reward based on state since no images in fake env
                        return int(sigmoid(classifier_func(obs)) > 0.85)

                    env = MultiCameraBinaryRewardClassifierWrapper(env, reward_func)
                except Exception as e:
                    print(f"Warning: Could not load classifier: {e}")
                    print("Continuing without classifier wrapper")
            
        else:
            # Real environment setup (original code)
            env = CubeEnv(
                fake_env=False,
                save_video=save_video,
                config=EnvConfig(),
            )
            env = GripperCloseEnv(env)
            env = JoystickIntervention(env)
            env = RelativeFrame(env)
            env = Quat2EulerWrapper(env)
            env = SERLObsWrapper(env, proprio_keys=self.proprio_keys)
            env = ChunkingWrapper(env, obs_horizon=1, act_exec_horizon=None)
            
            if classifier:
                classifier_func = load_classifier_func(
                    key=jax.random.PRNGKey(0),
                    sample=env.observation_space.sample(),
                    image_keys=self.classifier_keys,
                    checkpoint_path=os.path.abspath("/home/pine/hil-serl/examples/300000"),
                )

                def reward_func(obs):
                    sigmoid = lambda x: 1 / (1 + jnp.exp(-x))
                    # added check for z position to further robustify classifier
                    return int(sigmoid(classifier_func(obs)) > 0.85 and obs['state'][0, 6] > 0.04)

                env = MultiCameraBinaryRewardClassifierWrapper(env, reward_func)
        
        return env