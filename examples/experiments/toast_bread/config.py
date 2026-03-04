"""
Training configuration for A1_X robot.
Adapted from Insert_block for joint-space control.
"""
import os
import jax
import numpy as np
import jax.numpy as jnp
import gymnasium as gym
from franka_env.envs.wrappers import (
    SpacemouseIntervention,
    GelloIntervention,
    MultiCameraBinaryRewardClassifierWrapper,
    #ManualRewardWrapper,
)
from franka_env.envs.a1x_env import DefaultA1XEnvConfig
from serl_launcher.wrappers.serl_obs_wrappers import SERLObsWrapper
from serl_launcher.wrappers.chunking import ChunkingWrapper
from serl_launcher.networks.reward_classifier import load_classifier_func

from experiments.config import DefaultTrainingConfig
from experiments.insert_block.wrapper import A1XTaskEnv, A1XGripperPenaltyWrapper


class EnvConfig(DefaultA1XEnvConfig):
    """Environment configuration for A1_X robot."""
    
    # A1_X Robot Configuration
    A1X_NUM_DOFS = 7
    A1X_NODE_NAME = "a1x_serl_node"
    A1X_PORT = 6100
    A1X_PYTHON_PATH = "/usr/bin/python3"
    # Optional external CuRobo IK service address (e.g. tcp://127.0.0.1:6202)
    A1X_CUROBO_IK_SERVICE = os.environ.get("CUROBO_IK_SERVICE")
    
    # 🔧 IK配置选项
    # True: 使用CuRobo IK (GPU加速，更快)
    # False: 使用RelaxedIK (A1X默认，稳定)
    USE_CUROBO_IK = True  # 默认使用RelaxedIK
    
    # Camera Configuration - 腕部相机 + L515侧面相机
    REALSENSE_CAMERAS = {
        "wrist_1": {
            "serial_number": "044322073334",  # 腕部相机序列号
            "dim": (1280, 720),
            "exposure": 10500,
        },
        "side_policy_256": {
            "serial_number": "243222075799",  # 侧面相机序列号
            "dim": (1280, 720),
            "exposure": 10500,
        },
        # "side_classifier": {
        #     "serial_number": "044322073334",  # 复用腕部相机
        #     "dim": (1280, 720),
        #     "exposure": 10500,
        # },
        # "demo": {
        #     "serial_number": "f0265239",  # 复用L515相机
        #     "dim": (1280, 720),
        #     "exposure": 10500,
        # },
    }
    
    # Image cropping functions
    IMAGE_CROP = {
        "wrist_1": lambda img: img,
        "side_policy_256": lambda img: img[:, 245:-81],

        # side_classifier not available on this machine; leave commented out
        # "side_classifier": lambda img: img[390:-150, 420:-700],
        # "demo": lambda img: img[50:-150, 400:-400]
    }

    # Task Configuration - Joint Space
    # 目标关节配置 (根据你的任务调整)
    # 格式: [joint1, joint2, joint3, joint4, joint5, joint6, gripper(0-100mm)]
    # 注意：必须是7维！(6个关节 + 1个夹爪)
    TARGET_JOINT_STATE = np.array([0.7306, 2.2, -1.3127, 0.5768, -0.0374, 0.3708, 100.0])  # 抓取位置 (7维)
# - 0.20382978723404255
# - 1.7593617021276595
# - -0.5638297872340425
# - -0.9472340425531914
# - -0.18425531914893617
# - 0.043617021276595745


    RESET_JOINT_STATE = np.array([0.20382978723404255, 1.7593617021276595, -0.5638297872340425, -0.9472340425531914, -0.18425531914893617, 0.043617021276595745, 100.0])  # 中立位置 (7维)
    # 重置关节配置 (中立位置)
   # RESET_JOINT_STATE = np.array([-0.22404255319148936, 1.514255319148936, -0.684468085106383, -0.45063829787234044, 0.09957446808510638, -0.056595744680851066, 100.0])  # 夹爪张开
    
    USE_GRIPPER = True
    GRIPPER_CLOSED_MM = 2.5  # 夹爪闭合位置 (单位 mm)，当 USE_GRIPPER=False 时使用
    # 奖励阈值 (每个关节的容差) - 可调整使检测更宽松
    # 前6个是关节角度(弧度),最后一个是夹爪位置(mm)
    # 增大数值使成功检测更容易触发
    REWARD_THRESHOLD = np.array([0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 20.0])  # 更宽松的阈值
    ACTION_SPACE = gym.spaces.Box(
        low=np.ones((7,)) * -1.0,
        high=np.ones((7,)) * 1.0,
        dtype=np.float32,
    )
    # 动作缩放 - 控制每步的最大变化量
    # haoyuan for action scale tuning
    ACTION_SCALE = np.array([0.005, 0.005, 0.005, 0.0, 0.0, 0.0, 0.2]) # [x y z roll pitch yaw gripper]
    # ACTION_SCALE: np.ndarray = np.ones((7,))  # Scaling for joint actions
    
    # 关节限制 (安全范围)
    # 基于 A1_X 的实际关节限制
    JOINT_LIMIT_LOW = np.array([-2.880, -0.001, 0.0,  1.5, 1.521, -1.56, 2.0])
    JOINT_LIMIT_HIGH = np.array([2.880, 3.14, -2.95, -1.55, -1.52,  1.56, 99.0])
    
    # Display and Control
    DISPLAY_IMAGE = True
    MAX_EPISODE_LENGTH = 500
    
    # Random Reset
    RANDOM_RESET = False
    RANDOM_JOINT_RANGE = 0.1  # 随机扰动范围(弧度)


class TrainConfig(DefaultTrainingConfig):
    """Training configuration for A1_X SERL."""
    
    # 观察键 - A1_X 使用关节空间
    image_keys = ["wrist_1", "side_policy_256"]
    # classifier_keys disabled because only wrist camera is available
    classifier_keys = []
    
    # A1_X 的状态键 (关节空间,不是TCP pose)
    # 修复：使用实际存在的键名（不包括gripper_position，因为夹爪在joint_positions[6]中）
    #proprio_keys = ["joint_positions", "joint_velocities", "ee_pos_rot_gripper"]
    proprio_keys = ["ee_pos_rot_gripper"]
    # Training parameters
    checkpoint_period = 2000
    cta_ratio = 2
    random_steps = 0
    discount = 0.98
    buffer_period = 1000
    encoder_type = "resnet-pretrained"
    setup_mode = "single-arm-learned-gripper"
    reward_neg = -0.05
    
    # 🚀 Action Chunking 配置
    action_chunk_size = None # 一次输出4个连续的动作（滚动窗口）
    
    # Task description (用于语言条件化策略)
    task_desc = "toast the bread"
    
    # Octo model path (如果使用预训练模型)
    # octo_path = "/home/dungeon_master/conrft/octo_model/octo-small-1.5"
    octo_path = "hf://rail-berkeley/octo-small-1.5"
    teleoperation_device = "spacemouse"  # "gello", "spacemouse", or None
    
    # 🆕 新版 GelloIntervention 配置（基于 launch_yaml.py）
    gello_config_path = "/home/dungeon_master/conrft/Gello/gello_software/configs/yam_A1_X.yaml"  # YAML 配置文件路径
    
    # ⚠️ 旧版参数（已弃用，保留用于向后兼容）
    gello_port = "/dev/serial/by-id/usb-FTDI_USB__-__Serial_Converter_FTA7NNNU-if00-port0"

    def get_environment(self, fake_env=False, save_video=False, classifier=False, 
                       stack_obs_num=1, eval_mode=False, data_collection_mode=False):
        """
        Create A1_X environment with appropriate wrappers.
        
        Args:
            fake_env: 如果为 True，创建假环境（用于 learner）
            save_video: 是否保存视频
            classifier: 是否使用奖励分类器
            stack_obs_num: 观测堆叠数量
            eval_mode: 如果为 True，禁用 Gello 干预（评估模式）
            data_collection_mode: 如果为 True，使用数据采集同步模式
        """
        
        # 创建基础 A1_X 环境
        env = A1XTaskEnv(
            fake_env=fake_env,
            save_video=save_video,
            config=EnvConfig()
        )
        
        # 添加遥控设备干预 (用于人工示教)
        if not fake_env:
            if self.teleoperation_device == "gello":
                # 🆕 新版：使用 YAML 配置文件（基于 launch_yaml.py 架构）
                # 🎯 根据模式选择同步策略
                if data_collection_mode:
                    # 数据采集模式：Reset时同步 + 按空格也同步
                    # 因为数据采集时会多次按空格切换干预状态
                    enable_follower_val = True
                    sync_on_reset_val = False # 🔧 修复：Reset时同步
                    sync_on_intervention_val = True  # 🔧 修复：数据采集也要按空格同步
                    print("🎯 Gello 同步模式：数据采集（Reset + 空格键都同步）")
                else:
                    # 在线训练模式：初始化follower，Reset不同步，按空格时同步
                    enable_follower_val = True
                    sync_on_reset_val = False  # 不在reset时同步
                    sync_on_intervention_val = True
                    print("🎯 Gello 同步模式：在线训练（仅启用干预时同步）")
                
                env = GelloIntervention(
                    env,
                    left_config_path=self.gello_config_path,  # YAML 配置路径
                    control_rate_hz=500,                       # 控制频率 (改为500Hz)
                    eval_mode=eval_mode,                       # 🎯 评估模式
                    enable_follower=enable_follower_val,       # 🆕 是否初始化follower
                    sync_on_reset=sync_on_reset_val,          # 🎯 Reset同步
                    sync_on_intervention=sync_on_intervention_val,  # 🎯 干预同步
                )
                if eval_mode:
                    print(f"🎯 GelloIntervention 已创建（评估模式：干预已禁用）")
                else:
                    print(f"✅ GelloIntervention 已启用")
                    print(f"   配置文件: {self.gello_config_path}")
                    print(f"   💡 按空格键启用/禁用 Gello 干预")
            elif self.teleoperation_device == "spacemouse":
                env = SpacemouseIntervention(env)
        
        # SERL 观察包装器 - 标准化观察空间
        env = SERLObsWrapper(env, proprio_keys=self.proprio_keys)
        
        # 🚀 Chunking wrapper - 用于动作序列（启用 action chunking）
        # act_exec_horizon=4: 一次预测4个连续动作，每步执行一个
        env = ChunkingWrapper(
            env, 
            obs_horizon=stack_obs_num,      # 观测历史（默认1）
            act_exec_horizon=self.action_chunk_size  # 🚀 动作chunk大小（4个动作）
        )
        
        # 奖励分类器 (如果启用)
        if classifier:
            classifier_func = load_classifier_func(
                key=jax.random.PRNGKey(0),
                sample=env.observation_space.sample(),
                image_keys=self.classifier_keys,
                checkpoint_path=os.path.abspath("classifier_ckpt/"),
            )

            def reward_func(obs):
                """自定义奖励函数 - 根据任务调整"""
                def sigmoid(x):
                    return 1 / (1 + jnp.exp(-x))
                
                # 示例: 检查是否达到目标关节配置
                classifier_score = sigmoid(classifier_func(obs)[0])
                
                # 获取当前关节位置 (需要从环境中获取)
                current_joints = env.unwrapped.curr_joint_positions
                target_joints = env.unwrapped.config.TARGET_JOINT_STATE
                
                # 计算关节误差
                joint_error = np.linalg.norm(current_joints[:6] - target_joints[:6])
                
                # 组合奖励: 分类器 + 关节精度 + 夹爪状态
                if classifier_score > 0.9 and joint_error < 0.2:
                    return 10.0  # 成功奖励
                else:
                    return self.reward_neg  # 负奖励
            
            env = MultiCameraBinaryRewardClassifierWrapper(env, reward_func)
        
        # 夹爪惩罚包装器 - 避免频繁开关夹爪
        env = A1XGripperPenaltyWrapper(env, penalty=-0.2)
        
        # 🎖️ 手动奖励包装器 - 随时可以按 's'/'f' 键标记成功/失败
        # 放在最外层，这样无论是否有干预都能使用
        # if not fake_env:  # 只在真机上启用
        #     env = ManualRewardWrapper(env, success_reward=1.0)
        
        return env
