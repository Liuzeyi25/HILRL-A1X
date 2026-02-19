#!/usr/bin/env python
import time
import torch
import numpy as np

# ROS2兼容性检查
try:
    import rclpy
    from rclpy.node import Node
    from geometry_msgs.msg import Pose, PoseStamped
    from sensor_msgs.msg import JointState
    from std_msgs.msg import Header
    HAS_ROS2 = True
    HAS_ROS1 = False
except ImportError:
    HAS_ROS2 = False
    try:
        import rospy
        from geometry_msgs.msg import Pose, PoseStamped
        from sensor_msgs.msg import JointState
        from std_msgs.msg import Header
        HAS_ROS1 = True
    except ImportError:
        HAS_ROS1 = False
        print("Warning: Neither ROS1 nor ROS2 available, logging disabled")

from curobo.types.base import TensorDeviceType
from curobo.types.math import Pose as CuroboPose
from curobo.types.robot import RobotConfig
from curobo.wrap.reacher.ik_solver import IKSolver, IKSolverConfig

# 添加Pinocchio用于FK验证
try:
    import pinocchio as pin
    HAS_PINOCCHIO = True
except ImportError:
    HAS_PINOCCHIO = False
    print("Warning: Pinocchio not available, FK verification disabled")
class URDFInverseKinematics:
    def __init__(
        self,
        urdf_file="/home/dungeon_master/A1_X/arm/install/mobiman/share/mobiman/urdf/A1X/urdf/a1x.urdf",
        base_link="base_link",
        ee_link="arm_link6",  # 固定为 arm_link6
    ):
        
        self.urdf_file = urdf_file
        self.base_link = base_link
        self.ee_link = ee_link
        if self.ee_link is None:
            self.ee_link = self._find_ee_link()

        self._log_info(f"[IK] Using EE link: {self.ee_link}")
        
        # 初始化CuRobo IK solver
        self._init_ik_solver()
        
        # 初始化Pinocchio FK验证器（可选）
        self._init_fk_verifier()
    
    def _log_info(self, msg):
        """兼容ROS1/ROS2的日志输出"""
        if HAS_ROS2:
            # ROS2暂时用print，实际使用时需要node.get_logger()
            print(msg)
        elif HAS_ROS1:
            import rospy
            rospy.loginfo(msg)
        else:
            print(msg)
    
    def _log_warn(self, msg):
        """警告日志"""
        if HAS_ROS2:
            print(f"WARN: {msg}")
        elif HAS_ROS1:
            import rospy
            rospy.logwarn(msg)
        else:
            print(f"WARN: {msg}")
    
    def _log_error(self, msg):
        """错误日志"""
        if HAS_ROS2:
            print(f"ERROR: {msg}")
        elif HAS_ROS1:
            import rospy
            rospy.logerr(msg)
        else:
            print(f"ERROR: {msg}")
    
    def _log_debug(self, msg):
        """调试日志"""
        if HAS_ROS2:
            # print(f"DEBUG: {msg}")  # 调试时取消注释
            pass
        elif HAS_ROS1:
            import rospy
            rospy.logdebug(msg)
        else:
            # print(f"DEBUG: {msg}")  # 调试时取消注释
            pass
    
    def _find_ee_link(self):
        """自动查找end-effector link（参考FK代码）"""
        if not HAS_PINOCCHIO:
            self._log_warn("[IK] Pinocchio not available, using default EE link: Link6")
            return "Link6"
        
        try:
            model = pin.buildModelFromUrdf(self.urdf_file)
            
            # 按优先级查找end-effector frame（与FK代码一致）
            candidate_names = ['gripper_link', 'end_effector', 'ee_link', 'tool0', 'Link6']
            
            for name in candidate_names:
                if model.existFrame(name):
                    self._log_info(f"[IK] Found EE frame: {name}")
                    return name
            
            # 如果都没找到，使用最后一个frame
            last_frame_name = model.frames[-1].name
            self._log_warn(f"[IK] No standard EE frame found, using last frame: {last_frame_name}")
            return last_frame_name
            
        except Exception as e:
            self._log_error(f"[IK] Failed to auto-detect EE link: {e}")
            return "Link6"  # 回退到默认值
    
    def _init_ik_solver(self):
        """初始化CuRobo IK solver"""
        try:
            self.tensor_args = TensorDeviceType()
            print(f"[IK] CuRobo device: {self.tensor_args.device}")
            self.robot_cfg = RobotConfig.from_basic(
                self.urdf_file, 
                self.base_link, 
                self.ee_link, 
                self.tensor_args
            )
            
            self.ik_config = IKSolverConfig.load_from_robot_config(
                self.robot_cfg,
                None,
                rotation_threshold=0.1,  # 进一步放宽到0.1（约5.7度）
                position_threshold=0.01,  # 10mm
                num_seeds=32,  # 大幅增加seeds
                self_collision_check=False, 
                self_collision_opt=False,
                tensor_args=self.tensor_args,
                use_cuda_graph=True,
            )

            self.ik_solver = IKSolver(self.ik_config)
            self._log_info(f"[IK] CuRobo IK solver initialized successfully")
            self._log_info(f"[IK] DOF: {self.robot_cfg.kinematics.kinematics_config.n_dof}")
            
        except Exception as e:
            self._log_error(f"[IK] Failed to initialize IK solver: {e}")
            raise
    
    def _init_fk_verifier(self):
        """初始化FK验证器（用于检查IK结果）"""
        if not HAS_PINOCCHIO:
            self.fk_model = None
            self.fk_data = None
            self.fk_ee_frame_id = None
            return
        
        try:
            self.fk_model = pin.buildModelFromUrdf(self.urdf_file)
            self.fk_data = self.fk_model.createData()
            
            # 查找end-effector frame ID
            if self.fk_model.existFrame(self.ee_link):
                self.fk_ee_frame_id = self.fk_model.getFrameId(self.ee_link)
            else:
                self.fk_ee_frame_id = self.fk_model.nframes - 1
                
            self._log_info(
                f"[IK] FK verifier initialized (DOF: {self.fk_model.nq}, "
                f"EE frame: {self.fk_model.frames[self.fk_ee_frame_id].name})"
            )
            
        except Exception as e:
            self._log_warn(f"[IK] Failed to initialize FK verifier: {e}")
            self.fk_model = None
    
    def verify_ik_solution(self, joint_solution, target_position, target_orientation):
        """使用FK验证IK解的准确性"""
        if self.fk_model is None:
            return None
        
        try:
            # 准备关节位置（确保形状匹配）
            q = np.zeros(self.fk_model.nq)
            n_joints = min(len(joint_solution), self.fk_model.nq)
            q[:n_joints] = joint_solution[:n_joints]
            
            # 计算FK
            pin.forwardKinematics(self.fk_model, self.fk_data, q)
            pin.updateFramePlacements(self.fk_model, self.fk_data)
            
            # 获取end-effector pose
            ee_placement = self.fk_data.oMf[self.fk_ee_frame_id]
            fk_position = ee_placement.translation
            fk_quat = pin.Quaternion(ee_placement.rotation)
            fk_quat_array = np.array([fk_quat.x, fk_quat.y, fk_quat.z, fk_quat.w])
            
            # 计算误差
            pos_error = np.linalg.norm(fk_position - np.array(target_position))
            
            # 四元数误差（使用点积）
            target_quat = np.array(target_orientation)
            quat_dot = np.abs(np.dot(fk_quat_array, target_quat))
            quat_error = 1.0 - quat_dot  # 理想情况下接近0
            
            return {
                'position_error': pos_error,
                'orientation_error': quat_error,
                'fk_position': fk_position,
                'fk_quaternion': fk_quat_array
            }
            
        except Exception as e:
            self._log_warn(f"[IK] FK verification failed: {e}")
            return None
    
    def solve_ik(self, target_position, target_orientation, current_joints=None):
        """
        Calculates IK with validation and normalization.
        :param current_joints: List or Array of current joint angles (in radians).
                               Required for smooth motion tracking.
        """
        # 1. Normalize Quaternion
        quat = np.array(target_orientation)
        norm = np.linalg.norm(quat)
        if norm > 0:
            quat = quat / norm
        else:
            self._log_error("Invalid quaternion: Norm is zero")
            return None

        # Convert Target to Tensor
        target_position_tensor = torch.tensor(list(target_position), 
                                              device=self.tensor_args.device, 
                                              dtype=torch.float32)
        target_orientation_tensor = torch.tensor(list(quat), 
                                                 device=self.tensor_args.device, 
                                                 dtype=torch.float32)
        
        goal = CuroboPose(target_position_tensor, target_orientation_tensor)

        # CRITICAL CHANGE 3: Prepare the seed (initial state)
        # If we have current joints, we format them as the seed for the solver.
        # When using multiple seeds (num_seeds > 1), shape must be [batch_size, num_seeds, num_dof]
        seed_tensor = None
        if current_joints is not None:
            # Ensure it is a tensor with shape (1, num_seeds, DOF)
            num_seeds = self.ik_solver.num_seeds
            seed_tensor = torch.tensor(current_joints, 
                                     device=self.tensor_args.device, 
                                     dtype=torch.float32).view(1, 1, -1)
            # Replicate seed for all seeds
            seed_tensor = seed_tensor.repeat(1, num_seeds, 1)

        # CRITICAL CHANGE 4: Pass the seed to solve_batch
        # This tells the solver: "Start searching FROM here"
        if seed_tensor is not None:
            result = self.ik_solver.solve_batch(goal, seed_config=seed_tensor)
        else:
            # Fallback to random initialization if no current state provided (dangerous for tracking)
            result = self.ik_solver.solve_batch(goal)
            
        torch.cuda.synchronize()

        # 2. Check Success Status
        is_success = result.success.cpu().numpy().all()
        
        # Retry logic (Retaining your original logic, but be careful with this loop in real-time control)
        original_pos_thresh = self.ik_solver.position_threshold
        original_rot_thresh = self.ik_solver.rotation_threshold

        while not is_success:
            pos_err = result.position_error.cpu().numpy()[0,0]
            rot_err = result.rotation_error.cpu().numpy()[0,0] if hasattr(result, 'rotation_error') else 0.0
            self._log_warn(f"IK retry: Pos={pos_err*1000:.2f}mm, Rot={rot_err:.4f}")
            
            self.ik_solver.position_threshold *= 5
            self.ik_solver.rotation_threshold *= 2
            
            # Pass the seed again during retry!
            if seed_tensor is not None:
                result = self.ik_solver.solve_batch(goal, seed_config=seed_tensor)
            else:
                result = self.ik_solver.solve_batch(goal)
                
            is_success = result.success.cpu().numpy().all()
            
            # Break if thresholds get absurdly high to prevent infinite loops
            if self.ik_solver.position_threshold > 0.5:
                pos_err = result.position_error.cpu().numpy()[0,0]
                rot_err = result.rotation_error.cpu().numpy()[0,0] if hasattr(result, 'rotation_error') else 0.0
                self._log_error(f"IK Failed: Final Pos={pos_err*1000:.2f}mm, Rot={rot_err:.4f}")
                break

        # Reset thresholds for next run
        self.ik_solver.position_threshold = original_pos_thresh
        self.ik_solver.rotation_threshold = original_rot_thresh

        if is_success:
            pos_err = result.position_error.cpu().numpy()[0,0]
            rot_err = result.rotation_error.cpu().numpy()[0,0] if hasattr(result, 'rotation_error') else 0.0
            print(f"IK Success: Pos={pos_err*1000:.2f}mm, Rot={rot_err:.4f}")
            return result
        else:
            # 调试：打印最终失败时的详细信息
            pos_err = result.position_error.cpu().numpy()[0,0] if hasattr(result, 'position_error') else float('inf')
            rot_err = result.rotation_error.cpu().numpy()[0,0] if hasattr(result, 'rotation_error') else float('inf')
            # 获取最佳解（即使失败）
            best_solution = result.solution.cpu().numpy()[0]
            self._log_error(f"IK无解，最佳尝试: joints={np.array2string(best_solution, precision=3, separator=',')}")
            return None