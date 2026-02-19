#!/usr/bin/env python

import time
import torch
import numpy as np
from geometry_msgs.msg import Pose
from sensor_msgs.msg import JointState
from std_msgs.msg import Header

# to import curobo, run if error: 
# export PYTHONPATH=/home/chenxiang/ssd/miniconda3/envs/hilserl-rmdg/lib/python3.10/site-packages:$PYTHONPATH
from curobo.types.base import TensorDeviceType
from curobo.types.math import Pose as CuroboPose
from curobo.types.robot import RobotConfig
from curobo.wrap.reacher.ik_solver import IKSolver, IKSolverConfig
# from a1_meshcat import init_robot_visualizer, display_configs  
from curobo.cuda_robot_model.cuda_robot_model import CudaRobotModel, CudaRobotModelConfig

class A1Kinematics:
    def __init__(self,
                 urdf_file="/home/dungeon_master/A1_X/arm/install/mobiman/share/mobiman/urdf/A1X/urdf/a1x.urdf",
                 base_link="base_link",
                 ee_link="gripper_link"):  # 🔧 修改为 gripper_link
        self.tensor_args = TensorDeviceType()

        self.robot_cfg = RobotConfig.from_basic(urdf_file, base_link, ee_link, self.tensor_args)

        cuda_model_cfg = self.robot_cfg.kinematics

        self.robot_model = CudaRobotModel(cuda_model_cfg)
        self.ik_config = IKSolverConfig.load_from_robot_config(
            self.robot_cfg, None,
            num_seeds=32,  # 增加seeds
            position_threshold=0.002,  # 5mm
            rotation_threshold=0.005,  # ~2.9度
            regularization=True, 
            use_cuda_graph=False,
            tensor_args=self.tensor_args)
        self.ik_solver = IKSolver(self.ik_config)
        self.prev_q = None
    
    def solve_ik(self, pos, quat, max_joint_delta=0.2):
        # CuRobo的CuroboPose期望四元数格式为 [w,x,y,z]
        # 如果输入是[x,y,z,w]格式，需要转换
        if isinstance(quat, (np.ndarray, list, tuple)) and len(quat) == 4:
            # 假设输入是[x,y,z,w]，转换为[w,x,y,z]
            #print("假设输入是[x,y,z,w]，转换为[w,x,y,z]")
            quat_wxyz = [quat[3], quat[0], quat[1], quat[2]]
        else:
            quat_wxyz = quat
        
        # 确保 pos 和 quat_wxyz 都转换为正确设备上的 tensor
        pos_tensor = torch.as_tensor(pos, **self.tensor_args.as_torch_dict())
        quat_tensor = torch.as_tensor(quat_wxyz, **self.tensor_args.as_torch_dict())
            
        goal = CuroboPose(pos_tensor, quat_tensor)
        start_time = time.time()
        
        seed = retract = None
        if self.prev_q is not None:
            seed    = self.prev_q.repeat(self.ik_config.num_seeds, 1).unsqueeze(0)
            retract = seed[:, :1]
        res = self.ik_solver.solve_single(goal, seed, retract, return_seeds=8)

        if res.success.any():
            sols = res.solution[res.success]
            ref  = self.prev_q if self.prev_q is not None else sols[0]
            best = sols[torch.argmin(torch.norm(sols - ref, dim=1))]
            self.prev_q = best
            res.js_solution.position = best
            
            # print("prev_q:", self.prev_q.cpu().numpy())
            # print(f"Best IK solution: {best.cpu().numpy()}")
        else:
            print("⚠️ IK求解失败: 所有seeds都未找到可行解")
            if self.prev_q is not None:
                print(f"   保持上一次的关节位置: {self.prev_q.cpu().numpy()}")
            
        elapsed_time = (time.time() - start_time) * 1000  # ms
        # print(f"🌟 IK 求解耗时: {elapsed_time:.2f} ms")
        return res
        
        # # 如果提供了current_joints，使用它作为参考
        # if current_joints is not None:
        #     current_q = torch.as_tensor(current_joints[:6], **self.tensor_args.as_torch_dict())
        #     seed = current_q.repeat(self.ik_config.num_seeds, 1).unsqueeze(0)
        #     retract = seed[:, :1]
        # else:
        #     seed = retract = None
        #     if self.prev_q is not None:
        #         seed = self.prev_q.repeat(self.ik_config.num_seeds, 1).unsqueeze(0)
        #         retract = seed[:, :1]
        
        # res = self.ik_solver.solve_single(goal, seed, retract, return_seeds=8)

        # if res.success.any():
        #     sols = res.solution[res.success]
        #     ref = current_q if current_joints is not None else (self.prev_q if self.prev_q is not None else sols[0])
        #     best = sols[torch.argmin(torch.norm(sols - ref, dim=1))]
            
        #     # 检查解与当前关节的差距
        #     if current_joints is not None:
        #         joint_diff_tensor = torch.abs(best - current_q)
        #         print(f"[a1_x_kinemetic] Joint differences: {joint_diff_tensor.cpu().numpy()}")
        #         print(f"[a1_x_kinemetic] Max joint diff: {joint_diff_tensor.max().item():.4f} rad")
            
        #     # 检查解与当前关节的差距
        #     if current_joints is not None:
        #         joint_delta = torch.abs(best - current_q).max().item()
        #         if joint_delta > max_joint_delta:
        #             print(
        #                 f"⚠️ Joint delta {joint_delta:.3f} rad "
        #                 f"> {max_joint_delta} rad, re-solving..."
        #             )
        #             # 重新求解，使用更严格的种子（保持与初始化一致的 num_seeds）
        #             seed = current_q.repeat(
        #                 self.ik_config.num_seeds, 1
        #             ).unsqueeze(0)
        #             retract = seed[:, :1]
        #             res = self.ik_solver.solve_single(
        #                 goal, seed, retract, return_seeds=8
        #             )
        #             if res.success.any():
        #                 sols = res.solution[res.success]
        #                 best = sols[
        #                     torch.argmin(torch.norm(sols - current_q, dim=1))
        #                 ]
            
        #     self.prev_q = best
        #     res.js_solution.position = best
            
        # elapsed_time = (time.time() - start_time) * 1000  # ms
        # print(f"🌟 IK 求解耗时: {elapsed_time:.2f} ms")
        # return res
        
        # elapsed_time = (time.time() - start_time) * 1000  # ms
        # print(f"🌟 IK 求解耗时: {elapsed_time:.2f} ms")
        # torch.cuda.synchronize()
        # return result
    
    def forward_kinematics(self, joint_positions):
        joint_tensor = torch.tensor(joint_positions, device=self.tensor_args.device, dtype=torch.float32).unsqueeze(0)
        start_time_1 = time.time()
        state = self.robot_model.get_state(joint_tensor)
        elapsed_time_1 = (time.time() - start_time_1) * 1000
        print(f"🌟 FK 求解耗时: {elapsed_time_1:.2f} ms")
        pos = state.ee_position.squeeze().cpu().numpy()
        quat = state.ee_quaternion.squeeze().cpu().numpy() #（w x y z）
        return pos, quat