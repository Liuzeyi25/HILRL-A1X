#!/usr/bin/env python3
"""
ROS2 node for A1_X robot that communicates via ZMQ.
This script runs with system Python 3.10 and ROS2.
"""

import argparse
import os
import threading
import time

import numpy as np
import rclpy
import zmq
from scipy.spatial.transform import Rotation as R
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float32
from geometry_msgs.msg import PoseStamped

try:
    import pinocchio as pin
    HAS_PINOCCHIO = True
except ImportError:
    HAS_PINOCCHIO = False
    print("Warning: Pinocchio not available, FK computation disabled")

# 🔧 CuRobo IK支持（可选）
try:
    import sys
    # 添加项目路径到sys.path
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    
    from a1_x_kenimetic_haoyuan import A1Kinematics
    HAS_CUROBO_IK = True
    print("✅ CuRobo IK available")
except ImportError as e:
    HAS_CUROBO_IK = False
    print(f"⚠️  CuRobo IK not available: {e}")


class A1XRobotZMQNode(Node):
    """ROS2 node that bridges to ZMQ for A1_X robot control.
    
    🚀 方案1优化：使用两个独立的 Socket 分离命令和状态查询
    - command_socket (port): 专门接收关节命令
    - state_socket (port + 1): 专门响应状态查询
    
    这样控制线程和主线程不再竞争同一个socket，消除延迟。
    """
    
    # Safety thresholds
    SAFE_EFFORT_JOINT_2 = 8.0  # N·m
    SAFE_EFFORT_JOINT_3 = 7.0  # N·m
    SAFE_EFFORT_JOINT_4 = 5.0  # N·m
    SAFE_Z_MIN = 0.07  # meters 0.081

    def __init__(
        self,
        node_name: str = "a1x_gello_node",
        zmq_port: int = 6100,
        use_curobo_ik: bool = False,
        curobo_ik_service: str | None = None,
    ):
        super().__init__(node_name)
        
        # 🔧 IK模式选择
        self.use_curobo_ik = use_curobo_ik
        self.curobo_ik_solver = None
        self.curobo_ik_client = None
        self.curobo_ik_service = curobo_ik_service
        
        if self.use_curobo_ik:
            if self.curobo_ik_service:
                self._init_curobo_ik_client(self.curobo_ik_service)
                self.get_logger().info(
                    f"🚀 Using external CuRobo IK service: {self.curobo_ik_service}"
                )
            elif HAS_CUROBO_IK:
                self.get_logger().info("🚀 Initializing CuRobo IK solver...")
                try:
                    self.curobo_ik_solver = A1Kinematics(
                        urdf_file="/home/dungeon_master/A1_X/arm/install/mobiman/share/mobiman/urdf/A1X/urdf/a1x.urdf",
                        base_link="base_link",
                        ee_link="arm_link6"
                    )
                    self.get_logger().info("✅ CuRobo IK solver initialized successfully")
                except Exception as e:
                    self.get_logger().error(f"❌ Failed to initialize CuRobo IK: {e}")
                    self.use_curobo_ik = False
                    self.get_logger().warn("⚠️  Falling back to RelaxedIK (Cartesian control)")
            else:
                self.get_logger().warn("⚠️  CuRobo IK requested but not available, using RelaxedIK")
                self.use_curobo_ik = False
        if not self.use_curobo_ik:
            self.get_logger().info("🎯 Using RelaxedIK (Cartesian control)")
        
        # Publisher for commanding joint states (arm joints 0-5)
        self.joint_command_pub = self.create_publisher(
            JointState,
            "/motion_target/target_joint_state_arm",
            10
        )

        self.pose_command_pub = self.create_publisher(
            PoseStamped,
            "/motion_target/target_pose_arm",
            10
        )
        
        # Publisher for gripper control (joint 6) - range 0-100mm
        self.gripper_command_pub = self.create_publisher(
            Float32,
            "/motion_control/position_control_gripper",
            10
        )
        
        # Subscriber for current joint states
        self.joint_state_sub = self.create_subscription(
            JointState,
            "/hdas/feedback_arm",
            self.joint_state_callback,
            10
        )

        self.pose_ee_sub = self.create_subscription(
            PoseStamped,
            "/hdas/pose_ee_arm",
            self.pose_ee_callback,
            10
        )
        
        # Current joint state
        self._current_joint_positions = None
        self._current_joint_velocities = None
        self._current_joint_torques = None
        self._joint_names = None
        
        # Current end-effector pose
        self._current_pos = None  # [x, y, z]
        self._current_rot = None  # [qx, qy, qz, qw] quaternion
        
        # 🔧 时间戳：追踪关节和EE位姿的更新时间，用于检测时间不对齐
        self._joint_state_timestamp = 0.0  # 关节状态最后更新时间 (time.time())
        self._ee_pose_timestamp = 0.0      # EE位姿最后更新时间 (time.time())
        
        self._lock = threading.Lock()
        
        # FK solver
        self._fk_model = None
        self._fk_data = None
        self._fk_ee_frame_id = None
        self._init_fk_solver()
        
        # 🚀 方案1：双Socket模式
        self.zmq_context = zmq.Context()
        
        # Command socket - 专门处理命令 (控制线程使用)
        self.command_port = zmq_port
        self.command_socket = self.zmq_context.socket(zmq.REP)
        self.command_socket.bind(f"tcp://*:{self.command_port}")
        
        # State socket - 专门处理状态查询 (主线程使用)
        self.state_port = zmq_port + 1
        self.state_socket = self.zmq_context.socket(zmq.REP)
        self.state_socket.bind(f"tcp://*:{self.state_port}")
        
        self._running = True
        
        # Start ZMQ server threads (独立线程处理命令和状态)
        self.command_thread = threading.Thread(target=self.command_server_loop, daemon=False)
        self.state_thread = threading.Thread(target=self.state_server_loop, daemon=False)
        self.command_thread.start()
        self.state_thread.start()
        
        self.get_logger().info(f"🚀 A1XRobotZMQNode initialized with dual sockets:")
        self.get_logger().info(f"   Command port: {self.command_port}")
        self.get_logger().info(f"   State port: {self.state_port}")

    def joint_state_callback(self, msg: JointState):
        """Callback for receiving joint state updates."""
        with self._lock:
            self._joint_state_timestamp = time.time()
            self._joint_names = list(msg.name)
            self._current_joint_positions = list(msg.position)
            
            # Map gripper position if we have 7 joints (6 arm + 1 gripper)
            # ROS2 feedback: -2.83 (open) to 0 (closed)
            # Standard range: 0 (closed) to 100 (open)
            if len(self._current_joint_positions) >= 7:
                gripper_raw = self._current_joint_positions[6]
                # Linear mapping: out = (in - in_min) / (in_max - in_min) * (out_max - out_min) + out_min
                # in_range: [-2.83, 0], out_range: [100, 0]
                # Simplified: gripper_mapped = (0 - gripper_raw) / (0 - (-2.83)) * (0 - 100) + 100
                #                            = (-gripper_raw) / 2.83 * (-100) + 100
                #                            = (gripper_raw * 100) / 2.83
                gripper_mapped = (gripper_raw / -2.83) * 100.0
                # Clamp to valid range
                gripper_mapped = max(0.0, min(100.0, gripper_mapped))
                self._current_joint_positions[6] = gripper_mapped
            
            if msg.velocity:
                self._current_joint_velocities = list(msg.velocity)
            else:
                self._current_joint_velocities = [0.0] * len(msg.position)
            
            if msg.effort:
                self._current_joint_torques = list(msg.effort)
            else:
                self._current_joint_torques = [0.0] * len(msg.position)

    def pose_ee_callback(self, msg: PoseStamped):
        """Callback for receiving end-effector pose updates."""
        with self._lock:
            self._ee_pose_timestamp = time.time()
            # Extract position (x, y, z)
            self._current_pos = np.array([
                msg.pose.position.x,
                msg.pose.position.y,
                msg.pose.position.z
            ])
            
            # Extract orientation as quaternion (x, y, z, w)
            self._current_rot = np.array([
                msg.pose.orientation.x,
                msg.pose.orientation.y,
                msg.pose.orientation.z,
                msg.pose.orientation.w
            ])

    def _init_fk_solver(self):
        """Initialize FK solver with Pinocchio"""
        if not HAS_PINOCCHIO:
            self.get_logger().warn("FK solver disabled: Pinocchio not available")
            return
        
        urdf_path = '/home/dungeon_master/A1_X/arm/install/mobiman/share/mobiman/urdf/A1X/urdf/a1x.urdf'
        try:
            self._fk_model = pin.buildModelFromUrdf(urdf_path)
            self._fk_data = self._fk_model.createData()
            
            # Find end-effector frame
            for name in ['gripper_link', 'end_effector', 'ee_link', 'tool0']:
                if self._fk_model.existFrame(name):
                    self._fk_ee_frame_id = self._fk_model.getFrameId(name)
                    break
            
            if self._fk_ee_frame_id is None:
                self._fk_ee_frame_id = self._fk_model.nframes - 1
            
            self.get_logger().info(f"FK solver initialized (DOF: {self._fk_model.nq}, EE frame: {self._fk_model.frames[self._fk_ee_frame_id].name})")
        except Exception as e:
            self.get_logger().error(f"Failed to initialize FK solver: {e}")
            self._fk_model = None

    def _init_curobo_ik_client(self, service_addr: str):
        """Initialize external CuRobo IK service client."""
        self._curobo_ctx = zmq.Context.instance()
        self.curobo_ik_client = self._curobo_ctx.socket(zmq.REQ)
        self.curobo_ik_client.setsockopt(zmq.RCVTIMEO, 2000)
        self.curobo_ik_client.setsockopt(zmq.SNDTIMEO, 2000)
        self.curobo_ik_client.setsockopt(zmq.LINGER, 0)
        self.curobo_ik_client.connect(service_addr)

    def _reset_curobo_ik_client(self):
        if self.curobo_ik_client is None or not self.curobo_ik_service:
            return
        try:
            self.curobo_ik_client.close()
        except Exception:
            pass
        self._init_curobo_ik_client(self.curobo_ik_service)

    def _request_remote_ik(self, target_pos, target_quat, current_joints):
        if self.curobo_ik_client is None:
            return None
        try:
            self.curobo_ik_client.send_json(
                {
                    "cmd": "solve_ik",
                    "target_pos": list(target_pos),
                    "target_quat": list(target_quat),
                    "current_joints": list(current_joints),
                }
            )
            reply = self.curobo_ik_client.recv_json()
            if reply.get("status") != "ok":
                self.get_logger().error(
                    f"❌ External CuRobo IK failed: {reply.get('error', 'unknown error')}"
                )
                return None
            return reply
        except zmq.Again:
            self.get_logger().error("❌ External CuRobo IK timeout")
            self._reset_curobo_ik_client()
            return None
        except Exception as e:
            self.get_logger().error(f"❌ External CuRobo IK error: {e}")
            self._reset_curobo_ik_client()
            return None
    
    def compute_fk(self, joint_positions):
        """Compute forward kinematics from joint positions"""
        if self._fk_model is None or not joint_positions:
            return None
        
        try:
            # Prepare joint positions (pad with zeros if needed)
            q = np.zeros(self._fk_model.nq)
            q[:min(len(joint_positions), self._fk_model.nq)] = joint_positions[:min(len(joint_positions), self._fk_model.nq)]
            
            # Compute FK
            pin.forwardKinematics(self._fk_model, self._fk_data, q)
            pin.updateFramePlacements(self._fk_model, self._fk_data)
            
            # Get end-effector pose
            ee_placement = self._fk_data.oMf[self._fk_ee_frame_id]
            position = ee_placement.translation
            quat = pin.Quaternion(ee_placement.rotation)
            
            return {
                'position': {'x': float(position[0]), 'y': float(position[1]), 'z': float(position[2])},
                'orientation': {'x': float(quat.x), 'y': float(quat.y), 'z': float(quat.z), 'w': float(quat.w)}
            }
        except Exception as e:
            self.get_logger().error(f"FK computation failed: {e}")
            return None

    def command_server_loop(self):
        """🚀 处理命令的独立线程 - 控制线程专用，低延迟。"""
        while self._running:
            try:
                # 非阻塞接收，1ms超时确保低延迟
                if self.command_socket.poll(timeout=1):
                    request = self.command_socket.recv_json()
                    cmd = request.get("cmd")
                    
                    if cmd == "command_joint_state":
                        positions = request.get("positions", [])
                        intended = self.publish_joint_command(positions)
                        if intended is None:
                            self.command_socket.send_json({"status": "error", "error": "invalid positions"})
                        else:
                            self.command_socket.send_json({"status": "ok"})
                    
                    elif cmd == "command_eef_pose":
                        eef_pose = request.get("pose")
                        wait_for_completion = request.get("wait_for_completion", True)
                        timeout = float(request.get("timeout", 4.0))
                        
                        # 调用publish_eef_command，返回dict
                        result = self.publish_eef_command(
                            eef_pose, 
                            wait_for_completion=wait_for_completion,
                            pos_tolerance_m=0.0007,  # 0.7mm XYZ 位置容差
                            timeout=timeout
                        )
                        
                        if result is None:
                            self.command_socket.send_json({
                                "status": "error", 
                                "error": "invalid eef_pose or current pose unavailable"
                            })
                        else:
                            # 返回完整的result dict
                            self.command_socket.send_json({
                                "status": "ok",
                                **result  # 包含 reached, final_error, target_pos, target_quat
                            })
                    
                    elif cmd == "command_eef_chunk":
                        eef_poses = request.get("poses")
                        correction_threshold = float(request.get("correction_threshold", 0.005))
                        max_corrections = int(request.get("max_corrections", 2))
                        timeout_per_step = float(request.get("timeout_per_step", 10.0))
                        
                        if not eef_poses:
                            self.command_socket.send_json({"status": "error", "error": "no poses provided"})
                        else:
                            result = self.publish_eef_chunk_command(
                                eef_poses,
                                correction_threshold=correction_threshold,
                                max_corrections=max_corrections,
                                timeout_per_step=timeout_per_step
                            )
                            self.command_socket.send_json(result)
                    
                    elif cmd == "shutdown":
                        self._running = False
                        self.command_socket.send_json({"status": "shutting down"})
                    
                    else:
                        self.command_socket.send_json({"error": "unknown command"})
            
            except Exception as e:
                self.get_logger().error(f"Command server error: {e}")
                try:
                    self.command_socket.send_json({"error": str(e)})
                except:
                    pass

    def state_server_loop(self):
        """🚀 处理状态查询的独立线程 - 主线程专用，不影响控制。"""
        while self._running:
            try:
                # 非阻塞接收
                if self.state_socket.poll(timeout=1):
                    request = self.state_socket.recv_json()
                    cmd = request.get("cmd")
                    
                    if cmd == "get_state":
                        with self._lock:
                            response = {
                                "positions": self._current_joint_positions, # 关节raw位置，末尾是gripper位置 -2.7~0
                                "velocities": self._current_joint_velocities,
                                "joint_names": self._joint_names,
                                "torques": self._current_joint_torques,
                                "ee_pos": self._current_pos.tolist() if self._current_pos is not None else [0, 0, 0],
                                "ee_quat": self._current_rot.tolist() if self._current_rot is not None else [0, 0, 0, 1],
                                "joint_ts": self._joint_state_timestamp,
                                "ee_ts": self._ee_pose_timestamp,
                                "server_ts": time.time(),
                            }
                        self.state_socket.send_json(response)
                    
                    else:
                        self.state_socket.send_json({"error": "use get_state only on state socket"})
            
            except Exception as e:
                self.get_logger().error(f"State server error: {e}")
                try:
                    self.state_socket.send_json({"error": str(e)})
                except:
                    pass

    def zmq_server_loop(self):
        """Handle ZMQ requests in a separate thread (保留兼容性，但不再使用)."""
        while self._running:
            try:
                # Non-blocking receive with timeout
                if self.zmq_socket.poll(timeout=1):  # 1ms timeout for low latency
                    request = self.zmq_socket.recv_json()
                    
                    cmd = request.get("cmd")
                    
                    if cmd == "get_state":
                        with self._lock:
                            response = {
                                "positions": self._current_joint_positions,
                                "velocities": self._current_joint_velocities,
                                "joint_names": self._joint_names,
                                "torques": self._current_joint_torques,
                                "ee_pos": self._current_pos.tolist() if self._current_pos is not None else [0, 0, 0],
                                "ee_quat": self._current_rot.tolist() if self._current_rot is not None else [0, 0, 0, 1],
                            }
                        self.zmq_socket.send_json(response)
                    
                    elif cmd == "command_joint_state":
                        positions = request.get("positions", [])
                        
                        # Joint commands don't wait - return immediately after publishing
                        intended = self.publish_joint_command(positions)
                        if intended is None:
                            self.zmq_socket.send_json({"status": "error", "error": "invalid positions"})
                        else:
                            self.zmq_socket.send_json({"status": "ok"})

                    elif cmd == "command_eef_pose":
                        eef_pose = request.get("pose")
                        timeout = float(request.get("timeout", 10.0))  # Increased default timeout

                        # publish_eef_command now returns the intended absolute pose (pos, quat)
                        # Also extract the delta for tolerance computation
                        delta_pos = np.array(eef_pose[:3])  # [dx, dy, dz]
                        ret = self.publish_eef_command(eef_pose)
                        if ret is None:
                            self.zmq_socket.send_json({"status": "error", "error": "invalid eef_pose or no current pose available"})
                        else:
                            target_pos, target_quat = ret
                            # Pass delta magnitude for adaptive tolerance
                            reached, info = self.wait_for_eef_pose(target_pos, target_quat, timeout=timeout, delta_mag=np.linalg.norm(delta_pos))
                            if reached:
                                # Return info even on success so client can track cumulative error
                                self.zmq_socket.send_json({"status": "ok", "info": info})
                            else:
                                self.zmq_socket.send_json({"status": "timeout", "info": info})
                    
                    elif cmd == "command_eef_chunk":
                        eef_poses = request.get("poses")
                        correction_threshold = float(request.get("correction_threshold", 0.005))  # 5mm default
                        max_corrections = int(request.get("max_corrections", 2))
                        timeout_per_step = float(request.get("timeout_per_step", 10.0))
                        
                        if not eef_poses:
                            self.zmq_socket.send_json({"status": "error", "error": "no poses provided"})
                        else:
                            result = self.publish_eef_chunk_command(
                                eef_poses,
                                correction_threshold=correction_threshold,
                                max_corrections=max_corrections,
                                timeout_per_step=timeout_per_step
                            )
                            self.zmq_socket.send_json(result)
                    
                    elif cmd == "shutdown":
                        self._running = False
                        self.zmq_socket.send_json({"status": "shutting down"})
                    
                    else:
                        self.zmq_socket.send_json({"error": "unknown command"})
            
            except Exception as e:
                self.get_logger().error(f"ZMQ server error: {e}")
                try:
                    self.zmq_socket.send_json({"error": str(e)})
                except:
                    pass

    def publish_joint_command(self, positions):
        """Publish joint position commands to ROS2."""
        # 确保 positions 是列表或可索引的，处理 numpy 数组
        if hasattr(positions, 'tolist'):
            positions = positions.tolist()
        
        # Separate arm joints (0-5) and gripper (6)
        if len(positions) >= 7:
            arm_positions = positions[:6]
            gripper_position = positions[6]
        else:
            arm_positions = positions
            gripper_position = None
        
        # Safety check: FK-based Z limit
        fk_result = self.compute_fk(arm_positions)
        if fk_result:
            fk_z = fk_result['position']['z']

            # 默认阻止低于安全高度的命令
            block_for_z = False

            # 如果我们能读取到当前末端位姿，则允许一个例外：
            # - 当预测的 fk_z 虽然低于 SAFE_Z_MIN，但相对于当前的 z 是向上的（fk_z > current_z）
            #   则认为这是一个向上修正动作，允许发送（避免数值误差导致的误阻断）。
            # - 否则（fk_z <= current_z 或无法读取当前 pose），仍然阻止。
            with self._lock:
                cur_z = None if self._current_pos is None else float(self._current_pos[2])

            if fk_z < self.SAFE_Z_MIN:
                if cur_z is None:
                    # 无法获取当前位姿，保守阻止
                    block_for_z = True
                else:
                    # 允许向上（增高）的小修正动作通过
                    # 使用一个很小的容差避免数值噪声造成误判
                    eps = 1e-6
                    delta_z = fk_z - cur_z
                    if delta_z <= eps:
                        block_for_z = True

            if block_for_z:
                self.get_logger().warn(
                    f"Safety stop: FK Z position {fk_z:.3f} m below limit {self.SAFE_Z_MIN:.3f} m (current_z={cur_z})"
                )
                # Publish gripper command separately - Float32 with range 0-100mm
                if gripper_position is not None:
                    gripper_msg = Float32()
                    gripper_msg.data = float(gripper_position)
                    self.gripper_command_pub.publish(gripper_msg)

                return None
        
        # Publish arm joint command
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        
        with self._lock:
            if self._joint_names is not None:
                # Use first 6 joint names for arm
                msg.name = self._joint_names[:6] if len(self._joint_names) >= 6 else self._joint_names
            else:
                msg.name = [f"joint_{i+1}" for i in range(len(arm_positions))]
        
        msg.position = arm_positions
        msg.velocity = [0.6] * len(arm_positions)  # 恒定速度
        msg.effort = []
        # 🚀 高频循环中禁用 print，避免阻塞！
        # print(f"Publishing joint command: positions={arm_positions}, gripper={gripper_position}")
        self.joint_command_pub.publish(msg)
        
        # Publish gripper command separately - Float32 with range 0-100mm
        if gripper_position is not None:
            # haoyuan print - 🚀 高频循环中禁用
            # print(gripper_position)
            gripper_msg = Float32()
            gripper_msg.data = float(gripper_position)
            self.gripper_command_pub.publish(gripper_msg)

        # Return the intended positions as a flat list (arm + optional gripper)
        intended = list(arm_positions)
        if gripper_position is not None:
            intended.append(float(gripper_position))
        return intended

    def publish_eef_command(self, eef_pose, wait_for_completion=True,
                            pos_tolerance_m=0.001, timeout=4.0,
                            # 旧参数保留兼容，不再用于完成判断
                            joint_tolerance=None):
        """Publish end-effector pose command using external CuRobo IK service.
        
        Args:
            eef_pose: 7D array [delta_x, delta_y, delta_z, delta_rx, delta_ry, delta_rz, gripper]
                     First 3: delta position (m)
                     Next 3: delta rotation (euler angles in radians)
                     Last 1: gripper absolute position (0-100mm)
            wait_for_completion: 是否等待执行到位
            pos_tolerance_m: XYZ 位置误差容忍度（米），默认 0.001 = 1mm
                             直接用 /hdas/pose_ee_arm 反馈的 _current_pos 与目标 XYZ 比较，
                             物理含义明确，不受臂型/关节构型影响。
            timeout: 超时时间（秒）
            joint_tolerance: 已废弃，保留参数以免调用方报错，不再使用。
        
        Returns:
            dict with 'target_joints', 'reached', 'final_error_m', 'gripper'
            final_error_m: 超时时的实际 XYZ 位置误差（米）
        """
        if len(eef_pose) != 7:
            self.get_logger().error(f"Invalid eef_pose length: {len(eef_pose)}, expected 7")
            return None
        
        # 获取当前末端位姿
        with self._lock:
            if self._current_pos is None or self._current_rot is None:
                self.get_logger().warning("Current EE pose not available")
                return None
            current_pos = self._current_pos.copy()
            current_rot = self._current_rot.copy()
            
            if self._current_joint_positions is None:
                self.get_logger().error("Current joint positions not available for IK")
                return None
            current_joints = np.array(self._current_joint_positions[:6])

        # 处理delta action → 绝对目标位姿
        delta_pos = np.array(eef_pose[:3])
        delta_rot_euler = np.array(eef_pose[3:6])
        gripper_position = eef_pose[6]
        
        # 计算目标位置（这就是完成判断的基准）
        target_pos = current_pos + delta_pos
        
        # 计算目标旋转（旋转组合）
        current_rotation = R.from_quat(current_rot)
        delta_rotation = R.from_euler('xyz', delta_rot_euler)
        new_rotation = delta_rotation * current_rotation
        new_quat = new_rotation.as_quat()  # [x, y, z, w]
        
        print(f"[a1x_ros2_node] Action to be Solved - pos: {target_pos}, quat[x,y,z,w]: {new_quat}")

        # 调用外部CuRobo IK服务
        reply = self._request_remote_ik(target_pos, new_quat, current_joints)
        if reply is None:
            self.get_logger().error("External CuRobo IK service failed")
            return None
        
        joint_solution = np.array(reply.get("solution", []), dtype=float)[:6]
        if joint_solution.size < 6:
            self.get_logger().error("Invalid IK solution from service")
            return None
        
        # 计算关节差距（仅供参考打印，不用于完成判断）
        joint_diff = np.abs(joint_solution - current_joints)
        max_diff = joint_diff.max()
        
        print(f"[a1x_ros2_node] Current joints: {current_joints}")
        print(f"[a1x_ros2_node] Target joints:  {joint_solution}")
        print(f"[a1x_ros2_node] Joint diff:     {joint_diff}")
        print(f"[a1x_ros2_node] Max joint diff: {max_diff:.4f} rad ({np.rad2deg(max_diff):.2f}°)")
        print(f"[a1x_ros2_node] Completion criterion: XYZ pos error < {pos_tolerance_m*1000:.1f} mm")
        
        # 发布关节命令
        self.publish_joint_command(
            np.concatenate([joint_solution, [gripper_position]])
        )
        
        # ── 等待执行到位：直接监控 EEF XYZ 位置误差 ──────────────────────────
        # 用 /hdas/pose_ee_arm 反馈的 _current_pos 与 target_pos 比较。
        # 优点：物理意义直接（mm级），不受臂型/奇异点影响。
        reached = False
        final_error_m = float('inf')
        
        if wait_for_completion:
            start_time = time.time()
            poll_interval = 0.01
            
            while time.time() - start_time < timeout:
                with self._lock:
                    if self._current_pos is None:
                        time.sleep(poll_interval)
                        continue
                    current_pos_check = self._current_pos.copy()
                
                pos_error = float(np.linalg.norm(current_pos_check - target_pos))
                final_error_m = pos_error
                
                if pos_error < pos_tolerance_m:
                    reached = True
                    break
                
                time.sleep(poll_interval)
            
            if not reached:
                self.get_logger().warn(
                    f"EEF timeout: pos error={final_error_m*1000:.2f} mm "
                    f"(tolerance={pos_tolerance_m*1000:.1f} mm)"
                )
        
        return {
            'target_joints': joint_solution.tolist(),
            'reached': reached,
            'final_error_m': final_error_m,
            'final_error_mm': final_error_m * 1000,
            'gripper': float(gripper_position)
        }
    
    def publish_eef_chunk_command(self, eef_poses, correction_threshold: float = 0.005, max_corrections: int = 2, timeout_per_step: float = 10.0):
        """Execute an action chunk using single-step + correction strategy.
        
        Based on performance testing, this method uses:
        - Single large movement to target (instead of multiple small steps)
        - Iterative corrections if error exceeds threshold
        - Expected performance: 2-3s per 4cm movement with ~5mm final accuracy
        
        Args:
            eef_poses: List of 7D arrays [delta_x, delta_y, delta_z, delta_rx, delta_ry, delta_rz, gripper]
                      Each pose is a delta from the PREVIOUS pose (sequential deltas)
            correction_threshold: meters, trigger correction if error > this (default 5mm)
            max_corrections: maximum number of corrections per step (default 2)
            timeout_per_step: seconds timeout per movement (default 10s)
        
        Returns:
            dict with:
                - status: "ok" or "error"
                - chunk_results: list of dicts with per-step results
                - total_time: total execution time
                - final_error: final position error after last step
        """
        if not eef_poses or len(eef_poses) == 0:
            return {"status": "error", "error": "empty chunk"}
        
        chunk_results = []
        chunk_start_time = time.time()
        
        with self._lock:
            if self._current_pos is None or self._current_rot is None:
                return {"status": "error", "error": "no current pose available"}
            tracked_pos = self._current_pos.copy()
            tracked_quat = self._current_rot.copy()
        
        self.get_logger().info(f"Starting chunk execution: {len(eef_poses)} steps")
        
        for step_idx, eef_pose in enumerate(eef_poses):
            if len(eef_pose) != 7:
                return {"status": "error", "error": f"invalid pose length at step {step_idx}: {len(eef_pose)}"}
            
            step_result = {
                "step": step_idx,
                "movements": []
            }
            
            # Extract target delta from current tracked position
            delta_pos = np.array(eef_pose[:3])
            delta_rot_euler = np.array(eef_pose[3:6])
            gripper_position = eef_pose[6]
            
            # Compute absolute target for this step
            target_pos = tracked_pos + delta_pos
            
            # Compute target rotation
            tracked_rotation = R.from_quat(tracked_quat)
            delta_rotation = R.from_euler('xyz', delta_rot_euler)
            target_rotation = delta_rotation * tracked_rotation
            target_quat = target_rotation.as_quat()
            
            delta_mag = float(np.linalg.norm(delta_pos))
            
            self.get_logger().info(
                f"Step {step_idx+1}/{len(eef_poses)}: "
                f"delta=[{delta_pos[0]*100:.2f}, {delta_pos[1]*100:.2f}, {delta_pos[2]*100:.2f}]cm, "
                f"target=[{target_pos[0]:.4f}, {target_pos[1]:.4f}, {target_pos[2]:.4f}]"
            )
            
            # Initial movement: send full delta to target
            move_start_time = time.time()
            
            # Create pose message for absolute target
            pose_msg = PoseStamped()
            pose_msg.pose.position.x = float(target_pos[0])
            pose_msg.pose.position.y = float(target_pos[1])
            pose_msg.pose.position.z = float(target_pos[2])
            pose_msg.pose.orientation.x = float(target_quat[0])
            pose_msg.pose.orientation.y = float(target_quat[1])
            pose_msg.pose.orientation.z = float(target_quat[2])
            pose_msg.pose.orientation.w = float(target_quat[3])
            
            self.pose_command_pub.publish(pose_msg)
            
            # Publish gripper
            gripper_msg = Float32()
            gripper_msg.data = float(gripper_position)
            self.gripper_command_pub.publish(gripper_msg)
            
            # Wait for movement to complete
            reached, info = self.wait_for_eef_pose(
                target_pos, target_quat, 
                timeout=timeout_per_step, 
                delta_mag=delta_mag
            )
            
            move_time = time.time() - move_start_time
            
            if not reached:
                self.get_logger().warn(f"Step {step_idx+1} initial movement timeout")
                step_result["movements"].append({
                    "type": "initial",
                    "time": move_time,
                    "reached": False,
                    "info": info
                })
                chunk_results.append(step_result)
                return {
                    "status": "timeout",
                    "chunk_results": chunk_results,
                    "failed_at_step": step_idx,
                    "total_time": time.time() - chunk_start_time
                }
            
            # Update tracked position from actual feedback
            with self._lock:
                if self._current_pos is not None:
                    tracked_pos = self._current_pos.copy()
                if self._current_rot is not None:
                    tracked_quat = self._current_rot.copy()
            
            pos_error = float(np.linalg.norm(tracked_pos - target_pos))
            
            step_result["movements"].append({
                "type": "initial",
                "time": move_time,
                "reached": True,
                "error": pos_error,
                "current_pos": tracked_pos.tolist()
            })
            
            self.get_logger().info(
                f"  Initial move completed in {move_time:.2f}s, error={pos_error*1000:.2f}mm"
            )
            
            # Apply corrections if needed
            for corr_idx in range(max_corrections):
                if pos_error <= correction_threshold:
                    self.get_logger().info(f"  ✓ Error within threshold, no correction needed")
                    break
                
                self.get_logger().info(
                    f"  Correction {corr_idx+1}/{max_corrections}: "
                    f"error {pos_error*1000:.2f}mm > threshold {correction_threshold*1000:.1f}mm"
                )
                
                # Compute correction delta
                correction_delta = target_pos - tracked_pos
                correction_delta_mag = float(np.linalg.norm(correction_delta))
                
                # Compute correction rotation
                tracked_rotation = R.from_quat(tracked_quat)
                target_rotation = R.from_quat(target_quat)
                correction_rotation = target_rotation * tracked_rotation.inv()
                correction_quat = (correction_rotation * tracked_rotation).as_quat()
                
                # Publish correction
                corr_start_time = time.time()
                
                pose_msg = PoseStamped()
                pose_msg.pose.position.x = float(target_pos[0])
                pose_msg.pose.position.y = float(target_pos[1])
                pose_msg.pose.position.z = float(target_pos[2])
                pose_msg.pose.orientation.x = float(correction_quat[0])
                pose_msg.pose.orientation.y = float(correction_quat[1])
                pose_msg.pose.orientation.z = float(correction_quat[2])
                pose_msg.pose.orientation.w = float(correction_quat[3])
                
                self.pose_command_pub.publish(pose_msg)
                self.gripper_command_pub.publish(gripper_msg)
                
                # Wait for correction
                reached, info = self.wait_for_eef_pose(
                    target_pos, correction_quat,
                    timeout=timeout_per_step,
                    delta_mag=correction_delta_mag
                )
                
                corr_time = time.time() - corr_start_time
                
                if not reached:
                    self.get_logger().warn(f"  Correction {corr_idx+1} timeout")
                    step_result["movements"].append({
                        "type": f"correction_{corr_idx+1}",
                        "time": corr_time,
                        "reached": False,
                        "info": info
                    })
                    # Continue to next correction or finish
                    break
                
                # Update tracked position
                with self._lock:
                    if self._current_pos is not None:
                        tracked_pos = self._current_pos.copy()
                    if self._current_rot is not None:
                        tracked_quat = self._current_rot.copy()
                
                pos_error = float(np.linalg.norm(tracked_pos - target_pos))
                
                step_result["movements"].append({
                    "type": f"correction_{corr_idx+1}",
                    "time": corr_time,
                    "reached": True,
                    "error": pos_error,
                    "current_pos": tracked_pos.tolist()
                })
                
                self.get_logger().info(
                    f"  Correction {corr_idx+1} completed in {corr_time:.2f}s, error={pos_error*1000:.2f}mm"
                )
            
            step_result["final_error"] = pos_error
            step_result["total_time"] = sum(m["time"] for m in step_result["movements"])
            chunk_results.append(step_result)
        
        total_time = time.time() - chunk_start_time
        final_error = chunk_results[-1]["final_error"] if chunk_results else 0.0
        
        self.get_logger().info(
            f"Chunk execution complete: {len(eef_poses)} steps in {total_time:.2f}s, "
            f"final_error={final_error*1000:.2f}mm"
        )
        
        return {
            "status": "ok",
            "chunk_results": chunk_results,
            "total_time": total_time,
            "final_error": final_error
        }

    def wait_for_joint_positions(self, target_positions, timeout: float = 5.0, tol: float = 0.02):
        """Wait until current joint positions are within tol of target_positions.

        Args:
            target_positions: list or array of intended positions (arm [0:6] and optional gripper)
            timeout: seconds to wait
            tol: absolute tolerance per joint

        Returns:
            (reached: bool, info: dict)
        """
        start = time.time()
        last_error = None
        while time.time() - start < timeout:
            with self._lock:
                cur = self._current_joint_positions
            if cur is None:
                time.sleep(0.05)
                continue

            # compare up to length of target_positions
            n = min(len(cur), len(target_positions))
            cur_slice = np.array(cur[:n], dtype=float)
            tgt_slice = np.array(target_positions[:n], dtype=float)
            errors = np.abs(cur_slice - tgt_slice)
            max_err = float(np.max(errors)) if errors.size > 0 else 0.0
            last_error = {"max_error": max_err, "errors": errors.tolist(), "current": cur_slice.tolist(), "target": tgt_slice.tolist()}
            if max_err <= tol:
                return True, last_error
            time.sleep(0.05)

        return False, (last_error or {"error": "no feedback"})

    def wait_for_eef_pose(self, target_pos, target_quat, timeout: float = 5.0, pos_tol: float = None, ang_tol: float = 0.1, delta_mag: float = None):
        """Wait until end-effector pose reaches target within tolerances.

        Args:
            target_pos: [x,y,z]
            target_quat: [x,y,z,w]
            timeout: seconds
            pos_tol: meters tolerance (if None, auto-computed based on delta_mag)
            ang_tol: radians tolerance (default 0.1 rad ~5.7 degrees)
            delta_mag: magnitude of position delta commanded (used for adaptive tolerance)

        Returns:
            (reached: bool, info: dict)
        """
        start = time.time()
        last_info = None
        target_pos = np.array(target_pos, dtype=float)
        target_quat = np.array(target_quat, dtype=float)
        
        # Auto-compute position tolerance based on commanded delta magnitude
        if pos_tol is None:
            if delta_mag is not None and delta_mag > 0:
                # Use 70% of the commanded movement delta as tolerance
                # A1_X controller may not reach exact targets, accept larger tolerance
                pos_tol = max(0.005, delta_mag * 0.5)  # At least 5mm tolerance, or 50% of delta
                print(f"[wait_for_eef_pose] Auto pos_tol: {pos_tol:.4f}m (50% of delta {delta_mag:.4f}m)")
            else:
                # Fallback: fixed small tolerance for zero-delta commands
                pos_tol = 0.005  # 5mm
                print(f"[wait_for_eef_pose] Using fixed pos_tol: {pos_tol:.4f}m (zero delta)")

        while time.time() - start < timeout:
            with self._lock:
                cur_pos = None if self._current_pos is None else np.array(self._current_pos, dtype=float)
                cur_quat = None if self._current_rot is None else np.array(self._current_rot, dtype=float)

            if cur_pos is None or cur_quat is None:
                time.sleep(0.05)
                continue

            pos_err = float(np.linalg.norm(cur_pos - target_pos))

            # rotation error: compute relative rotation from current to target
            try:
                cur_r = R.from_quat(cur_quat)
                tgt_r = R.from_quat(target_quat)
                rel = tgt_r * cur_r.inv()
                ang_err = float(np.linalg.norm(rel.as_rotvec()))
            except Exception:
                ang_err = float('inf')

            last_info = {"pos_err": pos_err, "ang_err": ang_err, "current_pos": cur_pos.tolist(), "target_pos": target_pos.tolist()}
            if pos_err <= pos_tol and ang_err <= ang_tol:
                return True, last_info

            time.sleep(0.05)

        return False, (last_info or {"error": "no pose feedback"})

    def cleanup(self):
        """Clean up resources."""
        self._running = False
        
        # 等待两个线程结束
        if hasattr(self, 'command_thread'):
            self.command_thread.join(timeout=2)
        if hasattr(self, 'state_thread'):
            self.state_thread.join(timeout=2)
        
        # 关闭两个 socket
        if hasattr(self, 'command_socket'):
            self.command_socket.close()
        if hasattr(self, 'state_socket'):
            self.state_socket.close()
        
        self.zmq_context.term()
        
        self.get_logger().info("🚀 A1XRobotZMQNode cleanup completed")


def main():
    parser = argparse.ArgumentParser(description="A1_X ROS2 ZMQ Bridge Node")
    parser.add_argument("--port", type=int, default=6100, help="ZMQ port")
    parser.add_argument("--node-name", type=str, default="a1x_gello_node", help="ROS2 node name")
    parser.add_argument(
        "--use-curobo-ik", 
        action="store_true", 
        help="Use CuRobo IK solver instead of RelaxedIK (Cartesian control)"
    )
    parser.add_argument(
        "--curobo-ik-service",
        type=str,
        default=os.environ.get("CUROBO_IK_SERVICE"),
        help="External CuRobo IK service address (e.g. tcp://127.0.0.1:6202)",
    )
    args = parser.parse_args()
    
    rclpy.init()
    
    node = A1XRobotZMQNode(
        node_name=args.node_name, 
        zmq_port=args.port,
        use_curobo_ik=args.use_curobo_ik,
        curobo_ik_service=args.curobo_ik_service,
    )
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        # Silently ignore shutdown errors
        if "shutdown" not in str(e).lower():
            print(f"Error: {e}")
    finally:
        try:
            node.cleanup()
            node.destroy_node()
        except:
            pass
        try:
            if rclpy.ok():
                rclpy.shutdown()
        except:
            pass


if __name__ == "__main__":
    main()
