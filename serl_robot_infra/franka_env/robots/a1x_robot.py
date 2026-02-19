"""A1_X robot controlled via ROS2.

This module provides a Robot interface for the A1_X robot arm controlled via ROS2.
The robot publishes commands to /motion_target/target_joint_state_arm topic.

This implementation uses a subprocess with system Python to run ROS2 nodes,
communicating via ZMQ to work around Python version incompatibilities.
"""

import os
import subprocess
import sys
import threading
import time
from typing import Dict, Optional

import numpy as np
import zmq
from scipy.spatial.transform import Rotation as R
import torch

# Add project root to path for A1Kinematics import
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

try:
    from a1_x_kenimetic_haoyuan import A1Kinematics
    HAS_A1_KINEMATICS = True
except ImportError:
    HAS_A1_KINEMATICS = False
    print("Warning: A1Kinematics not available. IK will not work.")


class A1XRobotBridge:
    """Bridge to communicate with ROS2 node via ZMQ.
    
    🚀 方案1优化：使用两个独立的 Socket 分离命令和状态查询
    - command_socket: 专门发送关节命令（控制线程使用）
    - state_socket: 专门查询状态（主线程使用）
    
    这样两个线程不再共享锁，消除锁竞争导致的延迟。
    """
    
    def __init__(self, port: int = 6100, state_port: int = None):
        """初始化双Socket桥接。
        
        Args:
            port: 命令端口 (command socket)
            state_port: 状态端口 (state socket)，默认为 port + 1
        """
        self.command_port = port
        self.state_port = state_port if state_port is not None else port + 1
        
        self.context = zmq.Context()
        
        # Command socket - 专门发送命令（控制线程使用）
        self.command_socket = self.context.socket(zmq.REQ)
        self.command_socket.setsockopt(zmq.RCVTIMEO, 50)  # 50ms timeout
        self.command_socket.setsockopt(zmq.SNDTIMEO, 50)
        self.command_socket.setsockopt(zmq.LINGER, 0)
        self.command_socket.connect(f"tcp://localhost:{self.command_port}")
        self._command_lock = threading.Lock()  # 命令专用锁
        
        # State socket - 专门查询状态（主线程使用）
        self.state_socket = self.context.socket(zmq.REQ)
        self.state_socket.setsockopt(zmq.RCVTIMEO, 50)  # 50ms timeout
        self.state_socket.setsockopt(zmq.SNDTIMEO, 50)
        self.state_socket.setsockopt(zmq.LINGER, 0)
        self.state_socket.connect(f"tcp://localhost:{self.state_port}")
        self._state_lock = threading.Lock()  # 状态专用锁
        
        print(f"🚀 [A1XRobotBridge] 双Socket模式初始化")
        print(f"   Command port: {self.command_port}")
        print(f"   State port: {self.state_port}")
        
    def _reset_command_socket(self):
        """Reset command socket after timeout/error."""
        try:
            self.command_socket.close()
        except:
            pass
        self.command_socket = self.context.socket(zmq.REQ)
        self.command_socket.setsockopt(zmq.RCVTIMEO, 50)
        self.command_socket.setsockopt(zmq.SNDTIMEO, 50)
        self.command_socket.setsockopt(zmq.LINGER, 0)
        self.command_socket.connect(f"tcp://localhost:{self.command_port}")
        
    def _reset_state_socket(self):
        """Reset state socket after timeout/error."""
        try:
            self.state_socket.close()
        except:
            pass
        self.state_socket = self.context.socket(zmq.REQ)
        self.state_socket.setsockopt(zmq.RCVTIMEO, 50)
        self.state_socket.setsockopt(zmq.SNDTIMEO, 50)
        self.state_socket.setsockopt(zmq.LINGER, 0)
        self.state_socket.connect(f"tcp://localhost:{self.state_port}")
        
    def get_joint_state(self) -> Optional[Dict]:
        """Get current joint state from ROS2 node.
        
        使用独立的 state_socket，不会阻塞控制线程。
        """
        with self._state_lock:
            try:
                self.state_socket.send_json({"cmd": "get_state"})
                response = self.state_socket.recv_json()
                return response
            except zmq.Again:
                print("Timeout waiting for joint state, resetting state socket...")
                self._reset_state_socket()
                return None
            except Exception as e:
                print(f"Error getting joint state: {e}")
                self._reset_state_socket()
                return None
    
    def command_joint_positions(self, positions: np.ndarray):
        """Send joint position commands to ROS2 node.
        
        使用独立的 command_socket，不会阻塞状态查询线程。
        """
        with self._command_lock:
            try:
                self.command_socket.send_json({
                    "cmd": "command_joint_state",
                    "positions": positions.tolist()
                })
                # Receive immediate acknowledgment
                self.command_socket.recv_json()
            except zmq.Again:
                print("Timeout sending joint command, resetting command socket...")
                self._reset_command_socket()
            except Exception as e:
                print(f"Error commanding joints: {e}")
                self._reset_command_socket()
    
    def close(self):
        """Close ZMQ connections."""
        # Shutdown via command socket
        try:
            self.command_socket.send_json({"cmd": "shutdown"})
            self.command_socket.recv_json()
        except:
            pass
        
        self.command_socket.close()
        self.state_socket.close()
        self.context.term()


class A1XRobot:
    """A class representing an A1_X robot controlled via ROS2.
    
    The A1_X robot is controlled through ROS2 topics:
    - Commands are published to /motion_target/target_joint_state_arm
    - Joint states are received from /hdas/feedback_arm
    
    This implementation runs ROS2 node in a separate process using system Python 3.10
    and communicates via ZMQ.
    """

    def __init__(
        self,
        num_dofs: int = 7,
        node_name: str = "a1x_gello_node",
        port: int = 6100,
        python_path: str = "/usr/bin/python3",
        use_curobo_ik: bool = True,
        curobo_ik_service: Optional[str] = None,
    ):
        """Initialize the A1_X robot.
        
        Args:
            num_dofs: Number of degrees of freedom (joints) of the robot. Default is 7.
            node_name: Name of the ROS2 node. Default is "a1x_gello_node".
            port: ZMQ port for communication. Default is 6100.
            python_path: Path to system Python 3.10 with ROS2. Default is /usr/bin/python3.
            use_curobo_ik: If True, use CuRobo IK solver; otherwise use RelaxedIK. Default is False.
            curobo_ik_service: Optional external CuRobo IK service address (e.g. tcp://127.0.0.1:6202).
        """
        self._num_dofs = num_dofs
        self._port = port
        self._use_curobo_ik = use_curobo_ik
        self._curobo_ik_service = curobo_ik_service or os.environ.get("CUROBO_IK_SERVICE")
        
        # Start ROS2 node process
        print("Starting ROS2 node subprocess...")
        script_path = os.path.join(os.path.dirname(__file__), "a1x_ros2_node.py")
        
        # 🔧 Add --use-curobo-ik flag if requested
        ik_flag = " --use-curobo-ik" if use_curobo_ik else ""
        service_flag = (
            f" --curobo-ik-service {self._curobo_ik_service}"
            if self._curobo_ik_service
            else ""
        )
        
        # Source ROS2 and run the node (use .zsh for subprocess compatibility)
        cmd = (
            f"source /opt/ros/humble/setup.zsh && {python_path} {script_path} "
            f"--port {port} --node-name {node_name}{ik_flag}{service_flag}"
        )
        
        if use_curobo_ik:
            print("🚀 Using CuRobo IK solver for end-effector control")
        else:
            print("🎯 Using RelaxedIK (Cartesian control) for end-effector control")
        
        self._ros2_process = subprocess.Popen(
            cmd,
            shell=True,
            executable="/bin/zsh",
            stdout=None,  # Don't capture, let it print to terminal
            stderr=None,  # Don't capture, let it print to terminal
        )
        
        # Check if process started successfully
        time.sleep(1)
        if self._ros2_process.poll() is not None:
            # Process exited
            print(f"ROS2 node failed to start (exit code: {self._ros2_process.returncode})")
        
        # Wait a bit more for the node to initialize
        time.sleep(1)
        
        # Connect to ROS2 node via ZMQ (dual socket mode)
        print(f"Connecting to ROS2 node on ports {port} (cmd) and {port + 1} (state)...")
        self._bridge = A1XRobotBridge(port=port, state_port=port + 1)
        
        # Wait for first joint state message
        print("Waiting for joint states from A1_X robot...")
        timeout = 10.0  # seconds
        start_time = time.time()
        state = None
        while state is None:
            state = self._bridge.get_joint_state()
            time.sleep(0.1)
            if time.time() - start_time > timeout:
                print(f"Warning: No joint states received after {timeout}s")
                break
        
        if state is not None:
            print("A1_X robot connected successfully")
            if state.get("joint_names"):
                print(f"Joint names: {state['joint_names']}")
        
        # Initialize internal state
        self._last_commanded_state = np.zeros(self._num_dofs)
        
        # Gripper smoothing filter (EMA)
        self._gripper_filtered = None  # Will be initialized on first command
        # 🔧 alpha 值说明:
        #   - alpha=0.01: 极慢响应（99%旧值+1%新值），约需 460 步才能达到目标的 99%
        #   - alpha=0.1:  较慢响应，约需 44 步
        #   - alpha=0.3:  中等响应，约需 13 步
        #   - alpha=0.5:  较快响应，约需 7 步
        #   - alpha=1.0:  无滤波，直接响应
        # 在 500Hz 控制下: 13步 ≈ 26ms, 44步 ≈ 88ms
        self._gripper_alpha = 0.3  # 🔧 从 0.01 改为 0.3，提高夹爪响应速度
        
        # Initialize IK solver if using CuRobo IK
        self._ik_solver = None
        if use_curobo_ik and HAS_A1_KINEMATICS:
            print("Initializing A1Kinematics IK solver...")
            urdf_path = "/home/dungeon_master/A1_X/arm/install/mobiman/share/mobiman/urdf/A1X/urdf/a1x.urdf"
            try:
                self._ik_solver = A1Kinematics(
                    urdf_file=urdf_path,
                    base_link="base_link",
                    ee_link="gripper_link"
                )
                print("✅ A1Kinematics IK solver initialized successfully")
            except Exception as e:
                print(f"⚠️ Failed to initialize A1Kinematics: {e}")
                self._ik_solver = None

    def num_dofs(self) -> int:
        """Get the number of joints of the robot.
        
        Returns:
            int: The number of joints of the robot.
        """
        return self._num_dofs

    def get_joint_state(self) -> np.ndarray:
        """Get the current state of the robot.
        
        Returns:
            np.ndarray: The current joint positions of the robot.
        """
        state = self._bridge.get_joint_state()
        if state is None or state.get("positions") is None:
            # Return last commanded state if no feedback available
            return self._last_commanded_state
        
        joint_state = np.array(state["positions"])
        
        # Ensure correct number of DOFs
        if len(joint_state) != self._num_dofs:
            if len(joint_state) > self._num_dofs:
                joint_state = joint_state[:self._num_dofs]
            else:
                joint_state = np.pad(
                    joint_state,
                    (0, self._num_dofs - len(joint_state)),
                    "constant"
                )
        
        # haoyuan print - 🚀 高频循环中禁用
        # print("!!current gripper:", joint_state[6])
        return joint_state

    def get_eef_pose(self) -> tuple:
        """Get current end-effector pose.
        
        Returns:
            tuple: (position, quaternion) where:
                - position: np.ndarray [x, y, z] in meters
                - quaternion: np.ndarray [x, y, z, w]
        """
        state = self._bridge.get_joint_state()
        if state is None:
            return np.zeros(3), np.array([0, 0, 0, 1])
        
        ee_pos = np.array(state.get("ee_pos", [0, 0, 0]))
        ee_quat = np.array(state.get("ee_quat", [0, 0, 0, 1]))
        
        return ee_pos, ee_quat

    def command_joint_state(self, joint_state: np.ndarray, from_gello: bool = True) -> None:
        """Command the robot to a given state.
        
        Args:
            joint_state (np.ndarray): The joint positions to command the robot to.
            from_gello (bool): If True, apply Gello-to-A1X mapping. If False, treat as 
                             native A1X joint positions (e.g., from reset/environment).
        """
        assert len(joint_state) == self._num_dofs, (
            f"Expected {self._num_dofs} joint values, got {len(joint_state)}"
        )
        
        # Save requested state
        self._last_commanded_state = joint_state.copy()

        # Map from Gello joint ranges to A1_X joint ranges if needed
        if from_gello:
            try:
                mapped = self._map_to_a1x(joint_state)
            except Exception as e:
                print(f"Warning: Gello mapping failed: {e}, using raw values")
                # Fallback to sending raw values if mapping fails
                mapped = joint_state
        else:
            # Already in A1X joint space, no mapping needed
            mapped = joint_state
        
        # Apply gripper smoothing filter (EMA on gripper only - index 6)
        if len(mapped) >= 7:
            if self._gripper_filtered is None:
                # Initialize filter with first value
                self._gripper_filtered = mapped[6]
            else:
                # Apply exponential moving average: filtered = alpha * new + (1-alpha) * old
                self._gripper_filtered = (
                    self._gripper_alpha * mapped[6] + 
                    (1 - self._gripper_alpha) * self._gripper_filtered
                )
            # Replace gripper value with filtered value
            mapped = mapped.copy()
            mapped[6] = self._gripper_filtered

        # print(f"Commanding A1_X joints: {mapped}")
        self._bridge.command_joint_positions(mapped)

    def _map_to_a1x(self, joint_state: np.ndarray) -> np.ndarray:
        """Linearly map Gello joint ranges to A1_X joint ranges.

        Notes / assumptions:
        - Input `joint_state` is length 7 (including gripper).
        - Some ranges provided were given in reversed order; mapping handles that.
        - If an input falls outside the provided Gello range it will be clipped.
        """
        # Gello joint ranges - (start, end) as the physical motion direction
        gello_range_start = np.array([-2.87, 0.0, 0.0, -1.57, -1.34, -2.0, 0.103], dtype=float)
        gello_range_end   = np.array([ 2.87, 3.14, 3.14,  1.57,  1.34,  2.0, 1.0], dtype=float)

        # A1_X joint ranges (target output ranges)
        # Gripper (idx 6): range is 0-100mm
        #                  gello 0.103 (closed) -> A1_X 2mm (closed)
        #                  gello 1.0 (open) -> A1_X 99mm (open)
        a1x_range_start = np.array([-2.880, 0.0, 0.0,  1.55, 1.521, -1.56, 2.0], dtype=float)
        a1x_range_end   = np.array([ 2.880, 3.14, -2.95, -1.55, -1.52,  1.56, 99.0], dtype=float)

        js = np.asarray(joint_state, dtype=float).copy()
        if js.size != 7:
            raise ValueError("Expected 7 joint values for mapping to A1_X")

        # Clip inputs to gello ranges (handle reversed ranges where start > end)
        clipped = js.copy()
        for i in range(7):
            lo = min(gello_range_start[i], gello_range_end[i])
            hi = max(gello_range_start[i], gello_range_end[i])
            clipped[i] = np.clip(js[i], lo, hi)

        # Compute linear mapping per-joint
        # out = out_start + (in - in_start) * (out_end - out_start) / (in_end - in_start)
        out = np.zeros_like(clipped)
        for i in range(7):
            in_start = gello_range_start[i]
            in_end = gello_range_end[i]
            out_start = a1x_range_start[i]
            out_end = a1x_range_end[i]

            in_range = in_end - in_start
            # Avoid division by zero
            if abs(in_range) < 1e-9:
                out[i] = out_start
            else:
                t = (clipped[i] - in_start) / in_range  # normalized position [0, 1]
                out[i] = out_start + t * (out_end - out_start)

        return out

    def _map_from_a1x(self, a1x_joints: np.ndarray) -> np.ndarray:
        """Inverse mapping: A1_X joint ranges to Gello joint ranges.
        
        Given A1_X joint positions, compute the corresponding Gello positions.
        This is the inverse of _map_to_a1x().
        
        Args:
            a1x_joints: A1_X joint positions [7]
            
        Returns:
            Gello joint positions [7]
        """
        # Same ranges as forward mapping
        gello_range_start = np.array([-2.87, 0.0, 0.0, -1.57, -1.34, -2.0, 0.103], dtype=float)
        gello_range_end   = np.array([ 2.87, 3.14, 3.14,  1.57,  1.34,  2.0, 1.0], dtype=float)
        
        a1x_range_start = np.array([-2.880, 0.0, 0.0,  1.55, 1.521, -1.56, 2.0], dtype=float)
        a1x_range_end   = np.array([ 2.880, 3.14, -2.95, -1.55, -1.52,  1.56, 99.0], dtype=float)
        
        a1x = np.asarray(a1x_joints, dtype=float).copy()
        if a1x.size != 7:
            raise ValueError("Expected 7 joint values for inverse mapping from A1_X")
        
        # Clip inputs to A1X ranges
        clipped = a1x.copy()
        for i in range(7):
            lo = min(a1x_range_start[i], a1x_range_end[i])
            hi = max(a1x_range_start[i], a1x_range_end[i])
            clipped[i] = np.clip(a1x[i], lo, hi)
        
        # Compute inverse linear mapping per-joint
        # Solve for in: t = (in - in_start) / (in_end - in_start)
        # where: out = out_start + t * (out_end - out_start)
        # So: t = (out - out_start) / (out_end - out_start)
        # And: in = in_start + t * (in_end - in_start)
        
        gello = np.zeros_like(clipped)
        for i in range(7):
            out_start = a1x_range_start[i]
            out_end = a1x_range_end[i]
            in_start = gello_range_start[i]
            in_end = gello_range_end[i]
            
            out_range = out_end - out_start
            # Avoid division by zero
            if abs(out_range) < 1e-9:
                gello[i] = in_start
            else:
                # Normalize: where is clipped[i] in [out_start, out_end]?
                t = (clipped[i] - out_start) / out_range  # [0, 1]
                gello[i] = in_start + t * (in_end - in_start)
        
        print(f"[A1X] Inverse mapping:")
        print(f"  A1X input:     [{', '.join(f'{v:7.3f}' for v in a1x)}]")
        print(f"  Gello output:  [{', '.join(f'{v:7.3f}' for v in gello)}]")
        
        return gello

    def update_ik_seed(self):
        """Update IK solver's seed (prev_q) to current joint state.
        
        Should be called after robot reset or any large joint-space movement
        to ensure IK solver starts from the correct current position.
        """
        if self._ik_solver is None:
            return
        
        current_joints = self.get_joint_state()[:6]
        self._ik_solver.prev_q = torch.as_tensor(
            current_joints,
            dtype=torch.float32,
            device=self._ik_solver.tensor_args.device
        ).unsqueeze(0)
        print(
            f"[A1XRobot] Updated IK seed to current joints: "
            f"{current_joints}"
        )

    def command_eef_pose(self, eef_delta: np.ndarray, wait_for_completion: bool = True, timeout: float = 2.0) -> dict:
        """Command the robot with EEF delta pose using local IK solver.
        
        Args:
            eef_delta: 7D array [delta_x, delta_y, delta_z, delta_rx, delta_ry, delta_rz, gripper]
                      First 3: delta position (m)
                      Next 3: delta rotation (euler angles in radians)
                      Last 1: gripper absolute position (0-100mm) or delta if you scale it
            wait_for_completion: 是否等待执行到位（目前未实现）
            timeout: 超时时间（秒）（目前未实现）
        
        Returns:
            dict with execution status or None if failed
        """
        if self._ik_solver is None:
            print("⚠️ IK solver not available, cannot execute EEF command")
            return None
        
        if len(eef_delta) != 7:
            print(f"⚠️ Invalid eef_delta length: {len(eef_delta)}, expected 7")
            return None
        
        # 获取当前状态
        ee_pos, ee_quat = self.get_eef_pose()
        current_joints = self.get_joint_state()[:6]  # 只取前6个关节
        
        if ee_pos is None or ee_quat is None:
            print("⚠️ Current EE pose not available")
            return None
        
        # 处理 delta action → 绝对目标位姿
        delta_pos = np.array(eef_delta[:3])
        delta_rot_euler = np.array(eef_delta[3:6])
        gripper_position = eef_delta[6]
        
        # 计算目标位置
        target_pos = ee_pos + delta_pos
        
        # 计算目标旋转（旋转组合）
        current_rotation = R.from_quat(ee_quat)  # [x, y, z, w]
        delta_rotation = R.from_euler('xyz', delta_rot_euler)
        target_rotation = delta_rotation * current_rotation
        target_quat = target_rotation.as_quat()  # [x, y, z, w]
        
        print(f"[a1x_robot] Action to be Solved - pos: {target_pos}, quat[x,y,z,w]: {target_quat}")
        
        # 调用本地 IK 求解器
        try:
            # 确保 prev_q 在正确的设备上（与 IK solver 相同）
            self._ik_solver.prev_q = torch.as_tensor(
                current_joints,
                dtype=torch.float32,
                device=self._ik_solver.tensor_args.device
            ).unsqueeze(0)
            result = self._ik_solver.solve_ik(
                pos=target_pos,
                quat=target_quat
            )
        except Exception as e:
            print(f"⚠️ IK solve failed: {e}")
            return None
        
        # 检查求解是否成功
        if not result.success.cpu().numpy().any():
            print("⚠️ IK solver failed to find solution")
            return None
        
        # 获取关节解
        joint_solution = result.js_solution.position.cpu().numpy()[:6]
        
        # 计算关节差距
        joint_diff = np.abs(joint_solution - current_joints)
        max_diff = joint_diff.max()
        
        print(f"[a1x_robot] IK Solution Found - joints: {joint_solution}, max joint diff: {max_diff:.4f} rad ({np.rad2deg(max_diff):.2f}°)")
        
        # 构建完整的关节命令（包括夹爪）
        full_joint_command = np.concatenate([joint_solution, [gripper_position]])
        
        # 直接发送关节命令（不通过 Gello 映射，因为这是直接的关节控制）
        self.command_joint_state(full_joint_command, from_gello=False)
        
        return {
            'target_joints': joint_solution.tolist(),
            'reached': True,  # 简化：不等待到位
            'final_error': 0.0,
            'gripper': float(gripper_position)
        }

    def get_observations(self) -> Dict[str, np.ndarray]:
        """Get the current observations of the robot.
        
        Returns:
            Dict[str, np.ndarray]: A dictionary of observations including:
                - joint_positions: Current joint positions
                - joint_velocities: Current joint velocities
                - ee_pos_quat: End-effector position and quaternion [x, y, z, qx, qy, qz, qw]
                - gripper_position: Gripper position (if applicable)
        """
        joint_positions = self.get_joint_state()
        
        state = self._bridge.get_joint_state()
        if state is not None and state.get("velocities") is not None:
            joint_velocities = np.array(state["velocities"])
        else:
            joint_velocities = np.zeros(self._num_dofs)
        
        if len(joint_velocities) != self._num_dofs:
            if len(joint_velocities) > self._num_dofs:
                joint_velocities = joint_velocities[:self._num_dofs]
            else:
                joint_velocities = np.pad(
                    joint_velocities,
                    (0, self._num_dofs - len(joint_velocities)),
                    "constant"
                )
        
        # Get end-effector pose from ROS2
        ee_pos, ee_quat = self.get_eef_pose()
        ee_pos_quat = np.concatenate([ee_pos, ee_quat])  # [x, y, z, qx, qy, qz, qw]
        
        # Convert quaternion to RPY (roll, pitch, yaw) in radians
        rotation = R.from_quat(ee_quat)  # expects [x, y, z, w]
        ee_rpy = rotation.as_euler('xyz', degrees=False)  # returns [roll, pitch, yaw]
        ee_pos_rot = np.concatenate([ee_pos, ee_rpy])  # [x, y, z, roll, pitch, yaw]
        
        # If the last joint is a gripper, extract it
        if self._num_dofs >= 7:
            gripper_position = np.array([joint_positions[-1]]) # gripper [0-100]
        else:
            gripper_position = np.array([0.0])
        
        ee_pos_rot_gripper = np.concatenate([ee_pos_rot, gripper_position / 100])  # [x, y, z, roll, pitch, yaw, gripper]

        return {
            "joint_positions": joint_positions,
            "joint_velocities": joint_velocities,
            "ee_pos_quat": ee_pos_quat,
            "ee_pos_rot_gripper": ee_pos_rot_gripper, # gripper [0-1]
            "gripper_position": gripper_position,
        }

    def command_eef_chunk(
        self, 
        eef_poses: list, 
        correction_threshold: float = 0.005,
        max_corrections: int = 2,
        timeout_per_step: float = 10.0
    ) -> dict:
        """Execute an action chunk using single-step + correction strategy.
        
        This method implements the optimal strategy discovered through testing:
        - Single large movement to each target (not multiple small steps)
        - Iterative corrections if error exceeds threshold
        - Expected performance: ~2-3s per 4cm movement with ~5mm final accuracy
        
        Based on test results:
        - 4cm movement: 2.16s total, 3 movements (1 initial + 2 corrections)
        - Final error: ~5mm (acceptable for most applications)
        - 20× faster than multi-step approach (2.16s vs 16-20s)
        
        Args:
            eef_poses: List of 7D arrays [delta_x, delta_y, delta_z, delta_rx, delta_ry, delta_rz, gripper]
                      Each pose is a delta from the PREVIOUS pose (sequential deltas)
            correction_threshold: meters, trigger correction if error > this (default 5mm)
                                 Recommended: 5mm for speed, 2mm for precision
            max_corrections: maximum number of corrections per step (default 2)
                           Recommended: 2 for most cases, 3-4 for high precision
            timeout_per_step: seconds timeout per movement (default 10s)
        
        Returns:
            dict with:
                - status: "ok", "timeout", or "error"
                - chunk_results: list of dicts with per-step results (movements, errors, times)
                - total_time: total execution time in seconds
                - final_error: final position error in meters after last step
                - (if timeout/error) failed_at_step: step index where execution failed
        
        Example:
            >>> robot = A1XRobot()
            >>> # Move 4cm in X over 4 steps (each step is 1cm delta from previous)
            >>> chunk = [
            ...     [0.01, 0.0, 0.0, 0.0, 0.0, 0.0, 100.0],  # Step 1: +1cm X
            ...     [0.01, 0.0, 0.0, 0.0, 0.0, 0.0, 100.0],  # Step 2: +1cm X
            ...     [0.01, 0.0, 0.0, 0.0, 0.0, 0.0, 100.0],  # Step 3: +1cm X
            ...     [0.01, 0.0, 0.0, 0.0, 0.0, 0.0, 100.0],  # Step 4: +1cm X
            ... ]
            >>> result = robot.command_eef_chunk(chunk)
            >>> print(f"Status: {result['status']}")
            >>> print(f"Total time: {result['total_time']:.2f}s")
            >>> print(f"Final error: {result['final_error']*1000:.2f}mm")
        """
        try:
            self._bridge.socket.send_json({
                "cmd": "command_eef_chunk",
                "poses": eef_poses,
                "correction_threshold": correction_threshold,
                "max_corrections": max_corrections,
                "timeout_per_step": timeout_per_step
            })
            
            # Wait for response (timeout includes all movements in chunk)
            # Add 5s buffer per step for network/processing overhead
            total_timeout_ms = int((timeout_per_step + 5.0) * len(eef_poses) * 1000)
            self._bridge.socket.setsockopt(zmq.RCVTIMEO, total_timeout_ms)
            
            result = self._bridge.socket.recv_json()
            
            # Restore default timeout
            self._bridge.socket.setsockopt(zmq.RCVTIMEO, 20000)
            
            return result
            
        except zmq.Again:
            return {
                "status": "error",
                "error": "ZMQ timeout waiting for chunk execution"
            }
        except Exception as e:
            return {
                "status": "error",
                "error": str(e)
            }

    def close(self):
        """Clean up ROS2 resources."""
        if hasattr(self, "_bridge"):
            self._bridge.close()
        if hasattr(self, "_ros2_process"):
            self._ros2_process.terminate()
            self._ros2_process.wait(timeout=5)


def main():
    """Test the A1_X robot interface."""
    robot = A1XRobot(num_dofs=7)
    
    print("\nRobot initialized")
    print(f"Number of DOFs: {robot.num_dofs()}")
    
    print("\nCurrent joint state:")
    joint_state = robot.get_joint_state()
    print(joint_state)
    
    print("\nObservations:")
    obs = robot.get_observations()
    for key, value in obs.items():
        print(f"  {key}: {value}")
    
    print("\nTesting command (no movement, just publishing)...")
    test_command = joint_state.copy()
    robot.command_joint_state(test_command)
    time.sleep(0.5)
    
    print("\nClosing robot...")
    robot.close()
    print("Done")


if __name__ == "__main__":
    main()
