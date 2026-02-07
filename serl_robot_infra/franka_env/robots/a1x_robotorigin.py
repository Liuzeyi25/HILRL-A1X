"""A1_X robot controlled via ROS2.

This module provides a Robot interface for the A1_X robot arm controlled via ROS2.
The robot publishes commands to /motion_target/target_joint_state_arm topic.

This implementation uses a subprocess with system Python to run ROS2 nodes,
communicating via ZMQ to work around Python version incompatibilities.
"""

import os
import subprocess
import threading
import time
from typing import Dict, Optional

import numpy as np
import zmq
from scipy.spatial.transform import Rotation as R


class A1XRobotBridge:
    """Bridge to communicate with ROS2 node via ZMQ."""
    
    def __init__(self, port: int = 6100):
        self.port = port
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REQ)
        self.socket.setsockopt(zmq.RCVTIMEO, 50)  # 50ms receive timeout (reduced for low latency)
        self.socket.setsockopt(zmq.SNDTIMEO, 50)  # 50ms send timeout
        self.socket.setsockopt(zmq.LINGER, 0)  # Don't wait on close
        self.socket.connect(f"tcp://localhost:{port}")
        self._lock = threading.Lock()
        
    def _reset_socket(self):
        """Reset socket after timeout/error to recover from bad state."""
        try:
            self.socket.close()
        except:
            pass
        self.socket = self.context.socket(zmq.REQ)
        self.socket.setsockopt(zmq.RCVTIMEO, 50)
        self.socket.setsockopt(zmq.SNDTIMEO, 50)
        self.socket.setsockopt(zmq.LINGER, 0)
        self.socket.connect(f"tcp://localhost:{self.port}")
        
    def get_joint_state(self) -> Optional[Dict]:
        """Get current joint state from ROS2 node."""
        with self._lock:
            try:
                self.socket.send_json({"cmd": "get_state"})
                response = self.socket.recv_json()
                return response
            except zmq.Again:
                # Timeout - need to reset socket for REQ-REP pattern
                print("Timeout waiting for joint state, resetting socket...")
                self._reset_socket()
                return None
            except Exception as e:
                print(f"Error getting joint state: {e}")
                self._reset_socket()
                return None
    
    def command_joint_positions(self, positions: np.ndarray):
        """Send joint position commands to ROS2 node.
        
        Note: This method returns immediately after sending the command,
        without waiting for the robot to reach the target positions.
        """
        with self._lock:
            try:
                # Send joint command
                self.socket.send_json({
                    "cmd": "command_joint_state",
                    "positions": positions.tolist()
                })
                # haoyuan test
                # print(f"Commanding A1_X joints: {positions}")

                # Receive immediate acknowledgment (no wait for completion)
                self.socket.recv_json()
            except zmq.Again:
                # Timeout - need to reset socket
                print("Timeout sending joint command, resetting socket...")
                self._reset_socket()
            except Exception as e:
                print(f"Error commanding joints: {e}")
                self._reset_socket()
    
    def close(self):
        """Close ZMQ connection."""
        try:
            self.socket.send_json({"cmd": "shutdown"})
            self.socket.recv_json()
        except:
            pass
        self.socket.close()
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
    ):
        """Initialize the A1_X robot.
        
        Args:
            num_dofs: Number of degrees of freedom (joints) of the robot. Default is 7.
            node_name: Name of the ROS2 node. Default is "a1x_gello_node".
            port: ZMQ port for communication. Default is 6100.
            python_path: Path to system Python 3.10 with ROS2. Default is /usr/bin/python3.
        """
        self._num_dofs = num_dofs
        self._port = port
        
        # Start ROS2 node process
        print("Starting ROS2 node subprocess...")
        script_path = os.path.join(os.path.dirname(__file__), "a1x_ros2_node.py")
        
        # Source ROS2 and run the node (use .zsh for subprocess compatibility)
        cmd = f"source /opt/ros/humble/setup.zsh && {python_path} {script_path} --port {port} --node-name {node_name}"
        
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
        
        # Connect to ROS2 node via ZMQ
        print(f"Connecting to ROS2 node on port {port}...")
        self._bridge = A1XRobotBridge(port=port)
        
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
        self._gripper_alpha = 0.01  # Smoothing factor: 0=max smooth, 1=no filter

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
        
        # haoyuan print
        print("!!current gripper:", joint_state[6])
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
            gripper_position = np.array([joint_positions[-1]])
        else:
            gripper_position = np.array([0.0])
        
        ee_pos_rot_gripper = np.concatenate([ee_pos_rot, gripper_position])  # [x, y, z, roll, pitch, yaw, gripper]

        return {
            "joint_positions": joint_positions,
            "joint_velocities": joint_velocities,
            "ee_pos_quat": ee_pos_quat,
            "ee_pos_rot_gripper": ee_pos_rot_gripper,
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
