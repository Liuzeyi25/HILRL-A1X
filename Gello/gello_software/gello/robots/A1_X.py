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

from gello.robots.robot import Robot


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
        """Send joint position commands to ROS2 node."""
        with self._lock:
            try:
                self.socket.send_json({
                    "cmd": "command",
                    "positions": positions.tolist()
                })
                # Wait for acknowledgment
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


class A1XRobot(Robot):
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
        script_path = os.path.join(os.path.dirname(__file__), "A1_X_ros2_node.py")
        
        # Source ROS2 and run the node (use .zsh for subprocess compatibility)
        # Note: subprocess uses zsh, so we source setup.zsh not setup.zsh
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
        
        return joint_state

    def command_joint_state(self, joint_state: np.ndarray) -> None:
        """Command the robot to a given state.
        
        Args:
            joint_state (np.ndarray): The joint positions to command the robot to.
        """
        assert len(joint_state) == self._num_dofs, (
            f"Expected {self._num_dofs} joint values, got {len(joint_state)}"
        )
        
        # Save requested state
        self._last_commanded_state = joint_state.copy()

        # Map from Gello joint ranges to A1_X joint ranges before sending
        try:
            mapped = self._map_to_a1x(joint_state)
        except Exception:
            # Fallback to sending raw values if mapping fails
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
            print(f"[A1X] Gripper filtered: {self._gripper_filtered:.3f}")

        self._bridge.command_joint_positions(mapped)

    def _map_to_a1x(self, joint_state: np.ndarray) -> np.ndarray:
        """Linearly map Gello joint ranges to A1_X joint ranges.

        Notes / assumptions:
        - Input `joint_state` is length 7 (including gripper).
        - Some ranges provided were given in reversed order; mapping handles that.
        - If an input falls outside the provided Gello range it will be clipped.
        """
        # Gello joint ranges - (start, end) as the physical motion direction
        # Joint 5 (idx 4): user said 1.34 to -1.34, but raw shows small values around 0
        #                  Maybe the actual range is -1.34 to 1.34?
        # Gripper (idx 6): observed raw values range from ~0.1 (closed) to ~1.0 (open)
        gello_range_start = np.array([-2.87, 0.0, 0.0, -1.57, -1.34, -2.0, 0.103], dtype=float)
        gello_range_end   = np.array([ 2.87, 3.14, 3.14,  1.57,  1.34,  2.0, 1.0], dtype=float)

        # A1_X joint ranges (target output ranges)
        # Joint 5 (idx 4): 反转方向 - 当 gello 从 -1.34 到 1.34 时，A1_X 从 1.521 到 -1.52
        # Gripper (idx 6): NEW - range is 0-100mm
        #                  gello 0.103 (closed) -> A1_X 0mm (closed)
        #                  gello 1.0 (open) -> A1_X 100mm (open)
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

        # Debug print: show before/after mapping
        print(f"[A1X] Gello raw:    [{', '.join(f'{v:7.3f}' for v in js)}]")
        # print(f"[A1X] Gello clipped:[{', '.join(f'{v:7.3f}' for v in clipped)}]")
        # print(f"[A1X] A1X mapped:   [{', '.join(f'{v:7.3f}' for v in out)}]")

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
        # Same ranges as forward mapping (must match _map_to_a1x exactly!)
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
                - ee_pos_quat: End-effector position and quaternion (placeholder)
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
        
        # Placeholder for end-effector pose (would need forward kinematics)
        ee_pos_quat = np.zeros(7)
        
        # If the last joint is a gripper, extract it
        if self._num_dofs >= 7:
            gripper_position = np.array([joint_positions[-1]])
        else:
            gripper_position = np.array([0.0])
        
        return {
            "joint_positions": joint_positions,
            "joint_velocities": joint_velocities,
            "ee_pos_quat": ee_pos_quat,
            "gripper_position": gripper_position,
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
