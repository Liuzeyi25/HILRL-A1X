"""A1_X robot controlled via ROS2.

Communication architecture:
  - ROS2 node runs in a subprocess (system Python 3.10)
  - Main process communicates via dual ZMQ sockets (command + state)
  - Local CuRobo IK solver for EEF delta control
"""

import os
import subprocess
import sys
import threading
import time
from typing import Dict, Optional

import numpy as np
import torch
import zmq
from scipy.spatial.transform import Rotation as R

# A1Kinematics import
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

try:
    from a1_x_kenimetic_haoyuan import A1Kinematics
    HAS_A1_KINEMATICS = True
except ImportError:
    HAS_A1_KINEMATICS = False
    print("Warning: A1Kinematics not available. IK will not work.")

# Gello <-> A1X joint mapping ranges (7 joints including gripper)
_GELLO_RANGE_START = np.array([-2.87, 0.0, 0.0, -1.57, -1.34, -2.0, 0.103])
_GELLO_RANGE_END = np.array([2.87, 3.14, 3.14, 1.57, 1.34, 2.0, 1.0])
_A1X_RANGE_START = np.array([-2.880, 0.0, 0.0, 1.55, 1.521, -1.56, 2.0])
_A1X_RANGE_END = np.array([2.880, 3.14, -2.95, -1.55, -1.52, 1.56, 99.0])


def _linear_map(value: np.ndarray, in_start: np.ndarray, in_end: np.ndarray,
                out_start: np.ndarray, out_end: np.ndarray) -> np.ndarray:
    """Vectorized linear mapping with clipping for reversed ranges."""
    lo = np.minimum(in_start, in_end)
    hi = np.maximum(in_start, in_end)
    clipped = np.clip(value, lo, hi)

    in_range = in_end - in_start
    safe = np.abs(in_range) > 1e-9
    t = np.where(safe, (clipped - in_start) / np.where(safe, in_range, 1.0), 0.0)
    return out_start + t * (out_end - out_start)


def _pad_or_trim(arr: np.ndarray, target_len: int) -> np.ndarray:
    """Pad with zeros or trim array to target_len."""
    if len(arr) > target_len:
        return arr[:target_len]
    if len(arr) < target_len:
        return np.pad(arr, (0, target_len - len(arr)), "constant")
    return arr


# ──────────────────────────────────────────────────────────────────────
# ZMQ Bridge
# ──────────────────────────────────────────────────────────────────────

class A1XRobotBridge:
    """Dual-socket ZMQ bridge to ROS2 node.

    Separates command and state into independent sockets to eliminate
    lock contention between control and observation threads.
    """

    _ZMQ_TIMEOUT_MS = 50

    def __init__(self, port: int = 6100, state_port: int = None):
        self.command_port = port
        self.state_port = state_port or port + 1

        self.context = zmq.Context()
        self.command_socket = self._make_socket(self.command_port)
        self.state_socket = self._make_socket(self.state_port)
        self._command_lock = threading.Lock()
        self._state_lock = threading.Lock()

        print(f"[A1XRobotBridge] Command port: {self.command_port}, State port: {self.state_port}")

    def _make_socket(self, port: int) -> zmq.Socket:
        sock = self.context.socket(zmq.REQ)
        sock.setsockopt(zmq.RCVTIMEO, self._ZMQ_TIMEOUT_MS)
        sock.setsockopt(zmq.SNDTIMEO, self._ZMQ_TIMEOUT_MS)
        sock.setsockopt(zmq.LINGER, 0)
        sock.connect(f"tcp://localhost:{port}")
        return sock

    def _reset_socket(self, attr: str, port: int):
        old = getattr(self, attr)
        try:
            old.close()
        except Exception:
            pass
        setattr(self, attr, self._make_socket(port))

    def get_joint_state(self) -> Optional[Dict]:
        with self._state_lock:
            try:
                self.state_socket.send_json({"cmd": "get_state"})
                return self.state_socket.recv_json()
            except zmq.Again:
                print("Timeout waiting for joint state, resetting state socket...")
                self._reset_socket("state_socket", self.state_port)
                return None
            except Exception as e:
                print(f"Error getting joint state: {e}")
                self._reset_socket("state_socket", self.state_port)
                return None

    def command_joint_positions(self, positions: np.ndarray):
        with self._command_lock:
            try:
                self.command_socket.send_json({
                    "cmd": "command_joint_state",
                    "positions": positions.tolist(),
                })
                self.command_socket.recv_json()
            except zmq.Again:
                print("Timeout sending joint command, resetting command socket...")
                self._reset_socket("command_socket", self.command_port)
            except Exception as e:
                print(f"Error commanding joints: {e}")
                self._reset_socket("command_socket", self.command_port)

    def close(self):
        try:
            self.command_socket.send_json({"cmd": "shutdown"})
            self.command_socket.recv_json()
        except Exception:
            pass
        self.command_socket.close()
        self.state_socket.close()
        self.context.term()


# ──────────────────────────────────────────────────────────────────────
# Robot
# ──────────────────────────────────────────────────────────────────────

class A1XRobot:
    """A1_X robot interface with ROS2 subprocess + local IK solver.

    Communication via ZMQ dual-socket bridge.
    EEF control uses CuRobo IK to convert Cartesian deltas to joint commands.
    """

    URDF_PATH = "/home/dungeon_master/A1_X/arm/install/mobiman/share/mobiman/urdf/A1X/urdf/a1x.urdf"

    def __init__(
        self,
        num_dofs: int = 7,
        node_name: str = "a1x_gello_node",
        port: int = 6100,
        python_path: str = "/usr/bin/python3",
        use_curobo_ik: bool = True,
        curobo_ik_service: Optional[str] = None,
        reset_joint_state: Optional[np.ndarray] = None,
    ):
        self._num_dofs = num_dofs
        self._port = port
        self._use_curobo_ik = use_curobo_ik
        self._curobo_ik_service = curobo_ik_service or os.environ.get("CUROBO_IK_SERVICE")
        self._reset_joint_state = reset_joint_state
        self._last_commanded_state = np.zeros(num_dofs)

        # Gripper EMA filter
        self._gripper_filtered = None
        self._gripper_alpha = 0.3

        # EE target position locking (anti-drift)
        self._locked_ee_pos: Optional[np.ndarray] = None
        self._locked_ee_quat: Optional[np.ndarray] = None  # [x,y,z,w]

        # Start ROS2 subprocess
        self._start_ros2_node(node_name, port, python_path)

        # ZMQ bridge
        print(f"Connecting to ROS2 node on ports {port} (cmd) and {port + 1} (state)...")
        self._bridge = A1XRobotBridge(port=port, state_port=port + 1)
        self._wait_for_connection()

        # IK solver
        self._ik_solver = None
        if use_curobo_ik and HAS_A1_KINEMATICS:
            self._init_ik_solver()

    # ------------------------------------------------------------------
    # Initialization helpers
    # ------------------------------------------------------------------

    def _start_ros2_node(self, node_name: str, port: int, python_path: str):
        script_path = os.path.join(os.path.dirname(__file__), "a1x_ros2_node.py")

        cmd_parts = [
            f"source /opt/ros/humble/setup.zsh &&",
            f"{python_path} {script_path}",
            f"--port {port} --node-name {node_name}",
        ]
        if self._use_curobo_ik:
            cmd_parts.append("--use-curobo-ik")
        if self._curobo_ik_service:
            cmd_parts.append(f"--curobo-ik-service {self._curobo_ik_service}")

        print(f"Starting ROS2 node subprocess ({'CuRobo IK' if self._use_curobo_ik else 'RelaxedIK'})...")
        self._ros2_process = subprocess.Popen(
            " ".join(cmd_parts),
            shell=True,
            executable="/bin/zsh",
            stdout=None,
            stderr=None,
        )
        time.sleep(2)
        if self._ros2_process.poll() is not None:
            print(f"ROS2 node failed to start (exit code: {self._ros2_process.returncode})")

    def _wait_for_connection(self, timeout: float = 10.0):
        print("Waiting for joint states from A1_X robot...")
        start = time.time()
        state = None
        while state is None and (time.time() - start) < timeout:
            state = self._bridge.get_joint_state()
            time.sleep(0.1)

        if state is not None:
            print("A1_X robot connected successfully")
            if state.get("joint_names"):
                print(f"Joint names: {state['joint_names']}")
        else:
            print(f"Warning: No joint states received after {timeout}s")

    def _init_ik_solver(self):
        print("Initializing A1Kinematics IK solver...")
        try:
            self._ik_solver = A1Kinematics(
                urdf_file=self.URDF_PATH,
                base_link="base_link",
                ee_link="gripper_link",
            )
            if self._reset_joint_state is not None:
                self._ik_solver.prev_q = torch.as_tensor(
                    self._reset_joint_state[:6],
                    **self._ik_solver.tensor_args.as_torch_dict(),
                )
            print("A1Kinematics IK solver initialized")
        except Exception as e:
            print(f"Failed to initialize A1Kinematics: {e}")
            self._ik_solver = None

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    def num_dofs(self) -> int:
        return self._num_dofs

    # ------------------------------------------------------------------
    # State queries
    # ------------------------------------------------------------------

    def get_state_snapshot(self) -> Optional[Dict]:
        """Atomic state snapshot: joints + EE pose + timestamps."""
        t_start = time.time()
        state = self._bridge.get_joint_state()
        t_zmq = time.time() - t_start
        
        if state is None:
            return None

        positions = state.get("positions")
        joint_positions = (
            _pad_or_trim(np.array(positions), self._num_dofs)
            if positions is not None
            else self._last_commanded_state.copy()
        )

        velocities = state.get("velocities")
        joint_velocities = (
            _pad_or_trim(np.array(velocities), self._num_dofs)
            if velocities is not None
            else np.zeros(self._num_dofs)
        )

        ee_pos = np.array(state.get("ee_pos", [0, 0, 0]))
        ee_quat = np.array(state.get("ee_quat", [0, 0, 0, 1]))
        joint_ts = state.get("joint_ts", 0.0)
        ee_ts = state.get("ee_ts", 0.0)
        
        t_now = time.time()

        return {
            "joint_positions": joint_positions,
            "joint_velocities": joint_velocities,
            "ee_pos": ee_pos,
            "ee_quat": ee_quat,
            "joint_ts": joint_ts,
            "ee_ts": ee_ts,
            "ts_diff": abs(joint_ts - ee_ts),
            "zmq_latency_ms": t_zmq * 1000,
            "joint_age_ms": (t_now - joint_ts) * 1000 if joint_ts > 0 else 0.0,
            "ee_age_ms": (t_now - ee_ts) * 1000 if ee_ts > 0 else 0.0,
        }

    def get_joint_state(self) -> np.ndarray:
        state = self._bridge.get_joint_state()
        if state is None or state.get("positions") is None:
            return self._last_commanded_state.copy()
        return _pad_or_trim(np.array(state["positions"]), self._num_dofs)

    def get_eef_pose(self) -> tuple:
        """Returns (position[3], quaternion[4] xyzw)."""
        state = self._bridge.get_joint_state()
        if state is None:
            return np.zeros(3), np.array([0, 0, 0, 1.0])
        return (
            np.array(state.get("ee_pos", [0, 0, 0])),
            np.array(state.get("ee_quat", [0, 0, 0, 1])),
        )

    def get_observations(self) -> Dict[str, np.ndarray]:
        snapshot = self.get_state_snapshot()
        if snapshot is None:
            joint_positions = self._last_commanded_state.copy()
            joint_velocities = np.zeros(self._num_dofs)
            ee_pos = np.zeros(3)
            ee_quat = np.array([0, 0, 0, 1.0])
        else:
            joint_positions = snapshot["joint_positions"]
            joint_velocities = snapshot["joint_velocities"]
            ee_pos = snapshot["ee_pos"]
            ee_quat = snapshot["ee_quat"]

        ee_rpy = R.from_quat(ee_quat).as_euler("xyz", degrees=False)
        gripper_position = np.array([joint_positions[-1]]) if self._num_dofs >= 7 else np.array([0.0])

        return {
            "joint_positions": joint_positions,
            "joint_velocities": joint_velocities,
            "ee_pos_quat": np.concatenate([ee_pos, ee_quat]),
            "ee_pos_rot_gripper": np.concatenate([ee_pos, ee_rpy, gripper_position / 100]),
            "gripper_position": gripper_position,
        }

    # ------------------------------------------------------------------
    # Joint commands
    # ------------------------------------------------------------------

    def command_joint_state(self, joint_state: np.ndarray, from_gello: bool = True) -> None:
        assert len(joint_state) == self._num_dofs, (
            f"Expected {self._num_dofs} joint values, got {len(joint_state)}"
        )
        self._last_commanded_state = joint_state.copy()

        mapped = self._map_to_a1x(joint_state) if from_gello else joint_state

        # Apply gripper EMA filter (index 6)
        if len(mapped) >= 7:
            raw_gripper = mapped[6]
            if self._gripper_filtered is None:
                self._gripper_filtered = raw_gripper
            else:
                self._gripper_filtered += self._gripper_alpha * (raw_gripper - self._gripper_filtered)
            mapped = mapped.copy()
            mapped[6] = self._gripper_filtered

        self._bridge.command_joint_positions(mapped)

    # ------------------------------------------------------------------
    # EE target locking (anti-drift)
    # ------------------------------------------------------------------

    def lock_ee_target(self) -> bool:
        """Lock current EE pose as the target.

        Call after reset or whenever the robot reaches a desired position.
        Subsequent command_eef_pose calls accumulate deltas on this locked
        target instead of reading drifted feedback, preventing servo drift.
        """
        snapshot = self.get_state_snapshot()
        if snapshot is None:
            print("[lock_ee_target] Failed to get state snapshot")
            return False
        self._locked_ee_pos = snapshot["ee_pos"].copy()
        self._locked_ee_quat = snapshot["ee_quat"].copy()  # [x,y,z,w]
        print(f"[lock_ee_target] Locked EE pos={self._locked_ee_pos}, quat={self._locked_ee_quat}")
        return True

    def unlock_ee_target(self):
        """Unlock EE target, reverting to feedback-based control."""
        self._locked_ee_pos = None
        self._locked_ee_quat = None
        print("[unlock_ee_target] EE target unlocked")

    # ------------------------------------------------------------------
    # EEF control
    # ------------------------------------------------------------------

    def command_eef_pose(self, eef_delta: np.ndarray, wait_for_completion: bool = True,
                         timeout: float = 2.0, pos_tolerance_m: float = 0.003) -> Optional[dict]:
        """Apply EEF delta [dx,dy,dz, drx,dry,drz, gripper_mm] via IK.

        When EE target is locked (via lock_ee_target), deltas accumulate
        on the locked target — the robot always commands towards the
        intended target, not the drifted feedback.
        When unlocked, falls back to feedback-based control.
        """
        if self._ik_solver is None:
            print("IK solver not available")
            return None
        if len(eef_delta) != 7:
            print(f"Invalid eef_delta length: {len(eef_delta)}, expected 7")
            return None

        snapshot = self.get_state_snapshot()
        if snapshot is None:
            print("Failed to get state snapshot")
            return None

        if snapshot["ts_diff"] > 0.05:
            print(f"Warning: joint/EE timestamp diff = {snapshot['ts_diff']*1000:.1f}ms")

        current_joints = snapshot["joint_positions"][:6]
        ee_pos = snapshot["ee_pos"]
        ee_quat = snapshot["ee_quat"]  # [x, y, z, w]

        # --- Determine target pos & quat ---
        delta_pos = np.array(eef_delta[:3])
        delta_rot = np.array(eef_delta[3:6])
        gripper_position = eef_delta[6]

        if self._locked_ee_pos is not None:
            # ---- Locked mode: accumulate delta on locked target ----
            self._locked_ee_pos = self._locked_ee_pos + delta_pos

            if not np.allclose(delta_rot, 0.0, atol=1e-8):
                locked_rotation = R.from_quat(self._locked_ee_quat)
                delta_rotation = R.from_euler("xyz", delta_rot)
                self._locked_ee_quat = (delta_rotation * locked_rotation).as_quat()

            target_pos = self._locked_ee_pos.copy()
            target_quat = self._locked_ee_quat.copy()

            # Drift diagnostic
            pos_drift = np.linalg.norm(ee_pos - target_pos)
            print(f"[EE锁定] 目标pos={target_pos}, 反馈pos={ee_pos}, 位置偏差={pos_drift*1000:.2f}mm")
        else:
            # ---- Unlocked mode: feedback-based ----
            target_pos = ee_pos + delta_pos
            if np.allclose(delta_rot, 0.0, atol=1e-8):
                target_quat = ee_quat
            else:
                current_rotation = R.from_quat(ee_quat)
                delta_rotation = R.from_euler("xyz", delta_rot)
                target_quat = (delta_rotation * current_rotation).as_quat()

        print(f"当前关节角: {current_joints}")
        print(f"指令EEF delta: pos {delta_pos}, rot {delta_rot}, gripper {gripper_position}mm")

        # Update IK seed to actual joint feedback before solving
        self._ik_solver.prev_q = torch.as_tensor(
            current_joints,
            dtype=torch.float32,
            device=self._ik_solver.tensor_args.device,
        ).unsqueeze(0)

        try:
            result = self._ik_solver.solve_ik(pos=target_pos, quat=target_quat)
        except Exception as e:
            print(f"IK solve failed: {e}")
            return None

        if not result.success.cpu().numpy().any():
            print("IK solver failed to find solution")
            return None

        joint_solution = result.js_solution.position.cpu().numpy()[:6]
        max_diff = np.abs(joint_solution - current_joints).max()
        print(f"IK solution: max joint diff = {max_diff:.4f} rad ({np.rad2deg(max_diff):.2f} deg)")

        full_command = np.concatenate([joint_solution, [gripper_position]])
        self.command_joint_state(full_command, from_gello=False)

        # ── 等待执行到位：直接监控 EEF XYZ 位置误差 ──────────────────────────
        # 用 get_state_snapshot() 中的 ee_pos 与 target_pos 比较，
        # 物理含义直接（mm 级），不受臂型/构型影响。
        reached = False
        final_error_m = float("inf")

        if wait_for_completion:
            pos_tolerance_m = pos_tolerance_m   # 使用传入的容差
            poll_interval   = 0.01
            start_wait      = time.time()

            while time.time() - start_wait < timeout:
                snap = self.get_state_snapshot()
                if snap is None:
                    time.sleep(poll_interval)
                    continue

                pos_error = float(np.linalg.norm(snap["ee_pos"] - target_pos))
                final_error_m = pos_error

                if pos_error < pos_tolerance_m:
                    reached = True
                    break

                time.sleep(poll_interval)

            if not reached:
                print(
                    f"[command_eef_pose] 超时: pos error={final_error_m*1000:.2f} mm "
                    f"(tolerance=1.0 mm, timeout={timeout}s)"
                )

        return {
            "target_joints":  joint_solution.tolist(),
            "target_pos":     target_pos.tolist(),
            "reached":        reached,
            "final_error_m":  final_error_m,
            "final_error_mm": final_error_m * 1000,
            "gripper":        float(gripper_position),
        }

    def command_eef_chunk(
        self,
        eef_poses: list,
        correction_threshold: float = 0.005,
        max_corrections: int = 2,
        timeout_per_step: float = 10.0,
    ) -> dict:
        """Execute a sequence of EEF delta poses with iterative correction."""
        try:
            self._bridge.command_socket.send_json({
                "cmd": "command_eef_chunk",
                "poses": eef_poses,
                "correction_threshold": correction_threshold,
                "max_corrections": max_corrections,
                "timeout_per_step": timeout_per_step,
            })
            total_timeout_ms = int((timeout_per_step + 5.0) * len(eef_poses) * 1000)
            self._bridge.command_socket.setsockopt(zmq.RCVTIMEO, total_timeout_ms)
            result = self._bridge.command_socket.recv_json()
            self._bridge.command_socket.setsockopt(zmq.RCVTIMEO, self._bridge._ZMQ_TIMEOUT_MS)
            return result
        except zmq.Again:
            return {"status": "error", "error": "ZMQ timeout waiting for chunk execution"}
        except Exception as e:
            return {"status": "error", "error": str(e)}

    # ------------------------------------------------------------------
    # Gello <-> A1X joint mapping
    # ------------------------------------------------------------------

    @staticmethod
    def _map_to_a1x(joint_state: np.ndarray) -> np.ndarray:
        """Gello joint positions -> A1X joint positions (7 DOF)."""
        js = np.asarray(joint_state, dtype=float)
        if js.size != 7:
            raise ValueError("Expected 7 joint values for Gello->A1X mapping")
        return _linear_map(js, _GELLO_RANGE_START, _GELLO_RANGE_END, _A1X_RANGE_START, _A1X_RANGE_END)

    @staticmethod
    def _map_from_a1x(a1x_joints: np.ndarray) -> np.ndarray:
        """A1X joint positions -> Gello joint positions (7 DOF)."""
        a1x = np.asarray(a1x_joints, dtype=float)
        if a1x.size != 7:
            raise ValueError("Expected 7 joint values for A1X->Gello mapping")
        return _linear_map(a1x, _A1X_RANGE_START, _A1X_RANGE_END, _GELLO_RANGE_START, _GELLO_RANGE_END)

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def close(self):
        if hasattr(self, "_bridge"):
            self._bridge.close()
        if hasattr(self, "_ros2_process"):
            self._ros2_process.terminate()
            self._ros2_process.wait(timeout=5)


def main():
    robot = A1XRobot(num_dofs=7)

    print(f"\nDOFs: {robot.num_dofs()}")
    print(f"Joint state: {robot.get_joint_state()}")

    obs = robot.get_observations()
    for key, value in obs.items():
        print(f"  {key}: {value}")

    robot.command_joint_state(robot.get_joint_state(), from_gello=False)
    time.sleep(0.5)
    robot.close()
    print("Done")


if __name__ == "__main__":
    main()
