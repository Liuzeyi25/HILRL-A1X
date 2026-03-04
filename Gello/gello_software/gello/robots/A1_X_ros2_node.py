#!/usr/bin/env python3
"""
ROS2 node for A1_X robot that communicates via ZMQ.
This script runs with system Python 3.10 and ROS2.
"""

import argparse
import threading

import numpy as np
import rclpy
import zmq
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


class A1XRobotZMQNode(Node):
    """ROS2 node that bridges to ZMQ for A1_X robot control."""
    
    # Safety thresholds
    SAFE_EFFORT_JOINT_2 = 8.0  # N·m
    SAFE_EFFORT_JOINT_3 = 7.0  # N·m
    SAFE_EFFORT_JOINT_4 = 5.0  # N·m
    SAFE_Z_MIN = 0.083  # meters

    def __init__(self, node_name: str = "a1x_gello_node", zmq_port: int = 6100):
        super().__init__(node_name)
        
        # Publisher for commanding joint states (arm joints 0-5)
        self.joint_command_pub = self.create_publisher(
            JointState,
            "/motion_target/target_joint_state_arm",
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
        
        # haoyuan add
        self.eef_state_sub = self.create_subscription(
            PoseStamped,
            "/hdas/pose_ee_arm",
            self.eef_state_callback,
            10
        )
        
        # Current joint state
        self._current_joint_positions = None
        self._current_joint_torques = None
        self._current_joint_velocities = None
        self._current_end_effector_pose = None
        self._joint_names = None
        self._lock = threading.Lock()
        
        # FK solver
        self._fk_model = None
        self._fk_data = None
        self._fk_ee_frame_id = None
        self._init_fk_solver()
        
        # ZMQ server
        self.zmq_context = zmq.Context()
        self.zmq_socket = self.zmq_context.socket(zmq.REP)
        self.zmq_socket.bind(f"tcp://*:{zmq_port}")
        
        self._running = True
        
        # Start ZMQ server thread
        self.zmq_thread = threading.Thread(target=self.zmq_server_loop, daemon=False)
        self.zmq_thread.start()
        
        self.get_logger().info(f"A1XRobotZMQNode initialized on port {zmq_port}")

    def joint_state_callback(self, msg: JointState):
        """Callback for receiving joint state updates."""
        with self._lock:
            self._joint_names = list(msg.name)
            self._current_joint_positions = list(msg.position)
            if msg.velocity:
                self._current_joint_velocities = list(msg.velocity)
            else:
                self._current_joint_velocities = [0.0] * len(msg.position)
            if msg.effort:
                self._current_joint_torques = list(msg.effort)
            else:
                self._current_joint_torques = [0.0] * len(msg.position)
                
    # haoyuan add            
    def eef_state_callback(self, msg: PoseStamped):
        """Callback for receiving end-effector state updates."""
        with self._lock:
            self._current_end_effector_pose = {
                'position': {'x': msg.pose.position.x, 'y': msg.pose.position.y, 'z': msg.pose.position.z},
                'orientation': {'x': msg.pose.orientation.x, 'y': msg.pose.orientation.y, 
                                'z': msg.pose.orientation.z, 'w': msg.pose.orientation.w}
            }
    
    # haoyuan add 
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
    
    # haoyuan add 
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

    def zmq_server_loop(self):
        """Handle ZMQ requests in a separate thread."""
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
                                "eef_pose": self._current_end_effector_pose
                            }
                        self.zmq_socket.send_json(response)
                    
                    elif cmd == "command":
                        positions = request.get("positions", [])
                        self.publish_joint_command(positions)
                        self.zmq_socket.send_json({"status": "ok"})
                    
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
        # Separate arm joints (0-5) and gripper (6)
        if len(positions) >= 7:
            arm_positions = positions[:6]
            gripper_position = positions[6]
        else:
            arm_positions = positions
            gripper_position = None
        
        # Safety check: FK-based Z limit
        fk_result = self.compute_fk(arm_positions)
        print(f" @@@@@ FK Result: {fk_result['position']['z'] if fk_result else 'None'}")
        # print(f" @@@@@ FK Result: {fk_result['position'] if fk_result else 'None'}")
        if fk_result and fk_result['position']['z'] < self.SAFE_Z_MIN:
            print(f" Safety stop: FK Z position {fk_result['position']['z']:.3f} m below limit {self.SAFE_Z_MIN:.3f} m")
            return
        
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
        msg.velocity = [6.0] * len(arm_positions)  # 恒定速度 20.0
        msg.effort = []
        
        self.joint_command_pub.publish(msg)
        # Debug log: show published arm command
        self.get_logger().info(f"[ARM] Published: {[f'{p:.3f}' for p in arm_positions]}")
        
        # Publish gripper command separately - Float32 with range 0-100mm
        if gripper_position is not None:
            gripper_msg = Float32()
            gripper_msg.data = float(gripper_position)
            self.gripper_command_pub.publish(gripper_msg)
            # Debug log: show published gripper command
            self.get_logger().info(f"[GRIPPER] Published: {gripper_position:.3f} mm")

    def cleanup(self):
        """Clean up resources."""
        self._running = False
        self.zmq_thread.join(timeout=2)
        self.zmq_socket.close()
        self.zmq_context.term()


def main():
    parser = argparse.ArgumentParser(description="A1_X ROS2 ZMQ Bridge Node")
    parser.add_argument("--port", type=int, default=6100, help="ZMQ port")
    parser.add_argument("--node-name", type=str, default="a1x_gello_node", help="ROS2 node name")
    args = parser.parse_args()
    
    rclpy.init()
    
    node = A1XRobotZMQNode(node_name=args.node_name, zmq_port=args.port)
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.cleanup()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
