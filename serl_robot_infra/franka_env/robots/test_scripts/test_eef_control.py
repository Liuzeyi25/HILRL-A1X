#!/usr/bin/env python3
"""
Test end-effector pose control for A1_X robot.
This script tests the publish_eef_command function by moving the robot in Cartesian space.
"""

import sys
import time
import numpy as np
import zmq

# Add parent directory to path
sys.path.insert(0, '.')

from a1x_robot import A1XRobot


def test_eef_movement():
    """Test end-effector control by moving 3cm in X direction."""
    
    print("=" * 70)
    print("A1_X End-Effector Control Test")
    print("=" * 70)
    print("\nThis script will move the robot 3cm in the X direction")
    print("using Cartesian space control (end-effector pose).")
    print()
    
    # Confirm before proceeding
    response = input("⚠️  Make sure the path is clear. Continue? (y/n): ")
    if response.lower() != 'y':
        print("❌ Aborted by user.")
        return
    
    print("\n🤖 Initializing A1_X robot...")
    try:
        robot = A1XRobot(
            num_dofs=7,
            node_name="a1x_eef_test_node",
            port=6100,
            python_path="/usr/bin/python3"
        )
    except Exception as e:
        print(f"❌ Failed to initialize robot: {e}")
        return
    
    print("✅ Robot initialized successfully")
    
    # Wait for robot to receive initial pose feedback (already handled in A1XRobot init)
    print("\n⏳ Ready to start tests (node initialization waited for joint state)")
    
    try:
        # Test 1: Move 3cm in X direction
        print("\n" + "="*70)
        print("Test 1: Translation - Moving 3cm in X direction")
        print("="*70)
        print("   Command format: [delta_x, delta_y, delta_z, delta_rx, delta_ry, delta_rz, gripper]")
        
        # Define the movement
        delta_x = 0.03  # 3cm = 0.03m
        delta_y = 0.0
        delta_z = 0.0
        delta_rx = 0.0  # No rotation
        delta_ry = 0.0
        delta_rz = 0.0
        gripper = 100.0  # Keep gripper open
        
        eef_pose = [delta_x, delta_y, delta_z, delta_rx, delta_ry, delta_rz, gripper]
        
        print(f"   Sending command: {eef_pose}")
        
        # Send command via ZMQ
        context = zmq.Context()
        socket = context.socket(zmq.REQ)
        socket.setsockopt(zmq.RCVTIMEO, 15000)  # 15 second timeout (node waits for motion completion)
        socket.setsockopt(zmq.SNDTIMEO, 15000)
        socket.connect("tcp://localhost:6100")
        
        socket.send_json({
            "cmd": "command_eef_pose",
            "pose": eef_pose
        })
        
        response = socket.recv_json()
        print(f"   Response: {response}")
        
        if response.get("status") == "ok":
            print("✅ Command completed successfully!")
        else:
            print(f"❌ Command failed: {response}")
        
        socket.close()
        context.term()
        
        # Test 2: Rotation around Z axis (yaw) - 15 degrees
        print("\n" + "="*70)
        print("Test 2: Rotation - Rotating 15° around Z axis (yaw)")
        print("="*70)
        
        delta_rz = np.deg2rad(15)  # 15 degrees in radians
        eef_pose_rot = [0.0, 0.0, 0.0, 0.0, 0.0, delta_rz, gripper]
        print(f"   Sending command: {eef_pose_rot}")
        print(f"   (rotation: {np.rad2deg(delta_rz):.1f} degrees around Z axis)")
        
        context = zmq.Context()
        socket = context.socket(zmq.REQ)
        socket.setsockopt(zmq.RCVTIMEO, 15000)
        socket.setsockopt(zmq.SNDTIMEO, 15000)
        socket.connect("tcp://localhost:6100")
        
        socket.send_json({
            "cmd": "command_eef_pose",
            "pose": eef_pose_rot
        })
        
        response = socket.recv_json()
        print(f"   Response: {response}")
        
        if response.get("status") == "ok":
            print("✅ Rotation completed!")
        
        socket.close()
        context.term()
        
        # Test 3: Rotation around Y axis (pitch) - 10 degrees
        print("\n" + "="*70)
        print("Test 3: Rotation - Rotating 10° around Y axis (pitch)")
        print("="*70)
        
        delta_ry = np.deg2rad(10)  # 10 degrees in radians
        eef_pose_rot2 = [0.0, 0.0, 0.0, 0.0, delta_ry, 0.0, gripper]
        print(f"   Sending command: {eef_pose_rot2}")
        print(f"   (rotation: {np.rad2deg(delta_ry):.1f} degrees around Y axis)")
        
        context = zmq.Context()
        socket = context.socket(zmq.REQ)
        socket.setsockopt(zmq.RCVTIMEO, 15000)
        socket.setsockopt(zmq.SNDTIMEO, 15000)
        socket.connect("tcp://localhost:6100")
        
        socket.send_json({
            "cmd": "command_eef_pose",
            "pose": eef_pose_rot2
        })
        
        response = socket.recv_json()
        print(f"   Response: {response}")
        
        if response.get("status") == "ok":
            print("✅ Rotation completed!")
        
        socket.close()
        context.term()
        
        # Test 4: Combined translation and rotation
        print("\n" + "="*70)
        print("Test 4: Combined - Move 2cm in Y and rotate 10° around X")
        print("="*70)
        
        delta_y = 0.02  # 2cm
        delta_rx = np.deg2rad(10)  # 10 degrees around X (roll)
        eef_pose_combined = [0.0, delta_y, 0.0, delta_rx, 0.0, 0.0, gripper]
        print(f"   Sending command: {eef_pose_combined}")
        print(f"   (translation: 2cm in Y, rotation: {np.rad2deg(delta_rx):.1f}° around X)")
        
        context = zmq.Context()
        socket = context.socket(zmq.REQ)
        socket.setsockopt(zmq.RCVTIMEO, 15000)
        socket.setsockopt(zmq.SNDTIMEO, 15000)
        socket.connect("tcp://localhost:6100")
        
        socket.send_json({
            "cmd": "command_eef_pose",
            "pose": eef_pose_combined
        })
        
        response = socket.recv_json()
        print(f"   Response: {response}")
        
        if response.get("status") == "ok":
            print("✅ Combined movement completed!")
        
        socket.close()
        context.term()
        
        # Test 5: Return to original position (undo all movements)
        print("\n" + "="*70)
        print("Test 5: Returning to original position")
        print("="*70)
        
        # Reverse all movements
        eef_pose_back = [-delta_x, -delta_y, 0.0, -delta_rx, -delta_ry, -delta_rz, gripper]
        print(f"   Sending command: {eef_pose_back}")
        
        context = zmq.Context()
        socket = context.socket(zmq.REQ)
        socket.setsockopt(zmq.RCVTIMEO, 15000)
        socket.setsockopt(zmq.SNDTIMEO, 15000)
        socket.connect("tcp://localhost:6100")
        
        socket.send_json({
            "cmd": "command_eef_pose",
            "pose": eef_pose_back
        })
        
        response = socket.recv_json()
        print(f"   Response: {response}")
        
        if response.get("status") == "ok":
            print("✅ Return command completed successfully! Robot should be back at original position.")
        
        socket.close()
        context.term()
        
        # Test 6: Four small sequential X translations (+1cm each), each waits for previous to complete
        print("\n" + "="*70)
        print("Test 6: Sequential small translations - 4 x +1cm on X, wait for each to complete")
        print("="*70)
        print("   Strategy: Use absolute target tracking to minimize cumulative error")

        inc = 0.01  # 1cm per step
        steps = 4
        context = zmq.Context()
        socket = context.socket(zmq.REQ)
        socket.setsockopt(zmq.RCVTIMEO, 15000)
        socket.setsockopt(zmq.SNDTIMEO, 15000)
        socket.connect("tcp://localhost:6100")

        # Get starting position by querying current EE pose via get_state
        socket.send_json({"cmd": "get_state"})
        state_resp = socket.recv_json()
        
        # We need to get actual EE pose from the node
        # For now, we'll use a workaround: send a zero-delta command to get current pose in response
        # Better: add a "get_ee_pose" command to the node
        print("   Recording initial position for absolute target tracking...")
        
        # Send a zero movement to trigger node's pose print and establish baseline
        zero_cmd = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, gripper]
        socket.send_json({
            "cmd": "command_eef_pose",
            "pose": zero_cmd
        })
        baseline_resp = socket.recv_json()
        if baseline_resp.get("status") == "ok" and baseline_resp.get("info"):
            start_pos = np.array(baseline_resp["info"]["current_pos"])
            print(f"   Initial EE position: {start_pos}")
        else:
            print("   ⚠️  Could not get initial position, using relative deltas")
            start_pos = None

        all_ok = True
        for i in range(steps):
            if start_pos is not None:
                # Absolute target: start + (i+1) * increment
                # We need to compute delta from CURRENT to absolute target
                # But we send delta to node, so we compute: target - current
                # Since node uses current + delta, we need to be careful
                
                # Strategy: compute the absolute target position for this step
                abs_target_offset = np.array([inc * (i + 1), 0.0, 0.0])
                
                # Node will add delta to current position
                # So we just send the single-step increment as before
                # BUT we verify we're close to the absolute target in info
                eef_step = [inc, 0.0, 0.0, 0.0, 0.0, 0.0, gripper]
                print(f"   Step {i+1}/{steps}: sending delta {eef_step[:3]}")
                print(f"   Expected absolute position: {start_pos + abs_target_offset}")
            else:
                eef_step = [inc, 0.0, 0.0, 0.0, 0.0, 0.0, gripper]
                print(f"   Step {i+1}/{steps}: sending {eef_step}")
            
            socket.send_json({
                "cmd": "command_eef_pose",
                "pose": eef_step
            })
            try:
                response = socket.recv_json()
            except Exception as e:
                print(f"   ❌ Step {i+1} failed to receive response: {e}")
                all_ok = False
                break

            print(f"   Response: {response}")
            
            if response.get("status") == "ok":
                if response.get("info") and start_pos is not None:
                    actual_pos = np.array(response["info"]["current_pos"])
                    expected_pos = start_pos + np.array([inc * (i + 1), 0.0, 0.0])
                    cumulative_err = np.linalg.norm(actual_pos - expected_pos)
                    print(f"   ✅ Step {i+1} completed")
                    print(f"   📏 Cumulative error from start: {cumulative_err*1000:.2f}mm")
                else:
                    print(f"   ✅ Step {i+1} completed")
            else:
                print(f"   ❌ Step {i+1} failed: {response}")
                all_ok = False
                break

        socket.close()
        context.term()

        if all_ok:
            print("\n✅ Test 6 completed: all sequential X translations succeeded")
        else:
            print("\n⚠️ Test 6 aborted due to error in sequence")

        print("\n✅ End-effector control test complete!")
        
    except zmq.Again:
        print("\n❌ Timeout: No response from ROS2 node")
        print("   Make sure the ROS2 node is receiving pose feedback from /hdas/pose_ee_arm")
    except KeyboardInterrupt:
        print("\n⚠️  Interrupted by user (Ctrl+C)")
    except Exception as e:
        print(f"\n❌ Error during test: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("\n🧹 Closing robot connection...")
        robot.close()
        time.sleep(0.5)
        print("✅ Done!")


if __name__ == "__main__":
    test_eef_movement()
