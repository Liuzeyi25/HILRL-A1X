#!/usr/bin/env python3
"""
Quick test script for A1_X robot integration.
This helps verify that the bridge architecture is working.
"""

import time
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

def test_zmq_bridge():
    """Test ZMQ communication without ROS2."""
    import zmq
    
    print("Testing ZMQ communication...")
    context = zmq.Context()
    socket = context.socket(zmq.REQ)
    
    try:
        socket.connect("tcp://localhost:6100")
        socket.send_json({"cmd": "get_state"})
        
        if socket.poll(timeout=1000):  # 1 second timeout
            response = socket.recv_json()
            print(f"✓ ZMQ communication successful: {response}")
            return True
        else:
            print("✗ No response from ROS2 node (timeout)")
            return False
    except Exception as e:
        print(f"✗ ZMQ connection failed: {e}")
        print("  Make sure the ROS2 node is running:")
        print("  source /opt/ros/humble/setup.bash")
        print("  /usr/bin/python3 gello/robots/A1_X_ros2_node.py --port 6100")
        return False
    finally:
        socket.close()
        context.term()


def test_a1x_robot():
    """Test A1XRobot class."""
    print("\nTesting A1XRobot class...")
    
    try:
        from gello.robots.A1_X import A1XRobot
        
        print("Creating A1XRobot instance...")
        robot = A1XRobot(num_dofs=7, port=6100)
        
        print(f"✓ Robot initialized with {robot.num_dofs()} DOFs")
        
        # Test getting joint state
        joint_state = robot.get_joint_state()
        print(f"✓ Current joint state: {joint_state}")
        
        # Test observations
        obs = robot.get_observations()
        print(f"✓ Observations keys: {list(obs.keys())}")
        
        # Test commanding (same position, no movement)
        print("Testing command (commanding current position)...")
        robot.command_joint_state(joint_state)
        time.sleep(0.1)
        print("✓ Command successful")
        
        # Cleanup
        print("Closing robot...")
        robot.close()
        print("✓ Robot closed successfully")
        
        return True
        
    except Exception as e:
        print(f"✗ A1XRobot test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_ros2_topics():
    """Check if ROS2 topics are available."""
    print("\nChecking ROS2 topics...")
    
    import subprocess
    
    try:
        result = subprocess.run(
            ["bash", "-c", "source /opt/ros/humble/setup.bash && ros2 topic list"],
            capture_output=True,
            text=True,
            timeout=5
        )
        
        topics = result.stdout.strip().split('\n')
        
        required_topics = [
            "/motion_target/target_joint_state_arm",
            "/hdas/feedback_arm"
        ]
        
        for topic in required_topics:
            if topic in topics:
                print(f"✓ Found topic: {topic}")
            else:
                print(f"✗ Missing topic: {topic}")
        
        return all(topic in topics for topic in required_topics)
        
    except Exception as e:
        print(f"✗ Could not check ROS2 topics: {e}")
        return False


def main():
    print("="*60)
    print("A1_X Robot Integration Test")
    print("="*60)
    
    # Test 1: ROS2 topics (requires robot system running)
    test_ros2_topics()
    
    # Test 2: ZMQ bridge (requires ROS2 node running)
    zmq_ok = test_zmq_bridge()
    
    # Test 3: A1XRobot class
    if zmq_ok:
        robot_ok = test_a1x_robot()
    else:
        print("\nSkipping A1XRobot test (ZMQ bridge not available)")
        print("\nTo start the ROS2 bridge node manually:")
        print("  cd /tmp")
        print("  source /opt/ros/humble/setup.bash")
        print("  /usr/bin/python3 /home/dungeon_master/Gello/gello_software/gello/robots/A1_X_ros2_node.py --port 6100")
        robot_ok = False
    
    print("\n" + "="*60)
    if zmq_ok and robot_ok:
        print("✓ All tests passed!")
    else:
        print("✗ Some tests failed. See messages above.")
    print("="*60)


if __name__ == "__main__":
    main()
