#!/usr/bin/env python3
"""
Test Gello bidirectional control:
1. Gello follows robot to target position
2. Switch to teleoperation mode where robot follows Gello
"""

import sys
import time
import numpy as np
import threading

# Add paths
sys.path.insert(0, '.')
sys.path.insert(0, 'Gello/gello_software')

from a1x_robot import A1XRobot
from gello.agents.gello_agent import GelloAgent, DynamixelRobotConfig
from gello.agents.gello_follower import GelloFollower


class GelloBidirectionalTest:
    """Test bidirectional control between Gello and A1_X robot."""
    
    def __init__(
        self,
        robot_port: int = 6100,
        gello_port: str = "/dev/serial/by-id/usb-FTDI_USB__-__Serial_Converter_FTA7NNNU-if00-port0",
    ):
        """
        Initialize test with robot and Gello.
        
        Args:
            robot_port: ZMQ port for A1_X robot
            gello_port: Serial port for Gello device
        """
        print("=" * 70)
        print("Gello Bidirectional Control Test")
        print("=" * 70)
        
        # Initialize robot
        print("\n🤖 Initializing A1_X robot...")
        self.robot = A1XRobot(
            num_dofs=7,
            node_name="a1x_gello_test_node",
            port=robot_port,
            python_path="/usr/bin/python3"
        )
        print("✅ Robot initialized")
        
        # Wait for robot to be ready
        time.sleep(2.0)
        
        # Initialize Gello with explicit configuration (same as working diagnostic)
        print("\n🎮 Initializing Gello...")
        
        # Use the same configuration as the working diagnostic script
        gello_config = DynamixelRobotConfig(
            joint_ids=[1, 2, 3, 4, 5, 6],
            joint_offsets=[1.5708, 4.71239, 4.71239, 3.14159, 1.5708, 3.14159],
            joint_signs=[1.0, -1.0, -1.0, -1.0, 1.0, 1.0],
            gripper_config=[7, 139.66015625, 199.16015625]
        )
        
        try:
            self.gello = GelloAgent(
                port=gello_port,
                dynamixel_config=gello_config
            )
            print("✅ Gello initialized")
            
            # Verify Gello is working by reading position
            try:
                test_pos = self.gello.act({})
                print(f"   Gello position read successful: {test_pos.shape[0]} joints")
            except Exception as e:
                print(f"⚠️  Warning: Could not read Gello position: {e}")
                
        except Exception as e:
            print(f"❌ Failed to initialize Gello: {e}")
            print("\n💡 Troubleshooting:")
            print("   1. Make sure Gello device has power (12V adapter)")
            print("   2. Check USB connection")
            print("   3. Run diagnostic: python examples/diagnose_gello_follower.py")
            self.robot.close()
            raise
        
        # Initialize GelloFollower for proper position control
        self.gello_follower = GelloFollower(self.gello._robot)
        print("   GelloFollower initialized for position control")
        
        # Test state
        self.mode = "follow"  # "follow" or "teleoperation"
        self.running = True
        self.keyboard_thread = None
        
    def get_robot_joint_state(self) -> np.ndarray:
        """Get current robot joint state (A1_X format)."""
        return self.robot.get_joint_state()
    
    def get_gello_joint_state(self) -> np.ndarray:
        """Get current Gello joint state."""
        return self.gello.act({})  # GelloAgent.act() returns joint positions
    
    def a1x_to_gello_mapping(self, a1x_joints: np.ndarray) -> np.ndarray:
        """
        Convert A1_X joint positions to Gello joint positions.
        Uses the inverse mapping from A1XRobot.
        """
        if hasattr(self.robot, '_map_from_a1x'):
            gello_joints = self.robot._map_from_a1x(a1x_joints)
            return gello_joints
        else:
            print("⚠️  Robot does not have _map_from_a1x method")
            return a1x_joints  # Fallback: assume same joint space
    
    def gello_to_a1x_mapping(self, gello_joints: np.ndarray) -> np.ndarray:
        """
        Convert Gello joint positions to A1_X joint positions.
        Uses the forward mapping from A1XRobot.
        """
        if hasattr(self.robot, '_map_to_a1x'):
            a1x_joints = self.robot._map_to_a1x(gello_joints)
            return a1x_joints
        else:
            print("⚠️  Robot does not have _map_to_a1x method")
            return gello_joints  # Fallback: assume same joint space
    
    def slow_follow_to_target(self, target_gello_joints: np.ndarray, duration: float = 3.0):
        """
        Slowly move Gello to target position using GelloFollower.
        
        Args:
            target_gello_joints: Target Gello joint positions [7]
            duration: Time to reach target (seconds)
        """
        # Get current Gello position
        current_pos = self.get_gello_joint_state()
        
        # Number of steps for smooth motion (20 Hz control rate)
        num_steps = int(duration * 20)
        
        print(f"\n📍 Current Gello position:")
        print(f"   {current_pos}")
        print(f"🎯 Target Gello position:")
        print(f"   {target_gello_joints}")
        print(f"⏱️  Moving in {num_steps} steps over {duration:.1f}s...")
        
        # Start follower mode (enables position control)
        print("\n🔄 Starting GelloFollower position control...")
        self.gello_follower.start()
        time.sleep(0.5)  # Give time for mode switch
        
        try:
            # Linear interpolation with ease-in-out
            for step in range(num_steps + 1):
                t = step / num_steps  # 0.0 to 1.0
                
                # Smooth interpolation (ease-in-out cubic)
                t_smooth = 3 * t**2 - 2 * t**3
                
                interpolated_pos = current_pos + t_smooth * (target_gello_joints - current_pos)
                
                # Command Gello to move using follower
                self.gello_follower.command_follow(interpolated_pos)
                
                # Progress indicator
                if step % max(1, (num_steps // 20)) == 0:
                    progress = int(t * 100)
                    bar_length = 30
                    filled = int(bar_length * t)
                    bar = '█' * filled + '░' * (bar_length - filled)
                    print(f"  [{bar}] {progress}%", end='\r')
                
                time.sleep(duration / num_steps)
            
            print(f"\n  ✅ Gello reached target position")
            
        finally:
            # Stop follower mode (returns to free-wheeling)
            print("\n🛑 Stopping follower mode...")
            self.gello_follower.stop()
            time.sleep(0.3)
    
    def keyboard_listener(self):
        """Listen for keyboard input to switch modes."""
        print("\n" + "=" * 70)
        print("Keyboard Controls:")
        print("=" * 70)
        print("  [SPACE] - Switch between Follow and Teleoperation mode")
        print("  [Q]     - Quit test")
        print("=" * 70)
        
        while self.running:
            try:
                key = input()
                
                if key.lower() == 'q':
                    print("\n⏹️  Quit requested")
                    self.running = False
                    break
                
                elif key == ' ' or key.lower() == 's':
                    # Switch mode
                    if self.mode == "follow":
                        self.mode = "teleoperation"
                        print("\n🎮 Switched to TELEOPERATION mode")
                        print("   → Move Gello to control the robot")
                        # GelloFollower is already stopped at end of follow cycle
                    else:
                        self.mode = "follow"
                        print("\n🤖 Switched to FOLLOW mode")
                        print("   → Gello will follow robot position")
                        # GelloFollower will be started in next follow cycle
                
            except Exception as e:
                if self.running:
                    print(f"⚠️  Keyboard input error: {e}")
    
    def test_follow_mode(self):
        """
        Test 1: Gello follows robot to target position.
        Robot moves to a target, Gello follows slowly.
        """
        print("\n" + "=" * 70)
        print("Test 1: Gello Follows Robot")
        print("=" * 70)
        
        # Get current robot position
        current_robot_joints = self.get_robot_joint_state()
        print(f"\n📍 Current robot joint state (A1_X):")
        print(f"   {current_robot_joints}")
        
        # Convert to Gello space
        current_gello_target = self.a1x_to_gello_mapping(current_robot_joints)
        
        # Move Gello to match robot
        print("\n🔄 Moving Gello to match robot position...")
        self.slow_follow_to_target(current_gello_target, duration=3.0)
        
        print("\n✅ Test 1 complete: Gello now matches robot position")
    
    def test_teleoperation_mode(self):
        """
        Test 2: Robot follows Gello (teleoperation mode).
        User moves Gello, robot follows.
        """
        print("\n" + "=" * 70)
        print("Test 2: Robot Follows Gello (Teleoperation)")
        print("=" * 70)
        print("\n🎮 Teleoperation mode active")
        print("   → Move Gello manually to control the robot")
        print("   → Press [SPACE] to switch back to Follow mode")
        print("   → Press [Q] to quit")
        
        # Ensure follower is stopped for manual control
        if hasattr(self, 'gello_follower'):
            self.gello_follower.stop()
            time.sleep(0.3)
        
        # Teleoperation loop
        rate = 20  # Hz
        dt = 1.0 / rate
        
        while self.running and self.mode == "teleoperation":
            # Read Gello position
            gello_pos = self.get_gello_joint_state()
            
            # Convert to A1_X joint space
            a1x_target = self.gello_to_a1x_mapping(gello_pos)
            
            # Command robot (non-blocking for joint commands)
            self.robot.command_joint_state(a1x_target, from_gello=False)
            
            time.sleep(dt)
        
        print("\n⏸️  Teleoperation mode stopped")
    
    def run_interactive_test(self):
        """
        Run interactive test with mode switching.
        Allows user to switch between follow and teleoperation modes.
        """
        print("\n" + "=" * 70)
        print("Interactive Bidirectional Control Test")
        print("=" * 70)
        
        # Start keyboard listener in background
        self.keyboard_thread = threading.Thread(target=self.keyboard_listener, daemon=True)
        self.keyboard_thread.start()
        
        # Initial follow mode
        print("\n🤖 Starting in FOLLOW mode")
        self.test_follow_mode()
        
        # Wait a moment after initial sync
        time.sleep(1.0)
        
        print("\n💡 Press [SPACE] to switch to TELEOPERATION mode")
        
        # Main loop
        rate = 10  # Hz for mode checking
        dt = 1.0 / rate
        
        while self.running:
            if self.mode == "follow":
                # In follow mode, periodically update Gello to match robot
                try:
                    robot_joints = self.get_robot_joint_state()
                    gello_target = self.a1x_to_gello_mapping(robot_joints)
                    
                    # Smoothly move Gello (short duration for continuous following)
                    self.slow_follow_to_target(gello_target, duration=1.0)
                    
                    time.sleep(2.0)  # Update every 2 seconds
                    
                except Exception as e:
                    print(f"⚠️  Error in follow mode: {e}")
                    time.sleep(0.1)
            
            elif self.mode == "teleoperation":
                # Switch to teleoperation mode
                self.test_teleoperation_mode()
            
            time.sleep(dt)
        
        print("\n🏁 Test completed")
    
    def close(self):
        """Clean up resources."""
        print("\n🧹 Cleaning up...")
        
        # Stop Gello follower if active
        if hasattr(self, 'gello_follower'):
            try:
                self.gello_follower.stop()
                print("✅ Gello follower stopped")
            except:
                pass
        
        # Disable Gello torque
        if hasattr(self, 'gello'):
            try:
                self.gello._robot.set_torque_mode(False)
                print("✅ Gello torque disabled")
            except:
                pass
        
        # Close robot
        if hasattr(self, 'robot'):
            try:
                self.robot.close()
                print("✅ Robot connection closed")
            except:
                pass
        
        print("✅ Cleanup complete")


def main():
    """Main test function."""
    
    # Confirm before starting
    print("\n⚠️  This test will control both Gello and A1_X robot")
    print("   Make sure:")
    print("   1. Robot is in a safe starting position")
    print("   2. Gello is connected and powered on")
    print("   3. Path is clear for robot movement")
    print()
    
    response = input("Continue? (y/n): ")
    if response.lower() != 'y':
        print("❌ Test aborted")
        return
    
    # Create test instance
    test = GelloBidirectionalTest()
    
    try:
        # Run interactive test
        test.run_interactive_test()
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Test interrupted by user (Ctrl+C)")
    except Exception as e:
        print(f"\n❌ Error during test: {e}")
        import traceback
        traceback.print_exc()
    finally:
        test.close()
        print("\n👋 Goodbye!")


if __name__ == "__main__":
    main()

    
    def keyboard_listener(self):
        """Listen for keyboard input to switch modes."""
        print("\n" + "=" * 70)
        print("Keyboard Controls:")
        print("=" * 70)
        print("  [SPACE] - Switch between Follow and Teleoperation mode")
        print("  [Q]     - Quit test")
        print("=" * 70)
        
        while self.running:
            try:
                key = input()
                
                if key.lower() == 'q':
                    print("\n⏹️  Quit requested")
                    self.running = False
                    break
                
                elif key == ' ' or key.lower() == 's':
                    # Switch mode
                    if self.mode == "follow":
                        self.mode = "teleoperation"
                        print("\n🎮 Switched to TELEOPERATION mode")
                        print("   → Move Gello to control the robot")
                    else:
                        self.mode = "follow"
                        print("\n🤖 Switched to FOLLOW mode")
                        print("   → Gello will follow robot position")
                
            except Exception as e:
                if self.running:
                    print(f"⚠️  Keyboard input error: {e}")
    
    def test_follow_mode(self):
        """
        Test 1: Gello follows robot to target position.
        Robot moves to a target, Gello follows slowly.
        """
        print("\n" + "=" * 70)
        print("Test 1: Gello Follows Robot")
        print("=" * 70)
        
        # Get current robot position
        current_robot_joints = self.get_robot_joint_state()
        print(f"\n📍 Current robot joint state (A1_X):")
        print(f"   {current_robot_joints}")
        
        # Convert to Gello space
        current_gello_target = self.a1x_to_gello_mapping(current_robot_joints)
        
        # Move Gello to match robot
        print("\n🔄 Moving Gello to match robot position...")
        self.slow_follow_to_target(current_gello_target, duration=3.0)
        
        print("\n✅ Test 1 complete: Gello now matches robot position")
    
    def test_teleoperation_mode(self):
        """
        Test 2: Robot follows Gello (teleoperation mode).
        User moves Gello, robot follows.
        """
        print("\n" + "=" * 70)
        print("Test 2: Robot Follows Gello (Teleoperation)")
        print("=" * 70)
        print("\n🎮 Teleoperation mode active")
        print("   → Move Gello manually to control the robot")
        print("   → Press [SPACE] to switch back to Follow mode")
        print("   → Press [Q] to quit")
        
        # Teleoperation loop
        rate = 20  # Hz
        dt = 1.0 / rate
        
        while self.running and self.mode == "teleoperation":
            # Read Gello position
            gello_pos = self.gello.read_pos()
            
            # Convert to A1_X joint space
            a1x_target = self.gello_to_a1x_mapping(gello_pos)
            
            # Command robot (non-blocking for joint commands)
            self.robot.command_joint_state(a1x_target, from_gello=False)
            
            time.sleep(dt)
        
        print("\n⏸️  Teleoperation mode stopped")
    
    def run_interactive_test(self):
        """
        Run interactive test with mode switching.
        Allows user to switch between follow and teleoperation modes.
        """
        print("\n" + "=" * 70)
        print("Interactive Bidirectional Control Test")
        print("=" * 70)
        
        # Start keyboard listener in background
        self.keyboard_thread = threading.Thread(target=self.keyboard_listener, daemon=True)
        self.keyboard_thread.start()
        
        # Initial follow mode
        print("\n🤖 Starting in FOLLOW mode")
        self.test_follow_mode()
        
        # Main loop
        rate = 10  # Hz for mode checking
        dt = 1.0 / rate
        
        while self.running:
            if self.mode == "follow":
                # In follow mode, periodically update Gello to match robot
                try:
                    robot_joints = self.get_robot_joint_state()
                    gello_target = self.a1x_to_gello_mapping(robot_joints)
                    
                    # Smoothly move Gello (short duration for continuous following)
                    self.slow_follow_to_target(gello_target, duration=0.5)
                    
                    time.sleep(1.0)  # Update every 1 second
                    
                except Exception as e:
                    print(f"⚠️  Error in follow mode: {e}")
                    time.sleep(0.1)
            
            elif self.mode == "teleoperation":
                # Switch to teleoperation mode
                self.test_teleoperation_mode()
            
            time.sleep(dt)
        
        print("\n🏁 Test completed")
    
    def close(self):
        """Clean up resources."""
        print("\n🧹 Cleaning up...")
        
        # Stop Gello
        if hasattr(self, 'gello'):
            try:
                self.gello.stop_robot()
                print("✅ Gello stopped")
            except:
                pass
        
        # Close robot
        if hasattr(self, 'robot'):
            try:
                self.robot.close()
                print("✅ Robot connection closed")
            except:
                pass
        
        print("✅ Cleanup complete")


def main():
    """Main test function."""
    
    # Confirm before starting
    print("\n⚠️  This test will control both Gello and A1_X robot")
    print("   Make sure:")
    print("   1. Robot is in a safe starting position")
    print("   2. Gello is connected and powered on")
    print("   3. Path is clear for robot movement")
    print()
    
    response = input("Continue? (y/n): ")
    if response.lower() != 'y':
        print("❌ Test aborted")
        return
    
    # Create test instance
    test = GelloBidirectionalTest()
    
    try:
        # Run interactive test
        test.run_interactive_test()
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Test interrupted by user (Ctrl+C)")
    except Exception as e:
        print(f"\n❌ Error during test: {e}")
        import traceback
        traceback.print_exc()
    finally:
        test.close()
        print("\n👋 Goodbye!")


if __name__ == "__main__":
    main()
