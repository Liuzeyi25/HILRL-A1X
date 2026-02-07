#!/usr/bin/env python3
"""
Simplified Gello bidirectional control test:
1. Start in teleoperation mode - manually align Gello to robot position
2. Switch to follow mode (optional) - Gello follows robot small movements  
3. Switch back to teleoperation - robot follows Gello
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


class SimplifiedGelloBidirectionalTest:
    """Simplified bidirectional control test - starts with teleoperation."""
    
    def __init__(
        self,
        robot_port: int = 6100,
        gello_port: str = "/dev/serial/by-id/usb-FTDI_USB__-__Serial_Converter_FTA7NNNU-if00-port0",
    ):
        print("=" * 70)
        print("Simplified Gello Bidirectional Control Test")
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
        time.sleep(2.0)
        
        # Initialize Gello with explicit configuration
        print("\n🎮 Initializing Gello...")
        gello_config = DynamixelRobotConfig(
            joint_ids=[1, 2, 3, 4, 5, 6],
            joint_offsets=[1.5708, 4.71239, 4.71239, 3.14159, 1.5708, 3.14159],
            joint_signs=[1.0, -1.0, -1.0, -1.0, 1.0, 1.0],
            gripper_config=[7, 139.66015625, 199.16015625]
        )
        
        self.gello = GelloAgent(port=gello_port, dynamixel_config=gello_config)
        print("✅ Gello initialized")
        
        # Verify
        test_pos = self.gello.act({})
        print(f"   Gello has {test_pos.shape[0]} joints")
        
        # Initialize GelloFollower for follow mode
        self.gello_follower = GelloFollower(self.gello._robot)
        print("   GelloFollower initialized")
        
        # Test state
        self.mode = "teleoperation"  # Start in teleoperation!
        self.running = True
        self.keyboard_thread = None
        
    def get_robot_joint_state(self) -> np.ndarray:
        """Get current robot joint state (A1_X format)."""
        return self.robot.get_joint_state()
    
    def get_gello_joint_state(self) -> np.ndarray:
        """Get current Gello joint state."""
        return self.gello.act({})
    
    def a1x_to_gello_mapping(self, a1x_joints: np.ndarray) -> np.ndarray:
        """Convert A1_X joint positions to Gello joint positions."""
        if hasattr(self.robot, '_map_from_a1x'):
            return self.robot._map_from_a1x(a1x_joints)
        return a1x_joints
    
    def gello_to_a1x_mapping(self, gello_joints: np.ndarray) -> np.ndarray:
        """Convert Gello joint positions to A1_X joint positions."""
        if hasattr(self.robot, '_map_to_a1x'):
            return self.robot._map_to_a1x(gello_joints)
        return gello_joints
    
    def show_position_status(self):
        """Show current positions and difference."""
        robot_pos = self.get_robot_joint_state()
        gello_pos = self.get_gello_joint_state()
        
        # Convert robot to Gello space for comparison
        robot_in_gello_space = self.a1x_to_gello_mapping(robot_pos)
        
        # Calculate difference (ignore gripper - last joint)
        diff = np.abs(robot_in_gello_space[:6] - gello_pos[:6])
        max_diff_rad = np.max(diff)
        max_diff_deg = np.rad2deg(max_diff_rad)
        
        print(f"\n📊 Position Status:")
        print(f"   Robot (A1X):  {robot_pos[:3]}...")
        print(f"   Gello:        {gello_pos[:3]}...")
        print(f"   Max diff:     {max_diff_deg:.1f}° ({max_diff_rad:.3f} rad)")
        
        if max_diff_rad < 0.1:  # < 5.7 degrees
            print(f"   ✅ Positions are well aligned!")
        elif max_diff_rad < 0.5:  # < 28.6 degrees
            print(f"   ⚠️  Positions have some difference")
        else:
            print(f"   ❌ Positions are far apart - align in teleoperation mode first")
        
        return max_diff_rad
    
    def keyboard_listener(self):
        """Listen for keyboard input."""
        print("\n" + "=" * 70)
        print("Keyboard Controls:")
        print("=" * 70)
        print("  [SPACE] - Switch between Teleoperation and Follow mode")
        print("  [S]     - Show position status")
        print("  [Q]     - Quit test")
        print("=" * 70)
        
        while self.running:
            try:
                key = input()
                
                if key.lower() == 'q':
                    print("\n⏹️  Quit requested")
                    self.running = False
                    break
                
                elif key == ' ':
                    # Switch mode
                    if self.mode == "teleoperation":
                        self.mode = "follow"
                        print("\n🤖 Switched to FOLLOW mode")
                        print("   → Gello will try to follow robot (small movements only)")
                    else:
                        self.mode = "teleoperation"
                        print("\n🎮 Switched to TELEOPERATION mode")
                        print("   → Move Gello to control robot")
                
                elif key.lower() == 's':
                    self.show_position_status()
                
            except Exception as e:
                if self.running:
                    print(f"⚠️  Keyboard input error: {e}")
    
    def run_teleoperation_mode(self):
        """Robot follows Gello movements."""
        if not self.running or self.mode != "teleoperation":
            return
        
        print("\n" + "=" * 70)
        print("🎮 Teleoperation Mode Active")
        print("=" * 70)
        print("\n   → Move Gello manually - robot will follow")
        print("   → Use this mode to align Gello with robot position")
        print("   → Press [SPACE] to switch to Follow mode")
        print("   → Press [S] to check position alignment")
        print("   → Press [Q] to quit\n")
        
        rate = 20  # Hz
        dt = 1.0 / rate
        
        while self.running and self.mode == "teleoperation":
            try:
                # Read Gello position
                gello_pos = self.get_gello_joint_state()
                
                
                # Convert to A1_X joint space
                a1x_target = self.gello_to_a1x_mapping(gello_pos)
                # print(f"   Gello Pos: {a1x_target[:3]}...", end='\r')
                
                # Command robot (non-blocking)
                self.robot.command_joint_state(a1x_target, from_gello=False)
                
                time.sleep(dt)
            except Exception as e:
                print(f"⚠️  Teleoperation error: {e}")
                time.sleep(0.5)
        
        print("\n⏸️  Teleoperation mode stopped")
    
    def run_follow_mode(self):
        """Gello follows robot movements (small changes only)."""
        if not self.running or self.mode != "follow":
            return
        
        print("\n" + "=" * 70)
        print("🤖 Follow Mode Active")
        print("=" * 70)
        
        # Check if positions are aligned
        max_diff = self.show_position_status()
        
        if max_diff > 0.5:
            print("\n❌ Positions too far apart for follow mode!")
            print("   Switch back to teleoperation mode and manually align first.")
            self.mode = "teleoperation"
            return
        
        print("\n   → Gello will actively follow robot movements")
        print("   → Keep movements small and smooth!")
        print("   → Press [SPACE] to switch back to Teleoperation")
        print("   → Press [Q] to quit\n")
        
        # Start GelloFollower
        print("🔄 Starting GelloFollower...")
        try:
            self.gello_follower.start()
            print("✅ Gello is now in position control mode")
        except Exception as e:
            print(f"❌ Failed to start follower: {e}")
            self.mode = "teleoperation"
            return
        
        rate = 20  # Hz - active following
        dt = 1.0 / rate
        
        # Track previous target to detect large jumps
        prev_gello_target = None
        consecutive_errors = 0
        max_consecutive_errors = 10
        
        try:
            while self.running and self.mode == "follow":
                try:
                    # Get robot position
                    robot_pos = self.get_robot_joint_state()
                    
                    # Convert to Gello space
                    gello_target = self.a1x_to_gello_mapping(robot_pos)
                    
                    # Check for large jumps (safety check)
                    if prev_gello_target is not None:
                        jump = np.max(np.abs(gello_target[:6] - prev_gello_target[:6]))
                        if jump > 0.3:  # ~17 degrees in one step
                            print(f"\n⚠️  Large jump detected: {np.rad2deg(jump):.1f}°")
                            print(f"   Skipping this command to protect servos")
                            time.sleep(dt)
                            continue
                    
                    # Command Gello to follow
                    self.gello_follower.command_follow(gello_target)
                    
                    # Update previous target
                    prev_gello_target = gello_target.copy()
                    
                    # Reset error counter on success
                    consecutive_errors = 0
                    
                    # Show tracking status
                    gello_pos = self.get_gello_joint_state()
                    diff = np.abs(gello_target[:6] - gello_pos[:6])
                    max_diff = np.max(diff)
                    print(f"   Following - Max diff: {np.rad2deg(max_diff):.1f}°", end='\r')
                    
                    time.sleep(dt)
                    
                except Exception as e:
                    consecutive_errors += 1
                    if consecutive_errors <= 3:  # Only show first few errors
                        print(f"\n⚠️  Follow error: {e}")
                    elif consecutive_errors == 4:
                        print(f"\n⚠️  (suppressing further errors...)")
                    
                    if consecutive_errors >= max_consecutive_errors:
                        print(f"\n❌ Too many consecutive errors ({consecutive_errors})")
                        print(f"   Stopping follow mode for safety")
                        break
                    
                    time.sleep(0.5)
        finally:
            # Stop follower mode
            print("\n🛑 Stopping follower...")
            self.gello_follower.stop()
            print("✅ Returned to free-wheeling mode")
        
        print("\n⏸️  Follow mode stopped")
    
    def run_interactive_test(self):
        """Run the interactive test."""
        print("\n" + "=" * 70)
        print("Interactive Test Starting")
        print("=" * 70)
        
        # Show initial status
        self.show_position_status()
        
        # Start keyboard listener
        self.keyboard_thread = threading.Thread(target=self.keyboard_listener, daemon=True)
        self.keyboard_thread.start()
        
        print("\n💡 Starting in TELEOPERATION mode")
        print("   Use this to manually align Gello to robot position")
        
        # Main loop
        while self.running:
            if self.mode == "teleoperation":
                self.run_teleoperation_mode()
            elif self.mode == "follow":
                self.run_follow_mode()
            
            time.sleep(0.1)
        
        print("\n🏁 Test completed")
    
    def close(self):
        """Clean up resources."""
        print("\n🧹 Cleaning up...")
        
        # Stop follower if active
        if hasattr(self, 'gello_follower'):
            try:
                self.gello_follower.stop()
                print("✅ Gello follower stopped")
            except:
                pass
        
        if hasattr(self, 'gello'):
            try:
                self.gello._robot.set_torque_mode(False)
                print("✅ Gello torque disabled")
            except:
                pass
        
        if hasattr(self, 'robot'):
            try:
                self.robot.close()
                print("✅ Robot connection closed")
            except:
                pass
        
        print("✅ Cleanup complete")


def main():
    """Main test function."""
    
    print("\n⚠️  This test will control both Gello and A1_X robot")
    print("   Make sure:")
    print("   1. Robot is in a safe starting position")
    print("   2. Gello is powered on (check servo LEDs)")
    print("   3. Path is clear for robot movement")
    print()
    
    response = input("Continue? (y/n): ")
    if response.lower() != 'y':
        print("❌ Test aborted")
        return
    
    test = SimplifiedGelloBidirectionalTest()
    
    try:
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
