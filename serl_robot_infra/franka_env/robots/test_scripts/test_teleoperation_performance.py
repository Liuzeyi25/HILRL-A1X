#!/usr/bin/env python3
"""
Test teleoperation mode responsiveness.
Measure command rate, latency, and robot following performance.
"""

import sys
import time
import numpy as np
import threading

sys.path.insert(0, '.')
sys.path.insert(0, 'Gello/gello_software')

from a1x_robot import A1XRobot
from gello.agents.gello_agent import GelloAgent, DynamixelRobotConfig


class TeleoperationPerformanceTest:
    """Test teleoperation mode performance."""
    
    def __init__(
        self,
        robot_port: int = 6100,
        gello_port: str = "/dev/serial/by-id/usb-FTDI_USB__-__Serial_Converter_FTA7NNNU-if00-port0",
    ):
        print("=" * 70)
        print("Teleoperation Mode Performance Test")
        print("=" * 70)
        
        # Initialize robot
        print("\n🤖 Initializing A1_X robot...")
        self.robot = A1XRobot(
            num_dofs=7,
            node_name="a1x_teleoperation_test",
            port=robot_port,
            python_path="/usr/bin/python3"
        )
        print("✅ Robot initialized")
        time.sleep(1.0)
        
        # Initialize Gello
        print("\n🎮 Initializing Gello...")
        gello_config = DynamixelRobotConfig(
            joint_ids=[1, 2, 3, 4, 5, 6],
            joint_offsets=[1.5708, 4.71239, 4.71239, 3.14159, 1.5708, 3.14159],
            joint_signs=[1.0, -1.0, -1.0, -1.0, 1.0, 1.0],
            gripper_config=[7, 139.66015625, 199.16015625]
        )
        
        self.gello = GelloAgent(port=gello_port, dynamixel_config=gello_config)
        print("✅ Gello initialized")
        
        # Performance tracking
        self.running = False
        self.stats = {
            'read_times': [],
            'command_times': [],
            'total_times': [],
            'position_diffs': [],
            'iterations': 0
        }
        
    def gello_to_a1x_mapping(self, gello_joints: np.ndarray) -> np.ndarray:
        """Convert Gello joint positions to A1_X joint positions."""
        if hasattr(self.robot, '_map_to_a1x'):
            return self.robot._map_to_a1x(gello_joints)
        return gello_joints
    
    def run_teleoperation_test(self, duration: float = 10.0, target_rate: int = 50):
        """
        Run teleoperation mode and measure performance.
        
        Args:
            duration: Test duration in seconds
            target_rate: Target control rate in Hz
        """
        print("\n" + "=" * 70)
        print(f"🎮 Testing Teleoperation Mode")
        print("=" * 70)
        print(f"\n   Duration: {duration}s")
        print(f"   Target rate: {target_rate} Hz")
        print(f"   Target period: {1000.0/target_rate:.2f} ms")
        print("\n   → Move Gello to test robot following")
        print("   → Test will run automatically and show statistics\n")
        
        dt = 1.0 / target_rate
        start_time = time.time()
        self.running = True
        self.stats = {
            'read_times': [],
            'command_times': [],
            'total_times': [],
            'position_diffs': [],
            'iterations': 0
        }
        
        prev_gello_pos = None
        
        try:
            while self.running and (time.time() - start_time) < duration:
                loop_start = time.time()
                
                # Read Gello position
                read_start = time.time()
                gello_pos = self.gello.act({})
                read_time = time.time() - read_start
                
                # Convert to A1_X joint space
                a1x_target = self.gello_to_a1x_mapping(gello_pos)
                
                # Command robot
                command_start = time.time()
                self.robot.command_joint_state(a1x_target, from_gello=False)
                command_time = time.time() - command_start
                
                # Calculate position change
                if prev_gello_pos is not None:
                    diff = np.max(np.abs(gello_pos[:6] - prev_gello_pos[:6]))
                    self.stats['position_diffs'].append(diff)
                prev_gello_pos = gello_pos.copy()
                
                # Track timing
                total_time = time.time() - loop_start
                self.stats['read_times'].append(read_time * 1000)  # ms
                self.stats['command_times'].append(command_time * 1000)  # ms
                self.stats['total_times'].append(total_time * 1000)  # ms
                self.stats['iterations'] += 1
                
                # Sleep to maintain target rate
                elapsed = time.time() - loop_start
                sleep_time = max(0, dt - elapsed)
                if sleep_time > 0:
                    time.sleep(sleep_time)
                
        except KeyboardInterrupt:
            print("\n⚠️  Test interrupted by user")
        finally:
            self.running = False
        
        # Print statistics
        self.print_statistics()
    
    def print_statistics(self):
        """Print performance statistics."""
        if self.stats['iterations'] == 0:
            print("\n⚠️  No data collected")
            return
        
        print("\n" + "=" * 70)
        print("📊 Performance Statistics")
        print("=" * 70)
        
        # Basic stats
        print(f"\n🔢 Basic Metrics:")
        print(f"   Total iterations: {self.stats['iterations']}")
        
        # Timing statistics
        read_times = np.array(self.stats['read_times'])
        command_times = np.array(self.stats['command_times'])
        total_times = np.array(self.stats['total_times'])
        
        print(f"\n⏱️  Timing Breakdown (ms):")
        print(f"   Gello read:      {np.mean(read_times):6.2f} ± {np.std(read_times):5.2f}  (min: {np.min(read_times):5.2f}, max: {np.max(read_times):6.2f})")
        print(f"   Robot command:   {np.mean(command_times):6.2f} ± {np.std(command_times):5.2f}  (min: {np.min(command_times):5.2f}, max: {np.max(command_times):6.2f})")
        print(f"   Total loop:      {np.mean(total_times):6.2f} ± {np.std(total_times):5.2f}  (min: {np.min(total_times):5.2f}, max: {np.max(total_times):6.2f})")
        
        # Achieved rate
        actual_rate = 1000.0 / np.mean(total_times)
        print(f"\n📈 Actual Performance:")
        print(f"   Achieved rate:   {actual_rate:.1f} Hz")
        print(f"   Average period:  {np.mean(total_times):.2f} ms")
        
        # Position change statistics
        if len(self.stats['position_diffs']) > 0:
            pos_diffs = np.array(self.stats['position_diffs'])
            print(f"\n🎯 Position Tracking:")
            print(f"   Avg movement:    {np.rad2deg(np.mean(pos_diffs)):.3f}° per step")
            print(f"   Max movement:    {np.rad2deg(np.max(pos_diffs)):.3f}° per step")
            
            # Movement detection
            moving_threshold = 0.001  # ~0.06 degrees
            moving_count = np.sum(pos_diffs > moving_threshold)
            moving_pct = 100.0 * moving_count / len(pos_diffs)
            print(f"   Moving frames:   {moving_count}/{len(pos_diffs)} ({moving_pct:.1f}%)")
        
        # Bottleneck analysis
        print(f"\n🔍 Bottleneck Analysis:")
        read_pct = 100.0 * np.mean(read_times) / np.mean(total_times)
        command_pct = 100.0 * np.mean(command_times) / np.mean(total_times)
        other_pct = 100.0 - read_pct - command_pct
        
        print(f"   Gello read:      {read_pct:5.1f}%")
        print(f"   Robot command:   {command_pct:5.1f}%")
        print(f"   Other overhead:  {other_pct:5.1f}%")
        
        # Recommendations
        print(f"\n💡 Recommendations:")
        if np.mean(read_times) > 10:
            print(f"   ⚠️  Gello read is slow ({np.mean(read_times):.1f}ms)")
            print(f"      → Consider using faster baudrate or reduce smoothing")
        if np.mean(command_times) > 10:
            print(f"   ⚠️  Robot command is slow ({np.mean(command_times):.1f}ms)")
            print(f"      → Check ZMQ connection latency")
            print(f"      → Ensure robot control loop is responsive")
        if actual_rate < 30:
            print(f"   ⚠️  Achieved rate is low ({actual_rate:.1f} Hz)")
            print(f"      → Target rate may be too high for current setup")
        elif actual_rate >= 40:
            print(f"   ✅ Good performance! ({actual_rate:.1f} Hz)")
        
    def close(self):
        """Clean up resources."""
        print("\n🧹 Cleaning up...")
        self.running = False
        
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
    
    print("\n⚠️  This test will measure teleoperation performance")
    print("   Make sure:")
    print("   1. Robot is in a safe starting position")
    print("   2. Gello is powered on")
    print("   3. Path is clear for robot movement")
    print()
    
    response = input("Continue? (y/n): ")
    if response.lower() != 'y':
        print("❌ Test aborted")
        return
    
    # Test parameters
    print("\n📋 Test Configuration:")
    print("   1. Quick test:  10s @ 50Hz (default)")
    print("   2. Long test:   30s @ 50Hz")
    print("   3. High rate:   10s @ 100Hz")
    print("   4. Custom")
    
    choice = input("\nSelect test (1-4, default=1): ").strip() or "1"
    
    if choice == "1":
        duration, rate = 10.0, 50
    elif choice == "2":
        duration, rate = 30.0, 50
    elif choice == "3":
        duration, rate = 10.0, 100
    elif choice == "4":
        duration = float(input("Duration (seconds): "))
        rate = int(input("Target rate (Hz): "))
    else:
        duration, rate = 10.0, 50
    
    test = TeleoperationPerformanceTest()
    
    try:
        test.run_teleoperation_test(duration=duration, target_rate=rate)
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Test interrupted by user (Ctrl+C)")
    except Exception as e:
        print(f"\n❌ Error during test: {e}")
        import traceback
        traceback.print_exc()
    finally:
        test.close()
        print("\n👋 Test complete!")


if __name__ == "__main__":
    main()
