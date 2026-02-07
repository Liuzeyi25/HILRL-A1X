#!/usr/bin/env python3
"""
Test single-step movement with correction vs multi-step approach.
Strategy: Send target position directly, then apply one correction if needed.
"""

import sys
import time
import numpy as np
import zmq

# Add parent directory to path
sys.path.insert(0, '.')

from a1x_robot import A1XRobot


def test_single_step_with_correction():
    """Test moving to target in one step + optional correction."""
    
    print("=" * 70)
    print("A1_X Single-Step + Correction Test")
    print("=" * 70)
    print("\nStrategy: Send target position directly, then correct if needed")
    print("Compare this with multi-step approach for smoothness and accuracy")
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
            node_name="a1x_single_step_test_node",
            port=6100,
            python_path="/usr/bin/python3"
        )
    except Exception as e:
        print(f"❌ Failed to initialize robot: {e}")
        return
    
    print("✅ Robot initialized successfully")
    print("\n⏳ Ready to start test")
    
    try:
        # Test parameters
        target_distances = [0.02, 0.04]  # Test 2cm and 4cm movements
        correction_threshold = 0.005  # 5mm - if error > this, apply correction
        max_corrections = 2  # Maximum number of corrections per target
        gripper = 100.0
        
        # Create ZMQ connection
        context = zmq.Context()
        socket = context.socket(zmq.REQ)
        socket.setsockopt(zmq.RCVTIMEO, 20000)
        socket.setsockopt(zmq.SNDTIMEO, 20000)
        socket.connect("tcp://localhost:6100")
        
        # Get baseline position
        print("\n📍 Recording baseline position...")
        zero_cmd = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, gripper]
        socket.send_json({
            "cmd": "command_eef_pose",
            "pose": zero_cmd
        })
        
        baseline_resp = socket.recv_json()
        
        if baseline_resp.get("status") != "ok" or not baseline_resp.get("info"):
            print("   ❌ Failed to get baseline position")
            socket.close()
            context.term()
            robot.close()
            return
        
        start_pos = np.array(baseline_resp["info"]["current_pos"])
        print(f"   ✅ Baseline: [{start_pos[0]:.6f}, {start_pos[1]:.6f}, {start_pos[2]:.6f}]")
        
        # Storage for all test results
        all_results = []
        
        # Test each target distance
        for test_idx, target_distance in enumerate(target_distances):
            print(f"\n{'='*70}")
            print(f"Test {test_idx+1}/{len(target_distances)}: Move {target_distance*100:.0f}cm in X direction")
            print(f"{'='*70}")
            
            # Get current position before this test
            socket.send_json({
                "cmd": "command_eef_pose",
                "pose": zero_cmd
            })
            curr_resp = socket.recv_json()
            if curr_resp.get("status") != "ok":
                print("   ❌ Failed to get current position")
                continue
            
            test_start_pos = np.array(curr_resp["info"]["current_pos"])
            target_pos = test_start_pos + np.array([target_distance, 0.0, 0.0])
            
            print(f"\n   Start:  [{test_start_pos[0]:.6f}, {test_start_pos[1]:.6f}, {test_start_pos[2]:.6f}]")
            print(f"   Target: [{target_pos[0]:.6f}, {target_pos[1]:.6f}, {target_pos[2]:.6f}]")
            
            test_result = {
                "target_distance": target_distance,
                "start_pos": test_start_pos.copy(),
                "target_pos": target_pos.copy(),
                "movements": []
            }
            
            current_pos = test_start_pos.copy()
            
            # Initial movement to target
            print(f"\n   📤 Initial Move: Sending delta [{target_distance*100:.1f}cm, 0, 0]")
            delta = target_pos - current_pos
            eef_cmd = [delta[0], delta[1], delta[2], 0.0, 0.0, 0.0, gripper]
            
            socket.send_json({
                "cmd": "command_eef_pose",
                "pose": eef_cmd
            })
            
            start_time = time.time()
            try:
                response = socket.recv_json()
            except zmq.Again:
                print("   ❌ Timeout on initial move")
                continue
            move_time = time.time() - start_time
            
            if response.get("status") != "ok":
                print(f"   ❌ Initial move failed: {response}")
                continue
            
            actual_pos = np.array(response["info"]["current_pos"])
            error = np.linalg.norm(actual_pos - target_pos)
            
            movement_result = {
                "type": "initial",
                "commanded_delta": delta.copy(),
                "actual_pos": actual_pos.copy(),
                "error": error,
                "time": move_time
            }
            test_result["movements"].append(movement_result)
            
            print(f"   ✅ Completed in {move_time:.2f}s")
            print(f"   Actual: [{actual_pos[0]:.6f}, {actual_pos[1]:.6f}, {actual_pos[2]:.6f}]")
            print(f"   Error:  {error*1000:.3f}mm")
            
            current_pos = actual_pos.copy()
            
            # Apply corrections if needed
            for correction_idx in range(max_corrections):
                if error <= correction_threshold:
                    print(f"   ✓ Error within threshold ({correction_threshold*1000:.1f}mm), no correction needed")
                    break
                
                print(f"\n   🔧 Correction {correction_idx+1}: Error {error*1000:.3f}mm > threshold {correction_threshold*1000:.1f}mm")
                correction_delta = target_pos - current_pos
                print(f"   Sending correction delta: [{correction_delta[0]*100:.2f}cm, {correction_delta[1]*100:.2f}cm, {correction_delta[2]*100:.2f}cm]")
                
                eef_cmd = [correction_delta[0], correction_delta[1], correction_delta[2], 0.0, 0.0, 0.0, gripper]
                socket.send_json({
                    "cmd": "command_eef_pose",
                    "pose": eef_cmd
                })
                
                start_time = time.time()
                try:
                    response = socket.recv_json()
                except zmq.Again:
                    print("   ❌ Timeout on correction")
                    break
                move_time = time.time() - start_time
                
                if response.get("status") != "ok":
                    print(f"   ❌ Correction failed: {response}")
                    break
                
                actual_pos = np.array(response["info"]["current_pos"])
                error = np.linalg.norm(actual_pos - target_pos)
                
                movement_result = {
                    "type": f"correction_{correction_idx+1}",
                    "commanded_delta": correction_delta.copy(),
                    "actual_pos": actual_pos.copy(),
                    "error": error,
                    "time": move_time
                }
                test_result["movements"].append(movement_result)
                
                print(f"   ✅ Completed in {move_time:.2f}s")
                print(f"   Actual: [{actual_pos[0]:.6f}, {actual_pos[1]:.6f}, {actual_pos[2]:.6f}]")
                print(f"   Error:  {error*1000:.3f}mm")
                
                current_pos = actual_pos.copy()
            
            # Final summary for this test
            final_error = test_result["movements"][-1]["error"]
            total_time = sum(m["time"] for m in test_result["movements"])
            num_movements = len(test_result["movements"])
            
            print(f"\n   📊 Test {test_idx+1} Summary:")
            print(f"      Target distance: {target_distance*100:.1f}cm")
            print(f"      Number of movements: {num_movements}")
            print(f"      Total time: {total_time:.2f}s")
            print(f"      Final error: {final_error*1000:.3f}mm ({final_error/target_distance*100:.1f}% of target)")
            
            all_results.append(test_result)
        
        socket.close()
        context.term()
        
        # Print comprehensive summary
        print("\n" + "="*70)
        print("Overall Summary")
        print("="*70)
        
        for idx, result in enumerate(all_results):
            target_dist = result["target_distance"]
            num_moves = len(result["movements"])
            final_err = result["movements"][-1]["error"]
            total_time = sum(m["time"] for m in result["movements"])
            
            print(f"\nTest {idx+1}: {target_dist*100:.0f}cm movement")
            print(f"  Movements: {num_moves} (1 initial + {num_moves-1} correction(s))")
            print(f"  Total time: {total_time:.2f}s")
            print(f"  Final error: {final_err*1000:.3f}mm ({final_err/target_dist*100:.1f}%)")
            
            for move_idx, move in enumerate(result["movements"]):
                print(f"    {move['type']:15s}: error={move['error']*1000:5.2f}mm, time={move['time']:.2f}s")
        
        print("\n✅ Test complete!")
        
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
    test_single_step_with_correction()
