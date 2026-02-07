#!/usr/bin/env python3
"""
Test cumulative error in sequential EEF movements.
This script sends 8 consecutive 1cm X-direction movements and tracks cumulative error.
"""

import sys
import time
import numpy as np
import zmq

# Add parent directory to path
sys.path.insert(0, '.')

from a1x_robot import A1XRobot


def test_cumulative_error():
    """Test cumulative error over 8 consecutive 1cm X movements."""
    
    print("=" * 70)
    print("A1_X Cumulative Error Test")
    print("=" * 70)
    print("\nThis script will move the robot 8 times, each time +1cm in X direction")
    print("Total movement: 8cm in X direction")
    print("We will track cumulative error at each step.")
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
            node_name="a1x_cumulative_test_node",
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
        increment = 0.005  # 0.5cm per step
        num_steps = 8
        chunk_size = 8  # Use single chunk to let error converge more
        gripper = 100.0  # Keep gripper open
        
        print("\n" + "="*70)
        print(f"Starting Sequential Movement Test: {num_steps} steps × {increment*100:.0f}cm")
        print(f"Strategy: Split into {num_steps//chunk_size} chunks of {chunk_size} steps each")
        print("="*70)
        
        # Create ZMQ connection
        context = zmq.Context()
        socket = context.socket(zmq.REQ)
        socket.setsockopt(zmq.RCVTIMEO, 20000)  # 20 second timeout (increased for slower movements)
        socket.setsockopt(zmq.SNDTIMEO, 20000)
        socket.connect("tcp://localhost:6100")
        
        # Step 0: Get baseline position by sending zero-delta command
        print("\n📍 Step 0: Recording baseline position...")
        zero_cmd = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, gripper]
        socket.send_json({
            "cmd": "command_eef_pose",
            "pose": zero_cmd
        })
        
        baseline_resp = socket.recv_json()
        
        if baseline_resp.get("status") == "ok" and baseline_resp.get("info"):
            start_pos = np.array(baseline_resp["info"]["current_pos"])
            print(f"   ✅ Baseline position: [{start_pos[0]:.6f}, {start_pos[1]:.6f}, {start_pos[2]:.6f}]")
        else:
            print(f"   ❌ Failed to get baseline position: {baseline_resp}")
            socket.close()
            context.term()
            robot.close()
            return
        
        # Storage for results
        results = []
        all_ok = True
        
        # Execute movements in chunks
        num_chunks = num_steps // chunk_size
        print(f"\n🚀 Executing {num_chunks} chunks of {chunk_size} movements each...")
        print("-" * 70)
        
        for chunk_idx in range(num_chunks):
            # Get chunk start position
            if chunk_idx == 0:
                chunk_start_pos = start_pos.copy()
            else:
                # Update chunk start to current position after previous chunk
                zero_cmd = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, gripper]
                socket.send_json({
                    "cmd": "command_eef_pose",
                    "pose": zero_cmd
                })
                chunk_resp = socket.recv_json()
                if chunk_resp.get("status") == "ok" and chunk_resp.get("info"):
                    chunk_start_pos = np.array(chunk_resp["info"]["current_pos"])
                else:
                    print(f"   ❌ Failed to get chunk {chunk_idx+1} start position")
                    all_ok = False
                    break
            
            print(f"\n{'='*70}")
            print(f"Chunk {chunk_idx+1}/{num_chunks}: Steps {chunk_idx*chunk_size+1}-{(chunk_idx+1)*chunk_size}")
            print(f"Chunk start position: [{chunk_start_pos[0]:.6f}, {chunk_start_pos[1]:.6f}, {chunk_start_pos[2]:.6f}]")
            print(f"{'='*70}")
            
            # Execute chunk_size steps, each relative to chunk start position
            current_pos = chunk_start_pos.copy()  # Track position without querying each time
            
            for step_in_chunk in range(chunk_size):
                step_num = chunk_idx * chunk_size + step_in_chunk + 1
                
                # Calculate target position relative to chunk start
                # For step 1 in chunk: chunk_start + 0.5cm
                # For step 2 in chunk: chunk_start + 1.0cm
                # etc.
                delta_from_chunk_start = increment * (step_in_chunk + 1)
                target_pos_in_chunk = chunk_start_pos + np.array([delta_from_chunk_start, 0.0, 0.0])
                
                # Calculate delta needed from CURRENT (tracked) position to reach TARGET
                delta_from_current = target_pos_in_chunk - current_pos
                eef_cmd = [delta_from_current[0], delta_from_current[1], delta_from_current[2], 0.0, 0.0, 0.0, gripper]
                
                # Expected position relative to overall start
                expected_pos = start_pos + np.array([increment * step_num, 0.0, 0.0])
                
                print(f"\n📤 Step {step_num}/{num_steps} (Chunk {chunk_idx+1}, Step {step_in_chunk+1})")
                print(f"   Current (tracked): [{current_pos[0]:.6f}, {current_pos[1]:.6f}, {current_pos[2]:.6f}]")
                print(f"   Target (chunk_start + {delta_from_chunk_start*100:.1f}cm): [{target_pos_in_chunk[0]:.6f}, {target_pos_in_chunk[1]:.6f}, {target_pos_in_chunk[2]:.6f}]")
                print(f"   Delta from current: [{delta_from_current[0]*100:.2f}cm, {delta_from_current[1]*100:.2f}cm, {delta_from_current[2]*100:.2f}cm]")
                
                # Send command
                socket.send_json({
                    "cmd": "command_eef_pose",
                    "pose": eef_cmd
                })
                
                # Wait for response
                try:
                    response = socket.recv_json()
                except zmq.Again:
                    print(f"   ❌ Timeout waiting for response at step {step_num}")
                    all_ok = False
                    break
                except Exception as e:
                    print(f"   ❌ Error at step {step_num}: {e}")
                    all_ok = False
                    break
                
                # Process response
                if response.get("status") == "ok":
                    if response.get("info"):
                        actual_pos = np.array(response["info"]["current_pos"])
                        pos_err = response["info"]["pos_err"]
                        
                        # Update tracked position to actual position for next step
                        current_pos = actual_pos.copy()
                        
                        # Calculate cumulative error (deviation from expected absolute position)
                        cumulative_err = np.linalg.norm(actual_pos - expected_pos)
                        
                        # Calculate step error (deviation from previous position + increment)
                        if step_num == 1:
                            step_err = np.linalg.norm(actual_pos - (start_pos + np.array([increment, 0.0, 0.0])))
                        else:
                            prev_pos = results[-1]["actual_pos"]
                            step_err = np.linalg.norm(actual_pos - (prev_pos + np.array([increment, 0.0, 0.0])))
                        
                        # Store result
                        result = {
                            "step": step_num,
                            "chunk": chunk_idx + 1,
                            "expected_pos": expected_pos.copy(),
                            "actual_pos": actual_pos.copy(),
                            "cumulative_err": cumulative_err,
                            "step_err": step_err,
                            "pos_err": pos_err
                        }
                        results.append(result)
                        
                        print(f"   ✅ Completed")
                        print(f"   Actual position:  [{actual_pos[0]:.6f}, {actual_pos[1]:.6f}, {actual_pos[2]:.6f}]")
                        print(f"   Step error:       {step_err*1000:.3f}mm")
                        print(f"   Cumulative error: {cumulative_err*1000:.3f}mm")
                        print(f"   Position error (from target): {pos_err*1000:.3f}mm")
                        
                    else:
                        print(f"   ⚠️  Completed but no info returned")
                        all_ok = False
                        break
                        
                elif response.get("status") == "timeout":
                    print(f"   ⏱️  Motion timeout at step {step_num}")
                    if response.get("info"):
                        print(f"   Info: {response['info']}")
                    all_ok = False
                    break
                else:
                    print(f"   ❌ Failed: {response}")
                    all_ok = False
                    break
            
            if not all_ok:
                break
        
        socket.close()
        context.term()
        
        # Print summary
        print("\n" + "="*70)
        print("Test Summary")
        print("="*70)
        
        if all_ok and len(results) == num_steps:
            print(f"✅ All {num_steps} steps completed successfully!")
            print()
            
            # Statistics
            cumulative_errors = [r["cumulative_err"] for r in results]
            step_errors = [r["step_err"] for r in results]
            
            print("📊 Statistics:")
            print(f"   Total distance commanded: {increment * num_steps * 100:.1f}cm")
            print(f"   Final cumulative error:   {cumulative_errors[-1]*1000:.3f}mm")
            print(f"   Mean cumulative error:    {np.mean(cumulative_errors)*1000:.3f}mm")
            print(f"   Max cumulative error:     {np.max(cumulative_errors)*1000:.3f}mm")
            print(f"   Mean step error:          {np.mean(step_errors)*1000:.3f}mm")
            print(f"   Max step error:           {np.max(step_errors)*1000:.3f}mm")
            print()
            
            # Detailed table
            print("📋 Detailed Results:")
            print("-" * 85)
            print(f"{'Step':<6} {'Chunk':<7} {'Expected X (m)':<15} {'Actual X (m)':<15} {'Cumul Err (mm)':<15} {'Step Err (mm)':<15}")
            print("-" * 85)
            for r in results:
                print(f"{r['step']:<6} {r['chunk']:<7} {r['expected_pos'][0]:<15.6f} {r['actual_pos'][0]:<15.6f} "
                      f"{r['cumulative_err']*1000:<15.3f} {r['step_err']*1000:<15.3f}")
            print("-" * 85)
            
            # Final position comparison
            final_expected = start_pos + np.array([increment * num_steps, 0.0, 0.0])
            final_actual = results[-1]["actual_pos"]
            final_err = np.linalg.norm(final_actual - final_expected)
            
            print()
            print("🎯 Final Position:")
            print(f"   Start:    [{start_pos[0]:.6f}, {start_pos[1]:.6f}, {start_pos[2]:.6f}]")
            print(f"   Expected: [{final_expected[0]:.6f}, {final_expected[1]:.6f}, {final_expected[2]:.6f}]")
            print(f"   Actual:   [{final_actual[0]:.6f}, {final_actual[1]:.6f}, {final_actual[2]:.6f}]")
            print(f"   Error:    {final_err*1000:.3f}mm ({final_err/increment/num_steps*100:.1f}% of total distance)")
            
        elif len(results) > 0:
            print(f"⚠️  Test completed {len(results)}/{num_steps} steps")
            print(f"   Final cumulative error: {results[-1]['cumulative_err']*1000:.3f}mm")
        else:
            print("❌ Test failed - no successful steps")
        
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
    test_cumulative_error()
