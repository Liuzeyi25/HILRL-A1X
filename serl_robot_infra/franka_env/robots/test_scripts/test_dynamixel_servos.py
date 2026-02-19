#!/usr/bin/env python3
"""
Simple script to test Dynamixel servo communication and reset if needed.
"""

import sys
import time

sys.path.insert(0, 'Gello/gello_software')

from dynamixel_sdk import *

# Dynamixel settings
ADDR_TORQUE_ENABLE = 64
ADDR_PRESENT_POSITION = 132
PROTOCOL_VERSION = 2.0
BAUDRATE = 115200
DEVICENAME = '/dev/serial/by-id/usb-FTDI_USB__-__Serial_Converter_FTA7NNNU-if00-port0'

# Default joint IDs for Gello
JOINT_IDS = [1, 2, 3, 4, 5, 6, 7]

def test_servo_communication():
    """Test basic communication with each servo."""
    print("=" * 70)
    print("Dynamixel Servo Communication Test")
    print("=" * 70)
    
    # Initialize PortHandler and PacketHandler
    portHandler = PortHandler(DEVICENAME)
    packetHandler = PacketHandler(PROTOCOL_VERSION)
    
    # Open port
    print(f"\n📡 Opening port: {DEVICENAME}")
    if not portHandler.openPort():
        print(f"❌ Failed to open port")
        return False
    print("✅ Port opened")
    
    # Set baudrate
    print(f"\n⚙️  Setting baudrate: {BAUDRATE}")
    if not portHandler.setBaudRate(BAUDRATE):
        print(f"❌ Failed to set baudrate")
        portHandler.closePort()
        return False
    print("✅ Baudrate set")
    
    # Give hardware time to settle
    time.sleep(0.5)
    
    # Test each servo
    print(f"\n🔍 Testing communication with each servo...")
    print("-" * 70)
    
    working_servos = []
    failed_servos = []
    
    for servo_id in JOINT_IDS:
        print(f"\nServo ID {servo_id}:")
        
        # Try to read present position
        dxl_present_position, dxl_comm_result, dxl_error = packetHandler.read4ByteTxRx(
            portHandler, servo_id, ADDR_PRESENT_POSITION
        )
        
        if dxl_comm_result != COMM_SUCCESS:
            print(f"  ❌ Communication failed: {packetHandler.getTxRxResult(dxl_comm_result)}")
            print(f"     Error code: {dxl_comm_result}")
            failed_servos.append(servo_id)
        elif dxl_error != 0:
            print(f"  ⚠️  Communication OK but servo error: {packetHandler.getRxPacketError(dxl_error)}")
            print(f"     Error code: {dxl_error}")
            print(f"     Position: {dxl_present_position}")
            # Consider this as working but with warning
            working_servos.append(servo_id)
        else:
            print(f"  ✅ Communication OK")
            print(f"     Position: {dxl_present_position}")
            working_servos.append(servo_id)
        
        time.sleep(0.1)  # Small delay between servos
    
    # Summary
    print("\n" + "=" * 70)
    print("Test Summary:")
    print("=" * 70)
    print(f"✅ Working servos ({len(working_servos)}): {working_servos}")
    print(f"❌ Failed servos ({len(failed_servos)}): {failed_servos}")
    
    # Try to disable torque on working servos
    if working_servos:
        print("\n🔧 Attempting to disable torque on working servos...")
        for servo_id in working_servos:
            dxl_comm_result, dxl_error = packetHandler.write1ByteTxRx(
                portHandler, servo_id, ADDR_TORQUE_ENABLE, 0
            )
            if dxl_comm_result == COMM_SUCCESS and dxl_error == 0:
                print(f"  ✅ Servo {servo_id}: torque disabled")
            else:
                print(f"  ⚠️  Servo {servo_id}: could not disable torque (code: {dxl_comm_result}, error: {dxl_error})")
            time.sleep(0.1)
    
    # Close port
    portHandler.closePort()
    print("\n🔌 Port closed")
    
    if failed_servos:
        print("\n⚠️  Some servos failed to communicate!")
        print("Possible causes:")
        print("  1. Servo power supply issue")
        print("  2. Loose or damaged cables")
        print("  3. Wrong servo ID configuration")
        print("  4. Servo hardware failure")
        return False
    else:
        print("\n✅ All servos communicating successfully!")
        return True

def main():
    try:
        success = test_servo_communication()
        if success:
            print("\n💡 You can now try running the Gello test again")
            return 0
        else:
            print("\n❌ Fix the servo issues before proceeding")
            return 1
    except KeyboardInterrupt:
        print("\n\n⚠️  Test interrupted by user")
        return 1
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
