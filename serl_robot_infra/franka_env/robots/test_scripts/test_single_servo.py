#!/usr/bin/env python3
"""
Test single servo in position control mode.
"""

import sys
import time

sys.path.insert(0, 'Gello/gello_software')

from dynamixel_sdk import *

# Control table addresses
ADDR_OPERATING_MODE = 11
ADDR_TORQUE_ENABLE = 64
ADDR_PRESENT_POSITION = 132
ADDR_GOAL_POSITION = 116

# Operating modes
CURRENT_CONTROL_MODE = 0
POSITION_CONTROL_MODE = 3

PROTOCOL_VERSION = 2.0
BAUDRATE = 115200
DEVICE_NAME = "/dev/serial/by-id/usb-FTDI_USB__-__Serial_Converter_FTA7NNNU-if00-port0"

def main():
    print("=" * 70)
    print("Single Servo Test (ID 1)")
    print("=" * 70)
    
    # Initialize
    portHandler = PortHandler(DEVICE_NAME)
    packetHandler = PacketHandler(PROTOCOL_VERSION)
    
    if not portHandler.openPort():
        print("❌ Failed to open port")
        return
    
    if not portHandler.setBaudRate(BAUDRATE):
        print("❌ Failed to set baudrate")
        return
    
    print("✅ Port opened")
    
    servo_id = 1
    
    # First, ensure we can communicate - try to read operating mode
    print(f"\n📊 Checking servo {servo_id} status...")
    mode, result, error = packetHandler.read1ByteTxRx(portHandler, servo_id, ADDR_OPERATING_MODE)
    if result != COMM_SUCCESS:
        print(f"❌ Cannot communicate with servo: {packetHandler.getTxRxResult(result)}")
        return
    print(f"   Current operating mode: {mode}")
    if mode == CURRENT_CONTROL_MODE:
        print("   → Current control mode (free-wheeling)")
    elif mode == POSITION_CONTROL_MODE:
        print("   → Position control mode")
    
    # Check if torque is enabled
    torque, result, error = packetHandler.read1ByteTxRx(portHandler, servo_id, ADDR_TORQUE_ENABLE)
    if result == COMM_SUCCESS:
        print(f"   Torque enabled: {bool(torque)}")
    
    # Read current position (will only work if in position mode or torque disabled)
    print(f"\n📊 Reading current position...")
    pos, result, error = packetHandler.read4ByteTxRx(portHandler, servo_id, ADDR_PRESENT_POSITION)
    if result != COMM_SUCCESS:
        print(f"   ⚠️  Cannot read position in current mode: {packetHandler.getTxRxResult(result)}")
        print("   Will use position=2048 (neutral) for test")
        pos = 2048
    else:
        print(f"   Current raw position: {pos}")
        print(f"   Current angle: {pos * np.pi / 2048:.3f} rad")
    
    # Disable torque
    print(f"\n🔄 Disabling torque...")
    result, error = packetHandler.write1ByteTxRx(portHandler, servo_id, ADDR_TORQUE_ENABLE, 0)
    if result != COMM_SUCCESS:
        print(f"❌ Failed: {packetHandler.getTxRxResult(result)}")
        return
    time.sleep(0.2)
    
    # Set to position control mode
    print(f"\n🔄 Setting to position control mode...")
    result, error = packetHandler.write1ByteTxRx(portHandler, servo_id, ADDR_OPERATING_MODE, POSITION_CONTROL_MODE)
    if result != COMM_SUCCESS:
        print(f"❌ Failed: {packetHandler.getTxRxResult(result)}")
        return
    time.sleep(0.3)
    
    # Enable torque
    print(f"\n🔄 Enabling torque...")
    result, error = packetHandler.write1ByteTxRx(portHandler, servo_id, ADDR_TORQUE_ENABLE, 1)
    if result != COMM_SUCCESS:
        print(f"❌ Failed: {packetHandler.getTxRxResult(result)}")
        return
    time.sleep(0.5)
    
    print("✅ Position control mode enabled")
    
    # Try to write same position
    print(f"\n🔄 Commanding same position: {pos}")
    result, error = packetHandler.write4ByteTxRx(portHandler, servo_id, ADDR_GOAL_POSITION, pos)
    if result != COMM_SUCCESS:
        print(f"❌ Failed: {packetHandler.getTxRxResult(result)}")
        print(f"   Error code: {error}")
    else:
        print(f"✅ Command successful!")
    
    # Disable torque
    print(f"\n🔄 Disabling torque...")
    packetHandler.write1ByteTxRx(portHandler, servo_id, ADDR_TORQUE_ENABLE, 0)
    
    portHandler.closePort()
    print("\n👋 Done")

if __name__ == "__main__":
    import numpy as np
    main()
