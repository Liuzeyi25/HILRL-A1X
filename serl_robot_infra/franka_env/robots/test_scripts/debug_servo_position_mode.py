#!/usr/bin/env python3
"""
Debug servo state after enabling position control mode.
"""

import sys
import time
sys.path.insert(0, 'Gello/gello_software')

from dynamixel_sdk import *

DEVICE_NAME = "/dev/serial/by-id/usb-FTDI_USB__-__Serial_Converter_FTA7NNNU-if00-port0"
PROTOCOL_VERSION = 2.0
BAUDRATE = 57600

ADDR_OPERATING_MODE = 11
ADDR_TORQUE_ENABLE = 64
ADDR_PRESENT_POSITION = 132
ADDR_GOAL_POSITION = 116
ADDR_HARDWARE_ERROR = 70

CURRENT_CONTROL_MODE = 0
POSITION_CONTROL_MODE = 3

def main():
    print("=" * 70)
    print("Servo Position Control Debug")
    print("=" * 70)
    
    portHandler = PortHandler(DEVICE_NAME)
    packetHandler = PacketHandler(PROTOCOL_VERSION)
    
    if not portHandler.openPort() or not portHandler.setBaudRate(BAUDRATE):
        print("❌ Failed to open port")
        return
    
    print("✅ Port opened")
    
    servo_id = 1
    
    # Read current state
    print(f"\n📊 Current state of servo {servo_id}:")
    
    mode, _, _ = packetHandler.read1ByteTxRx(portHandler, servo_id, ADDR_OPERATING_MODE)
    print(f"   Operating mode: {mode}")
    
    torque, _, _ = packetHandler.read1ByteTxRx(portHandler, servo_id, ADDR_TORQUE_ENABLE)
    print(f"   Torque enabled: {bool(torque)}")
    
    pos, _, _ = packetHandler.read4ByteTxRx(portHandler, servo_id, ADDR_PRESENT_POSITION)
    print(f"   Current position: {pos}")
    
    error, _, _ = packetHandler.read1ByteTxRx(portHandler, servo_id, ADDR_HARDWARE_ERROR)
    print(f"   Hardware error: {error} (0x{error:02x})")
    
    # Switch to position control
    print(f"\n🔄 Switching to position control mode...")
    
    # Disable torque
    result, _ = packetHandler.write1ByteTxRx(portHandler, servo_id, ADDR_TORQUE_ENABLE, 0)
    print(f"   Disable torque: result={result}")
    time.sleep(0.2)
    
    # Set mode
    result, _ = packetHandler.write1ByteTxRx(portHandler, servo_id, ADDR_OPERATING_MODE, POSITION_CONTROL_MODE)
    print(f"   Set position mode: result={result}")
    time.sleep(0.3)
    
    # Enable torque
    result, _ = packetHandler.write1ByteTxRx(portHandler, servo_id, ADDR_TORQUE_ENABLE, 1)
    print(f"   Enable torque: result={result}")
    time.sleep(0.5)
    
    # Check state again
    print(f"\n📊 State after enabling position control:")
    
    mode, _, _ = packetHandler.read1ByteTxRx(portHandler, servo_id, ADDR_OPERATING_MODE)
    print(f"   Operating mode: {mode}")
    
    torque, _, _ = packetHandler.read1ByteTxRx(portHandler, servo_id, ADDR_TORQUE_ENABLE)
    print(f"   Torque enabled: {bool(torque)}")
    
    pos, result, _ = packetHandler.read4ByteTxRx(portHandler, servo_id, ADDR_PRESENT_POSITION)
    print(f"   Current position: {pos} (result={result})")
    
    error, _, _ = packetHandler.read1ByteTxRx(portHandler, servo_id, ADDR_HARDWARE_ERROR)
    print(f"   Hardware error: {error} (0x{error:02x})")
    
    # Try to write position
    print(f"\n🔄 Trying to write position {pos} (same as current)...")
    result, error = packetHandler.write4ByteTxRx(portHandler, servo_id, ADDR_GOAL_POSITION, pos)
    print(f"   Result: {result}")
    print(f"   Error: {error}")
    print(f"   Result string: {packetHandler.getTxRxResult(result)}")
    
    if error != 0:
        print(f"   Servo error details: {packetHandler.getRxPacketError(error)}")
    
    # Disable torque
    packetHandler.write1ByteTxRx(portHandler, servo_id, ADDR_TORQUE_ENABLE, 0)
    
    portHandler.closePort()
    print("\n👋 Done")

if __name__ == "__main__":
    main()
