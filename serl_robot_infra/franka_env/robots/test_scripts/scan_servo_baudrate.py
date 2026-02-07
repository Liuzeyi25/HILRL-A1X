#!/usr/bin/env python3
"""
Scan for Dynamixel servos at different baudrates.
"""

import sys
sys.path.insert(0, 'Gello/gello_software')

from dynamixel_sdk import *

DEVICE_NAME = "/dev/serial/by-id/usb-FTDI_USB__-__Serial_Converter_FTA7NNNU-if00-port0"
PROTOCOL_VERSION = 2.0
ADDR_MODEL_NUMBER = 0

BAUDRATES = [57600, 115200, 1000000, 2000000, 3000000, 4000000]
SERVO_IDS = [1, 2, 3, 4, 5, 6, 7]

def main():
    print("=" * 70)
    print("Scanning for Dynamixel servos...")
    print("=" * 70)
    
    for baudrate in BAUDRATES:
        print(f"\n🔍 Trying baudrate: {baudrate}")
        
        portHandler = PortHandler(DEVICE_NAME)
        packetHandler = PacketHandler(PROTOCOL_VERSION)
        
        if not portHandler.openPort():
            print("   ❌ Failed to open port")
            continue
        
        if not portHandler.setBaudRate(baudrate):
            print(f"   ❌ Failed to set baudrate")
            portHandler.closePort()
            continue
        
        found_any = False
        for servo_id in SERVO_IDS:
            model, result, error = packetHandler.read2ByteTxRx(
                portHandler, servo_id, ADDR_MODEL_NUMBER
            )
            if result == COMM_SUCCESS:
                print(f"   ✅ Found servo ID {servo_id}, model: {model}")
                found_any = True
        
        if not found_any:
            print(f"   ⚠️  No servos found at {baudrate}")
        
        portHandler.closePort()
    
    print("\n" + "=" * 70)
    print("Scan complete")
    print("=" * 70)

if __name__ == "__main__":
    main()
