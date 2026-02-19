#!/usr/bin/env python3
"""
Comprehensive Dynamixel servo scanner - tries multiple baudrates and IDs.
"""

import sys
import time

sys.path.insert(0, 'Gello/gello_software')

from dynamixel_sdk import *

# Dynamixel settings
ADDR_TORQUE_ENABLE = 64
ADDR_PRESENT_POSITION = 132
ADDR_MODEL_NUMBER = 0
PROTOCOL_VERSION = 2.0
DEVICENAME = '/dev/serial/by-id/usb-FTDI_USB__-__Serial_Converter_FTA7NNNU-if00-port0'

# Common baudrates for Dynamixel servos
BAUDRATES = [
    57600,    # Common default
    115200,   # Often used
    1000000,  # High speed
    2000000,  # Very high speed
    3000000,  # Maximum speed
    9600,     # Low speed fallback
]

def scan_baudrate(baudrate, max_id=15):
    """Scan for servos at a specific baudrate."""
    print(f"\n{'='*70}")
    print(f"Scanning at baudrate: {baudrate}")
    print(f"{'='*70}")
    
    portHandler = PortHandler(DEVICENAME)
    packetHandler = PacketHandler(PROTOCOL_VERSION)
    
    # Open port
    if not portHandler.openPort():
        print(f"❌ Failed to open port")
        return []
    
    # Set baudrate
    if not portHandler.setBaudRate(baudrate):
        print(f"❌ Failed to set baudrate")
        portHandler.closePort()
        return []
    
    print(f"✅ Port opened at {baudrate} baud")
    
    # Give hardware time to settle
    time.sleep(0.3)
    
    found_servos = []
    
    # Quick scan of common IDs (1-15)
    for servo_id in range(1, max_id + 1):
        # Try to read model number (more reliable than position)
        model_num, dxl_comm_result, dxl_error = packetHandler.read2ByteTxRx(
            portHandler, servo_id, ADDR_MODEL_NUMBER
        )
        
        if dxl_comm_result == COMM_SUCCESS and dxl_error == 0:
            # Try to read position to verify it's really working
            position, pos_result, pos_error = packetHandler.read4ByteTxRx(
                portHandler, servo_id, ADDR_PRESENT_POSITION
            )
            
            if pos_result == COMM_SUCCESS:
                print(f"  ✅ Found servo ID {servo_id}: Model {model_num}, Position {position}")
                found_servos.append({
                    'id': servo_id,
                    'model': model_num,
                    'position': position,
                    'baudrate': baudrate
                })
            else:
                print(f"  ⚠️  Servo ID {servo_id}: Model {model_num} but can't read position")
        
        # Small delay between attempts
        time.sleep(0.05)
    
    portHandler.closePort()
    
    if found_servos:
        print(f"\n✅ Found {len(found_servos)} servo(s) at {baudrate} baud")
    else:
        print(f"❌ No servos found at {baudrate} baud")
    
    return found_servos

def test_hardware_connection():
    """Test if the port can be opened at all."""
    print("\n" + "="*70)
    print("Hardware Connection Test")
    print("="*70)
    
    portHandler = PortHandler(DEVICENAME)
    
    print(f"\n📡 Testing port: {DEVICENAME}")
    
    if not portHandler.openPort():
        print(f"❌ Cannot open port - USB device issue")
        print(f"\nTroubleshooting:")
        print(f"  1. Check USB cable is connected")
        print(f"  2. Run: ls -l /dev/serial/by-id/ | grep FTA7NNNU")
        print(f"  3. Check permissions: ls -l {DEVICENAME}")
        return False
    
    print(f"✅ Port opened successfully")
    portHandler.closePort()
    return True

def main():
    print("="*70)
    print("Comprehensive Dynamixel Servo Scanner")
    print("="*70)
    print("\nThis will scan for servos at multiple baudrates.")
    print("This may take 1-2 minutes...\n")
    
    # First test if we can even open the port
    if not test_hardware_connection():
        return 1
    
    # Scan at each baudrate
    all_found_servos = []
    
    for baudrate in BAUDRATES:
        found = scan_baudrate(baudrate, max_id=15)
        all_found_servos.extend(found)
        time.sleep(0.5)  # Brief pause between baudrates
    
    # Final summary
    print("\n" + "="*70)
    print("FINAL SCAN RESULTS")
    print("="*70)
    
    if all_found_servos:
        print(f"\n✅ Total servos found: {len(all_found_servos)}")
        print("\nDetails:")
        for servo in all_found_servos:
            print(f"  • ID {servo['id']}: Model {servo['model']} at {servo['baudrate']} baud (pos: {servo['position']})")
        
        # Group by baudrate
        baudrate_groups = {}
        for servo in all_found_servos:
            br = servo['baudrate']
            if br not in baudrate_groups:
                baudrate_groups[br] = []
            baudrate_groups[br].append(servo['id'])
        
        print("\nServos by baudrate:")
        for br, ids in baudrate_groups.items():
            print(f"  {br} baud: IDs {ids}")
        
        print("\n💡 Next step: Update BAUDRATE in test_dynamixel_servos.py if needed")
        return 0
    else:
        print("\n❌ NO SERVOS FOUND AT ANY BAUDRATE")
        print("\nThis means:")
        print("  1. ⚡ SERVOS ARE NOT POWERED - Most likely cause!")
        print("     → Check 12V power supply is connected and turned on")
        print("     → Look for LED indicators on servos")
        print("     → Verify power cable is plugged in")
        print("\n  2. 🔌 Wrong device or all servos have non-standard IDs")
        print("     → Verify this is the correct FTDI device")
        print("\n  3. 🔧 Hardware failure")
        print("     → Damaged servos or communication bus")
        print("     → Try different USB cable")
        
        print("\n" + "="*70)
        print("POWER CHECK")
        print("="*70)
        print("\n❓ Questions to check:")
        print("  • Is there a power adapter connected to the Gello device?")
        print("  • Is the power adapter plugged into a wall outlet?")
        print("  • Is there a power switch on the Gello? If yes, is it ON?")
        print("  • Do any LED lights appear on the Dynamixel servos?")
        print("  • Can you hear or feel the servos when you move them?")
        print("    (Powered servos have slight resistance)")
        
        return 1

if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n\n⚠️  Scan interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
