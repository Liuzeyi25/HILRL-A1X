#!/usr/bin/env python3
"""
Diagnostic tool to check Gello device status and identify issues.
"""

import os
import sys
import subprocess
import serial.tools.list_ports

def check_device_permissions(device_path):
    """Check if we have read/write permissions on the device."""
    if not os.path.exists(device_path):
        return False, "Device does not exist"
    
    if not os.access(device_path, os.R_OK):
        return False, "No read permission"
    
    if not os.access(device_path, os.W_OK):
        return False, "No write permission"
    
    return True, "Read/write permissions OK"

def check_device_in_use(device_path):
    """Check if the device is being used by another process."""
    try:
        # Use lsof to check if device is open
        result = subprocess.run(
            ['lsof', device_path],
            capture_output=True,
            text=True
        )
        
        if result.returncode == 0 and result.stdout.strip():
            return True, result.stdout.strip()
        else:
            return False, "Device not in use"
    except FileNotFoundError:
        return None, "lsof command not found (install lsof to check)"

def list_serial_devices():
    """List all available serial devices."""
    ports = serial.tools.list_ports.comports()
    if not ports:
        return []
    
    devices = []
    for port in ports:
        devices.append({
            'device': port.device,
            'description': port.description,
            'hwid': port.hwid,
        })
    return devices

def main():
    print("=" * 70)
    print("Gello Device Diagnostic Tool")
    print("=" * 70)
    
    # Expected Gello device path
    gello_device = "/dev/serial/by-id/usb-FTDI_USB__-__Serial_Converter_FTA7NNNU-if00-port0"
    
    print(f"\n🔍 Checking Gello device: {gello_device}")
    print("-" * 70)
    
    # Check if device exists
    if os.path.exists(gello_device):
        print(f"✅ Device exists")
        
        # Check permissions
        has_perm, perm_msg = check_device_permissions(gello_device)
        if has_perm:
            print(f"✅ {perm_msg}")
        else:
            print(f"❌ {perm_msg}")
            print(f"   Try: sudo chmod 666 {gello_device}")
        
        # Check if in use
        in_use, use_msg = check_device_in_use(gello_device)
        if in_use is None:
            print(f"⚠️  {use_msg}")
        elif in_use:
            print(f"⚠️  Device is in use by another process:")
            print(f"\n{use_msg}\n")
            print(f"   You may need to stop the other process first")
        else:
            print(f"✅ {use_msg}")
        
        # Get device info
        try:
            real_path = os.path.realpath(gello_device)
            print(f"📍 Real device path: {real_path}")
        except Exception as e:
            print(f"⚠️  Could not resolve real path: {e}")
    else:
        print(f"❌ Device not found!")
        print(f"\n💡 Possible reasons:")
        print(f"   1. Gello is not plugged in")
        print(f"   2. Wrong USB port or device ID")
        print(f"   3. USB cable issue")
    
    # List all serial devices
    print("\n" + "=" * 70)
    print("Available Serial Devices:")
    print("=" * 70)
    
    devices = list_serial_devices()
    if devices:
        for i, dev in enumerate(devices, 1):
            print(f"\n{i}. {dev['device']}")
            print(f"   Description: {dev['description']}")
            print(f"   Hardware ID: {dev['hwid']}")
    else:
        print("\n❌ No serial devices found")
    
    # Check for FTDI devices specifically
    print("\n" + "=" * 70)
    print("FTDI Devices (potential Gello devices):")
    print("=" * 70)
    
    try:
        result = subprocess.run(
            ['ls', '-la', '/dev/serial/by-id/'],
            capture_output=True,
            text=True
        )
        
        if result.returncode == 0:
            ftdi_devices = [line for line in result.stdout.split('\n') if 'FTDI' in line]
            if ftdi_devices:
                for line in ftdi_devices:
                    print(line)
            else:
                print("❌ No FTDI devices found")
        else:
            print("⚠️  Could not list /dev/serial/by-id/")
    except Exception as e:
        print(f"⚠️  Error checking FTDI devices: {e}")
    
    print("\n" + "=" * 70)
    print("Diagnosis Complete")
    print("=" * 70)

if __name__ == "__main__":
    main()
