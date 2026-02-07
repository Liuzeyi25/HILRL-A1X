# Gello Servo Communication Fix Guide

## Problem Identified

The diagnostic shows **all 7 Dynamixel servos are not responding** with error code `-3001: "There is no status packet"`.

## Root Cause

This error means the servos are **not powered on** or have a power/communication issue.

## Solution Steps

### 1. Check Power Supply ⚡

The Gello device requires external power for the Dynamixel servos:

```bash
# The servos need 12V power supply
# Check if the power adapter is:
- ✅ Plugged into wall outlet
- ✅ Connected to the Gello device
- ✅ Power LED is ON
```

**Common issues:**
- Power adapter not plugged in
- Power switch on Gello is OFF
- Loose power connector
- Dead power supply (test with multimeter: should read ~12V)

### 2. Check Servo Power LED

Look at the Dynamixel servos on the Gello device:

- ✅ **Good**: LEDs are solid or off
- ❌ **Bad**: No LEDs at all → No power
- ⚠️  **Warning**: Blinking red LED → Servo error state

### 3. Check Connections

1. **USB Connection** (Data):
   ```bash
   ls -l /dev/serial/by-id/ | grep FTDI
   # Should show: usb-FTDI_USB__-__Serial_Converter_FTA7NNNU-if00-port0
   ```

2. **Power Connection**: 
   - Check barrel jack is fully inserted
   - Wiggle cable to check for loose connection

3. **Servo Daisy Chain**:
   - All servos connected in series
   - No loose connectors
   - Cables not damaged

### 4. Test After Powering On

Once power is connected, run the diagnostic again:

```bash
cd /home/dungeon_master/conrft/serl_robot_infra/franka_env/robots
python3 test_dynamixel_servos.py
```

Expected output when working:
```
✅ Working servos (7): [1, 2, 3, 4, 5, 6, 7]
❌ Failed servos (0): []
```

### 5. Run Gello Test

After confirming all servos respond:

```bash
./run_gello_test.sh
```

## Quick Checklist

Before running any Gello test:

- [ ] Power adapter plugged into wall
- [ ] Power cable connected to Gello device  
- [ ] Power switch ON (if present)
- [ ] Servo LEDs visible (not blinking red)
- [ ] USB cable connected
- [ ] USB device shows up: `ls -l /dev/serial/by-id/ | grep FTA7NNNU`

## Alternative: Different Baudrate

If servos are powered but still not responding, they might be configured for a different baudrate:

```bash
# Try 57600 or 1000000 instead of 115200
# Edit test_dynamixel_servos.py line 16:
BAUDRATE = 57600  # or 1000000
```

## Alternative: Different Servo IDs

If only some servos respond, check the IDs:

```bash
# Run servo scan (if you have dynamixel_wizard or similar tool)
# Or try different ID ranges in test_dynamixel_servos.py
```

## Current Status

```
❌ All 7 servos not responding
❌ Error: -3001 (no status packet)
→ Most likely: Servos not powered on
```

## Next Steps

1. **Power on the Gello servos**
2. Run `python3 test_dynamixel_servos.py` again
3. Once all servos show ✅, run `./run_gello_test.sh`

---

**Note**: The USB communication is working fine (port opens, no permission errors). The issue is purely that the Dynamixel servos themselves are not responding to commands, which 99% of the time means they don't have power.
