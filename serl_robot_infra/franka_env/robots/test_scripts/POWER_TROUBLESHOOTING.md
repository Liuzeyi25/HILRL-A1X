# 🔴 CRITICAL ISSUE: Gello Servos Not Powered

## Current Status

```
❌ ALL 7 SERVOS FAILING TO RESPOND
❌ Error Code: -3001 (No Status Packet)
❌ Tested at 6 different baudrates: ALL FAILED
✅ USB connection: WORKING
✅ Port permissions: OK
✅ Device detection: OK

DIAGNOSIS: SERVOS ARE NOT POWERED
```

## What This Means

The USB data connection is working perfectly, but the **Dynamixel servos have no power**. 

Think of it like this:
- ✅ USB cable = Phone's data cable (working)
- ❌ Power supply = Phone's charger (MISSING!)

## What You Need to Find

### 1. Power Adapter Specifications
```
Voltage: 12V DC (usually)
Current: 2A or higher (depends on model)
Connector: Barrel jack (2.1mm or 2.5mm center pin)
```

### 2. What It Looks Like
The power adapter is typically:
- A black "wall wart" power brick
- With a barrel plug connector
- Label should say "12V DC" and "2A" or similar
- Might say "Dynamixel" or "Robotis" on it

### 3. Where to Connect
Look for on the Gello device:
- A barrel jack power port (round connector)
- Usually labeled "12V", "POWER", "DC IN", or similar
- Often on the side or back of the control board

## Step-by-Step Power-On Procedure

### Step 1: Locate Power Components
```bash
# Run the interactive checklist
cd /home/dungeon_master/conrft/serl_robot_infra/franka_env/robots
./check_gello_power.sh
```

### Step 2: Physical Inspection

**Look at the Gello device and identify:**

1. **Control Board/Hub** - The main circuit board where servos connect
2. **Power Input** - Barrel jack connector (round hole for power plug)
3. **Power Switch** - May have ON/OFF switch (if present, must be ON)
4. **Servo Cables** - Daisy-chained servos (should all be connected)

**Look at the Dynamixel servos:**
- Small servos with cables connecting them
- Each should have a small LED
- LEDs should light up when powered

### Step 3: Connect Power

```
1. Plug power adapter into wall outlet
   ↓
2. Plug barrel connector into Gello power jack
   ↓
3. Turn on power switch (if present)
   ↓
4. Look for LED indicators on servos
   ↓
5. LEDs should turn on (solid or brief blink)
```

### Step 4: Verify Power

After connecting power, check:

```bash
# Test if servos now respond
python3 test_dynamixel_servos.py
```

**Expected result when working:**
```
✅ Working servos (7): [1, 2, 3, 4, 5, 6, 7]
```

## Common Power Issues

### Issue 1: No Power Adapter Found
```
Solution: 
- Check near the Gello device / in the packaging
- Look for spare power adapters (12V, 2A or higher)
- If lost, you need to purchase a 12V DC power adapter
  - Center positive: (⊕−)
  - 2.1mm or 2.5mm barrel jack
  - At least 2A current rating
```

### Issue 2: Power Adapter But No Lights
```
Troubleshooting:
- Test power adapter with multimeter (should read ~12V DC)
- Try different wall outlet
- Check if adapter has LED indicator (should be lit)
- Inspect barrel connector for damage
- Ensure connector is fully inserted
```

### Issue 3: Some LEDs Work, Some Don't
```
Possible causes:
- Insufficient power supply current (need higher amperage)
- Damaged servo in the chain
- Loose cable connection in servo daisy-chain
```

### Issue 4: LEDs Blinking Red
```
Cause: Servo error state (overload, overheating, voltage issue)

Solution:
1. Turn off power
2. Wait 10 seconds
3. Turn on power
4. Test again
```

## Alternative: Check Gello Documentation

If you have the original Gello documentation:
1. Look for "Power Requirements" section
2. Find power adapter specifications
3. Check setup instructions for power connections

## Visual Indicators

### ✅ GOOD - Servos Are Powered:
- Servo LEDs are visible (any color, solid or off)
- Servos have slight resistance when moved manually
- You might hear a faint buzzing sound
- Power adapter LED is on (if it has one)

### ❌ BAD - Servos Are NOT Powered:
- No LED lights visible on any servo
- Servos move freely with no resistance
- Complete silence (no buzzing)
- Power adapter LED is off or no adapter connected

## After Powering On

Once power is connected and LEDs are visible:

```bash
cd /home/dungeon_master/conrft/serl_robot_infra/franka_env/robots

# Test servos
python3 test_dynamixel_servos.py

# If successful, run the Gello test
./run_gello_test.sh
```

## Still Not Working?

If servos still don't respond after confirming power:

1. **Wrong Baudrate**: Try different baudrates
   ```bash
   python3 scan_dynamixel_servos.py  # Scans all common baudrates
   ```

2. **Non-Standard IDs**: Servos might have IDs other than 1-7
   ```python
   # Edit test_dynamixel_servos.py
   JOINT_IDS = range(1, 20)  # Scan IDs 1-19
   ```

3. **Hardware Failure**: Contact Gello device manufacturer

## Quick Reference Commands

```bash
# Check device connection
python3 check_gello_device.py

# Interactive power checklist
./check_gello_power.sh

# Test servos (standard)
python3 test_dynamixel_servos.py

# Scan all baudrates
python3 scan_dynamixel_servos.py

# Run Gello bidirectional test
./run_gello_test.sh
```

---

**Bottom Line**: The USB communication is working perfectly. You just need to connect the 12V power supply to the Dynamixel servos. Look for the power adapter and power jack on the Gello device! 🔌⚡
