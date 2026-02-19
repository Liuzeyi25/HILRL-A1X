#!/bin/bash
# Quick Gello Power Check Script

echo "========================================================================"
echo "Gello Device Power Checklist"
echo "========================================================================"
echo ""
echo "The servo diagnostic shows: NO SERVOS RESPONDING"
echo "Error: -3001 (no status packet) on all servos"
echo ""
echo "This means the servos are NOT POWERED."
echo ""
echo "========================================================================"
echo "PLEASE CHECK THESE ITEMS:"
echo "========================================================================"
echo ""

read -p "1. Is there a power adapter/cable connected to Gello? (y/n): " power_cable
if [ "$power_cable" != "y" ]; then
    echo "   ❌ Connect the power adapter to the Gello device"
    echo "   The Gello needs external 12V power for the Dynamixel servos"
    exit 1
fi

read -p "2. Is the power adapter plugged into a wall outlet? (y/n): " wall_power
if [ "$wall_power" != "y" ]; then
    echo "   ❌ Plug the power adapter into a wall outlet"
    exit 1
fi

read -p "3. Is there a power switch? If yes, is it ON? (y/n/no-switch): " power_switch
if [ "$power_switch" = "n" ]; then
    echo "   ❌ Turn ON the power switch on the Gello device"
    exit 1
fi

echo ""
read -p "4. Do you see ANY LED lights on the Dynamixel servos? (y/n): " led_lights
if [ "$led_lights" != "y" ]; then
    echo "   ❌ No LED lights = No power to servos"
    echo ""
    echo "   Troubleshooting:"
    echo "   • Check power adapter LED (should be lit)"
    echo "   • Try a different power outlet"
    echo "   • Check if power cable is fully inserted"
    echo "   • Verify power adapter voltage (should be ~12V DC)"
    echo "   • Check for a fuse or circuit breaker on the Gello"
    exit 1
fi

echo ""
read -p "5. When you manually move the servos, do they have resistance? (y/n): " resistance
if [ "$resistance" = "n" ]; then
    echo "   ⚠️  Servos move too freely = Powered but maybe in error state"
else
    echo "   ✅ Servos have resistance (good sign)"
fi

echo ""
echo "========================================================================"
echo "LED Status Check"
echo "========================================================================"
echo ""
echo "Look at the Dynamixel servo LEDs:"
echo "  • Solid/Off = Normal (good)"
echo "  • Blinking RED = Error state (needs reset)"
echo "  • No light = No power (bad)"
echo ""

read -p "Are any servo LEDs blinking RED? (y/n): " red_blink
if [ "$red_blink" = "y" ]; then
    echo "   ⚠️  Servo error state detected"
    echo "   This can happen if servo was commanded beyond limits"
    echo "   Solution: Power cycle the servos (turn off/on power)"
fi

echo ""
echo "========================================================================"
echo "Next Steps"
echo "========================================================================"
echo ""

if [ "$led_lights" = "y" ]; then
    echo "Since you see LED lights, servos should have power."
    echo ""
    echo "Try these steps:"
    echo "  1. Power cycle: Turn OFF power, wait 5 seconds, turn ON"
    echo "  2. Run the test again:"
    echo "     python3 test_dynamixel_servos.py"
    echo ""
    echo "  3. If still fails, try different baudrate:"
    echo "     Edit test_dynamixel_servos.py line 16"
    echo "     Change BAUDRATE = 115200 to BAUDRATE = 57600"
    echo ""
    echo "  4. Check if servos have non-standard configuration"
else
    echo "❌ SERVOS HAVE NO POWER"
    echo ""
    echo "Action required:"
    echo "  1. Find the 12V power adapter for the Gello"
    echo "  2. Connect it properly"
    echo "  3. Verify LED lights turn on"
    echo "  4. Then run: python3 test_dynamixel_servos.py"
fi

echo ""
echo "========================================================================"
