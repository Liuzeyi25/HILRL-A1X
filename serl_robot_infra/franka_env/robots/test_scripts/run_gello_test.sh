#!/bin/bash
# Quick start script for Gello bidirectional control test

echo "========================================================================"
echo "Gello Bidirectional Control Test - Quick Start"
echo "========================================================================"
echo ""
echo "This script will:"
echo "  1. Check Gello device connection"
echo "  2. Start the bidirectional control test"
echo ""

# Check if Gello is connected
echo "🔍 Checking Gello device..."
GELLO_PORT="/dev/serial/by-id/usb-FTDI_USB__-__Serial_Converter_FTA7NNNU-if00-port0"

if [ -e "$GELLO_PORT" ]; then
    echo "✅ Gello device found at: $GELLO_PORT"
else
    echo "❌ Gello device not found!"
    echo "   Expected: $GELLO_PORT"
    echo ""
    echo "Available USB devices:"
    ls -l /dev/serial/by-id/usb-FTDI* 2>/dev/null || echo "   No FTDI devices found"
    echo ""
    exit 1
fi

echo ""
echo "📋 Test Controls:"
echo "   [SPACE] - Switch between Follow and Teleoperation mode"
echo "   [Q]     - Quit test"
echo ""
echo "Press Enter to start..."
read

# Run the test
cd "$(dirname "$0")"
python test_gello_bidirectional.py
