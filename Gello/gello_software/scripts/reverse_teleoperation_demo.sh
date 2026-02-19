#!/bin/bash
# Quick demo for reverse teleoperation (Gello follows A1_X)

set -e

echo "======================================"
echo "Reverse Teleoperation Demo"
echo "======================================"
echo ""
echo "This demo shows Gello following A1_X robot movements."
echo ""

# Check if A1_X ROS2 node is running
echo "[1/3] Checking A1_X ROS2 node..."
if ! pgrep -f "a1x_ros2_node.py" > /dev/null; then
    echo "Error: A1_X ROS2 node is not running!"
    echo "Please start it first:"
    echo "  cd /home/dungeon_master/conrft/serl_robot_infra/robot_servers"
    echo "  python3.10 a1x_ros2_node.py"
    exit 1
fi
echo "✓ A1_X ROS2 node is running"

# Activate Python environment
echo ""
echo "[2/3] Activating Python environment..."
cd /home/dungeon_master/conrft
source venv/bin/activate 2>/dev/null || echo "Note: Using system Python"

# Run demo
echo ""
echo "[3/3] Starting reverse teleoperation..."
echo ""
echo "Instructions:"
echo "  - Gello will start in FREE-WHEELING mode"
echo "  - When reverse mode starts, Gello will FOLLOW A1_X"
echo "  - You can manually move Gello at any time (override)"
echo ""
read -p "Press Enter to continue..."

# Set Python path
export PYTHONPATH="/home/dungeon_master/conrft/Gello/gello_software:$PYTHONPATH"
export PYTHONPATH="/home/dungeon_master/conrft/serl_robot_infra:$PYTHONPATH"
export PYTHONPATH="/home/dungeon_master/conrft:$PYTHONPATH"

# Run the demo
python3 << 'PYTHON_SCRIPT'
import sys
sys.path.insert(0, '/home/dungeon_master/conrft/Gello/gello_software')
sys.path.insert(0, '/home/dungeon_master/conrft/serl_robot_infra')

from gello.agents.gello_agent import GelloAgent
from gello.agents.gello_follower import GelloFollower
from franka_env.robots.a1x_robot import A1XRobot
import time
import numpy as np

# Initialize
print("\nInitializing robots...")
gello_agent = GelloAgent(
    port="/dev/serial/by-id/usb-FTDI_USB__-__Serial_Converter_FTA7NNNU-if00-port0"
)
a1x = A1XRobot(num_dofs=7, port=6100)
follower = GelloFollower(gello_agent._robot)

# Demo sequence
try:
    # Phase 1: Normal mode
    print("\n" + "="*60)
    print("Phase 1: Normal Teleoperation (5 seconds)")
    print("="*60)
    print("→ Move Gello manually. A1_X will follow.\n")
    
    for i in range(250):  # 5 seconds at 50Hz
        gello_joints = gello_agent.act({})
        a1x.update_command(gello_joints)
        time.sleep(0.02)
    
    # Phase 2: Reverse mode
    print("\n" + "="*60)
    print("Phase 2: Reverse Teleoperation (10 seconds)")
    print("="*60)
    print("→ Gello will now FOLLOW A1_X!")
    print("→ Watch Gello move automatically...\n")
    
    input("Press Enter to enable reverse mode...")
    follower.start()
    
    # Simulate robot movement (or use actual policy)
    for i in range(500):  # 10 seconds at 50Hz
        # Get A1_X current position
        a1x_joints = a1x.get_joint_state()
        
        # Optional: Add small sinusoidal motion for demo
        if i < 250:
            a1x_joints[1] += 0.001 * np.sin(i * 0.1)  # Small motion
            a1x.update_command(a1x_joints)
        
        # Make Gello follow
        follower.command_follow(a1x_joints)
        time.sleep(0.02)
        
        if i % 100 == 0:
            print(f"Step {i}/500: Following A1_X position...")
    
    print("\n✓ Demo complete!")
    
finally:
    print("\nCleaning up...")
    follower.stop()
    a1x.close()
    print("Done! Gello returned to free-wheeling mode.")

PYTHON_SCRIPT

echo ""
echo "======================================"
echo "Demo finished!"
echo "======================================"
