# A1_X Robot Integration for Gello

## Overview

This implementation allows you to control the A1_X robot arm using Gello. The A1_X robot uses ROS2 for control, which requires Python 3.10, while the Gello software uses Python 3.11.

## Architecture

To solve the Python version compatibility issue, this implementation uses a **bridge architecture**:

1. **Main Process** (Python 3.11): Runs the Gello control loop
2. **ROS2 Node Process** (Python 3.10): Handles ROS2 communication with A1_X
3. **ZMQ Bridge**: Connects the two processes

```
┌─────────────────────────────────────┐
│  Gello Control Loop (Python 3.11)  │
│  - A1XRobot class                   │
│  - Agent (GelloAgent)               │
└──────────────┬──────────────────────┘
               │ ZMQ (port 6100)
┌──────────────▼──────────────────────┐
│  ROS2 Node (Python 3.10)            │
│  - A1_X_ros2_node.py                │
│  - Publishes to /motion_target/...  │
│  - Subscribes to /hdas/feedback_arm │
└─────────────────────────────────────┘
```

## Files

- `gello/robots/A1_X.py`: Main robot class (runs in Python 3.11)
- `gello/robots/A1_X_ros2_node.py`: ROS2 bridge node (runs in Python 3.10)
- `configs/yam_A1_X.yaml`: Example configuration file

## ROS2 Topics

- **Command topic**: `/motion_target/target_joint_state_arm` (sensor_msgs/msg/JointState)
- **Feedback topic**: `/hdas/feedback_arm` (sensor_msgs/msg/JointState)

## Prerequisites

1. ROS2 Humble installed at `/opt/ros/humble`
2. System Python 3.10 with ROS2 packages
3. Python 3.11 conda environment with Gello dependencies
4. ZMQ installed in both environments

## Usage

### 1. Test the A1_X robot class

```bash
cd /home/dungeon_master/Gello/gello_software
python gello/robots/A1_X.py
```

This will:
- Start the ROS2 node subprocess
- Connect via ZMQ
- Print current joint states
- Test commanding joints

### 2. Run with Gello

```bash
cd /home/dungeon_master/Gello/gello_software
python experiments/launch_yaml.py --left-config-path configs/yam_A1_X.yaml
```

## Configuration

Edit `configs/yam_A1_X.yaml` to customize:

```yaml
robot:
  _target_: gello.robots.A1_X.A1XRobot
  num_dofs: 7                      # Number of joints
  node_name: "a1x_gello_node"      # ROS2 node name
  port: 6100                        # ZMQ port (change if 6100 is busy)
  python_path: "/usr/bin/python3"   # Path to system Python 3.10

agent:
  _target_: gello.agents.gello_agent.GelloAgent
  port: "/dev/ttyUSB0"              # Dynamixel port
  
hz: 30                               # Control frequency
```

## Troubleshooting

### ZMQ port already in use

If port 6100 is occupied, change it in the config:

```yaml
robot:
  port: 6101  # Use a different port
```

### ROS2 topics not found

Make sure the A1_X robot system is running:

```bash
ros2 topic list
# Should show:
# /motion_target/target_joint_state_arm
# /hdas/feedback_arm
```

### Python version issues

Verify system Python has ROS2:

```bash
/usr/bin/python3 --version  # Should be 3.10.x
source /opt/ros/humble/setup.bash
/usr/bin/python3 -c "import rclpy; print('OK')"
```

Verify conda environment has ZMQ:

```bash
conda activate gello_software
python -c "import zmq; print('OK')"
```

### Node subprocess fails to start

Check the subprocess output manually:

```bash
cd /tmp
source /opt/ros/humble/setup.bash
/usr/bin/python3 /home/dungeon_master/Gello/gello_software/gello/robots/A1_X_ros2_node.py --port 6100
```

## Development

### Testing the ROS2 node independently

```bash
# Terminal 1: Start ROS2 node
cd /tmp
source /opt/ros/humble/setup.bash
/usr/bin/python3 /home/dungeon_master/Gello/gello_software/gello/robots/A1_X_ros2_node.py --port 6100

# Terminal 2: Test with ZMQ client
python -c "
import zmq
context = zmq.Context()
socket = context.socket(zmq.REQ)
socket.connect('tcp://localhost:6100')
socket.send_json({'cmd': 'get_state'})
print(socket.recv_json())
"
```

## Notes

- The subprocess is automatically started when `A1XRobot` is instantiated
- The subprocess is automatically cleaned up when `close()` is called
- Communication latency is typically <1ms via ZMQ on localhost
