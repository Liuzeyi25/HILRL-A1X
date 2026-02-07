"""
Bidirectional teleoperation demo for A1_X + Gello.

Demonstrates switching between:
1. Normal teleoperation: Human moves Gello → A1_X follows
2. Reverse teleoperation: A1_X autonomous → Gello follows
3. Safety mode: Stop both robots
"""

import numpy as np
import time
import threading
from enum import Enum
from typing import Optional

from gello.agents.gello_agent import GelloAgent
from gello.agents.gello_follower import GelloFollower
from franka_env.robots.a1x_robot import A1XRobot


class TeleoperationMode(Enum):
    """Teleoperation modes."""
    STOPPED = 0
    NORMAL = 1      # Gello → A1_X
    REVERSE = 2     # A1_X → Gello
    AUTONOMOUS = 3  # A1_X autonomous, Gello follows


class BidirectionalTeleoperation:
    """
    Manages bidirectional teleoperation between Gello and A1_X.
    
    Usage:
        teleop = BidirectionalTeleoperation()
        teleop.start()
        
        # Normal teleoperation
        teleop.set_mode(TeleoperationMode.NORMAL)
        
        # Autonomous exploration with Gello following
        teleop.set_mode(TeleoperationMode.AUTONOMOUS)
        
        teleop.stop()
    """
    
    def __init__(
        self,
        gello_port: str = "/dev/serial/by-id/usb-FTDI_USB__-__Serial_Converter_FTA7NNNU-if00-port0",
        a1x_port: int = 6100,
        control_freq: float = 50.0,
    ):
        """
        Args:
            gello_port: Serial port for Gello
            a1x_port: ZMQ port for A1_X
            control_freq: Control loop frequency (Hz)
        """
        print("Initializing bidirectional teleoperation...")
        
        # Initialize robots
        self.gello_agent = GelloAgent(port=gello_port)
        self.gello_robot = self.gello_agent._robot
        self.a1x_robot = A1XRobot(num_dofs=7, port=a1x_port)
        
        # Initialize follower for reverse teleoperation
        self.gello_follower = GelloFollower(self.gello_robot)
        
        # Control parameters
        self.control_freq = control_freq
        self.dt = 1.0 / control_freq
        
        # State
        self.mode = TeleoperationMode.STOPPED
        self.is_running = False
        self.control_thread: Optional[threading.Thread] = None
        
        print("Initialization complete!")
    
    def start(self):
        """Start teleoperation system."""
        if self.is_running:
            print("Already running!")
            return
        
        self.is_running = True
        self.control_thread = threading.Thread(target=self._control_loop)
        self.control_thread.start()
        print("Teleoperation system started.")
    
    def stop(self):
        """Stop teleoperation system."""
        print("Stopping teleoperation...")
        self.is_running = False
        
        if self.control_thread is not None:
            self.control_thread.join()
        
        # Ensure safe state
        if self.mode == TeleoperationMode.REVERSE:
            self.gello_follower.stop()
        
        self.a1x_robot.close()
        print("Teleoperation stopped.")
    
    def set_mode(self, mode: TeleoperationMode):
        """
        Switch teleoperation mode.
        
        Args:
            mode: Target teleoperation mode
        """
        if mode == self.mode:
            return
        
        print(f"Switching from {self.mode.name} to {mode.name}...")
        
        # Exit current mode
        if self.mode == TeleoperationMode.REVERSE:
            self.gello_follower.stop()
        
        # Enter new mode
        if mode == TeleoperationMode.NORMAL:
            print("Mode: Human moves Gello → A1_X follows")
        
        elif mode == TeleoperationMode.REVERSE:
            print("Mode: A1_X moves → Gello follows")
            self.gello_follower.start()
        
        elif mode == TeleoperationMode.AUTONOMOUS:
            print("Mode: A1_X autonomous exploration, Gello mirrors")
            self.gello_follower.start()
        
        elif mode == TeleoperationMode.STOPPED:
            print("Mode: All motors stopped")
        
        self.mode = mode
    
    def _control_loop(self):
        """Main control loop (runs in separate thread)."""
        while self.is_running:
            start_time = time.time()
            
            if self.mode == TeleoperationMode.NORMAL:
                self._normal_teleoperation()
            
            elif self.mode == TeleoperationMode.REVERSE:
                self._reverse_teleoperation()
            
            elif self.mode == TeleoperationMode.AUTONOMOUS:
                self._autonomous_mode()
            
            # Sleep to maintain control frequency
            elapsed = time.time() - start_time
            sleep_time = max(0, self.dt - elapsed)
            time.sleep(sleep_time)
    
    def _normal_teleoperation(self):
        """Normal mode: Gello → A1_X."""
        # Read Gello joint state
        gello_joints = self.gello_agent.act({})
        
        # Command A1_X to follow
        self.a1x_robot.update_command(gello_joints)
    
    def _reverse_teleoperation(self):
        """Reverse mode: A1_X → Gello."""
        # Read A1_X joint state
        a1x_joints = self.a1x_robot.get_joint_state()
        
        # Command Gello to follow
        self.gello_follower.command_follow(a1x_joints)
    
    def _autonomous_mode(self):
        """Autonomous mode: Policy controls A1_X, Gello mirrors."""
        # Get policy action (placeholder - replace with actual policy)
        action = self._get_policy_action()
        
        # Execute action on A1_X
        self.a1x_robot.update_command(action)
        
        # Make Gello mirror A1_X
        a1x_joints = self.a1x_robot.get_joint_state()
        self.gello_follower.command_follow(a1x_joints)
    
    def _get_policy_action(self) -> np.ndarray:
        """
        Get action from trained policy.
        
        TODO: Replace with actual policy inference.
        """
        # Placeholder: Stay at current position
        return self.a1x_robot.get_joint_state()
    
    def get_current_state(self) -> dict:
        """Get current system state."""
        return {
            "mode": self.mode.name,
            "gello_joints": self.gello_robot.get_joint_state(),
            "a1x_joints": self.a1x_robot.get_joint_state(),
            "is_running": self.is_running,
        }


# ============================================================================
# Demo Scripts
# ============================================================================

def demo_mode_switching():
    """Demo: Switch between different teleoperation modes."""
    teleop = BidirectionalTeleoperation()
    teleop.start()
    
    try:
        print("\n" + "="*60)
        print("Demo: Mode Switching")
        print("="*60)
        
        # Phase 1: Normal teleoperation
        print("\n[Phase 1] Normal teleoperation (10 seconds)")
        print("→ Move Gello manually, A1_X will follow\n")
        teleop.set_mode(TeleoperationMode.NORMAL)
        time.sleep(10)
        
        # Phase 2: Reverse teleoperation
        print("\n[Phase 2] Reverse teleoperation (10 seconds)")
        print("→ Gello will follow A1_X position\n")
        teleop.set_mode(TeleoperationMode.REVERSE)
        time.sleep(10)
        
        # Phase 3: Stop
        print("\n[Phase 3] Stopping...")
        teleop.set_mode(TeleoperationMode.STOPPED)
        
        # Print final state
        state = teleop.get_current_state()
        print(f"\nFinal state:")
        print(f"  Mode: {state['mode']}")
        print(f"  Gello joints: {state['gello_joints'][:3]}...")
        print(f"  A1_X joints: {state['a1x_joints'][:3]}...")
        
    finally:
        teleop.stop()


def demo_autonomous_with_follower():
    """Demo: Autonomous exploration with Gello following."""
    from examples.experiments.a1x_pick_banana.config20260127 import TrainConfig
    
    print("\n" + "="*60)
    print("Demo: Autonomous Exploration + Gello Follower")
    print("="*60)
    
    # Setup environment
    config = TrainConfig()
    env = config.get_environment(fake_env=False)
    
    # Initialize teleop
    teleop = BidirectionalTeleoperation()
    teleop.start()
    teleop.set_mode(TeleoperationMode.AUTONOMOUS)
    
    try:
        obs, _ = env.reset()
        print("\nRunning autonomous exploration...")
        print("→ Gello will mirror robot movements\n")
        
        for step in range(100):
            # Policy action (placeholder)
            action = env.action_space.sample()
            
            # Execute
            obs, reward, done, truncated, info = env.step(action)
            
            # Gello automatically follows (handled by teleop thread)
            
            if step % 25 == 0:
                print(f"Step {step}: Reward = {reward:.3f}")
            
            if done:
                break
        
        print("\nExploration complete!")
        
    finally:
        teleop.stop()
        env.close()


if __name__ == "__main__":
    import sys
    
    print("Bidirectional Teleoperation for A1_X + Gello")
    print("=" * 60)
    
    if len(sys.argv) > 1 and sys.argv[1] == "autonomous":
        demo_autonomous_with_follower()
    else:
        demo_mode_switching()
