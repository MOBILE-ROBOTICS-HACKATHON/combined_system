"""Arrow-key driving, like turtlesim's turtle_teleop_key, but safe for a real robot.

    ant1_keys                                   (helper from scripts/ant1_env.sh)
    = ros2 run team_pkg ant1_keys --ros-args -r __ns:=$NS [-p speed:=0.05 -p turn:=0.2]

  Up/Down    forward / backward        a / d   sideways left / right (mecanum)
  Left/Right turn left / right         space   stop        q   quit
  + / -      faster / slower (x1.25 per press, up to the caps)

Dead-man: it moves ONLY while a key is held (keyboard auto-repeat). Release the key and it
sends zeros. Speeds are hard-capped at MAX_V / MAX_W no matter what parameters ask for.
"""
import os
import select
import signal
import sys
import termios
import time
import tty
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.signals import SignalHandlerOptions
from geometry_msgs.msg import Twist
from std_msgs.msg import Bool

MAX_V, MAX_W = 0.15, 0.5   # hard caps (m/s, rad/s)
HOLD = 0.6                 # s without a key -> stop. Must exceed the OS auto-repeat delay (~0.5 s)
KEYS = {  # key -> (vx, vy, wz) as multiples of speed/turn
    "\x1b[A": (1, 0, 0), "\x1b[B": (-1, 0, 0), "\x1b[D": (0, 0, 1), "\x1b[C": (0, 0, -1),
    "a": (0, 1, 0), "d": (0, -1, 0),
}


class Ant1Keys(Node):
    def __init__(self):
        super().__init__("ant1_keys")
        self.v = min(self.declare_parameter("speed", 0.05).value, MAX_V)
        self.w = min(self.declare_parameter("turn", 0.2).value, MAX_W)
        self.pub = self.create_publisher(Twist, "cmd_vel", 10)
        self.create_subscription(Bool, "ems", self.on_ems, qos_profile_sensor_data)
        self.cmd, self.last_key, self.moving = (0, 0, 0), 0.0, False

    def on_ems(self, msg):
        if msg.data:
            self.get_logger().warn("E-STOP active: robot will not move", throttle_duration_sec=2.0)

    def key(self, k):
        if k in KEYS:
            self.cmd, self.last_key = KEYS[k], time.monotonic()
        elif k == " ":
            self.last_key = 0.0
        elif len(k) == 1 and k in "+=-_":   # '=' and '_' too: same keys without Shift
            f = 1.25 if k in "+=" else 0.8
            self.v = round(min(max(self.v * f, 0.01), MAX_V), 3)
            self.w = round(min(max(self.w * f, 0.05), MAX_W), 3)
            print(f"\rspeed {self.v} m/s, turn {self.w} rad/s   ", flush=True)

    def tick(self):
        """Call at ~10 Hz: drive while a key is held, else send zeros once on release."""
        if time.monotonic() - self.last_key < HOLD:
            t = Twist()
            t.linear.x, t.linear.y, t.angular.z = self.cmd[0] * self.v, self.cmd[1] * self.v, self.cmd[2] * self.w
            self.pub.publish(t)
            self.moving = True
        elif self.moving:
            self.stop()

    def stop(self):
        for _ in range(5):   # several zeros: one lost packet must not leave the robot driving
            self.pub.publish(Twist())
            time.sleep(0.02)
        self.moving = False


def read_key(timeout):
    """One keypress from the raw terminal (arrow keys arrive as 3-byte escape sequences)."""
    if not select.select([sys.stdin], [], [], timeout)[0]:
        return None
    k = os.read(sys.stdin.fileno(), 1).decode(errors="ignore")
    if k == "\x1b" and select.select([sys.stdin], [], [], 0.01)[0]:
        k += os.read(sys.stdin.fileno(), 2).decode(errors="ignore")
    return k


def _interrupt(*_):
    raise KeyboardInterrupt


def main():
    # Closing the terminal (SIGHUP) or `docker stop` (SIGTERM) must also reach stop() below.
    signal.signal(signal.SIGHUP, _interrupt)
    signal.signal(signal.SIGTERM, _interrupt)
    rclpy.init(signal_handler_options=SignalHandlerOptions.NO)
    node = Ant1Keys()
    print(__doc__.split("\n\n")[2])
    print(f"speed {node.v} m/s, turn {node.w} rad/s (caps {MAX_V} / {MAX_W})")
    old = termios.tcgetattr(sys.stdin)
    try:
        tty.setcbreak(sys.stdin.fileno())   # keys arrive immediately, Ctrl-C still works
        while rclpy.ok():
            k = read_key(0.1)
            if k == "q":
                break
            if k:
                node.key(k)
            node.tick()
            for _ in range(30):                          # ALL waiting messages, not one per loop (fell behind at ~20/s)
                rclpy.spin_once(node, timeout_sec=0)
    except KeyboardInterrupt:
        pass
    finally:
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old)
        node.stop()
        node.get_logger().info("stopped")
        node.destroy_node()
        rclpy.try_shutdown()
