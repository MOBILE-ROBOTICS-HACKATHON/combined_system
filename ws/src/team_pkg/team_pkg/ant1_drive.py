"""Drive the ANT1 (or any robot with cmd_vel) through timed moves: step 1 of the robot work.

    ant1_drive -p vx:=0.1 -p secs:=2.0        # forward 0.1 m/s for 2 s  (helper from scripts/ant1_env.sh)
    ant1_drive -p vy:=0.1                     # sideways left (mecanum)
    ant1_drive -p demo:=true                  # forward, back, left, right, turn left, turn right
    = ros2 run team_pkg ant1_drive --ros-args -r __ns:=$NS -p ...

vx/vy in m/s (+x forward, +y left), wz in rad/s (+ = counter-clockwise). Speeds are clamped to
max_v / max_w. Always ends with zero commands, also on Ctrl-C. The robot's own laser safety zone
and e-stop still apply on top of this.
"""
import signal
import time
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.signals import SignalHandlerOptions
from geometry_msgs.msg import Twist
from std_msgs.msg import Bool

DEMO = [  # (vx, vy, wz, secs): 0.1 m/s * 2 s = 20 cm per move, keep ~0.5 m free all around
    (0.1, 0.0, 0.0, 2.0), (-0.1, 0.0, 0.0, 2.0),
    (0.0, 0.1, 0.0, 2.0), (0.0, -0.1, 0.0, 2.0),
    (0.0, 0.0, 0.3, 2.0), (0.0, 0.0, -0.3, 2.0),
]


class Ant1Drive(Node):
    def __init__(self):
        super().__init__("ant1_drive")
        p = lambda name, default: self.declare_parameter(name, default).value
        max_v, max_w = p("max_v", 0.3), p("max_w", 1.0)
        moves = DEMO if p("demo", False) else [(p("vx", 0.0), p("vy", 0.0), p("wz", 0.0), p("secs", 2.0))]
        clamp = lambda x, m: max(-m, min(m, float(x)))
        self.moves = [(clamp(vx, max_v), clamp(vy, max_v), clamp(wz, max_w), float(s)) for vx, vy, wz, s in moves]
        self.pub = self.create_publisher(Twist, "cmd_vel", 10)
        self.create_subscription(Bool, "ems", self.on_ems, qos_profile_sensor_data)
        self.i, self.t0, self.done = -1, 0.0, False
        self.create_timer(0.1, self.tick)   # 10 Hz, like a joystick: the robot gets a steady command stream

    def on_ems(self, msg):
        if msg.data:
            self.get_logger().warn("E-STOP active: robot will not move (twist the button to release)",
                                   throttle_duration_sec=2.0)

    def tick(self):
        now = time.monotonic()
        if self.i < 0 or now - self.t0 >= self.moves[self.i][3]:
            self.i, self.t0 = self.i + 1, now
            if self.i == len(self.moves):
                self.done = True
                return
            self.get_logger().info("move %d/%d: vx=%.2f vy=%.2f wz=%.2f for %.1fs"
                                   % (self.i + 1, len(self.moves), *self.moves[self.i]))
        cmd = Twist()
        cmd.linear.x, cmd.linear.y, cmd.angular.z = self.moves[self.i][:3]
        self.pub.publish(cmd)

    def stop(self):
        for _ in range(5):   # several zeros: one lost packet must not leave the robot driving
            self.pub.publish(Twist())
            time.sleep(0.05)
        self.get_logger().info("stopped")


def _interrupt(*_):
    raise KeyboardInterrupt


def main():
    # Closing the terminal (SIGHUP) or `docker stop` (SIGTERM) must also reach stop() below.
    signal.signal(signal.SIGHUP, _interrupt)
    signal.signal(signal.SIGTERM, _interrupt)
    # No rclpy signal handler: Ctrl-C raises KeyboardInterrupt but keeps the context alive for stop().
    rclpy.init(signal_handler_options=SignalHandlerOptions.NO)
    node = Ant1Drive()
    try:
        while rclpy.ok() and not node.done:
            rclpy.spin_once(node, timeout_sec=0.1)
    except KeyboardInterrupt:
        pass
    finally:
        node.stop()
        node.destroy_node()
        rclpy.try_shutdown()
