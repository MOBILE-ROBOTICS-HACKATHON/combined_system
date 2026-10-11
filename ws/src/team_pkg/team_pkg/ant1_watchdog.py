"""Stop the ANT1 when commands stop. The robot itself has NO cmd_vel timeout: it keeps
executing the last command forever (found on 10 Oct 2026, needed the e-stop).

    ant1_watchdog            (helper from scripts/ant1_env.sh; keep it running all session)

Watches cmd_vel. If the last command was non-zero and nothing new arrived for `timeout` s,
publishes zeros. Covers: `ros2 topic pub -t N` ending, crashed or closed nodes on any machine.
Commands must therefore be streamed at >= 5 Hz (teleop, our nodes, `ros2 topic pub -r 10` do).
Does NOT cover the laptop or the network dying (zeros can't reach the robot): keep the e-stop in reach.
"""
import time
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from geometry_msgs.msg import Twist


class Ant1Watchdog(Node):
    def __init__(self):
        super().__init__("ant1_watchdog")
        self.timeout = self.declare_parameter("timeout", 0.5).value
        self.pub = self.create_publisher(Twist, "cmd_vel", 10)
        # best-effort subscriber matches both reliable and best-effort publishers
        self.create_subscription(Twist, "cmd_vel", self.on_cmd, qos_profile_sensor_data)
        self.last, self.moving = 0.0, False
        self.create_timer(0.05, self.check)
        self.get_logger().info(f"watching cmd_vel: stop after {self.timeout}s of silence")

    def on_cmd(self, m):
        self.last = time.monotonic()
        self.moving = any((m.linear.x, m.linear.y, m.linear.z, m.angular.x, m.angular.y, m.angular.z))

    def check(self):
        if self.moving and time.monotonic() - self.last > self.timeout:
            self.get_logger().warn("commands stopped while moving -> STOP")
            for _ in range(5):   # several zeros: one lost packet must not leave the robot driving
                self.pub.publish(Twist())
            self.moving = False


def main():
    rclpy.init()
    try:
        rclpy.spin(Ant1Watchdog())
    except KeyboardInterrupt:
        pass
