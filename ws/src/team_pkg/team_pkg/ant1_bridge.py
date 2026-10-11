"""ANT1 Robot Bridge Node.
Subscribes to high-level motion command topics (or standard cmd_vel),
enforces speed safety clamps and dead-man timeout / watchdog,
and continuously streams the exact Twist message format to the robot's hardware topic:
    /olive/olixRobot/ant1/id001/cmd_vel
"""

import time
import signal
import rclpy
from rclpy.node import Node
from rclpy.signals import SignalHandlerOptions
from geometry_msgs.msg import Twist
from std_msgs.msg import Bool

# ANT1 Pro Hardware Defaults & Limits
DEFAULT_ROBOT_TOPIC = "/olive/olixRobot/ant1/id001/cmd_vel"
DEFAULT_EMS_TOPIC = "/olive/olixRobot/ant1/id001/ems"

# Safety bounds: Olive ANT1 Pro is mecanum (x, y, yaw)
MAX_VX = 0.25   # m/s
MAX_VY = 0.25   # m/s
MAX_WZ = 0.6    # rad/s


class Ant1BridgeNode(Node):
    def __init__(self):
        super().__init__("ant1_bridge_node")

        # Declare parameters
        self.declare_parameter("robot_cmd_vel_topic", DEFAULT_ROBOT_TOPIC)
        self.declare_parameter("input_cmd_vel_topic", "cmd_vel_input")
        self.declare_parameter("ems_topic", DEFAULT_EMS_TOPIC)
        self.declare_parameter("publish_rate_hz", 10.0)
        self.declare_parameter("timeout_sec", 0.5)
        self.declare_parameter("max_vx", MAX_VX)
        self.declare_parameter("max_vy", MAX_VY)
        self.declare_parameter("max_wz", MAX_WZ)

        robot_topic = self.get_parameter("robot_cmd_vel_topic").value
        input_topic = self.get_parameter("input_cmd_vel_topic").value
        ems_topic = self.get_parameter("ems_topic").value
        rate_hz = float(self.get_parameter("publish_rate_hz").value)
        self.timeout = float(self.get_parameter("timeout_sec").value)
        self.max_vx = float(self.get_parameter("max_vx").value)
        self.max_vy = float(self.get_parameter("max_vy").value)
        self.max_wz = float(self.get_parameter("max_wz").value)

        # Hardware publisher to the ANT1 robot
        self.robot_pub = self.create_publisher(Twist, robot_topic, 10)

        # Input command subscription (from planner / teleop / test publisher)
        self.create_subscription(Twist, input_topic, self.on_input_cmd, 10)

        # E-stop listener
        self.create_subscription(Bool, ems_topic, self.on_ems, 10)

        self.last_cmd = Twist()
        self.last_cmd_time = 0.0
        self.is_moving = False
        self.e_stop_active = False

        timer_period = 1.0 / max(rate_hz, 1.0)
        self.timer = self.create_timer(timer_period, self.stream_loop)

        self.get_logger().info(
            f"ANT1 Bridge initialized:\n"
            f"  - Subscribing from: {input_topic} (type: geometry_msgs/msg/Twist)\n"
            f"  - Publishing to:   {robot_topic} @ {rate_hz:.1f}Hz\n"
            f"  - Safety watchdog:  {self.timeout:.2f}s timeout\n"
            f"  - Max speeds: vx={self.max_vx} m/s, vy={self.max_vy} m/s, wz={self.max_wz} rad/s"
        )

    def on_ems(self, msg: Bool):
        if msg.data and not self.e_stop_active:
            self.get_logger().warn("Hardware E-STOP engaged! Sending STOP commands.")
            self.e_stop_active = True
            self.send_stop(repeats=3)
        elif not msg.data and self.e_stop_active:
            self.get_logger().info("E-STOP released.")
            self.e_stop_active = False

    def on_input_cmd(self, msg: Twist):
        self.last_cmd_time = time.monotonic()

        # Clamp linear and angular velocities to safe thresholds
        clamped = Twist()
        clamped.linear.x = max(-self.max_vx, min(self.max_vx, float(msg.linear.x)))
        clamped.linear.y = max(-self.max_vy, min(self.max_vy, float(msg.linear.y)))
        clamped.linear.z = 0.0
        clamped.angular.x = 0.0
        clamped.angular.y = 0.0
        clamped.angular.z = max(-self.max_wz, min(self.max_wz, float(msg.angular.z)))

        self.last_cmd = clamped
        self.is_moving = any((
            abs(clamped.linear.x) > 1e-4,
            abs(clamped.linear.y) > 1e-4,
            abs(clamped.angular.z) > 1e-4
        ))

    def stream_loop(self):
        now = time.monotonic()

        if self.e_stop_active:
            self.robot_pub.publish(Twist())
            return

        # Watchdog: If no message was received for `timeout` seconds, stop immediately
        if self.is_moving and (now - self.last_cmd_time > self.timeout):
            self.get_logger().warn(
                f"Command timeout (> {self.timeout:.2f}s silence) while moving. Halting robot!"
            )
            self.send_stop(repeats=5)
            self.is_moving = False
            self.last_cmd = Twist()
            return

        # Continuously stream current command
        self.robot_pub.publish(self.last_cmd)

    def send_stop(self, repeats=5):
        zero = Twist()
        for _ in range(repeats):
            self.robot_pub.publish(zero)
            time.sleep(0.02)


def _interrupt(*_):
    raise KeyboardInterrupt


def main():
    signal.signal(signal.SIGHUP, _interrupt)
    signal.signal(signal.SIGTERM, _interrupt)
    rclpy.init(signal_handler_options=SignalHandlerOptions.NO)

    node = Ant1BridgeNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.get_logger().info("Stopping bridge and halting robot...")
        node.send_stop(repeats=5)
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
