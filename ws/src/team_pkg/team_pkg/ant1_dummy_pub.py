"""Dummy command publisher for testing AMT robot movements.
Publishes timed test profiles (forward, backward, strafe left/right, turn left/right)
at 10Hz to cmd_vel_input or directly to the robot topic.
"""

import time
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist

TEST_SEQUENCE = [
    # (vx, vy, wz, duration_sec, description)
    (0.05, 0.0, 0.0, 2.0, "Move Forward 5 cm/s"),
    (0.0, 0.0, 0.0, 1.0, "Pause"),
    (-0.05, 0.0, 0.0, 2.0, "Move Backward 5 cm/s"),
    (0.0, 0.0, 0.0, 1.0, "Pause"),
    (0.0, 0.05, 0.0, 2.0, "Strafe Left (mecanum) 5 cm/s"),
    (0.0, 0.0, 0.0, 1.0, "Pause"),
    (0.0, -0.05, 0.0, 2.0, "Strafe Right (mecanum) 5 cm/s"),
    (0.0, 0.0, 0.0, 1.0, "Pause"),
    (0.0, 0.0, 0.2, 2.0, "Rotate Counter-Clockwise 0.2 rad/s"),
    (0.0, 0.0, 0.0, 1.0, "Pause"),
    (0.0, 0.0, -0.2, 2.0, "Rotate Clockwise -0.2 rad/s"),
    (0.0, 0.0, 0.0, 1.0, "Stop complete"),
]


class DummyCmdPublisher(Node):
    def __init__(self):
        super().__init__("dummy_cmd_publisher")
        self.declare_parameter("topic", "cmd_vel_input")
        self.declare_parameter("loop", False)
        topic_name = self.get_parameter("topic").value
        self.loop = self.get_parameter("loop").value

        self.pub = self.create_publisher(Twist, topic_name, 10)
        self.step_idx = 0
        self.step_start_time = time.monotonic()
        self.timer = self.create_timer(0.1, self.timer_callback) # 10 Hz
        self.finished = False

        self.get_logger().info(f"Publishing dummy motion sequence to '{topic_name}' at 10Hz...")

    def timer_callback(self):
        if self.finished:
            return

        now = time.monotonic()
        vx, vy, wz, duration, desc = TEST_SEQUENCE[self.step_idx]

        if now - self.step_start_time >= duration:
            self.step_idx += 1
            if self.step_idx >= len(TEST_SEQUENCE):
                if self.loop:
                    self.step_idx = 0
                    self.step_start_time = now
                    vx, vy, wz, duration, desc = TEST_SEQUENCE[0]
                else:
                    self.get_logger().info("Test sequence completed. Sending final stop.")
                    self.pub.publish(Twist())
                    self.finished = True
                    return
            else:
                self.step_start_time = now
                vx, vy, wz, duration, desc = TEST_SEQUENCE[self.step_idx]
                self.get_logger().info(f"Step {self.step_idx + 1}/{len(TEST_SEQUENCE)}: {desc}")

        cmd = Twist()
        cmd.linear.x = float(vx)
        cmd.linear.y = float(vy)
        cmd.angular.z = float(wz)
        self.pub.publish(cmd)


def main():
    rclpy.init()
    node = DummyCmdPublisher()
    try:
        while rclpy.ok() and not node.finished:
            rclpy.spin_once(node, timeout_sec=0.1)
    except KeyboardInterrupt:
        pass
    finally:
        # Send zeros
        node.pub.publish(Twist())
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
