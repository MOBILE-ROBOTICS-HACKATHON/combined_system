import rclpy
from rclpy.node import Node
from std_msgs.msg import String


class Hello(Node):
    """Template node: copy this to start your real node, and register it in setup.py entry_points."""
    def __init__(self):
        super().__init__("hello_node")
        self.pub = self.create_publisher(String, "hello", 10)
        self.create_timer(1.0, lambda: self.pub.publish(String(data="hello hackathon")))


def main():
    rclpy.init()
    rclpy.spin(Hello())
