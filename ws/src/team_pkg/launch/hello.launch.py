from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    # Add use_sim_time, remappings, namespaces here. Real robots: use_sim_time=False!
    return LaunchDescription([Node(package="team_pkg", executable="hello_node", output="screen")])
