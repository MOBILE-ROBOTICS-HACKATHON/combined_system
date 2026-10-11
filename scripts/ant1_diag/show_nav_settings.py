import rclpy
from rcl_interfaces.srv import GetParameters
NS = "/olive/olixRobot/ant1/id001"
want = {
 f"{NS}/controller_server": ["FollowPath.max_vel_x", "FollowPath.min_vel_x", "FollowPath.max_vel_y", "FollowPath.vy_samples",
    "FollowPath.max_vel_theta", "FollowPath.PathAlign.scale", "FollowPath.GoalAlign.scale", "general_goal_checker.xy_goal_tolerance",
    "general_goal_checker.yaw_goal_tolerance", "failure_tolerance", "progress_checker.movement_time_allowance"],
 f"{NS}/velocity_smoother": ["max_velocity", "min_velocity"],
 f"{NS}/behavior_server": ["max_rotational_vel"],
 f"{NS}/local_costmap/local_costmap": ["inflation_layer.inflation_radius"],
 f"{NS}/global_costmap/global_costmap": ["inflation_layer.inflation_radius", "inflation_layer.cost_scaling_factor"],
 "/olive_olixRobot_id001_ant1": ["laser_safety_distance_cm"],
}
def val(v): return {1:v.bool_value,2:v.integer_value,3:v.double_value,8:list(v.double_array_value)}.get(v.type, "?")
rclpy.init(); n = rclpy.create_node("verify_nav")
for node, names in want.items():
    c = n.create_client(GetParameters, f"{node}/get_parameters"); c.wait_for_service(timeout_sec=8)
    f = c.call_async(GetParameters.Request(names=names)); rclpy.spin_until_future_complete(n, f, timeout_sec=8)
    print(node.split("/")[-1] + ": " + ", ".join(f"{k.split('.')[-2] + '.' if k.count('.') > 1 else ''}{k.split('.')[-1]}={val(v)}" for k, v in zip(names, f.result().values)))
