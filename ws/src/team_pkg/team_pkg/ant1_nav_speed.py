"""Configure the ANT1's onboard Nav2 for our use (everything resets on every robot reboot).

    ant1_nav_speed              # 0.2 m/s, 0.5 rad/s
    ant1_nav_speed 0.1 0.3      # slower
    ant1_nav_speed 0.3 0.6 --live   # change ONLY the speed, on the fly (also while driving a goal; no restart)
    ant1_nav_speed +0.1 --live      # relative: current speed + 0.1 m/s (turn unchanged)
    ant1_nav_speed -0.1 -0.2 --live # relative: slower and slower turning
    = ros2 run team_pkg ant1_nav_speed [v [w]] [--live] --ros-args -r __ns:=$NS

Capped at V_MAX / W_MAX (0.5 m/s, 1.0 rad/s; Olive's own default is 0.3 m/s).

Robot must be STANDING STILL: some settings are only read when Nav2 starts, so this RESTARTS Nav2
(controller, planner, behaviours; ~10 s; the SLAM map is kept).

What it sets (Olive defaults, 10 Oct 2026, in brackets):
  - speed v / turn w                          [0.3 m/s, 0.65 rad/s]
  - reverse + sideways driving                [forward + turn only; vy_samples 0]
  - no "point your nose along the path"       [PathAlign/GoalAlign 12.5] -> slides instead of turning
  - obstacle clearance 0.25 m, steeper cost   [0.325 m, scaling 2.0] -> may pass closer to things
  - goal reached within 0.3 m / 0.3 rad       [0.25 m / 0.1 rad]
  - controller may fail 1 s before aborting   [0.3 s]
  - progress: 0.25 m within 10 s              [0.5 m in 2 s -> slow drives aborted as "no progress"]
  - recovery spin at most w                   [1.0 rad/s]
  - robot laser safety stop 19 cm             [19 cm, Olive's default]
  - planner obstacles from the LIVE lidar: 25x25 m window around the robot, marked up to 10 m,
    cleared up to 12 m                        [the stored SLAM map + lidar marked within 2.5 m]
    Why: in a room where people walk and chairs/tables move, the SLAM map keeps ghosts (10 Oct:
    22 % of its obstacles were no longer there) and the planner detoured around them or found no
    path at all. The SLAM map is still built and used for localisation.
    ant1_nav_speed --static-map   goes back to Olive's map-based planning (for a room that doesn't change)
"""
import sys
import time
import rclpy
from rcl_interfaces.msg import Parameter, ParameterType, ParameterValue
from rcl_interfaces.srv import SetParameters
from rcl_interfaces.srv import GetParameters
from nav2_msgs.srv import ManageLifecycleNodes
from nav_msgs.msg import Odometry

V_MAX, W_MAX = 0.5, 1.0   # sanity caps (m/s, rad/s). The robot could do 2.5 m/s; Olive's Nav2 default is 0.3


def param(name, v):
    if isinstance(v, list) and v and isinstance(v[0], str):
        t, f = ParameterType.PARAMETER_STRING_ARRAY, "string_array_value"
    elif isinstance(v, list):
        t, f = ParameterType.PARAMETER_DOUBLE_ARRAY, "double_array_value"
    elif isinstance(v, bool):                  # before int: bool is a subclass of int
        t, f = ParameterType.PARAMETER_BOOL, "bool_value"
    elif isinstance(v, int):
        t, f = ParameterType.PARAMETER_INTEGER, "integer_value"
    else:
        t, f = ParameterType.PARAMETER_DOUBLE, "double_value"
    return Parameter(name=name, value=ParameterValue(type=t, **{f: v}))


def robot_moving(node):
    """True/False from 2 s of odometry, None if no odometry arrived."""
    speeds = []
    node.create_subscription(Odometry, "odom", lambda m: speeds.append(
        abs(m.twist.twist.linear.x) + abs(m.twist.twist.linear.y) + abs(m.twist.twist.angular.z)), 10)
    end = time.monotonic() + 2.0
    while time.monotonic() < end:
        rclpy.spin_once(node, timeout_sec=0.1)
    return max(speeds) > 0.03 if speeds else None


def call(node, typ, name, req, timeout=10.0):
    cli = node.create_client(typ, name)
    if not cli.wait_for_service(timeout_sec=5.0):
        return None
    f = cli.call_async(req)
    rclpy.spin_until_future_complete(node, f, timeout_sec=timeout)
    return f.result()


def main():
    args = rclpy.utilities.remove_ros_args(sys.argv)[1:]
    nums = [a for a in args if not a.startswith("--")]
    rclpy.init()
    node = rclpy.create_node("ant1_nav_speed")
    v, w = 0.2, 0.5
    if any(a[0] in "+-" for a in nums):    # relative change: start from what Nav2 uses right now
        r = call(node, GetParameters, "controller_server/get_parameters",
                 GetParameters.Request(names=["FollowPath.max_vel_x", "FollowPath.max_vel_theta"]))
        if r is None:
            print("controller_server not reachable: can't read the current speed")
            return
        v, w = r.values[0].double_value, r.values[1].double_value
        print(f"current: {v} m/s, {w} rad/s")
    for i, a in enumerate(nums[:2]):
        val = float(a)
        if i == 0:
            v = v + val if a[0] in "+-" else val
        else:
            w = w + val if a[0] in "+-" else val
    cv, cw = round(min(max(v, 0.05), V_MAX), 3), round(min(max(w, 0.1), W_MAX), 3)
    if (cv, cw) != (round(v, 3), round(w, 3)):
        print(f"clamped to {cv} m/s, {cw} rad/s (allowed 0.05..{V_MAX} m/s, 0.1..{W_MAX} rad/s)")
    v, w = cv, cw
    if "--live" in args:   # only the limits Nav2 re-reads while running: no restart, safe mid-goal
        live = {
            "controller_server": [param("FollowPath.max_vel_x", v), param("FollowPath.min_vel_x", -v),
                                  param("FollowPath.max_vel_y", v), param("FollowPath.min_vel_y", -v),
                                  param("FollowPath.max_speed_xy", v), param("FollowPath.max_vel_theta", w)],
            "velocity_smoother": [param("max_velocity", [v, v, w]), param("min_velocity", [-v, -v, -w])],
        }
        ok = True
        for srv, params in live.items():
            r = call(node, SetParameters, f"{srv}/set_parameters", SetParameters.Request(parameters=params))
            ok &= bool(r) and all(x.successful for x in r.results)
        print(f"live: {v} m/s, {w} rad/s ({'OK' if ok else 'FAILED'}; the recovery spin keeps its start-up speed)")
        node.destroy_node()
        rclpy.try_shutdown()
        return
    moving = robot_moving(node)
    if moving is None:
        print("no odometry from the robot: is it on and is ANT1_IFACE right? nothing changed")
        return
    if moving:
        print("robot is MOVING: this restarts Nav2, run it when the robot stands still. nothing changed")
        return
    if "--static-map" in args:   # Olive's original: plan on the stored SLAM map (+ lidar within 2.5 m)
        global_obstacles = [param("plugins", ["static_layer", "obstacle_layer", "inflation_layer"]),
                            param("rolling_window", False), param("width", 5), param("height", 5),
                            param("resolution", 0.02),
                            param("obstacle_layer.scan.obstacle_max_range", 2.5),
                            param("obstacle_layer.scan.raytrace_max_range", 3.0)]
    else:                        # live lidar only: moved chairs / walking people don't leave ghosts
        global_obstacles = [param("plugins", ["obstacle_layer", "inflation_layer"]),
                            param("rolling_window", True), param("width", 25), param("height", 25),
                            param("resolution", 0.05),
                            param("obstacle_layer.scan.obstacle_max_range", 10.0),
                            param("obstacle_layer.scan.raytrace_max_range", 12.0)]
    todo = {
        "controller_server": [
            param("FollowPath.max_vel_x", v), param("FollowPath.min_vel_x", -v),
            param("FollowPath.max_vel_y", v), param("FollowPath.min_vel_y", -v),
            param("FollowPath.max_speed_xy", v), param("FollowPath.max_vel_theta", w),
            param("FollowPath.acc_lim_y", 0.25), param("FollowPath.decel_lim_y", -0.3),
            # read at start-up only. 20x10x32 = 6400 trajectories ran the controller at 0.1-1 Hz instead
            # of 20 (10 Oct): goals timed out and the lifecycle manager kept resetting Nav2. 8x5x10 = 400.
            param("FollowPath.vx_samples", 8), param("FollowPath.vy_samples", 5),
            param("FollowPath.vtheta_samples", 10),
            param("FollowPath.PathAlign.scale", 0.0), param("FollowPath.GoalAlign.scale", 0.0),
            param("FollowPath.xy_goal_tolerance", 0.3),
            param("general_goal_checker.xy_goal_tolerance", 0.3),
            param("general_goal_checker.yaw_goal_tolerance", 0.3),
            param("failure_tolerance", 1.0),
            param("progress_checker.required_movement_radius", 0.25),
            param("progress_checker.movement_time_allowance", 10.0)],
        "velocity_smoother": [
            param("max_velocity", [v, v, w]), param("min_velocity", [-v, -v, -w]),
            param("max_accel", [0.3, 0.3, 1.0]), param("max_decel", [-0.3, -0.3, -0.7])],
        "behavior_server": [
            param("max_rotational_vel", w), param("min_rotational_vel", min(0.1, w)),
            param("rotational_acc_lim", 1.0)],
        "local_costmap/local_costmap": [param("inflation_layer.inflation_radius", 0.25)],
        "global_costmap/global_costmap": [param("inflation_layer.inflation_radius", 0.25),
                                          param("inflation_layer.cost_scaling_factor", 3.0)] + global_obstacles,
        "/olive_olixRobot_id001_ant1": [param("laser_safety_distance_cm", 19)],
    }
    ok = True
    for srv, params in todo.items():
        r = call(node, SetParameters, f"{srv}/set_parameters", SetParameters.Request(parameters=params))
        if r is None:
            print(f"{srv}: not reachable (is Mapping/Navigation running?)")
            ok = False
            continue
        for p, res in zip(params, r.results):
            if not res.successful:
                print(f"{srv} {p.name}: FAILED {res.reason}")
                ok = False
        print(f"{srv}: {len(params)} settings {'OK' if ok else 'with failures'}")
    if ok:
        print("restarting Nav2 so the start-up-only settings apply (map is kept) ...")
        for cmd, label in ((ManageLifecycleNodes.Request.RESET, "reset"),
                           (ManageLifecycleNodes.Request.STARTUP, "startup")):
            r = call(node, ManageLifecycleNodes, "lifecycle_manager_navigation/manage_nodes",
                     ManageLifecycleNodes.Request(command=cmd), timeout=60.0)
            print(f"  Nav2 {label}: {'OK' if r and r.success else 'FAILED'}")
            ok &= bool(r and r.success)
            time.sleep(1.0)
    planning = "stored SLAM map" if "--static-map" in args else "live lidar"
    print(f"Nav2: {v} m/s, {w} rad/s, forward/reverse/sideways, loose clearance, planning on {planning}" if ok
          else "NOT all settings applied: don't send goals yet (ant1_nav_speed --static-map restores Olive's planning)")
    node.destroy_node()
    rclpy.try_shutdown()
