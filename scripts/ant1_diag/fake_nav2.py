# Fake Nav2 for testing ant1_goto: TF map->base_link, goal_pose subscriber, action status + cancel service.
import sys, uuid, time, math, rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from geometry_msgs.msg import PoseStamped, TransformStamped, Twist
from action_msgs.msg import GoalStatus, GoalStatusArray
from action_msgs.srv import CancelGoal
from tf2_ros import TransformBroadcaster
from rcl_interfaces.msg import Log
from nav_msgs.msg import OccupancyGrid
from nav2_msgs.srv import ManageLifecycleNodes
from nav_msgs.msg import Odometry
from rcl_interfaces.srv import SetParameters
from rcl_interfaces.msg import SetParametersResult
F = "olive_olixRobot_ant1_id001_"
# drive: a moving ANT for the mission tests: 0.2 m/s toward the latest goal, turns to face its travel
# direction, odom + yaw in TF, stops on cancel, a newer goal preempts the old one.
mode = sys.argv[1]   # succeed | hang | dropfirst | abortfirst | latestatus | liesucceed | orphan | orphanpause | drive  (namespace /test, use an isolated ROS_DOMAIN_ID)
class Fake(Node):
    def __init__(self):
        super().__init__("fake_nav2", namespace="/test")
        q = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.st = self.create_publisher(GoalStatusArray, "navigate_to_pose/_action/status", q)
        self.create_subscription(PoseStamped, "goal_pose", self.on_goal, 10)
        self.create_subscription(Twist, "cmd_vel", lambda m: m.linear.x == 0 and print(f"FAKE got STOP x={m.linear.x} wz={m.angular.z}"), 10)
        self.create_service(CancelGoal, "navigate_to_pose/_action/cancel_goal", self.on_cancel)
        # orphan: navigate_to_pose aborts but the controller (follow_path) keeps driving until IT is cancelled
        # (real robot, 10 Oct). orphanpause: only a lifecycle PAUSE stops it.
        self.orphan = False
        self.cmd = self.create_publisher(Twist, "cmd_vel", 10)
        self.create_service(CancelGoal, "follow_path/_action/cancel_goal", self.on_cancel_follow)
        self.create_service(ManageLifecycleNodes, "lifecycle_manager_navigation/manage_nodes", self.on_lifecycle)
        self.log = self.create_publisher(Log, "/rosout", 10)
        cm = self.create_publisher(OccupancyGrid, "global_costmap/costmap", q)
        g = OccupancyGrid(); g.info.resolution = 0.05; g.info.width = g.info.height = 100   # 5x5 m from (-1,-1)
        g.info.origin.position.x = g.info.origin.position.y = -1.0
        d = [0] * 10000
        for cy in range(100):
            for cx in range(100):
                x, y = -1 + cx * 0.05, -1 + cy * 0.05
                if 2.5 <= x <= 3.0 and 0.0 <= y <= 2.0: d[cy * 100 + cx] = 100      # wall block
                elif 2.3 <= x <= 3.2 and -0.2 <= y <= 2.2: d[cy * 100 + cx] = 99    # robot can't fit
                elif 0.4 <= x <= 1.6 and 1.4 <= y <= 2.6 and not (0.7 <= x <= 1.3 and 1.7 <= y <= 2.3): d[cy * 100 + cx] = 100   # closed pocket
        g.data = d; cm.publish(g); self.cm = g
        self.dropped = False
        self.yaw, self.v = 0.0, 0.0
        self.odom = self.create_publisher(Odometry, "odom", 10)
        # drive mode: live limits like the real Nav2 (ant_link /ant/nav_params, ant1_nav_speed --live)
        self.vmax, self.tol = 0.2, 0.05
        def setp(req, res):
            for p in req.parameters:
                if p.name == "FollowPath.max_vel_x":
                    self.vmax = p.value.double_value; print(f"FAKE speed limit {self.vmax} m/s", flush=True)
                elif p.name == "general_goal_checker.xy_goal_tolerance":
                    self.tol = p.value.double_value; print(f"FAKE goal tolerance {self.tol} m", flush=True)
            res.results = [SetParametersResult(successful=True) for _ in req.parameters]
            return res
        for srv in ("controller_server", "velocity_smoother"):
            self.create_service(SetParameters, f"{srv}/set_parameters", setp)
        self.tfb = TransformBroadcaster(self); self.x = 0.0; self.y = 0.0; self.gxy = (1.0, 0.0); self.goal = None; self.gyaw = None
        old = GoalStatus(); old.goal_info.goal_id.uuid = list(uuid.uuid4().bytes); old.status = GoalStatus.STATUS_SUCCEEDED
        self.statuses = [old]
        if mode != 'latestatus': self.st.publish(GoalStatusArray(status_list=self.statuses))
        self.create_timer(0.1, self.tick)
    def on_goal(self, g):
        if mode == 'latestatus' and not getattr(self, 'late_done', False):
            self.late_done = True; self.st.publish(GoalStatusArray(status_list=self.statuses)); time.sleep(1.0)
        if mode == 'dropfirst' and not self.dropped:
            self.dropped = True; print('FAKE dropped first goal'); return
        q = g.pose.orientation
        print(f"FAKE goal: frame={g.header.frame_id} stamp={g.header.stamp.sec}.{g.header.stamp.nanosec} "
              f"x={g.pose.position.x} y={g.pose.position.y} yaw_deg={math.degrees(2*math.atan2(q.z, q.w)):.1f}")
        self.gxy = (g.pose.position.x, g.pose.position.y)
        q = g.pose.orientation; self.gyaw = math.atan2(2 * q.w * q.z, 1 - 2 * q.z * q.z) if (q.z or q.w) else None
        s = GoalStatus(); s.goal_info.goal_id.uuid = list(uuid.uuid4().bytes); s.status = GoalStatus.STATUS_EXECUTING
        for _ in range(2): self.log.publish(Log(level=30, name="olive.olixRobot.ant1.id001.controller_server", msg="Failed to make progress"))
        self.log.publish(Log(level=20, name="olive.olixRobot.ant1.id001.controller_server", msg="info noise"))
        if mode == 'drive' and self.goal and self.goal[0].status == GoalStatus.STATUS_EXECUTING: self.goal[0].status = GoalStatus.STATUS_CANCELED   # preempted
        self.statuses.append(s); self.goal = (s, time.monotonic())
        self.ngoals = getattr(self, 'ngoals', 0) + 1; self.st.publish(GoalStatusArray(status_list=self.statuses))
    def on_cancel(self, req, res):
        print(f"FAKE cancel called (all goals: {not any(req.goal_info.goal_id.uuid)})")
        if self.goal: self.goal[0].status = GoalStatus.STATUS_CANCELED; self.st.publish(GoalStatusArray(status_list=self.statuses))
        return res
    def on_cancel_follow(self, req, res):
        print("FAKE follow_path cancel called"); self.orphan = self.orphan and mode == "orphanpause"; return res
    def on_lifecycle(self, req, res):
        print(f"FAKE lifecycle command {req.command}"); self.orphan = False; res.success = True; return res
    def tick(self):
        if self.goal and mode.startswith("orphan") and self.goal[0].status == GoalStatus.STATUS_EXECUTING and time.monotonic() - self.goal[1] > 2:
            self.goal[0].status = GoalStatus.STATUS_ABORTED; self.st.publish(GoalStatusArray(status_list=self.statuses)); self.orphan = True; print("FAKE aborted, controller keeps driving")
        if self.orphan:
            m = Twist(); m.linear.x = 0.2; self.cmd.publish(m)
        if self.goal and mode == "liesucceed" and self.goal[0].status == GoalStatus.STATUS_EXECUTING and time.monotonic() - self.goal[1] > 1:
            self.goal[0].status = GoalStatus.STATUS_SUCCEEDED; self.st.publish(GoalStatusArray(status_list=self.statuses))
        if self.goal and mode == "abortfirst" and self.ngoals == 1 and time.monotonic() - self.goal[1] > 2 and self.goal[0].status == GoalStatus.STATUS_EXECUTING:
            self.goal[0].status = GoalStatus.STATUS_ABORTED; self.st.publish(GoalStatusArray(status_list=self.statuses)); print("FAKE aborted goal 1")
        if self.goal and (mode in ("succeed", "dropfirst", "latestatus") or (mode == "abortfirst" and self.ngoals > 1)):
            dx, dy = self.gxy[0] - self.x, self.gxy[1] - self.y; d = math.hypot(dx, dy); st = min(d, 0.15)
            if d > 1e-6: self.x += dx / d * st; self.y += dy / d * st
            if time.monotonic() - self.goal[1] > 3 and self.goal[0].status == GoalStatus.STATUS_EXECUTING:
                self.goal[0].status = GoalStatus.STATUS_SUCCEEDED; self.st.publish(GoalStatusArray(status_list=self.statuses))
        self.v = 0.0
        if mode == "drive" and self.goal and self.goal[0].status == GoalStatus.STATUS_EXECUTING:
            dx, dy = self.gxy[0] - self.x, self.gxy[1] - self.y; d = math.hypot(dx, dy)
            ye = math.atan2(math.sin((self.gyaw or self.yaw) - self.yaw), math.cos((self.gyaw or self.yaw) - self.yaw))
            if d < max(self.tol, 0.03) and abs(ye) > 0.3:      # like Nav2: at the spot, turn to the goal heading
                self.yaw += max(min(ye, 0.06), -0.06); self.v = 0.06
            elif d < max(self.tol, 0.03):
                self.goal[0].status = GoalStatus.STATUS_SUCCEEDED; self.st.publish(GoalStatusArray(status_list=self.statuses)); print(f"FAKE ANT arrived ({self.x:.2f},{self.y:.2f})", flush=True)
            else:
                st = min(d, self.vmax * 0.1); self.x += dx / d * st; self.y += dy / d * st; self.v = st / 0.1
                e = math.atan2(math.sin(math.atan2(dy, dx) - self.yaw), math.cos(math.atan2(dy, dx) - self.yaw)); self.yaw += max(min(e, 0.1), -0.1)
        o = Odometry(); o.twist.twist.linear.x = self.v; self.odom.publish(o)
        t = TransformStamped(); t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id, t.child_frame_id = F + "map", F + "base_link"; t.transform.translation.x = self.x; t.transform.translation.y = self.y
        t.transform.rotation.z, t.transform.rotation.w = math.sin(self.yaw / 2), math.cos(self.yaw / 2)
        self.tfb.sendTransform(t)
rclpy.init(); rclpy.spin(Fake())
