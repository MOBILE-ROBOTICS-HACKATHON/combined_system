"""ant_link: the bridge between the ANT1 world (domain 0, Wi-Fi) and the drone world (domain 42, localhost).

    mission_link                                (helper from scripts/mission_env.sh; needs THAT env)
    = ros2 run team_pkg ant_link     env: ANT1_DOMAIN (0), CF_DOMAIN (42), NS (ANT namespace),
                                          MISSION_CALIB (data/mission_calib.yaml)

ONE process, TWO ROS contexts (rclpy.init(domain_id=...)); CYCLONEDDS_URI has one <Domain> per id
(scripts/mission_env.sh). The drone stack never touches Wi-Fi: only this process does.

Publishes in the DRONE domain (42):
  /ant/pose_map    PoseStamped (frame ant_map)  ANT pose in its SLAM map            10 Hz
  /ant/pose_room   PoseStamped (frame world)    ANT pose in the drone frame, z = plate_z (needs calibration)
  /ant/moving      Bool                         odometry speed > 2 cm/s (or turning)
  /ant/nav_state   String                       idle | sent | executing | succeeded | aborted | canceled | stopped
  /ant/marker      Marker                       the ANT's 60x40 cm body for rviz (frame world)
  /mission/calib   String (latched)             the calibration (YAML), reloaded when the file changes
  TF world -> ant_map (static)                  so rviz shows both robots
Accepts in the DRONE domain:
  /ant/goal        PoseStamped  frame "world" (drone frame) or "ant_map". Orientation all zero = keep heading.
                   -> reachability check on the ANT costmap (closest reachable point if blocked)
                   -> goal_pose on the ANT (stamp 0), re-sent up to 3x if Nav2 doesn't confirm in 5 s
  /ant/nav_params  String       LIVE Nav2 limits, no restart: "speed=0.5 turn=0.6 tol=0.1" (rescue boost) or
                                "reset" (mission values 0.1 m/s, 0.3 rad/s, tol 0.3). Speed capped at 0.5 m/s.
  /ant/stop        Empty        cancel EVERY Nav2 action + zero velocity; if the ANT still gets
                                driving commands 1-2.5 s later: PAUSE Nav2 (ant1_nav_speed brings it back)
Keys: s = stop the ANT now, q = quit (stops the ANT first: never leave a goal running behind us).
"""
import math
import os
import select
import signal
import sys
import termios
import time
import tty
import rclpy
from rclpy.context import Context
from rclpy.executors import SingleThreadedExecutor
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
from rclpy.signals import SignalHandlerOptions
from rclpy.time import Time
from action_msgs.msg import GoalStatus, GoalStatusArray
from action_msgs.srv import CancelGoal
from geometry_msgs.msg import PoseStamped, TransformStamped, Twist
from nav_msgs.msg import OccupancyGrid, Odometry
from nav2_msgs.srv import ManageLifecycleNodes
from std_msgs.msg import Bool, Empty, String
from tf2_ros import Buffer, StaticTransformBroadcaster, TransformListener
from visualization_msgs.msg import Marker
from rcl_interfaces.srv import SetParameters
from team_pkg.ant1_nav import ACTIONS, FRAME, free_goal, yaw_of
from team_pkg.ant1_nav_speed import V_MAX, W_MAX, param
from team_pkg.mission_common import CALIB_PATH, LATCHED, Calib, load_calib

STATE = {GoalStatus.STATUS_ACCEPTED: "executing", GoalStatus.STATUS_EXECUTING: "executing",
         GoalStatus.STATUS_CANCELING: "canceling", GoalStatus.STATUS_SUCCEEDED: "succeeded",
         GoalStatus.STATUS_CANCELED: "canceled", GoalStatus.STATUS_ABORTED: "aborted"}
CONFIRM = 5.0   # s to wait for Nav2 to accept a goal before re-sending (loaded Nav2 took 3.6 s, 10 Oct)


def quat(yaw):
    return 0.0, 0.0, math.sin(yaw / 2), math.cos(yaw / 2)


class AntSide:
    """Node in the ANT domain: TF, odometry, costmap, Nav2 goals and cancels."""

    def __init__(self, ctx, ns):
        n = self.node = rclpy.create_node("ant_link", namespace=ns, context=ctx)
        self.tf = Buffer()
        self.tf_listener = TransformListener(self.tf, n)
        latched = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.goal_pub = n.create_publisher(PoseStamped, "goal_pose", 10)
        self.cmd_pub = n.create_publisher(Twist, "cmd_vel", 10)
        self.cancel_clis = [n.create_client(CancelGoal, f"{a}/_action/cancel_goal") for a in ACTIONS]
        self.lifecycle_cli = n.create_client(ManageLifecycleNodes, "lifecycle_manager_navigation/manage_nodes")
        self.param_clis = {k: n.create_client(SetParameters, f"{k}/set_parameters")
                           for k in ("controller_server", "velocity_smoother")}
        n.create_subscription(GoalStatusArray, "navigate_to_pose/_action/status", self.on_status, latched)
        n.create_subscription(OccupancyGrid, "global_costmap/costmap", lambda m: setattr(self, "costmap", m), latched)
        n.create_subscription(Odometry, "odom", self.on_odom, qos_profile_sensor_data)
        n.create_subscription(Twist, "cmd_vel", self.on_cmd, 10)
        self.costmap, self.speed, self.odom_at, self.driven_at = None, 0.0, 0.0, 0.0
        self.statuses = {}          # goal id -> status
        self.pose_stamp, self.pose_at, self.pose = None, 0.0, None

    def on_status(self, m):
        for s in m.status_list:
            self.statuses[bytes(s.goal_info.goal_id.uuid)] = s.status

    def on_odom(self, m):
        t = m.twist.twist
        self.speed = max(math.hypot(t.linear.x, t.linear.y), abs(t.angular.z) * 0.3)   # turning counts too
        self.odom_at = time.monotonic()

    def on_cmd(self, m):
        if max(abs(m.linear.x), abs(m.linear.y), abs(m.angular.z)) > 0.01:
            self.driven_at = time.monotonic()

    def read_pose(self):
        """(x, y, yaw) in the map; `pose_at` = when the transform last CHANGED (robot clock is off, so
        its stamps can't be compared with ours)."""
        try:
            t = self.tf.lookup_transform(FRAME + "map", FRAME + "base_link", Time())
        except Exception:
            return self.pose
        st = (t.header.stamp.sec, t.header.stamp.nanosec)
        if st != self.pose_stamp:
            self.pose_stamp, self.pose_at = st, time.monotonic()
            r = t.transform
            self.pose = (r.translation.x, r.translation.y, yaw_of(r.rotation))
        return self.pose

    def send_goal(self, x, y, yaw):
        g = PoseStamped()
        g.header.frame_id = FRAME + "map"          # stamp 0 = latest transform (robot clock is behind ours)
        g.pose.position.x, g.pose.position.y = x, y
        _, _, g.pose.orientation.z, g.pose.orientation.w = quat(yaw)
        self.goal_pub.publish(g)

    def cancel_all(self):
        missing = []
        for c in self.cancel_clis:
            if c.service_is_ready():
                c.call_async(CancelGoal.Request())   # empty goal_info = cancel ALL goals of that action
            else:
                missing.append(c.srv_name.split("/")[-3])
        for _ in range(5):
            self.cmd_pub.publish(Twist())
        return missing

    def set_limits(self, v, w, tol):
        """Live Nav2 limits (the parameters ant1_nav_speed --live sets, plus the goal tolerance).
        Returns the names of servers that weren't reachable."""
        req = {
            "controller_server": [param("FollowPath.max_vel_x", v), param("FollowPath.min_vel_x", -v),
                                  param("FollowPath.max_vel_y", v), param("FollowPath.min_vel_y", -v),
                                  param("FollowPath.max_speed_xy", v), param("FollowPath.max_vel_theta", w),
                                  param("FollowPath.xy_goal_tolerance", tol),
                                  param("general_goal_checker.xy_goal_tolerance", tol)],
            "velocity_smoother": [param("max_velocity", [v, v, w]), param("min_velocity", [-v, -v, -w])],
        }
        missing = []
        for k, params in req.items():
            if self.param_clis[k].service_is_ready():
                self.param_clis[k].call_async(SetParameters.Request(parameters=params))
            else:
                missing.append(k)
        return missing

    def pause_nav2(self):
        if self.lifecycle_cli.service_is_ready():
            self.lifecycle_cli.call_async(ManageLifecycleNodes.Request(command=ManageLifecycleNodes.Request.PAUSE))
            return True
        return False


class Link:
    def __init__(self, ant_ctx, cf_ctx, ns):
        self.ant = AntSide(ant_ctx, ns)
        n = self.node = rclpy.create_node("ant_link", context=cf_ctx)
        self.log = n.get_logger()
        self.pub_map = n.create_publisher(PoseStamped, "/ant/pose_map", 10)
        self.pub_room = n.create_publisher(PoseStamped, "/ant/pose_room", 10)
        self.pub_moving = n.create_publisher(Bool, "/ant/moving", 10)
        self.pub_state = n.create_publisher(String, "/ant/nav_state", LATCHED)
        self.pub_marker = n.create_publisher(Marker, "/ant/marker", 10)
        self.pub_calib = n.create_publisher(String, "/mission/calib", LATCHED)
        self.static_tf = StaticTransformBroadcaster(n)
        n.create_subscription(PoseStamped, "/ant/goal", self.on_goal, 10)
        n.create_subscription(Empty, "/ant/stop", lambda _: self.stop("/ant/stop received"), 10)
        n.create_subscription(String, "/ant/nav_params", self.on_nav_params, 10)
        self.limits = None                       # (v, w, tol) we set last, shown in the status line
        self.calib, self.calib_mtime = None, None
        self.goal = None            # (x, y, yaw) in map we are trying to get accepted
        self.known, self.mine, self.sent_at, self.tries = set(), None, 0.0, 0
        self.state, self.stop_at, self.paused = "idle", None, False
        self.note = ""
        self.reload_calib()

    # ---------- calibration ----------
    def reload_calib(self):
        try:
            mt = os.path.getmtime(CALIB_PATH)
        except OSError:
            mt = None
        if mt == self.calib_mtime:
            return
        self.calib_mtime, self.calib = mt, load_calib()
        if self.calib is None:
            self.log.warn(f"no calibration ({CALIB_PATH}): /ant/pose_room and world-frame goals are OFF. "
                          "Run mission_calib with the drone sitting on the ANT")
            self.pub_calib.publish(String(data=""))
            return
        self.log.info(f"calibration loaded: {self.calib.summary()}")
        self.pub_calib.publish(String(data=self.calib.to_yaml()))
        t = TransformStamped()                       # world -> ant_map, for rviz
        t.header.stamp = self.node.get_clock().now().to_msg()
        t.header.frame_id, t.child_frame_id = "world", "ant_map"
        t.transform.translation.x, t.transform.translation.y = self.calib.map_to_room(0.0, 0.0)
        r = t.transform.rotation
        r.x, r.y, r.z, r.w = quat(-self.calib.theta)
        self.static_tf.sendTransform(t)

    # ---------- goals and stop ----------
    def set_state(self, s, note=""):
        if s != self.state or note:
            self.state = s
            self.pub_state.publish(String(data=s))
            if note:
                self.note = note

    def on_goal(self, m):
        x, y = m.pose.position.x, m.pose.position.y
        q = m.pose.orientation
        keep = q.x == q.y == q.z == q.w == 0.0
        if m.header.frame_id == "world":
            if self.calib is None:
                self.log.error("goal in frame 'world' REFUSED: no calibration")
                return
            x, y = self.calib.room_to_map(x, y)
            yaw = None if keep else yaw_of(q) + self.calib.theta
        elif m.header.frame_id in ("ant_map", FRAME + "map"):
            yaw = None if keep else yaw_of(q)
        else:
            self.log.error(f"goal REFUSED: unknown frame '{m.header.frame_id}' (use 'world' or 'ant_map')")
            return
        here = self.ant.read_pose()
        if here is None:
            self.log.error("goal REFUSED: no ANT pose (TF map -> base_link) yet")
            return
        r = free_goal(self.ant.costmap, x, y, here)
        if r is None:
            self.log.error("goal REFUSED: the ANT can't reach anything from where it stands")
            return
        gx, gy, why = r
        if why:
            self.log.warn(why)
        self.goal = (gx, gy, here[2] if yaw is None else yaw)
        self.stop_at, self.tries = None, 0
        self.send()

    def send(self):
        self.known = set(self.ant.statuses)          # anything already in the list is not ours
        self.mine, self.sent_at = None, time.monotonic()
        self.tries += 1
        self.ant.send_goal(*self.goal)
        self.set_state("sent", f"goal ({self.goal[0]:+.2f}, {self.goal[1]:+.2f}) map, try {self.tries}")

    def on_nav_params(self, m):
        """"speed=0.5 turn=0.6 tol=0.1" or "reset" -> live Nav2 limits (no restart, safe mid-goal)."""
        text = m.data.strip()
        v, w, tol = 0.1, 0.3, 0.3                         # the mission values (ant1_nav_speed 0.1 0.3 --live)
        if text != "reset":
            kv = dict(p.split("=", 1) for p in text.split() if "=" in p)
            try:
                v = float(kv.get("speed", v))
                w = float(kv.get("turn", 0.6 if v >= 0.3 else w))
                tol = float(kv.get("tol", tol))
            except ValueError:
                self.log.error(f"/ant/nav_params: can't read '{text}' (use 'speed=0.5 turn=0.6 tol=0.1' or 'reset')")
                return
        v, w, tol = min(max(v, 0.05), V_MAX), min(max(w, 0.1), W_MAX), min(max(tol, 0.05), 0.5)
        missing = self.ant.set_limits(v, w, tol)
        self.limits = (v, w, tol)
        extra = f" (NOT reachable: {', '.join(missing)})" if missing else ""
        self.log.warn(f"Nav2 limits {'RESET to mission values' if text == 'reset' else 'set'}: "
                      f"{v} m/s, {w} rad/s, goal tolerance {tol} m{extra}")

    def stop(self, why):
        self.goal = None
        missing = self.ant.cancel_all()
        self.stop_at, self.paused = time.monotonic(), False
        extra = f" (not reachable: {', '.join(missing)})" if missing else ""
        self.log.warn(f"STOP ANT: {why}: cancelled all Nav2 actions + zero velocity{extra}")
        self.set_state("stopped", why)

    def track(self):
        """~20 Hz: goal confirmation / result, and the 'still driving after stop' check."""
        now = time.monotonic()
        if self.goal:
            if self.mine is None:
                for gid, st in self.ant.statuses.items():
                    if gid not in self.known and st in (GoalStatus.STATUS_ACCEPTED, GoalStatus.STATUS_EXECUTING):
                        self.mine = gid
                if self.mine is None and now - self.sent_at > CONFIRM:
                    if self.tries >= 3:
                        self.log.error("Nav2 never confirmed the goal (3 tries): is Navigation running on the ANT?")
                        self.goal = None
                        self.set_state("aborted", "never confirmed")
                    else:
                        self.log.warn("goal not confirmed by Nav2: re-sending")
                        self.send()
            if self.mine is not None:
                s = STATE.get(self.ant.statuses.get(self.mine), "executing")
                self.set_state(s)
                if s in ("succeeded", "aborted", "canceled"):
                    self.log.info(f"ANT goal {s}")
                    self.goal = None
        if self.stop_at and 1.0 < now - self.stop_at < 2.5 and self.ant.driven_at > self.stop_at + 1.0 and not self.paused:
            self.paused = True
            ok = self.ant.pause_nav2()
            self.log.error("the ANT is STILL being driven after the stop: "
                           + ("Nav2 PAUSED (run ant1_nav_speed with the ANT still to bring it back)" if ok
                              else "can't reach Nav2's lifecycle manager: USE THE E-STOP"))

    # ---------- publishing ----------
    def publish(self):
        p = self.ant.read_pose()
        now = self.node.get_clock().now().to_msg()
        if p is None or time.monotonic() - self.ant.pose_at > 1.0:
            return
        x, y, yaw = p
        moving = self.ant.speed > 0.02 and time.monotonic() - self.ant.odom_at < 1.0
        self.pub_moving.publish(Bool(data=moving))
        m = PoseStamped()
        m.header.stamp, m.header.frame_id = now, "ant_map"
        m.pose.position.x, m.pose.position.y = x, y
        o = m.pose.orientation
        o.x, o.y, o.z, o.w = quat(yaw)
        self.pub_map.publish(m)
        if self.calib is None:
            return
        r = PoseStamped()
        r.header.stamp, r.header.frame_id = now, "world"
        r.pose.position.x, r.pose.position.y = self.calib.map_to_room(x, y)
        r.pose.position.z = self.calib.plate_z
        o = r.pose.orientation
        o.x, o.y, o.z, o.w = quat(yaw - self.calib.theta)
        self.pub_room.publish(r)
        k = Marker(type=Marker.CUBE, action=Marker.ADD, ns="ant", id=0)
        k.header = r.header
        k.pose = r.pose
        k.pose.position.z = (self.calib.plate_z + self.calib.floor_z) / 2
        k.scale.x, k.scale.y, k.scale.z = 0.6, 0.4, max(self.calib.plate_z - self.calib.floor_z, 0.05)
        k.color.r, k.color.g, k.color.b, k.color.a = 0.9, 0.6, 0.1, 0.7
        self.pub_marker.publish(k)

    def status(self):
        a, now = self.ant, time.monotonic()
        if a.pose is None:
            return "ANT: no pose yet (Mapping/Navigation running? ANT1_IFACE/ANT1_PEER right in mission_env?)"
        age = now - a.pose_at
        s = f"ANT {'OK' if age < 1.0 else 'STALE'} ({age:.1f} s) map ({a.pose[0]:+.2f}, {a.pose[1]:+.2f}, {math.degrees(a.pose[2]):+.0f} deg)"
        if self.calib:
            rx, ry = self.calib.map_to_room(a.pose[0], a.pose[1])
            s += f" room ({rx:+.2f}, {ry:+.2f})"
        else:
            s += " | NO CALIB"
        s += f" | {a.speed:.2f} m/s"
        if self.limits:
            s += f" (max {self.limits[0]}, tol {self.limits[2]})"
        s += f" | nav {self.state}"
        if self.note:
            s += f": {self.note}"
        return s


def read_key():
    if sys.stdin.isatty() and select.select([sys.stdin], [], [], 0)[0]:
        return os.read(sys.stdin.fileno(), 1).decode(errors="ignore")
    return None


def _interrupt(*_):
    raise KeyboardInterrupt


def main():
    signal.signal(signal.SIGHUP, _interrupt)
    signal.signal(signal.SIGTERM, _interrupt)
    ant_dom, cf_dom = int(os.environ.get("ANT1_DOMAIN", "0")), int(os.environ.get("CF_DOMAIN", "42"))
    ns = os.environ.get("NS", "/olive/olixRobot/ant1/id001")
    if os.environ.get("ROS_LOCALHOST_ONLY") == "1":
        print("ROS_LOCALHOST_ONLY=1 would cut the ANT off: source scripts/mission_env.sh (not cf_env.sh)")
        return
    ant_ctx, cf_ctx = Context(), Context()
    rclpy.init(context=ant_ctx, domain_id=ant_dom, signal_handler_options=SignalHandlerOptions.NO)
    rclpy.init(context=cf_ctx, domain_id=cf_dom, signal_handler_options=SignalHandlerOptions.NO)
    link = Link(ant_ctx, cf_ctx, ns)
    ex_ant, ex_cf = SingleThreadedExecutor(context=ant_ctx), SingleThreadedExecutor(context=cf_ctx)
    ex_ant.add_node(link.ant.node)
    ex_cf.add_node(link.node)
    print(__doc__.split("\n\n")[3])
    print(f"ANT domain {ant_dom} ns {ns}  <->  drone domain {cf_dom}")
    old = termios.tcgetattr(sys.stdin) if sys.stdin.isatty() else None
    last_pub = last_print = last_calib = 0.0
    try:
        if old:
            tty.setcbreak(sys.stdin.fileno())
        while rclpy.ok(context=cf_ctx):
            ex_ant.spin_once(timeout_sec=0.02)
            ex_cf.spin_once(timeout_sec=0.02)
            now = time.monotonic()
            link.track()
            if now - last_pub > 0.1:
                link.publish()
                last_pub = now
            if now - last_calib > 1.0:
                link.reload_calib()
                last_calib = now
            if now - last_print > 0.5:
                print(f"\r{link.status()[:150]:<150}", end="", flush=True)
                last_print = now
            k = read_key()
            if k == "s":
                print()
                link.stop("s pressed")
            elif k == "q":
                break
    except KeyboardInterrupt:
        pass
    finally:
        if old:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old)
        print()
        link.stop("ant_link closing")            # never leave a goal running behind us
        end = time.monotonic() + 1.0
        while time.monotonic() < end:
            ex_ant.spin_once(timeout_sec=0.05)
        link.node.get_logger().info("ant_link stopped")
        rclpy.try_shutdown(context=ant_ctx)
        rclpy.try_shutdown(context=cf_ctx)
