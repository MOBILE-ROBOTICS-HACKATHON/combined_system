"""Drive the ANT1 to a map coordinate with its ONBOARD Nav2 (path planning + obstacle avoidance).
Needs Mapping or Navigation mode running (Web UI → SLAM). Run ant1_watchdog and ant1_nav_speed first.

    ant1_goto 1.0 0.5          # x y in metres (map frame); keeps its current heading (no extra turning)
    ant1_goto 1.0 0.5 90       # ... and ends facing 90° (= +y)
    ant1_goto 1.0 0.5 face     # ... and ends facing the direction it travelled
    = ros2 run team_pkg ant1_goto 1.0 0.5 --ros-args -r __ns:=$NS [-p retries:=3]

Before sending, it checks on the robot's costmap whether the robot can REACH the goal (enough room for
its 60x40 cm body all the way). If not (goal inside/at an object, in a closed-off pocket, or outside
the map) it drives to the closest reachable point instead and says so: "go to the box" works, and
no more spin-and-give-up at unreachable goals. If Nav2 still gives up, it retries `retries` times.

Map frame: origin = where the robot stood when Mapping started, +x = the way it faced, +y = its left.
Sends the goal on goal_pose (plain PoseStamped: works Humble -> Jazzy, unlike the Nav2 action types),
stamp 0 (= "latest": the robot clock is behind ours). Reads the result from the action status topic
(action_msgs, same in both distros). EVERY exit after a goal was sent (arrived, gave up, timeout, Ctrl-C,
crash) cancels all Nav2 goals incl. the controller's follow_path, sends zeros, and PAUSES Nav2 if the robot
is still being driven after that (10 Oct: Nav2 aborted navigate_to_pose but its follow_path kept driving;
e-stop needed). After a pause, ant1_nav_speed brings Nav2 back.
"""
import math
import signal
import sys
import time
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from rclpy.signals import SignalHandlerOptions
from rclpy.time import Time
from action_msgs.msg import GoalStatus, GoalStatusArray
from action_msgs.srv import CancelGoal
from geometry_msgs.msg import PoseStamped, Twist
from nav_msgs.msg import OccupancyGrid
from nav2_msgs.srv import ManageLifecycleNodes
from rcl_interfaces.msg import Log
from tf2_ros import Buffer, TransformListener
from team_pkg.ant1_nav import ACTIONS, FRAME, free_goal

NAV2 = ("controller_server", "planner_server", "bt_navigator", "behavior_server", "velocity_smoother")
DONE = {GoalStatus.STATUS_SUCCEEDED: "ARRIVED", GoalStatus.STATUS_CANCELED: "CANCELED",
        GoalStatus.STATUS_ABORTED: "ABORTED (Nav2 gave up: blocked or unreachable?)"}


class Ant1Goto(Node):
    def __init__(self):
        super().__init__("ant1_goto")
        self.map_frame = self.declare_parameter("map_frame", FRAME + "map").value
        self.base_frame = self.declare_parameter("base_frame", FRAME + "base_link").value
        self.timeout = self.declare_parameter("timeout", 180.0).value
        self.retries = self.declare_parameter("retries", 3).value
        self.tf = Buffer()
        self.tf_listener = TransformListener(self.tf, self)
        self.goal_pub = self.create_publisher(PoseStamped, "goal_pose", 10)
        self.stop_pub = self.create_publisher(Twist, "cmd_vel", 10)
        latched = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                             durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.create_subscription(GoalStatusArray, "navigate_to_pose/_action/status", self.on_status, latched)
        self.cancel_clis = [self.create_client(CancelGoal, f"{a}/_action/cancel_goal") for a in ACTIONS]
        self.lifecycle_cli = self.create_client(ManageLifecycleNodes, "lifecycle_manager_navigation/manage_nodes")
        self.driven_at = 0.0   # last time anybody commanded non-zero velocity
        self.create_subscription(Twist, "cmd_vel", self.on_cmd, 10)
        self.create_subscription(Log, "/rosout", self.on_log, 100)   # show WHY Nav2 struggles
        self.last_log = ""
        self.costmap = None
        self.create_subscription(OccupancyGrid, "global_costmap/costmap",
                                 lambda m: setattr(self, "costmap", m), latched)
        self.known = set()     # goal ids that existed before ours
        self.mine = None       # our goal id
        self.result = None
        self.sent = False
        self.got_status = False

    def on_status(self, msg):
        self.got_status = True
        for s in msg.status_list:
            gid = bytes(s.goal_info.goal_id.uuid)
            if not self.sent:
                self.known.add(gid)
            # A new id is ours only if it is still RUNNING when first seen. Over Wi-Fi the list of old,
            # finished goals can arrive after we sent ours: taking one of those printed "ARRIVED" while
            # the robot was still driving (10 Oct). A finished id we never saw running is never ours.
            elif self.mine is None and gid not in self.known:
                if s.status in (GoalStatus.STATUS_ACCEPTED, GoalStatus.STATUS_EXECUTING):
                    self.mine = gid
                else:
                    self.known.add(gid)
            if gid == self.mine and s.status in DONE:
                self.result = s.status

    def on_cmd(self, m):
        if max(abs(m.linear.x), abs(m.linear.y), abs(m.angular.z)) > 0.01:
            self.driven_at = time.monotonic()

    def on_log(self, msg):
        """Relay Nav2 warnings/errors from the robot while our goal runs (deduplicated)."""
        # 30 = WARN (Log.WARN is a bytes constant in rclpy, so compare with the number)
        if self.sent and msg.level >= 30 and any(k in msg.name for k in NAV2) and msg.msg != self.last_log:
            self.last_log = msg.msg
            self.get_logger().warn(f"[nav2 {msg.name.split('.')[-1]}] {msg.msg}")

    def pose(self):
        """(x, y, yaw) of the robot in the map frame, or None if TF isn't available yet."""
        try:
            t = self.tf.lookup_transform(self.map_frame, self.base_frame, Time()).transform
        except Exception:
            return None
        q = t.rotation
        return t.translation.x, t.translation.y, math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))

    def free_goal(self, x, y, robot):
        return free_goal(self.costmap, x, y, robot)

    def send(self, x, y, yaw):
        g = PoseStamped()
        g.header.frame_id = self.map_frame          # stamp stays 0 = use latest transform
        g.pose.position.x, g.pose.position.y = x, y
        g.pose.orientation.z, g.pose.orientation.w = math.sin(yaw / 2), math.cos(yaw / 2)
        self.goal_pub.publish(g)
        self.sent = True

    def zeros(self):
        for _ in range(5):
            self.stop_pub.publish(Twist())
            time.sleep(0.05)

    def cancel_and_stop(self):
        """Cancel ALL goals of every Nav2 action, send zeros, then check nobody still drives the robot.
        navigate_to_pose alone isn't enough: under load Nav2 aborted it but left follow_path running (10 Oct)."""
        log = self.get_logger()
        for c in self.cancel_clis:                 # empty goal_info = cancel all goals
            if c.wait_for_service(timeout_sec=1.0):
                rclpy.spin_until_future_complete(self, c.call_async(CancelGoal.Request()), timeout_sec=2.0)
            else:
                log.warn(f"{c.srv_name} not reachable")
        self.zeros()
        spin_for(self, 1.0)                        # velocity smoother ramps down
        check = time.monotonic()
        spin_for(self, 1.5)
        if self.driven_at <= check:
            log.info("all Nav2 goals cancelled, robot not commanded to move")
            return
        log.error("Nav2 STILL drives the robot after cancel: PAUSING Nav2 (ant1_nav_speed brings it back)")
        if self.lifecycle_cli.wait_for_service(timeout_sec=2.0):
            f = self.lifecycle_cli.call_async(ManageLifecycleNodes.Request(command=ManageLifecycleNodes.Request.PAUSE))
            rclpy.spin_until_future_complete(self, f, timeout_sec=10.0)
        self.zeros()
        check = time.monotonic()
        spin_for(self, 1.5)
        if self.driven_at > check:
            log.error("STILL MOVING COMMANDS: USE THE E-STOP")
        else:
            log.warn("Nav2 paused, robot stopped. Run ant1_nav_speed (robot still) before the next goal")


def spin_for(node, secs):
    end = time.monotonic() + secs
    while time.monotonic() < end:
        rclpy.spin_once(node, timeout_sec=0.05)


def _interrupt(*_):
    raise KeyboardInterrupt


def main():
    args = rclpy.utilities.remove_ros_args(sys.argv)[1:]
    if len(args) not in (2, 3):
        print(__doc__)
        return
    x, y = float(args[0]), float(args[1])
    signal.signal(signal.SIGHUP, _interrupt)
    signal.signal(signal.SIGTERM, _interrupt)
    rclpy.init(signal_handler_options=SignalHandlerOptions.NO)
    node = Ant1Goto()
    log = node.get_logger()
    try:
        # Over Wi-Fi, discovering the robot + first TF took 1-5 s (measured 10 Oct): wait up to 15 s.
        start, t0 = None, time.monotonic()
        while start is None and time.monotonic() - t0 < 15.0:
            spin_for(node, 0.2)
            start = node.pose()
        if start is None:
            log.error(f"no TF {node.map_frame} -> {node.base_frame} after 15 s: is Mapping/Navigation running? "
                      "right ANT1_IFACE in this terminal?")
            return
        t0 = time.monotonic()                      # learn the existing (old) goal ids before sending ours
        while not node.got_status and time.monotonic() - t0 < 5.0:
            spin_for(node, 0.1)                    # (no list at all = Nav2 freshly started, no old goals)
        t0 = time.monotonic()
        while node.goal_pub.get_subscription_count() == 0 and time.monotonic() - t0 < 15.0:
            spin_for(node, 0.1)                    # wait until Nav2's goal_pose subscriber is matched
        if node.goal_pub.get_subscription_count() == 0:
            log.error("nobody subscribes goal_pose: is Nav2 running?")
            return
        target = (x, y)                            # what you asked for (reachability is re-checked on retries)
        free = node.free_goal(x, y, start)
        if free is None:
            log.error("the robot can't reach anything from where it stands (boxed in?): not sent")
            return
        x, y, note = free
        if note:
            log.warn(note)
        if len(args) == 3 and args[2] == "face":
            yaw = math.atan2(y - start[1], x - start[0])
        elif len(args) == 3:
            yaw = math.radians(float(args[2]))
        else:
            yaw = start[2]                         # default: keep heading -> no turning just for the goal
        log.info(f"from ({start[0]:.2f}, {start[1]:.2f}) heading {math.degrees(start[2]):.0f} deg "
                 f"to ({x:.2f}, {y:.2f}) facing {math.degrees(yaw):.0f} deg")
        node.send(x, y, yaw)
        t0, last, sent_at, tries, attempt = time.monotonic(), 0.0, time.monotonic(), 1, 1
        while True:
            spin_for(node, 0.1)
            now = time.monotonic()
            if node.result == GoalStatus.STATUS_ABORTED and attempt <= node.retries:
                attempt += 1                       # Nav2 gave up: try again (map/costmap may have changed)
                log.warn(f"Nav2 gave up: retrying ({attempt - 1}/{node.retries})")
                node.known.add(node.mine)
                node.mine, node.result = None, None
                spin_for(node, 1.0)
                here = node.pose()
                free = node.free_goal(target[0], target[1], here) if here else None
                if free:                           # map may have grown: aim for the closest reachable again
                    x, y = free[0], free[1]
                    if free[2]:
                        log.warn(free[2])
                node.send(x, y, yaw)
                sent_at, tries = time.monotonic(), 1
                continue
            if node.result is not None:
                break
            if node.mine is None and now - sent_at > 5.0:   # loaded Nav2 took 3.6 s (10 Oct)
                if tries == 5:
                    log.error("Nav2 never confirmed the goal (5 tries): is Navigation running?")
                    return
                tries += 1
                log.warn(f"goal not confirmed by Nav2: re-sending ({tries}/5)")   # first message lost over Wi-Fi
                node.send(x, y, yaw)
                sent_at = now
            if now - t0 > node.timeout:
                log.warn(f"timeout after {node.timeout:.0f} s")
                return
            if now - last > 1.0:
                p = node.pose()
                if p:
                    log.info(f"at ({p[0]:.2f}, {p[1]:.2f})  {math.hypot(x - p[0], y - p[1]):.2f} m to go")
                last = now
        p = node.pose()
        if node.result == GoalStatus.STATUS_SUCCEEDED and p and math.hypot(x - p[0], y - p[1]) > 0.6:
            # Fail-safe: never report arrival (and leave a goal running) when the robot isn't there.
            log.error(f"Nav2 reported success but the robot is {math.hypot(x - p[0], y - p[1]):.2f} m from the goal: "
                      "cancelling all goals and stopping")
            return
        log.info(DONE[node.result])
    except KeyboardInterrupt:
        log.info("Ctrl-C")
    except Exception:
        log.error("crashed: cancelling the goal and stopping")
        raise
    finally:
        if node.sent:                              # never leave a goal running behind us, whatever happened
            node.cancel_and_stop()
        node.destroy_node()
        rclpy.try_shutdown()
