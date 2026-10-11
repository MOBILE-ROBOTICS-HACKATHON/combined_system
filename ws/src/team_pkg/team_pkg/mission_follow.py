"""METHOD 1, mission_follow: YOU fly the drone (cf_keys), the ANT follows underneath it with Nav2.

    mission_follow                              (helper from scripts/cf_env.sh; needs cf_server + ant_link)
    = ros2 run team_pkg mission_follow --ros-args -p cf:=cf1 [-p deadband:=0.4 -p min_move:=0.3 -p period:=2.0]

  f   follow ON / OFF (starts OFF)       x   stop the ANT now (follow OFF)       q   stop the ANT + quit

Follow rule (5 Hz): the target is the point under the drone. A new ANT goal is sent only when
  the ANT is more than `deadband` (0.4 m) from the target, AND the target moved more than `min_move`
  (0.3 m) since the last goal, AND the last goal is at least `period` (2 s) old.
Nav2 avoids obstacles on the way; if the point under the drone is blocked (drone over a table),
ant_link sends the ANT to the closest point it can reach.
The ANT is STOPPED (/ant/stop: every Nav2 action cancelled + zero velocity) when following goes OFF,
the drone isn't flying (landed / not armed), the drone's position or the ANT's data is older than 1 s,
or there is no calibration. The drone side stays protected by cf_guard (run it with -p leash:=2.0).
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
from rclpy.node import Node
from rclpy.signals import SignalHandlerOptions
from crazyflie_interfaces.msg import Status
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Empty, String
from team_pkg.mission_common import LATCHED, AntView


class Follow(Node):
    def __init__(self):
        super().__init__("mission_follow")
        cf = self.declare_parameter("cf", "cf1").value
        self.deadband = self.declare_parameter("deadband", 0.4).value
        self.min_move = self.declare_parameter("min_move", 0.3).value
        self.period = self.declare_parameter("period", 2.0).value
        self.antv = AntView(self)
        self.create_subscription(PoseStamped, f"/{cf}/pose", self.on_pose, 10)
        self.create_subscription(Status, f"/{cf}/status", lambda m: setattr(self, "status", m), 10)
        self.create_subscription(String, "/ant/nav_state", lambda m: setattr(self, "nav", m.data), LATCHED)
        self.goal_pub = self.create_publisher(PoseStamped, "/ant/goal", 10)
        self.stop_pub = self.create_publisher(Empty, "/ant/stop", 10)
        self.pos, self.pos_at, self.status, self.nav = None, 0.0, None, "?"
        self.on, self.commanding, self.last_goal, self.sent_at, self.why = False, False, None, 0.0, "follow is OFF (press f)"

    def on_pose(self, m):
        p = m.pose.position
        self.pos, self.pos_at = (p.x, p.y, p.z), time.monotonic()

    def flying(self):
        return bool(self.status and self.status.supervisor_info & Status.SUPERVISOR_INFO_IS_FLYING)

    def stop_ant(self, why):
        self.stop_pub.publish(Empty())
        self.commanding, self.last_goal = False, None
        print(f"\nANT STOPPED: {why}")

    def blocker(self):
        """Why we may NOT drive the ANT right now (None = OK)."""
        if not self.on:
            return "follow is OFF (press f)"
        if self.antv.calib is None:
            return "no calibration (run mission_calib)"
        if self.antv.fresh() is None:
            return "no ANT data (ant_link running? ANT connected?)"
        if self.pos is None or time.monotonic() - self.pos_at > 1.0:
            return "no drone position (cf_server running?)"
        if not self.flying():
            return "the drone isn't flying"
        return None

    def step(self):
        """5 Hz: decide whether the ANT gets a new goal, or must stop."""
        self.why = self.blocker()
        if self.why:
            if self.commanding:
                self.stop_ant(self.why)
            return
        ax, ay, _, _ = self.antv.fresh()
        tx, ty = self.pos[0], self.pos[1]
        now = time.monotonic()
        far = math.hypot(tx - ax, ty - ay) > self.deadband
        moved = self.last_goal is None or math.hypot(tx - self.last_goal[0], ty - self.last_goal[1]) > self.min_move
        if far and moved and now - self.sent_at > self.period:
            g = PoseStamped()
            g.header.frame_id = "world"                  # drone frame; ant_link converts + checks reachability
            g.pose.position.x, g.pose.position.y = tx, ty  # orientation all zero = the ANT keeps its heading
            self.goal_pub.publish(g)
            self.last_goal, self.sent_at, self.commanding = (tx, ty), now, True
            print(f"\nANT goal -> under the drone ({tx:+.2f}, {ty:+.2f}) room, {math.hypot(tx - ax, ty - ay):.2f} m away")

    def status_line(self):
        s = f"FOLLOW {'ON ' if self.on else 'OFF'}"
        if self.pos:
            s += f" | drone ({self.pos[0]:+.2f}, {self.pos[1]:+.2f}, {self.pos[2]:+.2f}) {'FLYING' if self.flying() else 'landed'}"
        ant = self.antv.fresh()
        if ant and self.pos:
            s += f" | ANT ({ant[0]:+.2f}, {ant[1]:+.2f}) {math.hypot(self.pos[0] - ant[0], self.pos[1] - ant[1]):.2f} m from the drone"
            s += " moving" if self.antv.moving else " still"
        s += f" | nav {self.nav}"
        if self.last_goal:
            s += f" | goal {time.monotonic() - self.sent_at:.0f} s ago"
        if self.why:
            s += f" | WAITING: {self.why}"
        return s


def read_key(timeout):
    if not select.select([sys.stdin], [], [], timeout)[0]:
        return None
    return os.read(sys.stdin.fileno(), 1).decode(errors="ignore")


def _interrupt(*_):
    raise KeyboardInterrupt


def main():
    signal.signal(signal.SIGHUP, _interrupt)
    signal.signal(signal.SIGTERM, _interrupt)
    rclpy.init(signal_handler_options=SignalHandlerOptions.NO)
    node = Follow()
    print(__doc__.split("\n\n")[2])
    print(__doc__.split("\n\n")[3])
    old = termios.tcgetattr(sys.stdin) if sys.stdin.isatty() else None
    last_step = last_print = 0.0
    try:
        if old:
            tty.setcbreak(sys.stdin.fileno())
        while rclpy.ok():
            k = read_key(0.05) if old else time.sleep(0.05)
            for _ in range(30):                          # ALL waiting messages, not one per loop (fell behind at ~20/s)
                rclpy.spin_once(node, timeout_sec=0)
            now = time.monotonic()
            if k == "f":
                node.on = not node.on
                print(f"\nfollow {'ON' if node.on else 'OFF'}")
            elif k == "x":
                node.on = False
                node.stop_ant("x pressed")
            elif k == "q":
                break
            if now - last_step > 0.2:
                node.step()
                last_step = now
            if now - last_print > 0.5:
                print(f"\r{node.status_line()[:160]:<160}", end="", flush=True)
                last_print = now
    except KeyboardInterrupt:
        pass
    finally:
        if old:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old)
        node.stop_ant("mission_follow closing")   # never leave the ANT driving behind us
        end = time.monotonic() + 0.5
        while time.monotonic() < end:
            rclpy.spin_once(node, timeout_sec=0.05)
        node.destroy_node()
        rclpy.try_shutdown()
