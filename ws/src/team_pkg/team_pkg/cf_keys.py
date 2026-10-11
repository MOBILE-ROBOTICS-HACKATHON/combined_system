"""Keyboard control for a Crazyflie: manual steps AND typed go-to targets, in one terminal.

    cf_keys                                     (helper from scripts/cf_env.sh)
    = ros2 run team_pkg cf_keys --ros-args -p cf:=cf1 [-p speed:=0.15 -p step:=0.1 -p zmax:=1.0]

  t          type FLY: ARM + take off 0.3 m   SPACE / l   land (disarms 2 s after touchdown)
  Up / Down  forward / back (x)           w / s       up / down (z)
  Left/Right left / right (y)             a / d       turn left / right (15 deg)
  + / -      bigger / smaller steps       h           back above home (with ant_link: above the ANT)
  g          go to: type "x y z" + Enter, metres FROM HOME (z = height above the floor)
  p          print position               q           LAND (if flying), disarm and quit; Ctrl-C too

SLOW BY DESIGN: every move is timed for `speed` (default 0.15 m/s, hard cap 0.3), takeoff and landing too.
Each press = one small step (default 0.1 m) with go_to: the drone holds its position by itself between
presses. Holding a key takes the next step only when the previous one is done, so releasing it stops
within one step.
HOME = where the drone stood (still) before takeoff. Everything here is relative to it, so it works with
Lighthouse room coordinates and Flow-deck takeoff coordinates alike. x/y follow the position system's
axes (Flow deck: x = the drone's front at connect; Lighthouse: the room's axes, shown in cf_rviz).
Takeoff is REFUSED while the position estimate jumps on the ground (e.g. Lighthouse stations not seen).
WITH THE ANT (ant_link running, docs/08): h flies above the ANT's current position; SPACE lands on the
plate when above the ANT, else on the floor; takeoff is refused while the ANT drives.
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
from builtin_interfaces.msg import Duration
from crazyflie_interfaces.msg import Status
from std_msgs.msg import String
from crazyflie_interfaces.srv import Arm, GoTo, Land, NotifySetpointsStop, Takeoff
from geometry_msgs.msg import Point, PoseStamped
from team_pkg.mission_common import AntView, supervisor_problem, supervisor_warning

STEP_MIN, STEP_MAX = 0.05, 0.2
SPEED_MAX = 0.3              # m/s hard cap, whatever the parameter says
MIN_MOVE = 0.7               # s, shortest move (a 15 deg turn too)
TURN = math.radians(15)
KEYS = {  # key -> (dx, dy, dz, dyaw) in steps / turns
    "\x1b[A": (1, 0, 0, 0), "\x1b[B": (-1, 0, 0, 0), "\x1b[D": (0, 1, 0, 0), "\x1b[C": (0, -1, 0, 0),
    "w": (0, 0, 1, 0), "s": (0, 0, -1, 0), "a": (0, 0, 0, 1), "d": (0, 0, 0, -1),
}


def dur(s):
    return Duration(sec=int(s), nanosec=int(s % 1 * 1e9))


class CfKeys(Node):
    def __init__(self):
        super().__init__("cf_keys")
        cf = self.declare_parameter("cf", "cf1").value
        self.speed = min(self.declare_parameter("speed", 0.15).value, SPEED_MAX)
        self.step = min(max(self.declare_parameter("step", 0.1).value, STEP_MIN), STEP_MAX)
        self.zmax = self.declare_parameter("zmax", 1.0).value
        self.takeoff_h = min(self.declare_parameter("takeoff_height", 0.3).value, self.zmax)
        self.cf = cf
        self.goto_cli = self.create_client(GoTo, f"/{cf}/go_to")
        self.takeoff_cli = self.create_client(Takeoff, f"/{cf}/takeoff")
        self.arm_cli = self.create_client(Arm, f"/{cf}/arm")   # newer firmware: no takeoff unless armed
        self.disarm_at = None                                # set after a landing
        self.land_cli = self.create_client(Land, f"/{cf}/land")
        self.stop_cli = self.create_client(NotifySetpointsStop, f"/{cf}/notify_setpoints_stop")
        self.create_subscription(PoseStamped, f"/{cf}/pose", self.on_pose, 10)
        self.create_subscription(Status, f"/{cf}/status", lambda m: setattr(self, "status", m), 10)
        self.pos, self.busy_until, self.status, self.home, self.recent = None, 0.0, None, None, []
        self.antv = AntView(self)
        self.guard_took = None                   # cf_guard landed/killed the drone: keys locked until it is down
        self.create_subscription(String, f"/{cf}/guard_event", lambda m: setattr(self, "guard_took", m.data), 10)

    def flying(self):
        if self.status is not None:
            return bool(self.status.supervisor_info & Status.SUPERVISOR_INFO_IS_FLYING)
        return bool(self.pos and self.home and self.pos[2] - self.home[2] > 0.08)

    def spread(self):
        """Largest change of the position over the last 2 s (m); None if too little data."""
        if len(self.recent) < 5:
            return None
        return max(max(p[i] for _, p in self.recent) - min(p[i] for _, p in self.recent) for i in range(3))

    def on_pose(self, m):
        p = m.pose.position
        now = time.monotonic()
        self.pos = (p.x, p.y, p.z)
        ant = self.antv.fresh()
        rel = (p.x - ant[0], p.y - ant[1], p.z) if ant else self.pos   # riding a moving ANT = standing still
        if bool(ant) != getattr(self, "rel_mode", None):        # reference changed (ANT data came/went):
            self.rel_mode, self.recent = bool(ant), []          # don't mix room and ANT-relative positions
        self.recent = [(t, q) for t, q in self.recent if now - t < 2.0] + [(now, rel)]
        sp = self.spread()
        if not self.flying() and sp is not None and sp < 0.20:   # standing (roughly) still on the ground
            self.home = self.pos

    def rel(self):
        """Position relative to home (x, y, height above the floor)."""
        return tuple(a - b for a, b in zip(self.pos, self.home))

    def ready(self, cli):
        if cli.service_is_ready():
            return True
        print(f"\r{cli.srv_name} not available: is cf_server running?")
        return False

    def go(self, x, y, z, yaw=0.0, relative=False, secs=1.0):
        if self.ready(self.goto_cli):
            self.goto_cli.call_async(GoTo.Request(relative=relative, goal=Point(x=float(x), y=float(y), z=float(z)),
                                                  yaw=float(yaw), duration=dur(secs)))

    def key(self, k):
        now = time.monotonic()
        if self.guard_took and not self.flying():
            self.guard_took = None
        if self.guard_took and k not in "p":
            print(f"\rcf_guard took over ({self.guard_took}): keys locked until the drone is down")
            return
        if (k in KEYS or k == "h") and not self.flying():
            print("\rnot flying: take off first (t). Moves are never sent to a drone on the ground")
            return
        if k in KEYS:
            if now < self.busy_until:           # key held (auto-repeat ~30/s): next step once this one is done
                return
            dx, dy, dz, dyaw = KEYS[k]
            if not (self.pos and self.home):
                print("\rrefused: no position / home yet")
                return
            if dz > 0 and self.rel()[2] + self.step > self.zmax:
                print(f"\rrefused: would go above zmax {self.zmax} m")
                return
            if dz < 0 and self.rel()[2] - self.step < 0.1:
                print("\rrefused: too low (use SPACE to land)")
                return
            secs = max(self.step / self.speed, MIN_MOVE)
            self.busy_until = now + secs
            self.go(dx * self.step, dy * self.step, dz * self.step, dyaw * TURN, relative=True, secs=secs)
        elif k == "t" and self.ready(self.takeoff_cli):
            if self.pos is None:
                print("\rrefused: no position at all from the drone (cf_server running?)")
                return
            if self.home is None:                        # FORCEd with a wobbly estimate: take off from here
                self.home = self.pos
            why = supervisor_problem(self.status.supervisor_info) if self.status else "no status from the drone (cf_server running?)"
            if why:
                print(f"\rrefused: {why}")
                return
            if supervisor_warning(self.status.supervisor_info):
                print(f"\rnote: {supervisor_warning(self.status.supervisor_info)}")
            if self.ready(self.arm_cli):
                self.arm_cli.call_async(Arm.Request(arm=True))
                end = time.monotonic() + 1.0                 # let the arming complete (as Bitcraze's examples)
                while time.monotonic() < end:
                    rclpy.spin_once(self, timeout_sec=0.05)
                print("\rarmed")
            self.disarm_at = None
            secs = max(self.takeoff_h / self.speed, 3.0)
            # takeoff height is absolute (floor + takeoff_h); capped: a FORCEd takeoff with a bad estimate gave 1.74 m
            h = min(max(self.home[2] + self.takeoff_h, 0.2), self.zmax)
            self.takeoff_cli.call_async(Takeoff.Request(height=h, duration=dur(secs)))
            print(f"\rtakeoff to {self.takeoff_h} m above the floor in {secs:.1f} s")
        elif k in (" ", "l") and self.ready(self.land_cli):
            self.stop_cli.call_async(NotifySetpointsStop.Request(remain_valid_millisecs=100))
            home_z = self.home[2] if self.home else 0.0
            floor = self.antv.land_z(self.pos[:2], home_z) if self.pos else home_z   # plate if above the ANT
            secs = max((self.pos[2] - floor) / self.speed, 5.0) if self.pos else 5.0
            self.land_cli.call_async(Land.Request(height=floor, duration=dur(secs)))
            self.disarm_at = time.monotonic() + secs + 2.0   # disarm once it's surely down
            print(f"\rlanding in {secs:.1f} s")
        elif k in "+=-_" and len(k) == 1:
            self.step = round(min(max(self.step * (1.5 if k in "+=" else 1 / 1.5), STEP_MIN), STEP_MAX), 3)
            print(f"\rstep {self.step} m")
        elif k == "h" and self.pos and self.antv.fresh():
            ax, ay, plate, _ = self.antv.fresh()
            target = (ax, ay, max(self.pos[2], plate + 0.3))
            secs = max(math.dist(self.pos, target) / self.speed, 2.0)
            self.go(*target, relative=False, secs=secs)
            print(f"\rgoing above the ANT in {secs:.1f} s (SPACE lands on its plate)")
        elif k == "h" and self.pos and self.home:
            z = self.rel()[2]
            self.goto(0.0, 0.0, z if z > 0.2 else 0.4)
        elif k == "p":
            if self.pos and self.home:
                r = self.rel()
                print(f"\rfrom home ({r[0]:+.2f}, {r[1]:+.2f}, {r[2]:+.2f})   room "
                      f"({self.pos[0]:+.2f}, {self.pos[1]:+.2f}, {self.pos[2]:+.2f})")
            else:
                print("\rno position / home yet")

    def soft_issues(self):
        """Takeoff concerns you may override with FORCE (11 Oct, user: lighter checks)."""
        out = []
        sp = self.spread()
        if sp is None or sp > 0.20:
            out.append("position estimate not stable on the ground" + (f" (jumps {sp * 100:.0f} cm)" if sp else "")
                       + ": Lighthouse stations in view? flying may drift or crash")
        if self.antv.fresh() and self.antv.moving:
            out.append("the ANT is driving")
        return out

    def tick(self):
        """Disarm after a landing, once the drone reports it's on the ground."""
        if self.disarm_at and time.monotonic() > self.disarm_at and not self.flying() and self.arm_cli.service_is_ready():
            self.arm_cli.call_async(Arm.Request(arm=False))
            self.disarm_at = None
            print("\rdisarmed")

    def goto(self, x, y, z):
        """Go to (x, y, z) metres FROM HOME (z = height above the floor), timed for speed."""
        if self.guard_took and self.flying():
            print(f"\rrefused: cf_guard took over ({self.guard_took})")
            return False
        if not self.flying():
            print("\rrefused: not flying (take off first with t). Moves are never sent to a drone on the ground")
            return False
        if not 0.1 <= z <= self.zmax:
            print(f"\rrefused: z={z} outside 0.1..{self.zmax} m")
            return False
        if not (self.pos and self.home):
            print("\rrefused: no position / home yet (is cf_server running?)")
            return False
        target = (self.home[0] + x, self.home[1] + y, self.home[2] + z)
        secs = max(math.dist(self.pos, target) / self.speed, 2.0)
        self.go(*target, relative=False, secs=secs)
        print(f"\rgoing to ({x:.2f}, {y:.2f}, {z:.2f}) from home in {secs:.1f} s")
        return True


def read_key(timeout):
    if not select.select([sys.stdin], [], [], timeout)[0]:
        return None
    k = os.read(sys.stdin.fileno(), 1).decode(errors="ignore")
    if k == "\x1b" and select.select([sys.stdin], [], [], 0.01)[0]:
        k += os.read(sys.stdin.fileno(), 2).decode(errors="ignore")
    return k


def ask_target(old):
    """Leave raw mode, read 'x y z', go back. None if cancelled or invalid."""
    termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old)
    try:
        s = input("\ngo to x y z (m, Enter = cancel): ").split()
        if s and len(s) != 3:
            raise ValueError
        return tuple(float(v) for v in s) if s else None
    except ValueError:
        print("need three numbers, e.g.  0.5 0 0.5")
        return None
    finally:
        tty.setcbreak(sys.stdin.fileno())


def confirm_fly(old):
    """Takeoff needs a typed FLY (Enter alone or anything else cancels)."""
    termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old)
    try:
        ok = input("\nClear flight area, people away from the props? Type FLY + Enter: ").strip() == "FLY"
        print("" if ok else "takeoff cancelled")
        return ok
    finally:
        tty.setcbreak(sys.stdin.fileno())


def confirm_word(old, word, prompt):
    termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old)
    try:
        ok = input(prompt).strip() == word
        print("" if ok else "takeoff cancelled")
        return ok
    finally:
        tty.setcbreak(sys.stdin.fileno())


def land_and_wait(node):
    """On quit: land if flying, wait until it's down, disarm. Never leave a flying drone behind."""
    if not node.flying():
        if not node.arm_cli.service_is_ready():     # no server: nothing to land or disarm
            return
        node.disarm_at = time.monotonic()           # disarm right away (harmless if not armed)
    else:
        node.key("l")
        print("waiting for the landing (Ctrl-C again to stop waiting; the guard still watches)")
    end = time.monotonic() + 15.0
    while node.disarm_at and time.monotonic() < end:
        rclpy.spin_once(node, timeout_sec=0.1)
        node.tick()


def _interrupt(*_):
    raise KeyboardInterrupt


def main():
    signal.signal(signal.SIGHUP, _interrupt)
    signal.signal(signal.SIGTERM, _interrupt)
    rclpy.init(signal_handler_options=SignalHandlerOptions.NO)
    node = CfKeys()
    print(__doc__.split("\n\n")[2])
    print(f"/{node.cf}: speed {node.speed} m/s, step {node.step} m, zmax {node.zmax} m. "
          "Keep cf_guard running in its own terminal.")
    old = termios.tcgetattr(sys.stdin)
    try:
        tty.setcbreak(sys.stdin.fileno())
        while rclpy.ok():
            k = read_key(0.05)
            if k == "q":
                break
            if k == "t":
                issues = node.soft_issues()
                if issues:
                    print("\nWARNING: " + "; ".join(issues))
                    if confirm_word(old, "FORCE", "Take off ANYWAY? Type FORCE + Enter (anything else cancels): "):
                        node.key("t")
                elif confirm_fly(old):
                    node.key("t")
            elif k == "g":
                t = ask_target(old)
                if t:
                    node.goto(*t)
            elif k:
                node.key(k)
            for _ in range(30):                          # ALL waiting messages, not one per loop (fell behind at ~20/s)
                rclpy.spin_once(node, timeout_sec=0)
            node.tick()
    except KeyboardInterrupt:
        pass
    finally:
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old)
        try:
            land_and_wait(node)
        except KeyboardInterrupt:
            print("stopped waiting: LAND WITH THE GUARD (SPACE) if it is still flying")
        print("\ncf_keys closed")
        node.destroy_node()
        rclpy.try_shutdown()
