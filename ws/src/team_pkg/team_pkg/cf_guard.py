"""Safety guard for one or more Crazyflies (the drones' "e-stop terminal"). Keep it running the whole flight.

    cf_guard                                    (helper from scripts/cf_env.sh; guards every drone in $CF_LIST)
    = ros2 run team_pkg cf_guard --ros-args -p cfs:=cf1,cf2 [-p fence_xy:=1.0 -p fence_z:=1.2 -p vbat_land:=3.3]
                                                [-p leash:=2.0] [-p area:="xmin,xmax,ymin,ymax"]

  SPACE or l   LAND ALL flying drones now (5 s, longer from higher up)
  K (capital)  KILL ALL: motors off immediately, the drones FALL (only if about to hit something)
  q            quit the guard (lands every flying drone first)

HOME = where the drone sits while landed (updated until takeoff). The fence is centred on it, so this
works with room coordinates (Lighthouse) and takeoff coordinates (Flow deck) alike.
AREA (-p area:="xmin,xmax,ymin,ymax", drone/Lighthouse frame, see mission_rviz): a FIXED box in the room
instead of the box around home (for ROAM, where drones fly anywhere in the arena, independent of the ANT).
Automatic LAND when:
  - the drone leaves the fence box: more than fence_xy from home in x or y, or fence_z above home;
  - the battery stays below vbat_land (V, under load) for 2 s.
WITH THE ANT (ant_link running, docs/08): -p leash:=2.0 lands the drone when it gets more than 2 m (xy)
from the ANT's CURRENT position instead of the box around home; landing height = the ANT's plate when
the drone is above the ANT, else the floor; riding a moving ANT doesn't count as "unstable".
UNSTABLE POSITION warning while landed (estimate jumps > 5 cm): do NOT take off (Lighthouse not seen?).
LINK LOST alarm when the position stops arriving for 1 s while flying. The drone then KEEPS HOVERING
at its last target (no firmware timeout with takeoff/go_to): catch it from below and turn it upside
down, which stops the motors (tumble detection). The guard keeps trying to land in case the link returns.
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
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from rclpy.signals import SignalHandlerOptions
from builtin_interfaces.msg import Duration
from crazyflie_interfaces.msg import Status
from crazyflie_interfaces.srv import Land, NotifySetpointsStop
from geometry_msgs.msg import PoseStamped
from geometry_msgs.msg import Point
from rclpy.qos import QoSProfile, DurabilityPolicy
from std_msgs.msg import String
from std_srvs.srv import Empty
from visualization_msgs.msg import Marker
from team_pkg.mission_common import CRASHED, AntView, supervisor_problem

STALE = 1.0       # s without a pose while flying -> link lost
LOW_BAT_FOR = 2.0  # s below vbat_land -> land


class CfGuard(Node):
    def __init__(self, cf):
        super().__init__(f"cf_guard_{cf}")
        self.fence_xy = self.declare_parameter("fence_xy", 1.0).value
        self.fence_z = self.declare_parameter("fence_z", 1.2).value
        self.vbat_land = self.declare_parameter("vbat_land", 3.3).value
        self.land_secs = self.declare_parameter("land_secs", 5.0).value
        self.leash = self.declare_parameter("leash", 0.0).value      # m around the ANT; 0 = off
        a = self.declare_parameter("area", "").value
        self.area = [float(v) for v in a.split(",")] if a else None  # xmin, xmax, ymin, ymax (world)
        self.antv = AntView(self)
        self.leash_at = 0.0
        self.cf = cf
        self.create_subscription(PoseStamped, f"/{cf}/pose", self.on_pose, 10)
        self.create_subscription(Status, f"/{cf}/status", self.on_status, 10)
        self.land_cli = self.create_client(Land, f"/{cf}/land")
        self.stop_cli = self.create_client(NotifySetpointsStop, f"/{cf}/notify_setpoints_stop")
        self.kill_cli = self.create_client(Empty, f"/{cf}/emergency")
        self.pos, self.pose_at, self.status = None, 0.0, None
        self.low_since, self.landed_at, self.alarm_at = None, 0.0, 0.0
        self.home, self.recent = None, []          # recent: (time, pos) of the last 2 s, for stability
        latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.fence_pub = self.create_publisher(Marker, f"/{cf}/fence", latched)
        # Missions / cf_keys listen here and STOP commanding this drone: a later go_to would cancel our landing.
        self.event_pub = self.create_publisher(String, f"/{cf}/guard_event", 10)
        self.land_check = None                     # (time to check, land height): did our landing really happen?

    def spread(self):
        """Largest change of the position over the last 2 s (m): ~0 for a drone standing still."""
        if len(self.recent) < 5:
            return None
        return max(max(p[i] for _, p in self.recent) - min(p[i] for _, p in self.recent) for i in range(3))

    def fence_marker(self):
        """The fence as a red wire box around home, for rviz."""
        m = Marker(type=Marker.LINE_LIST, action=Marker.ADD)
        m.header.frame_id, m.ns, m.scale.x = "world", "fence", 0.01
        m.color.r, m.color.a = 1.0, 0.8
        m.pose.orientation.w = 1.0
        f, (hx, hy, hz) = self.fence_xy, self.home
        z0, z1 = hz, hz + self.fence_z
        c = [(hx + x, hy + y) for x, y in ((-f, -f), (f, -f), (f, f), (-f, f))]
        for i in range(4):
            (x1, y1), (x2, y2) = c[i], c[(i + 1) % 4]
            for h in (z0, z1):
                m.points += [Point(x=x1, y=y1, z=h), Point(x=x2, y=y2, z=h)]
            m.points += [Point(x=x1, y=y1, z=z0), Point(x=x1, y=y1, z=z1)]
        return m

    def area_marker(self):
        """The fixed AREA as a red rectangle on the floor/home height, for rviz."""
        m = Marker(type=Marker.LINE_STRIP, action=Marker.ADD)
        m.header.frame_id, m.ns, m.scale.x = "world", "fence", 0.01
        m.color.r, m.color.a = 1.0, 0.8
        m.pose.orientation.w = 1.0
        x0, x1, y0, y1 = self.area
        z = self.home[2] if self.home else 0.0
        m.points = [Point(x=x, y=y, z=z) for x, y in ((x0, y0), (x1, y0), (x1, y1), (x0, y1), (x0, y0))]
        return m

    def leash_marker(self, ant):
        """The leash as a red circle around the ANT, for rviz (replaces the home box)."""
        m = Marker(type=Marker.LINE_STRIP, action=Marker.ADD)
        m.header.frame_id, m.ns, m.scale.x = "world", "fence", 0.01
        m.color.r, m.color.a = 1.0, 0.8
        m.pose.orientation.w = 1.0
        m.points = [Point(x=ant[0] + self.leash * math.cos(a / 20 * math.pi), y=ant[1] + self.leash * math.sin(a / 20 * math.pi),
                          z=ant[2]) for a in range(41)]
        return m

    def on_pose(self, m):
        p = m.pose.position
        now = time.monotonic()
        self.pos, self.pose_at = (p.x, p.y, p.z), now
        ant = self.antv.fresh()
        rel = (p.x - ant[0], p.y - ant[1], p.z) if ant else self.pos   # riding a moving ANT = standing still
        if bool(ant) != getattr(self, "rel_mode", None):        # reference changed (ANT data came/went):
            self.rel_mode, self.recent = bool(ant), []          # don't mix room and ANT-relative positions
        self.recent = [(t, q) for t, q in self.recent if now - t < 2.0] + [(now, rel)]
        # Home follows the drone while it stands (stable) on the ground; frozen once it flies.
        sp = self.spread()
        if not self.flying() and sp is not None and sp < 0.20:
            if self.home is None or math.dist(self.home, self.pos) > 0.05:
                self.home = self.pos
                if not self.leash:                     # leash mode draws its circle around the ANT instead
                    self.fence_pub.publish(self.area_marker() if self.area else self.fence_marker())
        elif self.home is None and self.flying():   # guard started mid-flight: fence around here
            self.home = (p.x, p.y, 0.0)
            self.fence_pub.publish(self.area_marker() if self.area else self.fence_marker())

    def on_status(self, m):
        self.status = m

    def flying(self):
        if self.status is not None:
            return bool(self.status.supervisor_info & Status.SUPERVISOR_INFO_IS_FLYING)
        return self.pos is not None and self.pos[2] > 0.08   # no status topic: judge by height

    def land(self, why):
        self.landed_at = time.monotonic()
        print()                                    # messages on their own line, not after the status line
        if not self.land_cli.service_is_ready():
            self.get_logger().error(f"LAND ({why}) FAILED: cf_server not reachable. Catch the drone, turn it over")
            return
        self.get_logger().warn(f"LANDING: {why}")
        self.event_pub.publish(String(data=f"land: {why}"))
        # Ends any velocity streaming first; without that the drone ignores high-level commands.
        self.stop_cli.call_async(NotifySetpointsStop.Request(remain_valid_millisecs=100))
        # Lighthouse: the floor isn't always exactly z = 0. With the ANT: its plate if above it, else the floor.
        floor = self.antv.land_z(self.pos[:2], self.home[2] if self.home else 0.0) if self.pos else 0.0
        s = max(self.land_secs, (self.pos[2] - floor) / 0.3) if self.pos else self.land_secs   # <= 0.3 m/s down
        self.land_cli.call_async(Land.Request(height=floor, duration=Duration(sec=int(s), nanosec=int(s % 1 * 1e9))))
        now = time.monotonic()
        self.land_check = [now + 1.5, now + s + 2.0, floor, self.pos[2] if self.pos else None]   # early, final, z, z0

    def kill(self):
        print()
        if self.kill_cli.service_is_ready():
            self.event_pub.publish(String(data="kill"))
            self.kill_cli.call_async(Empty.Request())
            self.get_logger().error("KILL: motors off. Drone stays LOCKED until you power-cycle it")
        else:
            self.get_logger().error("KILL FAILED: cf_server not reachable. Catch the drone, turn it over")

    def check(self):
        """~10 Hz: automatic landings and alarms. Returns the status line."""
        now = time.monotonic()
        fly = self.flying()
        relanded = now - self.landed_at < 4.0          # don't spam land requests while it descends
        lc = self.land_check
        if lc and fly and self.pos:
            # Our landing must make the drone DESCEND. If someone sent a go_to after it (a mission, cf_keys,
            # a race: seen in the 2-drone test) or the command got lost, it doesn't: the guard's landing wins.
            if lc[0] and now > lc[0]:
                lc[0] = None
                if lc[3] is not None and self.pos[2] > lc[3] - 0.03 and self.pos[2] - lc[2] > 0.1:
                    self.land("not descending 1.5 s after our landing (overridden?): landing AGAIN")
            elif now > lc[1] and self.pos[2] - lc[2] > 0.1:
                self.land("still up after our landing should have finished: landing AGAIN")
        elif lc and not fly:
            self.land_check = None
        if fly and self.pos and now - self.pose_at > STALE:
            if now - self.alarm_at > 2.0:
                self.alarm_at = now
                print()
                self.get_logger().error("LINK LOST: no position for 1 s. The drone keeps hovering: catch it "
                                        "from below and turn it over. Still trying to land.")
            if not relanded:
                self.land("link lost")
        elif fly and self.pos and self.home:
            dx, dy, dz = (a - b for a, b in zip(self.pos, self.home))
            ant = self.antv.fresh() if self.leash > 0 else None
            if ant:                                        # leash: distance to the ANT's current position
                d = math.hypot(self.pos[0] - ant[0], self.pos[1] - ant[1])
                if (d > self.leash or dz > self.fence_z) and not relanded:
                    self.land(f"off the leash: {d:.2f} m from the ANT (max {self.leash}), {dz:+.2f} m high")
            elif self.area:                                # fixed box in the room
                x0, x1, y0, y1 = self.area
                x, y = self.pos[:2]
                if (not (x0 <= x <= x1 and y0 <= y <= y1) or dz > self.fence_z) and not relanded:
                    self.land(f"outside the area at ({x:+.2f}, {y:+.2f}), {dz:+.2f} m high")
            elif (abs(dx) > self.fence_xy or abs(dy) > self.fence_xy or dz > self.fence_z) and not relanded:
                self.land(f"outside fence: ({dx:+.2f}, {dy:+.2f}, {dz:+.2f}) m from home")
            if ant and self.area and not relanded:         # leash AND area: both must hold
                x0, x1, y0, y1 = self.area
                if not (x0 <= self.pos[0] <= x1 and y0 <= self.pos[1] <= y1):
                    self.land(f"outside the area at ({self.pos[0]:+.2f}, {self.pos[1]:+.2f})")
        v = self.status.battery_voltage if self.status else None
        if fly and v is not None and v < self.vbat_land:
            self.low_since = self.low_since or now
            if now - self.low_since > LOW_BAT_FOR and not relanded:
                self.land(f"battery {v:.2f} V")
        else:
            self.low_since = None
        if self.pos and self.area:
            p = f"({self.pos[0]:+.2f}, {self.pos[1]:+.2f}, {self.pos[2]:+.2f}) room"
        elif self.pos and self.home:
            p = "from home ({:+.2f}, {:+.2f}, {:+.2f})".format(*(a - b for a, b in zip(self.pos, self.home)))
        elif self.pos:
            p = f"({self.pos[0]:+.2f}, {self.pos[1]:+.2f}, {self.pos[2]:+.2f}), no home yet"
        else:
            p = "(no position yet)"
        s = f"{self.cf}: {p} {'FLYING' if fly else 'landed'}"
        ant = self.antv.fresh()
        if ant and self.pos:
            s += f" | ANT {math.hypot(self.pos[0] - ant[0], self.pos[1] - ant[1]):.2f} m" + (f"/{self.leash} leash" if self.leash else "")
            if now - self.leash_at > 0.5 and self.leash:
                self.leash_at = now
                self.fence_pub.publish(self.leash_marker(ant))
        elif self.leash:
            s += " | NO ANT DATA: box fence around home"
        sp = self.spread()
        if not fly and sp is not None and sp > 0.20:
            s += f" | POSITION UNSTABLE ({sp * 100:.0f} cm): DON'T TAKE OFF"
        if self.status:
            s += f" | bat {v:.2f} V | rssi -{self.status.rssi} dBm"
            if self.status.supervisor_info & Status.SUPERVISOR_INFO_IS_TUMBLED:
                s += " | TUMBLED"
            if self.status.supervisor_info & Status.SUPERVISOR_INFO_IS_LOCKED:
                s += " | LOCKED (power-cycle the drone)"
            if self.status.supervisor_info & CRASHED:
                s += " | CRASHED (power-cycle the drone + restart cf_server)"
            elif not fly and supervisor_problem(self.status.supervisor_info):
                s += " | WON'T TAKE OFF (see cf_keys t)"
        if self.pos and now - self.pose_at > STALE:
            s += " | NO DATA"
        return s


def read_key(timeout):
    if not select.select([sys.stdin], [], [], timeout)[0]:
        return None
    return os.read(sys.stdin.fileno(), 1).decode(errors="ignore")


def _interrupt(*_):
    raise KeyboardInterrupt


def main():
    signal.signal(signal.SIGHUP, _interrupt)    # closed terminal / docker stop -> land below
    signal.signal(signal.SIGTERM, _interrupt)
    rclpy.init(signal_handler_options=SignalHandlerOptions.NO)
    boot = Node("cf_guard")                     # only to read which drones to guard
    names = [n for n in boot.declare_parameter("cfs", boot.declare_parameter("cf", "cf1").value).value.split(",") if n]
    boot.destroy_node()
    guards = [CfGuard(n) for n in names]
    ex = SingleThreadedExecutor()
    for g in guards:
        ex.add_node(g)
    g0 = guards[0]
    print(__doc__.split("\n\n")[2])
    fence = (f"area {g0.area}" if g0.area else f"{g0.fence_xy} m around home") + (f", leash {g0.leash} m to the ANT" if g0.leash else "")
    print(f"guarding {', '.join('/' + n for n in names)}: {fence}, {g0.fence_z} m above home, land below {g0.vbat_land} V")
    tty_ok = sys.stdin.isatty()
    old = termios.tcgetattr(sys.stdin) if tty_ok else None
    last_print = 0.0
    try:
        if tty_ok:
            tty.setcbreak(sys.stdin.fileno())
        while rclpy.ok():
            k = read_key(0.1) if tty_ok else time.sleep(0.1)
            if k in (" ", "l"):
                flying = [g for g in guards if g.flying()]
                for g in flying:
                    g.land("SPACE pressed")
                if not flying:
                    print("\nnothing is flying")
            elif k == "K":
                for g in guards:
                    g.kill()
            elif k == "q":
                break
            for _ in range(3 * len(guards)):
                ex.spin_once(timeout_sec=0)
            lines = [g.check() for g in guards]
            if time.monotonic() - last_print > 0.5:
                width = 120 if len(guards) == 1 else 200
                print(f"\r{' || '.join(lines)[:width]:<{width}}", end="", flush=True)
                last_print = time.monotonic()
    except KeyboardInterrupt:
        pass
    finally:
        if tty_ok:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old)
        print()
        flying = [g for g in guards if g.flying()]
        for g in flying:                        # never leave a flying drone without its guard
            g.land("guard closed")
        end = time.monotonic() + (1.0 if flying else 0.0)
        while time.monotonic() < end:
            ex.spin_once(timeout_sec=0.1)
        g0.get_logger().info("guard stopped")
        for g in guards:
            g.destroy_node()
        rclpy.try_shutdown()
