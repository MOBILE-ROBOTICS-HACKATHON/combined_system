"""Shared machinery for the ANT + drone missions with ONE OR TWO drones (mission_scout, mission_hop).

Per drone (class Drone): its ROS clients, position/status, a mode, and the RECALL state machine.
For the fleet (class Fleet, a Node): the ANT (AntView, /ant/goal, /ant/stop), the LANDING QUEUE and the
keyboard/status loop. Docs: docs/08_ANT_DRONE_MISSIONS.md, docs/09_MISSION_STEP_BY_STEP.md.

Two drones on one ANT:
  SLOTS   where each drone sits on the plate (ANT body frame: +x = ANT front): one drone in the centre,
          two drones 15 cm in front / behind the centre (cf1 front, cf2 back). Put them there, noses forward.
  LAYER   every drone flies 0.25 m higher than the previous one (cf2 above cf1's height band) and
          formations keep >= 0.6 m between them: no drone hangs in the other's downwash.
  QUEUE   only ONE drone lands at a time (the "owner" of the plate). The other one WAITS 0.9 m BESIDE
          the ANT (not above it) until the plate is free. The ANT stays paused until both are down.
  TAKEOFF staggered: cf1 first, cf2 4 s later.

RECALL (one drone):  APPROACH -> (wait beside if the plate is taken) -> PAUSE_ANT -> ALIGN over its slot
  -> DESCEND 5 cm/s -> LAND on the plate -> DISARM.  ALIGN fails twice -> BESIDE: land on the floor next
  to the ANT.  ANT data lost > 2 s: hover; > 10 s: land on the floor where it is.
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
from crazyflie_interfaces.srv import Arm, GoTo, Land, NotifySetpointsStop, Takeoff
from geometry_msgs.msg import Point, PoseStamped
from std_msgs.msg import Empty, String
from team_pkg.mission_common import LATCHED, AntView, supervisor_problem, yaw_of

SLOTS = {1: [(0.0, 0.0)], 2: [(0.15, 0.0), (-0.15, 0.0)]}
LAYER = 0.25
CRUISE = {1: [0.6], 2: [0.55, 0.9]}   # m above the floor in every formation: 35 cm between the two layers
STAGGER = 4.0
RECALL = ("APPROACH", "PAUSE_ANT", "ALIGN", "DESCEND", "LAND", "DISARM")
LANDING = RECALL + ("BESIDE", "LAND_FLOOR", "GUARD")   # GUARD = cf_guard took over: no more commands


def dur(s):
    return Duration(sec=int(s), nanosec=int(s % 1 * 1e9))


class Drone:
    def __init__(self, fleet, name, idx, n):
        self.f, self.name, self.idx, self.slot = fleet, name, idx, SLOTS[n][idx]
        self.cli = {k: fleet.create_client(t, f"/{name}/{k}") for k, t in
                    (("go_to", GoTo), ("takeoff", Takeoff), ("land", Land), ("arm", Arm),
                     ("notify_setpoints_stop", NotifySetpointsStop))}
        fleet.create_subscription(PoseStamped, f"/{name}/pose", self.on_pose, 10)
        fleet.create_subscription(Status, f"/{name}/status", lambda m: setattr(self, "status", m), 10)
        fleet.create_subscription(String, f"/{name}/guard_event", self.on_guard, 10)
        self.pos = self.status = self.target = self.aim = self.home = None
        self.pos_at = self.yaw = self.go_at = self.aim_at = 0.0
        self.mode, self.state_at, self.flight_at, self.takeoff_at = "IDLE", time.monotonic(), None, None
        self.low_since = self.align_ok_since = self.ant_lost_at = None
        self.align_tries = 0

    # ---- state ----
    def on_pose(self, m):
        p = m.pose.position
        self.pos, self.pos_at, self.yaw = (p.x, p.y, p.z), time.monotonic(), yaw_of(m.pose.orientation)

    def flying(self):
        return bool(self.status and self.status.supervisor_info & Status.SUPERVISOR_INFO_IS_FLYING)

    def on_guard(self, m):
        """cf_guard landed/killed this drone: hands off (a go_to from us would cancel its landing)."""
        if self.mode != "IDLE":
            if self.f.owner is self:
                self.f.owner = None
            self.target = None
            self.set_mode("GUARD", f"cf_guard took over ({m.data}): no more commands from the mission")

    def say(self, text):
        print(f"\n[{self.name} {self.mode}] {text}", flush=True)

    def set_mode(self, mode, why=""):
        if mode != self.mode:
            self.mode, self.state_at = mode, time.monotonic()
            self.say(f"-> {mode}" + (f": {why}" if why else ""))

    def slot_world(self, ant):
        """Its slot on the plate, in the drone frame (x, y)."""
        ax, ay, _, yaw = ant
        sx, sy = self.slot
        return ax + math.cos(yaw) * sx - math.sin(yaw) * sy, ay + math.sin(yaw) * sx + math.cos(yaw) * sy

    def on_slot(self, ant, tol=0.12):
        if not self.pos:
            return False
        sx, sy = self.slot_world(ant)
        return math.hypot(self.pos[0] - sx, self.pos[1] - sy) < tol and abs(self.pos[2] - ant[2]) < 0.05

    # ---- commands ----
    def ready(self, k):
        if self.cli[k].service_is_ready():
            return True
        self.say(f"/{self.name}/{k} not available: is cf_server running (with this drone)?")
        return False

    def go(self, x, y, z, min_secs=1.0, force=False):
        """go_to (absolute), timed for the fleet speed; re-sent only if the target moved > 3 cm or every 1.5 s."""
        now = time.monotonic()
        if not force and self.target and math.dist(self.target, (x, y, z)) < 0.03 and now - self.go_at < 1.5:
            return
        if not (self.pos and self.flying() and self.ready("go_to")):   # never move a drone on the ground
            return
        self.go_at = now
        secs = max(math.dist(self.pos, (x, y, z)) / self.f.speed, min_secs)
        self.cli["go_to"].call_async(GoTo.Request(relative=False, goal=Point(x=float(x), y=float(y), z=float(z)),
                                                  yaw=float(self.yaw), duration=dur(secs)))
        self.target = (x, y, z)

    def land(self, z, secs):
        if self.ready("land"):
            self.cli["notify_setpoints_stop"].call_async(NotifySetpointsStop.Request(remain_valid_millisecs=100))
            self.cli["land"].call_async(Land.Request(height=float(z), duration=dur(secs)))

    def arm(self, on):
        if self.ready("arm"):
            self.cli["arm"].call_async(Arm.Request(arm=on))

    def takeoff(self, h, secs):
        if self.ready("takeoff"):
            self.cli["takeoff"].call_async(Takeoff.Request(height=float(h), duration=dur(secs)))


class Fleet(Node):
    """Base node of a mission. Subclasses add behaviour in fly_step(d, ant), extra_step(), key(), extra_status()."""

    def __init__(self, node_name):
        super().__init__(node_name)
        names = [n for n in self.declare_parameter("cfs", "cf1").value.split(",") if n]
        if len(names) not in SLOTS:
            raise SystemExit(f"1 or 2 drones, not {names}")
        self.speed = min(self.declare_parameter("speed", 0.15).value, 0.3)
        self.recall_v = self.declare_parameter("recall_v", 3.55).value
        self.max_flight = self.declare_parameter("max_flight", 240.0).value
        self.antv = AntView(self)
        self.create_subscription(String, "/ant/nav_state", lambda m: setattr(self, "nav", m.data), LATCHED)
        self.goal_pub = self.create_publisher(PoseStamped, "/ant/goal", 10)
        self.stop_pub = self.create_publisher(Empty, "/ant/stop", 10)
        self.drones = [Drone(self, n, i, len(names)) for i, n in enumerate(names)]
        self.nav, self.owner, self.hold, self.quit_after = "?", None, False, False

    # ---------------- ANT ----------------
    def ant_goal(self, frame, x, y):
        g = PoseStamped()
        g.header.frame_id = frame
        g.pose.position.x, g.pose.position.y = float(x), float(y)   # orientation zero = keep heading
        self.goal_pub.publish(g)

    def ant_stop(self, why):
        self.stop_pub.publish(Empty())
        print(f"\n[ANT] stop: {why}", flush=True)

    def floor(self):
        return self.antv.calib.floor_z if self.antv.calib else 0.0

    def busy(self):
        """Some drone is recalling / landing: the ANT stays where it is (hold) until all are down."""
        return any(d.mode in LANDING for d in self.drones)

    # ---------------- takeoff ----------------
    def takeoff_all(self, old, which=None):
        """Every selected drone sitting on its slot of a STANDING ANT takes off (staggered), after FLY."""
        ant = self.antv.fresh()
        if ant is None or self.antv.calib is None:
            print("\ntakeoff refused: no ANT data / calibration (ant_link running? mission_calib done?)")
            return False
        ready = [d for d in (which or self.drones) if not d.flying() and d.mode in ("IDLE",)]
        for d in list(ready):
            why = supervisor_problem(d.status.supervisor_info) if d.status else "no status (cf_server running?)"
            if why:
                d.say(f"can't take off: {why}")
                ready.remove(d)
        ok = [d for d in ready if d.pos]
        for d in list(ok):                                   # 11 Oct: drones may also start from the FLOOR
            on_ant = math.hypot(d.pos[0] - ant[0], d.pos[1] - ant[1]) < 0.3
            if on_ant and self.antv.moving:
                d.say("sits on the DRIVING ANT: it stays down (pause the patrol first)")
                ok.remove(d)
            elif not on_ant:
                d.say("not on the ANT: taking off from where it is (floor)")
        if not ok:
            print("\ntakeoff refused: no drone is sitting on the ANT (landed, on its slot)")
            return False
        if not (old and confirm_fly(old)):
            return False
        for i, d in enumerate(ok):
            d.takeoff_at = time.monotonic() + i * STAGGER
            d.set_mode("TAKEOFF", f"in {i * STAGGER:.0f} s" if i else "now")
        return True

    def takeoff_step(self, d):
        now = time.monotonic()
        if d.takeoff_at and now >= d.takeoff_at:
            ant = self.antv.fresh()
            if ant is None or self.antv.moving:
                d.say("takeoff cancelled: ANT data lost or the ANT moved")
                d.takeoff_at = None
                d.set_mode("IDLE")
                return
            d.arm(True)
            end = time.monotonic() + 1.0                    # let the arming complete first
            while time.monotonic() < end:
                rclpy.spin_once(self, timeout_sec=0.05)
            h = ant[2] + 0.3 + LAYER * d.idx
            d.takeoff(h, max(0.3 / self.speed, 3.0))
            d.takeoff_at, d.flight_at, d.home = None, time.monotonic(), d.pos
            d.state_at = time.monotonic()
            d.say(f"armed, taking off to z {h:.2f} (plate + {0.3 + LAYER * d.idx:.2f} m)")
        elif d.takeoff_at is None and now - d.state_at > 5.0:
            d.set_mode("HOLD" if d.flying() else "IDLE", "takeoff done" if d.flying() else "takeoff FAILED (armed? battery?)")
            if d.flying() and d.pos:
                d.go(*d.pos, force=True)

    # ---------------- recall ----------------
    def start_recall(self, d, why):
        if d.mode in LANDING or not d.flying():
            return
        d.align_tries, d.align_ok_since = 0, None
        d.set_mode("APPROACH", why)

    def wait_point(self, d, ant):
        """Where a drone waits for the plate: 0.9 m BESIDE the ANT (left for cf1, right for cf2), own layer."""
        side = 1 if d.idx == 0 else -1
        return (ant[0] + 0.9 * math.cos(ant[3] + side * math.pi / 2), ant[1] + 0.9 * math.sin(ant[3] + side * math.pi / 2),
                ant[2] + 0.5 + LAYER * d.idx)

    def recall_step(self, d, ant):
        now, t = time.monotonic(), time.monotonic() - d.state_at
        sx, sy = d.slot_world(ant)
        plate = ant[2]
        xy = math.hypot(d.pos[0] - sx, d.pos[1] - sy) if d.pos else 99
        if d.mode == "APPROACH":
            if self.owner is None:
                self.owner = d
                d.say("the plate is free: coming in")
            if self.owner is not d:
                d.go(*self.wait_point(d, ant))              # wait beside, out of the other one's downwash
                return
            d.go(sx, sy, plate + 0.5)
            if xy < 0.25:
                if not self.hold:
                    self.hold = True
                    self.ant_stop(f"{d.name} is above it: pausing for the touchdown")
                d.set_mode("PAUSE_ANT")
        elif d.mode == "PAUSE_ANT":
            d.go(sx, sy, plate + 0.5)
            if self.antv.moving is False and t > 1.0:
                d.set_mode("ALIGN", "ANT stands still")
            elif t > 8.0:
                self.ant_stop("still moving after 8 s: stop again")
                d.state_at = now
        elif d.mode == "ALIGN":
            d.go(sx, sy, plate + 0.4)
            if xy < 0.08 and d.pos and abs(d.pos[2] - plate - 0.4) < 0.08:
                d.align_ok_since = d.align_ok_since or now
                if now - d.align_ok_since > 2.0:
                    d.set_mode("DESCEND", f"{xy * 100:.0f} cm off its slot")
                    d.go(sx, sy, plate + 0.15, min_secs=0.25 / 0.05, force=True)   # 5 cm/s
            else:
                d.align_ok_since = None
            if t > 20.0 and d.mode == "ALIGN":
                d.align_tries += 1
                if d.align_tries > 1:
                    d.go(*self.wait_point(d, ant)[:2], plate + 0.3, force=True)
                    self.owner = None                        # free the plate for the other drone
                    d.set_mode("BESIDE", "can't line up with the plate: landing on the floor next to the ANT")
                else:
                    d.state_at = now
                    d.say("not aligned after 20 s: trying once more")
        elif d.mode == "BESIDE":
            if t > 7.0:
                d.land(self.floor(), 6.0)
                d.set_mode("LAND_FLOOR")
        elif d.mode == "DESCEND":
            if t > 0.25 / 0.05 + 1.0:
                d.land(plate, 3.0)
                d.set_mode("LAND", "touching down on the plate")

    def ground_step(self, d):
        """After touchdown: check where it sits, disarm, free the plate."""
        now = time.monotonic()
        if d.mode in ("LAND", "LAND_FLOOR", "GUARD") and not d.flying() and now - d.state_at > 2.0:
            ant = self.antv.fresh(2.0)
            if ant and d.pos:
                sx, sy = d.slot_world(ant)
                d.say(f"touchdown {'ON ITS SLOT' if d.on_slot(ant) else 'NOT on its slot'}: "
                      f"{math.hypot(d.pos[0] - sx, d.pos[1] - sy) * 100:.0f} cm from it, z {d.pos[2]:+.2f}")
            d.arm(False)
            if self.owner is d:
                self.owner = None
            d.set_mode("DISARM", "disarmed")
        elif d.mode == "DISARM" and now - d.state_at > 3.0:
            d.flight_at = None
            ant = self.antv.fresh(2.0)
            on = bool(ant and d.on_slot(ant, 0.25))
            d.set_mode("IDLE", "on the ANT" if on else "landed, NOT on the ANT: put it back on its slot to fly again")
            if not self.busy():
                self.hold = False
                self.all_down()

    def all_down(self):
        """Hook: every recalling drone is down (subclass resumes the ANT patrol, etc.)."""

    # ---------------- per-drone step ----------------
    def step(self):
        now = time.monotonic()
        ant = self.antv.fresh(2.0)
        for d in self.drones:
            if d.mode == "TAKEOFF":
                self.takeoff_step(d)
                continue
            fly = d.flying()
            if d.mode == "IDLE" and fly:
                d.set_mode("HOLD", "it is flying")
            if not fly:
                if d.mode not in LANDING and d.mode not in ("IDLE", "TAKEOFF"):
                    if self.owner is d:
                        self.owner = None
                    d.set_mode("LAND_FLOOR", "not flying any more (landed by cf_guard? killed?)")
                if d.mode not in ("IDLE", "TAKEOFF"):
                    self.ground_step(d)
                continue
            v = d.status.battery_voltage if d.status else None
            if v is not None and v < self.recall_v:
                d.low_since = d.low_since or now
                if now - d.low_since > 3.0:
                    self.start_recall(d, f"battery {v:.2f} V < {self.recall_v} V")
            else:
                d.low_since = None
            if d.flight_at and now - d.flight_at > self.max_flight:
                self.start_recall(d, f"flight time > {self.max_flight:.0f} s")
            if d.mode == "GUARD":                       # cf_guard is landing it: hands off
                continue
            if ant is None or self.antv.calib is None:
                d.ant_lost_at = d.ant_lost_at or now
                if now - d.ant_lost_at < 0.3 and d.pos:
                    d.say("ANT DATA LOST: hovering (landing on the floor in 10 s if it doesn't come back)")
                    d.go(*d.pos, force=True)
                if now - d.ant_lost_at > 10.0 and d.mode != "LAND_FLOOR":
                    if self.owner is d:
                        self.owner = None
                    d.land(self.floor(), 6.0)
                    d.set_mode("LAND_FLOOR", "ANT data lost for 10 s")
                continue
            d.ant_lost_at = None
            if d.mode in RECALL or d.mode in ("BESIDE",):
                self.recall_step(d, ant)
            elif d.mode not in LANDING:
                self.fly_step(d, ant)
        self.extra_step()

    def fly_step(self, d, ant):
        """Hook: what a flying drone does outside the recall (formations, hop legs)."""

    def extra_step(self):
        """Hook: mission-level logic (patrol, hop sequence)."""

    def land_here(self, d, why):
        """Land where it is: on the plate if above the ANT, else the floor."""
        if d.flying() and d.pos:
            z = self.antv.land_z(d.pos[:2], self.floor())
            d.land(z, max((d.pos[2] - z) / self.speed, 5.0))
            if self.owner is d:
                self.owner = None
            d.set_mode("LAND" if self.antv.calib and z == self.antv.calib.plate_z else "LAND_FLOOR", why)

    # ---------------- status ----------------
    def status_line(self):
        parts = []
        for d in self.drones:
            s = f"{d.name} {d.mode}"
            if d.pos:
                s += f" ({d.pos[0]:+.2f},{d.pos[1]:+.2f},{d.pos[2]:+.2f})"
            if d.status:
                s += f" {d.status.battery_voltage:.2f}V"
            if d.flight_at:
                s += f" {time.monotonic() - d.flight_at:.0f}s"
            parts.append(s)
        ant = self.antv.fresh(2.0)
        parts.append(f"ANT ({ant[0]:+.2f},{ant[1]:+.2f}) {'moving' if self.antv.moving else 'still'}" if ant else "NO ANT DATA")
        if self.antv.calib is None:
            parts.append("NO CALIB")
        if self.owner:
            parts.append(f"plate: {self.owner.name} landing")
        extra = self.extra_status()
        if extra:
            parts.append(extra)
        return " | ".join(parts)

    def extra_status(self):
        return ""

    def key(self, k, old):
        """Hook: mission keys. Return True if handled."""
        return False

    # ---------------- the loop ----------------
    def run(self, doc):
        signal.signal(signal.SIGHUP, _interrupt)
        signal.signal(signal.SIGTERM, _interrupt)
        print(doc)
        print(f"drones: {', '.join(d.name for d in self.drones)} | speed {self.speed} m/s | recall at {self.recall_v} V "
              f"or {self.max_flight:.0f} s")
        old = termios.tcgetattr(sys.stdin) if sys.stdin.isatty() else None
        last_step = last_print = 0.0
        try:
            if old:
                tty.setcbreak(sys.stdin.fileno())
            while rclpy.ok():
                if self.quit_after and not any(d.flying() or d.mode in LANDING for d in self.drones):
                    print("\nall drones down: quitting")
                    break
                k = read_key(0.05) if old else time.sleep(0.05)
                for _ in range(30):                      # handle EVERY waiting message (e.g. a cf_guard takeover)
                    rclpy.spin_once(self, timeout_sec=0)  # before deciding anything: one per loop was too slow
                if k == "q":
                    flying = [d for d in self.drones if d.flying()]
                    if not flying:
                        break
                    self.quit_after = True
                    for d in flying:
                        self.start_recall(d, "q pressed: landing on the ANT, then quitting")
                elif k == "x":
                    self.ant_stop("x pressed")
                    for d in self.drones:
                        self.land_here(d, "x pressed")
                    self.on_stop_all()
                elif k:
                    self.key(k, old)
                now = time.monotonic()
                if now - last_step > 0.2:
                    self.step()
                    last_step = now
                if now - last_print > 0.5:
                    print(f"\r{self.status_line()[:200]:<200}", end="", flush=True)
                    last_print = now
        except KeyboardInterrupt:
            print("\nCtrl-C: stopping the ANT. Flying drones keep their last target: land them with cf_guard (SPACE)")
        finally:
            if old:
                termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old)
            self.stop_pub.publish(Empty())          # never leave the ANT driving behind us
            end = time.monotonic() + 0.5
            while time.monotonic() < end:
                rclpy.spin_once(self, timeout_sec=0.05)
            self.destroy_node()
            rclpy.try_shutdown()

    def on_stop_all(self):
        """Hook: x pressed (subclass switches the patrol off, etc.)."""


def confirm_fly(old):
    """Takeoff needs a typed FLY (Enter alone or anything else cancels)."""
    termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old)
    try:
        ok = input("\nClear flight area, people away from the props? Type FLY + Enter: ").strip() == "FLY"
        print("" if ok else "takeoff cancelled")
        return ok
    finally:
        tty.setcbreak(sys.stdin.fileno())


def ask(old, prompt):
    """Read a line (leaves raw key mode for it)."""
    termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old)
    try:
        return input(f"\n{prompt}").strip()
    finally:
        tty.setcbreak(sys.stdin.fileno())


def read_key(timeout):
    if not select.select([sys.stdin], [], [], timeout)[0]:
        return None
    return os.read(sys.stdin.fileno(), 1).decode(errors="ignore")


def _interrupt(*_):
    raise KeyboardInterrupt


def start(cls, doc):
    rclpy.init(signal_handler_options=SignalHandlerOptions.NO)
    cls().run(doc)
