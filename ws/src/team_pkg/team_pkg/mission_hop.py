"""CARRIER HOP, mission_hop: the ANT carries the drone as close as it can get to a target it CANNOT reach
(on/behind a table, behind a barrier); the drone hops over, hovers above the target, comes back, lands on
the ANT, and the ANT drives back to where it started.

    mission_hop                                 (helper from scripts/cf_env.sh; one drone: $CF)
    = ros2 run team_pkg mission_hop --ros-args -p cfs:=cf1 [-p hop_z:=0.8 -p dwell:=5.0 -p max_hop:=1.8]

  g   set the TARGET: type "x y" in the ANT MAP (read it in ant1_rviz) + Enter
      (or click it with "Publish Point" in mission_rviz, drone frame)
  h   start the hop: the ANT drives to the target or, if it can't get there, to the closest point it CAN reach
  t   launch (type FLY) once the ANT has arrived and stopped
  x   abort: the ANT stops, the drone lands where it is (plate if above the ANT, else floor)
  q   land (if flying) and quit

Sequence (printed as it goes):
  DRIVE   ANT -> target (ant_link redirects to the closest REACHABLE point if the target is blocked)
  READY   ANT stopped: shows how far the target still is. Farther than max_hop (1.8 m, inside the 2 m
          cf_guard leash): hop refused. Otherwise press t.
  UP      take off from the ANT, climb to hop_z (0.8 m above the floor; raise it ONLY if the obstacle is
          higher and cf_guard's fence_z allows it)
  OUT     fly to above the target (at 0.15 m/s)       DWELL  hover there `dwell` s (the "delivery")
  BACK    recall: back above the ANT, align, descend 5 cm/s, land on the plate, disarm (mission_fleet)
  HOME    the ANT drives back to where the hop started       DONE
"""
import math
import time
from geometry_msgs.msg import PointStamped
from team_pkg.mission_fleet import LANDING, Fleet, ask, start


class Hop(Fleet):
    def __init__(self):
        super().__init__("mission_hop")
        self.hop_z = min(self.declare_parameter("hop_z", 0.8).value, 1.5)
        self.dwell = self.declare_parameter("dwell", 5.0).value
        self.max_hop = self.declare_parameter("max_hop", 1.8).value
        self.create_subscription(PointStamped, "/clicked_point", self.on_click, 10)
        self.d = self.drones[0]                   # the hop drone (a second drone, if configured, stays down)
        self.target = None                         # (frame, x, y)
        self.phase, self.phase_at, self.start_map = "IDLE", time.monotonic(), None

    def on_click(self, m):
        self.target = (m.header.frame_id or "world", m.point.x, m.point.y)
        print(f"\n[HOP] target from rviz: ({m.point.x:+.2f}, {m.point.y:+.2f}) {self.target[0]}. h = start")

    def set_phase(self, p, why=""):
        self.phase, self.phase_at = p, time.monotonic()
        print(f"\n[HOP {p}] {why}", flush=True)

    def target_room(self):
        f, x, y = self.target
        return self.antv.calib.map_to_room(x, y) if f in ("ant_map",) else (x, y)

    # ---------------- the hop sequence ----------------
    def extra_step(self):
        ant, d, t = self.antv.fresh(2.0), self.d, time.monotonic() - self.phase_at
        if self.phase in ("IDLE", "DONE") or self.antv.calib is None:
            return
        if self.phase in ("CLIMB", "OUT", "DWELL") and d.mode in LANDING:   # e.g. battery recall mid-hop
            self.set_phase("BACK", f"{d.name} is already returning ({d.mode})")
            return
        if self.phase == "DRIVE":
            if t > 2.0 and self.nav in ("succeeded", "aborted", "canceled", "stopped") and self.antv.moving is False:
                tx, ty = self.target_room()
                dist = math.hypot(tx - ant[0], ty - ant[1]) if ant else 99
                if dist > self.max_hop:
                    self.ant_stop("hop refused")
                    self.set_phase("IDLE", f"the ANT stopped {dist:.2f} m from the target (more than max_hop "
                                           f"{self.max_hop} m): pick a closer target or move obstacles")
                else:
                    self.set_phase("READY", f"the ANT is {dist:.2f} m from the target (nav {self.nav}). "
                                            "Press t (type FLY) to launch the hop")
        elif self.phase == "UP":
            if d.flying() and d.mode == "HOLD":
                tx, ty = self.target_room()
                d.mode = "HOP"                          # our own flight mode: fly_step does nothing for it
                d.go(d.pos[0], d.pos[1], self.floor() + self.hop_z, force=True)
                self.set_phase("CLIMB", f"climbing to {self.hop_z} m above the floor")
            elif t > 12.0 and not d.flying():
                self.set_phase("IDLE", "takeoff failed (armed? battery? on the ANT?)")
        elif self.phase == "CLIMB":
            if d.pos and abs(d.pos[2] - (self.floor() + self.hop_z)) < 0.08:
                tx, ty = self.target_room()
                d.go(tx, ty, self.floor() + self.hop_z, force=True)
                self.set_phase("OUT", f"flying over to the target ({tx:+.2f}, {ty:+.2f})")
            elif t > 15.0:
                self.set_phase("OUT", "climb slow: going anyway")
        elif self.phase == "OUT":
            tx, ty = self.target_room()
            if d.pos and math.hypot(d.pos[0] - tx, d.pos[1] - ty) < 0.1:
                self.set_phase("DWELL", f"above the target: hovering {self.dwell:.0f} s")
            elif t > 30.0:
                self.set_phase("DWELL", "target not reached in 30 s: turning back")
        elif self.phase == "DWELL":
            if t > self.dwell:
                self.start_recall(d, "hop done: back to the ANT")
                self.set_phase("BACK", "returning, landing on the ANT")
        elif self.phase == "BACK":
            if d.mode == "IDLE" and not d.flying():
                f, x, y = self.start_map
                self.ant_goal("ant_map", x, y)
                self.set_phase("HOME", f"drone down: the ANT drives back to ({x:+.2f}, {y:+.2f}) map")
        elif self.phase == "HOME":
            if t > 2.0 and self.nav in ("succeeded", "aborted", "canceled", "stopped"):
                self.set_phase("DONE", f"ANT back (nav {self.nav}). g = new target, h = hop again")

    def fly_step(self, d, ant):
        """Flight legs are driven by extra_step (HOP mode): nothing to do per tick."""

    def all_down(self):
        """Nothing to resume (the HOME leg is started by the sequence)."""

    def on_stop_all(self):
        self.set_phase("IDLE", "aborted (x)")

    # ---------------- keys ----------------
    def key(self, k, old):
        d, ant = self.d, self.antv.fresh()
        if k == "g":
            s = ask(old, "target x y in the ANT MAP (ant1_rviz), Enter = cancel: ").split()
            try:
                self.target = ("ant_map", float(s[0]), float(s[1]))
                print(f"target ({self.target[1]:+.2f}, {self.target[2]:+.2f}) map. h = start the hop")
            except (ValueError, IndexError):
                print("need two numbers, e.g.  2.5 1.0")
        elif k == "h":
            why = None
            if self.target is None:
                why = "no target (g or rviz Publish Point)"
            elif self.antv.calib is None or ant is None:
                why = "no ANT data / calibration"
            elif d.flying() or d.mode in LANDING:
                why = "the drone is flying"
            elif not d.on_slot(ant, 0.25):
                why = "the drone isn't sitting on the ANT"
            elif self.phase not in ("IDLE", "DONE"):
                why = f"a hop is running ({self.phase})"
            if why:
                print(f"\nhop refused: {why}")
                return True
            sx, sy = self.antv.calib.room_to_map(ant[0], ant[1])
            self.start_map = ("ant_map", sx, sy)
            f, x, y = self.target
            self.ant_goal(f, x, y)
            self.set_phase("DRIVE", f"the ANT drives toward the target ({x:+.2f}, {y:+.2f}) {f}; start saved "
                                    f"({sx:+.2f}, {sy:+.2f}) map")
        elif k == "t":
            if self.phase != "READY":
                print(f"\nlaunch only when the ANT has arrived (phase {self.phase})")
            elif self.takeoff_all(old, [d]):
                self.set_phase("UP", "taking off from the ANT")
        return True

    def extra_status(self):
        s = f"HOP {self.phase}"
        if self.target:
            s += f" target ({self.target[1]:+.2f},{self.target[2]:+.2f}) {self.target[0]}"
            ant = self.antv.fresh()
            if ant and self.antv.calib:
                tx, ty = self.target_room()
                s += f" {math.hypot(tx - ant[0], ty - ant[1]):.2f} m from the ANT"
        return s + f" | nav {self.nav}"


def main():
    start(Hop, "\n\n".join(__doc__.split("\n\n")[2:]))
