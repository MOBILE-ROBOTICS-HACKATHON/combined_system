"""METHOD 2, mission_scout: the ANT patrols waypoints (Nav2, avoids obstacles) while ONE OR TWO drones fly by
themselves: around the ANT (ESCORT), ahead of it (SCOUT), or anywhere in an area, independent of the ANT
(ROAM). When you call them (or a battery runs low) they come back and land on the ANT, one at a time;
the ANT pauses only for the touchdowns.

    mission_scout                               (helper from scripts/cf_env.sh; CF_DRONES=2 for two drones)
    = ros2 run team_pkg mission_scout --ros-args -p cfs:=cf1[,cf2] -p waypoints:="1.0,0.0; 1.0,1.0; 0.0,0.5"
        [-p area:="xmin,xmax,ymin,ymax"] [-p speed:=0.15 -p recall_v:=3.55 -p max_flight:=240.0]

  t   take off (type FLY): every drone sitting on the standing ANT, cf2 4 s after cf1
  e   ESCORT: random points 0.6-1.1 m around the ANT's CURRENT position (each drone in its own half)
  s   SCOUT: 1.0 m ahead of the ANT (direction of travel); two drones side by side, 0.7 m apart
  f   ROAM (free): random points in the AREA, NOT related to the ANT at all. Area = -p area:=... (drone
      frame, read it in mission_rviz), else 0.8 m around where that drone took off. New point when reached / 8 s.
  o   HOLD: hover where it is        r   RECALL: come back and land on the ANT
  1 / 2   the NEXT key (t e s f o r) only for cf1 / cf2, e.g.  1 e  2 f  = cf1 escorts, cf2 roams
  p   ANT patrol ON / OFF (starts OFF; held automatically while drones land)
  x   STOP EVERYTHING: ANT stops, patrol OFF, every drone lands where it is (plate if above the ANT, else floor)
  q   recall every flying drone, quit when all are down          rviz "Publish Point" = new patrol waypoint
Heights (every formation): one drone 0.6 m above the floor; two drones 0.55 m (cf1) and 0.90 m (cf2), so
their paths may cross 35 cm apart vertically; ROAM targets also keep >= 0.6 m between them.
If cf_guard lands/kills a drone (fence, leash, battery, SPACE) the mission stops commanding it (GUARD).
KEEP THE ANT SLOWER THAN THE DRONES: ant1_nav_speed 0.1 0.3 --live.
Recall triggers per drone: r, battery < recall_v for 3 s, flight longer than max_flight, q.
"""
import math
import random
import time
from geometry_msgs.msg import PointStamped
from team_pkg.mission_fleet import CRUISE, LANDING, Fleet, start


class Scout(Fleet):
    def __init__(self):
        super().__init__("mission_scout")
        wps = self.declare_parameter("waypoints", "").value
        self.waypoints = [("ant_map", float(a), float(b)) for a, b in
                          (w.split(",") for w in wps.replace(" ", "").split(";") if w)]
        a = self.declare_parameter("area", "").value
        self.area = [float(v) for v in a.split(",")] if a else None
        self.create_subscription(PointStamped, "/clicked_point", self.on_click, 10)
        self.patrol, self.patrol_was, self.wp, self.wp_sent, self.wp_tries, self.wp_at = False, False, 0, False, 0, 0.0
        self.sel, self.last_ant, self.ant_dir = None, None, None

    def on_click(self, m):
        self.waypoints.append((m.header.frame_id or "world", m.point.x, m.point.y))
        print(f"\n[ANT] waypoint {len(self.waypoints)} added from rviz: ({m.point.x:+.2f}, {m.point.y:+.2f}) {m.header.frame_id}")

    # ---------------- ANT patrol ----------------
    def extra_step(self):
        if self.busy() and self.patrol:                       # drones landing: hold the patrol
            self.patrol_was, self.patrol = True, False
        if not (self.patrol and self.waypoints) or self.hold:
            return
        n = len(self.waypoints)
        if not self.wp_sent:
            f, x, y = self.waypoints[self.wp % n]
            self.ant_goal(f, x, y)
            self.wp_sent, self.wp_at = True, time.monotonic()
            print(f"\n[ANT] patrol -> waypoint {self.wp % n + 1}/{n} ({x:+.2f}, {y:+.2f}) {f}")
            return
        if time.monotonic() - self.wp_at < 2.0:
            return
        if self.nav == "succeeded":
            self.wp, self.wp_sent, self.wp_tries = self.wp + 1, False, 0
        elif self.nav in ("aborted", "canceled", "stopped", "idle"):
            self.wp_tries += 1
            if self.wp_tries > 2:
                print(f"\n[ANT] couldn't reach waypoint {self.wp % n + 1}: skipping it")
                self.wp, self.wp_tries = self.wp + 1, 0
            self.wp_sent = False

    def all_down(self):
        if self.patrol_was:
            self.patrol, self.patrol_was, self.wp_sent = True, False, False
            print("\n[ANT] patrol RESUMES")

    def on_stop_all(self):
        self.patrol = self.patrol_was = False

    # ---------------- formations ----------------
    def ant_heading(self, ant):
        if self.last_ant is None:
            self.last_ant = ant
        elif math.hypot(ant[0] - self.last_ant[0], ant[1] - self.last_ant[1]) > 0.05:
            self.ant_dir = math.atan2(ant[1] - self.last_ant[1], ant[0] - self.last_ant[0])
            self.last_ant = ant
        return self.ant_dir if (self.antv.moving and self.ant_dir is not None) else ant[3]

    def clear_of_others(self, d, p, gap=0.6):
        for o in self.drones:
            if o is d:
                continue
            for q in (o.pos, o.target):
                if q and o.flying() and math.hypot(p[0] - q[0], p[1] - q[1]) < gap:
                    return False
        return True

    def fly_step(self, d, ant):
        floor, n, now = self.floor(), len(self.drones), time.monotonic()
        z = floor + CRUISE[n][d.idx]                          # this drone's altitude layer
        reached = d.target and d.pos and math.dist(d.pos, d.target) < 0.1
        if d.mode == "ESCORT":
            if d.aim is None or reached or now - d.aim_at > 8.0:
                base = d.idx * 2 * math.pi / n                # each drone in its own sector around the ANT
                d.aim = (random.uniform(0.6, 1.1), base + random.uniform(0, 2 * math.pi / n))
                d.aim_at = now
            r, a = d.aim                                      # relative to the ANT: follows it while it drives
            d.go(ant[0] + r * math.cos(a), ant[1] + r * math.sin(a), z)
        elif d.mode == "SCOUT":
            h = self.ant_heading(ant)
            lat = (d.idx - (n - 1) / 2) * 0.7                 # side by side
            d.go(ant[0] + math.cos(h) - lat * math.sin(h), ant[1] + math.sin(h) + lat * math.cos(h), z)
        elif d.mode == "ROAM":
            if d.aim is None or reached or now - d.aim_at > 8.0:
                if self.area:
                    x0, x1, y0, y1 = self.area[0] + 0.3, self.area[1] - 0.3, self.area[2] + 0.3, self.area[3] - 0.3
                else:
                    c = d.home or d.pos
                    x0, x1, y0, y1 = c[0] - 0.8, c[0] + 0.8, c[1] - 0.8, c[1] + 0.8
                for _ in range(30):                           # a free spot >= 0.6 m from the other drone
                    p = (random.uniform(x0, x1), random.uniform(y0, y1))
                    if self.clear_of_others(d, p):
                        break
                d.aim, d.aim_at = (p[0], p[1], z), now
            d.go(*d.aim)

    # ---------------- keys ----------------
    def key(self, k, old):
        if k in "12" and len(k) == 1:
            i = int(k) - 1
            if i < len(self.drones):
                self.sel = [self.drones[i]]
                print(f"\nnext command only for {self.drones[i].name}")
            return True
        which, self.sel = self.sel or self.drones, None
        if k == "t":
            self.takeoff_all(old, which)
        elif k in "esfo" and len(k) == 1:
            mode = {"e": "ESCORT", "s": "SCOUT", "f": "ROAM", "o": "HOLD"}[k]
            if mode != "HOLD" and self.antv.calib is None:
                print("\nno calibration: run mission_calib first")
                return True
            for d in which:
                if not d.flying() or d.mode in LANDING or d.mode == "TAKEOFF":
                    d.say(f"{mode} only while flying and not landing")
                    continue
                d.aim = None
                d.set_mode(mode)
                if mode == "HOLD" and d.pos:
                    d.go(*d.pos, force=True)
        elif k == "r":
            for d in which:
                self.start_recall(d, "r pressed")
        elif k == "p":
            if self.busy():
                print("\n[ANT] patrol is held while drones land; it resumes by itself")
            elif not self.waypoints:
                print("\n[ANT] no waypoints: -p waypoints:=\"x1,y1; x2,y2\" (ANT map) or rviz Publish Point")
            else:
                self.patrol, self.wp_sent = not self.patrol, False
                if not self.patrol:
                    self.ant_stop("patrol OFF")
                print(f"\n[ANT] patrol {'ON' if self.patrol else 'OFF'}")
        return True

    def extra_status(self):
        if not self.waypoints:
            return "no waypoints"
        n = len(self.waypoints)
        return f"patrol {'ON' if self.patrol else 'HELD' if self.patrol_was else 'OFF'} wp {self.wp % n + 1}/{n} nav {self.nav}"


def main():
    start(Scout, "\n\n".join(__doc__.split("\n\n")[2:]))
