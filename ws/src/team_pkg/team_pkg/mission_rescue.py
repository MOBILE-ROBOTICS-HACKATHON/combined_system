"""RESCUE DEMO, mission_rescue: drones fly and report their position to the ANT every 2 s (beacons). You
simulate a MALFUNCTION: one drone goes silent and starts to "fall". The ANT notices the missing beacon,
speeds up (0.5 m/s), finds the drone WITH THE D555 (vision_link), drives under it and catches it on its plate.

    mission_rescue                              (helper from scripts/cf_env.sh; CF_DRONES=2 for two drones;
    = ros2 run team_pkg mission_rescue --ros-args -p cfs:=cf1[,cf2] -p waypoints:="1,0; 1,1; 0,0.5"   needs
        [-p rescue_speed:=0.5 -p fall_speed:=0.10 -p wait_max:=30.0 -p beacon_period:=2.0]   vision_link too)

Everything of mission_scout works here too: t takeoff, e/s/f/o formations, r recall, p patrol, x, q, 1/2.
  m   MALFUNCTION (with 1/2 first when two drones: "2 m"): that drone's beacon STOPS, it holds 2 s, then
      "falls" at fall_speed (10 cm/s) down to 0.30 m above the ANT's plate height and hovers there waiting.
      (It is simulated: we still control it; cf_guard still guards it.)
The ANT side (uses ONLY beacons + vision to find the drone, never its Lighthouse position):
  SIGNAL_LOST  no beacon for 2 periods + 0.5 s: patrol held, Nav2 boosted to rescue_speed + 0.1 m goal tolerance
  SEARCH       turn on the spot to face the drone's LAST beacon position (to bring it into the camera's view);
               after 8 s without a sighting: drive toward that last beacon position
  CHASE        vision sees it: drive to the point under it (goal re-sent when the fix moves > 0.15 m)
  UNDER        the ANT arrived under the last vision fix: tell the drone "the ANT is here"
  CATCH        the drone (which knows its own position) steers the last <= 0.3 m onto its slot at 5 cm/s,
               lands on the plate, disarms
  DONE         Nav2 limits reset, patrol resumes, times printed + appended to data/rescue_log.csv
If the ANT isn't under it within wait_max (30 s) of reaching its wait height, the drone lands on the FLOOR.
x stops everything (ANT stops, drones land where they are, Nav2 limits reset).
"""
import csv
import json
import math
import os
import time
from std_msgs.msg import String
from team_pkg.mission_fleet import start
from team_pkg.mission_scout import Scout

FAULT = ("FAULT_HOLD", "FAULT_FALL", "FAULT_WAIT", "CATCH")
LOG = "/hackathon/data/rescue_log.csv"


class Rescue(Scout):
    def __init__(self):
        super().__init__()
        self.rescue_speed = self.declare_parameter("rescue_speed", 0.5).value
        self.fall_speed = self.declare_parameter("fall_speed", 0.10).value
        self.wait_max = self.declare_parameter("wait_max", 30.0).value
        self.period = self.declare_parameter("beacon_period", 2.0).value
        self.beacon_pub = self.create_publisher(String, "/fleet/beacon", 10)
        self.params_pub = self.create_publisher(String, "/ant/nav_params", 10)
        self.create_subscription(String, "/fleet/beacon", self.on_beacon, 10)
        self.create_subscription(String, "/vision/drones", self.on_vision, 10)
        self.silent = set()                      # drones whose beacon we switched off (the malfunction)
        self.beacons = {}                        # what the ANT side KNOWS: name -> (x, y, z, flying, time)
        self.vision = {}                         # name -> (x, y, z, time) last vision fix of that drone
        self.vision_link_at = 0.0
        self.r = None                            # the running rescue (dict)
        self.create_timer(self.period, self.send_beacons)

    # ---------------- beacons (drone side -> ANT side) ----------------
    def send_beacons(self):
        for d in self.drones:
            if d.name in self.silent or not d.pos or time.monotonic() - d.pos_at > 1.0:
                continue
            self.beacon_pub.publish(String(data=json.dumps({
                "name": d.name, "x": round(d.pos[0], 3), "y": round(d.pos[1], 3), "z": round(d.pos[2], 3),
                "flying": d.flying(), "bat": round(d.status.battery_voltage, 2) if d.status else None, "t": time.time()})))

    def on_beacon(self, m):
        b = json.loads(m.data)
        self.beacons[b["name"]] = (b["x"], b["y"], b["z"], b["flying"], time.monotonic())

    def on_vision(self, m):
        self.vision_link_at = time.monotonic()
        for t in json.loads(m.data)["tracks"]:
            if t["name"] != "unknown" and t["age"] < 0.5:
                self.vision[t["name"]] = (t["x"], t["y"], t["z"], time.monotonic())

    # ---------------- the drone's (simulated) malfunction ----------------
    def start_recall(self, d, why):
        if d.mode in FAULT:                      # a "broken" drone doesn't come home by itself
            return
        super().start_recall(d, why)

    def fly_step(self, d, ant):
        if d.mode not in FAULT:
            return super().fly_step(d, ant)
        t = time.monotonic() - d.state_at
        wait_z = ant[2] + 0.30                   # 0.30 m above the plate height (in the drone frame)
        if d.mode == "FAULT_HOLD" and t > 2.0:
            d.fault_xy = d.pos[:2]
            d.go(d.fault_xy[0], d.fault_xy[1], wait_z, min_secs=max((d.pos[2] - wait_z) / self.fall_speed, 1.0), force=True)
            d.set_mode("FAULT_FALL", f"falling at {self.fall_speed * 100:.0f} cm/s to z {wait_z:.2f}")
        elif d.mode == "FAULT_FALL" and d.pos and abs(d.pos[2] - wait_z) < 0.05:
            d.set_mode("FAULT_WAIT", f"hovering at z {wait_z:.2f}, waiting for the ANT (max {self.wait_max:.0f} s)")
        elif d.mode in ("FAULT_FALL", "FAULT_WAIT"):
            sx, sy = d.slot_world(ant)
            under = math.hypot(d.pos[0] - sx, d.pos[1] - sy)
            if under < 0.30 and not self.antv.moving and self.r and self.r["state"] == "UNDER":
                self.owner, self.hold = d, True
                secs = max(math.dist(d.pos, (sx, sy, ant[2] + 0.15)) / 0.05, 3.0)
                d.go(sx, sy, ant[2] + 0.15, min_secs=secs, force=True)
                d.catch_at = time.monotonic() + secs + 1.0
                d.set_mode("CATCH", f"the ANT is under me ({under * 100:.0f} cm): steering onto my slot, 5 cm/s")
            elif d.mode == "FAULT_WAIT" and t > self.wait_max:
                self.land_here(d, f"the ANT didn't come within {self.wait_max:.0f} s: landing on the floor")
        elif d.mode == "CATCH" and time.monotonic() > d.catch_at:
            d.land(ant[2], 3.0)
            d.set_mode("LAND", "touching down on the plate")

    def malfunction(self, d):
        if not d.flying() or d.mode not in ("HOLD", "ESCORT", "SCOUT", "ROAM"):
            d.say("malfunction only while it flies a formation (t, then e/s/f/o)")
            return
        if self.r:
            d.say(f"one rescue at a time ({self.r['name']} is being rescued)")
            return
        self.silent.add(d.name)
        d.set_mode("FAULT_HOLD", "SIMULATED MALFUNCTION: beacon OFF, holding 2 s, then falling")
        d.go(*d.pos, force=True)
        d.fault_t = time.monotonic()

    # ---------------- the ANT side: rescue ----------------
    def r_state(self, s, why=""):
        self.r["state"], self.r["at"] = s, time.monotonic()
        self.r.setdefault("times", {})[s] = round(time.monotonic() - self.r["lost"], 1)
        print(f"\n[RESCUE {self.r['name']} {s}] t+{self.r['times'][s]:.1f} s {why}", flush=True)

    def ant_goal_world(self, x, y, yaw=None):
        from geometry_msgs.msg import PoseStamped
        g = PoseStamped()
        g.header.frame_id = "world"
        g.pose.position.x, g.pose.position.y = float(x), float(y)
        if yaw is not None:
            g.pose.orientation.z, g.pose.orientation.w = math.sin(yaw / 2), math.cos(yaw / 2)
        self.goal_pub.publish(g)

    def extra_step(self):
        if self.r:                                            # the patrol waits during a rescue
            if self.patrol:
                self.patrol_was, self.patrol = True, False
        else:
            super().extra_step()
        now = time.monotonic()
        ant = self.antv.fresh(2.0)
        if self.r is None:                                    # watch the beacons
            for name, b in self.beacons.items():
                if b[3] and now - b[4] > 2 * self.period + 0.5:
                    self.r = {"name": name, "lost": b[4] + self.period, "last": b[:3], "goal": None, "goal_at": 0.0}
                    self.ant_stop("rescue: stopping the patrol leg")
                    self.params_pub.publish(String(data=f"speed={self.rescue_speed} turn=0.6 tol=0.1"))
                    self.r_state("SIGNAL_LOST", f"no beacon from {name} for {now - b[4]:.1f} s: Nav2 boosted to "
                                                f"{self.rescue_speed} m/s, last beacon ({b[0]:+.2f}, {b[1]:+.2f}, {b[2]:.2f})")
                    break
            return
        r, d = self.r, next((x for x in self.drones if x.name == self.r["name"]), None)
        if ant is None:
            return
        fix = self.vision.get(r["name"])
        fresh = fix and now - fix[3] < 1.0
        t = now - r["at"]
        if r["state"] == "SIGNAL_LOST":
            lx, ly = r["last"][:2]
            self.ant_goal_world(ant[0], ant[1], math.atan2(ly - ant[1], lx - ant[0]))
            self.r_state("SEARCH", "turning to face the last beacon position")
        elif r["state"] == "SEARCH":
            if fresh:
                r["seen"] = fix
                self.r_state("CHASE", f"SEEN by the D555 at ({fix[0]:+.2f}, {fix[1]:+.2f}, {fix[2]:.2f})")
            elif t > 8.0 and not r.get("blind"):
                r["blind"] = True
                self.ant_goal_world(*r["last"][:2])
                print("\n[RESCUE] not seen in 8 s: driving toward the last beacon position (vision may pick it up)", flush=True)
            elif r.get("blind") and self.nav == "succeeded":
                r["target"] = r["last"][:2]
                self.r_state("UNDER", "at the last beacon position (found by beacon, NOT by vision)")
        elif r["state"] == "CHASE":
            if fresh:
                r["seen"] = fix
            tx, ty = r["seen"][:2]
            if (r["goal"] is None or math.hypot(tx - r["goal"][0], ty - r["goal"][1]) > 0.15) and now - r["goal_at"] > 1.0:
                self.ant_goal_world(tx, ty, math.atan2(ty - ant[1], tx - ant[0]))
                r["goal"], r["goal_at"] = (tx, ty), now
            dist = math.hypot(tx - ant[0], ty - ant[1])
            if dist < 0.30 and not self.antv.moving and now - r["goal_at"] > 1.5:
                r["target"] = (tx, ty)
                if d and d.pos:                       # evaluation only: how far off was the vision fix?
                    r["err"] = round(math.hypot(tx - d.pos[0], ty - d.pos[1]) * 100)
                self.r_state("UNDER", f"the ANT is {dist * 100:.0f} cm from the last vision fix")
        elif r["state"] == "UNDER":
            if d and d.mode in ("CATCH", "LAND", "DISARM", "IDLE", "LAND_FLOOR", "GUARD"):
                self.r_state("CATCH", f"{d.name} is coming down onto the plate")
        if r["state"] in ("UNDER", "CATCH") and d and d.mode in ("IDLE", "DISARM") and not d.flying():
            self.finish(d, ant)
        elif d and d.mode in ("LAND_FLOOR", "GUARD") and not d.flying() and r["state"] != "DONE":
            self.finish(d, ant, failed=f"{d.name} landed on the floor ({d.mode})")

    def finish(self, d, ant, failed=None):
        r = self.r
        on = d.on_slot(ant, 0.25) if d.pos else False
        self.r_state("DONE" if not failed else "FAILED", failed or ("CAUGHT ON THE PLATE" if on else "landed, NOT on the plate"))
        self.params_pub.publish(String(data="reset"))
        self.silent.discard(d.name)
        self.beacons.pop(d.name, None)                # its old "flying" beacon must not start a new rescue
        self.vision.pop(d.name, None)
        err = r.get("err")
        tm = r["times"]
        print(f"\n[RESCUE RESULT] {d.name}: signal lost -> seen {tm.get('CHASE', '-')} s -> ANT under {tm.get('UNDER', '-')} s "
              f"-> down {tm.get('DONE', tm.get('FAILED'))} s | found by {'VISION' if 'CHASE' in tm else 'last beacon'}"
              + (f", vision fix {err} cm from where the drone really was when the ANT stopped" if err is not None else "")
              + f" | {'ON THE PLATE' if on and not failed else 'NOT caught'}", flush=True)
        os.makedirs(os.path.dirname(LOG), exist_ok=True)
        new = not os.path.exists(LOG)
        with open(LOG, "a", newline="") as f:
            w = csv.writer(f)
            if new:
                w.writerow(["time", "drone", "seen_s", "under_s", "down_s", "found_by", "vision_err_cm", "result"])
            w.writerow([time.strftime("%H:%M:%S"), d.name, tm.get("CHASE"), tm.get("UNDER"), tm.get("DONE", tm.get("FAILED")),
                        "vision" if "CHASE" in tm else "beacon", err, "caught" if on and not failed else (failed or "missed")])
        self.r = None
        if not self.busy():
            self.hold = False
            self.all_down()

    def on_stop_all(self):
        super().on_stop_all()
        if self.r:
            self.params_pub.publish(String(data="reset"))
            self.silent.discard(self.r["name"])
            print(f"\n[RESCUE {self.r['name']}] ABORTED (x)", flush=True)
            self.r = None

    # ---------------- keys / status ----------------
    def key(self, k, old):
        if k == "m":
            which = self.sel or (self.drones if len(self.drones) == 1 else None)
            self.sel = None
            if not which:
                print("\nwhich drone? press 1 or 2 first, then m")
                return True
            self.malfunction(which[0])
            return True
        return super().key(k, old)

    def extra_status(self):
        s = super().extra_status()
        now = time.monotonic()
        s += " | beacons " + " ".join(f"{n}:{now - b[4]:.0f}s" for n, b in sorted(self.beacons.items()))
        if now - self.vision_link_at > 2.0:
            s += " | NO vision_link"
        if self.r:
            s += f" | RESCUE {self.r['name']} {self.r['state']}"
        return s


def main():
    start(Rescue, "\n\n".join(__doc__.split("\n\n")[2:]))
