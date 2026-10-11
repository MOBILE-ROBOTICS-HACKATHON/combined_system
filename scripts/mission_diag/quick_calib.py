#!/usr/bin/env python3
"""QUICK calibration IN PLACE (the ANT doesn't move), 2 runs, no assumption about the drone's heading:
    mission_calib_quick --reset   forget old points (new session / after a SLAM reset)
    mission_calib_quick front     drone sitting on the plate's FRONT edge, on the ANT's centre line, not flying
    mission_calib_quick back      drone moved by hand to the plate's BACK edge, centre line -> FIT -> SAVED
The two spots are 0.44 m apart along the ANT's own x axis (PLATE_OFFSET=0.22 m from the centre; change it if
your plate differs), which gives the rotation between the ANT map and the Lighthouse room.
    mission_calib_quick           (no word) = drone in the CENTRE; works too if the ANT is driven ~1 m between runs
ON THE FLOOR instead (11 Oct: on the plate the drone saw 0 Lighthouse stations, probably the ANT's lidars):
    mission_calib_quick floor-front   drone ON THE FLOOR touching the middle of the ANT's FRONT bumper
    mission_calib_quick floor-back    drone ON THE FLOOR touching the middle of the BACK bumper -> FIT -> SAVED
    (BUMPER_OFFSET=0.35 m = half the ANT length 0.30 + half the drone 0.05; ANT_HEIGHT=0.146 = plate above floor)
After run 1 alone it also saves a provisional calibration that assumes the drone's NOSE points to the ANT's
FRONT (11 Oct: a drone turned by 135 deg made the ANT turn on the spot instead of following; run 2 fixes that).
Each run averages 2.5 s; ant_link reloads data/mission_calib.yaml within 1 s."""
import math, os, sys, time
import numpy as np
import rclpy
import yaml
from geometry_msgs.msg import PoseStamped
from team_pkg.mission_common import CALIB_PATH, Calib, fit_2d, save_calib, wrap, yaw_of

POINTS = os.path.join(os.path.dirname(CALIB_PATH), "mission_calib_points.yaml")
OFF = float(os.environ.get("PLATE_OFFSET", "0.22"))
BUMP = float(os.environ.get("BUMPER_OFFSET", "0.35"))
spot = next((a for a in sys.argv[1:] if a in ("front", "back", "center", "centre", "floor-front", "floor-back")), "center")
off_x = {"front": OFF, "back": -OFF, "floor-front": BUMP, "floor-back": -BUMP}.get(spot, 0.0)
on_floor = spot.startswith("floor")
if "--reset" in sys.argv:
    if os.path.exists(POINTS):
        os.remove(POINTS)
    sys.exit("stored points forgotten: run mission_calib_quick with the drone on the ANT")

cf = os.environ.get("CF", "cf1")
rclpy.init(); n = rclpy.create_node("quick_calib"); A, D = [], []
n.create_subscription(PoseStamped, "/ant/pose_map", lambda m: A.append((m.pose.position.x, m.pose.position.y, yaw_of(m.pose.orientation))), 10)
n.create_subscription(PoseStamped, f"/{cf}/pose", lambda m: D.append((m.pose.position.x, m.pose.position.y, m.pose.position.z, yaw_of(m.pose.orientation))), 10)
t = time.time()
while time.time() - t < 2.5:
    rclpy.spin_once(n, timeout_sec=0.05)
if len(A) < 5 or len(D) < 5:
    sys.exit(f"no data: ANT poses {len(A)} (mission_link running?), {cf} poses {len(D)} (cf_server running?)")
a, d = np.array(A), np.array(D)
if np.ptp(a[:, :2], axis=0).max() > 0.02 or np.ptp(d[:, :3], axis=0).max() > 0.02:
    sys.exit("the ANT or the drone moved during the 2.5 s: hold still and run it again")
ax, ay = a[:, 0].mean(), a[:, 1].mean(); ayaw = math.atan2(np.sin(a[:, 2]).mean(), np.cos(a[:, 2]).mean())
dx, dy, dz = d[:, 0].mean(), d[:, 1].mean(), d[:, 2].mean(); dyaw = math.atan2(np.sin(d[:, 3]).mean(), np.cos(d[:, 3]).mean())
pts = yaml.safe_load(open(POINTS)) if os.path.exists(POINTS) else []
# where the DRONE is in the map = ANT centre + its spot on the plate (along the ANT's x axis)
pts.append({"map": [float(ax + off_x * math.cos(ayaw)), float(ay + off_x * math.sin(ayaw))],
            "room": [float(dx), float(dy)], "z": float(dz), "spot": spot, "floor": on_floor})
os.makedirs(os.path.dirname(POINTS), exist_ok=True)
yaml.safe_dump(pts, open(POINTS, "w"))
ANT_H = float(os.environ.get("ANT_HEIGHT", "0.146"))
floor = lambda z: float(z - ANT_H)
# heights: from plate spots directly, from floor spots + the ANT's height
plates = [p["z"] if not p.get("floor") else p["z"] + ANT_H for p in pts]
spread = max(math.dist(p["map"], q["map"]) for p in pts for q in pts)
if len(pts) >= 2 and spread >= 0.35:
    th, tx, ty, rms = fit_2d([p["room"] for p in pts], [p["map"] for p in pts])
    plate = float(np.median(plates))
    cal = Calib(theta=th, tx=tx, ty=ty, plate_z=plate, floor_z=floor(plate), rms=rms, n=len(pts),
                stamp=time.strftime("%Y-%m-%d %H:%M:%S"), method=f"quick {len(pts)}-point (no heading assumption)")
    save_calib(cal)
    print(f"point {len(pts)} stored. SAVED {CALIB_PATH}: {cal.summary()}")
    moved = max(math.dist(p["room"], q["room"]) for p in pts for q in pts)
    print(f"  the drone moved {moved:.2f} m between the spots (expected {spread:.2f} m): "
          + ("GOOD" if rms < 0.05 else "MISMATCH: was it really on the centre line, front/back edge? --reset and redo"))
else:
    th = wrap(ayaw - dyaw); c, s = math.cos(th), math.sin(th)
    mx, my = pts[-1]["map"]
    cal = Calib(theta=float(th), tx=float(mx - (c * dx - s * dy)), ty=float(my - (s * dx + c * dy)), plate_z=float(plates[-1]),
                floor_z=floor(plates[-1]), rms=None, n=1, stamp=time.strftime("%Y-%m-%d %H:%M:%S"),
                method="quick snapshot (assumes nose = ANT front)")
    save_calib(cal)
    print(f"point {len(pts)} stored ({spot}). PROVISIONAL calibration saved (assumes the drone's NOSE = ANT front).")
    nxt = {"front": "the plate's BACK edge -> mission_calib_quick back", "floor-front": "the floor at the BACK bumper -> mission_calib_quick floor-back",
           "back": "the plate's FRONT edge -> mission_calib_quick front", "floor-back": "the floor at the FRONT bumper -> mission_calib_quick floor-front"}
    print(f"  Now move the drone by hand to {nxt.get(spot, 'another spot (front/back), or drive the ANT ~1 m')}")
