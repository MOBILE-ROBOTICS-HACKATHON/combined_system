"""mission_calib: find how the drone's frame (Lighthouse room) and the ANT's SLAM map relate.

    mission_calib                               (helper from scripts/cf_env.sh; needs cf_server + ant_link)
    = ros2 run team_pkg mission_calib --ros-args -p cf:=cf1 [-p ant_height:=0.146 -p offset_x:=0.0 -p offset_y:=0.0]

SETUP: the drone sits (NOT flying, cf_server connected) CENTRED on the ANT's top plate, nose = ANT forward.
Then drive the ANT IN STEPS with ant1_keys: ~40 cm, STOP 2 s, again ... 4-6 stops over >= 1 m, with a turn.
At every stop this tool records one pair (drone position in the room, drone position in the ANT map) and
fits   map = R(theta) * room + t.  Only STOPS count: while driving, the two positions arrive with different
delays (a few cm apart), which tilted the fit by 2-3 deg in the test.

  w   fit + SAVE (needs >= 3 stops spread over >= 1 m; check the RMS: < 3 cm good, > 5 cm redo)
  s   SNAPSHOT instead: one pose pair, uses the drone's yaw (less accurate: needs the nose exactly forward)
  r   restart the recording            q   quit (nothing saved unless you pressed w / s)

Saved to data/mission_calib.yaml; ant_link reloads it within 1 s. Also writes data/mission_calib.png:
the ANT's SLAM path vs the Lighthouse-measured path (= how accurate SLAM was on this drive).
Redo after a SLAM reset (the map frame moves) or if the ANT/drone positions don't line up in mission_rviz.
"""
import math
import os
import select
import signal
import sys
import termios
import time
import tty
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.signals import SignalHandlerOptions
from crazyflie_interfaces.msg import Status
from geometry_msgs.msg import PoseStamped
from team_pkg.mission_common import CALIB_PATH, Calib, fit_2d, save_calib, wrap, yaw_of

MIN_SPAN = 1.0      # m: the stops must spread at least this far
MIN_STOPS = 3
STILL = 0.6         # s the ANT must stand still (< 5 mm) for a stop to count
NEW_STOP = 0.15     # m from the previous stop for a new one


class CalibNode(Node):
    def __init__(self):
        super().__init__("mission_calib")
        cf = self.declare_parameter("cf", "cf1").value
        self.ant_height = self.declare_parameter("ant_height", 0.146).value
        self.off = (self.declare_parameter("offset_x", 0.0).value, self.declare_parameter("offset_y", 0.0).value)
        self.create_subscription(PoseStamped, "/ant/pose_map", self.on_ant, 10)
        self.create_subscription(PoseStamped, f"/{cf}/pose", self.on_cf, 10)
        self.create_subscription(Status, f"/{cf}/status", lambda m: setattr(self, "status", m), 10)
        self.ant = self.cf = self.status = None
        self.ant_at = self.cf_at = 0.0
        self.reset()

    def reset(self):
        self.room, self.mapp, self.zs, self.yaws, self.recent = [], [], [], [], []

    @property
    def path(self):
        """Largest distance between two stops (m)."""
        return max((math.dist(a, b) for a in self.mapp for b in self.mapp), default=0.0)

    def on_ant(self, m):
        self.ant, self.ant_at = (m.pose.position.x, m.pose.position.y, yaw_of(m.pose.orientation)), time.monotonic()

    def on_cf(self, m):
        p = m.pose.position
        self.cf, self.cf_at = (p.x, p.y, p.z, yaw_of(m.pose.orientation)), time.monotonic()

    def flying(self):
        return bool(self.status and self.status.supervisor_info & Status.SUPERVISOR_INFO_IS_FLYING)

    def drone_in_map(self):
        """Where the drone sits, in the ANT map: ANT centre + plate offset turned with the ANT."""
        x, y, yaw = self.ant
        ox, oy = self.off
        return x + math.cos(yaw) * ox - math.sin(yaw) * oy, y + math.sin(yaw) * ox + math.cos(yaw) * oy

    def problem(self):
        now = time.monotonic()
        if self.ant is None or now - self.ant_at > 1.0:
            return "no ANT pose (is ant_link running and the ANT connected?)"
        if self.cf is None or now - self.cf_at > 1.0:
            return "no drone pose (is cf_server running?)"
        if self.flying():
            return "the drone is FLYING: it must sit on the ANT"
        return None

    def sample(self):
        """~5 Hz: keep the last STILL seconds; when the ANT stood still all that time and is >= NEW_STOP
        from the previous stop, store ONE pair (averaged over the window)."""
        if self.problem():
            self.recent = []
            return
        now = time.monotonic()
        self.recent = [r for r in self.recent if now - r[0] <= STILL] + [(now, self.drone_in_map(), self.cf[:3], self.ant[2])]
        if now - self.recent[0][0] < STILL - 0.25:
            return
        mps = np.array([r[1] for r in self.recent])
        if np.ptp(mps, axis=0).max() > 0.005:            # still moving
            return
        mp = tuple(mps.mean(0))
        if self.mapp and min(math.dist(mp, m) for m in self.mapp) < NEW_STOP:
            return
        cfs = np.array([r[2] for r in self.recent]).mean(0)
        self.room.append(tuple(cfs[:2]))
        self.mapp.append(mp)
        self.zs.append(float(cfs[2]))
        self.yaws.append(self.recent[-1][3])
        print(f"\nstop {len(self.mapp)} recorded at map ({mp[0]:+.2f}, {mp[1]:+.2f})")

    def calib(self, theta, tx, ty, rms, n, method):
        plate = float(np.median(self.zs)) if self.zs else self.cf[2]
        return Calib(theta=float(theta), tx=float(tx), ty=float(ty), plate_z=plate, floor_z=plate - self.ant_height,
                     rms=None if rms is None else float(rms), n=n, stamp=time.strftime("%Y-%m-%d %H:%M:%S"),
                     method=method)

    def fit(self):
        th, tx, ty, rms = fit_2d(self.room, self.mapp)
        return self.calib(th, tx, ty, rms, len(self.room), f"{len(self.room)} stops over {self.path:.1f} m")

    def snapshot(self):
        p = self.problem()
        if p:
            return None, p
        th = wrap(self.ant[2] - self.cf[3])           # nose = ANT forward: yaw_map = yaw_room + theta
        mx, my = self.drone_in_map()
        c, s = math.cos(th), math.sin(th)
        rx, ry = self.cf[:2]
        self.zs = [self.cf[2]]
        return self.calib(th, mx - (c * rx - s * ry), my - (s * rx + c * ry), None, 1, "snapshot"), None

    def status_line(self):
        p = self.problem()
        s = f"stops {len(self.room)}, spread {self.path:.2f} m"
        if self.yaws:
            s += f", turned {math.degrees(max(self.yaws) - min(self.yaws)):.0f} deg"
        if len(self.room) >= 2 and self.path > 0.3:
            c = self.fit()
            s += f" | preview: theta {math.degrees(c.theta):+.1f} deg, RMS {c.rms * 100:.1f} cm"
        if p:
            s += f" | WAITING: {p}"
        elif self.path < MIN_SPAN or len(self.room) < MIN_STOPS:
            s += f" | drive ~40 cm, STOP 2 s, repeat (need {MIN_STOPS}+ stops over {MIN_SPAN} m, with a turn)"
        else:
            s += " | enough: press w to fit + save"
        return s


def report(node, c):
    """Residuals and a picture: the ANT's SLAM path vs the Lighthouse path converted into the map."""
    room = np.array(node.room)
    mapped = np.array([c.room_to_map(*r) for r in room])
    slam = np.array(node.mapp)
    err = np.linalg.norm(mapped - slam, axis=1)
    print(f"SLAM vs Lighthouse at {len(room)} stops: mean {err.mean() * 100:.1f} cm, max {err.max() * 100:.1f} cm")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.plot(slam[:, 0], slam[:, 1], "o-", label="ANT SLAM pose at the stops (map)")
        ax.plot(mapped[:, 0], mapped[:, 1], "x--", label="drone Lighthouse pose, converted to map")
        ax.set_aspect("equal")
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        ax.set_title(f"calibration drive: RMS {c.rms * 100:.1f} cm, max {err.max() * 100:.1f} cm")
        ax.legend()
        out = os.path.join(os.path.dirname(CALIB_PATH), "mission_calib.png")
        fig.savefig(out, dpi=120)
        print(f"picture: {out}")
    except Exception as e:   # the picture is a bonus, the calibration is already saved
        print(f"(no picture: {e})")


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
    node = CalibNode()
    print(__doc__.split("\n\n")[2])
    print(__doc__.split("\n\n")[3])
    old = termios.tcgetattr(sys.stdin) if sys.stdin.isatty() else None
    last_s = last_p = 0.0
    try:
        if old:
            tty.setcbreak(sys.stdin.fileno())
        while rclpy.ok():
            k = read_key(0.05) if old else time.sleep(0.05)
            for _ in range(30):                          # ALL waiting messages, not one per loop (fell behind at ~20/s)
                rclpy.spin_once(node, timeout_sec=0)
            now = time.monotonic()
            if now - last_s > 0.2:
                node.sample()
                last_s = now
            if now - last_p > 0.5:
                print(f"\r{node.status_line()[:150]:<150}", end="", flush=True)
                last_p = now
            if k == "r":
                node.reset()
                print("\nrecording restarted")
            elif k == "w":
                if node.path < MIN_SPAN or len(node.room) < MIN_STOPS:
                    print(f"\nnot yet: {len(node.room)} stops over {node.path:.2f} m; need {MIN_STOPS}+ over {MIN_SPAN} m")
                    continue
                c = node.fit()
                save_calib(c)
                print(f"\nSAVED {CALIB_PATH}: {c.summary()}")
                print("  " + ("GOOD" if c.rms < 0.03 else "OK" if c.rms < 0.05 else "BAD: redo slower, drone centred, Lighthouse in view"))
                report(node, c)
            elif k == "s":
                c, why = node.snapshot()
                if c is None:
                    print(f"\nsnapshot not possible: {why}")
                else:
                    save_calib(c)
                    print(f"\nSAVED snapshot {CALIB_PATH}: {c.summary()}\n  (less accurate than a drive: needs the nose exactly forward)")
            elif k == "q":
                break
    except KeyboardInterrupt:
        pass
    finally:
        if old:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old)
        print()
        node.destroy_node()
        rclpy.try_shutdown()
