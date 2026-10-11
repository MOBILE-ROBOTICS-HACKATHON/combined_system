"""vision_calib: measure where the D555 really sits on the ANT (position + angles), from flying drones.

    vision_calib                                (helper from scripts/cf_env.sh; needs cf_server, ant_link,
    = ros2 run team_pkg vision_calib --ros-args -p cfs:=cf1   vision_link and jetson_vision running)

SETUP: the ANT stands STILL. Fly a drone (cf_keys) in front of the ANT, 0.8-3 m away, at different heights
and sideways positions (left / centre / right, low / high: ~8-12 places, HOVER 2 s at each). The drone's true
position comes from Lighthouse; vision_link reports what the camera sees. Only while the drone HOVERS STILL
(< 3 cm in 0.6 s: moving pairs blur the timing) and exactly ONE candidate is near where the current mount
guess expects it, the frames are averaged into one pair (camera point, true point in the ANT frame). A 3D rigid fit (Kabsch) then gives the camera -> ANT transform = the mount.

  w   fit + SAVE to data/camera_mount.yaml (vision_link reloads it within 1 s; needs >= 8 pairs spread out)
  r   restart                q   quit (nothing saved unless you pressed w)
Start values: config/camera_mount_default.yaml (tape measure). If NO pairs come in, the guess is too far off
(> ~0.5 m at the drone): fix the tape-measure values first (pitch sign: negative = tilted up).
"""
import json
import math
import os
import select
import sys
import termios
import time
import tty
import numpy as np
import rclpy
import yaml
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import String
from team_pkg.mission_common import LINK_FROM_OPT, MOUNT_PATH, AntView, Mount, fit_3d, load_mount, matrix_to_rpy

GATE = 0.5          # m: candidate must be this close to where the mount guess puts the drone
MIN_PAIRS = 8


class VisionCalib(Node):
    def __init__(self):
        super().__init__("vision_calib")
        self.names = [n for n in self.declare_parameter("cfs", "cf1").value.split(",") if n]
        self.antv = AntView(self)
        self.truth, self.hist, self.acc = {}, {}, {}
        for n in self.names:
            self.create_subscription(PoseStamped, f"/{n}/pose", lambda m, n=n: self.on_truth(n, m), 10)
        self.create_subscription(String, "/vision/candidates", self.on_cands, 10)
        self.mount, self.src = load_mount()
        self.reset()

    def on_truth(self, n, m):
        p, now = (m.pose.position.x, m.pose.position.y, m.pose.position.z), time.monotonic()
        self.truth[n] = (p, now)
        self.hist[n] = [(t, q) for t, q in self.hist.get(n, []) if now - t < 0.6] + [(now, p)]

    def still(self, n):
        h = self.hist.get(n, [])
        return len(h) >= 4 and h[-1][0] - h[0][0] > 0.45 and max(np.linalg.norm(np.subtract(q, h[-1][1])) for _, q in h) < 0.03

    def reset(self):
        self.pairs = []                 # (point in camera LINK frame, true point in ANT body frame)

    def on_cands(self, m):
        ant, calib = self.antv.fresh(), self.antv.calib
        if ant is None or calib is None or self.antv.moving:
            return
        msg = json.loads(m.data)
        now = time.monotonic()
        for name, (pw, t) in self.truth.items():
            if now - t > 0.3 or pw[2] - calib.floor_z < 0.25:      # stale, or the drone isn't flying
                continue
            pb_true = Mount.world_to_body(pw, ant, calib.floor_z)
            near = [c for c in msg["cands"] if np.linalg.norm(self.mount.opt_to_body(c["p_opt"]) - pb_true) < GATE]
            if len(near) != 1 or not self.still(name):
                self.acc[name] = []                                    # moving / ambiguous: start over
                continue
            self.acc.setdefault(name, []).append((LINK_FROM_OPT @ np.array(near[0]["p_opt"]), pb_true))
            if len(self.acc[name]) < 6:                                # ~0.4 s of frames at this spot
                continue
            p_link = np.mean([a for a, _ in self.acc[name]], axis=0)
            pb_true = np.mean([b for _, b in self.acc[name]], axis=0)
            self.acc[name] = []
            if self.pairs and min(np.linalg.norm(pb_true - q) for _, q in self.pairs) < 0.15:
                continue                                               # same place again: keep pairs spread out
            self.pairs.append((p_link, pb_true))
            print(f"\npair {len(self.pairs)}: {name} at ({pb_true[0]:.2f}, {pb_true[1]:+.2f}, {pb_true[2]:.2f}) m in front of the ANT", flush=True)

    def fit(self):
        src = np.array([p for p, _ in self.pairs])
        dst = np.array([q for _, q in self.pairs])
        R, t, rms = fit_3d(src, dst)
        for _ in range(2):                                            # drop outliers (3x median residual), refit
            res = np.linalg.norm(src @ R.T + t - dst, axis=1)
            keep = res < max(3 * np.median(res), 0.05)
            if keep.sum() >= MIN_PAIRS and keep.sum() < len(keep):
                R, t, rms = fit_3d(src[keep], dst[keep])
        roll, pitch, yaw = (math.degrees(a) for a in matrix_to_rpy(R))
        return Mount(float(t[0]), float(t[1]), float(t[2]), roll, pitch, yaw), rms

    def status(self):
        s = f"pairs {len(self.pairs)} | mount guess ({self.src}): pitch {self.mount.rpy[1]:+.0f} deg, z {self.mount.t[2]:.2f} m"
        if self.antv.fresh() is None:
            s += " | WAITING: no ANT pose (ant_link?)"
        elif self.antv.moving:
            s += " | WAITING: the ANT is moving"
        elif not any(time.monotonic() - t < 0.5 and p[2] > 0.25 for p, t in self.truth.values()):
            s += " | WAITING: no flying drone"
        if len(self.pairs) >= 4:
            m, rms = self.fit()
            s += f" | preview: pitch {m.rpy[1]:+.1f} deg, z {m.t[2]:.2f} m, RMS {rms * 100:.1f} cm"
        return s


def read_key(timeout):
    if not select.select([sys.stdin], [], [], timeout)[0]:
        return None
    return os.read(sys.stdin.fileno(), 1).decode(errors="ignore")


def main():
    rclpy.init()
    node = VisionCalib()
    print(__doc__.split("\n\n")[2])
    print(__doc__.split("\n\n")[3])
    old = termios.tcgetattr(sys.stdin) if sys.stdin.isatty() else None
    last = 0.0
    try:
        if old:
            tty.setcbreak(sys.stdin.fileno())
        while rclpy.ok():
            k = read_key(0.05) if old else time.sleep(0.05)
            for _ in range(30):                          # ALL waiting messages, not one per loop (fell behind at ~20/s)
                rclpy.spin_once(node, timeout_sec=0)
            if time.monotonic() - last > 0.5:
                print(f"\r{node.status()[:170]:<170}", end="", flush=True)
                last = time.monotonic()
            if k == "r":
                node.reset()
                print("\nrestarted")
            elif k == "w":
                if len(node.pairs) < MIN_PAIRS:
                    print(f"\nnot yet: {len(node.pairs)} pairs, need {MIN_PAIRS} spread over left/right/near/far/low/high")
                    continue
                m, rms = node.fit()
                os.makedirs(os.path.dirname(MOUNT_PATH), exist_ok=True)
                with open(MOUNT_PATH, "w") as f:
                    f.write(f"# measured by vision_calib {time.strftime('%Y-%m-%d %H:%M:%S')}: {len(node.pairs)} pairs, RMS {rms * 100:.1f} cm\n")
                    yaml.safe_dump(m.to_dict(), f, sort_keys=False)
                print(f"\nSAVED {MOUNT_PATH}: {m.summary()}, RMS {rms * 100:.1f} cm "
                      + ("GOOD" if rms < 0.05 else "OK" if rms < 0.10 else "BAD: more spread-out places, check the drone is the only object"))
                node.mount = m
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
