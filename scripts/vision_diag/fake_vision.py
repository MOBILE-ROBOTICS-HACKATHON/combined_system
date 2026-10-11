#!/usr/bin/env python3
"""fake_vision.py: stands in for the Jetson + D555 (no hardware). Sends the SAME UDP packets as
jetson/jetson_vision.py, computed from the fake drones' true positions as seen by a camera on the fake ANT.

    python3 scripts/vision_diag/fake_vision.py [--drones cf1,cf2] [--mount "0.27,0.02,0.33,1,-12,3"]
                                               [--clutter 1.0] [--noise 0.01] [--dropout 0.1] [--blind]
Runs in the drone domain (42, localhost: source scripts/cf_env.sh). Needs ant_link (for /ant/pose_room) and
fake drones (scripts/cf_diag/fake_cf.py) - scripts/mission_diag/sim_all.sh with VISION=1 starts it.

--mount   the HIDDEN TRUE mount x,y,z (m), roll,pitch,yaw (deg) - deliberately a bit different from
          config/camera_mount_default.yaml, so vision_calib has something to find
--clutter mean number of FALSE blobs per frame (random spots + one fixed "lamp")
--blind   sees nothing (to test the rescue without vision);  touch /tmp/fake_vision_blind toggles it live
"""
import argparse
import json
import math
import os
import socket
import time
import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import String
from team_pkg.mission_common import LATCHED, Calib, Mount, yaw_of

FX = FY = 430.0
W, H = 848, 480
CX, CY = W / 2, H / 2


class FakeVision(Node):
    def __init__(self, args):
        super().__init__("fake_vision")
        v = [float(a) for a in args.mount.split(",")]
        self.mount = Mount(*v)
        self.args = args
        self.drones = {}
        for name in args.drones.split(","):
            self.create_subscription(PoseStamped, f"/{name}/pose",
                                     lambda m, n=name: self.drones.__setitem__(n, (m.pose.position.x, m.pose.position.y, m.pose.position.z)), 10)
        self.ant, self.calib = None, None
        self.create_subscription(PoseStamped, "/ant/pose_room", self.on_ant, 10)
        self.create_subscription(String, "/mission/calib", lambda m: setattr(self, "calib", Calib.from_yaml(m.data)), LATCHED)
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.rng = np.random.default_rng(1)
        self.seq, self.last_jpg = 0, 0.0
        self.create_timer(1 / 15, self.tick)
        print(f"fake_vision: TRUE mount {self.mount.summary()} -> UDP {args.laptop}:{args.port}", flush=True)

    def on_ant(self, m):
        p = m.pose.position
        self.ant = (p.x, p.y, p.z, yaw_of(m.pose.orientation))

    def visible(self, po):
        x, y, z = po
        return 0.5 < z < 4.5 and abs(math.atan2(x, z)) < math.radians(43.5) and abs(math.atan2(y, z)) < math.radians(29)

    def cand(self, po, size=0.1):
        u, v = CX + FX * po[0] / po[2], CY + FY * po[1] / po[2]
        bw, bh = max(int(FX * size / po[2]), 2), max(int(FX * 0.03 / po[2]), 2)
        return {"p": [round(float(c), 3) for c in po], "ext": [size, 0.03, 0.03], "px": [int(u), int(v)],
                "box": [int(u - bw / 2), int(v - bh / 2), bw, bh], "n": bw * bh, "iso": 1.0, "free": 1.0}

    def tick(self):
        blind = self.args.blind ^ os.path.exists("/tmp/fake_vision_blind")
        cands = []
        if self.ant and self.calib and not blind:
            for name, pw in self.drones.items():
                po = self.mount.body_to_opt(Mount.world_to_body(pw, self.ant, self.calib.floor_z))
                if self.visible(po) and self.rng.random() > self.args.dropout:
                    po = po + self.rng.normal(0, self.args.noise * po[2], 3)
                    cands.append(self.cand(po))
            # clutter: random false blobs + one fixed "lamp" in the room (1.5 m high)
            for _ in range(self.rng.poisson(self.args.clutter)):
                cands.append(self.cand([self.rng.uniform(-1.5, 1.5), self.rng.uniform(-0.8, 0.8), self.rng.uniform(0.8, 4.0)], 0.08))
            lamp = self.mount.body_to_opt(Mount.world_to_body((0.5, 3.5, self.calib.floor_z + 1.5), self.ant, self.calib.floor_z))
            if self.visible(lamp):
                cands.append(self.cand(lamp, 0.2))
        self.seq += 1
        msg = {"v": 1, "t": time.time(), "seq": self.seq, "fps": 15.0, "w": W, "h": H,
               "status": "ok (FAKE vision" + (", BLIND)" if blind else ")"), "cands": cands}
        self.sock.sendto(json.dumps(msg).encode(), (self.args.laptop, self.args.port))
        if time.time() - self.last_jpg > 0.5:                       # synthetic preview
            img = np.full((240, 424, 3), 40, np.uint8)
            for c in cands:
                x, y, bw, bh = c["box"]
                cv2.rectangle(img, (x // 2, y // 2), ((x + bw) // 2, (y + bh) // 2), (0, 255, 0), 1)
            cv2.putText(img, "FAKE D555" + (" BLIND" if blind else ""), (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
            self.sock.sendto(cv2.imencode(".jpg", img)[1].tobytes(), (self.args.laptop, self.args.preview_port))
            self.last_jpg = time.time()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--drones", default="cf1")
    ap.add_argument("--mount", default="0.27,0.02,0.33,1,-12,3")
    ap.add_argument("--clutter", type=float, default=1.0)
    ap.add_argument("--noise", type=float, default=0.01)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--blind", action="store_true")
    ap.add_argument("--laptop", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=5600)
    ap.add_argument("--preview-port", type=int, default=5601)
    rclpy.init()
    rclpy.spin(FakeVision(ap.parse_args()))


if __name__ == "__main__":
    main()
