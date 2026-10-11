"""vision_link: the laptop end of the ANT's EYES. Turns the Jetson's depth blobs into named drone positions.

    vision_link                                 (helper from scripts/cf_env.sh; drone domain 42)
    = ros2 run team_pkg vision_link --ros-args -p cfs:=cf1,cf2 [-p port:=5600] [-p min_h:=0.3 -p max_h:=1.6]

Input  UDP 5600 (JSON candidates, camera optical frame) + 5601 (JPEG preview) from jetson/jetson_vision.py
       (or scripts/vision_diag/fake_vision.py), /ant/pose_room + /mission/calib (ant_link), /fleet/beacon.
Chain  candidate -> ANT body frame (camera mount: data/camera_mount.yaml from vision_calib, else
       config/camera_mount_default.yaml; reloaded when the file changes) -> room/world frame (ANT pose)
       -> FILTERS: 0.3-1.6 m above the floor, not on/at the ANT itself, 0.5-4.5 m from the camera
       -> TRACKS: nearest-neighbour; CONFIRMED only after >= 5 hits AND seen in >= 50 % of the frames since it
          appeared (a drone is seen almost every frame, random false blobs are not); dropped after 1 s unseen -> NAMES: a track is cf1/cf2 if it is the closest track to that drone's
          last beacon position (within 1.2 m); else "unknown".
Output /vision/drones  std_msgs/String JSON {"t", "tracks": [{"id","name","x","y","z","hits","age"}]}  10 Hz
       /vision/candidates  JSON: every raw candidate with camera + world coordinates (vision_calib uses it)
       /vision/markers  rviz spheres + names      /vision/image  CompressedImage (the Jetson's preview)
Status line: link age, Jetson fps/status, candidates, tracks, and - for evaluation only - each named track's
error against the drone's Lighthouse position (/cfN/pose). The rescue itself never uses /cfN/pose for finding.
"""
import json
import math
import os
import select
import socket
import time
import numpy as np
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import String
from visualization_msgs.msg import Marker, MarkerArray
from team_pkg.mission_common import MOUNT_DEFAULT, MOUNT_PATH, AntView, Mount, load_mount

FOOT_X, FOOT_Y = 0.40, 0.30          # ANT half-size + 10 cm: blobs here are the ANT itself / its load


class Track:
    n = 0

    def __init__(self, p, now):
        Track.n += 1
        self.id, self.p, self.hits, self.seen, self.born, self.name = Track.n, np.array(p), 1, now, now, "unknown"
        self.frames = 1                  # candidate batches since it appeared

    def ok(self):
        return self.hits >= 5 and self.hits >= 0.5 * self.frames


class VisionLink(Node):
    def __init__(self):
        super().__init__("vision_link")
        self.names = [n for n in self.declare_parameter("cfs", "cf1").value.split(",") if n]
        port = self.declare_parameter("port", 5600).value
        self.min_h = self.declare_parameter("min_h", 0.3).value
        self.max_h = self.declare_parameter("max_h", 1.6).value
        self.antv = AntView(self)
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind(("0.0.0.0", port))
        self.isock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.isock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.isock.bind(("0.0.0.0", port + 1))
        self.pub = self.create_publisher(String, "/vision/drones", 10)
        self.pub_c = self.create_publisher(String, "/vision/candidates", 10)
        self.pub_m = self.create_publisher(MarkerArray, "/vision/markers", 10)
        self.pub_i = self.create_publisher(CompressedImage, "/vision/image", 2)
        self.create_subscription(String, "/fleet/beacon", self.on_beacon, 10)
        self.truth = {}
        for n in self.names:
            self.create_subscription(PoseStamped, f"/{n}/pose",
                                     lambda m, n=n: self.truth.__setitem__(n, (m.pose.position.x, m.pose.position.y, m.pose.position.z)), 10)
        self.beacons, self.tracks = {}, []
        self.mount, self.mount_src, self.mount_mtime = None, "", None
        self.last_pkt, self.jet, self.n_cands, self.n_kept, self.sender = 0.0, {}, 0, 0, None
        self.reload_mount()
        self.create_timer(0.02, self.poll)
        self.create_timer(0.1, self.publish)
        self.create_timer(1.0, self.reload_mount)
        self.create_timer(0.5, self.status)
        print(f"vision_link: listening on UDP {port} (+{port + 1} preview) for drones {self.names}", flush=True)

    # ---------------- inputs ----------------
    def reload_mount(self):
        mt = next((os.path.getmtime(p) for p in (MOUNT_PATH, MOUNT_DEFAULT) if os.path.exists(p)), None)
        if mt != self.mount_mtime:
            self.mount_mtime = mt
            self.mount, self.mount_src = load_mount()
            print(f"\ncamera mount ({self.mount_src}): {self.mount.summary()}", flush=True)

    def on_beacon(self, m):
        try:
            b = json.loads(m.data)
            self.beacons[b["name"]] = (b["x"], b["y"], b["z"], time.monotonic())
        except (ValueError, KeyError):
            pass

    def poll(self):
        while select.select([self.isock], [], [], 0)[0]:
            data, _ = self.isock.recvfrom(65536)
            img = CompressedImage(format="jpeg", data=data)
            img.header.stamp, img.header.frame_id = self.get_clock().now().to_msg(), "d555"
            self.pub_i.publish(img)
        while select.select([self.sock], [], [], 0)[0]:
            data, addr = self.sock.recvfrom(65536)
            try:
                msg = json.loads(data)
            except ValueError:
                continue
            self.last_pkt, self.sender = time.monotonic(), addr[0]
            self.jet = {"fps": msg.get("fps"), "status": msg.get("status", "?")}
            if "setup test" in self.jet["status"] or "check_jetson" in self.jet["status"]:
                print(f"\nUDP test packet from {addr[0]}: '{self.jet['status']}' -> the Jetson can reach this laptop", flush=True)
            self.on_candidates(msg.get("cands", []))

    def on_candidates(self, cands):
        ant, calib = self.antv.fresh(), self.antv.calib
        self.n_cands = len(cands)
        if ant is None or calib is None:
            return
        now = time.monotonic()
        kept, raw = [], []
        for c in cands:
            pb = self.mount.opt_to_body(c["p"])
            pw = Mount.body_to_world(pb, ant, calib.floor_z)
            h = pw[2] - calib.floor_z
            ok = (self.min_h <= h <= self.max_h and not (abs(pb[0]) < FOOT_X and abs(pb[1]) < FOOT_Y))
            raw.append({"p_opt": c["p"], "p_world": [round(float(v), 3) for v in pw], "kept": bool(ok)})
            if ok:
                kept.append(pw)
        self.n_kept = len(kept)
        self.pub_c.publish(String(data=json.dumps({"t": time.time(), "ant": list(ant), "cands": raw})))
        # nearest-neighbour association, gate grows with the time a track was unseen
        free = list(range(len(kept)))
        for tr in self.tracks:
            tr.frames += 1
        for tr in sorted(self.tracks, key=lambda t: -t.hits):
            if not free:
                break
            gate = 0.25 + 1.0 * (now - tr.seen)
            d = [np.linalg.norm(kept[i] - tr.p) for i in free]
            j = int(np.argmin(d))
            if d[j] < gate:
                tr.p = 0.5 * tr.p + 0.5 * kept[free[j]]
                tr.hits, tr.seen = tr.hits + 1, now
                free.pop(j)
        self.tracks += [Track(kept[i], now) for i in free]
        self.tracks = [t for t in self.tracks if now - t.seen < 1.0]
        # names: each drone takes the confirmed track nearest its last beacon (one track per drone)
        for t in self.tracks:
            t.name = "unknown"
        for name, b in self.beacons.items():
            cands_t = [t for t in self.tracks if t.ok() and t.name == "unknown"]
            if cands_t:
                t = min(cands_t, key=lambda t: np.linalg.norm(t.p - np.array(b[:3])))
                if np.linalg.norm(t.p - np.array(b[:3])) < 1.2:
                    t.name = name

    # ---------------- outputs ----------------
    def confirmed(self):
        return [t for t in self.tracks if t.ok()]

    def publish(self):
        now = time.monotonic()
        tr = [{"id": t.id, "name": t.name, "x": round(float(t.p[0]), 3), "y": round(float(t.p[1]), 3),
               "z": round(float(t.p[2]), 3), "hits": t.hits, "age": round(now - t.seen, 2)} for t in self.confirmed()]
        self.pub.publish(String(data=json.dumps({"t": time.time(), "tracks": tr,
                                                 "link_age": round(now - self.last_pkt, 2) if self.last_pkt else None})))
        ma = MarkerArray()
        ma.markers.append(Marker(action=Marker.DELETEALL))
        for t in tr:
            for kind in (Marker.SPHERE, Marker.TEXT_VIEW_FACING):
                m = Marker(type=kind, action=Marker.ADD, ns="vision", id=t["id"] * 2 + (kind == Marker.TEXT_VIEW_FACING))
                m.header.frame_id, m.header.stamp = "world", self.get_clock().now().to_msg()
                m.pose.position.x, m.pose.position.y, m.pose.position.z = t["x"], t["y"], t["z"] + (0.15 if kind != Marker.SPHERE else 0)
                m.pose.orientation.w = 1.0
                m.scale.x = m.scale.y = m.scale.z = 0.15 if kind == Marker.SPHERE else 0.1
                m.color.r, m.color.g, m.color.b, m.color.a = (0.1, 1.0, 0.3, 0.8) if t["name"] != "unknown" else (1.0, 1.0, 0.2, 0.6)
                m.text = f"{t['name']} (seen)"
                ma.markers.append(m)
        self.pub_m.publish(ma)

    def status(self):
        now = time.monotonic()
        if not self.last_pkt:
            s = "NO PACKETS from the Jetson yet (jetson_vision.py running? LAPTOP_IP right? firewall?)"
        else:
            age = now - self.last_pkt
            s = f"Jetson {self.sender} {'OK' if age < 1 else 'SILENT'} ({age:.1f} s) {self.jet.get('fps')} fps '{self.jet.get('status')}'"
        if self.antv.fresh() is None:
            s += " | NO ANT POSE (ant_link?)"
        s += f" | cands {self.n_cands} kept {self.n_kept} | tracks:"
        for t in self.confirmed():
            s += f" {t.name}({t.p[0]:+.2f},{t.p[1]:+.2f},{t.p[2]:.2f})"
            if self.truth:                                  # evaluation only: nearest true drone
                s += f" err {min(np.linalg.norm(t.p - np.array(v)) for v in self.truth.values()) * 100:.0f}cm"
        print(f"\r{s[:190]:<190}", end="", flush=True)


def main():
    rclpy.init()
    rclpy.spin(VisionLink())
