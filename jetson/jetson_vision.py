#!/usr/bin/env python3
"""jetson_vision.py: the ANT's EYES. Runs ON THE JETSON (plain Python 3: pyrealsense2 + numpy + OpenCV, no ROS).

Reads the RealSense D555 (PoE/DDS or USB), finds DRONE CANDIDATES = small objects floating in free air
in the depth image (no marker, no training), and sends them to the laptop over UDP (JSON, ~15 Hz).
The laptop node `vision_link` turns them into room coordinates, filters, tracks and names them.

    python3 jetson_vision.py                         # LAPTOP_IP from ~/.jetson_vision.env or the environment
    python3 jetson_vision.py --laptop 192.168.43.20  # explicit laptop IP
    python3 jetson_vision.py --broadcast             # no laptop IP: UDP broadcast on the hotspot
    python3 jetson_vision.py --test                  # camera check: fps + candidates, prints only
    python3 jetson_vision.py --show /tmp/dbg         # also save debug images (depth, mask, boxes) every 2 s
    python3 jetson_vision.py --selftest              # no camera: detector on a synthetic depth image

Packets (UDP):
  port 5600  JSON {"v":1, "t", "seq", "fps", "w", "h", "status", "cands": [{"p": [x,y,z] (m, camera OPTICAL
             frame: x right, y down, z forward), "ext": [dx,dy,dz] (m), "px": [u,v], "box": [x,y,w,h], "n",
             "iso": m}]}       one per depth frame, also when nothing is seen (heartbeat)
  port 5601  JPEG preview (colour if available, else depth), candidates boxed, ~2 Hz, <= 60 kB

How a candidate is found (all tunable with the flags below):
  1. depth valid between --min-range and --max-range;
  2. "edges" = a neighbour is invalid or differs by more than --edge-rel x depth  (object borders);
  3. connected pieces of non-edge pixels = objects; tiny / huge pieces dropped by pixel count;
  4. 3D size (10-90 % spread) between --min-size and --max-size (a Crazyflie is ~0.1 m);
  5. ISOLATED: the ring of pixels around it is >= --iso m farther away or invalid (it floats in front of
     what's behind it; walls, floor, tables, people are big and connected -> rejected in 3/4).
"""
import argparse
import json
import os
import socket
import sys
import time

import cv2
import numpy as np


# ------------------------------------------------------------------------------------------- detector
def find_blobs(z, fx, fy, cx, cy, min_range=0.5, max_range=4.5, edge_rel=0.08, min_px=15, max_px=6000,
               min_size=0.03, max_size=0.35, iso=0.30, max_cands=10):
    """z: depth in metres (HxW float32, 0 = invalid). Returns a list of candidate dicts (best first)."""
    valid = (z > min_range) & (z < max_range)
    zz = np.where(valid, z, 0.0).astype(np.float32)
    edge = ~valid
    for dy, dx in ((0, 1), (0, -1), (1, 0), (-1, 0)):
        nb = np.roll(np.roll(zz, dy, axis=0), dx, axis=1)
        edge |= (nb <= 0) | (np.abs(nb - zz) > edge_rel * np.maximum(zz, 1e-3))
    inner = (valid & ~edge).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(inner, connectivity=4)
    h, w = z.shape
    out = []
    for i in range(1, n):
        x0, y0, bw, bh, area = stats[i]
        if area < min_px or area > max_px:
            continue
        ys, xs = np.nonzero(labels[y0:y0 + bh, x0:x0 + bw] == i)
        ys, xs = ys + y0, xs + x0
        d = zz[ys, xs]
        X, Y = (xs - cx) * d / fx, (ys - cy) * d / fy
        pts = np.stack([X, Y, d], axis=1)
        lo, hi = np.percentile(pts, 10, axis=0), np.percentile(pts, 90, axis=0)
        ext = hi - lo
        size = float(max(ext[0], ext[1]))
        if size < min_size or size > max_size:
            continue
        zc = float(np.median(d))
        # isolation ring: pixels 3..8 px around the bounding box, outside the object
        m = 8
        rx0, ry0, rx1, ry1 = max(x0 - m, 0), max(y0 - m, 0), min(x0 + bw + m, w), min(y0 + bh + m, h)
        roi = zz[ry0:ry1, rx0:rx1]
        obj = labels[ry0:ry1, rx0:rx1] == i
        near = cv2.dilate(obj.astype(np.uint8), np.ones((7, 7), np.uint8)).astype(bool)
        ring = ~near
        rv = roi[ring]
        if rv.size == 0:
            continue
        behind = (rv <= 0) | (rv > zc + iso)                # invalid or clearly farther = free air
        frac = float(behind.mean())
        if frac < 0.7:
            continue
        valid_ring = rv[rv > 0]
        margin = float(np.median(valid_ring) - zc) if valid_ring.size else 9.9
        c = np.median(pts, axis=0)
        out.append({"p": [round(float(c[0]), 3), round(float(c[1]), 3), round(float(c[2]), 3)],
                    "ext": [round(float(e), 3) for e in ext], "px": [int(xs.mean()), int(ys.mean())],
                    "box": [int(x0), int(y0), int(bw), int(bh)], "n": int(area), "iso": round(min(margin, 9.9), 2),
                    "free": round(frac, 2)})
    out.sort(key=lambda c: (-c["free"], c["p"][2]))
    return out[:max_cands]


def selftest():
    """Synthetic 848x480 scene: floor, back wall, a table, a person-sized box and ONE drone-sized blob."""
    h, w, fx, fy, cx, cy = 480, 848, 430.0, 430.0, 424.0, 240.0
    v, u = np.mgrid[0:h, 0:w].astype(np.float32)
    z = np.full((h, w), 4.0, np.float32)                     # back wall at 4 m
    cam_h = 0.35                                              # camera 35 cm above the floor, level
    below = v > cy + 2
    zf = cam_h * fy / np.maximum(v - cy, 1e-3)                # floor plane
    z[below] = np.minimum(z[below], zf[below])
    def box(x0, x1, y0, y1, d):                               # a solid object at depth d
        z[y0:y1, x0:x1] = d
    box(80, 300, 200, 330, 2.2)                               # table front
    box(560, 660, 60, 400, 2.8)                               # person
    # drone: ~0.12 x 0.04 m at 2.0 m, 0.3 m above the camera
    du, dv = int(cx + 0.35 * fx / 2.0), int(cy - 0.30 * fy / 2.0)
    hw, hh = int(0.06 * fx / 2.0), int(0.02 * fy / 2.0)
    box(du - hw, du + hw, dv - hh, dv + hh, 2.0)
    rng = np.random.default_rng(0)
    z = z * (1 + rng.normal(0, 0.005, z.shape)).astype(np.float32)
    z[rng.random(z.shape) < 0.01] = 0                         # dropouts
    t0 = time.time()
    c = find_blobs(z, fx, fy, cx, cy)
    dt = (time.time() - t0) * 1000
    print(f"selftest: {len(c)} candidate(s) in {dt:.0f} ms: {c}")
    assert len(c) == 1, "expected exactly the drone"
    x, y, d = c[0]["p"]
    assert abs(d - 2.0) < 0.05 and abs(x - 0.35) < 0.05 and abs(y + 0.30) < 0.05, c[0]
    print("selftest OK: the drone found at", c[0]["p"], "(true 0.35, -0.30, 2.00); table, person, wall, floor rejected")


# ------------------------------------------------------------------------------------------- camera
def load_env():
    """KEY=VALUE lines from ~/.jetson_vision.env (written by setup_jetson.sh); real env vars win."""
    path = os.path.expanduser("~/.jetson_vision.env")
    if os.path.exists(path):
        for line in open(path):
            if "=" in line and not line.strip().startswith("#"):
                k, v = line.strip().split("=", 1)
                os.environ.setdefault(k, v)


def open_camera(rs, args):
    """Start the pipeline: wanted profiles first, else the camera's defaults. Returns (pipeline, profile)."""
    ctx = rs.context()
    devs = ctx.query_devices()
    if len(devs) == 0:
        raise RuntimeError("no RealSense found (PoE: ping 192.168.11.55, ~/.realsense-config.json DDS on, "
                           "USB cable UNPLUGGED; USB: lsusb). rs-enumerate-devices to check")
    d = devs[0]
    info = {k: d.get_info(getattr(rs.camera_info, k)) for k in ("name", "serial_number", "firmware_version")
            if d.supports(getattr(rs.camera_info, k))}
    print(f"camera: {info}", flush=True)
    pipe = rs.pipeline(ctx)
    for color in (True, False):
        cfg = rs.config()
        cfg.enable_device(info.get("serial_number", ""))
        cfg.enable_stream(rs.stream.depth, args.width, args.height, rs.format.z16, args.fps)
        if color:
            cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 15)
        try:
            prof = pipe.start(cfg)
            print(f"streams: depth {args.width}x{args.height}@{args.fps}" + (", colour 640x480@15" if color else ""), flush=True)
            return pipe, prof
        except RuntimeError as e:
            print(f"profile not available ({e}); trying {'depth only' if color else 'defaults'}", flush=True)
    prof = pipe.start()
    print("streams: camera defaults", flush=True)
    return pipe, prof


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--laptop", help="laptop IP (default: LAPTOP_IP from env / ~/.jetson_vision.env)")
    ap.add_argument("--port", type=int, default=5600)
    ap.add_argument("--preview-port", type=int, default=5601)
    ap.add_argument("--broadcast", action="store_true", help="send to 255.255.255.255 instead of the laptop IP")
    ap.add_argument("--test", action="store_true", help="print fps and candidates, send nothing")
    ap.add_argument("--show", metavar="DIR", help="save debug images to DIR every 2 s")
    ap.add_argument("--selftest", action="store_true", help="run the detector on a synthetic image (no camera)")
    ap.add_argument("--width", type=int, default=848)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--min-range", type=float, default=0.5)
    ap.add_argument("--max-range", type=float, default=4.5)
    ap.add_argument("--edge-rel", type=float, default=0.08)
    ap.add_argument("--min-px", type=int, default=15)
    ap.add_argument("--max-px", type=int, default=6000)
    ap.add_argument("--min-size", type=float, default=0.03)
    ap.add_argument("--max-size", type=float, default=0.35)
    ap.add_argument("--iso", type=float, default=0.30)
    args = ap.parse_args()
    if args.selftest:
        selftest()
        return
    load_env()
    import pyrealsense2 as rs                                  # only needed with a camera
    laptop = "255.255.255.255" if args.broadcast else (args.laptop or os.environ.get("LAPTOP_IP"))
    if not laptop and not args.test:
        sys.exit("no laptop IP: --laptop IP, LAPTOP_IP=... in ~/.jetson_vision.env, or --broadcast")
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
    det = {k: getattr(args, k) for k in ("min_range", "max_range", "edge_rel", "min_px", "max_px", "min_size", "max_size", "iso")}
    print(f"jetson_vision -> {laptop or '(test, nothing sent)'}:{args.port} | detector {det}", flush=True)
    seq, last_prev, last_show, last_fail = 0, 0.0, 0.0, 0.0
    times = []
    spatial = rs.spatial_filter()
    while True:
        try:
            pipe, prof = open_camera(rs, args)
            dprof = prof.get_stream(rs.stream.depth).as_video_stream_profile()
            intr = dprof.get_intrinsics()
            scale = prof.get_device().first_depth_sensor().get_depth_scale()
            print(f"intrinsics fx {intr.fx:.1f} fy {intr.fy:.1f} cx {intr.ppx:.1f} cy {intr.ppy:.1f}, depth scale {scale}", flush=True)
            while True:
                frames = pipe.wait_for_frames(2000)
                df = frames.get_depth_frame()
                if not df:
                    continue
                df = spatial.process(df).as_depth_frame()
                z = np.asanyarray(df.get_data()).astype(np.float32) * scale
                cands = find_blobs(z, intr.fx, intr.fy, intr.ppx, intr.ppy, **det)
                now = time.time()
                times = [t for t in times if now - t < 2.0] + [now]
                fps = len(times) / 2.0
                seq += 1
                msg = {"v": 1, "t": now, "seq": seq, "fps": round(fps, 1), "w": z.shape[1], "h": z.shape[0],
                       "status": "ok", "cands": cands}
                if args.test:
                    if seq % 15 == 0:
                        print(f"fps {fps:4.1f} | {len(cands)} candidate(s): " +
                              "; ".join(f"({c['p'][0]:+.2f},{c['p'][1]:+.2f},{c['p'][2]:.2f}) m {c['ext'][0]:.2f}x{c['ext'][1]:.2f} free {c['free']}" for c in cands[:3]), flush=True)
                else:
                    sock.sendto(json.dumps(msg).encode(), (laptop, args.port))
                if now - last_prev > 0.5 or (args.show and now - last_show > 2.0):
                    cf = frames.get_color_frame()
                    if cf:
                        img = np.asanyarray(cf.get_data()).copy()
                        sx, sy = img.shape[1] / z.shape[1], img.shape[0] / z.shape[0]
                    else:
                        img = cv2.applyColorMap(cv2.convertScaleAbs(z, alpha=255 / args.max_range), cv2.COLORMAP_JET)
                        sx = sy = 1.0
                    for c in cands:
                        x, y, bw, bh = c["box"]
                        cv2.rectangle(img, (int(x * sx), int(y * sy)), (int((x + bw) * sx), int((y + bh) * sy)), (0, 255, 0), 2)
                        cv2.putText(img, f"{c['p'][2]:.1f}m", (int(x * sx), int(y * sy) - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                    if args.show and now - last_show > 2.0:
                        os.makedirs(args.show, exist_ok=True)
                        cv2.imwrite(os.path.join(args.show, f"{seq:06d}_view.jpg"), img)
                        cv2.imwrite(os.path.join(args.show, f"{seq:06d}_depth.png"), (np.clip(z, 0, 10) * 1000).astype(np.uint16))
                        last_show = now
                    if not args.test and now - last_prev > 0.5:
                        small = cv2.resize(img, (320, int(320 * img.shape[0] / img.shape[1])))
                        ok, jpg = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 60])
                        if ok and len(jpg) < 60000:
                            sock.sendto(jpg.tobytes(), (laptop, args.preview_port))
                    last_prev = now
        except KeyboardInterrupt:
            print("stopped")
            return
        except Exception as e:                                # camera gone / DDS hiccup: keep the laptop informed
            now = time.time()
            if now - last_fail > 5:
                print(f"camera problem: {e} -> retrying", flush=True)
                last_fail = now
            if laptop and not args.test:
                sock.sendto(json.dumps({"v": 1, "t": now, "seq": seq, "fps": 0, "status": f"no camera: {e}"[:200],
                                        "cands": []}).encode(), (laptop, args.port))
            try:
                pipe.stop()
            except Exception:
                pass
            time.sleep(1.0)


if __name__ == "__main__":
    main()
