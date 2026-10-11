"""Shared pieces for the ANT1 + Crazyflie missions (docs/08_ANT_DRONE_MISSIONS.md).

Two frames:
  world    the drone's position system (Lighthouse room coordinates), used by cf_server / cf_* tools
  ant_map  the ANT1's SLAM map (olive_olixRobot_ant1_id001_map), used by Nav2
Linked by a 2D rigid transform found by mission_calib:   map = R(theta) * room + (tx, ty)
plus two heights in the drone's frame: plate_z (top of the ANT, where the drone sits) and floor_z.

AntView: what every drone-side node knows about the ANT, from ant_link's topics in domain 42:
  /ant/pose_room  geometry_msgs/PoseStamped  ANT centre in the drone frame, z = plate_z (10 Hz)
  /ant/moving     std_msgs/Bool              ANT driving (odometry speed > 2 cm/s)
  /mission/calib  std_msgs/String (latched)  the calibration as YAML
"""
import math
import os
import time
import numpy as np
import yaml
from rclpy.qos import DurabilityPolicy, QoSProfile
from std_msgs.msg import Bool, String
from geometry_msgs.msg import PoseStamped

CALIB_PATH = os.environ.get("MISSION_CALIB", "/hackathon/data/mission_calib.yaml")
MOUNT_PATH = os.environ.get("CAMERA_MOUNT", "/hackathon/data/camera_mount.yaml")              # calibrated
MOUNT_DEFAULT = "/hackathon/ws/src/team_pkg/config/camera_mount_default.yaml"                # tape measure
LATCHED = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
ON_PLATE_R = 0.25   # m: a drone this close (xy) to the ANT centre is above the plate


# Crazyflie supervisor.info bits (firmware 2024+; crazyflie_interfaces only names bits 0-6). 11 Oct: a drone
# reported 640 = isCrashed + hlTrajFinished and silently ignored arm/takeoff until it was power-cycled.
SUP_BITS = ["canBeArmed", "isArmed", "autoArm", "canFly", "isFlying", "isTumbled", "isLocked", "isCrashed",
            "hlControlActive", "hlTrajFinished", "hlControlDisabled"]
CRASHED, LOCKED, TUMBLED, CAN_BE_ARMED = 128, 64, 32, 1


def supervisor_problem(info):
    """Why the drone won't take off (None if the supervisor allows it). `info` = Status.supervisor_info."""
    if info & CRASHED:
        return "CRASHED state (it tipped over / hit something): switch the drone OFF and ON, then restart cf_server"
    if info & LOCKED:
        return "LOCKED (after a kill / emergency stop): switch the drone OFF and ON, then restart cf_server"
    if info & TUMBLED:
        return "TUMBLED (not level): put it flat on its feet"
    return None


def supervisor_warning(info):
    """Not a refusal (firmware versions differ here), just worth printing."""
    if not info & (CAN_BE_ARMED | 2):
        return f"the firmware doesn't report 'can be armed' (supervisor_info {info}): if it doesn't take off, check battery / level / power-cycle"
    return None


def yaw_of(q):
    return math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))


def wrap(a):
    return math.atan2(math.sin(a), math.cos(a))


class Calib:
    """map = R(theta) * room + t.  Heights (plate_z, floor_z) are in the drone (room) frame."""

    def __init__(self, theta, tx, ty, plate_z, floor_z, rms=None, n=None, stamp=None, method=""):
        self.theta, self.tx, self.ty = theta, tx, ty
        self.plate_z, self.floor_z = plate_z, floor_z
        self.rms, self.n, self.stamp, self.method = rms, n, stamp, method

    def room_to_map(self, x, y):
        c, s = math.cos(self.theta), math.sin(self.theta)
        return c * x - s * y + self.tx, s * x + c * y + self.ty

    def map_to_room(self, x, y):
        c, s = math.cos(self.theta), math.sin(self.theta)
        x, y = x - self.tx, y - self.ty
        return c * x + s * y, -s * x + c * y

    def to_yaml(self):
        return yaml.safe_dump({k: v for k, v in vars(self).items()}, sort_keys=False)

    @staticmethod
    def from_yaml(text):
        d = yaml.safe_load(text) if text else None
        return Calib(**d) if d else None

    def summary(self):
        rms = f", fit RMS {self.rms * 100:.1f} cm from {self.n} points" if self.rms is not None else ""
        return (f"theta {math.degrees(self.theta):+.1f} deg, t ({self.tx:+.2f}, {self.ty:+.2f}) m, "
                f"plate z {self.plate_z:+.2f}, floor z {self.floor_z:+.2f}{rms} [{self.method}]")


def load_calib(path=CALIB_PATH):
    try:
        with open(path) as f:
            return Calib.from_yaml(f.read())
    except FileNotFoundError:
        return None


def save_calib(c, path=CALIB_PATH):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(c.to_yaml())


def fit_2d(room, mapp):
    """Least-squares 2D rigid fit (Kabsch): map ~ R(theta) room + t. Returns theta, tx, ty, rms (m)."""
    p, q = np.asarray(room, float), np.asarray(mapp, float)
    pc, qc = p.mean(0), q.mean(0)
    h = (p - pc).T @ (q - qc)
    theta = math.atan2(h[0, 1] - h[1, 0], h[0, 0] + h[1, 1])
    r = np.array([[math.cos(theta), -math.sin(theta)], [math.sin(theta), math.cos(theta)]])
    t = qc - r @ pc
    rms = float(np.sqrt(np.mean(np.sum((p @ r.T + t - q) ** 2, axis=1))))
    return theta, float(t[0]), float(t[1]), rms


def fit_3d(src, dst):
    """Least-squares 3D rigid fit (Kabsch/SVD): dst ~ R @ src + t. Returns R (3x3), t (3,), rms (m)."""
    p, q = np.asarray(src, float), np.asarray(dst, float)
    pc, qc = p.mean(0), q.mean(0)
    u, _, vt = np.linalg.svd((p - pc).T @ (q - qc))
    d = np.sign(np.linalg.det(vt.T @ u.T))                  # no reflections
    r = vt.T @ np.diag([1.0, 1.0, d]) @ u.T
    t = qc - r @ pc
    rms = float(np.sqrt(np.mean(np.sum((p @ r.T + t - q) ** 2, axis=1))))
    return r, t, rms


def rpy_to_matrix(roll, pitch, yaw):
    """Rotation matrix from roll/pitch/yaw (rad, fixed axes x, y, z: R = Rz(yaw) Ry(pitch) Rx(roll))."""
    cr, sr, cp, sp, cy, sy = math.cos(roll), math.sin(roll), math.cos(pitch), math.sin(pitch), math.cos(yaw), math.sin(yaw)
    return np.array([[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                     [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                     [-sp, cp * sr, cp * cr]])


def matrix_to_rpy(r):
    return (math.atan2(r[2, 1], r[2, 2]), math.atan2(-r[2, 0], math.hypot(r[2, 1], r[2, 2])), math.atan2(r[1, 0], r[0, 0]))


# Camera OPTICAL frame (RealSense: x right, y down, z forward) -> camera LINK frame (x forward, y left, z up)
LINK_FROM_OPT = np.array([[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]])


class Mount:
    """Where the D555 sits on the ANT. Body frame: origin = ANT centre ON THE FLOOR, x forward, y left, z up.
    x, y, z (m) = camera position; roll/pitch/yaw (deg) = camera link orientation, pitch NEGATIVE = tilted UP."""

    def __init__(self, x=0.25, y=0.0, z=0.30, roll_deg=0.0, pitch_deg=-15.0, yaw_deg=0.0, **extra):
        self.t = np.array([x, y, z], float)
        self.rpy = (roll_deg, pitch_deg, yaw_deg)
        self.R = rpy_to_matrix(*(math.radians(a) for a in self.rpy)) @ LINK_FROM_OPT   # body <- optical
        self.extra = extra

    def opt_to_body(self, p):
        return self.R @ np.asarray(p, float) + self.t

    def body_to_opt(self, p):
        return self.R.T @ (np.asarray(p, float) - self.t)

    @staticmethod
    def body_to_world(pb, ant, floor_z):
        """ant = (x, y, plate_z, yaw) in the drone/world frame (AntView.fresh())."""
        c, s = math.cos(ant[3]), math.sin(ant[3])
        return np.array([ant[0] + c * pb[0] - s * pb[1], ant[1] + s * pb[0] + c * pb[1], floor_z + pb[2]])

    @staticmethod
    def world_to_body(pw, ant, floor_z):
        c, s = math.cos(ant[3]), math.sin(ant[3])
        dx, dy = pw[0] - ant[0], pw[1] - ant[1]
        return np.array([c * dx + s * dy, -s * dx + c * dy, pw[2] - floor_z])

    def to_dict(self):
        return {"x": float(self.t[0]), "y": float(self.t[1]), "z": float(self.t[2]),
                "roll_deg": float(self.rpy[0]), "pitch_deg": float(self.rpy[1]), "yaw_deg": float(self.rpy[2]), **self.extra}

    def summary(self):
        return (f"camera at ({self.t[0]:+.2f}, {self.t[1]:+.2f}, {self.t[2]:.2f}) m on the ANT, roll {self.rpy[0]:+.1f}, "
                f"pitch {self.rpy[1]:+.1f} (negative = up), yaw {self.rpy[2]:+.1f} deg")


def load_mount():
    """(Mount, path it came from): the calibrated file if there is one, else the tape-measure default."""
    for path in (MOUNT_PATH, MOUNT_DEFAULT):
        try:
            with open(path) as f:
                return Mount(**(yaml.safe_load(f) or {})), path
        except FileNotFoundError:
            continue
    return Mount(), "built-in default"


class AntView:
    """Subscribes to ant_link's topics on `node`. All fields None until data arrives."""

    def __init__(self, node):
        self.pose, self.pose_at, self.moving, self.calib = None, 0.0, None, None
        node.create_subscription(PoseStamped, "/ant/pose_room", self._pose, 10)
        node.create_subscription(Bool, "/ant/moving", lambda m: setattr(self, "moving", m.data), 10)
        node.create_subscription(String, "/mission/calib", lambda m: setattr(self, "calib", Calib.from_yaml(m.data)),
                                 LATCHED)

    def _pose(self, m):
        p = m.pose.position
        self.pose, self.pose_at = (p.x, p.y, p.z, yaw_of(m.pose.orientation)), time.monotonic()

    def fresh(self, max_age=1.0):
        """ANT pose (x, y, plate_z, yaw) in the drone frame if newer than max_age s, else None."""
        return self.pose if self.pose and time.monotonic() - self.pose_at < max_age else None

    def land_z(self, drone_xy, default):
        """Height to land at: the plate if the drone is above the ANT, the floor otherwise.
        `default` (the drone's own home/floor estimate) when there is no ANT data."""
        ant = self.fresh()
        if ant is None or self.calib is None:
            return default
        if math.hypot(drone_xy[0] - ant[0], drone_xy[1] - ant[1]) < ON_PLATE_R:
            return self.calib.plate_z
        return self.calib.floor_z


if __name__ == "__main__":   # self-check: the fit recovers a known transform; conversions invert
    rng = np.random.default_rng(1)
    true = (math.radians(30), 2.0, -1.0)
    room = rng.uniform(-2, 2, (40, 2))
    c = Calib(*true, plate_z=0.16, floor_z=0.0)
    mapp = np.array([c.room_to_map(*p) for p in room]) + rng.normal(0, 0.01, (40, 2))
    th, tx, ty, rms = fit_2d(room, mapp)
    assert abs(math.degrees(th - true[0])) < 1 and abs(tx - 2) < 0.02 and abs(ty + 1) < 0.02 and rms < 0.02, (th, tx, ty, rms)
    x, y = c.map_to_room(*c.room_to_map(0.3, -0.7))
    assert abs(x - 0.3) < 1e-9 and abs(y + 0.7) < 1e-9
    assert Calib.from_yaml(c.to_yaml()).tx == 2.0
    R0 = rpy_to_matrix(0.1, -0.3, 1.2)
    assert np.allclose(rpy_to_matrix(*matrix_to_rpy(R0)), R0)
    src = rng.uniform(-2, 2, (30, 3))
    dst = src @ R0.T + np.array([0.2, -0.1, 0.4]) + rng.normal(0, 0.005, (30, 3))
    R, t, rms3 = fit_3d(src, dst)
    assert np.allclose(R, R0, atol=0.01) and np.allclose(t, [0.2, -0.1, 0.4], atol=0.01) and rms3 < 0.01, (R, t, rms3)
    m = Mount(0.3, 0.05, 0.32, 2, -12, 3)
    pb = np.array([2.0, 0.4, 0.8])
    assert np.allclose(m.opt_to_body(m.body_to_opt(pb)), pb)
    assert m.body_to_opt([1.25, 0.0, 0.30])[2] > 0.9               # straight ahead = optical +z
    ant = (1.0, -2.0, 0.16, 0.7)
    assert np.allclose(Mount.world_to_body(Mount.body_to_world(pb, ant, 0.0), ant, 0.0), pb)
    up = Mount(pitch_deg=-15).opt_to_body([0, 0, 1.0])            # optical axis of a camera tilted UP
    assert up[2] > 0.30 + 0.2, up
    print(f"ok: fit theta {math.degrees(th):.2f} deg, t ({tx:.3f}, {ty:.3f}), rms {rms * 100:.1f} cm")
