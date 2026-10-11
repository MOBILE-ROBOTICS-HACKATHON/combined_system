# Fake Crazyflie (no radio, no drone): same services and topics as cf_server for /<name>, with simple motion.
#   python3 scripts/cf_diag/fake_cf.py MODE [NAME=cf1] [SLOT_X=0.0]   (SLOT_X: where it sits on the ANT, m along its x)
# Practice the whole setup (cf_guard, cf_rviz, cf_keys, cf_goto) or test code changes before flying.
#   python3 scripts/cf_diag/fake_cf.py [normal | drift | lowbat | linkloss | room | nolh | onant | crashed]  (instead of cf_server)
# drift: after takeoff it slides +x out of the fence (guard must land it); lowbat: 3.2 V after 3 s;
# linkloss: no data 4..8 s after takeoff. room: Lighthouse-like, stands at room (-3.4, 2.3), floor z=-0.15.
# nolh: Lighthouse without base stations: the estimate jumps around on the ground (takeoff must be refused).
# onant: for the ANT missions (needs ant_link + a fake ANT). While landed ON the ANT the drone rides along:
#   room pose = HIDDEN TRUE transform (theta 30 deg, t (2.0, -1.0); map = R room + t) of /ant/pose_map,
#   z = plate 0.16 (floor 0). mission_calib must recover it. Flies at <= 0.3 m/s. Landing at a height that
#   isn't the plate (when above the ANT) or the floor: it DROPS (prints it). FAKE_BAT_DROP=V/s battery drain.
import os, sys, time, math, rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy
from crazyflie_interfaces.srv import Takeoff, Land, GoTo, NotifySetpointsStop, Arm
from crazyflie_interfaces.msg import Status
from std_srvs.srv import Empty
from std_msgs.msg import String
from geometry_msgs.msg import PoseStamped
TH, TX, TY, PLATE = math.radians(30), 2.0, -1.0, 0.16
from nav_msgs.msg import Odometry
from geometry_msgs.msg import TransformStamped
from tf2_ros import TransformBroadcaster
from ament_index_python.packages import get_package_share_directory
mode = sys.argv[1] if len(sys.argv) > 1 else "normal"
NAME = sys.argv[2] if len(sys.argv) > 2 else "cf1"
SLOT_X = float(sys.argv[3]) if len(sys.argv) > 3 else 0.0
class F(Node):
    def __init__(self):
        super().__init__(f"fake_{NAME}")
        self.p = [-3.4, 2.3, -0.15] if mode == 'room' else [0.0, 0.0, 0.0]; self.target = list(self.p); self.flying = False; self.locked = False; self.t_off = None; self.landing = False
        def d(r): return r.duration.sec + r.duration.nanosec / 1e9
        def to(q, s):
            if not self.armed: print("FAKE takeoff REFUSED: not armed", flush=True); return s
            self.landing = False; self.target = [self.p[0], self.p[1], q.height]; self.flying = True; self.t_off = time.monotonic(); print(f"FAKE takeoff h={q.height:.2f} d={d(q)}", flush=True); return s
        def la(q, s): self.target = [self.p[0], self.p[1], q.height]; self.landing = True; print(f"FAKE land d={d(q)}", flush=True); return s
        def go(q, s):
            g = [q.goal.x, q.goal.y, q.goal.z]
            self.target = [a + b for a, b in zip(self.target, g)] if q.relative else g
            if self.landing: print("FAKE go_to CANCELS the landing in progress (like the real firmware)", flush=True)
            self.landing = False
            print(f"FAKE go_to rel={q.relative} goal=({q.goal.x:.2f},{q.goal.y:.2f},{q.goal.z:.2f}) yaw={q.yaw:.3f} d={d(q):.2f} -> target=({self.target[0]:.2f},{self.target[1]:.2f},{self.target[2]:.2f})", flush=True); return s
        def em(q, s): self.locked = True; self.flying = False; self.p[2] = 0.0; self.target = list(self.p); print("FAKE EMERGENCY", flush=True); return s
        def ns(q, s): print("FAKE notify_setpoints_stop", flush=True); return s
        self.create_service(Takeoff, f"/{NAME}/takeoff", to); self.create_service(Land, f"/{NAME}/land", la)
        self.create_service(GoTo, f"/{NAME}/go_to", go); self.create_service(Empty, f"/{NAME}/emergency", em)
        self.create_service(NotifySetpointsStop, f"/{NAME}/notify_setpoints_stop", ns)
        self.armed = False
        def arm(q, s): self.armed = q.arm; print(f"FAKE arm={q.arm}", flush=True); return s
        self.create_service(Arm, f"/{NAME}/arm", arm)
        self.pose = self.create_publisher(PoseStamped, f"/{NAME}/pose", 10); self.odom = self.create_publisher(Odometry, f"/{NAME}/odom", 10)
        self.st = self.create_publisher(Status, f"/{NAME}/status", 10)
        self.create_publisher(String, f"/{NAME}/robot_description", QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)).publish(String(data=open(get_package_share_directory("crazyflie_description") + "/urdf/crazyflie_description.urdf").read().replace("$NAME", NAME)))
        self.tfb = TransformBroadcaster(self)
        self.yaw, self.ant, self.riding, self.bat = 0.0, None, mode == "onant", 4.1
        if mode == "onant":
            def ant(m):
                q = m.pose.orientation; c, s_ = math.cos(TH), math.sin(TH); x, y = m.pose.position.x - TX, m.pose.position.y - TY
                yr = math.atan2(2 * q.w * q.z, 1 - 2 * q.z * q.z) - TH
                self.ant = (c * x + s_ * y + SLOT_X * math.cos(yr), -s_ * x + c * y + SLOT_X * math.sin(yr), yr)   # its slot on the plate
            self.create_subscription(PoseStamped, "/ant/pose_map", ant, 10)
        self.create_timer(0.1, self.tick)
    def tick(self):
        if mode == "drift" and self.flying and self.p[2] > 0.3 and self.t_off and time.monotonic() - self.t_off > 3: self.target[0] += 0.05
        step = 0.03 if mode == "onant" else 0.05
        if mode == "onant" and self.riding and self.ant and not self.flying:
            self.p = [self.ant[0], self.ant[1], PLATE]; self.yaw = self.ant[2]; self.target = list(self.p)
        for i in range(3):
            e = self.target[i] - self.p[i]; self.p[i] += max(min(e, step), -step)
        if mode == "onant" and self.flying: self.bat -= float(os.environ.get("FAKE_BAT_DROP", "0")) * 0.1
        if mode == 'nolh' and not self.flying: self.p = [__import__('random').uniform(-5, 5) for _ in range(3)]
        if self.flying and abs(self.p[2] - self.target[2]) < 0.01 and self.landing:
            self.flying = False
            if mode == "onant":
                over = self.ant and math.hypot(self.p[0] - self.ant[0], self.p[1] - self.ant[1]) < 0.25
                self.riding = bool(over and abs(self.p[2] - PLATE) < 0.03)
                if not self.riding and abs(self.p[2]) > 0.03:
                    print(f"FAKE landed IN THE AIR at z={self.p[2]:.2f}: DROPPED to the floor", flush=True); self.p[2] = 0.0; self.target = list(self.p)
                print(f"FAKE landed {'ON THE ANT' if self.riding else 'on the floor'} at ({self.p[0]:.2f},{self.p[1]:.2f},{self.p[2]:.2f}), "
                      f"{math.hypot(self.p[0] - self.ant[0], self.p[1] - self.ant[1]) if self.ant else -1:.2f} m from the ANT centre", flush=True)
            else:
                print("FAKE landed", flush=True)
        lost = mode == "linkloss" and self.t_off and 4 < time.monotonic() - self.t_off < 8
        if not lost:
            m = PoseStamped(); m.header.frame_id = "world"; m.pose.position.x, m.pose.position.y, m.pose.position.z = self.p; m.pose.orientation.z, m.pose.orientation.w = math.sin(self.yaw / 2), math.cos(self.yaw / 2); self.pose.publish(m)
            m.header.stamp = self.get_clock().now().to_msg()
            o = Odometry(); o.header = m.header; o.child_frame_id = NAME; o.pose.pose = m.pose; self.odom.publish(o)
            t = TransformStamped(); t.header = m.header; t.child_frame_id = NAME
            t.transform.translation.x, t.transform.translation.y, t.transform.translation.z = self.p; t.transform.rotation.w = 1.0; self.tfb.sendTransform(t)
            s = Status(); s.battery_voltage = 3.2 if (mode == "lowbat" and self.flying and time.monotonic() - self.t_off > 3) else (self.bat if mode == "onant" else 3.9)
            s.supervisor_info = (Status.SUPERVISOR_INFO_IS_FLYING if self.flying else 0) | (Status.SUPERVISOR_INFO_IS_LOCKED if self.locked else 0) \
                | (0 if self.locked else Status.SUPERVISOR_INFO_CAN_BE_ARMED) | (Status.SUPERVISOR_INFO_IS_ARMED if self.armed else 0) \
                | (128 if mode == "crashed" else 0); s.rssi = 40; self.st.publish(s)   # like real firmware; mode crashed = 11 Oct bug
rclpy.init(); rclpy.spin(F())
