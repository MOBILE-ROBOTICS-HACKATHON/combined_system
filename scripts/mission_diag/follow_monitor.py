#!/usr/bin/env python3
"""Passive monitor of the follow chain (drone domain 42): what mission_follow sends, where it lands in the
ANT map, what Nav2 reports. Only subscribes.   python3 scripts/mission_diag/follow_monitor.py [seconds]"""
import math
import sys
import time
import rclpy
from rclpy.qos import DurabilityPolicy, QoSProfile
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Bool, Empty, String
from team_pkg.mission_common import Calib, yaw_of

rclpy.init()
n = rclpy.create_node("follow_monitor")
L = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
st = {"calib": None, "ant": None, "antr": None, "cf": None, "nav": "?", "moving": None}
t0 = time.time()


def p(s):
    print(f"{time.time() - t0:5.1f}s {s}", flush=True)


def nav(m):
    if m.data != st["nav"]:
        p(f"NAV STATE -> {m.data}")
    st["nav"] = m.data


def goal(m):
    x, y = m.pose.position.x, m.pose.position.y
    s = f"GOAL from mission ({m.header.frame_id}) ({x:+.2f},{y:+.2f})"
    c = st["calib"]
    if c and m.header.frame_id == "world":
        mx, my = c.room_to_map(x, y)
        s += f" -> map ({mx:+.2f},{my:+.2f})"
        if st["ant"]:
            s += f", {math.hypot(mx - st['ant'][0], my - st['ant'][1]):.2f} m from the ANT"
    p(s)


n.create_subscription(String, "/mission/calib", lambda m: st.__setitem__("calib", Calib.from_yaml(m.data)), L)
n.create_subscription(PoseStamped, "/ant/pose_map", lambda m: st.__setitem__(
    "ant", (m.pose.position.x, m.pose.position.y, math.degrees(yaw_of(m.pose.orientation)))), 10)
n.create_subscription(PoseStamped, "/ant/pose_room", lambda m: st.__setitem__("antr", (m.pose.position.x, m.pose.position.y)), 10)
n.create_subscription(PoseStamped, "/cf1/pose", lambda m: st.__setitem__("cf", (m.pose.position.x, m.pose.position.y, m.pose.position.z)), 10)
n.create_subscription(Bool, "/ant/moving", lambda m: st.__setitem__("moving", m.data), 10)
n.create_subscription(String, "/ant/nav_state", nav, L)
n.create_subscription(PoseStamped, "/ant/goal", goal, 10)
n.create_subscription(Empty, "/ant/stop", lambda m: p("STOP sent"), 10)
last = 0.0
while time.time() - t0 < float(sys.argv[1] if len(sys.argv) > 1 else 60):
    for _ in range(30):
        rclpy.spin_once(n, timeout_sec=0)
    time.sleep(0.05)
    if time.time() - last > 1.0 and st["ant"]:
        last = time.time()
        a, c, r = st["ant"], st["cf"], st["antr"]
        d = math.hypot(c[0] - r[0], c[1] - r[1]) if c and r else float("nan")
        p(f"ANT map ({a[0]:+.2f},{a[1]:+.2f}) yaw {a[2]:+4.0f} {'MOVING' if st['moving'] else 'still '} | drone room "
          + (f"({c[0]:+.2f},{c[1]:+.2f},{c[2]:.2f})" if c else "?") + f" | drone-ANT {d:.2f} m | nav {st['nav']}")
