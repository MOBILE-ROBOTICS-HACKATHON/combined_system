# Passive Nav2 recorder v2 (only subscribes): pose, commands incl. y, DWB critic scores, logs, goal states.
import time, math, rclpy
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from rclpy.time import Time
from rcl_interfaces.msg import Log
from action_msgs.msg import GoalStatusArray
from geometry_msgs.msg import Twist, PoseStamped
from nav_msgs.msg import Path
from tf2_ros import Buffer, TransformListener
try:
    from dwb_msgs.msg import LocalPlanEvaluation
except Exception:
    LocalPlanEvaluation = None
NS = "/olive/olixRobot/ant1/id001"; F = "olive_olixRobot_ant1_id001_"
rclpy.init(); n = rclpy.create_node("nav_recorder2"); t0 = time.monotonic()
def p(s): print(f"{time.monotonic()-t0:6.1f}s {s}", flush=True)
tf = Buffer(); TransformListener(tf, n)
latched = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL)
NAMES = {1: "ACCEPTED", 2: "EXECUTING", 3: "CANCELING", 4: "SUCCEEDED", 5: "CANCELED", 6: "ABORTED"}
last = {}
def status(action):
    def cb(m):
        for s in m.status_list:
            k = (action, bytes(s.goal_info.goal_id.uuid))
            if last.get(k) != s.status and time.monotonic() - t0 > 2:
                p(f"ACTION {action}: {NAMES.get(s.status, s.status)}")
            last[k] = s.status
    return cb
for a in ["navigate_to_pose", "spin", "backup", "wait"]:
    n.create_subscription(GoalStatusArray, f"{NS}/{a}/_action/status", status(a), latched)
seen = {}
def log(m):
    if m.level >= 20 and any(k in m.name for k in ("controller", "planner", "bt_navigator", "behavior", "smoother")):
        key = m.msg[:70]
        if time.monotonic() - seen.get(key, -99) > 5: p(f"LOG {m.name.split('.')[-1]}: {m.msg}")
        seen[key] = time.monotonic()
n.create_subscription(Log, "/rosout", log, 200)
cmds = []
n.create_subscription(Twist, f"{NS}/cmd_vel", lambda m: cmds.append((m.linear.x, m.linear.y, m.angular.z)), 50)
n.create_subscription(PoseStamped, f"{NS}/goal_pose", lambda m: p(f"GOAL_POSE x={m.pose.position.x:.2f} y={m.pose.position.y:.2f} yaw={math.degrees(2*math.atan2(m.pose.orientation.z, m.pose.orientation.w)):.0f}"), 10)
plan = [None]
n.create_subscription(Path, f"{NS}/plan", lambda m: plan.__setitem__(0, m), 5)
ev = [None]
if LocalPlanEvaluation:
    n.create_subscription(LocalPlanEvaluation, f"{NS}/evaluation", lambda m: ev.__setitem__(0, m), 5)
def pose():
    try:
        t = tf.lookup_transform(F + "map", F + "base_link", Time()).transform
        q = t.rotation
        return t.translation.x, t.translation.y, math.degrees(math.atan2(2*(q.w*q.z+q.x*q.y), 1-2*(q.y*q.y+q.z*q.z)))
    except Exception:
        return None
nxt = t0 + 1; active = 0
while time.monotonic() - t0 < float(__import__("sys").argv[1] if len(__import__("sys").argv) > 1 else 600):
    rclpy.spin_once(n, timeout_sec=0.05)
    if time.monotonic() < nxt: continue
    nxt += 1
    c = cmds; cmds = []
    moving = any(abs(v) > 0.005 for t in c for v in t)
    if not moving:
        active = 0; continue
    ps = pose(); pl = plan[0]
    s = f"POSE {ps[0]:.2f},{ps[1]:.2f} {ps[2]:.0f}deg | " if ps else ""
    avg = [sum(t[i] for t in c)/len(c) for i in range(3)]
    s += f"CMD avg vx={avg[0]:+.2f} vy={avg[1]:+.2f} wz={avg[2]:+.2f} ({len(c)} msgs)"
    if pl and pl.poses:
        L = sum(math.hypot(a.pose.position.x-b.pose.position.x, a.pose.position.y-b.pose.position.y) for a, b in zip(pl.poses, pl.poses[1:]))
        s += f" | PLAN {L:.2f} m -> ({pl.poses[-1].pose.position.x:.2f},{pl.poses[-1].pose.position.y:.2f})"
    p(s)
    e = ev[0]
    if e and e.twists and active % 3 == 0:
        b = e.twists[e.best_index]
        crit = ", ".join(f"{sc.name}={sc.raw_score*sc.scale:.1f}" for sc in b.scores if sc.raw_score*sc.scale != 0)
        nvalid = sum(1 for t in e.twists if t.total >= 0)
        p(f"   DWB best vx={b.traj.velocity.x:+.2f} vy={b.traj.velocity.y:+.2f} wz={b.traj.velocity.theta:+.2f} total={b.total:.1f} [{crit}] valid {nvalid}/{len(e.twists)}")
    active += 1
p("recorder done")
