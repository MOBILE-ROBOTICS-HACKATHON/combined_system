import time, rclpy
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from action_msgs.srv import CancelGoal
from action_msgs.msg import GoalStatusArray
from geometry_msgs.msg import Twist
from std_msgs.msg import Bool
from rclpy.qos import qos_profile_sensor_data
NS = "/olive/olixRobot/ant1/id001"
NAMES = {0: "UNKNOWN", 1: "ACCEPTED", 2: "EXECUTING", 3: "CANCELING", 4: "SUCCEEDED", 5: "CANCELED", 6: "ABORTED"}
rclpy.init(); n = rclpy.create_node("secure")
latched = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL)
st = {}; ems = []
for a in ["navigate_to_pose", "follow_path", "spin", "backup"]:
    n.create_subscription(GoalStatusArray, f"{NS}/{a}/_action/status", lambda m, a=a: st.__setitem__(a, m), latched)
n.create_subscription(Bool, f"{NS}/ems", lambda m: ems.append(m.data), qos_profile_sensor_data)
pub = n.create_publisher(Twist, f"{NS}/cmd_vel", 10)
t0 = time.time()
while time.time() - t0 < 6: rclpy.spin_once(n, timeout_sec=0.1)
def show(tag):
    for a, m in st.items():
        print(f"{tag} {a}: " + ", ".join(f"{bytes(s.goal_info.goal_id.uuid).hex()[:6]}={NAMES[s.status]}@{s.goal_info.stamp.sec}" for s in m.status_list))
show("BEFORE")
print("e-stop pressed:", ems[-1] if ems else "?")
for a in ["navigate_to_pose", "follow_path", "spin", "backup"]:
    c = n.create_client(CancelGoal, f"{NS}/{a}/_action/cancel_goal")
    if c.wait_for_service(timeout_sec=5):
        f = c.call_async(CancelGoal.Request()); rclpy.spin_until_future_complete(n, f, timeout_sec=5)
        r = f.result(); print(f"cancel {a}: return_code={r.return_code if r else '?'} cancelled={len(r.goals_canceling) if r else '?'}")
for _ in range(10): pub.publish(Twist()); rclpy.spin_once(n, timeout_sec=0.05)
t0 = time.time()
while time.time() - t0 < 3: rclpy.spin_once(n, timeout_sec=0.1)
show("AFTER")
