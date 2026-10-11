# Compare the SLAM map with what the lidar sees RIGHT NOW: map obstacles that the current scan sees through = stale.
import time, math, rclpy, numpy as np
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, qos_profile_sensor_data
from rclpy.time import Time
from nav_msgs.msg import OccupancyGrid
from sensor_msgs.msg import LaserScan
from rcl_interfaces.srv import GetParameters
from tf2_ros import Buffer, TransformListener
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
NS = "/olive/olixRobot/ant1/id001"; F = "olive_olixRobot_ant1_id001_"
rclpy.init(); n = rclpy.create_node("stale_check"); tf = Buffer(); TransformListener(tf, n)
q = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL)
got = {}; scans = []
n.create_subscription(OccupancyGrid, f"{NS}/map", lambda m: got.__setitem__("map", m), q)
n.create_subscription(LaserScan, f"{NS}/scan", lambda m: scans.append(m), qos_profile_sensor_data)
t0 = time.time()
while time.time() - t0 < 30 and ("map" not in got or len(scans) < 10 or not tf.can_transform(F + "map", F + "laser", Time())): rclpy.spin_once(n, timeout_sec=0.1)
scans = []
t1 = time.time()
while time.time() - t1 < 3: rclpy.spin_once(n, timeout_sec=0.05)   # fresh scans with TF available
print("tf ok:", tf.can_transform(F + "map", F + "laser", Time()), "scans:", len(scans))
m = got["map"]; W, H, res = m.info.width, m.info.height, m.info.resolution; ox, oy = m.info.origin.position.x, m.info.origin.position.y
M = np.array(m.data, dtype=np.int16).reshape(H, W)
seen_free = np.zeros((H, W), bool); hits = []
for s in scans[-10:]:
    try:
        t = tf.lookup_transform(F + "map", s.header.frame_id, Time()).transform
    except Exception as e:
        continue
    qz, qw = t.rotation.z, t.rotation.w; yaw = 2 * math.atan2(qz, qw); lx, ly = t.translation.x, t.translation.y
    for i, r in enumerate(s.ranges):
        if not (s.range_min < r < s.range_max) or not math.isfinite(r): continue
        a = yaw + s.angle_min + i * s.angle_increment
        hx, hy = lx + r * math.cos(a), ly + r * math.sin(a); hits.append((hx, hy))
        for k in np.arange(0.1, r - 0.15, res):               # cells the beam passes through = currently free
            cx, cy = int((lx + k * math.cos(a) - ox) / res), int((ly + k * math.sin(a) - oy) / res)
            if 0 <= cx < W and 0 <= cy < H: seen_free[cy, cx] = True
occ = M >= 65
stale = occ & seen_free
print(f"map: {W}x{H} cells, {occ.sum()} occupied cells")
print(f"occupied in the map but the lidar sees THROUGH them now (stale): {stale.sum()} cells ({100*stale.sum()/max(occ.sum(),1):.0f}% of all mapped obstacles)")
print(f"robot/laser at ({lx:.2f}, {ly:.2f})")
img = np.ones((H, W, 3)); img[M == -1] = (0.8, 0.8, 0.85); img[occ] = (0.15, 0.15, 0.15); img[stale] = (1.0, 0.1, 0.1)
fig, ax = plt.subplots(figsize=(10, 10 * H / W + 0.5))
ax.imshow(img, origin="lower", extent=[ox, ox + W * res, oy, oy + H * res])
h = np.array(hits); ax.plot(h[:, 0], h[:, 1], ".", ms=1.5, color=(0.1, 0.4, 1.0), label="lidar hits now")
ax.plot(lx, ly, "bo", ms=9, label="robot"); ax.plot(2.5, 2.5, "g*", ms=16, label="goal (2.5, 2.5)")
ax.legend(loc="upper left"); ax.grid(True, alpha=0.4)
ax.set_title("SLAM map: black = obstacle, RED = mapped obstacle the lidar now sees through (stale), blue = lidar now")
fig.savefig("stale_map.png", dpi=90, bbox_inches="tight")
# global costmap / planner settings
def get(node, names):
    c = n.create_client(GetParameters, f"{NS}/{node}/get_parameters"); c.wait_for_service(timeout_sec=8)
    f = c.call_async(GetParameters.Request(names=names)); rclpy.spin_until_future_complete(n, f, timeout_sec=8)
    v = lambda x: {1: x.bool_value, 2: x.integer_value, 3: x.double_value, 4: x.string_value, 9: list(x.string_array_value)}.get(x.type, "(unset)")
    return {k: v(x) for k, x in zip(names, f.result().values)}
print(get("global_costmap/global_costmap", ["plugins", "rolling_window", "width", "height", "update_frequency", "track_unknown_space",
      "static_layer.map_subscribe_transient_local", "obstacle_layer.scan.obstacle_max_range", "obstacle_layer.scan.raytrace_max_range",
      "obstacle_layer.scan.clearing", "obstacle_layer.scan.marking", "obstacle_layer.footprint_clearing_enabled"]))
print(get("local_costmap/local_costmap", ["width", "height", "obstacle_layer.scan.obstacle_max_range", "obstacle_layer.scan.raytrace_max_range"]))
print(get("planner_server", ["GridBased.tolerance", "GridBased.use_astar", "GridBased.allow_unknown"]))
print(get("slam_toolbox", ["mode", "map_update_interval", "minimum_travel_distance", "minimum_travel_heading", "do_loop_closing", "max_laser_range"]))
