# Grab global costmap + SLAM map + robot pose, run a reachability flood fill, draw a picture.
import time, math, collections, rclpy, numpy as np
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from rclpy.time import Time
from nav_msgs.msg import OccupancyGrid
from tf2_ros import Buffer, TransformListener
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
NS = "/olive/olixRobot/ant1/id001"; F = "olive_olixRobot_ant1_id001_"
rclpy.init(); n = rclpy.create_node("grab_map"); tf = Buffer(); TransformListener(tf, n)
q = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL)
got = {}
n.create_subscription(OccupancyGrid, f"{NS}/global_costmap/costmap", lambda m: got.__setitem__("cost", m), q)
n.create_subscription(OccupancyGrid, f"{NS}/map", lambda m: got.__setitem__("map", m), q)
t0 = time.time(); pose = None
while time.time() - t0 < 25 and (len(got) < 2 or pose is None):
    rclpy.spin_once(n, timeout_sec=0.1)
    try:
        t = tf.lookup_transform(F + "map", F + "base_link", Time()).transform; pose = (t.translation.x, t.translation.y)
    except Exception: pass
c = got["cost"]; m = got["map"]
W, H, res, ox, oy = c.info.width, c.info.height, c.info.resolution, c.info.origin.position.x, c.info.origin.position.y
C = np.array(c.data, dtype=np.int16).reshape(H, W)
M = np.array(m.data, dtype=np.int16).reshape(m.info.height, m.info.width)
print(f"costmap {W}x{H} @ {res} m, x {ox:.2f}..{ox+W*res:.2f}, y {oy:.2f}..{oy+H*res:.2f}")
print(f"map     {m.info.width}x{m.info.height}, x {m.info.origin.position.x:.2f}..{m.info.origin.position.x+m.info.width*m.info.resolution:.2f}, y {m.info.origin.position.y:.2f}..{m.info.origin.position.y+m.info.height*m.info.resolution:.2f}")
print(f"robot at {pose}")
cell = lambda x, y: (int((x - ox) / res), int((y - oy) / res))
free = (C < 99)            # -1 (unknown) counts as free too: allow_unknown = true
rx, ry = cell(*pose)
print("cost at robot:", C[ry, rx] if 0 <= rx < W and 0 <= ry < H else "outside")
# start = nearest free cell to robot
fs = np.argwhere(free); d = (fs[:, 1] - rx) ** 2 + (fs[:, 0] - ry) ** 2; sy, sx = fs[np.argmin(d)]
reach = np.zeros_like(free); reach[sy, sx] = True; dq = collections.deque([(sx, sy)])
while dq:
    x, y = dq.popleft()
    for dx, dy in ((1,0),(-1,0),(0,1),(0,-1)):
        nx, ny = x + dx, y + dy
        if 0 <= nx < W and 0 <= ny < H and free[ny, nx] and not reach[ny, nx]:
            reach[ny, nx] = True; dq.append((nx, ny))
print(f"reachable area: {reach.sum()*res*res:.1f} m2 of {free.sum()*res*res:.1f} m2 free")
for gx, gy in [(2.5, 2.5), (2.52, 2.47), (2.0, 1.0), (1.0, 0.5)]:
    cx, cy = cell(gx, gy)
    inside = 0 <= cx < W and 0 <= cy < H
    print(f"goal ({gx},{gy}): " + (f"cost={C[cy,cx]} reachable={bool(reach[cy,cx])}" if inside else "OUTSIDE the costmap"))
rp = np.argwhere(reach); gx, gy = cell(2.5, 2.5); k = np.argmin((rp[:,1]-gx)**2 + (rp[:,0]-gy)**2)
print(f"closest reachable point to (2.5,2.5): ({ox+(rp[k,1]+.5)*res:.2f}, {oy+(rp[k,0]+.5)*res:.2f})")
img = np.zeros((H, W, 3)); img[:] = (0.75, 0.75, 0.75)
img[C == -1] = (0.55, 0.55, 0.6); img[(C >= 1) & (C < 99)] = (0.95, 0.85, 0.6)
img[C >= 99] = (0.9, 0.3, 0.3); img[C == 100] = (0.1, 0.1, 0.1); img[(C == 0)] = (1, 1, 1)
img[reach & (C == 0)] = (0.8, 1.0, 0.8)
fig, ax = plt.subplots(figsize=(9, 9 * H / W + 0.5))
ax.imshow(img, origin="lower", extent=[ox, ox + W * res, oy, oy + H * res])
ax.plot(*pose, "bo", ms=10, label="robot"); ax.plot(2.52, 2.47, "r*", ms=18, label="goal (2.52, 2.47)")
ax.plot(ox+(rp[k,1]+.5)*res, oy+(rp[k,0]+.5)*res, "g^", ms=12, label="closest reachable")
ax.grid(True, alpha=0.4); ax.set_xticks(np.arange(math.floor(ox), ox + W*res + 1, 1)); ax.set_yticks(np.arange(math.floor(oy), oy + H*res + 1, 1))
ax.legend(loc="upper left"); ax.set_title("global costmap: black=obstacle, red=robot can't fit, green=reachable, grey-blue=unknown")
fig.savefig("costmap.png", dpi=90, bbox_inches="tight")
