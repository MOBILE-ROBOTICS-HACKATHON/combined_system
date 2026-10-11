"""Shared ANT1 navigation pieces (used by ant1_goto and ant_link)."""
import collections
import math
import numpy as np

FRAME = "olive_olixRobot_ant1_id001_"
# Everything in Nav2 that can drive the robot: cancel ALL of them to stop it (10 Oct: navigate_to_pose
# aborted, but follow_path kept driving).
ACTIONS = ("navigate_to_pose", "follow_path", "spin", "backup", "drive_on_heading", "wait")


def yaw_of(q):
    return math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))


def free_goal(m, x, y, robot):
    """(x, y, note) to actually send: the goal itself if the robot can REACH it, else the closest
    point it can reach (e.g. next to the box you asked for, or the nearest spot outside a closed-off
    pocket). None if the robot can't move at all. m = global costmap (OccupancyGrid) or None. Unreachable goals make the planner fail ->
    Nav2 only spins/waits/backs up -> gives up (10 Oct: goal in a pocket at the top wall)."""
    if m is None:
        return x, y, "no costmap received: goal sent unchecked"
    W, H, res = m.info.width, m.info.height, m.info.resolution
    ox, oy = m.info.origin.position.x, m.info.origin.position.y
    cost = np.array(m.data, dtype=np.int16).reshape(H, W)
    free = cost < 99                             # 99 = robot can't fit, 100 = obstacle; -1 unknown = allowed
    cell = lambda px, py: (min(max(int((px - ox) / res), 0), W - 1), min(max(int((py - oy) / res), 0), H - 1))
    cells = np.argwhere(free)
    if cells.size == 0:
        return None
    rx, ry = cell(robot[0], robot[1])           # start from the free cell nearest the robot
    sy, sx = cells[np.argmin((cells[:, 1] - rx) ** 2 + (cells[:, 0] - ry) ** 2)]
    reach = np.zeros_like(free)
    reach[sy, sx] = True
    todo = collections.deque([(sx, sy)])
    while todo:                                  # flood fill: everywhere the robot can drive to
        cx, cy = todo.popleft()
        for nx, ny in ((cx + 1, cy), (cx - 1, cy), (cx, cy + 1), (cx, cy - 1)):
            if 0 <= nx < W and 0 <= ny < H and free[ny, nx] and not reach[ny, nx]:
                reach[ny, nx] = True
                todo.append((nx, ny))
    gx, gy = cell(x, y)
    inside = ox <= x < ox + W * res and oy <= y < oy + H * res
    if inside and reach[gy, gx]:
        return x, y, ("goal is in UNMAPPED space: the plan may change as the lidar sees it"
                      if cost[gy, gx] == -1 else None)
    pts = np.argwhere(reach)
    k = np.argmin((pts[:, 1] - gx) ** 2 + (pts[:, 0] - gy) ** 2)
    nx, ny = ox + (pts[k, 1] + 0.5) * res, oy + (pts[k, 0] + 0.5) * res
    why = ("is inside/at an obstacle or closed off (no gap wide enough for the robot)" if inside
           else "is outside the known map")
    return nx, ny, (f"({x:.2f}, {y:.2f}) {why}: going to the closest REACHABLE point "
                    f"({nx:.2f}, {ny:.2f}), {math.hypot(nx - x, ny - y):.2f} m from it")
