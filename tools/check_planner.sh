source /opt/ros/jazzy/setup.bash
source "${ROS_WS:-$HOME/ros2_ws}/install/setup.bash"   # ROS_WS: your colcon workspace
python3 - <<'PY'
import math
import numpy as np
import rclpy
from nav_msgs.msg import OccupancyGrid
from sentinel_patrol.path_planner_node import PathPlanner, BLOCKED, LETHAL

rclpy.init()
n = PathPlanner()

RES, N = 0.05, 160                 # 8 x 8 m at 5 cm
ORIGIN = -4.0
def cell(x, y):                    # world -> (row, col)
    return int((y - ORIGIN) / RES), int((x - ORIGIN) / RES)

grid = np.zeros((N, N), dtype=np.int16)
# a wall across the middle at x = 0, with two doorways of different widths
c_wall = int((0.0 - ORIGIN) / RES)
grid[:, c_wall - 1:c_wall + 2] = 100
def open_gap(y0, y1):
    grid[cell(0, y0)[0]:cell(0, y1)[0], c_wall - 1:c_wall + 2] = 0
open_gap(1.00, 1.30)               # 0.30 m: narrower than the robot
open_gap(-1.50, -0.90)             # 0.60 m: wide enough
grid[0, :] = grid[-1, :] = grid[:, 0] = grid[:, -1] = 100   # outer walls

msg = OccupancyGrid()
msg.info.resolution = RES
msg.info.width = msg.info.height = N
msg.info.origin.position.x = msg.info.origin.position.y = ORIGIN
msg.data = grid.reshape(-1).astype(np.int8).tolist()
n._map_cb(msg)

r_needed = n.r_block
print(f"robot footprint radius incl. margin: {r_needed:.3f} m  (a gap must exceed {2*r_needed:.3f} m)")

narrow_mid = n.cost[cell(0.0, 1.15)]
wide_mid = n.cost[cell(0.0, -1.20)]
print(f"middle of the 0.30 m gap: costmap = {narrow_mid}  -> {'BLOCKED by footprint' if narrow_mid >= BLOCKED else 'passable'}")
print(f"middle of the 0.60 m gap: costmap = {wide_mid}  -> {'BLOCKED by footprint' if wide_mid >= BLOCKED else 'passable'}")

start, goal = cell(-2.0, 0.0), cell(2.0, 0.0)
cells = n._astar(start, goal)
if cells is None:
    print("A*: NO PATH")
else:
    pts = [n._to_world(r, c) for r, c in n._shortcut(cells)]
    ys_at_wall = [y for x, y in [n._to_world(r, c) for r, c in cells] if abs(x) < 0.08]
    length = sum(math.dist(pts[i], pts[i + 1]) for i in range(len(pts) - 1))
    print(f"A*: {len(cells)} cells -> {len(pts)} waypoints, {length:.2f} m long")
    print(f"    crosses the wall at y = {min(ys_at_wall):.2f} .. {max(ys_at_wall):.2f}")
    used = "wide (0.60 m)" if min(ys_at_wall) < 0 else "narrow (0.30 m)"
    print(f"    doorway used: {used}")
    print(f"    waypoints: {[(round(x,2), round(y,2)) for x, y in pts]}")
    clear = min(n.cost[n._to_cell(x, y)] for x, y in pts)
    print(f"    worst costmap value on the route: {clear} (must stay below {BLOCKED})")

# and a goal placed inside a wall should be moved to somewhere the robot fits
bad = cell(0.0, 0.0)
snap = n._nearest_passable(*bad, n.snap_r)
print("goal clicked on the wall ->", "no free spot within the snap radius" if snap is None
      else f"moved to {tuple(round(v,2) for v in n._to_world(*snap))}")
n.destroy_node(); rclpy.shutdown()
PY
