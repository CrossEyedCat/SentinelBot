"""Costmap and A* path planner over the map the robot built itself.

Two jobs, and the first is the one that makes the second honest:

1. **Costmap.** The occupancy map says where the *world* is solid. It says nothing about where a robot
   250 x 190 mm across can actually fit: the centre of the robot can never reach a cell closer to a
   wall than its own radius, and a gap narrower than the robot is not a route however free its cells
   look. So every lethal cell is inflated by the circumscribed radius of the chassis plus a safety
   margin, and everything inside that band is marked as blocked-by-footprint. Beyond it the cost
   decays, which makes the planner prefer the middle of a corridor to scraping along its wall.
   Published on /costmap: 100 = obstacle, 99 = blocked because of the robot's size, 1..98 = passable
   but close to something, 0 = clear, -1 = never seen.

2. **Planner.** A* over the cells the robot can actually occupy, 8-connected, with the costmap value
   added to the step cost. The resulting cell path is then shortened by line-of-sight so the robot
   drives straight lines between corners instead of a staircase. Published on /path_planned, which
   the FSM follows in its NAVIGATING state.

A goal clicked on top of an obstacle, or inside the inflated band around one, is snapped to the
nearest cell the robot can stand in; if there is none within goal_snap_radius, or no route exists,
an empty path is published and the FSM abandons the goal instead of driving at a wall for two minutes.

Subscribes: /map (nav_msgs/OccupancyGrid), /odom (nav_msgs/Odometry), /goal_pose (PoseStamped)
Publishes:  /costmap (nav_msgs/OccupancyGrid), /path_planned (nav_msgs/Path), /plan_status (String)
"""
import heapq
import math

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import OccupancyGrid, Odometry, Path
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from scipy.ndimage import distance_transform_edt
from std_msgs.msg import String

LETHAL = 100        # a real obstacle
BLOCKED = 99        # free in the world, but the robot's own size cannot fit here


class PathPlanner(Node):

    def __init__(self) -> None:
        super().__init__('path_planner')
        # 250 x 190 mm body, 248 mm over the wheels -> the circle that contains the robot at any yaw
        self.declare_parameter('robot_radius', 0.176)
        self.declare_parameter('safety_margin', 0.05)
        self.declare_parameter('inflation_radius', 0.55)   # cost decays to zero at this distance
        self.declare_parameter('lethal_threshold', 65)     # map value at or above this is an obstacle
        self.declare_parameter('allow_unknown', True)      # plan through cells never seen
        self.declare_parameter('unknown_cost', 40)         # ... but prefer known-free ones
        self.declare_parameter('cost_weight', 0.6)         # m of detour worth avoiding a cost-100 cell
        self.declare_parameter('replan_period', 2.0)
        self.declare_parameter('goal_snap_radius', 0.50)
        self.declare_parameter('map_frame', 'odom')

        self.r_block = float(self.get_parameter('robot_radius').value) + \
            float(self.get_parameter('safety_margin').value)
        self.r_inflate = float(self.get_parameter('inflation_radius').value)
        self.lethal_th = int(self.get_parameter('lethal_threshold').value)
        self.allow_unknown = bool(self.get_parameter('allow_unknown').value)
        self.unknown_cost = int(self.get_parameter('unknown_cost').value)
        self.cost_weight = float(self.get_parameter('cost_weight').value)
        self.snap_r = float(self.get_parameter('goal_snap_radius').value)
        self.map_frame = self.get_parameter('map_frame').value

        self.cost = None            # int16 costmap, same shape as the map
        self.info = None            # map metadata (resolution, origin, size)
        self.pose = None
        self.goal = None

        map_qos = QoSProfile(reliability=ReliabilityPolicy.RELIABLE, history=HistoryPolicy.KEEP_LAST,
                             depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        odom_qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,
                              history=HistoryPolicy.KEEP_LAST, depth=5)
        self.create_subscription(OccupancyGrid, '/map', self._map_cb, map_qos)
        self.create_subscription(Odometry, '/odom', self._odom_cb, odom_qos)
        self.create_subscription(PoseStamped, '/goal_pose', self._goal_cb, 10)
        self.cost_pub = self.create_publisher(OccupancyGrid, '/costmap', map_qos)
        self.path_pub = self.create_publisher(Path, '/path_planned', 10)
        self.status_pub = self.create_publisher(String, '/plan_status', 10)
        self.create_timer(float(self.get_parameter('replan_period').value), self._replan)
        self.get_logger().info(
            f'path_planner: footprint radius {self.r_block:.3f} m blocks the map, '
            f'cost decays to 0 at {self.r_inflate:.2f} m')

    # ------------------------------------------------------------------ inputs
    def _odom_cb(self, msg: Odometry) -> None:
        p = msg.pose.pose.position
        self.pose = (p.x, p.y)

    def _goal_cb(self, msg: PoseStamped) -> None:
        self.goal = (msg.pose.position.x, msg.pose.position.y)
        self._replan()

    def _map_cb(self, msg: OccupancyGrid) -> None:
        self.info = msg.info
        grid = np.asarray(msg.data, dtype=np.int16).reshape(msg.info.height, msg.info.width)
        res = msg.info.resolution
        lethal = grid >= self.lethal_th
        unknown = grid < 0

        # distance from every cell to the nearest obstacle, in metres
        if lethal.any():
            dist = distance_transform_edt(~lethal, sampling=res)
        else:
            dist = np.full(grid.shape, np.inf, dtype=np.float64)

        cost = np.zeros(grid.shape, dtype=np.int16)
        cost[lethal] = LETHAL
        # the band the robot's own body cannot enter, however free the world is there
        cost[(~lethal) & (dist <= self.r_block)] = BLOCKED
        # beyond it, a decaying preference for staying away from walls
        band = (~lethal) & (dist > self.r_block) & (dist < self.r_inflate)
        span = max(1e-6, self.r_inflate - self.r_block)
        cost[band] = np.clip(98.0 * (1.0 - (dist[band] - self.r_block) / span), 1, 98).astype(np.int16)
        cost[unknown & (cost < BLOCKED)] = -1        # keep "never seen" distinguishable
        self.cost = cost
        self._publish_costmap(msg.header)

    def _publish_costmap(self, header) -> None:
        msg = OccupancyGrid()
        msg.header = header
        msg.header.frame_id = self.map_frame
        msg.info = self.info
        msg.data = self.cost.reshape(-1).astype(np.int8).tolist()
        self.cost_pub.publish(msg)

    # ------------------------------------------------------------------ grid helpers
    def _to_cell(self, x, y):
        return (int((y - self.info.origin.position.y) / self.info.resolution),
                int((x - self.info.origin.position.x) / self.info.resolution))

    def _to_world(self, r, c):
        return (self.info.origin.position.x + (c + 0.5) * self.info.resolution,
                self.info.origin.position.y + (r + 0.5) * self.info.resolution)

    def _passable(self, r, c) -> bool:
        if not (0 <= r < self.cost.shape[0] and 0 <= c < self.cost.shape[1]):
            return False
        v = self.cost[r, c]
        if v < 0:
            return self.allow_unknown
        return v < BLOCKED

    def _cell_cost(self, r, c) -> float:
        v = int(self.cost[r, c])
        return self.unknown_cost if v < 0 else v

    def _nearest_passable(self, r, c, radius_m):
        """A goal clicked on a wall, or in the band the robot cannot fit into, is moved to the closest
        cell the robot could actually stand in."""
        if self._passable(r, c):
            return (r, c)
        rad = int(math.ceil(radius_m / self.info.resolution))
        best, best_d = None, None
        for dr in range(-rad, rad + 1):
            for dc in range(-rad, rad + 1):
                d = dr * dr + dc * dc
                if d > rad * rad or (best_d is not None and d >= best_d):
                    continue
                if self._passable(r + dr, c + dc):
                    best, best_d = (r + dr, c + dc), d
        return best

    # ------------------------------------------------------------------ planning
    def _replan(self) -> None:
        if self.cost is None or self.pose is None or self.goal is None:
            return
        start = self._to_cell(*self.pose)
        goal = self._to_cell(*self.goal)
        # the robot itself may be standing inside the inflated band (right next to a wall); let it out
        start = self._nearest_passable(*start, 0.40) or start
        snapped = self._nearest_passable(*goal, self.snap_r)
        if snapped is None:
            self._fail(f'goal is inside an obstacle or too tight for the robot '
                       f'(footprint radius {self.r_block:.2f} m)')
            return
        if snapped != goal:
            gx, gy = self._to_world(*snapped)
            self.get_logger().info(f'goal moved to the nearest place the robot fits: {gx:.2f}, {gy:.2f}')
        cells = self._astar(start, snapped)
        if cells is None:
            self._fail('no route over the costmap')
            return
        pts = [self._to_world(r, c) for r, c in self._shortcut(cells)]
        pts[-1] = self._to_world(*snapped)
        self._publish_path(pts)
        self.status_pub.publish(String(data=f'OK points={len(pts)}'))

    def _fail(self, why: str) -> None:
        self.get_logger().warning(f'no plan: {why}')
        self.status_pub.publish(String(data=f'NO_PATH {why}'))
        self._publish_path([])

    def _astar(self, start, goal):
        if not self._passable(*goal):
            return None
        res = self.info.resolution
        h = lambda rc: math.hypot(rc[0] - goal[0], rc[1] - goal[1]) * res
        open_q = [(h(start), 0.0, start)]
        came, g_of = {start: None}, {start: 0.0}
        nbrs = [(-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
                (-1, -1, 1.4142), (-1, 1, 1.4142), (1, -1, 1.4142), (1, 1, 1.4142)]
        limit = 400000
        while open_q and limit > 0:
            limit -= 1
            _, g, cur = heapq.heappop(open_q)
            if cur == goal:
                path, node = [], cur
                while node is not None:
                    path.append(node)
                    node = came[node]
                return path[::-1]
            if g > g_of.get(cur, float('inf')):
                continue
            for dr, dc, step in nbrs:
                nxt = (cur[0] + dr, cur[1] + dc)
                if not self._passable(*nxt):
                    continue
                if dr and dc and not (self._passable(cur[0] + dr, cur[1]) and
                                      self._passable(cur[0], cur[1] + dc)):
                    continue                      # no cutting diagonally through a corner
                ng = g + step * res * (1.0 + self.cost_weight * self._cell_cost(*nxt) / 100.0)
                if ng < g_of.get(nxt, float('inf')):
                    g_of[nxt] = ng
                    came[nxt] = cur
                    heapq.heappush(open_q, (ng + h(nxt), ng, nxt))
        return None

    def _line_clear(self, a, b) -> bool:
        """Bresenham between two cells: is every cell on the way one the robot may occupy?"""
        r0, c0 = a
        r1, c1 = b
        dr, dc = abs(r1 - r0), abs(c1 - c0)
        sr, sc = (1 if r1 > r0 else -1), (1 if c1 > c0 else -1)
        err = dr - dc
        while True:
            if not self._passable(r0, c0):
                return False
            if (r0, c0) == (r1, c1):
                return True
            e2 = 2 * err
            if e2 > -dc:
                err -= dc
                r0 += sr
            if e2 < dr:
                err += dr
                c0 += sc

    def _shortcut(self, cells):
        """Keep only the corners: from each kept cell, jump to the furthest cell still in line of sight."""
        out, i = [cells[0]], 0
        while i < len(cells) - 1:
            j = len(cells) - 1
            while j > i + 1 and not self._line_clear(cells[i], cells[j]):
                j -= 1
            out.append(cells[j])
            i = j
        return out

    def _publish_path(self, pts) -> None:
        path = Path()
        path.header.frame_id = self.map_frame
        path.header.stamp = self.get_clock().now().to_msg()
        for x, y in pts:
            ps = PoseStamped()
            ps.header = path.header
            ps.pose.position.x, ps.pose.position.y = float(x), float(y)
            ps.pose.orientation.w = 1.0
            path.poses.append(ps)
        self.path_pub.publish(path)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = PathPlanner()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
