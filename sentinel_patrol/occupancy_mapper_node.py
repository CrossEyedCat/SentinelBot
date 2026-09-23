"""Occupancy grid of the patrolled area, built from the LiDAR and the odometry.

The robot has no map when it starts: it discovers where it may drive by patrolling. This node turns
each /scan into evidence about the world and accumulates it in a log-odds occupancy grid published on
/map, which RViz2 draws as the field of free (white), occupied (black) and not-yet-seen (grey) cells.
Clicking a free cell with RViz's "2D Goal Pose" tool sends the robot there (see the NAVIGATING state
of the FSM), so this map is what makes click-to-drive possible.

It also broadcasts the transforms nothing else publishes in this simulation:
    odom -> base_link   from /odom, every message
    base_link -> <sensor frames>   static, at the poses the CAD gives them

Log-odds update per scan, for every ray that returned a hit:
    cells along the ray   l += l_free   (something would have blocked the beam, so they are empty)
    cell at the endpoint  l += l_occ    (the beam stopped there)
A ray that returns nothing (out of range) only clears, up to its maximum range. Clamping the total
keeps a cell able to change its mind when the world changes, which a plain counter cannot do.

It also records the trail the robot has actually driven and publishes it as /path_travelled, which
RViz draws over the map, so the route of a patrol or of a drive to a clicked goal is visible.

Subscribes: /scan (sensor_msgs/LaserScan), /odom (nav_msgs/Odometry)
Publishes:  /map (nav_msgs/OccupancyGrid, transient_local so RViz gets it on connect),
            /path_travelled (nav_msgs/Path), /tf, /tf_static
"""
import math

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped, TransformStamped
from nav_msgs.msg import OccupancyGrid, Odometry, Path
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import LaserScan
from tf2_ros import StaticTransformBroadcaster, TransformBroadcaster

# where each sensor sits on the chassis (metres, from the Fusion model) -> static transforms for RViz
SENSOR_FRAMES = {
    'lidar_link': (-0.025, 0.0, 0.142, 0.0),
    'imu_link': (0.0, 0.0, 0.023, 0.0),
    'us_front_link': (0.125, 0.0, 0.052, 0.0),
    'us_rear_link': (-0.125, 0.0, 0.052, math.pi),
    'us_left_link': (0.0, 0.095, 0.052, math.pi / 2),
    'us_right_link': (0.0, -0.095, 0.052, -math.pi / 2),
    'ir_fl_link': (0.113, 0.083, 0.045, math.pi / 4),
    'ir_fr_link': (0.113, -0.083, 0.045, -math.pi / 4),
    'ir_rl_link': (-0.113, 0.083, 0.045, 3 * math.pi / 4),
    'ir_rr_link': (-0.113, -0.083, 0.045, -3 * math.pi / 4),
}


def yaw_to_quat(yaw: float):
    return (0.0, 0.0, math.sin(yaw / 2.0), math.cos(yaw / 2.0))


def quat_to_yaw(q) -> float:
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))


class OccupancyMapper(Node):

    def __init__(self) -> None:
        super().__init__('occupancy_mapper')
        self.declare_parameter('resolution', 0.05)      # m per cell
        self.declare_parameter('size_m', 16.0)          # square map side, centred on the odom origin
        self.declare_parameter('publish_period', 1.0)   # s
        self.declare_parameter('map_frame', 'odom')     # no loop closure here, so map == odom
        self.declare_parameter('base_frame', 'base_link')
        self.declare_parameter('l_free', -0.4)
        self.declare_parameter('l_occ', 0.85)
        self.declare_parameter('l_min', -2.5)
        self.declare_parameter('l_max', 3.5)
        self.declare_parameter('occupied_above', 0.65)  # probability shown as occupied
        self.declare_parameter('free_below', 0.35)
        self.declare_parameter('max_clear_range', 5.0)  # how far a ray that hit nothing may clear
        self.declare_parameter('publish_tf', True)

        self.res = float(self.get_parameter('resolution').value)
        size_m = float(self.get_parameter('size_m').value)
        self.n = int(round(size_m / self.res))
        self.origin = -size_m / 2.0                     # world coordinate of cell (0, 0)
        self.map_frame = self.get_parameter('map_frame').value
        self.base_frame = self.get_parameter('base_frame').value
        self.l_free = float(self.get_parameter('l_free').value)
        self.l_occ = float(self.get_parameter('l_occ').value)
        self.l_min = float(self.get_parameter('l_min').value)
        self.l_max = float(self.get_parameter('l_max').value)
        self.occ_above = float(self.get_parameter('occupied_above').value)
        self.free_below = float(self.get_parameter('free_below').value)
        self.max_clear = float(self.get_parameter('max_clear_range').value)

        self.grid = np.zeros((self.n, self.n), dtype=np.float32)   # log-odds, 0 = unknown
        self.pose = None                                            # (x, y, yaw) in the odom frame
        self.scan_frame = None
        self.updates = 0

        sensor_qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,
                                history=HistoryPolicy.KEEP_LAST, depth=5)
        map_qos = QoSProfile(reliability=ReliabilityPolicy.RELIABLE, history=HistoryPolicy.KEEP_LAST,
                             depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.create_subscription(LaserScan, '/scan', self._scan_cb, sensor_qos)
        self.create_subscription(Odometry, '/odom', self._odom_cb, sensor_qos)
        self.map_pub = self.create_publisher(OccupancyGrid, '/map', map_qos)
        self.create_timer(float(self.get_parameter('publish_period').value), self._publish_map)

        # the trail the robot has actually driven, drawn over the map in RViz
        self.declare_parameter('path_step', 0.05)      # m of travel between recorded points
        self.declare_parameter('path_max_points', 3000)
        self.path_step = float(self.get_parameter('path_step').value)
        self.path_max = int(self.get_parameter('path_max_points').value)
        self.path = Path()
        self.path.header.frame_id = self.map_frame
        self.path_pub = self.create_publisher(Path, '/path_travelled', 10)
        self.create_timer(0.5, self._publish_path)

        self.tf_on = bool(self.get_parameter('publish_tf').value)
        if self.tf_on:
            self.tf = TransformBroadcaster(self)
            self.static_tf = StaticTransformBroadcaster(self)
            self._send_static_tf()
        self.get_logger().info(
            f'occupancy_mapper: {self.n}x{self.n} cells at {self.res} m '
            f'({size_m:.0f} x {size_m:.0f} m centred on the {self.map_frame} origin)')

    # ------------------------------------------------------------------ transforms
    def _send_static_tf(self) -> None:
        stamp = self.get_clock().now().to_msg()
        out = []
        for frame, (x, y, z, yaw) in SENSOR_FRAMES.items():
            tf = TransformStamped()
            tf.header.stamp = stamp
            tf.header.frame_id = self.base_frame
            tf.child_frame_id = frame
            tf.transform.translation.x, tf.transform.translation.y, tf.transform.translation.z = x, y, z
            q = yaw_to_quat(yaw)
            tf.transform.rotation.x, tf.transform.rotation.y = q[0], q[1]
            tf.transform.rotation.z, tf.transform.rotation.w = q[2], q[3]
            out.append(tf)
        self.static_tf.sendTransform(out)

    def _odom_cb(self, msg: Odometry) -> None:
        p, q = msg.pose.pose.position, msg.pose.pose.orientation
        self.pose = (p.x, p.y, quat_to_yaw(q))

        # record the trail: a new point every path_step of travel, so a long patrol stays cheap
        last = self.path.poses[-1].pose.position if self.path.poses else None
        if last is None or math.hypot(p.x - last.x, p.y - last.y) >= self.path_step:
            ps = PoseStamped()
            ps.header.frame_id = self.map_frame
            ps.header.stamp = msg.header.stamp
            ps.pose = msg.pose.pose
            self.path.poses.append(ps)
            if len(self.path.poses) > self.path_max:
                del self.path.poses[:len(self.path.poses) - self.path_max]

        if not self.tf_on:
            return
        tf = TransformStamped()
        tf.header.stamp = msg.header.stamp
        tf.header.frame_id = self.map_frame
        tf.child_frame_id = self.base_frame
        tf.transform.translation.x, tf.transform.translation.y = p.x, p.y
        tf.transform.translation.z = 0.0
        tf.transform.rotation = q
        self.tf.sendTransform(tf)

    # ------------------------------------------------------------------ mapping
    def _scan_cb(self, scan: LaserScan) -> None:
        if self.pose is None:
            return                                        # no pose yet: the scan cannot be placed
        self.scan_frame = scan.header.frame_id
        px, py, yaw = self.pose
        # the LiDAR sits ahead of / above the chassis centre, so cast the rays from where it really is
        lx_b, ly_b = SENSOR_FRAMES.get(self.scan_frame, (0.0, 0.0, 0.0, 0.0))[:2]
        ox = px + lx_b * math.cos(yaw) - ly_b * math.sin(yaw)
        oy = py + lx_b * math.sin(yaw) + ly_b * math.cos(yaw)

        rng = np.asarray(scan.ranges, dtype=np.float32)
        ang = scan.angle_min + np.arange(rng.size, dtype=np.float32) * scan.angle_increment + yaw
        hit = np.isfinite(rng) & (rng > scan.range_min) & (rng < scan.range_max)
        # A ray that hit something clears everything in front of that hit. A ray that returned nothing
        # only says "empty for a while": it is the weakest evidence there is, and where two wall panels
        # of the arena meet with a millimetre gap a handful of such rays escape and would otherwise
        # paint a 12 m corridor of free space through a solid wall. Cap how far they may clear.
        reach = np.where(hit, rng, np.float32(min(self.max_clear, scan.range_max)))
        keep = np.isfinite(reach) & (reach > scan.range_min)
        if not keep.any():
            return
        reach, ang, hit = reach[keep], ang[keep], hit[keep]

        # sample every ray at half-cell spacing: column j of `dist` is one ray, rows are steps along it
        step = self.res * 0.5
        steps = int(math.ceil(float(reach.max()) / step))
        dist = (np.arange(1, steps + 1, dtype=np.float32)[:, None]) * step
        inside = dist < (reach[None, :] - self.res)       # stop short of the endpoint cell
        fx = ox + dist * np.cos(ang)[None, :]
        fy = oy + dist * np.sin(ang)[None, :]
        self._add(fx[inside], fy[inside], self.l_free)
        if hit.any():
            self._add(ox + reach[hit] * np.cos(ang[hit]), oy + reach[hit] * np.sin(ang[hit]), self.l_occ)
        self.updates += 1

    def _add(self, xs, ys, delta: float) -> None:
        """Add `delta` to the log-odds of every cell the given world points fall in."""
        ix = ((np.asarray(xs) - self.origin) / self.res).astype(np.int32)
        iy = ((np.asarray(ys) - self.origin) / self.res).astype(np.int32)
        ok = (ix >= 0) & (ix < self.n) & (iy >= 0) & (iy < self.n)
        if not ok.any():
            return
        # np.add.at accumulates repeated cells instead of overwriting them, so a cell crossed by
        # several rays in one scan gains the evidence of each
        np.add.at(self.grid, (iy[ok], ix[ok]), np.float32(delta))
        np.clip(self.grid, self.l_min, self.l_max, out=self.grid)

    def _publish_path(self) -> None:
        if not self.path.poses:
            return
        self.path.header.stamp = self.get_clock().now().to_msg()
        self.path_pub.publish(self.path)

    def _publish_map(self) -> None:
        if self.updates == 0:
            return
        prob = 1.0 - 1.0 / (1.0 + np.exp(self.grid))      # log-odds -> probability
        cells = np.full(self.grid.shape, -1, dtype=np.int8)          # unknown
        cells[prob >= self.occ_above] = 100                          # occupied
        free = (prob <= self.free_below) & (self.grid != 0.0)
        cells[free] = 0                                              # free and driveable
        mid = (self.grid != 0.0) & (prob > self.free_below) & (prob < self.occ_above)
        cells[mid] = np.clip(prob[mid] * 100.0, 1, 99).astype(np.int8)

        msg = OccupancyGrid()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.map_frame
        msg.info.resolution = self.res
        msg.info.width = msg.info.height = self.n
        msg.info.origin.position.x = msg.info.origin.position.y = self.origin
        msg.info.origin.orientation.w = 1.0
        msg.data = cells.reshape(-1).tolist()
        self.map_pub.publish(msg)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = OccupancyMapper()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
