"""Record everything the mission video's costmap panel needs, on the simulation clock.

    ros2 run ... python3 tools/mission_logger.py <outdir> <max_seconds>

Writes <outdir>/log.json (poses, planned routes, goals, states, mission progress, costmap index)
and <outdir>/costmaps.npz (one int8 grid per /costmap message, compressed - the grid is mostly
uniform so it packs down hard). Everything is stamped in simulation seconds, which is the clock the
camera frames carry too, so the panel can be rendered one frame per camera frame afterwards.

Stops on MISSION_COMPLETE (after a short tail so the last arrival is in the log) or max_seconds.
"""
import json
import math
import os
import sys
import time

import numpy as np
import rclpy
from rclpy.executors import ExternalShutdownException
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import OccupancyGrid, Odometry, Path
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import String

TAIL = 6.0          # s of wall time to keep logging after MISSION_COMPLETE
FLUSH = 20.0        # s of wall time between partial writes of log.json
POSE_PERIOD = 0.1   # s between logged poses


def stamp_of(header) -> float:
    return header.stamp.sec + header.stamp.nanosec * 1e-9


class MissionLogger(Node):

    def __init__(self, outdir: str, max_seconds: float) -> None:
        super().__init__('mission_logger')
        # String topics have no header, so they are stamped from the node clock: that clock has
        # to be the simulation one, or those stamps cannot be compared with the camera frames
        self.set_parameters([Parameter('use_sim_time', Parameter.Type.BOOL, True)])
        self.outdir = outdir
        self.max_seconds = max_seconds
        os.makedirs(outdir, exist_ok=True)
        self.t0 = None
        self.finish_at = None
        self.done = False

        self.poses = []      # [t, x, y, yaw]
        self.plans = []      # [t, [[x, y], ...]]
        self.goals = []      # [t, x, y]
        self.states = []     # [t, text]
        self.mission = []    # [t, text]
        self.n_grids = 0     # grids are saved as they arrive, not held
        self.grid_t = []
        self.meta = None
        self.last_pose_t = -1e9

        latched = QoSProfile(reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL,
                             history=HistoryPolicy.KEEP_LAST, depth=1)
        sensor = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST, depth=5)
        self.create_subscription(OccupancyGrid, '/costmap', self._cost_cb, latched)
        self.create_subscription(Odometry, '/odom', self._odom_cb, sensor)
        self.create_subscription(Path, '/path_planned', self._plan_cb, 10)
        self.create_subscription(PoseStamped, '/goal_pose', self._goal_cb, 10)
        self.create_subscription(String, '/patrol_state', self._state_cb, 10)
        self.create_subscription(String, '/mission_status', self._mission_cb, 10)

    def now(self) -> float:
        return self.get_clock().now().nanoseconds * 1e-9

    def _mark(self, t: float) -> None:
        if self.t0 is None:
            self.t0 = t

    def _cost_cb(self, msg: OccupancyGrid) -> None:
        t = stamp_of(msg.header)
        self._mark(t)
        if self.meta is None:
            self.meta = {'resolution': msg.info.resolution, 'width': msg.info.width,
                         'height': msg.info.height,
                         'origin': [msg.info.origin.position.x, msg.info.origin.position.y]}
        grid = np.asarray(msg.data, dtype=np.int8).reshape(msg.info.height, msg.info.width)
        # written out as it arrives, not stacked and saved at the end: a run that dies on the way
        # out then still leaves everything it recorded on disk
        np.save(os.path.join(self.outdir, 'grid_%05d.npy' % len(self.grid_t)), grid)
        self.grid_t.append(t)
        self.n_grids += 1

    def _odom_cb(self, msg: Odometry) -> None:
        t = stamp_of(msg.header)
        self._mark(t)
        if t - self.last_pose_t < POSE_PERIOD:
            return
        self.last_pose_t = t
        p, q = msg.pose.pose.position, msg.pose.pose.orientation
        yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))
        self.poses.append([t, p.x, p.y, yaw])
        if self.max_seconds and self.t0 is not None and t - self.t0 > self.max_seconds:
            self.done = True
        if self.finish_at is not None and time.monotonic() > self.finish_at:
            self.done = True

    def _plan_cb(self, msg: Path) -> None:
        t = stamp_of(msg.header)
        self._mark(t)
        self.plans.append([t, [[p.pose.position.x, p.pose.position.y] for p in msg.poses]])

    def _goal_cb(self, msg: PoseStamped) -> None:
        t = stamp_of(msg.header)
        self._mark(t)
        self.goals.append([t, msg.pose.position.x, msg.pose.position.y])

    def _state_cb(self, msg: String) -> None:
        self.states.append([self.now(), msg.data])

    def _mission_cb(self, msg: String) -> None:
        t = self.now()
        self._mark(t)
        self.mission.append([t, msg.data])
        if ('MISSION_COMPLETE' in msg.data or 'MISSION_ABORTED' in msg.data) and self.finish_at is None:
            self.finish_at = time.monotonic() + TAIL
            self.get_logger().info('[LOG] mission complete, closing the log')

    def write(self, quiet: bool = False) -> None:
        log = {'t0': self.t0, 'meta': self.meta, 'poses': self.poses, 'plans': self.plans,
               'goals': self.goals, 'states': self.states, 'mission': self.mission,
               'grid_t': self.grid_t}
        with open(os.path.join(self.outdir, 'log.json'), 'w') as fh:
            json.dump(log, fh)
        if quiet:
            return
        print('   log: %d poses, %d plans, %d goals, %d costmaps, %.0f s'
              % (len(self.poses), len(self.plans), len(self.goals), self.n_grids,
                 (self.poses[-1][0] - self.t0) if self.poses and self.t0 else 0.0))


def main() -> None:
    outdir, max_seconds = sys.argv[1], float(sys.argv[2])
    rclpy.init()
    node = MissionLogger(outdir, max_seconds)
    last_flush = time.monotonic()
    try:
        while rclpy.ok() and not node.done:
            rclpy.spin_once(node, timeout_sec=0.05)
            if time.monotonic() - last_flush > FLUSH:
                node.write(quiet=True)      # a partial log on disk beats a perfect one in memory
                last_flush = time.monotonic()
    except BaseException as exc:
        # rclpy raises several different things when the context is torn down under it
        # (ExternalShutdownException, RCLError, KeyboardInterrupt); none of them may cost the run
        print('   logger stopping: %s' % type(exc).__name__)
    node.write()
    node.destroy_node()
    rclpy.try_shutdown()


if __name__ == '__main__':
    main()
