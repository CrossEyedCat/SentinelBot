"""Mission: drive a full lap around every column in the arena.

    ros2 run sentinel_patrol column_tour
    ros2 run sentinel_patrol column_tour --ros-args -p ring_radius:=0.55 -p points_per_lap:=10

The node is only a mission sequencer: it hands one goal at a time to the FSM on /goal_pose, the
same topic an RViz click uses, and waits for /goal_status. Everything else - planning around the
column, obstacle avoidance, the final approach - is the existing stack. A lap is a ring of
waypoints around one column, entered on the side the robot arrives from and left on the side the
next column is on:

    theta_in  = bearing of the previous column (or of the robot, for the first lap)
    theta_out = bearing of the next column (or theta_in, for the last lap)
    sweep     = 2pi + the shorter of the two arcs from theta_in to theta_out
    waypoints every sweep / steps, walked in the direction that made that arc the shorter one

The ring radius is half the column spacing, which is both the widest circle that clears the
neighbouring columns and the narrowest that clears this one. Measured on the recorded costmap: the
band the planner closes around a column ends at r = 0.43 m, and a 0.55 m circle has a worst cost of
60 the whole way round, so the centre of the robot has 0.22 m of corner clearance by construction.
Half the spacing also makes neighbouring rings touch, which is what lets one lap's exit be the next
lap's entry.

The chord between neighbouring ring points passes ring_radius * cos(pi/n) from the column centre -
0.52 m at the defaults, well clear of the 0.43 m blocked band, so every hop is a straight line and
the lap is a polygon walked round the column. Two earlier versions were worse: four points at
0.45 m put the chord inside the blocked band, and six points at 0.55 m left the robot enough room
between waypoints to drift off the ring.

Circling a column at all means living inside the distances the reactive layer exists to protect. On
the ring the column is 0.40 m from the chassis and about 0.22 m from a corner, against a patrol
d_stop of 0.50 m and an ir_critical of 0.12 m, so an untouched profile reads the lap itself as a
continuous emergency. tools/record_column_tour.sh therefore sets a mission profile before the tour
starts and leaves the shipped defaults alone; it is the same trade the final approach already makes
when it drops to d_stop_final 0.22 m to reach a goal parked against a wall.

A waypoint that comes back GOAL_UNREACHABLE, or that takes longer than goal_timeout, is skipped and
counted: the tour must not stall on one bad point.

Subscribes: /odom, /goal_status, /patrol_state
Publishes:  /goal_pose (geometry_msgs/PoseStamped), /mission_status (std_msgs/String)
"""
import math

import rclpy
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry, Path
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import String

# the nine columns of turtlebot3_world, in world coordinates, walked as a serpentine so the
# robot never crosses the arena to reach the next one
COLUMNS = [-1.1, -1.1, -1.1, 0.0, -1.1, 1.1,
           0.0, 1.1, 0.0, 0.0, 0.0, -1.1,
           1.1, -1.1, 1.1, 0.0, 1.1, 1.1]


class ColumnTour(Node):

    CONTINUOUS_GOALS = 3    # progress goals per lap when the ring is driven as a path
    ALERT_GRACE = 3.0       # s in ALERT before the mission asks the operator channel for a reset
    MAX_ALERT_RESETS = 3    # after this many, something is wrong that resetting will not fix

    def __init__(self) -> None:
        super().__init__('column_tour')
        self.declare_parameter('columns', COLUMNS)        # flat [x0, y0, x1, y1, ...], world frame
        self.declare_parameter('ring_radius', 0.55)       # m; column 0.15 + footprint 0.226 = 0.376 min
        self.declare_parameter('points_per_lap', 10)
        self.declare_parameter('spawn_x', -2.0)           # where the model was spawned in the world
        self.declare_parameter('spawn_y', -0.5)
        self.declare_parameter('goal_timeout', 90.0)      # s before a waypoint is given up on
        self.declare_parameter('settle', 0.3)             # s of stop between waypoints
        self.declare_parameter('restate_after', 6.0)      # s not navigating before the goal is re-sent
        self.declare_parameter('continuous', False)       # one goal per lap, with the ring as a path
        self.declare_parameter('transit_via_planner', False)  # reach each ring's entry through A* first
        self.declare_parameter('transit_reach', 0.6)      # m; closer than this and the lap starts at once

        flat = list(self.get_parameter('columns').value)
        self.columns = list(zip(flat[0::2], flat[1::2]))
        self.radius = float(self.get_parameter('ring_radius').value)
        self.n_lap = int(self.get_parameter('points_per_lap').value)
        self.spawn = (float(self.get_parameter('spawn_x').value), float(self.get_parameter('spawn_y').value))
        self.goal_timeout = float(self.get_parameter('goal_timeout').value)
        self.settle = float(self.get_parameter('settle').value)
        self.restate_after = float(self.get_parameter('restate_after').value)
        self.continuous = bool(self.get_parameter('continuous').value)
        self.transit_via_planner = bool(self.get_parameter('transit_via_planner').value)
        self.transit_reach = float(self.get_parameter('transit_reach').value)
        self.transit = False        # driving to the next ring's entry on the planner's path
        self._lap = None            # (dense ring, progress goals) held back until the transit is over

        self.offset = None          # world -> odom, worked out from the first odometry sample
        self.pose = None            # robot position in the odom frame
        self.state = ''
        self.col_i = 0              # which column
        self.way = []               # ring waypoints (odom frame) of the current column
        self.way_i = 0
        self.sent_at = None         # time the current waypoint was published
        self.navigating_at = None   # last time the FSM reported NAVIGATING
        self.hold_until = None      # settle timer between waypoints
        self.reached = 0
        self.skipped = 0
        self.alert_since = None     # when the FSM went to ALERT, if it is there now
        self.alert_resets = 0
        self.done = False

        qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST, depth=5)
        self.create_subscription(Odometry, '/odom', self._odom_cb, qos)
        self.create_subscription(String, '/goal_status', self._status_cb, 10)
        self.create_subscription(String, '/patrol_state', self._state_cb, 10)
        self.goal_pub = self.create_publisher(PoseStamped, '/goal_pose', 10)
        self.status_pub = self.create_publisher(String, '/mission_status', 10)
        self.cmd_pub = self.create_publisher(String, '/patrol_cmd', 10)
        self.path_pub = self.create_publisher(Path, '/mission_path', 10)
        self.create_timer(0.2, self._tick)
        self.get_logger().info(
            f'column_tour: {len(self.columns)} columns, ring {self.radius:.2f} m, '
            f'{self.n_lap} waypoints per full turn plus the arc out to the next column')

    # ------------------------------------------------------------------ inputs
    def _odom_cb(self, msg: Odometry) -> None:
        p = msg.pose.pose.position
        if self.offset is None:
            # /odom may start at the world origin or at the spawn pose depending on how the
            # OdometryPublisher was configured; measure it instead of assuming either
            self.offset = (self.spawn[0] - p.x, self.spawn[1] - p.y)
            self.get_logger().info(f'world -> odom offset {self.offset[0]:+.2f} {self.offset[1]:+.2f}')
        self.pose = (p.x, p.y)

    def _state_cb(self, msg: String) -> None:
        self.state = msg.data
        if msg.data == 'NAVIGATING':
            self.navigating_at = self._now()
        if msg.data != 'ALERT':
            self.alert_since = None
        elif self.alert_since is None:
            self.alert_since = self._now()

    def _status_cb(self, msg: String) -> None:
        if self.done or self.sent_at is None:
            return
        if msg.data.startswith('GOAL_REACHED'):
            self.reached += 1
            self._advance('reached')
        elif msg.data.startswith('GOAL_UNREACHABLE'):
            self.skipped += 1
            self._advance(f'skipped ({msg.data})')

    # ------------------------------------------------------------------ sequencing
    def _now(self) -> float:
        return self.get_clock().now().nanoseconds * 1e-9

    def _to_odom(self, x: float, y: float) -> tuple:
        return x - self.offset[0], y - self.offset[1]

    def _plan_lap(self) -> None:
        """Ring waypoints around the current column: in at the near side, round, out at the far side.

        The lap is entered at the ring point facing where the robot came from and left at the point
        facing the next column. Neighbouring columns are 1.1 m apart and the ring is half of that,
        so the two rings touch: one column's exit point IS the next column's entry point and the
        tour never crosses the arena between laps. The direction of travel is whichever of the two
        makes the arc from entry to exit shorter, so no lap is walked the long way round.
        """
        cx, cy = self._to_odom(*self.columns[self.col_i])
        if self.col_i == 0:
            th_in = math.atan2(self.pose[1] - cy, self.pose[0] - cx)
        else:
            px, py = self._to_odom(*self.columns[self.col_i - 1])
            th_in = math.atan2(py - cy, px - cx)
        if self.col_i + 1 < len(self.columns):
            nx, ny = self._to_odom(*self.columns[self.col_i + 1])
            th_out = math.atan2(ny - cy, nx - cx)
        else:
            th_out = th_in          # nothing to leave for: close the lap where it started

        ccw = (th_out - th_in) % (2.0 * math.pi)
        turn = 1.0 if ccw <= math.pi else -1.0
        delta = ccw if turn > 0 else 2.0 * math.pi - ccw
        sweep = 2.0 * math.pi + delta       # a full turn, then on to the exit
        steps = max(self.n_lap, int(math.ceil(sweep / (2.0 * math.pi / self.n_lap))))
        step = turn * sweep / steps
        self.way = [(cx + self.radius * math.cos(th_in + k * step),
                     cy + self.radius * math.sin(th_in + k * step))
                    for k in range(steps + 1)]
        self.way_i = 0

        if self.continuous:
            # The lap as a curve rather than a queue of arrivals. Stopping at every waypoint is what
            # made the robot stand still for a quarter of the run; published as one path, the
            # controller drives the whole lap and only has to arrive a few times.
            #
            # A few, not once: the FSM's goal is a point, and a lap is not. With a single goal at the
            # ring's end the robot reports arrival the moment it starts, because a closed ring ends
            # where it begins. Three goals spaced a third of the sweep apart are far enough apart to
            # be unambiguous, and still cut the stops from eleven a lap to three.
            dense = [(cx + self.radius * math.cos(th_in + turn * a),
                      cy + self.radius * math.sin(th_in + turn * a))
                     for a in self._dense_angles(sweep)]
            goals = [(cx + self.radius * math.cos(th_in + turn * sweep * k / self.CONTINUOUS_GOALS),
                      cy + self.radius * math.sin(th_in + turn * sweep * k / self.CONTINUOUS_GOALS))
                     for k in range(1, self.CONTINUOUS_GOALS + 1)]
            entry = dense[0]
            far = math.hypot(self.pose[0] - entry[0], self.pose[1] - entry[1]) > self.transit_reach
            if self.transit_via_planner and far:
                # In the base arena neighbouring rings touch, so one lap's exit is the next lap's
                # entry and the straight line between them is the path. With shelves or walls in
                # between it is not: hand the FSM the entry as a plain goal with no mission path,
                # so the A* route drives the transit, and hold the ring back until it arrives.
                self._lap = (dense, goals)
                self._publish_path([])
                self.way = [entry]
                self.transit = True
                self._report(f'column {self.col_i + 1}/{len(self.columns)}: transit to the ring entry '
                             f'({entry[0]:+.2f}, {entry[1]:+.2f}) on the planned route')
                return
            self._publish_path(dense)
            self.way = goals
        self._report(f'column {self.col_i + 1}/{len(self.columns)} at '
                     f'({self.columns[self.col_i][0]:+.1f}, {self.columns[self.col_i][1]:+.1f}): lap started')

    def _start_lap(self) -> None:
        """The transit is over (arrived, or gave up): now the ring."""
        dense, goals = self._lap
        self._lap, self.transit = None, False
        self._publish_path(dense)
        self.way, self.way_i = goals, 0
        self._report(f'column {self.col_i + 1}/{len(self.columns)} at '
                     f'({self.columns[self.col_i][0]:+.1f}, {self.columns[self.col_i][1]:+.1f}): lap started')

    def _dense_angles(self, sweep: float, step_m: float = 0.05):
        n = max(8, int(round(sweep * self.radius / step_m)))
        return [sweep * k / n for k in range(n + 1)]

    def _publish_path(self, pts) -> None:
        msg = Path()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'odom'
        for x, y in pts:
            ps = PoseStamped()
            ps.header = msg.header
            ps.pose.position.x = float(x)
            ps.pose.position.y = float(y)
            ps.pose.orientation.w = 1.0
            msg.poses.append(ps)
        self.path_pub.publish(msg)

    def _send(self) -> None:
        x, y = self.way[self.way_i]
        msg = PoseStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'odom'
        msg.pose.position.x = x
        msg.pose.position.y = y
        msg.pose.orientation.w = 1.0
        self.goal_pub.publish(msg)
        self.sent_at = self._now()
        self.navigating_at = None
        self._report(f'column {self.col_i + 1}/{len(self.columns)} '
                     f'waypoint {self.way_i + 1}/{len(self.way)} -> odom ({x:+.2f}, {y:+.2f})')

    def _advance(self, why: str) -> None:
        self.get_logger().info(f'[MISSION] column {self.col_i + 1} waypoint {self.way_i + 1}: {why}')
        self.sent_at = None
        self.hold_until = self._now() + self.settle
        self.way_i += 1
        if self.way_i >= len(self.way) and self.transit:
            self._start_lap()
            return
        if self.way_i >= len(self.way):
            self.col_i += 1
            self.way = []
            if self.col_i >= len(self.columns):
                self.done = True
                if self.continuous:
                    self._publish_path([])
                self._report(f'MISSION_COMPLETE laps={len(self.columns)} '
                             f'reached={self.reached} skipped={self.skipped}')

    def _report(self, text: str) -> None:
        self.status_pub.publish(String(data=text))
        self.get_logger().info(f'[MISSION] {text}')

    def _tick(self) -> None:
        if self.done or self.pose is None or self.offset is None:
            return
        t = self._now()
        if self.hold_until is not None and t < self.hold_until:
            return
        self.hold_until = None
        if not self.way:
            self._plan_lap()
        if self.sent_at is None:
            self._send()
            return
        if t - self.sent_at > self.goal_timeout:
            self.skipped += 1
            self._advance(f'skipped (no arrival in {self.goal_timeout:.0f} s)')
            return
        # An alert is not something to drive through. The FSM refuses goals while it is in ALERT, so
        # re-sending one is pointless: a first run spent 220 goals on a robot that had already hit a
        # column and was never going to move again. Ask the operator channel for a reset, and if that
        # does not clear it a few times over, stop the mission rather than pretend it is running.
        if self.state == 'ALERT':
            if self.alert_since is not None and t - self.alert_since > self.ALERT_GRACE:
                if self.alert_resets >= self.MAX_ALERT_RESETS:
                    self.done = True
                    if self.continuous:
                        self._publish_path([])
                    self._report(f'MISSION_ABORTED in ALERT after {self.alert_resets} resets, '
                                 f'column {self.col_i + 1}/{len(self.columns)}, '
                                 f'reached={self.reached} skipped={self.skipped}')
                    return
                self.alert_resets += 1
                self.alert_since = t
                self.cmd_pub.publish(String(data='reset'))
                self._report(f'ALERT: asked for a reset ({self.alert_resets}/{self.MAX_ALERT_RESETS})')
            return

        # the FSM drops its goal on an operator stop; re-send rather than wait forever
        idle_for = t - (self.navigating_at or self.sent_at)
        if self.state != 'NAVIGATING' and idle_for > self.restate_after:
            self.get_logger().warning(f'[MISSION] state {self.state or "?"} for {idle_for:.0f} s, re-sending goal')
            self._send()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = ColumnTour()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
