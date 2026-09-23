"""ROS2 node wrapping the PatrolFSM.

Subscribes:  /ultrasonic, /ultrasonic_rear, /ultrasonic_left, /ultrasonic_right (sensor_msgs/Range)
             /ir_left, /ir_right                                                (sensor_msgs/Range)
             /imu                                                               (sensor_msgs/Imu)
             /patrol_cmd                                                        (std_msgs/String: start|stop|reset)
Publishes:   /cmd_vel      (geometry_msgs/Twist)
             /robot_alert  (std_msgs/String)
             /patrol_state (std_msgs/String, every tick, for rqt / screenshots)
             /goal_status  (std_msgs/String)
             /path_to_goal (nav_msgs/Path, the line currently being followed)

Every transition is logged as:  <old state> -> <new state> | trigger <sensor> = <value>
"""
import math
import os

import rclpy
from ament_index_python.packages import get_package_share_directory
from geometry_msgs.msg import PoseStamped, Twist, TwistStamped
from nav_msgs.msg import Odometry, Path
from rcl_interfaces.msg import SetParametersResult
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Imu, LaserScan, Range
from std_msgs.msg import String

from sentinel_patrol.fsm_core import FsmParams, PatrolFSM, SensorFrame


def _tilt_deg_from_quat(x: float, y: float, z: float, w: float) -> float:
    """Largest of |roll|, |pitch| in degrees from an orientation quaternion."""
    sinr = 2.0 * (w * x + y * z)
    cosr = 1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(sinr, cosr)
    sinp = max(-1.0, min(1.0, 2.0 * (w * y - z * x)))
    pitch = math.asin(sinp)
    return math.degrees(max(abs(roll), abs(pitch)))


class PatrolFsmNode(Node):

    def __init__(self) -> None:
        super().__init__('patrol_fsm')
        defaults = FsmParams()
        # every threshold is a ROS parameter so it can be tuned at runtime (ros2 param set)
        for name, value in vars(defaults).items():
            self.declare_parameter(name, value)
        self.declare_parameter('loop_rate', 10.0)
        # ROS2 Humble TurtleBot3 expects geometry_msgs/Twist on /cmd_vel; the Jazzy ros_gz bridge
        # expects geometry_msgs/TwistStamped. Same FSM, different envelope.
        self.declare_parameter('cmd_vel_stamped', False)
        self.stamped = bool(self.get_parameter('cmd_vel_stamped').value)
        self.declare_parameter('map_frame', 'odom')     # frame the map and the clicked goals live in
        self.declare_parameter('pilot', False)          # learned steering inside NAVIGATING
        self.declare_parameter('pilot_weights', '')     # .npz from tools/train_pilot.py
        self.declare_parameter('prefer_mission_path', True)   # /mission_path beats /path_planned
        self.map_frame = self.get_parameter('map_frame').value
        self.p = self._read_params()
        self.pilot = self._load_pilot()
        self.mission_path = []
        self.fsm = PatrolFSM(self.p, t0=self._now(), pilot=self.pilot)
        self.frame = SensorFrame()

        sensor_qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,
                                history=HistoryPolicy.KEEP_LAST, depth=5)
        self.create_subscription(Range, '/ultrasonic', self._us_front_cb, sensor_qos)
        self.create_subscription(Range, '/ultrasonic_rear', self._us_rear_cb, sensor_qos)
        self.create_subscription(Range, '/ultrasonic_left', self._us_left_cb, sensor_qos)
        self.create_subscription(Range, '/ultrasonic_right', self._us_right_cb, sensor_qos)
        self.create_subscription(Range, '/ir_left', self._ir_left_cb, sensor_qos)
        self.create_subscription(Range, '/ir_right', self._ir_right_cb, sensor_qos)
        self.create_subscription(Imu, '/imu', self._imu_cb, sensor_qos)
        self.create_subscription(String, '/patrol_cmd', self._cmd_cb, 10)
        # click-to-drive: RViz's "2D Goal Pose" tool publishes the clicked point on /goal_pose
        self.create_subscription(Odometry, '/odom', self._odom_cb, sensor_qos)
        self.create_subscription(PoseStamped, '/goal_pose', self._goal_cb, 10)
        # the route the planner worked out over the costmap, and whether it managed to find one
        self.create_subscription(Path, '/path_planned', self._plan_cb, 10)
        self.create_subscription(Path, '/mission_path', self._mission_path_cb, 10)
        self.create_subscription(String, '/plan_status', self._plan_status_cb, 10)

        self.cmd_pub = self.create_publisher(TwistStamped if self.stamped else Twist, '/cmd_vel', 10)
        self.alert_pub = self.create_publisher(String, '/robot_alert', 10)
        self.state_pub = self.create_publisher(String, '/patrol_state', 10)
        self.goal_pub = self.create_publisher(String, '/goal_status', 10)
        # the line the robot is currently trying to follow, drawn over the map in RViz
        self.goal_path_pub = self.create_publisher(Path, '/path_to_goal', 10)

        # live tuning: `ros2 param set /patrol_fsm d_stop 0.6` must reach the running FSM, not only
        # the node's parameter server. FsmParams is shared with the FSM, so writing into it is enough.
        self.add_on_set_parameters_callback(self._on_params)

        rate = self.get_parameter('loop_rate').value
        self.create_timer(1.0 / rate, self._tick)
        self.get_logger().info(f'PatrolFSM ready in state {self.fsm.state} '
                               f'(holonomic={self.p.holonomic}, auto_start={self.p.auto_start})')

    # ------------------------------------------------------------------ learned steering
    def _load_pilot(self):
        """Load the imitation-trained policy, if one is asked for and present.

        A missing or unreadable file is a warning, not a failure: the robot then drives with the
        hand-written law, which is the behaviour every test in this package covers.
        """
        if not self.get_parameter('pilot').value:
            return None
        path = str(self.get_parameter('pilot_weights').value)
        if not path:
            share = get_package_share_directory('sentinel_patrol')
            path = os.path.join(share, 'config', 'pilot_mlp.npz')
        try:
            from sentinel_patrol.pilot.runtime import PilotAdapter
            pilot = PilotAdapter(path, logger=self.get_logger())
        except Exception as exc:                                    # noqa: BLE001
            self.get_logger().warning(f'pilot disabled: {type(exc).__name__}: {exc}')
            return None
        self.create_subscription(LaserScan, '/scan', self._scan_cb,
                                 QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,
                                            history=HistoryPolicy.KEEP_LAST, depth=5))
        self.get_logger().info(f'pilot: steering NAVIGATING from {os.path.basename(path)}')
        return pilot

    # ------------------------------------------------------------------ params / time
    def _read_params(self) -> FsmParams:
        p = FsmParams()
        for name in vars(p):
            setattr(p, name, self.get_parameter(name).value)
        return p

    def _on_params(self, params) -> SetParametersResult:
        """Apply `ros2 param set` changes to the live FsmParams (type-checked against the defaults)."""
        for prm in params:
            if not hasattr(self.p, prm.name):
                continue                                   # loop_rate, cmd_vel_stamped, use_sim_time
            current = getattr(self.p, prm.name)
            if isinstance(current, bool) != isinstance(prm.value, bool) or \
                    (not isinstance(current, bool) and not isinstance(prm.value, (int, float))):
                return SetParametersResult(successful=False, reason=f'{prm.name}: expected {type(current).__name__}')
            setattr(self.p, prm.name, type(current)(prm.value))
            self.get_logger().info(f'parameter {prm.name} = {getattr(self.p, prm.name)} (live)')
        return SetParametersResult(successful=True)

    def _now(self) -> float:
        return self.get_clock().now().nanoseconds * 1e-9

    # ------------------------------------------------------------------ sensor callbacks
    def _us_front_cb(self, msg: Range) -> None:
        self.frame.us_front = msg.range
        self.frame.t_us_front = self._now()

    def _us_rear_cb(self, msg: Range) -> None:
        self.frame.us_rear = msg.range

    def _us_left_cb(self, msg: Range) -> None:
        self.frame.us_left = msg.range

    def _us_right_cb(self, msg: Range) -> None:
        self.frame.us_right = msg.range

    def _ir_left_cb(self, msg: Range) -> None:
        self.frame.ir_left = msg.range
        self.frame.t_ir = self._now()

    def _ir_right_cb(self, msg: Range) -> None:
        self.frame.ir_right = msg.range
        self.frame.t_ir = self._now()

    def _imu_cb(self, msg: Imu) -> None:
        t = self._now()
        self.frame.t_imu = t
        # horizontal acceleration: gravity sits on z when the robot is level
        a_xy = math.hypot(msg.linear_acceleration.x, msg.linear_acceleration.y)
        q = msg.orientation
        tilt = _tilt_deg_from_quat(q.x, q.y, q.z, q.w)
        tr = self.fsm.on_imu(a_xy, tilt, t)
        if tr is not None:
            self._on_transition(tr)
            self._stop_now()

    def _cmd_cb(self, msg: String) -> None:
        self.get_logger().info(f'operator command: {msg.data}')
        self.fsm.on_command(msg.data)

    def _plan_cb(self, msg: Path) -> None:
        # A mission that publishes its own dense reference owns the route: the planner keeps running
        # (its costmap is what proves the route is clear) but its per-goal A* line, which is only as
        # long as one hop, would replace a reference the pilot needs several times that to steer by.
        if self.mission_path and self.get_parameter('prefer_mission_path').value:
            return
        pts = [(p.pose.position.x, p.pose.position.y) for p in msg.poses]
        if pts:
            self.fsm.on_path(pts)

    def _mission_path_cb(self, msg: Path) -> None:
        self.mission_path = [(p.pose.position.x, p.pose.position.y) for p in msg.poses]
        if self.mission_path:
            self.fsm.on_path(self.mission_path)

    def _plan_status_cb(self, msg: String) -> None:
        """The planner has looked at the costmap and cannot get the robot there. Say so at once rather
        than driving at the obstacle until t_goal_max expires."""
        if not msg.data.startswith('NO_PATH') or self.fsm.goal() is None:
            return
        why = msg.data[len('NO_PATH'):].strip()
        self.get_logger().warning(f'abandoning the goal: {why}')
        self.goal_pub.publish(String(data=f'GOAL_UNREACHABLE {why}'))
        tr = self.fsm.abandon_goal(self._now())
        if tr is not None:
            self.get_logger().info(f'[FSM] {tr}')
            self._stop_now()

    def _odom_cb(self, msg: Odometry) -> None:
        p, q = msg.pose.pose.position, msg.pose.pose.orientation
        self.frame.pose_x, self.frame.pose_y = p.x, p.y
        self.frame.yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))
        self.frame.t_pose = self._now()
        if self.pilot is not None:
            v = msg.twist.twist
            self.pilot.set_velocity(v.linear.x, v.linear.y, v.angular.z, self.frame.t_pose)

    def _scan_cb(self, msg: LaserScan) -> None:
        self.pilot.set_scan(
            self.pilot.sectors_from_scan(msg.ranges, msg.angle_min, msg.angle_increment, msg.range_max),
            self._now())

    def _goal_cb(self, msg: PoseStamped) -> None:
        """A point clicked on the map in RViz. Its frame must be the one the map is published in."""
        frame = msg.header.frame_id or self.map_frame
        if frame not in (self.map_frame, 'map', 'odom'):
            self.get_logger().warning(
                f'ignoring goal in frame "{frame}": this node has no transform for it; '
                f'set the RViz fixed frame to "{self.map_frame}"')
            return
        x, y = msg.pose.position.x, msg.pose.position.y
        if self.fsm.state == 'ALERT':
            self.get_logger().warning(f'goal ({x:.2f}, {y:.2f}) refused: robot is in ALERT, reset it first')
            self.goal_pub.publish(String(data=f'REFUSED x={x:.2f} y={y:.2f} reason=ALERT'))
            return
        self.fsm.on_goal(x, y, self._now())
        # Arriving takes the FSM through IDLE, which drops the route with the goal. A mission
        # driving a lap publishes its reference once and expects it to hold for the whole lap,
        # so it is reinstated here - otherwise the second goal of every lap is steered at
        # blind, and a pilot that needs a route to look along never gets to run at all.
        if self.mission_path and self.get_parameter('prefer_mission_path').value:
            self.fsm.on_path(self.mission_path)
        self.get_logger().info(f'goal accepted: x={x:.2f} y={y:.2f} ({frame})')
        self.goal_pub.publish(String(data=f'ACCEPTED x={x:.2f} y={y:.2f}'))

    # ------------------------------------------------------------------ main loop
    def _tick(self) -> None:
        t = self._now()
        cmd, tr = self.fsm.step(self.frame, t)
        if tr is not None:
            self._on_transition(tr)
        self._publish_cmd(cmd.vx, cmd.vy, cmd.wz)
        self.state_pub.publish(String(data=self.fsm.state))
        self._publish_goal_path()

    def _publish_goal_path(self) -> None:
        """Straight line from where the robot is to the goal it is driving at, or an empty path when
        there is no goal, which makes RViz erase the previous one."""
        path = Path()
        path.header.frame_id = self.map_frame
        path.header.stamp = self.get_clock().now().to_msg()
        goal = self.fsm.goal()
        if goal is not None and self.frame.has_pose():
            for x, y in ((self.frame.pose_x, self.frame.pose_y), goal):
                ps = PoseStamped()
                ps.header = path.header
                ps.pose.position.x, ps.pose.position.y = float(x), float(y)
                ps.pose.orientation.w = 1.0
                path.poses.append(ps)
        self.goal_path_pub.publish(path)

    def _publish_cmd(self, vx: float, vy: float, wz: float) -> None:
        twist = Twist()
        twist.linear.x = float(vx)
        twist.linear.y = float(vy)
        twist.angular.z = float(wz)
        if self.stamped:
            msg = TwistStamped()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.header.frame_id = 'base_link'
            msg.twist = twist
            self.cmd_pub.publish(msg)
        else:
            self.cmd_pub.publish(twist)

    def _on_transition(self, tr) -> None:
        # required logger output: current state, triggering sensor value, new state
        self.get_logger().info(f'[FSM] {tr}')
        if tr.reason in ('GOAL_REACHED', 'GOAL_UNREACHABLE'):
            self.goal_pub.publish(String(data=f'{tr.reason} {tr.sensor}={tr.value:.2f}'))
        if tr.dst == 'ALERT' and self.fsm.last_alert is not None:
            text = self.fsm.last_alert.as_text()
            self.alert_pub.publish(String(data=text))
            self.get_logger().error(text)

    def _stop_now(self) -> None:
        """Publish a zero velocity immediately (do not wait for the next 10 Hz tick)."""
        self._publish_cmd(0.0, 0.0, 0.0)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = PatrolFsmNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        if rclpy.ok():                      # context still valid: stop the robot before leaving
            node._publish_cmd(0.0, 0.0, 0.0)
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
