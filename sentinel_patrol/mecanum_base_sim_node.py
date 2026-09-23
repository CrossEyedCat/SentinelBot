"""Simulation-side base controller for the SentinelBot Mk II mecanum platform.

Gazebo in this build cannot reproduce mecanum rollers (anisotropic wheel friction is ignored by both
dartsim and bullet-featherstone), so the model is a free dynamic body and this node applies the
RESULTANT force/torque that the four wheels would produce:

    cmd_vel (vx, vy, wz, base frame)  ->  F = m * clamp(Kp * e + Ki * int(e), +-a_max),   e = v_cmd - v
                                          tau = Iz * clamp(Kw * e_w + Kiw * int(e_w), +-alpha_max)

with the acceleration limits from the design check (a_max 0.6 m/s^2, traction-limited). The force is
rotated into the world frame and handed to the Gazebo ApplyLinkWrench system as persistent wrench
increments (see _apply for why). Everything downstream stays physical: contacts, tipping, and the
IMU, which is what the FSM's ALERT state relies on.

Subscribes: /cmd_vel (geometry_msgs/Twist), /odom (nav_msgs/Odometry, twist in base frame)
Publishes:  /wrench_clear (ros_gz_interfaces/Entity), /wrench_persistent (ros_gz_interfaces/EntityWrench),
            /wheel_{fl,fr,rl,rr}_cmd (std_msgs/Float64) - the angular velocity each wheel is turned at
"""
import math
import time

import rclpy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from ros_gz_interfaces.msg import Entity, EntityWrench
from std_msgs.msg import Float64


class MecanumBaseSim(Node):

    def __init__(self) -> None:
        super().__init__('mecanum_base_sim')
        self.declare_parameter('link_name', 'sentinel_mk2::base_link')
        self.declare_parameter('mass', 4.0)
        self.declare_parameter('inertia_z', 0.033)
        self.declare_parameter('a_max', 0.6)        # m/s^2, design check
        self.declare_parameter('alpha_max', 2.0)    # rad/s^2
        self.declare_parameter('kp_lin', 4.0)       # 1/s
        self.declare_parameter('ki_lin', 3.0)       # 1/s^2, removes the contact-friction offset
        self.declare_parameter('kp_ang', 8.0)       # 1/s
        self.declare_parameter('ki_ang', 8.0)       # 1/s^2
        self.declare_parameter('cmd_timeout', 0.5)  # s without cmd_vel -> brake
        self.link = self.get_parameter('link_name').value
        self.m = float(self.get_parameter('mass').value)
        self.iz = float(self.get_parameter('inertia_z').value)
        self.a_max = float(self.get_parameter('a_max').value)
        self.alpha_max = float(self.get_parameter('alpha_max').value)
        self.kp_lin = float(self.get_parameter('kp_lin').value)
        self.ki_lin = float(self.get_parameter('ki_lin').value)
        self.kp_ang = float(self.get_parameter('kp_ang').value)
        self.ki_ang = float(self.get_parameter('ki_ang').value)
        self.cmd_timeout = float(self.get_parameter('cmd_timeout').value)
        self.int_err = [0.0, 0.0, 0.0]      # integral of (v_cmd - v) for x, y, yaw
        self.last_t = None

        self.cmd = Twist()
        self.cmd_time = None
        self.total = [0.0, 0.0, 0.0]        # wrench currently summed inside ApplyLinkWrench
        self.last_compact = time.monotonic()
        self.compact_period = 2.0           # s
        qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST, depth=5)
        self.create_subscription(Twist, '/cmd_vel', self._cmd_cb, 10)
        self.create_subscription(Odometry, '/odom', self._odom_cb, qos)
        self.clear_pub = self.create_publisher(Entity, '/wrench_clear', 10)
        self.wrench_pub = self.create_publisher(EntityWrench, '/wrench_persistent', 10)
        # the four wheels are visual links; they are spun at the rate the mecanum inverse kinematics
        # demands so that what is on screen matches how the robot is actually moving
        self.declare_parameter('wheel_radius', 0.042)
        self.declare_parameter('lx', 0.090)          # half the wheelbase
        self.declare_parameter('ly', 0.124)          # half the track
        self.r = float(self.get_parameter('wheel_radius').value)
        self.lxy = float(self.get_parameter('lx').value) + float(self.get_parameter('ly').value)
        self.wheel_pubs = {t: self.create_publisher(Float64, f'/wheel_{t}_cmd', 10)
                           for t in ('fl', 'fr', 'rl', 'rr')}
        self.get_logger().info(f'mecanum_base_sim: driving {self.link} (m={self.m} kg, a_max={self.a_max} m/s^2)')

    def _cmd_cb(self, msg: Twist) -> None:
        self.cmd = msg
        self.cmd_time = self.get_clock().now()

    def _odom_cb(self, odom: Odometry) -> None:
        # command watchdog: no cmd_vel -> brake to zero
        if self.cmd_time is None or (self.get_clock().now() - self.cmd_time).nanoseconds * 1e-9 > self.cmd_timeout:
            cmd = Twist()
        else:
            cmd = self.cmd
        vx, vy = odom.twist.twist.linear.x, odom.twist.twist.linear.y     # base frame
        wz = odom.twist.twist.angular.z
        t = self.get_clock().now().nanoseconds * 1e-9
        dt = 0.0 if self.last_t is None else max(0.0, min(0.1, t - self.last_t))
        self.last_t = t
        errs = (cmd.linear.x - vx, cmd.linear.y - vy, cmd.angular.z - wz)
        gains = ((self.kp_lin, self.ki_lin, self.a_max), (self.kp_lin, self.ki_lin, self.a_max),
                 (self.kp_ang, self.ki_ang, self.alpha_max))
        out = []
        for i, (e, (kp, ki, lim)) in enumerate(zip(errs, gains)):
            # PI with clamping anti-windup: the integral only grows while the output is not saturated
            u = kp * e + ki * self.int_err[i]
            if abs(u) < lim:
                self.int_err[i] += e * dt
            out.append(max(-lim, min(lim, u)))
        if cmd.linear.x == 0.0 and cmd.linear.y == 0.0 and cmd.angular.z == 0.0 and \
                abs(vx) < 0.01 and abs(vy) < 0.01 and abs(wz) < 0.02:
            self.int_err = [0.0, 0.0, 0.0]      # at rest with zero command: no bias to hold
        ax, ay, alpha = out
        # base -> world rotation (yaw from the odometry quaternion)
        q = odom.pose.pose.orientation
        yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))
        fx = self.m * (ax * math.cos(yaw) - ay * math.sin(yaw))
        fy = self.m * (ax * math.sin(yaw) + ay * math.cos(yaw))
        tz = self.iz * alpha

        self._apply(fx, fy, tz)
        self._spin_wheels(vx, vy, wz)

    def _spin_wheels(self, vx: float, vy: float, wz: float) -> None:
        """Mecanum inverse kinematics: body velocity -> the angular velocity of each wheel.

            w_FL = (vx - vy - (lx+ly).wz) / r        w_FR = (vx + vy + (lx+ly).wz) / r
            w_RL = (vx + vy - (lx+ly).wz) / r        w_RR = (vx - vy + (lx+ly).wz) / r

        These are the rates the real drivetrain would run at, and the same formulas the micro-ROS
        board on the physical robot would use. Feeding them the MEASURED body velocity rather than the
        commanded one keeps the wheels consistent with the ground the robot is covering, so it never
        looks like it is skating; the cost is that a wedged robot's wheels stop instead of slipping.
        """
        k = self.lxy * wz
        for tag, w in (('fl', vx - vy - k), ('fr', vx + vy + k),
                       ('rl', vx + vy - k), ('rr', vx - vy + k)):
            self.wheel_pubs[tag].publish(Float64(data=w / self.r))

    # ------------------------------------------------------------------ wrench bookkeeping
    # ApplyLinkWrench keeps a LIST of persistent wrenches and sums them every step. A "clear" sent in
    # the same millisecond as a new "set" wipes the new one too (the clear queue is drained first), and
    # the two topics may cross in the bridge, so clear+set per cycle gives a random duty cycle.
    # Instead: publish only the DIFFERENCE between the wanted wrench and the running sum already in
    # the simulator; the list then holds a few entries that add up to exactly F. It is compacted
    # (clear, short pause, full re-send) every couple of seconds so it cannot grow without bound.
    def _apply(self, fx: float, fy: float, tz: float) -> None:
        now = time.monotonic()
        if now - self.last_compact > self.compact_period:
            ent = Entity(); ent.name = self.link; ent.type = Entity.LINK
            self.clear_pub.publish(ent)
            time.sleep(0.004)
            self.total = [0.0, 0.0, 0.0]
            self.last_compact = now
        dx, dy, dt = fx - self.total[0], fy - self.total[1], tz - self.total[2]
        if abs(dx) < 0.02 and abs(dy) < 0.02 and abs(dt) < 0.002:
            return
        msg = EntityWrench()
        msg.entity.name = self.link
        msg.entity.type = Entity.LINK
        msg.wrench.force.x = dx
        msg.wrench.force.y = dy
        msg.wrench.torque.z = dt
        self.wrench_pub.publish(msg)
        self.total = [fx, fy, tz]


def main(args=None) -> None:
    rclpy.init(args=args)
    node = MecanumBaseSim()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
