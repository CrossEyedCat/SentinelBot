"""Adapter: emulate ultrasonic and IR range sensors from the TurtleBot3 LiDAR.

The stock TurtleBot3 in Gazebo publishes /scan (LaserScan) but no sensor_msgs/Range topics.
This node slices /scan into angular sectors and republishes each as a Range message with the
field of view and min/max range of the real sensor it stands in for:

    /ultrasonic        HC-SR04, front,  15 deg cone, 0.02 - 4.00 m
    /ultrasonic_rear   HC-SR04, rear
    /ultrasonic_left   HC-SR04, left
    /ultrasonic_right  HC-SR04, right
    /ir_left           Sharp GP2Y0A21, +30 deg, 5 deg beam, 0.10 - 0.80 m
    /ir_right          Sharp GP2Y0A21, -30 deg

Readings beyond the sensor's max range are clamped to max_range (reads as "clear"); readings below
min_range, which Gazebo returns as -inf, are clamped to min_range, exactly as the hardware would
report, so that an obstacle inside the blind zone never reads as clear.
"""
import math
from typing import List

import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import LaserScan, Range


class VirtualRange:
    def __init__(self, topic: str, centre_deg: float, fov_deg: float,
                 min_range: float, max_range: float, radiation: int):
        self.topic = topic
        self.centre = math.radians(centre_deg)
        self.fov = math.radians(fov_deg)
        self.min_range = min_range
        self.max_range = max_range
        self.radiation = radiation
        self.pub = None


class ScanToRangeNode(Node):

    def __init__(self) -> None:
        super().__init__('scan_to_range')
        self.declare_parameter('scan_topic', '/scan')
        self.declare_parameter('us_fov_deg', 15.0)
        self.declare_parameter('ir_fov_deg', 5.0)
        self.declare_parameter('ir_angle_deg', 30.0)
        us_fov = self.get_parameter('us_fov_deg').value
        ir_fov = self.get_parameter('ir_fov_deg').value
        ir_ang = self.get_parameter('ir_angle_deg').value

        US, IR = Range.ULTRASOUND, Range.INFRARED
        self.sensors: List[VirtualRange] = [
            VirtualRange('/ultrasonic', 0.0, us_fov, 0.02, 4.00, US),
            VirtualRange('/ultrasonic_rear', 180.0, us_fov, 0.02, 4.00, US),
            VirtualRange('/ultrasonic_left', 90.0, us_fov, 0.02, 4.00, US),
            VirtualRange('/ultrasonic_right', -90.0, us_fov, 0.02, 4.00, US),
            VirtualRange('/ir_left', ir_ang, ir_fov, 0.10, 0.80, IR),
            VirtualRange('/ir_right', -ir_ang, ir_fov, 0.10, 0.80, IR),
        ]
        qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,
                         history=HistoryPolicy.KEEP_LAST, depth=5)
        for s in self.sensors:
            s.pub = self.create_publisher(Range, s.topic, qos)

        # mode "sectors": one 360 deg /scan (stock TurtleBot3) is sliced into virtual sensors.
        # mode "per_sensor": every physical sensor of the Mk II model publishes its own narrow
        #                    LaserScan (us_front, ir_fl, ...) and is converted 1:1 into a Range.
        self.declare_parameter('mode', 'sectors')
        self.mode = self.get_parameter('mode').value
        if self.mode == 'per_sensor':
            per_sensor_topics = {
                '/ultrasonic': 'us_front', '/ultrasonic_rear': 'us_rear',
                '/ultrasonic_left': 'us_left', '/ultrasonic_right': 'us_right',
                '/ir_left': 'ir_fl', '/ir_right': 'ir_fr',
            }
            for s in self.sensors:
                src = per_sensor_topics[s.topic]
                self.create_subscription(LaserScan, src, lambda msg, s=s: self._single_cb(msg, s), qos)
            self.get_logger().info('scan_to_range (per_sensor): ' +
                                   ', '.join(f'{v} -> {k}' for k, v in per_sensor_topics.items()))
        else:
            self.create_subscription(LaserScan, self.get_parameter('scan_topic').value, self._scan_cb, qos)
            self.get_logger().info('scan_to_range (sectors of /scan): publishing ' +
                                   ', '.join(s.topic for s in self.sensors))

    @staticmethod
    def _resolve(rays, scan: LaserScan, s: VirtualRange) -> float:
        """Turn a set of raw rays into the distance the real sensor would report.

        Gazebo follows REP 117: a ray shorter than the sensor's minimum comes back as -inf, one that
        hits nothing as +inf. Both used to be discarded and replaced by max_range, so an obstacle
        INSIDE the blind zone read as "clear" - which inverts the BACKING_OFF trigger: pressing a
        corner IR to within 10 cm of a pillar made it jump from 0.11 m to 0.80 m and the FSM kept
        rotating into the contact. A too-close return is reported as min_range instead, which is what
        an HC-SR04 or a GP2Y0A21 does, and leaves ir_critical (0.12 m) above the published floor.
        """
        valid = [r for r in rays if math.isfinite(r) and scan.range_min < r < scan.range_max]
        if valid:
            return min(valid)
        if any((not math.isfinite(r) and r < 0) or (math.isfinite(r) and r <= scan.range_min)
               for r in rays):
            return s.min_range
        return s.max_range

    def _single_cb(self, scan: LaserScan, s: VirtualRange) -> None:
        """One physical sensor -> one Range: the closest valid ray of its own narrow scan."""
        d = self._resolve(scan.ranges, scan, s)
        msg = Range()
        msg.header = scan.header
        msg.radiation_type = s.radiation
        msg.field_of_view = s.fov
        msg.min_range = s.min_range
        msg.max_range = s.max_range
        msg.range = float(min(max(d, s.min_range), s.max_range))
        s.pub.publish(msg)

    def _scan_cb(self, scan: LaserScan) -> None:
        n = len(scan.ranges)
        if n == 0 or scan.angle_increment == 0.0:
            return
        for s in self.sensors:
            d = self._resolve(self._sector_rays(scan, s.centre, s.fov), scan, s)
            msg = Range()
            msg.header = scan.header
            msg.header.frame_id = 'base_scan'
            msg.radiation_type = s.radiation
            msg.field_of_view = s.fov
            msg.min_range = s.min_range
            msg.max_range = s.max_range
            msg.range = float(min(max(d, s.min_range), s.max_range))
            s.pub.publish(msg)

    @staticmethod
    def _sector_rays(scan: LaserScan, centre: float, fov: float):
        """Raw rays within [centre - fov/2, centre + fov/2], angles wrapped to the scan."""
        n = len(scan.ranges)
        half = fov / 2.0
        steps = max(1, int(round(fov / scan.angle_increment)))
        rays = []
        for k in range(steps + 1):
            ang = centre - half + k * scan.angle_increment
            # wrap into [angle_min, angle_min + 2*pi)
            rel = (ang - scan.angle_min) % (2.0 * math.pi)
            idx = int(round(rel / scan.angle_increment)) % n
            rays.append(scan.ranges[idx])
        return rays


def main(args=None) -> None:
    rclpy.init(args=args)
    node = ScanToRangeNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
