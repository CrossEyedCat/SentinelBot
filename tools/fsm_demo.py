"""The FSM demonstration: every state of the patrol FSM, and the transitions Section 5.1 of the report
provokes in Gazebo, driven in one run and logged for the video.

    python3 tools/fsm_demo.py <outdir> [spawn_x spawn_y]

Runs beside a Gazebo simulation of the Mk II in the TurtleBot3 world with the patrol stack up
(tools/record_fsm_demo.sh starts all of it). The scenario acts only through what an operator or the
world can do: /patrol_cmd, /goal_pose, a parameter change, objects created in and removed from the
world, a pose set on the robot, a push applied to it, and one node stopped. It never publishes a
state or a sensor value, so every transition in the log is the FSM's own.

Logged on the simulation clock: /patrol_state, /robot_alert, the ranges and the IMU the FSM reads,
the FSM's transition lines from /rosout, and a caption for each scenario step. Writes
<outdir>/demo.json, and publishes MISSION_COMPLETE on /mission_status at the end so the camera
recorder (tools/mission_grab.py) stops.
"""
import json
import math
import os
import re
import subprocess
import sys
import threading
import time

import rclpy
from rcl_interfaces.msg import Log
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from ros_gz_interfaces.msg import Entity, EntityWrench
from sensor_msgs.msg import Imu, Range
from std_msgs.msg import String

WORLD = 'default'
ROBOT = 'sentinel_mk2'
LINK = 'sentinel_mk2::base_link'
HERE = os.path.dirname(os.path.abspath(__file__))
BOX = os.path.join(HERE, 'fsm_demo_box.sdf')     # 0.14 m cube, static
PEN = os.path.join(HERE, 'fsm_demo_pen.sdf')     # four static walls, 0.84 m inside


def quat_yaw_roll(yaw, roll):
    """q = q_z(yaw) * q_x(roll)."""
    cy, sy = math.cos(yaw / 2), math.sin(yaw / 2)
    cr, sr = math.cos(roll / 2), math.sin(roll / 2)
    return cy * sr, sy * sr, sy * cr, cy * cr          # x, y, z, w


class Demo(Node):
    def __init__(self, out, spawn):
        super().__init__('fsm_demo', parameter_overrides=[Parameter('use_sim_time', Parameter.Type.BOOL, True)])
        self.out, self.spawn = out, spawn
        self.state, self.lock = None, threading.Lock()
        self.log = {'states': [], 'alerts': [], 'fsm_lines': [], 'captions': [], 'sensors': [],
                    'events': []}
        self.us = self.irl = self.irr = None
        self.acc, self.tilt = 0.0, 0.0
        be = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST)
        self.create_subscription(String, '/patrol_state', self._state_cb, 10)
        self.create_subscription(String, '/robot_alert', self._alert_cb, 10)
        self.create_subscription(Range, '/ultrasonic', lambda m: setattr(self, 'us', m.range), be)
        self.create_subscription(Range, '/ir_left', lambda m: setattr(self, 'irl', m.range), be)
        self.create_subscription(Range, '/ir_right', lambda m: setattr(self, 'irr', m.range), be)
        self.create_subscription(Imu, '/imu', self._imu_cb, be)
        self.odom_xy = (0.0, 0.0)
        self.create_subscription(Odometry, '/odom', lambda m: setattr(
            self, 'odom_xy', (m.pose.pose.position.x, m.pose.pose.position.y)), be)
        self.create_subscription(Log, '/rosout', self._rosout_cb, 100)
        self.cmd_pub = self.create_publisher(String, '/patrol_cmd', 10)
        self.goal_pub = self.create_publisher(PoseStamped, '/goal_pose', 10)
        self.status_pub = self.create_publisher(String, '/mission_status', 10)
        self.wrench_pub = self.create_publisher(EntityWrench, '/wrench_persistent', 10)
        self.clear_pub = self.create_publisher(Entity, '/wrench_clear', 10)
        self.create_timer(0.1, self._sample)

    # ------------------------------------------------------------------ logging
    def now(self):
        return self.get_clock().now().nanoseconds * 1e-9

    def _state_cb(self, msg):
        with self.lock:
            if msg.data != self.state:
                self.state = msg.data
                self.log['states'].append([self.now(), msg.data])

    def _alert_cb(self, msg):
        self.log['alerts'].append([self.now(), msg.data])

    def _imu_cb(self, m):
        a = m.linear_acceleration
        self.acc = math.hypot(a.x, a.y)
        q = m.orientation
        # tilt: angle between the body z axis and the world z axis
        zz = 1.0 - 2.0 * (q.x * q.x + q.y * q.y)
        self.tilt = math.degrees(math.acos(max(-1.0, min(1.0, zz))))

    def _rosout_cb(self, m):
        if m.name.endswith('patrol_fsm') and ('[FSM]' in m.msg or 'ALERT' in m.msg):
            self.log['fsm_lines'].append([self.now(), m.msg])

    def _sample(self):
        t = self.now()
        if t > 0:
            self.log['sensors'].append([round(t, 2), self.us, self.irl, self.irr, round(self.acc, 3),
                                        round(self.tilt, 2)])

    def caption(self, text, kind='step'):
        self.log['captions'].append([self.now(), text, kind])
        print('%8.2f  %s' % (self.now(), text), flush=True)

    def save(self):
        with open(os.path.join(self.out, 'demo.json'), 'w') as fh:
            json.dump(self.log, fh)

    # ------------------------------------------------------------------ acting on the world
    def cmd(self, text):
        self.cmd_pub.publish(String(data=text))

    def wait_state(self, targets, timeout):
        t_end = self.now() + timeout
        while self.now() < t_end:
            if self.state in targets:
                return True
            time.sleep(0.05)
        return False

    def hold(self, seconds):
        t_end = self.now() + seconds
        while self.now() < t_end:
            time.sleep(0.05)

    def gz_service(self, service, reqtype, req, timeout=5000):
        return subprocess.run(['gz', 'service', '-s', service, '--reqtype', reqtype, '--reptype',
                               'gz.msgs.Boolean', '--timeout', str(timeout), '--req', req],
                              capture_output=True, text=True).stdout

    def pause(self, on):
        self.gz_service('/world/%s/control' % WORLD, 'gz.msgs.WorldControl', 'pause: %s' % str(on).lower())

    def step(self, n):
        self.gz_service('/world/%s/control' % WORLD, 'gz.msgs.WorldControl', 'multi_step: %d' % n)

    def pose(self):
        """World pose of the robot: x, y, yaw."""
        txt = subprocess.run(['gz', 'topic', '-e', '-n', '1', '-t', '/world/%s/dynamic_pose/info' % WORLD],
                             capture_output=True, text=True, timeout=20).stdout
        for block in re.split(r'\n(?=pose \{)', txt):
            if re.search(r'name: "%s"' % ROBOT, block):
                num = lambda k, b: float((re.search(r'\b%s: (-?[\d.e+-]+)' % k, b) or [0, 0])[1])
                pos = re.search(r'position \{(.*?)\}', block, re.S).group(1)
                ori = re.search(r'orientation \{(.*?)\}', block, re.S)
                ori = ori.group(1) if ori else 'w: 1'
                qx, qy, qz, qw = (num(k, ori) for k in ('x', 'y', 'z', 'w'))
                yaw = math.atan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))
                return num('x', pos), num('y', pos), yaw
        raise RuntimeError('no pose for %s' % ROBOT)

    def spawn_model(self, name, sdf, x, y, z, yaw):
        subprocess.run(['ros2', 'run', 'ros_gz_sim', 'create', '-world', WORLD, '-file', sdf, '-name', name,
                        '-x', '%.3f' % x, '-y', '%.3f' % y, '-z', '%.3f' % z, '-Y', '%.4f' % yaw],
                       capture_output=True, text=True, timeout=30)
        self.log['events'].append([self.now(), 'spawn', name, x, y, yaw])

    def remove_model(self, name):
        self.gz_service('/world/%s/remove' % WORLD, 'gz.msgs.Entity', 'name: "%s", type: MODEL' % name)

    def set_pose(self, x, y, z, q):
        self.gz_service('/world/%s/set_pose' % WORLD, 'gz.msgs.Pose',
                        'name: "%s", position: {x: %.3f, y: %.3f, z: %.3f}, orientation: {x: %.4f, y: %.4f, '
                        'z: %.4f, w: %.4f}' % ((ROBOT, x, y, z) + tuple(q)))

    def param(self, name, value):
        subprocess.run(['ros2', 'param', 'set', '/patrol_fsm', name, str(value)], capture_output=True, timeout=30)

    def push(self, fx, fy, seconds):
        """A lateral strike: with the simulation paused, a persistent force is set on the chassis,
        exactly `seconds` of simulation are stepped, and the force is cleared again."""
        self.pause(True)
        w = EntityWrench()
        w.entity.name, w.entity.type = LINK, Entity.LINK
        w.wrench.force.x, w.wrench.force.y = fx, fy
        self.wrench_pub.publish(w)
        time.sleep(0.3)
        t0 = self.now()
        self.step(int(round(seconds * 1000)))
        t_end = time.time() + 20
        while self.now() < t0 + seconds - 0.002 and time.time() < t_end:
            time.sleep(0.01)
        e = Entity()
        e.name, e.type = LINK, Entity.LINK
        self.clear_pub.publish(e)
        time.sleep(0.3)
        self.step(2)
        time.sleep(0.3)
        self.pause(False)


def scenario(d):
    sx, sy = d.spawn
    ok = {}
    d.hold(2.0)
    d.caption('IDLE: sensors fresh, waiting for the operator')
    d.hold(4.0)

    # 1-3. start, then free patrol until an obstacle is avoided and cleared
    d.caption('Operator: start (ros2 topic pub /patrol_cmd start)')
    d.cmd('start')
    ok['IDLE->PATROLLING'] = d.wait_state({'PATROLLING'}, 10)
    d.caption('Patrol: drive straight until the front ultrasonic reads under 0.50 m')
    ok['PATROLLING->OBSTACLE_AVOIDANCE'] = d.wait_state({'OBSTACLE_AVOIDANCE'}, 90)
    d.caption('Obstacle inside the stop band: turn towards the freer infrared side')
    ok['OBSTACLE_AVOIDANCE->PATROLLING'] = d.wait_state({'PATROLLING'}, 30)
    d.caption('Front clear above 0.70 m for 0.5 s: patrol resumes')

    # 4. a box against a front corner while avoiding: BACKING_OFF
    if d.wait_state({'OBSTACLE_AVOIDANCE'}, 90):
        d.pause(True)
        x, y, yaw = d.pose()
        bx, by = 0.113 + 0.18 * math.cos(math.pi / 4), 0.083 + 0.18 * math.sin(math.pi / 4)
        c, s = math.cos(yaw), math.sin(yaw)
        d.spawn_model('corner_box', BOX, x + c * bx - s * by, y + s * bx + c * by, 0.07, yaw + math.pi / 4)
        d.pause(False)
        d.caption('A box placed against the front-left corner, 0.11 m from the infrared sensor')
        ok['OBSTACLE_AVOIDANCE->BACKING_OFF'] = d.wait_state({'BACKING_OFF'}, 8)
        d.caption('Infrared under 0.12 m: reverse until both corners read over 0.18 m')
        d.wait_state({'OBSTACLE_AVOIDANCE', 'PATROLLING'}, 15)
        d.wait_state({'PATROLLING'}, 30)
        d.hold(1.0)
        d.remove_model('corner_box')

    # 5. boxed in: the avoidance episode outlasts 10 s, ALERT (STUCK)
    d.hold(3.0)
    d.pause(True)
    x, y, yaw = d.pose()
    d.spawn_model('pen', PEN, x, y, 0.0, yaw)
    d.pause(False)
    d.caption('The robot boxed in on four sides')
    d.wait_state({'OBSTACLE_AVOIDANCE', 'BACKING_OFF'}, 10)
    d.caption('No free side: the avoidance episode runs past its 10 s limit')
    ok['OBSTACLE_AVOIDANCE->ALERT (stuck)'] = d.wait_state({'ALERT'}, 20)
    d.caption('ALERT: STUCK. Zero velocity; only the operator can release it')
    d.hold(3.0)
    d.remove_model('pen')
    d.caption('Operator: reset')
    d.cmd('reset')
    ok['ALERT->IDLE (stuck)'] = d.wait_state({'IDLE'}, 10)
    d.hold(2.0)
    d.cmd('start')
    d.wait_state({'PATROLLING'}, 10)
    d.caption('Operator: start')

    # 6. tipping
    d.hold(5.0)
    x, y, yaw = d.pose()
    d.caption('The robot tipped onto its side (pose set in Gazebo)')
    d.set_pose(x, y, 0.20, quat_yaw_roll(yaw, math.radians(80)))
    ok['PATROLLING->ALERT (tipping)'] = d.wait_state({'ALERT'}, 10)
    d.caption('ALERT: TIPPING, tilt over 30 deg from the IMU orientation')
    d.hold(4.0)
    d.set_pose(x, y, 0.03, quat_yaw_roll(yaw, 0.0))
    d.hold(2.0)
    d.caption('Robot set upright. Operator: reset')
    d.cmd('reset')
    ok['ALERT->IDLE (tipping)'] = d.wait_state({'IDLE'}, 10)
    d.hold(2.0)
    d.cmd('start')
    d.wait_state({'PATROLLING'}, 10)
    d.caption('Operator: start')

    # 7. collision, with the threshold lowered for the demonstration as in Section 5.1
    d.hold(3.0)
    d.param('imu_accel_spike', 2.0)
    d.caption('Collision threshold lowered to 2 m/s2 for the test (ros2 param set)')
    d.hold(3.0)
    d.wait_state({'PATROLLING'}, 30)            # strike a robot that is driving, not one mid-turn
    if d.state in ('PATROLLING', 'OBSTACLE_AVOIDANCE', 'BACKING_OFF', 'NAVIGATING'):
        x, y, yaw = d.pose()
        f = 20.0                                   # N on 4 kg: 5 m/s2 sideways for 50 ms
        d.caption('A strike from the left, outside the ultrasonic cone')
        d.push(f * math.sin(yaw), -f * math.cos(yaw), 0.05)     # towards the robot's right
        ok['PATROLLING->ALERT (collision)'] = d.wait_state({'ALERT'}, 5)
        d.caption('ALERT: COLLISION, horizontal acceleration over the threshold twice')
        d.hold(4.0)
    d.param('imu_accel_spike', 8.0)
    d.caption('Threshold restored to 8 m/s2. Operator: reset')
    d.cmd('reset')
    ok['ALERT->IDLE (collision)'] = d.wait_state({'IDLE'}, 10)
    d.hold(2.0)
    d.cmd('start')
    d.wait_state({'PATROLLING'}, 10)
    d.caption('Operator: start')

    # 8. sensor lost: the range adapter stopped while driving
    d.hold(4.0)
    d.caption('Range adapter node stopped while driving')
    subprocess.run(['pkill', '-f', 'lib/sentinel_patrol/scan_to_range'])
    ok['PATROLLING->ALERT (sensor lost)'] = d.wait_state({'ALERT'}, 6)
    d.caption('ALERT: SENSOR_LOST, no range reading for 1 s')
    d.hold(3.0)
    share = subprocess.run(['ros2', 'pkg', 'prefix', '--share', 'sentinel_patrol'], capture_output=True,
                           text=True).stdout.strip()
    subprocess.Popen(['ros2', 'run', 'sentinel_patrol', 'scan_to_range', '--ros-args', '--params-file',
                      os.path.join(share, 'config', 'patrol_params.yaml'), '-p', 'use_sim_time:=true',
                      '-p', 'mode:=per_sensor'], stdout=open(os.path.join(d.out, 'adapter2.log'), 'w'),
                     stderr=subprocess.STDOUT)
    d.caption('Adapter restarted. Operator: reset')
    d.hold(3.0)
    d.cmd('reset')
    ok['ALERT->IDLE (sensor lost)'] = d.wait_state({'IDLE'}, 10)
    d.hold(3.0)

    # 9. NAVIGATING: a goal on /goal_pose, planned with A* over the costmap the robot built
    # an open cell between four pillars (0.47 m clear of each) on the line the first patrol drove, so
    # the map already holds it; the far one of two from the robot. Tried and dropped: the spawn point
    # (too close to the west wall to finish an approach) and a cell the robot had not yet mapped
    # (the planner, rightly, called it unreachable)
    x, y, _ = d.pose()
    gx, gy = max([(-0.55, -0.55), (0.55, -0.55)], key=lambda c: math.hypot(x - c[0], y - c[1]))
    # world -> odom measured now, as tools' column_tour does, rather than assumed from the spawn:
    # /odom may start at the world origin or at the spawn pose
    ox, oy = d.odom_xy
    off = (x - ox, y - oy)
    d.log['events'].append([d.now(), 'goal', gx, gy, off[0], off[1]])
    g = PoseStamped()
    g.header.frame_id = 'odom'
    g.header.stamp = d.get_clock().now().to_msg()
    g.pose.position.x, g.pose.position.y = gx - off[0], gy - off[1]
    g.pose.orientation.w = 1.0
    d.caption('Goal sent on /goal_pose: A* over the costmap the robot built')
    d.goal_pub.publish(g)
    ok['IDLE->NAVIGATING'] = d.wait_state({'NAVIGATING'}, 10)
    d.wait_state({'IDLE'}, 90)
    d.hold(0.5)
    # the caption says what the FSM logged, not what was hoped for
    end = [ln for _, ln in d.log['fsm_lines'] if 'NAVIGATING -> IDLE' in ln]
    reached = bool(end) and 'GOAL_REACHED' in end[-1]
    ok['NAVIGATING->IDLE (arrival)'] = reached
    if reached:
        d.caption('Goal reached within 0.10 m: back to IDLE')
    elif end:
        d.caption('The planner found no route to the goal: back to IDLE')
    else:
        d.caption('The goal was not reached in 90 s')
    d.hold(4.0)
    d.caption('End of the demonstration', kind='end')
    d.log['ok'] = ok
    return ok


def main():
    out = sys.argv[1]
    spawn = (float(sys.argv[2]), float(sys.argv[3])) if len(sys.argv) > 3 else (-2.0, -0.5)
    os.makedirs(out, exist_ok=True)
    rclpy.init()
    d = Demo(out, spawn)
    th = threading.Thread(target=rclpy.spin, args=(d,), daemon=True)
    th.start()
    while d.now() <= 0:
        time.sleep(0.1)
    try:
        ok = scenario(d)
        print(json.dumps(ok, indent=1), flush=True)
    except Exception as e:                                    # keep what was logged
        d.caption('scenario stopped: %r' % e, kind='error')
        print('ERROR', repr(e), flush=True)
    finally:
        d.save()
        for _ in range(5):
            d.status_pub.publish(String(data='MISSION_COMPLETE'))
            time.sleep(0.2)
        d.save()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
