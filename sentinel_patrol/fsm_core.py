"""Pure-Python finite state machine for the SentinelBot patrol robot.

This module has NO ROS dependency on purpose. The ROS2 node (patrol_fsm_node.py)
feeds it sensor frames plus a clock and gets back velocity commands, transitions
and alert messages. Every transition can therefore be unit-tested with pytest on
any machine before the code is ever run in Gazebo.

States (each one is a method of PatrolFSM):
    IDLE                stopped, waiting for valid sensors + operator "start"
    PATROLLING          drives forward, ultrasonic watches the front
    NAVIGATING          drives to a point the operator clicked on the map
    OBSTACLE_AVOIDANCE  stops and rotates (or strafes) towards free space using the IR pair
    BACKING_OFF         short reverse when an obstacle is inside the turning radius
    ALERT               collision / tipping / stuck: full stop, alert published, operator reset only
"""
import math
from dataclasses import dataclass, field
from typing import Optional, Tuple

IDLE = 'IDLE'
PATROLLING = 'PATROLLING'
NAVIGATING = 'NAVIGATING'
OBSTACLE_AVOIDANCE = 'OBSTACLE_AVOIDANCE'
BACKING_OFF = 'BACKING_OFF'
ALERT = 'ALERT'
STATES = (IDLE, PATROLLING, NAVIGATING, OBSTACLE_AVOIDANCE, BACKING_OFF, ALERT)


@dataclass
class FsmParams:
    """All thresholds of the FSM. Mirrors config/patrol_params.yaml."""
    # distances [m]
    d_stop: float = 0.50          # ultrasonic: PATROLLING -> OBSTACLE_AVOIDANCE
    d_clear: float = 0.70         # ultrasonic: OBSTACLE_AVOIDANCE -> PATROLLING (hysteresis 0.20 m)
    ir_steer: float = 0.25        # both IR must exceed this before returning to PATROLLING
    ir_critical: float = 0.12     # any IR below this -> BACKING_OFF (inside the turning radius)
    ir_back_exit: float = 0.18    # BACKING_OFF ends only once both IR exceed this (hysteresis vs ir_critical)
    d_side_clear: float = 0.60    # holonomic only: side ultrasonic needed to strafe instead of rotating
    d_rear_min: float = 0.25      # holonomic only: rear ultrasonic aborts BACKING_OFF
    ir_wall_delta: float = 0.08   # holonomic only: |ir_left - ir_right| below this = flat wall -> rotate, not strafe
    t_strafe_max: float = 2.5     # holonomic only: strafe at most this long (~0.37 m), then rotate instead
    # IMU
    imu_accel_spike: float = 8.0  # |a_xy| [m/s^2] -> ALERT (collision)
    imu_tilt_max: float = 30.0    # |roll| or |pitch| [deg] -> ALERT (tipping)
    n_imu_confirm: int = 2        # consecutive IMU samples above threshold
    # velocities
    v_patrol: float = 0.15        # [m/s]
    w_turn: float = 0.60          # [rad/s]
    v_back: float = -0.08         # [m/s]
    v_strafe: float = 0.15        # [m/s] holonomic only
    v_goal: float = 0.18          # [m/s] approach speed in NAVIGATING
    # goal seeking (NAVIGATING)
    goal_tolerance: float = 0.10  # [m] close enough to count as arrived
    d_final: float = 0.60         # [m] distance to the goal below which the final approach starts
    d_stop_final: float = 0.22    # [m] reduced front stop distance during the final approach
    v_final: float = 0.10         # [m/s] speed cap during the final approach
    goal_yaw_align: float = 0.35  # [rad] bearing error above which the robot turns before driving
    k_goal_yaw: float = 1.2       # [1/s] heading gain while driving towards the goal
    k_goal_lat: float = 0.8       # [1/s] holonomic only: sideways correction while driving
    t_goal_max: float = 120.0     # [s] give up on a goal that cannot be reached
    t_escape: float = 4.0         # [s] drive on past an obstacle before re-aiming at the goal
    n_avoid_max: int = 8          # give up after this many avoidance episodes for one goal
    lookahead: float = 0.35       # [m] how far along the planned path the robot steers at
    carrot_window: float = 1.20   # [m] of route searched for the nearest point; a lap comes
    #                               back on itself, so an unbounded search finds the far side
    # progress watchdog: commanded to move but not moving means wedged against something
    stuck_dist: float = 0.05      # [m] movement below this does not count as progress
    t_stuck: float = 4.0          # [s] of no progress while driving before acting on it
    v_min_progress: float = 0.02  # [m/s] commanded speed below which the robot is not being asked
    #                               to translate at all, so distance covered says nothing
    # timing [s]
    t_back: float = 0.60          # minimum reverse duration
    t_back_max: float = 1.50      # reverse never lasts longer than this
    t_clear: float = 0.50         # path must stay clear this long before PATROLLING
    t_avoid_max: float = 10.0     # OBSTACLE_AVOIDANCE longer than this = stuck -> ALERT
    sensor_timeout: float = 1.0   # data older than this is not "valid" in IDLE
    # debounce
    n_confirm: int = 3            # consecutive ultrasonic samples below d_stop
    # behaviour switches
    holonomic: bool = False       # True for the mecanum platform (Mk II)
    auto_start: bool = False      # True: leave IDLE as soon as sensors are valid (no operator command)


@dataclass
class SensorFrame:
    """Latest reading of every sensor plus the time it was received."""
    us_front: Optional[float] = None
    us_rear: Optional[float] = None
    us_left: Optional[float] = None
    us_right: Optional[float] = None
    ir_left: Optional[float] = None
    ir_right: Optional[float] = None
    # pose in the map/odom frame, needed only to drive to a clicked goal
    pose_x: Optional[float] = None
    pose_y: Optional[float] = None
    yaw: Optional[float] = None
    t_us_front: float = float('-inf')
    t_ir: float = float('-inf')
    t_imu: float = float('-inf')
    t_pose: float = float('-inf')

    def has_pose(self) -> bool:
        return self.pose_x is not None and self.pose_y is not None and self.yaw is not None

    def ir_min(self) -> Optional[float]:
        vals = [v for v in (self.ir_left, self.ir_right) if v is not None]
        return min(vals) if vals else None


@dataclass
class Command:
    vx: float = 0.0
    vy: float = 0.0
    wz: float = 0.0

    def is_zero(self) -> bool:
        return self.vx == 0.0 and self.vy == 0.0 and self.wz == 0.0


@dataclass
class Transition:
    src: str
    dst: str
    sensor: str      # what triggered it, e.g. "ultrasonic.front"
    value: float     # the sensor value at the moment of the transition
    reason: str = ''  # free text, e.g. "COLLISION"

    def __str__(self) -> str:
        return f'{self.src} -> {self.dst} | trigger {self.sensor} = {self.value:.2f} {self.reason}'.rstrip()


@dataclass
class AlertEvent:
    reason: str
    sensor: str
    value: float
    from_state: str

    def as_text(self) -> str:
        return (f'ALERT reason={self.reason} sensor={self.sensor} '
                f'value={self.value:.2f} from={self.from_state}')


class PatrolFSM:
    """The state machine. Call on_imu() from the IMU callback and step() at a fixed rate."""

    def __init__(self, params: Optional[FsmParams] = None, t0: float = 0.0, pilot=None):
        self.p = params or FsmParams()
        # Optional learned steering for NAVIGATING: pilot(frame, path) -> (vx, vy, wz) or None.
        # It is deliberately the last thing consulted and the first thing overruled - see
        # s_navigating - so installing one cannot widen what the robot is allowed to do.
        self.pilot = pilot
        self.state = IDLE
        self.entered_at = t0
        self.last_alert: Optional[AlertEvent] = None
        self._start_requested = False
        self._stop_requested = False
        self._reset_requested = False
        self._obstacle_count = 0
        self._imu_spike_count = 0
        self._clear_since: Optional[float] = None
        self._turn_dir = 1          # +1 = left (CCW), -1 = right
        self._strafe_dir = 0        # +1 = left, -1 = right, 0 = rotate instead
        self._strafe_start: Optional[float] = None    # when the current strafe began (t_strafe_max)
        self._episode_start: Optional[float] = None   # when PATROLLING was last left (stuck timer base)
        self._goal: Optional[Tuple[float, float]] = None   # clicked destination, map/odom frame
        self._goal_start: Optional[float] = None      # when the current goal was accepted
        self._escape_until: Optional[float] = None    # drive straight on until then (see s_navigating)
        self._avoid_count = 0                         # avoidance episodes for the current goal
        self._path: list = []                         # route from the planner, map/odom frame
        self._path_i = 0                              # which point of it the robot is steering at
        self._progress_ref: Optional[Tuple[float, float, float]] = None   # x, y, t of the last progress
        self._handlers = {
            IDLE: self.s_idle,
            PATROLLING: self.s_patrolling,
            NAVIGATING: self.s_navigating,
            OBSTACLE_AVOIDANCE: self.s_obstacle_avoidance,
            BACKING_OFF: self.s_backing_off,
            ALERT: self.s_alert,
        }

    # ------------------------------------------------------------------ operator
    def on_command(self, cmd: str) -> None:
        """Operator commands from /patrol_cmd: start | stop | reset."""
        cmd = cmd.strip().lower()
        if cmd == 'start':
            self._start_requested = True
        elif cmd == 'stop':
            self._stop_requested = True
        elif cmd == 'reset':
            self._reset_requested = True

    def on_goal(self, x: float, y: float, t: float) -> None:
        """A destination clicked on the map (RViz '2D Goal Pose' -> /goal_pose).

        Accepted in any state except ALERT: a robot that has been hit or tipped over must be cleared
        by a human before it drives anywhere, whatever the map says.
        """
        if self.state == ALERT:
            return
        self._goal = (x, y)
        self._goal_start = t
        self._escape_until = None
        self._avoid_count = 0
        self._path = []               # the old route led somewhere else
        self._path_i = 0

    def on_path(self, points) -> None:
        """A route from the planner: a list of (x, y) waypoints ending at the goal.

        Following a planned route is what lets the robot go *around* an obstacle field instead of
        pushing at it. With no path set the FSM still works, heading straight at the goal and reacting
        to what it meets, so a planner that is absent or slow degrades the behaviour instead of
        breaking it.
        """
        self._path = list(points)
        self._path_i = 0

    def path(self) -> list:
        return self._path

    def abandon_goal(self, t: float) -> Optional[Transition]:
        """Drop the current goal, e.g. because the planner reports that no route to it exists.
        Returns the transition if the robot was driving to it, so the node can log it."""
        was_navigating = self.state == NAVIGATING
        self._goal = None
        self._goal_start = None
        self._path = []
        self._path_i = 0
        if was_navigating:
            return self._go(IDLE, t, 'planner', 0.0, 'GOAL_UNREACHABLE')
        return None

    def goal(self) -> Optional[Tuple[float, float]]:
        return self._goal

    def _carrot(self, f: SensorFrame) -> Optional[Tuple[float, float]]:
        """The point on the route the robot should steer at now.

        Pure pursuit: find the waypoint the robot is nearest to, then take the first one at least a
        lookahead beyond it. Picking the nearest first matters - simply skipping waypoints that are
        close by leaves the robot steering backwards at one it has already driven past, which is
        exactly what happens the moment an avoidance manoeuvre pushes it off the route. The search
        never runs backwards along the path, so the robot cannot be sent back to the start.

        The nearest point is looked for only a short way forward, not over the whole remaining
        route. A route that comes back on itself - a lap around a column is the obvious case - has
        a sample near the robot both just ahead of it and most of a loop later, and the unbounded
        search takes whichever happens to be a centimetre closer. Steering at the far one sends the
        robot across the middle of its own lap; in one run it drove into the column it was circling
        and raised a collision ALERT.
        """
        if not self._path:
            return None
        near, near_d, travelled = self._path_i, None, 0.0
        for i in range(self._path_i, len(self._path)):
            if i > self._path_i:
                travelled += math.hypot(self._path[i][0] - self._path[i - 1][0],
                                        self._path[i][1] - self._path[i - 1][1])
                if travelled > self.p.carrot_window:
                    break
            d = math.hypot(self._path[i][0] - f.pose_x, self._path[i][1] - f.pose_y)
            if near_d is None or d < near_d:
                near, near_d = i, d
        j = near
        while j < len(self._path) - 1 and \
                math.hypot(self._path[j][0] - f.pose_x, self._path[j][1] - f.pose_y) < self.p.lookahead:
            j += 1
        self._path_i = j
        return self._path[j]

    def goal_distance(self, f: SensorFrame) -> Optional[float]:
        if self._goal is None or not f.has_pose():
            return None
        return math.hypot(self._goal[0] - f.pose_x, self._goal[1] - f.pose_y)

    # ------------------------------------------------------------------ IMU (priority path)
    def on_imu(self, a_xy: float, tilt_deg: float, t: float) -> Optional[Transition]:
        """Runs at IMU rate (100 Hz). ALERT pre-empts every state except IDLE and ALERT itself."""
        if self.state in (IDLE, ALERT):
            return None
        # TRANSITION any -> ALERT (collision): horizontal acceleration spike, debounced
        if a_xy > self.p.imu_accel_spike:
            self._imu_spike_count += 1
        else:
            self._imu_spike_count = 0
        if self._imu_spike_count >= self.p.n_imu_confirm:
            return self._go(ALERT, t, 'imu.a_xy', a_xy, 'COLLISION')
        # TRANSITION any -> ALERT (tipping): orientation beyond the safe tilt
        if abs(tilt_deg) > self.p.imu_tilt_max:
            return self._go(ALERT, t, 'imu.tilt_deg', tilt_deg, 'TIPPING')
        return None

    # ------------------------------------------------------------------ main tick
    def step(self, f: SensorFrame, t: float) -> Tuple[Command, Optional[Transition]]:
        """One FSM cycle (10 Hz). Returns the velocity command and the transition, if any."""
        # Two conditions must hold in EVERY moving state, so they are checked here rather than in
        # the individual handlers, where it is easy to forget one:
        if self.state not in (IDLE, ALERT):
            # TRANSITION any moving state -> IDLE: operator stop. A stop issued during a manoeuvre
            # used to be honoured only once the robot got back to PATROLLING, i.e. up to t_avoid_max
            # later; an operator pressing stop expects the robot to halt now.
            if self._stop_requested:
                self._stop_requested = False
                return Command(), self._go(IDLE, t, 'operator', 0.0, 'stop')
            # TRANSITION any moving state -> ALERT: the front ultrasonic went silent. Freshness used
            # to be checked only in IDLE, so a dead sensor topic left the robot driving blind on the
            # last reading it ever received.
            age = t - f.t_us_front
            if age > self.p.sensor_timeout:
                return Command(), self._go(ALERT, t, 'sensor.age_s', age, 'SENSOR_LOST')
        cmd, tr = self._handlers[self.state](f, t)
        if tr is None:
            # Nothing else fired this tick, so check that the robot is actually getting anywhere.
            # A state's own logic always wins: an arrival or a timeout is a better explanation than
            # "not moving", and this must not pre-empt them.
            tr = self._check_progress(f, t, cmd)
            if tr is not None:
                return Command(), tr
        return cmd, tr

    def _check_progress(self, f: SensorFrame, t: float, cmd: Command) -> Optional[Transition]:
        """Commanded to drive but not moving: the robot is wedged against something none of its
        forward-looking sensors can see. Seen in simulation after a planned route took the robot
        between two pillars: it sat perfectly still for 50 s, in NAVIGATING, until the goal timer
        expired, because nothing was watching whether the commands had any effect.
        """
        # Only a command to TRANSLATE can be judged by how far the robot translated. Turning on the
        # spot moves the robot nowhere on purpose, and counting that as no progress made the robot
        # abandon legitimate turns: every in-place turn longer than t_stuck was read as being wedged.
        if self.state not in (PATROLLING, NAVIGATING) or not f.has_pose() \
                or math.hypot(cmd.vx, cmd.vy) < self.p.v_min_progress:
            self._progress_ref = None
            return None
        if self._progress_ref is None:
            self._progress_ref = (f.pose_x, f.pose_y, t)
            return None
        rx, ry, rt = self._progress_ref
        if math.hypot(f.pose_x - rx, f.pose_y - ry) >= self.p.stuck_dist:
            self._progress_ref = (f.pose_x, f.pose_y, t)      # moving: reset the reference
            return None
        if t - rt < self.p.t_stuck:
            return None
        # TRANSITION PATROLLING or NAVIGATING -> OBSTACLE_AVOIDANCE: no progress while driving. The
        # avoidance behaviour turns the robot, which is what frees a body wedged on a corner, and its
        # own stuck timer escalates to ALERT if that does not work either.
        self._choose_direction(f, t)
        self._episode_start = t
        if self.state == NAVIGATING:
            self._avoid_count += 1
        return self._go(OBSTACLE_AVOIDANCE, t, 'progress.m', 0.0, 'NO_PROGRESS')

    # ------------------------------------------------------------------ states
    def s_idle(self, f: SensorFrame, t: float) -> Tuple[Command, Optional[Transition]]:
        """IDLE: stand still until both sensors are fresh and the operator says start."""
        self._reset_requested = False
        self._stop_requested = False
        fresh = (t - f.t_us_front) < self.p.sensor_timeout and (t - f.t_imu) < self.p.sensor_timeout
        # TRANSITION IDLE -> NAVIGATING: the operator clicked a destination on the map. No 'start' is
        # needed - clicking a point on the map IS the command to go there.
        if fresh and self._goal is not None and f.has_pose():
            return Command(), self._go(NAVIGATING, t, 'goal.distance', self.goal_distance(f) or 0.0, 'goal')
        # TRANSITION IDLE -> PATROLLING: valid data on ultrasonic AND imu, plus start (or auto_start)
        if fresh and (self._start_requested or self.p.auto_start):
            self._start_requested = False
            return Command(), self._go(PATROLLING, t, 'operator', 1.0, 'start')
        return Command(), None

    def s_patrolling(self, f: SensorFrame, t: float) -> Tuple[Command, Optional[Transition]]:
        """PATROLLING: drive straight; the front ultrasonic decides when to stop.

        Operator stop and sensor loss are handled for every moving state in step().
        """
        # TRANSITION PATROLLING -> NAVIGATING: a destination was clicked during the patrol
        if self._goal is not None and f.has_pose():
            return Command(), self._go(NAVIGATING, t, 'goal.distance', self.goal_distance(f) or 0.0, 'goal')
        # debounce: n_confirm consecutive readings below d_stop
        if f.us_front is not None and f.us_front < self.p.d_stop:
            self._obstacle_count += 1
        else:
            self._obstacle_count = 0
        # TRANSITION PATROLLING -> OBSTACLE_AVOIDANCE: obstacle closer than d_stop, confirmed
        if self._obstacle_count >= self.p.n_confirm:
            self._choose_direction(f, t)
            self._episode_start = t          # the stuck timer counts the whole avoidance episode
            return Command(), self._go(OBSTACLE_AVOIDANCE, t, 'ultrasonic.front', f.us_front)
        return Command(vx=self.p.v_patrol), None

    def s_navigating(self, f: SensorFrame, t: float) -> Tuple[Command, Optional[Transition]]:
        """NAVIGATING: drive to the point the operator clicked on the map.

        The robot turns to face the goal before it drives, so the front ultrasonic always looks where
        the robot is going. That lets this state reuse the same obstacle rule as PATROLLING instead of
        needing one of its own, and it is the reason the mecanum platform does not simply strafe to the
        goal sideways: it would be travelling in a direction no long-range sensor covers.
        """
        if self._goal is None or not f.has_pose():
            # the goal was withdrawn, or the pose source died under us: stop rather than guess
            return Command(), self._go(IDLE, t, 'goal', 0.0, 'no_goal')
        dx, dy = self._goal[0] - f.pose_x, self._goal[1] - f.pose_y
        dist = math.hypot(dx, dy)
        # TRANSITION NAVIGATING -> IDLE: arrived
        if dist <= self.p.goal_tolerance:
            return Command(), self._go(IDLE, t, 'goal.distance', dist, 'GOAL_REACHED')
        # TRANSITION NAVIGATING -> IDLE: the goal is not reachable (blocked, or outside the arena)
        elapsed = t - (self._goal_start if self._goal_start is not None else t)
        if elapsed > self.p.t_goal_max:
            return Command(), self._go(IDLE, t, 'timer.goal', elapsed, 'GOAL_UNREACHABLE')
        # Final approach. A goal is often placed near something - a doorway, a corner, the wall of the
        # arena - and the patrol stop distance (0.50 m) is much larger than the arrival tolerance, so
        # the robot would halt half a metre short of any such point and never arrive. Inside d_final it
        # therefore slows down and accepts a much smaller front clearance, because it is deliberately
        # approaching that spot rather than being blocked by it. The IR ring and BACKING_OFF still
        # protect the corners, so the robot cannot drive into anything.
        final = dist <= self.p.d_final
        if final:
            self._escape_until = None      # never drive past a goal that is already this close
        stop_distance = self.p.d_stop_final if final else self.p.d_stop

        # TRANSITION NAVIGATING -> BACKING_OFF: a corner sensor is against something. The front
        # ultrasonic cannot see a contact on the flank, and a mecanum robot travelling with a lateral
        # component can make one, so the IR ring is checked here as well as during avoidance.
        ir_min = f.ir_min()
        if ir_min is not None and ir_min < self.p.ir_critical:
            self._episode_start = t
            return Command(), self._go(BACKING_OFF, t, 'ir.min', ir_min)

        # debounce, exactly as in PATROLLING
        if f.us_front is not None and f.us_front < stop_distance:
            self._obstacle_count += 1
        else:
            self._obstacle_count = 0
        # TRANSITION NAVIGATING -> OBSTACLE_AVOIDANCE: something blocks the way. The goal is kept, so
        # the avoidance behaviour hands control back here once the path is clear.
        if self._obstacle_count >= self.p.n_confirm:
            self._avoid_count += 1
            # TRANSITION NAVIGATING -> IDLE: the same obstacle keeps coming back. Without a global
            # planner the robot cannot reason its way around a large one, so it says so rather than
            # shuffling in front of it until t_goal_max.
            if self._avoid_count > self.p.n_avoid_max:
                return Command(), self._go(IDLE, t, 'avoid.episodes', float(self._avoid_count),
                                           'GOAL_UNREACHABLE')
            self._choose_direction(f, t)
            self._episode_start = t
            return Command(), self._go(OBSTACLE_AVOIDANCE, t, 'ultrasonic.front', f.us_front)

        # Just came back from an avoidance: drive on along the heading the avoidance freed instead of
        # immediately turning back towards the goal. Re-aiming at once put the obstacle straight back
        # in front of the robot, which produced a NAVIGATING <-> OBSTACLE_AVOIDANCE livelock in
        # simulation (one hop every 1.8 s, front distance alternating 0.34 m / 2.60 m, indefinitely).
        if self._escape_until is not None:
            if t < self._escape_until:
                return Command(vx=self.p.v_goal), None
            self._escape_until = None

        # An installed pilot replaces the steering law below, and nothing else. Every transition in
        # this state - arrival, the goal timer, the IR ring, the front ultrasonic, the avoidance
        # counter - has already been decided above, so a learned controller can only choose how to
        # drive between those decisions, never whether the robot is allowed to. It is skipped on the
        # final approach, where accuracy against the goal matters more than smoothness.
        if self.pilot is not None and not final and self._path:
            out = self.pilot(f, self._path)
            if out is not None:
                return Command(vx=out[0], vy=out[1], wz=out[2]), None

        # Steer at the route, not at the goal. Distance and the final approach are always measured to
        # the real goal; only the direction comes from the path, and on the last stretch the path is
        # ignored so the robot settles on the point itself.
        tx, ty = self._goal
        if not final:
            carrot = self._carrot(f)
            if carrot is not None:
                tx, ty = carrot
        sx, sy = tx - f.pose_x, ty - f.pose_y
        # error expressed in the robot's own frame
        ex = sx * math.cos(f.yaw) + sy * math.sin(f.yaw)
        ey = -sx * math.sin(f.yaw) + sy * math.cos(f.yaw)
        bearing = (math.atan2(sy, sx) - f.yaw + math.pi) % (2 * math.pi) - math.pi

        if final and self.p.holonomic:
            # The mecanum platform can translate in any direction, so on the last stretch it slides
            # straight onto the point instead of turning, driving and overshooting. This is what makes
            # arrival accurate: the speed falls with the remaining distance and both axes are servoed.
            v = max(0.04, min(self.p.v_final, 0.8 * dist))
            norm = math.hypot(ex, ey) or 1.0
            return Command(vx=v * ex / norm, vy=v * ey / norm), None

        # how far to drive is set by the distance to the goal, not to the nearby carrot
        drive_dist = dist

        if abs(bearing) > self.p.goal_yaw_align:
            return Command(wz=math.copysign(self.p.w_turn, bearing)), None      # turn on the spot first
        # facing the goal: drive, easing off over the last stretch so the robot settles
        v_cap = self.p.v_final if final else self.p.v_goal
        cmd = Command(vx=max(0.04, min(v_cap, 0.8 * drive_dist)), wz=self.p.k_goal_yaw * bearing)
        if self.p.holonomic:
            cmd.vy = max(-self.p.v_strafe, min(self.p.v_strafe, self.p.k_goal_lat * ey))
        return cmd, None

    def s_obstacle_avoidance(self, f: SensorFrame, t: float) -> Tuple[Command, Optional[Transition]]:
        """OBSTACLE_AVOIDANCE: rotate (or strafe) towards the side the IR pair reports as freer."""
        ir_min = f.ir_min()
        # TRANSITION OBSTACLE_AVOIDANCE -> BACKING_OFF: obstacle inside the turning radius
        if ir_min is not None and ir_min < self.p.ir_critical:
            return Command(), self._go(BACKING_OFF, t, 'ir.min', ir_min)
        # TRANSITION OBSTACLE_AVOIDANCE -> ALERT: stuck. Measured since PATROLLING was left, not since
        # this state was entered: an OBSTACLE_AVOIDANCE <-> BACKING_OFF limit cycle (seen in simulation,
        # 12 cycles in 20 s against a pillar) would otherwise reset the timer at every hop.
        elapsed = t - (self._episode_start if self._episode_start is not None else self.entered_at)
        if elapsed > self.p.t_avoid_max:
            return Command(), self._go(ALERT, t, 'timer.avoid', elapsed, 'STUCK')
        # TRANSITION OBSTACLE_AVOIDANCE -> PATROLLING: front clear (with hysteresis) AND both IR
        # above ir_steer, and it has stayed that way for t_clear seconds
        front_clear = f.us_front is not None and f.us_front > self.p.d_clear
        ir_clear = (f.ir_left is None or f.ir_left > self.p.ir_steer) and \
                   (f.ir_right is None or f.ir_right > self.p.ir_steer)
        if front_clear and ir_clear:
            if self._clear_since is None:
                self._clear_since = t
            elif t - self._clear_since >= self.p.t_clear:
                # resume whatever was interrupted: the clicked goal if one is still active
                nxt = NAVIGATING if (self._goal is not None and f.has_pose()) else PATROLLING
                if nxt is NAVIGATING:
                    self._escape_until = t + self.p.t_escape
                return Command(), self._go(nxt, t, 'ultrasonic.front', f.us_front)
        else:
            self._clear_since = None
        # action: strafe if the platform is holonomic and that side is open, otherwise rotate in place.
        # Strafing is bounded: past t_strafe_max the obstacle is wider than a pillar (a wall seen at an
        # angle, for example) and sliding along it never frees the front, so fall back to rotating.
        if self.p.holonomic and self._strafe_dir != 0:
            if self._strafe_start is not None and t - self._strafe_start > self.p.t_strafe_max:
                self._strafe_dir = 0
            else:
                return Command(vy=self._strafe_dir * self.p.v_strafe), None
        return Command(wz=self._turn_dir * self.p.w_turn), None

    def s_backing_off(self, f: SensorFrame, t: float) -> Tuple[Command, Optional[Transition]]:
        """BACKING_OFF: reverse a few centimetres to make room for the turn."""
        elapsed = t - self.entered_at
        # TRANSITION BACKING_OFF -> ALERT: the whole avoidance episode is taking too long (stuck)
        episode = t - (self._episode_start if self._episode_start is not None else self.entered_at)
        if episode > self.p.t_avoid_max:
            return Command(), self._go(ALERT, t, 'timer.avoid', episode, 'STUCK')
        # TRANSITION BACKING_OFF -> OBSTACLE_AVOIDANCE (holonomic): rear ultrasonic says stop reversing
        if self.p.holonomic and f.us_rear is not None and f.us_rear < self.p.d_rear_min:
            self._choose_direction(f, t)
            return Command(), self._go(OBSTACLE_AVOIDANCE, t, 'ultrasonic.rear', f.us_rear)
        ir_min = f.ir_min()
        # hysteresis: entered below ir_critical (0.12 m), leaves only above ir_back_exit (0.18 m);
        # a shared threshold produced a BACKING_OFF <-> OBSTACLE_AVOIDANCE chatter at 0.11-0.12 m
        ir_ok = ir_min is None or ir_min > self.p.ir_back_exit
        # TRANSITION BACKING_OFF -> OBSTACLE_AVOIDANCE: minimum time elapsed and IR clear of the
        # hysteresis band, or the hard time limit reached (no rear sensor on the diff-drive robot)
        if (elapsed >= self.p.t_back and ir_ok) or elapsed >= self.p.t_back_max:
            self._choose_direction(f, t)     # re-decide the side with the sensors as they are now
            return Command(), self._go(OBSTACLE_AVOIDANCE, t, 'timer.back', elapsed)
        return Command(vx=self.p.v_back), None

    def s_alert(self, f: SensorFrame, t: float) -> Tuple[Command, Optional[Transition]]:
        """ALERT: full stop. Only the operator's reset leaves this state."""
        # TRANSITION ALERT -> IDLE: operator reset (no automatic exit, by design)
        if self._reset_requested:
            self._reset_requested = False
            return Command(), self._go(IDLE, t, 'operator', 0.0, 'reset')
        return Command(), None

    # ------------------------------------------------------------------ helpers
    def _choose_direction(self, f: SensorFrame, t: float) -> None:
        """Turn towards the side with the larger IR distance (more free space). Left wins ties.

        Holonomic platform: strafe instead of rotating only when the obstacle is narrow and off to
        one side (the two corner IRs disagree) and the side ultrasonic on the chosen side is open.
        A flat wall lights both IRs at nearly the same distance; strafing along it never clears the
        front ultrasonic (seen in simulation: 10 s stuck against the arena wall), so rotate instead.
        """
        left = f.ir_left if f.ir_left is not None else float('inf')
        right = f.ir_right if f.ir_right is not None else float('inf')
        self._turn_dir = 1 if left >= right else -1
        self._strafe_dir = 0
        self._strafe_start = None
        if self.p.holonomic:
            both_seen = f.ir_left is not None and f.ir_right is not None
            wall_like = both_seen and abs(f.ir_left - f.ir_right) < self.p.ir_wall_delta
            side = f.us_left if self._turn_dir > 0 else f.us_right
            if not wall_like and side is not None and side > self.p.d_side_clear:
                self._strafe_dir = self._turn_dir
                self._strafe_start = t

    def _go(self, dst: str, t: float, sensor: str, value: float, reason: str = '') -> Transition:
        tr = Transition(self.state, dst, sensor, value, reason)
        if dst == ALERT:
            self.last_alert = AlertEvent(reason or 'UNKNOWN', sensor, value, self.state)
        self.state = dst
        self.entered_at = t
        self._obstacle_count = 0
        self._imu_spike_count = 0
        self._clear_since = None
        self._progress_ref = None
        if dst in (IDLE, ALERT):
            # A 'start' sent while the robot was moving (or sitting in ALERT) must not survive into
            # IDLE: it would re-launch the patrol on the very next tick, defeating both the operator's
            # stop and the rule that a human clears an ALERT. A clicked goal is dropped for the same
            # reason - after a stop or a collision the robot must not resume driving to an old point.
            self._start_requested = False
            self._goal = None
            self._goal_start = None
            self._escape_until = None
            self._avoid_count = 0
            self._path = []
            self._path_i = 0
        return tr
