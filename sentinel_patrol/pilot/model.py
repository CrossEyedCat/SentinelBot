"""Plant model and arena geometry shared by the MPC teacher, the training loop and the tests.

Deliberately ROS-free and vectorised: the teacher rolls out several hundred candidate control
sequences per control step, and training rolls out thousands of episodes, so everything here takes
a leading batch dimension and never loops in Python over samples.

The base is modelled as what mecanum_base_sim actually is: a PI velocity loop whose output is an
acceleration, clamped to the design's limits (0.6 m/s^2, 2 rad/s^2), integrated in a body frame
that is itself rotating.

An earlier version of this file used a first-order tracker instead - the command was reached at
a_max with no lag beyond that. A policy trained against it matched its teacher to 2% here and then
wandered in Gazebo, covering 48.8 m of a 35.3 m route, because commands tuned for a plant that
responds immediately overshoot one that does not. The gains below are read from
mecanum_base_sim_node.py; if they are retuned there, they must be retuned here, and the policy
retrained.
"""
from dataclasses import dataclass

import numpy as np

# turtlebot3_world: nine columns on a 1.1 m grid, 0.15 m radius, inside a wall about 2.5 m out
COLUMNS = np.array([[x, y] for x in (-1.1, 0.0, 1.1) for y in (-1.1, 0.0, 1.1)], dtype=float)
COLUMN_RADIUS = 0.15
ARENA_RADIUS = 2.45

# State layout: pose in the world frame, velocities in the body frame, and the three integrator
# terms of the velocity loop - the plant has memory, so a rollout has to carry it.
X, Y, TH, VX, VY, W, IX, IY, IW = range(9)
STATE_DIM = 9


@dataclass(frozen=True)
class Limits:
    v_max: float = 0.30         # m/s, the mission profile's v_goal
    w_max: float = 1.00         # rad/s
    a_max: float = 0.60         # m/s^2, traction limited (design check)
    alpha_max: float = 2.00     # rad/s^2
    dt: float = 0.10            # s, the FSM's control period
    # the velocity loop of mecanum_base_sim, which is what turns a command into motion
    kp_lin: float = 4.0         # 1/s
    ki_lin: float = 3.0         # 1/s^2
    kp_ang: float = 8.0         # 1/s
    ki_ang: float = 8.0         # 1/s^2
    substeps: int = 4           # the real loop runs on odometry at ~50 Hz, not at the command rate


def step(state: np.ndarray, cmd: np.ndarray, lim: Limits) -> np.ndarray:
    """One control period of the FSM, integrated at the rate the velocity loop really runs at.

    state (..., 9), cmd (..., 3) body-frame velocity request. The command is held for the whole
    period, as it is on the robot: the FSM publishes at 10 Hz and the base controller runs on every
    odometry message.
    """
    s = np.array(state, dtype=float, copy=True)
    # The linear limit is on the speed, not on each axis: clamping vx and vy separately would let a
    # diagonal command ask for 1.41 * v_max, which the wheels cannot deliver and which flattered
    # every controller that strafes.
    c = np.array(cmd, dtype=float, copy=True)
    speed = np.hypot(c[..., 0], c[..., 1])
    shrink = np.minimum(1.0, lim.v_max / np.maximum(speed, 1e-9))
    c[..., 0] *= shrink
    c[..., 1] *= shrink
    c[..., 2] = np.clip(c[..., 2], -lim.w_max, lim.w_max)

    h = lim.dt / lim.substeps
    for _ in range(lim.substeps):
        # PI on the velocity error, output clamped to the acceleration limit, with the same
        # clamping anti-windup the node uses: the integral only grows while the output is free
        ex, ey = c[..., 0] - s[..., VX], c[..., 1] - s[..., VY]
        ew = c[..., 2] - s[..., W]
        ux = lim.kp_lin * ex + lim.ki_lin * s[..., IX]
        uy = lim.kp_lin * ey + lim.ki_lin * s[..., IY]
        uw = lim.kp_ang * ew + lim.ki_ang * s[..., IW]
        s[..., IX] += np.where(np.abs(ux) < lim.a_max, ex * h, 0.0)
        s[..., IY] += np.where(np.abs(uy) < lim.a_max, ey * h, 0.0)
        s[..., IW] += np.where(np.abs(uw) < lim.alpha_max, ew * h, 0.0)
        ax = np.clip(ux, -lim.a_max, lim.a_max)
        ay = np.clip(uy, -lim.a_max, lim.a_max)
        alpha = np.clip(uw, -lim.alpha_max, lim.alpha_max)

        # the body frame is rotating, so a velocity held constant in the world turns in the body:
        # d/dt v_body = a_body - omega x v_body
        vx, vy, w = s[..., VX].copy(), s[..., VY].copy(), s[..., W].copy()
        s[..., VX] = vx + (ax + w * vy) * h
        s[..., VY] = vy + (ay - w * vx) * h
        s[..., W] = w + alpha * h

        th = s[..., TH]
        cos, sin = np.cos(th), np.sin(th)
        s[..., X] += (s[..., VX] * cos - s[..., VY] * sin) * h
        s[..., Y] += (s[..., VX] * sin + s[..., VY] * cos) * h
        s[..., TH] = wrap(th + s[..., W] * h)
    return s


def make_state(x=0.0, y=0.0, th=0.0, vx=0.0, vy=0.0, w=0.0) -> np.ndarray:
    """A plant state with the integrators at rest. The policy never sees the last three."""
    out = np.zeros(STATE_DIM)
    out[X], out[Y], out[TH], out[VX], out[VY], out[W] = x, y, th, vx, vy, w
    return out


def wrap(a):
    """Angle(s) into (-pi, pi]."""
    return (np.asarray(a) + np.pi) % (2.0 * np.pi) - np.pi


def clearance(xy: np.ndarray, columns: np.ndarray = COLUMNS) -> np.ndarray:
    """Distance from point(s) to the nearest obstacle surface: columns and the arena wall."""
    d = np.linalg.norm(xy[..., None, :] - columns, axis=-1).min(axis=-1) - COLUMN_RADIUS
    wall = ARENA_RADIUS - np.linalg.norm(xy, axis=-1)
    return np.minimum(d, wall)


def ring_path(centre, radius: float, th_in: float, th_out: float, step_m: float = 0.05):
    """The reference for one lap: in on the side the robot arrives from, a full turn, out towards
    the next column. Same geometry column_tour_node hands to the FSM, but sampled densely enough to
    be followed as a curve instead of visited as a sequence of points."""
    ccw = (th_out - th_in) % (2.0 * np.pi)
    turn = 1.0 if ccw <= np.pi else -1.0
    sweep = 2.0 * np.pi + (ccw if turn > 0 else 2.0 * np.pi - ccw)
    n = max(8, int(round(sweep * radius / step_m)))
    a = th_in + turn * np.linspace(0.0, sweep, n + 1)
    return np.stack([centre[0] + radius * np.cos(a), centre[1] + radius * np.sin(a)], axis=1)


def arc_length(path: np.ndarray) -> np.ndarray:
    """Cumulative distance along a polyline, starting at 0."""
    d = np.linalg.norm(np.diff(path, axis=0), axis=1)
    return np.concatenate([[0.0], np.cumsum(d)])


def project(xy: np.ndarray, path: np.ndarray, s: np.ndarray) -> np.ndarray:
    """Arc length of the nearest path sample to each query point. Nearest-sample rather than
    nearest-segment: the path is sampled every 5 cm, so the difference is below the noise.

    Only ever call this on a *window* of the reference. A lap is a closed ring, so the point the
    robot starts at is also the point it finishes at; asked globally, this returns the finish line
    on the first control step and the episode ends before it begins.
    """
    d = np.linalg.norm(xy[..., None, :] - path, axis=-1)
    return s[np.argmin(d, axis=-1)]


class Reference:
    """A path plus where along it the robot currently is.

    Progress is tracked forwards through a window instead of being recomputed from scratch, which
    is what keeps a closed ring unambiguous: the robot is at arc length 0.1 m, not at 3.5 m.
    """

    BACK = 0.25         # m, how far back the search may look (a robot pushed off the path)
    AHEAD = 1.20        # m, how far forward - more than one control horizon of travel
    STEP_AHEAD = 0.35   # m, how far progress may advance in one call: the robot covers 0.03 m in a
    #                     control period, so anything more than this is the ring's own ambiguity
    MAX_OFF = 0.60      # m, a query further than this from the reference says nothing about progress

    def __init__(self, path: np.ndarray, pos: float = 0.0) -> None:
        self.path = np.asarray(path, dtype=float)
        self.s = arc_length(self.path)
        self.pos = pos

    def reset(self, pos: float = 0.0) -> None:
        self.pos = pos

    @property
    def length(self) -> float:
        return float(self.s[-1])

    def window(self, back: float = None, ahead: float = None):
        lo = self.pos - (self.BACK if back is None else back)
        hi = self.pos + (self.AHEAD if ahead is None else ahead)
        keep = (self.s >= lo) & (self.s <= hi)
        if not keep.any():
            keep = np.zeros_like(self.s, dtype=bool)
            keep[min(int(np.searchsorted(self.s, self.pos)), len(self.s) - 1)] = True
        return self.path[keep], self.s[keep]

    def locate(self, xy: np.ndarray) -> float:
        """Place the marker on a reference the robot is already somewhere along.

        `advance` may only creep forwards, which is right once the robot is being tracked and wrong
        when a route is handed over mid-lap: walking up from zero through a 0.35 m window would pin
        the marker near the start and leave every lookahead point behind the robot. This searches
        the whole path once, and among the samples that are equally close it takes the earliest -
        on a closed ring the robot's position matches two arc lengths, and the one it has not driven
        yet is the one worth steering along.
        """
        xy = np.asarray(xy, dtype=float)
        d = np.linalg.norm(self.path - xy, axis=1)
        close = np.flatnonzero(d <= max(d.min() + 0.05, 0.10))
        self.pos = float(self.s[close[0]]) if len(close) else 0.0
        return self.pos

    def advance(self, xy: np.ndarray) -> float:
        """Move the progress marker forwards, never backwards, and never by a leap.

        A lap is a closed ring, so a position on it is genuinely ambiguous - it is both arc 0.1 and
        arc 3.5. Taking the nearest sample within a short forward window resolves that the only way
        that makes sense, and refusing a query that is far off the reference stops an avoidance
        manoeuvre from being scored as having driven half the lap.
        """
        path, s = self.window(ahead=self.STEP_AHEAD)
        d = np.linalg.norm(path - np.asarray(xy, dtype=float), axis=1)
        i = int(np.argmin(d))
        if d[i] <= self.MAX_OFF:
            self.pos = max(self.pos, float(s[i]))
        return self.pos


def cross_track(xy: np.ndarray, path: np.ndarray) -> np.ndarray:
    """Distance from each query point to the nearest path sample."""
    return np.linalg.norm(xy[..., None, :] - path, axis=-1).min(axis=-1)
