"""Run the trained policy on the robot, as the steering law inside NAVIGATING.

The adapter is the only place where the robot's messages meet the network, and it builds the
observation by calling the same features module the training loop called. Anything it cannot supply
honestly - a reference too short to look ahead along, a stale velocity, missing weights - makes it
return None, and the FSM falls back to the hand-written law. A learned controller that quietly does
something odd is worse than one that hands back.
"""
import math
import os

import numpy as np

from . import features, mlp, model


class PilotAdapter:

    MIN_PATH = 0.60         # m of reference; shorter than the lookahead and the policy is guessing
    TRACE_ENV = 'PILOT_TRACE'   # set to a file path and every inference is appended to it as CSV
    STALE = 0.5             # s without odometry or a scan

    def __init__(self, weights: str, lim: model.Limits = None, logger=None) -> None:
        self.net = mlp.MLP.load(weights)
        self.lim = lim or model.Limits()
        self.log = logger
        self.scale = np.array([self.lim.v_max, self.lim.v_max, self.lim.w_max])
        self.vel = (0.0, 0.0, 0.0)          # body-frame velocity from /odom
        self.vel_t = -1e9
        self.ranges = None                  # twelve body-frame sectors from /scan
        self.ranges_t = -1e9
        self.ref = None
        self._path_key = None
        # the trace file is opened on the first step that is recorded; None until then
        self._trace = None
        self.calls = 0
        self.handbacks = 0
        self.engaged = False

    # ------------------------------------------------------------------ inputs from the node
    def set_velocity(self, vx: float, vy: float, wz: float, t: float) -> None:
        self.vel, self.vel_t = (vx, vy, wz), t

    def set_scan(self, ranges_12, t: float) -> None:
        self.ranges, self.ranges_t = np.asarray(ranges_12, dtype=float), t

    @staticmethod
    def sectors_from_scan(ranges, angle_min: float, angle_increment: float, range_max: float):
        """Reduce a LaserScan to the twelve body-frame sectors the policy was trained on."""
        r = np.asarray(ranges, dtype=float)
        ok = np.isfinite(r) & (r > 0.0)
        ang = model.wrap(angle_min + np.arange(len(r)) * angle_increment)
        out = np.full(features.N_SECTORS, features.RANGE_MAX)
        half = np.pi / features.N_SECTORS
        for k, mid in enumerate(features.SECTOR_MID):
            sel = ok & (np.abs(model.wrap(ang - mid)) <= half)
            if sel.any():
                out[k] = min(features.RANGE_MAX, float(r[sel].min()))
        return out

    # ------------------------------------------------------------------ the pilot itself
    def __call__(self, f, path):
        """fsm_core calls this with the sensor frame and the current route."""
        self.calls += 1
        t = max(f.t_pose or 0.0, 0.0)
        if self.ranges is None or t - self.vel_t > self.STALE or t - self.ranges_t > self.STALE:
            return self._hand_back('no fresh odometry or scan')

        xy = np.asarray(path, dtype=float)
        if len(xy) < 2:
            return self._hand_back('route too short')
        key = (len(xy), float(xy[0, 0]), float(xy[0, 1]), float(xy[-1, 0]), float(xy[-1, 1]))
        if key != self._path_key:
            # a new route: find where on it the robot already is, rather than assuming the start
            self.ref = model.Reference(xy)
            self.ref.locate((f.pose_x, f.pose_y))
            self._path_key = key
            self._dump_route(xy)
        if self.ref.length < self.MIN_PATH:
            return self._hand_back('route shorter than the lookahead')

        self.ref.advance((f.pose_x, f.pose_y))
        state = model.make_state(f.pose_x, f.pose_y, f.yaw,
                                 self.vel[0], self.vel[1], self.vel[2])
        obs = features.observation(state, self.ref, self.lim, ranges=self.ranges)
        act = self._infer(obs)
        self._record(t, state, obs, act)
        out = act * self.scale
        if not np.all(np.isfinite(out)):
            return self._hand_back('policy produced a non-finite command')
        # a last clamp on the speed, so the policy cannot ask for more than the base can give
        speed = math.hypot(out[0], out[1])
        if speed > self.lim.v_max:
            out[0] *= self.lim.v_max / speed
            out[1] *= self.lim.v_max / speed
        if not self.engaged:
            self.engaged = True
            if self.log is not None:
                self.log.info(f'pilot engaged: route {self.ref.length:.2f} m')
        return float(out[0]), float(out[1]), float(np.clip(out[2], -self.lim.w_max, self.lim.w_max))

    def _infer(self, obs):
        return self.net(obs)

    def _dump_route(self, xy: np.ndarray) -> None:
        path = os.environ.get(self.TRACE_ENV)
        if path:
            with open(path + '.routes', 'a') as fh:
                fh.write('%d %s' % (hash(self._path_key) % 100000, ' '.join('%.4f' % v for v in xy.ravel())) + chr(10))

    def _record(self, t: float, state, obs, act) -> None:
        """What the policy saw and what it did, at every step: the record that explains a failure on
        the robot, and the states the policy actually visits in Gazebo, which the MPPI teacher can
        label for another round of DAgger."""
        path = os.environ.get(self.TRACE_ENV)
        if not path:
            return
        if self._trace is None:
            self._trace = open(path, 'a')
            self._trace.write('t,x,y,yaw,vx,vy,wz,ref_pos,ref_len,route,' +
                              ','.join('o%d' % i for i in range(len(obs))) + ',ax,ay,aw' + chr(10))
        row = [t, state[model.X], state[model.Y], state[model.TH], state[model.VX], state[model.VY], state[model.W],
               self.ref.pos, self.ref.length, hash(self._path_key) % 100000] + list(obs) + list(act)
        self._trace.write(','.join('%.5f' % v for v in row) + chr(10))
        self._trace.flush()

    def _hand_back(self, why: str):
        self.handbacks += 1
        if self.log is not None and self.handbacks % 50 == 1:
            self.log.warning(f'pilot handing back to the hand-written law: {why}')
        return None
