"""The teacher: a sampling model-predictive controller (MPPI) for the mecanum base.

It samples a few hundred candidate control sequences around its previous solution, rolls each one
forward through the plant model, scores them, and takes the softmax-weighted average. That handles
the non-convex part of the problem - obstacles - without a gradient, and vectorises into numpy, so
one control step costs a couple of milliseconds instead of the tens a nonlinear solve would.

It is a teacher rather than the deployed controller for two reasons: it needs privileged
information (exact column positions, the whole reference), and a few milliseconds every 100 ms on a
Raspberry Pi is a real cost. The student distils it into a fixed 27 -> 64 -> 64 -> 3 forward pass.

The cost is where the behaviour the robot was missing gets specified:

    progress    reward arc length gained along the reference - the reason it drives instead of idling
    cross-track keep to the ring
    heading     point the front roughly along the path, so the forward sensors stay useful
    yaw rate    penalise spinning: this is the term that stops the turn-drive-turn shuffle
    jerk        penalise changes in command, which is what makes the motion look smooth
    obstacle    a soft margin outside the footprint, and a wall inside it
"""
from dataclasses import dataclass

import numpy as np

from . import model


@dataclass(frozen=True)
class MppiWeights:
    progress: float = 26.0
    cross_track: float = 12.0
    heading: float = 1.2
    yaw_rate: float = 0.65
    jerk: float = 0.35
    obstacle: float = 40.0
    margin: float = 0.30        # m of clearance the soft penalty starts at
    hard: float = 0.20          # m of clearance treated as a collision


class MppiTeacher:

    def __init__(self, lim: model.Limits = model.Limits(), horizon: int = 15, samples: int = 512,
                 temperature: float = 0.6, sigma=(0.12, 0.12, 0.35),
                 weights: MppiWeights = MppiWeights(), seed: int = 0) -> None:
        self.lim = lim
        self.h = horizon
        self.m = samples
        self.temp = temperature
        self.sigma = np.asarray(sigma, dtype=float)
        self.w = weights
        self.rng = np.random.default_rng(seed)
        self.nominal = np.zeros((horizon, 3))

    def reset(self) -> None:
        self.nominal[:] = 0.0

    # ------------------------------------------------------------------ cost
    def _cost(self, states: np.ndarray, cmds: np.ndarray, path: np.ndarray, s: np.ndarray,
              s0: np.ndarray, tangent: np.ndarray) -> np.ndarray:
        """states (M, H+1, 6), cmds (M, H, 3) -> total cost per sample."""
        xy = states[:, 1:, :2]
        w = self.w
        cost = np.zeros(states.shape[0])

        # progress: how much arc length the sample gained, wrapped so the lap's seam is not a cliff
        gained = model.project(xy[:, -1, :], path, s) - s0
        cost -= w.progress * gained

        cost += w.cross_track * (model.cross_track(xy, path) ** 2).sum(axis=1)

        # heading: the front should point roughly where the robot is going
        want = tangent[np.searchsorted(s, np.clip(model.project(xy, path, s), 0, s[-1]))]
        cost += w.heading * (model.wrap(states[:, 1:, model.TH] - want) ** 2).sum(axis=1)

        cost += w.yaw_rate * (cmds[..., 2] ** 2).sum(axis=1)
        cost += w.jerk * (np.diff(cmds, axis=1) ** 2).sum(axis=(1, 2))

        clear = model.clearance(xy)
        soft = np.clip(w.margin - clear, 0.0, None)
        cost += w.obstacle * (soft ** 2).sum(axis=1)
        cost += 1e3 * (clear < w.hard).sum(axis=1)
        return cost

    # ------------------------------------------------------------------ control
    def __call__(self, state: np.ndarray, ref: model.Reference) -> np.ndarray:
        # Only the stretch of reference the horizon can reach matters, and scoring against the whole
        # lap costs a 512 x 15 x N distance matrix every control step. The window also keeps a
        # closed ring unambiguous: without it the finish line is a neighbour of the start line.
        s0 = ref.pos
        path, s = ref.window(back=0.4, ahead=1.6)

        d = np.diff(path, axis=0, append=path[-1:] + (path[-1:] - path[-2:-1]))
        tangent = np.arctan2(d[:, 1], d[:, 0])

        eps = self.rng.normal(0.0, 1.0, (self.m, self.h, 3)) * self.sigma
        cand = self.nominal[None] + eps
        cand = np.clip(cand, [-self.lim.v_max, -self.lim.v_max, -self.lim.w_max],
                       [self.lim.v_max, self.lim.v_max, self.lim.w_max])

        roll = np.empty((self.m, self.h + 1, model.STATE_DIM))
        roll[:, 0] = state
        for k in range(self.h):
            roll[:, k + 1] = model.step(roll[:, k], cand[:, k], self.lim)

        cost = self._cost(roll, cand, path, s, s0, tangent)
        adv = cost - cost.min()
        weight = np.exp(-adv / max(self.temp, 1e-6))
        weight /= weight.sum()
        self.nominal = np.einsum('m,mhc->hc', weight, cand)

        out = self.nominal[0].copy()
        self.nominal = np.roll(self.nominal, -1, axis=0)
        self.nominal[-1] = 0.0
        return out
