"""The observation the student sees, built the same way in training and on the robot.

Train/serve skew is the usual way an imitation-learned controller fails, so there is exactly one
implementation of this and both sides call it. Everything is in the body frame and normalised to
roughly [-1, 1], because a small MLP trained on raw metres and radians spends its capacity learning
the scaling instead of the task.

    0 .. 11   six lookahead points on the reference, at fixed arc lengths ahead, body frame
    12 .. 14  the robot's own velocity, body frame
    15 .. 26  nearest obstacle in each of twelve body-frame sectors, clipped

The teacher sees more than this - exact column positions and the whole path - which is the point:
the student is distilled from privileged information into what the robot can actually measure.
"""
import numpy as np

from . import model

LOOKAHEAD = (0.15, 0.30, 0.50, 0.75, 1.05, 1.40)     # m ahead along the reference
N_SECTORS = 12
RANGE_MAX = 1.5                                       # m, clip for the sector distances
POS_SCALE = 1.5                                       # m, normalises the lookahead points
OBS_DIM = 2 * len(LOOKAHEAD) + 3 + N_SECTORS

_EDGES = np.linspace(-np.pi, np.pi, N_SECTORS + 1)
SECTOR_MID = 0.5 * (_EDGES[:-1] + _EDGES[1:])


def lookahead_points(state: np.ndarray, ref: model.Reference) -> np.ndarray:
    """The reference sampled ahead of the robot, expressed in the body frame."""
    want = np.clip(ref.pos + np.asarray(LOOKAHEAD), 0.0, ref.length)
    idx = np.minimum(np.searchsorted(ref.s, want), len(ref.path) - 1)
    d = ref.path[idx] - state[:2]
    cos, sin = np.cos(state[model.TH]), np.sin(state[model.TH])
    return np.stack([d[:, 0] * cos + d[:, 1] * sin,
                     -d[:, 0] * sin + d[:, 1] * cos], axis=-1)


def sector_ranges(state: np.ndarray, columns: np.ndarray = model.COLUMNS) -> np.ndarray:
    """Nearest obstacle surface in each body-frame sector, the way a coarse lidar would see it.

    On the robot this comes from /scan reduced to the same twelve sectors; here it is computed from
    the known geometry. Both answer 'how far to the nearest thing in that direction', clipped.
    """
    xy, th = state[:2], state[model.TH]
    d = columns - xy
    rng = np.linalg.norm(d, axis=-1) - model.COLUMN_RADIUS
    bearing = model.wrap(np.arctan2(d[:, 1], d[:, 0]) - th)

    half = np.pi / N_SECTORS
    hit = np.abs(model.wrap(bearing[None, :] - SECTOR_MID[:, None])) <= half
    out = np.where(hit, rng[None, :], np.inf).min(axis=1)
    wall = model.ARENA_RADIUS - float(np.linalg.norm(xy))
    return np.clip(np.minimum(np.minimum(out, RANGE_MAX), wall), 0.0, RANGE_MAX)


def observation(state: np.ndarray, ref: model.Reference, lim: model.Limits,
                ranges: np.ndarray = None) -> np.ndarray:
    """Assemble and normalise. `ranges` overrides the geometric sectors (the robot passes /scan)."""
    pts = lookahead_points(state, ref) / POS_SCALE
    vel = np.array([state[model.VX] / lim.v_max,
                    state[model.VY] / lim.v_max,
                    state[model.W] / lim.w_max])
    rng = sector_ranges(state) if ranges is None else np.asarray(ranges, dtype=float)
    rng = np.clip(rng, 0.0, RANGE_MAX) / RANGE_MAX
    return np.concatenate([pts.reshape(-1), vel, rng])
