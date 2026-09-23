"""Top-down drawing of a generated arena, with the path the robot actually drove if a log is given.

    python3 tools/map_figure.py warehouse corridors forest --out docs/evidence_planner/maps/arenas.png
    python3 tools/map_figure.py warehouse corridors forest --logs docs/evidence_planner/maps/mission_log_{name}_pilot --out arenas_driven.png

Walls and shelves in grey, columns in red, the rings the mission asks for as dashed circles, the
spawn as a green square, the driven path (odom frame, shifted by the spawn offset into the world
frame) in green, and every point where the FSM took the wheel from the pilot in orange.
"""
import argparse
import json
import math
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt                         # noqa: E402
from matplotlib.patches import Circle, Rectangle          # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import make_worlds as mw                                  # noqa: E402


def draw(ax, name, spec, log=None):
    for o in mw.arena() + spec['obstacles']:
        t = matplotlib.transforms.Affine2D().rotate_around(o['x'], o['y'], o['yaw']) + ax.transData
        ax.add_patch(Rectangle((o['x'] - o['sx'] / 2, o['y'] - o['sy'] / 2), o['sx'], o['sy'],
                               facecolor='#8a8f94' if o['name'].startswith('wall') else '#a0785a',
                               edgecolor='none', transform=t))
    for k, (x, y) in enumerate(spec['columns']):
        ax.add_patch(Circle((x, y), mw.COLUMN_R, facecolor='#c2410c', edgecolor='none'))
        ax.add_patch(Circle((x, y), spec['ring'], fill=False, linestyle='--', linewidth=0.9, edgecolor='#c2410c', alpha=0.6))
        ax.text(x, y + spec['ring'] + 0.12, str(k + 1), ha='center', va='bottom', fontsize=8, color='#c2410c')
    sx, sy = spec['spawn']
    ax.add_patch(Rectangle((sx - 0.1, sy - 0.1), 0.2, 0.2, facecolor='#1f9e6e', edgecolor='none'))
    if log is not None:
        # /odom starts at the spawn pose or at the origin; the mission logs which by its offset
        poses = log['poses']
        ox, oy = poses[0][1], poses[0][2]
        dx, dy = (sx - ox, sy - oy) if math.hypot(ox, oy) < 0.05 else (0.0, 0.0)
        xs = [p[1] + dx for p in poses]
        ys = [p[2] + dy for p in poses]
        ax.plot(xs, ys, color='#1f9e6e', linewidth=1.1, alpha=0.9)
        prev, pi = None, 0
        for t, s in log.get('states', []):
            if s in ('OBSTACLE_AVOIDANCE', 'BACKING_OFF') and prev not in ('OBSTACLE_AVOIDANCE', 'BACKING_OFF'):
                while pi + 1 < len(poses) and poses[pi + 1][0] <= t:
                    pi += 1
                ax.plot(poses[pi][1] + dx, poses[pi][2] + dy, 'o', color='#c97a16', markersize=5)
            prev = s
    ax.set_xlim(-mw.HALF - 0.2, mw.HALF + 0.2)
    ax.set_ylim(-mw.HALF - 0.2, mw.HALF + 0.2)
    ax.set_aspect('equal')
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(name, fontsize=11)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('maps', nargs='+')
    ap.add_argument('--log', default='', help='mission_log dir (for a single map)')
    ap.add_argument('--logs', default='', help='template with {name}, e.g. docs/evidence_planner/maps/mission_log_{name}_pilot')
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    fig, axes = plt.subplots(1, len(args.maps), figsize=(4.2 * len(args.maps), 4.4), dpi=150)
    axes = [axes] if len(args.maps) == 1 else list(axes)
    for ax, name in zip(axes, args.maps):
        d = args.log or (args.logs.format(name=name) if args.logs else '')
        log = json.load(open(os.path.join(d, 'log.json'))) if d and os.path.exists(os.path.join(d, 'log.json')) else None
        draw(ax, name, mw.MAPS[name], log)
    fig.tight_layout()
    fig.savefig(args.out)
    print('wrote', args.out)


if __name__ == '__main__':
    main()
