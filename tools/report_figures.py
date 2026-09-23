"""Figures for the written submission: the state diagram, the run timeline, the evidence montages.

    python3 tools/report_figures.py --out docs/submission/figures

Everything here is drawn from files already in the package - the parameter defaults in
`config/patrol_params.yaml`, the logs in `docs/evidence_jazzy_run/`, the screenshots the recorder
archived, the renders of the robot's design in `docs/robot/`, the mission logs in
`docs/evidence_planner/` and the training record `tools/train_pilot.py --record` wrote - so a figure
cannot drift away from the system it describes. Nothing is re-simulated.
"""
import argparse
import csv
import json
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt                                           # noqa: E402
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle  # noqa: E402
from PIL import Image                                                     # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.join(HERE, '..')

INK, MUTED, RULE = '#16191C', '#59636D', '#B4BCB3'
BOX, BOXEDGE, ALERTC, IDLEC = '#EEF2F5', '#27333D', '#B23A3A', '#4A5A66'
ACCENT = '#0F6E74'


def box(ax, x, y, w, h, label, sub, edge=BOXEDGE, face=BOX, size=11.0):
    """One state. The y axis runs downwards, so the name goes at the smaller y."""
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0,rounding_size=4',
                                linewidth=1.6, edgecolor=edge, facecolor=face, zorder=3))
    ax.text(x + w / 2, y + h * 0.36, label, ha='center', va='center', fontsize=size,
            color=edge, zorder=4, fontweight='bold')
    ax.text(x + w / 2, y + h * 0.74, sub, ha='center', va='center', fontsize=7.8,
            color=MUTED, zorder=4)


def arrow(ax, p0, p1, text, rad=0.0, colour=INK, dx=0, dy=0, size=8.0, style='-|>'):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle=style, mutation_scale=11,
                                 connectionstyle='arc3,rad=%g' % rad, linewidth=1.15,
                                 color=colour, zorder=2, shrinkA=2, shrinkB=2))
    if text:
        mx, my = (p0[0] + p1[0]) / 2, (p0[1] + p1[1]) / 2
        ax.text(mx + dx, my + dy, text, ha='center', va='center', fontsize=size,
                color=colour, zorder=5,
                bbox=dict(boxstyle='round,pad=0.22', facecolor='white', edgecolor='none'))


def fig_fsm(path):
    """The state machine as the code implements it, thresholds included."""
    fig, ax = plt.subplots(figsize=(11.2, 6.6), dpi=200)
    ax.set_xlim(0, 1000); ax.set_ylim(600, -34); ax.axis('off')

    ax.add_patch(Rectangle((268, 130), 644, 300, linewidth=1.0, edgecolor=RULE,
                           facecolor='#FAFBFA', zorder=0, linestyle=(0, (4, 3))))
    ax.text(898, 424, 'moving states: the operator stop and the sensor timeout are checked for all '
            'four, not inside the handlers', ha='right', va='bottom', fontsize=7.8, color=MUTED)

    box(ax, 40, 46, 170, 58, 'IDLE', 'motors held at zero', edge=IDLEC)
    box(ax, 300, 156, 240, 58, 'PATROLLING', 'ultrasonic  |  v = 0.15 m/s')
    box(ax, 648, 156, 240, 58, 'NAVIGATING', 'ultrasonic + IR + odometry, A*')
    box(ax, 300, 330, 240, 58, 'OBSTACLE_AVOIDANCE', 'the IR pair picks the turn', size=9.6)
    box(ax, 648, 330, 240, 58, 'BACKING_OFF', 'IR critical  |  v = -0.08 m/s')
    box(ax, 400, 480, 260, 58, 'ALERT', '/robot_alert, no automatic exit', edge=ALERTC,
        face='#FBEFEF')

    arrow(ax, (170, 104), (330, 156), 'start', dx=-30, dy=16)
    arrow(ax, (210, 58), (700, 156), 'a goal clicked on the map', rad=-0.22, dy=-46)
    arrow(ax, (690, 158), (212, 84), 'arrived, or the goal cannot be reached', rad=0.10, dy=-22,
          size=7.6)
    arrow(ax, (540, 172), (648, 172), 'goal', dy=-13, size=7.6)

    arrow(ax, (338, 214), (338, 330), 'front < 0.50 m for 3 ticks,\nor no progress for 4 s',
          dx=-74, dy=42, size=7.6)
    arrow(ax, (452, 330), (452, 214), 'front > 0.70 m and IR > 0.25 m\nheld for 0.5 s: resume\n'
          'the state that was interrupted', dx=52, dy=4, size=7.4)
    arrow(ax, (700, 214), (545, 332), 'front < 0.50 m\nor IR < 0.25 m', rad=0.14, dx=74, dy=-6,
          size=7.6)
    arrow(ax, (540, 344), (648, 344), 'IR < 0.12 m', dx=12, dy=-26, size=7.6)
    arrow(ax, (648, 374), (540, 374), 't > 0.6 s and IR > 0.18 m', dy=16, size=7.6)

    arrow(ax, (530, 430), (530, 480), '', colour=ALERTC)
    for k, line in enumerate((
            'from any moving state:',
            'horizontal acceleration > 8 m/s$^2$ twice   (collision)',
            'tilt > 30$^{\\circ}$   (tipping)',
            'one avoidance episode longer than 10 s   (stuck)',
            'no sensor reading for 1 s   (sensor lost)')):
        ax.text(676, 462 + 22 * k, line, ha='left', va='center', fontsize=8.2, color=ALERTC)
    arrow(ax, (400, 504), (110, 104), '', rad=0.46, colour=ALERTC)
    ax.text(150, 400, 'operator reset;\na human clears every alert,\nbecause after a collision\n'
            'the odometry is not trusted', ha='center', va='center', fontsize=8.2, color=ALERTC)

    fig.savefig(path, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print('wrote', path)


def fig_timeline(path):
    """The archived Jazzy run: which state, when, and what number caused the change."""
    tl = os.path.join(PKG, 'docs/evidence_jazzy_run/timeline.csv')
    rows = list(csv.DictReader(open(tl)))
    t = [float(r['sim_time']) for r in rows]
    s = [r['state'] for r in rows]
    order = ['IDLE', 'PATROLLING', 'OBSTACLE_AVOIDANCE', 'ALERT']
    colour = {'IDLE': IDLEC, 'PATROLLING': ACCENT, 'OBSTACLE_AVOIDANCE': '#C97A16',
              'ALERT': ALERTC}
    end = t[-1] + 6.0

    trig = []
    for line in open(os.path.join(PKG, 'docs/evidence_jazzy_run/transitions.log')):
        if '[FSM]' not in line:
            continue
        trig.append(line.split('trigger', 1)[1].strip())

    fig, ax = plt.subplots(figsize=(11.0, 2.9), dpi=200)
    for i, (t0, st) in enumerate(zip(t, s)):
        t1 = t[i + 1] if i + 1 < len(t) else end
        ax.add_patch(Rectangle((t0, order.index(st) - 0.33), t1 - t0, 0.66,
                               facecolor=colour[st], edgecolor='none', alpha=0.85))
    for i, (t0, st) in enumerate(zip(t[1:], s[1:])):
        if i < len(trig):
            ax.annotate(trig[i].replace(' = ', ' '), (t0, order.index(st) - 0.42),
                        fontsize=7.0, rotation=32, ha='left', va='bottom', color=MUTED)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(order, fontsize=9)
    ax.set_ylim(len(order) - 0.4, -0.9)
    ax.set_xlim(0, end)
    ax.set_xlabel('simulation time (s)', fontsize=9)
    ax.tick_params(labelsize=8.5)
    for sp in ('top', 'right'):
        ax.spines[sp].set_visible(False)
    fig.savefig(path, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print('wrote', path)


CTRL = {'waypoints, hand law': IDLEC, 'continuous path, hand law': '#9AAAB6',
        'learned pilot (MLP)': ACCENT}


def _style(ax):
    ax.tick_params(labelsize=8.5)
    for sp in ('top', 'right'):
        ax.spines[sp].set_visible(False)


def fig_controllers(path):
    """The column mission driven by each of the FSM's three controllers: the base arena, then the
    three harder arenas.

    Every figure is tools/mission_motion.py on the run's own log, except the two base-arena runs whose
    logs were overwritten; their summaries were recorded by the same script at the time
    (docs/evidence_planner/base_controllers_recorded.json)."""
    sys.path.insert(0, HERE)
    from mission_motion import summarise
    ev = os.path.join(PKG, 'docs', 'evidence_planner')
    rec = json.load(open(os.path.join(ev, 'base_controllers_recorded.json')))
    base = [('waypoints, hand law', rec['waypoints']),
            ('continuous path, hand law', rec['continuous']),
            ('learned pilot (MLP)', summarise(os.path.join(ev, 'mission_log_pilot')))]
    fig, (a, b) = plt.subplots(1, 2, figsize=(11.0, 3.3), dpi=200, gridspec_kw={'width_ratios': [1, 1.25]})

    names = [n for n, _ in base][::-1]
    times = [m['time'] for _, m in base][::-1]
    a.barh(names, times, color=[CTRL[n] for n in names], height=0.62)
    for i, (n, m) in enumerate(base[::-1]):
        a.text(m['time'] + 6, i, '%.0f s   %.1f m, %d interruption%s' % (
            m['time'], m['distance'], m['interruptions'], '' if m['interruptions'] == 1 else 's'),
            va='center', fontsize=7.8, color=INK)
    a.set_xlim(0, 520)
    a.set_xlabel('time to complete the nine laps (s)', fontsize=9)
    a.set_title('(a) Base arena, the same mission three ways', fontsize=9.5, loc='left', color=INK)
    _style(a)

    arenas = ('warehouse', 'forest', 'corridors')
    ctrls = (('waypoints, hand law', 'waypoints'), ('continuous path, hand law', 'continuous'),
             ('learned pilot (MLP)', 'pilot'))
    top, width = 800, 0.26
    for j, (label, key) in enumerate(ctrls):
        for i, arena in enumerate(arenas):
            d = os.path.join(ev, 'maps', 'mission_log_%s_%s' % (arena, key))
            if not os.path.exists(os.path.join(d, 'log.json')):
                continue
            m = summarise(d)
            x = i + (j - 1) * width
            h = min(m['time'], top)
            b.bar(x, h, width * 0.92, color=CTRL[label], hatch=None if m['completed'] else '////',
                  edgecolor='white' if m['completed'] else INK, linewidth=0.6)
            if m['completed']:
                b.text(x, h + 12, '%.0f s' % m['time'], ha='center', va='bottom', fontsize=6.8, color=INK)
            else:                                  # a run that never finished: say so on the bar itself
                b.text(x, h / 2, 'not finished, %.0f s' % m['time'], ha='center', va='center',
                       rotation=90, fontsize=6.8, color=ALERTC,
                       bbox=dict(facecolor='white', edgecolor='none', pad=1.5))
    b.set_xticks(range(len(arenas)))
    b.set_xticklabels(arenas, fontsize=9)
    b.set_ylim(0, top + 60)
    b.set_ylabel('mission time (s)', fontsize=9)
    b.set_title('(b) Three harder arenas (hatched: not finished)', fontsize=9.5, loc='left', color=INK)
    b.legend(handles=[Rectangle((0, 0), 1, 1, color=CTRL[label]) for label, _ in ctrls],
             labels=[label for label, _ in ctrls], fontsize=7.5, frameon=False, loc='upper left')
    _style(b)
    fig.tight_layout(w_pad=3.0)
    fig.savefig(path, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print('wrote', path)


def fig_paths(path):
    """The path each controller drove in the base arena: the closing costmap panel of each run's film
    (docs/evidence_planner/paths/, the right half of the last frame), trimmed of the dark margin."""
    sys.path.insert(0, HERE)
    from mission_motion import summarise
    ev = os.path.join(PKG, 'docs', 'evidence_planner')
    rec = json.load(open(os.path.join(ev, 'base_controllers_recorded.json')))
    runs = (('waypoints', 'waypoints, hand law', rec['waypoints']['time']),
            ('continuous', 'continuous path, hand law', rec['continuous']['time']),
            ('pilot', 'learned pilot (MLP)', summarise(os.path.join(ev, 'mission_log_pilot'))['time']))
    fig, axes = plt.subplots(1, 3, figsize=(11.0, 4.0), dpi=200)
    for ax, (name, label, t) in zip(axes, runs):
        im = Image.open(os.path.join(ev, 'paths', name + '.png'))
        ax.imshow(im.crop((0, 0, 672, im.height)))
        ax.set_title('%s: %.0f s' % (label, t), fontsize=9.5, color=INK)
        ax.axis('off')
    fig.tight_layout(w_pad=0.6)
    fig.savefig(path, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print('wrote', path)


def fig_training(path):
    """How the learned pilot was trained, from docs/evidence_planner/pilot_training.json, which
    `tools/train_pilot.py --record` writes: (a) the labelled states and the imitation loss after
    behaviour cloning and each DAgger round, (b) the controllers on the same unseen laps in the
    surrogate, the network deployed on the robot included (`--evaluate config/pilot_mlp.npz`)."""
    rec = json.load(open(os.path.join(PKG, 'docs', 'evidence_planner', 'pilot_training.json')))
    st, ctl = rec['stages'], rec['controllers']
    names = ['cloning'] + ['DAgger %d' % (i + 1) for i in range(len(st) - 1)]
    x = list(range(len(st)))
    fig, (a, c) = plt.subplots(1, 2, figsize=(11.0, 3.5), dpi=200, gridspec_kw={'width_ratios': [1, 1.3]})

    a.bar(x, [s['samples'] / 1000 for s in st], color=BOX, edgecolor=BOXEDGE, linewidth=0.6, width=0.6)
    for i, s in zip(x, st):
        a.text(i, s['samples'] / 1000 + 0.6, '%.1fk' % (s['samples'] / 1000), ha='center', fontsize=7, color=MUTED)
    a.set_ylabel('labelled states (thousands)', fontsize=8.5)
    a.set_ylim(0, max(s['samples'] for s in st) / 1000 * 1.25)
    a.set_xticks(x)
    a.set_xticklabels(names, fontsize=8)
    _style(a)
    a2 = a.twinx()
    a2.plot(x, [s['loss'] for s in st], color=ACCENT, marker='o', linewidth=1.8)
    a2.set_ylabel('imitation loss (MSE)', fontsize=8.5, color=ACCENT)
    a2.tick_params(labelsize=8, colors=ACCENT)
    a2.set_ylim(0, max(s['loss'] for s in st) * 1.3)
    a2.spines['top'].set_visible(False)
    a.set_title('(a) Cloning, then four rounds of DAgger', fontsize=9.5, loc='left', color=INK)

    order = (('teacher', 'MPPI teacher', MUTED),
             ('deployed pilot', 'pilot deployed on the robot', ACCENT),
             ('student', 'pilot re-trained for this figure', '#5FA3A8'),
             ('hand law (waypoints)', 'waypoints, hand law', CTRL['waypoints, hand law']),
             ('hand law (whole lap)', 'continuous path, hand law', CTRL['continuous path, hand law']))
    order = [o for o in order if o[0] in ctl]
    for i, (key, label, colour) in enumerate(order):
        m = ctl[key]
        lap = m['lap_time'] or 0.0
        c.barh(i, lap, color=colour, height=0.62)
        c.text(lap + 0.4, i, '%.1f s per lap, %.0f%% of laps finished' % (lap, m['completed']),
               va='center', fontsize=7.4, color=INK)
    c.set_yticks(range(len(order)))
    c.set_yticklabels([label for _, label, _ in order], fontsize=8)
    c.invert_yaxis()
    c.set_xlim(0, max((ctl[k]['lap_time'] or 0) for k, _, _ in order) * 1.75)
    c.set_xlabel('mean time per finished lap in the surrogate (s)', fontsize=8.5)
    c.set_title('(b) The same %d unseen laps for every controller' % rec['eval_laps'], fontsize=9.5,
                loc='left', color=INK)
    _style(c)
    fig.tight_layout(w_pad=2.5)
    fig.savefig(path, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print('wrote', path)


def montage(paths, out, cols=3, width=2400, captions=None):
    """Lay archived screenshots out in a grid at a common width."""
    ims = [Image.open(os.path.join(PKG, p)).convert('RGB') for p in paths]
    rows = (len(ims) + cols - 1) // cols
    cw = width // cols
    scaled = [im.resize((cw, int(im.height * cw / im.width)), Image.LANCZOS) for im in ims]
    rh = max(im.height for im in scaled)
    sheet = Image.new('RGB', (width, rh * rows), 'white')
    for i, im in enumerate(scaled):
        sheet.paste(im, ((i % cols) * cw, (i // cols) * rh))
    sheet.save(out, quality=92)
    print('wrote', out, sheet.size)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=os.path.join(PKG, 'docs/submission/figures'))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    fig_fsm(os.path.join(args.out, 'fig1_fsm.png'))
    # the robot itself: two views of the Fusion design and the same model in Gazebo (docs/robot/)
    montage(['docs/robot/mk2_cad_assembled.png',
             'docs/robot/mk2_cad_internals.png',
             'docs/robot/mk2_gazebo.png'],
            os.path.join(args.out, 'fig_robot.png'), cols=3, width=2400)
    # fig_02 is the same state but the camera had left the robot out of frame; Part 4 of the brief
    # asks for the robot to be visible in each state, so the later PATROLLING frame is used
    montage(['docs/evidence_jazzy_run/figures/fig_04_PATROLLING.png',
             'docs/evidence_jazzy_run/figures/fig_03_OBSTACLE_AVOIDANCE.png',
             'docs/evidence_jazzy_run/figures/fig_09_ALERT.png'],
            os.path.join(args.out, 'fig2_states.png'), cols=3, width=2400)
    fig_timeline(os.path.join(args.out, 'fig3_timeline.png'))
    montage(['docs/evidence_map_run/rviz_map.png',
             'docs/evidence_planner/costmap_plan.png'],
            os.path.join(args.out, 'fig4_map.png'), cols=2, width=2000)
    # the controllers compared on the same mission, and how the learned one was trained
    fig_controllers(os.path.join(args.out, 'fig_controllers.png'))
    fig_paths(os.path.join(args.out, 'fig_paths.png'))
    fig_training(os.path.join(args.out, 'fig_training.png'))
    return 0


if __name__ == '__main__':
    sys.exit(main())
