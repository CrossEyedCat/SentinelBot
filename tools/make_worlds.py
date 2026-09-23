"""Three harder arenas for the column-lap mission, generated and checked rather than drawn.

    python3 tools/make_worlds.py            # writes worlds/<name>.sdf and worlds/<name>.map

Each world is a 5 m square with columns to lap around and walls, shelves or crates the planner has
to route around between laps. The base arena (turtlebot3_world) has nine columns on a grid and
nothing in between; these do not:

    warehouse   three shelf rows with 0.9 m aisles; four columns at the aisle mouths
    corridors   two staggered walls make an S-shaped route; four columns, two per side
    forest      six columns at irregular spacing, a wall segment and a crate in the gaps

The numbers are constrained, not chosen freely. The FSM stops at 0.20 m of front range and steers
away at 0.15 m of side range, measured from the sensors on a body 0.176 m in circumscribed radius;
so a ring the robot drives must stay at least CLEAR = 0.42 m from every surface, and a passage the
planner is expected to use must be wider than 2 * (0.176 + 0.05) = 0.45 m plus room to turn. The
checker below enforces the first on every ring point and refuses to write a world that fails it -
the first two drafts of every one of these failed.

The .map file beside each world is what tools/record_column_tour.sh reads for MAP=<name>: the
world file, the spawn pose, the column list in the order they are lapped, the ring radius, and
whether transits go through the planner.
"""
import math
import os

HERE = os.path.dirname(os.path.abspath(__file__))
WORLDS = os.path.join(HERE, '..', 'worlds')
BASE = os.path.join(WORLDS, 'sentinel_world.sdf')

HALF = 2.5            # inner half-size of the arena
WALL_T = 0.10         # wall thickness
H = 0.6               # obstacle height; the LiDAR and the range sensors are below this
COLUMN_R = 0.15       # same as the base arena's columns
BODY_R = 0.176
CLEAR = 0.42          # ring-to-surface minimum: body plus the FSM's stop band plus a little
# The planner's default safety margin (0.05 m beyond the body) was tuned for columns. A planned
# route past a wall end cut it so close that the robot stopped against the end and never got past
# it: the first corridors run aborted there. Set live on the planner for these worlds.
PLANNER_MARGIN = 0.15


def box(name, x, y, sx, sy, yaw=0.0, colour=(0.55, 0.57, 0.60)):
    return dict(kind='box', name=name, x=x, y=y, sx=sx, sy=sy, yaw=yaw, colour=colour)


def column(name, x, y):
    return dict(kind='cyl', name=name, x=x, y=y, r=COLUMN_R, colour=(0.80, 0.30, 0.25))


def arena():
    t = WALL_T
    return [box('wall_n', 0, HALF + t / 2, 2 * HALF + 2 * t, t), box('wall_s', 0, -HALF - t / 2, 2 * HALF + 2 * t, t),
            box('wall_e', HALF + t / 2, 0, t, 2 * HALF), box('wall_w', -HALF - t / 2, 0, t, 2 * HALF)]


MAPS = {
    'warehouse': dict(
        ring=0.55, spawn=(-2.0, 0.0),
        columns=[(-1.2, 1.5), (1.2, 1.5), (1.2, -1.5), (-1.2, -1.5)],
        obstacles=[box('shelf_w', -1.2, 0.0, 0.3, 1.0, colour=(0.45, 0.35, 0.25)),
                   box('shelf_c', 0.0, 0.0, 0.3, 1.0, colour=(0.45, 0.35, 0.25)),
                   box('shelf_e', 1.2, 0.0, 0.3, 1.0, colour=(0.45, 0.35, 0.25))]),
    'corridors': dict(
        ring=0.45, spawn=(-1.6, 0.0),
        columns=[(-1.625, -1.4), (-1.625, 1.4), (1.625, -1.4), (1.625, 1.4)],
        obstacles=[box('wall_a', -0.7, -0.75, WALL_T, 3.5),      # y in (-2.5, 1.0): open at the top
                   box('wall_b', 0.7, 0.75, WALL_T, 3.5)]),      # y in (-1.0, 2.5): open at the bottom
    'forest': dict(
        ring=0.50, spawn=(-2.0, 2.0),
        columns=[(-1.5, 1.4), (0.3, 1.5), (1.55, 0.6), (1.2, -1.4), (-0.4, -0.4), (-1.55, -1.4)],
        obstacles=[box('wall_seg', -0.2, -1.9, 1.0, WALL_T),
                   box('crate', -2.0, 0.0, 0.3, 0.3, colour=(0.45, 0.35, 0.25))]),
}


# ---------------------------------------------------------------- geometry checks
def dist_to_box(px, py, b):
    c, s = math.cos(-b['yaw']), math.sin(-b['yaw'])
    dx, dy = px - b['x'], py - b['y']
    lx, ly = dx * c - dy * s, dx * s + dy * c
    ox = max(abs(lx) - b['sx'] / 2, 0.0)
    oy = max(abs(ly) - b['sy'] / 2, 0.0)
    inside = abs(lx) < b['sx'] / 2 and abs(ly) < b['sy'] / 2
    return -1.0 if inside else math.hypot(ox, oy)


def clearance(px, py, solids):
    d = float('inf')
    for o in solids:
        if o['kind'] == 'box':
            d = min(d, dist_to_box(px, py, o))
        else:
            d = min(d, math.hypot(px - o['x'], py - o['y']) - o['r'])
    return d


def check(name, spec):
    solids = arena() + spec['obstacles'] + [column('c%d' % i, x, y) for i, (x, y) in enumerate(spec['columns'])]
    worst = float('inf')
    for i, (cx, cy) in enumerate(spec['columns']):
        others = [o for o in solids if o.get('name') != 'c%d' % i]
        for k in range(72):
            a = 2 * math.pi * k / 72
            px, py = cx + spec['ring'] * math.cos(a), cy + spec['ring'] * math.sin(a)
            d = clearance(px, py, others)
            if d < worst:
                worst, where = d, (i, px, py)
    sx, sy = spec['spawn']
    ds = clearance(sx, sy, solids)
    ok = worst >= CLEAR and ds >= BODY_R + 0.15
    print('  %-10s ring clearance min %.2f m (column %d at %.2f, %.2f)  spawn clearance %.2f m  %s'
          % (name, worst, where[0] + 1, where[1], where[2], ds, 'ok' if ok else 'FAIL'))
    return ok


# ---------------------------------------------------------------- SDF
def model_xml(o):
    colour = ' '.join('%.2f' % c for c in o['colour'])
    if o['kind'] == 'box':
        geom = '<box><size>%.3f %.3f %.3f</size></box>' % (o['sx'], o['sy'], H)
        pose = '%.3f %.3f %.3f 0 0 %.4f' % (o['x'], o['y'], H / 2, o['yaw'])
    else:
        geom = '<cylinder><radius>%.3f</radius><length>%.3f</length></cylinder>' % (o['r'], H)
        pose = '%.3f %.3f %.3f 0 0 0' % (o['x'], o['y'], H / 2)
    return ('    <model name="%s"><static>true</static><pose>%s</pose>\n'
            '      <link name="link">\n'
            '        <collision name="c"><geometry>%s</geometry></collision>\n'
            '        <visual name="v"><geometry>%s</geometry>'
            '<material><ambient>%s 1</ambient><diffuse>%s 1</diffuse></material></visual>\n'
            '      </link>\n    </model>\n' % (o['name'], pose, geom, geom, colour, colour))


def write_world(name, spec):
    base = open(BASE).read()
    i = base.index('    <model name="turtlebot3_world">')
    j = base.index('</model>', i) + len('</model>\n')
    solids = arena() + spec['obstacles'] + [column('column_%d' % (k + 1), x, y) for k, (x, y) in enumerate(spec['columns'])]
    body = ''.join(model_xml(o) for o in solids)
    out = base[:i] + '    <!-- generated by tools/make_worlds.py: %s -->\n' % name + body + base[j:]
    out = out.replace('<world name="sentinel_world">', '<world name="%s">' % name)
    with open(os.path.join(WORLDS, name + '.sdf'), 'w', newline=chr(10)) as fh:
        fh.write(out)
    flat = ','.join('%.3f' % v for xy in spec['columns'] for v in xy)   # no spaces: the recorder passes it unquoted
    with open(os.path.join(WORLDS, name + '.map'), 'w', newline=chr(10)) as fh:   # bash sources this: no CR
        fh.write('# read by tools/record_column_tour.sh for MAP=%s\n' % name)
        fh.write('WORLD=%s.sdf\nSPAWN_X=%.2f\nSPAWN_Y=%.2f\nRING=%.2f\nTRANSIT=true\nCOLUMNS="[%s]"\nPLANNER_MARGIN=%.2f\n'
                 % (name, spec['spawn'][0], spec['spawn'][1], spec['ring'], flat, PLANNER_MARGIN))


def main() -> None:
    print('checking rings against every surface (minimum %.2f m):' % CLEAR)
    bad = [n for n, s in MAPS.items() if not check(n, s)]
    if bad:
        raise SystemExit('not written: %s' % ', '.join(bad))
    for n, s in MAPS.items():
        write_world(n, s)
        print('  wrote worlds/%s.sdf and .map (%d columns, ring %.2f m)' % (n, len(s['columns']), s['ring']))


if __name__ == '__main__':
    main()
