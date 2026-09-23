"""The right-hand panel of the FSM demonstration video, one frame per camera frame.

    python3 tools/fsm_demo_panel.py <demo_dir> <cam.mp4.stamps> <panel.mp4> <fps>

Everything drawn comes from <demo_dir>/demo.json, which tools/fsm_demo.py logged from the same run
on the simulation clock: the state the FSM published, the ranges and IMU values it read, its own
transition lines from /rosout, and the scenario caption. Nothing is interpolated or synthesised;
each frame shows the last value logged at or before that frame's simulation time.
"""
import bisect
import json
import re
import subprocess
import sys
import textwrap

from PIL import Image, ImageDraw, ImageFont

W, H = 720, 720
BG, FG, MUTED, LINE = (17, 20, 23), (230, 233, 228), (152, 162, 171), (58, 64, 70)
COLOUR = {'IDLE': (110, 118, 126), 'PATROLLING': (46, 139, 87), 'NAVIGATING': (52, 110, 200),
          'OBSTACLE_AVOIDANCE': (214, 150, 30), 'BACKING_OFF': (222, 110, 40), 'ALERT': (200, 50, 50)}
GRID = [['IDLE', 'PATROLLING'], ['NAVIGATING', 'OBSTACLE_AVOIDANCE'], ['BACKING_OFF', 'ALERT']]
DV = '/usr/share/fonts/truetype/dejavu/'


def font(name, size):
    return ImageFont.truetype(DV + name, size)


F_TITLE, F_STATE, F_BODY, F_SMALL = font('DejaVuSans-Bold.ttf', 25), font('DejaVuSans-Bold.ttf', 17), \
    font('DejaVuSans.ttf', 17), font('DejaVuSans.ttf', 14)
F_CAP, F_MONO = font('DejaVuSans-Bold.ttf', 19), font('DejaVuSansMono.ttf', 13)


def last(series, t):
    """The last entry of a time-sorted [[t, ...], ...] list at or before t."""
    i = bisect.bisect_right([row[0] for row in series], t)
    return series[i - 1] if i else None


def bar(d, x, y, w, value, lo, hi, marks, fmt, label):
    d.text((x, y), label, font=F_BODY, fill=FG)
    txt = '-' if value is None else fmt % value
    d.text((x + w, y), txt, font=F_BODY, fill=FG, anchor='ra')
    yb = y + 26
    d.rectangle([x, yb, x + w, yb + 8], outline=LINE, fill=(28, 32, 36))
    if value is not None:
        f = max(0.0, min(1.0, (value - lo) / (hi - lo)))
        d.rectangle([x, yb, x + int(f * w), yb + 8], fill=(120, 170, 210))
    for m, text, anchor in marks:          # anchor 'ra' ends the label at its tick, 'la' starts it there
        xm = x + int((m - lo) / (hi - lo) * w)
        d.line([xm, yb - 3, xm, yb + 11], fill=(230, 80, 80), width=2)
        d.text((xm + (-4 if anchor == 'ra' else 4 if anchor == 'la' else 0), yb + 12), text, font=F_SMALL,
               fill=MUTED, anchor=anchor)


def frame(log, t, lines_fsm):
    img = Image.new('RGB', (W, H), BG)
    d = ImageDraw.Draw(img)
    d.text((22, 16), 'Patrol FSM, live', font=F_TITLE, fill=FG)
    d.text((W - 22, 22), 'simulation time %.1f s' % t, font=F_SMALL, fill=MUTED, anchor='ra')

    st = last(log['states'], t)
    state = st[1] if st else None
    al = last(log['alerts'], t)
    reason = re.search(r'reason=(\w+)', al[1]).group(1) if al and re.search(r'reason=(\w+)', al[1]) else ''
    bw, bh, x0, y0 = 330, 46, 22, 60
    for r, row in enumerate(GRID):
        for c, name in enumerate(row):
            x, y = x0 + c * (bw + 16), y0 + r * (bh + 10)
            label = 'ALERT: %s' % reason if name == 'ALERT' and name == state and reason else name
            if name == state:
                d.rounded_rectangle([x, y, x + bw, y + bh], 8, fill=COLOUR[name])
                d.text((x + bw / 2, y + bh / 2), label, font=F_STATE, fill=(255, 255, 255), anchor='mm')
            else:
                d.rounded_rectangle([x, y, x + bw, y + bh], 8, outline=LINE, width=2)
                d.text((x + bw / 2, y + bh / 2), label, font=F_STATE, fill=MUTED, anchor='mm')

    # the current scenario step, and the one before it: two steps can land within a second
    i = bisect.bisect_right([row[0] for row in log['captions']], t)
    y = 232
    d.text((22, y), 'Scenario', font=F_SMALL, fill=MUTED)
    if i:
        for k, ln in enumerate(textwrap.wrap(log['captions'][i - 1][1], 56)[:2]):
            d.text((22, y + 18 + k * 24), ln, font=F_CAP, fill=FG)
    if i > 1:
        prev = log['captions'][i - 2][1]
        d.text((22, y + 68), 'before: ' + (prev if len(prev) < 78 else prev[:75] + '...'), font=F_SMALL, fill=MUTED)

    s = last(log['sensors'], t)
    us, irl, irr, acc, tilt = (s[1:] if s else [None] * 5)
    y = 322
    bar(d, 22, y, 676, us, 0.0, 2.0, [(0.5, 'stop 0.50', 'ra'), (0.7, 'clear 0.70', 'la')], '%.2f m',
        'Front ultrasonic')
    irmin = None if irl is None and irr is None else min(v for v in (irl, irr) if v is not None)
    bar(d, 22, y + 62, 676, irmin, 0.0, 0.8, [(0.12, 'back off 0.12', 'ra'), (0.18, 'release 0.18', 'la')],
        '%.2f m', 'Closer infrared corner (left %s, right %s)'
        % tuple('-' if v is None else '%.2f' % v for v in (irl, irr)))
    bar(d, 22, y + 124, 676, acc, 0.0, 12.0, [(2.0, 'test 2', 'ma'), (8.0, 'collision 8', 'ma')], '%.1f m/s2',
        'Horizontal acceleration (IMU)')
    bar(d, 22, y + 186, 676, tilt, 0.0, 90.0, [(30.0, 'tipping 30', 'ma')], '%.0f deg', 'Tilt (IMU orientation)')

    y = 574
    d.line([22, y - 8, W - 22, y - 8], fill=LINE, width=1)
    d.text((22, y), "The FSM's own log (/rosout)", font=F_SMALL, fill=MUTED)
    i = bisect.bisect_right([row[0] for row in lines_fsm], t)
    for k, row in enumerate(lines_fsm[max(0, i - 6):i]):
        txt = '%6.1f  %s' % (row[0], row[1].replace('[FSM] ', ''))
        d.text((22, y + 20 + k * 19), txt[:92], font=F_MONO, fill=FG if k == min(i, 6) - 1 else MUTED)
    return img


def main():
    demo_dir, stamps, out, fps = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
    log = json.load(open(demo_dir + '/demo.json'))
    for k in ('states', 'captions', 'sensors', 'fsm_lines', 'alerts'):
        log[k].sort(key=lambda r: r[0])
    lines_fsm = [r for r in log['fsm_lines'] if '->' in r[1] or 'ALERT reason' in r[1]]
    times = [float(x) for x in open(stamps).read().split()]
    ff = subprocess.Popen(['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24',
                           '-s', '%dx%d' % (W, H), '-r', str(fps), '-i', '-', '-c:v', 'libx264', '-preset',
                           'veryfast', '-crf', '20', '-pix_fmt', 'yuv420p', out], stdin=subprocess.PIPE)
    for t in times:
        ff.stdin.write(frame(log, t, lines_fsm).tobytes())
    ff.stdin.close()
    ff.wait()
    print('%s: %d frames' % (out, len(times)))


if __name__ == '__main__':
    main()
