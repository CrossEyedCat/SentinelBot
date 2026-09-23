#!/usr/bin/env python3
"""Compose report figures from a run_demo.sh output directory.

    python3 tools/make_figures.py <demo_dir> [<figures_dir>]

For every shot_NN_<STATE>.png it renders a panel: the Gazebo frame on top and, below it, the
real ROS2 logger lines from transitions.log up to that state change plus the /patrol_state
value. Nothing is synthesised: the text is copied from the logs of the same run.
Requires Pillow (pip install pillow).
"""
import csv
import os
import re
import sys

from PIL import Image, ImageDraw, ImageFont

IDLE_STATE = 'IDLE'
STATE_COLOUR = {
    'IDLE': (100, 116, 139), 'PATROLLING': (15, 118, 110), 'OBSTACLE_AVOIDANCE': (180, 83, 9),
    'BACKING_OFF': (109, 40, 217), 'ALERT': (185, 28, 28),
}


def load_font(size: int):
    for name in ('DejaVuSansMono.ttf', 'consola.ttf', 'Consolas.ttf', 'cour.ttf'):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def main() -> None:
    demo_dir = sys.argv[1]
    fig_dir = sys.argv[2] if len(sys.argv) > 2 else os.path.join(demo_dir, 'figures')
    os.makedirs(fig_dir, exist_ok=True)

    with open(os.path.join(demo_dir, 'transitions.log'), encoding='utf-8') as fh:
        transitions = [ln.rstrip() for ln in fh if ln.strip()]
    with open(os.path.join(demo_dir, 'timeline.csv'), encoding='utf-8') as fh:
        timeline = list(csv.DictReader(fh))

    # One "[FSM] A -> B" line per state change, in order, each paired with the "ALERT reason" line that
    # follows an ALERT transition. Frame seq N (from the file name) is the N-th state seen by
    # state_watcher: seq 1 is the initial IDLE (no transition), seq N >= 2 is produced by the
    # (N-1)-th transition, i.e. fsm[N-2]. Indexing by sequence number (not by scanning for "-> STATE")
    # keeps every frame attached to its own transition even when frames are missing or a state
    # name recurs (the old forward scan matched the final "ALERT -> IDLE" line for the first IDLE
    # frame and then consumed the whole log).
    fsm = []
    for k, ln in enumerate(transitions):
        if '[FSM]' in ln:
            dst = re.search(r'-> (\w+) \|', ln)
            extra = transitions[k + 1] if k + 1 < len(transitions) and 'ALERT reason' in transitions[k + 1] else None
            fsm.append((dst.group(1) if dst else '?', ln, extra))

    shots = sorted(f for f in os.listdir(demo_dir) if re.match(r'shot_\d+_.*\.png$', f))
    parsed = [(int(m.group(1)), m.group(2)) for m in (re.match(r'shot_(\d+)_(.*)\.png$', s) for s in shots)]

    # Frame seq N is the N-th state the watcher saw. Whether that is transition N-1 or N-2 depends on
    # whether it managed to capture the initial IDLE before the FSM left it, so pick the offset that
    # makes the most frames agree with the log rather than assuming either.
    def matches(base):
        n = 0
        for seq, state in parsed:
            k = seq - base
            n += (k < 0 and state == IDLE_STATE) or (0 <= k < len(fsm) and fsm[k][0] == state)
        return n
    base = max((2, 1), key=matches)
    if matches(base) < len(parsed):
        print(f'note: {len(parsed) - matches(base)} of {len(parsed)} frames do not line up with the log')

    font = load_font(15)
    font_b = load_font(20)
    for shot in shots:
        m = re.match(r'shot_(\d+)_(.*)\.png$', shot)
        seq, state = int(m.group(1)), m.group(2)
        sim_t = timeline[seq - 1]['sim_time'] if 0 <= seq - 1 < len(timeline) else '?'
        k = seq - base
        if 0 <= k < len(fsm) and fsm[k][0] == state:
            pass                                      # frame and log agree: this is its transition
        elif k < 0 and state == IDLE_STATE:
            k = None                                  # genuine first frame: IDLE before any transition
        else:
            # a state change the 10 Hz watcher missed: take the first transition into this state at or
            # after the expected position, and say so instead of guessing silently
            k = next((j for j in range(max(0, k), len(fsm)) if fsm[j][0] == state), None)
            print(f'note: {shot}: no transition at the expected index; using',
                  'no line' if k is None else fsm[k][1][-70:])
        if k is None:
            lines = ['(initial state: node started, waiting for /patrol_cmd start)'] if state == IDLE_STATE \
                else ['(no matching transition line in transitions.log)']
        else:
            lines = [p[1] for p in fsm[max(0, k - 3):k + 1]]
            if fsm[k][2]:
                lines.append(fsm[k][2])

        img = Image.open(os.path.join(demo_dir, shot)).convert('RGB')
        w = 1100
        img = img.resize((w, int(img.height * w / img.width)))
        pad, line_h = 14, 22
        panel_h = pad * 3 + 30 + line_h * (len(lines) + 1)
        canvas = Image.new('RGB', (w, img.height + panel_h), (24, 27, 31))
        canvas.paste(img, (0, 0))
        d = ImageDraw.Draw(canvas)
        y = img.height + pad
        colour = STATE_COLOUR.get(state, (200, 200, 200))
        d.rectangle([pad, y, pad + 14, y + 22], fill=colour)
        d.text((pad + 24, y), f'/patrol_state: {state}    sim time {sim_t} s    frame {seq:02d}',
               font=font_b, fill=(235, 235, 235))
        y += 30 + pad // 2
        d.text((pad, y), 'ros2 launch sentinel_patrol patrol.launch.py   (logger output)', font=font, fill=(150, 160, 170))
        y += line_h
        for ln in lines:
            ln = re.sub(r'^\[(INFO|ERROR)\] \[(\d+)\.\d+\]', lambda mm: f'[{mm.group(1)}] [{mm.group(2)}]', ln)
            d.text((pad, y), ln[:150], font=font, fill=(255, 120, 120) if 'ALERT' in ln else (220, 220, 220))
            y += line_h
        out = os.path.join(fig_dir, f'fig_{seq:02d}_{state}.png')
        canvas.save(out)
        print('wrote', out)


if __name__ == '__main__':
    main()
