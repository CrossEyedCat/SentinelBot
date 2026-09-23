"""Render the costmap panel of the mission video, one frame per camera frame.

    python3 tools/mission_panel.py <logdir> <stamps file> <out.mp4> <fps_out>

Runs entirely offline from mission_logger's output, so the panel can be redrawn without touching
the simulation. The stamps file is the simulation time of every camera frame, so the two videos
have the same frame count and stay locked together when they are stacked side by side.

What the panel shows, on top of the planner's own costmap:
    grey    never seen            white   free and driveable
    orange  near an obstacle      red     closed by the robot's own footprint (cost 99)
    black   obstacle (cost 100)
    green   where the robot has been      blue   the route the planner is following now
    amber   the waypoint being driven to
"""
import glob
import json
import os
import subprocess
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont

SIZE = 720          # panel is square and as tall as the camera frame
HEADER = 48
FONT_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
PLAN_MAX_AGE = 8.0  # s; a route older than this is stale and is not drawn

BG = (24, 27, 31)
INK = (235, 238, 242)
UNKNOWN = (150, 157, 165)
FREE = (255, 255, 255)
NEAR = (250, 224, 186)      # cost just under the footprint band
BLOCKED = (233, 141, 141)   # 99: the robot's own width closes this
LETHAL = (32, 36, 42)       # 100
TRAVELLED = (31, 158, 110)
PLANNED = (47, 128, 237)
GOAL = (217, 119, 6)
ROBOT = (17, 21, 26)


def colour_table() -> np.ndarray:
    """value + 1 -> RGB, so index 0 is the -1 'unknown' cell."""
    table = np.zeros((102, 3), dtype=np.uint8)
    table[0] = UNKNOWN
    for v in range(0, 99):
        f = v / 98.0
        table[v + 1] = [int(FREE[i] + (NEAR[i] - FREE[i]) * f) for i in range(3)]
    table[100] = BLOCKED
    table[101] = LETHAL
    return table


def load(logdir: str):
    with open(os.path.join(logdir, "log.json")) as fh:
        log = json.load(fh)
    files = sorted(glob.glob(os.path.join(logdir, "grid_*.npy")))
    grids = [np.load(f) for f in files]
    # a log cut short mid-write can list more stamps than there are grids on disk
    log["grid_t"] = log["grid_t"][:len(grids)]
    return log, grids


def bounds(log) -> tuple:
    """A square window in odom metres that holds everything the robot did, plus a margin."""
    xs = [p[1] for p in log["poses"]] + [g[1] for g in log["goals"]]
    ys = [p[2] for p in log["poses"]] + [g[2] for g in log["goals"]]
    cx, cy = (min(xs) + max(xs)) / 2.0, (min(ys) + max(ys)) / 2.0
    half = max(max(xs) - min(xs), max(ys) - min(ys)) / 2.0 + 0.75
    half = max(half, 2.6)       # a short mission must not zoom in until the arena is unreadable
    return cx - half, cy - half, 2.0 * half


def at_or_before(items, t: float, i: int) -> int:
    """Advance i while items[i + 1] is still at or before t; the callers walk time forwards."""
    while i + 1 < len(items) and items[i + 1][0] <= t:
        i += 1
    return i


def main() -> None:
    logdir, stamps_path, out, fps = sys.argv[1], sys.argv[2], sys.argv[3], float(sys.argv[4])
    log, grids = load(logdir)
    stamps = [float(s) for s in open(stamps_path) if s.strip()]
    meta, table = log["meta"], colour_table()
    res, ox, oy = meta["resolution"], meta["origin"][0], meta["origin"][1]
    x0, y0, span = bounds(log)
    map_px = SIZE - HEADER

    # the crop, in cells, that the panel shows
    c0, r0 = int((x0 - ox) / res), int((y0 - oy) / res)
    n = max(1, int(span / res))
    c0 = max(0, min(c0, meta["width"] - n))
    r0 = max(0, min(r0, meta["height"] - n))
    scale = map_px / float(n)

    def to_px(x: float, y: float) -> tuple:
        col = (x - ox) / res - c0
        row = (y - oy) / res - r0
        return col * scale, HEADER + (n - row) * scale

    font = ImageFont.truetype(FONT_PATH, 21)
    small = ImageFont.truetype(FONT_PATH, 15)
    ff = subprocess.Popen(
        ["ffmpeg", "-y", "-loglevel", "error",
         "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", "%dx%d" % (SIZE, SIZE), "-framerate", "%.4f" % fps,
         "-i", "-", "-c:v", "libx264", "-preset", "ultrafast", "-crf", "20", "-pix_fmt", "yuv420p", out],
        stdin=subprocess.PIPE,
    )

    gi = pi = gli = mi = si = 0
    grid_t, plans, goals, mission, poses = log["grid_t"], log["plans"], log["goals"], log["mission"], log["poses"]
    grid_items = [[u] for u in grid_t]
    cached_gi, cached_img = -1, None
    trail = []

    for t in stamps:
        gi = at_or_before(grid_items, t, gi)
        pi = at_or_before(plans, t, pi)
        gli = at_or_before(goals, t, gli)
        mi = at_or_before(mission, t, mi)
        si = at_or_before(poses, t, si)

        if gi != cached_gi:
            crop = grids[gi][r0:r0 + n, c0:c0 + n]
            rgb = table[np.clip(crop.astype(np.int16), -1, 100) + 1]
            cached_img = Image.fromarray(np.flipud(rgb), "RGB").resize((map_px, map_px), Image.NEAREST)
            cached_gi = gi

        frame = Image.new("RGB", (SIZE, SIZE), BG)
        frame.paste(cached_img, (0, HEADER))
        d = ImageDraw.Draw(frame)

        while len(trail) < len(poses) and poses[len(trail)][0] <= t:
            p = poses[len(trail)]
            trail.append(to_px(p[1], p[2]))
        if len(trail) > 1:
            d.line(trail, fill=TRAVELLED, width=3)

        if plans and t - plans[pi][0] < PLAN_MAX_AGE and len(plans[pi][1]) > 1:
            d.line([to_px(x, y) for x, y in plans[pi][1]], fill=PLANNED, width=3)

        if goals:
            gx, gy = to_px(goals[gli][1], goals[gli][2])
            d.ellipse([gx - 9, gy - 9, gx + 9, gy + 9], outline=GOAL, width=3)
            d.line([gx - 13, gy, gx + 13, gy], fill=GOAL, width=2)
            d.line([gx, gy - 13, gx, gy + 13], fill=GOAL, width=2)

        if poses:
            _, rx, ry, yaw = poses[si]
            px, py = to_px(rx, ry)
            hx, hy = to_px(rx + 0.28 * np.cos(yaw), ry + 0.28 * np.sin(yaw))
            d.line([px, py, hx, hy], fill=ROBOT, width=4)
            d.ellipse([px - 7, py - 7, px + 7, py + 7], fill=ROBOT)

        text = mission[mi][1] if mission else ""
        d.text((16, 7), text[:52], font=font, fill=INK)
        d.text((16, 30), "grey: never seen   red: closed by the robot footprint   blue: planned route",
               font=small, fill=(150, 157, 165))

        ff.stdin.write(frame.tobytes())

    ff.stdin.close()
    ff.wait()
    print("   panel: %d frames -> %.1f s" % (len(stamps), len(stamps) / fps))


if __name__ == "__main__":
    main()
