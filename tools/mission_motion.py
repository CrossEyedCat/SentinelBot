"""How the robot actually spent a recorded mission: driving, turning, both, or standing still.

    python3 tools/mission_motion.py <mission_log dir>        # the directory record_column_tour.sh leaves

Reads mission_logger's log.json. Every figure in the headline comparison table comes from this script
run on the log of the mission it describes, with the same thresholds: a pose interval counts as moving
above 0.03 m/s and as turning above 0.15 rad/s.
"""
import json
import math
import os
import sys

V_MOVING = 0.03      # m/s
W_TURNING = 0.15     # rad/s
LAPS, LAP_LENGTH = 9, 39.6      # the nine-column tour: ideal yaw per metre is 360 * 9 / 39.6


def summarise(logdir: str) -> dict:
    """The figures main() prints, as a dict, for tools that tabulate or plot several missions."""
    log = json.load(open(os.path.join(logdir, "log.json")))
    p = log["poses"]

    drive = turn = both = idle = dist = yaw_total = 0.0
    for a, b in zip(p, p[1:]):
        dt = b[0] - a[0]
        if dt <= 0 or dt > 0.5:
            continue
        ds = math.dist(a[1:3], b[1:3])
        dyaw = abs((b[3] - a[3] + math.pi) % (2 * math.pi) - math.pi)
        v, w = ds / dt, dyaw / dt
        dist += ds
        yaw_total += dyaw
        moving, spinning = v > V_MOVING, w > W_TURNING
        if moving and spinning:
            both += dt
        elif moving:
            drive += dt
        elif spinning:
            turn += dt
        else:
            idle += dt
    total = drive + turn + both + idle

    states = [s for _, s in log.get("states", [])]
    interruptions = sum(1 for a, b in zip(states, states[1:])
                        if b in ("OBSTACLE_AVOIDANCE", "BACKING_OFF") and a not in ("OBSTACLE_AVOIDANCE", "BACKING_OFF"))
    alerts = sum(1 for a, b in zip(states, states[1:]) if b == "ALERT" and a != "ALERT")
    mission = [m for _, m in log.get("mission", [])]
    completed = any("MISSION_COMPLETE" in m for m in mission)

    return dict(time=total, distance=dist, yaw_deg=math.degrees(yaw_total), drive=drive, turn=turn,
                both=both, idle=idle, interruptions=interruptions, alerts=alerts,
                completed=completed, goals=len(log.get("goals", [])))


def main() -> None:
    m = summarise(sys.argv[1] if len(sys.argv) > 1 else "mission_log")
    total, dist, yaw_total = m["time"], m["distance"], math.radians(m["yaw_deg"])
    drive, turn, both, idle = m["drive"], m["turn"], m["both"], m["idle"]
    interruptions, alerts, completed = m["interruptions"], m["alerts"], m["completed"]
    print("mission %.0f s of simulation, %.1f m driven, %.0f deg of yaw turned, %s"
          % (total, dist, math.degrees(yaw_total), "completed" if completed else "NOT completed"))
    print()
    for name, t in (("driving straight", drive), ("turning on the spot", turn),
                    ("driving and turning at once", both), ("stationary", idle)):
        print("  %-28s %6.1f s   %4.1f %%" % (name, t, 100.0 * t / total if total else float("nan")))
    print()
    print("  yaw turned per metre driven:  %.0f deg/m   (a perfect ring tour: %.0f deg/m)"
          % (math.degrees(yaw_total) / dist if dist else float("nan"), 360.0 * LAPS / LAP_LENGTH))
    print("  goals issued: %d   obstacle interruptions: %d   alerts: %d"
          % (m["goals"], interruptions, alerts))


if __name__ == "__main__":
    main()
