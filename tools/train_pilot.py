"""Train the driving policy by imitation: MPPI teacher -> behaviour cloning -> DAgger.

    python3 tools/train_pilot.py                        # full run, writes config/pilot_mlp.npz
    python3 tools/train_pilot.py --quick                # a short run, for checking the plumbing

Why imitation and not reinforcement learning: the teacher already knows how to drive. An MPC with
the right cost solves this problem on the first try; what it cannot do is run in a few microseconds
on the robot, and it needs to be told exactly where the columns are. So the expensive controller is
used offline as a labelling oracle and its behaviour is compressed into a network that reads only
what the robot can measure. That is cheaper than RL by orders of magnitude, and when the policy
does something odd there is a teacher to compare it against, which matters for a module about
explainable behaviour.

Why DAgger and not plain cloning: a cloned policy is only ever shown states the teacher visits. The
first time it drifts off the ring it is off-distribution and has no idea what to do, and the error
compounds. DAgger fixes exactly this by rolling out the *student*, asking the teacher what it
should have done in the states the student actually reached, and adding those to the set.

Three controllers are measured on the same episodes:
  teacher    the MPPI oracle
  student    the network being trained
  hand law   the carrot law from fsm_core, in both the way the mission drives it today (a point
             goal every 0.34 m, each one an arrival) and with the whole lap as one path
"""
import argparse
import json
import math
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from sentinel_patrol.pilot import features, mlp, model, mppi   # noqa: E402

LIM = model.Limits()
STEPS = 420                 # control periods per episode: 42 s, plenty for one lap
GOAL_TOLERANCE = 0.10       # m, what the mission counts as an arrival
WAYPOINT_SPACING = 0.34     # m, the hop length the mission uses today (10 points on a 0.55 m ring)


# ---------------------------------------------------------------------- references
def random_lap(rng) -> model.Reference:
    """A lap around one of the nine columns, entered and left at random bearings."""
    centre = model.COLUMNS[rng.integers(len(model.COLUMNS))]
    th_in = rng.uniform(-np.pi, np.pi)
    th_out = th_in + rng.uniform(-np.pi, np.pi)
    return model.Reference(model.ring_path(centre, 0.55, th_in, th_out))


def start_state(ref: model.Reference, rng, jitter: bool = True) -> np.ndarray:
    """On the ring at the entry point, facing along it, with the errors a real handover has."""
    path = ref.path
    heading = math.atan2(path[1, 1] - path[0, 1], path[1, 0] - path[0, 0])
    st = model.make_state(path[0, 0], path[0, 1], heading)
    if jitter:
        st[model.X] += rng.normal(0.0, 0.06)
        st[model.Y] += rng.normal(0.0, 0.06)
        st[model.TH] = model.wrap(heading + rng.normal(0.0, 0.45))
        st[model.VX] = rng.uniform(0.0, LIM.v_max)
    return st


# ---------------------------------------------------------------------- the hand-written law
class HandLaw:
    """fsm_core's NAVIGATING command, reproduced against the surrogate.

    Two ways of driving the same steering law, because the robot's behaviour is as much about the
    interface as the controller:

    `waypoints=True` is what the mission does today. The lap is chopped into point goals 0.34 m
    apart and each one has to be arrived at, within GOAL_TOLERANCE, before the next is issued. The
    planner returns a straight line for a hop that short, so the carrot is the goal itself - which
    is why this is modelled as steering straight at the goal.

    `waypoints=False` hands the same law the whole lap at once, taking the carrot a lookahead
    further along the reference. Note that fsm_core's own `_carrot` could NOT be used for this: it
    picks the nearest path sample from its current index onwards, and on a closed ring that latches
    onto the far branch of the loop - the end of the lap is a few centimetres from the middle of it.
    The law is fine; its bookkeeping assumes an open path, which is all the planner ever gives it.
    """

    def __init__(self, ref: model.Reference, waypoints: bool, v_goal=0.30, v_final=0.15,
                 d_final=0.20, goal_yaw_align=0.75, w_turn=0.60, k_goal_yaw=1.2, k_goal_lat=0.8,
                 lookahead=0.35):
        self.ref = ref
        self.waypoints = waypoints
        self.v_goal, self.v_final, self.d_final = v_goal, v_final, d_final
        self.align, self.w_turn = goal_yaw_align, w_turn
        self.k_yaw, self.k_lat, self.lookahead = k_goal_yaw, k_goal_lat, lookahead
        if waypoints:
            want = np.arange(WAYPOINT_SPACING, ref.length, WAYPOINT_SPACING)
            self.goals = ref.path[np.searchsorted(ref.s, want)]
        else:
            self.goals = ref.path[-1:]
        self.goal_i = 0
        self.arrivals = 0

    def __call__(self, st) -> np.ndarray:
        if self.waypoints:
            goal = self.goals[min(self.goal_i, len(self.goals) - 1)]
            dist = float(np.hypot(goal[0] - st[model.X], goal[1] - st[model.Y]))
            if dist < GOAL_TOLERANCE and self.goal_i < len(self.goals) - 1:
                self.goal_i += 1
                self.arrivals += 1
                return np.zeros(3)      # the mission's settle: a full stop between waypoints
            target = goal
        else:
            dist = self.ref.length - self.ref.pos            # remaining arc, not a straight line
            want = min(self.ref.pos + self.lookahead, self.ref.length)
            target = self.ref.path[min(int(np.searchsorted(self.ref.s, want)), len(self.ref.path) - 1)]
        final = dist < self.d_final

        sx, sy = target[0] - st[model.X], target[1] - st[model.Y]
        cos, sin = math.cos(st[model.TH]), math.sin(st[model.TH])
        ex, ey = sx * cos + sy * sin, -sx * sin + sy * cos
        bearing = model.wrap(math.atan2(sy, sx) - st[model.TH])

        if final:
            v = max(0.04, min(self.v_final, 0.8 * dist))
            norm = math.hypot(ex, ey) or 1.0
            return np.array([v * ex / norm, v * ey / norm, 0.0])
        if abs(bearing) > self.align:
            return np.array([0.0, 0.0, math.copysign(self.w_turn, bearing)])
        return np.array([max(0.04, min(self.v_goal, 0.8 * dist)),
                         float(np.clip(self.k_lat * ey, -0.15, 0.15)),
                         self.k_yaw * bearing])


# ---------------------------------------------------------------------- rollout and scoring
def rollout(policy, ref: model.Reference, st0, steps=None, record=None, lim=None, delay=0):
    """Run one episode. `record` collects (observation, teacher action) pairs when given.

    `lim` is the plant for this episode (the nominal one by default) and `delay` the number of
    control periods a command takes to reach it - the robot has a bridge and a base controller in
    between, the surrogate by default has neither. A start away from the path's first point is
    located on the path rather than assumed to be at it."""
    # STEPS is read here rather than bound as a default, so a tool can shorten the episode by
    # setting train_pilot.STEPS and have every rollout in the process follow.
    steps = STEPS if steps is None else steps
    lim = lim or LIM
    st = st0.copy()
    ref.reset()
    ref.locate(st[:2])
    queue = [np.zeros(3)] * int(delay)
    log = {'xy': [], 'v': [], 'w': [], 'clear': [], 'collided': False, 'reached': 0.0}
    for _ in range(steps):
        cmd = policy(st)
        if record is not None:
            record(st)
        queue.append(np.asarray(cmd, dtype=float))
        st = model.step(st, queue.pop(0), lim)
        log['xy'].append(st[:2].copy())
        log['v'].append(math.hypot(st[model.VX], st[model.VY]))
        log['w'].append(abs(st[model.W]))
        c = float(model.clearance(st[:2]))
        log['clear'].append(c)
        here = ref.advance(st[:2])
        if c < 0.176:                       # the circumscribed radius: this is a collision
            log['collided'] = True
            break
        if here > ref.length - 0.12:
            break
    log['reached'] = ref.pos / ref.length
    log['xy'] = np.asarray(log['xy'])
    return log


def score(logs) -> dict:
    ok = [l for l in logs if not l['collided']]
    done = [l for l in logs if l['reached'] > 0.95 and not l['collided']]
    # no surviving episode means these numbers do not exist; zeros would read as 'stood still'
    v = np.concatenate([l['v'] for l in ok]) if ok else np.full(1, np.nan)
    w = np.concatenate([l['w'] for l in ok]) if ok else np.full(1, np.nan)
    return {
        'completed': 100.0 * len(done) / max(1, len(logs)),
        'collided': 100.0 * sum(l['collided'] for l in logs) / max(1, len(logs)),
        'lap_time': float(np.mean([len(l['v']) * LIM.dt for l in done])) if done else float('nan'),
        'speed': float(v.mean()),
        'stationary': 100.0 * float((v < 0.03).mean()),
        'yaw_rate': float(w.mean()),
        'min_clear': float(np.min([np.min(l['clear']) for l in logs])),
    }


def _clean(m: dict) -> dict:
    """Scores for JSON: a figure that does not exist is null, not NaN."""
    return {k: (None if isinstance(v, float) and math.isnan(v) else v) for k, v in m.items()}


def report(name: str, m: dict) -> None:
    print('  %-22s complete %5.1f%%  collide %4.1f%%  lap %5.1f s  speed %4.2f m/s  '
          'still %4.1f%%  |w| %4.2f rad/s  clear %.3f m'
          % (name, m['completed'], m['collided'], m['lap_time'], m['speed'],
             m['stationary'], m['yaw_rate'], m['min_clear']))


# ---------------------------------------------------------------------- training
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--quick', action='store_true', help='a short run, for checking the plumbing')
    ap.add_argument('--out', default=None)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--record', default=None, metavar='JSON',
                    help="write the loss and the student's scores after every stage, and the final "
                         'comparison of all four controllers, to this file')
    ap.add_argument('--evaluate', default=None, metavar='NPZ',
                    help='score a saved network on the same unseen laps instead of training; with '
                         '--record, add it to that existing file as "deployed pilot"')
    args = ap.parse_args()

    bc_eps = 12 if args.quick else 60
    dagger_rounds = 1 if args.quick else 4
    dagger_eps = 10 if args.quick else 40
    epochs = 30 if args.quick else 120
    eval_eps = 8 if args.quick else 40
    out = args.out or os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                   '..', 'config', 'pilot_mlp.npz')

    rng = np.random.default_rng(args.seed)
    teacher = mppi.MppiTeacher(LIM, seed=args.seed)
    net = mlp.MLP((features.OBS_DIM, 64, 64, 3), seed=args.seed)
    scale = np.array([LIM.v_max, LIM.v_max, LIM.w_max])

    xs, ys = [], []

    # the unseen laps everything is scored on, drawn from their own generator: scoring the student
    # between stages uses neither the training generator nor the teacher, so it changes no weight
    seeds = np.random.default_rng(args.seed + 999)
    laps = []
    for _ in range(eval_eps):
        ref = random_lap(seeds)
        laps.append((ref, start_state(ref, seeds)))
    stages = []

    if args.evaluate:                        # a network trained earlier, on exactly these laps
        saved = mlp.MLP.load(args.evaluate)
        logs = [rollout(lambda st, _r=ref: saved(features.observation(st, _r, LIM)) * scale, ref, st0.copy())
                for ref, st0 in laps]
        m = score(logs)
        report(os.path.basename(args.evaluate), m)
        if args.record:
            with open(args.record) as fh:
                rec = json.load(fh)
            rec['controllers']['deployed pilot'] = _clean(m)
            with open(args.record, 'w') as fh:
                json.dump(rec, fh, indent=1)
            print('added to %s' % args.record)
        return

    def record_stage(name, loss):
        if args.record:
            logs = [rollout(lambda st, _r=ref: net(features.observation(st, _r, LIM)) * scale, ref, st0.copy())
                    for ref, st0 in laps]
            stages.append(dict(stage=name, samples=len(xs), loss=float(loss), **score(logs)))
            report(name, stages[-1])

    def gather(ref, driver):
        """Run one episode under `driver`, labelling every visited state with the teacher."""
        teacher.reset()
        st0 = start_state(ref, rng)

        def record(st):
            xs.append(features.observation(st, ref, LIM))
            ys.append(np.clip(teacher(st, ref) / scale, -1.0, 1.0))

        return rollout(driver, ref, st0, record=record)

    # --- behaviour cloning: the teacher drives, and is its own label
    t0 = time.time()
    print('behaviour cloning: %d episodes under the teacher' % bc_eps)
    for i in range(bc_eps):
        ref = random_lap(rng)
        teacher.reset()
        st0 = start_state(ref, rng)

        def teacher_driver(st, _r=ref):
            xs.append(features.observation(st, _r, LIM))
            act = teacher(st, _r)
            ys.append(np.clip(act / scale, -1.0, 1.0))
            return act

        rollout(teacher_driver, ref, st0)
    x = np.asarray(xs)
    y = np.asarray(ys)
    print('   %d samples in %.0f s' % (len(x), time.time() - t0))
    loss = net.fit(x, y, epochs=epochs, lr=3e-3, seed=args.seed)
    print('   cloning loss %.5f' % loss)
    record_stage('behaviour cloning', loss)

    # --- DAgger: the student drives, the teacher says what it should have done
    for r in range(dagger_rounds):
        before = len(xs)
        for _ in range(dagger_eps):
            ref = random_lap(rng)

            def student_driver(st, _r=ref):
                return net(features.observation(st, _r, LIM)) * scale

            gather(ref, student_driver)
        x = np.asarray(xs)
        y = np.asarray(ys)
        loss = net.fit(x, y, epochs=epochs, lr=2e-3, seed=args.seed + r + 1)
        print('DAgger round %d: +%d samples (%d total), loss %.5f'
              % (r + 1, len(xs) - before, len(xs), loss))
        record_stage('DAgger round %d' % (r + 1), loss)

    # --- evaluation, all four controllers on the same episodes
    print('\nevaluation over %d unseen laps' % eval_eps)
    results = {}
    for name in ('teacher', 'student', 'hand law (waypoints)', 'hand law (whole lap)'):
        logs = []
        for ref, st0 in laps:
            if name == 'teacher':
                teacher.reset()
                pol = lambda st, _r=ref: teacher(st, _r)
            elif name == 'student':
                pol = lambda st, _r=ref: net(features.observation(st, _r, LIM)) * scale
            else:
                pol = HandLaw(ref, waypoints=name.endswith('(waypoints)'))
            logs.append(rollout(pol, ref, st0.copy()))
        results[name] = score(logs)
        report(name, results[name])

    net.save(out)
    print('\nwrote %s (%d parameters)'
          % (os.path.normpath(out), sum(w.size for w in net.w) + sum(b.size for b in net.b)))
    if args.record:
        with open(args.record, 'w') as fh:
            json.dump({'seed': args.seed, 'eval_laps': eval_eps, 'epochs': epochs,
                       'stages': [_clean(s) for s in stages],
                       'controllers': {k: _clean(v) for k, v in results.items()}}, fh, indent=1)
        print('wrote %s' % args.record)


if __name__ == '__main__':
    main()
