"""Unit tests for the imitation-learned pilot. No ROS, no Gazebo, no trained weights needed."""
import math
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from sentinel_patrol.pilot import features, mlp, model, mppi   # noqa: E402

LIM = model.Limits()


def a_lap(th_in=0.0, th_out=math.pi):
    return model.Reference(model.ring_path(np.array([0.0, 0.0]), 0.55, th_in, th_out))


# ---------------------------------------------------------------- plant
def test_speed_limit_is_on_the_speed_not_on_each_axis():
    """A diagonal command must not buy 1.41 * v_max."""
    st = model.make_state()
    for _ in range(60):
        st = model.step(st, np.array([LIM.v_max, LIM.v_max, 0.0]), LIM)
    # a PI loop may overshoot a little on the way; what must hold is that it settles at the limit
    assert math.hypot(st[model.VX], st[model.VY]) <= LIM.v_max * 1.02


def test_acceleration_is_limited():
    st = model.make_state()
    nxt = model.step(st, np.array([LIM.v_max, 0.0, LIM.w_max]), LIM)
    assert nxt[model.VX] <= LIM.a_max * LIM.dt + 1e-12
    assert nxt[model.W] <= LIM.alpha_max * LIM.dt + 1e-12


def test_driving_forward_moves_along_the_heading():
    st = model.make_state(th=math.pi / 2)
    for _ in range(20):
        st = model.step(st, np.array([LIM.v_max, 0.0, 0.0]), LIM)
    assert st[model.Y] > 0.2 and abs(st[model.X]) < 1e-9


# ---------------------------------------------------------------- reference
def test_progress_along_a_closed_ring_does_not_jump_to_the_finish():
    """Regression: a lap starts and ends at nearly the same point, so a nearest-sample search over
    the whole path reports the robot as finished on its first control step."""
    ref = a_lap(th_in=0.0, th_out=0.2)
    ref.reset()
    ref.advance(ref.path[0])
    assert ref.pos < 0.1, 'the start of the lap must not project onto its end'
    assert ref.length > 3.0


def test_progress_only_moves_forwards():
    ref = a_lap()
    ref.reset()
    for p in ref.path[:40]:
        ref.advance(p)
    ahead = ref.pos
    ref.advance(ref.path[0])            # shoved back to the start, a metre off the window
    assert ref.pos == pytest.approx(ahead), 'progress must not run backwards, or leap forwards'
    for p in ref.path[40:60]:           # and it keeps going once the robot is back on the ring
        ref.advance(p)
    assert ref.pos > ahead


def test_a_route_handed_over_mid_lap_is_located_not_walked_up_to():
    """Regression: rebuilding the reference when the mission's path replaced the planner's left the
    marker pinned near the lap start, so every lookahead point was behind the robot and the policy
    stalled for the rest of the lap."""
    ref = a_lap()
    half = ref.path[len(ref.path) // 2]
    ref.locate(half)
    assert ref.pos == pytest.approx(ref.length / 2, abs=0.15)

    walked = a_lap()
    walked.reset()
    walked.advance(half)
    assert walked.pos < 0.4, 'advance is meant to creep; this is why locate exists'


def test_locate_prefers_the_arc_the_robot_has_not_driven():
    """On a closed ring one point is two arc lengths; the earlier one is still ahead of the robot."""
    ref = a_lap(th_in=0.0, th_out=0.1)          # sweep just over a full turn
    ref.locate(ref.path[0])
    assert ref.pos < 0.1


# ---------------------------------------------------------------- observation
def test_observation_shape_and_range():
    ref = a_lap()
    ref.reset()
    st = model.make_state(0.55, 0.0, math.pi / 2, 0.1)
    obs = features.observation(st, ref, LIM)
    assert obs.shape == (features.OBS_DIM,)
    assert np.all(np.abs(obs) <= 1.5), 'features are meant to be normalised'


def test_sector_ranges_see_a_column_ahead():
    """Facing the centre column from 0.55 m, the forward sector must report about 0.40 m."""
    st = model.make_state(0.55, 0.0, math.pi)                # at +x, looking back at the origin
    rng = features.sector_ranges(st)
    ahead = rng[features.N_SECTORS // 2]                      # the sector centred on the heading
    assert ahead == pytest.approx(0.55 - model.COLUMN_RADIUS, abs=0.05)


def test_observation_is_the_same_whether_ranges_are_given_or_derived():
    ref = a_lap()
    ref.reset()
    st = model.make_state(0.55, 0.0, 1.0, 0.05)
    derived = features.observation(st, ref, LIM)
    passed = features.observation(st, ref, LIM, ranges=features.sector_ranges(st))
    assert np.allclose(derived, passed), 'the robot must see what training saw'


# ---------------------------------------------------------------- student
def test_mlp_output_is_bounded_and_survives_a_round_trip(tmp_path):
    net = mlp.MLP((features.OBS_DIM, 16, 16, 3), seed=1)
    x = np.random.default_rng(0).normal(0, 3.0, (32, features.OBS_DIM))
    out = net(x)
    assert out.shape == (32, 3) and np.all(np.abs(out) <= 1.0)

    path = str(tmp_path / 'net.npz')
    net.save(path)
    again = mlp.MLP.load(path)
    assert np.allclose(net(x), again(x))


def test_the_mlp_pilot_writes_a_trace_instead_of_dying(tmp_path, monkeypatch):
    """Regression: the trace handle was once initialised only in a subclass, so with PILOT_TRACE
    set - as the recorder always sets it - the MLP pilot raised on its first step and the FSM
    process died, and the mission ran on the hand law under a pilot label."""
    from sentinel_patrol.pilot import runtime
    net = mlp.MLP((features.OBS_DIM, 16, 16, 3), seed=1)
    weights = str(tmp_path / 'net.npz'); net.save(weights)
    trace = str(tmp_path / 'trace.csv')
    monkeypatch.setenv(runtime.PilotAdapter.TRACE_ENV, trace)
    a = runtime.PilotAdapter(weights)

    class F:
        t_pose, pose_x, pose_y, yaw = 1.0, 0.0, 0.0, 0.0

    a.set_velocity(0.1, 0.0, 0.0, 1.0)
    a.set_scan(np.full(12, 3.0), 1.0)
    assert a(F(), [(0.0, 0.0), (0.5, 0.0), (1.0, 0.0), (1.5, 0.0)]) is not None
    assert os.path.exists(trace) and len(open(trace).read().splitlines()) == 2, 'header plus one recorded step'


def test_mlp_can_learn_a_linear_map():
    rng = np.random.default_rng(0)
    x = rng.normal(0, 0.5, (600, 4))
    y = np.tanh(x @ np.array([[0.8, 0.0, -0.3], [0.0, 0.5, 0.2], [0.1, 0.1, 0.1], [0.0, -0.4, 0.0]]))
    net = mlp.MLP((4, 32, 32, 3), seed=0)
    before, _ = net.loss_and_grads(x, y)
    after = net.fit(x, y, epochs=200, lr=5e-3)
    assert after < before / 5.0, f'training did not converge ({before:.4f} -> {after:.4f})'


# ---------------------------------------------------------------- teacher
def test_teacher_drives_forwards_along_the_ring():
    """A cheap teacher, a few steps: it must make progress and keep off the column."""
    ref = a_lap()
    ref.reset()
    st = model.make_state(0.55, 0.0, math.pi / 2)
    ref.advance(st[:2])
    teach = mppi.MppiTeacher(LIM, horizon=10, samples=96, seed=0)
    for _ in range(40):
        st = model.step(st, teach(st, ref), LIM)
        ref.advance(st[:2])
        assert model.clearance(st[:2]) > 0.176, 'the teacher drove into the column'
    assert ref.pos > 0.3, f'the teacher made no progress (pos={ref.pos:.2f})'


# ---------------------------------------------------------------- the hook into the FSM
def _navigating_fsm(pilot, us=2.0):
    """An FSM in NAVIGATING with a long route ahead of it and a pilot installed."""
    from sentinel_patrol.fsm_core import NAVIGATING, FsmParams, PatrolFSM, SensorFrame
    fsm = PatrolFSM(FsmParams(), pilot=pilot)
    fsm.on_goal(4.0, 0.0, 0.0)
    fsm.on_path([(x * 0.1, 0.0) for x in range(41)])
    frame = SensorFrame(us_front=us, ir_left=0.8, ir_right=0.8,
                        t_us_front=0.0, t_ir=0.0, t_imu=0.0,
                        pose_x=0.0, pose_y=0.0, yaw=0.0, t_pose=0.0)
    cmd, tr = fsm.step(frame, 0.0)
    assert fsm.state == NAVIGATING
    return fsm, frame


def test_an_installed_pilot_provides_the_navigating_command():
    seen = {}

    def pilot(f, path):
        seen['path'] = len(path)
        return (0.21, -0.05, 0.11)

    fsm, frame = _navigating_fsm(pilot)
    cmd, tr = fsm.step(frame, 0.1)
    assert tr is None
    assert (cmd.vx, cmd.vy, cmd.wz) == (0.21, -0.05, 0.11)
    assert seen['path'] == 41, 'the pilot is given the whole route'


def test_a_pilot_that_hands_back_falls_through_to_the_hand_written_law():
    fsm, frame = _navigating_fsm(lambda f, path: None)
    cmd, tr = fsm.step(frame, 0.1)
    assert tr is None and cmd.vx > 0.0, 'the carrot law should have driven'


def test_a_pilot_cannot_drive_through_an_obstacle():
    """The safety transitions are decided before the pilot is consulted, so a policy asking for full
    speed into something still ends up stopped. This is the whole reason the hook sits where it does."""
    from sentinel_patrol.fsm_core import OBSTACLE_AVOIDANCE
    fsm, frame = _navigating_fsm(lambda f, path: (0.3, 0.0, 0.0), us=0.20)
    tr = None
    for k in range(1, 8):
        cmd, tr = fsm.step(frame, 0.1 * k)
        if tr is not None:
            break
    assert tr is not None and tr.dst == OBSTACLE_AVOIDANCE
    assert cmd.is_zero(), 'the robot must stop, whatever the policy asked for'


def test_a_pilot_cannot_suppress_a_corner_sensor():
    from sentinel_patrol.fsm_core import BACKING_OFF
    fsm, frame = _navigating_fsm(lambda f, path: (0.3, 0.0, 0.0))
    frame.ir_left = 0.05
    cmd, tr = fsm.step(frame, 0.1)
    assert tr is not None and tr.dst == BACKING_OFF and cmd.is_zero()


def test_the_episode_budget_is_read_at_call_time_not_bound_as_a_default():
    """A tool shortens every rollout in the process by setting train_pilot.STEPS, so the module
    attribute must not be captured in rollout's signature.
    """
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'tools'))
    import train_pilot as tp

    rng = np.random.default_rng(0)
    ref = tp.random_lap(rng)
    st0 = tp.start_state(ref, rng)
    stand_still = lambda st: np.zeros(3)                                  # noqa: E731

    was = tp.STEPS
    try:
        tp.STEPS = 25
        assert len(tp.rollout(stand_still, ref, st0.copy())['xy']) == 25
        tp.STEPS = 7
        assert len(tp.rollout(stand_still, ref, st0.copy())['xy']) == 7
        # an explicit argument still wins over the module setting
        assert len(tp.rollout(stand_still, ref, st0.copy(), steps=11)['xy']) == 11
    finally:
        tp.STEPS = was
