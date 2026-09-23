"""Unit tests for the ROS-free FSM core. Run with:  pytest test/  (no ROS needed)."""
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from sentinel_patrol.fsm_core import (  # noqa: E402
    ALERT, BACKING_OFF, IDLE, NAVIGATING, OBSTACLE_AVOIDANCE, PATROLLING,
    FsmParams, PatrolFSM, SensorFrame,
)

DT = 0.1  # 10 Hz tick


def fresh_frame(t, us=2.0, ir_l=0.8, ir_r=0.8, **kw):
    return SensorFrame(us_front=us, ir_left=ir_l, ir_right=ir_r,
                       t_us_front=t, t_ir=t, t_imu=t, **kw)


def posed_frame(t, x=0.0, y=0.0, yaw=0.0, **kw):
    """A frame that also carries the robot pose, as the odometry callback fills it in."""
    return fresh_frame(t, pose_x=x, pose_y=y, yaw=yaw, t_pose=t, **kw)


def run_until_patrolling(fsm, t=0.0):
    fsm.on_command('start')
    cmd, tr = fsm.step(fresh_frame(t), t)
    assert tr is not None and tr.dst == PATROLLING
    return t + DT


# ---------------------------------------------------------------- IDLE
def test_idle_waits_for_start_and_fresh_sensors():
    fsm = PatrolFSM()
    cmd, tr = fsm.step(fresh_frame(0.0), 0.0)
    assert fsm.state == IDLE and tr is None and cmd.is_zero()
    fsm.on_command('start')
    stale = SensorFrame(us_front=2.0, t_us_front=-5.0, t_imu=0.0)   # ultrasonic 5 s old
    cmd, tr = fsm.step(stale, 0.0)
    assert fsm.state == IDLE, 'must not start on stale sensors'
    cmd, tr = fsm.step(fresh_frame(0.1), 0.1)
    assert tr.src == IDLE and tr.dst == PATROLLING and tr.sensor == 'operator'


def test_auto_start_parameter():
    fsm = PatrolFSM(FsmParams(auto_start=True))
    _, tr = fsm.step(fresh_frame(0.0), 0.0)
    assert tr is not None and tr.dst == PATROLLING


# ---------------------------------------------------------------- PATROLLING
def test_patrolling_drives_forward_and_debounces_obstacle():
    p = FsmParams()
    fsm = PatrolFSM(p)
    t = run_until_patrolling(fsm)
    cmd, _ = fsm.step(fresh_frame(t), t)
    assert cmd.vx == pytest.approx(p.v_patrol) and cmd.wz == 0.0
    # a single noisy reading below d_stop must NOT switch state
    cmd, tr = fsm.step(fresh_frame(t, us=0.3), t)
    assert tr is None and fsm.state == PATROLLING
    cmd, tr = fsm.step(fresh_frame(t, us=2.0), t)     # counter resets
    assert tr is None
    # three consecutive readings do
    for _ in range(2):
        _, tr = fsm.step(fresh_frame(t, us=0.45), t)
        assert tr is None
    cmd, tr = fsm.step(fresh_frame(t, us=0.45), t)
    assert tr.src == PATROLLING and tr.dst == OBSTACLE_AVOIDANCE
    assert tr.sensor == 'ultrasonic.front' and tr.value == pytest.approx(0.45)
    assert cmd.is_zero(), 'robot must stop on the transition tick'


def test_operator_stop_returns_to_idle():
    fsm = PatrolFSM()
    t = run_until_patrolling(fsm)
    fsm.on_command('stop')
    cmd, tr = fsm.step(fresh_frame(t), t)
    assert tr.dst == IDLE and cmd.is_zero()


# ---------------------------------------------------------------- operator command safety
def test_stop_is_honoured_during_a_manoeuvre():
    """Regression: stop was only checked in PATROLLING, so the robot finished the manoeuvre first."""
    fsm = PatrolFSM()
    t = run_until_patrolling(fsm)
    enter_avoidance(fsm, t)
    fsm.on_command('stop')
    cmd, tr = fsm.step(fresh_frame(t, us=0.4), t)
    assert tr is not None and tr.dst == IDLE and cmd.is_zero()


def test_stale_start_does_not_restart_after_stop():
    """Regression: 'start' set the flag in any state and nothing cleared it on the way into IDLE."""
    fsm = PatrolFSM()
    t = run_until_patrolling(fsm)
    fsm.on_command('start')          # redundant start while already patrolling
    fsm.on_command('stop')
    _, tr = fsm.step(fresh_frame(t), t)
    assert tr.dst == IDLE
    for k in range(1, 20):           # must stay stopped without a new operator command
        cmd, tr = fsm.step(fresh_frame(t + k * DT), t + k * DT)
        assert tr is None and fsm.state == IDLE and cmd.is_zero()


def test_stale_start_does_not_restart_after_alert_reset():
    """A start pressed while the robot lay in ALERT must not launch it the moment it is reset."""
    fsm = PatrolFSM()
    t = run_until_patrolling(fsm)
    fsm.on_imu(0.2, 80.0, t)         # tipping
    assert fsm.state == ALERT
    fsm.on_command('start')          # ignored in ALERT, but used to stay latched
    fsm.on_command('reset')
    _, tr = fsm.step(fresh_frame(t), t)
    assert tr.dst == IDLE
    cmd, tr = fsm.step(fresh_frame(t + DT), t + DT)
    assert tr is None and fsm.state == IDLE and cmd.is_zero()
    fsm.on_command('start')          # only a fresh start, given in IDLE, may resume the patrol
    _, tr = fsm.step(fresh_frame(t + 2 * DT), t + 2 * DT)
    assert tr.dst == PATROLLING


# ---------------------------------------------------------------- sensor loss
def test_silent_ultrasonic_raises_alert_while_moving():
    """Regression: freshness was checked only in IDLE, so a dead topic left the robot driving blind."""
    p = FsmParams()
    fsm = PatrolFSM(p)
    t = run_until_patrolling(fsm)
    stamp = t                                  # last time the ultrasonic published anything
    t += p.sensor_timeout / 2                  # still fresh
    cmd, tr = fsm.step(SensorFrame(us_front=2.0, ir_left=0.8, ir_right=0.8,
                                   t_us_front=stamp, t_ir=stamp, t_imu=t), t)
    assert tr is None and cmd.vx == pytest.approx(p.v_patrol)
    t += p.sensor_timeout                      # now stale
    cmd, tr = fsm.step(SensorFrame(us_front=2.0, ir_left=0.8, ir_right=0.8,
                                   t_us_front=stamp, t_ir=stamp, t_imu=t), t)
    assert tr is not None and tr.dst == ALERT and tr.reason == 'SENSOR_LOST'
    assert cmd.is_zero() and fsm.last_alert.sensor == 'sensor.age_s'


def test_sensor_loss_is_checked_in_every_moving_state():
    p = FsmParams()
    for enter in (lambda fsm, t: enter_avoidance(fsm, t),
                  lambda fsm, t: fsm.step(fresh_frame(t, us=0.3, ir_l=0.05, ir_r=0.5), t)):
        fsm = PatrolFSM(p)
        t = run_until_patrolling(fsm)
        enter_avoidance(fsm, t)
        enter(fsm, t)
        assert fsm.state in (OBSTACLE_AVOIDANCE, BACKING_OFF)
        t2 = t + p.sensor_timeout + DT
        cmd, tr = fsm.step(SensorFrame(us_front=0.3, ir_left=0.5, ir_right=0.5,
                                       t_us_front=t, t_ir=t, t_imu=t2), t2)
        assert tr is not None and tr.dst == ALERT and tr.reason == 'SENSOR_LOST'


# ---------------------------------------------------------------- OBSTACLE_AVOIDANCE
def enter_avoidance(fsm, t, ir_l=0.8, ir_r=0.8):
    for _ in range(fsm.p.n_confirm):
        _, tr = fsm.step(fresh_frame(t, us=0.4, ir_l=ir_l, ir_r=ir_r), t)
    assert fsm.state == OBSTACLE_AVOIDANCE
    return t


def test_turns_towards_freer_side():
    fsm = PatrolFSM()
    t = run_until_patrolling(fsm)
    enter_avoidance(fsm, t, ir_l=0.2, ir_r=0.6)     # right side is freer
    cmd, _ = fsm.step(fresh_frame(t, us=0.4, ir_l=0.2, ir_r=0.6), t)
    assert cmd.wz == pytest.approx(-fsm.p.w_turn) and cmd.vx == 0.0
    fsm2 = PatrolFSM()
    t = run_until_patrolling(fsm2)
    enter_avoidance(fsm2, t, ir_l=0.6, ir_r=0.2)    # left side is freer
    cmd, _ = fsm2.step(fresh_frame(t, us=0.4, ir_l=0.6, ir_r=0.2), t)
    assert cmd.wz == pytest.approx(+fsm2.p.w_turn)


def test_returns_to_patrol_only_after_hysteresis_and_hold_time():
    p = FsmParams()
    fsm = PatrolFSM(p)
    t = run_until_patrolling(fsm)
    enter_avoidance(fsm, t)
    # 0.6 m is above d_stop but below d_clear: still avoiding (hysteresis)
    for _ in range(10):
        t += DT
        _, tr = fsm.step(fresh_frame(t, us=0.6), t)
        assert tr is None and fsm.state == OBSTACLE_AVOIDANCE
    # clear for less than t_clear: still avoiding
    t += DT
    _, tr = fsm.step(fresh_frame(t, us=1.5), t)
    assert tr is None
    # clear for >= t_clear: patrol resumes
    for _ in range(int(p.t_clear / DT) + 1):
        t += DT
        _, tr = fsm.step(fresh_frame(t, us=1.5), t)
        if tr is not None:
            break
    assert tr is not None and tr.dst == PATROLLING and tr.sensor == 'ultrasonic.front'


def test_ir_critical_triggers_backing_off_and_returns():
    p = FsmParams()
    fsm = PatrolFSM(p)
    t = run_until_patrolling(fsm)
    enter_avoidance(fsm, t)
    _, tr = fsm.step(fresh_frame(t, us=0.3, ir_l=0.10, ir_r=0.5), t)
    assert tr.dst == BACKING_OFF and tr.sensor == 'ir.min' and tr.value == pytest.approx(0.10)
    cmd, _ = fsm.step(fresh_frame(t, us=0.3, ir_l=0.10, ir_r=0.5), t)
    assert cmd.vx == pytest.approx(p.v_back)
    # still reversing before t_back even if IR is clear again
    t2 = t + p.t_back / 2
    _, tr = fsm.step(fresh_frame(t2, us=0.3, ir_l=0.3, ir_r=0.5), t2)
    assert tr is None and fsm.state == BACKING_OFF
    # after t_back but IR only just above ir_critical (inside the hysteresis band): keep reversing
    t3 = t + p.t_back + DT
    _, tr = fsm.step(fresh_frame(t3, us=0.3, ir_l=0.14, ir_r=0.5), t3)
    assert tr is None and fsm.state == BACKING_OFF, 'exit needs ir_back_exit, not ir_critical'
    # IR above ir_back_exit: back to avoidance
    t4 = t + p.t_back + 2 * DT
    _, tr = fsm.step(fresh_frame(t4, us=0.3, ir_l=0.3, ir_r=0.5), t4)
    assert tr.dst == OBSTACLE_AVOIDANCE and tr.sensor == 'timer.back'


def test_avoid_backoff_limit_cycle_raises_stuck_alert():
    """Regression from the Mk II Gazebo run: OBSTACLE_AVOIDANCE <-> BACKING_OFF hopping every ~1 s
    against a pillar for 20 s never tripped the 10 s stuck timer, because every state entry reset it.
    The timer must count from the moment PATROLLING was left."""
    p = FsmParams()
    fsm = PatrolFSM(p)
    t = run_until_patrolling(fsm)
    enter_avoidance(fsm, t)
    hops = 0
    while t < 30.0:
        t += DT
        if fsm.state == OBSTACLE_AVOIDANCE:
            _, tr = fsm.step(fresh_frame(t, us=0.3, ir_l=0.11, ir_r=0.5), t)       # IR critical -> back off
        else:
            _, tr = fsm.step(fresh_frame(t, us=0.3, ir_l=0.30, ir_r=0.5), t)       # clears while reversing
        if tr is not None:
            hops += 1
            if tr.dst == ALERT:
                break
    assert fsm.state == ALERT and fsm.last_alert.reason == 'STUCK'
    assert t <= p.t_avoid_max + 2.0, f'stuck alert came too late (t={t:.1f}s after {hops} hops)'
    assert hops >= 4, 'the scenario should have produced several hops before the alert'


def test_backing_off_hard_time_limit():
    p = FsmParams()
    fsm = PatrolFSM(p)
    t = run_until_patrolling(fsm)
    enter_avoidance(fsm, t)
    fsm.step(fresh_frame(t, us=0.3, ir_l=0.05, ir_r=0.05), t)
    assert fsm.state == BACKING_OFF
    t4 = t + p.t_back_max + DT      # IR still critical, but the hard limit wins
    _, tr = fsm.step(fresh_frame(t4, us=0.3, ir_l=0.05, ir_r=0.05), t4)
    assert tr.dst == OBSTACLE_AVOIDANCE


def test_stuck_timeout_raises_alert():
    p = FsmParams()
    fsm = PatrolFSM(p)
    t = run_until_patrolling(fsm)
    enter_avoidance(fsm, t)
    t5 = t + p.t_avoid_max + DT
    _, tr = fsm.step(fresh_frame(t5, us=0.3), t5)
    assert tr.dst == ALERT and tr.reason == 'STUCK' and fsm.last_alert.reason == 'STUCK'


def test_timers_use_absolute_time():
    """Regression: sim time never starts at 0. Entering a state must record the real entry time,
    otherwise t_avoid_max fires immediately (seen in Gazebo: STUCK with value 36 s after 0.1 s)."""
    p = FsmParams()
    t0 = 5000.0
    fsm = PatrolFSM(p, t0=t0)
    fsm.on_command('start')
    _, tr = fsm.step(fresh_frame(t0), t0)
    assert tr.dst == PATROLLING and fsm.entered_at == t0
    t = t0 + 30.0
    for _ in range(p.n_confirm):
        _, tr = fsm.step(fresh_frame(t, us=0.4), t)
    assert tr is not None and tr.dst == OBSTACLE_AVOIDANCE and fsm.entered_at == t
    _, tr = fsm.step(fresh_frame(t + DT, us=0.4), t + DT)
    assert tr is None and fsm.state == OBSTACLE_AVOIDANCE, 'stuck timer must count from state entry'
    _, tr = fsm.step(fresh_frame(t + p.t_avoid_max + DT, us=0.4), t + p.t_avoid_max + DT)
    assert tr is not None and tr.reason == 'STUCK'


# ---------------------------------------------------------------- ALERT via IMU
def test_imu_spike_alert_is_debounced_and_sticky():
    p = FsmParams()
    fsm = PatrolFSM(p)
    t = run_until_patrolling(fsm)
    assert fsm.on_imu(12.0, 0.0, t) is None, 'first sample alone must not trigger'
    assert fsm.on_imu(0.5, 0.0, t) is None      # counter resets
    assert fsm.on_imu(12.0, 0.0, t) is None
    tr = fsm.on_imu(12.0, 0.0, t)
    assert tr is not None and tr.dst == ALERT and tr.reason == 'COLLISION'
    assert fsm.last_alert.as_text().startswith('ALERT reason=COLLISION sensor=imu.a_xy')
    # no automatic exit: clear sensors for a long time keep it in ALERT
    for k in range(200):
        cmd, tr2 = fsm.step(fresh_frame(t + k * DT), t + k * DT)
        assert tr2 is None and fsm.state == ALERT and cmd.is_zero()
    # ALERT ignores further IMU
    assert fsm.on_imu(20.0, 0.0, t) is None
    # only the operator reset leaves ALERT
    fsm.on_command('reset')
    _, tr3 = fsm.step(fresh_frame(t), t)
    assert tr3.src == ALERT and tr3.dst == IDLE


def test_tilt_alert():
    fsm = PatrolFSM()
    t = run_until_patrolling(fsm)
    tr = fsm.on_imu(0.2, 35.0, t)
    assert tr is not None and tr.reason == 'TIPPING' and tr.sensor == 'imu.tilt_deg'


def test_imu_ignored_in_idle():
    fsm = PatrolFSM()
    assert fsm.on_imu(50.0, 80.0, 0.0) is None and fsm.state == IDLE


# ---------------------------------------------------------------- NAVIGATING (clicked map goal)
def test_clicked_goal_leaves_idle_without_a_start_command():
    """Clicking a point on the map is itself the command to go there."""
    fsm = PatrolFSM()
    _, tr = fsm.step(posed_frame(0.0), 0.0)
    assert fsm.state == IDLE and tr is None
    fsm.on_goal(2.0, 0.0, 0.0)
    _, tr = fsm.step(posed_frame(DT), DT)
    assert tr is not None and tr.dst == NAVIGATING and tr.reason == 'goal'
    cmd, _ = fsm.step(posed_frame(2 * DT), 2 * DT)
    assert cmd.vx > 0 and abs(cmd.wz) < 1e-6, 'goal dead ahead: drive straight at it'


def test_navigating_turns_towards_a_goal_behind_the_robot():
    p = FsmParams()
    fsm = PatrolFSM(p)
    fsm.on_goal(-2.0, 0.0, 0.0)                       # directly behind
    fsm.step(posed_frame(0.0), 0.0)
    assert fsm.state == NAVIGATING
    cmd, _ = fsm.step(posed_frame(DT), DT)
    assert cmd.vx == 0.0 and abs(cmd.wz) == pytest.approx(p.w_turn), 'turn before driving'


def test_goal_reached_returns_to_idle_and_clears_the_goal():
    p = FsmParams()
    fsm = PatrolFSM(p)
    fsm.on_goal(1.0, 0.0, 0.0)
    fsm.step(posed_frame(0.0), 0.0)
    assert fsm.state == NAVIGATING
    inside = p.goal_tolerance / 2
    cmd, tr = fsm.step(posed_frame(DT, x=1.0 - inside), DT)
    assert tr.dst == IDLE and tr.reason == 'GOAL_REACHED' and cmd.is_zero()
    assert fsm.goal() is None
    for k in range(2, 12):                            # and it stays there
        cmd, tr = fsm.step(posed_frame(k * DT, x=1.0 - inside), k * DT)
        assert tr is None and fsm.state == IDLE and cmd.is_zero()


def test_navigating_yields_to_an_obstacle_and_resumes_the_goal():
    """The avoidance behaviour is shared with PATROLLING; afterwards the goal must be resumed."""
    p = FsmParams()
    fsm = PatrolFSM(p)
    fsm.on_goal(3.0, 0.0, 0.0)
    fsm.step(posed_frame(0.0), 0.0)
    assert fsm.state == NAVIGATING
    t = DT
    for _ in range(p.n_confirm):                      # obstacle on the way to the goal
        _, tr = fsm.step(posed_frame(t, us=0.4), t)
    assert tr is not None and tr.dst == OBSTACLE_AVOIDANCE and fsm.goal() == (3.0, 0.0)
    for k in range(1, int(p.t_clear / DT) + 3):       # path clears again
        t += DT
        _, tr = fsm.step(posed_frame(t, us=1.5), t)
        if tr is not None:
            break
    assert tr is not None and tr.dst == NAVIGATING, 'must resume the goal, not the free patrol'


def test_alert_drops_the_goal_so_a_reset_does_not_resume_driving():
    fsm = PatrolFSM()
    fsm.on_goal(3.0, 0.0, 0.0)
    fsm.step(posed_frame(0.0), 0.0)
    assert fsm.state == NAVIGATING
    fsm.on_imu(0.2, 80.0, DT)                         # tipped over on the way
    assert fsm.state == ALERT and fsm.goal() is None
    fsm.on_goal(3.0, 0.0, DT)                         # a click while in ALERT is refused
    assert fsm.goal() is None
    fsm.on_command('reset')
    _, tr = fsm.step(posed_frame(2 * DT), 2 * DT)
    assert tr.dst == IDLE
    cmd, tr = fsm.step(posed_frame(3 * DT), 3 * DT)
    assert tr is None and fsm.state == IDLE and cmd.is_zero()


def test_goal_that_cannot_be_reached_times_out():
    p = FsmParams()
    fsm = PatrolFSM(p)
    fsm.on_goal(5.0, 0.0, 0.0)
    fsm.step(posed_frame(0.0), 0.0)
    assert fsm.state == NAVIGATING
    t = p.t_goal_max + DT                             # never got any closer
    cmd, tr = fsm.step(posed_frame(t), t)
    assert tr.dst == IDLE and tr.reason == 'GOAL_UNREACHABLE' and cmd.is_zero()
    assert fsm.goal() is None


def test_operator_stop_abandons_the_goal():
    fsm = PatrolFSM()
    fsm.on_goal(3.0, 0.0, 0.0)
    fsm.step(posed_frame(0.0), 0.0)
    assert fsm.state == NAVIGATING
    fsm.on_command('stop')
    _, tr = fsm.step(posed_frame(DT), DT)
    assert tr.dst == IDLE and fsm.goal() is None


def test_holonomic_navigating_trims_the_sideways_error():
    p = FsmParams(holonomic=True)
    fsm = PatrolFSM(p)
    fsm.on_goal(2.0, 0.3, 0.0)                        # ahead, slightly to the left
    fsm.step(posed_frame(0.0), 0.0)
    cmd, _ = fsm.step(posed_frame(DT), DT)
    assert cmd.vx > 0 and cmd.vy > 0, 'mecanum platform corrects sideways while driving on'
    plain = PatrolFSM(FsmParams(holonomic=False))
    plain.on_goal(2.0, 0.3, 0.0)
    plain.step(posed_frame(0.0), 0.0)
    cmd, _ = plain.step(posed_frame(DT), DT)
    assert cmd.vy == 0.0, 'a differential drive can only steer'


def test_navigating_drives_on_before_re_aiming_after_an_avoidance():
    """Regression: re-aiming at the goal the instant the path cleared put the same obstacle straight
    back in front of the robot, giving a NAVIGATING <-> OBSTACLE_AVOIDANCE livelock in simulation."""
    p = FsmParams()
    fsm = PatrolFSM(p)
    fsm.on_goal(0.0, 3.0, 0.0)                        # goal to the LEFT of the robot's heading
    fsm.step(posed_frame(0.0), 0.0)
    t = DT
    for _ in range(p.n_confirm):                      # obstacle -> avoidance
        _, tr = fsm.step(posed_frame(t, us=0.4), t)
    assert tr.dst == OBSTACLE_AVOIDANCE
    for _ in range(int(p.t_clear / DT) + 3):          # path clears -> back to NAVIGATING
        t += DT
        _, tr = fsm.step(posed_frame(t, us=2.6), t)
        if tr is not None:
            break
    assert tr.dst == NAVIGATING
    # for t_escape the robot drives straight on instead of turning back towards the goal
    cmd, _ = fsm.step(posed_frame(t + DT, us=2.6), t + DT)
    assert cmd.vx > 0 and cmd.wz == 0.0, 'must clear the obstacle before re-aiming'
    late = t + p.t_escape + DT
    cmd, _ = fsm.step(posed_frame(late, us=2.6), late)
    assert abs(cmd.wz) > 0, 'once past the obstacle it aims at the goal again'


def test_goal_is_abandoned_after_too_many_avoidance_episodes():
    p = FsmParams()
    fsm = PatrolFSM(p)
    fsm.on_goal(0.0, 5.0, 0.0)
    fsm.step(posed_frame(0.0), 0.0)
    t, tr = DT, None
    for episode in range(p.n_avoid_max + 2):
        for _ in range(p.n_confirm):                  # blocked again
            t += DT
            _, tr = fsm.step(posed_frame(t, us=0.4), t)
            if tr is not None:
                break
        if tr is not None and tr.dst == IDLE:
            break
        assert tr.dst == OBSTACLE_AVOIDANCE
        for _ in range(int(p.t_clear / DT) + 3):      # and clears again
            t += DT
            _, tr = fsm.step(posed_frame(t, us=2.6), t)
            if tr is not None:
                break
        fsm._escape_until = None                      # skip the escape leg to reach the cap quickly
    assert tr is not None and tr.dst == IDLE and tr.reason == 'GOAL_UNREACHABLE'
    assert fsm.goal() is None


def test_final_approach_reaches_a_goal_placed_near_a_wall():
    """The patrol stop distance is 0.50 m but the tolerance is 0.10 m, so without a reduced final
    clearance the robot would halt half a metre short of any goal next to a wall."""
    p = FsmParams()
    fsm = PatrolFSM(p)
    fsm.on_goal(1.0, 0.0, 0.0)
    fsm.step(posed_frame(0.0), 0.0)
    assert fsm.state == NAVIGATING
    # a wall 0.30 m ahead: outside the final approach that is an obstacle, inside it is the goal
    x, t = 0.55, DT                                   # 0.45 m to go, inside d_final
    for _ in range(p.n_confirm + 1):
        cmd, tr = fsm.step(posed_frame(t, x=x, us=0.30), t)
        t += DT
    assert tr is None and fsm.state == NAVIGATING and cmd.vx > 0, 'must keep closing on the goal'
    cmd, tr = fsm.step(posed_frame(t, x=1.0 - p.goal_tolerance / 2, us=0.30), t)
    assert tr is not None and tr.dst == IDLE and tr.reason == 'GOAL_REACHED'


def test_far_from_the_goal_the_normal_stop_distance_still_applies():
    p = FsmParams()
    fsm = PatrolFSM(p)
    fsm.on_goal(5.0, 0.0, 0.0)                        # far away, so no final approach
    fsm.step(posed_frame(0.0), 0.0)
    t, tr = DT, None
    for _ in range(p.n_confirm):
        _, tr = fsm.step(posed_frame(t, us=0.30), t)  # 0.30 m is inside d_stop
        t += DT
    assert tr is not None and tr.dst == OBSTACLE_AVOIDANCE


def test_holonomic_final_approach_slides_straight_onto_the_point():
    p = FsmParams(holonomic=True)
    fsm = PatrolFSM(p)
    fsm.on_goal(0.30, 0.30, 0.0)                      # inside d_final, 45 degrees off the nose
    fsm.step(posed_frame(0.0), 0.0)
    assert fsm.state == NAVIGATING
    cmd, _ = fsm.step(posed_frame(DT), DT)
    assert cmd.vx > 0 and cmd.vy > 0 and cmd.wz == 0.0, 'translate onto the goal without turning'
    speed = (cmd.vx ** 2 + cmd.vy ** 2) ** 0.5
    assert speed <= p.v_final + 1e-9, 'and do it slowly'


def test_escape_leg_is_cancelled_close_to_the_goal():
    """The escape leg exists to get past an obstacle; near the goal it would drive straight past it."""
    p = FsmParams()
    fsm = PatrolFSM(p)
    fsm.on_goal(0.40, 0.0, 0.0)
    fsm.step(posed_frame(0.0), 0.0)
    fsm._escape_until = 100.0                         # as if an avoidance had just handed back control
    cmd, _ = fsm.step(posed_frame(DT), DT)
    assert fsm._escape_until is None and cmd.vx > 0


# ---------------------------------------------------------------- following a planned route
def test_navigating_steers_at_the_planned_route_not_at_the_goal():
    """The point of a planner is that the robot goes around things. It must therefore steer at the
    next waypoint, while distance and arrival are still measured to the real goal."""
    p = FsmParams()
    fsm = PatrolFSM(p)
    fsm.on_goal(3.0, 0.0, 0.0)                       # goal straight ahead
    fsm.step(posed_frame(0.0), 0.0)
    assert fsm.state == NAVIGATING
    cmd, _ = fsm.step(posed_frame(DT), DT)
    assert abs(cmd.wz) < 1e-6, 'with no route it drives straight at the goal'
    fsm.on_path([(1.0, 1.5), (2.0, 1.5), (3.0, 0.0)])  # a detour to the left around something
    cmd, _ = fsm.step(posed_frame(2 * DT), 2 * DT)
    assert cmd.wz > 0, 'now it must turn towards the first waypoint, not the goal'


def test_the_steering_point_advances_along_the_route():
    p = FsmParams()
    fsm = PatrolFSM(p)
    fsm.on_goal(4.0, 0.0, 0.0)
    fsm.step(posed_frame(0.0), 0.0)
    fsm.on_path([(1.0, 0.0), (2.0, 0.0), (3.0, 0.0), (4.0, 0.0)])
    fsm.step(posed_frame(DT), DT)
    assert fsm._path_i == 0
    fsm.step(posed_frame(2 * DT, x=1.9), 2 * DT)      # driven past the first two waypoints
    assert fsm._path_i == 2, 'waypoints inside the lookahead are dropped'


def test_the_route_is_ignored_on_the_final_approach():
    """Near the goal the robot settles on the point itself, not on a waypoint beside it."""
    p = FsmParams(holonomic=True)
    fsm = PatrolFSM(p)
    fsm.on_goal(0.30, 0.0, 0.0)
    fsm.step(posed_frame(0.0), 0.0)
    fsm.on_path([(0.30, 1.0)])                        # a stale waypoint well off to the side
    cmd, _ = fsm.step(posed_frame(DT), DT)
    assert cmd.vx > 0 and abs(cmd.vy) < 1e-6, 'drive at the goal, not at the stale waypoint'


def test_a_new_goal_and_a_reset_both_drop_the_route():
    fsm = PatrolFSM()
    fsm.on_goal(3.0, 0.0, 0.0)
    fsm.step(posed_frame(0.0), 0.0)
    fsm.on_path([(1.0, 1.0), (3.0, 0.0)])
    assert fsm.path()
    fsm.on_goal(2.0, -1.0, DT)
    assert fsm.path() == [], 'the old route led to the old goal'
    fsm.on_path([(1.0, -1.0), (2.0, -1.0)])
    fsm.on_command('stop')
    fsm.step(posed_frame(2 * DT), 2 * DT)
    assert fsm.state == IDLE and fsm.path() == [] and fsm.goal() is None


def test_planner_can_abandon_an_unreachable_goal_at_once():
    """When the planner reports that no route exists there is no point driving at it for 120 s."""
    fsm = PatrolFSM()
    fsm.on_goal(3.0, 0.0, 0.0)
    fsm.step(posed_frame(0.0), 0.0)
    assert fsm.state == NAVIGATING
    tr = fsm.abandon_goal(DT)
    assert tr is not None and tr.dst == IDLE and tr.reason == 'GOAL_UNREACHABLE'
    assert fsm.goal() is None and fsm.path() == []
    cmd, tr = fsm.step(posed_frame(2 * DT), 2 * DT)
    assert tr is None and fsm.state == IDLE and cmd.is_zero()


# ---------------------------------------------------------------- wedged, but still commanding
def test_no_progress_while_driving_is_noticed():
    """Regression: a planned route took the robot between two pillars, it wedged, and sat perfectly
    still in NAVIGATING for 50 s because nothing checked whether the commands had any effect."""
    p = FsmParams()
    fsm = PatrolFSM(p)
    fsm.on_goal(4.0, 0.0, 0.0)
    fsm.step(posed_frame(0.0), 0.0)
    assert fsm.state == NAVIGATING
    t, tr = DT, None
    for _ in range(int(p.t_stuck / DT) + 4):          # commanded to drive, pose never changes
        cmd, tr = fsm.step(posed_frame(t, x=1.0, y=0.0), t)
        if tr is not None:
            break
        t += DT
    assert tr is not None and tr.dst == OBSTACLE_AVOIDANCE and tr.reason == 'NO_PROGRESS'


def test_moving_normally_never_trips_the_progress_watchdog():
    p = FsmParams()
    fsm = PatrolFSM(p)
    fsm.on_goal(9.0, 0.0, 0.0)
    fsm.step(posed_frame(0.0), 0.0)
    t, x = DT, 0.0
    for _ in range(int(p.t_stuck / DT) * 3):
        x += p.v_goal * DT                            # creeping along at the commanded speed
        _, tr = fsm.step(posed_frame(t, x=x), t)
        assert tr is None, f'unexpected {tr}'
        t += DT
    assert fsm.state == NAVIGATING


def test_turning_on_the_spot_is_not_mistaken_for_being_wedged():
    """Regression: the progress watchdog only exempted a zero command, so an in-place turn - which
    covers no ground by design - was read as being wedged once it lasted longer than t_stuck. On the
    column tour that fired ten times, each one throwing the robot off its lap into avoidance."""
    p = FsmParams()
    fsm = PatrolFSM(p)
    fsm.on_goal(-4.0, 0.0, 0.0)                       # behind the robot: it must turn before driving
    fsm.step(posed_frame(0.0), 0.0)
    assert fsm.state == NAVIGATING
    t = DT
    for _ in range(int(p.t_stuck / DT) * 3):          # yaw is held fixed, so it keeps turning
        cmd, tr = fsm.step(posed_frame(t, x=0.0, y=0.0), t)
        assert abs(cmd.wz) > 0.0 and abs(cmd.vx) < p.v_min_progress, 'expected an in-place turn'
        assert tr is None, f'turning on the spot must not read as no progress ({tr})'
        t += DT
    assert fsm.state == NAVIGATING


def test_the_carrot_search_stays_local_on_a_route_that_loops():
    """Regression: a lap is a closed ring, so the sample nearest the robot exists both just ahead of
    it and a whole loop later. The unbounded search took whichever was a hair closer; steering at the
    far one sent the robot across the middle of its own lap and into the column it was circling."""
    import math as _m
    n = 70
    ring = [(0.55 * _m.cos(2 * _m.pi * k / n), 0.55 * _m.sin(2 * _m.pi * k / n)) for k in range(n + 1)]
    p = FsmParams()
    fsm = PatrolFSM(p)
    fsm.on_goal(0.55, 0.0, 0.0)
    fsm.on_path(ring)
    here = ring[3]
    frame = posed_frame(0.0, x=here[0], y=here[1], yaw=1.6)
    carrot = fsm._carrot(frame)
    assert carrot is not None
    # the carrot must be a lookahead ahead of index 3, not most of a loop away
    idx = ring.index(carrot)
    assert 3 < idx < 20, f'carrot latched onto index {idx} of {n}'


def test_a_corner_sensor_stops_the_robot_while_navigating():
    """The front ultrasonic cannot see a contact on the flank; the IR ring must be checked here too."""
    p = FsmParams()
    fsm = PatrolFSM(p)
    fsm.on_goal(3.0, 0.0, 0.0)
    fsm.step(posed_frame(0.0), 0.0)
    assert fsm.state == NAVIGATING
    cmd, tr = fsm.step(posed_frame(DT, ir_l=0.09, ir_r=0.8), DT)   # left corner against something
    assert tr is not None and tr.dst == BACKING_OFF and tr.sensor == 'ir.min'
    assert cmd.is_zero()


def test_navigating_without_a_pose_stops():
    fsm = PatrolFSM()
    fsm.on_goal(2.0, 0.0, 0.0)
    fsm.step(posed_frame(0.0), 0.0)
    assert fsm.state == NAVIGATING
    cmd, tr = fsm.step(fresh_frame(DT), DT)           # odometry stopped publishing a pose
    assert tr.dst == IDLE and cmd.is_zero()


# ---------------------------------------------------------------- holonomic (Mk II)
def test_holonomic_strafes_when_side_is_open():
    p = FsmParams(holonomic=True)
    fsm = PatrolFSM(p)
    t = run_until_patrolling(fsm)
    for _ in range(p.n_confirm):
        fsm.step(fresh_frame(t, us=0.4, ir_l=0.6, ir_r=0.2, us_left=1.5, us_right=0.3), t)
    assert fsm.state == OBSTACLE_AVOIDANCE
    cmd, _ = fsm.step(fresh_frame(t, us=0.4, ir_l=0.6, ir_r=0.2, us_left=1.5), t)
    assert cmd.vy == pytest.approx(+p.v_strafe) and cmd.wz == 0.0


def test_holonomic_rotates_at_a_flat_wall_even_if_side_is_open():
    """Regression from the Mk II Gazebo run: strafing along the arena wall never clears the front
    ultrasonic. Both corner IRs reading the same distance means a wall: rotate instead."""
    p = FsmParams(holonomic=True)
    fsm = PatrolFSM(p)
    t = run_until_patrolling(fsm)
    for _ in range(p.n_confirm):
        fsm.step(fresh_frame(t, us=0.45, ir_l=0.62, ir_r=0.60, us_left=2.0, us_right=2.0), t)
    assert fsm.state == OBSTACLE_AVOIDANCE
    cmd, _ = fsm.step(fresh_frame(t, us=0.45, ir_l=0.62, ir_r=0.60, us_left=2.0), t)
    assert cmd.vy == 0.0 and abs(cmd.wz) == pytest.approx(p.w_turn)


def test_holonomic_strafe_gives_up_after_t_strafe_max():
    """An obstacle wider than a pillar: after t_strafe_max of sliding sideways with the front still
    blocked, the FSM must switch to rotating (which is what eventually frees the front)."""
    p = FsmParams(holonomic=True)
    fsm = PatrolFSM(p)
    t = run_until_patrolling(fsm)
    for _ in range(p.n_confirm):
        fsm.step(fresh_frame(t, us=0.4, ir_l=0.6, ir_r=0.2, us_left=1.5), t)
    cmd, _ = fsm.step(fresh_frame(t, us=0.4, ir_l=0.6, ir_r=0.2, us_left=1.5), t)
    assert cmd.vy == pytest.approx(p.v_strafe)
    t2 = t + p.t_strafe_max + DT
    cmd, tr = fsm.step(fresh_frame(t2, us=0.4, ir_l=0.6, ir_r=0.2, us_left=1.5), t2)
    assert tr is None and fsm.state == OBSTACLE_AVOIDANCE
    assert cmd.vy == 0.0 and cmd.wz == pytest.approx(+p.w_turn), 'strafe must fall back to rotation'


def test_holonomic_rotates_when_side_is_blocked():
    p = FsmParams(holonomic=True)
    fsm = PatrolFSM(p)
    t = run_until_patrolling(fsm)
    for _ in range(p.n_confirm):
        fsm.step(fresh_frame(t, us=0.4, ir_l=0.6, ir_r=0.2, us_left=0.3), t)
    cmd, _ = fsm.step(fresh_frame(t, us=0.4, ir_l=0.6, ir_r=0.2, us_left=0.3), t)
    assert cmd.vy == 0.0 and cmd.wz == pytest.approx(+p.w_turn)


def test_holonomic_rear_ultrasonic_aborts_backing_off():
    p = FsmParams(holonomic=True)
    fsm = PatrolFSM(p)
    t = run_until_patrolling(fsm)
    enter_avoidance(fsm, t)
    fsm.step(fresh_frame(t, us=0.3, ir_l=0.05, ir_r=0.5), t)
    assert fsm.state == BACKING_OFF
    _, tr = fsm.step(fresh_frame(t + DT, us=0.3, ir_l=0.05, ir_r=0.5, us_rear=0.2), t + DT)
    assert tr.dst == OBSTACLE_AVOIDANCE and tr.sensor == 'ultrasonic.rear'
