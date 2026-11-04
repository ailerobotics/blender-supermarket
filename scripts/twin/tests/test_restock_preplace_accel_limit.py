"""sim-preplace-accel-limit-reject-v1 -- CPU regression for the pre-submission arm-plan acceleration
gate + uniform time reparameterisation (twin.restock_manipulation.traj_limits), wired into
RosMotionPort, and the named-trip / heartbeat-alive path (controlled_stop).

Root cause (warmacc4, carton_level OFF): the collect place pre-place is a cuMotion joint-space plan,
MoveIt-TOTG-timed against joint_limits.yaml (3.0 rad/s^2 * 0.02 = 0.06 rad/s^2), resampled onto a 0.1 s
grid with a sub-0.1 s terminal remainder, then handed to the simulation FollowJointTrajectory executor
(trajectory_host.Plan), which rebuilds every segment as a CUBIC HERMITE from the (position, velocity)
pair and checks its peak against max_speed*(0.5/0.15) = 1.0 rad/s^2. These resampled plans sit right at
that ceiling (the recorded 157-point lift reconstructs to 0.936 rad/s^2 with no margin); the larger
pre-place move crossed it at its tiny terminal-remainder segment and was REJECTED at execution
(host-stack.log 7306: 'interpolated acceleration limit violation' -> HOST_FAULT_ESTOP phase=preplace,
moveit_err=-4), never caught at plan time. The carton_level-ON path (coldwin33) is the explicit leveled
begin_joint_path, sized far under the ceiling, so it passes.

Fixtures are REAL recorded plans (host-stack.log TRAJ_HOST_PLAN dumps): positions only; the executor's
own cubic-Hermite velocity reconstruction is applied in the test (central differences, rest endpoints),
except the leveled path whose begin_joint_path waypoints are rest points (zero velocity, exact).
"""
import json
import pathlib
from types import SimpleNamespace as NS

import numpy as np
import pytest

from twin.restock_manipulation import traj_limits as T
from twin.restock_manipulation.cad_tool import ARM_JOINTS
from twin.restock.trajectory_host import Plan
from twin.restock import controlled_stop

# UR10 joint limits in ARM order [pan, lift, elbow, wrist_1, wrist_2, wrist_3] (mobile-import.urdf).
LIMITS = [(-6.2832, 6.2832), (-6.2832, 6.2832), (0.1, 3.1416),
          (-6.2832, 6.2832), (-6.2832, 6.2832), (-6.2832, 6.2832)]
# The simulation executor's ceiling: DEFAULT_EXEC_MAX_SPEED (0.3) * MAX_ACCEL_PER_SPEED (0.5/0.15) = 1.0.
EXEC_MAX_SPEED = T.DEFAULT_EXEC_MAX_SPEED
EXEC_MAX_ACCEL = EXEC_MAX_SPEED * T.MAX_ACCEL_PER_SPEED
BUDGET_S = 110.0

_FIX = json.loads((pathlib.Path(__file__).parent / 'fixtures'
                   / 'preplace_accel_limit_plans.json').read_text())
NAMES = _FIX['joint_names']  # the plan/log joint order (NOT ARM order)


def _executor_velocities(times, Q):
    """The per-waypoint velocities the executor's cubic-Hermite reconstruction integrates: central
    differences of the sampled positions with both endpoints pinned to rest (TOTG plans start and end at
    rest; the executor's Plan requires the first waypoint at ~0 and the last ~0)."""
    V = np.zeros_like(Q)
    if len(Q) > 2:
        V[1:-1] = (Q[2:] - Q[:-2]) / (times[2:] - times[:-2])[:, None]
    V[0] = 0.0
    V[-1] = 0.0
    return V


def _fake_traj(times, Q, V):
    """A trajectory_msgs/JointTrajectory-shaped object the real trajectory_host.Plan + the port's
    _retime_plan_for_exec_limits accept (joint_names + points[].positions/.velocities/.time_from_start)."""
    pts = []
    for i in range(len(times)):
        sec = int(times[i])
        nanosec = int(round((times[i] - sec) * 1e9))
        if nanosec >= 1_000_000_000:
            sec += 1
            nanosec -= 1_000_000_000
        pts.append(NS(positions=list(Q[i]), velocities=list(V[i]), effort=[],
                      time_from_start=NS(sec=sec, nanosec=nanosec)))
    return NS(joint_names=list(NAMES), points=pts)


def _arm_start(positions_name_order):
    """A plan's first waypoint re-expressed in ARM order (the 'measured' start the Plan checks against)."""
    by = dict(zip(NAMES, positions_name_order))
    return [by[j] for j in ARM_JOINTS]


def _plan_verdict(times, Q, V):
    """ACCEPT or the executor's exact rejection string, from the REAL trajectory_host.Plan."""
    try:
        Plan(_fake_traj(times, Q, V), _arm_start(Q[0]), LIMITS,
             max_speed=EXEC_MAX_SPEED, max_acceleration=EXEC_MAX_ACCEL)
        return 'ACCEPT'
    except Exception as exc:  # noqa: BLE001 -- the rejection reason is the assertion target
        return str(exc)


def _lift_plan():
    p = _FIX['warmacc4_lift_plan']
    return np.array(p['times'], float), np.array(p['positions'], float)


def _preplace_over_limit_plan():
    """The warmacc4 carton_level-OFF pre-place condition: the REAL recorded cuMotion+TOTG plan whose
    TOTG stop landed a SMALL terminal remainder off the 0.1 s grid (the pre-place move is larger than the
    recorded lift, so its terminal-remainder decel is sharper). Shrinking the recorded lift's terminal
    remainder from 0.0115 s to 0.006 s reproduces the executor cubic-Hermite acceleration crossing the
    1.0 rad/s^2 ceiling on ur_wrist_2 -- the warmacc4 rejection -- WITHOUT raising any joint speed past
    the 0.3 rad/s profile (so it is an acceleration violation, exactly like the recorded one)."""
    t, Q = _lift_plan()
    t = t.copy()
    t[-1] = t[-2] + 0.006
    return t, Q, _executor_velocities(t, Q)


# --------------------------------------------------------------------- shared formula identity

def test_segment_accel_peaks_matches_trajectory_host_plan_formula():
    """The mission-side gate must compute the IDENTICAL per-segment acceleration the executor rejects on."""
    rng = np.random.default_rng(1)
    for _ in range(2000):
        q0, q1, v0, v1 = (rng.normal(size=6) for _ in range(4))
        d = float(rng.uniform(0.005, 3.0))
        a = (2 * (q0 - q1) + d * (v0 + v1)) / d ** 3
        b = (3 * (q1 - q0) - d * (2 * v0 + v1)) / d ** 2
        inline = np.maximum(np.abs(2 * b), np.abs(6 * a * d + 2 * b))  # trajectory_host.Plan line
        assert np.allclose(inline, T.segment_accel_peaks(q0, q1, v0, v1, d))


# --------------------------------------------------------------------- root condition (real data)

def test_warmacc4_cumotion_plan_sits_at_the_executor_ceiling():
    """The recorded carton_level-OFF cuMotion+TOTG lift plan runs right at the executor's cubic-Hermite
    acceleration ceiling with essentially no margin -- the reason the (larger) pre-place crossed it."""
    t, Q = _lift_plan()
    V = _executor_velocities(t, Q)
    peak, jidx, _ = T.trajectory_accel_peak(t, Q, V)
    assert 0.90 < peak < EXEC_MAX_ACCEL          # at the ceiling, but under -> accepted
    assert NAMES[jidx] == 'ur_wrist_2_joint'
    assert _plan_verdict(t, Q, V) == 'ACCEPT'
    assert np.abs(V).max() < EXEC_MAX_SPEED      # the near-ceiling is acceleration, not speed


# ------------------------------------------------- (1) warmacc4 pre-place: reject -> retime -> pass

def test_warmacc4_preplace_over_limit_is_rejected_by_the_executor():
    """Reproduce the warmacc4 pre-place rejection against the REAL trajectory_host.Plan: an interpolated
    ACCELERATION violation on ur_wrist_2 (host-stack.log 7306), with every joint speed still in-profile."""
    t, Q, V = _preplace_over_limit_plan()
    peak, jidx, _ = T.trajectory_accel_peak(t, Q, V)
    assert peak > EXEC_MAX_ACCEL
    assert NAMES[jidx] == 'ur_wrist_2_joint'
    assert np.abs(V).max() < EXEC_MAX_SPEED
    assert _plan_verdict(t, Q, V) == 'interpolated acceleration limit violation'


def test_warmacc4_preplace_plan_passes_the_gate_after_retiming():
    """The fix: the same over-limit pre-place plan, uniformly retimed against the executor ceiling, is
    accepted by the REAL trajectory_host.Plan; the geometric joint path (positions) is preserved exactly
    and no joint speed is raised."""
    t, Q, V = _preplace_over_limit_plan()
    new_t, new_V, scale, peak_before, peak_after = T.retime_to_accel_limit(
        t, Q, V, max_acceleration=EXEC_MAX_ACCEL, budget_s=BUDGET_S, joint_names=NAMES)
    assert scale > 1.0
    assert peak_before > EXEC_MAX_ACCEL >= peak_after
    assert np.isclose(peak_after, peak_before / scale ** 2)          # accel scales 1/s^2
    assert np.all(new_t >= t - 1e-12) and new_t[-1] <= BUDGET_S      # slower, within the budget
    assert np.abs(new_V).max() <= np.abs(V).max() + 1e-9             # speeds never raised
    assert _plan_verdict(new_t, Q, new_V) == 'ACCEPT'                # the executor now accepts it
    # positions are untouched -> the carton follows the same path, only slower
    assert T.retime_to_accel_limit(t, Q, V, max_acceleration=EXEC_MAX_ACCEL, budget_s=BUDGET_S,
                                   joint_names=NAMES)[0] is not Q    # times changed, positions not returned


# ------------------------------------------------- (2) coldwin33 levelled plan still passes (no-op)

def test_coldwin33_levelled_preplace_passes_and_retime_is_a_noop():
    """The carton_level-ON leveled begin_joint_path (recorded coldwin33 9-waypoint rest-to-rest plan)
    is far under the executor ceiling, so the gate leaves it byte-identical (scale == 1)."""
    p = _FIX['coldwin33_levelled_preplace']
    t = np.array(p['times'], float)
    Q = np.array(p['positions'], float)
    V = np.zeros_like(Q)                         # begin_joint_path waypoints are rest points
    peak, _, _ = T.trajectory_accel_peak(t, Q, V)
    assert peak < 0.2 * EXEC_MAX_ACCEL           # 7x+ under the ceiling
    assert _plan_verdict(t, Q, V) == 'ACCEPT'
    new_t, new_V, scale, pb, pa = T.retime_to_accel_limit(
        t, Q, V, max_acceleration=EXEC_MAX_ACCEL, budget_s=BUDGET_S, joint_names=NAMES)
    assert scale == 1.0 and new_t is t and new_V is V and pb == pa   # byte-identical (no-op)


# ------------------------------------------------- (3) synthetic over-limit: refuse by name

def _unretimable_plan():
    """A plan already near the bounded execution budget AND over the ceiling at a sharp terminal segment
    (ur_elbow_joint): the uniform time-stretch needed to meet the ceiling would push its duration well
    past EXEC_DURATION_BUDGET_S, so it cannot be retimed to fit -- the 'cannot meet the limits' case."""
    times = np.array([0.0, 100.0, 100.01])                      # ~100 s base, 0.01 s terminal remainder
    Q = np.zeros((3, 6)); Q[:, 0] = [1.30, 1.30, 1.32]          # ur_elbow_joint (NAMES index 0)
    V = np.zeros((3, 6)); V[1, 0] = 0.3                         # 0.3 rad/s at the mid waypoint -> sharp decel
    return times, Q, V


def test_synthetic_unretimable_plan_refuses_with_named_reason():
    """A plan too aggressive to meet the ceiling within the bounded execution budget is refused AT PLAN
    TIME with the exact named reason, naming the worst joint."""
    times, Q, V = _unretimable_plan()
    peak, jidx, _ = T.trajectory_accel_peak(times, Q, V)
    assert peak > EXEC_MAX_ACCEL and NAMES[jidx] == 'ur_elbow_joint'
    with pytest.raises(T.PlanLimitError) as exc:
        T.retime_to_accel_limit(times, Q, V, max_acceleration=EXEC_MAX_ACCEL,
                                budget_s=BUDGET_S, joint_names=NAMES)
    msg = str(exc.value)
    assert msg.startswith('plan exceeds acceleration limit: ur_elbow_joint ')
    assert ('> %.3f' % EXEC_MAX_ACCEL) in msg
    assert isinstance(exc.value, ValueError)     # existing except ValueError plan handling still catches


def test_retime_math_is_a_pure_time_reparameterisation():
    """Uniform time scaling: positions are the SAME samples, velocities scale 1/s, acceleration 1/s^2,
    and a plan already within the limit is returned verbatim."""
    rng = np.random.default_rng(7)
    times = np.array([0, .1, .2, .3, .37])
    Q = np.cumsum(rng.normal(scale=0.2, size=(5, 6)), axis=0)
    V = rng.normal(scale=0.5, size=(5, 6)); V[0] = 0; V[-1] = 0
    peak0, _, _ = T.trajectory_accel_peak(times, Q, V)
    # under-limit -> no-op
    nt, nv, s, pb, pa = T.retime_to_accel_limit(times, Q, V, max_acceleration=peak0 * 2,
                                                budget_s=1e9, joint_names=NAMES)
    assert s == 1.0 and nt is times
    # over-limit -> scaled
    nt, nv, s, pb, pa = T.retime_to_accel_limit(times, Q, V, max_acceleration=peak0 / 4,
                                                budget_s=1e9, joint_names=NAMES)
    assert np.isclose(s, 2 * (1 + 1e-3), atol=1e-2)
    assert np.allclose(nt, times * s) and np.allclose(nv, V / s)
    assert pa <= peak0 / 4 + 1e-9


# ------------------------------------------------- port wrapper (ROS-msg adapter, CPU)

def _call_port_retime(exec_max_accel, traj):
    """Invoke RosMotionPort._retime_plan_for_exec_limits with a minimal fake self (CPU, no ROS node)."""
    from twin.restock_manipulation import ros_adapter
    fake_self = NS(_exec_max_accel=exec_max_accel)
    return ros_adapter.RosMotionPort._retime_plan_for_exec_limits(fake_self, traj)


def test_port_wrapper_retimes_an_over_limit_cumotion_plan_in_place():
    t, Q, V = _preplace_over_limit_plan()
    traj = NS(joint_trajectory=_fake_traj(t, Q, V))
    before = [p.time_from_start.sec + p.time_from_start.nanosec * 1e-9 for p in traj.joint_trajectory.points]
    out = _call_port_retime(EXEC_MAX_ACCEL, traj)
    after_t = [p.time_from_start.sec + p.time_from_start.nanosec * 1e-9 for p in out.joint_trajectory.points]
    assert after_t[-1] > before[-1]                                   # slowed
    # positions unchanged; the retimed msg is accepted by the executor
    newQ = np.array([list(p.positions) for p in out.joint_trajectory.points])
    newV = np.array([list(p.velocities) for p in out.joint_trajectory.points])
    assert np.allclose(newQ, Q)
    assert _plan_verdict(np.array(after_t), newQ, newV) == 'ACCEPT'


def test_port_wrapper_is_byte_identical_for_an_under_limit_plan():
    p = _FIX['coldwin33_levelled_preplace']
    t = np.array(p['times'], float); Q = np.array(p['positions'], float); V = np.zeros_like(Q)
    traj = NS(joint_trajectory=_fake_traj(t, Q, V))
    times_before = [(pt.time_from_start.sec, pt.time_from_start.nanosec) for pt in traj.joint_trajectory.points]
    out = _call_port_retime(EXEC_MAX_ACCEL, traj)
    assert out is traj                                                # same object, untouched
    assert [(pt.time_from_start.sec, pt.time_from_start.nanosec) for pt in out.joint_trajectory.points] == times_before


def test_port_wrapper_refuses_an_unretimable_plan():
    times, Q, V = _unretimable_plan()
    traj = NS(joint_trajectory=_fake_traj(times, Q, V))
    with pytest.raises(T.PlanLimitError):
        _call_port_retime(EXEC_MAX_ACCEL, traj)


# ------------------------------------------------- (point 3) named trip + heartbeat-alive path

def test_goal_rejection_and_plan_limit_are_preserved_named_trips():
    """A controller goal rejection (moveit_err=-4 class) and a plan-time acceleration refusal are both
    NAMED first trips that survive over the downstream 'arm command heartbeat lost' the controlled stop's
    deliberate arm hold produces -- so the terminal mission reason names the real cause, not the symptom.
    This verifies the reason-preservation wiring PR 316 keeps the heartbeat alive with (suppress_heartbeat)
    -- it does not re-implement the controlled stop."""
    goal_reject = ('arm controller aborted trajectory (goal rejected or execution fault, moveit_err=-4; '
                   'see host-stack.log TRAJ_HOST_GOAL_REJECT/TRAJ_HOST_FAULT)')
    plan_limit = 'plan exceeds acceleration limit: ur_wrist_2_joint 1.887 > 1.000'
    downstream = 'arm command heartbeat lost'
    for reason in (goal_reject, plan_limit):
        assert controlled_stop.is_trip_reason(reason)
        # the first real trip wins over the later heartbeat symptom
        assert controlled_stop.prefer_first_trip(reason, downstream) == reason
    # a bare heartbeat symptom is NOT itself a trip (so it never masks the real cause)
    assert not controlled_stop.is_trip_reason(downstream)
    assert controlled_stop.is_downstream_symptom(downstream)
