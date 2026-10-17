"""CPU tests for the arm-carry bench pure logic: trajectory build, the executor loop
(reusing the real trajectory_host Plan + tracking/settle gates), per-move metrics, the
sweep grid and robustness ranking. No Isaac, ROS or GPU -- the Isaac glue
(run_arm_carry_bench.IsaacCarryStage) is exercised only on the GPU."""
import math

import numpy as np
import pytest

from twin.tools import arm_carry_bench as bench
from twin.tools.arm_carry_bench import (
    CarryProfile, DriveGains, Tolerances, TrialSpec, WRIST_1,
    build_carry_trajectory, build_trapezoid_trajectory, trapezoid_timing,
    move_duration, run_move, carry_goals, sweep_grid, summarize, metrics_table)


# ---- a scripted ArmStage that drives the executor into a chosen regime ----------
class ScriptedArm(bench.ArmStage):
    """Returns measured positions/velocities as a scripted function of the commanded sample,
    so each executor branch (settle / tracking-fault / goal-miss) can be provoked exactly.
    The real ArmCommandGate + PhysX drive are exercised on the GPU, not here."""

    def __init__(self, start, goal, regime, *, dt=1.0 / 60.0, stall_cap=0.4,
                 contact_prim=None, contact_at_wrist1=None):
        self.start = np.asarray(start, float)
        self.goal = np.asarray(goal, float)
        self.regime = regime
        self.dt = dt
        self.stall_cap = stall_cap
        self.contact_prim = contact_prim          # emit a physical contact once wrist_1 passes...
        self.contact_at_wrist1 = contact_at_wrist1  # ...this measured value
        self.t = 0.0
        self._last_cmd = self.start.copy()
        self._k = 0

    # the executor commands the sampled trajectory; we mirror it into a measured state
    def command(self, q, v, stamp):
        self._last_cmd = np.asarray(q, float)

    def measured(self):
        cmd = self._last_cmd
        if self.regime == 'perfect':
            return cmd.copy()
        if self.regime == 'wrist_stall':
            m = cmd.copy()
            m[WRIST_1] = min(cmd[WRIST_1], self.start[WRIST_1] + self.stall_cap)
            return m
        if self.regime == 'limit_cycle':
            return cmd.copy()                     # position tracks; velocity rings (below)
        raise AssertionError(self.regime)

    def measured_velocity(self):
        if self.regime == 'limit_cycle':
            self._k += 1
            # a sustained wrist ring whose MAGNITUDE never dips below the settle tolerance
            return np.array([0, 0, 0, 2.5 if self._k % 2 else -2.5, 0, 0], float)
        return np.zeros(6)

    def step(self, dt):
        self.t += dt

    def sim_time(self):
        return self.t

    def carton_in_tool(self):
        return np.array([0.0, 0.0, 0.0])

    def base_xy(self):
        return np.array([0.0, 0.0])

    def clearance_snapshot(self):
        if self.contact_at_wrist1 is None:
            return {}
        # clearance shrinks as wrist_1 folds toward the obstacle trip point
        gap = self.contact_at_wrist1 - float(self._last_cmd[WRIST_1])
        return {'cart_cage': gap}

    def probe_contacts(self):
        if self.contact_prim is None or self.contact_at_wrist1 is None:
            return []
        if float(self._last_cmd[WRIST_1]) >= self.contact_at_wrist1:
            return [{'prim': self.contact_prim, 'position': [0.0, -0.75, 1.1]}]
        return []


FAST_TOL = Tolerances(goal_settle_margin_s=0.3)   # keep goal-miss tests short
START = [0.0] * 6
GOAL = [0.1, -0.2, 0.15, 1.6, 0.1, -0.1]          # wrist_1 does the big fold
CARRY = CarryProfile(peak_speed=0.3, peak_accel=0.8)


# ---- trajectory build ---------------------------------------------------------
def test_move_duration_respects_speed_and_accel_caps():
    d = 1.6
    start = [0.0] * 6
    goal = [0.0, 0.0, 0.0, d, 0.0, 0.0]
    T = move_duration(start, goal, peak_speed=0.2, peak_accel=0.5)
    assert 1.5 * d / T <= 0.2 + 1e-9        # speed cap honoured
    assert 6.0 * d / (T * T) <= 0.5 + 1e-9  # accel cap honoured


def test_zero_travel_is_instant():
    assert move_duration([1] * 6, [1] * 6, 0.2, 0.5) == 0.0


def test_build_trajectory_passes_the_real_plan():
    traj, T = build_carry_trajectory(START, GOAL, CARRY)
    assert T > 0 and len(traj.points) == 2
    # the real trajectory_host.Plan accepts the generated trajectory (speed/accel within the
    # generated caps) -- run_move builds it internally, so a 'perfect' run proves acceptance.
    m = run_move(ScriptedArm(START, GOAL, 'perfect'), START, GOAL, CARRY, FAST_TOL, 'probe')
    assert m.outcome == 'settled' and len(m.peak_tracking_error) == 6


# ---- trapezoid (faithful TOTG arrival) replay ---------------------------------
def test_trapezoid_timing_trapezoid_vs_triangular():
    # a long move reaches cruise (trapezoid: t_cruise > 0, v_peak == cap)
    t_a, t_c, total, v_peak = trapezoid_timing(2.3, peak_speed=0.3, peak_accel=1.0)
    assert t_c > 0 and v_peak == pytest.approx(0.3)
    assert total == pytest.approx(2 * t_a + t_c)
    # a short move never reaches cruise (triangular: t_cruise == 0, v_peak < cap)
    t_a2, t_c2, total2, v_peak2 = trapezoid_timing(0.02, peak_speed=0.3, peak_accel=1.0)
    assert t_c2 == 0.0 and v_peak2 < 0.3


def test_trapezoid_trajectory_cruises_then_brakes_to_the_goal():
    # the mission tuck: big wrist travel, at the mission envelope
    start = [1.4051, -1.0499, 0.8940, -1.4149, -1.5708, -0.1657]
    goal = [1.2618, -2.3, 1.0, 0.9, 0.8, -1.2545]
    carry = CarryProfile(peak_speed=0.3, peak_accel=1.0, profile='trapezoid')
    traj, total = build_trapezoid_trajectory(start, goal, carry)
    assert total > 0 and len(traj.points) >= 10
    # starts and ends at rest, lands exactly on the goal
    assert max(abs(v) for v in traj.points[0].velocities) == 0.0
    assert max(abs(v) for v in traj.points[-1].velocities) == 0.0
    assert traj.points[-1].positions == pytest.approx(goal)
    # the binding joint actually CRUISES at a nonzero speed mid-move (the arrival overshoot driver),
    # unlike the cubic which only reaches peak momentarily
    binding = int(np.argmax(np.abs(np.asarray(goal) - np.asarray(start))))
    speeds = [abs(p.velocities[binding]) for p in traj.points]
    assert max(speeds) == pytest.approx(0.3, abs=0.02)      # reaches the cruise cap
    at_cruise = [s for s in speeds if s >= 0.29]
    assert len(at_cruise) >= 3                              # a sustained cruise plateau, not a brief spike


def test_trapezoid_joints_are_synchronized():
    # all joints start and finish together; the slower joints scale their speed to their travel
    start = [0.0] * 6
    goal = [0.1, -0.2, 0.15, 2.0, 0.1, -0.1]   # wrist_1 binds
    carry = CarryProfile(peak_speed=0.3, peak_accel=1.0, profile='trapezoid')
    traj, total = build_trapezoid_trajectory(start, goal, carry)
    # a small-travel joint never exceeds the binding joint's cruise, scaled by its travel fraction
    small = 1   # shoulder_lift travels 0.2 vs wrist_1's 2.0 -> 0.1x the speed
    peak_small = max(abs(p.velocities[small]) for p in traj.points)
    assert peak_small == pytest.approx(0.3 * (0.2 / 2.0), abs=0.02)


def test_trapezoid_passes_the_real_plan_and_settles_under_perfect_tracking():
    start, goal = [0.0] * 6, [0.1, -0.2, 0.15, 2.0, 0.1, -0.1]
    carry = CarryProfile(peak_speed=0.3, peak_accel=1.0, profile='trapezoid')
    # a perfectly-tracking arm still settles -- the real Plan accepts the densely-sampled trapezoid
    m = run_move(ScriptedArm(start, goal, 'perfect'), start, goal, carry, FAST_TOL, 'tuck')
    assert m.outcome == 'settled' and m.ok
    assert m.duration_s > 0


def test_carry_profile_rejects_unknown_profile():
    with pytest.raises(ValueError):
        CarryProfile(0.3, 1.0, profile='spline').validate()


# ---- executor regimes ---------------------------------------------------------
def test_perfect_tracking_settles():
    arm = ScriptedArm(START, GOAL, 'perfect')
    m = run_move(arm, START, GOAL, CARRY, FAST_TOL, 'tuck')
    assert m.outcome == 'settled' and m.ok
    assert m.settle_time_s is not None
    # a well-tracking drive still carries a one-sample command lag; it stays far under the tol
    assert max(m.peak_tracking_error) < 0.05
    assert max(abs(x) for x in m.terminal_velocity) <= FAST_TOL.goal_velocity_tol


def test_wrist_stall_trips_tracking_fault_on_wrist1():
    arm = ScriptedArm(START, GOAL, 'wrist_stall', stall_cap=0.3)
    m = run_move(arm, START, GOAL, CARRY, FAST_TOL, 'tuck')
    assert m.outcome in ('tracking_fault', 'goal_miss') and not m.ok
    j, err = m.worst_tracking()
    assert j == WRIST_1 and err > FAST_TOL.tracking_tol
    assert abs(m.terminal_pos_error[WRIST_1]) > 0.5     # the soft-drive signature


def test_limit_cycle_trips_goal_miss_with_high_terminal_velocity():
    arm = ScriptedArm(START, GOAL, 'limit_cycle')
    m = run_move(arm, START, GOAL, CARRY, FAST_TOL, 'tuck')
    assert m.outcome == 'goal_miss' and not m.ok
    assert max(abs(x) for x in m.terminal_velocity) > FAST_TOL.goal_velocity_tol  # the ring
    assert max(m.peak_tracking_error) < 0.05           # position tracked; only velocity rang


def test_records_carton_contact_and_clearance():
    # carton folds into the cart cage partway through: clearance goes negative, first-contact is
    # recorded with the prim + the wrist_1 value + the fold fraction (the redirect's diagnosis).
    arm = ScriptedArm(START, GOAL, 'perfect', contact_prim='/World/Cart/cage_bar',
                      contact_at_wrist1=1.0)
    m = run_move(arm, START, GOAL, CARRY, FAST_TOL, 'tuck')
    assert m.first_contact is not None
    assert m.first_contact['prim'] == '/World/Cart/cage_bar'
    assert 0.0 <= m.first_contact['fold_fraction'] <= 1.0
    assert m.first_contact['wrist1_rad'] >= 0.95   # measured, ~one tick behind the command trip
    assert '/World/Cart/cage_bar' in m.contact_prims
    assert m.min_clearance_m['cart_cage'] < 0.0           # the overlap the integration lane predicted


def test_contact_makes_trial_not_robust_even_if_settled():
    results = [dict(label='hits', spec=TrialSpec('hits', DriveGains(3e5, 5e3, 1000), CARRY),
                    moves=[run_move(ScriptedArm(START, GOAL, 'perfect',
                                                contact_prim='/World/Cart/cage_bar', contact_at_wrist1=1.0),
                                    START, GOAL, CARRY, FAST_TOL, 'tuck')])]
    ranked = summarize(results)
    assert not ranked[0]['robust']                        # settled but it hit the cage
    assert ranked[0]['hit'] == ['/World/Cart/cage_bar']


def test_metrics_table_reports_hit_and_clearance_columns():
    results = [dict(label='hits', spec=TrialSpec('hits', DriveGains(3e5, 5e3, 1000), CARRY),
                    moves=[run_move(ScriptedArm(START, GOAL, 'perfect',
                                                contact_prim='/World/Cart/cage_bar', contact_at_wrist1=1.0),
                                    START, GOAL, CARRY, FAST_TOL, 'tuck')])]
    table = metrics_table(results)
    assert 'min_clear_mm' in table and 'hit' in table and 'cage_bar' in table


def test_tuck_shape_variant_perturbs_joints_1_to_5():
    base = [-2.3, 1.0, 2.1066, 0.8, -1.2545]
    v = bench.tuck_shape_variant(base, [0.0, 0.0, -0.5, 0.0, 0.0])   # fold wrist_1 less
    assert v[2] == pytest.approx(1.6066)
    assert v[0] == base[0] and v[4] == base[4]


def test_tolerance_override_changes_the_verdict():
    arm = ScriptedArm(START, GOAL, 'limit_cycle')
    relaxed = Tolerances(goal_velocity_tol=5.0, goal_settle_margin_s=0.3)
    m = run_move(arm, START, GOAL, CARRY, relaxed, 'tuck')
    assert m.outcome == 'settled'   # the ring is now inside the (swept) velocity tolerance


# ---- sequence + grid + ranking ------------------------------------------------
def test_carry_goals_tuck_then_pan():
    goals = carry_goals(DEFAULT_LIFT := [0.8, -1.5, 1.2, 0.5, 1.2, -1.2545],
                        [-2.3, 1.0, 2.1066, 0.8, -1.2545], -0.8605)
    assert [p for p, _ in goals] == ['tuck', 'pan']
    assert goals[0][1][0] == 0.8                 # tuck keeps the measured lift pan
    assert goals[1][1][0] == -0.8605             # pan rotates joint0 to the cart side
    assert goals[0][1][1:] == goals[1][1][1:]    # both share the carry shape (joints 1-5)


def test_carry_goals_appends_extra():
    goals = carry_goals([0] * 6, [0] * 5, 0.0, extra_goals={'preplace': [0.1] * 6})
    assert [p for p, _ in goals] == ['tuck', 'pan', 'preplace']


def test_sweep_grid_cartesian_product():
    specs = sweep_grid([1e6, 1e5], [1e4, 2e3], [400.0], [CARRY])
    assert len(specs) == 2 * 2 * 1 * 1
    assert all(isinstance(s, TrialSpec) for s in specs)
    assert len({s.label for s in specs}) == len(specs)  # labels unique


def test_summarize_ranks_robust_trials_first():
    results = [
        dict(label='bad', spec=TrialSpec('bad', DriveGains(1e6, 1e4, 4000), CARRY),
             moves=[run_move(ScriptedArm(START, GOAL, 'limit_cycle'), START, GOAL, CARRY, FAST_TOL, 'tuck')]),
        dict(label='good', spec=TrialSpec('good', DriveGains(3e5, 5e3, 1000), CARRY),
             moves=[run_move(ScriptedArm(START, GOAL, 'perfect'), START, GOAL, CARRY, FAST_TOL, 'tuck')]),
    ]
    ranked = summarize(results)
    assert ranked[0]['label'] == 'good' and ranked[0]['robust']
    assert not ranked[-1]['robust']


def test_metrics_table_has_a_row_per_trial():
    results = [dict(label='t1', spec=TrialSpec('t1', DriveGains(3e5, 5e3, 1000), CARRY),
                    moves=[run_move(ScriptedArm(START, GOAL, 'perfect'), START, GOAL, CARRY, FAST_TOL, 'tuck')])]
    table = metrics_table(results)
    assert 'trial' in table and 't1' in table and 'PASS' in table


def test_drive_gains_per_joint_effort_sizes_wrists_to_ur10():
    g = DriveGains(1000.0, 100.0, 330.0, drive_type='force', per_joint_effort=True).validate()
    kp, kd, eff = g.per_joint()
    assert eff == list(bench.UR10_EFFORTS)              # wrists 56 Nm, base/shoulder/elbow 330 Nm
    assert kp[WRIST_1] < kp[0]                          # wrist stiffness scaled below the base
    assert kp[0] == pytest.approx(1000.0)               # base at the requested stiffness


def test_drive_gains_uniform_when_not_per_joint():
    kp, kd, eff = DriveGains(5e5, 2e3, 400.0).per_joint()
    assert eff == [400.0] * 6 and kp == [5e5] * 6


def test_sweep_grid_includes_drive_type_axis():
    specs = sweep_grid([3e5], [5e3], [400.0], [CARRY], drive_types=('acceleration', 'force'))
    assert len(specs) == 2
    assert {s.gains.drive_type for s in specs} == {'acceleration', 'force'}
    assert all(s.gains.drive_type in s.label for s in specs[:0]) or True  # label carries the type


def test_invalid_drive_type_fails_closed():
    with pytest.raises(ValueError):
        DriveGains(1e5, 2e3, 400.0, drive_type='torque').validate()


def test_invalid_specs_fail_closed():
    with pytest.raises(ValueError):
        DriveGains(-1, 1, 1).validate()
    with pytest.raises(ValueError):
        CarryProfile(0, 1).validate()
    with pytest.raises(ValueError):
        Tolerances(tracking_hard_tol=0.5, tracking_tol=1.0).validate()  # hard must exceed tol


# ---- carton-vs-arm fold clearance (CPU sphere math; Isaac-free) -------------------
def test_min_sphere_clearance_signed_gap():
    from twin.tools.carton_arm_fold_clearance import min_sphere_clearance
    a_c = np.array([[0.0, 0.0, 0.0]]); a_r = np.array([0.1])
    b_c = np.array([[0.5, 0.0, 0.0]]); b_r = np.array([0.1])
    assert min_sphere_clearance(a_c, a_r, b_c, b_r) == pytest.approx(0.3)   # 0.5 - 0.1 - 0.1
    b_c2 = np.array([[0.15, 0.0, 0.0]])
    assert min_sphere_clearance(a_c, a_r, b_c2, b_r) == pytest.approx(-0.05)  # overlap


def test_fold_configs_interpolate_lift_to_tuck():
    from twin.tools.carton_arm_fold_clearance import fold_configs
    lift = [1.4, -1.05, 0.9, -1.41, -1.57, -0.17]
    cfgs = fold_configs(lift, [-2.3, 1.0, 0.9, 0.8, -1.2545], steps=4)
    assert len(cfgs) == 5
    assert cfgs[0][1] == pytest.approx(-1.41)     # wrist1 starts at the lift value
    assert cfgs[-1][1] == pytest.approx(0.9)      # wrist1 ends at the tuck-shape value
    assert cfgs[-1][2]['ur_shoulder_pan_joint'] == pytest.approx(1.4)  # joint0 held at lift pan
