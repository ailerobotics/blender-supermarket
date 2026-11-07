"""CPU tests for the place-descent false start-stall fix (sim-place-descent-stall-false-start-v1).

coldwin100/101c declared the collect set-down OBSTRUCTED (``PLACE_OBSTRUCTED prim=above-modelled-seat``,
carton bottom +108.8 mm) 83 ms / 5 ticks after the descent trajectory was dispatched, with the carton
centre z FROZEN at the descent-start height. The dispatched CARTON_LEVEL_PLACE_DESCENT plan (n=12, speed
0.020, accel 0.040 rad/s) has a 3.5 s (coldwin100) / 21.3 s (coldwin101c) rest-to-rest FIRST segment -- so
the plan cannot move the carton 1 mm (``stall_eps``) in 83 ms, and the trajectory had not even begun
executing (``TRAJ_HOST_START anchored start`` was in the future). The effort was at baseline (no press) and
the PR 384 contact logger logged ZERO contacts. The ``place_release.PlaceReleaseGate`` STALL trigger was
counting a render-paced startup window as a stall and the seat gate read it as a contact above the seat.

The fix arms the stall trigger only once descent progress was EXPECTED (opt-in HOST key
``carton_level.place_release_stall_arming``: ``'progress-lag'`` preferred, ``'descent-begun'`` fallback),
leaving the EFFORT trigger unchanged so a genuine descent-start jam (coldwin68's cage post) still ends by
name. Default off => byte-identical: the coldwin100 tick sequence still produces ``action='obstructed'``.

    PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=scripts python3 -m pytest \
        scripts/twin/tests/test_place_descent_stall_false_start_v1.py
"""
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from twin.restock_manipulation import carton_level as cl
from twin.restock_manipulation import TransferController, ProductProfile
from twin.restock_manipulation.place_release import (
    PlaceReleaseGate, PlannedDescent, waypoint_segment_times)

# coldwin100 TRAJ_HOST_PLAN joint waypoints (names order; segment timing is per-joint-max so order-free)
# and the logged time_from_start of each point -- the REAL dispatched plan shape (n=12 => 13 points).
CW100_WPS = [
    [1.8106, -1.8708, -0.0694, 0.059, 1.7243, -0.0005],
    [1.7781, -1.8239, -0.0701, 0.0587, 1.7112, -0.0005],
    [1.7433, -1.7773, -0.0713, 0.0396, 1.6977, -0.0005],
    [1.7059, -1.7309, -0.0721, 0.0231, 1.6844, -0.0004],
    [1.6663, -1.6845, -0.0727, 0.009, 1.6714, -0.0004],
    [1.6245, -1.6382, -0.0732, -0.0028, 1.6587, -0.0003],
    [1.5803, -1.5918, -0.0735, -0.0124, 1.6463, -0.0002],
    [1.5338, -1.5454, -0.0738, -0.0109, 1.634, -0.0001],
    [1.4847, -1.4988, -0.074, -0.0071, 1.6219, -0.0001],
    [1.4751, -1.4713, -0.0731, -0.0242, 1.6175, -0.0],
    [1.5244, -1.4702, -0.0708, -0.0745, 1.6239, -0.0],
    [1.5704, -1.4658, -0.0687, -0.125, 1.6297, -0.0],
    [1.614, -1.4589, -0.0667, -0.1755, 1.6351, -0.0001],
]
CW100_LOGGED_TIMES = [0.0, 3.5184, 7.01, 10.4942, 13.9715, 17.4455, 20.9207, 24.4108,
                      28.0884, 30.1504, 33.9279, 37.7138, 41.4978]

# coldwin100 PLACE_RELEASE_TICK / PLACE_SEAT_GATE_ARM geometry (ur_base_link, carton CENTRE frame):
CW100_START_Z = 0.6397          # carton centre z, FROZEN at the descent start the whole 83 ms window
CW100_SEAT_CZ = 0.5308          # modelled seat carton-centre z (support_top 0.4408 + carton half 0.0900)
CW100_EFFORT = 120.4            # arm effort at baseline (no press) through the window
CW100_DISPATCH_NOW = 290.800    # first PLACE_RELEASE_TICK sim time (stall=0)
TICK_DT = 1.0 / 60.0            # 60 Hz controller ticks


def _planned_cw100():
    """A PlannedDescent on the REAL coldwin100 plan timing. The knot descents are modelled as a monotone
    descent proportional to the segment timing (total 0.096 m, coldwin100's descent_drop) -- the stall
    arming only needs the early-window planned descent, which the 3.5 s rest-to-rest first segment holds
    sub-mm regardless of the total magnitude."""
    times = waypoint_segment_times(CW100_WPS, speed=0.020, accel=0.040)
    total = 0.096
    descents = tuple(total * (t / times[-1]) for t in times)
    return PlannedDescent(tuple(times), descents)


def _seat_gate(*, stall_arming=None, planned=None, **over):
    kw = dict(release_z=CW100_SEAT_CZ, effort_margin_nm=100.0, effort_consecutive=3,
              stall_eps_m=0.001, stall_ticks=5, support_tokens=('shelf', 'deck'), obstruction_gap_mm=2.0,
              effort_warmup_ticks=3, seat_center_z=CW100_SEAT_CZ, seat_tol_m=0.02, require_seat=True,
              stall_arming=stall_arming)
    kw.update(over)
    g = PlaceReleaseGate(**kw)
    if planned is not None:
        g.reset(release_z=CW100_SEAT_CZ, seat_center_z=CW100_SEAT_CZ, planned=planned)
    return g


def _run_frozen(gate, *, ticks, z=CW100_START_Z, eff=CW100_EFFORT, now0=CW100_DISPATCH_NOW):
    """Feed ``ticks`` frozen-carton descending ticks (the coldwin100 startup window) and return the list of
    decisions."""
    out = []
    for i in range(ticks):
        out.append(gate.observe(now=now0 + i * TICK_DT, carton_z=z, commanded_down=True,
                                arm_effort_max=eff))
    return out


# --------------------------------------------------------- (1) coldwin100 false start-stall: on => not stalled

def test_byte_identical_off_coldwin100_still_obstructs():
    """Arming OFF (default) must reproduce the coldwin100 failure: the 6th frozen tick (stall=5) is read as
    a measured contact and, being +108.9 mm above the seat, is OBSTRUCTED. This pins the byte-identical-off
    requirement -- the tick sequence still produces action=obstructed when the arming is off."""
    g = _seat_gate(stall_arming=None)
    decs = _run_frozen(g, ticks=8)
    assert all(d.action == 'press' for d in decs[:5]), [d.action for d in decs[:5]]
    assert decs[5].action == 'obstructed'
    assert decs[5].hit_prim == 'above-modelled-seat'
    assert decs[5].stall_count == 5


def test_progress_lag_coldwin100_not_obstructed_descent_continues():
    """Brief test #1: the coldwin100 sequence replayed with progress-lag arming ON (the real plan profile)
    is NOT obstructed -- the planned descent at 83 ms is sub-mm, so the frozen carton is not lagging the
    plan, and the descent continues (every tick 'press')."""
    g = _seat_gate(stall_arming='progress-lag', planned=_planned_cw100())
    decs = _run_frozen(g, ticks=20)             # well past the 6-tick / 83 ms false-stall window
    assert all(d.action == 'press' for d in decs), [(round(d.carton_z, 4), d.action) for d in decs]
    assert all(d.trigger is None for d in decs)


def test_progress_lag_coldwin101c_long_first_segment_not_obstructed():
    """coldwin101c's first plan segment is 21.3 s -- an even larger ramp. A frozen carton for the whole
    83 ms window still never lags the plan."""
    # one dominant-joint 21.3 s first segment (pan moves ~0.1035 rad => seg = 1.5*0.1035/0.02 = 7.76? no --
    # build it straight from the logged first segment: dmax 0.284 (wrist2) => sqrt(6*0.284/0.04)=6.53,
    # 1.5*0.284/0.02=21.3 -> 21.3 s). Use a 2-knot plan with that first segment.
    planned = PlannedDescent((0.0, 21.3, 44.0), (0.0, 0.05, 0.096))
    g = _seat_gate(stall_arming='progress-lag', planned=planned)
    decs = _run_frozen(g, ticks=20, now0=365.033)
    assert all(d.action == 'press' for d in decs)


# ------------------------------------------------------------- (2) a REAL stall after >= 30 mm of descent

def test_progress_lag_real_stall_after_30mm_is_obstructed():
    """Brief test #2: a carton that descends ~30 mm tracking the plan and THEN freezes while the plan still
    expects progress is a REAL stall -- the lag grows past the threshold and, since the carton is still
    above the modelled seat, the verdict is OBSTRUCTED (named). This is NOT suppressed."""
    start_z, seat_cz = 0.700, 0.600            # seat 100 mm below the start
    planned = PlannedDescent((0.0, 20.0), (0.0, 0.100))    # 100 mm descent over 20 s
    g = _seat_gate(stall_arming='progress-lag', planned=planned, release_z=seat_cz,
                   seat_center_z=seat_cz)
    g.reset(release_z=seat_cz, seat_center_z=seat_cz, planned=planned)
    # descend tracking the plan until 30 mm, then FREEZE the carton while sim-time (and the plan) advance
    dt, now0 = 0.1, 100.0
    frozen_z = None
    decs = []
    for i in range(400):
        now = now0 + i * dt
        elapsed = now - now0
        pdesc = planned.at(elapsed)
        if pdesc < 0.030 and frozen_z is None:
            z = start_z - pdesc                # tracks the plan (lag ~0 -> press)
        else:
            if frozen_z is None:
                frozen_z = start_z - 0.030     # jam at 30 mm descended
            z = frozen_z
        d = g.observe(now=now, carton_z=z, commanded_down=True, arm_effort_max=120.0)
        decs.append(d)
        if d.action != 'press':
            break
    assert decs[-1].action == 'obstructed', [d.action for d in decs[-6:]]
    assert decs[-1].hit_prim == 'above-modelled-seat'
    # the stall only fired AFTER real descent (>= 30 mm), never during the tracked descent
    assert frozen_z is not None and frozen_z == pytest.approx(0.670)
    assert sum(1 for d in decs if d.action == 'press') >= 30     # it pressed through the real descent


# ----------------------------------------------- (3) a descent-start jam with effort rising => effort trigger

def test_descent_start_jam_effort_rising_is_obstructed_by_effort():
    """Brief test #3: with progress-lag arming suppressing the startup STALL, a genuine descent-start jam
    (carton frozen at the start) is still caught by the UNCHANGED effort trigger -- the arm effort rising
    over the pre-contact baseline for effort_consecutive ticks is a measured contact, and being above the
    seat the verdict is OBSTRUCTED. The effort trigger is not touched by the stall-arming fix."""
    g = _seat_gate(stall_arming='progress-lag', planned=_planned_cw100())
    # 3 warmup ticks at baseline, then effort rises 120 -> 260 (>= baseline 120 + margin 100)
    efforts = [120.0, 120.0, 120.0] + [260.0] * 6
    decs = []
    for i, eff in enumerate(efforts):
        d = g.observe(now=CW100_DISPATCH_NOW + i * TICK_DT, carton_z=CW100_START_Z,
                      commanded_down=True, arm_effort_max=eff)
        decs.append(d)
        if d.action != 'press':
            break
    assert decs[-1].action == 'obstructed'           # above the seat
    assert 'effort' in decs[-1].reason               # the EFFORT trigger won
    # the stall never fired (progress-lag suppressed the startup window)
    assert all('stall' not in (d.trigger or '') for d in decs)


def test_effort_trigger_releases_at_seat_byte_identical_under_arming():
    """The effort trigger is unchanged: a rising effort AT the seat (within tol) still RELEASES -- the fix
    only gates the stall, never effort."""
    g = _seat_gate(stall_arming='progress-lag', planned=_planned_cw100(),
                   seat_center_z=CW100_START_Z, release_z=None)   # seat at the carton; no z fallback
    g.reset(release_z=None, seat_center_z=CW100_START_Z, planned=_planned_cw100())
    efforts = [120.0, 120.0, 120.0] + [260.0] * 6
    last = None
    for i, eff in enumerate(efforts):
        last = g.observe(now=CW100_DISPATCH_NOW + i * TICK_DT, carton_z=CW100_START_Z,
                         commanded_down=True, arm_effort_max=eff)
        if last.action != 'press':
            break
    assert last.action == 'release' and last.trigger == 'effort'


# -------------------------------------------- (4) the planned-progress source with the real plan shape

def test_planned_progress_source_matches_real_plan_timing():
    """Brief test #4: waypoint_segment_times on the REAL coldwin100 plan shape (n=12, speed 0.020,
    accel 0.040) reproduces the dispatched TRAJ_HOST_PLAN time_from_start (within the 4-decimal joint
    rounding of the logged waypoints)."""
    times = waypoint_segment_times(CW100_WPS, speed=0.020, accel=0.040)
    assert len(times) == len(CW100_LOGGED_TIMES) == 13
    assert times[1] == pytest.approx(CW100_LOGGED_TIMES[1], abs=0.02)   # first segment 3.5184 s
    assert max(abs(a - b) for a, b in zip(times, CW100_LOGGED_TIMES)) < 0.02
    assert times[-1] == pytest.approx(41.4978, abs=0.05)               # total ~41.5 s


def test_planned_descent_early_window_is_sub_mm():
    """The planned carton descent at the coldwin100 stall window (83 ms into a 3.5 s first segment) is
    sub-mm -- below the 1 mm stall_eps the gate demanded -- so the frozen measured carton has ~0 lag."""
    pd = _planned_cw100()
    assert pd.first_segment_s == pytest.approx(3.5175, abs=0.02)
    assert pd.at(0.083) * 1000.0 < 0.1               # sub-0.1 mm at the stall verdict
    assert pd.at(0.0) == pytest.approx(0.0)
    # by the time the plan has actually ramped (several seconds) it expects real descent
    assert pd.at(5.0) * 1000.0 > 5.0
    assert pd.at(100.0) == pytest.approx(pd.total_descent)   # clamps to the total after the plan ends


def test_planned_descent_smoothstep_is_monotone_within_segment():
    pd = PlannedDescent((0.0, 2.0, 4.0), (0.0, 0.05, 0.10))
    prev = -1.0
    for i in range(41):
        v = pd.at(i * 0.1)
        assert v >= prev - 1e-12
        prev = v
    assert pd.at(1.0) == pytest.approx(0.05 * 0.5, abs=1e-9)   # smoothstep(0.5)=0.5 => half the segment


# ------------------------------------------------------------------ descent-begun fallback mode

def test_descent_begun_suppresses_false_start_until_grace():
    """descent-begun arming: a frozen carton at the start does NOT stall until the descent has begun or the
    ramp grace has elapsed. With an explicit 1 s grace, the 83 ms window never arms."""
    g = _seat_gate(stall_arming='descent-begun', stall_arm_grace_s=1.0)
    decs = _run_frozen(g, ticks=10)
    assert all(d.action == 'press' for d in decs)


def test_descent_begun_arms_after_measured_descent():
    """Once the carton has actually descended >= begun_m the legacy no-progress stall arms; a later freeze
    above the seat is OBSTRUCTED."""
    g = _seat_gate(stall_arming='descent-begun', stall_begun_m=0.002, stall_arm_grace_s=1000.0,
                   release_z=0.600, seat_center_z=0.600)
    g.reset(release_z=0.600, seat_center_z=0.600)
    # descend 5 mm (arms descent-begun), then freeze above the seat
    z_seq = [0.700, 0.698, 0.696, 0.695] + [0.695] * 8   # 5 mm descent then frozen at +95 mm above seat
    decs = [g.observe(now=100.0 + i * TICK_DT, carton_z=z, commanded_down=True, arm_effort_max=120.0)
            for i, z in enumerate(z_seq)]
    assert any(d.action == 'obstructed' for d in decs), [d.action for d in decs]


def test_descent_begun_arms_after_grace_for_total_jam():
    """A genuine jam from the very start that never moves is still caught: after the ramp grace elapses the
    no-progress stall arms (so a total start jam is not left to the effort trigger alone)."""
    g = _seat_gate(stall_arming='descent-begun', stall_arm_grace_s=0.05,   # tiny grace for the test
                   release_z=0.600, seat_center_z=0.600)
    g.reset(release_z=0.600, seat_center_z=0.600)
    decs = [g.observe(now=100.0 + i * TICK_DT, carton_z=0.700, commanded_down=True, arm_effort_max=120.0)
            for i in range(12)]
    assert any(d.action == 'obstructed' for d in decs)


def test_progress_lag_without_profile_falls_back_to_descent_begun():
    """progress-lag with NO planned profile armed degrades to the descent-begun fallback (the documented
    behaviour when the planned progress is not available at the decider)."""
    g = _seat_gate(stall_arming='progress-lag', planned=None, stall_arm_grace_s=1.0)
    assert g._planned is None
    decs = _run_frozen(g, ticks=10)     # within the 1 s grace, not armed -> press
    assert all(d.action == 'press' for d in decs)


# ------------------------------------------------------------------ config resolve + validation

def test_resolve_arming_keys():
    lc = cl.resolve_level_config({'place_release_on_contact': True,
                                  'place_release_stall_arming': 'progress-lag',
                                  'place_release_stall_lag_m': 0.004,
                                  'place_release_stall_begun_m': 0.003,
                                  'place_release_stall_arm_grace_s': 2.5})
    assert lc.place_release_stall_arming == 'progress-lag'
    assert lc.place_release_stall_lag_m == pytest.approx(0.004)
    assert lc.place_release_stall_begun_m == pytest.approx(0.003)
    assert lc.place_release_stall_arm_grace_s == pytest.approx(2.5)


def test_resolve_arming_default_is_none_byte_identical():
    lc = cl.resolve_level_config({'place_release_on_contact': True})
    assert lc.place_release_stall_arming is None
    assert lc.place_release_stall_arm_grace_s is None


@pytest.mark.parametrize('bad', ['always', 'lag', 'begun', 'on'])
def test_validate_rejects_bad_arming(bad):
    with pytest.raises(ValueError):
        cl.resolve_level_config({'place_release_on_contact': True, 'place_release_stall_arming': bad})


def test_validate_rejects_arming_without_release_on_contact():
    with pytest.raises(ValueError):
        cl.resolve_level_config({'place_release_stall_arming': 'progress-lag'})


@pytest.mark.parametrize('bad', [{'place_release_stall_lag_m': 0.0},
                                 {'place_release_stall_lag_m': -1},
                                 {'place_release_stall_begun_m': 0.0},
                                 {'place_release_stall_arm_grace_s': 0.0},
                                 {'place_release_stall_arm_grace_s': -1}])
def test_validate_rejects_nonpositive_thresholds(bad):
    spec = {'place_release_on_contact': True, 'place_release_stall_arming': 'descent-begun'}
    spec.update(bad)
    with pytest.raises(ValueError):
        cl.resolve_level_config(spec)


def test_gate_rejects_unknown_arming():
    with pytest.raises(ValueError):
        PlaceReleaseGate(stall_arming='nope')


# ------------------------------------------------------------------ controller boot-path WIRING

PROFILE = ProductProfile('sku-box', (0.45, 0.30, 0.18), (0.5, 0.5, 0.5, 0.5),
                         max_reach=1.3, grasp_mode='top')


class _NullMotion:
    def poll(self):
        return 'running'


def _controller(**carton_level_over):
    carton = {'place_from_above': True, 'place_release_on_contact': True,
              'place_seat_gate': True, 'place_pad_down_arrival': True,
              'place_release_stall_arming': 'progress-lag'}
    carton.update(carton_level_over)
    lc = cl.resolve_level_config(carton)
    return TransferController(_NullMotion(), {'sku-box': PROFILE}, fixture_mode=True,
                              carton_level=lc, collect_place_straight_line=True)


def test_controller_constructs_gate_with_arming(capsys):
    """Boot path: the gate is constructed with the HOST-configured arming (never silently inert), and a
    named PLACE_STALL_ARMING line is printed for the run worker to grep."""
    c = _controller()
    assert c._place_release_gate is not None
    assert c._place_release_gate.stall_arming == 'progress-lag'
    out = capsys.readouterr().out
    assert 'PLACE_STALL_ARMING armed=True mode=progress-lag' in out


def test_controller_arming_off_default_byte_identical(capsys):
    """Arming absent => the gate's stall_arming is None (the legacy raw stall) and the boot line names it
    legacy -- byte-identical."""
    c = _controller(place_release_stall_arming=None)
    assert c._place_release_gate is not None
    assert c._place_release_gate.stall_arming is None
    out = capsys.readouterr().out
    assert 'PLACE_STALL_ARMING armed=False mode=legacy' in out


def test_controller_descent_begun_wiring():
    c = _controller(place_release_stall_arming='descent-begun', place_release_stall_begun_m=0.004)
    assert c._place_release_gate.stall_arming == 'descent-begun'
    assert c._place_release_gate.stall_begun_m == pytest.approx(0.004)


def test_gate_reset_planned_reaches_decider():
    """The plan progress reaches the decider: reset(planned=...) arms the gate's planned model, and a
    subsequent observe reads it (progress-lag compares against it)."""
    pd = PlannedDescent((0.0, 3.5, 41.5), (0.0, 0.008, 0.096))
    g = _seat_gate(stall_arming='progress-lag')
    assert g._planned is None
    g.reset(release_z=CW100_SEAT_CZ, seat_center_z=CW100_SEAT_CZ, planned=pd)
    assert g._planned is pd
    # a frozen tick well inside the first segment never trips (sub-mm planned, ~0 lag)
    d = g.observe(now=0.05, carton_z=CW100_START_Z, commanded_down=True, arm_effort_max=120.0)
    assert d.action == 'press'


# ------------------------------------------------------------------ FK knot builder (real kinematics)

CW100_URDF = '/home/manishsaini/datasets/twin-robot-in-store-boot-v1/mission/coldwin100/mobile-import.urdf'
need_urdf = pytest.mark.skipif(not os.path.exists(CW100_URDF), reason='coldwin100 URDF not present')


@need_urdf
def test_place_descent_time_knots_real_kinematics():
    """place_descent_time_knots builds the (knot_times, knot_descents) the controller feeds the gate from
    the REAL robot FK: a 12-waypoint descent (TCP lowered ~0.09 m) yields a monotone non-negative descent
    reaching ~0.09 m, the first-segment time is the rest-to-rest ramp, and a PlannedDescent on it is sub-mm
    in the startup window."""
    from twin.urdf_probe import parse_urdf
    from twin.restock_manipulation.cad_tool import ARM_JOINTS
    robot = parse_urdf(CW100_URDF)
    tool_link = 'ur_tool0'
    # a standoff config and a straight vertical descent of 0.09 m to the set-down, as a 12-waypoint path
    standoff = {j: v for j, v in zip(ARM_JOINTS, CW100_WPS[0])}
    tcp0 = cl.tool_position(robot, [standoff[j] for j in ARM_JOINTS], tool_link)
    target = tcp0.copy()
    target[2] -= 0.09
    sd = cl.level_place_ik(robot, target, cl.carton_up_in_tool(cl.tool_rotation(
        robot, [standoff[j] for j in ARM_JOINTS], tool_link)),
        [standoff[j] for j in ARM_JOINTS], tool_link=tool_link)
    sdv = [sd[j] for j in ARM_JOINTS]
    s0 = [standoff[j] for j in ARM_JOINTS]
    waypoints = [[(1 - a) * s0[k] + a * sdv[k] for k in range(6)]
                 for a in [i / 11.0 for i in range(12)]]
    kt, kd = cl.place_descent_time_knots(robot, waypoints, speed=0.020, accel=0.040, tool_link=tool_link)
    assert len(kt) == len(kd) == 12
    assert kt[0] == 0.0 and all(b > a for a, b in zip(kt, kt[1:]))      # strictly increasing times
    assert kd[0] == pytest.approx(0.0, abs=1e-6)
    assert all(b >= a - 1e-9 for a, b in zip(kd, kd[1:]))               # monotone non-decreasing descent
    assert kd[-1] == pytest.approx(0.09, abs=0.01)                      # reaches ~the planned drop
    pd = PlannedDescent(tuple(kt), tuple(kd))
    assert pd.at(0.083) * 1000.0 < 1.0                                  # sub-mm in the 83 ms startup window
