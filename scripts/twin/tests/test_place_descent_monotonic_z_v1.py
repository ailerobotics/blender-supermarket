"""CPU tests for the monotonic carton-z descent guard (sim-place-descent-reach-executed-vs-planned-v1).

coldwin103 dispatched the first real pad-down collect set-down since cw98 and it did NOT reach the shelf.
Two facts, both pinned here:

  * the EXECUTED arm TRACKED the dispatched ``TRAJ_HOST_PLAN`` faithfully -- FK of the recorded
    ``CARTON_TICK`` joints vs the plan at matched elapsed time stays within ~0.011 rad per joint, and the
    FK-of-executed carton centre reproduces the recorded ``PLACE_RELEASE_TICK`` carton_z to < 1 mm -- so the
    set-down failure is NOT a tracking failure, a competing command or a mid-way replacement; and
  * the dispatched PLAN itself does not descend: its FK carton-centre z drops 7 mm then RISES to +10.6 mm
    above the standoff over the first 8 of 13 waypoints, holding up through ~t=28 s, and only dives to the
    seat in the last 4. The run was declared ``PLACE_OBSTRUCTED`` at t=28.6 s, at the hump, before the real
    descent ran. The hump is introduced AFTER the base ``plan_compensated_place_descent`` (which is monotonic
    and reaches the seat) by the sensor-anchored closed-loop servo (``closed_loop_sensor``), which saturates
    the +-3 deg wrist authority and swings the ~0.19 m-laterally-offset welded carton up ~10.7 mm.

The fix (opt-in ``carton_level.place_descent_monotonic_z``) caps the correction's upward carton-z excursion
back to the monotonic base descent (``place_clearance.enforce_monotonic_carton_z``), so the dispatched
descent is monotonically non-increasing to the seat. Default off => byte-identical.

    PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=scripts python3 -m pytest \
        scripts/twin/tests/test_place_descent_monotonic_z_v1.py
"""
import math
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from twin.restock_manipulation import carton_level as cl
from twin.restock_manipulation import place_clearance as pc
from twin.restock_manipulation import TransferController, ProductProfile
from twin.restock_manipulation.cad_tool import ARM_JOINTS

PROFILE = ProductProfile('sku-box', (0.45, 0.30, 0.18), (0.5, 0.5, 0.5, 0.5),
                         max_reach=1.3, grasp_mode='top')

# The dataset mobile URDF the coldwin103 host ran (the FK-dependent replays are gated on it).
_URDF = Path.home() / 'datasets/twin-robot-in-store-boot-v1/host-cad/mobile.urdf'
need_urdf = pytest.mark.skipif(not _URDF.is_file(), reason='dataset mobile URDF not on this box')
_TCPLINK = 'gripper_pad_tcp'

# -------------------------------------------------------- recorded coldwin103 evidence (host-stack.log)
# TRAJ_HOST_PLAN names=[elbow, lift, pan, w1, w2, w3], 13 waypoints (line ~30828).
_PLAN_NAMES = ['ur_elbow_joint', 'ur_shoulder_lift_joint', 'ur_shoulder_pan_joint',
               'ur_wrist_1_joint', 'ur_wrist_2_joint', 'ur_wrist_3_joint']
_CW103_PLAN = [
    [1.8096, -1.8693, -0.0703, 0.0586, 1.7243, -0.0005], [1.777, -1.8224, -0.071, 0.0585, 1.7112, -0.0005],
    [1.7421, -1.7758, -0.0722, 0.0394, 1.6977, -0.0004], [1.7047, -1.7294, -0.073, 0.023, 1.6844, -0.0004],
    [1.665, -1.683, -0.0736, 0.009, 1.6715, -0.0003], [1.6231, -1.6367, -0.0741, -0.0028, 1.6588, -0.0002],
    [1.5789, -1.5904, -0.0745, -0.0123, 1.6463, -0.0002], [1.5323, -1.544, -0.0747, -0.0107, 1.6341, -0.0001],
    [1.4832, -1.4973, -0.075, -0.0069, 1.6219, -0.0], [1.4734, -1.4698, -0.0741, -0.0239, 1.6175, 0.0],
    [1.5227, -1.4687, -0.0717, -0.0742, 1.624, 0.0], [1.5687, -1.4643, -0.0696, -0.1247, 1.6298, 0.0],
    [1.6124, -1.4575, -0.0677, -0.1751, 1.6352, -0.0]]
# TRAJ_HOST_PLAN point times (s) + the anchored execution start (TRAJ_HOST_START anchored start=291.883).
_CW103_PLAN_TIMES = [0.0, 3.5175, 7.0084, 10.4918, 13.9684, 17.4418, 20.9166, 24.412, 28.0951,
                     30.1582, 33.9353, 37.7205, 41.5035]
_ANCHOR = 291.883
# CARTON_TICK phase=place  now  q=[pan, lift, elbow, w1, w2, w3]  (ARM order), sampled across the descent.
_CW103_CARTON_TICK = [
    (291.317, [-0.0703, -1.8693, 1.8096, 0.0586, 1.7243, -0.0005]),
    (292.917, [-0.0706, -1.8501, 1.7963, 0.0586, 1.7189, -0.0005]),
    (294.967, [-0.0709, -1.8223, 1.7769, 0.0584, 1.7112, -0.0005]),
    (296.983, [-0.0717, -1.7923, 1.7545, 0.0461, 1.7025, -0.0004]),
    (299.567, [-0.0724, -1.7635, 1.7322, 0.0350, 1.6942, -0.0004]),
]
# PLACE_SEAT_GATE_ARM: support_top 0.4408, carton_half 0.090 -> seat_center_z 0.5308, start_carton_z 0.6396.
_CW103_SEAT_CENTRE_Z = 0.5308
_CW103_START_CARTON_Z = 0.6396
_SEAT_TOL = 0.020


def _plan_arm(p):
    d = dict(zip(_PLAN_NAMES, p))
    return [d[j] for j in ARM_JOINTS]


@pytest.fixture(scope='module')
def robot():
    from twin.urdf_probe import parse_urdf
    return parse_urdf(str(_URDF))


@pytest.fixture(scope='module')
def geom():
    return pc.GripperCartonGeometry.shipped()


def _carton_z(robot, geom, w):
    return pc.carton_centre_z(robot, w, geom)


def _max_rise_mm(zs):
    return 1000.0 * max(0.0, max(zs) - zs[0])


def _is_monotonic(zs, tol=1e-4):
    return all(zs[i + 1] <= zs[i] + tol for i in range(len(zs) - 1))


# =============================================================== pure-logic capping algorithm (no URDF)
# An analytic carton-z: proportional to a "height" dimension carried in the waypoint's 2nd slot, so a test
# can construct a known hump + base and verify the one-sided cap without the FK.
def _analytic_cz(w):
    return float(w[1])


def test_guard_caps_a_hump_back_to_the_monotonic_base():
    # base descends monotonically 0.00 -> -0.12; the "corrected" path humps UP in the middle (the servo swing).
    base = [[0.0, 0.00 - 0.01 * i, 0.0] for i in range(13)]
    final = [list(b) for b in base]
    for i in (3, 4, 5, 6, 7):                         # push those carton-z UP above the start (a +hump)
        final[i][1] = 0.02
    out, rpt = pc.enforce_monotonic_carton_z(None, final, base, None, carton_z_fn=_analytic_cz)
    zs = [_analytic_cz(w) for w in out]
    assert rpt['mode'] == 'capped'
    assert _is_monotonic(zs)
    assert rpt['max_rise_mm_before'] >= 19.5 and rpt['max_rise_mm_after'] <= 1.0
    # reaches the base's final descent (capped to base, not left high)
    assert abs(zs[-1] - _analytic_cz(base[-1])) < 1e-6


def test_guard_is_noop_on_an_already_monotonic_path():
    base = [[0.0, -0.01 * i, 0.0] for i in range(10)]
    out, rpt = pc.enforce_monotonic_carton_z(None, [list(b) for b in base], base, None,
                                             carton_z_fn=_analytic_cz)
    assert rpt['mode'] == 'noop' and rpt['n_adjusted'] == 0
    assert out == [list(b) for b in base]


def test_guard_keeps_a_one_sided_downward_correction():
    # the servo LOWERS some waypoints toward level (carton-z below base) -- the cap must KEEP those.
    base = [[0.0, -0.01 * i, 0.0] for i in range(8)]
    final = [list(b) for b in base]
    final[3][1] = base[3][1] - 0.005                 # a downward correction (below base) -- must be kept
    out, rpt = pc.enforce_monotonic_carton_z(None, final, base, None, carton_z_fn=_analytic_cz)
    assert _is_monotonic([_analytic_cz(w) for w in out])
    assert out[3][1] == pytest.approx(final[3][1])   # kept verbatim, not pulled up to base
    assert rpt['n_adjusted'] == 0 and rpt['mode'] == 'noop'


def test_guard_rejects_length_mismatch_and_short_paths():
    with pytest.raises(ValueError, match='counts differ'):
        pc.enforce_monotonic_carton_z(None, [[0.0]], [[0.0], [0.0]], None, carton_z_fn=_analytic_cz)
    with pytest.raises(ValueError, match='at least two'):
        pc.enforce_monotonic_carton_z(None, [[0.0]], [[0.0]], None, carton_z_fn=_analytic_cz)


# ======================================================= coldwin103 diagnosis: arm tracked the plan
@need_urdf
def test_executed_tracked_the_dispatched_plan(robot, geom):
    """The executed CARTON_TICK joints match the dispatched plan at matched elapsed time (anchored start):
    the arm TRACKED the plan, so the set-down failure is the PLAN, not tracking."""
    plan_arm = [_plan_arm(p) for p in _CW103_PLAN]

    def plan_at(elapsed):
        ts = _CW103_PLAN_TIMES
        if elapsed <= ts[0]:
            return plan_arm[0]
        if elapsed >= ts[-1]:
            return plan_arm[-1]
        for k in range(len(ts) - 1):
            if ts[k] <= elapsed <= ts[k + 1]:
                a = (elapsed - ts[k]) / (ts[k + 1] - ts[k])
                return [(1 - a) * plan_arm[k][j] + a * plan_arm[k + 1][j] for j in range(6)]
        return plan_arm[-1]

    for now, q in _CW103_CARTON_TICK:
        ref = plan_at(max(0.0, now - _ANCHOR))
        err = max(abs(q[j] - ref[j]) for j in range(6))
        assert err < 0.02, f'executed joint error {err:.4f} at now={now} (arm did NOT track the plan)'


@need_urdf
def test_dispatched_cw103_plan_is_non_monotonic(robot, geom):
    """FK the recorded dispatched plan: the carton-centre z RISES above the standoff (the +10.6 mm hump)
    before descending -- the before-state the fix removes."""
    plan_arm = [_plan_arm(p) for p in _CW103_PLAN]
    zs = [_carton_z(robot, geom, w) for w in plan_arm]
    assert not _is_monotonic(zs)
    assert _max_rise_mm(zs) > 8.0                     # the recorded ~10.6 mm hump


# ======================================================= coldwin103 fix: base monotonic, servo humps, guard fixes
def _regen_base_and_hump(robot, geom):
    """Regenerate the coldwin103 BASE descent (monotonic) and the sensor-anchored servo HUMP (the dispatched
    plan's extra), reproducing the recorded shape on CPU."""
    import numpy as np
    plan_arm = [_plan_arm(p) for p in _CW103_PLAN]
    std = plan_arm[0]
    def tcp(a):
        return cl.tool_position(robot, list(a), tool_link=_TCPLINK)
    dxy = np.array([tcp(plan_arm[-1])[0] - tcp(std)[0], tcp(plan_arm[-1])[1] - tcp(std)[1], 0.0])
    raw_tgt = tcp(plan_arm[-1]) - 0.05 * (dxy / float(np.linalg.norm(dxy)))
    gc = cl.GravityCompensation(tool_link='ur_tool0').validate()
    up = cl.carton_up_in_tool(cl.tool_rotation(robot, list(std), tool_link='ur_tool0'))
    base = cl.plan_compensated_place_descent(
        robot, list(std), list(raw_tgt), list(std), up, gc, bound_deg=2.0, tool_link=_TCPLINK,
        limits={'ur_elbow_joint': (0.1, math.pi)}, waypoints=12, approach_from_above=True,
        above_clearance_m=0.08, depth_bias_m=0.05)
    base_wps = [list(w) for w in base.waypoints]
    clc = cl.ClosedLoopConfig(authority_deg=3.0, rate_deg_per_s=30.0, gain=0.6,
                              joints=('ur_wrist_1_joint', 'ur_wrist_2_joint', 'ur_wrist_3_joint'),
                              tool_link=_TCPLINK)
    servo = cl.CartonLevelServo(robot, up, clc)
    humped = servo.correct_path_sensor_anchored([list(w) for w in base_wps], gc, list(base_wps[0]))
    return base_wps, humped


@need_urdf
def test_base_plan_is_monotonic_and_reaches_the_seat(robot, geom):
    base_wps, _ = _regen_base_and_hump(robot, geom)
    zs = [_carton_z(robot, geom, w) for w in base_wps]
    assert _is_monotonic(zs)
    # FK<->measured constant calibration offset (the ~125 mm pad-model bias), from the standoff config.
    off = zs[0] - _CW103_START_CARTON_Z
    final_meas = zs[-1] - off
    assert abs(final_meas - _CW103_SEAT_CENTRE_Z) <= _SEAT_TOL


@need_urdf
def test_servo_reproduces_the_hump(robot, geom):
    _, humped = _regen_base_and_hump(robot, geom)
    zs = [_carton_z(robot, geom, w) for w in humped]
    assert not _is_monotonic(zs)
    assert _max_rise_mm(zs) > 8.0                     # reproduces the recorded hump


@need_urdf
def test_guard_makes_cw103_descent_monotonic_and_reaches_seat(robot, geom):
    base_wps, humped = _regen_base_and_hump(robot, geom)
    fixed, rpt = pc.enforce_monotonic_carton_z(robot, humped, base_wps, geom, z_tol=0.001)
    zs = [_carton_z(robot, geom, w) for w in fixed]
    assert rpt['monotonic_after'] and _is_monotonic(zs)
    assert rpt['max_rise_mm_after'] <= 1.0
    off = _carton_z(robot, geom, base_wps[0]) - _CW103_START_CARTON_Z
    assert abs((zs[-1] - off) - _CW103_SEAT_CENTRE_Z) <= _SEAT_TOL
    # the guard keeps the descent within the tilt bound (it blends toward the within-bound base plan).
    up = cl.carton_up_in_tool(cl.tool_rotation(robot, list(base_wps[0]), tool_link='ur_tool0'))
    gc = cl.GravityCompensation(tool_link='ur_tool0').validate()
    mr, mp, _ = cl.predict_path_tilt_executed(robot, [list(w) for w in fixed], up,
                                              tool_link=_TCPLINK, gravity_comp=gc)
    assert max(mr, mp) <= 2.0


# ================================================================= wiring on/off (boot-path through ros_host)
def test_resolve_defaults_off_byte_identical():
    lc = cl.resolve_level_config({'level_goals': True})
    assert lc.place_descent_monotonic_z is False
    assert lc.place_descent_monotonic_z_tol_m == pytest.approx(0.001)


def test_resolve_on_sets_the_flag_and_tol():
    lc = cl.resolve_level_config({'level_goals': True, 'place_descent_monotonic_z': True,
                                  'place_descent_monotonic_z_tol_m': 0.002})
    assert lc.place_descent_monotonic_z is True
    assert lc.place_descent_monotonic_z_tol_m == pytest.approx(0.002)


def test_resolve_rejects_negative_tol():
    with pytest.raises(ValueError, match='place_descent_monotonic_z_tol_m'):
        cl.resolve_level_config({'level_goals': True, 'place_descent_monotonic_z': True,
                                 'place_descent_monotonic_z_tol_m': -0.001})


class _NullMotion:
    def poll(self):
        return 'running'


def test_controller_receives_the_flag_from_the_resolved_host_config():
    lc = cl.resolve_level_config({'level_goals': True, 'place_descent_monotonic_z': True})
    c = TransferController(_NullMotion(), {'sku-box': PROFILE}, fixture_mode=True, carton_level=lc)
    assert c._carton_level.place_descent_monotonic_z is True
    off = TransferController(_NullMotion(), {'sku-box': PROFILE}, fixture_mode=True,
                             carton_level=cl.resolve_level_config({'level_goals': True}))
    assert off._carton_level.place_descent_monotonic_z is False


def test_ros_host_surfaces_the_key_in_effective_config():
    """ros_host must (a) pass the resolved carton_level (so the flag reaches the controller) and (b) surface
    place_descent_monotonic_z in HOST_EFFECTIVE_CONFIG, so the run worker can grep that the guard armed."""
    from twin.restock_manipulation import ros_host
    src = Path(ros_host.__file__).read_text()
    assert "carton_level=carton_level.resolve_level_config(config.get('carton_level'))" in src
    assert "'place_descent_monotonic_z':" in src
    # the controller logs a named armed line the run worker checks.
    from twin.restock_manipulation import controller as _ctrl
    assert 'PLACE_DESCENT_MONOTONIC_Z' in Path(_ctrl.__file__).read_text()
