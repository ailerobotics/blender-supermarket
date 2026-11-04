"""CPU tests for the collect set-down DROOP-AWARE fix (sim-place-descent-residual-v2).

Grounded in the measured coldwin39/coldwin41 commanded joint PATHS (integration worker's
cw39-vs-cw41-droop.txt). The sweep + selector + plan gate need ``usd-core`` (cart geometry) and the
dataset mobile URDF, so those skip where absent; the config-resolve + servo-bound tests are pure.

    PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=scripts python3 -m pytest \
        scripts/twin/tests/test_place_descent_residual_v2.py
"""
import math
from pathlib import Path

import pytest

from twin.restock_manipulation import carton_level as cl

pxr = pytest.importorskip('pxr')  # noqa: F841 -- author_cart needs usd-core

from twin.restock_manipulation import place_clearance as pc  # noqa: E402
from twin.restock_manipulation import place_select as ps     # noqa: E402

_URDF = Path.home() / 'datasets/twin-robot-in-store-boot-v1/mobile-casterlift/mobile_ur10.urdf'
TCP = 'gripper_pad_tcp'

# Measured commanded configs (TRAJ order elbow,shoulder_lift,pan,w1,w2,w3 -> ARM order).
_CW39_PLACE = (1.5833, -1.2932, -0.094, -0.2526, 1.6056, 0.0001)      # coldwin39 place-end (exec tilt 2.02)
_CW41_ABOVE = (1.4951, -1.4719, -0.0522, 0.0087, 1.6011, -0.0005)     # coldwin41 above-place (trip 4.94)


def _traj_to_arm(t):
    el, sl, pan, w1, w2, w3 = t
    return [pan, sl, el, w1, w2, w3]


@pytest.fixture(scope='module')
def robot():
    if not _URDF.is_file():
        pytest.skip('dataset mobile URDF not present')
    from twin.urdf_probe import parse_urdf
    return parse_urdf(str(_URDF))


@pytest.fixture(scope='module')
def geom():
    return pc.GripperCartonGeometry.shipped()


@pytest.fixture(scope='module')
def bodies():
    return pc.rollwagen_collider_boxes()


# ---------------------------------------------------- part 1: droop model fit against the measured runs

def test_droop_model_reproduces_cw39_executed_tilt(robot):
    """K=2850 (UNCHANGED default) reproduces coldwin39's measured set-down EXECUTED pitch 2.02 deg at the
    real place reach 1.086 m (model 2.16, residual ~0.14 deg)."""
    gc = cl.GravityCompensation().validate()
    cfg = _traj_to_arm(_CW39_PLACE)
    assert cl.reach_xy(robot, cfg, tool_link=TCP) == pytest.approx(1.086, abs=0.02)
    _, pitch, tilt = cl.droop_induced_rpy(robot, cfg, gc, tool_link=TCP)
    assert tilt == pytest.approx(2.02, abs=0.25)          # measured executed 2.02
    assert pitch < 0                                      # droops nose-down


def test_droop_grows_with_reach(robot):
    """The droop tilt grows monotonically with reach -- the basis for the least-extended lever."""
    gc = cl.GravityCompensation().validate()
    far = cl.droop_induced_rpy(robot, _traj_to_arm(_CW39_PLACE), gc, tool_link=TCP)[2]
    # the above-place config is less extended in xy but the measured 4.94 is droop (1.94) + post pivot (~3)
    near = cl.droop_induced_rpy(robot, _traj_to_arm(_CW41_ABOVE), gc, tool_link=TCP)[2]
    assert far > near                                     # reach 1.086 droops more than 1.004


def test_cw41_residual_is_post_pivot_not_droop(robot):
    """coldwin41's 4.94 deg is NOT pure droop: the K=2850 model gives ~1.94 at that config, so ~3 deg is the
    post-edge pivot from the mouth-drift (documented residual)."""
    gc = cl.GravityCompensation().validate()
    tilt = cl.droop_induced_rpy(robot, _traj_to_arm(_CW41_ABOVE), gc, tool_link=TCP)[2]
    assert tilt < 2.5 and 4.94 - tilt > 2.0


# ---------------------------------------------------- part 1: drooped sweep reproduces the cw41 clip

def test_drooped_sweep_reproduces_cw41_post2_clip_vs_level_89mm(robot, geom, bodies):
    """The EXECUTED (drooped, FK) sweep shows the above-place pose STRIKES post_2 (gap < 0, within margin),
    while the v1 idealised LEVEL sweep at the configured load-centre reads ~89 mm clear -- proving the
    level 89 mm is the WRONG quantity."""
    anchor = _traj_to_arm(_CW39_PLACE)
    gc = cl.GravityCompensation().validate()
    Tbc = pc.base_from_cart_anchor(robot, anchor, geom, load_center_xy=(0.18, 0.0), support_z=0.9325,
                                   yaw_cart_rel=-math.pi / 2)
    cal = pc.sweep_executed([anchor], robot, geom, bodies, Tbc, gravity_comp=None)[1]['post_2']
    # the drooped executed sweep on the measured above-place config strikes post_2 (<= margin)
    _, drooped = pc.sweep_executed([_traj_to_arm(_CW41_ABOVE)], robot, geom, bodies, Tbc,
                                   gravity_comp=gc, cal_offset_m=cal)
    assert drooped['post_2'] < pc.DEFAULT_MARGIN_M        # STRIKE / within the 30 mm margin
    # the v1 idealised LEVEL sweep of the carton box at (0.18, 0) reads post_2 ~ +89 mm (the wrong quantity)
    new = pc.place_descent_poses((0.18, 0.0), 0.9325, -math.pi / 2, carton_h=pc.CARTON_DIMS_M[2])
    _, level = pc.sweep_clearance(new, geom, pc.rollwagen_collider_boxes())
    assert level['post_2'] > 0.080                        # ~89 mm "clear" -- describes a pose never held


def test_cw41_load_center_drift_toward_mouth(robot, geom):
    """The above-place IK DRIFTS the carton ~93 mm toward the +x mouth (cart ~0.27 vs the intended 0.18) --
    the coldwin41 regression mechanism, branch-exact from FK."""
    anchor = _traj_to_arm(_CW39_PLACE)
    Tbc = pc.base_from_cart_anchor(robot, anchor, geom, load_center_xy=(0.18, 0.0), support_z=0.9325,
                                   yaw_cart_rel=-math.pi / 2)
    assert pc.load_center_in_cart(robot, anchor, geom, Tbc)[0] == pytest.approx(0.18, abs=0.01)
    drift_x = pc.load_center_in_cart(robot, _traj_to_arm(_CW41_ABOVE), geom, Tbc)[0]
    assert drift_x > 0.25                                 # drifted toward the mouth


# ---------------------------------------------------- part 2: least-extended selection

def test_least_extended_reduces_reach_and_droop(robot, geom, bodies):
    anchor = _traj_to_arm(_CW39_PLACE)
    best, cands = ps.select_least_extended(robot, geom, bodies, anchor, tilt_target_deg=1.5)
    assert best is not None                               # a feasible least-extended pick exists
    assert best.reach_m < 1.086                           # less extended than coldwin39
    assert best.droop_tilt_deg <= 1.5                     # predicted droop under the target
    assert best.load_center_drift_mm < 50.0               # carton stays over the load-centre (no mouth drift)
    # the anchor (coldwin39) itself is over the bound -- the selection is a strict improvement
    anchor_c = next(c for c in cands if c.load_center_dx == 0 and c.base_closer_m == 0)
    assert anchor_c.droop_tilt_deg > 2.0


# ---------------------------------------------------- part 1/3: plan gate still passes at K=2850

def test_plan_gate_is_honest_about_reach(robot, geom):
    """The carton-level plan gate (plan_compensated_place_descent, K=2850 UNCHANGED) REJECTS the far
    coldwin39-reach descent (the real droop ~2.15 > 2.0) and PASSES the less-extended fixed-shape standoff
    -- i.e. the gate is honest and the fix is reach reduction, not a changed droop stiffness."""
    import math as _m
    import numpy as np
    gc = cl.GravityCompensation().validate()
    # far: the coldwin39 place reach 1.086 -> the gate rejects (droop over the bound)
    far_seed = _traj_to_arm(_CW39_PLACE)
    up = cl.carton_up_in_tool(cl.tool_rotation(robot, far_seed, TCP))
    tcp0 = cl.tool_position(robot, far_seed, TCP)
    far = cl.plan_compensated_place_descent(
        robot, far_seed, tuple(tcp0 - np.array([0.0, 0.0, 0.12])), far_seed, up, gc, bound_deg=2.0,
        tool_link=TCP, waypoints=8, approach_from_above=True, above_clearance_m=0.08)
    assert far.reject(2.0) is not None and far.max_pitch > 2.0
    # less-extended: the fixed preplace shape (reach ~0.74) passes with the SAME model (v1 result)
    near_seed = [0.093, -1.91, 2.014, -0.103, 1.664, 0.0]
    up2 = cl.carton_up_in_tool(cl.tool_rotation(robot, near_seed, TCP))
    tcp1 = cl.tool_position(robot, near_seed, TCP)
    near = cl.plan_compensated_place_descent(
        robot, near_seed, tuple(tcp1 - np.array([0.0, 0.0, 0.12])), near_seed, up2, gc, bound_deg=2.0,
        tool_link=TCP, waypoints=8, approach_from_above=True, above_clearance_m=0.08)
    assert near.reject(2.0) is None


# ---------------------------------------------------- part 3: closed-loop backstop bounded + converging

def test_closed_loop_servo_converges_and_is_bounded(robot):
    """The closed-loop wrist servo drives the EXECUTED droop tilt under 1.5 deg on the coldwin39 descent and
    never exceeds its authority (brief part 3, proven offline by replay)."""
    gc = cl.GravityCompensation().validate()
    cfg = _traj_to_arm(_CW39_PLACE)
    up = cl.carton_up_in_tool(cl.tool_rotation(robot, cfg, TCP))
    clc = cl.ClosedLoopConfig(authority_deg=3.0, tool_link=TCP)
    servo = cl.CartonLevelServo(robot, up, clc)
    no_servo, with_servo = servo.replay([cfg] * 30, gc)
    assert no_servo[0][2] > 2.0                           # uncompensated droop tripped
    assert with_servo[-1][2] < 1.5                        # servo pulls it under the acceptance
    assert max(abs(v) for v in servo.accumulated.values()) <= math.radians(3.0) + 1e-9


def test_closed_loop_authority_caps_a_large_residual(robot):
    """The servo cannot exceed its authority: a disturbance larger than the authority is reduced by at most
    the authority, never over-corrected (it is a bounded backstop, not the primary clearance)."""
    gc = cl.GravityCompensation(stiffness=1200).validate()   # exaggerated droop
    cfg = _traj_to_arm(_CW39_PLACE)
    up = cl.carton_up_in_tool(cl.tool_rotation(robot, cfg, TCP))
    servo = cl.CartonLevelServo(robot, up, cl.ClosedLoopConfig(authority_deg=3.0, tool_link=TCP))
    no_servo, with_servo = servo.replay([cfg] * 40, gc)
    assert with_servo[-1][2] < no_servo[0][2]            # improved
    assert max(abs(v) for v in servo.accumulated.values()) <= math.radians(3.0) + 1e-9


# ---------------------------------------------------- config resolve (pure, no URDF)

def test_v2_config_keys_default_off():
    lc = cl.resolve_level_config(True)
    assert lc.place_clearance_drooped is False
    assert lc.closed_loop_level is False
    assert lc.closed_loop_authority_deg == pytest.approx(3.0)


def test_v2_config_keys_resolve():
    lc = cl.resolve_level_config({'place_clearance_drooped': True, 'closed_loop_level': True,
                                  'closed_loop_authority_deg': 2.5, 'closed_loop_gain': 0.8})
    assert lc.place_clearance_drooped is True and lc.closed_loop_level is True
    assert lc.closed_loop_authority_deg == pytest.approx(2.5)
    assert lc.closed_loop_gain == pytest.approx(0.8)


@pytest.mark.parametrize('bad', [{'closed_loop_authority_deg': 0}, {'closed_loop_rate_deg_per_s': -1},
                                 {'closed_loop_gain': 0}, {'closed_loop_gain': 3.0}])
def test_v2_config_rejects_bad(bad):
    bad = dict(bad); bad['closed_loop_level'] = True
    with pytest.raises(ValueError):
        cl.resolve_level_config(bad)
