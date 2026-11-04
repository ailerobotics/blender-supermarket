"""CPU tests for the collect place-on-cart clearance sweep + the preplace-above-place remedy
(sim-place-descent-residual-v1).

The clearance sweep + cart geometry need ``usd-core`` (``author_cart``); ``from_mission`` + the plan gate
also need the dataset mobile URDF (and shelfbot source), so those are skipped where absent. The config
resolve / config_preflight tests are pure.

    PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=scripts python3 -m pytest \
        scripts/twin/tests/test_place_clearance.py
"""
import math
from pathlib import Path

import numpy as np
import pytest

from twin.restock_manipulation import carton_level as cl

pxr = pytest.importorskip('pxr')  # noqa: F841 -- author_cart needs usd-core

from twin.restock_manipulation import place_clearance as pc  # noqa: E402

_URDF = Path.home() / 'datasets/twin-robot-in-store-boot-v1/mobile-casterlift/mobile_ur10.urdf'
_SHELFBOT = Path.home() / 'workspaces/shelfbot_newgripper'
CARTON_H = pc.CARTON_DIMS_M[2]
SUPPORT_Z = 0.9325
PLACE = (0.18, 0.0)


@pytest.fixture(scope='module')
def bodies():
    return pc.rollwagen_collider_boxes()


@pytest.fixture(scope='module')
def geom():
    return pc.GripperCartonGeometry.shipped()


# --------------------------------------------------------------- the sweep + refuse gate

def test_rollwagen_bodies_include_posts_supports_drawbar(bodies):
    names = set(bodies)
    assert {'post_0', 'post_1', 'post_2', 'post_3', 'deck', 'shelf', 'drawbar'} <= names
    assert any(n.startswith('side_') for n in names)
    assert any(n.startswith('back_') for n in names)


def test_old_over_mouth_standoff_strikes_post_2(bodies, geom):
    """coldwin39 replay: the fixed over-mouth preplace standoff (cart (0.243,-0.185), carton yaw -97 deg)
    jams the gripper pad on post_2 -- the clearance check reproduces the contact and REFUSES it."""
    old = [pc.carton_pose_cart((0.243, -0.185), SUPPORT_Z, math.radians(-97.0), carton_h=CARTON_H)]
    _, per_body = pc.sweep_clearance(old, geom, bodies)
    assert per_body['post_2'] < 0.030            # within (indeed inside) the 30 mm margin
    assert per_body['post_2'] < 0.001            # flush/penetrating -- the SEAL_DIAG min_gap 0.0 contact
    reject = pc.clearance_reject(per_body)
    assert reject is not None and 'post_2' in reject


def test_new_above_place_descent_clears_all_non_support(bodies, geom):
    """The remedy: a preplace standoff DIRECTLY ABOVE the place load-centre + a vertical descent clears
    every non-support Rollwagen body by a wide margin; only the support (shelf) touches."""
    new = pc.place_descent_poses(PLACE, SUPPORT_Z, -math.pi / 2, carton_h=CARTON_H, standoff_m=0.12, n=12)
    per_wp, per_body = pc.sweep_clearance(new, geom, bodies)
    assert pc.clearance_reject(per_body) is None
    nonsupport = [v for k, v in per_body.items() if k not in pc.SUPPORT_BODIES]
    assert min(nonsupport) >= pc.DEFAULT_MARGIN_M
    assert min(nonsupport) > 0.080               # CPU-proven ~89 mm
    # every waypoint clears, not just the endpoints
    for g in per_wp:
        assert pc.clearance_reject(g) is None


def test_place_target_itself_clears_post_2(bodies, geom):
    """The PLACE target (0.18,0) is already clear of post_2 by ~90 mm -- the failure was the STANDOFF,
    not the place target (so the remedy moves the standoff, not the place)."""
    place = [pc.carton_pose_cart(PLACE, SUPPORT_Z, -math.pi / 2, carton_h=CARTON_H)]
    _, per_body = pc.sweep_clearance(place, geom, bodies)
    assert per_body['post_2'] > 0.080


def test_support_contact_is_allowed(bodies, geom):
    """A ~0 gap to the SUPPORT (shelf/deck) is NOT refused -- the carton lands on it."""
    place = [pc.carton_pose_cart(PLACE, SUPPORT_Z, -math.pi / 2, carton_h=CARTON_H)]
    _, per_body = pc.sweep_clearance(place, geom, bodies)
    assert per_body['shelf'] < 0.030             # touching the support
    assert pc.clearance_reject(per_body) is None  # but not refused


def test_planted_post_inside_margin_is_refused(bodies, geom):
    """A planted obstacle inside the clearance margin of the (otherwise clear) new descent is refused."""
    new = pc.place_descent_poses(PLACE, SUPPORT_Z, -math.pi / 2, carton_h=CARTON_H, standoff_m=0.12, n=12)
    planted = dict(bodies)
    planted['planted'] = (np.array([0.10, -0.06, 1.00]), np.array([0.14, -0.02, 1.12]))
    _, per_body = pc.sweep_clearance(new, geom, planted)
    reject = pc.clearance_reject(per_body)
    assert reject is not None and 'planted' in reject


def test_margin_boundary(bodies, geom):
    """A body just OUTSIDE the margin passes; the same body nudged just inside is refused."""
    new = pc.place_descent_poses(PLACE, SUPPORT_Z, -math.pi / 2, carton_h=CARTON_H, standoff_m=0.12, n=4)
    _, per_body = pc.sweep_clearance(new, geom, bodies)
    nonsupport_min = min(v for k, v in per_body.items() if k not in pc.SUPPORT_BODIES)
    assert pc.clearance_reject(per_body, margin=nonsupport_min - 1e-4) is None
    assert pc.clearance_reject(per_body, margin=nonsupport_min + 1e-3) is not None


# --------------------------------------------------------------- geometry provenance

@pytest.mark.skipif(not _URDF.is_file() or not (_SHELFBOT / 'shelfbot').is_dir(),
                    reason='dataset mobile URDF or shelfbot source not on this box')
def test_from_mission_matches_baked_constant():
    g = pc.GripperCartonGeometry.from_mission(_URDF, _SHELFBOT)
    assert np.allclose(g.pad_center, pc.PAD_CENTER_IN_CARTON, atol=1e-5)
    assert np.allclose(g.pad_size, pc.PAD_SIZE_M, atol=1e-6)
    assert np.allclose(np.array(g.pad_R), np.array(pc.PAD_R_IN_CARTON), atol=1e-5)


# --------------------------------------------------------------- plan-gate honesty (K=2850 unchanged)

@pytest.mark.skipif(not _URDF.is_file(), reason='dataset mobile URDF not on this box')
def test_plan_gate_passes_with_unchanged_droop_model():
    """The carton-level PLAN gate (plan_compensated_place_descent, K=2850) evaluates the NEW vertical
    descent and predicts a tilt well under the 2.0 bound (so the gate still passes)."""
    from twin.urdf_probe import parse_urdf
    robot = parse_urdf(str(_URDF))
    gc = cl.GravityCompensation().validate()
    assert gc.stiffness == pytest.approx(cl.DEFAULT_DROOP_STIFFNESS)   # 2850, unchanged
    tcp = 'gripper_pad_tcp'
    seed = [0.093, -1.91, 2.014, -0.103, 1.664, 0.0]
    up = cl.carton_up_in_tool(cl.tool_rotation(robot, seed, tcp))
    standoff_tcp = cl.tool_position(robot, seed, tcp)
    place_tcp = standoff_tcp - np.array([0.0, 0.0, 0.12])
    big = 2 * math.pi
    limits = {'ur_elbow_joint': (0.1, math.pi), 'ur_wrist_1_joint': (-big, 0.65),
              'ur_wrist_2_joint': (0.0, big)}
    from twin.restock_manipulation.cad_tool import ARM_JOINTS
    standoff_cfg = cl.level_place_ik(robot, standoff_tcp, up, seed, tool_link=tcp, limits=limits,
                                     gravity_comp=gc)
    start = [float(standoff_cfg[j]) for j in ARM_JOINTS]
    plan = cl.plan_compensated_place_descent(robot, start, tuple(place_tcp), start, up, gc,
                                             bound_deg=2.0, tool_link=tcp, limits=limits, waypoints=8,
                                             approach_from_above=True, above_clearance_m=0.08)
    assert plan.reject(2.0) is None
    assert max(plan.max_pitch, plan.max_roll) < 0.5


# --------------------------------------------------------------- config resolve

def test_config_keys_default_off():
    lc = cl.resolve_level_config(True)
    assert lc.preplace_above_place is False
    assert lc.place_clearance_check is False
    assert lc.preplace_standoff_m == pytest.approx(0.12)
    assert lc.place_clearance_margin_m == pytest.approx(0.030)


def test_config_keys_resolve():
    lc = cl.resolve_level_config({'preplace_above_place': True, 'preplace_standoff_m': 0.15,
                                  'place_clearance_check': True, 'place_clearance_margin_m': 0.04})
    assert lc.preplace_above_place is True and lc.place_clearance_check is True
    assert lc.preplace_standoff_m == pytest.approx(0.15)
    assert lc.place_clearance_margin_m == pytest.approx(0.04)


@pytest.mark.parametrize('bad', [{'preplace_standoff_m': 0}, {'preplace_standoff_m': -0.1},
                                 {'place_clearance_margin_m': -0.01},
                                 {'preplace_standoff_m': float('nan')}])
def test_config_rejects_bad(bad):
    with pytest.raises(ValueError):
        cl.resolve_level_config(bad)


# --------------------------------------------------------------- config_preflight

def test_preflight_passes_v2_clean_block():
    # sim-place-descent-residual-v2 production block: the droop-aware gate + closed-loop backstop, with the
    # regressing v1 preplace_above_place OFF.
    from twin.restock_manipulation import config_preflight as cp
    host = {'carton_level': {'place_clearance_drooped': True, 'closed_loop_level': True,
                             'gravity_compensation': {'stiffness_nm_per_rad': 2850}}}
    assert cp.check_carton_level(host) == []


def test_preflight_refuses_preplace_above_place_on():
    # sim-place-descent-residual-v2: the v1 remedy regresses the set-down, so it is refused.
    from twin.restock_manipulation import config_preflight as cp
    host = {'carton_level': {'preplace_above_place': True, 'place_clearance_check': True}}
    problems = cp.check_carton_level(host)
    assert problems and any('preplace_above_place' in p and 'REGRESSES' in p for p in problems)


def test_preflight_catches_level_gate_without_remedy():
    from twin.restock_manipulation import config_preflight as cp
    host = {'carton_level': {'place_clearance_check': True}}   # level gate armed, no remedy
    problems = cp.check_carton_level(host)
    assert problems and 'no remedy' in problems[0]


def test_preflight_catches_malformed_block():
    from twin.restock_manipulation import config_preflight as cp
    host = {'carton_level': {'preplace_standoff_m': -1}}
    problems = cp.check_carton_level(host)
    assert problems and 'malformed' in problems[0]


def test_preflight_noop_when_off():
    from twin.restock_manipulation import config_preflight as cp
    assert cp.check_carton_level({}) == []
    assert cp.check_carton_level({'carton_level': True}) == []
