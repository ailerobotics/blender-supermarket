"""sim-robust-collect-stow-branch-v1 -- CPU proof (no Isaac/ROS/GPU) that the collect deck RETREAT escapes
the self-colliding place fold to a clear standoff, so the stow-to-transport after the carton release plans
without cuMotion's self-collision rejection.

Root cause (collect5, ~/datasets/twin-robot-in-store-boot-v1/mission/collect5/host-stack.log; write-up
data/twin-robot-in-store-boot-v1/collect5-pr217-evidence.md): PR #217 routes the collect deck PLACE descent
and empty RETREAT collision-OFF straight-line (the set-down fold is a conservative attached-carton
self-collision). But with collision-avoidance off, the empty-retreat pose IK branches into a DROOPED config
where the forearm sits ~18 mm inside the gripper cube (``cube_without_surface_plates``) sphere margin. The
next move is the collision-AWARE staged ``retreat_tuck``; cuMotion rejects that self-colliding START
(``INVALID_INITIAL_CSPACE_POSITION`` -- ``Invalid c-space position: self collision detected``), the host
raises ``host emergency stop`` and the run ends on ``arm command heartbeat lost``.

The logged failing start (``initial_cspace_position``) is reconstructed here on the real URDF + the cuMotion
sphere model (the mission XRDF), the way test_restock_cart_place_goal reconstructs the deck poses, and the
fix is checked at the sphere level: the FIXED upright escape shape (collect_place_preplace_shape) is
self-clear, the collision-off straight-line JOINT smoothstep from the failing start to that shape un-folds
MONOTONICALLY out of the cube margin (no waypoint worse than the self-colliding start), and the subsequent
staged ``retreat_tuck`` goal and the transport ``stow`` goal are self-clear -- so every move AFTER the
collision-off escape plans from a clean start. The controller-tier routing + fail-by-name assertions live in
test_restock_manipulation.py (test_collect_deck_place_routes_place_and_retreat_straight_line /
test_collect_deck_retreat_fails_by_name_without_a_clear_escape_shape).
"""
import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

SCRIPTS = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(SCRIPTS))

from twin import urdf_probe as U  # noqa: E402

FIX = Path(__file__).resolve().parent / 'fixtures'
URDF = FIX / 'collect_stow_arm.urdf'
SPHERES = FIX / 'collect_stow_spheres.json'

# The dataset sources the committed fixtures were cut from (parity check; skipped when absent).
RUN_URDF = Path('/home/manishsaini/datasets/twin-robot-in-store-boot-v1/mission/collect5/mobile-import.urdf')
RUN_XRDF = Path('/home/manishsaini/datasets/twin-robot-in-store-boot-v1/host-cad/mobile.xrdf')

# collect5 HOST_EFFECTIVE_CONFIG (host-stack.log) -- the mission's own values.
RETREAT_ENDPOINT = (0.213638, -1.75545, 2.0753, -0.319821, 1.97496, 1.18688e-06)  # the rejected start
UPRIGHT_ESCAPE = (0.093, -1.910, 2.014, -0.103, 1.664, 0.0)   # collect_place_preplace_shape
CART_TUCK_SHAPE = (-2.27, 1.0, 0.65, 0.8, -1.2545)            # retreat_tuck folds arm joints 1-5 to this
TRANSPORT = (0.0, -2.5, 1.7, -0.7, -1.5708, 0.0)             # the XRDF transport/stow nominal


@pytest.fixture(scope='module')
def model():
    robot = U.parse_urdf(str(URDF))
    fx = json.loads(SPHERES.read_text())
    ignore = {frozenset(p) for p in fx['ignore']}
    return robot, fx['spheres'], fx['arm_joints'], ignore


def _worst_self_clearance(model, q):
    """The minimum sphere-to-sphere clearance (m) over every non-ignored link pair at arm config ``q``,
    with the colliding pair. Negative == self-collision (what cuMotion rejects as INVALID_CSPACE). This is
    the mission XRDF ``mobile_cad`` sphere model placed by the real URDF forward kinematics."""
    robot, spheres, arm, ignore = model
    cfg = {n: float(v) for n, v in zip(arm, q)}
    world = U.link_world_transforms(robot, cfg, root='ur_base_link')
    placed = {}
    for link, slist in spheres.items():
        if link not in world:
            continue
        T = world[link]
        placed[link] = [((T @ np.array([s[0], s[1], s[2], 1.0]))[:3], float(s[3])) for s in slist]
    links = list(placed)
    worst, pair = math.inf, None
    for i in range(len(links)):
        for j in range(i + 1, len(links)):
            la, lb = links[i], links[j]
            if frozenset((la, lb)) in ignore:
                continue
            d = min(float(np.linalg.norm(ca - cb) - ra - rb)
                    for ca, ra in placed[la] for cb, rb in placed[lb])
            if d < worst:
                worst, pair = d, (la, lb)
    return worst, pair


def test_fixtures_match_the_dataset_when_present(model):
    """The committed URDF + sphere fixtures are a verbatim cut of the mission's own URDF and XRDF
    (self_collision ignore + mobile_cad spheres). Parity guards against a stale fixture."""
    if not (RUN_URDF.is_file() and RUN_XRDF.is_file()):
        pytest.skip('dataset mobile-import.urdf / mobile.xrdf not present')
    import yaml
    real = U.parse_urdf(str(RUN_URDF))
    for q in (UPRIGHT_ESCAPE, RETREAT_ENDPOINT):
        a = U.link_world_transforms(real, {n: float(v) for n, v in zip(model[2], q)}, root='ur_base_link')
        b = U.link_world_transforms(model[0], {n: float(v) for n, v in zip(model[2], q)}, root='ur_base_link')
        for link in model[1]:
            if link in a and link in b:
                assert np.allclose(a[link], b[link], atol=1e-9), link
    x = yaml.safe_load(RUN_XRDF.read_text())
    assert set(x['geometry']['mobile_cad']['spheres']) == set(model[1])


def test_logged_retreat_start_is_self_colliding(model):
    """Reproduce the collect5 failure: the collision-off straight-line retreat's drooped endpoint (the
    rejected initial_cspace_position) self-collides -- the forearm inside the gripper-cube sphere margin."""
    worst, pair = _worst_self_clearance(model, RETREAT_ENDPOINT)
    assert worst < 0, f'expected the logged retreat start to self-collide, got {worst:.4f} m'
    assert set(pair) == {'ur_forearm_link', 'cube_without_surface_plates'}, pair


def test_upright_escape_shape_is_self_clear(model):
    """The FIXED clear escape (collect_place_preplace_shape) the retreat backs the empty gripper to is
    self-collision-free, so the next collision-aware move starts from a valid c-space position."""
    worst, pair = _worst_self_clearance(model, UPRIGHT_ESCAPE)
    assert worst > 0, f'the upright escape shape must be self-clear, got {worst:.4f} m ({pair})'


def test_straight_line_retreat_escape_unfolds_monotonically_to_clear(model):
    """The fix: a collision-OFF straight-line JOINT smoothstep from the self-colliding retreat start to the
    upright escape shape. Every waypoint un-folds OUT of the cube margin (clearance never worse than the
    self-colliding start), and the endpoint is self-clear -- so an in-sim execution of the smoothstep does
    not drive the forearm deeper into the gripper, and ``retreat_tuck`` then plans from a clean start."""
    start = np.array(RETREAT_ENDPOINT)
    goal = np.array(UPRIGHT_ESCAPE)
    clearances = []
    for u in np.linspace(0.0, 1.0, 21):
        worst, _ = _worst_self_clearance(model, start + (goal - start) * u)
        clearances.append(worst)
    # monotonic non-decreasing clearance (un-folding), within a tiny numeric tolerance
    for a, b in zip(clearances, clearances[1:]):
        assert b >= a - 1e-4, f'clearance regressed along the escape: {a:.4f} -> {b:.4f}'
    assert clearances[-1] > 0, f'escape endpoint must be self-clear, got {clearances[-1]:.4f} m'
    assert clearances[0] < 0, 'the start must be the self-colliding fold (else there is nothing to fix)'


def test_staged_tuck_and_transport_goals_are_self_clear(model):
    """After the clean escape, the staged retreat_tuck (arm joints 1-5 folded to cart_tuck_shape at the
    escape pan) and the transport stow goal are both self-collision-free, so the collision-aware plans
    from the escape shape have valid endpoints."""
    retreat_tuck_goal = (UPRIGHT_ESCAPE[0],) + CART_TUCK_SHAPE
    for name, q in (('retreat_tuck', retreat_tuck_goal), ('transport', TRANSPORT)):
        worst, pair = _worst_self_clearance(model, q)
        assert worst > 0, f'{name} goal must be self-clear, got {worst:.4f} m ({pair})'
