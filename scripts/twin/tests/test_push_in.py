"""CPU tests for the post-release collect push-in nudge (sim-place-post-release-push-in-v1).

PR 385 lands the side-grip carton on the rollwagen shelf with a mouth-out lip overhang -- a close-to-tipping
rest (CoM ~14 mm inboard of the +x lip, ~47 % base support). This lane adds an opt-in POST-RELEASE push-in:
after the seat gate releases the carton and the gripper backs clear, re-contact the free carton and push it
inboard (cart -x) toward a more centred rest, the stroke CAPPED so the side-grip housing (coaxial with the
pad normal, hangs ~0.14 m below the carton bottom) keeps a clearance to the shelf board -- a bounded nudge,
since a CENTRED rest needs a top-down grip (out of scope).

These tests pin:
  * ``resolve_level_config``: off => byte-identical; ``place_push_in`` requires ``place_housing_seat``; bad
    values rejected;
  * the pure helpers (``classify_result`` done/short, ``push_direction_base``, ``clip_area``);
  * the stroke cap + the whole plan against the composed cw100 cart (guarded on usd-core + dataset);
  * the boot-path wiring (``place_cage_wiring.wire_push_in``): the 'PLACE_PUSH_IN armed' log + raises on a
    half-wired config / missing cart geometry; and ros_host wires it at boot;
  * the PHASE registration + that the GEOMETRY reaches the dispatcher (``_issue_push_in`` hands the wired
    push dir + shelf-clearance closure to ``push_in.plan_push_in``);
  * the bounded retry (``_push_in_complete``: a 'short' carton re-pushes at most ``push_in_retry_max``).

No Isaac / ROS / GPU (the composed-USD replay is guarded).

    PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=scripts python3 -m pytest \
        scripts/twin/tests/test_push_in.py
"""
import math
import sys
import types
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from twin.restock_manipulation import carton_level as cl
from twin.restock_manipulation import place_cage_wiring as wiring
from twin.restock_manipulation import push_in as pin
from twin.restock_manipulation import controller as controller_mod
from twin.restock_manipulation import TransferController, ProductProfile

PROFILE = ProductProfile('sku-box', (0.45, 0.30, 0.18), (0.5, 0.5, 0.5, 0.5),
                         max_reach=1.3, grasp_mode='top')


class _NullMotion:
    def poll(self):
        return 'running'


def _pushin_cfg(**over):
    base = {'level_goals': True, 'level_place': True, 'place_from_above': True,
            'place_pad_down_arrival': True, 'place_housing_seat': True,
            'housing_seat_lip_overhang_m': 0.28, 'housing_seat_z_lift_m': 0.0314,
            'place_push_in': True, 'push_in_target_x_m': 0.0, 'push_in_shelf_clear_min_m': 0.010}
    base.update(over)
    return cl.resolve_level_config(base)


def _controller(carton_level=None):
    c = TransferController(_NullMotion(), {'sku-box': PROFILE}, fixture_mode=True, carton_level=carton_level)
    c.operation = 'collect'
    c.collect_place_straight_line = True
    c.cart_cage_place = False
    return c


# ----------------------------------------------------------------------- resolve_level_config
def test_resolve_defaults_off_byte_identical():
    lc = cl.resolve_level_config({'level_goals': True})
    assert lc.place_push_in is False
    assert lc.push_in_target_x_m == 0.0 and lc.push_in_shelf_clear_min_m == 0.010
    assert lc.push_in_retry_max == 1


def test_resolve_requires_housing_seat():
    with pytest.raises(ValueError, match='place_housing_seat'):
        cl.resolve_level_config({'level_goals': True, 'level_place': True, 'place_from_above': True,
                                 'place_pad_down_arrival': True, 'place_push_in': True})


def test_resolve_rejects_bad_values():
    for bad, val, msg in (('push_in_shelf_clear_min_m', -0.01, 'positive'),
                          ('push_in_speed', 0.0, 'positive'),
                          ('push_in_waypoints', 1, '>= 2'),
                          ('push_in_retry_max', -1, 'non-negative')):
        with pytest.raises(ValueError, match=msg):
            _pushin_cfg(**{bad: val})


def test_resolve_roundtrips_knobs():
    lc = _pushin_cfg(push_in_target_x_m=0.05, push_in_shelf_clear_min_m=0.015, push_in_speed=0.03,
                     push_in_waypoints=8, push_in_retry_max=2, push_in_standoff_m=0.1)
    assert lc.push_in_target_x_m == 0.05 and lc.push_in_shelf_clear_min_m == 0.015
    assert lc.push_in_speed == 0.03 and lc.push_in_waypoints == 8 and lc.push_in_retry_max == 2


# ----------------------------------------------------------------------- pure helpers
def test_classify_result_done_within_settle():
    verdict, reason = pin.classify_result(0.030, 0.028, settle_m=0.010)
    assert verdict == 'done' and 'tracked' in reason


def test_classify_result_short_beyond_settle():
    verdict, reason = pin.classify_result(0.030, 0.009, settle_m=0.010)
    assert verdict == 'short' and 'short' in reason


def test_push_direction_is_cart_minus_x_unit():
    import numpy as np
    # cart-body -> base is identity => cart +x is base +x, so the inboard push (cart -x) is base -x.
    d = pin.push_direction_base(np.eye(4))
    assert math.isclose(float(d[0]), -1.0, abs_tol=1e-9)
    assert math.isclose(float(np.linalg.norm(d)), 1.0, abs_tol=1e-9)


def test_clip_area_full_and_partial():
    # a 0.2 x 0.2 square fully inside a big AABB => full area
    sq = [(-0.1, -0.1), (0.1, -0.1), (0.1, 0.1), (-0.1, 0.1)]
    assert math.isclose(pin.clip_area(sq, (-1, -1), (1, 1)), 0.04, abs_tol=1e-9)
    # half of it hangs off the +x edge at x=0 => half the area
    assert math.isclose(pin.clip_area(sq, (-1, -1), (0.0, 1)), 0.02, abs_tol=1e-9)
    # fully outside => 0
    assert pin.clip_area(sq, (1, 1), (2, 2)) == 0.0


# ----------------------------------------------------------------------- boot-path wiring + armed log
def test_wire_push_in_silent_when_off():
    lines = []
    assert wiring.wire_push_in(_controller(None), {}, '/nonexistent.urdf',
                               log=lambda *a, **k: lines.append(a[0])) is False
    off = _controller(cl.resolve_level_config({'level_goals': True, 'level_place': True,
                                               'place_from_above': True, 'place_pad_down_arrival': True}))
    assert wiring.wire_push_in(off, {}, '/nonexistent.urdf', log=lambda *a, **k: lines.append(a[0])) is False
    assert lines == []


def test_wire_push_in_raises_on_half_wired_housing_seat():
    # place_push_in on but place_housing_seat off (hand-built, bypasses resolve.validate) -> loud at boot.
    c = _controller(types.SimpleNamespace(place_push_in=True, place_housing_seat=False))
    with pytest.raises(ValueError, match='place_housing_seat'):
        wiring.wire_push_in(c, {}, '/nonexistent.urdf', log=lambda *a, **k: None)


def test_wire_push_in_raises_when_cart_geometry_absent():
    # place_push_in on, place_housing_seat on, but NO place_descent_cage_gate block => cannot find the shelf.
    c = _controller(_pushin_cfg())
    with pytest.raises(ValueError, match='place_descent_cage_gate'):
        wiring.wire_push_in(c, {}, '/nonexistent.urdf', log=lambda *a, **k: None)


def test_ros_host_calls_wire_push_in_at_boot():
    from twin.restock_manipulation import ros_host
    src = Path(ros_host.__file__).read_text()
    assert 'from .place_cage_wiring import wire_push_in' in src
    assert 'wire_push_in(controller, config, args.urdf)' in src


def test_boot_report_exposes_place_push_in():
    from twin.restock_manipulation import ros_host
    src = Path(ros_host.__file__).read_text()
    assert "'place_push_in':" in src


# ----------------------------------------------------------------------- phase registered + routing
def test_push_in_phase_is_a_general_move():
    assert 'push_in' in controller_mod._GENERAL_MOVE_PHASES


def test_next_after_release_routes_to_push_in_only_when_active():
    # off => retreat (byte-identical)
    off = _controller(cl.resolve_level_config({'level_goals': True, 'level_place': True,
                                               'place_from_above': True, 'place_pad_down_arrival': True}))
    assert off._next_after_release() == 'retreat'
    # on but geometry NOT wired => still retreat (never armed-but-blind)
    on = _controller(_pushin_cfg())
    assert on._push_in_geometry is None
    assert on._next_after_release() == 'retreat'
    # on AND geometry wired => push_in
    on._push_in_geometry = object()
    assert on._push_in_active() is True
    assert on._next_after_release() == 'push_in'
    # a non-collect-deck place never routes to push_in
    on.operation = 'transfer'
    assert on._next_after_release() == 'retreat'


# ------------------------- the GEOMETRY reaches the dispatcher (_issue_push_in -> plan_push_in)
def test_issue_push_in_hands_wired_geometry_to_planner(monkeypatch):
    """The chosen geometry reaches the dispatcher: ``_issue_push_in`` passes the WIRED push direction +
    shelf-clearance closure (and the target computed from ``carton_com_cart_x``) to ``push_in.plan_push_in``.
    Stubbed (no robot FK/USD): the planner spy records what it was handed and short-circuits."""
    c = _controller(_pushin_cfg(push_in_target_x_m=0.10))
    c.phase = 'push_in'
    c.poses = {'place': (0.9, 0.1, 0.5)}
    c._carton_up_in_tool = (0.0, 0.0, 1.0)
    sentinel_dir = object()
    sentinel_fn = object()
    pg = pin.PushInGeometry(push_dir_base=sentinel_dir, shelf_clear_fn=sentinel_fn,
                            carton_com_cart_x=lambda cfg: 0.3741, carton_support_fraction=lambda cfg: 0.47)
    c._push_in_geometry = pg
    monkeypatch.setattr(c, '_fk_model', lambda: object())
    monkeypatch.setattr(c, '_place_level_limits', lambda: None)
    monkeypatch.setattr(c, '_place_fk_link', lambda robot, seed: 'gripper_pad_tcp')
    # avoid the housing-seat transform needing _pad_down_arrival_on's real path (keep seat = place pose)
    monkeypatch.setattr(c, '_housing_seat_tcp', lambda seat: tuple(seat))
    monkeypatch.setattr(cl, 'level_place_ik', lambda *a, **k: {j: 0.0 for j in
                        __import__('twin.restock_manipulation.cad_tool', fromlist=['ARM_JOINTS']).ARM_JOINTS})

    seen = {}

    def _spy_plan(robot, seat_tcp, up, start, push_dir_base, *, target_stroke_m, shelf_clear_fn, **kw):
        seen['push_dir'] = push_dir_base
        seen['shelf_fn'] = shelf_clear_fn
        seen['target_stroke'] = target_stroke_m
        raise ValueError('stop after recording')

    monkeypatch.setattr(pin, 'plan_push_in', _spy_plan)
    c._issue_push_in(object(), [0.0] * 6, types.SimpleNamespace(placed_pose=None))
    assert seen['push_dir'] is sentinel_dir
    assert seen['shelf_fn'] is sentinel_fn
    # target stroke = seat CoM cart-x (0.3741) - target_x (0.10)
    assert math.isclose(seen['target_stroke'], 0.3741 - 0.10, abs_tol=1e-6)


# ----------------------------------------------------------------------- bounded retry
def _complete_controller(retry_max=1):
    c = _controller(_pushin_cfg(push_in_retry_max=retry_max))
    c.phase = 'push_in'
    c.target_surface = (0.0, 0.0, 0.9639)
    c._push_in_geometry = pin.PushInGeometry(
        push_dir_base=None, shelf_clear_fn=None,
        carton_com_cart_x=lambda cfg: 0.344, carton_support_fraction=lambda cfg: 0.55)
    c._push_in_expected_shift = 0.030
    c._push_in_last_config = [0.0] * 6
    c._carton_up_in_tool = None   # => tilt reported 0.0, no FK needed
    return c


def test_push_in_short_retries_once_then_advances():
    c = _complete_controller(retry_max=1)
    # carton moved only 5 mm vs 30 mm expected => short. before/after 25 mm apart (world xy).
    c._push_in_com_before = [1.000, 2.000]
    s_short = types.SimpleNamespace(placed_pose=(1.005, 2.000, 0.96))
    # attempt 0 -> short, retry (phase stays push_in, issued False, retries=1)
    c._push_in_complete(0.0, s_short)
    assert c._push_in_retries == 1
    assert c.phase == 'push_in'
    assert c.issued is False
    # attempt 1 -> short again, budget exhausted => advance to retreat anyway (no infinite retry)
    c._push_in_complete(0.1, s_short)
    assert c.phase == 'retreat'


def test_push_in_done_advances_without_retry():
    c = _complete_controller(retry_max=1)
    c._push_in_com_before = [1.000, 2.000]
    # carton moved ~30 mm (world xy) == expected => done, no retry, advance to retreat.
    s_done = types.SimpleNamespace(placed_pose=(1.030, 2.000, 0.96))
    c._push_in_complete(0.0, s_done)
    assert c._push_in_retries == 0
    assert c.phase == 'retreat'


def test_push_in_missing_pose_trusts_model_and_advances():
    c = _complete_controller(retry_max=1)
    c._push_in_com_before = None
    c._push_in_complete(0.0, types.SimpleNamespace(placed_pose=None))
    assert c.phase == 'retreat'   # no measured pose => trust the model (on-target) => advance


# ----------------------------------------------------------------------- composed-USD replay (guarded)
_URDF = Path.home() / 'datasets/twin-robot-in-store-boot-v1/host-cad/mobile.urdf'
_XRDF = Path.home() / 'datasets/twin-robot-in-store-boot-v1/host-cad/mobile.xrdf'
_MISSION = Path.home() / 'datasets/twin-robot-in-store-boot-v1/mission/coldwin100/mission.usda'


def _usd_available():
    try:
        import pxr  # noqa: F401
        return True
    except Exception:
        return False


@pytest.mark.skipif(not (_URDF.is_file() and _XRDF.is_file() and _MISSION.is_file() and _usd_available()),
                    reason='dataset URDF/XRDF, coldwin100 mission.usda, or usd-core absent')
def test_push_in_descent_proof_passes():
    """The CPU proof (scripts/check_push_in_descent.py): the stroke is capped to keep the housing off the
    shelf board, the arm clears cage + deck by >= 30 mm, the carton stays level, and its CoM moves inboard
    off the +x lip -- the bounded shelf-safe nudge, against the composed cw100 cart."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        'check_push_in_descent', Path(__file__).resolve().parents[2] / 'check_push_in_descent.py')
    chk = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(chk)
    assert chk.main(['--quick']) == 0
