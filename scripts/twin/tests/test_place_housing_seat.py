"""CPU tests for the side-grip-housing-compatible pad-down seat transform (sim-place-housing-compatible-seat-v1).

The pad-down collect set-down strikes the Rollwagen shelf board with the gripper suction HOUSING (PR 383).
This lane adds an opt-in seat transform both the pad-down branch selector AND the place descent read:
``housing_seat_z_lift_m`` raises the carton onto the pinned COMPOSED shelf flat top (vs the author_cart
BOX-MODEL top the descent over-drove toward), and ``housing_seat_lip_overhang_m`` shifts the seat mouth-out
toward the +x shelf lip so the housing hangs outboard of the board.

These tests pin:
  * the pure transform (``TransferController._housing_seat_tcp``): off => byte-identical; on => the seat is
    shifted along the mouth-out unit by the overhang AND raised by the z-lift; gated on the pad-down place;
  * the boot-path wiring (``place_cage_wiring.wire_housing_seat``): the 'PLACE_HOUSING_SEAT armed' log fires
    when on (and NOT when off), and it raises loudly on a half-wired config;
  * the selector INTEGRATION: ``_issue_pad_down_reorient`` passes the TRANSFORMED seat to
    ``place_reorient.select_pad_down_branch`` (the chosen geometry reaches the dispatcher).

No Isaac / ROS / GPU / USD.

    PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=scripts python3 -m pytest \
        scripts/twin/tests/test_place_housing_seat.py
"""
import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from twin.restock_manipulation import carton_level as cl
from twin.restock_manipulation import place_cage_wiring as wiring
from twin.restock_manipulation import place_reorient as pr
from twin.restock_manipulation import TransferController, ProductProfile

PROFILE = ProductProfile('sku-box', (0.45, 0.30, 0.18), (0.5, 0.5, 0.5, 0.5),
                         max_reach=1.3, grasp_mode='top')


class _NullMotion:
    def poll(self):
        return 'running'


def _housing_cfg(overhang=0.30, lift=0.0314):
    return cl.resolve_level_config({
        'level_goals': True, 'level_place': True, 'place_from_above': True,
        'place_pad_down_arrival': True, 'place_housing_seat': True,
        'housing_seat_lip_overhang_m': overhang, 'housing_seat_z_lift_m': lift})


def _controller(carton_level=None):
    c = TransferController(_NullMotion(), {'sku-box': PROFILE}, fixture_mode=True,
                           carton_level=carton_level)
    # make _pad_down_arrival_on() / _collect_deck_place() true (the NON-cage collect deck place)
    c.operation = 'collect'
    c.collect_place_straight_line = True
    c.cart_cage_place = False
    return c


# ----------------------------------------------------------------------- resolve_level_config
def test_resolve_defaults_off_byte_identical():
    lc = cl.resolve_level_config({'level_goals': True})
    assert lc.place_housing_seat is False
    assert lc.housing_seat_lip_overhang_m == 0.0 and lc.housing_seat_z_lift_m == 0.0


def test_resolve_requires_pad_down_arrival():
    with pytest.raises(ValueError, match='place_pad_down_arrival'):
        cl.resolve_level_config({'level_goals': True, 'level_place': True, 'place_from_above': True,
                                 'place_housing_seat': True})


def test_resolve_rejects_negative_overhang_and_lift():
    for bad in ('housing_seat_lip_overhang_m', 'housing_seat_z_lift_m'):
        with pytest.raises(ValueError, match='non-negative'):
            cl.resolve_level_config({'level_goals': True, 'level_place': True, 'place_from_above': True,
                                     'place_pad_down_arrival': True, 'place_housing_seat': True, bad: -0.01})


# ----------------------------------------------------------------------- the pure transform
def test_housing_seat_off_is_byte_identical():
    c = _controller(cl.resolve_level_config({'level_goals': True, 'level_place': True,
                                              'place_from_above': True, 'place_pad_down_arrival': True}))
    seat = (0.9, 0.1, 0.5)
    assert c._housing_seat_tcp(seat) == seat


def test_housing_seat_none_carton_level_is_byte_identical():
    c = _controller(None)
    seat = (0.9, 0.1, 0.5)
    assert c._housing_seat_tcp(seat) == seat


def test_housing_seat_applies_overhang_and_lift():
    c = _controller(_housing_cfg(overhang=0.30, lift=0.0314))
    seat = (0.9, 0.1, 0.5)
    out = c._housing_seat_tcp(seat)
    # mouth-out unit = -unit(seat_xy): shifts the seat toward the arm base (= the +x cart/shelf lip),
    # and z is raised by the lift.
    d = pr.mouth_out_unit(seat)
    assert math.isclose(out[0], seat[0] + d[0] * 0.30, abs_tol=1e-9)
    assert math.isclose(out[1], seat[1] + d[1] * 0.30, abs_tol=1e-9)
    assert math.isclose(out[2], seat[2] + d[2] * 0.30 + 0.0314, abs_tol=1e-9)
    # mouth-out is horizontal, so the only z change is the lift; and the seat moves toward the base origin
    assert out[2] > seat[2]
    assert math.hypot(out[0], out[1]) < math.hypot(seat[0], seat[1])  # pulled toward the base (the lip)


def test_housing_seat_lift_only_is_pure_vertical():
    c = _controller(_housing_cfg(overhang=0.0, lift=0.0314))
    seat = (0.9, 0.1, 0.5)
    out = c._housing_seat_tcp(seat)
    assert math.isclose(out[0], 0.9) and math.isclose(out[1], 0.1)
    assert math.isclose(out[2], 0.5 + 0.0314, abs_tol=1e-9)


def test_housing_seat_preserves_orientation_tail():
    c = _controller(_housing_cfg(overhang=0.10, lift=0.02))
    seat = (0.9, 0.1, 0.5, 0.5, 0.5, 0.5, 0.5)
    out = c._housing_seat_tcp(seat)
    assert tuple(out[3:]) == (0.5, 0.5, 0.5, 0.5)


def test_housing_seat_gated_off_when_not_pad_down_place():
    c = _controller(_housing_cfg())
    c.operation = 'transfer'   # not a collect deck place -> _pad_down_arrival_on() False -> untouched
    seat = (0.9, 0.1, 0.5)
    assert c._housing_seat_tcp(seat) == seat


# ----------------------------------------------------------------------- boot-path wiring + armed log
def test_wire_housing_seat_armed_logs_when_on():
    c = _controller(_housing_cfg(overhang=0.30, lift=0.0314))
    lines = []
    armed = wiring.wire_housing_seat(c, log=lambda *a, **k: lines.append(a[0]))
    assert armed is True
    assert any('PLACE_HOUSING_SEAT armed' in ln and 'lip_overhang=0.3000' in ln
               and 'z_lift=0.0314' in ln for ln in lines)


def test_wire_housing_seat_silent_when_off():
    c = _controller(cl.resolve_level_config({'level_goals': True, 'level_place': True,
                                             'place_from_above': True, 'place_pad_down_arrival': True}))
    lines = []
    assert wiring.wire_housing_seat(c, log=lambda *a, **k: lines.append(a[0])) is False
    assert lines == []
    # also byte-identical when carton_level is absent entirely
    assert wiring.wire_housing_seat(_controller(None), log=lambda *a, **k: lines.append(a[0])) is False
    assert lines == []


def test_ros_host_calls_wire_housing_seat_at_boot():
    """The boot path (ros_host.main) must wire the housing seat right after it builds the controller, so the
    'PLACE_HOUSING_SEAT armed' line + the transform are live at boot -- not only in a unit test."""
    from twin.restock_manipulation import ros_host
    src = Path(ros_host.__file__).read_text()
    assert 'from .place_cage_wiring import wire_housing_seat' in src
    assert 'wire_housing_seat(controller)' in src


def test_wire_housing_seat_raises_on_half_wired():
    # a hand-built config with the seat on but pad-down arrival off (bypasses resolve's validate) must be
    # caught loudly at the boot seam, never silently half-wired.
    import types
    c = _controller(types.SimpleNamespace(place_housing_seat=True, place_pad_down_arrival=False,
                                          housing_seat_lip_overhang_m=0.3, housing_seat_z_lift_m=0.0))
    with pytest.raises(ValueError, match='place_pad_down_arrival'):
        wiring.wire_housing_seat(c, log=lambda *a, **k: None)


# --------------------------------------------- the transformed seat reaches the branch selector
def test_issue_pad_down_reorient_passes_transformed_seat_to_selector(monkeypatch):
    """The chosen geometry reaches the dispatcher: _issue_pad_down_reorient calls select_pad_down_branch
    with the HOUSING-COMPATIBLE seat (poses['place'] + overhang + lift), not the raw place pose. Stubbed
    (no robot FK/USD): the selector spy records the seat_tcp it was handed and short-circuits."""
    c = _controller(_housing_cfg(overhang=0.30, lift=0.0314))
    c.phase = 'place'
    c.poses = {'place': (0.9, 0.1, 0.5)}
    c._carton_up_in_tool = (0.0, 0.0, 1.0)
    c.collect_place_preplace_shape = [0.0] * 6
    c._place_descent_cage_gate = lambda w: None
    c._place_descent_cage_min_clearance = lambda w: 0.1
    # stub the FK-dependent helpers so no robot is needed before the selector call
    monkeypatch.setattr(c, '_fk_model', lambda: object())
    monkeypatch.setattr(c, '_place_gravity_compensation', lambda: object())
    monkeypatch.setattr(c, '_place_level_limits', lambda: None)
    monkeypatch.setattr(c, '_place_fk_link', lambda robot, seed: 'gripper_pad_tcp')
    monkeypatch.setattr(cl, 'tool_position', lambda *a, **k: (0.0, 0.0, 0.0))

    seen = {}

    def _spy_select(robot, standoff_tcp, up, seed, *, seat_tcp, **kw):
        seen['seat'] = tuple(seat_tcp)
        raise ValueError('stop after recording the seat')   # caught by the reorient's try/except

    monkeypatch.setattr(pr, 'select_pad_down_branch', _spy_select)

    c._issue_pad_down_reorient(object(), [0.0] * 6)
    expected = c._housing_seat_tcp((0.9, 0.1, 0.5))
    assert seen['seat'] == tuple(expected)
    # and it is NOT the raw place pose (the transform really moved it)
    assert seen['seat'] != (0.9, 0.1, 0.5)


# --------------------------------------------- composed-USD replay (guarded on usd-core + dataset)
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
def test_housing_seat_descent_proof_passes():
    """The CPU proof (scripts/check_housing_seat_descent.py): the z-lift targets the composed flat top, the
    coldwin102 overhang clears the shelf board >= 30 mm, and no stable+clear side-grip seat exists (the
    honest tradeoff). Replays the production pad-down descent against the composed cw100 cart."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        'check_housing_seat_descent',
        Path(__file__).resolve().parents[2] / 'check_housing_seat_descent.py')
    chk = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(chk)
    assert chk.main(['--quick']) == 0
