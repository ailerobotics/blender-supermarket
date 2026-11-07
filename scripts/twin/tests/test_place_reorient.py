"""CPU unit tests for the three-stage collect place set-down (sim-place-pad-down-arrival-v1).

Stage (a) standoff math + stage (b) pad-down-branch selection are PURE (``place_reorient``); the FK-heavy
``carton_level`` calls the selector makes are monkeypatched here so the ranking/refuse LOGIC is tested with
no URDF. The LevelConfig validation + parsing is pure. The full URDF/usd-core chain is proven by
``scripts/check_place_pad_down_arrival.py`` (guarded test at the bottom).

    PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=scripts python3 -m pytest \
        scripts/twin/tests/test_place_reorient.py
"""
import importlib.util
import math
import os
from pathlib import Path

import pytest

from twin.restock_manipulation import carton_level as cl
from twin.restock_manipulation import place_reorient as pr
from twin.restock_manipulation.cad_tool import ARM_JOINTS


# ------------------------------------------------------------------ stage (a): the shifted standoff
def test_mouth_out_unit_points_from_the_seat_toward_the_base():
    # seat at +x of the base -> mouth-out is -x (toward the arm base, into the open approach corridor)
    d = pr.mouth_out_unit((0.683, 0.101, 0.634))
    assert d[2] == 0.0
    assert math.isclose(math.hypot(d[0], d[1]), 1.0, rel_tol=1e-9)
    assert d[0] < 0 and abs(d[1]) < abs(d[0])          # mostly -x, slight -y
    # it is exactly -unit(seat_xy)
    n = math.hypot(0.683, 0.101)
    assert d[0] == pytest.approx(-0.683 / n) and d[1] == pytest.approx(-0.101 / n)


def test_mouth_out_unit_degenerate_raises():
    with pytest.raises(ValueError, match='base origin'):
        pr.mouth_out_unit((0.0, 0.0, 0.5))


def test_reorient_standoff_tcp_shifts_toward_base_and_raises():
    seat = (0.683, 0.101, 0.634)
    std = pr.reorient_standoff_tcp(seat, 0.15, 0.10)
    # z raised by the height above the seat
    assert std[2] == pytest.approx(0.734)
    # xy shifted toward the base by exactly shift along mouth-out
    d = pr.mouth_out_unit(seat)
    assert std[0] == pytest.approx(seat[0] + d[0] * 0.15)
    assert std[1] == pytest.approx(seat[1] + d[1] * 0.15)
    assert math.hypot(std[0], std[1]) < math.hypot(seat[0], seat[1])   # closer to the base


# ------------------------------------------------------------------ config: validation + parsing
def test_pad_down_requires_place_from_above():
    with pytest.raises(ValueError, match='place_from_above'):
        cl.resolve_level_config({'place_pad_down_arrival': True, 'place_from_above': False})


def test_pad_down_requires_level_place():
    with pytest.raises(ValueError, match='level_place'):
        cl.resolve_level_config({'place_pad_down_arrival': True, 'place_from_above': True,
                                 'level_place': False})


def test_pad_down_parses_fields():
    lc = cl.resolve_level_config({'place_pad_down_arrival': True, 'place_from_above': True,
                                  'pad_down_arrival_shift_m': 0.18, 'pad_down_arrival_height_m': 0.12,
                                  'pad_down_reorient_waypoints': 16, 'pad_down_reorient_speed': 0.05,
                                  'pad_down_reorient_accel': 0.09})
    assert lc.place_pad_down_arrival is True
    assert lc.pad_down_arrival_shift_m == pytest.approx(0.18)
    assert lc.pad_down_arrival_height_m == pytest.approx(0.12)
    assert lc.pad_down_reorient_waypoints == 16
    assert lc.pad_down_reorient_speed == pytest.approx(0.05)
    assert lc.pad_down_reorient_accel == pytest.approx(0.09)


def test_pad_down_defaults_and_off():
    assert cl.resolve_level_config(True).place_pad_down_arrival is False       # default off
    lc = cl.resolve_level_config({'place_pad_down_arrival': True, 'place_from_above': True})
    assert lc.pad_down_arrival_shift_m == pytest.approx(0.20)
    assert lc.pad_down_arrival_height_m == pytest.approx(0.10)
    assert lc.pad_down_reorient_waypoints == 12


@pytest.mark.parametrize('bad', [{'pad_down_arrival_shift_m': 0.0}, {'pad_down_arrival_height_m': -1},
                                 {'pad_down_reorient_speed': 0.0}, {'pad_down_reorient_waypoints': 1},
                                 {'pad_down_reorient_waypoints': True}])
def test_pad_down_rejects_bad_fields(bad):
    with pytest.raises(ValueError):
        cl.resolve_level_config(dict({'place_pad_down_arrival': True, 'place_from_above': True}, **bad))


# ------------------------------------------------------------------ stage (b): branch selection logic (pure)
class _FakePlan:
    def __init__(self, tilt):
        self.max_roll = tilt
        self.max_pitch = 0.0
        # a per-branch marker so the fakes can map plan -> branch for the gate/clearance
        self.waypoints = ((tilt,) * 6,)


def _install_fakes(monkeypatch, branches, tilt_by_w2):
    """enumerate returns the given branch joint-maps; plan_compensated_place_descent returns a plan whose
    tilt is keyed by the branch wrist_2 (so the gate/clearance can read it back from the waypoint marker)."""
    monkeypatch.setattr(cl, 'enumerate_level_place_branches',
                        lambda *a, **k: [dict(zip(ARM_JOINTS, b)) for b in branches])

    def _plan(robot, start, target, seed, up, gc, **k):
        w2 = round(float(start[4]), 3)
        return _FakePlan(tilt_by_w2[w2])
    monkeypatch.setattr(cl, 'plan_compensated_place_descent', _plan)


def test_select_pad_down_ranks_by_clearance(monkeypatch):
    # three cage-clear branches (by wrist_2): clearances 0.06, 0.12, 0.09 -> pick the 0.12 one
    branches = [[0, -1, 1, 0, 0.6, 0], [0, -1, 1, 0, 1.6, 0], [0, -1, 1, 0, 1.1, 0]]
    tilt_by_w2 = {0.6: 1.2, 1.6: 1.28, 1.1: 1.24}
    _install_fakes(monkeypatch, branches, tilt_by_w2)
    clear_by_tilt = {1.2: 0.06, 1.28: 0.12, 1.24: 0.09}
    gate = lambda wps: None                                             # all cage-clear
    clear_fn = lambda wps: clear_by_tilt[round(wps[0][0], 3)]
    chosen, rep = pr.select_pad_down_branch(
        object(), (0.5, 0.1, 0.7), [0, 0, 1], [0] * 6, seat_tcp=(0.68, 0.1, 0.63),
        gravity_comp=object(), bound_deg=2.0, descent_cage_gate=gate, descent_cage_clearance=clear_fn)
    assert chosen['ur_wrist_2_joint'] == pytest.approx(1.6)             # the max-clearance branch
    assert rep['chosen_descent_clearance_m'] == pytest.approx(0.12)
    assert rep['n_feasible'] == 3


def test_select_pad_down_discards_cage_rakers(monkeypatch):
    # the 0.6 branch rakes the cage (gate rejects); the 1.6 branch is clear -> pick 1.6
    branches = [[0, -1, 1, 0, 0.6, 0], [0, -1, 1, 0, 1.6, 0]]
    tilt_by_w2 = {0.6: 1.2, 1.6: 1.28}
    _install_fakes(monkeypatch, branches, tilt_by_w2)
    gate = lambda wps: 'RAKE' if round(wps[0][0], 3) == 1.2 else None   # reject the 0.6 branch's descent
    chosen, rep = pr.select_pad_down_branch(
        object(), (0.5, 0.1, 0.7), [0, 0, 1], [0] * 6, seat_tcp=(0.68, 0.1, 0.63),
        gravity_comp=object(), bound_deg=2.0, descent_cage_gate=gate, descent_cage_clearance=None)
    assert chosen['ur_wrist_2_joint'] == pytest.approx(1.6)
    assert rep['n_feasible'] == 1


def test_select_pad_down_raises_when_all_rake(monkeypatch):
    branches = [[0, -1, 1, 0, 0.6, 0], [0, -1, 1, 0, 1.6, 0]]
    _install_fakes(monkeypatch, branches, {0.6: 1.2, 1.6: 1.28})
    gate = lambda wps: 'RAKE'                                           # every descent rakes
    with pytest.raises(ValueError, match='no pad-down branch'):
        pr.select_pad_down_branch(object(), (0.5, 0.1, 0.7), [0, 0, 1], [0] * 6,
                                  seat_tcp=(0.68, 0.1, 0.63), gravity_comp=object(), bound_deg=2.0,
                                  descent_cage_gate=gate)


def test_select_pad_down_discards_over_bound_tilt(monkeypatch):
    # the 0.6 branch descent tilts 3.0 deg (> bound) even though cage-clear -> discard; pick 1.6
    branches = [[0, -1, 1, 0, 0.6, 0], [0, -1, 1, 0, 1.6, 0]]
    _install_fakes(monkeypatch, branches, {0.6: 3.0, 1.6: 1.28})
    chosen, rep = pr.select_pad_down_branch(
        object(), (0.5, 0.1, 0.7), [0, 0, 1], [0] * 6, seat_tcp=(0.68, 0.1, 0.63),
        gravity_comp=object(), bound_deg=2.0, descent_cage_gate=lambda wps: None)
    assert chosen['ur_wrist_2_joint'] == pytest.approx(1.6)


def test_select_pad_down_needs_gravity_comp():
    with pytest.raises(ValueError, match='GravityCompensation'):
        pr.select_pad_down_branch(object(), (0.5, 0.1, 0.7), [0, 0, 1], [0] * 6,
                                  seat_tcp=(0.68, 0.1, 0.63), gravity_comp=None)


# ------------------------------------------------------------------ full URDF/usd-core chain (guarded)
_URDF = Path.home() / 'datasets/twin-robot-in-store-boot-v1/host-cad/mobile.urdf'
_XRDF = Path.home() / 'datasets/twin-robot-in-store-boot-v1/host-cad/mobile.xrdf'


def _usd():
    try:
        import pxr  # noqa: F401
        return True
    except Exception:
        return False


@pytest.mark.skipif(not (_URDF.is_file() and _XRDF.is_file() and _usd()),
                    reason='dataset URDF/XRDF or usd-core absent')
def test_full_chain_clears_the_cage_at_every_park():
    """The production three-stage chain (select_pad_down_branch + plan_level_path_to_goal + the descent)
    clears the cage + stays level + reaches the seat at every cw64-cw92 park, and the old single-standoff
    descent is still refused -- the CPU proof tool returns 0."""
    spec = importlib.util.spec_from_file_location(
        'check_place_pad_down_arrival',
        Path(__file__).resolve().parents[2] / 'check_place_pad_down_arrival.py')
    chk = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(chk)
    assert chk.main(['--shift', '0.20', '--height', '0.10', '--margin', '0.03']) == 0
