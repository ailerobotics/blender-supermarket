"""CPU tests for the collect place-DESCENT cage-clearance gate (sim-place-descent-reach-cage-v1).

Pure-logic tests (reject naming, support exclusion, margin, provider inert-off, controller byte-identical
off) run everywhere; the coldwin87 descent geometry proof is guarded on usd-core + the dataset URDF/XRDF
(same pattern as test_preplace_sweep_clearance). No Isaac / ROS / GPU.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from twin.restock_manipulation import place_clearance as pc


# --------------------------------------------------------------------------- pure reject / selection logic
def test_descent_reject_none_when_all_clear():
    per_body = {'side_1_horizontal_6': 0.10, 'post_2': 0.08, 'deck': -0.2, 'shelf': -0.2}
    assert pc.place_descent_sweep_reject(per_body) is None


def test_descent_reject_names_worst_penetrating_cage_body():
    per_body = {'side_1_horizontal_6': -0.0019, 'post_3': -0.0002, 'post_2': 0.05}
    msg = pc.place_descent_sweep_reject(per_body)
    assert msg is not None
    assert msg.startswith(pc.PLACE_DESCENT_CAGE_CLEARANCE)
    assert 'prim=side_1_horizontal_6' in msg               # the WORST (most negative) body first
    assert 'min_clearance=-1.9mm' in msg
    assert 'cube_without_surface_plates' in msg            # the named wedge link


def test_descent_reject_excludes_support_bodies():
    per_body = {'deck': -0.5, 'shelf': -0.5, 'side_1_horizontal_6': 0.1}
    assert pc.place_descent_sweep_reject(per_body) is None


def test_descent_default_margin_is_30mm_design():
    assert pc.DEFAULT_PLACE_DESCENT_CAGE_MARGIN_M == 0.030


def test_descent_margin_is_configurable():
    per_body = {'side_1_horizontal_6': 0.02}               # 20 mm clear: below the 30 mm design margin
    assert pc.place_descent_sweep_reject(per_body) is not None
    assert pc.place_descent_sweep_reject(per_body, margin=0.0) is None   # clears a penetration-only margin


def test_descent_clearance_is_the_same_arm_sphere_sweep_as_preplace():
    # place_descent_cage_clearance is the PR 366 arm-sphere sweep re-exposed for the descent; identity keeps
    # a verdict matching the gate the selector uses.
    assert pc.place_descent_cage_clearance.__module__ == pc.preplace_cage_arm_clearance.__module__


# --------------------------------------------------------------------------- provider contract (no USD/FK)
def _fake_gate(per_body_fn, margin=pc.DEFAULT_PLACE_DESCENT_CAGE_MARGIN_M):
    def _g(waypoints):
        if not waypoints:
            return None
        return pc.place_descent_sweep_reject(per_body_fn(waypoints), margin=margin)
    return _g


def test_provider_gate_empty_waypoints_is_none():
    assert _fake_gate(lambda w: {'side_1_horizontal_6': -0.1})([]) is None


def test_fake_gate_refuses_penetration_passes_clear():
    assert _fake_gate(lambda w: {'side_1_horizontal_6': -0.019})([[0] * 6]) is not None
    assert _fake_gate(lambda w: {'side_1_horizontal_6': 0.10})([[0] * 6]) is None


# --------------------------------------------------------------------------- controller byte-identical off
def test_controller_defaults_leave_descent_gate_inert():
    from twin.restock_manipulation import controller as ctrl
    src = Path(ctrl.__file__).read_text()
    # The injected place-descent cage hook defaults to None (inert => byte-identical); assert the source
    # wires that default and that the dispatch + selector both guard on it, so a future rename can't silently
    # arm the gate or leave the dispatch ungated.
    assert 'self._place_descent_cage_gate = None' in src
    assert 'if self._place_descent_cage_gate is not None:' in src
    # the dispatch safety net and the selector descent-feasibility criterion both reference the hook
    assert src.count('self._place_descent_cage_gate') >= 3


# --------------------------------------------------------------------------- coldwin87 descent proof (guarded)
_URDF = Path.home() / 'datasets/twin-robot-in-store-boot-v1/host-cad/mobile.urdf'
_XRDF = Path.home() / 'datasets/twin-robot-in-store-boot-v1/host-cad/mobile.xrdf'


def _usd_available():
    try:
        import pxr  # noqa: F401
        return True
    except Exception:
        return False


@pytest.mark.skipif(not (_URDF.is_file() and _XRDF.is_file() and _usd_available()),
                    reason='dataset URDF/XRDF or usd-core absent')
def test_coldwin87_descent_old_refused_new_cleared():
    """The cw87 reconstruction flags the executed in-cage-reorient descent at every collect-place park and
    passes the pad-down remedy descent to the corrected seat (clear >= 30 mm, tilt <= 2.0 deg, reaches seat)."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        'check_place_descent_reach',
        Path(__file__).resolve().parents[2] / 'check_place_descent_reach.py')
    chk = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(chk)
    assert chk.main([]) == 0
