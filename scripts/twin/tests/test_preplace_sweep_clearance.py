"""CPU tests for the carried pre-place leg-2 cage-clearance gate (sim-preplace-sweep-cage-clearance-v1).

Pure-logic tests (reject naming, support exclusion, pan-first reorder, provider inert-off) run everywhere;
the coldwin70 geometry proof is guarded on usd-core + the dataset URDF/XRDF (same pattern as
test_cart_world_obstacle). No Isaac / ROS / GPU.
"""
import math
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from twin.restock_manipulation import place_clearance as pc
from twin.restock_manipulation import preplace_sequence as ps


# --------------------------------------------------------------------------- pure reject / selection logic
def test_reject_none_when_all_clear():
    per_body = {'post_2': 0.10, 'side_0_horizontal_6': 0.08, 'deck': -0.2, 'shelf': -0.2}
    assert pc.preplace_sweep_reject(per_body, margin=0.0) is None


def test_reject_names_worst_penetrating_cage_body():
    per_body = {'post_2': -0.048, 'side_0_horizontal_6': -0.029, 'post_3': 0.05}
    msg = pc.preplace_sweep_reject(per_body, margin=0.0)
    assert msg is not None
    assert msg.startswith(pc.PREPLACE_SWEEP_CLEARANCE)
    assert 'prim=post_2' in msg               # the WORST (most negative) body is named first
    assert 'min_clearance=-48.0mm' in msg
    assert 'side_0_horizontal_6=-29.0mm' in msg


def test_reject_excludes_support_bodies():
    # deck/shelf are the surfaces the carton lands on -- the preplace approach rides above them, so a
    # negative reading there must NOT refuse.
    per_body = {'deck': -0.5, 'shelf': -0.5, 'post_2': 0.1}
    assert pc.preplace_sweep_reject(per_body, margin=0.0) is None


def test_reject_margin_is_configurable():
    per_body = {'post_2': 0.02}               # 20 mm clear
    assert pc.preplace_sweep_reject(per_body, margin=0.0) is None
    assert pc.preplace_sweep_reject(per_body, margin=0.04) is not None   # below a 40 mm requirement


def test_default_margin_is_penetration():
    assert pc.DEFAULT_PREPLACE_CAGE_MARGIN_M == 0.0
    assert pc.PREPLACE_CAGE_CLEAR_TARGET_M == 0.040


def test_cage_body_names_selects_posts_sides_backs_only():
    bodies = {'deck': None, 'shelf': None, 'drawbar': None, 'post_2': None,
              'side_0_horizontal_6': None, 'side_1_vertical_3': None, 'back_horizontal_1': None}
    cage = set(pc.cage_body_names(bodies))
    assert cage == {'post_2', 'side_0_horizontal_6', 'side_1_vertical_3', 'back_horizontal_1'}
    assert 'deck' not in cage and 'shelf' not in cage and 'drawbar' not in cage


# --------------------------------------------------------------------------- pan-first reorder (pure)
def test_pan_first_holds_everything_but_pan_at_carry():
    carry = [1.2678, -2.27, 1.0, 0.65, 0.0019, 0.6191]
    standoff = [-0.4053, -1.4856, 1.4788, 0.0348, 0.6626, -0.0329]
    leg1, leg2 = ps.pan_first_subgoals(carry, standoff)
    # leg1 = standoff PAN (index 0), carry everything else; leg2 = standoff verbatim.
    assert leg1[0] == pytest.approx(standoff[0])
    assert list(leg1[1:]) == pytest.approx(list(carry[1:]))
    assert list(leg2) == pytest.approx(list(standoff))


def test_pan_first_is_inverse_of_wrist_first():
    carry = [1.0, -2.0, 1.1, 0.6, 0.0, 0.5]
    standoff = [-0.4, -1.5, 1.5, 0.03, 0.66, -0.03]
    wrist1, _ = ps.wrist_first_subgoals(carry, standoff)
    pan1, _ = ps.pan_first_subgoals(carry, standoff)
    # wrist-first holds ONLY pan at carry; pan-first holds everything BUT pan at carry.
    assert wrist1[0] == pytest.approx(carry[0]) and list(wrist1[1:]) == pytest.approx(list(standoff[1:]))
    assert pan1[0] == pytest.approx(standoff[0]) and list(pan1[1:]) == pytest.approx(list(carry[1:]))


def test_pan_first_rejects_bad_input():
    with pytest.raises(ValueError):
        ps.pan_first_subgoals([1, 2, 3], [1, 2, 3, 4, 5, 6])


# --------------------------------------------------------------------------- provider inert / active
def test_provider_gate_empty_waypoints_is_none():
    # A provider built with a trivial cart pose returns None for an empty path without touching geometry.
    gate = _fake_gate(lambda wps: {'post_2': -0.1})
    assert gate([]) is None


def _fake_gate(per_body_fn, margin=0.0):
    """A provider-shaped closure over a fake per-body function, to test the gate contract without USD/FK."""
    def _g(waypoints):
        if not waypoints:
            return None
        return pc.preplace_sweep_reject(per_body_fn(waypoints), margin=margin)
    return _g


def test_fake_gate_refuses_penetration_passes_clear():
    assert _fake_gate(lambda w: {'post_2': -0.048})([[0] * 6]) is not None
    assert _fake_gate(lambda w: {'post_2': 0.05})([[0] * 6]) is None


# --------------------------------------------------------------------------- controller byte-identical off
def test_controller_defaults_leave_gate_inert():
    import inspect
    from twin.restock_manipulation import controller as ctrl
    sig = inspect.signature(ctrl.TransferController.__init__)
    assert sig.parameters['preplace_pan_first'].default is False
    # The injected cage-clearance hook defaults to None (inert => byte-identical); assert the source wires
    # that default so a future rename can't silently arm the gate.
    src = Path(ctrl.__file__).read_text()
    assert 'self._preplace_cage_clearance = None' in src


# --------------------------------------------------------------------------- coldwin70 geometry proof (guarded)
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
def test_coldwin70_old_refused_new_cleared():
    """The cw70 reconstruction flags the executed wrist-first leg-2 and passes the pan-first reorder."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        'check_preplace_sweep_clearance',
        Path(__file__).resolve().parents[2] / 'check_preplace_sweep_clearance.py')
    chk = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(chk)
    assert chk.main(['--margin', '0.0']) == 0
