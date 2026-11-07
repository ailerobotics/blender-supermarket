"""CPU tests for the pad-down place gripper-housing-vs-SHELF jam model fix (sim-pad-down-descent-jam-physics-v1).

Pure-logic tests (the support-board sweep set + the two-margin reject) run everywhere; the coldwin100
geometry replay against the COMPOSED mission USD is guarded on usd-core + the dataset URDF/XRDF + the
coldwin100 mission.usda (same guard pattern as test_place_descent_reach). No Isaac / ROS / GPU.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from twin.restock_manipulation import place_clearance as pc


# --------------------------------------------------------------------------- pure: the support-board sweep set
def test_sweep_bodies_adds_support_to_cage():
    bodies = {'post_2': ((0, 0, 0), (1, 1, 1)), 'side_1_horizontal_6': ((0, 0, 0), (1, 1, 1)),
              'shelf': ((0, 0, 0), (1, 1, 1)), 'deck': ((0, 0, 0), (1, 1, 1)),
              'drawbar': ((0, 0, 0), (1, 1, 1))}
    sweep = pc.place_descent_sweep_bodies(bodies)
    # cage bodies kept, in order, then the support board(s) appended; drawbar (neither cage nor support)
    # stays out (the gripper never nears the tow bar at the pad-down descent).
    assert pc.cage_body_names(bodies) == ['post_2', 'side_1_horizontal_6']
    assert sweep[:2] == ['post_2', 'side_1_horizontal_6']
    assert set(sweep) == {'post_2', 'side_1_horizontal_6', 'shelf', 'deck'}
    assert 'drawbar' not in sweep


def test_sweep_bodies_no_support_present_is_cage_only():
    bodies = {'post_2': ((0, 0, 0), (1, 1, 1))}
    assert pc.place_descent_sweep_bodies(bodies) == ['post_2']


def test_support_margin_default_is_penetration_only():
    assert pc.DEFAULT_PLACE_DESCENT_SUPPORT_MARGIN_M == 0.0


def test_reject_names_shelf_at_penetration_only_but_cage_at_design_margin():
    # a shelf PENETRATION (the pad-down housing driven into the board) is refused by name...
    msg = pc.place_descent_sweep_reject({'shelf': -0.053, 'post_2': 0.15})
    assert msg is not None and 'prim=shelf' in msg and 'SUPPORT board' in msg
    # ...a shelf CLOSE RIDE (the housing skimming the board as the carton lands) passes...
    assert pc.place_descent_sweep_reject({'shelf': 0.017, 'post_2': 0.15}) is None
    # ...but a CAGE body below the 30 mm design margin still refuses (the support margin never relaxes the cage)
    assert pc.place_descent_sweep_reject({'shelf': 0.017, 'post_2': 0.02}) is not None


# --------------------------------------------------------------------------- coldwin100 composed-USD replay (guarded)
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
def test_coldwin100_shelf_jam_reproduced_by_corrected_model():
    """Replaying the recorded coldwin100 TRAJ_HOST_PLAN pad-down descent against the COMPOSED mission cart
    colliders: the cage-only sweep reads it clear (~+108 mm, the host's blind descent_clearance), while the
    corrected shelf-inclusive sweep finds the gripper housing -59 mm INTO /Rollwagen/chassis/shelf and
    refuses it by name -- the cw100 jam reproduced from the real geometry, not the box model."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        'check_pad_down_shelf_contact',
        Path(__file__).resolve().parents[2] / 'check_pad_down_shelf_contact.py')
    chk = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(chk)
    assert chk.main([]) == 0
