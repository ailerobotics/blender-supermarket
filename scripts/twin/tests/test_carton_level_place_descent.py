"""CPU-only: the GENTLED, OBSERVABLE collect place-on-cart SET-DOWN descent
(sim-carton-level-preplace-v2).

collectlevel3 tripped CARTON_TILT at phase=PLACE (roll+0.10 pitch+2.22 deg, sustained 0.167 s > the
0.15 s grace) -- NOT at the pre-place segment PR 298 covered (its leveled approach ran at predicted
0.11 deg pitch and never tripped). The place-on-cart SET-DOWN is the ONE carried segment still driven
as a single 2-point straight-line begin_joint_goal. Its COMMANDED straight joint line is already LEVEL
(standoff and set-down both full-DOF leveled; the straight interior stays < 0.05 deg -- proved here on
the REAL mission kinematics), so the 2.22 deg is an EXECUTION transient of the loaded set-down: at the
standard 0.08 rad/s the welded carton's DIFFERENTIAL joint tracking during the heavy shoulder/elbow
descent leaves level. The fix TIME-PARAMETERISES the SAME level path slower
(carton_level.place_descent_speed/accel); the bound and grace are UNCHANGED. This replays the executed
joints through a representative tracking-lag model to reproduce 2.22 and shows the gentled descent stays
under 1 deg, and checks the controller wiring + the new CARTON_SEGMENT_TILT observability.

    PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=scripts python3 -m pytest \
        scripts/twin/tests/test_carton_level_place_descent.py
"""
import math
import os

import pytest

from twin.restock_manipulation import carton_level as cl
from twin.tools import carton_level_bench as bench

URDF = '/home/manishsaini/datasets/twin-robot-in-store-boot-v1/mission/collectlevel3/mobile-import.urdf'
need_urdf = pytest.mark.skipif(not os.path.exists(URDF), reason='mission URDF not present')


# ------------------------------------------------------------------ config (no URDF)

def test_resolve_place_descent_defaults():
    lc = cl.resolve_level_config(True)
    assert lc.place_descent_speed == pytest.approx(0.02)
    assert lc.place_descent_accel == pytest.approx(0.04)


def test_resolve_place_descent_override():
    lc = cl.resolve_level_config({'place_descent_speed': 0.03, 'place_descent_accel': 0.05})
    assert lc.place_descent_speed == pytest.approx(0.03)
    assert lc.place_descent_accel == pytest.approx(0.05)


@pytest.mark.parametrize('bad', [{'place_descent_speed': 0.0}, {'place_descent_speed': -1},
                                 {'place_descent_accel': 0.0}, {'place_descent_accel': float('inf')}])
def test_resolve_place_descent_rejects_nonpositive(bad):
    with pytest.raises(ValueError):
        cl.resolve_level_config(bad)


def test_off_leaves_place_descent_inert():
    assert cl.resolve_level_config(False) is None
    assert cl.resolve_level_config('off') is None


# ------------------------------------------------------------------ bench proof (REAL kinematics)

@need_urdf
def test_commanded_place_descent_straight_line_is_level():
    """The COMMANDED single-2-point straight joint line standoff->set-down is LEVEL: the trip is NOT a
    planned-pose tilt. (This is the correction to the brief's pre-place framing.)"""
    r = bench.run_place_descent(URDF)
    assert r.commanded_max_pitch < 0.1, r.commanded_max_pitch
    assert r.commanded_max_tilt < 0.1, r.commanded_max_tilt


@need_urdf
def test_ungentled_execution_reproduces_the_2p22_trip():
    """The representative tracking-lag model reproduces the collectlevel3 executed peak (~2.22 deg pitch)
    on the standard 0.08 rad/s descent -- an EXECUTION transient, while the command is level."""
    r = bench.run_place_descent(URDF)
    assert r.ungentled_speed == pytest.approx(0.08)
    assert r.ungentled_max_pitch == pytest.approx(2.22, abs=0.1), r.ungentled_max_pitch
    # it is the execution, not the plan: the ungentled executed pitch is >> the commanded pitch.
    assert r.ungentled_max_pitch > 10 * r.commanded_max_pitch


@need_urdf
def test_gentled_execution_stays_under_one_degree():
    """Time-parameterising the SAME level path at the default gentled speed keeps the executed carton
    under 1 deg (the brief's target) with comfortable margin to the 2.0 deg bound -- bound unchanged."""
    r = bench.run_place_descent(URDF)
    assert r.gentled_speed == pytest.approx(0.02)
    assert r.gentled_max_pitch < 1.0, r.gentled_max_pitch
    assert r.gentled_max_pitch < 0.5 * cl.DEFAULT_TILT_BOUND_DEG      # < 1.0 deg, well under the 2.0 bound
    assert r.gentled_duration > r.ungentled_duration                 # slower (time-parameterised)


@need_urdf
def test_required_speed_ceiling_monotone():
    """State the required ceiling: the executed peak falls monotonically as the descent is gentled, and
    crosses under 1 deg at or below the default 0.02 rad/s."""
    peaks = {}
    for s, a in [(0.08, 0.15), (0.04, 0.07), (0.03, 0.05), (0.02, 0.04)]:
        peaks[s] = bench.run_place_descent(URDF, gentled_speed=s, gentled_accel=a).gentled_max_pitch
    assert peaks[0.08] > peaks[0.04] > peaks[0.03] > peaks[0.02]      # monotone in speed
    assert peaks[0.02] < 1.0


# ------------------------------------------------------------------ controller wiring + observability

@need_urdf
def test_place_descent_passes_gentled_ceilings_to_motion():
    """The collect-deck level place descent dispatches as a straight-line begin_joint_goal carrying the
    gentled place_descent_speed/accel; carton_level OFF leaves it byte-identical (no speed/accel)."""
    from twin.tests import test_restock_manipulation as T
    # level_path OFF so the preplace uses the single leveled goal (the leveled-waypoint preplace hits a
    # pre-existing fixture-URDF inversion unrelated to this change); the place descent is independent.
    c = T.make_controller(
        'collect', transport_joints=T._SL_STOW, staged_cart_swing=True,
        cart_pan_joint0=-0.8605, cart_tuck_shape=T._SL_TUCK, reposition_before_place=True,
        collect_place_preplace_shape=T._SL_PREPLACE_SHAPE, collect_place_wrist_guard=True,
        collect_place_straight_line=True,
        carton_level=cl.resolve_level_config({'trip': False, 'level_path': False}), fk_urdf=URDF,
        carry_wrist1_cap=0.65, carry_wrist2_min=0.0)[0]
    m = c.motion
    _by_phase, _seq, jg_by_phase = T._drive_collect_place(c, m)
    place = jg_by_phase['place']
    assert place['straight_line'] is True
    assert place['speed'] == pytest.approx(0.02)
    assert place['accel'] == pytest.approx(0.04)


def test_place_descent_off_is_byte_identical():
    """carton_level OFF: the collect place descent is unchanged -- a begin() POSE straight-line move
    (the leveled begin_joint_goal descent branch is gated off), so no gentled joint goal is issued."""
    from twin.tests import test_restock_manipulation as T
    c, m = T._collect_place_controller(straight_line=True)
    assert c._carton_level is None
    by_phase, _seq, jg_by_phase = T._drive_collect_place(c, m)
    assert by_phase.get('place') is True            # place stays a collision-off begin() pose straight-line
    assert 'place' not in jg_by_phase               # ... never a (gentled) joint goal when the level fix is off
