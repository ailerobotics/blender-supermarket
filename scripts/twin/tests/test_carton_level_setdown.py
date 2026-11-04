"""CPU-only regression: the PR-307 collect place SET-DOWN carton level (sim-carton-level-setdown-v1).

coldwin33 was the first complete side-bay cold cycle with every fix of the day in (PR 309..315) and
sim-carton-level-southdock-v1's gravity_compensation (stiffness 2850, carton_mass 0.45) ON. The carry stayed
level through lift/tuck/transit/preplace, but the PLACE set-down -- the most extended welded pose -- still
pitched +2.06 deg, sustained exactly the 0.150 s grace, and the carton-level guard tripped (host-stack.log
`CARTON_TILT ... pitch=+2.06 ... phase=place`). PR 311's one global stiffness closed the preplace droop but
moved the set-down only from 2.07 (coldwin31, uncompensated) to 2.06 (coldwin33, compensated).

Root cause (CPU-reproduced here): PR 311's full-DOF level place IK spread the droop-cancelling pre-tilt onto
the HEAVY shoulder_lift/elbow pitch joints (shoulder_lift +0.020 rad, elbow +0.015 rad). At the far-extended
loaded set-down those joints are gravity-PINNED: the GPU moved the executed carton 0.01 deg for a +2.04 deg
heavy pre-tilt (2.07 -> 2.06), i.e. d(executed)/d(command) ~ 0. The WRIST, which carries ~2 N*m, DOES track
its command (coldwin33 lift/tuck/preplace executed matched the geometric plan within 0.02 deg). The
sim-carton-level-southdock-v1 "executed = command + droop(command)" model is CIRCULAR -- level_place_ik solves
the command so command+droop is level and the predictor evaluates command+droop -> always ~0 -- so it could
never express the residual. The honest TWO-REGIME model (heavy pinned, wrist tracks), calibrated to BOTH GPU
points, reproduces 2.07/2.06 and shows the fix: route the droop-cancelling pre-tilt onto the TRACKING WRIST
(GravityCompensation.correct_joints, default the wrist), so the physical welded carton lands level.

    PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=scripts python3 -m pytest scripts/twin/tests/test_carton_level_setdown.py
"""
import math
import os

import pytest

from twin.restock_manipulation import carton_level as cl
from twin.restock_manipulation.cad_tool import ARM_JOINTS

URDF = '/home/manishsaini/datasets/twin-robot-in-store-boot-v1/mission/coldwin31/mobile-import.urdf'
need_urdf = pytest.mark.skipif(not os.path.exists(URDF), reason='mission URDF not present')

BOUND = cl.DEFAULT_TILT_BOUND_DEG
MARGIN = 1.0
TL = 'gripper_pad_tcp'


# ------------------------------------------------------------------ reproduce 2.07 / 2.06 (two-regime)

@need_urdf
def test_two_regime_model_reproduces_both_gpu_setdown_points():
    """The honest two-regime model (heavy pinned, wrist tracks) reproduces coldwin31's 2.07 deg UNCOMPENSATED
    AND coldwin33's 2.06 deg with the PR311 HEAVY-routed pre-tilt -- the pre-tilt on the gravity-pinned joints
    does NOT realise, so the executed set-down barely moves. One calibration (K=2850), two independent points."""
    from twin.tools import carton_level_bench as b
    south = b.run_setdown_tworegime(URDF, geometry='south')
    # uncompensated geometric-level command droops to ~2 deg (coldwin31 observed +2.07; the one-constant
    # K=2850 model reproduces it to within ~0.1 deg -- the lane re-pins K from the CARTON_SEGMENT_TILT log).
    assert south.uncomp_exec_pitch == pytest.approx(2.03, abs=0.25)
    assert south.uncomp_exec_pitch > 1.5                                # clearly off level (the trip regime)
    # PR311's HEAVY-routed pre-tilt executes essentially the SAME (the coldwin33 residual 2.06): the pre-tilt
    # on the gravity-pinned shoulder/elbow does not change the executed carton.
    assert south.heavy_exec_pitch == pytest.approx(2.03, abs=0.25)
    assert south.heavy_exec_pitch > 1.5                                # still ~2 deg -- the trip the GPU hit
    assert abs(south.heavy_exec_pitch - south.uncomp_exec_pitch) < 0.2   # d(exec)/d(heavy pre-tilt) ~ 0


# ------------------------------------------------------------------ the fix: wrist-routed feedforward

@need_urdf
def test_wrist_routed_feedforward_lands_the_setdown_level_both_docks():
    from twin.tools import carton_level_bench as b
    for geom in ('south', 'east'):
        r = b.run_setdown_tworegime(URDF, geometry=geom)
        # the fix lands the EXECUTED set-down level within the brief's <= 1 deg margin at BOTH docks ...
        assert r.wrist_exec_tilt <= MARGIN, (geom, r.wrist_exec_tilt)
        assert r.wrist_exec_pitch <= MARGIN, (geom, r.wrist_exec_pitch)
        assert r.wrist_exec_tilt < 0.1, (geom, r.wrist_exec_tilt)        # in fact ~0
        # ... strictly better than BOTH the uncompensated droop AND the PR311 heavy-routed command ...
        assert r.wrist_exec_pitch < r.uncomp_exec_pitch
        assert r.wrist_exec_pitch < r.heavy_exec_pitch
        # ... the correction lands on the TRACKING wrist (not the gravity-pinned heavy joints) ...
        assert r.wrist_pretilt_joint.startswith('ur_wrist_')
        # ... the executed set-down still reaches the sensed deck within the mission place goal tolerance ...
        assert r.wrist_reach_err_mm <= 120.0, (geom, r.wrist_reach_err_mm)
        assert r.wrist_clear                                             # in the clear carry wrist region
        assert r.passes(MARGIN)


@need_urdf
def test_level_place_ik_gravity_comp_routes_pretilt_to_the_wrist_not_heavy():
    """The source change: level_place_ik with gravity_comp pre-rotates the WRIST, leaving the heavy pitch
    joints at their geometric value (coldwin33 proved the heavy pre-tilt does not realise)."""
    from twin.urdf_probe import parse_urdf
    robot = parse_urdf(URDF)
    u = cl.carton_up_in_tool(cl.tool_rotation(robot, [1.1994, -1.2682, 2.0862, -0.8177, 1.2002, 0.0], TL))
    big = 2 * math.pi
    limits = {'ur_wrist_1_joint': (-big, 0.65), 'ur_wrist_2_joint': (0.0, big), 'ur_elbow_joint': (0.1, math.pi)}
    seed = [cl.level_preserve_tcp(robot, [0.093, -1.91, 2.014, -0.103, 1.664, 0.0], u,
                                  tool_link=TL, limits=limits)[j] for j in ARM_JOINTS]
    tgt = (1.043, 0.055, 0.435)
    gc = cl.GravityCompensation(stiffness=2850, carton_mass=0.45, tool_link=TL)
    geom = cl.level_place_ik(robot, tgt, u, seed, tool_link=TL, limits=limits)
    comp = cl.level_place_ik(robot, tgt, u, seed, tool_link=TL, limits=limits, gravity_comp=gc)
    dj = {j: comp[j] - geom[j] for j in ARM_JOINTS}
    # the heavy pitch joints are NOT pre-tilted (unchanged from geometric); the wrist carries the correction
    assert abs(dj['ur_shoulder_lift_joint']) < 1e-3
    assert abs(dj['ur_elbow_joint']) < 1e-3
    assert abs(dj['ur_wrist_1_joint']) > 0.01
    # the geometric carton is pre-tilted (> 1 deg) by that wrist rotation -- the mechanism
    assert cl.carried_rpy(robot, [comp[j] for j in ARM_JOINTS], u, TL)[2] > 1.0


# ------------------------------------------------------------------ config: correct_joints

def test_gravity_compensation_defaults_correct_joints_to_the_wrist():
    gc = cl.GravityCompensation()
    assert gc.correct_joints == cl.DEFAULT_LEVEL_JOINTS
    assert all(j.startswith('ur_wrist_') for j in gc.correct_joints)


def test_resolve_gravity_compensation_accepts_correct_joints_and_rejects_bad():
    gc = cl.resolve_gravity_compensation({'stiffness_nm_per_rad': 2850, 'correct_joints': ['ur_wrist_1_joint']})
    assert gc.correct_joints == ('ur_wrist_1_joint',)
    with pytest.raises(ValueError):
        cl.resolve_gravity_compensation({'correct_joints': ['not_a_joint']})
    # absent => default wrist
    assert cl.resolve_gravity_compensation(True).correct_joints == cl.DEFAULT_LEVEL_JOINTS


@need_urdf
def test_wrist_feedforward_is_extension_following():
    """The wrist pre-rotation GROWS with reach (the sag is tau/K and tau grows with extension), so the
    deeper south set-down gets a larger wrist correction than the nearer east cart -- a per-pose solve,
    not a hand-dropped constant."""
    from twin.tools import carton_level_bench as b
    south = b.run_setdown_tworegime(URDF, geometry='south')
    east = b.run_setdown_tworegime(URDF, geometry='east')
    # the south dock is more extended => more uncompensated droop => needs (and gets) more correction
    assert south.uncomp_exec_pitch > east.uncomp_exec_pitch
    assert south.passes(MARGIN) and east.passes(MARGIN)


# ------------------------------------------------------------------ CARTON_SEGMENT_TILT: set-down + release

def test_segment_tilt_line_reports_planned_and_executed_for_place_and_release(capsys):
    """The per-segment line now leads with planned=<deg> executed=<deg> (log-tunable without a trip) and
    emits for the set-down AND the release, which previously reported only through the trip."""
    from twin.tests.test_restock_manipulation import make_controller
    for seg in ('place', 'release'):
        c, _ = make_controller('collect', carton_level=cl.resolve_level_config(True), fk_urdf=URDF)
        c.phase = seg
        c._carton_seg_exec[seg] = (0.10, 0.42)
        c._carton_seg_planned[seg] = (0.00, 0.05)
        c.evidence = []
        c._transition('retreat')
        out = capsys.readouterr().out
        assert f'CARTON_SEGMENT_TILT phase={seg}' in out
        assert 'planned=0.05 executed=0.42' in out                      # the brief's compact tokens
        assert 'executed_max roll=0.10 pitch=0.42' in out               # the roll/pitch breakdown kept
        assert any(isinstance(e, dict) and 'carton_segment_tilt' in e for e in c.evidence)


@need_urdf
def test_release_sample_accumulates_executed_tilt_without_a_trip():
    """_carton_seg_sample stashes the welded carton's release attitude (stash-only, never fails the move)."""
    from twin.tests.test_restock_manipulation import make_controller
    c, _ = make_controller('collect', carton_level=cl.resolve_level_config(True), fk_urdf=URDF)
    c._capture_level_up((1.1994, -1.2682, 2.0862, -0.8177, 1.2002, 0.0))
    c.phase = 'release'
    c._carton_seg_sample('release', (1.1994, -1.2682, 2.0862, -0.8177, 1.2002, 0.0))
    assert 'release' in c._carton_seg_exec
    er, ep = c._carton_seg_exec['release']
    assert er >= 0.0 and ep >= 0.0
