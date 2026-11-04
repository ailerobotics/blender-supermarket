"""CPU-only regression: the PR-307 SOUTH-DOCK collect place-on-cart carton level (sim-carton-level-southdock-v1).

coldwin31 welded the carton, cleared the collision-freshness check and drove to the corrected south-dock
place pose, then the carton-level guard tripped in the PLACE set-down: roll -0.04, pitch +2.07 deg, sustained
0.167 s > the 0.150 s grace (host-stack.log `CARTON_TILT ... phase=place`). Both the planned endpoints and the
commanded straight joint line are level to < 0.05 deg -- the +2.07 is a QUASI-STATIC loaded gravity droop of
the far-extended arm at the sensed deck set-down (shoulder_lift gravity torque ~58 N*m), NOT a planned-pose
tilt and NOT velocity lag, which is why PR 302's 0.02 rad/s gentling could not save it (0.02 still gave 2.07).

This pins the 307 geometry (the exact place TCP, standoff and grasp reconstructed from coldwin31's host-stack)
and proves the fix: the loaded gravity-droop FEEDFORWARD on the level place IK (carton_level.gravity_compensation)
re-solves the COMMANDED config so the EXECUTED (drooped) welded carton lands level -- for BOTH the south dock and
the pre-307 east cart -- with the clear wrist region preserved, and reproduces the observed droop side by side
with the east cart (which collectlevel4 saw as a sub-grace transient and PASSED). It also pins that the
per-segment CARTON_SEGMENT_TILT diagnostic now actually emits (it keyed off the reassigned phase and never fired).

    PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=scripts python3 -m pytest scripts/twin/tests/test_carton_level_southdock.py
"""
import math
import os

import pytest

from twin.restock_manipulation import carton_level as cl
from twin.restock_manipulation.cad_tool import ARM_JOINTS

URDF = '/home/manishsaini/datasets/twin-robot-in-store-boot-v1/mission/coldwin31/mobile-import.urdf'
need_urdf = pytest.mark.skipif(not os.path.exists(URDF), reason='mission URDF not present')

BOUND = cl.DEFAULT_TILT_BOUND_DEG
TL = 'gripper_pad_tcp'


# ------------------------------------------------------------------ gravity-droop model + compensation

@need_urdf
def test_gravity_torque_model_reproduces_coldwin31_and_east_droop():
    from twin.tools import carton_level_bench as b
    south = b.run_southdock_place(URDF, geometry='south')
    east = b.run_southdock_place(URDF, geometry='east')
    # the far-extended south-dock set-down carries MORE loaded gravity torque than the east cart ...
    assert south.shoulder_lift_torque > east.shoulder_lift_torque > 40.0
    # ... and the geometrically-level command DROOPS off level: south over the 2.0 bound (the coldwin31 trip,
    # observed +2.07), east just under it (collectlevel4's sub-grace transient that PASSED).
    assert south.current_exec_pitch == pytest.approx(2.07, abs=0.25)
    assert south.current_exec_tilt > BOUND
    assert east.current_exec_tilt < BOUND
    # the COMMANDED set-down is geometrically level in BOTH cases -- the trip is execution, not a planned pose.
    assert south.current_cmd_tilt < 0.05 and east.current_cmd_tilt < 0.05


@need_urdf
def test_gravity_compensation_lands_the_executed_carton_level_both_geometries():
    from twin.tools import carton_level_bench as b
    for geom in ('south', 'east'):
        r = b.run_southdock_place(URDF, geometry=geom)
        # the fix: the EXECUTED (drooped) set-down is level within the brief's < 1 deg margin (vs ~2 deg before)
        assert r.comp_exec_tilt <= 1.0, (geom, r.comp_exec_tilt)
        assert r.comp_exec_tilt < 0.1, (geom, r.comp_exec_tilt)          # in fact ~0
        assert r.comp_exec_tilt < r.current_exec_tilt                    # strictly better than uncompensated
        # the whole descent stays under the real 2.0 bound (its max is the unchanged, already-passing standoff)
        assert r.descent_exec_max <= BOUND
        # the compensated command is INTENTIONALLY pre-tilted (not geometric level) -- that is the mechanism
        assert r.comp_cmd_tilt > 1.0 if geom == 'south' else r.comp_cmd_tilt >= 0.0
        assert r.setdown_clear                                            # stays in the clear carry wrist region
        assert r.passes(BOUND)


@need_urdf
def test_compensation_is_speed_independent_unlike_pr302_gentling():
    # PR 302's lever was time-parameterising the descent slower (a velocity-lag assumption). coldwin31 proves
    # the droop is quasi-static: the SAME geometrically-level set-down config droops the carton the same ~2 deg
    # regardless of how slowly it is approached, so only re-solving the GOAL (gravity FF) fixes it.
    from twin.tools import carton_level_bench as b
    r = b.run_southdock_place(URDF, geometry='south')
    assert r.current_exec_tilt > BOUND                                   # geometric-level command still droops
    assert r.comp_exec_tilt < 0.1                                        # only the FF goal re-solve lands level


@need_urdf
def test_arm_gravity_torques_shoulder_lift_dominates_at_the_setdown():
    from twin.urdf_probe import parse_urdf
    robot = parse_urdf(URDF)
    setdown = (-0.0917, -1.366, 1.6759, -0.3097, 1.6063, 0.0)            # coldwin31 place descent endpoint (ARM)
    tau = cl.arm_gravity_torques(robot, setdown, carton_mass=0.45, tool_link=TL)
    assert tau['ur_shoulder_lift_joint'] > 50.0                          # the heavy pitch joint carries the load
    assert abs(tau['ur_shoulder_pan_joint']) < 1.0                       # the vertical-axis pan carries ~none


# ------------------------------------------------------------------ config resolution

def test_resolve_gravity_compensation_forms():
    assert cl.resolve_gravity_compensation(None) is None
    assert cl.resolve_gravity_compensation(False) is None
    assert cl.resolve_gravity_compensation('off') is None
    gc = cl.resolve_gravity_compensation(True)
    assert isinstance(gc, cl.GravityCompensation) and gc.stiffness == cl.DEFAULT_DROOP_STIFFNESS
    gc = cl.resolve_gravity_compensation({'stiffness_nm_per_rad': 3000, 'carton_mass_kg': 0.5})
    assert gc.stiffness == 3000.0 and gc.carton_mass == 0.5
    with pytest.raises(ValueError):
        cl.resolve_gravity_compensation({'stiffness': -1})
    with pytest.raises(ValueError):
        cl.resolve_gravity_compensation('nonsense')


def test_level_config_parses_gravity_compensation_off_is_none():
    assert cl.resolve_level_config(True).gravity_compensation is None    # absent => off => byte-identical
    lc = cl.resolve_level_config({'gravity_compensation': {'stiffness': 2850}})
    assert isinstance(lc.gravity_compensation, cl.GravityCompensation)
    assert lc.gravity_compensation.stiffness == 2850.0
    # the compensation inherits the level tool_link so the droop torque uses the attitude frame
    lc = cl.resolve_level_config({'tool_link': 'ur_tool0', 'gravity_compensation': True})
    assert lc.gravity_compensation.tool_link == 'ur_tool0'


@need_urdf
def test_level_place_ik_gravity_comp_pretilts_command_so_executed_is_level():
    from twin.urdf_probe import parse_urdf
    robot = parse_urdf(URDF)
    u = cl.carton_up_in_tool(cl.tool_rotation(robot, [1.1994, -1.2682, 2.0862, -0.8177, 1.2002, 0.0], TL))
    big = 2 * math.pi
    limits = {'ur_wrist_1_joint': (-big, 0.65), 'ur_wrist_2_joint': (0.0, big), 'ur_elbow_joint': (0.1, math.pi)}
    seed = [cl.level_preserve_tcp(robot, [0.093, -1.91, 2.014, -0.103, 1.664, 0.0], u,
                                  tool_link=TL, limits=limits)[j] for j in ARM_JOINTS]
    target = (1.043, 0.055, 0.435)
    gc = cl.GravityCompensation(stiffness=2850, carton_mass=0.45, tool_link=TL)
    geom = cl.level_place_ik(robot, target, u, seed, tool_link=TL, limits=limits)
    comp = cl.level_place_ik(robot, target, u, seed, tool_link=TL, limits=limits, gravity_comp=gc)
    # the geometric command is level; the compensated command is PRE-TILTED (differs by the droop)
    assert cl.carried_rpy(robot, [geom[j] for j in ARM_JOINTS], u, TL)[2] < 0.1
    assert cl.carried_rpy(robot, [comp[j] for j in ARM_JOINTS], u, TL)[2] > 1.0
    # under the droop, the compensated command EXECUTES level while the geometric one droops off level
    dq = gc.droop(robot, comp, tool_link=TL)
    comp_exec = [comp[ARM_JOINTS[i]] + dq[ARM_JOINTS[i]] for i in range(6)]
    assert cl.carried_rpy(robot, comp_exec, u, TL)[2] < 0.1
    dq0 = gc.droop(robot, geom, tool_link=TL)
    geom_exec = [geom[ARM_JOINTS[i]] + dq0[ARM_JOINTS[i]] for i in range(6)]
    assert cl.carried_rpy(robot, geom_exec, u, TL)[2] > BOUND
    # the executed compensated pose still reaches the sensed deck (reach preserved)
    import numpy as np
    assert float(np.linalg.norm(cl.tool_position(robot, comp_exec, TL) - np.asarray(target))) < 5e-3


@need_urdf
def test_predict_path_tilt_executed_reads_the_physical_carton_not_the_pretilt():
    from twin.urdf_probe import parse_urdf
    robot = parse_urdf(URDF)
    u = cl.carton_up_in_tool(cl.tool_rotation(robot, [1.1994, -1.2682, 2.0862, -0.8177, 1.2002, 0.0], TL))
    big = 2 * math.pi
    limits = {'ur_wrist_1_joint': (-big, 0.65), 'ur_wrist_2_joint': (0.0, big), 'ur_elbow_joint': (0.1, math.pi)}
    seed = [cl.level_preserve_tcp(robot, [0.093, -1.91, 2.014, -0.103, 1.664, 0.0], u,
                                  tool_link=TL, limits=limits)[j] for j in ARM_JOINTS]
    gc = cl.GravityCompensation(stiffness=2850, carton_mass=0.45, tool_link=TL)
    comp = [cl.level_place_ik(robot, (1.043, 0.055, 0.435), u, seed, tool_link=TL, limits=limits,
                              gravity_comp=gc)[j] for j in ARM_JOINTS]
    # GEOMETRIC replay reads the pre-tilted command (> 1 deg) -> would falsely reject the valid plan
    _, geo_pitch, _ = cl.predict_path_tilt(robot, [comp, comp], u, tool_link=TL)
    assert geo_pitch > 1.0
    # droop-aware replay reads what the PHYSICAL carton (and the CARTON_TILT trip) sees -> level
    _, exe_pitch, _ = cl.predict_path_tilt_executed(robot, [comp, comp], u, tool_link=TL, gravity_comp=gc)
    assert exe_pitch < 0.1


# ------------------------------------------------------------------ CARTON_SEGMENT_TILT emit fix

def test_segment_tilt_emits_for_the_departing_phase(capsys):
    from twin.tests.test_restock_manipulation import make_controller
    c, _ = make_controller('collect', carton_level=cl.resolve_level_config(True), fk_urdf=URDF)
    c.phase = 'place'
    c._carton_seg_exec['place'] = (0.3, 1.1)           # an executed sample measured during the place phase
    c._carton_seg_planned['place'] = (0.0, 0.05)
    c.evidence = []
    c._transition('retreat')                           # leaving place -> retreat
    out = capsys.readouterr().out
    assert 'CARTON_SEGMENT_TILT phase=place' in out    # keyed off the DEPARTING phase (was the reassigned one)
    assert 'executed_max roll=0.30 pitch=1.10' in out
    # and it is relayed through evidence so the orchestrator surfaces it in orch-console.log
    assert any(isinstance(e, dict) and 'carton_segment_tilt' in e for e in c.evidence)


def test_segment_tilt_silent_when_carton_level_off():
    from twin.tests.test_restock_manipulation import make_controller
    c, _ = make_controller('collect')                  # no carton_level
    c.phase = 'place'
    c.evidence = []
    c._transition('retreat')
    assert not any(isinstance(e, dict) and 'carton_segment_tilt' in e for e in c.evidence)


# ------------------------------------------------------------------ controller wiring

@need_urdf
def test_solve_level_place_applies_gravity_comp_when_configured():
    from twin.tests.test_restock_manipulation import make_controller, _SL_PREPLACE_SHAPE
    lvl = cl.resolve_level_config({'gravity_compensation': {'stiffness': 2850, 'carton_mass': 0.45}})
    c, _ = make_controller('collect', carton_level=lvl, fk_urdf=URDF,
                           carry_wrist1_cap=0.65, carry_wrist2_min=0.0)
    robot = c._fk_model()
    c._capture_level_up((1.1994, -1.2682, 2.0862, -0.8177, 1.2002, 0.0))
    u = c._carton_up_in_tool
    # the set-down solve (target_pos given) is gravity-compensated -> a pre-tilted command that executes level
    lev = c._solve_level_place(robot, list(_SL_PREPLACE_SHAPE), (1.043, 0.055, 0.435))
    comp = [lev[j] for j in ARM_JOINTS]
    assert cl.carried_rpy(robot, comp, u, c._carton_level.tool_link)[2] > 1.0     # command pre-tilted
    gc = c._carton_level.gravity_compensation
    dq = gc.droop(robot, lev, tool_link=c._carton_level.tool_link)
    exe = [comp[i] + dq[ARM_JOINTS[i]] for i in range(6)]
    assert cl.carried_rpy(robot, exe, u, c._carton_level.tool_link)[2] < 0.2      # executes level


@need_urdf
def test_solve_level_place_off_is_geometric_level():
    from twin.tests.test_restock_manipulation import make_controller, _SL_PREPLACE_SHAPE
    c, _ = make_controller('collect', carton_level=cl.resolve_level_config(True), fk_urdf=URDF,
                           carry_wrist1_cap=0.65, carry_wrist2_min=0.0)
    robot = c._fk_model()
    c._capture_level_up((1.1994, -1.2682, 2.0862, -0.8177, 1.2002, 0.0))
    u = c._carton_up_in_tool
    lev = c._solve_level_place(robot, list(_SL_PREPLACE_SHAPE), (1.043, 0.055, 0.435))
    # no gravity_compensation configured => the command itself is geometrically level (byte-identical solve)
    assert cl.carried_rpy(robot, [lev[j] for j in ARM_JOINTS], u, c._carton_level.tool_link)[2] < 0.1
