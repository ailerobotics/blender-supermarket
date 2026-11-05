"""CPU tests for the carried pre-place APPROACH carton-level gate + carried-carton dynamics cap
(sim-preplace-level-gate-v1).

coldwin50 tripped CARTON_TILT at the HIGH from-above collect preplace standoff (pitch +7.64 deg, base
rock-steady, the PR 339 full D6 lock never reached). The integration-worker forensics
(coldwin50-preplace.txt) pinned it: cw49 and cw50 commanded the SAME standoff IK branch (every joint within
0.003 rad) whose STATIC droop-executed tilt is only ~1.6 deg; the 7.64 is run-to-run DYNAMIC deflection
during a COARSE, FAST preplace sweep (a 9-point cuMotion plan). Two static gaps enabled it:

  1. the carried APPROACH to the standoff levelled only the GEOMETRIC interior wrists, so the EXECUTED
     (drooped) interior still pitched ~1.7 deg, and the approach plan gate replayed the GEOMETRIC path
     (``predict_path_tilt``, droop-blind), so it never saw it;
  2. the move was a coarse fast sweep -> dynamic deflection.

``preplace_level_gate`` (default OFF => byte-identical) closes both: the standoff is solved on the branch
with the smallest predicted EXECUTED tilt (``solve_level_standoff``), the carried approach to it is a
DROOP-COMPENSATED, DENSIFIED, speed/accel-capped leveled path (``plan_compensated_approach``), the predicted
EXECUTED (droop-aware) tilt is gated and refused before any motion if it exceeds the true bound, and the
dynamics cap (dense waypoints + carried speed/accel) lowers the peak joint velocity that drives the dynamic
deflection.

    PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=scripts python3 -m pytest \
        scripts/twin/tests/test_preplace_level_gate_v1.py
"""
import math
import os

import pytest

from twin.restock_manipulation import carton_level as cl
from twin.restock_manipulation.cad_tool import ARM_JOINTS

URDF = '/home/manishsaini/datasets/twin-robot-in-store-boot-v1/mobile-casterlift/mobile_ur10.urdf'
need_urdf = pytest.mark.skipif(not os.path.exists(URDF), reason='dataset mobile URDF not present')
TL = 'gripper_pad_tcp'

# coldwin48/49/50 geometry (ur_base_link): the resolved shelf place target + the deck-tuned fixed standoff.
FIXED_SHAPE = (0.093, -1.91, 2.014, -0.103, 1.664, 0.0)
CW_PLACE = (0.957, 0.064, 0.538)                   # the resolved shelf place target
STANDOFF = (CW_PLACE[0], CW_PLACE[1], CW_PLACE[2] + 0.08)   # the +0.08 m above-support standoff
# a realistic level side-grasp weld (consistent with the southdock/setdown fixtures): carton up = tool +Y.
WELD = [1.1994, -1.2682, 2.0862, -0.8177, 1.2002, 0.0]


@pytest.fixture(scope='module')
def robot():
    from twin.urdf_probe import parse_urdf
    return parse_urdf(URDF)


@pytest.fixture(scope='module')
def up(robot):
    return cl.carton_up_in_tool(cl.tool_rotation(robot, WELD, TL))


@pytest.fixture(scope='module')
def gc():
    return cl.GravityCompensation(stiffness=2850.0, carton_mass=0.45, tool_link=TL).validate()


def _limits():
    big = 2 * math.pi
    return {'ur_elbow_joint': (0.1, math.pi),
            'ur_wrist_1_joint': (-big, 0.65), 'ur_wrist_2_joint': (0.0, big)}


# --------------------------------------------------------------- config resolve / validate

def test_resolve_off_is_byte_identical_default():
    assert cl.resolve_level_config(False) is None
    lc = cl.resolve_level_config(True)
    assert lc.preplace_level_gate is False
    # the new keys carry their documented defaults (the bound minus a 1.0 deg margin, etc.)
    assert lc.preplace_gate_bound_deg == pytest.approx(1.0)
    assert lc.preplace_max_segment_rad == pytest.approx(0.25)
    assert lc.preplace_approach_speed == pytest.approx(0.05)
    assert lc.preplace_approach_accel == pytest.approx(0.1)


def test_resolve_gate_keys():
    lc = cl.resolve_level_config({'preplace_level_gate': True, 'preplace_gate_bound_deg': 0.8,
                                  'preplace_max_segment_rad': 0.2, 'preplace_approach_speed': 0.04,
                                  'preplace_approach_accel': 0.08})
    assert lc.preplace_level_gate is True
    assert lc.preplace_gate_bound_deg == pytest.approx(0.8)
    assert lc.preplace_max_segment_rad == pytest.approx(0.2)
    assert lc.preplace_approach_speed == pytest.approx(0.04)
    assert lc.preplace_approach_accel == pytest.approx(0.08)


def test_resolve_rejects_gate_bound_over_bound():
    with pytest.raises(ValueError):
        cl.resolve_level_config({'preplace_gate_bound_deg': 2.5})   # > bound_deg 2.0
    with pytest.raises(ValueError):
        cl.resolve_level_config({'preplace_gate_bound_deg': 0.0})
    with pytest.raises(ValueError):
        cl.resolve_level_config({'preplace_max_segment_rad': -0.1})
    with pytest.raises(ValueError):
        cl.resolve_level_config({'preplace_approach_speed': 0.0})
    with pytest.raises(ValueError):
        cl.resolve_level_config({'preplace_approach_accel': -1})


# --------------------------------------------------------------- densify_count (dynamics)

def test_densify_count_scales_with_segment_cap():
    start = (0.0,) * 6
    goal = (1.35, 0.0, 0.0, 0.0, 1.64, 0.0)       # the cw50 pan/wrist_2 sweep magnitudes
    # max per-joint step is 1.64 rad; at 0.25 rad/segment -> ceil(1.64/0.25) = 7 segments.
    assert cl.densify_count(start, goal, 0.25) == 7
    # a tighter cap -> more segments (finer); a looser cap -> fewer (never below 2).
    assert cl.densify_count(start, goal, 0.1) > cl.densify_count(start, goal, 0.25)
    assert cl.densify_count(start, start, 0.25) == 2
    with pytest.raises(ValueError):
        cl.densify_count(start, goal, 0.0)


# --------------------------------------------------------------- level standoff solve (part 1)

@need_urdf
def test_solve_level_standoff_is_executed_level(robot, up, gc):
    sol, (roll, pitch, tilt) = cl.solve_level_standoff(robot, STANDOFF, up, list(FIXED_SHAPE), gc,
                                                       tool_link=TL, limits=_limits())
    # the welded carton's PREDICTED EXECUTED pose at the standoff is level (droop folded in), not merely
    # the geometric command.
    assert tilt < 0.3
    solv = [sol[j] for j in ARM_JOINTS]
    # the standoff is near the target reach: the droop pre-tilt + the loaded sag trade a few cm for a level
    # EXECUTED attitude (as the place descent does; the from-above descent re-targets the true place TCP
    # per-waypoint, so a cm-scale standoff offset is corrected during the descent, not carried into the set-down).
    assert math.dist(cl.tool_position(robot, solv, TL), STANDOFF) < 0.03
    # and stays in the clear carry wrist region.
    assert cl._in_clear_wrist(sol)


@need_urdf
def test_solve_level_standoff_requires_gravity_compensation(robot, up):
    with pytest.raises(ValueError):
        cl.solve_level_standoff(robot, STANDOFF, up, list(FIXED_SHAPE), None, tool_link=TL)


# --------------------------------------------------------- approach gate: the cw50 gap (part 2)

@need_urdf
def test_geometric_interior_approach_droops_but_compensated_is_level(robot, up, gc):
    """Reproduce the coldwin50 gap and the fix: the GEOMETRIC-interior approach's EXECUTED tilt exceeds the
    1.0 deg gate target (the droop-blind gate missed it), while the DROOP-COMPENSATED, DENSIFIED approach
    keeps the predicted EXECUTED carton level (<= 1.0) over the whole path."""
    goal, (_, _, gtilt) = cl.solve_level_standoff(robot, STANDOFF, up, list(FIXED_SHAPE), gc,
                                                  tool_link=TL, limits=_limits())
    goalv = [goal[j] for j in ARM_JOINTS]
    # a realistic LOW-DROOP carry/transit start (the real cw50 transit executed ~0.03 deg).
    transit = cl.level_place_ik(robot, (0.55, 0.05, 0.80), up, list(FIXED_SHAPE),
                                tool_link=TL, limits=_limits(), gravity_comp=gc)
    transit_v = [transit[j] for j in ARM_JOINTS]
    assert cl.droop_induced_rpy(robot, transit_v, gc, tool_link=TL, up_in_tool=up)[2] < 0.3

    # A) the legacy GEOMETRIC-interior approach (plan_level_path_to_goal levels only geometrically): its
    #    EXECUTED (droop-aware) interior exceeds the 1.0 deg gate target -- the gap the old geometric gate
    #    (predict_path_tilt) never saw.
    geo = cl.plan_level_path_to_goal(robot, list(transit_v), list(goalv), up, bound_deg=2.0,
                                     tool_link=TL, limits=_limits(), waypoints=12)
    _, _, geo_exec = cl.predict_path_tilt_executed(robot, [list(w) for w in geo.waypoints], up,
                                                   tool_link=TL, gravity_comp=gc)
    assert geo_exec > 1.0

    # B) the droop-compensated + densified approach keeps the EXECUTED carton level THROUGHOUT.
    comp = cl.plan_compensated_approach(robot, list(transit_v), list(goalv), up, gc,
                                        tool_link=TL, limits=_limits(), bound_deg=2.0, max_segment_rad=0.25)
    assert comp.max_tilt < geo_exec            # the interior droop is removed
    assert max(comp.max_roll, comp.max_pitch) <= 1.0   # <= the gate target
    # the path starts exactly at the measured start and ends exactly at the executed-level standoff.
    assert list(comp.waypoints[0]) == pytest.approx(transit_v)
    assert list(comp.waypoints[-1]) == pytest.approx(goalv)
    # densified: at least densify_count segments.
    assert comp.waypoints_n >= cl.densify_count(transit_v, goalv, 0.25)


@need_urdf
def test_compensated_approach_requires_gravity_compensation(robot, up):
    with pytest.raises(ValueError):
        cl.plan_compensated_approach(robot, list(FIXED_SHAPE), list(FIXED_SHAPE), up, None, tool_link=TL)


# --------------------------------------------------------- dynamics cap: peak joint velocity (part 1/3)

@need_urdf
def test_densified_capped_approach_has_lower_peak_joint_velocity(robot, up, gc):
    """The CPU velocity replay: a COARSE fast cuMotion-style sweep (few points, arm_max_speed timing) drives
    a high peak joint velocity (the source of the run-to-run dynamic deflection), while the densified,
    speed/accel-capped begin_joint_path drives a much lower peak -- same geometric endpoints."""
    goal, _ = cl.solve_level_standoff(robot, STANDOFF, up, list(FIXED_SHAPE), gc, tool_link=TL,
                                      limits=_limits())
    goalv = [goal[j] for j in ARM_JOINTS]
    transit = cl.level_place_ik(robot, (0.55, 0.05, 0.80), up, list(FIXED_SHAPE),
                                tool_link=TL, limits=_limits(), gravity_comp=gc)
    transit_v = [transit[j] for j in ARM_JOINTS]
    comp = cl.plan_compensated_approach(robot, list(transit_v), list(goalv), up, gc,
                                        tool_link=TL, limits=_limits(), bound_deg=2.0, max_segment_rad=0.25)
    pk_v, pk_a = cl.path_peak_joint_speed([list(w) for w in comp.waypoints], speed=0.05, accel=0.1)
    assert pk_v <= 0.05 + 1e-6               # the carried speed cap bounds the peak joint velocity

    # a coarse 9-point sweep timed at the arm_max_speed profile (what cw50's cuMotion preplace did).
    coarse = [list(transit_v)]
    for i in range(1, 9):
        a = i / 8.0
        coarse.append([(1 - a) * transit_v[k] + a * goalv[k] for k in range(6)])
    t = [0.0]
    for i in range(1, 9):
        dmax = max(abs(coarse[i][k] - coarse[i - 1][k]) for k in range(6))
        t.append(t[-1] + dmax / 0.3)         # arm_max_speed 0.3 rad/s TOTG-style timing
    coarse_v, coarse_a = cl.timed_path_peak_joint_speed(t, coarse)
    # the capped densified approach is strictly gentler on both peak velocity AND peak acceleration.
    assert pk_v < coarse_v
    assert pk_a < coarse_a


def test_path_peak_joint_speed_zero_for_degenerate():
    assert cl.path_peak_joint_speed([(0.0,) * 6]) == (0.0, 0.0)
    assert cl.timed_path_peak_joint_speed([0.0], [(0.0,) * 6]) == (0.0, 0.0)


# --------------------------------------------------------- the refusal path (genuinely un-levelable)

@need_urdf
def test_compensated_approach_over_bound_is_rejectable(robot, up, gc):
    """A compensated approach whose predicted EXECUTED tilt would still exceed the TRUE bound yields a
    LevelPathPlan.reject message (the caller fails by name BEFORE any motion). Built by starting the carried
    approach from a genuinely tilted start the leveller cannot reach back from within the clear region."""
    goal, _ = cl.solve_level_standoff(robot, STANDOFF, up, list(FIXED_SHAPE), gc, tool_link=TL,
                                      limits=_limits())
    goalv = [goal[j] for j in ARM_JOINTS]
    # a start whose carton is far off level AND outside the wrist clear region levels to its best in-region
    # residual; with a very TIGHT bound the gate rejects.
    tilted_start = (goalv[0], goalv[1], goalv[2], 1.4, -1.2, 0.0)
    comp = cl.plan_compensated_approach(robot, list(tilted_start), list(goalv), up, gc,
                                        tool_link=TL, limits=_limits(), bound_deg=0.05, max_segment_rad=0.25)
    # against a 0.05 deg bound the first-waypoint residual (the pinned tilted start) is over it -> reject.
    assert comp.reject(0.05) is not None
    assert comp.reject(0.05).startswith(cl.CARTON_LEVEL_PLAN_REJECTED)


# --------------------------------------------------------- config_preflight coherence

def test_preflight_gate_without_above_flag_refused():
    from twin.restock_manipulation import config_preflight as cp
    probs = cp.check_carton_level({'carton_level': {'preplace_level_gate': True}})
    assert any('preplace_level_gate' in p and 'inert' in p for p in probs)


def test_preflight_gate_with_above_support_not_flagged():
    from twin.restock_manipulation import config_preflight as cp
    probs = cp.check_carton_level({'carton_level': {'preplace_level_gate': True,
                                                    'preplace_above_support': True}})
    assert not any('preplace_level_gate' in p for p in probs)


def test_preflight_off_is_byte_identical():
    from twin.restock_manipulation import config_preflight as cp
    assert cp.check_carton_level({'carton_level': True}) == []
    assert cp.check_carton_level({}) == []


def test_preflight_rejects_malformed_gate_key():
    from twin.restock_manipulation import config_preflight as cp
    probs = cp.check_carton_level({'carton_level': {'preplace_gate_bound_deg': 9.0}})   # > bound 2.0
    assert probs and 'malformed' in probs[0]


# --------------------------------------------------------- controller dispatch wiring

def _ctrl(**kw):
    from twin.restock_manipulation import TransferController
    from twin.tests.test_restock_manipulation import PROFILE, FixtureMotion
    return TransferController(FixtureMotion(), {'sku-box': PROFILE}, fixture_mode=True, **kw)


@need_urdf
def test_gated_preplace_dispatch_drives_densified_capped_begin_joint_path(robot, up, gc):
    """The controller's gated preplace dispatch drives the droop-compensated, densified approach as a
    begin_joint_path at the carried speed/accel caps, logs the gate, and stashes the planned preplace tilt
    for CARTON_SEGMENT_TILT."""
    lc = cl.resolve_level_config({'preplace_level_gate': True, 'preplace_above_support': True,
                                  'gravity_compensation': {'stiffness_nm_per_rad': 2850, 'carton_mass_kg': 0.45},
                                  'closed_loop_level': True, 'closed_loop_sensor': True})
    c = _ctrl(carton_level=lc, fk_urdf=URDF, carry_wrist1_cap=0.65, carry_wrist2_min=0.0)
    c.operation = 'collect'
    c.phase = 'preplace'
    c.issued = False
    c._capture_level_up(list(WELD))
    goal, _ = cl.solve_level_standoff(robot, STANDOFF, up, list(FIXED_SHAPE), gc, tool_link=TL,
                                      limits=_limits())
    goalv = tuple(goal[j] for j in ARM_JOINTS)
    transit = cl.level_place_ik(robot, (0.55, 0.05, 0.80), up, list(FIXED_SHAPE),
                                tool_link=TL, limits=_limits(), gravity_comp=gc)
    measured = tuple(transit[j] for j in ARM_JOINTS)
    res = c._issue_gated_preplace_approach(goalv, measured)
    assert res.status != 'failed', res.reason
    assert c.issued is True
    # ONE begin_joint_path dispatched at the carried caps, ending at the leveled standoff.
    jp = c.motion.joint_paths
    assert len(jp) == 1
    assert jp[0]['speed'] == pytest.approx(lc.preplace_approach_speed)
    assert jp[0]['accel'] == pytest.approx(lc.preplace_approach_accel)
    assert list(jp[0]['waypoints'][0]) == pytest.approx(list(measured))
    assert list(jp[0]['waypoints'][-1]) == pytest.approx(list(goalv))
    # densified (more than a trivial 2-point move) and the planned preplace tilt is stashed for the log.
    assert len(jp[0]['waypoints']) >= cl.densify_count(measured, goalv, lc.preplace_max_segment_rad)
    assert 'preplace' in c._carton_seg_planned
    pr, pp = c._carton_seg_planned['preplace']
    assert max(pr, pp) <= 1.0


def test_gate_off_does_not_build_the_gated_path():
    # off => no gate attribute engagement: resolve(True) has preplace_level_gate False, so the controller's
    # preplace dispatch takes the legacy _issue_level_approach_move (byte-identical). Guard the flag only.
    assert cl.resolve_level_config(True).preplace_level_gate is False
