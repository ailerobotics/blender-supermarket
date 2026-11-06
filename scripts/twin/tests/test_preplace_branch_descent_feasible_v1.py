"""CPU proof for the DESCENT-FEASIBLE carry-connected standoff select (sim-preplace-branch-descent-feasible-v1).

coldwin79 sealed the pick and converged the pre-place, then the carton-level PLACE-DESCENT gate refused the
set-down BEFORE motion because the descent from the chosen standoff would have rolled the carton 3.43 deg
(> the 2.0 deg bound). Runs 76/77/78 descended level from the SAME configuration. The difference: the
carry-connected standoff selector (``select_carry_connected_standoff``) ranked candidate standoffs only by
carry travel + the leg-2 continuity gate and greedily returned the FIRST leg-2-accepted branch; it never
checked whether the place descent FROM that standoff keeps the carton level. The candidate set and each
branch's leg-2 continuity result depend on the run's measured carry pose / sensed place target (which vary a
few mm / mrad run-to-run), so coldwin79 greedily accepted the low wrist_2=0.052 branch whose descent is
infeasible, while 76/77/78 accepted a ~0.59 branch whose descent is level -- making the set-down a lottery.

The fix (opt-in ``preplace_descent_feasible``): the selector is handed a ``descent_predictor`` (the SAME
``plan_compensated_place_descent`` the place-descent gate runs, from each CANDIDATE standoff to the resolved
support), DISCARDS any candidate whose predicted descent exceeds the bound, and chooses deterministically by
the stable key (least carry travel, then descent tilt, then leg-2 step, then wrist_2 closest to carry) --
independent of the search time / enumeration order.

This reconstructs the coldwin79 / coldwin76 pre-place on the real URDF + logged vectors (host-stack.log:
place=(0.956,0.077,0.538) cw79 / (0.957,0.07,0.538) cw76, measured carry [-0.4248,-2.27,1.0,0.65,0.002,0.619],
first-carried/weld pose [1.3157,-1.2905,2.1095,-0.8202,1.2939,0], collect_place_preplace_shape
[0.093,-1.91,2.014,-0.103,1.664,0], base pitch -1.17) and shows:

  1. the REAL place-descent predictor classifies coldwin79's low-wrist_2 standoff as descent-INFEASIBLE
     (>2.0 deg, reproducing the logged 3.43 deg signature) and a mid-wrist_2 standoff as FEASIBLE (~1.8 deg);
  2. the selector DISCARDS a leg-2-accepted-but-descent-infeasible branch and chooses the next descent-feasible
     one (the coldwin79 pattern: the previously chosen branch is discarded, a level branch is chosen);
  3. the choice is DETERMINISTIC across 20 runs under artificial timing jitter;
  4. coldwin76's choice is UNCHANGED by the filter (its greedy branch was already descent-feasible);
  5. ``descent_predictor=None`` is byte-identical to the legacy call; the HOST key resolves, default OFF is
     byte-identical, and the preflight refuses the key without the carry-connected selector.

    PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=scripts python3 -m pytest \
        scripts/twin/tests/test_preplace_branch_descent_feasible_v1.py
"""
import math
import os
import random

import numpy as np
import pytest

from twin.restock_manipulation import carton_level as cl
from twin.restock_manipulation.cad_tool import ARM_JOINTS

_URDF_CANDIDATES = (
    '/home/manishsaini/datasets/twin-robot-in-store-boot-v1/mission/coldwin79/mobile-import.urdf',
    '/home/manishsaini/datasets/twin-robot-in-store-boot-v1/mission/coldwin76/mobile-import.urdf',
    '/home/manishsaini/datasets/twin-robot-in-store-boot-v1/mission/coldwin72/mobile-import.urdf',
)
URDF = next((p for p in _URDF_CANDIDATES if os.path.exists(p)), _URDF_CANDIDATES[0])
need_urdf = pytest.mark.skipif(not any(os.path.exists(p) for p in _URDF_CANDIDATES),
                               reason='no mobile URDF present')
TL = 'gripper_pad_tcp'
PITCH_DEG = -1.17
LJ = cl.DEFAULT_LEVEL_JOINTS
W2I = ARM_JOINTS.index('ur_wrist_2_joint')
# coldwin79 logged vectors (host-stack.log). Order [pan, shoulder_lift, elbow, wrist_1, wrist_2, wrist_3].
WELD = [1.3157, -1.2905, 2.1095, -0.8202, 1.2939, 0.0]            # first carried/lift pose (tilt 0)
CARRY = [-0.4248, -2.2700, 1.0002, 0.6502, 0.0020, 0.6189]        # measured carry at the pan-first leg-2 start
SEED = [0.093, -1.91, 2.014, -0.103, 1.664, 0.0]                  # collect_place_preplace_shape
PLACE_CW79 = (0.956, 0.077, 0.538)
PLACE_CW76 = (0.957, 0.07, 0.538)
STANDOFF_CLEARANCE = 0.08                                         # preplace_above_support default clearance
# the place-descent options the gate uses (from the cw79 CARTON_LEVEL_PLACE_DESCENT log line)
ABOVE_CLEARANCE = 0.08
DEPTH_BIAS = 0.05
WAYPOINTS = 12
MAX_SEG = 0.08
GATE = 1.0
BOUND = 2.0


def _limits():
    big = 2 * math.pi
    return {'ur_elbow_joint': (0.1, math.pi),
            'ur_wrist_1_joint': (-big, 0.65), 'ur_wrist_2_joint': (0.0, big)}


@pytest.fixture(scope='module')
def ctx():
    from twin.urdf_probe import parse_urdf
    robot = parse_urdf(URDF)
    base_R = cl.base_rotation_rp(0.0, PITCH_DEG)
    R = cl.tool_rotation(robot, WELD, TL)
    up_world = cl.carton_up_in_tool(np.asarray(base_R, float) @ R)
    up_base = cl.carton_up_in_tool(R)
    lim = _limits()
    gc = cl.GravityCompensation(tool_link=TL).validate()
    # the descent IK seed = the leveled fixed pre-place shape (exactly what _level_place_goal passes)
    place_seed = cl.level_preserve_tcp(robot, SEED, up_base, tool_link=TL, limits=lim, gravity_comp=None)
    place_seed_vec = [float(place_seed[j]) for j in ARM_JOINTS]
    return robot, base_R, up_world, up_base, lim, gc, place_seed_vec


def _standoff(place):
    return (float(place[0]), float(place[1]), float(place[2]) + STANDOFF_CLEARANCE)


def _real_descent_predictor(ctx, place):
    """The REAL place-descent tilt predictor the controller builds: the same ``plan_compensated_place_descent``
    the CARTON_LEVEL_PLACE_DESCENT gate runs, from a candidate standoff to the resolved support."""
    robot, base_R, up_world, up_base, lim, gc, place_seed_vec = ctx

    def pred(std_map):
        std_vec = [float(std_map[j]) for j in ARM_JOINTS]
        plan = cl.plan_compensated_place_descent(
            robot, std_vec, place, list(place_seed_vec), up_base, gc, bound_deg=BOUND, tool_link=TL,
            limits=lim, waypoints=WAYPOINTS, approach_from_above=True, above_clearance_m=ABOVE_CLEARANCE,
            depth_bias_m=DEPTH_BIAS)
        return max(plan.max_roll, plan.max_pitch)

    return pred


def _select_kw(ctx):
    _, base_R, _, _, lim, _, _ = ctx
    return dict(tool_link=TL, limits=lim, base_R=base_R, level_joints=LJ, max_segment_rad=MAX_SEG,
                bound_deg=BOUND, max_wrist_step_rad=0.15, seeded_interior=True, level_accept_deg=GATE,
                budget_s=30.0)


# --------------------------------------------------------------------------- 1. real descent classification

@need_urdf
def test_cw79_low_wrist2_descent_infeasible_mid_feasible(ctx):
    """The REAL place-descent predictor (the gate's own ``plan_compensated_place_descent``) classifies the
    coldwin79 LOW-wrist_2 standoff as descent-INFEASIBLE (> 2.0 deg, reproducing the logged 3.43 deg) and a
    MID-wrist_2 standoff as FEASIBLE (< 2.0 deg) -- the physical fact the selector must act on."""
    robot, base_R, up_world, up_base, lim, gc, _ = ctx
    std_t = _standoff(PLACE_CW79)
    pred = _real_descent_predictor(ctx, PLACE_CW79)
    branches = cl.enumerate_carry_connected_branches(robot, std_t, up_world, SEED, CARRY, tool_link=TL,
                                                     limits=lim, base_R=base_R, fast=True)
    by_w2 = {round(float(b['ur_wrist_2_joint']), 3): b for b in branches}
    low = min(by_w2, key=lambda w: abs(w))          # the carry-connected low branch (coldwin79 chose ~0.05)
    mids = [w for w in by_w2 if 0.4 <= w <= 0.7]
    assert mids, 'expected a mid-wrist_2 standoff branch'
    low_tilt = pred(by_w2[low])
    mid_tilt = pred(by_w2[min(mids)])
    # the low branch descent exceeds the bound (coldwin79's refusal, logged 3.43 deg); the mid is level
    assert low_tilt > BOUND, f'low wrist_2={low} descent {low_tilt:.2f} should exceed {BOUND}'
    assert low_tilt > 3.0, f'low branch descent {low_tilt:.2f} should reproduce the ~3.43 deg signature'
    assert mid_tilt <= BOUND, f'mid wrist_2 descent {mid_tilt:.2f} should be within {BOUND}'


# ------------------------------------------------------------ 2. the selector discards the infeasible branch

@need_urdf
def test_selector_discards_descent_infeasible_branch(ctx):
    """The selector DISCARDS a leg-2-accepted-but-descent-infeasible branch and chooses the next
    descent-feasible one -- the coldwin79 pattern. A descent_predictor that marks the lowest-travel feasible
    branch as INFEASIBLE (the real cw79 monotonic structure: low wrist_2 => high descent tilt) forces the
    discard; without the predictor the selector returns that very branch (the greedy lottery choice)."""
    robot, base_R, up_world, up_base, lim, gc, _ = ctx
    std_t = _standoff(PLACE_CW79)
    kw = _select_kw(ctx)
    # legacy (no descent filter): the greedy lowest-travel leg-2-accepted branch
    sol_legacy, _ = cl.select_carry_connected_standoff(robot, std_t, up_world, SEED, CARRY,
                                                       descent_predictor=None, **kw)
    w2_legacy = float(sol_legacy['ur_wrist_2_joint'])
    # a predictor matching the real monotonic structure: a branch at/below the greedy choice has an
    # infeasible descent (like coldwin79's low wrist_2), branches above it are level.
    thresh = w2_legacy + 0.01
    real = _real_descent_predictor(ctx, PLACE_CW79)

    def pred(std_map):
        if float(std_map['ur_wrist_2_joint']) <= thresh:
            return 3.43           # the logged coldwin79 infeasible descent
        return real(std_map)      # the real (level) descent for the higher branches

    sol_filt, rep = cl.select_carry_connected_standoff(robot, std_t, up_world, SEED, CARRY,
                                                       descent_predictor=pred, **kw)
    w2_filt = float(sol_filt['ur_wrist_2_joint'])
    # the greedy branch is discarded; a DIFFERENT, descent-feasible branch is chosen
    assert w2_filt > w2_legacy, f'filter should pick a higher branch than the discarded {w2_legacy:.3f}'
    assert rep['descent_feasible_gate'] is True
    assert rep['chosen_descent_tilt_deg'] is not None and rep['chosen_descent_tilt_deg'] <= BOUND
    # the chosen branch's descent (via the real predictor) is within the bound -- the place gate accepts it
    assert real(sol_filt) <= BOUND
    # the ranked report records both scores and marks the discarded branch as descent-infeasible
    disc = [b for b in rep['branches'] if b['accepted'] and b.get('descent_ok') is False]
    assert disc, 'the report should record at least one leg-2-accepted-but-descent-infeasible branch'


# -------------------------------------------------------------------------------------- 3. determinism

@need_urdf
def test_choice_deterministic_under_timing_jitter(ctx):
    """20 selects with an artificially JITTERED clock return the IDENTICAL chosen standoff -- the ranking is
    independent of the search time (the lottery was run-to-run INPUT drift, never the clock)."""
    robot, base_R, up_world, up_base, lim, gc, _ = ctx
    std_t = _standoff(PLACE_CW79)
    kw = _select_kw(ctx)
    real = _real_descent_predictor(ctx, PLACE_CW79)
    w2_legacy = float(cl.select_carry_connected_standoff(robot, std_t, up_world, SEED, CARRY,
                                                         descent_predictor=None, **kw)[0]['ur_wrist_2_joint'])
    thresh = w2_legacy + 0.01

    def pred(std_map):
        return 3.43 if float(std_map['ur_wrist_2_joint']) <= thresh else real(std_map)

    rng = random.Random(1234)
    chosen = set()
    for _ in range(20):
        jitter = rng.uniform(-0.4, 0.4)

        def jclock(_j=jitter):
            import time
            return time.perf_counter() + _j

        sol, _ = cl.select_carry_connected_standoff(robot, std_t, up_world, SEED, CARRY,
                                                    descent_predictor=pred, clock=jclock,
                                                    **{k: v for k, v in kw.items() if k != 'budget_s'},
                                                    budget_s=60.0)
        chosen.add(round(float(sol['ur_wrist_2_joint']), 6))
    assert len(chosen) == 1, f'non-deterministic choices under timing jitter: {chosen}'


# ------------------------------------------------------------------------------ 4. coldwin76 unchanged

@need_urdf
def test_cw76_choice_unchanged_by_filter(ctx):
    """coldwin76's greedy branch was already descent-feasible, so the descent filter leaves its choice
    UNCHANGED (the fix discards only the infeasible lottery branches, never a run that was already level)."""
    robot, base_R, up_world, up_base, lim, gc, _ = ctx
    std_t = _standoff(PLACE_CW76)
    kw = _select_kw(ctx)
    pred = _real_descent_predictor(ctx, PLACE_CW76)
    sol_legacy, _ = cl.select_carry_connected_standoff(robot, std_t, up_world, SEED, CARRY,
                                                       descent_predictor=None, **kw)
    sol_filt, rep = cl.select_carry_connected_standoff(robot, std_t, up_world, SEED, CARRY,
                                                       descent_predictor=pred, **kw)
    assert float(sol_filt['ur_wrist_2_joint']) == pytest.approx(float(sol_legacy['ur_wrist_2_joint']),
                                                                abs=1e-6)
    # and the chosen branch's descent is level (within the bound) -- the gate would accept it
    assert rep['chosen_descent_tilt_deg'] is not None and rep['chosen_descent_tilt_deg'] <= BOUND


# --------------------------------------------------------------------------- 5. byte-identical when off

@need_urdf
def test_descent_predictor_none_byte_identical(ctx):
    """``descent_predictor=None`` is byte-identical to the no-param (legacy) call: same standoff, no descent
    evaluation, on BOTH coldwin79 and coldwin76 inputs."""
    robot, base_R, up_world, up_base, lim, gc, _ = ctx
    kw = _select_kw(ctx)
    for place in (PLACE_CW79, PLACE_CW76):
        std_t = _standoff(place)
        sol_a, rep_a = cl.select_carry_connected_standoff(robot, std_t, up_world, SEED, CARRY, **kw)
        sol_b, rep_b = cl.select_carry_connected_standoff(robot, std_t, up_world, SEED, CARRY,
                                                          descent_predictor=None, **kw)
        assert [float(sol_a[j]) for j in ARM_JOINTS] == [float(sol_b[j]) for j in ARM_JOINTS]
        assert rep_a['descent_feasible_gate'] is False
        assert rep_b['descent_feasible_gate'] is False
        assert rep_a['chosen_descent_tilt_deg'] is None


# ------------------------------------------------------------------------------ 6. config resolve + preflight

def test_resolve_enables_descent_feasible():
    """The HOST carton_level key resolves the flag; default off => byte-identical."""
    assert cl.resolve_level_config(True).preplace_descent_feasible is False
    cfg = cl.resolve_level_config({'preplace_descent_feasible': True})
    assert cfg.preplace_descent_feasible is True


def test_preflight_descent_feasible_without_carry_connected_refused():
    """``preplace_descent_feasible`` ON with ``preplace_branch_prefer_carry_connected`` OFF is inert -> refused."""
    from twin.restock_manipulation import config_preflight as cp
    probs = cp.check_carton_level({'carton_level': {'preplace_descent_feasible': True}})
    assert any('preplace_descent_feasible' in p and 'inert' in p for p in probs)


def test_preflight_descent_feasible_with_carry_connected_not_flagged():
    """With the carry-connected selector (+ reaction branch + level gate + above-support) on, the
    descent-feasible key is not flagged."""
    from twin.restock_manipulation import config_preflight as cp
    probs = cp.check_carton_level({'carton_level': {
        'preplace_descent_feasible': True, 'preplace_branch_prefer_carry_connected': True,
        'preplace_reaction_branch': True, 'preplace_level_gate': True, 'preplace_above_support': True,
        'preplace_seeded_interior': True}})
    assert not any('preplace_descent_feasible' in p for p in probs)
