"""CPU proof test: the carton-level bench keeps every CARRIED segment under the 2 deg bound while
the unleveled fold tilts the carton well off level. No Isaac/ROS/GPU.

    PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=scripts python3 -m pytest scripts/twin/tests/test_carton_level_bench.py
"""
import os

import pytest

from twin.restock_manipulation import carton_level as cl

URDF = '/home/manishsaini/datasets/twin-robot-in-store-boot-v1/mission/coldwin16/mobile-import.urdf'
pytestmark = pytest.mark.skipif(not os.path.exists(URDF), reason='mission URDF not present')


def test_bench_before_tilts_after_is_level_and_clear():
    from twin.tools import carton_level_bench as b
    u, res = b.run(URDF, bound_deg=cl.DEFAULT_TILT_BOUND_DEG)
    assert {r.name for r in res} == {'lift', 'tuck', 'pan', 'preplace', 'place'}
    # the unleveled fold really does tilt the carton off level somewhere in the carry
    assert max(r.before_max_tilt for r in res) > 10.0
    # the level constraint holds every carried segment under the bound AND in the clear wrist region
    for r in res:
        assert r.after_max_roll <= cl.DEFAULT_TILT_BOUND_DEG, r.name
        assert r.after_max_pitch <= cl.DEFAULT_TILT_BOUND_DEG, r.name
        assert r.after_in_clear_wrist, f'{r.name} leveled out of the clear wrist region'
    assert b.all_pass(res)


def test_bench_tighter_bound_still_holds():
    from twin.tools import carton_level_bench as b
    _, res = b.run(URDF, bound_deg=1.0, n=12)
    assert all(r.after_max_roll <= 1.0 and r.after_max_pitch <= 1.0 for r in res)


# ----------------------------------------- sim-carton-tilt-trip-margin-v1: real collect carry

def test_top_grasp_weld_is_top_down():
    """The reconstructed collect grasp is TOP-DOWN: the suction plates point world-down and the level
    carton's up-axis reads ~[0,-1,0] in the tool frame (the coldwin24 grasp, not the coldwin16 side grip)."""
    import numpy as np
    from twin.urdf_probe import parse_urdf
    from twin.tools.carton_level_bench import TOP_GRASP_WELD
    r = parse_urdf(URDF)
    R = cl.tool_rotation(r, list(TOP_GRASP_WELD))
    plate_forward_world = R @ np.array([0.0, 1.0, 0.0])   # plate +Z is [0,1,0] in tool0
    assert plate_forward_world == pytest.approx([0, 0, -1], abs=1e-2)
    u = cl.carton_up_in_tool(R)
    assert u == pytest.approx([0, -1, 0], abs=1e-2)
    assert cl.up_axis_rpy(R @ u)[2] == pytest.approx(0, abs=1e-6)   # carton level at the weld


def test_real_collect_carry_tilts_unleveled_and_levels_after():
    """Reproduce coldwin24: every REAL collect carry shape (carry_lift_shape, tuck/pan,
    collect_place_preplace_shape) tilts the top-grasped carton well off level unleveled, and the fixed
    leveling drives each UNDER 0.1 deg within the clear wrist region (the brief's before/after proof)."""
    from twin.tools import carton_level_bench as b
    u, res = b.run_collect(URDF, bound_deg=cl.DEFAULT_TILT_BOUND_DEG)
    assert {name for name, *_ in res} == {'lift', 'pan', 'place_preplace'}
    for name, before, after, wrists, clear in res:
        assert before > 2.0, f'{name} should reproduce the off-level carry ({before:.1f} deg)'
        assert after < 0.1, f'{name} not leveled under 0.1 deg (got {after:.3f})'
        assert clear, f'{name} leveled out of the clear wrist region'


def _old_single_start_2d_solver(robot, goal, u, joints, limits, max_iter=80, tol_deg=0.05):
    """The PRE-FIX leveling: ONE start (the goal), damped Gauss-Newton on the 2-D horizontal residual
    ``up_world x world_up`` -- which vanishes at BOTH the upright and the inverted (180 deg) pole. Kept
    here only to prove the fix: it raises on a >90 deg start where a clear level branch actually exists."""
    import numpy as np
    cfg = cl._as_joint_map(goal)
    joints = [j for j in joints if j in cfg]
    uu = cl._unit(u)

    def up_world(m):
        return cl.tool_rotation(robot, m, 'ur_tool0') @ uu

    def resid(m):
        return np.cross(up_world(m), cl.WORLD_UP)[:2]

    def tilt(m):
        return abs(cl.up_axis_rpy(up_world(m))[2])

    for _ in range(max_iter):
        if tilt(cfg) <= tol_deg:
            return cfg
        r0 = resid(cfg)
        J = np.zeros((2, len(joints)))
        for k, j in enumerate(joints):
            m2 = dict(cfg); m2[j] = cfg[j] + 1e-4
            J[:, k] = (resid(m2) - r0) / 1e-4
        dq = -np.linalg.solve(J.T @ J + 1e-6 * np.eye(len(joints)), J.T @ r0)
        mx = float(np.max(np.abs(dq))) if dq.size else 0.0
        if mx > 0.3:
            dq *= 0.3 / mx
        for k, j in enumerate(joints):
            v = cfg[j] + float(dq[k])
            if limits and j in limits:
                lo, hi = limits[j]; v = min(max(v, lo), hi)
            cfg[j] = v
    if tilt(cfg) <= max(tol_deg, 1e-3):
        return cfg
    raise ValueError('no level-orientation solution (old 2-D-residual single-start solver)')


def test_level_solver_escapes_the_180_basin():
    """Regression guard for the fix: the place-on-cart standoff (collect_place_preplace_shape) holds the
    top-grasped carton ~180 deg off level -- squarely in the inverted basin where the OLD single-start
    2-D-residual solver stalled and raised 'no solution' (coldwin24/24b). A clear-region level branch
    EXISTS; the fixed multi-start 3-D-residual solver must find it."""
    from twin.urdf_probe import parse_urdf
    from twin.tools.carton_level_bench import TOP_GRASP_WELD, clear_wrist_limits
    r = parse_urdf(URDF)
    import json
    cfg = json.load(open(os.path.join(os.path.dirname(cl.__file__), 'ros_host_config.json')))
    preplace = [float(v) for v in cfg['collect_place_preplace_shape']]
    u = cl.carton_up_in_tool(cl.tool_rotation(r, list(TOP_GRASP_WELD)))
    limits = clear_wrist_limits()
    assert cl.carried_rpy(r, preplace, u)[2] > 90.0        # start is in the inverted (180) basin
    # the OLD solver cannot escape the inverted basin from this start
    with pytest.raises(ValueError):
        _old_single_start_2d_solver(r, preplace, u, cl.DEFAULT_LEVEL_JOINTS, limits)
    # the FIXED solver finds the clear-region upright branch
    lev = cl.level_config(r, preplace, u, limits=limits, accept_deg=cl.DEFAULT_TILT_BOUND_DEG)
    assert cl.carried_rpy(r, lev, u)[2] < 0.1
    assert lev['ur_wrist_1_joint'] <= 0.65 + 1e-9 and lev['ur_wrist_2_joint'] >= -1e-9


def test_run_place_levels_the_whole_place_at_the_same_reach():
    """sim-carton-level-place-on-cart-v1: the LEVEL place-on-cart proof. The reach-authored standoff
    holds the top-grasped carton ~180 deg off level; the re-planned level place reaches the SAME deck
    TCP (a near-free wrist_3 flange roll on the same arm branch) and the single straight-line descent to
    a level set-down on the deck stays under 0.1 deg off level throughout."""
    from twin.tools import carton_level_bench as b
    r = b.run_place(URDF)
    assert r.preplace_before_tilt > 175.0            # reach shape holds it inverted
    assert r.preplace_after_tilt < 0.1               # standoff leveled
    assert r.preplace_tcp_shift_mm < 1.0             # reach preserved
    assert r.setdown_after_tilt < 0.1                # level set-down
    assert r.setdown_pos_err_mm < 5.0                # reaches the sensed deck
    assert r.descent_max_tilt < 0.1                  # level THROUGHOUT the descent
    assert r.preplace_clear and r.setdown_clear      # in the clear carry wrist region
    assert r.preplace_wrist3_roll_deg > 90.0         # leveling is a flange roll
    assert r.preplace_arm_shift_deg < 2.0            # ... on the same arm branch (reach validation held)
    assert r.passes(cl.DEFAULT_TILT_BOUND_DEG)
