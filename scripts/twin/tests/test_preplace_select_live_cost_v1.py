"""CPU proof for the reaction-branch SELECT cost fix (sim-preplace-select-live-cost-v1, coldwin60).

coldwin60 refused the preplace because ``select_reaction_natural_standoff`` took 28.4 s on the REAL inputs
(not the ~0.08 s the coldwin57 fixture measured -- the fixture's standoff target IS its seed's own TCP, so
``level_place_ik`` reaches it in zero iterations, while the real target is ~0.5 m off the seed and forces
the full damped-least-squares descent). Profiling showed ~61 000 whole-robot FK calls (each rebuilding the
36-link mobile tree though only ``ur_tool0`` is needed). The fix (``fast=True``, default for the selector)
routes every solve through a cached tool-chain FK -- the SAME finite-difference arithmetic on only the
root->tool_link chain -- cutting the real select to ~1.5 s while keeping the chosen standoff within ~1e-4
rad of the slow whole-tree FK.

These tests reconstruct the exact coldwin60 inputs (coldwin60-select-cost.txt) and assert:
  1. the fast select is well under the 2 s budget on CPU, and
  2. its chosen standoff is identical (within 1e-3 rad) to the slow (whole-tree FK) reference.

    PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=scripts python3 -m pytest \
        scripts/twin/tests/test_preplace_select_live_cost_v1.py
"""
import math
import os
import time

import numpy as np
import pytest

from twin.restock_manipulation import carton_level as cl
from twin.restock_manipulation.cad_tool import ARM_JOINTS

_URDF_CANDIDATES = (
    '/home/manishsaini/datasets/twin-robot-in-store-boot-v1/mission/coldwin60/mobile-import.urdf',
    '/home/manishsaini/datasets/twin-robot-in-store-boot-v1/mission/coldwin57/mobile-import.urdf',
    '/home/manishsaini/datasets/twin-robot-in-store-boot-v1/mobile-casterlift/mobile_ur10.urdf',
)
URDF = next((p for p in _URDF_CANDIDATES if os.path.exists(p)), _URDF_CANDIDATES[0])
need_urdf = pytest.mark.skipif(not any(os.path.exists(p) for p in _URDF_CANDIDATES),
                               reason='no mobile URDF present')
TL = 'ur_tool0'
PITCH_DEG = -1.17
GRASP = [1.1923, -1.2833, 2.1071, -0.8249, 1.1932, -0.0001]
SEED = [0.093, -1.91, 2.014, -0.103, 1.664, 0.0]
CARRY = [1.2675, -2.27, 1.0, 0.65, 0.8, -1.2545]
TARGET = (0.53, 0.45, 0.54)
BUDGET_S = 2.0


def _limits():
    big = 2 * math.pi
    return {'ur_elbow_joint': (0.1, math.pi),
            'ur_wrist_1_joint': (-big, 0.65), 'ur_wrist_2_joint': (0.0, big)}


@pytest.fixture(scope='module')
def ctx():
    from twin.urdf_probe import parse_urdf
    robot = parse_urdf(URDF)
    base_R = cl.base_rotation_rp(0.0, PITCH_DEG)
    up = cl.carton_up_in_tool(np.asarray(base_R, float) @ cl.tool_rotation(robot, GRASP, TL))
    return robot, base_R, up, _limits()


def _select(ctx, fast, budget_s=None):
    robot, base_R, up, lim = ctx
    return cl.select_reaction_natural_standoff(
        robot, TARGET, up, SEED, CARRY, tool_link=TL, limits=lim, base_R=base_R,
        effort_limit=56.0, bound_deg=2.0, max_segment_rad=0.08, fast=fast, budget_s=budget_s)


@need_urdf
def test_fast_select_is_under_budget_on_real_inputs(ctx):
    """The fast select finishes well under the 2 s CPU budget on the reconstructed coldwin60 inputs
    (coldwin60 refused at 28.4 s with the whole-tree FK)."""
    t0 = time.perf_counter()
    std, rep = _select(ctx, fast=True, budget_s=BUDGET_S)   # budget raises if it ever regresses past 2 s
    elapsed = time.perf_counter() - t0
    assert elapsed < BUDGET_S, 'fast select took %.3f s (budget %.1f s)' % (elapsed, BUDGET_S)
    assert rep['select_seconds'] < BUDGET_S
    assert set(std) >= set(ARM_JOINTS)


@need_urdf
def test_fast_select_matches_slow_within_1e_3(ctx):
    """The fast (cached tool-chain FK) select returns the SAME chosen standoff as the slow whole-tree FK
    reference -- same branch, joints within 1e-3 rad (the identity coldwin60 would have produced)."""
    std_fast, rep_fast = _select(ctx, fast=True)
    std_slow, rep_slow = _select(ctx, fast=False)
    assert abs(rep_fast['chosen_wrist_2'] - rep_slow['chosen_wrist_2']) < 1e-3
    assert max(abs(float(std_fast[j]) - float(std_slow[j])) for j in ARM_JOINTS) < 1e-3
    # the enumerated branch SET matches within 1e-3 rad branch-by-branch
    bw_f = sorted(b['wrist_2'] for b in rep_fast['branches'])
    bw_s = sorted(b['wrist_2'] for b in rep_slow['branches'])
    assert len(bw_f) == len(bw_s)
    assert max(abs(a - b) for a, b in zip(bw_f, bw_s)) < 1e-3


@need_urdf
def test_fast_is_much_cheaper_than_slow(ctx):
    """Sanity: the fast FK is a large speedup over the whole-tree FK on the real inputs (not just equal)."""
    t0 = time.perf_counter(); _select(ctx, fast=True); t_fast = time.perf_counter() - t0
    t0 = time.perf_counter(); _select(ctx, fast=False); t_slow = time.perf_counter() - t0
    assert t_slow > 5 * t_fast, 'expected a large speedup; fast=%.3f slow=%.3f' % (t_fast, t_slow)
