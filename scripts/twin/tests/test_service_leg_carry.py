"""CPU guard for the sim-service-leg-carry-v1 proof (scripts/check_service_leg_carry.py). Pure helpers +
reach + tilt are always exercised; the full store-map path proof runs when the generated environment
manifest + mission URDF are present (skip-with-reason otherwise, matching test_carton_level_bench)."""
import math
import os
import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(SCRIPTS))
import importlib.util

_spec = importlib.util.spec_from_file_location('check_service_leg_carry',
                                               SCRIPTS / 'check_service_leg_carry.py')
slc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(slc)

REPO = SCRIPTS.parent
ENV = Path.home() / 'datasets/sim-store-realism-v3-portal26/environment.json'
URDF = Path.home() / 'datasets/twin-robot-in-store-boot-v1/mission/coldwin16/mobile-import.urdf'
SCENARIO = SCRIPTS / 'restock_scenario.json'
LANES = REPO / 'docs/examples/service-leg-carry-mission-fragment.json'
LANES_FALLBACK = REPO / 'docs/examples/tow-lanes-mission-fragment.json'


# ---- pure helpers (always on) -------------------------------------------------------------------

def test_rect_forward_offset_places_the_carton_ahead_of_the_base():
    # base at origin facing +x; a fwd=0.4 carton rect is centred 0.4 m ahead, base rect centred at 0.
    base = slc._rect((0.0, 0.0, 0.0), 0.5, 0.3)
    carton = slc._rect((0.0, 0.0, 0.0), 0.225, 0.15, fwd=0.4)
    assert min(c[0] for c in base) == pytest.approx(-0.5)
    assert min(c[0] for c in carton) == pytest.approx(0.4 - 0.225)
    assert max(c[0] for c in carton) == pytest.approx(0.4 + 0.225)


def test_approach_path_never_rotates_at_the_goal():
    # The return-to-staging path rotates to the goal yaw at the OPEN via point, then translates in holding
    # that yaw -- so every sample within the last translate leg is already at the goal yaw (no in-place spin
    # adjacent to the parked cart, which _staged_path's terminal rotate would do).
    start, via, goal = (-3.05, 6.0, math.pi), (-3.875, 6.5, math.pi), (-3.875, 8.0, -math.pi / 2)
    path = slc._approach_path(start, via, goal, n=20)
    assert path[-1][:2] == pytest.approx(goal[:2])
    # the final translate leg (last n samples) all hold the goal yaw
    for p in path[-20:]:
        assert slc._wrap(p[2] - goal[2]) == pytest.approx(0.0, abs=1e-9)
    # and the staged (carry) path DOES rotate in place at its goal -- the contrast that motivates _approach
    staged = slc._staged_path(start, via, goal, n=20)
    assert any(abs(slc._wrap(p[2] - goal[2])) > 1e-6 and p[:2] == pytest.approx(goal[:2]) for p in staged)


def test_sweep_with_carton_catches_a_clip_the_bare_base_clears():
    # A box just ahead of the base path is cleared by the bare base but HIT by the carton footprint union --
    # proving the swept envelope actually includes the carried carton.
    path = [(0.0, 0.0, 0.0)]
    from twin.mobile import MobileSpec
    spec = MobileSpec()                       # length 1.0, width 0.6 -> base reaches x=+0.5
    box = (0.55, -0.2, 0.75, 0.2)             # ahead of the base front (0.5) but within the fwd-0.4 carton
    bare_s, bare_c, _ = slc._sweep_clear(path, spec, (99, 99, 100, 100), [box], margin=0.0)
    with_s, with_c, hit = slc._sweep_clear(path, spec, (99, 99, 100, 100), [box],
                                           carton_fwd=0.4, carton_half=(0.225, 0.15), margin=0.0)
    assert bare_s > 0.0                        # the bare base clears the box
    assert with_s == 0.0 and hit is not None   # the carton footprint hits it


def test_reach_matches_the_proven_coldwin22_envelope():
    # The pick (carton on the parked cart, ~1.09 m behind the base at staging) and the place (endcap slot)
    # both sit inside the UR10 1.3 m envelope -- the same base-relative geometry coldwin22 picked/placed at.
    staging = (-3.875, 8.0, -math.pi / 2)
    service = (-3.05, 6.0, math.pi)
    pick = (-3.875, 9.09, 1.0125)                       # carton on the cart upper shelf at staging
    place = (-3.05, 4.826, 0.73 + 0.18 / 2.0)           # endcap slot surface + carton half-height
    assert slc._reach(staging[:2], pick) <= slc.DEFAULT_MAX_REACH
    assert slc._reach(service[:2], place) <= slc.DEFAULT_MAX_REACH
    assert 1.1 < slc._reach(staging[:2], pick) < 1.3    # tight-but-feasible, matches run-356 reach analysis


# ---- tilt (real carton_level FK; skip if the mission URDF is absent) ----------------------------

@pytest.mark.skipif(not URDF.exists(), reason='mission URDF not present')
def test_carried_carton_stays_level_under_the_bound():
    tilt = slc._tilt_proof(URDF)
    assert tilt is not None
    worst, rows = tilt
    assert worst <= slc.TILT_BOUND_DEG
    assert all(ok for _, _, _, ok in rows)              # every carried fold segment leveled under the bound


# ---- full store-map path proof (skip if the generated manifest is absent) -----------------------

@pytest.mark.skipif(not ENV.exists(), reason='generated environment manifest not present')
def test_full_service_leg_carry_proof_passes():
    import json
    from twin.navigation.scenario import environment_grid
    from twin.navigation.lanes import TowLaneGraph
    from twin.navigation.articulated import ArticulatedMap
    from twin.mobile import MobileSpec
    manifest = json.loads(ENV.read_text())
    scenario = json.loads(SCENARIO.read_text())
    lanes_path = LANES if LANES.exists() else LANES_FALLBACK
    cfg = json.loads(lanes_path.read_text())
    grid = environment_grid(manifest, scenario, None)
    spec = MobileSpec()
    world = ArticulatedMap(grid, spec)
    poses = cfg['mission_poses']
    staging, service = tuple(poses['shelf_staging']), tuple(poses['shelf_approach'])
    dims = scenario['handling_unit']['dimensions_m']
    half = (dims[0] / 2.0, dims[1] / 2.0)
    graph = TowLaneGraph.from_config(cfg['navigation']['tow_lanes'])
    graph.expand_all(world)
    s = next(ln for ln in graph.lanes if 'staging' in ln.name).route.states[-1]
    cart_box = slc._cart_aabb(spec, s)
    via = (staging[0], 6.5, staging[2])
    carry = slc._staged_path(staging, via, service)
    back = slc._approach_path(service, (service[0], 6.5, service[2]), staging)
    cs, cc, chit = slc._sweep_clear(carry, spec, cart_box, grid.obstacles,
                                    carton_fwd=slc.DEFAULT_CARRY_FWD_M, carton_half=half)
    rs, rc, rhit = slc._sweep_clear(back, spec, cart_box, grid.obstacles)
    assert cs > 0.0 and cc >= 0.0 and chit is None      # carry (with carton) clears static + parked cart
    assert rs > 0.0 and rc >= 0.0 and rhit is None      # empty return clears too
    pick = slc._carton_on_cart_world(spec, s, dims)
    assert slc._reach((s.x, s.y), pick) <= slc.DEFAULT_MAX_REACH
    slot = scenario['initial_staging']['target_surface_pose_world']
    place = (slot[0], slot[1], slot[2] + dims[2] / 2.0)
    assert slc._reach((service[0], service[1]), place) <= slc.DEFAULT_MAX_REACH
