"""CPU guard for sim-service-place-target-v1: the TRANSFER place target must be ONE explicit
function of the shelf-staging flow switch (twin.restock.transfer_place_target).

coldwinint1 drove the re-sequenced sales-floor run to the fm461 BACKROOM bay because
transfer_ground_truth.target was read flat, ignoring shelf_staging. These tests pin:
  * legacy flow (shelf_staging off)  -> the configured backroom bay target, unchanged;
  * shelf_staging flow               -> the sales-floor endcap (scenario target_surface_pose_world);
  * shelf_staging + a mis-filed (unreachable) configured target -> loud, NAMED failure;
  * the reach gate + the missing-input guards.
No Isaac/ROS/GPU; pure.
"""
import math
import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(SCRIPTS))

from twin.restock.transfer_place_target import (
    ARM_BASE_Z, DEFAULT_MAX_REACH_M, DEFAULT_VERIFY_CENTER_TOL_M, TransferPlaceTarget,
    TransferPlaceTargetError, VerifyRegion, VerifyRegionError, resolve_transfer_place_target,
    resolve_verify_region, shelf_staging_flow_active, verify_bay_for_place_target)

# Real geometry (scripts/restock_scenario.json + the proven flow, docs/restock_navigation.md).
ENDCAP = [-3.05, 4.826, 0.73, 0.0]          # scenario.initial_staging.target_surface_pose_world
SHELF_APPROACH = [-3.05, 6.0, math.pi]      # the endcap service pose
BACKROOM_BAY = [-2.0, 16.424, 0.73]         # the fm461 demo-fallback backroom target (coldwinint1)
CART_SIDE_BAY = [-4.3, 6.95, -math.pi / 2]


def _scenario(endcap=ENDCAP, carton=(0.45, 0.30, 0.18), max_reach=1.3, shelf_id='G3_back_s2'):
    return {
        'initial_staging': {'target_surface_pose_world': list(endcap)},
        'handling_unit': {'dimensions_m': list(carton), 'max_reach_m': max_reach},
        'job': {'shelf_id': shelf_id},
    }


def _config(*, shelf_staging, gt_target=None, poses=None, cart_side_bay=False,
            transfer_shelf_location='G3_back_s2'):
    cfg = {'mission_poses': poses if poses is not None else {
        'shelf_approach': list(SHELF_APPROACH), 'shelf_staging': [-3.875, 8.0, -math.pi / 2],
        'cart_side_bay': list(CART_SIDE_BAY)}}
    if shelf_staging:
        cfg['shelf_staging'] = True
    if cart_side_bay:
        cfg['cart_side_bay'] = True
    if transfer_shelf_location is not None:
        cfg['transfer_shelf_location'] = transfer_shelf_location
    if gt_target is not None:
        cfg['transfer_ground_truth'] = {'enabled': True, 'location': 'G3_back_s2',
                                        'target': list(gt_target)}
    return cfg


# ---- flow switch ------------------------------------------------------------------------------

def test_flow_switch_reads_shelf_staging():
    assert shelf_staging_flow_active({'shelf_staging': True}) is True
    assert shelf_staging_flow_active({'shelf_staging': False}) is False
    assert shelf_staging_flow_active({}) is False
    assert shelf_staging_flow_active(None) is False


# ---- legacy flow: byte-identical, the configured backroom target unchanged --------------------

def test_legacy_flow_returns_configured_backroom_target_unchanged():
    cfg = _config(shelf_staging=False, gt_target=BACKROOM_BAY)
    r = resolve_transfer_place_target(cfg, _scenario())
    assert r.flow == 'legacy'
    assert r.source == 'configured_bay'
    assert r.overridden is False
    assert r.target_xyz == pytest.approx(tuple(BACKROOM_BAY))
    assert r.service_pose is None   # legacy does not re-derive a service pose


def test_legacy_flow_with_no_configured_target_is_none_not_an_error():
    r = resolve_transfer_place_target(_config(shelf_staging=False), _scenario())
    assert r.flow == 'legacy' and r.target_xyz is None


# ---- shelf_staging flow: the sales-floor endcap, the proven shelf place target ----------------

def test_shelf_staging_selects_the_endcap_over_a_backroom_config_is_rejected_loudly():
    # the EXACT coldwinint1 config: shelf_staging ON, transfer target still the backroom bay.
    cfg = _config(shelf_staging=True, cart_side_bay=True, gt_target=BACKROOM_BAY)
    with pytest.raises(TransferPlaceTargetError) as exc:
        resolve_transfer_place_target(cfg, _scenario())
    msg = str(exc.value)
    assert 'shelf_staging' in msg and 'drive-and-reach' in msg
    assert 'endcap' in msg          # names the place the service leg actually services


def test_shelf_staging_with_the_endcap_target_resolves_to_the_endcap():
    cfg = _config(shelf_staging=True, cart_side_bay=True, gt_target=ENDCAP[:3])
    r = resolve_transfer_place_target(cfg, _scenario())
    assert r.flow == 'shelf_staging'
    assert r.source == 'shelf_staging_endcap'
    assert r.target_xyz == pytest.approx(tuple(ENDCAP[:3]))
    assert r.service_pose == pytest.approx(tuple(SHELF_APPROACH))
    assert r.overridden is False                 # configured target already the endcap
    assert r.reach_m is not None and r.reach_m <= DEFAULT_MAX_REACH_M
    assert r.location == 'G3_back_s2'


def test_shelf_staging_with_a_DISABLED_gt_block_does_not_trip_on_its_stale_target():
    # a disabled transfer_ground_truth block is the camera-sensed path; a stale backroom target
    # under it must NOT be treated as the active place target (no false refusal).
    cfg = _config(shelf_staging=True)
    cfg['transfer_ground_truth'] = {'enabled': False, 'target': list(BACKROOM_BAY)}
    r = resolve_transfer_place_target(cfg, _scenario())   # must not raise
    assert r.flow == 'shelf_staging'
    assert r.target_xyz == pytest.approx(tuple(ENDCAP[:3]))


def test_shelf_staging_with_no_configured_target_selects_the_endcap():
    cfg = _config(shelf_staging=True)
    r = resolve_transfer_place_target(cfg, _scenario())
    assert r.flow == 'shelf_staging'
    assert r.target_xyz == pytest.approx(tuple(ENDCAP[:3]))
    assert r.overridden is False                 # nothing to override


def test_endcap_reach_matches_the_arm_base_to_carton_rest_distance():
    r = resolve_transfer_place_target(_config(shelf_staging=True), _scenario())
    sx, sy, _ = SHELF_APPROACH
    ex, ey, ez = ENDCAP[:3]
    carton_half = 0.18 / 2.0
    expected = math.dist((sx, sy, ARM_BASE_Z), (ex, ey, ez + carton_half))
    assert r.reach_m == pytest.approx(expected)


# ---- the reach gate: fail loud when the endcap itself is out of reach -------------------------

def test_unreachable_endcap_fails_loudly():
    far = [-3.05, 1.0, 0.73, 0.0]    # > 1.3 m south of the service pose
    with pytest.raises(TransferPlaceTargetError) as exc:
        resolve_transfer_place_target(_config(shelf_staging=True), _scenario(endcap=far))
    # sim-service-carry-drive-to-shelf-v1: the plan-time trip is named 'unreachable place target:
    # <dist> m > <reach> m' (fired at boot, before any arm command).
    msg = str(exc.value)
    assert msg.startswith('unreachable place target:') and ' m > ' in msg and 'reach' in msg


# ---- missing-input guards (named, not a bare KeyError) ----------------------------------------

def test_missing_shelf_approach_pose_is_named():
    cfg = _config(shelf_staging=True)
    cfg['mission_poses'] = {'shelf_staging': [-3.875, 8.0, -math.pi / 2]}   # no shelf_approach
    with pytest.raises(TransferPlaceTargetError) as exc:
        resolve_transfer_place_target(cfg, _scenario())
    assert 'shelf_approach' in str(exc.value)


def test_missing_scenario_endcap_is_named():
    with pytest.raises(TransferPlaceTargetError) as exc:
        resolve_transfer_place_target(_config(shelf_staging=True), {'handling_unit': {}})
    assert 'target_surface_pose_world' in str(exc.value)


# ---- verify bay stays consistent with the place target ----------------------------------------

def test_verify_bay_is_centred_on_the_place_target():
    bay = verify_bay_for_place_target(tuple(ENDCAP[:3]), _scenario())
    assert bay['center_xy'] == pytest.approx(ENDCAP[:2])
    assert bay['top_z'] == pytest.approx(ENDCAP[2])
    assert len(bay['size_xy']) == 2 and all(v > 0 for v in bay['size_xy'])


def test_verify_bay_is_not_the_backroom_after_an_endcap_place():
    # the whole point: verify must NOT keep checking the backroom [-2.0, 16.42] after an endcap place.
    scenario = {'initial_staging': {'support_dimensions_m': [0.68, 0.48, 0.6]},
                'transfer_bay': {'center_xy': [-2.0, 16.42], 'dimensions_m': [0.7, 0.5, 0.73]}}
    bay = verify_bay_for_place_target(tuple(ENDCAP[:3]), scenario)
    assert bay['center_xy'] == pytest.approx(ENDCAP[:2])   # endcap, not the backroom
    assert math.dist(bay['center_xy'], [-2.0, 16.42]) > 9.0


# ---- sim-verify-region-endcap-v1: the VERIFY region is gated through the same resolver family ---

def _place(cfg, scenario=None):
    return resolve_transfer_place_target(cfg, scenario if scenario is not None else _scenario())


def test_verify_region_legacy_flow_is_left_unchanged():
    # legacy: no injection, no gate -> byte-identical verify region.
    cfg = _config(shelf_staging=False, gt_target=BACKROOM_BAY)
    cfg['verify_ground_truth'] = {'enabled': True}
    region = resolve_verify_region(cfg, _scenario(), _place(cfg))
    assert isinstance(region, VerifyRegion)
    assert region.flow == 'legacy'
    assert region.bay is None and region.source == 'unchanged'


def test_verify_region_disabled_block_is_left_unchanged():
    cfg = _config(shelf_staging=True)
    cfg['verify_ground_truth'] = {'enabled': False, 'bay': {'center_xy': list(BACKROOM_BAY[:2])}}
    region = resolve_verify_region(cfg, _scenario(), _place(cfg))
    assert region.bay is None and region.source == 'unchanged'   # off => never touched / never gated


def test_verify_region_shelf_staging_injects_the_endcap_bay_when_bay_less():
    cfg = _config(shelf_staging=True)
    cfg['verify_ground_truth'] = {'enabled': True}               # no bay override
    region = resolve_verify_region(cfg, _scenario(), _place(cfg))
    assert region.flow == 'shelf_staging' and region.source == 'shelf_staging_endcap'
    assert region.overridden is True
    assert region.bay is not None
    assert region.bay['center_xy'] == pytest.approx(ENDCAP[:2])
    assert region.bay['top_z'] == pytest.approx(ENDCAP[2])
    assert math.dist(region.bay['center_xy'], BACKROOM_BAY[:2]) > 9.0


def test_verify_region_shelf_staging_rejects_a_backroom_bay_by_name():
    # the EXACT verify-side coldwinint trap: shelf_staging on, verify bay still the backroom.
    cfg = _config(shelf_staging=True)
    cfg['verify_ground_truth'] = {'enabled': True,
                                  'bay': {'center_xy': list(BACKROOM_BAY[:2]),
                                          'size_xy': [0.7, 0.5], 'top_z': 0.73}}
    with pytest.raises(VerifyRegionError) as exc:
        resolve_verify_region(cfg, _scenario(), _place(cfg))
    msg = str(exc.value)
    assert 'shelf_staging' in msg and 'endcap' in msg
    assert 'verify_ground_truth.bay' in msg


def test_verify_region_error_is_a_transfer_place_target_error():
    # so the boot gate can catch the whole resolver family with one except.
    assert issubclass(VerifyRegionError, TransferPlaceTargetError)


def test_verify_region_keeps_an_endcap_consistent_configured_bay():
    # a bay already centred on the endcap is kept as authored (lane sizing preserved), never overwritten.
    cfg = _config(shelf_staging=True)
    cfg['verify_ground_truth'] = {'enabled': True,
                                  'bay': {'center_xy': ENDCAP[:2], 'size_xy': [0.68, 0.48], 'top_z': 0.73}}
    region = resolve_verify_region(cfg, _scenario(), _place(cfg))
    assert region.source == 'configured' and region.bay is None   # left as-is
    assert region.overridden is False
    assert region.center_offset_m == pytest.approx(0.0, abs=1e-9)


def test_verify_region_tolerance_boundary():
    # a bay just inside the tolerance is accepted; just outside is refused.
    cfg = _config(shelf_staging=True)
    inside = [ENDCAP[0] + DEFAULT_VERIFY_CENTER_TOL_M - 0.01, ENDCAP[1]]
    cfg['verify_ground_truth'] = {'enabled': True, 'bay': {'center_xy': inside, 'top_z': 0.73}}
    assert resolve_verify_region(cfg, _scenario(), _place(cfg)).source == 'configured'
    outside = [ENDCAP[0] + DEFAULT_VERIFY_CENTER_TOL_M + 0.01, ENDCAP[1]]
    cfg['verify_ground_truth'] = {'enabled': True, 'bay': {'center_xy': outside, 'top_z': 0.73}}
    with pytest.raises(VerifyRegionError):
        resolve_verify_region(cfg, _scenario(), _place(cfg))


# ---- against the real committed scenario + the merged fragment --------------------------------

def test_real_scenario_and_merged_fragment_resolve_to_the_endcap():
    import json
    repo = SCRIPTS.parent
    scenario = json.loads((SCRIPTS / 'restock_scenario.json').read_text())
    frag = json.loads((repo / 'docs/examples/service-leg-carry-mission-fragment.json').read_text())
    cfg = {k: v for k, v in frag.items() if not k.startswith('_')}
    # the fragment carries the transfer keys at top level (RUNTIME); supply the GT place target the
    # integration lane merges in (now the endcap) so the resolver has a configured target to check.
    endcap = scenario['initial_staging']['target_surface_pose_world'][:3]
    cfg['transfer_ground_truth'] = {'enabled': True, 'location': 'G3_back_s2', 'target': endcap}
    r = resolve_transfer_place_target(cfg, scenario)
    assert r.flow == 'shelf_staging'
    assert r.target_xyz == pytest.approx(tuple(endcap))
    assert r.reach_m <= DEFAULT_MAX_REACH_M
    # the merged fragment's verify bay is endcap-consistent -> accepted (never refused, never re-injected).
    region = resolve_verify_region(cfg, scenario, r)
    assert region.flow == 'shelf_staging'
    assert region.source == 'configured' and region.bay is None
    assert region.center_offset_m <= DEFAULT_VERIFY_CENTER_TOL_M
