#!/usr/bin/env python3
"""Offline proof (CPU-only) for sim-service-place-target-v1: the TRANSFER place target is ONE
explicit function of the shelf-staging flow switch, and the merged config is correct + loadable.

coldwinint1 (start_at=couple, side bay, carton pre-staged) proved couple + the straight tow into
the sales-floor side bay + park, then FAILED at transfer: the place target resolved to the fm461
BACKROOM bay [-2.0, 16.424, 0.73], 9.7 m from where the free base parked, because
``transfer_ground_truth.target`` was read FLAT with no dependence on shelf_staging. This proves, on CPU:

  (1) FLOW SELECTION -- resolve_transfer_place_target under BOTH flows against the REAL committed
      scenario: legacy (shelf_staging off) returns the configured backroom bay unchanged; shelf_staging
      returns the sales-floor ENDCAP (scenario.initial_staging.target_surface_pose_world).
  (2) LOUD FAILURE -- the EXACT coldwinint1 merged config (shelf_staging on + the backroom target) is
      REFUSED by name (TransferPlaceTargetError), naming drive-and-reach + the endcap.
  (3) REACH -- the resolved endcap place sits inside the UR10 reach from the shelf_approach service pose.
  (4) VERIFY CONSISTENCY -- the injected verify bay is centred on the endcap, not the backroom.
  (5) CONFIG-PREFLIGHT -- every batch key the merged fragment (cart-side-bay nav + service-leg manip)
      declares loads from the config its consumer actually reads (restock_manipulation.config_preflight).

Run:  PYTHONPATH=scripts python3 scripts/check_service_place_target.py \
        --scenario scripts/restock_scenario.json \
        --service-fragment docs/examples/service-leg-carry-mission-fragment.json \
        --nav-fragment docs/examples/cart-side-bay-mission-fragment.json
Exit 0 iff all five pass (prints SERVICE_PLACE_TARGET PASS/FAIL)."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))   # scripts/ -> twin.*
from twin.restock.transfer_place_target import (
    DEFAULT_MAX_REACH_M, DEFAULT_VERIFY_CENTER_TOL_M, TransferPlaceTargetError, VerifyRegionError,
    resolve_transfer_place_target, resolve_verify_region, verify_bay_for_place_target)
from twin.restock_manipulation import config_preflight as cp

BACKROOM_BAY = [-2.0, 16.424, 0.73]   # the fm461 demo-fallback target coldwinint1 wrongly used


def _strip(d):
    return {k: v for k, v in d.items() if not k.startswith('_')}


def _deep_merge(a, b):
    out = dict(a)
    for k, v in b.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def _record(report, name, passed, **kw):
    report['ok'] = report['ok'] and passed
    report['checks'].append(dict(name=name, passed=passed, **kw))
    print(('PASS ' if passed else 'FAIL ') + name + ' ' + json.dumps(kw), flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--scenario', type=Path, required=True)
    ap.add_argument('--service-fragment', type=Path, required=True)
    ap.add_argument('--nav-fragment', type=Path, required=True)
    ap.add_argument('--json', type=Path, default=None)
    args = ap.parse_args()

    scenario = json.loads(args.scenario.read_text())
    service = json.loads(args.service_fragment.read_text())
    nav = json.loads(args.nav_fragment.read_text())
    report = {'ok': True, 'checks': []}

    # The merged RUNTIME config = nav fragment (outbound/park) + service fragment (manip), top-level
    # keys deep-merged; the HOST config = the service fragment's host_config block.
    runtime_cfg = _deep_merge(_strip(nav), _strip(service))
    runtime_cfg.pop('host_config', None)
    runtime_cfg.pop('env', None)
    host_cfg = _strip(service.get('host_config', {}))
    endcap = tuple(scenario['initial_staging']['target_surface_pose_world'][:3])

    # (1) FLOW SELECTION -- legacy vs shelf_staging.
    legacy_cfg = dict(runtime_cfg)
    legacy_cfg.pop('shelf_staging', None)
    legacy_cfg.pop('cart_side_bay', None)
    legacy_cfg['transfer_ground_truth'] = {'enabled': True, 'location': 'G3_back_s2',
                                           'target': BACKROOM_BAY}
    legacy = resolve_transfer_place_target(legacy_cfg, scenario)
    _record(report, 'legacy_flow_keeps_backroom_target', legacy.flow == 'legacy'
            and legacy.target_xyz == tuple(BACKROOM_BAY),
            flow=legacy.flow, target=legacy.target_xyz)

    staged = resolve_transfer_place_target(runtime_cfg, scenario)
    _record(report, 'shelf_staging_flow_selects_endcap', staged.flow == 'shelf_staging'
            and tuple(round(v, 3) for v in staged.target_xyz) == tuple(round(v, 3) for v in endcap),
            flow=staged.flow, source=staged.source, target=[round(v, 3) for v in staged.target_xyz],
            service_pose=[round(v, 3) for v in staged.service_pose])

    # (2) LOUD FAILURE -- the coldwinint1 merged config (backroom target + shelf_staging).
    bad_cfg = dict(runtime_cfg)
    bad_cfg['transfer_ground_truth'] = {'enabled': True, 'location': 'G3_back_s2',
                                        'target': BACKROOM_BAY}
    raised = None
    try:
        resolve_transfer_place_target(bad_cfg, scenario)
    except TransferPlaceTargetError as exc:
        raised = str(exc)
    _record(report, 'coldwinint1_backroom_config_refused_by_name',
            raised is not None and 'drive-and-reach' in raised and 'endcap' in raised,
            raised=raised)

    # (3) REACH -- the resolved endcap place within the UR10 envelope from shelf_approach.
    _record(report, 'endcap_place_within_reach',
            staged.reach_m is not None and staged.reach_m <= DEFAULT_MAX_REACH_M,
            reach_m=round(staged.reach_m, 3) if staged.reach_m else None,
            max_reach_m=DEFAULT_MAX_REACH_M,
            margin_m=round(DEFAULT_MAX_REACH_M - staged.reach_m, 3) if staged.reach_m else None)

    # (4) VERIFY CONSISTENCY -- the injected verify bay is centred on the endcap, not the backroom.
    bay = verify_bay_for_place_target(staged.target_xyz, scenario)
    import math
    _record(report, 'verify_bay_centred_on_endcap',
            bay['center_xy'] == [round(endcap[0], 6), round(endcap[1], 6)] or
            math.dist(bay['center_xy'], endcap[:2]) < 1e-6,
            bay=bay, dist_to_backroom_m=round(math.dist(bay['center_xy'], BACKROOM_BAY[:2]), 2))

    # (4a) VERIFY REGION GATE -- the merged fragment's verify_ground_truth bay is accepted (endcap-
    # consistent, never re-injected); a backroom verify bay under shelf_staging REFUSES by name.
    region = resolve_verify_region(runtime_cfg, scenario, staged)
    _record(report, 'verify_region_merged_fragment_consistent',
            region.flow == 'shelf_staging' and region.center_offset_m is not None
            and region.center_offset_m <= DEFAULT_VERIFY_CENTER_TOL_M,
            source=region.source, center_offset_m=round(region.center_offset_m, 4)
            if region.center_offset_m is not None else None,
            overridden=region.overridden)

    bad_verify = dict(runtime_cfg)
    bad_verify['verify_ground_truth'] = {'enabled': True,
                                         'bay': {'center_xy': BACKROOM_BAY[:2],
                                                 'size_xy': [0.7, 0.5], 'top_z': 0.73}}
    vraised = None
    try:
        resolve_verify_region(bad_verify, scenario, staged)
    except VerifyRegionError as exc:
        vraised = str(exc)
    _record(report, 'backroom_verify_region_refused_by_name',
            vraised is not None and 'shelf_staging' in vraised and 'endcap' in vraised
            and 'verify_ground_truth.bay' in vraised,
            raised=vraised)

    # (4b) legacy byte-identical -- the verify region is untouched when shelf_staging is off.
    legacy_verify_cfg = dict(legacy_cfg)
    legacy_verify_cfg['verify_ground_truth'] = {'enabled': True,
                                                'bay': {'center_xy': BACKROOM_BAY[:2], 'top_z': 0.73}}
    legacy_region = resolve_verify_region(legacy_verify_cfg, scenario,
                                          resolve_transfer_place_target(legacy_verify_cfg, scenario))
    _record(report, 'legacy_verify_region_unchanged',
            legacy_region.flow == 'legacy' and legacy_region.bay is None
            and legacy_region.source == 'unchanged',
            source=legacy_region.source)

    # (5) CONFIG-PREFLIGHT -- the merged fragment's batch keys load from the right config file.
    required = ['shelf_staging', 'cart_side_bay', 'carry_drive_profile', 'transfer_service_carry',
                'carton_level', 'transfer_ground_truth', 'transfer_shelf_location', 'mission_poses',
                'verify_ground_truth']
    # inject the endcap transfer target into the merged runtime so the preflight sees a correct,
    # loadable config (the integration lane merges these exact lines from the service fragment).
    runtime_for_preflight = dict(runtime_cfg)
    problems = cp.check_activation(host_cfg, runtime_for_preflight, required)
    for key in required:
        consumer = cp.KEY_CONSUMER.get(key, '?')
        present = key in (host_cfg if consumer == cp.HOST else runtime_for_preflight)
        print('  CONFIG_PREFLIGHT %-24s -> %-7s %s' % (key, consumer,
              'PRESENT' if present else 'ABSENT'), flush=True)
    _record(report, 'config_preflight_merged_fragment_loads', not problems, problems=problems)

    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report, indent=2))
        print(f'wrote {args.json}', flush=True)
    print('SERVICE_PLACE_TARGET ' + ('PASS' if report['ok'] else 'FAIL'), flush=True)
    return 0 if report['ok'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
