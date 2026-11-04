#!/usr/bin/env python3
"""sim-service-carry-drive-to-shelf-v1 CPU proof: the side-bay TRANSFER carries the carton to the
endcap service pose BEFORE placing, as separate behaviour-tree steps -- the service-drive leg that
transferobs1 skipped (base pinned at the side-bay park, 'host emergency stop (host fault: unreachable
or undersized target)').

Proves, with no Isaac/ROS/GPU (same inputs the repo check_* scripts use):

  (1) BT NODE ORDER -- the cart_side_bay behaviour tree decomposes OUTBOUND + TRANSFER into the
      position-first leaves: ... uncouple@side bay | pick | carry->endcap (NAVIGATE) | perceive slot |
      place@endcap ... The carry is its OWN navigation node (twin.restock.bt.diagram / stage_phases).

  (2) REACH CHECK -- the resolved endcap place is WITHIN the UR10 reach from the shelf_approach service
      pose (resolve_transfer_place_target, the plan-time guard) and FAR OUT OF reach from the side-bay
      park (the observe-gate distance that tripped transferobs1). shelf_approach stays put (reachable);
      the reach bound is never moved. An unreachable endcap trips the NAMED plan-time reason
      'unreachable place target: <dist> m > <reach> m' BEFORE any arm command.

  (3) REPLAY transferobs1's start state through the FIXED controller observe gate -- with the service
      carry on, the far endcap place no longer fails at observe (it is deferred to the post-carry
      re-register); with it off, the same far place still fails fast by name (byte-identical).

  (4) CONFIG PREFLIGHT -- the cross-layer shelf_staging(runtime)/transfer_service_carry(host) split is
      refused at plan time; the paired config (and the committed repo host default) passes.

Usage:
    python3 scripts/check_service_carry_drive.py \
        --scenario scripts/restock_scenario.json \
        [--shelf-approach -3.05 6.0] [--side-bay -4.35 6.99] [--json OUT.json]
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / 'twin'))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from twin.restock.bt import diagram as D  # noqa: E402
from twin.restock.bt import stage_phases as SP  # noqa: E402
from twin.restock.transfer_place_target import (  # noqa: E402
    ARM_BASE_Z, TransferPlaceTargetError, resolve_transfer_place_target)
from twin.restock_manipulation import config_preflight as CP  # noqa: E402


def _stage_gates(model, stage):
    out = []

    def walk(n):
        if n.get('role') == 'stage_steps' and n.get('stage') == stage:
            out.extend((g.get('name', '').split(':', 1)[-1], g.get('kind')) for g in n['children'])
        for c in n.get('children', ()):
            walk(c)
    walk(model)
    return out


def _reach(service_xy, target_xyz, carton_half):
    return math.dist((service_xy[0], service_xy[1], ARM_BASE_Z),
                     (target_xyz[0], target_xyz[1], target_xyz[2] + carton_half))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--scenario', type=Path, default=Path(__file__).resolve().parent / 'restock_scenario.json')
    ap.add_argument('--shelf-approach', type=float, nargs=2, default=[-3.05, 6.0])
    ap.add_argument('--side-bay', type=float, nargs=2, default=[-4.35, 6.99])
    ap.add_argument('--json', type=Path)
    args = ap.parse_args(argv)

    scenario = json.loads(args.scenario.read_text())
    report = {'ok': True, 'checks': []}

    def record(name, ok, detail):
        report['checks'].append({'name': name, 'ok': bool(ok), 'detail': detail})
        report['ok'] = report['ok'] and bool(ok)
        print(f'[{"PASS" if ok else "FAIL"}] {name}: {detail}')

    # (1) BT node order for the side-bay flow.
    model = D.build_model(cart_side_bay=True)
    outbound = _stage_gates(model, 'outbound')
    transfer = _stage_gates(model, 'transfer')
    print('\n-- (1) BT node order (cart_side_bay) --')
    print('OUTBOUND:', ' -> '.join(f'{lbl}[{k}]' for lbl, k in outbound))
    print('TRANSFER:', ' -> '.join(f'{lbl}[{k}]' for lbl, k in transfer))
    carry_idx = next((i for i, (lbl, k) in enumerate(transfer) if 'carry' in lbl and k == SP.NAV), None)
    place_idx = next((i for i, (lbl, _) in enumerate(transfer) if 'place@endcap' in lbl), None)
    uncouple_ok = any(lbl == 'uncouple' and k == SP.UNCOUPLE for lbl, k in outbound)
    order_ok = (uncouple_ok and carry_idx is not None and place_idx is not None
                and carry_idx < place_idx)
    record('bt_node_order', order_ok,
           f'uncouple@side-bay in outbound={uncouple_ok}; transfer has carry->endcap NAVIGATE node '
           f'(idx {carry_idx}) BEFORE place@endcap (idx {place_idx})')

    # (2) reach check: endcap reachable from shelf_approach, not from the side-bay park.
    print('\n-- (2) reach check --')
    endcap = tuple(scenario['initial_staging']['target_surface_pose_world'][:3])
    carton_half = float(scenario['handling_unit']['dimensions_m'][2]) / 2.0
    max_reach = float(scenario['handling_unit'].get('max_reach_m', 1.30))
    r_service = _reach(args.shelf_approach, endcap, carton_half)
    r_sidebay = _reach(args.side_bay, endcap, carton_half)
    record('endcap_reachable_from_shelf_approach', r_service <= max_reach,
           f'{r_service:.3f} m <= {max_reach:.2f} m from shelf_approach {tuple(args.shelf_approach)} '
           f'(shelf_approach stays put; reach bound untouched)')
    record('endcap_unreachable_from_side_bay', r_sidebay > max_reach,
           f'{r_sidebay:.3f} m > {max_reach:.2f} m from the side-bay park {tuple(args.side_bay)} '
           f'(the observe-gate distance that tripped transferobs1)')

    # the plan-time resolver accepts the reachable endcap and names an unreachable one.
    cfg = {'shelf_staging': True, 'mission_poses': {'shelf_approach': list(args.shelf_approach) + [0.0],
                                                    'shelf_staging': [-3.875, 8.0, -math.pi / 2]}}
    resolved = resolve_transfer_place_target(cfg, scenario)
    record('plan_time_resolver_accepts_reachable_endcap',
           resolved.flow == 'shelf_staging' and resolved.reach_m <= max_reach,
           f'flow={resolved.flow} target={resolved.target_xyz} reach={resolved.reach_m:.3f} m')

    far_scenario = json.loads(args.scenario.read_text())
    far_scenario['initial_staging']['target_surface_pose_world'] = [args.shelf_approach[0],
                                                                    args.shelf_approach[1] - 2.0, 0.73, 0.0]
    try:
        resolve_transfer_place_target(cfg, far_scenario)
        record('plan_time_trip_names_unreachable', False, 'resolver did NOT raise on a far endcap')
    except TransferPlaceTargetError as exc:
        msg = str(exc)
        record('plan_time_trip_names_unreachable',
               msg.startswith('unreachable place target:') and ' m > ' in msg,
               f'raised: {msg.splitlines()[0]}')

    # (3) replay transferobs1's start state through the FIXED controller observe gate.
    print('\n-- (3) controller observe gate replay (transferobs1 far endcap) --')
    replay = _replay_observe_gate(max_reach)
    record('service_carry_defers_far_place', replay['on_proceeds'],
           f"service carry ON: observe proceeds past the far endcap (phase={replay['on_phase']}, "
           f"no 'unreachable or undersized target' emergency stop)")
    record('no_service_carry_still_fails_fast', replay['off_fails'],
           f"service carry OFF: observe fails fast '{replay['off_reason']}' (byte-identical transferobs1)")

    # (4) config preflight cross-layer consistency.
    print('\n-- (4) config preflight flow consistency --')
    split = CP.check_flow_consistency({}, {'shelf_staging': True})
    paired = CP.check_flow_consistency({'transfer_service_carry': True}, {'shelf_staging': True})
    host_cfg = json.loads((Path(__file__).resolve().parent / 'twin' / 'restock_manipulation'
                           / 'ros_host_config.json').read_text())
    repo_default = CP.check_flow_consistency(host_cfg, {'shelf_staging': True})
    record('preflight_refuses_split', len(split) == 1 and 'SPLIT' in split[0],
           'runtime shelf_staging ON + host transfer_service_carry OFF is refused at plan time')
    record('preflight_passes_paired', paired == [] and repo_default == []
           and bool(host_cfg.get('transfer_service_carry')),
           'paired config passes; the committed repo host default has transfer_service_carry=true')

    print('\nRESULT:', 'PASS' if report['ok'] else 'FAIL')
    if args.json:
        args.json.write_text(json.dumps(report, indent=2))
        print('wrote', args.json)
    return 0 if report['ok'] else 1


def _replay_observe_gate(max_reach):
    """Step the REAL TransferController observe gate with transferobs1's geometry: a reachable pick on
    the cart beside the base and the endcap place ~2.5 m out (far past reach). ON (service carry) must
    proceed; OFF must fail fast with the named reach reason."""
    import os
    import time
    os.environ['RESTOCK_TRANSFER_MODE'] = 'live_pose'
    os.environ['RESTOCK_TRANSFER_CARRY'] = 'wrist_guard'
    from twin.restock_manipulation.controller import TransferController
    from twin.restock_manipulation.types import Observation, ProductProfile, Sensors, Target

    profile = ProductProfile(sku='juice_orange', dimensions=(0.45, 0.30, 0.18), approach=0.18,
                             tcp_surface_offset=0.0, tcp_quaternion=(0., 0., 0., 1.),
                             max_reach=max_reach, grasp_mode='top')
    job = dict(job_id='svc', sku='juice_orange', quantity=1, source_bin='rollwagen', shelf_id='G3_back_s2')

    def source(t):
        return Observation(f'cart:{t}', 'sensor', t, 'juice_orange', 'rollwagen',
                           (0.0, -0.4, -0.1, 0., 0., 0., 1.), 1.0)   # on the cart beside the base: in reach

    def far_target(t):
        return Target(f'shelf:{t}', 'ground_truth', t, 'G3_back_s2', (2.5, 0.0, 0.1, 0., 0., 0., 1.),
                      True, (0.7, 0.5, 0.5))   # the endcap ~2.5 m out in the base frame (transferobs1)

    class Motion:
        def begin(self, *a, **k): pass
        def begin_joint_goal(self, *a, **k): pass
        def poll(self): return 'succeeded'
        def last_fault_was_sim_deadline(self): return False
        def stop(self): pass
        def update_payload(self, *a, **k): return True

    def run(service_carry):
        c = TransferController(Motion(), {'juice_orange': profile}, transfer_ground_truth=True,
                               transfer_service_carry=service_carry, phase_timeout=120.)
        c.destination = 'rollwagen'
        c.start('transfer', job)
        r, phases = None, []
        for i in range(30):
            t = 1.0 + 0.1 * i
            if c.phase in ('succeeded', 'failed', 'transit'):
                break
            phases.append(c.phase)
            contact = c.held or (c.phase == 'grasp' and c.issued)
            r = c.step(0.1, Sensors(t, (source(t),), far_target(t), True, contact, t, t,
                                    (0.,) * 6, (), t, time.monotonic() + 100.,
                                    (0.0, -0.4, -0.15, 0., 0., 0., 1.)))
        return c, r, phases

    con, ron, _ = run(True)
    coff, roff, _ = run(False)
    return {
        'on_proceeds': con.phase not in ('failed',) and 'unreachable' not in (getattr(ron, 'reason', '') or ''),
        'on_phase': con.phase,
        'off_fails': coff.phase == 'failed' and roff.reason == 'unreachable or undersized target',
        'off_reason': getattr(roff, 'reason', ''),
    }


if __name__ == '__main__':
    raise SystemExit(main())
