#!/usr/bin/env python3
"""Arm collision bench -- bounded GPU mode (sim-arm-collision-bench-v1).

Boots Isaac ONCE (reusing the proven ``run_arm_carry_bench`` stage, guards and trial loop -- the scout
steer: extend the boot-once bench, do not write a second one), authors the scenarios' planted
obstacles into the assembled stage as world colliders, drives each scenario's arm leg, and reads the
LIVE per-trial clearance (carton/arm AABB vs the obstacle) + PhysX contact into the SAME per-scenario
schema the CPU mock-ESDF mode emits -- so the GPU numbers drop straight into the same report/CSV.

This is the bounded proof (<= 20 min) the brief's deliverable 4 asks for.  It runs under the canonical
GPU discipline: the integration lane has priority; take the lock only when free, never ``--force``,
``HEAVY_MB=2500`` only if whispr holds VRAM, renice +10.  Recommended invocation (outer flock holds the
lock so the run is serialised behind the integration lane):

    renice +10 -p $$ >/dev/null
    flock -w 5 /tmp/aile-gpu.lock env_isaaclab/bin/python scripts/run_arm_collision_bench_gpu.py \
        --runtime-config <mission-runtime.json> \
        --scenarios docs/examples/arm-collision-scenarios \
        --legs collect_pick \
        --heavy-threshold-mb 2500 \
        --no-gpu-lock \
        --out /home/manishsaini/firstmate/data/sim-arm-collision-bench-v1/gpu-collect

``--no-gpu-lock`` is correct ONLY under an outer ``flock`` (the ``gpu_guards_no_flock`` path still runs
the VRAM/Blender/RAM guards).  Without an outer flock, omit it and the runner takes ``gpu_slot`` itself.

NOTE: the Isaac glue here is GPU-only and, like ``IsaacCarryStage`` itself, is exercised only on the
GPU (the CPU tests cover the pure obstacle authoring + schema ingest in ``arm_collision_scene``).  The
VERY FIRST live boot must be validated against a known run; until then this runner is the filed
follow-up for the AFTER/BEFORE GPU numbers.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from twin.tools import arm_collision_bench as bench          # noqa: E402
from twin.tools import arm_collision_scene as scene          # noqa: E402


def _select(scenarios, legs, ids):
    out = scenarios
    if legs:
        out = [s for s in out if s.arm_leg in set(legs)]
    if ids:
        out = [s for s in out if s.id in set(ids)]
    return out


def run(args):
    import run_arm_carry_bench as carry  # the proven boot-once bench we extend
    from twin.restock.guard import gpu_slot

    scenarios = _select(bench.load_scenarios(args.scenarios), args.legs, args.ids)
    if not scenarios:
        raise SystemExit('no scenarios selected (legs=%s ids=%s)' % (args.legs, args.ids))
    thresholds = bench.Thresholds.from_dict(
        json.loads(args.thresholds.read_text()) if args.thresholds else None)

    config = json.loads(args.runtime_config.read_text())
    mission_scenario = json.loads(Path(config['scenario']).read_text())
    md = dict(config.get('masked_depth') or {}); md.pop('runtime_env', None)
    config['masked_depth'] = md
    arm_cfg = config.get('armbench', {})
    lift = arm_cfg.get('lift_joints', carry.DEFAULT_LIFT)
    grasp = arm_cfg.get('grasp_joints', carry.DEFAULT_GRASP)
    carton_dims = tuple(mission_scenario.get('handling_unit', {}).get('dimensions', bench.CARTON_DIMS_M))
    args.out.mkdir(parents=True, exist_ok=True)

    slot = (carry.gpu_guards_no_flock(heavy_threshold_mb=args.heavy_threshold_mb) if args.no_gpu_lock
            else gpu_slot(heavy_threshold_mb=args.heavy_threshold_mb))
    results = []
    with slot:
        from isaacsim import SimulationApp
        app = SimulationApp({'headless': True, 'width': 640, 'height': 360})
        from twin.restock.physics_setup import assemble
        assembly = assemble(app=app, config=config, scenario=mission_scenario, output=args.out / 'assembly')
        base_prim = scene.find_base_prim(assembly.stage) or args.base_prim
        # Author every selected scenario's obstacles up front (per-leg disjoint workspace regions).
        authored = {}
        for s in scenarios:
            authored[s.id] = scene.author_obstacles(assembly.stage, s, base_prim_path=base_prim,
                                                    root='/ArmCollisionBench/%s' % scene._safe(s.id))
        stage = carry.IsaacCarryStage(assembly, lift, grasp, render=args.render, carton_dims=carton_dims,
                                      obstacle_prims={n: p for d in authored.values() for n, p in d.items()},
                                      arm_contacts=True)
        import twin.tools.arm_carry_bench as cbench
        # The mission's soft carry drive gains (run_arm_carry_bench REF_SOFT: stiffness, damping, max_effort).
        drive_gains = cbench.DriveGains(1.0e5, 2.0e3, 400.0)
        for s in scenarios:
            # Restrict the clearance read to this scenario's obstacles.
            stage.obstacles = dict(authored[s.id])
            spec = cbench.TrialSpec(s.id, drive_gains, cbench.CarryProfile(
                peak_speed=args.carry_speed, peak_accel=args.carry_accel))
            goals = [(f'wp{i}', list(w)) for i, w in enumerate(bench.build_trajectory(s))]
            moves = carry._run_trial(stage, spec, goals, args.dt, args.max_sim_s)
            min_gap = min((min(m.min_clearance_m.values()) for m in moves if m.min_clearance_m),
                          default=float('inf'))
            contact = any(m.contact_prims for m in moves)
            measured = {
                'min_true_clearance_m': float(min_gap),
                'contact': bool(contact),
                'gate_behavior': 'none',  # BEFORE: current stack has no perceived gate on this leg
                'worst_obstacle': next((p.rsplit('/', 1)[-1] for m in moves for p in m.contact_prims), None),
            }
            r = scene.ingest_trial(s, measured, thresholds)
            results.append(r)
            print('ARMCOLLISION_BENCH scenario=%s min_gap_mm=%.1f contact=%s' % (
                s.id, min_gap * 1e3, contact), flush=True)
        app.close()

    summary = bench.summarise(results, 'gpu/baseline', thresholds)
    (args.out / 'report.md').write_text(bench.report_markdown(results, summary, title='Arm collision bench (GPU)'))
    (args.out / 'results.csv').write_text(bench.results_csv(results))
    (args.out / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print('wrote %s/{report.md,results.csv,summary.json}' % args.out)
    return 0 if summary['all_passed'] else 1


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--runtime-config', type=Path, required=True)
    ap.add_argument('--scenarios', type=Path,
                    default=Path(__file__).resolve().parent.parent / 'docs/examples/arm-collision-scenarios')
    ap.add_argument('--legs', nargs='*', default=['collect_pick'], help='arm legs to bound the run to')
    ap.add_argument('--ids', nargs='*', default=None, help='restrict to these scenario ids')
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--thresholds', type=Path)
    ap.add_argument('--base-prim', default='/World/ur_base_link')
    ap.add_argument('--render', action='store_true')
    ap.add_argument('--dt', type=float, default=1.0 / 60.0)
    ap.add_argument('--max-sim-s', type=float, default=12.0)
    ap.add_argument('--carry-speed', type=float, default=0.1)
    ap.add_argument('--carry-accel', type=float, default=0.3)
    ap.add_argument('--heavy-threshold-mb', type=int, choices=(1500, 2500), default=1500)
    ap.add_argument('--no-gpu-lock', action='store_true', help='use ONLY under an outer flock on /tmp/aile-gpu.lock')
    return run(ap.parse_args(argv))


if __name__ == '__main__':
    raise SystemExit(main())
