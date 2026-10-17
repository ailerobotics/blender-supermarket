#!/usr/bin/env python3
"""CPU evidence for the simulated-people human-safety layer (no simulator/GPU).

Drives the scripted crowd + ground-truth detection + reaction mechanism on a
virtual clock and writes a trace (JSONL), a summary (JSON) and a top-down SVG that
shows the base slowing, stopping, the arm pausing and everything resuming as a
person crosses the robot and enters the arm envelope. The poses come from a mission
manifest+scenario when given (``twin.restock.scenario.poses_for_job``), else from a
built-in synthetic pose set so the demo runs anywhere.

Example::

    python3 scripts/run_humans_safety_demo.py \
        --out ~/datasets/sim-restock-humans-v1/demo

This is LABELLED ground-truth evidence, not sensor pixels; see the report.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from twin.humans import evidence
from twin.humans.config import HumansConfig
from twin.humans.perception import SafetyParams

# A generic store-scale pose set (map frame, metres) used when no manifest is given.
SYNTHETIC_POSES = {
    'pickup': (2.0, 1.0, 0.0),
    'cart_dock': (3.5, 1.0, 3.14159),
    'shelf_approach': (9.0, 5.0, 1.5708),
    'home': (2.0, 5.0, 0.0),
}

DEFAULT_HUMANS = {
    'enabled': True,
    'detection': {'source': 'ground_truth',
                  'cameras': ['restock_camera_wrist'],
                  'gt_visible_range_m': 6.0, 'gt_fov_halfangle_rad': 1.0},
    'reaction': {'slow_base_scale': 0.35, 'stop_base_scale': 0.0,
                 'resume_clear_dwell_s': 2.0, 'reach_radius_m': 1.3,
                 'arm_hold_margin_m': 0.3},
    # people omitted -> the default crowd is derived from the poses
}


def _poses_from_manifest(manifest_path, scenario_path):
    from twin.restock.scenario import poses_for_job
    from twin.restock.inventory import Job
    manifest = json.loads(Path(manifest_path).read_text())
    scenario = json.loads(Path(scenario_path).read_text())
    return poses_for_job(manifest, Job(**scenario['job']))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out', type=Path, required=True, help='output directory (created)')
    ap.add_argument('--manifest', type=Path, help='mission environment manifest (optional)')
    ap.add_argument('--scenario', type=Path, help='mission scenario (optional, with --manifest)')
    ap.add_argument('--humans-config', type=Path,
                    help='JSON with a "humans" block; default = built-in demo crowd')
    ap.add_argument('--duration', type=float, default=22.0)
    ap.add_argument('--dt', type=float, default=0.1)
    ap.add_argument('--drive-s', type=float, default=8.0)
    args = ap.parse_args()

    if args.manifest and args.scenario:
        poses = _poses_from_manifest(args.manifest, args.scenario)
    else:
        poses = SYNTHETIC_POSES

    if args.humans_config:
        humans = HumansConfig.from_config(json.loads(args.humans_config.read_text()))
    else:
        humans = HumansConfig.from_config({'humans': DEFAULT_HUMANS})
    if not humans.enabled:
        raise SystemExit('humans config is disabled; nothing to demonstrate')

    params = SafetyParams()
    rows = evidence.run(poses, humans, duration_s=args.duration, dt=args.dt,
                        drive_s=args.drive_s, params=params)
    summary = evidence.summarize(rows, params)

    # snapshots: first tick, closest approach, an arm-hold tick, last tick
    closest = min(rows, key=lambda r: (r['nearest_distance_base_m']
                                       if r['nearest_distance_base_m'] is not None else 1e9))
    held = next((r for r in rows if r['arm_hold']), rows[0])
    snaps = sorted({rows[0]['t'], held['t'], closest['t'], rows[-1]['t']})
    svg = evidence.render_svg(poses, rows, params, snaps)

    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    (out / 'trace.jsonl').write_text('\n'.join(json.dumps(r) for r in rows) + '\n')
    (out / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    (out / 'demo.svg').write_text(svg)
    print(json.dumps(summary, indent=2))
    print(f'wrote {out}/trace.jsonl, summary.json, demo.svg')
    return 0 if summary['verdict'].startswith('OK') else 1


if __name__ == '__main__':
    raise SystemExit(main())
