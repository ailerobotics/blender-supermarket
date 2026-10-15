#!/usr/bin/env python3
"""Commission empty-support baselines for the restock mission, with provenance.

Default (`--plan`, CPU, no Isaac): resolve the per-station commissioning plan from the
camera design + mission scenario (station -> mission location -> camera -> capture pose ->
required footprint / markers / lifetime) and print it. This proves the wiring and is the
plan the native run executes; it captures nothing.

`--execute`: run the choreography. With each target station EMPTY and the robot at its
capture pose, capture the eye-in-hand/station RGB-D frame, commission the support via the
production `commission()` path, and persist the baseline with provenance under the dataset
dir. All robot motion, empty staging, frame capture and empty-polygon selection are
integration/GPU-owned and supplied by a `--sources module:function` hook (this script
imports no Isaac/ROS); the choreography here only commissions and persists. A station the
hook does not cover is refused, never fabricated.

The hook is called `sources(scenario=..., camera_config=..., output=...)` and must return
`{'contexts': [RenderedSupportContext, ...], 'capture': capture}` where `capture(context)`
returns `{'frame', 'now', 'empty_polygons', 'capture_poses'?, 'raw_frames'?}` (see
`twin.restock_camera.commissioning.run_commissioning`).
"""
import argparse
import importlib
import json
from pathlib import Path

# The three mission stations that need an empty-support baseline, matching the
# ros_host_config.json `walls` keys (collect source, cart rollwagen, transfer shelf). The cart
# `rollwagen` context is commissioned empty at the staging pose before pickup; sensing_sources
# refuses it today as uncommissioned, so it must be in the default run.
DEFAULT_STATIONS = ('backstock_00', 'rollwagen', 'G3_back_s2')


def _resolve(spec):
    module, _, name = spec.partition(':')
    if not module or not name:
        raise ValueError(f'expected module:function, got {spec!r}')
    return getattr(importlib.import_module(module), name)


def support_locations(scenario):
    """Map design support names to mission locations from the scenario job.

    The camera design names supports 'source'/'cart'/'shelf'; the mission addresses them by
    the job's ``source_bin`` (e.g. 'backstock_00'), the rollwagen, and the ``shelf_id``.
    """
    job = scenario.get('job', {})
    return {'source': job.get('source_bin', 'source'), 'cart': 'rollwagen',
            'shelf': job.get('shelf_id', 'shelf')}


def _station_pose(staging, camera_id):
    """The robot base capture pose for a station, from scenario staging.

    The shelf station is reached at ``shelf_robot_pose``; the collect source (and cart
    pickup) at ``robot_base_pose``. Cart-arrival/transit poses are navigation-owned.
    """
    return staging.get('shelf_robot_pose') if camera_id == 'shelf' else staging.get('robot_base_pose')


def _rendered_intrinsics(intrinsics, render_scale):
    if render_scale == 1.0:
        return intrinsics
    from twin.restock_camera.geometry import Intrinsics, scale_intrinsics
    from dataclasses import asdict
    return asdict(scale_intrinsics(Intrinsics(**intrinsics), render_scale))


def commissioning_plan(camera_config, scenario, *, stations, locations=None, lifetime_s=600.,
                       render_scale=1.0):
    """Per-station commissioning plan from the camera design + scenario (no capture)."""
    staging = scenario['initial_staging']
    locations = locations if locations is not None else support_locations(scenario)
    by_location = {}
    for name, support in camera_config['supports'].items():
        location = locations.get(name, name)
        by_location[location] = (name, support)
    plan = []
    for location in stations:
        if location not in by_location:
            raise ValueError(f'no support in the camera design maps to station {location!r}; '
                             f'known: {sorted(by_location)}')
        design_name, support = by_location[location]
        camera_id = support['camera_id']
        markers = [m['id'] for m in support['markers']]
        if not markers:
            raise ValueError(f'support {design_name!r} has no registration marker')
        plan.append(dict(station=location, design_support=design_name, camera_id=camera_id,
                         support_id=support['id'], revision=support['revision'],
                         marker_id=markers[0], auxiliary_id=markers[1] if len(markers) > 1 else None,
                         required_xy_m=support['required_xy_m'],
                         capture_pose=_station_pose(staging, camera_id),
                         intrinsics=camera_config['cameras'][camera_id]['intrinsics'],
                         rendered_intrinsics=_rendered_intrinsics(
                             camera_config['cameras'][camera_id]['intrinsics'], render_scale),
                         render_scale=render_scale,
                         depth_error_bound_m=camera_config['required_depth_error_bound_m'],
                         lifetime_s=lifetime_s))
    return dict(schema='restock-commissioning-plan.v1', render_scale=render_scale, stations=plan,
                invariant='baselines come only from observed empty-support commissioning; '
                          'a missing station is refused, never fabricated')


def _execute(args, camera_config, scenario, stations):
    from twin.restock_camera import commissioning as C
    sources = _resolve(args.sources)(scenario=scenario, camera_config=camera_config, output=args.output)
    contexts = sources['contexts'] if isinstance(sources, dict) else sources.contexts
    capture = sources['capture'] if isinstance(sources, dict) else sources.capture
    records = C.run_commissioning(contexts, capture, dataset_dir=args.output, lifetime_s=args.lifetime_s)
    by_station = {r.station: r for r in records}
    missing = [s for s in stations if s not in by_station]
    if missing:
        raise C.CommissioningBaselineMissing(f'commissioning did not produce required station(s): {missing}')
    report = dict(schema='restock-commissioning-report.v1', dataset_dir=str(args.output),
                  stations={s: by_station[s].provenance() for s in stations})
    (Path(args.output) / 'commissioning-report.json').write_text(json.dumps(report, indent=1, sort_keys=True))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--camera-config', type=Path, required=True,
                        help='restock_camera.config.camera_config(...) output JSON')
    parser.add_argument('--scenario', type=Path, required=True, help='mission scenario JSON')
    parser.add_argument('--output', type=Path, help='dataset dir for persisted baselines (execute)')
    parser.add_argument('--station', action='append', dest='stations',
                        help='mission location to commission (repeatable; default: collect source + cart + shelf)')
    parser.add_argument('--sources', help='module:function returning booted {contexts, capture} (execute)')
    parser.add_argument('--lifetime-s', type=float, default=600.)
    parser.add_argument('--render-scale', type=float, default=1.0,
                        help='render-product scale to commission/plan at (intrinsics scale with it)')
    parser.add_argument('--render-preset', choices=('full', 'fast'),
                        help='opt into a sensing render preset (full|fast); resolves --render-scale '
                             '(fast=0.6). Use the SAME preset for the mission so the baseline matches.')
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--plan', action='store_true', help='print the commissioning plan (default; CPU, no capture)')
    mode.add_argument('--execute', action='store_true', help='run the choreography and persist baselines')
    args = parser.parse_args()
    if args.render_preset:                        # one opt-in flag, shared with the mission
        from twin.restock_camera.config import preset_render_scale
        args.render_scale = preset_render_scale(args.render_preset)
    try:
        camera_config = json.loads(args.camera_config.read_text())
        scenario = json.loads(args.scenario.read_text())
        stations = tuple(args.stations) if args.stations else DEFAULT_STATIONS
        plan = commissioning_plan(camera_config, scenario, stations=stations, lifetime_s=args.lifetime_s,
                                  render_scale=args.render_scale)
        if not args.execute:
            print(json.dumps(plan, indent=2))
            return 0
        if not args.sources or not args.output:
            raise ValueError('--execute requires --sources module:function and --output')
        print(json.dumps(_execute(args, camera_config, scenario, stations), indent=2), flush=True)
        return 0
    except (ValueError, KeyError, TypeError, OSError, ImportError) as exc:
        print(json.dumps({'errors': [f'{type(exc).__name__}: {exc}'], 'committed': False}))
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
