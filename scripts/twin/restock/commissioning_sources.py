"""Integration `--sources` hook for `scripts/run_restock_commissioning.py --execute`.

Boots Isaac, instantiates the composed store as a STATIC optical bench (reusing
`view_scene.build`, dynamics disabled) with the collect source EMPTY (the authored
`/RestockScenario/HandlingUnit` carton is deactivated so the bare support plane is
observed), renders the fixed `source` station camera's RGB-D, selects the empty
support + witness polygons from the camera design, and returns the contract
`{'contexts': [RenderedSupportContext], 'capture': capture}` that
`twin.restock_camera.commissioning.run_commissioning` consumes.

Scope: the collect `source` (`backstock_00`), the transfer `shelf` (`shelf_id`, e.g.
`G3_back_s2`) and the `cart` (`rollwagen`). The source/shelf cameras are world-fixed
(`parent='map'`); the cart camera is parented to the MOVING `cart_chassis` (`/ViewCart/chassis`
in the bench), so its extrinsic + projected deck polygons are composed through the parent's bench
world pose by the camera-lane `twin.restock_camera.sensing.commissioning_geometry`. The stored
baseline is support-local and so invariant to the parent pose (the cart's bench pose cannot corrupt
it; the live transfer re-registers the deck markers against the live base_from_camera).
`capture_poses` records the per-station capture pose as provenance only (`shelf_robot_pose` for the
shelf, `cart_dock_robot_pose`/`robot_base_pose` for the cart, else `robot_base_pose`). Commission
ONE station per boot, selected by `COMMISSIONING_STATIONS` (default `backstock_00`); loop the
guarded parent per station for more (baselines accumulate in the shared dataset dir).

No fabrication: the baseline is produced solely by the production
`RenderedSupportContext.commission` on the real empty-support render; a station
whose markers do not decode or whose plane is not flat is refused upstream.

Run via the guarded parent `scripts/run_restock_commissioning_gpu.py`, which holds
the GPU slot and launches the commissioning CLI under the task-local NVIDIA
runtime + isaac-python. The bench paths (store_usd, package_roots, ...) come from
the JSON named by the `COMMISSIONING_BENCH_CONFIG` environment variable.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

# Default commissioning location when COMMISSIONING_STATIONS is unset (the collect source).
DEFAULT_LOCATION = 'backstock_00'


def _resolve_station(scenario):
    """Resolve the single requested commissioning (design_station, location) for this boot.

    ``COMMISSIONING_STATIONS`` is a comma list of mission LOCATIONS (default ``backstock_00``);
    exactly one station is commissioned per boot. The map-parented source/shelf and the
    moving-parent cart (cart_chassis, via commissioning_geometry) are supported; an unknown
    location is refused."""
    requested = [s for s in os.environ.get('COMMISSIONING_STATIONS', DEFAULT_LOCATION).split(',') if s]
    if len(requested) != 1:
        raise ValueError('commission exactly one map-parented station per boot '
                         f'(got {requested!r}); loop the guarded parent per station for more')
    location = requested[0]
    job = scenario.get('job', {})
    # design support name -> mission location (mirrors run_restock_commissioning.support_locations)
    support_locations = {'source': job.get('source_bin', 'source'),
                         'cart': 'rollwagen', 'shelf': job.get('shelf_id', 'shelf')}
    design_of = {loc: name for name, loc in support_locations.items()}
    if location not in design_of:
        raise ValueError(f'unknown commissioning location {location!r}; known {sorted(design_of)}')
    station = design_of[location]
    if station not in ('source', 'shelf', 'cart'):
        raise ValueError(f'unsupported commissioning station {station!r}; known source/shelf/cart')
    return station, location


def build_sources(*, scenario, camera_config, output):
    """Boot the empty-source optical bench and return {contexts, capture}.

    `scenario`/`camera_config` are the loaded dicts passed by the commissioning CLI;
    `output` is the baselines dataset dir. Bench input paths come from
    `COMMISSIONING_BENCH_CONFIG`.
    """
    from isaacsim import SimulationApp
    app = SimulationApp({'headless': True, 'width': 640, 'height': 360,
                         'renderer': 'RayTracedLighting', 'anti_aliasing': 0, 'multi_gpu': False})

    import omni.usd
    from pxr import UsdGeom, UsdLux, Gf  # noqa: F401
    from isaacsim.core.api import World
    from twin.restock.view_scene import build as build_bench
    from twin.restock_camera.geometry import Intrinsics, pose_matrix
    from twin.restock_camera.sensing import support_slots, rendered_context

    bench_cfg = json.loads(Path(os.environ['COMMISSIONING_BENCH_CONFIG']).read_text())
    bench_out = Path(output) / 'bench'
    bench_out.mkdir(parents=True, exist_ok=True)

    design = camera_config
    staging = scenario['initial_staging']
    mobile = json.loads(Path(bench_cfg['mobile_manifest']).read_text())
    STATION, LOCATION = _resolve_station(scenario)
    support = design['supports'][STATION]
    cam = design['cameras'][STATION]
    if cam.get('parent') not in ('map', 'cart_chassis'):
        raise ValueError(f'station {STATION!r} camera parent {cam.get("parent")!r} unsupported; '
                         'only map-parented (source/shelf) or cart_chassis (cart) cameras are commissioned here')
    k = Intrinsics(**cam['intrinsics'])

    world = World(physics_dt=1 / 60., rendering_dt=1 / 60., stage_units_in_meters=1.)
    stage = omni.usd.get_context().get_stage()
    UsdGeom.Xform.Define(stage, '/World')
    light = UsdLux.DomeLight.Define(stage, '/World/Light')
    light.CreateIntensityAttr(1000.)
    # Author the full composed store + physical cameras + printed markers as a static bench.
    camera, _spec, _codes = build_bench(stage, bench_cfg, bench_out, view=STATION)

    # Present the EMPTY support to the station camera. For the collect `source`, deactivate the
    # authored handling-unit carton so the bare support plane at z=unit_bottom_z is observed. For the
    # `shelf`, the slot is already depleted in the authored scenario (target_initially_depleted), so
    # no deactivation is needed; the source carton (elsewhere, out of the shelf camera's view) is
    # still deactivated if present so a stray unit can never intrude on either baseline.
    unit = stage.GetPrimAtPath('/RestockScenario/HandlingUnit')
    if STATION == 'source' and (not unit or not unit.IsValid()):
        raise ValueError('expected authored /RestockScenario/HandlingUnit to empty for source commissioning')
    if unit and unit.IsValid():
        unit.SetActive(False)

    world.reset()
    for _ in range(90):
        world.step(render=True)

    # Station commissioning geometry: the camera extrinsic (base_from_camera) and the empty
    # support + witness quads projected to image UV, all parent-aware. For the MAP-parented
    # source/shelf the parent is the world-fixed /World identity; the shelf is reached at
    # shelf_robot_pose, the collect source at robot_base_pose. For the CART the camera is parented
    # to the MOVING cart_chassis (/ViewCart/chassis in the bench), so the parent's bench world pose
    # must be composed or the extrinsic and projected deck polygons are silently wrong -- that is
    # the camera-lane commissioning_geometry helper. The stored baseline is support-local, so it is
    # invariant to the cart's bench pose (support_uv/witness_uv cancel world_from_parent); the live
    # transfer re-registers the deck markers against the live base_from_camera.
    from twin.restock_camera.sensing import commissioning_geometry
    if cam['parent'] == 'cart_chassis':
        chassis = stage.GetPrimAtPath('/ViewCart/chassis')
        if not chassis or not chassis.IsValid():
            raise ValueError('expected authored /ViewCart/chassis for cart commissioning')
        world_from_parent = np.array(UsdGeom.XformCache().GetLocalToWorldTransform(chassis)).T
        base_pose = staging.get('cart_dock_robot_pose', staging['robot_base_pose'])
    else:
        world_from_parent = np.eye(4)
        base_pose = staging['shelf_robot_pose'] if STATION == 'shelf' else staging['robot_base_pose']
    world_from_base = pose_matrix(base_pose, float(mobile['spec']['base_z']))
    geom = commissioning_geometry(design, STATION, world_from_parent=world_from_parent,
                                  world_from_base=world_from_base)
    base_from_camera = geom['base_from_camera']
    support_uv, witness_uv = geom['support_uv'], geom['witness_uv']

    label_masks = {LOCATION: np.ones((k.height, k.width), bool)}
    # Full design-support -> mission-location map (so support_slots resolves THIS station's slot to
    # its own location, e.g. shelf -> shelf_id); the label mask is keyed by this station's LOCATION.
    support_locations = {'source': scenario['job'].get('source_bin', 'backstock_00'),
                         'cart': 'rollwagen', 'shelf': scenario['job']['shelf_id']}
    slots = support_slots(design, STATION, label_masks=label_masks, support_locations=support_locations)

    import time as _time
    from twin.restock.sensing_sources import COMMISSIONING_STEADY_DOMAIN

    def measure_context(frame, now):
        # Static bench: parked, tf stamped at the render time (|stamp - tf| = 0 passes the gate).
        return (base_from_camera, float(now), 'commissioning', True)

    def clock_context(frame, now):
        # Persistent WALL-clock acquisition stamp under the shared domain the mission also uses,
        # so sensed.age_at measures real elapsed wall seconds (not an incomparable sim/monotonic
        # delta) across the separate commissioning and mission processes. See sensing_sources.
        return ('commission', COMMISSIONING_STEADY_DOMAIN, _time.time())

    context = rendered_context(design, STATION, slots, measure_context=measure_context,
                              clearance_for_slot=lambda loc, view, frame, now: None,
                              camera_revision=cam['revision'], clock_context=clock_context)

    from twin.restock_camera.qr import FixedTileQR
    from twin.restock_camera.sensed import FrameEvidence
    need = {support['markers'][0]['id']}
    if len(support['markers']) > 1:
        need.add(support['markers'][1]['id'])

    def capture(ctx):
        # RTX RayTracedLighting needs frames to converge and the paired markers must both
        # decode with clean (untruncated, convex) quads under the PRODUCTION FixedTileQR --
        # the same gate commission applies internally. Scan converged frames and hand
        # commission the first that the production decoder reads both markers from.
        last = None
        attempts = []
        chosen = None
        last_rgb = None
        for i in range(240):
            world.step(render=True)
            frame = camera.get_current_frame(clone=True)
            identity = frame.get('native_rendering_frame', frame.get('rendering_frame'))
            if identity is None or identity == last:
                continue
            last = identity
            last_rgb = np.asarray(frame['rgb'])[:, :, :3]
            stamp = float(frame['rendering_time'])
            probe = FixedTileQR(STATION)
            evidence = FrameEvidence(STATION, 1_000_000 + i, stamp, stamp, 'probe', 'probe')
            try:
                qr = probe.decode(last_rgb, evidence, now=stamp, parked=True)
                codes = {d.text for d in qr.detections}
                if need <= codes:
                    chosen = frame
                    break
                attempts.append((i, sorted(codes)))
            except Exception as exc:  # truncated quad / spurious detection on this frame
                attempts.append((i, str(exc)))
        if chosen is None:
            if last_rgb is not None:
                np.save(Path(output) / 'debug-last-rgb.npy', last_rgb)
            raise ValueError(f'no converged frame cleanly decoded source markers {sorted(need)} '
                             f'in {len(attempts)} frames; last: {attempts[-5:]}')
        rgb = np.asarray(chosen['rgb'])[:, :, :3]
        depth = np.asarray(chosen['distance_to_image_plane'], float)
        now = float(chosen['rendering_time'])
        np.savez_compressed(Path(output) / 'commission-frame.npz', rgb=rgb, depth=depth,
                            support_uv=support_uv, witness_uv=witness_uv)
        return {'frame': chosen, 'now': now,
                'empty_polygons': {LOCATION: (support_uv, witness_uv)},
                'capture_poses': {LOCATION: list(base_pose)},
                'raw_frames': {LOCATION: dict(rgb=rgb, depth=depth, support_uv=support_uv,
                                              witness_uv=witness_uv, base_from_camera=base_from_camera)}}

    return {'contexts': [context], 'capture': capture, '_app': app}
