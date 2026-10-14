"""Integration `--sources` hook for `scripts/run_restock_commissioning.py --execute`.

Boots Isaac, instantiates the composed store as a STATIC optical bench (reusing
`view_scene.build`, dynamics disabled) with the collect source EMPTY (the authored
`/RestockScenario/HandlingUnit` carton is deactivated so the bare support plane is
observed), renders the fixed `source` station camera's RGB-D, selects the empty
support + witness polygons from the camera design, and returns the contract
`{'contexts': [RenderedSupportContext], 'capture': capture}` that
`twin.restock_camera.commissioning.run_commissioning` consumes.

Scope: the collect `source` station (`backstock_00`) only — the one the collect
grasp needs. The `source`/`shelf` design cameras are world-fixed (`parent='map'`),
so the baseline geometry (stored support-local) is invariant to the robot base
pose; `capture_poses` records the scenario `robot_base_pose` as provenance only.

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

# Design support name -> mission location (mirrors run_restock_commissioning.support_locations).
STATION = 'source'
LOCATION = 'backstock_00'


def _quad_world(center_xy, size_xy, yaw, z):
    """Ordered (4,2->3) convex rectangle corners in world XY at height z."""
    c, s = np.cos(yaw), np.sin(yaw)
    hx, hy = size_xy[0] / 2., size_xy[1] / 2.
    local = [(-hx, -hy), (hx, -hy), (hx, hy), (-hx, hy)]
    rot = np.array([[c, -s], [s, c]])
    out = []
    for lx, ly in local:
        dx, dy = rot @ (lx, ly)
        out.append([center_xy[0] + dx, center_xy[1] + dy, z])
    return np.asarray(out, float)


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
    from twin.restock_camera.geometry import Intrinsics, rigid, pose_matrix, transform_points, project
    from twin.restock_camera.sensing import support_slots, rendered_context

    bench_cfg = json.loads(Path(os.environ['COMMISSIONING_BENCH_CONFIG']).read_text())
    bench_out = Path(output) / 'bench'
    bench_out.mkdir(parents=True, exist_ok=True)

    design = camera_config
    staging = scenario['initial_staging']
    mobile = json.loads(Path(bench_cfg['mobile_manifest']).read_text())
    support = design['supports'][STATION]
    cam = design['cameras'][STATION]
    k = Intrinsics(**cam['intrinsics'])

    world = World(physics_dt=1 / 60., rendering_dt=1 / 60., stage_units_in_meters=1.)
    stage = omni.usd.get_context().get_stage()
    UsdGeom.Xform.Define(stage, '/World')
    light = UsdLux.DomeLight.Define(stage, '/World/Light')
    light.CreateIntensityAttr(1000.)
    # Author the full composed store + physical cameras + printed markers as a static bench.
    camera, _spec, _codes = build_bench(stage, bench_cfg, bench_out, view=STATION)

    # Empty the collect source: deactivate the authored handling-unit carton (and its label)
    # so the bare support plane at z=unit_bottom_z is what the source camera observes. The
    # support cube itself and the shelf (already depleted) are left intact.
    unit = stage.GetPrimAtPath('/RestockScenario/HandlingUnit')
    if not unit or not unit.IsValid():
        raise ValueError('expected authored /RestockScenario/HandlingUnit to empty for commissioning')
    unit.SetActive(False)

    world.reset()
    for _ in range(90):
        world.step(render=True)

    # Fixed source camera: parent='map' and /World is identity, so parent_from_optical is
    # world_from_optical. base_from_camera is the true camera-relative-to-base at capture
    # (provenance + base-frame support measurement; stored geometry is support-local).
    world_from_optical = rigid(cam['parent_from_optical'])
    base_pose = staging['robot_base_pose']
    world_from_base = pose_matrix(base_pose, float(mobile['spec']['base_z']))
    base_from_camera = np.linalg.inv(world_from_base) @ world_from_optical

    # Empty support + witness quads from the design geometry, projected to image UV. The
    # support quad is enlarged slightly over the required footprint (so the measured corners
    # contain it with the registration margin) while staying on the flat support-cube top;
    # the witness is the design off-slot reference patch (disjoint from the support).
    center = np.asarray(support['nominal_center'], float)
    req = support['required_xy_m']
    support_world = _quad_world(center[:2], (req[0] + .04, req[1] + .04), float(support['yaw']), float(center[2]))
    wd = support['witness_design']
    witness_world = _quad_world(wd['center'][:2], wd['size_xy'], float(wd['yaw']), float(wd['center'][2]))

    def _project(points_world):
        uv, valid = project(transform_points(np.linalg.inv(world_from_optical), points_world), k)
        if not valid.all():
            raise ValueError('empty-support polygon projects outside the source view')
        return uv

    support_uv, witness_uv = _project(support_world), _project(witness_world)

    label_masks = {LOCATION: np.ones((k.height, k.width), bool)}
    support_locations = {'source': LOCATION, 'cart': 'rollwagen', 'shelf': scenario['job']['shelf_id']}
    slots = support_slots(design, STATION, label_masks=label_masks, support_locations=support_locations)

    def measure_context(frame, now):
        # Static bench: parked, tf stamped at the render time (|stamp - tf| = 0 passes the gate).
        return (base_from_camera, float(now), 'commissioning', True)

    context = rendered_context(design, STATION, slots, measure_context=measure_context,
                              clearance_for_slot=lambda loc, view, frame, now: None,
                              camera_revision=cam['revision'])

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
