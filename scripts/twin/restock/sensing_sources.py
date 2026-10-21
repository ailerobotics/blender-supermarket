"""Integration-owned `assembly.sensing_sources` for the camera lane's sensing_factory.

`twin.restock_camera.sensing:build_sensing` (sensing_factory, PR#35) requires the live
sensing bindings on `assembly.sensing_sources` (its module docstring + docs/restock_camera.md
are the contract): the station cameras, per-camera `RenderedSupportContext` providers, the
`location_cameras` map, the grasp `profiles`, the `pose_bridge_root`, the mapper/odometry
`motion` port, the `contact` reader and the mission `recording_camera`. This module builds
that mapping from the composed assembly + the camera design, wiring:

- cameras / `base_from_camera` from the restock `CameraRig` (built on the composed stage);
- per-camera slots/contexts from the camera design via `support_slots` + `rendered_context`;
- `measure_context` from the rig TF + the manipulation motion's parked state;
- `clearance_for_slot` CONSERVATIVE: it returns ``None`` (no incorporated clearance) unless a
  live nvblox mapper is co-launched in the mission runtime and supplied here. With no live
  mapper the sensing stages fail closed on absent clearance — the honest measured outcome,
  never a fabricated "clear". (Wiring a co-launched mapper's incorporated slice into clearance
  is the remaining mission-runtime integration; the incorporation path itself is proven by
  run_incorporation_identity_gpu.py.)

The Isaac rig construction is isolated behind `_rig_builder` so the assembly/wiring is CPU
tested; the real builder runs only inside the guarded app. No observation is fabricated.
"""
from __future__ import annotations

import os
import time as _time

from twin.restock_camera.config import camera_config as build_camera_design
from twin.restock_camera.sensing import support_slots, rendered_context

# Shared persistent acquisition-clock domain for the commissioned-support age gate
# (sensed.RegisteredSupport.age_at). Commissioning and the mission run in SEPARATE processes
# with independent sim clocks and incomparable per-process monotonic clocks, so the baseline
# and the live capture both stamp their evidence with WALL time (time.time()) under this one
# fixed domain; age_at then measures real elapsed wall seconds since commissioning, bounded by
# the baseline lifetime, instead of a reversed sim-time delta. The commissioning hook
# (twin.restock.commissioning_sources) imports this exact domain so the two agree.
COMMISSIONING_STEADY_DOMAIN = 'restock-commission-wallclock-v1'


def _wall_clock_context(frame, now):
    return ('mission', COMMISSIONING_STEADY_DOMAIN, _time.time())

def _all_true_mask(design, camera_id):
    import numpy as np
    intr = design['cameras'][camera_id]['intrinsics']
    return np.ones((int(intr['height']), int(intr['width'])), bool)


def build_sensing_sources(*, assembly, scenario, manifest, manip_runtime, pose_bridge_root,
                          render_scale=1.0, full_res_locations=(), clearance_for_slot=None,
                          mast_depth=False, mast_depth_settings=None,
                          _rig_builder=None, _mast_bridge_builder=None):
    """Assemble the sensing_sources mapping. Pure wiring; the rig is built behind the seam.

    pose_bridge_root is the FILESYSTEM root of the tray-perception pose bridge
    (``<root>/labeling/pose_bridge.py``) that IsaacCartonSensors loads for depth
    geometry -- not a USD prim path.

    ``render_scale`` is the default (VRAM-bounded) render-product scale for the mapper and
    other cameras; ``full_res_locations`` are mission locations whose serving camera must
    render at 1.0 instead (decode-grade, e.g. the collect source whose paired markers are
    only ~6 px/module). Each camera's context intrinsics are scaled by the SAME per-camera
    factor so the context, its commissioned baseline and the live capture share one calibration.
    """
    # ``mast_depth`` (default off) adds the body-mounted mast depth camera (nvblox camera_1) to the
    # design and, below, authors its prim + masked-depth publisher. False/absent => byte-identical:
    # no mast camera in the design, no mast prim/render product, no second publisher.
    design = build_camera_design(scenario, manifest, mast_depth=mast_depth)
    shelf_id = scenario['job']['shelf_id']
    # Map the design 'source' support to the job's source_bin (the mission's address for the
    # pick location, e.g. 'backstock_00') exactly as run_restock_commissioning does, so the
    # commissioned baseline, the source context slot and collect_source_location all agree.
    support_locations = {'source': scenario['job'].get('source_bin', 'source'),
                         'cart': 'rollwagen', 'shelf': shelf_id}

    # location -> camera_id, from each design support's assigned camera (needed before the rig
    # to resolve which cameras must render full-res).
    location_cameras = {}
    for name, support in design['supports'].items():
        location_cameras[support_locations.get(name, name)] = support['camera_id']
    full_res_cameras = {location_cameras[loc] for loc in full_res_locations if loc in location_cameras}
    camera_scales = {cid: (1.0 if cid in full_res_cameras else render_scale)
                     for cid in {s['camera_id'] for s in design['supports'].values()}}

    rig = (_rig_builder or _real_rig)(assembly=assembly, design=design, render_scale=render_scale,
                                      camera_render_scales=camera_scales)
    cameras = rig['cameras']
    base_from_camera = rig['base_from_camera']       # callable(camera_id) -> 4x4
    recording_camera = rig['recording_camera']

    # No co-launched live mapper by default: clearance is conservatively absent, so sensing
    # fails closed rather than fabricating free space. A real mapper clearance callback may
    # be injected once the mission runtime co-launches the incorporating mapper.
    if clearance_for_slot is None:
        def clearance_for_slot(location, support_view, frame, now):
            return None

    def measure_context_for(camera_id):
        def measure_context(frame, now):
            parked = bool(manip_runtime.motion.parked())
            return (base_from_camera(camera_id), float(now), 'mission', parked)
        return measure_context

    context_providers = {}
    for camera_id in sorted({s['camera_id'] for s in design['supports'].values()}):
        masks = {support_locations.get(n, n): _all_true_mask(design, camera_id)
                 for n, s in design['supports'].items() if s['camera_id'] == camera_id}
        slots = support_slots(design, camera_id, label_masks=masks, support_locations=support_locations)
        if not slots:
            continue
        context_providers[camera_id] = rendered_context(
            design, camera_id, slots, measure_context=measure_context_for(camera_id),
            clearance_for_slot=clearance_for_slot,
            camera_revision=design['cameras'][camera_id]['revision'],
            render_scale=camera_scales[camera_id], clock_context=_wall_clock_context)

    profiles = dict(getattr(manip_runtime.controller, 'profiles', {}))

    # Body-mounted mast depth camera (nvblox camera_1): author its prim on robot_base + wire a second
    # masked_depth_ipc publisher on the MAST_* topics/frame (rig+carton self-masked), fed to the port's
    # per-tick masked_depth hook. None unless opted in; the camera design already excludes 'mast' when
    # off, so this is byte-identical when disabled.
    mast_depth_bridge = None
    if mast_depth:
        from .masked_depth import build_mast_depth_bridge
        build_mast = _mast_bridge_builder or build_mast_depth_bridge
        mast_depth_bridge = build_mast(stage=assembly.stage, assembly=assembly, design=design,
                                       settings=mast_depth_settings or {})

    def stage_scoped_teardown():
        """fm381: DESTROY (free VRAM, not just pause) the station cameras this mission NEVER reads --
        i.e. the SHELF camera (transfer-only; no transfer stage here), via _unused_sensing_camera_ids.
        Called ONCE by IsaacPort at grasp-confirmed (before the sim-223 release peak). Deliberately does
        NOT touch the SOURCE camera: run-300b showed the collect capture HOLDS source through the whole
        cart-swing (destroying it mid-collect starved /sensors -> sensor-stream watchdog). source stays
        resident; shelf alone (~0.95 GB freed before the peak) keeps the peak under the card. cart + nav
        + source untouched; fail-safe (never raises)."""
        from .native_cameras import destroy_camera_render_product
        freed = []
        for cid in _unused_sensing_camera_ids(design):
            cam = cameras.get(cid)
            if cam is not None:
                destroy_camera_render_product(cam, getattr(assembly, 'stage', None))
                freed.append(cid)
        print('STAGE_SCOPED_SENSING torn down: %s' % (freed or 'none'), flush=True)
        return freed

    return {
        'cameras': cameras,
        'stage_scoped_teardown': stage_scoped_teardown,
        # Mast masked-depth publisher (nvblox camera_1), or None when mast depth is off. The port
        # publishes it on the sensing cadence via its masked_depth hook (isaac_bundle wires it).
        'mast_depth': mast_depth_bridge,
        'context_providers': context_providers,
        'location_cameras': location_cameras,
        'profiles': profiles,
        'pose_bridge_root': pose_bridge_root,
        'motion': manip_runtime.motion,
        'contact': assembly.contact,
        'recording_camera': recording_camera,
        # SIM/node clock for the capture's acquisition-to-use freshness gate (build_sensing passes
        # it to StationSensors as steady_clock). Measuring the standing 0.5 s budget on the SIM
        # clock -- which barely advances across a render tick -- is what stops a full-res source
        # render+decode (>0.5 s WALL under render pacing) from false-tripping it; a wall backstop
        # (config sensor_wall_backstop_s, default 60 s) still hard-stops a frozen render. Matches
        # the domain of the capture's `now` and the frame rendering_time (both sim seconds); any
        # constant offset cancels since the budget bounds (clock()-started)+(now-stamp).
        'steady_clock': lambda: float(assembly.world.current_time),
    }


def _unused_sensing_camera_ids(design):
    """fm379 mitigation 2 (pure): camera ids serving ONLY the shelf support -- read by the transfer
    stage, which this mission does not run. Returns the shelf support's camera_id unless that camera
    also serves another (still-used) support. Empty if there is no shelf support."""
    supports = design.get('supports', {})
    shelf = supports.get('shelf')
    if not shelf:
        return ()
    shelf_cam = shelf['camera_id']
    others = {s['camera_id'] for n, s in supports.items() if n != 'shelf'}
    return () if shelf_cam in others else (shelf_cam,)


def _stage_scoped_teardown_camera_ids(design):
    """fm380 (pure): station camera ids safe to DESTROY at the collect->couple boundary -- those
    serving ONLY the ``source`` and/or ``shelf`` supports (read in collect / transfer-only). NEVER the
    cart camera (kept for couple/verify + nav) and never a camera shared with any support outside
    {source, shelf}. Empty if neither exists. Mirrors the usage analysis: couple/outbound/park read no
    station QR camera, and source is read only in collect."""
    supports = design.get('supports', {})
    scoped = {s['camera_id'] for n, s in supports.items() if n in ('source', 'shelf')}
    keep = {s['camera_id'] for n, s in supports.items() if n not in ('source', 'shelf')}
    return tuple(sorted(scoped - keep))


def station_camera_specs(design, parent_paths, *, drop_camera_ids=()):
    """CameraRig specs for the FIXED station cameras only (pure; no Isaac).

    ``drop_camera_ids`` (fm379 mitigation 2): camera ids to OMIT so their RTX render product is never
    created -- for a mission whose active stages never read that camera (e.g. the shelf camera when
    there is no transfer stage). Empty (default) => every eligible camera, byte-identical.

    The camera design keeps camera bodies and their mount collision boxes in separate lists
    (``cameras`` + ``solids`` with ``dimensions_m``); CameraRig expects each camera to carry its
    own ``collision_solids`` (``size_m``), so attach each camera's housing (and bracket, when
    present) from the solids list.

    Eye-in-hand cameras (e.g. the ``wrist`` collision camera on ur_tool0, frame
    ``restock_camera_wrist``) are parented to a robot LINK, not a fixed station parent, and are
    authored by their own path (``mobile_assets.author_wrist_camera`` /
    ``native_cameras.initialize_collision_camera``), not this fixed-station CameraRig. They are
    skipped here (parent absent from ``parent_paths``); the sensing supports/context/location
    maps never reference an eye-in-hand camera, so nothing downstream expects it in the rig.
    """
    solids_by_id = {s['id']: s for s in design.get('solids', [])}
    specs = []
    drop = set(drop_camera_ids)
    for name, camera in design['cameras'].items():
        if camera.get('parent') not in parent_paths:
            continue
        if name in drop:
            continue
        collision_solids = []
        for suffix in ('_housing', '_bracket'):
            solid = solids_by_id.get(name + suffix)
            if solid is not None:
                collision_solids.append({'parent_from_solid': solid['parent_from_solid'],
                                         'size_m': solid['dimensions_m']})
        if not collision_solids:
            raise ValueError(f'camera {name!r} has no housing/bracket collision solid in the design')
        specs.append(dict(camera, collision_solids=collision_solids))
    return specs


def _real_rig(*, assembly, design, render_scale=1.0, camera_render_scales=None):
    """Build the Isaac CameraRig on the composed stage and expose the sensing cameras."""
    from pxr import UsdGeom
    from .native_cameras import CameraRig
    cart_root = assembly.evidence['composition']['assets']['cart']['root']
    # 'map'-parented cameras (source, shelf) use world-origin poses. The view_scene bench
    # anchors them under an identity '/World'; the composed mission stage has no '/World'
    # (its default prim is the sublayered store), so author that identity anchor here.
    if not assembly.stage.GetPrimAtPath('/World'):
        UsdGeom.Xform.Define(assembly.stage, '/World')
    parent_paths = {'map': '/World', 'cart_chassis': cart_root}
    # fm379 mitigation 2: SIM-ONLY opt-in to skip creating sensing cameras no active stage reads, so
    # their RTX render product never allocates. RESTOCK_DROP_UNUSED_SENSING set => drop the SHELF
    # camera, which only serves the TRANSFER stage; this mission has no transfer (stages collect/
    # couple/outbound/park, transfer_shelf_location None) so it is pure overhead. Unset => byte-
    # identical. (Explicit to the shelf camera: a mission WITH transfer must not set this.)
    _drop_ids = _unused_sensing_camera_ids(design) if os.environ.get('RESTOCK_DROP_UNUSED_SENSING') else ()
    specs = station_camera_specs(design, parent_paths, drop_camera_ids=_drop_ids)
    rig = CameraRig(stage=assembly.stage, specs=specs, parent_paths=parent_paths, frequency=-1,
                    render_scale=render_scale, camera_render_scales=camera_render_scales)
    # CameraRig authors only the camera prims + their mount collision boxes. The support
    # REGISTRATION markers (the printed QR planes the sensing decode reads every frame) are not
    # among them, so author them for the SOURCE (collect), CART (rollwagen) and SHELF (G3_back_s2)
    # supports the transfer/verify stages scan -- otherwise their captures fail "fresh support
    # marker(s) not decoded".
    author_mission_markers(assembly.stage, design, parent_paths)
    base_root = assembly.paths['ur_base_link']
    # Captain-facing walkthrough video (firstmate 222): the destination-fixed shelf camera only ever
    # framed dark floor while the robot was at the backstock / in transit, so record instead from a
    # third-person CHASE camera parented to the MOVING base. It rides 2.6 m above ur_base_link looking
    # straight down with a wide (~96 deg) field, so the base, the UR10, the towed rollwagen and the
    # adjacent support (backstock at collect, shelf at place) stay in frame through collect/couple/
    # tow/place, lit by the store ceiling rig, and it follows the base's turns during tow. Added as an
    # extra camera (the fixed station cameras are unchanged for sensing); rendered on the recorder
    # cadence. Framing (height/FOV/tilt) is tuned from the first recorded run.
    import numpy as _np
    # fm376: SIM-ONLY opt-in to drop the walkthrough_chase render product (a 960x720 RTX camera, a
    # major VRAM consumer on a long run) WITHOUT touching the station sensing cameras or any mission
    # logic. RESTOCK_NO_WALKTHROUGH_RENDER set => no chase camera / no fill light, recording_camera=None
    # (the port self-disables the record cadence); the walkthrough video is then composed offline from
    # the captured sensing frames + telemetry. Unset => byte-identical (chase + fill created as before).
    recording_camera = None
    if not os.environ.get('RESTOCK_NO_WALKTHROUGH_RENDER'):
        from .native_cameras import initialize_camera
        from twin.restock_camera.geometry import Intrinsics
        chase_path = base_root.rstrip('/') + '/walkthrough_chase'
        # fm377: SIM-ONLY env knob for the walkthrough_chase render resolution (the 960x720 RTX product).
        # Default 960x720 => byte-identical; a lower res (e.g. 480x360, quarter pixels) scales fx/fy/cx/cy
        # by the SAME factor to keep the identical FOV/framing, just fewer rendered pixels. Bounded sane.
        def _wt_dim(var, default):   # tolerate unset / empty / malformed => default (never crash boot)
            try:
                return int(os.environ.get(var) or default)
            except (TypeError, ValueError):
                return default
        _cw = max(120, min(960, _wt_dim('RESTOCK_WALKTHROUGH_WIDTH', 960)))
        _ch = max(90, min(720, _wt_dim('RESTOCK_WALKTHROUGH_HEIGHT', 720)))
        _sx, _sy = _cw / 960.0, _ch / 720.0
        chase_k = Intrinsics(width=_cw, height=_ch, fx=430. * _sx, fy=430. * _sy,
                             cx=480. * _sx, cy=360. * _sy, near=.1, far=14.)
        chase = initialize_camera(chase_path, chase_k, name='walkthrough_chase')
        # Prim frame == base frame at identity orientation; a USD camera looks down its own -Z, so an
        # identity-oriented child 2.6 m up the base +Z points straight down at the base. camera_axes='usd'
        # keeps the raw prim-frame convention (no world/ROS optical-axis remap).
        chase.set_local_pose(translation=_np.array([0., 0., 2.6]),
                             orientation=_np.array([1., 0., 0., 0.]), camera_axes='usd')
        # Fill light riding with the chase camera so the overhead walkthrough is LIT (firstmate 222): the
        # store ceiling rig alone leaves the top-down view dark (run-110). A broad downward disk light
        # above the base illuminates the robot, arm, towed cart and the support in frame without touching
        # the sensing cameras' own exposure. UsdLux lights emit along -Z, so an identity-oriented child up
        # the base +Z shines straight down. Intensity tuned from the recorded run.
        from pxr import UsdLux, Gf
        fill_path = base_root.rstrip('/') + '/walkthrough_fill'
        fill = UsdLux.DiskLight.Define(assembly.stage, fill_path)
        fill.CreateIntensityAttr(28000.)
        fill.CreateRadiusAttr(1.6)
        fill.CreateColorAttr(Gf.Vec3f(1., .97, .92))
        UsdGeom.Xformable(fill.GetPrim()).AddTranslateOp().Set(Gf.Vec3d(0., 0., 2.4))
        recording_camera = chase
    return {'cameras': rig.cameras,
            'base_from_camera': lambda camera_id: rig.base_from_camera(camera_id, base_root),
            'recording_camera': recording_camera}


def author_mission_markers(stage, design, parent_paths, *, support_names=('source', 'cart', 'shelf'),
                           _author=None, _write_label=None, _mkdtemp=None):
    """Author the registration (QR) markers for the named mission supports on the compose stage.

    CameraRig authors only camera prims + mount colliders; the printed QR planes the sensing decode
    reads every frame come from `author_camera_design`, which the mission compose otherwise never
    calls -- so a mission capture of a support whose markers were not authored fails "fresh support
    marker(s) not decoded". The collect SOURCE was authored from 2026-10-14 (commit 2cc106c); the
    transfer/verify stages additionally scan the CART (rollwagen) and the SHELF station, so author
    those supports' markers here too. Markers-only per support: cameras + mount-collision solids
    are stripped so we neither duplicate the rig's cameras nor add colliders. A support's `<name>_
    marker_tab` substrate (the witness-patch backing) is kept when it exists -- the SOURCE markers
    float on 'map' and need their printed tab, while the cart/shelf witnesses rest on the physical
    cart deck / shelf tier and need no added tab. Idempotent: a support whose first marker prim
    already exists is skipped. Returns the support names actually authored. Pure wiring behind the
    injectable `_author`/`_write_label`/`_mkdtemp` seams; the real builder runs only in the app.
    """
    import tempfile
    from pathlib import Path as _Path
    author = _author
    write_label = _write_label
    if author is None:
        from twin.restock_camera.config import author_camera_design as author
    if write_label is None:
        from twin.restock_manipulation.isaac_sensors import write_carton_label as write_label
    mkdtemp = _mkdtemp or (lambda: _Path(tempfile.mkdtemp(prefix='restock_markers_')))

    authored = []
    for name in support_names:
        support = design['supports'].get(name)
        if support is None:
            continue
        parent_root = parent_paths.get(support['parent'])
        if parent_root is None:
            raise ValueError(f'support {name!r} parent {support["parent"]!r} not in parent_paths')
        # author_camera_design authors each marker at
        # <parent_root>/RestockCamera_<name>_marker_<index>; skip if already authored.
        if stage.GetPrimAtPath(parent_root + '/RestockCamera_' + name + '_marker_0'):
            continue
        marker_dir = mkdtemp()
        textures = {}
        for marker in support['markers']:
            png = marker_dir / (marker['id'].replace(':', '_') + '.png')
            write_label(marker['id'], png)
            textures[marker['id']] = str(png)
        tab_solids = [s for s in design.get('solids', []) if s['id'] == name + '_marker_tab']
        author(stage, dict(design, cameras={}, solids=tab_solids, supports={name: support}),
               parent_paths=parent_paths, marker_textures=textures)
        authored.append(name)
    return authored
