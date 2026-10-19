"""Reference ``sensing_factory`` for the restock mission bundle (camera lane).

Register it in the runtime-config as::

    config['sensing_factory'] = 'twin.restock_camera.sensing:build_sensing'

The integration bundle (`twin.restock.isaac_bundle`) resolves the ``module:function`` and
calls it as ``build_sensing(app=app, assembly=assembly, plan=plan, config=config,
output=output)``, then uses the returned mapping::

    {'capture': capture, 'recording_camera': recording_camera}

When ``config['collect_source_location']`` is set, the mapping also carries
``source_target_provider(now) -> SourceTarget | None`` — the fixed-source collect pick target
(see :mod:`twin.restock_camera.source_target` and the "Collect sensing contract").

When ``config['transfer_shelf_location']`` is set, the mapping also carries
``shelf_target_provider(now) -> ShelfSlotTarget | None`` (the empty-slot placement target) and
``shelf_verification_provider`` → ``verify(now, expected_sku, expected_pose) -> ShelfVerification``
(the post-placement check) — both from the fixed shelf station camera against its commissioned
baseline (see :mod:`twin.restock_camera.shelf_target` and ``data/sim-restock-camera-v2/shelf-contract.md``).

``capture(now, source, destination)`` returns a ``restock_manipulation.types.Sensors``
packet (or ``None``) from the multi-camera :class:`binding.StationSensors`, under the
500 ms freshness, exact-frame identity, source-immutability/tamper and fresh-clearance
gates — it never fabricates an observation and consumes the mapper ``collision_time`` via
the motion port. The ``InventoryVerifier`` then requires two post-boundary confirmations.

This lane owns the geometry/decode wiring: :class:`binding.StationSensors` and the
per-camera :class:`binding.RenderedSupportContext`, whose slots come from the camera
design (:func:`support_slots`). The live cameras, the TF ``measure_context`` and mapper
``clearance_for_slot`` callbacks, the motion port and the contact reader are
integration-owned and supplied through ``assembly.sensing_sources`` (contract below); this
module never imports Isaac/ROS or boots the app.

``assembly.sensing_sources`` (a mapping or an object with these attributes) must provide:

- ``cameras``            ``{camera_id: camera}`` — each exposes ``get_current_frame()`` /
                         ``get_intrinsics_matrix()`` (``native_cameras.RenderedCamera``).
- ``context_providers``  ``{camera_id: RenderedSupportContext}`` — built with
                         :func:`rendered_context` from the design plus integration's
                         ``measure_context``/``clearance_for_slot`` callbacks, and
                         commissioned from observed empty supports before the first capture.
- ``location_cameras``   ``{mission_location: camera_id}`` — e.g. ``{'rollwagen': 'cart',
                         <shelf_id>: 'shelf'}``; every location a slot reports must map here.
- ``profiles``           ``{sku: ProductProfile}``.
- ``pose_bridge_root``   str — the articulation base path for the TF pose bridge.
- ``motion``             the mapper/odometry port: ``parked()``, ``collision_time``,
                         ``joint_positions()``; its ``collision_time`` gates clearance.
- ``contact``            the gripper contact reader: ``read()``.
- ``recording_camera``   the camera used for the mission video.

Boot-time commissioning: if ``config['commissioning_dir']`` names a dataset dir written by
the commissioning choreography (:mod:`twin.restock_camera.commissioning` /
``scripts/run_restock_commissioning.py``), :func:`build_sensing` loads the persisted
empty-support baselines and attaches them to the contexts by station before the first
capture; ``config['commissioning_required_locations']`` (optional) is the set that must be
present or boot fails closed. Without it, integration must commission the contexts
in-process. Baselines are only ever produced by an observed commissioning run; a missing
one is refused, never fabricated.
"""
import math
import os

import numpy as np

from .binding import RenderedSupportContext, StationSensors, SupportSlot
from .geometry import Intrinsics, scale_intrinsics, rigid, transform_points, project

_REQUIRED = ('cameras', 'context_providers', 'location_cameras', 'profiles',
             'pose_bridge_root', 'motion', 'contact', 'recording_camera')


def _parent_frame_quad(center_xy, size_xy, yaw, z):
    """Ordered convex rectangle corners (4,3) at height ``z`` in the support's parent frame."""
    c, s = math.cos(yaw), math.sin(yaw)
    hx, hy = size_xy[0] / 2., size_xy[1] / 2.
    rot = np.array([[c, -s], [s, c]])
    out = []
    for lx, ly in ((-hx, -hy), (hx, -hy), (hx, hy), (-hx, hy)):
        dx, dy = rot @ (lx, ly)
        out.append([center_xy[0] + dx, center_xy[1] + dy, z])
    return np.asarray(out, float)


def commissioning_geometry(camera_config, station, *, world_from_parent, world_from_base,
                           render_scale=1.0):
    """Empty-support commissioning geometry for one station, correct for a world-fixed
    (``parent='map'``) OR a moving-parent (e.g. ``'cart_chassis'``) camera.

    Returns ``{intrinsics, base_from_camera, support_uv, witness_uv, support_world, witness_world}``:
    the camera extrinsic (relative to the arm base) and the empty support + witness quads both in
    the parent frame projected to image UV and in world. ``render_scale`` (<1) projects the UV onto
    the REDUCED render-product grid (:func:`geometry.scale_intrinsics`), so a ``fast`` commissioning
    frame, its selected empty polygons and the scaled context intrinsics all agree; the returned
    ``intrinsics`` is then the scaled calibration. ``render_scale=1.0`` (default) is byte-identical
    to the native design projection. Everything composes through
    ``world_from_parent`` -- the world pose of the camera+support's shared parent frame. For
    ``'map'`` that is identity (and this reduces exactly to the historical source-station inline
    computation); for the cart the caller passes the ``cart_chassis`` prim's world pose read off the
    stage, in the SAME configuration the baseline will be USED in (transfer).

    Why this matters (the silent-corruption guard): the design keys ``parent_from_optical`` and the
    support ``nominal_center``/``witness_design`` to the station's ``parent`` frame. For a moving
    parent, treating those as world (``world_from_optical = parent_from_optical``, as the source-only
    path did) ignores the cart's world pose, so the extrinsic and the projected deck polygons would
    be wrong. Because the camera and its support share the parent, ``support_uv``/``witness_uv`` are
    INVARIANT to ``world_from_parent`` (they cancel -- the camera sees the deck identically wherever
    the cart is), while ``base_from_camera`` reflects the actual configuration. The stored baseline
    is support-local and therefore pose-invariant; the live transfer capture re-registers the deck
    markers against the live ``base_from_camera`` (``sensing_sources`` reads it off the moving cart
    frame), so place targets stay consistent. Pure -- no Isaac/stage access; the caller supplies
    ``world_from_parent`` from the stage.
    """
    cam = camera_config['cameras'][station]
    support = camera_config['supports'][station]
    k = Intrinsics(**cam['intrinsics'])
    k = k if render_scale == 1.0 else scale_intrinsics(k, render_scale)
    wfp = rigid(world_from_parent)
    world_from_optical = wfp @ rigid(cam['parent_from_optical'])
    base_from_camera = np.linalg.inv(rigid(world_from_base)) @ world_from_optical
    center = np.asarray(support['nominal_center'], float)
    req = support['required_xy_m']
    support_parent = _parent_frame_quad(center[:2], (req[0] + .04, req[1] + .04),
                                        float(support['yaw']), float(center[2]))
    wd = support['witness_design']
    wc = np.asarray(wd['center'], float)
    witness_parent = _parent_frame_quad(wc[:2], wd['size_xy'], float(wd['yaw']), float(wc[2]))
    support_world = transform_points(wfp, support_parent)
    witness_world = transform_points(wfp, witness_parent)

    def _uv(points_world):
        uv, valid = project(transform_points(np.linalg.inv(world_from_optical), points_world), k)
        if not valid.all():
            raise ValueError(f'empty-support polygon projects outside the {station!r} commissioning view')
        return uv

    return dict(intrinsics=k, base_from_camera=base_from_camera,
                support_uv=_uv(support_world), witness_uv=_uv(witness_world),
                support_world=support_world, witness_world=witness_world)


def _get(sources, key):
    value = sources.get(key) if isinstance(sources, dict) else getattr(sources, key, None)
    if value is None:
        raise ValueError(f'sensing_sources is missing required binding: {key!r}')
    return value


def support_slots(camera_config, camera_id, *, label_masks, support_locations=None):
    """:class:`SupportSlot` list for one camera from the restock camera design.

    ``camera_config`` is ``restock_camera.config.camera_config(...)`` output. Each support
    assigned to ``camera_id`` becomes a slot whose primary marker registers the support and
    whose second marker (if any) is the auxiliary; the paired ``auxiliary_offset_xy`` and
    the required ``.62x.42`` footprint are carried through unchanged. ``label_masks`` maps a
    mission location to its top-label ROI mask; ``support_locations`` optionally renames a
    design support ('cart', 'shelf', ...) to the mission location ('rollwagen', shelf id).
    """
    support_locations = support_locations or {}
    slots = []
    for name, support in camera_config['supports'].items():
        if support['camera_id'] != camera_id:
            continue
        location = support_locations.get(name, name)
        markers = support['markers']
        if not markers:
            raise ValueError(f'support {name!r} has no registration marker')
        primary = markers[0]
        auxiliary = markers[1] if len(markers) > 1 else None
        slots.append(SupportSlot(
            location=location, support_id=support['id'], marker_id=primary['id'],
            marker_side_m=primary['side_m'], revision=support['revision'],
            label_mask=label_masks[location], required_xy=tuple(support['required_xy_m']),
            auxiliary_id=auxiliary['id'] if auxiliary else None,
            auxiliary_offset_xy=tuple(support['auxiliary_offset_xy']) if auxiliary else None))
    return slots


def rendered_context(camera_config, camera_id, slots, *, measure_context,
                     clearance_for_slot, camera_revision, clock_context=None, render_scale=1.0,
                     occlusion_window_s=20.0):
    """A :class:`RenderedSupportContext` for one camera from the design + integration callbacks.

    The intrinsics and depth-error bound come from the camera design; ``measure_context``
    (TF base_from_camera + parked/epoch), ``clearance_for_slot`` (mapper vacancy) and
    ``clock_context`` are integration-owned. The caller commissions it from observed empty
    supports before any mission capture.

    ``render_scale`` (<1) scales the intrinsics to match a reduced render-product resolution
    (:func:`geometry.scale_intrinsics`) so the context, its commissioned baseline and the live
    capture all use the SAME calibration as the actually-rendered pixels — the fix for
    resolution scaled without intrinsics. Commission at the same scale the mission renders at;
    note decode degrades with scale (small markers), so a decode-grade camera may need 1.0.

    ``occlusion_window_s`` enables approach-occlusion tolerance (see RenderedSupportContext): during
    a grasp the reaching arm occludes the fixed camera's view of the support it grasps toward, so
    the live capture holds the last clean registration (flagged occluded) up to this window instead
    of refusing and starving the host sensor-stream watchdog. ``camera_config['occlusion_window_s']``
    overrides it; set ``None`` to disable (strict every-frame full-footprint). commission()/observe
    stay strict regardless. The default covers a bounded grasp; the held registration still obeys the
    baseline lifetime and per-frame freshness, and a contradicting/lost view still refuses.
    """
    k = Intrinsics(**camera_config['cameras'][camera_id]['intrinsics'])
    intrinsics = k if render_scale == 1.0 else scale_intrinsics(k, render_scale)
    window = camera_config.get('occlusion_window_s', occlusion_window_s)
    return RenderedSupportContext(
        camera_id=camera_id, intrinsics=intrinsics, slots=slots,
        measure_context=measure_context, clearance_for_slot=clearance_for_slot,
        depth_error_m=camera_config['required_depth_error_bound_m'],
        camera_revision=camera_revision, clock_context=clock_context, occlusion_window_s=window,
        render_scale=render_scale)


def build_sensing(*, app, assembly, plan, config, output):
    """Assemble the station sensing runtime; return ``{'capture', 'recording_camera'}``.

    ``app``/``plan``/``config``/``output`` are accepted for the bundle factory signature;
    the live sensing pieces come from ``assembly.sensing_sources`` (see the module docstring).
    The returned ``capture`` closes over the integration motion port and contact reader so
    the bundle can call it as ``capture(now, source, destination)``.
    """
    sources = _get_sources(assembly)
    # fm376: SIM-ONLY opt-in to drop the walkthrough render product (recording_camera). When set, the
    # recording camera is optional (None) -- the bundle skips the Recorder and the port self-disables
    # the record cadence; the walkthrough is composed offline from sensing frames. Unset => required
    # exactly as before (byte-identical).
    _no_walkthrough = bool(os.environ.get('RESTOCK_NO_WALKTHROUGH_RENDER'))
    for key in _REQUIRED:
        if _no_walkthrough and key == 'recording_camera':
            continue
        _get(sources, key)
    context_providers = _get(sources, 'context_providers')
    # Optional boot-time commissioning: if the runtime-config names a commissioning dataset
    # dir, load the persisted empty-support baselines and attach them to the contexts by
    # station (fail closed — a required station with no baseline refuses here, and any
    # uncommissioned slot still refuses at capture). Baselines come only from an observed
    # commissioning run; see twin.restock_camera.commissioning.
    commissioning_dir = config.get('commissioning_dir') if isinstance(config, dict) else None
    if commissioning_dir:
        from .commissioning import load_baselines, apply_to_contexts
        required = config.get('commissioning_required_locations') if isinstance(config, dict) else None
        apply_to_contexts(context_providers, load_baselines(commissioning_dir), required_locations=required)
    # Acquisition-to-use freshness clock: the render-paced mission supplies a SIM/node clock via
    # sensing_sources['steady_clock'] (e.g. the ROS node clock, seconds), so a slow full-resolution
    # render is not misread as stale (a 0.5 s WALL budget false-trips when one capture's render +
    # decode takes 1.7-3.5 s wall under render pacing). A generous wall backstop
    # (config['sensor_wall_backstop_s'], default 60 s) still hard-stops a hung render whose sim clock
    # is frozen. No sim clock supplied => legacy wall behaviour (real hardware, where sim == wall).
    station_kwargs = {}
    sim_clock = (sources.get('steady_clock') if isinstance(sources, dict)
                 else getattr(sources, 'steady_clock', None))
    if sim_clock is not None:
        station_kwargs['steady_clock'] = sim_clock
    if isinstance(config, dict) and config.get('sensor_wall_backstop_s') is not None:
        station_kwargs['wall_backstop_s'] = float(config['sensor_wall_backstop_s'])
    elif os.environ.get('RESTOCK_SENSOR_WALL_BACKSTOP_S') is not None:
        # fm366: when config does not set it, the sim-only env governs the binding sensor-stream wall
        # backstop too (default unset => binding's own 60 s, byte-identical real-rig).
        station_kwargs['wall_backstop_s'] = max(5.0, min(600.0, float(os.environ['RESTOCK_SENSOR_WALL_BACKSTOP_S'])))
    # Opt-in sensing gate (default off): when config['sensing_execution_gate'] is set, the capture
    # path can SUSPEND the decode-camera render during a controller execution phase and republish the
    # held registration (keeps /sensors alive without the render -- the run-126 wall lever). The
    # runtime drives it via the returned 'set_render_suspended' each tick from the controller phase
    # (isaac_sensors.sensing_render_required). Off => every gate is byte-identical to before.
    if isinstance(config, dict) and config.get('sensing_execution_gate'):
        station_kwargs['execution_gate'] = True
    station = StationSensors(
        cameras=_get(sources, 'cameras'), profiles=_get(sources, 'profiles'),
        pose_bridge_root=_get(sources, 'pose_bridge_root'),
        context_providers=context_providers,
        location_cameras=_get(sources, 'location_cameras'), **station_kwargs)
    motion, contact = _get(sources, 'motion'), _get(sources, 'contact')

    def capture(now, source, destination):
        """Fresh multi-camera inventory scan under the standing gates; never fabricated."""
        return station.capture(now, source, destination, motion, contact)

    result = {'capture': capture,
              'recording_camera': None if _no_walkthrough else _get(sources, 'recording_camera'),
              # Integration seam for the opt-in sensing gate: the runtime calls this each tick with
              # ``not sensing_render_required(controller.phase)`` to suspend/resume the decode-camera
              # render. A no-op when config['sensing_execution_gate'] is unset (default).
              'set_render_suspended': station.set_render_suspended}
    # Collect-stage pick target from the FIXED source station camera + its commissioned
    # baseline (the arm eye-in-hand camera is masked-depth/fine-alignment only; it receives
    # a target reprojected from this by integration). See the module docstring and
    # twin.restock_camera.source_target / the "Collect sensing contract".
    source_location = config.get('collect_source_location') if isinstance(config, dict) else None
    if source_location:
        from .source_target import build_source_target_provider
        location_cameras = _get(sources, 'location_cameras')
        if source_location not in location_cameras:
            raise ValueError(f'collect_source_location {source_location!r} has no assigned camera')
        adapter = station.adapters[location_cameras[source_location]]
        result['source_target_provider'] = build_source_target_provider(
            adapter, motion, contact, source_location=source_location,
            station_frame=config.get('station_frame', 'map'))
        # fm384 phase-aware collect capture: a handle the integration calls ONCE at grasp-confirm (behind
        # RESTOCK_PHASE_AWARE_COLLECT_CAPTURE) to (1) switch the capture to destination-only
        # (station.release_source) and (2) DESTROY the now-unused source camera render product, freeing
        # its VRAM before the release peak. Honest: the destination still renders a real fresh frame.
        # Only meaningful when called; unset => never invoked => byte-identical.
        _source_camera_id = location_cameras[source_location]
        def release_source_capture():
            from twin.restock.native_cameras import destroy_camera_render_product
            station.release_source()
            cams = sources.get('cameras') if isinstance(sources, dict) else getattr(sources, 'cameras', {})
            cam = cams.get(_source_camera_id)
            if cam is not None:
                destroy_camera_render_product(cam, getattr(assembly, 'stage', None))
            print('PHASE_AWARE_COLLECT source released + render product destroyed: %s'
                  % _source_camera_id, flush=True)
            return _source_camera_id
        result['release_source_capture'] = release_source_capture
    # Transfer-stage shelf sensing from the FIXED shelf station camera + its commissioned
    # G3_back_s2 baseline: the empty-slot placement target and the post-placement verification,
    # both against the same baseline (see twin.restock_camera.shelf_target / the shelf contract).
    shelf_location = config.get('transfer_shelf_location') if isinstance(config, dict) else None
    if shelf_location:
        from .shelf_target import build_shelf_target_provider, build_shelf_verification_provider
        location_cameras = _get(sources, 'location_cameras')
        if shelf_location not in location_cameras:
            raise ValueError(f'transfer_shelf_location {shelf_location!r} has no assigned camera')
        adapter = station.adapters[location_cameras[shelf_location]]
        station_frame = config.get('station_frame', 'map')
        result['shelf_target_provider'] = build_shelf_target_provider(
            adapter, motion, contact, shelf_location=shelf_location, station_frame=station_frame)
        result['shelf_verification_provider'] = build_shelf_verification_provider(
            adapter, motion, contact, shelf_location=shelf_location, station_frame=station_frame)
    return result


def _get_sources(assembly):
    sources = (assembly.get('sensing_sources') if isinstance(assembly, dict)
               else getattr(assembly, 'sensing_sources', None))
    if sources is None:
        raise ValueError('assembly does not expose sensing_sources for the camera lane')
    return sources
