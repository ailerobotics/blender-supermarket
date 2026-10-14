"""Reference ``sensing_factory`` for the restock mission bundle (camera lane).

Register it in the runtime-config as::

    config['sensing_factory'] = 'twin.restock_camera.sensing:build_sensing'

The integration bundle (`twin.restock.isaac_bundle`) resolves the ``module:function`` and
calls it as ``build_sensing(app=app, assembly=assembly, plan=plan, config=config,
output=output)``, then uses the returned mapping::

    {'capture': capture, 'recording_camera': recording_camera}

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
from .binding import RenderedSupportContext, StationSensors, SupportSlot
from .geometry import Intrinsics

_REQUIRED = ('cameras', 'context_providers', 'location_cameras', 'profiles',
             'pose_bridge_root', 'motion', 'contact', 'recording_camera')


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
                     clearance_for_slot, camera_revision, clock_context=None):
    """A :class:`RenderedSupportContext` for one camera from the design + integration callbacks.

    The intrinsics and depth-error bound come from the camera design; ``measure_context``
    (TF base_from_camera + parked/epoch), ``clearance_for_slot`` (mapper vacancy) and
    ``clock_context`` are integration-owned. The caller commissions it from observed empty
    supports before any mission capture.
    """
    intrinsics = Intrinsics(**camera_config['cameras'][camera_id]['intrinsics'])
    return RenderedSupportContext(
        camera_id=camera_id, intrinsics=intrinsics, slots=slots,
        measure_context=measure_context, clearance_for_slot=clearance_for_slot,
        depth_error_m=camera_config['required_depth_error_bound_m'],
        camera_revision=camera_revision, clock_context=clock_context)


def build_sensing(*, app, assembly, plan, config, output):
    """Assemble the station sensing runtime; return ``{'capture', 'recording_camera'}``.

    ``app``/``plan``/``config``/``output`` are accepted for the bundle factory signature;
    the live sensing pieces come from ``assembly.sensing_sources`` (see the module docstring).
    The returned ``capture`` closes over the integration motion port and contact reader so
    the bundle can call it as ``capture(now, source, destination)``.
    """
    sources = _get_sources(assembly)
    for key in _REQUIRED:
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
    station = StationSensors(
        cameras=_get(sources, 'cameras'), profiles=_get(sources, 'profiles'),
        pose_bridge_root=_get(sources, 'pose_bridge_root'),
        context_providers=context_providers,
        location_cameras=_get(sources, 'location_cameras'))
    motion, contact = _get(sources, 'motion'), _get(sources, 'contact')

    def capture(now, source, destination):
        """Fresh multi-camera inventory scan under the standing gates; never fabricated."""
        return station.capture(now, source, destination, motion, contact)

    return {'capture': capture, 'recording_camera': _get(sources, 'recording_camera')}


def _get_sources(assembly):
    sources = (assembly.get('sensing_sources') if isinstance(assembly, dict)
               else getattr(assembly, 'sensing_sources', None))
    if sources is None:
        raise ValueError('assembly does not expose sensing_sources for the camera lane')
    return sources
