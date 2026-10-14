"""Commissioning choreography for empty-support baselines (camera lane).

No Isaac/ROS/USD imports: pure NumPy + stdlib, so it runs and tests on CPU. A
baseline is ONLY ever produced by an observed ``RenderedSupportContext.commission``
on a real empty-support RGB-D frame (``sensed.calibrate_rigid_support`` refuses an
occupied, tilted or occluded patch). This module persists *exactly* what that
commission measured, with provenance (session id, station, capture pose, stamp),
and reloads it at mission boot by reconstructing the ``RegisteredSupport`` and
attaching it to the matching slot. A missing baseline is refused, never fabricated:
geometry/config never manufacture an empty observation.

Lifecycle:
    commission_and_persist(context, frame, now, empty_polygons, dataset_dir=...,
                           capture_poses=...)   # a commissioning run, per station
    records = load_baselines(dataset_dir)                      # at mission boot
    apply_baselines(context, records, required_locations=[...])# attach or refuse

The persisted JSON is the loadable baseline; an optional ``.npz`` sidecar keeps the
raw commissioning RGB-D frame + selected polygons so a reviewer can re-run
``calibrate_rigid_support`` and reproduce the baseline (audit against fabrication).
"""
from dataclasses import asdict, dataclass
import json
import time
from pathlib import Path

import numpy as np

from .geometry import Intrinsics
from .sensed import SupportGeometry, RegisteredSupport

SCHEMA = 'restock-commissioning-baseline.v1'
MANIFEST = 'commissioning-manifest.json'
SOURCE = 'commissioned_empty_support'


class CommissioningBaselineMissing(ValueError):
    """No commissioned empty-support baseline exists for a required station."""


class CommissioningBaselineInvalid(ValueError):
    """A persisted baseline is malformed or does not match the live slot/camera."""


_TUPLE_GEOMETRY_FIELDS = ('polygon_support', 'witness_support', 'center_support', 'required_xy')


def _tuples(value):
    """Restore JSON lists to the nested tuples the frozen geometry carries."""
    if isinstance(value, (list, tuple)):
        return tuple(_tuples(v) for v in value)
    return value


def geometry_to_json(geometry):
    """Plain-JSON mapping for a :class:`sensed.SupportGeometry` (lossless).

    ``asdict`` recurses into the nested ``Intrinsics`` dataclass already, so the
    result is pure dict/list/scalar and round-trips through JSON.
    """
    return asdict(geometry)


def geometry_from_json(data):
    """Reconstruct a :class:`sensed.SupportGeometry` from :func:`geometry_to_json`."""
    fields = dict(data)
    intr = fields['intrinsics']
    fields['intrinsics'] = Intrinsics(**intr) if isinstance(intr, dict) else intr
    for key in _TUPLE_GEOMETRY_FIELDS:
        fields[key] = _tuples(fields[key])
    # baseline_acquired_steady stays None or float; frame_id must survive as int.
    fields['baseline_frame_id'] = int(fields['baseline_frame_id'])
    return SupportGeometry(**fields)


@dataclass(frozen=True)
class BaselineRecord:
    """One commissioned station baseline plus the provenance a mission reloads by.

    ``geometry`` is the loadable artifact (what ``RegisteredSupport`` wraps). The
    provenance mirrors the ``isaac_sensors`` capture-evidence vocabulary so an audit
    can line a reloaded baseline up with the capture that produced it.
    """
    station: str
    geometry: SupportGeometry
    capture_pose: tuple | None = None          # robot base planar pose [x,y,yaw] at capture
    base_from_camera: tuple | None = None       # measured camera transform at capture (audit)
    commissioning_freshness_s: float = 5.       # acquisition-to-use bound used at commission
    render_scale: float = 1.                     # render-product scale the baseline was commissioned at
    source: str = SOURCE
    schema: str = SCHEMA

    @property
    def camera_id(self):
        return self.geometry.camera_id

    @property
    def support_id(self):
        return self.geometry.support_id

    def provenance(self):
        g = self.geometry
        return dict(session_id=g.baseline_session_id, station=self.station,
                    support_id=g.support_id, camera_id=g.camera_id,
                    camera_revision=g.camera_revision, marker_id=g.marker_id,
                    revision=g.revision, calibration_id=g.calibration_id,
                    baseline_frame_id=g.baseline_frame_id, baseline_stamp=g.baseline_stamp,
                    steady_domain=g.steady_domain, baseline_acquired_steady=g.baseline_acquired_steady,
                    lifetime_s=g.lifetime_s, commissioning_freshness_s=self.commissioning_freshness_s,
                    render_scale=self.render_scale, rendered_resolution=[g.intrinsics.width, g.intrinsics.height],
                    capture_pose=self.capture_pose, source=self.source)

    def to_json(self):
        return dict(schema=self.schema, source=self.source, station=self.station,
                    capture_pose=None if self.capture_pose is None else list(self.capture_pose),
                    base_from_camera=None if self.base_from_camera is None else
                    [list(r) for r in self.base_from_camera],
                    commissioning_freshness_s=self.commissioning_freshness_s,
                    render_scale=self.render_scale,
                    provenance=self.provenance(), geometry=geometry_to_json(self.geometry))

    @classmethod
    def from_json(cls, data):
        if not isinstance(data, dict) or data.get('schema') != SCHEMA:
            raise CommissioningBaselineInvalid(f'unknown commissioning baseline schema: {data.get("schema")!r}')
        if data.get('source') != SOURCE:
            raise CommissioningBaselineInvalid('baseline is not a commissioned empty-support observation')
        geometry = geometry_from_json(data['geometry'])
        # Provenance must match the geometry it describes; a hand-edited provenance
        # block can never substitute for the measured baseline it claims to stamp.
        prov = data.get('provenance', {})
        if (prov.get('calibration_id') != geometry.calibration_id
                or prov.get('baseline_stamp') != geometry.baseline_stamp
                or prov.get('session_id') != geometry.baseline_session_id
                or prov.get('station') != data['station']):
            raise CommissioningBaselineInvalid('baseline provenance does not match its geometry')
        pose = data.get('capture_pose')
        bfc = data.get('base_from_camera')
        return cls(station=data['station'], geometry=geometry,
                   capture_pose=None if pose is None else tuple(pose),
                   base_from_camera=None if bfc is None else tuple(tuple(r) for r in bfc),
                   commissioning_freshness_s=data.get('commissioning_freshness_s', 5.),
                   render_scale=data.get('render_scale', 1.),
                   source=data['source'], schema=data['schema'])


def _validate_measured(geometry):
    """Refuse to persist anything without genuine acquisition provenance."""
    if (not geometry.calibration_id or not geometry.baseline_session_id
            or type(geometry.baseline_frame_id) is not int or geometry.baseline_frame_id < 0
            or not np.isfinite(geometry.baseline_stamp)):
        raise CommissioningBaselineInvalid('baseline lacks genuine commissioning provenance; refusing to persist')
    if bool(geometry.steady_domain) != (geometry.baseline_acquired_steady is not None):
        raise CommissioningBaselineInvalid('persistent acquisition clock identity and time required together')


def persist_baseline(dataset_dir, record, *, raw_frame=None):
    """Write one station baseline JSON (and optional raw-frame npz) under ``dataset_dir``.

    ``raw_frame`` (optional) = ``dict(rgb, depth, support_uv, witness_uv[, base_from_camera])``
    from the commissioning capture; persisted as an audit sidecar, not required to reload.
    Returns the written JSON path.
    """
    _validate_measured(record.geometry)
    directory = Path(dataset_dir)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f'{record.station}.json'
    path.write_text(json.dumps(record.to_json(), indent=1, sort_keys=True))
    if raw_frame is not None:
        np.savez_compressed(directory / f'{record.station}.npz',
            rgb=np.asarray(raw_frame['rgb']), depth=np.asarray(raw_frame['depth'], float),
            support_uv=np.asarray(raw_frame['support_uv'], float),
            witness_uv=np.asarray(raw_frame['witness_uv'], float),
            base_from_camera=np.asarray(raw_frame.get('base_from_camera', np.eye(4)), float))
    return path


def rebuild_manifest(dataset_dir, *, session_id=None, extra=None):
    """Rewrite the manifest as the union of every station baseline present in the dir.

    Scans the persisted ``<station>.json`` files (so successive commissioning passes for
    different cameras accumulate into one manifest) and records each station's provenance.
    """
    directory = Path(dataset_dir)
    directory.mkdir(parents=True, exist_ok=True)
    stations, provenance, sessions = {}, {}, set()
    for path in sorted(directory.glob('*.json')):
        if path.name == MANIFEST:
            continue
        record = BaselineRecord.from_json(json.loads(path.read_text()))
        stations[record.station] = path.name
        provenance[record.station] = record.provenance()
        sessions.add(record.geometry.baseline_session_id)
    manifest = dict(schema=SCHEMA, source=SOURCE, written=time.time(),
                    session_id=session_id or (sorted(sessions)[0] if len(sessions) == 1 else 'multiple'),
                    session_ids=sorted(sessions), stations=stations, provenance=provenance,
                    invariant='baselines come only from observed empty-support commissioning; '
                              'a missing station is refused, never fabricated')
    if extra:
        manifest.update(extra)
    path = directory / MANIFEST
    path.write_text(json.dumps(manifest, indent=1, sort_keys=True))
    return path


def commission_and_persist(context, frame, now, empty_polygons, *, dataset_dir,
                           capture_poses=None, lifetime_s=600., commissioning_freshness_s=5.,
                           render_scale=1., raw_frames=None):
    """Run ``context.commission`` on an observed empty frame and persist each baseline.

    ``empty_polygons[location] = (support_quad_uv, witness_quad_uv)`` selected from this
    measured empty frame (exactly the ``commission`` contract). ``capture_poses[location]``
    is the robot base pose used for the capture (recorded as provenance). ``raw_frames``
    optionally maps location -> raw-frame dict for the npz audit sidecar. Returns the list
    of :class:`BaselineRecord`.
    """
    capture_poses = capture_poses or {}
    raw_frames = raw_frames or {}
    context.commission(frame, now, empty_polygons, lifetime_s=lifetime_s,
                       commissioning_freshness_s=commissioning_freshness_s)
    records = []
    for slot in context.slots:
        tracker = slot.tracker
        if tracker is None:                       # commission did not cover this slot
            continue
        pose = capture_poses.get(slot.location)
        record = BaselineRecord(station=slot.location, geometry=tracker.geometry,
                                capture_pose=None if pose is None else tuple(pose),
                                commissioning_freshness_s=commissioning_freshness_s,
                                render_scale=render_scale)
        persist_baseline(dataset_dir, record, raw_frame=raw_frames.get(slot.location))
        records.append(record)
    rebuild_manifest(dataset_dir)
    return records


def run_commissioning(contexts, capture, *, dataset_dir, lifetime_s=600., commissioning_freshness_s=5.,
                      render_scale=1.):
    """Commission a set of per-camera contexts from observed empty supports and persist.

    ``contexts`` is an iterable of :class:`binding.RenderedSupportContext` (one per camera,
    each covering one or more station slots). ``capture(context)`` is the caller-owned
    choreography step that, with the station(s) EMPTY and the robot at the capture pose,
    returns a dict for that camera::

        {'frame': {...RGB-D...}, 'now': float,
         'empty_polygons': {location: (support_quad_uv, witness_quad_uv)},
         'capture_poses': {location: [x,y,yaw]}, 'raw_frames': {location: {...}}}  # last two optional

    This keeps all Isaac/robot motion and polygon selection in the caller (integration
    natively, a fixture in tests); the choreography here only commissions + persists.
    Returns the full list of :class:`BaselineRecord`.
    """
    records = []
    for context in contexts:
        got = capture(context)
        records.extend(commission_and_persist(
            context, got['frame'], got['now'], got['empty_polygons'], dataset_dir=dataset_dir,
            capture_poses=got.get('capture_poses'), raw_frames=got.get('raw_frames'),
            lifetime_s=lifetime_s, commissioning_freshness_s=commissioning_freshness_s,
            render_scale=render_scale))
    rebuild_manifest(dataset_dir)
    return records


def load_baselines(dataset_dir):
    """Load every commissioned baseline under ``dataset_dir`` as ``{station: BaselineRecord}``.

    Reads the manifest, then each station file. Raises :class:`CommissioningBaselineMissing`
    if the directory has never been commissioned (no manifest).
    """
    directory = Path(dataset_dir)
    manifest_path = directory / MANIFEST
    if not manifest_path.is_file():
        raise CommissioningBaselineMissing(f'no commissioning manifest under {dataset_dir!r}; '
                                           'run the commissioning choreography first')
    manifest = json.loads(manifest_path.read_text())
    if manifest.get('schema') != SCHEMA:
        raise CommissioningBaselineInvalid(f'unknown commissioning manifest schema: {manifest.get("schema")!r}')
    records = {}
    for station, name in manifest.get('stations', {}).items():
        path = directory / name
        if not path.is_file():
            raise CommissioningBaselineInvalid(f'manifest references missing baseline file: {name!r}')
        record = BaselineRecord.from_json(json.loads(path.read_text()))
        if record.station != station:
            raise CommissioningBaselineInvalid(f'baseline file {name!r} is for station {record.station!r}, not {station!r}')
        records[station] = record
    return records


def load_baseline(dataset_dir, station):
    """Load one station's baseline, refusing clearly when it was never commissioned."""
    records = load_baselines(dataset_dir)
    if station not in records:
        raise CommissioningBaselineMissing(
            f'station {station!r} has no commissioned empty-support baseline under {dataset_dir!r}; '
            f'commissioned stations: {sorted(records)}')
    return records[station]


def _check_match(context, slot, geometry):
    """Refuse a baseline that does not describe this exact slot/camera."""
    if (geometry.support_id != slot.support_id or geometry.marker_id != slot.marker_id
            or geometry.revision != slot.revision):
        raise CommissioningBaselineInvalid(
            f'baseline for {slot.location!r} describes support '
            f'{geometry.support_id!r}/{geometry.marker_id!r}/{geometry.revision!r}, '
            f'not the slot {slot.support_id!r}/{slot.marker_id!r}/{slot.revision!r}')
    if (geometry.camera_id != context.camera_id or geometry.camera_revision != context.camera_revision
            or geometry.intrinsics != context.intrinsics):
        raise CommissioningBaselineInvalid(
            f'baseline for {slot.location!r} was commissioned on camera '
            f'{geometry.camera_id!r}/{geometry.camera_revision!r}, not {context.camera_id!r}/{context.camera_revision!r}')


def apply_baselines(context, records, *, required_locations=None):
    """Attach commissioned baselines to a context's slots by station; refuse any missing.

    For every slot in ``context``, look up ``records[slot.location]``, verify it was
    commissioned for exactly this support and camera, reconstruct a
    :class:`sensed.RegisteredSupport` and set ``slot.tracker`` (the field
    ``RenderedSupportContext.__call__`` checks). A slot whose station is absent raises
    :class:`CommissioningBaselineMissing` — the mission fails closed at boot rather than
    capturing without a baseline. ``required_locations`` (default: every slot's location)
    is the set that must be present. Returns the list of attached station locations.
    """
    required = set(required_locations) if required_locations is not None else {s.location for s in context.slots}
    attached = []
    for slot in context.slots:
        if slot.location not in required:
            continue
        record = records.get(slot.location)
        if record is None:
            raise CommissioningBaselineMissing(
                f'station {slot.location!r} has no commissioned empty-support baseline; '
                f'commission it before the mission (available: {sorted(records)})')
        _check_match(context, slot, record.geometry)
        slot.tracker = RegisteredSupport(record.geometry)
        attached.append(slot.location)
    missing = required - set(attached)
    if missing:
        raise CommissioningBaselineMissing(f'context has no slot for required station(s): {sorted(missing)}')
    return attached


def apply_to_contexts(context_providers, records, *, required_locations=None):
    """Attach baselines across several per-camera contexts at mission boot.

    For every slot in every context, attach its station's baseline when one exists
    (verifying support/camera match); slots with no baseline keep ``tracker=None`` and
    fail closed at capture time. ``required_locations`` (if given) must all be attached —
    a required station with no commissioned baseline raises
    :class:`CommissioningBaselineMissing`. Returns the attached station locations.
    """
    contexts = list(context_providers.values()) if isinstance(context_providers, dict) else list(context_providers)
    attached = []
    for context in contexts:
        for slot in context.slots:
            record = records.get(slot.location)
            if record is None:
                continue
            _check_match(context, slot, record.geometry)
            slot.tracker = RegisteredSupport(record.geometry)
            attached.append(slot.location)
    if required_locations is not None:
        missing = set(required_locations) - set(attached)
        if missing:
            raise CommissioningBaselineMissing(
                f'required station(s) have no commissioned empty-support baseline: {sorted(missing)} '
                f'(attached: {sorted(set(attached))})')
    return attached
