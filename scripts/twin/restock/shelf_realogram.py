"""sim-smart-verify-v2: the post-transfer shelf observation (realogram) export (CPU).

This builds the *post-transfer shelf observation* the smart-verify v2 step runs
through store-vision ``realogram``/``planoreset`` (see
:mod:`smart_verify_storevision`). It is assembled from a COMPLETED run's artifacts:

* the mission ``verify`` stage evidence (the landed handling unit's SKU + quantity
  at the destination slot -- the only thing the mission verify stage observes), and
* the **resolved verify region** -- the bay the verify stage framed, carried on the
  verify evidence as ``tolerances`` (``bay_center_xy``/``bay_half_xy``/``bay_top_z``)
  and the placed unit's ``ground_truth_check.unit_center_world``. This is the region
  the shelf-staging resolver (``transfer_place_target.resolve_verify_region``, PR 306)
  pinned to the sales-floor endcap, and
* the **saved verify frames** -- the walkthrough frames whose ``simulation_time`` falls
  in the verify window (``frames.json`` in ``mission-frames/``). In the current
  ground-truth verify mode these are the chase-camera frames, NOT a dedicated shelf
  sensor, so they are labelled ``walkthrough_chase_camera`` honestly; a camera verify
  mode (or an out-of-band full-shelf record) is labelled accordingly.

Honesty about evidence is load-bearing and inherited from :mod:`smart_verify`: the
realogram's ``observed`` map is ONLY the landed unit the verify stage confirmed, so
its ``evidence_basis`` is ``mission_verify_sensor`` (a full-shelf multi-SKU scan would
be ``full_shelf_sensor``). The region + frames enrich provenance; they do not invent
observed stock.

Nothing here imports Isaac, ROS, Blender, pydantic or store-vision. It depends only
on :mod:`smart_verify` (the planogram index + the per-slot diff) and the standard
library, so it runs in the plain CPU test environment.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from . import smart_verify as sv

# A slot id (``G3_back_s2`` / ``G1L_b0_s0``) -> its bay fixture id by stripping the
# trailing facing index ``_s<n>`` and the deck index ``<n>``.
_SLOT_RE = re.compile(r'^(?P<bay>.+)_s(?P<deck>\d+)$')

# Frame-set labels: how the saved verify frames were captured. The ground-truth
# verify mode saves no dedicated shelf-sensor frames, so the only frames whose
# sim_time covers the verify window are the walkthrough chase camera -- labelled as
# such so the realogram never overclaims a shelf measurement it did not make.
FRAME_WALKTHROUGH = 'walkthrough_chase_camera'
FRAME_SHELF_SENSOR = 'shelf_sensor'
FRAME_NONE = 'none'


def bay_and_deck(slot_id):
    """``('G3_back', 2)`` for ``'G3_back_s2'``; ``(slot_id, None)`` if no ``_s<n>``."""
    m = _SLOT_RE.match(slot_id)
    if not m:
        return slot_id, None
    return m.group('bay'), int(m.group('deck'))


@dataclass(frozen=True)
class VerifyRegionWorld:
    """The resolved verify region (the bay the verify stage framed), world frame.

    ``center_xy``/``size_xy``/``top_z`` describe the bay footprint; ``source`` records
    where it was read from (``verify_tolerances`` is the bay actually used at verify
    time). ``unit_center_world``/``on_bay``/``tilt_deg`` come from the placed-unit
    ground-truth check when present.
    """
    center_xy: Optional[tuple]
    size_xy: Optional[tuple]
    top_z: Optional[float]
    source: str
    unit_center_world: Optional[tuple] = None
    on_bay: Optional[bool] = None
    tilt_deg: Optional[float] = None

    @property
    def available(self):
        return self.center_xy is not None and self.size_xy is not None

    def store_region(self):
        """``[x, y, w, h]`` footprint (store-vision bay store_region shape), or None."""
        if not self.available:
            return None
        cx, cy = float(self.center_xy[0]), float(self.center_xy[1])
        w, h = float(self.size_xy[0]), float(self.size_xy[1])
        return [cx - w / 2.0, cy - h / 2.0, w, h]

    def to_dict(self):
        return {
            'center_xy': list(self.center_xy) if self.center_xy is not None else None,
            'size_xy': list(self.size_xy) if self.size_xy is not None else None,
            'top_z': self.top_z,
            'source': self.source,
            'store_region_xywh': self.store_region(),
            'unit_center_world': (list(self.unit_center_world)
                                  if self.unit_center_world is not None else None),
            'on_bay': self.on_bay,
            'tilt_deg': self.tilt_deg,
        }


@dataclass(frozen=True)
class FrameRef:
    file: str
    simulation_time: Optional[float]
    sha256: Optional[str]
    camera: str
    path: Optional[str] = None

    def to_dict(self):
        return {'file': self.file, 'simulation_time': self.simulation_time,
                'sha256': self.sha256, 'camera': self.camera, 'path': self.path}


@dataclass(frozen=True)
class ShelfRealogram:
    """A post-transfer realogram for one destination slot, with its honest evidence.

    ``observed`` is the ``{sku: units}`` the mission verify stage confirmed at the
    slot. ``region`` is the resolved verify region; ``frames`` are the saved verify
    frames with their capture label. This is the artifact the store-vision
    ``realogram``/``planoreset`` cross-check consumes (see
    :mod:`smart_verify_storevision`).
    """
    slot_id: str
    bay_id: str
    deck_index: Optional[int]
    observed: dict
    evidence_basis: str
    region: VerifyRegionWorld
    frames: tuple = ()
    frames_basis: str = FRAME_NONE
    confidence: Optional[float] = None
    observation_ids: tuple = ()
    provenance: dict = field(default_factory=dict)

    def to_slot_observation(self):
        """A :class:`smart_verify.SlotObservation` for the planogram diff."""
        return sv.SlotObservation(
            slot_id=self.slot_id, observed=dict(self.observed),
            evidence_basis=self.evidence_basis, confidence=self.confidence,
            observation_ids=self.observation_ids)

    def to_dict(self):
        return {
            'schema': 'smart-verify.shelf-observation.v2',
            'slot_id': self.slot_id,
            'bay_id': self.bay_id,
            'deck_index': self.deck_index,
            'observed': dict(self.observed),
            'evidence_basis': self.evidence_basis,
            'confidence': self.confidence,
            'region': self.region.to_dict(),
            'frames_basis': self.frames_basis,
            'frames': [f.to_dict() for f in self.frames],
            'observation_ids': list(self.observation_ids),
            'provenance': dict(self.provenance),
        }


# --------------------------------------------------------------------------- #
# Resolved verify region                                                       #
# --------------------------------------------------------------------------- #

def extract_verify_region(evidence):
    """Resolve the verify region the verify stage framed, from its evidence.

    Reads ``tolerances`` (the bay actually used: ``bay_center_xy``/``bay_half_xy``/
    ``bay_top_z``) and the placed-unit ``ground_truth_check`` when present. Returns a
    :class:`VerifyRegionWorld`; if the evidence carries no bay geometry the region is
    marked unavailable (``source='unavailable'``) rather than invented.
    """
    tol = evidence.get('tolerances') if isinstance(evidence, dict) else None
    gt = evidence.get('ground_truth_check') if isinstance(evidence, dict) else None
    center = size = top = None
    source = 'unavailable'
    if isinstance(tol, dict) and tol.get('bay_center_xy') is not None:
        c = tol['bay_center_xy']
        center = (float(c[0]), float(c[1]))
        half = tol.get('bay_half_xy')
        if half is not None and len(half) >= 2:
            size = (2.0 * float(half[0]), 2.0 * float(half[1]))
        top = (float(tol['bay_top_z']) if tol.get('bay_top_z') is not None else None)
        source = 'verify_tolerances'
    unit_center = on_bay = tilt = None
    if isinstance(gt, dict):
        uc = gt.get('unit_center_world')
        if uc is not None and len(uc) >= 3:
            unit_center = (float(uc[0]), float(uc[1]), float(uc[2]))
        on_bay = gt.get('on_bay')
        tilt = gt.get('tilt_deg')
        if center is None and unit_center is not None:
            # No bay footprint authored, but we know where the unit landed: frame a
            # minimal region on the unit so the realogram still carries a location.
            center = (unit_center[0], unit_center[1])
            top = unit_center[2]
            source = 'placed_unit_only'
    return VerifyRegionWorld(
        center_xy=center, size_xy=size, top_z=top, source=source,
        unit_center_world=unit_center,
        on_bay=on_bay if isinstance(on_bay, bool) else None,
        tilt_deg=float(tilt) if isinstance(tilt, (int, float)) else None)


# --------------------------------------------------------------------------- #
# Saved verify frames                                                          #
# --------------------------------------------------------------------------- #

def _verify_window(mission, evidence):
    """``(start, end)`` sim-time window of the verify stage, or ``(None, None)``.

    Prefers the evidence's own ``verification_started``..``simulation_time``; else the
    verify stage span in ``stage_timeline``.
    """
    start = end = None
    if isinstance(evidence, dict):
        s = evidence.get('verification_started')
        e = evidence.get('simulation_time')
        if isinstance(s, (int, float)):
            start = float(s)
        if isinstance(e, (int, float)):
            end = float(e)
    if start is not None and end is not None:
        return (min(start, end), max(start, end))
    # fall back to the verify stage span in the timeline
    begin = None
    for ev in (mission.get('stage_timeline') or []) if isinstance(mission, dict) else []:
        data = ev.get('data', {}) if isinstance(ev, dict) else {}
        if data.get('stage') != 'verify':
            continue
        st = data.get('sim_time')
        if not isinstance(st, (int, float)):
            continue
        if ev.get('type') == 'stage_begin':
            begin = float(st)
        end = float(st)
    if begin is not None:
        return (begin, end if end is not None else begin)
    return (None, None)


def load_frames_manifest(run_dir):
    """Parse ``mission-frames/frames.json`` into its ``frames`` list, or ``[]``."""
    if run_dir is None:
        return []
    path = Path(run_dir) / 'mission-frames' / 'frames.json'
    if not path.is_file():
        return []
    try:
        data = json.loads(path.read_text())
    except (OSError, ValueError):
        return []
    frames = data.get('frames') if isinstance(data, dict) else data
    return frames if isinstance(frames, list) else []


def locate_verify_frames(run_dir, mission, evidence, *, camera=FRAME_WALKTHROUGH,
                         max_frames=8, pad_s=2.0):
    """Saved frames whose ``simulation_time`` falls in (a padded) verify window.

    Returns ``(frames, basis)``: ``frames`` is a list of :class:`FrameRef` (at most
    ``max_frames``, nearest the verify end last), ``basis`` the capture label. The
    ground-truth verify stage is near-instantaneous (a 2-frame ground-truth check), so
    the window is padded by ``pad_s`` seconds either side to capture the frames the
    chase camera recorded of the shelf AROUND the placement/verify. In that mode these
    are walkthrough chase-camera frames, NOT a dedicated shelf sensor, so the default
    ``camera`` labels them honestly. If no run dir / no manifest / no window overlap,
    returns ``([], FRAME_NONE)``.
    """
    manifest = load_frames_manifest(run_dir)
    if not manifest:
        return [], FRAME_NONE
    lo, hi = _verify_window(mission, evidence)
    if lo is None:
        return [], FRAME_NONE
    lo, hi = lo - max(0.0, pad_s), hi + max(0.0, pad_s)
    frames_dir = Path(run_dir) / 'mission-frames'
    hits = []
    for fr in manifest:
        if not isinstance(fr, dict):
            continue
        st = fr.get('simulation_time')
        if not isinstance(st, (int, float)):
            continue
        if lo <= float(st) <= hi:
            file = fr.get('file')
            hits.append(FrameRef(
                file=file, simulation_time=float(st), sha256=fr.get('sha256'),
                camera=camera,
                path=str(frames_dir / file) if file else None))
    hits.sort(key=lambda f: f.simulation_time if f.simulation_time is not None else 0.0)
    if max_frames and len(hits) > max_frames:
        hits = hits[-max_frames:]
    return hits, (camera if hits else FRAME_NONE)


# --------------------------------------------------------------------------- #
# Build the shelf observation                                                  #
# --------------------------------------------------------------------------- #

def build_shelf_realogram(mission, evidence, slot_id, *, run_dir=None,
                          shelf_record=None, extra_observed=None):
    """Assemble the post-transfer :class:`ShelfRealogram` for ``slot_id``.

    * ``shelf_record`` (an out-of-band full-shelf labelled observation, the
      :func:`smart_verify.observation_from_shelf_record` shape) -> ``full_shelf_sensor``
      basis, observed = its map; else
    * the mission verify evidence -> ``mission_verify_sensor`` basis, observed = the
      single landed unit the verify stage confirmed.

    ``extra_observed`` (optional ``{sku: units}``, e.g. a bookkeeping belief of other
    stock on the shelf) is merged in and recorded in provenance as a weaker belief;
    it does not change the evidence basis label of the *verified* unit.
    """
    region = extract_verify_region(evidence)
    bay_id, deck = bay_and_deck(slot_id)
    frames, frames_basis = ([], FRAME_NONE)
    if shelf_record is not None:
        base_obs = sv.observation_from_shelf_record(shelf_record)
        frames_basis = FRAME_SHELF_SENSOR
    else:
        base_obs = sv.observation_from_verify(evidence, slot_id)
        frames, frames_basis = locate_verify_frames(run_dir, mission, evidence)
    observed = dict(base_obs.observed)
    merged_note = None
    if extra_observed:
        merged_note = {k: int(v) for k, v in extra_observed.items() if int(v) > 0}
        for k, v in merged_note.items():
            observed[k] = observed.get(k, 0) + int(v)
    prov = {
        'emitted_by': 'sim-smart-verify-v2',
        'verify_source': (evidence.get('source') if isinstance(evidence, dict) else None),
        'verify_mode': (evidence.get('verify_mode') if isinstance(evidence, dict) else None),
        'observation_id': (evidence.get('observation_id') if isinstance(evidence, dict) else None),
        'region_source': region.source,
    }
    if merged_note:
        prov['extra_observed_bookkeeping'] = merged_note
    return ShelfRealogram(
        slot_id=slot_id, bay_id=bay_id, deck_index=deck, observed=observed,
        evidence_basis=base_obs.evidence_basis, region=region,
        frames=tuple(frames), frames_basis=frames_basis,
        confidence=base_obs.confidence, observation_ids=base_obs.observation_ids,
        provenance=prov)
