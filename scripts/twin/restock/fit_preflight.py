"""CPU tray-and-slot fit preflight: honest unsupported cases, off the hot path.

The brief asks the mission to *refuse* a handling unit that will not fit the slot
rather than fail opaquely at the place phase. This module decides, BEFORE any
reservation or robot motion, whether a job's handling unit fits the destination
slot, given a *measured* free-space record. It never drives hardware and imports
no Isaac/ROS/Blender code.

Fit verdict:
  * ``ok``        -- a trustworthy measurement shows the unit fits by the SAME
                     axiswise rule the place-phase gate enforces
                     (:func:`twin.restock_manipulation.fit.axis_fits`).
  * ``wont_fit``  -- a trustworthy measurement shows the slot is occupied/blocked
                     or too small on some axis. Refuse with an honest message.
  * ``unknown``   -- no measurement, a configured-geometry stand-in, or a
                     low-confidence measurement. NEVER silently passes: it is
                     reported with a confidence flag so the caller falls back to
                     configured geometry explicitly, with the place-phase gate as
                     the measured backstop and ``verify`` the stock-commit truth.

The measured record may come from the mission's own depth capture (already the
arm-base ``free_dimensions`` the place gate compares -- the native, mapping-free
path) or from a store-vision ``twin3d`` slot-space export (millimetres in the bay
frame; mapped to the comparison frame only via an EXPLICIT axis order, never a
silent identity).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

from twin.restock_manipulation.fit import PLACEMENT_MARGIN_M, axis_fits

# A measured free-space verdict is only trusted to REFUSE at/above this confidence;
# below it the record is treated as 'unknown' and the caller falls back explicitly.
MIN_CONFIDENCE = 0.5

_AXIS_LABELS = ('x', 'y', 'z')


@dataclass(frozen=True)
class HandlingUnit:
    """A sealed handling unit's grasp-frame footprint (arm-base x, y, z) in metres."""
    sku: str
    dimensions_m: tuple[float, float, float]


@dataclass(frozen=True)
class SlotFreeSpace:
    """Measured free space at a destination slot, in the comparison (arm-base) frame.

    ``free_dimensions_m`` is axis-for-axis with :class:`HandlingUnit.dimensions_m`.
    ``measured`` is False for a configured-geometry stand-in; ``confidence`` (0..1)
    and ``blocked``/``clear`` carry the measurement's own provenance so the verdict
    can refuse only on trustworthy evidence and defer otherwise.
    """
    free_dimensions_m: tuple[float, float, float]
    clear: bool = True
    measured: bool = True
    confidence: float = 1.0
    blocked: bool = False
    source: str = 'measured'
    slot_id: str | None = None


@dataclass(frozen=True)
class FitVerdict:
    status: str                       # 'ok' | 'wont_fit' | 'unknown'
    reasons: tuple[str, ...] = ()
    measured: bool = False
    confidence: float = 0.0
    margin_m: float = PLACEMENT_MARGIN_M
    dimensions_m: tuple[float, float, float] | None = None
    free_dimensions_m: tuple[float, float, float] | None = None
    slot_id: str | None = None
    source: str = ''

    def __bool__(self) -> bool:  # allow `if verdict:` -> only a positive fit
        return self.status == 'ok'

    @property
    def low_confidence(self) -> bool:
        """True for a non-measured / low-confidence verdict a caller must flag."""
        return self.status == 'unknown' or not self.measured or self.confidence < MIN_CONFIDENCE

    @property
    def message(self) -> str:
        detail = '; '.join(self.reasons)
        if self.status == 'ok':
            return 'handling unit fits measured slot free space'
        if self.status == 'wont_fit':
            return detail or 'tray exceeds slot free dimensions'
        return detail or 'slot free space is not measured; falling back to configured geometry'

    def event(self) -> dict:
        """Durable, log-friendly record of the verdict (JSON-safe)."""
        return {
            'fit': self.status,
            'reasons': list(self.reasons),
            'measured': self.measured,
            'confidence': round(float(self.confidence), 4),
            'margin_m': self.margin_m,
            'slot_id': self.slot_id,
            'source': self.source,
            'dimensions_m': list(self.dimensions_m) if self.dimensions_m else None,
            'free_dimensions_m': list(self.free_dimensions_m) if self.free_dimensions_m else None,
            'low_confidence': self.low_confidence,
        }


def evaluate_fit(unit: HandlingUnit, free: SlotFreeSpace | None, *,
                 margin: float = PLACEMENT_MARGIN_M,
                 min_confidence: float = MIN_CONFIDENCE) -> FitVerdict:
    """Decide fit = ok | wont_fit | unknown for one unit against one slot record.

    'unknown' (absent / configured-geometry / low-confidence record) is checked
    first: we only REFUSE on a trustworthy measurement, and otherwise defer to the
    configured-geometry fallback plus the place-phase backstop. A trustworthy
    measurement then refuses an occupied/blocked slot or an axis undersize, else ok.
    """
    dims = tuple(unit.dimensions_m)
    base = dict(dimensions_m=dims, margin_m=margin)
    if free is None:
        return FitVerdict('unknown', ('no measured free-space record for slot',),
                          source='absent', **base)
    free_dims = tuple(free.free_dimensions_m)
    base.update(free_dimensions_m=free_dims, slot_id=free.slot_id,
                measured=free.measured, confidence=float(free.confidence), source=free.source)
    if not free.measured:
        return FitVerdict('unknown',
                          ('free-space record is configured geometry, not a measurement',), **base)
    if not math.isfinite(free.confidence) or free.confidence < min_confidence:
        return FitVerdict('unknown',
                          (f'free-space confidence {free.confidence:.2f} below required '
                           f'{min_confidence:.2f}',), **base)
    if free.blocked or not free.clear:
        return FitVerdict('wont_fit', ('slot occupied or approach blocked (measured)',), **base)
    if not axis_fits(free_dims, dims, margin=margin):
        reasons = []
        if len(free_dims) != 3:
            reasons.append(f'measured free space is not 3 axes: {free_dims}')
        else:
            for label, a, b in zip(_AXIS_LABELS, free_dims, dims):
                if not math.isfinite(a) or a < b + margin:
                    reasons.append(
                        f'tray exceeds slot free dimensions on {label}: free '
                        f'{a * 1000:.0f}mm < unit {b * 1000:.0f}mm + margin {margin * 1000:.0f}mm')
        return FitVerdict('wont_fit', tuple(reasons), **base)
    return FitVerdict('ok', (), **base)


def handling_unit_from_scenario(scenario: dict, *, sku: str | None = None,
                                units: dict | None = None) -> HandlingUnit:
    """Build the in-scope :class:`HandlingUnit` from a mission scenario.

    Tray SKUs in scope default to the scenario's own ``handling_unit``; ``units``
    (a ``{sku: (x, y, z)}`` map) extends/overrides that set, keeping the catalogue
    configurable. ``sku`` selects which unit to preflight (default: the job's SKU).
    """
    catalogue = {k: tuple(float(x) for x in v) for k, v in (units or {}).items()}
    hu = scenario.get('handling_unit') or {}
    if hu.get('sku') and 'dimensions_m' in hu:
        catalogue.setdefault(hu['sku'], tuple(float(x) for x in hu['dimensions_m']))
    target = sku or (scenario.get('job') or {}).get('sku') or hu.get('sku')
    if target not in catalogue:
        raise ValueError(f'no handling-unit geometry in scope for sku {target!r}')
    dims = catalogue[target]
    if len(dims) != 3 or not all(math.isfinite(d) and d > 0 for d in dims):
        raise ValueError(f'handling-unit dimensions must be 3 positive metres: {dims}')
    return HandlingUnit(sku=str(target), dimensions_m=dims)


def free_space_from_depth_capture(free_dimensions_m, *, clear: bool = True,
                                  confidence: float = 1.0, measured: bool = True,
                                  slot_id: str | None = None,
                                  source: str = 'mission_depth_capture') -> SlotFreeSpace:
    """Wrap the mission's own arm-base ``free_dimensions`` (metres) as a record.

    This is the native, mapping-free path: ``free_dimensions`` is already what the
    place-phase gate compares, so no frame mapping is applied or assumed.
    """
    dims = tuple(float(v) for v in free_dimensions_m)
    if len(dims) != 3:
        raise ValueError(f'free_dimensions must be 3 axes in metres: {dims}')
    return SlotFreeSpace(dims, clear=bool(clear), measured=bool(measured),
                         confidence=float(confidence), slot_id=slot_id, source=source)


def free_space_from_twin3d(record: dict, *, slot_id: str | None = None,
                           axis_order=(0, 1, 2)) -> SlotFreeSpace:
    """Map a store-vision ``twin3d`` slot-space record (mm, bay frame) to a record.

    ``record`` carries ``free_width_mm`` / ``free_height_mm`` / ``free_depth_mm``
    plus ``measured`` / ``confidence`` / ``blocked`` (the ``MeasuredSlotSpace``
    shape). The bay frame (width, height, depth) is mapped to the comparison
    (arm-base x, y, z) frame via ``axis_order``: a permutation of (0, 1, 2) where
    comparison axis i takes bay dimension ``axis_order[i]``. There is no sensible
    default correspondence, so the caller MUST supply the mapping for the shelf's
    orientation; a wrong/absent one is not assumed silently -- it is validated.
    """
    try:
        source_mm = (float(record['free_width_mm']), float(record['free_height_mm']),
                     float(record['free_depth_mm']))
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f'twin3d record missing free_*_mm dimensions: {exc}') from exc
    order = tuple(int(i) for i in axis_order)
    if sorted(order) != [0, 1, 2]:
        raise ValueError(f'axis_order must be a permutation of (0,1,2): {axis_order}')
    free_dims = tuple(source_mm[order[i]] / 1000.0 for i in range(3))
    return SlotFreeSpace(
        free_dims,
        clear=not bool(record.get('blocked', False)),
        measured=bool(record.get('measured', True)),
        confidence=float(record.get('confidence', 1.0)),
        blocked=bool(record.get('blocked', False)),
        source=str(record.get('source', 'twin3d')),
        slot_id=slot_id if slot_id is not None else record.get('slot_id'))


def _twin3d_slot(doc: dict, slot_id: str) -> dict:
    """Pull one slot's measured record from a twin3d slot-volumes export document."""
    for slot in doc.get('slots', ()):
        if slot.get('slot_id') == slot_id:
            fill = slot.get('fill') or {}
            # A measured slot space may be recorded inline on the slot or under 'fill'.
            for candidate in (slot, fill):
                if 'free_width_mm' in candidate:
                    return candidate
            break
    raise ValueError(f'twin3d export has no measured free space for slot {slot_id!r}')


def load_free_space(spec: dict | None) -> SlotFreeSpace | None:
    """Interpret a ``fit_preflight`` config block into a :class:`SlotFreeSpace`.

    Shapes (first match wins); any other / absent block yields ``None`` (-> unknown):
      * ``{"free_space": {...SlotFreeSpace fields...}}``       -- a normalized record.
      * ``{"free_dimensions_m": [x, y, z], ...}``              -- mission depth capture.
      * ``{"twin3d_export": "<path>", "slot_id": ...,
           "axis_order": [...]}``                              -- store-vision export.
    """
    if not isinstance(spec, dict):
        return None
    if isinstance(spec.get('free_space'), dict):
        fs = spec['free_space']
        return SlotFreeSpace(
            tuple(float(v) for v in fs['free_dimensions_m']),
            clear=bool(fs.get('clear', True)), measured=bool(fs.get('measured', True)),
            confidence=float(fs.get('confidence', 1.0)), blocked=bool(fs.get('blocked', False)),
            source=str(fs.get('source', 'config')), slot_id=fs.get('slot_id'))
    if 'free_dimensions_m' in spec:
        return free_space_from_depth_capture(
            spec['free_dimensions_m'], clear=bool(spec.get('clear', True)),
            confidence=float(spec.get('confidence', 1.0)),
            measured=bool(spec.get('measured', True)), slot_id=spec.get('slot_id'),
            source=str(spec.get('source', 'mission_depth_capture')))
    if 'twin3d_export' in spec:
        import json
        from pathlib import Path
        doc = json.loads(Path(spec['twin3d_export']).read_text())
        slot_id = spec.get('slot_id')
        record = _twin3d_slot(doc, slot_id) if slot_id else doc
        return free_space_from_twin3d(record, slot_id=slot_id,
                                      axis_order=spec.get('axis_order', (0, 1, 2)))
    return None


def preflight_from_config(scenario: dict, config: dict | None) -> FitVerdict | None:
    """Launcher helper: build the verdict from scenario + a ``config['fit_preflight']``.

    Returns ``None`` when the option is absent, so an off run is byte-identical.
    """
    if not isinstance(config, dict):
        return None
    spec = config.get('fit_preflight')
    if not spec:
        return None
    spec = spec if isinstance(spec, dict) else {}
    unit = handling_unit_from_scenario(scenario, sku=spec.get('sku'), units=spec.get('units'))
    free = load_free_space(spec)
    return evaluate_fit(unit, free, margin=float(spec.get('margin_m', PLACEMENT_MARGIN_M)),
                        min_confidence=float(spec.get('min_confidence', MIN_CONFIDENCE)))
