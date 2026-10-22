"""Smart post-restock verification: planogram/realogram compliance (CPU, opt-in).

This is an ADDITIONAL report layered on top of the mission's ``verify`` stage. It
NEVER changes the mission's stock-commit gate: the mission ``verify`` stage stays
the source of truth for committing stock (see :mod:`inventory` ``verify``). After
``verify`` succeeds this module diffs the resulting shelf realogram against the
store planogram and emits a compliance verdict plus any residual reset jobs that
would bring the slot back to the planogram -- the closed-loop analogue of
store-vision ``realogram``/``planoreset``.

Honesty about evidence basis is load-bearing. Three realogram bases are possible,
each labelled on every slot verdict:

* ``mission_verify_sensor`` -- only the just-placed handling unit was scanned at
  the destination slot by the mission ``verify`` stage. This is NOT a full-shelf
  multi-SKU scan; it confirms identity+quantity of the landed unit only.
* ``bookkeeping`` -- the transactional inventory DB's committed counts at the
  slot. A belief derived from reservations+commits, not a fresh sensor scan.
* ``full_shelf_sensor`` -- a complete labelled shelf observation supplied out of
  band (e.g. a store-vision ``realogram`` export or a scenario labelled view).

Nothing here imports Isaac, ROS, Blender or store-vision. The optional
store-vision cross-check lives in :mod:`smart_verify_storevision` and is a
lazy, best-effort add-on that degrades to this built-in diff when store-vision
is not importable.
"""
from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

# Quantity policies for judging a slot compliant against the planogram target.
QUANTITY_POLICIES = ('at_least', 'exact', 'identity')

# Evidence bases, strongest first.
EVIDENCE_BASES = ('full_shelf_sensor', 'mission_verify_sensor', 'bookkeeping')


@dataclass(frozen=True)
class SlotExpectation:
    """Planogram target for a slot: aggregated units per SKU."""
    slot_id: str
    expected: dict  # {sku: units}
    surface_z: Optional[float] = None


@dataclass(frozen=True)
class SlotObservation:
    """A realogram for one slot with its honest evidence basis."""
    slot_id: str
    observed: dict  # {sku: units}
    evidence_basis: str
    confidence: Optional[float] = None
    observation_ids: tuple = ()

    def __post_init__(self):
        if self.evidence_basis not in EVIDENCE_BASES:
            raise ValueError(f'evidence_basis must be one of {EVIDENCE_BASES}')


@dataclass(frozen=True)
class SlotDiff:
    sku: str
    expected: int
    observed: int
    delta: int   # expected - observed (>0 shortfall, <0 overstock)
    kind: str    # compliant | shortfall | overstock | misplaced


@dataclass(frozen=True)
class ResetJob:
    """Residual reset job in the sim-smart-job-feed-v1 ``--job-from`` contract.

    The feed adapter maps these JSON fields onto the mission :class:`Job`
    (job_id/sku/quantity/source_bin/shelf_id). ``kind`` distinguishes a top-up
    ``restock`` from a ``remove`` of a misplaced/overstocked SKU; ``provenance``
    records that smart-verify emitted it and why.
    """
    sku: str
    quantity: int
    shelf_id: str
    kind: str                     # restock | remove
    priority: int
    source_bin: Optional[str] = None
    provenance: dict = field(default_factory=dict)

    def to_job_feed(self):
        """JSON dict in the job-feed ``--job-from`` shape (source cell + slot)."""
        return {
            'sku': self.sku,
            'quantity': self.quantity,
            'source_bin': self.source_bin,
            'shelf_id': self.shelf_id,
            'slot': self.shelf_id,
            'priority': self.priority,
            'kind': self.kind,
            'provenance': dict(self.provenance),
        }


@dataclass(frozen=True)
class SlotVerdict:
    slot_id: str
    compliant: bool
    evidence_basis: str
    expected: dict
    observed: dict
    diffs: list
    confidence: Optional[float] = None

    def to_dict(self):
        return {
            'slot_id': self.slot_id,
            'compliant': self.compliant,
            'evidence_basis': self.evidence_basis,
            'confidence': self.confidence,
            'expected': dict(self.expected),
            'observed': dict(self.observed),
            'diffs': [vars(d) for d in self.diffs],
        }


@dataclass(frozen=True)
class SmartVerifyReport:
    job_id: str
    compliant: bool
    evidence_basis: str
    slots: list
    residual_jobs: list
    authority_note: str
    provenance: dict = field(default_factory=dict)

    def to_dict(self):
        return {
            'schema': 'smart-verify.v1',
            'job_id': self.job_id,
            'compliant': self.compliant,
            'evidence_basis': self.evidence_basis,
            'authority_note': self.authority_note,
            'slots': [s.to_dict() for s in self.slots],
            'residual_jobs': [j.to_job_feed() for j in self.residual_jobs],
            'provenance': dict(self.provenance),
        }


AUTHORITY_NOTE = (
    'Mission verify stage is the stock-commit source of truth; this planogram '
    'compliance report is additional and never gates the mission.'
)


# --------------------------------------------------------------------------- #
# Planogram slot resolution (mirrors scripts/store_environment.py id scheme)   #
# --------------------------------------------------------------------------- #

def _aggregate_items(items):
    """[[sku, count], ...] (per-facing counts) -> {sku: total_units}."""
    agg = {}
    for sku, count in items:
        if not isinstance(sku, str) or not sku:
            raise ValueError(f'planogram item has invalid sku: {sku!r}')
        if type(count) is not int or count < 0:
            raise ValueError(f'planogram item {sku!r} has invalid count: {count!r}')
        agg[sku] = agg.get(sku, 0) + count
    return agg


def planogram_slot_index(plan):
    """Build ``{slot_id: SlotExpectation}`` from a planogram.

    The slot-id scheme mirrors ``scripts/store_environment.py`` exactly:

    * gondola bay shelves -> ``f"{g_id}{side}_b{bay_index}_s{i}"``
    * the ``G3_back`` endcap shelves -> ``f"G3_back_s{i}"``

    A consistency test asserts these ids match ``store_environment.layout`` so
    the two can never silently drift.
    """
    index = {}
    for g in plan.get('gondolas', []):
        for side in ('L', 'R'):
            side_cfg = g.get('sides', {}).get(side)
            if not side_cfg:
                continue
            for bay in side_cfg.get('bays', []):
                for i, shelf in enumerate(bay.get('shelves', [])):
                    slot_id = f"{g['id']}{side}_b{bay['index']}_s{i}"
                    index[slot_id] = SlotExpectation(
                        slot_id=slot_id,
                        expected=_aggregate_items(shelf.get('items', [])),
                        surface_z=shelf.get('z'))
    for endcap in plan.get('endcaps', []):
        for i, shelf in enumerate(endcap.get('shelves', [])):
            slot_id = f"{endcap['id']}_s{i}"
            index[slot_id] = SlotExpectation(
                slot_id=slot_id,
                expected=_aggregate_items(shelf.get('items', [])),
                surface_z=shelf.get('z'))
    return index


def slot_expectation(plan, slot_id):
    index = planogram_slot_index(plan)
    if slot_id not in index:
        raise KeyError(f'slot {slot_id!r} is not in the planogram')
    return index[slot_id]


# --------------------------------------------------------------------------- #
# Observations                                                                 #
# --------------------------------------------------------------------------- #

def mission_verify_evidence(mission):
    """Extract the verify-stage sensor evidence from a parsed mission.json.

    Returns the ``inventory_verified`` (preferred) or ``verify_succeeded`` event
    evidence dict, or ``None`` if verify did not succeed. Both carry the same
    post-transfer destination-identity observation produced by the mission's
    independent :class:`InventoryVerifier`.
    """
    events = mission.get('events', [])
    chosen = None
    for ev in events:
        if ev.get('state') in ('inventory_verified', 'verify_succeeded'):
            chosen = ev.get('evidence') or {}
    return chosen


def observation_from_verify(evidence, slot_id):
    """A ``mission_verify_sensor`` realogram: ONLY the landed unit was scanned.

    This honestly reflects that the mission verify stage scanned the destination
    slot's just-placed handling unit, not the whole shelf.
    """
    sku = evidence.get('sku')
    qty = evidence.get('quantity')
    if not isinstance(sku, str) or type(qty) is not int:
        raise ValueError('verify evidence missing sku/quantity')
    loc = evidence.get('location')
    if loc is not None and loc != slot_id:
        raise ValueError(f'verify evidence location {loc!r} != slot {slot_id!r}')
    obs_ids = tuple(evidence.get('confirming_observation_ids')
                    or ([evidence['observation_id']] if evidence.get('observation_id') else []))
    conf = evidence.get('confidence')
    return SlotObservation(slot_id=slot_id, observed={sku: qty},
                           evidence_basis='mission_verify_sensor',
                           confidence=conf if isinstance(conf, (int, float)) else None,
                           observation_ids=obs_ids)


def observation_from_bookkeeping(stock, slot_id):
    """A ``bookkeeping`` realogram from a ``{(location, sku): units}`` snapshot."""
    observed = {sku: qty for (loc, sku), qty in stock.items()
                if loc == slot_id and qty > 0}
    return SlotObservation(slot_id=slot_id, observed=observed,
                           evidence_basis='bookkeeping')


def observation_from_shelf_record(record):
    """A ``full_shelf_sensor`` realogram from an external labelled shelf view.

    ``record`` shape: ``{"slot_id": str, "observed": {sku: units},
    "confidence": float?, "observation_ids": [...]?}`` -- the same shape a
    store-vision ``realogram`` export (or scenario ground truth) can produce.
    """
    slot_id = record['slot_id']
    observed = {str(k): int(v) for k, v in record.get('observed', {}).items()}
    conf = record.get('confidence')
    return SlotObservation(slot_id=slot_id, observed=observed,
                           evidence_basis='full_shelf_sensor',
                           confidence=conf if isinstance(conf, (int, float)) else None,
                           observation_ids=tuple(record.get('observation_ids', ())))


# --------------------------------------------------------------------------- #
# Diff + verdict                                                               #
# --------------------------------------------------------------------------- #

def diff_slot(expectation, observation, *, quantity_policy='at_least'):
    """Diff one slot's observation against its planogram expectation."""
    if quantity_policy not in QUANTITY_POLICIES:
        raise ValueError(f'quantity_policy must be one of {QUANTITY_POLICIES}')
    expected, observed = expectation.expected, observation.observed
    diffs = []
    compliant = True
    for sku in sorted(set(expected) | set(observed)):
        exp = expected.get(sku, 0)
        obs = observed.get(sku, 0)
        delta = exp - obs
        if exp == 0 and obs > 0:
            kind = 'misplaced'
            compliant = False
        elif quantity_policy == 'identity':
            kind = 'compliant' if obs > 0 else 'shortfall'
            if kind == 'shortfall':
                compliant = False
        elif quantity_policy == 'exact':
            if delta > 0:
                kind, compliant = 'shortfall', False
            elif delta < 0:
                kind, compliant = 'overstock', False
            else:
                kind = 'compliant'
        else:  # at_least
            if delta > 0:
                kind, compliant = 'shortfall', False
            else:
                kind = 'compliant'
        diffs.append(SlotDiff(sku=sku, expected=exp, observed=obs, delta=delta, kind=kind))
    return SlotVerdict(slot_id=expectation.slot_id, compliant=compliant,
                       evidence_basis=observation.evidence_basis,
                       expected=dict(expected), observed=dict(observed),
                       diffs=diffs, confidence=observation.confidence)


def reset_jobs_for(verdict, *, source_bin=None, priority=50, job_id_prefix='reset'):
    """Residual reset jobs that would bring a slot back to the planogram."""
    jobs = []
    for d in verdict.diffs:
        if d.kind == 'shortfall':
            jobs.append(ResetJob(
                sku=d.sku, quantity=d.delta, shelf_id=verdict.slot_id,
                kind='restock', priority=priority, source_bin=source_bin,
                provenance={'emitted_by': 'smart-verify', 'reason': 'planogram_shortfall',
                            'expected': d.expected, 'observed': d.observed,
                            'evidence_basis': verdict.evidence_basis,
                            'job_id_hint': f'{job_id_prefix}-{verdict.slot_id}-{d.sku}'}))
        elif d.kind in ('misplaced', 'overstock'):
            jobs.append(ResetJob(
                sku=d.sku, quantity=-d.delta, shelf_id=verdict.slot_id,
                kind='remove', priority=priority, source_bin=source_bin,
                provenance={'emitted_by': 'smart-verify', 'reason': f'planogram_{d.kind}',
                            'expected': d.expected, 'observed': d.observed,
                            'evidence_basis': verdict.evidence_basis,
                            'job_id_hint': f'{job_id_prefix}-{verdict.slot_id}-{d.sku}'}))
    return jobs


def build_report(*, job_id, plan, observations, source_bin=None,
                 quantity_policy='at_least', priority=50, provenance=None):
    """Assemble a :class:`SmartVerifyReport` from per-slot observations.

    ``observations`` is a list of :class:`SlotObservation`. Each slot is resolved
    against the planogram, diffed, and any residual reset jobs collected. The
    overall ``compliant`` is the AND over checked slots; ``evidence_basis`` is the
    WEAKEST basis among the checked slots (honest: the report is only as strong as
    its weakest evidence).
    """
    index = planogram_slot_index(plan)
    slots, residual = [], []
    bases = []
    for obs in observations:
        if obs.slot_id not in index:
            raise KeyError(f'slot {obs.slot_id!r} is not in the planogram')
        verdict = diff_slot(index[obs.slot_id], obs, quantity_policy=quantity_policy)
        slots.append(verdict)
        bases.append(obs.evidence_basis)
        residual.extend(reset_jobs_for(verdict, source_bin=source_bin, priority=priority))
    overall = all(s.compliant for s in slots) if slots else False
    weakest = max(bases, key=EVIDENCE_BASES.index) if bases else 'bookkeeping'
    return SmartVerifyReport(
        job_id=job_id, compliant=overall, evidence_basis=weakest,
        slots=slots, residual_jobs=residual, authority_note=AUTHORITY_NOTE,
        provenance=dict(provenance or {}))


# --------------------------------------------------------------------------- #
# Convenience loaders                                                          #
# --------------------------------------------------------------------------- #

def load_planogram(path):
    return json.loads(Path(path).read_text())


def load_stock(db_path):
    """Read the inventory ``stock`` table into ``{(location, sku): units}``.

    Opens the sqlite DB read-only so smart-verify can never mutate inventory.
    """
    uri = f'file:{Path(db_path).resolve()}?mode=ro'
    con = sqlite3.connect(uri, uri=True)
    try:
        rows = con.execute('SELECT location, sku, quantity FROM stock').fetchall()
    finally:
        con.close()
    return {(loc, sku): qty for loc, sku, qty in rows}


def report_from_mission(*, mission, plan, slot_id, stock=None, shelf_record=None,
                        source_bin=None, quantity_policy='at_least', priority=50,
                        job_id=None, mission_json_path=None):
    """End-to-end report for a completed mission's destination slot.

    Picks the strongest available observation for ``slot_id``:
    full_shelf_sensor (if ``shelf_record`` supplied) else bookkeeping (if
    ``stock`` supplied) else the mission verify sensor evidence. Refuses when the
    mission did not reach a succeeded verify (smart-verify only runs after
    verify succeeds).
    """
    evidence = mission_verify_evidence(mission)
    if evidence is None:
        raise ValueError('mission verify did not succeed; smart-verify has nothing to check')
    job_id = job_id or mission.get('events', [{}])[0].get('job_id') or 'unknown'
    if shelf_record is not None:
        obs = observation_from_shelf_record(shelf_record)
    elif stock is not None:
        obs = observation_from_bookkeeping(stock, slot_id)
    else:
        obs = observation_from_verify(evidence, slot_id)
    prov = {'source': 'smart-verify',
            'slot_id': slot_id,
            'quantity_policy': quantity_policy,
            'verify_observation_ids': list(obs.observation_ids),
            'mission_state': mission.get('state')}
    if mission_json_path is not None:
        prov['mission_json'] = str(mission_json_path)
    return build_report(job_id=job_id, plan=plan, observations=[obs],
                        source_bin=source_bin, quantity_policy=quantity_policy,
                        priority=priority, provenance=prov)
