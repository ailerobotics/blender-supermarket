"""Lazy, best-effort store-vision cross-check for smart-verify (CPU, read-only).

store-vision (``/home/manishsaini/firstmate/projects/store-vision``) is a separate,
local-only project. This lane treats it as READ-ONLY: we import its public
``planoreset.verify`` interfaces to turn smart-verify's per-slot verdicts into
store-vision ``SlotVerification`` records + ``planoreset.deviation`` events -- the
same realogram/planoreset shapes store-vision's own ``restockloop`` consumes.

Design constraints:

* **Never required.** store-vision is imported lazily; if it is not importable
  the cross-check degrades to ``{'available': False, 'reason': ...}`` and the
  built-in :mod:`smart_verify` diff remains the authoritative close of the loop.
* **No model coupling.** ``planoreset.verify.verify_from_compliance`` is duck-typed
  on a ``ComplianceResult``-like object (``.status`` / ``.compliant`` /
  ``.expected_sku`` / ``.dominant_sku`` / ``.facings_present`` /
  ``.facings_expected`` / ``.shelf_id``), so we hand it a tiny shim built from a
  :class:`smart_verify.SlotVerdict` without constructing store-vision's full
  ``StoreRealogram`` model.
* **Pure observer.** Nothing here mutates store-vision; we only read its code.
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

# Default location of the sibling store-vision checkout (overridable). This module
# never writes there; it only adds ``src`` to sys.path to import the package.
DEFAULT_STOREVISION_SRC = '/home/manishsaini/firstmate/projects/store-vision/src'

# smart-verify diff kind -> a store-vision-style compliance status. Statuses in
# planoreset.verify._LABEL_STATUSES route to a label_error task; the rest route to
# a restock task. A shortfall/overstock is a stock problem (restock); a misplaced
# SKU (wrong product present) reads as a wrong_sku labelling/placement deviation.
_STATUS_FOR_KIND = {
    'shortfall': 'understocked',
    'overstock': 'overstocked',
    'misplaced': 'wrong_sku',
}


@dataclass(frozen=True)
class _ComplianceShim:
    """Minimal duck-typed ``ComplianceResult`` for ``verify_from_compliance``."""
    shelf_id: str
    status: str
    compliant: bool
    expected_sku: Optional[str]
    dominant_sku: Optional[str]
    facings_expected: Optional[int]
    facings_present: Optional[int]


def _dominant(units):
    """SKU with the most units in a ``{sku: units}`` map (ties broken by name)."""
    if not units:
        return None
    return sorted(units.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]


def _compliance_from_verdict(verdict):
    """Build a ``_ComplianceShim`` from a :class:`smart_verify.SlotVerdict`."""
    status = 'ok'
    if not verdict.compliant:
        # Surface the strongest (first non-compliant) diff kind as the status.
        for d in verdict.diffs:
            if d.kind != 'compliant':
                status = _STATUS_FOR_KIND.get(d.kind, d.kind)
                break
    return _ComplianceShim(
        shelf_id=verdict.slot_id,
        status=status,
        compliant=verdict.compliant,
        expected_sku=_dominant(verdict.expected),
        dominant_sku=_dominant(verdict.observed),
        facings_expected=sum(verdict.expected.values()) or None,
        facings_present=sum(verdict.observed.values()) or None,
    )


def _import_planoreset(storevision_src=None):
    """Import ``storevision.planoreset.verify`` lazily; return the module or None.

    ``storevision_src`` (or ``$STOREVISION_SRC``, or the default checkout) is added
    to ``sys.path`` only if needed. Any import failure returns ``(None, reason)``.
    """
    src = storevision_src or os.environ.get('STOREVISION_SRC') or DEFAULT_STOREVISION_SRC
    try:
        import storevision.planoreset.verify as verify  # noqa: F401
        return verify, None, None
    except Exception:
        pass
    added = None
    if src and Path(src).is_dir() and src not in sys.path:
        sys.path.insert(0, src)
        added = src
    try:
        import storevision.planoreset.verify as verify
        return verify, added, None
    except Exception as exc:  # pragma: no cover - exercised when store-vision absent
        if added is not None:
            try:
                sys.path.remove(added)
            except ValueError:
                pass
        return None, None, f'{type(exc).__name__}: {exc}'


def cross_check(report, *, storevision_src=None, now=0.0):
    """Cross-check a :class:`smart_verify.SmartVerifyReport` through store-vision.

    Returns a JSON-safe dict:

    * available store-vision -> ``{'available': True, 'source': <path>,
      'verifications': [...], 'deviation_events': [...], 'compliant': bool}``
      where the lists are store-vision ``SlotVerification`` / ``planoreset.deviation``
      ``to_dict()`` outputs;
    * otherwise -> ``{'available': False, 'reason': <why>}``.

    This is additive and advisory: it never changes ``report`` or the mission's
    stock-commit gate. ``report.compliant`` (the built-in diff) remains the close.
    """
    verify, source, reason = _import_planoreset(storevision_src)
    if verify is None:
        return {'available': False,
                'reason': reason or 'storevision.planoreset.verify not importable'}
    src_path = str(source or getattr(verify, '__file__', 'storevision'))
    results = {v.slot_id: _compliance_from_verdict(v) for v in report.slots}
    try:
        verifications = verify.verify_from_compliance(results)
    except Exception as exc:  # pragma: no cover - needs a partial store-vision env
        # The module imported but a transitive dependency (e.g. pydantic in
        # storevision.core) is missing at call time. Degrade: advisory only.
        return {'available': False, 'source': src_path,
                'reason': f'verify_from_compliance failed: {type(exc).__name__}: {exc}'}
    out = {
        'available': True,
        'source': src_path,
        'compliant': all(v.ok for v in verifications) if verifications else report.compliant,
        'verifications': [v.to_dict() for v in verifications],
        'deviation_events': [],
    }
    # deviation_events pulls storevision.core (pydantic); tolerate its absence so the
    # cross-check still yields the slot verifications it could compute.
    try:
        out['deviation_events'] = [e.to_dict() for e in verify.deviation_events(verifications, now=now)]
    except Exception as exc:  # pragma: no cover - needs a partial store-vision env
        out['deviation_events_note'] = f'deviation_events unavailable: {type(exc).__name__}: {exc}'
    return out
