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

import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

# Default location of the sibling store-vision checkout (overridable). This module
# never writes there; it only adds ``src`` to sys.path to import the package.
DEFAULT_STOREVISION_SRC = '/home/manishsaini/firstmate/projects/store-vision/src'

# The store-vision ``.venv`` python (pydantic + store-vision installed). The realogram
# MODEL pulls pydantic, which the plain CPU test python lacks, so the realogram cross-
# check runs in this interpreter as a READ-ONLY subprocess. Overridable.
DEFAULT_STOREVISION_VENV_PYTHON = '/home/manishsaini/firstmate/projects/store-vision/.venv/bin/python'

_REALOGRAM_DRIVER = str(Path(__file__).resolve().parent / '_storevision_realogram_driver.py')

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


def _dominant(units):
    """SKU with the most units in a ``{sku: units}`` map (ties by name), or None."""
    if not units:
        return None
    return sorted(units.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]


def _venv_python(venv_python=None):
    """Resolve the store-vision venv python, or None if not an executable file."""
    py = venv_python or os.environ.get('STOREVISION_VENV_PYTHON') or DEFAULT_STOREVISION_VENV_PYTHON
    return py if (py and Path(py).is_file() and os.access(py, os.X_OK)) else None


def cross_check_realogram(realogram, expectation, *, venv_python=None,
                          storevision_src=None, min_overlap=0.25, timeout=60):
    """Run the post-transfer realogram through store-vision realogram->planoreset.

    Builds a store-vision ``BayRealogram`` (observed) + core ``Shelf`` (the planogram
    target) from a :class:`shelf_realogram.ShelfRealogram` and its
    :class:`smart_verify.SlotExpectation`, then runs ``realogram.diff.diff_bay`` ->
    ``planoreset.verify.verify_from_realogram`` -> ``deviation_events`` inside the
    store-vision ``.venv`` as a READ-ONLY subprocess (the realogram model needs
    pydantic the CPU test python lacks).

    Returns a JSON-safe dict: ``{'available': True, 'source', 'compliant', 'diff',
    'verifications', 'deviation_events'}`` on success, else ``{'available': False,
    'reason': ...}``. Never raises, never mutates store-vision, never changes the
    mission gate -- advisory only, alongside the built-in :mod:`smart_verify` diff.

    One shared SKU per slot is modelled (dominant observed vs dominant expected); the
    mission restocks a single SKU per slot, so the facing/sku deviation is exact for
    that case. A genuine multi-SKU shelf needs a full store-vision ``realogram`` scan,
    which this lane does not capture (the mission verify stage sees only the landed
    unit).
    """
    py = _venv_python(venv_python)
    if py is None:
        return {'available': False,
                'reason': 'store-vision venv python not found (set $STOREVISION_VENV_PYTHON)'}
    if not Path(_REALOGRAM_DRIVER).is_file():  # pragma: no cover - driver ships with the module
        return {'available': False, 'reason': f'realogram driver missing: {_REALOGRAM_DRIVER}'}

    region = realogram.region
    store_region = region.store_region()
    level_z = region.top_z
    exp_units = dict(getattr(expectation, 'expected', {}) or {})
    obs_units = dict(realogram.observed)
    req = {
        'bay_id': realogram.bay_id,
        'store_region': store_region,
        'level_z': level_z,
        'min_overlap': min_overlap,
        'planogram_slots': [{
            'id': realogram.slot_id,
            'expected_sku': _dominant(exp_units),
            'facings': int(sum(exp_units.values())),
            'store_region': store_region,
            'level_z': level_z,
        }],
        'observed_slots': [{
            'id': realogram.slot_id,
            'deck_index': realogram.deck_index or 0,
            'sku': _dominant(obs_units),
            'facings': int(sum(obs_units.values())),
            'store_region': store_region,
            'level_z': level_z,
            'confidence': realogram.confidence or 0.0,
        }],
        'storevision_src': storevision_src or os.environ.get('STOREVISION_SRC') or DEFAULT_STOREVISION_SRC,
    }
    try:
        proc = subprocess.run(
            [py, '-I', _REALOGRAM_DRIVER], input=json.dumps(req),
            capture_output=True, text=True, timeout=timeout)
    except (OSError, subprocess.SubprocessError) as exc:
        return {'available': False, 'reason': f'subprocess failed: {type(exc).__name__}: {exc}'}
    if proc.returncode != 0:
        return {'available': False,
                'reason': f'driver exit {proc.returncode}: {(proc.stderr or "").strip()[:300]}'}
    try:
        return json.loads(proc.stdout)
    except ValueError as exc:
        return {'available': False,
                'reason': f'driver output not JSON: {exc}; stdout={proc.stdout[:200]!r}'}
