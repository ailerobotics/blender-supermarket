"""Per-stage phase plans + a pure milestone classifier for the mission tree.

The behaviour tree drives each mission stage as a **subtree of explicit navigate -> perceive ->
manipulate leaves** (the position-first order the runtime already follows). This module is the
single, pure (no py_trees, no simulator) source of truth for:

* ``PLANS`` -- the ordered list of ``(label, kind)`` steps per stage. One entry == one leaf, so
  *adding a new step is a new entry here* (and a new leaf appears in the tree + the live view).
* ``milestone(action, adapter)`` -- which step the running adapter is currently on, derived from
  the adapter's *observed* ``phase`` / manipulation phase / reposition + return flags. It is
  monotonic within a stage attempt, so a leaf for step ``i`` is RUNNING while the milestone ``== i``
  and SUCCEEDS once it has advanced past ``i``. The tree never issues navigation/manipulation
  commands itself -- it steps the real adapter once per tick and reads the phase back here, so the
  decomposition is a faithful, byte-identical *view* of the same stepping (see ``leaves.py``).

An adapter that exposes no ``phase`` (a plain test/mock fixture) has no milestones -- ``stage_plan``
returns ``None`` and the stage is driven by a single leaf (byte-identical to the pre-decomposition
``Stage``), so the existing offline/mock parity harnesses are unaffected.
"""
from __future__ import annotations

# kinds are purely descriptive labels for the view / the reader; the classifier below owns ordering.
NAV, PERCEIVE, MANIP, COUPLE, UNCOUPLE, VERIFY = (
    'nav', 'perceive', 'manip', 'couple', 'uncouple', 'verify')

# Position-first phase plans: drive into position, perceive, then act. One tuple == one leaf.
PLANS = {
    'collect': [('drive→pick', NAV), ('perceive carton', PERCEIVE), ('grasp·lift·tuck', MANIP),
                ('drive→rollwagen', NAV), ('perceive deck', PERCEIVE), ('place', MANIP),
                ('verify placed', VERIFY)],
    'couple':  [('drive→cart', NAV), ('dock drawbar', PERCEIVE), ('attach', COUPLE)],
    'outbound': [('tow→shelf', NAV)],
    'park':    [('stationarize', MANIP)],
    'transfer': [('perceive slot', PERCEIVE), ('place on shelf', MANIP), ('verify placed', VERIFY)],
    'verify':  [('observe inventory', PERCEIVE)],
    'return':  [('tow→home', NAV)],
}

# The uncouple-at-parking-spot return plan (captain 2026-10-20): tow the cart to its parking spot,
# brake it, release the coupling, back away and drive home empty. Selected when the adapter opts into
# uncouple-on-return (see stage_plan); the default 'return' plan above tows the cart home coupled.
RETURN_UNCOUPLE_PLAN = [('tow→cart park', NAV), ('brake cart', MANIP), ('uncouple', UNCOUPLE),
                        ('back away', NAV), ('drive→home', NAV)]


def _return_is_uncouple(adapter) -> bool:
    fn = getattr(adapter, '_uncouple_return', None)
    try:
        return bool(fn()) if callable(fn) else False
    except Exception:
        return False


def adapter_is_phased(adapter) -> bool:
    """True when the adapter exposes per-stage phases (the real RuntimeAdapter sets ``phase`` in its
    constructor, even before a stage begins); a plain fixture that never sets ``phase`` is not phased
    and its stages are driven by a single leaf."""
    return hasattr(adapter, 'phase')


def stage_plan(action, adapter):
    """The ordered ``(label, kind)`` plan for ``action`` on ``adapter``, or ``None`` for a non-phased
    adapter (single-leaf fallback). ``return`` resolves to the uncouple plan when the adapter opts in."""
    if not adapter_is_phased(adapter):
        return None
    if action == 'return' and _return_is_uncouple(adapter):
        return RETURN_UNCOUPLE_PLAN
    return PLANS.get(action)


def _collect_milestone(adapter) -> int:
    phase = getattr(adapter, 'phase', None)
    manip = getattr(adapter, '_last_manip_phase', None)
    repositioned = bool(getattr(adapter, '_collect_repositioned', False))
    pending = bool(getattr(adapter, '_collect_reposition_pending', False))
    if not repositioned:
        if phase == 'navigate' and pending:
            return 3                       # staging / reposition drive to the rollwagen mouth
        if phase == 'navigate':
            return 0                       # drive to the pick pose
        if phase in ('warm',):
            return 1                       # perceive the carton (warm the collision map)
        return 2                           # execute: grasp / lift / tuck (pick)
    if phase == 'navigate':
        return 3                           # still repositioning
    if phase in ('rewarm', 'warm'):
        return 4                           # perceive the deck at place_approach
    return 6 if manip == 'verify' else 5   # place, then the internal placement verify


def _couple_milestone(adapter) -> int:
    phase = getattr(adapter, 'phase', None)
    if phase == 'navigate':
        return 0
    if phase == 'fine_dock':
        return 1
    return 2                               # execute / attached


def _transfer_milestone(adapter) -> int:
    phase = getattr(adapter, 'phase', None)
    manip = getattr(adapter, '_last_manip_phase', None)
    if phase in ('warm', 'rewarm'):
        return 0
    return 2 if manip == 'verify' else 1


_RETURN_LEG_MILESTONE = {None: 0, 'cart_parking': 0, 'brake': 1, 'uncouple': 2,
                         'uncouple_wait': 2, 'cart_parking_backaway': 3, 'home': 4}


def _return_milestone(adapter) -> int:
    if not _return_is_uncouple(adapter):
        return 0
    return _RETURN_LEG_MILESTONE.get(getattr(adapter, '_return_leg', None), 0)


_CLASSIFIERS = {'collect': _collect_milestone, 'couple': _couple_milestone,
                'transfer': _transfer_milestone, 'return': _return_milestone}


def milestone(action, adapter) -> int:
    """The current (monotonic) step index for ``action`` on ``adapter``. Stages with a single
    navigate/confirm leaf (outbound, park, verify) are always on step 0 until they succeed."""
    fn = _CLASSIFIERS.get(action)
    return fn(adapter) if fn else 0
