"""Pure (no py_trees) view labels for the behaviour-tree mission snapshot.

fm390: the collect stage is a POSITION-FIRST placement sequence -- the base drives to place_approach in
front of the rollwagen mouth, re-perceives the deck, then lowers + verifies -- so the live view should
render that arc (nav -> perceive -> place -> verify), not an opaque 'collect: running'. This module maps
the running adapter's nav phase + the manipulation controller's phase to a short step label. It imports
nothing heavy so it validates under any interpreter (the pure test tier); ``leaves.py`` uses it.
"""
from __future__ import annotations

# The headline arc is nav -> perceive -> place -> verify; the honest finer steps (pick / carry / retreat
# / stow) sit in between so the view does not hide real motion behind the four headline words.
COLLECT_STEP = {
    'navigate': 'nav', 'warm': 'perceive', 'rewarm': 'perceive', 'fine_dock': 'nav',
    'observe': 'perceive', 'transit': 'nav',
    'approach': 'pick', 'contact': 'pick', 'grasp': 'pick', 'lift': 'pick',
    'tuck': 'carry', 'pan': 'carry',
    'preplace': 'place', 'place': 'place', 'release': 'place',
    'retreat': 'retreat', 'retreat_tuck': 'retreat', 'stow': 'stow', 'verify': 'verify',
}

# The adapter phases that are a BASE drive / collision-map (re-)warm: they override the finer
# manipulation phase for the label (the arm is paused/held while the base moves or the map warms).
_DRIVE_PHASES = ('navigate', 'warm', 'rewarm', 'fine_dock')


def collect_substep(mission):
    """Best-effort, read-only collect sub-step label ``'<step> (<raw phase>)'`` for the view.

    The adapter drive/warm phase overrides the finer manipulation phase; otherwise the controller phase
    is the signal. Returns None when no phase is exposed (a plain mission/adapter without these attrs =>
    no label, byte-identical snapshot)."""
    adapter = getattr(mission, 'adapter', None)
    if adapter is None:
        return None
    phase = getattr(adapter, 'phase', None)
    manip = getattr(adapter, '_last_manip_phase', None)
    raw = phase if phase in _DRIVE_PHASES else (manip or phase)
    if raw is None:
        return None
    return '%s (%s)' % (COLLECT_STEP.get(raw, raw), raw)
