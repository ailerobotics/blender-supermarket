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

# sim-shelf-staging-v1 (captain follow-up): the re-sequenced sales-floor OUTBOUND. Instead of towing the
# COUPLED train straight to the west-facing endcap (a ~46 deg in-aisle turn + the cart exiting the doorway
# angled ~40 deg), tow the rollwagen STRAIGHT down the portal axis to a staging pose (0 deg articulation,
# cart crosses the doorway aligned), brake + release the cart there (reusing the uncouple mechanic). The
# base is left PARKED (uncoupled) at the staging pose beside the parked cart; the solo carton-carry service
# to the endcap is the TRANSFER service leg below (sim-service-leg-carry-v1), not an empty drive here. Each
# step is a separate position-first leaf; the sharp turn to the endcap is done by the free base carrying the
# carton, not the coupled train. Selected when the adapter opts into shelf-staging (see stage_plan /
# adapter.shelf_staging); absent => the single 'tow→shelf' leaf above.
OUTBOUND_STAGING_PLAN = [('tow→staging', NAV), ('brake cart', MANIP), ('uncouple', UNCOUPLE)]

# sim-service-leg-carry-v1 (captain 2026-11-03): the re-sequenced sales-floor TRANSFER, the manipulation half
# of shelf-staging. With the cart parked + uncoupled at the staging pose (OUTBOUND_STAGING_PLAN above), the
# free base services the endcap ALONE: pick the carton off the parked cart AT the staging pose (reuse the
# transfer pick, re-targeted to the parked-cart frame), carry it to the endcap with the carton held LEVEL
# (carton_level, CARTON_TILT trip) and a drive-with-payload base speed/accel profile so the base cannot shove
# the carton, place it on the endcap (reuse the transfer place, re-registered in the endcap base frame), then
# drive the empty base BACK to the staging pose beside the parked cart. navigate -> perceive -> manipulate,
# one leaf each; the sharp endcap turn + the carry are done by the free base, the coupled train never turns.
# The carry-drive reuses fm390's TRANSIT carry-hold handshake (the arm holds the level fold while the base
# drives, then re-registers + places at the new pose) -- the SAME mechanic the collect place-on-cart uses.
# Selected when the adapter opts into shelf-staging; absent => the legacy 'perceive slot / place / verify'
# transfer at a single parked pose (byte-identical). The cart return to the backroom is a separate captain
# decision and is deliberately NOT part of this leg (it ends with the robot back at the staging pose).
TRANSFER_STAGING_PLAN = [('perceive carton', PERCEIVE), ('transfer-pick@staging', MANIP),
                         ('carry→endcap', NAV), ('perceive slot', PERCEIVE),
                         ('place@endcap', MANIP), ('verify placed', VERIFY),
                         ('drive→staging', NAV)]

# sim-cart-side-bay-v1 (captain option 1): the side bay extends shelf-staging. The OUTBOUND tows the
# COUPLED train straight down the portal axis (dead-centre doorway crossing) then tucks it WEST into an
# OFF-AXIS side bay (~14 deg articulation), brakes + releases it there -- same three position-first leaves
# as the staging plan, re-labelled for the bay. The free base then services the endcap (TRANSFER_STAGING_PLAN,
# the service leg re-targeted to the bay park).
OUTBOUND_SIDE_BAY_PLAN = [('tow→side bay', NAV), ('brake cart', MANIP), ('uncouple', UNCOUPLE)]

# The RETURN baseline: with the cart parked OFF the portal axis, the free base drives HOME EMPTY through the
# FREED doorway (the on-axis staging cart blocked this -- grid A* NO PATH; the side bay clears it). The cart
# stays parked (uncoupled) at the side bay. One position-first leaf.
RETURN_SIDE_BAY_PLAN = [('drive→home (empty)', NAV)]

# The COUPLED tow-home (cart retrieved to the backroom) is a NAMED, config-selectable FOLLOW-UP, not the
# baseline: the conserved ~180 deg cart reorientation cannot be gentle on a single-doorway floor (south bay
# loops; west bay needs a 46 deg outbound -- see docs/examples/cart-side-bay-mission-fragment.json +
# data/sim-cart-side-bay-v1/report.md). Defined here so the diagram can render the five retrieval steps when
# the follow-up is enabled (config['cart_tow_home']); the runtime wiring for it is deliberately not built here.
RETURN_SIDE_BAY_RECOUPLE_PLAN = [('re-couple@bay', COUPLE), ('tow→portal lead-in', NAV),
                                 ('tow→cart park', NAV), ('brake cart', MANIP), ('uncouple', UNCOUPLE),
                                 ('drive→home', NAV)]


def _return_is_uncouple(adapter) -> bool:
    fn = getattr(adapter, '_uncouple_return', None)
    try:
        return bool(fn()) if callable(fn) else False
    except Exception:
        return False


def _shelf_staging(adapter) -> bool:
    """True when the adapter opts into the sim-shelf-staging-v1 sales-floor re-sequence (outbound tows to
    a staging pose, uncouples, then the base services the endcap alone)."""
    fn = getattr(adapter, '_shelf_staging_active', None)
    try:
        return bool(fn()) if callable(fn) else False
    except Exception:
        return False


def _cart_side_bay(adapter) -> bool:
    """True when the adapter opts into the sim-cart-side-bay-v1 off-axis side bay (outbound tows the cart to
    the bay, uncouples, and the return drives the base home empty through the freed doorway)."""
    fn = getattr(adapter, '_cart_side_bay_active', None)
    try:
        return bool(fn()) if callable(fn) else False
    except Exception:
        return False


def adapter_is_phased(adapter) -> bool:
    """True when the adapter exposes per-stage phases (the real RuntimeAdapter sets ``phase`` in its
    constructor, even before a stage begins); a plain fixture that never sets ``phase`` is not phased
    and its stages are driven by a single leaf."""
    return hasattr(adapter, 'phase')


# vr1: steps whose evidence comes from a labelled SIM GROUND-TRUTH workaround when the adapter declares
# that stage's mode (adapter.stage_modes, set by the bundle from the runtime config): the transfer's slot
# target + placement check (fm461) and the mission-level inventory verify. The tag rides in the leaf name,
# the live view and the stage timeline, so the walkthrough caption names the workaround.
_MODE_TAGGED_KINDS = {'transfer': (PERCEIVE, VERIFY), 'verify': (PERCEIVE,), 'collect': (VERIFY,)}


def _stage_mode(action, adapter):
    modes = getattr(adapter, 'stage_modes', None)
    return modes.get(action) if isinstance(modes, dict) else None


def _tag_mode(action, plan, adapter):
    mode = _stage_mode(action, adapter)
    kinds = _MODE_TAGGED_KINDS.get(action, ())
    if not mode or not plan or not kinds:
        return plan
    tag = ' [%s]' % str(mode).replace('_', ' ')
    return [((label + tag) if kind in kinds else label, kind) for label, kind in plan]


def stage_plan(action, adapter):
    """The ordered ``(label, kind)`` plan for ``action`` on ``adapter``, or ``None`` for a non-phased
    adapter (single-leaf fallback). ``return`` resolves to the uncouple plan when the adapter opts in."""
    if not adapter_is_phased(adapter):
        return None
    if action == 'return' and _cart_side_bay(adapter):
        return RETURN_SIDE_BAY_PLAN
    if action == 'return' and _return_is_uncouple(adapter):
        return RETURN_UNCOUPLE_PLAN
    if action == 'outbound' and _cart_side_bay(adapter):
        return OUTBOUND_SIDE_BAY_PLAN
    if action == 'outbound' and _shelf_staging(adapter):
        return OUTBOUND_STAGING_PLAN
    if action == 'transfer' and _shelf_staging(adapter):
        return _tag_mode(action, TRANSFER_STAGING_PLAN, adapter)
    return _tag_mode(action, PLANS.get(action), adapter)


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
    if not _shelf_staging(adapter):
        # Legacy transfer at a single parked pose: perceive (warm/rewarm) -> place -> verify.
        if phase in ('warm', 'rewarm'):
            return 0
        return 2 if manip == 'verify' else 1
    # sim-service-leg-carry-v1 service leg (see TRANSFER_STAGING_PLAN): seven position-first steps.
    # Keyed on the observed carry leg (_transfer_leg) + phase + manip phase; monotonic within an attempt.
    leg = getattr(adapter, '_transfer_leg', None)
    if leg == 'return_staging':
        return 6                                   # drive→staging (empty base back to the staging pose)
    if manip == 'verify':
        return 5                                   # verify placed (controller's internal placement verify)
    if leg == 'place':
        return 4 if phase == 'execute' else 3      # rewarm = perceive slot (3); execute = place@endcap (4)
    if leg == 'carry':
        return 2                                   # carry→endcap (TRANSIT carry-drive, base moving)
    return 1 if phase == 'execute' else 0          # pick: warm = perceive carton (0); execute = pick (1)


_RETURN_LEG_MILESTONE = {None: 0, 'cart_parking': 0, 'brake': 1, 'uncouple': 2,
                         'uncouple_wait': 2, 'cart_parking_backaway': 3, 'home': 4}


def _return_milestone(adapter) -> int:
    if not _return_is_uncouple(adapter):
        return 0
    return _RETURN_LEG_MILESTONE.get(getattr(adapter, '_return_leg', None), 0)


_OUTBOUND_LEG_MILESTONE = {None: 0, 'staging': 0, 'brake': 1, 'uncouple': 2, 'uncouple_wait': 2}


def _outbound_milestone(adapter) -> int:
    # Single 'tow→shelf' leaf (always step 0) unless the shelf-staging re-sequence is active, when the
    # observed leg maps to tow→staging(0) / brake cart(1) / uncouple(2). The base is left parked at the
    # staging pose; the carton-carry service to the endcap is the transfer service leg, not an outbound leg.
    if not _shelf_staging(adapter):
        return 0
    return _OUTBOUND_LEG_MILESTONE.get(getattr(adapter, '_outbound_leg', None), 0)


_CLASSIFIERS = {'collect': _collect_milestone, 'couple': _couple_milestone,
                'transfer': _transfer_milestone, 'return': _return_milestone,
                'outbound': _outbound_milestone}


def milestone(action, adapter) -> int:
    """The current (monotonic) step index for ``action`` on ``adapter``. Stages with a single
    navigate/confirm leaf (outbound, park, verify) are always on step 0 until they succeed."""
    fn = _CLASSIFIERS.get(action)
    return fn(adapter) if fn else 0
