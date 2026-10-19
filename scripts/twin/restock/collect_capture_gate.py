"""fm394: WHEN the collect phase-aware capture teardown fires (pure, no Isaac -- CPU-testable).

The stage-scoped sensing teardown (fm381) and the phase-aware source->cart capture switch (fm384) both
DESTROY collect camera render products to free VRAM before the release peak. They used to fire at
grasp-confirm (the moment the carton weld latches), which tears the sensing/render loop down DURING the
carton LIFT -- and the lift then goal-missed 2 of 2 times it was active (runs 300d, 305: wrist_1 err
~0.63, arm still moving, 30 s settle, MOVE_RETRY could not rescue), versus a clean lift in run-300c which
did NOT have the switch. The retry cannot dodge it because the miss is a SYSTEMATIC consequence of the
mid-lift teardown (stale motion-completion feedback while sensing is held), not a stochastic transient.

Fix: fire only AFTER the lift has completed -- the controller has transitioned past 'lift' to a post-lift
phase (tuck/pan/transit/preplace/... for the staged/reposition flow, or preplace directly otherwise) -- so
the lift runs with an undisturbed sensing/render loop. The release peak is later (preplace->place->release),
so the ~1.6 GB is still freed in time.
"""

# Phases at or before the lift completes: the teardown must NOT fire during any of these. None = no phase
# gated yet (startup). Everything else (tuck/pan/transit/preplace/place/release/retreat/retreat_tuck/stow/
# verify) is post-lift and safe.
_PRE_LIFT_OR_LIFT = frozenset({None, 'observe', 'approach', 'contact', 'grasp', 'lift'})


def release_after_lift(phase, carton_welded):
    """True once the carton is welded AND the controller has moved PAST the lift (a post-lift phase).

    Pre-lift / lift / not-yet-welded => False (teardown holds, lift runs undisturbed). Idempotence is the
    caller's (the _fire_* methods fire once); this only decides the earliest tick the teardown may run."""
    return bool(carton_welded) and phase not in _PRE_LIFT_OR_LIFT
