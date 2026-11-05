"""sim-station-full-brake-place-v1: full base brake for the LOADED collect place descent.

Pure, CPU-testable logic shared by the port brake selector (``isaac_port._apply_base_brake``) and the
weld-loop effort guard. No Isaac/USD/numpy dependency.

The cause (coldwin49, shelf place at z=0.9707): during the loaded from-above descent onto the cart
support the chassis is held ONLY by the rotation-lock station hold (``CHASSIS_STATION_PITCH_HOLD``),
which deliberately leaves TRANSLATION FREE ("weld-loop relief"). The welded carton/arm reaction then
recoils the base ~0.08-0.18 m south (away from the cart), past the 0.080 m ``collect_place_base_settle_m``
stationary guard, and the place never completes. The fix is to FULL-lock the base (translation AND
rotation) over the loaded place window -- from the moment the arm leaves the pre-place standoff (``place``)
through detach + withdraw (``release`` / ``retreat``) -- then return to the rotation-only hold.

Because a fully-locked base closes the world->base->arm->weld->carton->support->world loop once the carton
meets the support, :class:`EffortGuard` is the belt-and-suspenders bound: it watches the max arm joint
effort during the locked window and ABORTS to a HOLD (never a brake release, which would let the loop
shove the base) if the effort spikes past a configured limit. The contact-triggered weld release
(``place_contact_release``) fires at first contact and opens the loop, so in the nominal case the loop is
closed only for the instant between contact and release; the guard catches the pathological case where it
is not.
"""
import math

#: Controller phases that make up the LOADED collect place window: the arm has left the pre-place
#: standoff and is descending onto the support (``place``), the weld is released (``release``) and the
#: empty arm withdraws (``retreat``). The full lock covers exactly this span, scoped to the collect
#: station (``action == 'collect'``), and returns to the rotation-only hold at ``stow`` / ``verify``.
LOADED_PLACE_PHASES = frozenset({'place', 'release', 'retreat'})

# sim-station-lock-from-preplace-v1: the opt-in RUNTIME key ``station_full_brake_engage_phase`` chooses
# WHEN the full D6 lock engages relative to the carried pre-place. coldwin68/69 parked at the same spot as
# the no-shove reference (coldwin64) but, with the firm wrist hold now stopping the wrist giving way, the
# carried pre-place's SECOND leg (shoulder_pan sweep) drove its reaction into the base -- and during
# pre-place only the base ROTATION is held (``engage_brake``), the full six-axis lock engages only once the
# place descent begins. So the base was shoved ~0.33 m sideways DURING the pre-place and the carton came
# down ~0.5 m off its cart-relative target onto a cage post. The base must be FULLY locked before the
# firm-held carried pre-place starts, so the carton lands where it is aimed.
#   * 'place'      (default == today): engage at the descent (``LOADED_PLACE_PHASES``), pre-place rotation-
#                  only. Byte-identical to the behaviour before this key.
#   * 'preplace':  engage at the START of the carried pre-place (before leg 1), so both legs run under the
#                  full lock. The carton is airborne through pre-place, so the weld-loop concern (which is
#                  why ``place`` left translation free) does not apply until contact at ``place``.
#   * 'hold_firm': engage the moment ANY carried pre-place joint is firmed (the ``sim-preplace-wrist2-hold``
#                  hold-firm-on-arrival event, end of leg 1) -- the base is locked exactly when a joint has
#                  arrived and before the leg-2 sweep that shoves the base. Requires the preplace hold-firm
#                  config (``phase_drive.hold_firm_on_arrival.enabled``); config_preflight refuses it
#                  otherwise (else the value would silently fall back to ``place``).
# One engage, one release, no cycling: the lock engages once at the chosen trigger and holds through
# place/release/retreat, then returns to the rotation-only hold at stow/verify exactly as before.
ENGAGE_PHASE_PLACE = 'place'
ENGAGE_PHASE_PREPLACE = 'preplace'
ENGAGE_PHASE_HOLD_FIRM = 'hold_firm'
ENGAGE_PHASES = (ENGAGE_PHASE_PLACE, ENGAGE_PHASE_PREPLACE, ENGAGE_PHASE_HOLD_FIRM)
DEFAULT_ENGAGE_PHASE = ENGAGE_PHASE_PLACE


def resolve_engage_phase(value):
    """Normalise/validate ``station_full_brake_engage_phase``. None/'' => the default 'place' (today's
    behaviour). Raises ValueError on an unknown value so a misconfiguration fails loudly at boot."""
    if value is None or value == '':
        return DEFAULT_ENGAGE_PHASE
    v = str(value).strip().lower()
    if v not in ENGAGE_PHASES:
        raise ValueError('station_full_brake_engage_phase must be one of %r; got %r'
                         % (list(ENGAGE_PHASES), value))
    return v


def full_lock_window(action, phase, engage_phase, *, hold_firm_fired=False):
    """Pure: whether the loaded-place full base lock should be engaged this tick.

    Always locked once the descent has begun (``phase in LOADED_PLACE_PHASES``) at the collect station --
    today's window, unchanged. The ``engage_phase`` key only EXTENDS the window BACKWARD into pre-place:
      * 'preplace'  => locked for the whole ``preplace`` phase (both legs).
      * 'hold_firm' => locked during ``preplace`` once ``hold_firm_fired`` (a carried joint has arrived and
                       been firmed), i.e. from the end of leg 1.
      * 'place'     => pre-place is NOT locked (rotation-only hold), engage at the descent.
    No-op (False) outside the collect station. Pure/CPU; no Isaac/USD dependency."""
    if action != 'collect':
        return False
    if phase in LOADED_PLACE_PHASES:
        return True
    if phase == ENGAGE_PHASE_PREPLACE:
        ep = engage_phase or DEFAULT_ENGAGE_PHASE
        if ep == ENGAGE_PHASE_PREPLACE:
            return True
        if ep == ENGAGE_PHASE_HOLD_FIRM:
            return bool(hold_firm_fired)
    return False


def in_loaded_place_window(action, phase):
    """True when ``(action, phase)`` is the LOADED collect place window the full lock covers (the descent
    window; the ``station_full_brake_engage_phase`` pre-place extension is in :func:`full_lock_window`)."""
    return action == 'collect' and phase in LOADED_PLACE_PHASES


def max_abs_effort(efforts):
    """The maximum absolute FINITE value in ``efforts`` (an iterable of numbers), or None when the
    reading is absent/empty/all-non-finite. Pure; tolerant of None entries and a None argument."""
    if efforts is None:
        return None
    vals = []
    for e in efforts:
        try:
            f = float(e)
        except (TypeError, ValueError):
            continue
        if math.isfinite(f):
            vals.append(abs(f))
    return max(vals) if vals else None


class EffortGuard:
    """Weld-loop safety guard for the full-brake locked window.

    Folds one max-arm-joint-effort sample per tick. Latches an ABORT (a HOLD, not a shove) once the
    effort exceeds ``limit`` for ``consecutive`` consecutive IN-WINDOW samples, so a single-tick contact
    transient (the expected brief loop closure at first contact) does not trip it. Inert (always 'ok')
    when ``limit`` is None, out of the window, or no effort reading is available.
    """

    def __init__(self, limit, *, consecutive=3):
        if limit is not None:
            try:
                limit = float(limit)
            except (TypeError, ValueError):
                raise ValueError('station_place_full_brake_effort_limit must be a positive finite N·m')
            if not (limit > 0 and math.isfinite(limit)):
                raise ValueError('station_place_full_brake_effort_limit must be a positive finite N·m')
        self.limit = limit
        self.consecutive = max(1, int(consecutive))
        self.peak = 0.0
        self._over = 0
        self.aborted = False
        self.reason = None

    def observe(self, max_effort, in_window):
        """Fold one sample. Returns 'abort' on the tick the guard latches (and every tick after, while
        latched), else 'ok'. ``max_effort`` is the max absolute arm joint effort (N·m) or None."""
        if self.aborted:
            return 'abort'
        if self.limit is None or not in_window:
            self._over = 0           # out of window / no limit: reset the streak
            return 'ok'
        if max_effort is None:
            return 'ok'              # no reading this tick: cannot judge, keep placing
        if max_effort > self.peak:
            self.peak = max_effort
        if max_effort > self.limit:
            self._over += 1
            if self._over >= self.consecutive:
                self.aborted = True
                self.reason = ('station-full-brake weld-loop effort limit exceeded: max arm joint '
                               'effort %.1f N·m > %.1f N·m for %d consecutive ticks in the locked place '
                               'window' % (max_effort, self.limit, self._over))
                return 'abort'
        else:
            self._over = 0
        return 'ok'

    def reset(self):
        """Clear the latch + streak (a fresh collect attempt / a new mission op)."""
        self._over = 0
        self.aborted = False
        self.reason = None
