"""sim-place-release-on-contact-v1: contact-gated weld release for the collect place descent.

Pure, CPU-testable decision logic. No Isaac / USD / numpy / ROS import.

WHY (coldwin68 vs coldwin64, same config/support /Rollwagen/chassis/shelf):
  * coldwin64 (clean): the welded carton descended the full planned drop and the fixed carton_z gate
    fired at ``release_z`` (``PLACE_CONTACT_RELEASE`` at z<=0.5624); peak arm effort 125.8 N*m. The pad
    raycast saw NOTHING (``SEAL_DIAG ... hit_bodies=[]``).
  * coldwin68 (jam): the PR359 pre-place hold-firm held the welded carton rigid into a ~1.46 deg roll, so
    its edge met the cage post ``/Rollwagen/chassis/post_2`` at the DESCENT START (carton_z frozen at
    0.6257, 0.075 m ABOVE the computed ``release_z`` 0.5624). The fixed carton_z gate therefore never
    fired, and the stiff place drive (kp1e6/eff2000) pressed the blocked carton to 757 N*m -- past the
    600 N*m station full-brake guard, three StageRetry cycles, then a 180 s sensor-stream backstop.
    ``SEAL_DIAG ... min_gap_mm=0.0 hit_bodies=['/Rollwagen/chassis/post_2']`` names the touched body.

The fix this module decides (controller consumes it; the existing cancel->wait-terminal->detach
serialization performs the release): RELEASE the weld the instant the support is ACTUALLY touched --
whichever of (effort rising over the pre-contact hold baseline, the descent STALLING, or the legacy
carton_z threshold) fires FIRST -- so the press never builds to the 600 N*m guard. BUT a measured
contact alone would drop the carton while it is jammed against a cage POST, so the release is GATED on
the touched body being the selected SUPPORT (shelf/deck): a flush contact on a NON-support body is an
OBSTRUCTION -- do not press, do not release; the controller fails by name (``PLACE_OBSTRUCTED``) or does
one bounded lateral re-approach. ``closed_loop_sensor`` / the seal signals are the measured inputs.

Opt-in (``carton_level.place_release_on_contact``); disabled => this module is never constructed =>
byte-identical. Every trigger is unit-tested and replay-tested against the coldwin68 / coldwin64 traces.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

from twin.restock_manipulation import place_seat_geometry as _psg

#: Default substrings that identify the SELECTED support SURFACE in a hit-prim path. A hit body whose
#: path contains one of these is the shelf/deck the carton sets down on (release-eligible); any OTHER
#: hit body (a cage ``post_*``, drawbar, rail) is an OBSTRUCTION. Tokens, not an exact path, so the
#: controller need not relay the runtime-resolved support prim.
DEFAULT_SUPPORT_TOKENS: Tuple[str, ...] = ('shelf', 'deck')


def is_support_hit(prim: str, support_tokens: Sequence[str] = DEFAULT_SUPPORT_TOKENS) -> bool:
    """True when the hit-prim path names the SUPPORT surface (contains a support token). Pure."""
    p = str(prim)
    return any(str(tok) in p for tok in support_tokens)


@dataclass(frozen=True)
class PlaceReleaseDecision:
    """One place-descent tick verdict.

    ``action`` is ``'press'`` (keep descending), ``'release'`` (fire the weld release NOW -- the support
    was touched AT the modelled seat), ``'obstructed'`` (a NON-support body is flush under the descending
    carton, OR -- with the seat gate on -- a measured contact occurred well ABOVE the modelled seat) or
    ``'fault'`` (seat gate on: the descent reached/passed the modelled seat with NO contact -- the support
    is not where modelled). ``trigger`` is the winning release trigger (``'effort'`` | ``'stall'`` |
    ``'contact'`` | ``'z'``) or None. ``reason`` is a ready-to-log sentence. The numbers are the evidence
    the controller logs (``PLACE_RELEASE_TICK`` / ``PLACE_CONTACT_RELEASE`` / ``PLACE_OBSTRUCTED``)."""
    action: str
    trigger: Optional[str]
    reason: str
    carton_z: Optional[float]
    effort: Optional[float]
    baseline: Optional[float]
    stall_count: int
    hit_prim: Optional[str]


class PlaceReleaseGate:
    """Per-place-descent contact-gated release decider.

    Construct once per set-down (the controller :meth:`reset` s it when it arms the descent and supplies
    the legacy ``release_z`` fallback). :meth:`observe` folds one tick of the measured signals and returns
    a :class:`PlaceReleaseDecision`.

    Parameters (all from ``carton_level``; see its ``place_release_*`` / ``place_support_tokens`` fields):
      * ``release_z``: the legacy carton-centre-z contact threshold (the fallback trigger); None => no z
        fallback (the descent-independent triggers still fire).
      * ``effort_margin_nm`` / ``effort_consecutive``: the arm effort must rise this many N*m over the
        pre-contact HOLD baseline (the running min of the observed effort) for this many consecutive ticks
        to trigger ``'effort'``.
      * ``stall_eps_m`` / ``stall_ticks``: the carton centre z must fail to make ``stall_eps_m`` of
        downward progress for this many consecutive ticks to trigger ``'stall'``.
      * ``contact_gap_mm``: a measured seal gap at/below this to a SUPPORT body triggers ``'contact'``;
        None => the contact trigger is off.
      * ``support_tokens``: substrings marking the support surface (see :data:`DEFAULT_SUPPORT_TOKENS`).
      * ``obstruction_gap_mm``: a NON-support hit body at/below this measured gap is a flush OBSTRUCTION.
      * ``effort_warmup_ticks``: ignore the effort trigger for this many opening ticks so the baseline
        settles before it can be exceeded (the descent's first stiff tracking transient is not contact).
    """

    def __init__(self, *, release_z: Optional[float] = None, effort_margin_nm: float = 100.0,
                 effort_consecutive: int = 3, stall_eps_m: float = 0.001, stall_ticks: int = 5,
                 contact_gap_mm: Optional[float] = None,
                 support_tokens: Sequence[str] = DEFAULT_SUPPORT_TOKENS,
                 obstruction_gap_mm: float = 2.0, effort_warmup_ticks: int = 3,
                 stall_height_margin_m: Optional[float] = None,
                 seat_center_z: Optional[float] = None,
                 seat_tol_m: float = _psg.DEFAULT_SEAT_TOL_M,
                 require_seat: bool = False):
        self.release_z = None if release_z is None else float(release_z)
        self.effort_margin_nm = float(effort_margin_nm)
        self.effort_consecutive = max(1, int(effort_consecutive))
        self.stall_eps_m = float(stall_eps_m)
        self.stall_ticks = max(1, int(stall_ticks))
        self.contact_gap_mm = None if contact_gap_mm is None else float(contact_gap_mm)
        self.support_tokens = tuple(support_tokens)
        self.obstruction_gap_mm = float(obstruction_gap_mm)
        self.effort_warmup_ticks = max(0, int(effort_warmup_ticks))
        # sim-place-verify-rest-tolerance-v1: a descent STALL only means 'support met' when the carton is
        # at/near the planned set-down height ``release_z``. None (default) => the stall trigger is
        # unchanged (byte-identical). A finite margin gates it: a stall whose carton centre z is still
        # above ``release_z + stall_height_margin_m`` is a render-paced STARTUP stall (the descent
        # trajectory has not produced motion yet -- carton_z frozen at the standoff), NOT a set-down, so
        # it is suppressed and the descent keeps pressing until the carton actually reaches the support
        # (the z / effort / support-contact triggers then fire at the true set-down). Inert when
        # ``release_z`` is None (no reference height). See coldwin83/coldwin85: both released on this
        # startup stall ~0.06 m HIGH, leaving the carton cradled above the shelf -- cw83's slipped free
        # and rested, cw85's was carried off by the retreat and fell to the floor (verify xy=1.219
        # bottom_gap=-0.954). Making the release fire at the true set-down height makes the rest
        # deterministic in both.
        self.stall_height_margin_m = (None if stall_height_margin_m is None
                                       else float(stall_height_margin_m))
        # sim-place-seat-geometry-v1: the HONEST seat gate. ``seat_center_z`` is the carton CENTRE z at the
        # corrected modelled seat (support TOP + carton half, in the SAME ur_base_link frame as the measured
        # carton centre ``carton_z``). When set, a measured CONTACT (effort/stall/legacy-z) releases the weld
        # ONLY when the carton is within ``seat_tol_m`` of the seat; a contact well ABOVE the seat is an
        # OBSTRUCTION (the coldwin87 14 cm stall -- do not drop the carton from there) and a descent that
        # reaches/passes the seat with NO contact is a named fault (the support is not where modelled). None
        # (default) => the seat gate is OFF => the legacy trigger behaviour (byte-identical). The shared
        # predicate is twin.restock_manipulation.place_seat_geometry.classify_set_down.
        self.seat_center_z = None if seat_center_z is None else float(seat_center_z)
        self.seat_tol_m = float(seat_tol_m)
        # sim-seat-gate-pad-down-path-v1: make the honest seat gate the SINGLE release authority. When the
        # seat gate is the authority (``require_seat`` -- the controller passes ``place_seat_gate``), a
        # would-be release trigger that fires while NO modelled seat is armed for this descent
        # (``seat_center_z`` is None) must NEVER release: coldwin98's release gate observed during the PR 380
        # pad-down REORIENT leg (carton centre z frozen at the standoff), so the raw STALL trigger fired a
        # SEAT-BLIND release 106 mm above the seat and the carton was let go in mid-air. With ``require_seat``
        # on, such a trigger is held as an OBSTRUCTION (named ``seat-gate-unarmed``), never 'support met'.
        # Default off => the legacy raw trigger cascade stands (byte-identical).
        self.require_seat = bool(require_seat)
        self.reset()

    def reset(self, release_z: Optional[float] = ...,  # type: ignore[assignment]
              seat_center_z: Optional[float] = ...) -> None:  # type: ignore[assignment]
        """Clear the per-descent latches. Pass ``release_z`` to re-arm the z fallback for a new set-down
        (the controller passes the freshly computed threshold); omit it to keep the current one. Pass
        ``seat_center_z`` to re-arm the honest seat gate for the new set-down (the corrected modelled seat
        carton-centre z); omit it to keep the current one."""
        if release_z is not ...:
            self.release_z = None if release_z is None else float(release_z)
        if seat_center_z is not ...:
            self.seat_center_z = None if seat_center_z is None else float(seat_center_z)
        self._ticks = 0
        self._baseline: Optional[float] = None
        self._over = 0
        self._min_z: Optional[float] = None
        self._stall = 0
        self.peak_effort = 0.0

    @staticmethod
    def _finite(v) -> Optional[float]:
        try:
            f = float(v)
        except (TypeError, ValueError):
            return None
        return f if math.isfinite(f) else None

    def _stall_is_setdown(self, z: float) -> bool:
        """sim-place-verify-rest-tolerance-v1: whether a no-downward-progress STALL at carton centre ``z``
        is close enough to the planned set-down height ``release_z`` to count as 'support met'. True (no
        gating) when the margin or ``release_z`` is unset; otherwise the stall only counts once the carton
        has descended to within ``stall_height_margin_m`` of ``release_z`` -- a stall still well above it is
        the render-paced startup stall (trajectory not yet moving), not a set-down. Pure."""
        if self.stall_height_margin_m is None or self.release_z is None:
            return True
        return z <= self.release_z + self.stall_height_margin_m

    def _classify_obstruction(self, seal_gap_mm: Optional[float],
                              seal_hit_prims: Optional[Sequence[str]]) -> Optional[str]:
        """Return the FIRST non-support hit-prim path that is flush under the pad (a real obstruction), or
        None. Flush = a measured seal gap at/below ``obstruction_gap_mm`` OR the descent has already
        stalled (a blocked carton whose gap reading is absent). Pure; tolerant of None / malformed."""
        if not seal_hit_prims:
            return None
        non_support = [str(p) for p in seal_hit_prims if p and not is_support_hit(p, self.support_tokens)]
        if not non_support:
            return None
        gap = self._finite(seal_gap_mm)
        flush = gap is not None and gap <= self.obstruction_gap_mm
        if flush or self._stall >= self.stall_ticks:
            return non_support[0]
        return None

    def observe(self, *, now: float, carton_z, commanded_down: bool, arm_effort_max=None,
                seal_gap_mm=None, seal_hit_prims: Optional[Sequence[str]] = None) -> PlaceReleaseDecision:
        """Fold one descent tick and decide. Only accumulates while ``commanded_down`` (the arm is driving
        the set-down); a non-descending tick returns a no-op ``'press'`` without disturbing the latches."""
        z = self._finite(carton_z)
        eff = self._finite(arm_effort_max)
        if not commanded_down:
            return PlaceReleaseDecision('press', None, 'not in place descent', z, eff, self._baseline,
                                        self._stall, None)
        self._ticks += 1
        # --- effort baseline (pre-contact HOLD effort = running min of the observed effort) + streak ---
        effort_trigger = False
        if eff is not None:
            if eff > self.peak_effort:
                self.peak_effort = eff
            if self._baseline is None or eff < self._baseline:
                self._baseline = eff
            if (self._ticks > self.effort_warmup_ticks and self._baseline is not None
                    and eff >= self._baseline + self.effort_margin_nm):
                self._over += 1
                effort_trigger = self._over >= self.effort_consecutive
            else:
                self._over = 0
        # --- stall (no downward progress of the carton centre z) ---
        if z is not None:
            if self._min_z is None or z <= self._min_z - self.stall_eps_m:
                self._min_z = z if self._min_z is None else min(self._min_z, z)
                self._stall = 0
            else:
                self._stall += 1
        stall_trigger = (z is not None and self._stall >= self.stall_ticks
                         and self._stall_is_setdown(z))
        # --- obstruction: a NON-support body flush under the descending carton (gate on the touched body) ---
        obstruction_prim = self._classify_obstruction(seal_gap_mm, seal_hit_prims)
        # --- contact: a SUPPORT body at/below the configured seal gap ---
        gap = self._finite(seal_gap_mm)
        contact_trigger = (self.contact_gap_mm is not None and gap is not None
                           and gap <= self.contact_gap_mm and seal_hit_prims is not None
                           and any(is_support_hit(p, self.support_tokens) for p in seal_hit_prims if p))
        # --- z fallback (the legacy threshold) ---
        z_trigger = self.release_z is not None and z is not None and z <= self.release_z

        if obstruction_prim is not None:
            return PlaceReleaseDecision(
                'obstructed', None,
                'descending carton met NON-support body %s (gap=%s mm, stall=%d) -- not pressing/releasing'
                % (obstruction_prim, 'n/a' if gap is None else '%.2f' % gap, self._stall),
                z, eff, self._baseline, self._stall, obstruction_prim)

        # --- sim-seat-gate-pad-down-path-v1: the seat gate is the SINGLE release authority. If it is required
        # (place_seat_gate on) but NO modelled seat is armed for THIS descent (seat_center_z unset), a release
        # trigger must NEVER fire a seat-blind release -- it could drop the carton anywhere. coldwin98 fired the
        # raw STALL trigger during the pad-down REORIENT leg (carton centre z frozen at the standoff, 106 mm
        # above the seat) with seat_center_z still None, and let the carton go in mid-air. HOLD and name it an
        # OBSTRUCTION (sentinel 'seat-gate-unarmed') so the controller re-approaches bounded / fails by name --
        # it never presses on, never releases. Inert when require_seat is off OR the seat is armed (the normal
        # pad-down descent arms the seat before this gate observes a descent). ---
        if self.require_seat and self.seat_center_z is None:
            if effort_trigger or stall_trigger or contact_trigger or z_trigger:
                winning = ('effort' if effort_trigger else 'stall' if stall_trigger
                           else 'contact' if contact_trigger else 'z')
                return PlaceReleaseDecision(
                    'obstructed', None,
                    'release trigger (%s) fired at carton_z=%s but the honest seat gate (place_seat_gate) is '
                    'the release authority and NO modelled seat is armed for this descent (seat_center_z '
                    'unset) -- refusing a seat-blind release that could drop the carton off the seat '
                    '(coldwin98: a frozen-standoff stall 106 mm above the seat); holding'
                    % (winning, 'n/a' if z is None else '%.4f' % z),
                    z, eff, self._baseline, self._stall, 'seat-gate-unarmed')
            return PlaceReleaseDecision('press', None, 'descending (seat gate required; awaiting seat arm)',
                                        z, eff, self._baseline, self._stall, None)

        # --- sim-place-seat-geometry-v1: the HONEST seat gate (supersedes the raw trigger cascade below
        # when armed). A measured CONTACT (effort / stall / support seal-gap) releases the weld ONLY when
        # the carton bottom is within seat_tol_m of the corrected modelled seat. A contact well ABOVE the
        # seat is an OBSTRUCTION -- the carton is held/blocked high (coldwin87's 14 cm stall was NOT a shelf
        # contact; the cage-filtered carton rested on nothing and the arm jammed a side-grip link on the
        # cage), so releasing there drops the carton; do not press/release, re-approach or fail by name. A
        # descent that reaches/passes the seat with NO contact is a named 'fault' (the support is not where
        # modelled). The legacy carton_z fallback still arms a release only AT the seat. Byte-identical when
        # seat_center_z is None (the gate is not armed). ---
        if self.seat_center_z is not None and z is not None:
            measured_contact = bool(effort_trigger or stall_trigger or contact_trigger)
            seat_gap = z - self.seat_center_z          # carton CENTRE minus seat CENTRE == carton bottom gap
            cls = _psg.classify_by_gap(seat_gap, contact=measured_contact, tol_m=self.seat_tol_m)
            winning = ('effort' if effort_trigger else 'stall' if stall_trigger
                       else 'contact' if contact_trigger else 'z' if z_trigger else None)
            if cls == 'obstructed':
                return PlaceReleaseDecision(
                    'obstructed', None,
                    'measured contact (%s) while the carton bottom is %+.1f mm vs the modelled seat '
                    '(> seat_tol %.0f mm) -- the carton is held/blocked ABOVE the seat, NOT a set-down; not '
                    'pressing/releasing (would drop it)'
                    % (winning or 'stall/effort', seat_gap * 1000.0, self.seat_tol_m * 1000.0),
                    z, eff, self._baseline, self._stall, 'above-modelled-seat')
            if cls == 'passed':
                return PlaceReleaseDecision(
                    'fault', None,
                    'the place descent reached %+.1f mm BELOW the modelled seat with NO measured contact -- '
                    'the support is not where modelled (would drop through / the carton is falling)'
                    % (seat_gap * 1000.0,),
                    z, eff, self._baseline, self._stall, None)
            if cls == 'seated' and measured_contact:
                return PlaceReleaseDecision(
                    'release', winning or 'contact',
                    'measured contact (%s) AT the modelled seat (carton bottom %+.1f mm, within seat_tol '
                    '%.0f mm) -- true set-down, releasing the weld'
                    % (winning or 'contact', seat_gap * 1000.0, self.seat_tol_m * 1000.0),
                    z, eff, self._baseline, self._stall, None)
            if z_trigger and not measured_contact and abs(seat_gap) <= self.seat_tol_m:
                # the legacy z threshold reached the seat with no stall/effort yet: release at the seat
                return PlaceReleaseDecision(
                    'release', 'z',
                    'carton z=%.4f reached the modelled seat (bottom %+.1f mm within seat_tol %.0f mm)'
                    % (z, seat_gap * 1000.0, self.seat_tol_m * 1000.0),
                    z, eff, self._baseline, self._stall, None)
            return PlaceReleaseDecision('press', None, 'descending (seat gate: %+.1f mm above the seat)'
                                        % (seat_gap * 1000.0,), z, eff, self._baseline, self._stall, None)
        if effort_trigger:
            return PlaceReleaseDecision(
                'release', 'effort',
                'arm effort %.1f N*m rose >= baseline %.1f + margin %.1f for %d ticks (support touched)'
                % (eff, self._baseline, self.effort_margin_nm, self._over),
                z, eff, self._baseline, self._stall, None)
        if stall_trigger:
            return PlaceReleaseDecision(
                'release', 'stall',
                'carton descent stalled: no %.4f m progress for %d ticks at z=%s (support met)'
                % (self.stall_eps_m, self._stall, 'n/a' if z is None else '%.4f' % z),
                z, eff, self._baseline, self._stall, None)
        if contact_trigger:
            return PlaceReleaseDecision(
                'release', 'contact',
                'seal gap %.2f mm <= %.2f on a support body (support touched)' % (gap, self.contact_gap_mm),
                z, eff, self._baseline, self._stall, None)
        if z_trigger:
            return PlaceReleaseDecision(
                'release', 'z',
                'carton z=%.4f <= release_z %.4f (planned set-down height reached)' % (z, self.release_z),
                z, eff, self._baseline, self._stall, None)
        return PlaceReleaseDecision('press', None, 'descending', z, eff, self._baseline, self._stall, None)
