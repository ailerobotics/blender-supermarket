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
    was touched) or ``'obstructed'`` (a NON-support body is flush under the descending carton -- do not
    press, do not release). ``trigger`` is the winning release trigger (``'effort'`` | ``'stall'`` |
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
                 stall_height_margin_m: Optional[float] = None):
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
        self.reset()

    def reset(self, release_z: Optional[float] = ...) -> None:  # type: ignore[assignment]
        """Clear the per-descent latches. Pass ``release_z`` to re-arm the z fallback for a new set-down
        (the controller passes the freshly computed threshold); omit it to keep the current one."""
        if release_z is not ...:
            self.release_z = None if release_z is None else float(release_z)
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
