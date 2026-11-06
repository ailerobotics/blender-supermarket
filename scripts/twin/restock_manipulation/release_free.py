"""sim-release-threshold-carton-frame-v1: prove the carton is FREE of the gripper and RESTING before the
arm retreats, and clear the gripper straight up before any lateral / pan retreat motion.

Pure decision logic (no Isaac / USD / numpy / ROS import), CPU-testable like ``place_release.py`` /
``release_reissue.py``. The controller owns the actual hold, the vertical-clearance lift motion, and the
RELEASE_NOT_FREE fail; this module decides, from the measured carton pose, when the carton has come to rest
and whether it followed the gripper up on the clearance lift.

WHY (coldwin83 vs coldwin85/coldwin86, same config):
  * All three released the weld at the set-down (``trigger=stall``) and the open gripper left the carton
    on/near the shelf. In coldwin83 the carton stayed; in coldwin85 and coldwin86 the carton ended on the
    floor after the retreat (``verify_place``: 'carton not resting on deck', cw85 ``xy=1.219
    bottom_gap=-0.954``, cw86 ``xy=1.220 bottom_gap=-0.815``). The retreat's first motion is (for the
    non-deck place) a near-horizontal retract, so an open gripper still cradling the barely-rested carton
    swept it off the shelf -- the fall was a RETREAT problem, not a release-height problem.

The fix (opt-in ``carton_level.release_free_check``): after the weld detach, HOLD (no retreat motion) until
the measured carton CENTRE z is stable and the carton BOTTOM is resting on the support (``|bottom_gap|`` small)
for ``settle_ticks`` consecutive ticks -- proving the carton is at rest, not cradled. THEN raise the gripper
straight up by ``lift_dz_m`` (a pure vertical clearance lift) before any lateral / pan retreat motion, and
verify the carton did NOT rise with the gripper. If the carton never comes to rest within ``settle_window_ticks``
or follows the gripper up on the clearance lift, FAIL BY NAME (``RELEASE_NOT_FREE``) rather than retreat and
sweep / carry the carton off.

Default off => the gate is never constructed => byte-identical (the detach transitions straight to retreat).
Every branch is unit-tested and replay-tested against the coldwin83/85/86/87 traces.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class ReleaseFreeDecision:
    """One post-detach verdict.

    ``action`` is ``'settle'`` (still holding -- keep waiting for the carton to come to rest), ``'lift'``
    (the carton is free and resting -> raise the gripper straight up for the clearance lift), ``'proceed'``
    (the clearance lift confirmed the carton did NOT follow -> retreat is safe) or ``'refuse'`` (the carton
    never came to rest, or it followed the gripper up -> RELEASE_NOT_FREE). ``reason`` is a ready-to-log
    sentence. The numbers are the measured evidence the controller logs (``RELEASE_FREE_*``)."""
    action: str
    reason: str
    settle_ticks: int
    carton_z: Optional[float]
    bottom_gap: Optional[float]
    rise: Optional[float]


class ReleaseFreeGate:
    """Per-set-down free-and-resting + clearance-lift decider (pure; the controller owns the hold, the lift
    motion and the fail).

    Construct once per controller when ``carton_level.release_free_check`` is on; :meth:`reset` it when the
    weld detach is confirmed. The controller drives two stages:

      1. HOLD stage: call :meth:`settle` once per tick with the measured carton CENTRE z and the carton
         ``bottom_gap`` (``carton_bottom - support_top``). It returns ``'settle'`` while waiting, ``'lift'``
         once the carton has been stable AND resting for ``settle_ticks`` consecutive ticks (it latches the
         carton z as the clearance-lift baseline), or ``'refuse'`` if the carton never rests within
         ``settle_window_ticks``.
      2. LIFT stage: once the controller's pure vertical clearance lift completes, call :meth:`follow` with
         the measured carton CENTRE z. It returns ``'proceed'`` when the carton stayed put (rose less than
         ``follow_frac`` of ``lift_dz_m``) or ``'refuse'`` when the carton rose with the gripper (still held).

    Parameters (all from ``carton_level``; see its ``release_free_*`` fields):
      * ``settle_ticks`` / ``stable_eps_m``: the carton CENTRE z must change by no more than ``stable_eps_m``
        between consecutive ticks (stable) while resting for this many consecutive ticks.
      * ``rest_gap_m``: the carton is RESTING when ``|bottom_gap| <= rest_gap_m`` (bottom on the support).
      * ``settle_window_ticks``: the bounded hold -- if the carton has not settled+rested within this many
        ticks, the carton is not at rest (still cradled / fell), so refuse.
      * ``lift_dz_m``: the pure vertical clearance-lift height the controller commands.
      * ``follow_frac``: the carton FOLLOWED the gripper when it rose at least this fraction of ``lift_dz_m``.
    """

    def __init__(self, *, settle_ticks: int = 5, stable_eps_m: float = 0.005, rest_gap_m: float = 0.03,
                 settle_window_ticks: int = 60, lift_dz_m: float = 0.06, follow_frac: float = 0.5):
        self.settle_ticks = max(1, int(settle_ticks))
        self.stable_eps_m = float(stable_eps_m)
        self.rest_gap_m = float(rest_gap_m)
        self.settle_window_ticks = max(self.settle_ticks, int(settle_window_ticks))
        self.lift_dz_m = float(lift_dz_m)
        self.follow_frac = float(follow_frac)
        if not (math.isfinite(self.stable_eps_m) and self.stable_eps_m > 0):
            raise ValueError('release_free_stable_eps_m must be a finite positive number of metres')
        if not (math.isfinite(self.rest_gap_m) and self.rest_gap_m > 0):
            raise ValueError('release_free_rest_gap_m must be a finite positive number of metres')
        if not (math.isfinite(self.lift_dz_m) and self.lift_dz_m > 0):
            raise ValueError('release_free_lift_dz_m must be a finite positive number of metres')
        if not (math.isfinite(self.follow_frac) and 0 < self.follow_frac <= 1.0):
            raise ValueError('release_free_follow_frac must be a finite number in (0, 1]')
        self.reset()

    def reset(self) -> None:
        """Clear the per-set-down latches (call at the confirmed weld detach)."""
        self._ticks = 0
        self._stable = 0
        self._prev_z: Optional[float] = None
        self.lift_baseline: Optional[float] = None

    @staticmethod
    def _finite(v) -> Optional[float]:
        try:
            f = float(v)
        except (TypeError, ValueError):
            return None
        return f if math.isfinite(f) else None

    def settle(self, *, carton_z, bottom_gap) -> ReleaseFreeDecision:
        """Fold one post-detach HOLD tick. ``carton_z`` is the measured carton CENTRE z, ``bottom_gap`` the
        carton-bottom-minus-support-top gap. Returns ``'settle'`` / ``'lift'`` / ``'refuse'``."""
        self._ticks += 1
        z = self._finite(carton_z)
        g = self._finite(bottom_gap)
        resting = g is not None and abs(g) <= self.rest_gap_m
        stable = z is not None and self._prev_z is not None and abs(z - self._prev_z) <= self.stable_eps_m
        self._prev_z = z if z is not None else self._prev_z
        if resting and stable:
            self._stable += 1
        else:
            self._stable = 0
        if self._stable >= self.settle_ticks:
            self.lift_baseline = z
            return ReleaseFreeDecision(
                'lift',
                'carton at rest (bottom_gap=%s m, stable for %d ticks) -- clearance-lift the gripper before '
                'the lateral retreat' % ('n/a' if g is None else '%.4f' % g, self._stable),
                self._stable, z, g, None)
        if self._ticks >= self.settle_window_ticks:
            return ReleaseFreeDecision(
                'refuse',
                'carton never came to rest within %d ticks after the detach (last bottom_gap=%s m, stable '
                'streak=%d) -- not free+resting, refusing to retreat and sweep it off'
                % (self.settle_window_ticks, 'n/a' if g is None else '%.4f' % g, self._stable),
                self._stable, z, g, None)
        return ReleaseFreeDecision(
            'settle',
            'holding for the carton to come to rest (bottom_gap=%s m, resting=%s stable_streak=%d/%d)'
            % ('n/a' if g is None else '%.4f' % g, resting, self._stable, self.settle_ticks),
            self._stable, z, g, None)

    def follow(self, *, carton_z) -> ReleaseFreeDecision:
        """Verdict after the vertical clearance lift completed: did the carton follow the gripper up?
        ``'proceed'`` when it stayed put, ``'refuse'`` (RELEASE_NOT_FREE) when it rose with the gripper. A
        missing baseline / carton z cannot prove following, so it proceeds (no false refusal)."""
        z = self._finite(carton_z)
        base = self._finite(self.lift_baseline)
        if z is None or base is None:
            return ReleaseFreeDecision(
                'proceed', 'no measured carton rise to evaluate -- proceeding to retreat',
                self._stable, z, None, None)
        rise = z - base
        if rise >= self.follow_frac * self.lift_dz_m:
            return ReleaseFreeDecision(
                'refuse',
                'carton rose %.4f m with the gripper on the %.3f m clearance lift (>= %.0f%% -- still held by '
                'the gripper)' % (rise, self.lift_dz_m, self.follow_frac * 100.0),
                self._stable, z, None, rise)
        return ReleaseFreeDecision(
            'proceed',
            'carton stayed on the support (rose %.4f m << %.3f m lift) -- free and resting, retreat is safe'
            % (rise, self.lift_dz_m),
            self._stable, z, None, rise)
