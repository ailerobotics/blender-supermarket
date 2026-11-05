"""Frozen-joint convergence guard for the carried preplace approach (sim-preplace-per-joint-gains-v1).

coldwin53 reached the carried preplace on the SOFT drive (the swing fix), but shoulder_pan could not
HOLD its goal against the 1.64 rad wrist_2 sweep reaction: it drifted back +0.49 rad and FROZE 0.56 rad
short of the standoff goal, which never converged, so the stage ran the full 119.7 s and failed with the
opaque ``phase timeout: preplace`` -- naming neither the frozen joint nor its error.

This pure/CPU guard watches the per-joint error to the issued joint goal. When the WORST joint's error
stops improving (does not shrink by ``eps_rad`` for ``stall_s`` of SIM time) while the move is still
un-converged (worst error > ``tol_rad``), it reports which joint is frozen and by how much, as a
PHASE_CONVERGENCE line, so a stall is named in seconds instead of at the 120 s timeout. It never fires on
a move that is converging (every improvement past ``eps_rad`` rearms the window) nor on a converged move
(worst error <= tol). Off (``stall_s`` <= 0) => inert.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import math


@dataclass(frozen=True)
class ConvergenceStall:
    """A named frozen-joint verdict. ``joint`` is the worst joint's name, ``error`` its |measured-goal|
    (rad), ``best_error`` the smallest worst-error seen, ``stalled_s`` the SIM seconds since it last
    improved. ``message`` is the PHASE_CONVERGENCE log/fault line."""
    joint: str
    error: float
    best_error: float
    stalled_s: float
    message: str


class ConvergenceGuard:
    """Per-joint goal-error stall detector. Construct with the arm joint NAMES, the stall window
    ``stall_s`` (SIM seconds; <= 0 => inert), the improvement epsilon ``eps_rad`` and the converged
    tolerance ``tol_rad``. Call :meth:`arm` with the issued 6-joint goal when the preplace move is issued,
    then :meth:`observe(now, measured)` each tick: it returns a :class:`ConvergenceStall` the first tick
    the worst joint has been frozen for the whole window, else None. One-shot (``_fired``) so it reports
    once. :meth:`disarm` clears it when the phase leaves preplace (or the move re-issues)."""

    def __init__(self, joint_names: Sequence[str], *, stall_s: float, eps_rad: float = 0.01,
                 tol_rad: float = 0.05):
        self.joint_names = tuple(joint_names)
        self.stall_s = float(stall_s)
        self.eps_rad = float(eps_rad)
        self.tol_rad = float(tol_rad)
        self.disarm()

    @property
    def enabled(self) -> bool:
        return self.stall_s > 0

    def disarm(self) -> None:
        self._goal: Optional[Tuple[float, ...]] = None
        self._best: Optional[float] = None
        self._best_t: Optional[float] = None
        self._fired = False

    def arm(self, goal: Sequence[float]) -> None:
        """Arm against a fresh 6-joint goal (the issued preplace standoff). Resets the stall window."""
        g = tuple(float(v) for v in goal)
        if len(g) != len(self.joint_names) or not all(math.isfinite(v) for v in g):
            self._goal = None
            return
        self._goal = g
        self._best = None
        self._best_t = None
        self._fired = False

    def _worst(self, measured: Sequence[float]) -> Optional[Tuple[int, float]]:
        if self._goal is None:
            return None
        m = list(measured)
        if len(m) != len(self._goal) or not all(math.isfinite(float(v)) for v in m):
            return None
        errs = [abs(float(mi) - gi) for mi, gi in zip(m, self._goal)]
        idx = max(range(len(errs)), key=lambda i: errs[i])
        return idx, errs[idx]

    def observe(self, now: float, measured: Sequence[float]) -> Optional[ConvergenceStall]:
        if not self.enabled or self._fired or self._goal is None:
            return None
        if not math.isfinite(float(now)):
            return None
        worst = self._worst(measured)
        if worst is None:
            return None
        idx, err = worst
        # Converged / converging: a worst error within tol never stalls; any improvement past eps rearms.
        if err <= self.tol_rad:
            self._best = err
            self._best_t = now
            return None
        if self._best is None or err < self._best - self.eps_rad:
            self._best = err
            self._best_t = now
            return None
        stalled = now - (self._best_t if self._best_t is not None else now)
        if stalled >= self.stall_s:
            self._fired = True
            jn = self.joint_names[idx]
            msg = ('PHASE_CONVERGENCE phase=preplace stalled joint=%s error=%.4f rad best=%.4f rad for '
                   '%.1f s SIM (goal error stopped shrinking; worst joint frozen >= tol %.3f) -- the carried '
                   'preplace is not converging (coldwin53: shoulder_pan could not hold against the wrist '
                   'sweep). Failing by name instead of the opaque 120 s phase timeout.'
                   % (jn, err, self._best, stalled, self.tol_rad))
            return ConvergenceStall(joint=jn, error=err, best_error=float(self._best),
                                    stalled_s=float(stalled), message=msg)
        return None
