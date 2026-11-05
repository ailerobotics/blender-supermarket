"""Frozen-joint convergence guard for the carried preplace approach (sim-preplace-per-joint-gains-v1,
fixed in sim-preplace-convergence-guard-fix-v1).

coldwin53 reached the carried preplace on the SOFT drive (the swing fix), but shoulder_pan could not
HOLD its goal against the 1.64 rad wrist_2 sweep: it descended to the standoff (err 0.06 rad), then as
the wrist_2 leg swept it was driven BACK +0.49 rad and FROZE 0.56 rad short of the goal, which never
converged, so the stage ran the full 119.7 s and failed with the opaque ``phase timeout: preplace`` --
naming neither the frozen joint nor its error.

PR 342's first guard watched each joint's error to the FINAL goal and failed when the single worst joint's
error stopped shrinking. The densified preplace path is SEQUENTIAL (shoulder_pan leg first, then the
wrist_2 sweep), so a joint that is legitimately IDLE during another joint's leg -- commanded to hold at its
carry value, far from its final goal only because its own leg has not started -- looked ``frozen`` by the
final-goal measure. coldwin54 FALSE-TRIPPED at 30 s: wrist_2 sat at its carry value (0.002, 1.63 rad from
its eventual goal) through the pan leg while the pan descended cleanly, and the guard aborted the run.

This rewrite measures the right thing (sim-preplace-convergence-guard-fix-v1): per-joint, it distinguishes
a joint that is IDLE (not moving, so not yet commanded -- the wrist_2-during-the-pan-leg case) from one
that is FROZEN. A joint is frozen when it HAS been driven (its executed position moved from the armed
start) yet its executed position does not converge -- it either jams part-way short of the goal and stops
improving for the stall window, or RETREATS from the goal after approaching it (the coldwin53 pan pattern,
driven back by the wrist sweep) and holds there for the window. A joint that has never moved from its
armed start is treated as idle (its leg has not begun) and is excluded. A joint within ``tol_rad`` of its
goal is converged and continuously rearms, so it never trips. The guard reports which joint is frozen, its
commanded (goal) and executed positions, how far it has drifted from its closest approach, and the window,
as a PHASE_CONVERGENCE line -- so a genuine stall is named in seconds instead of at the 120 s timeout,
while the sequential path's not-yet-moved joints never false-trip. Off (``stall_s`` <= 0) => inert.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple

import math


@dataclass(frozen=True)
class ConvergenceStall:
    """A named frozen-joint verdict. ``joint`` is the frozen joint's name, ``error`` its |executed-goal|
    (rad), ``commanded`` the joint's commanded goal position, ``executed`` its measured position,
    ``best_error`` the smallest |executed-goal| it reached (so ``error - best_error`` is how far it has
    drifted/retreated from its closest approach), ``stalled_s`` the SIM seconds since it last improved.
    ``message`` is the PHASE_CONVERGENCE log/fault line."""
    joint: str
    error: float
    commanded: float
    executed: float
    best_error: float
    stalled_s: float
    message: str


@dataclass(frozen=True)
class DriftRecapture:
    """sim-preplace-wrist2-hold-v1: a NON-terminal drift re-capture signal. Emitted the first tick a
    DRIVEN, un-converged preplace joint has drifted away from its closest approach by more than
    ``recapture_rad`` while re-capture budget remains -- instead of waiting out the full stall window and
    faulting. The controller re-issues the armed preplace goal (re-driving the back-driven joint toward its
    target) and logs PREPLACE_DRIFT_RECAPTURE once. ``joint`` is the drifting joint, ``from_pos`` its
    executed (drifted) position, ``to_pos`` its commanded goal, ``best_error`` the closest approach it had
    reached, ``count``/``limit`` the re-capture used/budget for this joint."""
    joint: str
    from_pos: float
    to_pos: float
    best_error: float
    error: float
    count: int
    limit: int
    message: str


class ConvergenceGuard:
    """Per-joint frozen-joint detector for the carried preplace. Construct with the arm joint NAMES, the
    stall window ``stall_s`` (SIM seconds; <= 0 => inert), the improvement epsilon ``eps_rad``, the
    converged tolerance ``tol_rad`` and the ``engage_rad`` motion a joint must show from its armed start to
    count as driven (below it the joint is idle -- not yet commanded -- and never trips).

    Call :meth:`arm` with the issued 6-joint goal when the preplace move is issued, then
    :meth:`observe(now, measured)` each tick: it returns a :class:`ConvergenceStall` the first tick a
    driven joint has been frozen (un-converged AND not improving) for the whole window, else None. The
    first observe after arming captures the start pose (returns None). One-shot (``_fired``) so it reports
    once. :meth:`disarm` clears it when the phase leaves preplace (or the move re-issues)."""

    def __init__(self, joint_names: Sequence[str], *, stall_s: float, eps_rad: float = 0.01,
                 tol_rad: float = 0.05, engage_rad: float = 0.1,
                 recapture_rad: float = 0.0, recapture_max: int = 0):
        self.joint_names = tuple(joint_names)
        self.stall_s = float(stall_s)
        self.eps_rad = float(eps_rad)
        self.tol_rad = float(tol_rad)
        self.engage_rad = float(engage_rad)
        # sim-preplace-wrist2-hold-v1: opt-in drift re-capture. recapture_rad (> 0) is the drift from the
        # closest approach that triggers a re-issue; recapture_max (> 0) the per-joint re-capture budget.
        # Both must be positive to arm re-capture; either <= 0 => inert (only the terminal stall path runs,
        # byte-identical to sim-preplace-convergence-guard-fix-v1).
        self.recapture_rad = float(recapture_rad)
        self.recapture_max = int(recapture_max)
        self.disarm()

    @property
    def enabled(self) -> bool:
        return self.stall_s > 0

    @property
    def recapture_enabled(self) -> bool:
        return self.enabled and self.recapture_rad > 0 and self.recapture_max > 0

    @property
    def goal(self) -> Optional[Tuple[float, ...]]:
        """The armed 6-joint preplace goal (the controller re-issues this on a :class:`DriftRecapture`)."""
        return self._goal

    def disarm(self) -> None:
        self._goal: Optional[Tuple[float, ...]] = None
        self._start: Optional[Tuple[float, ...]] = None
        self._best: Dict[int, float] = {}
        self._best_t: Dict[int, float] = {}
        self._abs_best: Dict[int, float] = {}
        self._moved: Dict[int, bool] = {}
        self._recaptures: Dict[int, int] = {}
        self._fired = False

    def arm(self, goal: Sequence[float]) -> None:
        """Arm against a fresh 6-joint goal (the issued preplace standoff). Resets all per-joint state; the
        armed start pose is captured on the first :meth:`observe` tick."""
        g = tuple(float(v) for v in goal)
        if len(g) != len(self.joint_names) or not all(math.isfinite(v) for v in g):
            self._goal = None
            return
        self._goal = g
        self._start = None
        self._best = {}
        self._best_t = {}
        self._abs_best = {}
        self._moved = {}
        self._recaptures = {}
        self._fired = False

    @staticmethod
    def _finite6(seq: Sequence[float], n: int) -> Optional[Tuple[float, ...]]:
        vals = [float(v) for v in seq]
        if len(vals) != n or not all(math.isfinite(v) for v in vals):
            return None
        return tuple(vals)

    def observe(self, now: float, measured: Sequence[float]):
        """Return the first non-None of:
          * a :class:`DriftRecapture` (re-capture armed, a driven joint drifted > ``recapture_rad`` from its
            closest approach and budget remains) -- the controller re-issues the goal, NOT terminal; or
          * a :class:`ConvergenceStall` (a driven joint un-converged and not improving for the whole stall
            window with no re-capture left) -- terminal; else None.
        One-shot for the terminal stall (``_fired``); re-captures are per-joint budgeted and non-terminal."""
        if not self.enabled or self._fired or self._goal is None:
            return None
        if not math.isfinite(float(now)):
            return None
        m = self._finite6(measured, len(self._goal))
        if m is None:
            return None
        now = float(now)
        # Capture the armed start pose on the first observe; need it to tell a driven joint from an idle one.
        if self._start is None:
            self._start = m
            for i, (mi, gi) in enumerate(zip(m, self._goal)):
                self._best[i] = abs(mi - gi)
                self._abs_best[i] = abs(mi - gi)
                self._best_t[i] = now
                self._moved[i] = False
                self._recaptures[i] = 0
            return None

        stall_idx: Optional[int] = None
        stall_err = -1.0
        recap_idx: Optional[int] = None
        recap_err = -1.0
        for i, (mi, gi) in enumerate(zip(m, self._goal)):
            err = abs(mi - gi)
            # A joint that has moved beyond engage_rad from its armed start has been driven (its leg ran);
            # one that never moves is idle (its leg has not started) and is excluded from the stall test.
            if abs(mi - self._start[i]) > self.engage_rad:
                self._moved[i] = True
            # Converged / improving rearms the window (within tol is converged; any shrink past eps rearms).
            if err <= self.tol_rad or err < self._best[i] - self.eps_rad:
                self._best[i] = min(self._best[i], err)
                self._abs_best[i] = min(self._abs_best[i], err)
                self._best_t[i] = now
                continue
            # Un-converged and not improving: only a DRIVEN joint stalls (an idle one is not frozen).
            if not self._moved[i]:
                continue
            # sim-preplace-wrist2-hold-v1: a driven joint that has drifted away from its closest approach by
            # more than recapture_rad is RE-CAPTURED (re-issue the goal) before the stall window elapses, as
            # long as its budget remains -- so a back-driven wrist is re-driven in a few sim-s instead of
            # faulting at 30 s (cw66: the leg-2 pan sweep back-drove the arrived wrist_2). Prefer re-capture.
            if (self.recapture_enabled and err > self._best[i] + self.recapture_rad
                    and self._recaptures[i] < self.recapture_max):
                if err > recap_err:
                    recap_err, recap_idx = err, i
                continue
            if now - self._best_t[i] >= self.stall_s and err > stall_err:
                stall_err, stall_idx = err, i

        if recap_idx is not None:
            i = recap_idx
            self._recaptures[i] += 1
            jn = self.joint_names[i]
            cmd = float(self._goal[i])
            ex = float(m[i])
            err = abs(ex - cmd)
            best = float(self._abs_best[i])
            count, limit = self._recaptures[i], self.recapture_max
            # Restart this joint's stall window from the drifted point so the re-issue has a fresh window to
            # re-converge in; the closest approach (_abs_best) is preserved for any later terminal message.
            self._best[i] = err
            self._best_t[i] = now
            msg = ('PREPLACE_DRIFT_RECAPTURE joint=%s from=%.4f to=%.4f (drifted %.4f rad from its closest '
                   'approach %.4f; re-issuing the preplace goal, re-capture %d/%d) -- a driven preplace joint '
                   'was back-driven off its target; re-driving it instead of waiting out the stall window.'
                   % (jn, ex, cmd, err - best, best, count, limit))
            return DriftRecapture(joint=jn, from_pos=ex, to_pos=cmd, best_error=best, error=err,
                                  count=count, limit=limit, message=msg)

        if stall_idx is None:
            return None
        self._fired = True
        i = stall_idx
        jn = self.joint_names[i]
        cmd = float(self._goal[i])
        ex = float(m[i])
        err = abs(ex - cmd)
        best = float(self._abs_best[i])
        stalled = now - float(self._best_t[i])
        drift = err - best
        msg = ('PHASE_CONVERGENCE phase=preplace stalled joint=%s commanded=%.4f executed=%.4f error=%.4f '
               'rad (drifted %.4f rad from its closest approach %.4f) not improving for %.1f s SIM (> tol '
               '%.3f) -- the carried preplace is not converging: this joint has been driven but its '
               'executed position is frozen/retreating (coldwin53: shoulder_pan was driven back by the '
               'wrist_2 sweep and could not hold). Failing by name instead of the opaque 120 s phase '
               'timeout.' % (jn, cmd, ex, err, drift, best, stalled, self.tol_rad))
        return ConvergenceStall(joint=jn, error=err, commanded=cmd, executed=ex, best_error=best,
                                stalled_s=float(stalled), message=msg)
