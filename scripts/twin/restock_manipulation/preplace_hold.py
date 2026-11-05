"""Hold-firm-on-arrival decider for the carried preplace (sim-preplace-wrist2-hold-v1).

cw66 sealed its pick and carried the carton to the set-down pose, then stalled in the carried preplace:
wrist_2 reached its commanded 0.666 rad (closest approach 0.0006), was held there on the SOFT carried
drive (kp 3e5 / eff 400) through the end of leg 1, and was then BACK-DRIVEN off its target by leg 2's
1.69 rad shoulder_pan sweep -- collapsing to a -0.32 rad limit cycle that never re-converged, so the
convergence guard faulted at 30 s. cw64 ran the identical config/trajectory, dipped at the same pan angle,
and recovered: the soft held wrist holding the dynamic pan-sweep reaction is a coin toss.

The static hold is trivial (gravity on wrist_2 is 0.045 N*m); the give-way is purely the DYNAMIC pan-sweep
reaction on the soft HELD wrist. So once a carried-preplace joint has ARRIVED at its (settled) commanded
target it is swing-SAFE to stiffen -- it is no longer moving, so firming it cannot reintroduce the carton
swing the soft profile was chosen to kill (physics_setup: "never re-stiffen a wrist and reintroduce the
swing" applies to a MOVING wrist, not a held one). This decider names which joints have so arrived; the
Isaac port then switches those joints' PhysX drive to the firm hold profile, so a later leg's reaction
cannot back-drive an arrived joint.

Pure / CPU: it only reads the streamed command target vs the measured position per tick and tracks a
per-joint "command settled AND arrived" latch. Monotonic within a phase (once firm, stays firm); the
caller :meth:`reset` s it on leaving preplace. Disabled or fed None => no joints firm => byte-identical.
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Set


class HoldFirmOnArrival:
    """Per-joint arrival latch for the carried preplace.

    Construct with the arm joint NAMES, the arrival tolerance ``tol_rad`` (|command - measured| at or below
    which the joint is "at" its target), the command-settled epsilon ``settle_eps_rad`` (the command must
    move no more than this between ticks to count as settled), the ``dwell_ticks`` the command must stay
    settled-and-arrived before the joint latches firm, and ``enabled``. ``hold_joints`` (optional) restricts
    which joints may be firmed (e.g. the wrists + leveling cluster -- shoulder_pan is already firm via the
    carried_joint_profile); None => every arm joint is eligible.

    Call :meth:`observe(commanded, measured)` each preplace tick (both ARM_JOINTS-ordered); it returns the
    list of joint NAMES that NEWLY latched firm this tick (for the caller to log once and apply the firm
    gains). :attr:`firm` is the accumulated firm set. :meth:`reset` clears all state (call on phase exit)."""

    def __init__(self, joint_names: Sequence[str], *, tol_rad: float = 0.05, settle_eps_rad: float = 0.01,
                 dwell_ticks: int = 3, enabled: bool = False, hold_joints: Optional[Sequence[str]] = None):
        self.joint_names = tuple(joint_names)
        self.tol_rad = float(tol_rad)
        self.settle_eps_rad = float(settle_eps_rad)
        self.dwell_ticks = int(dwell_ticks)
        self.enabled = bool(enabled)
        self._eligible: Set[int] = (set(range(len(self.joint_names))) if hold_joints is None
                                    else {i for i, n in enumerate(self.joint_names) if n in set(hold_joints)})
        self.reset()

    def reset(self) -> None:
        self._prev_cmd: Optional[List[float]] = None
        self._settled: Dict[int, int] = {i: 0 for i in range(len(self.joint_names))}
        self._firm: Set[int] = set()

    @property
    def firm(self) -> Set[str]:
        return {self.joint_names[i] for i in self._firm}

    @staticmethod
    def _finite(seq: Sequence[float], n: int) -> Optional[List[float]]:
        vals = [float(v) for v in seq]
        if len(vals) != n or not all(math.isfinite(v) for v in vals):
            return None
        return vals

    def observe(self, commanded: Sequence[float], measured: Sequence[float]) -> List[str]:
        """Return the joint names that NEWLY latched firm this tick (empty when disabled / malformed /
        nothing new). A joint latches when its command has been settled (|cmd - prev_cmd| <= settle_eps for
        dwell_ticks consecutive ticks) AND it is arrived (|cmd - measured| <= tol). Monotonic."""
        if not self.enabled:
            return []
        n = len(self.joint_names)
        cmd = self._finite(commanded, n)
        meas = self._finite(measured, n)
        if cmd is None or meas is None:
            return []
        newly: List[str] = []
        prev = self._prev_cmd
        for i in range(n):
            if i not in self._eligible or i in self._firm:
                continue
            # Command settled this tick? (first tick has no prev -> not settled yet.)
            if prev is not None and abs(cmd[i] - prev[i]) <= self.settle_eps_rad:
                self._settled[i] += 1
            else:
                self._settled[i] = 0
            # Latch firm once the command has held settled for the dwell AND the joint is at its target.
            if self._settled[i] >= self.dwell_ticks and abs(cmd[i] - meas[i]) <= self.tol_rad:
                self._firm.add(i)
                newly.append(self.joint_names[i])
        self._prev_cmd = list(cmd)
        return newly
