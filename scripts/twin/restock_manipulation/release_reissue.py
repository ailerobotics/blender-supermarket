"""sim-release-open-reissue-v1: re-emit the gripper-open + weld-detach control every release tick.

Pure decision logic (no Isaac / USD / numpy / ROS import), CPU-testable like ``place_release.py``.

WHY (coldwin84 vs coldwin83, BYTE-IDENTICAL config):
  * coldwin83 (released cleanly): the contact-gated weld release fired, the release phase emitted the
    gripper-open control ONCE, and the Isaac side ran ``cad_tool.command(enabled=False)`` exactly once
    (``coldwin83.console.log`` has exactly ONE ``SEAL_DIAG command:open``). The weld
    (``SIM_ATTACH_FALLBACK``) released, ``s.contact`` cleared, the run detached and completed.
  * coldwin84 (hang, same config): the SAME single gripper-open control was emitted (``self.issued``
    latch in ``controller.py`` release phase) but the Isaac side NEVER ran ``command(open)`` --
    ``coldwin84.console.log`` has ZERO ``SEAL_DIAG command:open`` lines and the LAST ``SEAL_DIAG`` is
    still ``command:close ... enabled=True``. The one open-carrying ``/manipulation/status`` was
    overwritten on the LAST-VALUE control relay (``control_ipc`` keeps only the latest status per topic)
    by a newer host status -- the next, no-control release tick's higher ``sequence`` -- before
    ``IsaacRosGraph.poll`` read it. This is the SAME one-shot drop PR 356 fixed for the contact ENABLE
    (``contact_reissue_suction_each_tick``); the open was never made resilient. Five earlier runs on the
    identical config released cleanly, so the release was a dice roll on relay timing. The weld stayed
    engaged and the run e-stopped: ``release did not detach the held carton within 8.0s sim of the
    gripper-open command``.

The fix (opt-in ``carton_level.release_reissue``): during the release phase RE-EMIT the gripper-open
control -- which on the Isaac side is BOTH the open AND the ``SIM_ATTACH_FALLBACK`` weld detach
(``cad_tool.command(False)`` -> ``open_gripper`` + ``_release_fallback``) -- on EVERY release tick until
the per-tick feedback (``RESTOCK_CONTACT_STATUS_FEEDBACK`` / PR 358: ``gripper_enabled`` +
``gripper_status_time``, and the ``s.contact`` latch) confirms the gripper is OPEN and the carton is NOT
held (seal released), THEN proceed to the cuMotion planning-scene detach. Every release status then
carries ``controls=[open]``, so whichever one the relay hands ``IsaacRosGraph.poll`` lands the open --
the relay can no longer drop it. If no confirmation arrives within a bounded sim window, FAIL BY NAME
(``RELEASE_NOT_CONFIRMED``) with the feedback fields -- NEVER restart the stage with the carton welded.

Default off => the gate is never constructed => byte-identical (the single-shot open + the 8 s
``release_detach_timeout_s`` path). Every branch is unit-tested and replay-tested against the coldwin84
(hang off, confirmed on) and coldwin83 (single-shot success still confirms) traces.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class ReleaseReissueDecision:
    """One release-tick verdict.

    ``action`` is ``'reissue'`` (carton still held -- re-emit the open+detach control this tick),
    ``'confirmed'`` (the feedback confirms gripper open AND carton not held -> proceed to the detach) or
    ``'timeout'`` (no confirmation within the bounded sim window -> fail by name). ``reissue_control`` is
    True only on ``'reissue'`` (re-emit the one-shot). ``log`` is a ready-to-print ``RELEASE_REISSUE`` line
    when one is due (once per ``log_every`` ticks), else None. The remaining fields are the measured
    evidence the controller logs on ``RELEASE_CONFIRMED`` / ``RELEASE_NOT_CONFIRMED``."""
    action: str
    reissue_control: bool
    log: Optional[str]
    reason: str
    ticks: int
    elapsed_s: float
    held: bool
    gripper_open: bool
    gripper_status_fresh: bool


class ReleaseReissueGate:
    """Per-set-down release-open re-issue decider (pure; the controller owns the actual control emit and
    the cuMotion detach).

    Construct once per controller when ``carton_level.release_reissue`` is on; :meth:`reset` it when the
    first gripper-open is emitted (so the bounded window starts at the first open, not controller
    construction). :meth:`observe` folds one release tick of the measured signals and returns a
    :class:`ReleaseReissueDecision`.

    Confirmation uses the PR 358 per-tick feedback:
      * carton NOT held: ``not held`` -- the ``s.contact`` latch cleared, so the ``SIM_ATTACH_FALLBACK``
        weld / native suction released (the authoritative physical seal-released signal).
      * gripper OPEN: when a FRESH gripper status is present (``gripper_status_time`` is not None), the
        plugin reports Open (``gripper_enabled`` False); when no fresh status is present the feedback
        cannot CONTRADICT, so ``not held`` carries the verdict. This never confirms a still-held carton.
    """

    def __init__(self, *, confirm_timeout_s: float, log_every: int = 30):
        if not (isinstance(confirm_timeout_s, (int, float)) and math.isfinite(confirm_timeout_s)
                and confirm_timeout_s > 0):
            raise ValueError('release_confirm_timeout_s must be a finite positive number of sim seconds')
        self.confirm_timeout_s = float(confirm_timeout_s)
        self.log_every = max(1, int(log_every))
        self.reset(None)

    def reset(self, now: Optional[float]) -> None:
        """Re-arm the gate for a set-down. ``now`` seeds the bounded window start (None => seed lazily on
        the first :meth:`observe`)."""
        self._start: Optional[float] = None if now is None else float(now)
        self._ticks = 0
        self.confirmed_ticks: Optional[int] = None

    @staticmethod
    def _gripper_open(gripper_enabled, gripper_status_time):
        """(gripper_open, status_fresh). A fresh status (``gripper_status_time`` not None) reports Open as
        ``gripper_enabled`` False; with no fresh status the feedback cannot contradict (gripper_open True,
        so ``not held`` decides). Pure; tolerant of None."""
        fresh = gripper_status_time is not None
        if not fresh:
            return True, False
        return (not bool(gripper_enabled)), True

    def observe(self, *, now: float, held: bool, gripper_enabled: bool,
                gripper_status_time) -> ReleaseReissueDecision:
        """Fold one release tick and decide. The first call seeds the window start if :meth:`reset` was
        not given a ``now``."""
        now = float(now)
        if self._start is None:
            self._start = now
        self._ticks += 1
        elapsed = max(0.0, now - self._start)
        gripper_open, fresh = self._gripper_open(gripper_enabled, gripper_status_time)
        carton_free = not bool(held)
        if carton_free and gripper_open:
            self.confirmed_ticks = self._ticks
            return ReleaseReissueDecision(
                'confirmed', False, None,
                'gripper open + carton not held confirmed after %d tick(s)' % self._ticks,
                self._ticks, elapsed, bool(held), gripper_open, fresh)
        if elapsed > self.confirm_timeout_s:
            return ReleaseReissueDecision(
                'timeout', False, None,
                'release not confirmed within %.1fs sim (held=%s gripper_open=%s status_fresh=%s)'
                % (self.confirm_timeout_s, bool(held), gripper_open, fresh),
                self._ticks, elapsed, bool(held), gripper_open, fresh)
        log = None
        if self._ticks % self.log_every == 1:
            log = ('RELEASE_REISSUE n=%d elapsed=%.2fs held=%s gripper_open=%s status_fresh=%s'
                   % (self._ticks, elapsed, bool(held), gripper_open, fresh))
        return ReleaseReissueDecision(
            'reissue', True, log, 'carton still held -- re-emit gripper-open + weld-detach',
            self._ticks, elapsed, bool(held), gripper_open, fresh)
