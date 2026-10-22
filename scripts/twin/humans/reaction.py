"""The slow/stop base + pause/resume arm reaction -- the safety MECHANISM.

This is a pure, deterministic controller that turns the published safety zone (from
:mod:`twin.humans.detection`) into three levers:

* ``base_scale``   -- a multiplier in ``[0, 1]`` applied to the navigation velocity
  command (1.0 = full, ``slow_base_scale`` while SLOW, ``stop_base_scale`` while
  STOP). The navigation lane may equivalently read ``costmap_inflation_m``.
* ``arm_hold``     -- ``True`` pauses arm motion (a person is inside the arm's reach
  envelope, or the zone is STOP); ``False`` lets it run.
* ``costmap_inflation_m`` -- optional extra inflation for the nav obstacle layer.

It is the *mechanism* the behaviour-tree lane (``sim-restock-bt-v1``) calls from its
``human detected -> slow/stop/pause/resume`` subtree, and the integration shim calls
from the mission adapter; it does NOT subscribe to topics or own a control loop.

Fail-toward-safe, mirroring the fleet gate (``humangate/gate_core.py``): a missing or
stale state is treated as STOP, and a return to full base speed / arm motion is
granted only after the clear condition holds continuously for ``resume_clear_dwell_s``
(defence-in-depth hysteresis over the camera-side dwell). All thresholds come from
:class:`twin.humans.config.ReactionConfig`; every one is a provisional placeholder,
and the owner decisions on the real numbers are OPEN (see the report).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

from .config import ReactionConfig
from .perception import SafetyState, Zone


@dataclass(frozen=True)
class ReactionOutput:
    base_scale: float
    arm_hold: bool
    costmap_inflation_m: float
    zone: str                     # the effective zone used (STOP when stale)
    stale: bool
    person_in_arm_envelope: bool
    base_clear_dwell_s: float
    arm_clear_dwell_s: float
    reason: str

    def to_log(self) -> dict:
        return {'base_scale': self.base_scale, 'arm_hold': self.arm_hold,
                'costmap_inflation_m': self.costmap_inflation_m, 'zone': self.zone,
                'stale': self.stale, 'person_in_arm_envelope': self.person_in_arm_envelope,
                'base_clear_dwell_s': round(self.base_clear_dwell_s, 3),
                'arm_clear_dwell_s': round(self.arm_clear_dwell_s, 3), 'reason': self.reason}


class ReactionController:
    """Stateful mechanism. Call :meth:`update` once per tick with the latest state."""

    def __init__(self, config: Optional[ReactionConfig] = None):
        self.cfg = config or ReactionConfig()
        # Restriction latches: the resume dwell gates RELEASE from a restricted
        # state, so a mission that starts clear (no people) runs at full speed with
        # the arm free from tick one; only once slowed/stopped/paused does the
        # continuous-CLEAR dwell apply before returning to full / resuming the arm.
        self._base_restricted = False
        self._arm_held = False
        self._base_clear_since: Optional[float] = None
        self._arm_clear_since: Optional[float] = None

    def _stale(self, state: Optional[SafetyState], now: float) -> bool:
        if state is None:
            return True
        age = now - state.stamp
        return not math.isfinite(age) or age > self.cfg.state_timeout_s

    def update(self, state: Optional[SafetyState], now: float) -> ReactionOutput:
        c = self.cfg
        stale = self._stale(state, now)
        if stale:
            zone = Zone.STOP
            in_env = True                       # unknown proximity -> assume inside, fail safe
            reason = ('no safety state yet' if state is None
                      else f'stale safety state ({now - state.stamp:.2f}s > '
                           f'{c.state_timeout_s:.2f}s) -> STOP')
        else:
            zone = state.zone
            d_base = state.nearest_distance_base
            in_env = (math.isfinite(d_base)
                      and d_base <= c.reach_radius_m + c.arm_hold_margin_m)
            reason = f'zone {zone.value}'
            if in_env:
                reason += f'; person {d_base:.2f} m from base (<= reach envelope)'

        # --- base speed: restrict on SLOW/STOP/stale, resume on CLEAR dwell ---
        base_clear = (not stale) and zone == Zone.CLEAR
        if base_clear and self._base_restricted:
            if self._base_clear_since is None:
                self._base_clear_since = now
            base_dwell = now - self._base_clear_since
            if base_dwell >= c.resume_clear_dwell_s:
                self._base_restricted = False
                self._base_clear_since = None
        else:
            if not base_clear:
                self._base_clear_since = None
            base_dwell = 0.0 if self._base_clear_since is None else now - self._base_clear_since

        if not base_clear:                      # SLOW / STOP / stale: slow or stop now, latch
            self._base_restricted = True
            if zone == Zone.STOP:
                base_scale, inflation = c.stop_base_scale, c.stop_inflation_m
            else:
                base_scale, inflation = c.slow_base_scale, c.slow_inflation_m
        elif self._base_restricted:             # CLEAR but still confirming the resume
            base_scale, inflation = c.slow_base_scale, c.slow_inflation_m
            reason += (f'; base resume dwell {base_dwell:.2f}/{c.resume_clear_dwell_s:.2f}s')
        else:                                   # CLEAR and resume confirmed (or never restricted)
            base_scale, inflation = 1.0, 0.0

        # --- arm pause: hold on in-envelope/STOP/stale, resume on clear dwell -
        arm_clear = (not stale) and zone != Zone.STOP and not in_env
        if arm_clear and self._arm_held:
            if self._arm_clear_since is None:
                self._arm_clear_since = now
            arm_dwell = now - self._arm_clear_since
            if arm_dwell >= c.resume_clear_dwell_s:
                self._arm_held = False
                self._arm_clear_since = None
        else:
            if not arm_clear:
                self._arm_clear_since = None
            arm_dwell = 0.0 if self._arm_clear_since is None else now - self._arm_clear_since

        if not arm_clear:                       # in reach envelope / STOP / stale: hold, latch
            self._arm_held = True
        arm_hold = self._arm_held
        if arm_hold and arm_clear:
            reason += (f'; arm resume dwell {arm_dwell:.2f}/{c.resume_clear_dwell_s:.2f}s')

        return ReactionOutput(
            base_scale=base_scale, arm_hold=arm_hold, costmap_inflation_m=inflation,
            zone=zone.value, stale=stale, person_in_arm_envelope=in_env,
            base_clear_dwell_s=base_dwell, arm_clear_dwell_s=arm_dwell, reason=reason)
