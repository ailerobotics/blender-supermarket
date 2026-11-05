"""Human-aware RUN/SLOW/STOP/BACKSTOP authority for the simulated restock arm
(sim-arm-reactive-replan-v1, P1).

This is a near-verbatim port of the real cell's plan-only safety gate
(``ur10-setup/humangate/gate_core.py`` + ``camera_state.py`` + ``gate_config.py``),
re-homed onto the simulated arm and fed by the ALREADY vendored human-safety
evaluator in ``scripts/twin/humans`` (whose ``SafetyState`` carries the
CLEAR/SLOW/STOP zone + nearest distance + health). The gate adds the motion
AUTHORITY the vendored evaluator does not: a fail-toward-stop RUN/SLOW/STOP/
BACKSTOP decision with a self-observed resume dwell, the "lost-close never
resumes" refusal, and the STOP->BACKSTOP fall-through. Per the perception-
architecture scout (sim-perception-architecture-s1 §6/§4) the gate is the SINGLE
motion authority for people and the nvblox map is NEVER the stop mechanism.

Design invariants, all fail-toward-stop (identical to the real cell):
  * absence of evidence is NEVER clearance;
  * RUN is reachable ONLY through an explicit healthy CLEAR held for the full
    dwell the gate observes ITSELF (defence in depth over the camera-side dwell);
  * an unknown distance is never treated as "far";
  * a person LOST while close does not enable a resume; only a person TRACKED
    leaving (...->SLOW->CLEAR) does;
  * after a STOP the gate escalates to BACKSTOP if the arm is not confirmed
    stopped within a deadline, or on any gate self-fault.

PURE: standard library only. All time is passed in (sim seconds for timeouts/
dwell; a wall/evidence clock for staleness/skew), so every rule is unit-tested
offline and replayed. Disabled (``ArmHumanGate`` simply not constructed by the
controller when the HOST key is off) => byte-identical.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import FrozenSet, Optional, Tuple

# --- camera-side zone enum (mirrors humans.perception.messages.Zone values) ---
ZONE_CLEAR = "CLEAR"
ZONE_SLOW = "SLOW"
ZONE_STOP = "STOP"
ZONES = (ZONE_CLEAR, ZONE_SLOW, ZONE_STOP)

# Health flag the gate keys on by name for the lost-close resume refusal.
HEALTH_TRACK_LOST_CLOSE = "TRACK_LOST_CLOSE"

# --- the gate's own action vocabulary ---
RUN = "RUN"
SLOW = "SLOW"
STOP = "STOP"
BACKSTOP = "BACKSTOP"
ACTIONS = (RUN, SLOW, STOP, BACKSTOP)

# --- how the most recent CLEAR was reached (drives the resume refusal) ---
PROV_CLEAN_START = "CLEAN_START"
PROV_TRACKED_LEAVING = "TRACKED_LEAVING"
PROV_LOST_CLOSE = "LOST_CLOSE"


@dataclass(frozen=True)
class ArmSafetyFrame:
    """One human-safety sample as the arm gate consumes it (the port of
    ``camera_state.CameraState``). ``zone`` is the CLEAR/SLOW/STOP judgement the
    vendored evaluator already computed from the reach envelope; the gate never
    re-thresholds distance (it only reads the zone + honours an unknown distance).
    ``stamp`` is the EVIDENCE time (when the sensing happened, not publication)."""

    stamp: float
    zone: str
    nearest_distance_m: Optional[float] = None
    health_ok: bool = True
    health_flags: FrozenSet[str] = field(default_factory=frozenset)
    heartbeat_seq: int = 0

    def __post_init__(self) -> None:
        if self.zone not in ZONES:
            raise ValueError(f"zone {self.zone!r} not one of {ZONES}")
        if self.nearest_distance_m is not None and isinstance(self.nearest_distance_m, float) \
                and math.isnan(self.nearest_distance_m):
            object.__setattr__(self, "nearest_distance_m", None)

    @property
    def distance_known(self) -> bool:
        return self.nearest_distance_m is not None

    @property
    def is_clear(self) -> bool:
        return self.zone == ZONE_CLEAR and self.health_ok and not self.health_flags


@dataclass(frozen=True)
class ArmGateConfig:
    """Gate thresholds. Conservative defaults carried over from the real cell
    (``humangate/gate_params.yaml``); tune UP, never down, until measured."""

    # interface freshness / liveness
    state_max_age_s: float = 0.200
    heartbeat_timeout_s: float = 0.300
    clock_skew_tolerance_s: float = 0.050
    # resume gating: continuous healthy-CLEAR the gate must observe ITSELF
    clear_dwell_s: float = 2.0
    # speed-slider fractions (RUN = operating, SLOW = low, STOP = zero)
    operating_speed_fraction: float = 1.0
    slow_speed_fraction: float = 0.1
    stop_speed_fraction: float = 0.0
    # STOP -> BACKSTOP fall-through: cancel the goal if the arm is not confirmed
    # stopped within this deadline.
    backstop_deadline_s: float = 0.5
    # Clean-start semantics (mirrors humans.reaction.ReactionController): when
    # False (the arm default), a mission that has NEVER been restricted runs at
    # full speed from tick one — the self-observed ``clear_dwell_s`` is required
    # only to RESUME out of a SLOW/STOP/BACKSTOP restriction, not to earn the very
    # first RUN. When True, the real-cell humangate behaviour applies: every RUN,
    # including the first, is earned through a full healthy-CLEAR dwell.
    require_initial_dwell: bool = False

    def validate(self) -> "ArmGateConfig":
        if not (0.0 < self.state_max_age_s <= 1.0):
            raise ValueError("state_max_age_s must be in (0, 1] s")
        if not (0.0 < self.heartbeat_timeout_s <= 2.0):
            raise ValueError("heartbeat_timeout_s must be in (0, 2] s")
        if self.heartbeat_timeout_s < self.state_max_age_s:
            raise ValueError("heartbeat_timeout_s must be >= state_max_age_s")
        if self.clock_skew_tolerance_s < 0.0:
            raise ValueError("clock_skew_tolerance_s must be >= 0")
        if self.clear_dwell_s <= 0.0:
            raise ValueError("clear_dwell_s must be > 0")
        if not (0.0 <= self.stop_speed_fraction < self.slow_speed_fraction
                <= self.operating_speed_fraction <= 1.0):
            raise ValueError("speed fractions must satisfy 0 <= stop < slow <= operating <= 1")
        if self.backstop_deadline_s <= 0.0:
            raise ValueError("backstop_deadline_s must be > 0")
        return self


@dataclass(frozen=True)
class ArmGateDecision:
    """One gate decision: everything the controller needs to act + log."""

    action: str                       # one of ACTIONS
    reason: str
    speed_fraction: Optional[float]   # slider fraction for RUN/SLOW/STOP; None for BACKSTOP
    demand: str                       # pre-dwell/pre-backstop demand (STOP/SLOW/CLEAR)
    clear_dwell_s: float
    provenance: str
    healthy: bool

    @property
    def is_run(self) -> bool:
        return self.action == RUN

    @property
    def arm_hold(self) -> bool:
        """True when the arm must not move (STOP or BACKSTOP). SLOW keeps moving
        (scaled); RUN runs."""
        return self.action in (STOP, BACKSTOP)


class ArmHumanGate:
    """Stateful decision core. Feed it frames with on_state(); ask decide()."""

    def __init__(self, config: Optional[ArmGateConfig] = None) -> None:
        self.cfg = (config or ArmGateConfig()).validate()
        self._latest: Optional[ArmSafetyFrame] = None
        self._latest_arrival: Optional[float] = None
        self._ever_received = False
        self._last_seq: Optional[int] = None
        self._last_seq_change: Optional[float] = None
        self._clear_since: Optional[float] = None
        self._stop_since: Optional[float] = None
        self._backstop_latched = False
        self._self_fault: Optional[str] = None
        self._provenance = PROV_CLEAN_START
        self._prev_zone: Optional[str] = None
        self._lost_close_flag_seen = False
        # Restriction latch (clean-start semantics): True once the gate has ever
        # demanded SLOW/STOP/BACKSTOP. Until then a healthy CLEAR runs immediately
        # (no startup dwell); after a restriction the resume dwell applies.
        self._ever_restricted = False

    # ---- inputs -----------------------------------------------------------

    def on_state(self, state: ArmSafetyFrame, now: float,
                 now_wall: Optional[float] = None) -> None:
        """Ingest one freshly-arrived safety frame. ``now`` is sim/monotonic
        seconds; ``now_wall`` is the clock the evidence ``stamp`` is in (staleness/
        skew). When ``now_wall`` is None, per-message staleness is skipped (the
        caller's freshness gate owns it)."""
        self._ever_received = True
        self._latest = state
        self._latest_arrival = now
        if self._last_seq is None or state.heartbeat_seq != self._last_seq:
            self._last_seq = state.heartbeat_seq
            self._last_seq_change = now
        self._update_provenance(state)
        self._prev_zone = state.zone
        # Dwell continuity is a property of EVERY received message, not just the
        # decide-tick samples: any non-trusted-CLEAR breaks the continuous-CLEAR
        # run immediately, even if a later CLEAR overwrites it before the next tick.
        if not self._is_trusted_clear(state, now_wall):
            self._clear_since = None

    def inject_self_fault(self, reason: str) -> None:
        """Flag an internal fault (decide loop overran, exception caught). Forces
        BACKSTOP until a healthy resume."""
        self._self_fault = reason

    def _is_trusted_clear(self, state: ArmSafetyFrame, now_wall: Optional[float]) -> bool:
        if not state.is_clear:
            return False
        if now_wall is not None:
            age = now_wall - state.stamp
            if age > self.cfg.state_max_age_s:
                return False
            if age < -self.cfg.clock_skew_tolerance_s:
                return False
        return True

    # ---- provenance -------------------------------------------------------

    def _update_provenance(self, state: ArmSafetyFrame) -> None:
        if state.zone != ZONE_CLEAR and HEALTH_TRACK_LOST_CLOSE in state.health_flags:
            self._lost_close_flag_seen = True
        if state.zone != ZONE_CLEAR:
            return
        if self._prev_zone == ZONE_CLEAR:
            return
        if self._prev_zone == ZONE_STOP or self._lost_close_flag_seen:
            self._provenance = PROV_LOST_CLOSE
        elif self._prev_zone == ZONE_SLOW:
            self._provenance = PROV_TRACKED_LEAVING
        else:
            self._provenance = PROV_CLEAN_START
        self._lost_close_flag_seen = False

    # ---- decision ---------------------------------------------------------

    def _demand(self, now: float, now_wall: float) -> Tuple[str, bool, str]:
        c = self.cfg
        if not self._ever_received or self._latest is None or self._latest_arrival is None:
            return STOP, False, "startup: no human-safety frame received yet"
        if (now - self._latest_arrival) > c.heartbeat_timeout_s:
            dt_ms = (now - self._latest_arrival) * 1e3
            return STOP, False, f"heartbeat timeout: no frame for {dt_ms:.0f} ms"
        if self._last_seq_change is not None and \
                (now - self._last_seq_change) > c.heartbeat_timeout_s:
            return STOP, False, "heartbeat_seq stalled (detector stuck)"
        s = self._latest
        age = now_wall - s.stamp
        if age > c.state_max_age_s:
            return STOP, False, f"stale frame: evidence {age * 1e3:.0f} ms old"
        if age < -c.clock_skew_tolerance_s:
            return STOP, False, f"clock skew: stamp {(-age) * 1e3:.0f} ms in the future"
        if not s.health_ok or s.health_flags:
            if s.zone == ZONE_SLOW:
                return SLOW, True, f"health fault, still tracking (SLOW) {sorted(s.health_flags)}"
            return STOP, True, f"health fault forcing STOP {sorted(s.health_flags)}"
        if s.zone == ZONE_STOP:
            return STOP, True, self._zone_reason("STOP", s)
        if s.zone == ZONE_SLOW:
            return SLOW, True, self._zone_reason("SLOW", s)
        return ZONE_CLEAR, True, "human-safety reports CLEAR and healthy"

    @staticmethod
    def _zone_reason(zone: str, s: ArmSafetyFrame) -> str:
        d = "unknown" if not s.distance_known else f"{s.nearest_distance_m:.2f} m"
        return f"zone {zone}, nearest person {d}"

    def decide(self, now: float, now_wall: float,
               driver_stopped: Optional[bool] = None) -> ArmGateDecision:
        """Produce a decision. ``now`` is sim/monotonic seconds (timeouts/dwell);
        ``now_wall`` the evidence clock (staleness/skew); ``driver_stopped`` the
        injected arm-stopped feedback (True/False/None)."""
        c = self.cfg
        demand, healthy, reason = self._demand(now, now_wall)

        if demand == ZONE_CLEAR:
            if self._provenance == PROV_LOST_CLOSE:
                self._clear_since = None
                primary, reason = STOP, (
                    "CLEAR not trusted: last person was lost close to the arm, "
                    "not tracked leaving — holding until a clean tracked departure")
            elif not self._ever_restricted and not c.require_initial_dwell:
                # Clean start: the arm has never been restricted, so a healthy
                # CLEAR runs immediately — the resume dwell gates only RELEASE from
                # a restriction, not the very first RUN (humans.reaction semantics).
                primary = RUN
                reason = "CLEAR healthy (clean start, no prior restriction)"
            else:
                if self._clear_since is None:
                    self._clear_since = now
                dwell = now - self._clear_since
                if dwell >= c.clear_dwell_s:
                    primary = RUN
                    reason = f"CLEAR healthy for {dwell:.2f} s >= dwell {c.clear_dwell_s:.2f} s"
                else:
                    primary = STOP
                    reason = (f"CLEAR dwell {dwell:.2f}/{c.clear_dwell_s:.2f} s — "
                              "holding until the resume dwell completes")
        else:
            self._clear_since = None
            primary = demand

        if primary in (SLOW, STOP, BACKSTOP):
            self._ever_restricted = True

        dwell_s = 0.0 if self._clear_since is None else (now - self._clear_since)
        provenance = self._provenance

        hazard_stop = (demand == STOP)
        if hazard_stop:
            if self._stop_since is None:
                self._stop_since = now
        else:
            self._stop_since = None

        if primary == RUN:
            if self._backstop_latched:
                reason = reason + " (resume requires a fresh plan after backstop)"
            self._backstop_latched = False
            self._self_fault = None
            action = RUN
            speed = c.operating_speed_fraction
        else:
            action = primary
            speed = c.slow_speed_fraction if primary == SLOW else c.stop_speed_fraction
            if self._self_fault is not None:
                self._backstop_latched = True
                reason = f"gate self-fault: {self._self_fault}"
            if hazard_stop and self._stop_since is not None and \
                    (now - self._stop_since) > c.backstop_deadline_s and \
                    driver_stopped is not True:
                self._backstop_latched = True
                conf = "still moving" if driver_stopped is False else "unconfirmed"
                reason = (f"primary STOP not confirmed within {c.backstop_deadline_s:.2f} s "
                          f"(arm {conf}) — escalating to backstop")
            if self._backstop_latched:
                action = BACKSTOP
                speed = None

        return ArmGateDecision(
            action=action, reason=reason, speed_fraction=speed, demand=demand,
            clear_dwell_s=dwell_s, provenance=provenance, healthy=healthy,
        )


def frame_from_safety_state(state, now_wall: float, *, heartbeat_seq: int = 0) -> ArmSafetyFrame:
    """Adapt a vendored ``humans.perception.messages.SafetyState`` (or any object
    exposing ``zone`` (a ``Zone`` enum or str), ``nearest_distance`` and an
    optional ``evidence_stamp``/``health_ok``/``health_flags``) onto the gate's
    :class:`ArmSafetyFrame`. The vendored evaluator already computed the zone from
    the reach envelope and fails toward STOP, so the gate consumes its verdict
    directly without re-thresholding distance. ``None`` state => a STOP frame
    (absence of evidence is never clearance)."""
    if state is None:
        return ArmSafetyFrame(stamp=now_wall, zone=ZONE_STOP, health_ok=False,
                              health_flags=frozenset({"no_state"}),
                              heartbeat_seq=heartbeat_seq)
    zone = getattr(state, "zone", ZONE_STOP)
    zone = getattr(zone, "value", zone)  # Zone enum -> str
    if zone not in ZONES:
        zone = ZONE_STOP
    dist = getattr(state, "nearest_distance", None)
    if dist is not None and (not isinstance(dist, (int, float)) or not math.isfinite(dist)):
        dist = None  # inf/NaN => unknown (never "far")
    stamp = getattr(state, "evidence_stamp", None)
    if stamp is None or not isinstance(stamp, (int, float)) or not math.isfinite(stamp):
        stamp = now_wall
    health_ok = bool(getattr(state, "health_ok", True))
    flags = getattr(state, "health_flags", None) or frozenset()
    return ArmSafetyFrame(stamp=float(stamp), zone=zone,
                          nearest_distance_m=(float(dist) if dist is not None else None),
                          health_ok=health_ok, health_flags=frozenset(flags),
                          heartbeat_seq=heartbeat_seq)
