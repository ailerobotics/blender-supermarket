"""Final fine-docking servo: close the nav→attach residual on the MEASURED hitch pose.

Navigation parks the base at the `cart_dock` goal within its arrival tolerance (0.08 m /
0.08 rad). The physical coupling `attach` (``twin.mobile.Coupler.attach``) is far tighter —
hitch-point distance ≤ 25 mm, hitch-z ≤ 15 mm, yaw ≤ 5°, fully stationary — because it models
a real hitch, and those tolerances are NOT ours to loosen. This servo spans the gap: once the
couple nav goal is reached (with the cart pinned to the world through the whole approach, so it
cannot be shoved — see ``mobile_runtime`` cart-hold), a slow closed-loop creep drives the robot's
rear hitch onto the cart's drawbar using the per-tick MEASURED poses, then hands over to `attach`.

Controller: a translation-dominant inverse-hitch-Jacobian creep. The rear hitch sits rigidly
``hitch_x`` behind the base; its planar velocity is a non-singular linear map of (v, ω)
(determinant ``hitch_x`` ≠ 0), so a proportional law on the measured hitch error yields

    v = -gain_lin · (hitch_error · heading)                       # along-heading, REVERSE-capable
    ω = -(gain_lin / hitch_x) · (hitch_error · heading_perp) - gain_yaw · yaw_error

which commands mostly translation (small ω) toward the drawbar. It NEVER rotates in place — a
real differential base does not pivot about its measured centre, so an in-place spin drifts the
base and never converges (observed on cn-002). Bounded by ``time_cap_s``; on expiry it fails
HONESTLY with the residuals it could not close, never by loosening attach.

Motion only: this emits a base velocity command (consumed by the same
``mobile.command(v, omega, dock, transport_ready=)`` seam navigation uses, with ``hold_cart=True``
so the cart stays pinned while the base creeps). The physical latch in ``mobile.py`` stays owned
by the coupling-joint side. Config-driven and default-off: absent ``config['fine_dock']`` ⇒ no
servo ⇒ the couple stage attaches directly, byte-identical. The sim and real-rig creep/gain/
tolerance values live in SEPARATE config blocks.

The default creep is the translation-dominant inverse-hitch-Jacobian law above. An OPT-IN
``approach_mode: 'crosstrack'`` (default ``'hitch_jacobian'`` ⇒ byte-identical) swaps it for a
cross-track servo of the rear hitch onto the cart axis (``_crosstrack``), which removes the legacy
law's degenerate zero-command equilibrium LINE -- the coldlivefinal couple failure, where the creep
limit-cycled at ~26 mm / ~4.8° yaw (just outside the 4° yaw tol) and never latched.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from twin.mobile import MobileSpec

# The attach handshake this servo must get inside (twin.mobile.Coupler.attach); the servo's own
# success tolerances must be strictly tighter so a 'ready' verdict actually satisfies attach.
_ATTACH_HITCH_M = 0.025
_ATTACH_YAW_RAD = math.radians(5)


def wrap(a):
    return math.atan2(math.sin(a), math.cos(a))


@dataclass(frozen=True)
class FineDockConfig:
    """Sim OR real-rig fine-docking tuning (keep the two in separate config blocks)."""
    creep_speed: float          # max linear creep magnitude (m/s)
    creep_yaw_rate: float       # max angular creep magnitude (rad/s)
    gain_lin: float             # P gain: hitch-error-along-heading -> linear
    gain_yaw: float             # P gain: yaw error -> angular
    hitch_tol_m: float          # 'ready' when hitch residual <= this (must be < 25 mm)
    yaw_tol_rad: float          # and |yaw residual| <= this (must be < 5°)
    settle_speed: float         # measured speeds/yaw-rates <= this => stationary (for 'ready')
    time_cap_s: float           # bounded; past this the servo fails with the live residuals
    # --- OPTIONAL terminal-docking mode (default 'hitch_jacobian' == the legacy law, byte-identical) ---
    # 'crosstrack' swaps the creep command for a cross-track servo on the cart axis (see _crosstrack):
    # it removes the legacy law's degenerate zero-command equilibrium line (``lat = (gain_yaw*|hitch_x|/
    # gain_lin)*yaw_err``, where ``w==0`` while NEITHER hitch nor yaw is seated -- the coldlivefinal
    # limit cycle at ~26 mm / ~4.8°). The extra ``ct_*`` gains are only read in that mode.
    approach_mode: str = 'hitch_jacobian'
    ct_gain_approach: float = 2.0   # P gain: along-axis distance -> reverse creep toward the drawbar
    ct_gain_cross: float = 8.0      # cross-track gain: hitch off the cart axis -> steer
    ct_gain_yaw: float = 1.5        # heading-error gain (base yaw vs cart yaw) -> steer
    ct_align_hold_m: float = 3.0    # hold-back distance per unit (|cross|+0.3|yaw|): align BEFORE seating
    ct_min_creep: float = 0.015     # minimum along-axis creep while unseated (beats the drive floor)
    # --- OPTIONAL progress-based cap (default off => the fixed ``time_cap_s`` above, byte-identical) ---
    # ``time_cap_s`` is a blunt fixed deadline: coupledock's cross-track creep was CONVERGING monotonically
    # (469 -> 88 mm at ~13 mm/s) and the fixed 30 s cut it off mid-approach, while coldlivefinal's limit
    # cycle (stuck ~26 mm / ~4.8°) burned the full 30 s before failing. ``progress_cap`` replaces the fixed
    # deadline with: keep creeping while the measured residual keeps IMPROVING (the hitch-equivalent metric
    # ``hitch + |yaw|*|hitch_x|`` drops by >= ``progress_min_improve_m`` within the last
    # ``progress_patience_s`` SIM seconds), bounded by a hard total ``progress_max_time_cap_s``. A genuine
    # monotone approach runs to the latch; a limit cycle fails FAST (at the patience window, not the hard
    # cap) with its live residual. SIM-clock throughout (``freshness-sim-clock-not-wall``).
    progress_cap: bool = False
    progress_patience_s: float = 12.0
    progress_min_improve_m: float = 0.002
    progress_max_time_cap_s: float = 120.0

    def __post_init__(self):
        vals = (self.creep_speed, self.creep_yaw_rate, self.gain_lin, self.gain_yaw,
                self.hitch_tol_m, self.yaw_tol_rad, self.settle_speed, self.time_cap_s)
        if not all(math.isfinite(v) and v > 0 for v in vals):
            raise ValueError('fine_dock values must be finite and positive')
        if self.hitch_tol_m >= _ATTACH_HITCH_M:
            raise ValueError(f'fine_dock hitch_tol_m must be < attach {_ATTACH_HITCH_M} m')
        if self.yaw_tol_rad >= _ATTACH_YAW_RAD:
            raise ValueError(f'fine_dock yaw_tol_rad must be < attach {math.degrees(_ATTACH_YAW_RAD):.0f}°')
        if self.approach_mode not in ('hitch_jacobian', 'crosstrack'):
            raise ValueError("fine_dock approach_mode must be 'hitch_jacobian' or 'crosstrack'")
        if self.approach_mode == 'crosstrack':
            ct = (self.ct_gain_approach, self.ct_gain_cross, self.ct_gain_yaw,
                  self.ct_align_hold_m, self.ct_min_creep)
            if not all(math.isfinite(v) and v > 0 for v in ct):
                raise ValueError('fine_dock crosstrack gains must be finite and positive')
            if self.ct_min_creep >= self.creep_speed:
                raise ValueError('fine_dock ct_min_creep must be < creep_speed')
        if self.progress_cap:
            pc = (self.progress_patience_s, self.progress_min_improve_m, self.progress_max_time_cap_s)
            if not all(math.isfinite(v) and v > 0 for v in pc):
                raise ValueError('fine_dock progress_cap values must be finite and positive')
            if self.progress_max_time_cap_s < self.time_cap_s:
                raise ValueError('fine_dock progress_max_time_cap_s must be >= time_cap_s')


@dataclass(frozen=True)
class FineDockResult:
    status: str  # 'creeping' | 'ready' | 'failed'
    linear: float = 0.0
    angular: float = 0.0
    hitch_residual_m: float = float('inf')
    yaw_residual_rad: float = float('inf')
    reason: str = ''


def residuals(dock, spec=MobileSpec()):
    """Measured hitch-point distance (m) and yaw misalignment (rad) for a DockObservation."""
    a = dock.robot.point(spec.hitch_x)
    b = dock.cart.point(spec.cart_hitch_x)
    return math.dist(a, b), wrap(dock.robot.yaw - dock.cart.yaw)


def target_base(cart, spec=MobileSpec()):
    """Base pose (x, y, yaw) that seats the rear hitch exactly on the measured cart drawbar."""
    bx, by = cart.point(spec.cart_hitch_x)  # drawbar point in world
    return (bx - spec.hitch_x * math.cos(cart.yaw),
            by - spec.hitch_x * math.sin(cart.yaw), cart.yaw)


def standoff_from_cart(cart, offset_m, spec=MobileSpec()):
    """Base pose (x, y, yaw) ON the measured cart's hitch axis, ``offset_m`` short of seating.

    This is the fix for the couple approach landing the base against a STALE map-frame standoff:
    the authored couple_standoff carries a FIXED yaw (0 rad), but the pinned cart's measured yaw
    varies run to run (coldlivefinal -1.6°, coupledock +19.2°), so navigating to the fixed pose
    leaves the base misaligned to the real cart by that difference -- a residual the fine-dock creep
    cannot null (it lands on the ``ct_gain_cross*cross == ct_gain_yaw*yaw_err`` equilibrium). Seating
    the standoff on the MEASURED cart axis (base yaw = cart yaw, rear hitch held ``offset_m`` back along
    the axis so the servo creeps STRAIGHT in) makes the fine-dock start from a consistent, on-axis,
    yaw-aligned pose regardless of where the cart settled. ``offset_m`` > 0 holds the hitch back on the
    base side of the drawbar. Pure geometry; the physical attach + latch tolerances are unchanged."""
    if not (math.isfinite(offset_m) and offset_m > 0):
        raise ValueError('standoff_from_cart offset_m must be finite and positive')
    tbx, tby, tyaw = target_base(cart, spec)
    return (tbx + offset_m * math.cos(cart.yaw), tby + offset_m * math.sin(cart.yaw), tyaw)


def cart_relative_residuals(dock, spec=MobileSpec()):
    """Rear-hitch alignment to the measured cart axis: (hitch_m, cross_m, yaw_rad).

    ``cross`` is the signed offset of the rear hitch from the cart's drawbar axis and ``yaw`` the
    base-vs-cart heading error -- the two the standoff must get small so the fine-dock enters near the
    latch box (``hitch`` includes the deliberate along-axis standoff offset, so it is NOT the capture
    gate). Shares ``residuals``' hitch definition exactly."""
    hm, yr = residuals(dock, spec)
    cy = dock.cart.yaw
    nx, ny = -math.sin(cy), math.cos(cy)
    hx, hy = dock.robot.point(spec.hitch_x)
    dbx, dby = dock.cart.point(spec.cart_hitch_x)
    cross = (hx - dbx) * nx + (hy - dby) * ny
    return hm, cross, yr


class FineDock:
    """Translation-dominant inverse-hitch-Jacobian creep onto the measured cart drawbar."""

    def __init__(self, config: FineDockConfig, spec: MobileSpec = MobileSpec()):
        if not isinstance(config, FineDockConfig):
            raise ValueError('config must be a FineDockConfig')
        self.config, self.spec = config, spec
        self.t0 = None
        self._best_metric = float('inf')   # progress_cap: best (lowest) hitch-equivalent metric seen
        self._best_t = None                # and the sim time it was last improved

    def start(self, now):
        if not math.isfinite(now):
            raise ValueError('fine_dock start needs a finite clock')
        self.t0 = float(now)
        self._best_metric = float('inf')
        self._best_t = float(now)

    def _cap_failure(self, hm, yr, now):
        """The time-cap verdict: a FineDockResult('failed', ...) when the bound is hit, else None.

        Default (``progress_cap`` off): the fixed ``time_cap_s`` deadline, byte-identical. With
        ``progress_cap`` on: track the hitch-equivalent metric ``hitch + |yaw|*|hitch_x|`` (a yaw error
        shows up as the hitch displacement its lever arm causes); keep going while it keeps dropping,
        and fail at the patience window (a limit cycle) or the hard total cap (a genuinely too-slow
        approach), reporting the live residual either way. SIM-clock (``now``) throughout."""
        cfg = self.config
        if not cfg.progress_cap:
            if now - self.t0 > cfg.time_cap_s:
                return FineDockResult('failed', 0.0, 0.0, hm, yr,
                                      reason=f'fine_dock time cap {cfg.time_cap_s:g}s: residual '
                                             f'hitch={hm*1000:.1f}mm yaw={math.degrees(yr):.2f}deg')
            return None
        metric = hm + abs(yr) * abs(self.spec.hitch_x)
        if metric <= self._best_metric - cfg.progress_min_improve_m:
            self._best_metric, self._best_t = metric, now
        if now - self.t0 > cfg.progress_max_time_cap_s:
            return FineDockResult('failed', 0.0, 0.0, hm, yr,
                                  reason=f'fine_dock progress cap {cfg.progress_max_time_cap_s:g}s: residual '
                                         f'hitch={hm*1000:.1f}mm yaw={math.degrees(yr):.2f}deg')
        if now - self._best_t > cfg.progress_patience_s:
            return FineDockResult('failed', 0.0, 0.0, hm, yr,
                                  reason=f'fine_dock no progress for {cfg.progress_patience_s:g}s: residual '
                                         f'hitch={hm*1000:.1f}mm yaw={math.degrees(yr):.2f}deg')
        return None

    def _stationary(self, dock):
        vals = (dock.robot_speed, dock.cart_speed, dock.robot_yaw_rate, dock.cart_yaw_rate)
        return (all(math.isfinite(v) for v in vals)
                and math.isfinite(dock.age) and 0 <= dock.age <= 0.1
                and max(abs(v) for v in vals) <= self.config.settle_speed)

    def step(self, dock, now) -> FineDockResult:
        """One servo tick against the MEASURED dock observation."""
        if self.t0 is None:
            raise ValueError('fine_dock.step before start')
        cfg = self.config
        hm, yr = residuals(dock, self.spec)
        within = hm <= cfg.hitch_tol_m and abs(yr) <= cfg.yaw_tol_rad

        if within and self._stationary(dock):
            return FineDockResult('ready', 0.0, 0.0, hm, yr,
                                  reason='seated within attach tolerance, stopped')
        capped = self._cap_failure(hm, yr, now)
        if capped is not None:
            return capped
        if within:
            # Close enough but still moving: brake and let it settle (no new creep command).
            return FineDockResult('creeping', 0.0, 0.0, hm, yr, reason='settling within tolerance')

        if cfg.approach_mode == 'crosstrack':
            v, w, reason = self._crosstrack(dock, hm)
            return FineDockResult('creeping', v, w, hm, yr, reason=reason)

        # Inverse-hitch-Jacobian proportional creep toward the measured drawbar.
        b, hx_spec = dock.robot, self.spec.hitch_x
        hx, hy = b.point(hx_spec)
        dbx, dby = dock.cart.point(self.spec.cart_hitch_x)
        ex, ey = hx - dbx, hy - dby                       # hitch error (world)
        fwd = ex * math.cos(b.yaw) + ey * math.sin(b.yaw)     # along heading (reverse if +)
        lat = -ex * math.sin(b.yaw) + ey * math.cos(b.yaw)    # heading-perpendicular
        yaw_err = wrap(b.yaw - dock.cart.yaw)
        v = max(-cfg.creep_speed, min(cfg.creep_speed, -cfg.gain_lin * fwd))
        w = max(-cfg.creep_yaw_rate,
                min(cfg.creep_yaw_rate, -(cfg.gain_lin / hx_spec) * lat - cfg.gain_yaw * yaw_err))
        return FineDockResult('creeping', v, w, hm, yr,
                              reason=f'creep hitch={hm*1000:.0f}mm yaw={math.degrees(yaw_err):.1f}deg')

    def _crosstrack(self, dock, hm):
        """Cross-track servo of the REAR HITCH onto the cart's drawbar axis (opt-in approach mode).

        The legacy inverse-hitch-Jacobian law folds the lateral-hitch and yaw corrections into one
        ``w``; wherever ``lat = (gain_yaw*|hitch_x|/gain_lin)*yaw_err`` the two cancel (``w==0``) with
        NEITHER seated -- a whole equilibrium LINE outside the latch box. The approach lands on it at
        whatever yaw the standoff cross-track implies; coldlivefinal landed at ~4.8° (> the 4° yaw tol,
        hitch ~26 mm) and the creep could not null the last mm/degree (FINE_DOCK limit cycle).

        Here the rear hitch is the LEADING point of the reverse approach, so steering it onto the cart
        axis is a stable (minimum-phase) trailer-free cross-track problem -- forward motion provides the
        steering authority, so the only equilibrium is the seated pose:

            along = (hitch - drawbar) . cart_axis      # +ahead of the drawbar on the base side
            cross = (hitch - drawbar) . cart_normal    # signed offset from the axis
            v = -ct_gain_approach * (along - along_sp)  # reverse toward the drawbar, held back while
                                                        #   mis-aligned so cross/yaw null BEFORE seating
            w =  ct_gain_cross * cross - ct_gain_yaw * yaw_err

        ``along_sp = min(along, ct_align_hold_m*(|cross|+0.3|yaw_err|))`` keeps the hitch a touch short of
        the drawbar until it is on the axis and aligned, then finishes in; a ``ct_min_creep`` floor keeps
        the base moving (so steering stays live) until the residual check latches. Pure motion -- the
        latch tolerances and the physical attach are unchanged.
        """
        cfg, spec = self.config, self.spec
        cy = dock.cart.yaw
        ux, uy = math.cos(cy), math.sin(cy)               # cart hitch axis (forward)
        nx, ny = -math.sin(cy), math.cos(cy)              # cart axis normal
        dbx, dby = dock.cart.point(spec.cart_hitch_x)     # drawbar (world)
        hx, hy = dock.robot.point(spec.hitch_x)           # rear hitch (world)
        along = (hx - dbx) * ux + (hy - dby) * uy
        cross = (hx - dbx) * nx + (hy - dby) * ny
        yaw_err = wrap(dock.robot.yaw - cy)
        along_sp = min(along, cfg.ct_align_hold_m * (abs(cross) + 0.3 * abs(yaw_err)))
        v = max(-cfg.creep_speed, min(cfg.creep_speed, -cfg.ct_gain_approach * (along - along_sp)))
        if abs(v) < cfg.ct_min_creep and along > 0.003:   # keep creeping in so steering stays effective
            v = -cfg.ct_min_creep
        w = max(-cfg.creep_yaw_rate,
                min(cfg.creep_yaw_rate, cfg.ct_gain_cross * cross - cfg.ct_gain_yaw * yaw_err))
        return v, w, (f'crosstrack hitch={hm*1000:.0f}mm cross={cross*1000:.0f}mm '
                      f'yaw={math.degrees(yaw_err):.1f}deg')


def build(config, spec=None):
    """Build a FineDock from a `config['fine_dock']` object, or None when absent (disabled)."""
    cfg = config.get('fine_dock') if isinstance(config, dict) else None
    if cfg is None:
        return None
    if not isinstance(cfg, dict):
        raise ValueError('config.fine_dock must be an object')
    return FineDock(FineDockConfig(**cfg), spec=spec or MobileSpec())
