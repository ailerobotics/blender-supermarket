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

    def __post_init__(self):
        vals = (self.creep_speed, self.creep_yaw_rate, self.gain_lin, self.gain_yaw,
                self.hitch_tol_m, self.yaw_tol_rad, self.settle_speed, self.time_cap_s)
        if not all(math.isfinite(v) and v > 0 for v in vals):
            raise ValueError('fine_dock values must be finite and positive')
        if self.hitch_tol_m >= _ATTACH_HITCH_M:
            raise ValueError(f'fine_dock hitch_tol_m must be < attach {_ATTACH_HITCH_M} m')
        if self.yaw_tol_rad >= _ATTACH_YAW_RAD:
            raise ValueError(f'fine_dock yaw_tol_rad must be < attach {math.degrees(_ATTACH_YAW_RAD):.0f}°')


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


class FineDock:
    """Translation-dominant inverse-hitch-Jacobian creep onto the measured cart drawbar."""

    def __init__(self, config: FineDockConfig, spec: MobileSpec = MobileSpec()):
        if not isinstance(config, FineDockConfig):
            raise ValueError('config must be a FineDockConfig')
        self.config, self.spec = config, spec
        self.t0 = None

    def start(self, now):
        if not math.isfinite(now):
            raise ValueError('fine_dock start needs a finite clock')
        self.t0 = float(now)

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
        if now - self.t0 > cfg.time_cap_s:
            return FineDockResult('failed', 0.0, 0.0, hm, yr,
                                  reason=f'fine_dock time cap {cfg.time_cap_s:g}s: residual '
                                         f'hitch={hm*1000:.1f}mm yaw={math.degrees(yr):.2f}deg')
        if within:
            # Close enough but still moving: brake and let it settle (no new creep command).
            return FineDockResult('creeping', 0.0, 0.0, hm, yr, reason='settling within tolerance')

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


def build(config, spec=None):
    """Build a FineDock from a `config['fine_dock']` object, or None when absent (disabled)."""
    cfg = config.get('fine_dock') if isinstance(config, dict) else None
    if cfg is None:
        return None
    if not isinstance(cfg, dict):
        raise ValueError('config.fine_dock must be an object')
    return FineDock(FineDockConfig(**cfg), spec=spec or MobileSpec())
