"""Opt-in physics timestep for the Isaac restock mission (sim-realtime-factor-v1).

The mission runs one 1/60 s physics step per tick, and a tick costs 40-72 ms of wall even
when it renders nothing (physics solve + lidar + ROS graph + nav/manip). So the single lever
that cuts the *number* of those ticks for a given span of SIM time -- and therefore the wall
cost of the whole cycle -- is a coarser physics step. Every mission guard, settle window and
watchdog is already on the SIM clock (``elapsed += dt`` vs a SIM budget), so a coarser step
changes none of their behaviour per SIM second; it only changes PhysX contact fidelity, which
is why it is gated behind one opt-out key and a stated contact-tolerance check.

One top-level runtime-config key turns it on::

    "physics_dt": 0.025                 # seconds per physics step (40 Hz)
    "physics_dt": {"hz": 40}            # equivalently
    "physics_dt": {"dt": 0.025}         # equivalently

Absent (or ``null``) => 1/60 s => every path is byte-identical to before (the resolver returns
the exact float ``1/60`` and ``summary_fields`` is ``{}``). The value is threaded to
``World(physics_dt=, rendering_dt=)`` AND to the mission loop's ``tick(dt)`` / ``advance(dt)``
from the SAME config, and ``IsaacPort.advance`` asserts ``world.get_physics_dt() == dt`` -- so a
mis-thread fails the boot loudly instead of silently running a wrong step.

The contact-tolerance CHECK (``max_safe_physics_dt``): a coarser step advances every body by
``v * dt`` between solver updates, so for a CONTACT-APPROACH event the step must stay small enough
that the closing body still lands at least one solver sample inside the tolerance window (the
contact is not skipped). The mission's contact-approach events and their windows:

* the surface-gripper WELD descent vs the weld window (``SIM_ATTACH_CONTACT_MM``, 3 mm default):
  ``v_grasp_descent * dt <= safety * weld_window_m`` -- the finest, and the usual binding one;
* the couple fine-dock closing vs the latch box: ``v_dock_closing * dt <= safety * dock_tolerance_m``
  -- the fine-dock crawls (cm/s) into a ~cm box, so it is essentially never binding.

The HORIZONTAL tow speed (base ``rv``, ~0.56 m/s) is deliberately NOT a term here: a towed wheel
rests in persistent contact with the floor and translates ALONG it, so it is not a contact-approach
that can tunnel -- its vertical closing speed onto the floor is ~0. What a coarser step CAN hurt for
the coupled tow is solver STABILITY (the documented traction / break-away / caster-contact
sensitivity), which is not reducible to a per-step penetration formula; ``tow_step_mm`` reports the
per-step horizontal travel as an INFORMATIONAL stability indicator, and the honest acceptance is one
GPU confirmation that the coupled tow and the grasp weld are unchanged at the chosen step. Speeds
come from the measured run (``max_safe_physics_dt`` is PURE and unit tested; the report feeds it the
recorded grasp-descent / dock-closing speeds and states the values used).
"""
from __future__ import annotations

import math
import os

# Legacy / default step. Returned verbatim when the key is absent, so the boot path is byte-identical.
DEFAULT_DT = 1.0 / 60.0

# Env the launcher/host-session export with the RESOLVED physics step (seconds) so every process in
# the mission -- the Isaac/mission process AND the out-of-Isaac host children (controller, station-host
# incorporation ack, trajectory host) -- can rebase its per-frame freshness windows onto the running
# step without re-reading the runtime config. Absent => DEFAULT_DT => every freshness window is its
# legacy sim-seconds value (scale factor exactly 1.0 => byte-identical).
PHYSICS_DT_ENV = 'RESTOCK_PHYSICS_DT'

# Bounds on the configurable step. Finer than 1/240 buys nothing and coarser than 1/15 is past any
# plausible contact tolerance for this rig; a value outside refuses the boot rather than running it.
MIN_DT = 1.0 / 240.0
MAX_DT = 1.0 / 15.0

# Contact-tolerance defaults for the safety check (all overridable). The weld window mirrors
# restock_manipulation SIM_ATTACH_CONTACT_MM (sim-robust-weld-attach-v1, 3 mm). The grasp descent speed
# is a conservative stand-in for the slow contact straight-line descent (the weld gate bottoms ~0.67 mm
# from contact, so the descent is slow near contact); override it with a measured value when one is
# available. The dock latch box (~cm) and its crawl closing speed are essentially never binding.
WELD_WINDOW_M = 0.003
GRASP_DESCENT_MPS = 0.05
DOCK_TOLERANCE_M = 0.01
DOCK_CLOSING_MPS = 0.05
SAFETY = 0.5


def _as_dt(value, where):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        raise ValueError(f'{where} must be a positive, finite number of seconds')
    dt = float(value)
    if not (MIN_DT - 1e-12 <= dt <= MAX_DT + 1e-12):
        raise ValueError(f'{where}={dt:g}s is outside the allowed physics-step range '
                          f'[{MIN_DT:g}, {MAX_DT:g}] s (1/{1/MAX_DT:.0f} .. 1/{1/MIN_DT:.0f} Hz)')
    return dt


def resolve_dt(config):
    """The physics step (seconds) for this run from ``config['physics_dt']``.

    Absent / ``None`` => ``DEFAULT_DT`` (exactly ``1/60``) => byte-identical boot. A bare number is
    seconds; a dict accepts ``{"dt": seconds}`` or ``{"hz": frequency}`` (exactly one). Fails closed on a
    malformed value or an out-of-range step so a typo refuses the boot instead of running a wrong step."""
    block = config.get('physics_dt') if isinstance(config, dict) else None
    if block is None:
        return DEFAULT_DT
    if isinstance(block, (int, float)) and not isinstance(block, bool):
        return _as_dt(block, 'physics_dt')
    if isinstance(block, dict):
        has_dt, has_hz = 'dt' in block, 'hz' in block
        unknown = set(block) - {'dt', 'hz'}
        if unknown:
            raise ValueError(f'physics_dt has unknown keys {sorted(unknown)}; use {{"dt": s}} or {{"hz": f}}')
        if has_dt == has_hz:
            raise ValueError('physics_dt dict must set exactly one of "dt" (seconds) or "hz" (frequency)')
        if has_hz:
            hz = block['hz']
            if isinstance(hz, bool) or not isinstance(hz, (int, float)) or not math.isfinite(hz) or hz <= 0:
                raise ValueError('physics_dt.hz must be a positive, finite frequency')
            return _as_dt(1.0 / float(hz), 'physics_dt.hz')
        return _as_dt(block['dt'], 'physics_dt.dt')
    raise ValueError('physics_dt must be a number of seconds or {"dt": s} / {"hz": f}')


def is_default(dt):
    """Whether ``dt`` is the legacy 1/60 step (so summaries/labels stay byte-identical)."""
    return math.isclose(float(dt), DEFAULT_DT, abs_tol=1e-9)


def max_safe_physics_dt(*, v_grasp_descent_mps=GRASP_DESCENT_MPS, v_dock_closing_mps=DOCK_CLOSING_MPS,
                        weld_window_m=WELD_WINDOW_M, dock_tolerance_m=DOCK_TOLERANCE_M,
                        v_tow_mps=None, safety=SAFETY):
    """Largest physics step (seconds) that keeps every CONTACT-APPROACH event sampled within its
    tolerance window, given the measured/assumed closing speeds. PURE.

    The hard bound is ``min`` over the approach events (weld descent, dock closing). The horizontal
    tow speed (``v_tow_mps``, optional) is NOT a hard term -- a towed wheel rests on the floor and
    does not tunnel -- it only populates ``tow_step_mm`` as an informational stability indicator.
    Returns ``(dt_max, detail)``; ``dt_max`` is not clamped to ``MAX_DT`` (the caller states the
    margin against a chosen dt)."""
    for name, val in (('v_grasp_descent_mps', v_grasp_descent_mps),
                      ('v_dock_closing_mps', v_dock_closing_mps),
                      ('weld_window_m', weld_window_m), ('dock_tolerance_m', dock_tolerance_m),
                      ('safety', safety)):
        if isinstance(val, bool) or not isinstance(val, (int, float)) or not math.isfinite(val) or val <= 0:
            raise ValueError(f'max_safe_physics_dt: {name} must be a positive finite number')
    weld_dt = safety * weld_window_m / v_grasp_descent_mps
    dock_dt = safety * dock_tolerance_m / v_dock_closing_mps
    dt_max = min(weld_dt, dock_dt)
    detail = {
        'weld_dt': weld_dt, 'weld_hz': 1.0 / weld_dt,
        'dock_dt': dock_dt, 'dock_hz': 1.0 / dock_dt,
        'binding': 'weld' if weld_dt <= dock_dt else 'dock',
        'inputs': {'v_grasp_descent_mps': float(v_grasp_descent_mps),
                   'v_dock_closing_mps': float(v_dock_closing_mps),
                   'weld_window_m': float(weld_window_m), 'dock_tolerance_m': float(dock_tolerance_m),
                   'safety': float(safety)},
        'dt_max': dt_max, 'hz_max': 1.0 / dt_max,
    }
    if v_tow_mps is not None:
        if isinstance(v_tow_mps, bool) or not isinstance(v_tow_mps, (int, float)) or not math.isfinite(v_tow_mps) or v_tow_mps < 0:
            raise ValueError('max_safe_physics_dt: v_tow_mps must be a nonnegative finite number or None')
        detail['tow_step_mm'] = 1e3 * float(v_tow_mps) * dt_max
        detail['inputs']['v_tow_mps'] = float(v_tow_mps)
    return dt_max, detail


def describe(dt):
    """One boot-log line naming the step (so a raised step is visible at boot)."""
    return (f'PHYSICS_STEP dt={dt:.6g}s hz={1.0/dt:.3g} '
            f'{"(legacy 1/60)" if is_default(dt) else "(RAISED from 1/60 -- sim-realtime-factor-v1)"}')


def freshness_scale(dt):
    """Multiplier that rebases a per-FRAME freshness window expressed in sim-seconds-at-1/60 onto a
    run at physics step ``dt``. PURE. Exactly ``1.0`` at ``DEFAULT_DT`` (so a default run is
    byte-identical).

    Why a per-frame freshness window must scale with the step (sim-realtime-factor-v1 /
    sim-physics-step-collision-band-v1). The mission's sensor-freshness bands -- the collision/ESDF
    incorporation age (``collision_dt``), the arm-feedback age, and the one-to-few-step delivery-order
    "render lead" -- bound how STALE (or marginally future) a relayed stamp may be before the
    controller refuses to move the arm. Their binding constraint in this rig is NOT a sim-time safety
    margin (the collect/transfer scene is PARKED/static, so an older packet is strictly consistent --
    see controller.max_age fm476); it is the ESDF/feedback PRODUCTION CADENCE. nvblox integrates
    RENDERED depth frames, and ``rendering_dt`` tracks ``physics_dt`` (physics_setup.assemble), so one
    rendered frame spans exactly ``dt`` sim-seconds. The real physical quantity each band encodes is a
    FRAME-COUNT budget -- "tolerate up to N un-refreshed frames of staleness" -- and the sim-seconds
    band is ``N * dt``. A coarser step stretches every frame's sim-time footprint, so a fixed
    sim-seconds band silently SHRINKS the frame budget and false-trips a healthy-but-coarser pipeline
    (the rtfcollect failure: ``collision_dt=1.5278`` over a 1.5 s band = the SAME ~55-frame staleness
    that is 0.92 s and well inside the band at 1/60). Scaling the sim-seconds band by ``dt / (1/60)``
    holds the frame budget constant across the step -- it is NOT a widening: the frame-count margin is
    identical, only re-expressed in the running clock. ``dt`` is bounded [MIN_DT, MAX_DT] so the factor
    is bounded [0.25, 4.0]."""
    d = float(dt)
    if not math.isfinite(d) or d <= 0:
        return 1.0
    return d / DEFAULT_DT


# sim-collision-dt-render-lead-floor-v1: the collision/ESDF incorporation RELAY-LEAD window (seconds at
# 1/60). The mirrored collision stamp the controller reads (``collision_time``) is nvblox's ESDF
# integration time, which rides the Isaac RENDER clock; under heavy live_pose render pacing that clock
# runs AHEAD of the host packet clock the controller reads as ``now``, so a perfectly FRESH collision
# stamp legitimately LEADS ``now`` (is marginally future). coldwin30 tripped the collect preplace with
# ``collision_dt=-0.5333`` against a hand-set -0.5 floor -- the collision world led ``now`` by 0.53 s,
# which is FRESHER, not stale. The lead is bounded by one incorporation window on top of the
# one-to-few-step delivery render lead, and like every other freshness window it is a per-FRAME budget
# (nvblox integrates rendered frames; ``rendering_dt`` tracks ``physics_dt``), so it scales with the
# step via ``freshness_scale``. 0.5 s matches the nvblox incorporation freshness window
# (``restock_host.incorporation_ack._esdf_fresh_window_s`` / ``isaac_sensors`` ``esdf_fresh_window``).
COLLISION_RELAY_LEAD_S = 0.5


def collision_band_floor(render_lead_base_s, dt):
    """Magnitude (>= 0 seconds) of the controller collision/feedback freshness-band FLOOR: how far a
    stamp may LEAD ``now`` (be future) and still count as fresh. PURE.

    DERIVED (sim-collision-dt-render-lead-floor-v1) from the configured delivery render lead plus the
    step-scaled ESDF incorporation window, so the heavy-live_pose relay lead (measured ~0.53 s at the
    default step, coldwin30) sits inside the band WITHOUT a hand-set ``RESTOCK_FRESH_RENDER_LEAD_S``
    override::

        floor = (render_lead_base_s + COLLISION_RELAY_LEAD_S) * freshness_scale(dt)

    At the DEFAULT step (1/60) and the default delivery lead (0.2 s) this is 0.7 s -- exactly the value
    the integration lane had to bump the env to for coldwin31. It GENERALISES PR 288's step-aware rule
    (``freshness_scale``) to the default step: the incorporation-window term contributes 0.5 s even at
    1/60, where the pure ``base * freshness_scale`` scaling was a no-op and left the floor at the bare
    delivery lead. Only this (future-lead) floor gains headroom; the POSITIVE (stale) ceiling -- the
    safety side -- is unchanged, so a genuinely stale stamp still trips. A non-finite/negative base is
    treated as 0."""
    base = float(render_lead_base_s)
    if not math.isfinite(base) or base < 0:
        base = 0.0
    return (base + COLLISION_RELAY_LEAD_S) * freshness_scale(dt)


def dt_from_env(environ=None):
    """The resolved physics step (seconds) a process should scale its freshness windows by, read from
    the ``RESTOCK_PHYSICS_DT`` env the launcher/host-session export. Absent/malformed/out-of-range =>
    ``DEFAULT_DT`` (factor 1.0 => byte-identical), never raises -- a freshness read must not crash a
    mission tick on a bad env."""
    env = os.environ if environ is None else environ
    raw = env.get(PHYSICS_DT_ENV)
    if raw is None:
        return DEFAULT_DT
    try:
        return _as_dt(float(raw), PHYSICS_DT_ENV)
    except (TypeError, ValueError):
        return DEFAULT_DT


def freshness_scale_from_env(environ=None):
    """``freshness_scale(dt_from_env())`` -- the per-frame freshness multiplier for THIS process. 1.0
    when ``RESTOCK_PHYSICS_DT`` is absent/default => byte-identical."""
    return freshness_scale(dt_from_env(environ))


def scale_freshness_window(base_s, dt):
    """A sim-seconds freshness window ``base_s`` (its value AT 1/60) rebased onto step ``dt``. PURE.
    ``base_s`` unchanged at ``DEFAULT_DT``."""
    return float(base_s) * freshness_scale(dt)


def export_physics_dt(dt, environ=None):
    """Publish the resolved step into ``RESTOCK_PHYSICS_DT`` so host children + later reads rebase
    their freshness windows. Writes the exact ``repr`` of DEFAULT_DT's float at the default step, so a
    default run's env value round-trips to factor 1.0."""
    env = os.environ if environ is None else environ
    env[PHYSICS_DT_ENV] = repr(float(dt))
    return env[PHYSICS_DT_ENV]


def summary_fields(config):
    """mission.json provenance: ``{'physics_dt': dt, 'physics_hz': hz}`` only when the step is raised
    from 1/60 (so a default run's summary is byte-identical). A labelled workaround the walkthrough
    cut names, because a coarser step is a sim-fidelity relaxation."""
    dt = resolve_dt(config)
    if is_default(dt):
        return {}
    return {'physics_dt_mode': 'raised', 'physics_dt': dt, 'physics_hz': 1.0 / dt}
