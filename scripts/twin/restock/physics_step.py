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

# Legacy / default step. Returned verbatim when the key is absent, so the boot path is byte-identical.
DEFAULT_DT = 1.0 / 60.0

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


def summary_fields(config):
    """mission.json provenance: ``{'physics_dt': dt, 'physics_hz': hz}`` only when the step is raised
    from 1/60 (so a default run's summary is byte-identical). A labelled workaround the walkthrough
    cut names, because a coarser step is a sim-fidelity relaxation."""
    dt = resolve_dt(config)
    if is_default(dt):
        return {}
    return {'physics_dt_mode': 'raised', 'physics_dt': dt, 'physics_hz': 1.0 / dt}
