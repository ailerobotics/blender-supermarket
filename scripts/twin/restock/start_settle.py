"""Sim-time settle + rest/heading verify for the start-phase shortcut (sim-couple-shortcut-settle-dt-v1).

``shortcut.apply_start_state`` synthesizes a later stage's physical precondition (carton on the cart,
hitch, ...) and then has to let physics quiesce before the first mission tick. The legacy settle was a
FIXED tick count (``settle_steps=30`` world.step calls). That is wrong for three reasons the
coldwin25c/25d couple-start failures exposed:

* A fixed tick count is NOT a fixed amount of SIM time: 30 ticks is 0.5 s at 1/60 but 0.83 s at 1/36,
  and neither is tied to whether the bodies have actually stopped moving. The captain's rule is that
  every settle window is in SIM time and verified by the pose actually converging, so a physics-step or
  pacing change can never reintroduce a timing bug.
* On the ``portal_floor_box`` flat-floor box collider (PR 260/261, ``drop_subfloor`` drops the redundant
  ``Ground_exterior`` z=-0.010 sub-floor) a freshly spawned, UN-held cart is briefly unstable: it rises
  and yaws a few degrees before it settles (coldwin25 cart rose z 0.199 -> 0.275, yaw 0 -> 3.5 deg in
  ~1 s). The short fixed settle froze the couple-start cart mid-wobble at a tipped, mis-headed pose; the
  couple then measured the wrong dock yaw and routed the standoff into the backstock corridor
  (``blocked_no_route``). ``towstaging3`` (``portal_floor_box`` OFF) froze at the exact dock pose and
  couples. The fix holds the synthesized train at its authored seat THROUGH the settle so the box-floor
  wobble cannot tip or slide it, exactly reproducing the stable ``towstaging3`` seat at any floor.
* The stage used to be released without ever checking the synthesized cart actually came to REST at the
  configured dock. The verify below refuses to drive (naming ``START_STATE_UNSUPPORTED`` /
  ``START_STATE_SETTLE_MISMATCH``) when it did not, instead of towing from a mis-seated train.

This module is PURE (no Isaac/pxr/numpy-of-Isaac import): the settle driver takes injected ``step`` /
``sample`` callables and the checks take plain measured values, so the sim-time convergence and the
rest/heading verdicts are unit-tested on CPU at 1/60 and 1/36, with and without pacing. ``apply_start_state``
supplies the live-stage ``world.step`` and ``body_measurement`` on the Isaac side.
"""
from __future__ import annotations

import math

from .settle_metrics import settle_report

# Defaults. Every window is SIM SECONDS (not ticks), so a coarser physics step or a faster pacing
# profile leaves the settle behaviour unchanged per sim-second. min_settle_s matches the legacy 30
# ticks at 1/60 (0.5 s) as a floor; convergence (net pose-displacement RATE below tol over a trailing
# window) decides when to stop, bounded by max_settle_s.
_DEFAULTS = dict(
    min_settle_s=0.5,       # never release before this much SIM time, whatever the step
    window_s=0.2,           # trailing SIM window the convergence rate is measured over
    max_settle_s=4.0,       # bound: a never-settling body releases here (the verify then decides)
    linear_tol=0.005,       # m/s net displacement rate for "converged"
    angular_tol=0.005,      # rad/s net yaw rate for "converged"
    heading_tol_rad=math.radians(5.0),   # cart yaw vs configured dock (matches the 5 deg attach gate)
    rest_z_tol_m=0.02,      # cart z drift from its authored spawn z (a tip/sink exceeds this)
    rest_tilt_tol_rad=math.radians(3.0), # cart roll/pitch off level (a tip exceeds this)
    rest_vel_tol=0.02,      # residual speed (m/s, rad/s) for "stationary"
    hold_during_settle=True, # pin base + cart at the authored seat through the settle (the fix)
)

_SETTLE_MAX_S = 60.0


def _num(value, name, *, lo=None, hi=None, allow_zero=True):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f'start_state_settle.{name} must be a finite number')
    v = float(value)
    if not allow_zero and v == 0:
        raise ValueError(f'start_state_settle.{name} must be nonzero')
    if lo is not None and v < lo:
        raise ValueError(f'start_state_settle.{name}={v:g} must be >= {lo:g}')
    if hi is not None and v > hi:
        raise ValueError(f'start_state_settle.{name}={v:g} must be <= {hi:g}')
    return v


def resolve_settle_params(config):
    """Normalised settle/verify parameters from ``config['start_state_settle']`` (all optional; the
    defaults above). Angles may be given in degrees via ``*_deg`` keys. Absent block => the defaults,
    so a run that does not set it behaves identically to the tuned defaults. Pure; fails closed on a
    malformed value."""
    spec = config.get('start_state_settle') if isinstance(config, dict) else None
    params = dict(_DEFAULTS)
    if spec is None:
        return params
    if not isinstance(spec, dict):
        raise ValueError('start_state_settle must be an object of settle/verify parameters')
    if 'min_settle_s' in spec:
        params['min_settle_s'] = _num(spec['min_settle_s'], 'min_settle_s', lo=0.0, hi=_SETTLE_MAX_S)
    if 'window_s' in spec:
        params['window_s'] = _num(spec['window_s'], 'window_s', lo=0.0, hi=_SETTLE_MAX_S, allow_zero=False)
    if 'max_settle_s' in spec:
        params['max_settle_s'] = _num(spec['max_settle_s'], 'max_settle_s', lo=0.0, hi=_SETTLE_MAX_S,
                                      allow_zero=False)
    if 'linear_tol' in spec:
        params['linear_tol'] = _num(spec['linear_tol'], 'linear_tol', lo=0.0, allow_zero=False)
    if 'angular_tol' in spec:
        params['angular_tol'] = _num(spec['angular_tol'], 'angular_tol', lo=0.0, allow_zero=False)
    if 'heading_tol_deg' in spec:
        params['heading_tol_rad'] = math.radians(_num(spec['heading_tol_deg'], 'heading_tol_deg',
                                                      lo=0.0, allow_zero=False))
    elif 'heading_tol_rad' in spec:
        params['heading_tol_rad'] = _num(spec['heading_tol_rad'], 'heading_tol_rad', lo=0.0, allow_zero=False)
    if 'rest_z_tol_m' in spec:
        params['rest_z_tol_m'] = _num(spec['rest_z_tol_m'], 'rest_z_tol_m', lo=0.0, allow_zero=False)
    if 'rest_tilt_tol_deg' in spec:
        params['rest_tilt_tol_rad'] = math.radians(_num(spec['rest_tilt_tol_deg'], 'rest_tilt_tol_deg',
                                                        lo=0.0, allow_zero=False))
    elif 'rest_tilt_tol_rad' in spec:
        params['rest_tilt_tol_rad'] = _num(spec['rest_tilt_tol_rad'], 'rest_tilt_tol_rad', lo=0.0,
                                           allow_zero=False)
    if 'rest_vel_tol' in spec:
        params['rest_vel_tol'] = _num(spec['rest_vel_tol'], 'rest_vel_tol', lo=0.0, allow_zero=False)
    if 'hold_during_settle' in spec:
        if not isinstance(spec['hold_during_settle'], bool):
            raise ValueError('start_state_settle.hold_during_settle must be true or false')
        params['hold_during_settle'] = spec['hold_during_settle']
    if params['max_settle_s'] < params['min_settle_s']:
        raise ValueError('start_state_settle.max_settle_s must be >= min_settle_s')
    return params


def _wrap(a):
    return math.atan2(math.sin(a), math.cos(a))


def converge_settle(*, step, sample, dt, params, names=('carton',)):
    """Step physics in SIM TIME until the sampled bodies stop moving, then stop.

    ``step()`` advances one physics step (``world.step``). ``sample()`` returns ``{name: (x, y, z, yaw)}``
    for each name in ``names``, read AFTER the step. ``dt`` is the physics step in seconds (read from the
    live World, not a tick count). Convergence = over the trailing ``window_s`` of SIM time the net
    pose-displacement RATE of every tracked body is below ``linear_tol`` / ``angular_tol`` (the measured-
    motion metric in ``settle_metrics``), AND at least ``min_settle_s`` of SIM time has elapsed. Bounded
    by ``max_settle_s`` (a body that never settles releases there -- the caller's verify then decides).

    Returns a report dict ``{converged, elapsed_s, steps, window_span_s, dt, report}``. ``steps`` differs
    with ``dt`` (more fine steps for the same sim time) but ``elapsed_s`` at convergence does NOT -- that
    is the whole point. Pure (no Isaac): fully exercised on CPU with fake ``step``/``sample``.
    """
    dt = float(dt)
    if not math.isfinite(dt) or dt <= 0:
        raise ValueError('converge_settle needs a positive finite physics dt')
    names = tuple(names)
    min_s, win_s, max_s = params['min_settle_s'], params['window_s'], params['max_settle_s']
    window = []
    steps = 0
    elapsed = 0.0
    converged = False
    last_report = None
    # +1e-9 so a max_s that is an exact multiple of dt runs its last whole step.
    while elapsed < max_s - 1e-9:
        step()
        steps += 1
        elapsed = steps * dt
        poses = sample()
        window.append({'simulation_time': elapsed,
                       **{n: {'pose_xyz_yaw': tuple(poses[n])} for n in names}})
        # Keep the trailing window ~win_s long (drop the front while the span still covers it).
        while len(window) > 2 and (window[-1]['simulation_time'] - window[1]['simulation_time']) >= win_s:
            window.pop(0)
        span = window[-1]['simulation_time'] - window[0]['simulation_time']
        if elapsed + 1e-9 >= min_s and span + 1e-9 >= win_s and len(window) >= 2:
            settled, rep = settle_report(window, names=names,
                                         linear_tol=params['linear_tol'],
                                         angular_tol=params['angular_tol'])
            last_report = rep
            if settled:
                converged = True
                break
    return {'converged': converged, 'elapsed_s': round(elapsed, 6), 'steps': steps,
            'window_span_s': round((window[-1]['simulation_time'] - window[0]['simulation_time'])
                                   if len(window) >= 2 else 0.0, 6),
            'dt': dt, 'report': last_report}


def resting_check(measured, expected, params):
    """``(ok, reason)`` -- is the synthesized cart RESTING on supported floor?

    ``measured`` / ``expected`` carry ``z`` (m), ``roll_rad``, ``pitch_rad``, ``lin_speed`` (m/s),
    ``ang_speed`` (rad/s). The cart has NOT come to rest when it has risen/sunk from its authored spawn
    z beyond ``rest_z_tol_m`` (a tip or a sink through an unsupported floor), is tilted off level beyond
    ``rest_tilt_tol_rad``, or is still moving beyond ``rest_vel_tol`` -- each a sign the floor under the
    spawn footprint cannot hold it. Names ``START_STATE_UNSUPPORTED`` so the trip is attributable. Pure.
    """
    dz = abs(float(measured['z']) - float(expected['z']))
    tilt = max(abs(float(measured.get('roll_rad', 0.0))), abs(float(measured.get('pitch_rad', 0.0))))
    lin = abs(float(measured.get('lin_speed', 0.0)))
    ang = abs(float(measured.get('ang_speed', 0.0)))
    problems = []
    if dz > params['rest_z_tol_m']:
        problems.append(f'z {float(measured["z"]):.4f} m vs spawn {float(expected["z"]):.4f} m '
                        f'(drift {dz * 1000:.1f} mm > {params["rest_z_tol_m"] * 1000:.0f} mm)')
    if tilt > params['rest_tilt_tol_rad']:
        problems.append(f'tilt {math.degrees(tilt):.2f} deg off level '
                        f'> {math.degrees(params["rest_tilt_tol_rad"]):.1f} deg')
    if lin > params['rest_vel_tol'] or ang > params['rest_vel_tol']:
        problems.append(f'still moving (lin {lin:.3f} m/s, ang {ang:.3f} rad/s '
                        f'> {params["rest_vel_tol"]:.3f})')
    if problems:
        return False, ('START_STATE_UNSUPPORTED: the synthesized cart is not resting on supported floor '
                       '-- ' + '; '.join(problems) + ' (likely cause: the spawn/parking footprint sits '
                       'over dropped floor, e.g. portal_floor_box drop_subfloor with no floor under the '
                       'shortcut cart spawn)')
    return True, None


def heading_check(measured_yaw, configured_yaw, tol):
    """``(ok, residual_rad, reason)`` -- does the settled cart heading match the configured dock yaw?

    The couple stage measures the live cart yaw to derive its cart-relative standoff; a cart mis-seated
    off the dock heading sends the standoff to the wrong place (coldwin25c: 6.7 deg vs the dock -90 deg
    put the standoff in the backstock corridor -> blocked_no_route). Names ``START_STATE_SETTLE_MISMATCH``.
    Pure."""
    res = abs(_wrap(float(measured_yaw) - float(configured_yaw)))
    if res > tol:
        return False, res, (f'START_STATE_SETTLE_MISMATCH: the synthesized cart settled at heading '
                            f'{math.degrees(float(measured_yaw)):.1f} deg, not the configured dock '
                            f'{math.degrees(float(configured_yaw)):.1f} deg (residual '
                            f'{math.degrees(res):.1f} deg > {math.degrees(tol):.1f} deg); the couple '
                            'standoff would be derived from the wrong cart axis')
    return True, res, None
