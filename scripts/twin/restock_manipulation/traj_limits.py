"""sim-preplace-accel-limit-reject-v1 -- shared cubic-Hermite trajectory limit gate + retiming.

The simulation FollowJointTrajectory executor (``twin.restock.trajectory_host.Plan``) reconstructs
every two-waypoint cubic-Hermite segment of a submitted plan and REJECTS the goal if any joint's
interpolated position, speed, or ACCELERATION leaves the controller's envelope. A cuMotion + MoveIt
``AddTimeOptimalParameterization`` (TOTG) joint plan -- the ``carton_level``-OFF pre-place path
(``RosMotionPort.begin_joint_goal`` -> ``MoveGroup`` joint goal) -- is time-parameterised against
MoveIt's own joint-limits.yaml (3.0 rad/s^2, scaled 0.02), NOT the executor's cubic-Hermite
acceleration ceiling (``max_speed * 0.5/0.15`` = 1.0 rad/s^2 at the sim 0.3 rad/s profile). A coarse
0.1 s TOTG sampling + the near-rest terminal remainder resamples that plan right at / over the
executor ceiling (warmacc4: the accepted lift plan reconstructs to peak 0.936 rad/s^2, with no margin),
so the pre-place crossed it and was REJECTED only at execution -- surfacing as a host fault
(``HOST_FAULT_ESTOP phase=preplace``, moveit_err=-4), never caught at plan time.

This module states the per-segment cubic-Hermite extrema ONCE (so ``trajectory_host.Plan`` and the
mission-side pre-submission gate check the IDENTICAL quantity) and provides a uniform time
reparameterisation that brings a plan's interpolated acceleration under the executor's limit BEFORE
submission:

  * the geometric joint path is preserved EXACTLY -- positions are untouched; a uniform time scale
    ``s`` maps ``t -> s*t`` and ``v -> v/s`` (the Hermite is reparameterised in time, not reshaped), so
    every per-segment acceleration scales by ``1/s^2`` and every speed by ``1/s`` (never worse), and the
    terminal-rest / start-at-rest / monotonic-time invariants the executor checks are preserved;
  * a plan ALREADY within the limit is returned unchanged (``s == 1`` -> byte-identical), so the gate is
    a pure safety net: it activates only for a plan the executor would otherwise reject;
  * a plan whose required time scale would push the total duration past the bounded execution budget is
    refused at plan time with a NAMED reason -- ``plan exceeds acceleration limit: <joint> <value> >
    <limit>`` -- before any arm command is streamed, rather than being submitted and rejected at the
    controller (which is a lost heartbeat).

Pure (numpy + math only), so the whole gate is CPU unit-testable with no ROS / Isaac / GPU.
"""
from __future__ import annotations

import math

import numpy as np

# The trajectory_host ratio: its interpolated-acceleration ceiling is ``max_speed * MAX_ACCEL_PER_SPEED``
# (preserve the conservative 0.5 rad/s^2 per 0.15 rad/s profile). Shared verbatim with
# ``trajectory_host.main`` so the mission-side gate and the executor compute the IDENTICAL ceiling.
MAX_ACCEL_PER_SPEED = 0.5 / 0.15
# The simulation FollowJointTrajectory executor's default joint-speed profile (rad/s). The host launches
# ``trajectory_host`` with this default (no ``--max-speed`` override), so the executor's acceleration
# ceiling is ``DEFAULT_EXEC_MAX_SPEED * MAX_ACCEL_PER_SPEED`` = 1.0 rad/s^2. Shared so the port's
# pre-submission gate matches the executor it is protecting.
DEFAULT_EXEC_MAX_SPEED = 0.3


class PlanLimitError(ValueError):
    """A plan whose interpolated acceleration cannot meet the executor's limit within the bounded
    execution-time budget (raised with the ``plan exceeds acceleration limit: ...`` named reason). A
    subclass of ValueError so existing ``except ValueError`` plan-handling is unchanged."""


def hermite_coeffs(q0, q1, v0, v1, d):
    """The cubic-Hermite coefficients (a, b) of a segment from (q0, v0) to (q1, v1) over duration d,
    in seconds from the segment start -- IDENTICAL to ``trajectory_host.Plan``. Position is
    ``((a*t+b)*t+v0)*t+q0``; acceleration is ``6a*t + 2b`` (linear, extremal at t=0 and t=d)."""
    d = float(d)
    a = (2 * (q0 - q1) + d * (v0 + v1)) / d ** 3
    b = (3 * (q1 - q0) - d * (2 * v0 + v1)) / d ** 2
    return a, b


def segment_accel_peaks(q0, q1, v0, v1, d):
    """Per-joint peak |acceleration| over the cubic-Hermite segment (array). The acceleration is linear
    in t, so its magnitude is maximal at an endpoint: ``max(|2b|, |6a*d + 2b|)`` -- the EXACT quantity
    ``trajectory_host.Plan`` tests against ``max_acceleration``."""
    a, b = hermite_coeffs(np.asarray(q0, float), np.asarray(q1, float),
                          np.asarray(v0, float), np.asarray(v1, float), d)
    return np.maximum(np.abs(2 * b), np.abs(6 * a * float(d) + 2 * b))


def trajectory_accel_peak(times, positions, velocities):
    """The worst per-segment cubic-Hermite acceleration over a whole plan.

    ``times`` is (N,) monotonically increasing (seconds from start); ``positions`` / ``velocities`` are
    (N, J) in a consistent joint order. Returns ``(peak, joint_index, seg_index)`` -- the maximal
    |acceleration| (rad/s^2), the joint column it occurs on, and the segment start index. A plan with
    fewer than two points has no segment -> ``(0.0, 0, 0)``."""
    t = np.asarray(times, float)
    Q = np.asarray(positions, float)
    V = np.asarray(velocities, float)
    peak, jidx, sidx = 0.0, 0, 0
    for i in range(len(t) - 1):
        d = t[i + 1] - t[i]
        if not (d > 0):
            continue  # non-increasing time is rejected by validate_trajectory upstream; skip defensively
        acc = segment_accel_peaks(Q[i], Q[i + 1], V[i], V[i + 1], d)
        j = int(np.argmax(acc))
        if acc[j] > peak:
            peak, jidx, sidx = float(acc[j]), j, i
    return peak, jidx, sidx


def accel_time_scale(peak, max_acceleration, *, eps=1e-3):
    """The smallest uniform time scale ``s >= 1`` that brings a cubic-Hermite acceleration ``peak`` to
    ``<= max_acceleration`` (acceleration scales by ``1/s^2``). ``s == 1`` when already within the limit.
    ``eps`` leaves a small margin so a floating-point edge case is comfortably under the ceiling."""
    peak = float(peak)
    limit = float(max_acceleration)
    if not (limit > 0):
        raise ValueError('max_acceleration must be positive')
    if peak <= limit:
        return 1.0
    return math.sqrt(peak / limit) * (1.0 + float(eps))


def retime_to_accel_limit(times, positions, velocities, *, max_acceleration, budget_s,
                          joint_names, eps=1e-3):
    """Uniformly time-reparameterise a plan so every segment's cubic-Hermite acceleration is within
    ``max_acceleration``.

    Returns ``(new_times, new_velocities, scale, peak_before, peak_after)``:
      * ``new_times = times * scale`` (positions are returned to the caller UNCHANGED);
      * ``new_velocities = velocities / scale``;
      * ``scale >= 1`` (``1.0`` and the inputs returned verbatim when the plan already meets the limit);
      * ``peak_before`` / ``peak_after`` are the worst interpolated acceleration before / after retiming.

    Raises ``PlanLimitError('plan exceeds acceleration limit: <joint> <value> > <limit>')`` when the
    scale needed would push the total duration past ``budget_s`` -- the plan is too aggressive to meet
    the executor's limit within the bounded execution budget, so it is refused at plan time (naming the
    worst joint and its interpolated acceleration) rather than submitted and rejected at the controller.
    Pure / CPU."""
    t = np.asarray(times, float)
    V = np.asarray(velocities, float)
    peak, jidx, sidx = trajectory_accel_peak(t, positions, velocities)
    if peak <= float(max_acceleration):
        return t, V, 1.0, peak, peak
    s = accel_time_scale(peak, max_acceleration, eps=eps)
    total = float(t[-1]) * s if len(t) else 0.0
    if total > float(budget_s):
        name = joint_names[jidx] if jidx < len(joint_names) else 'joint[%d]' % jidx
        raise PlanLimitError('plan exceeds acceleration limit: %s %.3f > %.3f'
                             % (name, peak, float(max_acceleration)))
    return t * s, V / s, s, peak, peak / (s * s)
