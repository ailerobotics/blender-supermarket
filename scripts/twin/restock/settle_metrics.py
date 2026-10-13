"""Measured-motion settling metric shared by the native stationary diagnostics.

PhysX TGS reports an instantaneous rigid-body velocity that carries a position/
velocity solve bias and does not integrate to the body's displacement (documented
upstream and in this task's report). The honest measure of "stationary" is how far
a body actually travels over the settle window - its net pose change as a rate -
which is robust to the sub-millimetre in-place jitter the solver leaves. This module
is pure numpy so it is unit-testable on CPU without a simulator.
"""
import math


def _wrap(a):
    return abs((a + math.pi) % (2 * math.pi) - math.pi)


def settle_report(window, names=('robot', 'cart'), *, linear_tol=0.005, angular_tol=0.005):
    """Return (settled, report) from a window of recorded stationary-feedback rows.

    Each row must carry ``simulation_time`` and, per body name, a dict with
    ``pose_xyz_yaw`` (and optionally ``linear_velocity``/``angular_velocity``). The
    settle verdict is gated on the net measured pose displacement rate over the
    window; the solver-biased reported velocity and the per-step jitter rate are
    recorded only as transparency diagnostics.
    """
    if len(window) < 2:
        raise ValueError('settle window needs at least two samples')
    span = window[-1]['simulation_time'] - window[0]['simulation_time']
    if span <= 0:
        raise ValueError('non-monotonic settle window')
    step = span / (len(window) - 1)
    report = {}
    settled = True
    for name in names:
        poses = [row[name]['pose_xyz_yaw'] for row in window]
        net_lin_rate = math.dist(poses[-1][:3], poses[0][:3]) / span
        net_ang_rate = _wrap(poses[-1][3] - poses[0][3]) / span
        peak_step_lin_rate = max(math.dist(poses[i + 1][:3], poses[i][:3])
                                 for i in range(len(poses) - 1)) / step
        reported = max((math.hypot(*row[name][key])
                        for row in window
                        for key in ('linear_velocity', 'angular_velocity')
                        if key in row[name]), default=float('nan'))
        report[name] = dict(net_linear_rate_m_s=net_lin_rate,
                            net_angular_rate_rad_s=net_ang_rate,
                            window_s=float(span),
                            peak_step_linear_rate_m_s=peak_step_lin_rate,
                            reported_solver_speed_biased=reported)
        if net_lin_rate >= linear_tol or net_ang_rate >= angular_tol:
            settled = False
    return settled, report
