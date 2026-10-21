"""Uncoupled base-drive fidelity bench -- pure core (no Isaac/ROS/GPU).

fm434 base-drive-fidelity investigation: run-325/326 showed the coupled base under-tracks the commanded
(v, omega) on the tow lane -- but the UNCOUPLED comparison proved it under-tracks even without a cart, so
it is a base-drive fidelity issue, not cart drag. The static check ruled out a wheel-radius/diff-drive
units error (the conversion uses radius 0.16 / track 0.70 matching the URDF, and the DriveAPI target is set
in the correct degrees convention; a units error would lurch FAST, not crawl). So the fidelity gap is
measured on the GPU with short UNCOUPLED bench runs, one physics lever per run.

This module is the PURE, Isaac-free core: the commanded step profile (straight then a min-radius arc) and
the commanded-vs-achieved tracking verdict. The Isaac glue (boot the mobile uncoupled, apply the wheel
targets, read the achieved body velocity) lives in the guarded runner ``scripts/run_base_drive_bench.py``.
CPU-tested in ``scripts/twin/tests/test_base_drive_bench.py``.
"""
from __future__ import annotations
from dataclasses import dataclass
import math


@dataclass(frozen=True)
class DriveCommand:
    """One commanded step of the profile."""
    t: float            # sim time at the start of the step (s)
    v: float            # commanded forward speed (m/s)
    omega: float        # commanded yaw rate (rad/s)
    phase: str          # 'straight' | 'arc'


def step_profile(*, dt=1.0 / 60.0, straight_s=4.0, arc_s=6.0, v=0.25, min_radius=1.5):
    """fm434 profile: hold a straight `v` for `straight_s`, then an arc at the `min_radius` (omega =
    v / min_radius, the coupled turn-radius limit) for `arc_s`. Pure list of DriveCommand at `dt`.

    A single open-loop command profile (NOT the closed-loop lane follower) so the bench isolates the
    DRIVE's ability to achieve the command from the follower/gate logic."""
    if not (dt > 0 and straight_s >= 0 and arc_s >= 0 and v > 0 and min_radius > 0):
        raise ValueError('invalid step-profile parameters')
    omega_arc = v / min_radius
    cmds = []
    n_straight = int(round(straight_s / dt))
    n_arc = int(round(arc_s / dt))
    for i in range(n_straight):
        cmds.append(DriveCommand(i * dt, v, 0.0, 'straight'))
    for j in range(n_arc):
        cmds.append(DriveCommand((n_straight + j) * dt, v, omega_arc, 'arc'))
    return cmds


def _ratio(achieved, commanded):
    return float('nan') if abs(commanded) < 1e-9 else achieved / commanded


def tracking_report(commands, achieved, *, settle_frac=0.25, tol=0.15):
    """Compare commanded vs achieved (both lists of (v, omega) aligned per tick) and grade tracking.

    `achieved` is a list of (rv, rw) measured base speeds. For each phase ('straight', 'arc') the first
    `settle_frac` of that phase's ticks are dropped (drive spin-up) and the median achieved/commanded
    ratio is reported for v (straight+arc) and for omega (arc). `within_tol` is True iff BOTH the straight
    forward ratio AND the arc forward+yaw ratios are within `tol` of 1.0 -- the fm434 'tracks within ~15%'
    stop condition. Pure; the runner feeds it the recorded telemetry."""
    if len(commands) != len(achieved) or not commands:
        raise ValueError('commands and achieved must be equal, non-empty lengths')
    import statistics as st

    def phase_ratios(phase, field_cmd, field_ach):
        idx = [i for i, c in enumerate(commands) if c.phase == phase]
        if not idx:
            return None
        drop = int(len(idx) * settle_frac)
        kept = idx[drop:] or idx
        rs = []
        for i in kept:
            c = commands[i].v if field_cmd == 'v' else commands[i].omega
            a = achieved[i][0] if field_ach == 'rv' else achieved[i][1]
            r = _ratio(abs(a), abs(c))
            if not math.isnan(r):
                rs.append(r)
        return round(st.median(rs), 3) if rs else None

    straight_v = phase_ratios('straight', 'v', 'rv')
    arc_v = phase_ratios('arc', 'v', 'rv')
    arc_w = phase_ratios('arc', 'omega', 'rw')
    ratios = [r for r in (straight_v, arc_v, arc_w) if r is not None]
    within_tol = bool(ratios) and all(abs(r - 1.0) <= tol for r in ratios)
    return dict(straight_v_ratio=straight_v, arc_v_ratio=arc_v, arc_omega_ratio=arc_w,
                within_tol=within_tol, tol=tol)
