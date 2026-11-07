"""The three-stage collect place set-down: separate the carried APPROACH from the gripper REORIENTATION
(sim-place-pad-down-arrival-v1).

coldwin92 was the first run with the honest seat gate AND both cage-clearance gates live, and it exposed
the last contradiction of the SINGLE-standoff place design:

  * every pre-place standoff whose carried approach keeps the carton LEVEL lands on the LOW-wrist_2 CARRY
    branch (the branch connected to the tucked carry shape), and that branch's straight place DESCENT rakes
    the +x mouth posts -- the carry-connected selector's descent-feasibility filter therefore found NO
    descent-feasible branch (11 ranked, 8 leg-2-accepted, all descent-infeasible);
  * the only CAGE-CLEAR descent holds the HIGH-wrist_2 PAD-DOWN branch (wrist_2 ~ 1.6), which the carried
    approach cannot reach LEVEL (the single-standoff fallback tilted the carton 13.96 deg ->
    CARTON_LEVEL_PLAN_REJECTED -> HOST_FAULT_ESTOP at collect step 5 'place').

The fix is to SEPARATE the approach from the reorientation, exactly as the captain's intent states: arrive
level above the shelf, turn the gripper pad-down there with the carton kept level, then descend clear of
the cage to the proven seat. CPU-proven (data/sim-place-pad-down-arrival-v1):

  * an in-place reorientation directly over the seat rakes post_2 at EVERY standoff height -- the carton
    rotates ~73 deg about the vertical axis (yaw 155 -> 82) and the suction housing sweeps THROUGH the tall
    mouth post (post_2 spans z[-0.196, 1.254]); raising the standoff to clear the post top degenerates the
    level IK (no in-clear-region level solution above ~0.40 m);
  * but reorienting in the OPEN APPROACH CORRIDOR -- the standoff shifted ``shift_m`` toward the arm base
    and ``height_m`` above the seat, so the housing orbits IN FRONT of the mouth posts -- keeps every
    gripper link clear of the cage (at the default 0.20 m / 0.10 m, controller gripper_pad_tcp frame:
    >= +66 mm reorient, >= +103 mm descent) while the carton stays level (<= 0.04 deg reorient, <= 1.25 deg
    descent) AND branch-continuous (reorient wrist step <= 0.080 rad) at every cw64-cw92 park; the PR 376
    ``place_from_above`` L-path descent then returns the pad-down carton over the seat and lands it on the
    0.9325 seat.

This module is the PURE / CPU core:

  * :func:`mouth_out_unit` -- the horizontal "out of the cage mouth, toward the arm base" direction (from
    the seat tool-origin toward the ur_base_link origin); the reorientation standoff is offset along it.
  * :func:`reorient_standoff_tcp` -- the shifted above-shelf reorientation standoff tool-origin.
  * :func:`select_pad_down_branch` -- among the reach+level branches at the standoff, the PAD-DOWN branch
    whose ``place_from_above`` descent to the seat is cage-clear and level (ranked by descent clearance when
    a clearance fn is supplied, else by the smallest descent tilt). Raises when none is cage-clear.

The reorientation leg itself is planned with ``carton_level.plan_level_world_approach(anchor='seeded')`` --
the carry-wrist arrival and the pad-down branch are distinct wrist-flip IK solutions, so a BRANCH-CONTINUOUS
seeded ramp (not a per-waypoint independent re-leveler, which jumps ~1.5 rad) is needed to keep the carton
level AND continuous -- and it is gated by the SAME descent cage gate the place dispatch runs -- see
:meth:`controller.TransferController` and
``scripts/check_place_pad_down_arrival.py`` (the CPU sweep proof). No Isaac / ROS / GPU; needs only the URDF
FK + numpy (the cage sweep adds the XRDF sphere model + usd-core, like the PR 376 descent gate).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np

# Log tokens the controller prints.
PLACE_REORIENT = 'PLACE_REORIENT'
PLACE_REORIENT_REJECTED = 'PLACE_REORIENT_REJECTED'
PAD_DOWN_BRANCH = 'PAD_DOWN_BRANCH'


def mouth_out_unit(seat_tcp: Sequence[float]) -> Tuple[float, float, float]:
    """The horizontal unit vector pointing OUT of the cage mouth -- from the seat tool-origin
    (``seat_tcp``, in ur_base_link) back toward the ur_base_link origin, i.e. toward the arm base.

    The rollwagen docks beside the arm with its open +x face toward the robot, so the seat sits at
    ur_base_link +x and the open approach corridor is the horizontal direction back toward the base
    origin: ``-unit(seat_xy)``. Pulling the reorientation standoff this way lands the gripper housing IN
    FRONT of the mouth posts (CPU-proven clear at every cw park), while the PR 376 ``place_from_above``
    L-path descent returns the pad-down carton over the seat. Raises on a degenerate (origin) seat xy.
    Pure/CPU."""
    x, y = float(seat_tcp[0]), float(seat_tcp[1])
    n = math.hypot(x, y)
    if n <= 1e-9:
        raise ValueError('mouth_out_unit: the seat tool-origin xy is at the base origin; no mouth-out '
                         'direction is defined (expected the seat at ur_base_link +x)')
    return (-x / n, -y / n, 0.0)


def reorient_standoff_tcp(seat_tcp: Sequence[float], shift_m: float, height_m: float
                          ) -> Tuple[float, float, float]:
    """The reorientation standoff tool-origin: the seat tool-origin offset ``shift_m`` along
    :func:`mouth_out_unit` (into the open approach corridor, in front of the mouth posts) and raised
    ``height_m`` above the seat. Pure/CPU."""
    s = np.asarray(seat_tcp, float)[:3]
    d = np.asarray(mouth_out_unit(seat_tcp), float)
    out = s + d * float(shift_m)
    out[2] = float(s[2]) + float(height_m)
    return (float(out[0]), float(out[1]), float(out[2]))


@dataclass
class PadDownBranch:
    """A ranked candidate pad-down branch at the reorientation standoff.

    ``config`` is the ARM-order joint map (reach+level at the standoff tool-origin). ``wrist_2`` is its
    branch discriminant. ``descent_tilt_deg`` is the predicted executed |roll|/|pitch| peak of its
    ``place_from_above`` descent to the seat; ``descent_clearance_m`` the minimum cuMotion-arm-sphere gap
    to any cage body over that descent (``None`` when no clearance fn was supplied -- then ``cage_ok`` came
    from the pass/fail gate alone). ``cage_ok`` is whether the descent cleared the cage. ``chosen`` marks
    the selected branch."""
    config: dict
    wrist_2: float
    descent_tilt_deg: float
    descent_clearance_m: Optional[float]
    cage_ok: bool
    chosen: bool = False


def select_pad_down_branch(robot, standoff_tcp: Sequence[float], up_in_tool, seed, *, seat_tcp,
                           limits=None, gravity_comp=None, bound_deg=2.0, tool_link='ur_tool0',
                           root='ur_base_link', descent_waypoints=12, approach_from_above=True,
                           above_clearance_m=0.08, depth_bias_m=0.0,
                           descent_cage_gate: Optional[Callable[[List[list]], Optional[str]]] = None,
                           descent_cage_clearance: Optional[Callable[[List[list]], float]] = None,
                           fast=True):
    """Pick the PAD-DOWN branch at the reorientation ``standoff_tcp``: the reach+level branch whose
    ``place_from_above`` descent to ``seat_tcp`` is CAGE-CLEAR and within the ``bound_deg`` level bound
    (sim-place-pad-down-arrival-v1).

    ``gravity_comp`` must be a ``carton_level.GravityCompensation`` (the loaded droop model the place descent
    uses). ``descent_cage_gate`` is the pass/fail cage sweep closure (``waypoints -> reject | None``, the
    controller's wired ``_place_descent_cage_gate``); a branch whose descent it REJECTS is discarded.
    ``descent_cage_clearance`` is the optional raw min-clearance closure (``waypoints -> metres``); when
    supplied the surviving branches are ranked by it (MOST clear first), else by the smallest descent tilt.
    Returns ``(pad_down_config, report)`` where ``report`` carries the ranked :class:`PadDownBranch` list and
    the chosen branch's metrics. Raises ``ValueError`` when NO branch has a cage-clear, in-bound descent (the
    caller fails by name / the report documents the next lever). Pure/CPU (URDF FK; the cage closures add the
    XRDF sphere model)."""
    from . import carton_level as cl
    from .cad_tool import ARM_JOINTS
    if gravity_comp is None:
        raise ValueError('select_pad_down_branch requires a GravityCompensation (the place-descent droop model)')
    seat = tuple(float(c) for c in np.asarray(seat_tcp, float)[:3])
    branches = cl.enumerate_level_place_branches(robot, standoff_tcp, up_in_tool, seed, limits=limits,
                                                 tool_link=tool_link, root=root, fast=fast)
    if not branches:
        raise ValueError('no reach+level standoff branch at the reorientation standoff %s'
                         % list(np.round(np.asarray(standoff_tcp, float)[:3], 4)))
    cands: List[PadDownBranch] = []
    for b in branches:
        bv = [float(b[j]) for j in ARM_JOINTS]
        try:
            plan = cl.plan_compensated_place_descent(
                robot, bv, seat, bv, up_in_tool, gravity_comp, bound_deg=bound_deg, tool_link=tool_link,
                root=root, limits=limits, waypoints=descent_waypoints,
                approach_from_above=approach_from_above, above_clearance_m=above_clearance_m,
                depth_bias_m=depth_bias_m)
        except ValueError:
            continue                     # no in-limits level descent from this branch -> discard
        tilt = float(max(plan.max_roll, plan.max_pitch))
        if tilt > float(bound_deg) + 1e-9:
            cands.append(PadDownBranch(dict(b), bv[4], tilt, None, cage_ok=False))
            continue
        wps = [list(w) for w in plan.waypoints]
        cage_ok = True
        if descent_cage_gate is not None:
            try:
                cage_ok = descent_cage_gate(wps) is None
            except Exception:            # a provider error must not crash the select
                cage_ok = True
        clr = None
        if descent_cage_clearance is not None:
            try:
                clr = float(descent_cage_clearance(wps))
            except Exception:
                clr = None
        cands.append(PadDownBranch(dict(b), bv[4], tilt, clr, cage_ok=cage_ok))
    feasible = [c for c in cands if c.cage_ok and c.descent_tilt_deg <= float(bound_deg) + 1e-9]
    if not feasible:
        raise ValueError(
            'no pad-down branch at the reorientation standoff has a cage-clear, level (<= %.2f deg) '
            'place-from-above descent to the seat; enumerated %d branches '
            '(wrist_2 / descent_tilt / cage_ok: %s)'
            % (float(bound_deg), len(cands),
               ', '.join('%.2f/%.2f/%s' % (c.wrist_2, c.descent_tilt_deg, c.cage_ok) for c in cands)))
    # Rank: MOST descent clearance first when known, else the smallest descent tilt. Deterministic
    # tie-break by wrist_2 so the choice is stable run-to-run.
    if any(c.descent_clearance_m is not None for c in feasible):
        feasible.sort(key=lambda c: (-(c.descent_clearance_m if c.descent_clearance_m is not None else -1e9),
                                     round(c.descent_tilt_deg, 6), round(c.wrist_2, 6)))
    else:
        feasible.sort(key=lambda c: (round(c.descent_tilt_deg, 6), round(c.wrist_2, 6)))
    chosen = feasible[0]
    chosen.chosen = True
    report = {
        'chosen_wrist_2': chosen.wrist_2,
        'chosen_descent_tilt_deg': chosen.descent_tilt_deg,
        'chosen_descent_clearance_m': chosen.descent_clearance_m,
        'n_branches': len(cands),
        'n_feasible': len(feasible),
        'branches': cands,
    }
    return dict(chosen.config), report
