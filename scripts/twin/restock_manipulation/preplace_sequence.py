"""Wrist-first preplace sequencing fallback (sim-preplace-per-joint-gains-v1, deliverable 3).

The primary fix is the per-joint drive override (stiffen shoulder_pan, keep the wrists soft). This is the
FALLBACK the brief asks for in case the model shows the pan cannot hold at any wrist-soft effort: remove
the fight instead of out-authoring it. Sequence the carried preplace approach into two joint-space moves:

  1. WRIST-SWEEP leg: drive every joint EXCEPT shoulder_pan from the carry/tuck pose to the standoff -- so
     the big 1.64 rad wrist_2 rotation (and the leveling wrist_1 / wrist_3 / shoulder_lift / elbow moves)
     happen while shoulder_pan is HELD at its carry value. The carton is kept level through this leg by the
     caller's leveled approach, exactly as the direct standoff approach is. During this leg shoulder_pan is
     NOT travelling to a far goal, so the wrist-sweep reaction is rejected around a FIXED setpoint, not
     fought while simultaneously traversing 1.35 rad.
  2. PAN leg: move shoulder_pan to its standoff value with the wrists already set, so the pan travels with
     NO wrist sweep acting -- the sweep reaction and the pan travel are DECOUPLED in time.

shoulder_pan (a near-vertical YAW joint) is the one held: its carton-tilt sensitivity at the standoff is
small, so a held/late pan keeps the carton within the level bound across both legs. The wrists / lift /
elbow are NOT held alone (their d(carton tilt)/d(joint) is large and config-dependent -- isolating them
would tilt the carton) -- they move together with the full leveling approach in the wrist-sweep leg. Both
sub-goals are plain 6-joint configs for ``begin_joint_goal`` / ``begin_joint_path``; this module only
computes them (pure/CPU).
"""
from __future__ import annotations

from typing import Sequence, Tuple

import math

# The joint HELD through the wrist-sweep leg (moved only in the second leg): shoulder_pan, the yaw joint.
HOLD_LAST_JOINTS = (0,)   # ur_shoulder_pan_joint


def wrist_first_subgoals(start: Sequence[float], standoff: Sequence[float], *,
                         hold_last: Sequence[int] = HOLD_LAST_JOINTS
                         ) -> Tuple[Tuple[float, ...], Tuple[float, ...]]:
    """Return ``(wrist_sweep_leg, standoff)`` sub-goals for the wrist-first preplace sequence.

    ``start`` is the current (tucked / carry) 6-joint config; ``standoff`` the preplace standoff goal.
    ``wrist_sweep_leg`` = the standoff with the ``hold_last`` joints (default shoulder_pan) kept at their
    START value, so every OTHER joint (the wrists + leveling cluster) reaches the standoff while pan holds;
    ``standoff`` is returned verbatim as the second (pan) leg. Pure/CPU; raises ValueError on a non-6 /
    non-finite input."""
    s = _vec6(start, 'start')
    g = _vec6(standoff, 'standoff')
    leg1 = list(g)
    for i in set(int(i) for i in hold_last):
        if 0 <= i < 6:
            leg1[i] = s[i]
    return tuple(leg1), tuple(g)


def _vec6(v, name):
    out = [float(x) for x in v]
    if len(out) != 6 or not all(math.isfinite(x) for x in out):
        raise ValueError('%s must be 6 finite joint values; got %r' % (name, v))
    return tuple(out)
