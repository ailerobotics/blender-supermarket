"""Perceived-clearance collision gate for every arm leg (pure, CPU).

Before the controller dispatches any arm leg -- pick approach, lift, tuck, pre-place legs,
place descent, withdraw, every transfer leg -- this gate queries the live nvblox ESDF
(:class:`twin.restock_manipulation.esdf_grid.EsdfSnapshot`, fed by :mod:`esdf_source`) along
the planned path for the swept gripper + welded-carton volume and returns a
:class:`GateVerdict`: proceed, warn, hold or refuse, with the minimum perceived clearance and
the body it is nearest to when the ESDF carries labels. It logs one ``PERCEIVED_CLEARANCE``
line per leg.

This is the additive perception layer the lane adds on top of the existing analytic / ground
-truth collect path. It is **opt-in and byte-identical when off**: the gate object is built
and installed (wrapped around the motion port, see :mod:`gated_motion`) only when the host key
``perception_collision_gate`` opts in; with the key absent nothing here runs.

Decision rule (pure; all thresholds configurable):

* leg phase not in the gate's active set           -> ``proceed`` (gate inactive for this leg)
* no fresh ESDF snapshot, or the swept volume is    -> ``unknown_action`` (default ``hold``):
  more than ``unobserved_tol`` unobserved               the map cannot certify this leg yet
* minimum observed clearance < ``margin_m``         -> ``mode`` (default ``refuse``): a mapped
                                                        surface is too close to the swept volume
* otherwise                                         -> ``proceed``

``hold``/``refuse`` are both honest stops: the gate never lets a leg through on an unobserved
or too-close map. The difference is the *reason* and how the mission reacts -- a ``refuse``
raised at the motion-port seam is caught by the controller and fails the phase by name
(``PERCEIVED_CLEARANCE_REFUSE``), which the mission's StageRetry re-begins against a fresher
map (a bounded hold-and-retry); ``warn`` logs and proceeds (observability-only, for the first
bring-up runs). Execution-time preempt/replan when a NEW obstacle enters an already-dispatched
leg's corridor is a SIBLING lane (sim-arm-reactive-replan-v1) built on the hook points below,
NOT this gate, which gates at dispatch.

Hook points for the sibling lanes (stable interface):

* :meth:`PerceptionCollisionGate.evaluate` -- the per-leg verdict. The collision-bench lane
  (sim-arm-collision-bench-v1) reads ``GateVerdict.min_clearance`` / ``action`` to score
  collision-free rate, minimum clearance and refusals per scenario.
* :func:`swept_corridor_aabbs` -- the swept gripper+carton AABB union for a leg. The reactive
  -replan lane watches this corridor for newly-entered obstacles between dispatch and
  completion; the bench plants obstacles relative to it.
* :class:`twin.restock_manipulation.esdf_source.EsdfSource` -- the snapshot provider the bench
  replaces with a scripted static/moving-obstacle field and the dynamic layer feeds.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Optional

import numpy as np

from . import esdf_grid

#: The arm legs the gate is active on by default -- every leg that moves the gripper (and, when
#: held, the welded carton). ``observe``/``grasp``/``release``/``stow``/``idle`` are not swept
#: gripper translations through the mapped workspace, so the gate defaults inactive on them.
DEFAULT_GATED_PHASES = frozenset({
    'approach', 'contact', 'lift', 'tuck', 'pan', 'transit', 'preplace', 'place', 'retreat'})

_PROCEED = 'proceed'
_WARN = 'warn'
_HOLD = 'hold'
_REFUSE = 'refuse'
_MODES = (_WARN, _HOLD, _REFUSE)


@dataclass(frozen=True)
class GateVerdict:
    """Per-leg decision + the measured perceived clearance (a hook-point value).

    ``action`` is one of ``proceed``/``warn``/``hold``/``refuse``. ``min_clearance`` is the
    minimum OBSERVED signed distance (metres) over the swept volume (``inf`` when nothing was
    observed); ``nearest_body`` is the body it is nearest to when the ESDF carries labels, else
    ``None``. ``unobserved_fraction`` is how much of the swept volume the map had not seen.
    """
    phase: str
    action: str
    min_clearance: float
    margin_m: float
    nearest_body: Optional[str]
    unobserved_fraction: float
    reason: str
    aabb_count: int
    stamp: float
    snapshot_age: float

    @property
    def ok(self) -> bool:
        """True when the leg may be dispatched (``proceed`` or observability-only ``warn``)."""
        return self.action in (_PROCEED, _WARN)

    @property
    def blocked(self) -> bool:
        return self.action in (_HOLD, _REFUSE)


def swept_corridor_aabbs(start_pose, goal_pose, pad_min, pad_max, *, samples=6):
    """Swept gripper(+carton) AABB union for a leg from ``start_pose`` to ``goal_pose``.

    Both poses are 7-tuples ``(x, y, z, qx, qy, qz, qw)`` of the ``gripper_pad_tcp`` frame;
    ``[pad_min, pad_max]`` is the oriented envelope in that frame (fold a held carton in with
    :func:`twin.restock_manipulation.esdf_grid.combined_envelope`). The leg is sampled into
    ``samples`` interpolated poses (positions lerp, orientation :func:`esdf_grid.nlerp`) and
    each pose's oriented box is turned into an AABB -- their union approximates the corridor.
    This is the hook the reactive-replan lane watches and the bench plants against.
    """
    start = np.asarray(start_pose, float)
    goal = np.asarray(goal_pose, float)
    if start.shape != (7,) or goal.shape != (7,):
        raise ValueError('finite 7-tuple start/goal poses required')
    n = max(2, int(samples))
    poses = []
    for i in range(n):
        t = i / (n - 1)
        pos = (1 - t) * start[:3] + t * goal[:3]
        quat = esdf_grid.nlerp(start[3:], goal[3:], t)
        poses.append((float(pos[0]), float(pos[1]), float(pos[2]), *map(float, quat)))
    return esdf_grid.swept_pose_aabbs(poses, pad_min, pad_max)


class PerceptionCollisionGate:
    """Decide, log and report a per-leg perceived-clearance verdict.

    :param margin_m: minimum perceived clearance (m) the swept volume must keep from any mapped
        surface; below it the gate applies ``mode``.
    :param mode: ``warn`` | ``hold`` | ``refuse`` -- the action when clearance is below margin.
    :param unknown_action: the action when there is no snapshot or the swept volume is more than
        ``unobserved_tol`` unobserved (default ``hold`` -- the map cannot certify the leg yet).
    :param phases: the leg phases the gate is active on (default :data:`DEFAULT_GATED_PHASES`);
        a leg outside the set always ``proceed``\\ s.
    :param samples: swept-corridor sampling density.
    :param unobserved_tol: tolerated unobserved fraction of a swept volume before ``unknown``.
    :param max_snapshot_age_s: a snapshot older than this (now - stamp) is treated as absent.
    :param logger: ``print``-like sink for the ``PERCEIVED_CLEARANCE`` line.
    """

    def __init__(self, *, margin_m=0.03, mode=_REFUSE, unknown_action=_HOLD,
                 phases=None, samples=6, unobserved_tol=0.2, max_snapshot_age_s=0.5,
                 logger=print):
        if not (math.isfinite(margin_m) and margin_m >= 0):
            raise ValueError('margin_m must be a finite non-negative metre clearance')
        if mode not in _MODES:
            raise ValueError('mode must be one of %s' % (_MODES,))
        if unknown_action not in _MODES:
            raise ValueError('unknown_action must be one of %s' % (_MODES,))
        if not (math.isfinite(unobserved_tol) and 0 <= unobserved_tol <= 1):
            raise ValueError('unobserved_tol must be a fraction in [0, 1]')
        if not (math.isfinite(max_snapshot_age_s) and max_snapshot_age_s > 0):
            raise ValueError('max_snapshot_age_s must be a positive sim-second budget')
        self.margin_m = float(margin_m)
        self.mode = mode
        self.unknown_action = unknown_action
        self.phases = frozenset(phases) if phases is not None else DEFAULT_GATED_PHASES
        self.samples = max(2, int(samples))
        self.unobserved_tol = float(unobserved_tol)
        self.max_snapshot_age_s = float(max_snapshot_age_s)
        self.logger = logger

    def active(self, phase) -> bool:
        """Whether the gate evaluates this leg phase (a leg outside the set always proceeds)."""
        return phase in self.phases

    def evaluate(self, *, phase, start_pose, goal_pose, pad_min, pad_max, snapshot, now,
                 held=False) -> GateVerdict:
        """Perceived-clearance verdict for one leg; logs one ``PERCEIVED_CLEARANCE`` line.

        ``snapshot`` is the live :class:`esdf_grid.EsdfSnapshot` (or ``None`` when the ESDF is
        unavailable); ``now`` is sim seconds. ``[pad_min, pad_max]`` is the swept envelope (fold
        a held carton in before calling). Returns a :class:`GateVerdict` regardless of ``mode``
        so a caller in ``warn`` still sees the measured clearance.
        """
        if not self.active(phase):
            return GateVerdict(phase, _PROCEED, float('inf'), self.margin_m, None, 0.0,
                               'phase not gated', 0, -1.0, float('inf'))
        boxes = swept_corridor_aabbs(start_pose, goal_pose, pad_min, pad_max, samples=self.samples)
        return self.evaluate_corridor(phase=phase, boxes=boxes, snapshot=snapshot, now=now,
                                      held=held)

    def evaluate_corridor(self, *, phase, boxes, snapshot, now, held=False) -> GateVerdict:
        """Verdict for a pre-built swept corridor (list of ``(low, high)`` AABBs).

        The multi-waypoint leg path (:meth:`begin_joint_path`) and the collision-bench lane use
        this directly; :meth:`evaluate` is the two-pose convenience that builds the corridor
        first. Logs the same ``PERCEIVED_CLEARANCE`` line.
        """
        if not self.active(phase):
            return GateVerdict(phase, _PROCEED, float('inf'), self.margin_m, None, 0.0,
                               'phase not gated', 0, -1.0, float('inf'))
        age = (now - snapshot.stamp) if snapshot is not None else float('inf')
        if snapshot is None or not math.isfinite(age) or age > self.max_snapshot_age_s:
            why = ('no ESDF snapshot' if snapshot is None
                   else 'ESDF snapshot stale (age=%.3f s > %.3f s)' % (age, self.max_snapshot_age_s))
            return self._verdict(phase, self.unknown_action, float('inf'), None, 1.0,
                                 why, 0, -1.0 if snapshot is None else snapshot.stamp,
                                 age, now, held=held)
        q = snapshot.clearance_over_boxes(boxes)
        nearest = snapshot.label_at(q.nearest_index)
        unobserved = q.unobserved_fraction
        if unobserved > self.unobserved_tol:
            action = self.unknown_action
            reason = ('swept volume %.0f%% unobserved (> %.0f%% tol): the ESDF has not mapped '
                      'this corridor yet' % (100 * unobserved, 100 * self.unobserved_tol))
        elif q.min_observed < self.margin_m:
            action = self.mode
            reason = ('perceived clearance %.4f m < margin %.3f m%s' %
                      (q.min_observed, self.margin_m,
                       ' (nearest %s)' % nearest if nearest else ''))
        else:
            action = _PROCEED
            reason = 'clear'
        return self._verdict(phase, action, q.min_observed, nearest, unobserved, reason,
                             len(boxes), snapshot.stamp, age, now, held=held)

    def _verdict(self, phase, action, min_clearance, nearest, unobserved, reason, aabb_count,
                 stamp, age, now, held=False) -> GateVerdict:
        verdict = GateVerdict(phase, action, min_clearance, self.margin_m, nearest, unobserved,
                              reason, aabb_count, stamp, age)
        clr = '%.4f' % min_clearance if math.isfinite(min_clearance) else 'none'
        if self.logger is not None:
            self.logger(
                'PERCEIVED_CLEARANCE phase=%s action=%s min=%s margin=%.3f nearest=%s '
                'unobserved=%.2f held=%s boxes=%d stamp=%.3f age=%s now=%.3f reason=%r'
                % (phase, action, clr, self.margin_m, nearest or '-', unobserved, bool(held),
                   aabb_count, stamp, ('%.3f' % age) if math.isfinite(age) else 'inf', now, reason),
                flush=True)
        return verdict
