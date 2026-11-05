"""Adapter: drive the MERGED perceived-clearance gate (sim-rgbd-arm-collision-aware-v1, PR #351)
from the arm collision bench, so the AFTER numbers come from the real gate, not a model.

Plugs into the bench via the published `module:function` seam:

    PYTHONPATH=scripts python3 scripts/run_arm_collision_bench.py \
        --gate twin.tools.arm_collision_perception_gate:perception_gate --observe-all

It uses the sibling lane's OWN published hook points verbatim (README_perception.md):
``collision_gate.PerceptionCollisionGate`` + ``esdf_source.AnalyticBoxEsdfSource`` +
``tools.replay_collect_gate.replay_gate``.  For each scenario it builds an analytic ESDF from the
scenario's OBSERVED, NON-human obstacle boxes, FKs the leg's ``gripper_pad_tcp`` corridor from the
bench's nominal trajectory, runs the real gate leg-by-leg, and maps the real verdict
(``proceed``/``warn``/``hold``/``refuse``) into the bench's behaviour vocabulary.

Scope, stated honestly (captain section-9 + README_perception "Set-down + the gate"):

* The merged gate is a **dispatch BLOCKER** -- it ``refuse``/``hold``s before motion.  It does NOT
  itself ``avoid`` (re-plan a detour) or ``slow_stop_resume`` -- those are the sim-arm-reactive-replan
  lane, built on these hook points, not yet merged.  So a ``refuse`` here is the real gate PREVENTING
  the strike (no contact), which is what the AFTER "no contact" number measures; it will not match a
  scenario whose *expected* behaviour is the richer ``avoid``.
* **People are never segmented into the arm ESDF** -- the human gate owns the STOP.  So the moving
  person/hand obstacles are EXCLUDED from this gate's ESDF; the gate correctly proceeds and the STOP
  is the human gate's job (out of this gate's scope).  Score the real gate on the STATIC set.
"""
from __future__ import annotations

import math

import numpy as np

from . import arm_collision_bench as bench

# bench arm_leg -> a gated leg phase name (collision_gate.DEFAULT_GATED_PHASES).
PHASE_FOR = {
    'collect_pick': 'approach', 'carry': 'tuck', 'preplace_approach': 'preplace',
    'set_down': 'place', 'withdraw': 'retreat', 'transfer': 'approach',
}
_HUMAN = ('person', 'hand')

_ARM = None  # cached ArmModel (FK is the only per-call cost)
_GATE = None


def _arm():
    global _ARM
    if _ARM is None:
        _ARM = bench.ArmModel.load()
    return _ARM


def _rotmat_to_quat(R):
    """3x3 rotation -> (qx, qy, qz, qw)."""
    R = np.asarray(R, float)
    t = np.trace(R)
    if t > 0:
        s = math.sqrt(t + 1.0) * 2
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    else:
        i = int(np.argmax([R[0, 0], R[1, 1], R[2, 2]]))
        if i == 0:
            s = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
            w = (R[2, 1] - R[1, 2]) / s; x = 0.25 * s
            y = (R[0, 1] + R[1, 0]) / s; z = (R[0, 2] + R[2, 0]) / s
        elif i == 1:
            s = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
            w = (R[0, 2] - R[2, 0]) / s; x = (R[0, 1] + R[1, 0]) / s
            y = 0.25 * s; z = (R[1, 2] + R[2, 1]) / s
        else:
            s = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
            w = (R[1, 0] - R[0, 1]) / s; x = (R[0, 2] + R[2, 0]) / s
            y = (R[1, 2] + R[2, 1]) / s; z = 0.25 * s
    q = np.array([x, y, z, w], float)
    return tuple(q / (np.linalg.norm(q) or 1.0))


def _pad_pose7(arm, cfg):
    T = np.asarray(arm._fk(cfg).get('gripper_pad_tcp', arm._fk(cfg)['ur_tool0']), float)
    return (float(T[0, 3]), float(T[1, 3]), float(T[2, 3]), *_rotmat_to_quat(T[:3, :3]))


def _static_boxes(scenario, t=0.0):
    """Observed, NON-human obstacle AABBs (low, high) + labels at time ``t`` (ur_base_link)."""
    boxes, labels = [], []
    for ob in scenario.obstacles:
        if not ob.observed or ob.motion.obstacle_class in _HUMAN:
            continue
        lo, hi = ob.aabb_at(t)
        boxes.append((tuple(float(v) for v in lo), tuple(float(v) for v in hi)))
        labels.append(ob.name)
    return boxes, labels


def perception_gate(scenario: bench.Scenario, frames) -> bench.GateVerdict:
    """Bench-compatible Gate backed by the real merged ``PerceptionCollisionGate``.

    Returns ``refuse`` when the real gate blocks any leg (collision prevented before motion),
    ``clear`` when it lets every leg through.  ``frames`` is unused -- the real gate recomputes its
    own swept gripper+carton corridor + perceived clearance."""
    global _GATE
    from twin.restock_manipulation import collision_gate, esdf_source
    from twin.tools.replay_collect_gate import replay_gate
    if _GATE is None:
        _GATE = collision_gate.PerceptionCollisionGate(margin_m=scenario.margin_m, mode='refuse',
                                                       logger=None)
    arm = _arm()
    traj = bench.build_trajectory(scenario)
    poses = [_pad_pose7(arm, cfg) for cfg in traj]
    held = scenario.arm_leg in bench.CARRY_LEGS
    phase = PHASE_FOR.get(scenario.arm_leg, 'approach')
    legs = [(phase, poses[i], poses[i + 1], held) for i in range(len(poses) - 1)]
    boxes, labels = _static_boxes(scenario)
    if not boxes:
        # no ESDF obstacle in this gate's scope (e.g. the moving person/hand -> human gate owns stop)
        return bench.GateVerdict('clear', 0, 0.0,
                                 'no perceived-ESDF obstacle in scope (human-class -> human gate owns the STOP)')
    source = esdf_source.AnalyticBoxEsdfSource(boxes, labels=labels, voxel_size=0.02,
                                               origin=(-1.5, -1.5, -0.6), size=(3.2, 3.2, 2.2))
    verdicts = replay_gate(legs, source, gate=_GATE, carton_dims=bench.CARTON_DIMS_M)
    min_clr = min((v.min_clearance for v in verdicts if math.isfinite(v.min_clearance)),
                  default=float('inf'))
    blocked = [v for v in verdicts if v.blocked]
    if blocked:
        return bench.GateVerdict('refuse', 0, 0.0,
                                 'real gate blocked %d/%d leg(s) (min perceived %.4f m < margin): '
                                 'strike prevented before motion'
                                 % (len(blocked), len(verdicts), min_clr))
    return bench.GateVerdict('clear', 0, 0.0,
                             'real gate: every leg proceeds (min perceived %.4f m)' % min_clr)
