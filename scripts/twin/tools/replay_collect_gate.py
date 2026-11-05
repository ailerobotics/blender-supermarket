"""Replay a collect run's arm legs through the perceived-clearance gate (CPU, no GPU/ROS).

Drives :class:`twin.restock_manipulation.collision_gate.PerceptionCollisionGate` leg-by-leg over a
collect sequence of ``gripper_pad_tcp`` poses against an analytic ESDF (the known cart-cage
geometry + any planted obstacle), printing the per-leg ``PERCEIVED_CLEARANCE`` verdict. It shows
what the gate would decide for each leg of a collect WITHOUT a GPU run, so a camera-side / margin
change is provable in milliseconds. The leg poses come from ``--legs <json>`` (a recorded run's
poses) or the built-in representative collect sequence; obstacles come from the cart-cage boxes and
optional ``--planted-post``. The verdicts are labelled analytic, not sensor evidence.

Hook for the bench lane (sim-arm-collision-bench-v1): :func:`replay_gate` takes any list of legs and
any :class:`esdf_source.EsdfSource`, so a scripted static/moving-obstacle field scores the same way.
"""
from __future__ import annotations

import argparse
import json

from twin.restock_manipulation import collision_gate, esdf_source

PAD_MIN = (-0.145, -0.067, -0.272)
PAD_MAX = (0.146, 0.059, 0.002)

#: A representative collect leg sequence in ur_base_link (top grasp, pad +Z down). Each leg is
#: ``(phase, start_pose7, goal_pose7, held)``. The reach target sits at base +Y ~1 m (the source
#: shelf), the cart to the robot's side (+x). Replace with ``--legs`` for a recorded run's poses.
REPRESENTATIVE_COLLECT_LEGS = [
    ('approach', (0.0, 0.30, 0.60, 1, 0, 0, 0), (0.0, 1.00, 0.52, 1, 0, 0, 0), False),
    ('contact',  (0.0, 1.00, 0.52, 1, 0, 0, 0), (0.0, 1.00, 0.36, 1, 0, 0, 0), False),
    ('lift',     (0.0, 1.00, 0.36, 1, 0, 0, 0), (0.0, 1.00, 0.62, 1, 0, 0, 0), True),
    ('preplace', (0.0, 1.00, 0.62, 1, 0, 0, 0), (0.35, 0.45, 0.62, 1, 0, 0, 0), True),
    ('place',    (0.35, 0.45, 0.62, 1, 0, 0, 0), (0.35, 0.45, 0.47, 1, 0, 0, 0), True),
    ('retreat',  (0.35, 0.45, 0.47, 1, 0, 0, 0), (0.35, 0.45, 0.62, 1, 0, 0, 0), False),
]

#: Representative cart-cage-ish boxes in ur_base_link (illustrative; a real replay crops the live
#: ESDF). Posts flank the set-down column but clear of the carton-laden representative legs above
#: (the folded carton envelope reaches world x ~0.575, y ~0.60 at the place TCP, so the posts sit
#: beyond that with margin). No deck slab under the place column -- the baseline is clear; a
#: planted obstacle is what the gate must catch.
REPRESENTATIVE_CART_BOXES = [
    ((0.78, 0.15, 0.30), (0.85, 0.22, 1.00)),   # a front post, clear of the set-down carton
    ((0.78, 0.70, 0.30), (0.85, 0.77, 1.00)),   # another post
]


def _carton_dims_for(leg_held, carton_dims):
    return carton_dims if leg_held else None


def replay_gate(legs, source, *, gate=None, now0=10.0, carton_dims=(0.45, 0.30, 0.18)):
    """Return the per-leg :class:`GateVerdict` list for ``legs`` against ``source``.

    ``legs`` is an iterable of ``(phase, start_pose, goal_pose, held)``. A fresh snapshot is pulled
    per leg (sim time advances a little each leg), the held carton is folded into the swept envelope.
    """
    gate = gate or collision_gate.PerceptionCollisionGate(margin_m=0.03, mode='refuse')
    verdicts = []
    for i, (phase, start, goal, held) in enumerate(legs):
        now = now0 + 0.1 * i
        from twin.restock_manipulation.esdf_grid import combined_envelope
        pad_min, pad_max = combined_envelope(PAD_MIN, PAD_MAX,
                                             _carton_dims_for(held, carton_dims))
        snapshot = source.snapshot(now)
        verdicts.append(gate.evaluate(phase=phase, start_pose=start, goal_pose=goal,
                                      pad_min=pad_min, pad_max=pad_max, snapshot=snapshot,
                                      now=now, held=held))
    return verdicts


def _build_source(boxes, planted_post, *, voxel=0.03):
    boxes = list(boxes)
    labels = ['cart_%d' % i for i in range(len(boxes))]
    if planted_post is not None:
        x, y, z = planted_post
        boxes.append(((x - 0.03, y - 0.03, z - 0.30), (x + 0.03, y + 0.03, z + 0.30)))
        labels.append('planted_post')
    return esdf_source.AnalyticBoxEsdfSource(
        boxes, origin=(-0.6, 0.0, 0.0), size=(1.4, 1.5, 1.3), voxel_size=voxel, labels=labels)


def main(argv=None):
    ap = argparse.ArgumentParser(description='Replay a collect run through the perceived-clearance gate.')
    ap.add_argument('--legs', help='JSON list of [phase, start_pose7, goal_pose7, held] legs '
                                    '(default: a representative collect sequence)')
    ap.add_argument('--planted-post', help='x,y,z of a planted post obstacle in ur_base_link')
    ap.add_argument('--margin-m', type=float, default=0.03)
    ap.add_argument('--mode', default='refuse', choices=['warn', 'hold', 'refuse'])
    args = ap.parse_args(argv)

    legs = (json.loads(open(args.legs).read()) if args.legs
            else [list(l) for l in REPRESENTATIVE_COLLECT_LEGS])
    post = tuple(float(v) for v in args.planted_post.split(',')) if args.planted_post else None
    source = _build_source(REPRESENTATIVE_CART_BOXES, post)
    gate = collision_gate.PerceptionCollisionGate(margin_m=args.margin_m, mode=args.mode)
    verdicts = replay_gate([tuple(l) for l in legs], source, gate=gate)
    blocked = [v for v in verdicts if v.blocked]
    print('COLLECT_GATE_REPLAY legs=%d blocked=%d (%s)'
          % (len(verdicts), len(blocked), ', '.join('%s:%s' % (v.phase, v.action) for v in verdicts)))
    return 1 if blocked else 0


if __name__ == '__main__':
    import sys
    sys.exit(main())
