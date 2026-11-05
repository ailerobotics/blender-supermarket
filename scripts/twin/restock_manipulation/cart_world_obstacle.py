"""The known-pose rollwagen cart as cuMotion world CollisionObjects (pure, CPU).

The highest-value collision-awareness fix for the arm is NOT a camera: the rollwagen is rigidly
coupled, so its pose is known after ``couple``, yet the forward-looking wrist ESDF never covers
it (measured grid y_min -0.08 m; cart at base y ~ -0.8 m -- sim-perception-architecture-s1 s4/
world_obstacles.py). So the set-down phase is structurally blind and today leans on the
``cage_collision`` pair-filter + collision-OFF straight-line inserts. This module turns the
cart's authored collider geometry into the ``world_obstacles`` host-key box spec in the arm base
frame, so cuMotion plans the arm + attached carton around the real deck / cage uprights / drawbar
with **zero VRAM and no perception** -- the shared dependency the scout sequenced first (S0).

It is **opt-in and byte-identical off**: the emitted spec is only consumed when it is placed in
``config['world_obstacles']`` (today ``[]``); with it absent the host publishes nothing new. It
pairs with ``ros_host`` so a retired static obstacle and an added cart obstacle live behind the
same explicit host key. The cart pose in the base frame is a KNOWN mission geometry supplied by
the caller (the coupled hitch transform); this module only transforms + groups the collider AABBs
and proves (via :mod:`twin.restock_host.world_obstacles`) that they clear the arm at rest.

Pure: the box transform/grouping needs only numpy. Reading the authored cart colliders needs
``pxr`` (usd-core) via :func:`twin.restock_manipulation.place_clearance.rollwagen_collider_boxes`;
callers without USD pass an explicit ``{name: (lo, hi)}`` collider dict (the CPU tests do).
"""
from __future__ import annotations

import math
from typing import Dict, Optional, Sequence, Tuple

import numpy as np

#: How the authored rollwagen collider names group into published world obstacles. The cage
#: uprights/side/back bars are lumped into one obstacle (they are a rigid frame), the horizontal
#: supports (deck, upper shelf, drawbar) each stay their own box so a place onto the deck is
#: checked against exactly the deck. Matches :func:`place_clearance.is_cage_body` /
#: ``honest_collider_parts`` body classes.
CAGE_OBSTACLE_ID = 'rollwagen_cage'


def _rotz(yaw: float) -> np.ndarray:
    c, s = math.cos(yaw), math.sin(yaw)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def cart_pose_matrix(pose) -> np.ndarray:
    """4x4 cart-body -> ur_base_link transform from a ``(x, y, z, yaw)`` or a 4x4 matrix.

    The cart is level (the carton-level constraint), so a position + yaw is the full pose; a 4x4
    is accepted verbatim for a caller that already has the hitch transform.
    """
    arr = np.asarray(pose, float)
    if arr.shape == (4, 4):
        if not np.allclose(arr[3], [0, 0, 0, 1]):
            raise ValueError('cart pose matrix must be a homogeneous transform')
        return arr
    if arr.shape != (4,) or not np.isfinite(arr).all():
        raise ValueError('cart pose must be (x, y, z, yaw) or a 4x4 matrix')
    T = np.eye(4)
    T[:3, :3] = _rotz(float(arr[3]))
    T[:3, 3] = arr[:3]
    return T


def _aabb_to_base_box(lo, hi, T_base_cart, pad_m):
    """Transform a cart-frame AABB ``[lo, hi]`` by ``T_base_cart`` into a base-frame oriented box
    ``(size, center, rpy)``. The box keeps the cart's yaw (from ``T``) and its own extents; a yaw-
    only cart transform preserves the box exactly (no bounding-box inflation)."""
    lo = np.asarray(lo, float)
    hi = np.asarray(hi, float)
    size = (hi - lo) + 2.0 * float(pad_m)
    center_cart = (lo + hi) / 2.0
    R = np.asarray(T_base_cart, float)[:3, :3]
    t = np.asarray(T_base_cart, float)[:3, 3]
    center_base = R @ center_cart + t
    yaw = math.atan2(R[1, 0], R[0, 0])
    return (tuple(float(s) for s in size), tuple(float(c) for c in center_base),
            (0.0, 0.0, float(yaw)))


def rollwagen_world_obstacle_specs(collider_boxes: Dict[str, Tuple], cart_pose, *,
                                   pad_m: float = 0.0, include=None,
                                   cage_id: str = CAGE_OBSTACLE_ID):
    """``world_obstacles`` host-key specs for the coupled rollwagen, in the arm base frame.

    :param collider_boxes: ``{name: (lo, hi)}`` cart-body AABBs from
        :func:`twin.restock_manipulation.place_clearance.rollwagen_collider_boxes` (or an explicit
        dict in a no-USD test).
    :param cart_pose: the cart-body -> ur_base_link pose (``(x, y, z, yaw)`` or 4x4).
    :param pad_m: optional isotropic inflation (m) added to every box extent (a safety skin).
    :param include: optional iterable of collider names to keep (default all).
    :returns: a list of ``{'id', 'boxes': [{'size', 'center', 'rpy'}]}`` dicts, ready to drop into
        ``config['world_obstacles']``. Cage bodies are grouped under ``cage_id``; each horizontal
        support (deck/shelf/drawbar) is its own obstacle so a place is checked against exactly it.
    """
    from . import place_clearance
    T = cart_pose_matrix(cart_pose)
    names = list(collider_boxes) if include is None else [n for n in collider_boxes if n in set(include)]
    if not names:
        raise ValueError('no rollwagen colliders selected')
    cage_boxes = []
    specs = []
    for name in names:
        lo, hi = collider_boxes[name]
        size, center, rpy = _aabb_to_base_box(lo, hi, T, pad_m)
        box = {'size': size, 'center': center, 'rpy': rpy}
        if place_clearance.is_cage_body(name):
            cage_boxes.append(box)
        else:
            specs.append({'id': 'rollwagen_%s' % name, 'boxes': [box]})
    if cage_boxes:
        specs.insert(0, {'id': cage_id, 'boxes': cage_boxes})
    return specs


def rollwagen_world_obstacle_specs_from_usd(cart_pose, *, spec=None, pad_m=0.0, include=None,
                                            cage_id=CAGE_OBSTACLE_ID):
    """Convenience: read the authored cart colliders (needs ``pxr``) then build the specs."""
    from .place_clearance import rollwagen_collider_boxes
    return rollwagen_world_obstacle_specs(rollwagen_collider_boxes(spec=spec), cart_pose,
                                          pad_m=pad_m, include=include, cage_id=cage_id)


def check_cart_obstacle_clears_arm(specs, urdf_path, xrdf_path, *, configs=None,
                                   manifest_path=None):
    """Prove the emitted cart obstacles clear the arm at its rest configs (the S0 CPU gate).

    Reuses :mod:`twin.restock_host.world_obstacles`: a cuMotion world obstacle overlapping the
    start/stow configuration rejects every plan with ``INVALID_INITIAL_CSPACE`` (firstmate 295), so
    the cart boxes MUST clear the arm where it begins. Returns ``{id: min_clearance_m}`` over the
    supplied ``configs`` (default: the zero config; pass the transport/stow config for a stricter
    check). Raises when any obstacle intersects the arm (negative clearance).
    """
    from twin.restock_host import world_obstacles as wo
    boxes = []
    for spec in specs:
        for b in spec['boxes']:
            boxes.append(wo.Box(spec['id'], tuple(b['size']), tuple(b['center']),
                                tuple(b.get('rpy', (0.0, 0.0, 0.0)))))
    configs = configs if configs is not None else [{}]
    out = {}
    for cfg in configs:
        centres, radii, _owners = wo.arm_spheres_world(urdf_path, xrdf_path, cfg)
        for box in boxes:
            gap = wo.clearance(box, centres, radii)
            out[box.id] = min(gap, out.get(box.id, gap))
    worst = min(out.values()) if out else float('inf')
    if worst < 0:
        raise ValueError('rollwagen world obstacle intersects the arm at rest (min clearance '
                         '%.4f m) -- the coupled cart pose or padding is wrong' % worst)
    return out
