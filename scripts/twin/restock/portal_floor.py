"""CPU geometry of the sales_backroom_portal from-rest tow pin.

The coupled outbound tow pins at x=-3.935 to the millimetre, from rest, while aligned,
with the drive wheels at constant z (no climb) and both casters airborne. A full CPU
scan of the composed stage finds NO collider edge, seam, or z-step there: the
``Backroom_floor`` is a single flat continuous slab with top z=0. The pin is a PhysX
**triangle-mesh internal-edge ("ghost") contact** on that flat floor:

* ``Backroom_floor`` is authored with collision ``approximation = "none"`` (a full
  triangle mesh). PhysX cooks its top quad into two triangles sharing one internal
  diagonal edge. On a flat surface that edge carries an in-plane (horizontal) contact
  normal -- exactly the "vertical in-plane edge, no z climb" the trace shows.
* The cart's front-LEFT swivel caster lands on that diagonal. Measured perpendicular
  distance of the front-left caster to the cooked diagonal: launch1b -0.002 m,
  coldwin18 -0.014 m, coldwin9 +0.040 m (all pinned); launch1a +0.238 m (the one run
  that crossed -- its caster was clear of the edge). So the discriminator is caster-on-
  edge, not momentum (coldwin9 approached FASTER than launch1a yet pinned).
* The base is a rigid-ish coupled train: base_x = cart_x + (cart_hitch_x - hitch_x) =
  cart_x + 1.27. The cart caster catches at a near-fixed cart pose (cart_x ~ -5.205), so
  the base halts at -5.205 + 1.27 = -3.935 to the millimetre -- the whole 7-run cluster.

Fix: author the flat drive floor as a BOX (``boundingCube``) collider. A box has six
flat faces and NO internal edge, so the casters roll across cleanly. This module is the
pure-geometry proof shared by ``scripts/check_portal_floor_seam.py`` and the tests; it
imports no pxr. See ``data/sim-robust-portal-floor-seam-v1/report.md``.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Sequence, Tuple

from ..mobile import MobileSpec

# Backroom_floor top-face quad, world XY, in the authored USD vertex order
# (store_environment.box('Backroom_floor', (13, 8.25, .12), (-5.5, 15.125, -.12))).
# PhysX/USD triangulates a quad [v0,v1,v2,v3] as a fan from v0 -> triangles (v0,v1,v2)
# and (v0,v2,v3); the single shared (internal) edge is the diagonal v0-v2.
BACKROOM_FLOOR_TOP_QUAD: Tuple[Tuple[float, float], ...] = (
    (-12.0, 11.0), (1.0, 11.0), (1.0, 19.25), (-12.0, 19.25))

# The tow lane runs east along y~=15.25 and the cart casters sweep this XY band as the
# train threads the portal (cart body trails the base by the hitch offset). "At the
# portal" = near the authored throat x=-3.875; the caster band reaches ~1.3 m west of it.
PORTAL_CASTER_BAND_X: Tuple[float, float] = (-5.6, -3.8)
PORTAL_CASTER_BAND_Y: Tuple[float, float] = (14.8, 15.7)


def hitch_offset(spec: MobileSpec = MobileSpec()) -> float:
    """Coupled base_x - cart_x when the train is aligned (yaw 0): the hitch fixes
    base_x + hitch_x == cart_x + cart_hitch_x."""
    return spec.cart_hitch_x - spec.hitch_x


def drive_wheel_contact_x(base_x: float, spec: MobileSpec = MobileSpec()) -> float:
    """World x of the drive-wheel ground contact. The wheels are jointed at base-local
    x=0 (``left/right_wheel_joint`` origin ``(0, +-track/2, r-base_z)``), so the contact
    patch sits directly under the base origin: contact x == base origin x."""
    return base_x


def caster_contact_z(spec: MobileSpec = MobileSpec()) -> float:
    """Height of a trailing-caster wheel's lowest point above the floor. ``caster_lift_m``
    raises it that far above the drive-wheel plane; >0 => the casters are airborne and the
    two drive wheels bear the load (as the pinned-run trace shows)."""
    return spec.caster_lift_m


def cart_caster_local(spec: MobileSpec = MobileSpec()) -> List[Tuple[float, float]]:
    """The four cart caster WHEEL body centres in cart-local XY (author_cart:
    x in (-l/2+0.10, l/2-0.10) then wheel offset -0.025; y in (-w/2+0.08, w/2-0.08))."""
    l, w = spec.cart_length, spec.cart_width
    xs = (-l / 2 + 0.10, l / 2 - 0.10)
    ys = (-w / 2 + 0.08, w / 2 - 0.08)
    return [(x - 0.025, y) for x in xs for y in ys]


def cart_caster_world(cart_pose: Sequence[float],
                      spec: MobileSpec = MobileSpec()) -> List[Tuple[float, float]]:
    """World XY of the four cart casters for a cart pose ``(x, y, yaw_rad)``."""
    cx, cy, cyaw = cart_pose[0], cart_pose[1], cart_pose[2]
    c, s = math.cos(cyaw), math.sin(cyaw)
    return [(cx + c * lx - s * ly, cy + s * lx + c * ly) for lx, ly in cart_caster_local(spec)]


def _line_from_segment(a, b):
    ax, ay = a
    bx, by = b
    dx, dy = bx - ax, by - ay
    return ax, ay, dx, dy


def perp_distance(point, seg_a, seg_b) -> float:
    """Signed perpendicular distance from ``point`` to the infinite line through the
    segment endpoints (normal = segment direction rotated -90 deg, normalised)."""
    ax, ay, dx, dy = _line_from_segment(seg_a, seg_b)
    n = math.hypot(dx, dy)
    if n == 0:
        return math.hypot(point[0] - ax, point[1] - ay)
    nx, ny = dy / n, -dx / n
    return (point[0] - ax) * nx + (point[1] - ay) * ny


def cooked_diagonal(top_quad=BACKROOM_FLOOR_TOP_QUAD):
    """The internal diagonal edge PhysX cooks for a triangle-mesh quad top (fan from v0):
    the segment v0 -> v2."""
    return top_quad[0], top_quad[2]


def internal_floor_edges(collider_kind: str, top_quad=BACKROOM_FLOOR_TOP_QUAD):
    """Internal collision edges of the floor top under the drive surface.

    A triangle mesh (``approximation="none"``) cooks one internal diagonal; a box
    (``boundingCube``) / convex hull of the slab has none (only the outer box edges,
    which lie at the floor perimeter, far from the lane). Returns a list of
    ``(a, b)`` segments."""
    from .. import colliders
    if collider_kind in (colliders.TRIANGLE_MESH, "none", "triangle_mesh"):
        return [cooked_diagonal(top_quad)]
    if collider_kind in (colliders.BOX, "boundingCube", "box",
                         colliders.CONVEX_HULL, "convexHull", "convex_hull"):
        return []
    raise ValueError(f"unknown collider kind {collider_kind!r}")


def _seg_crosses_band(seg, xband, yband) -> bool:
    """Does the segment pass through the axis-aligned XY band? Sampled densely (the
    segment is a straight diagonal, so a fine sample is exact enough for a few-cm band)."""
    (ax, ay), (bx, by) = seg
    steps = 2000
    for i in range(steps + 1):
        t = i / steps
        x = ax + t * (bx - ax)
        y = ay + t * (by - ay)
        if xband[0] <= x <= xband[1] and yband[0] <= y <= yband[1]:
            return True
    return False


def edges_in_portal_band(collider_kind: str, top_quad=BACKROOM_FLOOR_TOP_QUAD,
                         xband=PORTAL_CASTER_BAND_X, yband=PORTAL_CASTER_BAND_Y):
    """The internal floor edges that cross the portal caster band -- the ones a tow
    caster can ghost-collide. Empty for a box floor (the fix); one diagonal for a
    triangle-mesh floor (the defect)."""
    return [seg for seg in internal_floor_edges(collider_kind, top_quad)
            if _seg_crosses_band(seg, xband, yband)]


@dataclass(frozen=True)
class CasterOnEdge:
    label: str
    caster_xy: Tuple[float, float]
    perp_m: float


def front_caster_edge_analysis(cart_pose, spec: MobileSpec = MobileSpec(),
                               top_quad=BACKROOM_FLOOR_TOP_QUAD) -> List[CasterOnEdge]:
    """For a cart pose, the perpendicular distance of each cart caster to the cooked
    diagonal -- the root-cause evidence (front-left caster perp ~0 on every pinned run)."""
    a, b = cooked_diagonal(top_quad)
    out = []
    labels = ['rear_right', 'rear_left', 'front_right', 'front_left']
    for label, xy in zip(labels, cart_caster_world(cart_pose, spec)):
        out.append(CasterOnEdge(label, (round(xy[0], 4), round(xy[1], 4)),
                                round(perp_distance(xy, a, b), 4)))
    return out


# --------------------------------------------------------------------------- #
# Option C — the flat drive floor is a box; make a ROLLING wheel rest ON it at #
# z=0 (not sink to the redundant Ground_exterior sub-floor at z=-0.010).       #
# The whole-slab boundingCube keeps the collect station on the box too, so the #
# key is that its resting contact matches the mesh (wheel_z = wheel_radius).   #
# Pure geometry/contact helpers for the CPU proof; no pxr.                     #
# --------------------------------------------------------------------------- #

# Measured top-surface z of the drive-floor colliders (store USD, metres). The box's
# boundingCube top equals the triangle-mesh top exactly, so the switch authors NO z step.
DRIVE_FLOOR_TOP_Z = 0.0
# The redundant sub-floor backdrop (Ground_exterior) top, 10 mm below the real floors.
SUBFLOOR_TOP_Z = -0.010


def wheel_rest_contact_z(box_top_z: float = DRIVE_FLOOR_TOP_Z,
                         box_rest_offset: float = 0.0,
                         wheel_rest_offset: float = 0.0) -> float:
    """World z of a wheel's lowest contact point when it rests on the flat-floor box.

    PhysX rests two shapes separated by the sum of their rest offsets, so the wheel's
    lowest point sits at ``box_top_z + box_rest_offset + wheel_rest_offset``. With both
    offsets 0 and the box top at z=0 the contact is at z=0 (so the axle/wheel_z sits one
    wheel radius above => 0.160 for the 0.16 m drive wheel)."""
    return box_top_z + box_rest_offset + wheel_rest_offset


def wheel_z_on_box(spec: MobileSpec = MobileSpec(), **kw) -> float:
    """The drive-wheel axle height (the trace's ``casters.wheel_z``) when resting on the
    box: contact z + wheel radius. Expected 0.160 with zero offsets and box top z=0."""
    return wheel_rest_contact_z(**kw) + spec.wheel_radius


def ground_exterior_top_quad(half: float = 60.0, top_z: float = SUBFLOOR_TOP_Z):
    """The Ground_exterior backdrop's top quad (an axis-aligned square, authored v0..v3)."""
    return ((-half, -half), (half, -half), (half, half), (-half, half))


def subfloor_edges_on_path(points, half: float = 60.0) -> List[float]:
    """Perp distances from each probe point to Ground_exterior's cooked top diagonal
    (fan v0->v2 = corner to corner). Large everywhere on the tow lane (>13 m), i.e. the
    sub-floor has NO internal edge on any caster/wheel path -- the box-run pin is not a
    Ground_exterior edge."""
    q = ground_exterior_top_quad(half)
    a, c = q[0], q[2]
    return [abs(perp_distance(p, a, c)) for p in points]


def resolve_config(portal_floor_box):
    """Normalise the opt-in ``portal_floor_box`` runtime value into a settings dict.

    Accepts a bool or a mapping; absent/falsey => ``enabled=False`` (byte-identical off).
    Keys (all optional, with the option-C defaults):
      * ``enabled``        -- author the flat-floor box at all.
      * ``rest_offset``    -- PhysX restOffset on the box (default 0.0 => contact at box top).
      * ``contact_offset`` -- PhysX contactOffset on the box (default 0.02 m margin).
      * ``drop_subfloor``  -- drop the redundant Ground_exterior sub-floor collider so the
                              rolling wheel cannot sink to z=-0.010 (default True when enabled).
    """
    from .. import colliders
    if not portal_floor_box:
        return dict(enabled=False, rest_offset=None, contact_offset=None, drop_subfloor=False)
    if not isinstance(portal_floor_box, dict):
        cfg = {}
        enabled = bool(portal_floor_box)
    else:
        cfg = portal_floor_box
        enabled = bool(cfg.get('enabled', True))
    if not enabled:
        return dict(enabled=False, rest_offset=None, contact_offset=None, drop_subfloor=False)
    return dict(
        enabled=True,
        rest_offset=float(cfg.get('rest_offset', colliders.DRIVE_FLOOR_BOX_REST_OFFSET)),
        contact_offset=float(cfg.get('contact_offset', colliders.DRIVE_FLOOR_BOX_CONTACT_OFFSET)),
        drop_subfloor=bool(cfg.get('drop_subfloor', True)),
    )
