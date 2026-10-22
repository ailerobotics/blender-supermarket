"""CPU proof that the demo-005 OUTBOUND tow leaves the backroom THROUGH THE DOOR.

The sales/backroom dividing wall lives at y=11.125 with a single clear opening, the
``sales_backroom_portal`` (``store_environment.py``: ``PORTAL_X``/``PORTAL_WIDTH`` and the
``Wall_back_{west,east}`` / ``Portal_jamb_*`` boxes). The earlier arc hugged x~-1.7 the whole
descent, so it crossed the wall line at x~-2.06 -- straight through the solid east wall. These
tests sample the base and cart footprints along every OUTBOUND frame of the real choreography
timeline and assert: zero intersection with the wall and both jambs, a positive clearance,
the crossing happens inside the door opening, the cart articulation stays under the 55 deg
budget, and the motion is a gentle monotonic forward arc (no reverse, no doubling back). They
also pin the COLLECT/COUPLE/TRANSFER anchor poses so this path change cannot silently move
them. Pure geometry -- no Isaac, no Blender.
"""
from __future__ import annotations

import math

import pytest

from twin.demo import choreography as C
from twin.mobile import MobileSpec

# --- Wall/door geometry, mirrored from scripts/store_environment.py -----------------
# PORTAL_X/PORTAL_WIDTH are imported so a design change there fails this test loudly.
from store_environment import PORTAL_X, PORTAL_WIDTH

WALL_Y = 11.125          # Wall_back_* centre (store_environment.build_environment)
WALL_T = 0.25            # wall box depth in y -> band [11.0, 11.25]
JAMB_T = 0.30            # Portal_jamb_* depth in y -> band [10.975, 11.275]
JAMB_W = 0.07            # jamb box width in x
LEFT = PORTAL_X - PORTAL_WIDTH / 2.0     # -5.125  west edge of the clear opening
RIGHT = PORTAL_X + PORTAL_WIDTH / 2.0    # -2.625  east edge of the clear opening

# Each box is (xmin, xmax, ymin, ymax) in the map frame.
WALL_BOXES = {
    'wall_west': (-13.25, LEFT, WALL_Y - WALL_T / 2, WALL_Y + WALL_T / 2),
    'wall_east': (RIGHT, 13.25, WALL_Y - WALL_T / 2, WALL_Y + WALL_T / 2),
    'jamb_west': (LEFT - JAMB_W, LEFT, WALL_Y - JAMB_T / 2, WALL_Y + JAMB_T / 2),
    'jamb_east': (RIGHT, RIGHT + JAMB_W, WALL_Y - JAMB_T / 2, WALL_Y + JAMB_T / 2),
}

SPEC = MobileSpec()
MARGIN = 0.05            # same conservative inflation twin.mobile.occupied_envelope uses
CLEARANCE_MIN = 0.12     # required gap from every footprint edge to the wall/jambs (m)
ART_BUDGET = math.radians(55)

# Base footprint radius: the circumscribed transport disk twin.mobile.occupied_envelope
# builds for the robot (arm + wheels), inflated by MARGIN then by the 16-gon circumscription.
BASE_R = (max(SPEC.transport_radius,
              math.hypot(SPEC.length / 2, (SPEC.wheel_track + SPEC.wheel_width) / 2))
          + MARGIN) / math.cos(math.pi / 16)


def _box_poly(b):
    xmin, xmax, ymin, ymax = b
    return [(xmin, ymin), (xmax, ymin), (xmax, ymax), (xmin, ymax)]


def _cart_poly(x, y, yaw):
    """Map-frame convex quad of the towed cart (deck + drawbar), inflated by MARGIN,
    matching the cart rectangle in twin.mobile.occupied_envelope."""
    c, s = math.cos(yaw), math.sin(yaw)
    pts = []
    for lx in (-SPEC.cart_length / 2 - MARGIN, SPEC.cart_hitch_x + MARGIN):
        for ly in (-SPEC.cart_width / 2 - MARGIN, SPEC.cart_width / 2 + MARGIN):
            pts.append((x + c * lx - s * ly, y + s * lx + c * ly))
    # order as a convex ring
    return [pts[0], pts[1], pts[3], pts[2]]


def _pt_box_dist(px, py, b):
    xmin, xmax, ymin, ymax = b
    dx = max(xmin - px, 0.0, px - xmax)
    dy = max(ymin - py, 0.0, py - ymax)
    return math.hypot(dx, dy)


def _sat_overlap(poly_a, poly_b):
    """True if two convex polygons overlap (separating-axis theorem)."""
    for poly in (poly_a, poly_b):
        n = len(poly)
        for i in range(n):
            ax, ay = poly[i]
            bx, by = poly[(i + 1) % n]
            nx, ny = -(by - ay), (bx - ax)  # edge normal
            amin = min(nx * px + ny * py for px, py in poly_a)
            amax = max(nx * px + ny * py for px, py in poly_a)
            bmin = min(nx * px + ny * py for px, py in poly_b)
            bmax = max(nx * px + ny * py for px, py in poly_b)
            if amax < bmin or bmax < amin:
                return False  # a separating axis exists
    return True


def _seg_pt_dist(px, py, ax, ay, bx, by):
    dx, dy = bx - ax, by - ay
    L2 = dx * dx + dy * dy
    t = 0.0 if L2 == 0 else max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / L2))
    return math.hypot(px - (ax + t * dx), py - (ay + t * dy))


def _poly_poly_dist(poly_a, poly_b):
    """Min distance between two convex polygons (0 if they overlap)."""
    if _sat_overlap(poly_a, poly_b):
        return 0.0
    best = math.inf
    for poly, other in ((poly_a, poly_b), (poly_b, poly_a)):
        n = len(other)
        for px, py in poly:
            for i in range(n):
                ax, ay = other[i]
                bx, by = other[(i + 1) % n]
                best = min(best, _seg_pt_dist(px, py, ax, ay, bx, by))
    return best


def _outbound_frames(fps=10):
    frames, _ = C.build_timeline(fps=fps)
    return [f for f in frames if f['stage'] == 'outbound']


# ---------------------------------------------------------------------------
def test_anchor_poses_untouched():
    """The door path must not move the COLLECT/COUPLE/TRANSFER anchors."""
    assert C.PICKUP == (-2.3, 12.75, C.SOUTH)
    assert C.LANE_OUT_START == (-2.3, 13.0, C.SOUTH)
    assert C.SHELF == (-3.05, 6.0, math.pi)
    assert C.OUTBOUND_WAYPOINTS[0] == C.LANE_OUT_START
    assert C.OUTBOUND_WAYPOINTS[-1] == C.SHELF


def test_base_and_cart_clear_the_wall_and_jambs():
    """No base or cart footprint intersects the wall or a jamb; all keep CLEARANCE_MIN."""
    min_base = math.inf
    min_cart = math.inf
    for f in _outbound_frames():
        bx, by, _ = f['base']
        cx, cy, cyaw = f['cart']
        cart = _cart_poly(cx, cy, cyaw)
        for name, box in WALL_BOXES.items():
            base_gap = _pt_box_dist(bx, by, box) - BASE_R
            cart_gap = _poly_poly_dist(cart, _box_poly(box))
            assert base_gap > 0.0, f'base intersects {name} at y={by:.2f}'
            assert cart_gap > 0.0, f'cart intersects {name} at y={cy:.2f}'
            min_base = min(min_base, base_gap)
            min_cart = min(min_cart, cart_gap)
    assert min_base >= CLEARANCE_MIN, f'base clearance {min_base:.3f} < {CLEARANCE_MIN}'
    assert min_cart >= CLEARANCE_MIN, f'cart clearance {min_cart:.3f} < {CLEARANCE_MIN}'


def _crossing_x(points):
    """x where the polyline of (x, y) first crosses the wall line y=WALL_Y."""
    for i in range(1, len(points)):
        x0, y0 = points[i - 1]
        x1, y1 = points[i]
        if (y0 - WALL_Y) * (y1 - WALL_Y) <= 0 and y0 != y1:
            t = (WALL_Y - y0) / (y1 - y0)
            return x0 + t * (x1 - x0)
    return None


def test_crossing_is_inside_the_door_opening():
    """Both the base and the towed cart cross the wall line within the clear opening."""
    frames = _outbound_frames()
    base_x = _crossing_x([(f['base'][0], f['base'][1]) for f in frames])
    cart_x = _crossing_x([(f['cart'][0], f['cart'][1]) for f in frames])
    assert base_x is not None and cart_x is not None
    # Inside the opening, with the base disk / cart half-width kept off both jambs.
    assert LEFT + BASE_R < base_x < RIGHT - BASE_R, f'base crosses at x={base_x:.3f}'
    assert LEFT < cart_x < RIGHT, f'cart crosses at x={cart_x:.3f}'
    # And comfortably near the portal centre, not scraping an edge.
    assert abs(base_x - PORTAL_X) < 0.6
    assert abs(cart_x - PORTAL_X) < 0.8


def test_articulation_under_budget():
    """Peak cart articulation across the whole tow stays under the 55 deg budget."""
    worst = 0.0
    for f in _outbound_frames():
        art = abs((f['cart'][2] - f['base'][2] + math.pi) % (2 * math.pi) - math.pi)
        worst = max(worst, art)
    assert worst < ART_BUDGET, f'articulation {math.degrees(worst):.1f} deg >= 55'


def test_gentle_monotonic_forward_arc():
    """y strictly decreases (no doubling back), no reverse, and the turn radius stays gentle."""
    frames = _outbound_frames()
    prev = None
    max_rate = 0.0
    for f in frames:
        bx, by, byaw = f['base']
        if prev is not None:
            px, py, pyaw = prev
            assert by <= py + 1e-6, f'y doubles back at {by:.3f} (prev {py:.3f})'
            ds = math.hypot(bx - px, by - py)
            if ds > 1e-4:
                # forward motion: displacement projects positively on the heading
                along = (bx - px) * math.cos(byaw) + (by - py) * math.sin(byaw)
                assert along > -1e-3, f'reverse motion at y={by:.3f}'
                dyaw = abs((byaw - pyaw + math.pi) % (2 * math.pi) - math.pi)
                max_rate = max(max_rate, dyaw / ds)
        prev = (bx, by, byaw)
    # Min turn radius >= ~0.85 m, on par with the proven demo-005 shelf arc (~0.94 m).
    assert 1.0 / max_rate > 0.85, f'min turn radius {1.0 / max_rate:.2f} m too tight'
