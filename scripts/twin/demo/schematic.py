"""A top-down schematic render of the demo scene (CPU, pure PIL).

Used by the ``--synthetic`` preview/fallback path to draw a readable map view of the
base, cart, carton, tow lanes, backstock and shelf from the choreography poses. The
PRIMARY deliverable renders the real Isaac viewport instead (:mod:`twin.demo.run_demo`);
this exists so the whole offline pipeline (panel + composite + encode) can be validated
and previewed without the GPU, and as a safety-net scene source.
"""
from __future__ import annotations

import math

from PIL import Image, ImageDraw

from . import choreography as C

BG = (24, 26, 30)
FLOOR = (36, 40, 46)
LANE_OUT = (210, 170, 40)
LANE_RET = (90, 150, 210)
SHELF = (180, 70, 60)
SUPPORT = (120, 120, 130)
ROBOT = (235, 235, 240)
ARM = (200, 205, 215)
CART = (150, 200, 150)
CARTON = (240, 200, 70)
TEXT = (150, 158, 168)

# World window (metres) covering the whole mission.
X0, X1 = -10.2, -0.8
Y0, Y1 = 3.0, 18.4


def _mk_projector(w, h, pad=24):
    def proj(x, y):
        # +x world -> right; +y world -> up (flip pixel y).
        px = pad + (x - X0) / (X1 - X0) * (w - 2 * pad)
        py = pad + (Y1 - y) / (Y1 - Y0) * (h - 2 * pad)
        return (px, py)
    scale = (w - 2 * pad) / (X1 - X0)
    return proj, scale


def _rect(draw, proj, cx, cy, yaw, lx, ly, fill, outline=None, width=2):
    c, s = math.cos(yaw), math.sin(yaw)
    corners = [(-lx / 2, -ly / 2), (lx / 2, -ly / 2), (lx / 2, ly / 2), (-lx / 2, ly / 2)]
    pts = []
    for ox, oy in corners:
        wx = cx + c * ox - s * oy
        wy = cy + s * ox + c * oy
        pts.append(proj(wx, wy))
    draw.polygon(pts, fill=fill, outline=outline or fill, width=width)
    return pts


def render_scene(frame, size=(960, 720)):
    """Render one choreography frame as a top-down schematic image."""
    w, h = size
    img = Image.new('RGB', (w, h), BG)
    d = ImageDraw.Draw(img)
    proj, scale = _mk_projector(w, h)

    # Floor.
    d.rectangle([proj(X0, Y1), proj(X1, Y0)], fill=FLOOR)

    # Tow lanes (authored centrelines).
    d.line([proj(x, y) for x, y, _ in C.OUTBOUND_WAYPOINTS], fill=LANE_OUT, width=3, joint='curve')
    d.line([proj(x, y) for x, y, _ in C.RETURN_WAYPOINTS], fill=LANE_RET, width=3, joint='curve')

    # Shelf endcap G3_back (near the slot target) and backstock support.
    _rect(d, proj, C.P_SHELF[0], C.P_SHELF[1] - 0.25, 0.0, 1.2, 0.5, SHELF)
    _rect(d, proj, C.P_SUPPORT[0], C.P_SUPPORT[1], 0.0, 0.68, 0.48, SUPPORT)
    # Cart parking marker.
    px, py = proj(*C.CART_PARKING[:2])
    d.ellipse([px - 6, py - 6, px + 6, py + 6], outline=CART, width=2)

    # Cart (rollwagen) with open +x mouth hint.
    bx, by, byaw = frame['base']
    cx, cy, cyaw = frame['cart']
    _rect(d, proj, cx, cy, cyaw, 0.80, 0.68, CART, outline=(90, 140, 90), width=2)
    # draw cage mouth (+x face) as a brighter edge
    mc, ms = math.cos(cyaw), math.sin(cyaw)
    m0 = proj(cx + mc * 0.4 - ms * 0.34, cy + ms * 0.4 + mc * 0.34)
    m1 = proj(cx + mc * 0.4 + ms * 0.34, cy + ms * 0.4 - mc * 0.34)
    d.line([m0, m1], fill=(200, 240, 200), width=3)

    # Robot base + a short arm stub + heading arrow.
    _rect(d, proj, bx, by, byaw, 0.9, 0.62, ROBOT, outline=(120, 120, 130), width=2)
    hx, hy = proj(bx + math.cos(byaw) * 0.6, by + math.sin(byaw) * 0.6)
    d.line([proj(bx, by), (hx, hy)], fill=(80, 160, 230), width=3)
    d.ellipse([hx - 4, hy - 4, hx + 4, hy + 4], fill=(80, 160, 230))

    # Carton.
    kx, ky, kz, kyaw = frame['carton']
    _rect(d, proj, kx, ky, kyaw, 0.60, 0.40, CARTON, outline=(180, 150, 40), width=2)

    # Labels.
    d.text(proj(C.P_SHELF[0] - 0.55, C.P_SHELF[1] + 0.2), 'G3_back_s2', fill=TEXT)
    d.text(proj(C.CART_PARKING[0] - 0.4, C.CART_PARKING[1] - 0.5), 'cart parking', fill=TEXT)
    d.text((10, h - 20), 'TOP-DOWN SCHEMATIC (synthetic preview)', fill=TEXT)
    return img
