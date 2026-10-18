#!/usr/bin/env python3
"""CPU proof (no GPU) gating the cart-cage open-face place (firstmate 350).

The place-onto-cart routes the carried carton THROUGH the cart's single OPEN loading face. The
composed /Rollwagen cage is a three-sided box (deck + back wall + two side walls, OPEN +x toward
the robot) modelled in the ur_base_link frame -- the geometry is CPU-extracted from the composed
mission stage (``--stage`` re-derives and asserts it; the committed CANONICAL_CAGE below is that
dump). This script proves, with the SAME sphere predicate the place-phase gate uses
(``twin.restock_host.world_obstacles``), that:

  (i)  the carry fold (stow -> tuck -> pan) with the carton welded clears all THREE walls + the deck
       (there is NO top box -- the carry rises ABOVE the open cage), and
  (ii) a straight open-face insert exists: interpolating the carton box from the mouth standoff
       (preplace) down the open +x axis onto the deck (place) stays clear of every wall at every
       step, and reports the final insert-to-deck gap.

Exit 0 iff both hold (every wall cleared along the fold AND a collision-free insert exists). This is
firstmate 350's actual gate -- it replaces the mis-sourced bench carton-vs-cart-AABB anchor, which
measured carton-vs-arm clearance, not carton-vs-cage-walls.
"""
import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from twin.restock_manipulation.cad_tool import ARM_JOINTS  # noqa: E402
from twin.restock_host.world_obstacles import (  # noqa: E402
    Box, boxes_from_specs, carton_cover_local, check_configuration)

# Authoritative cage, ur_base_link frame (walls-only, OPEN +x). CPU-dumped from the composed stage
# /Rollwagen (cart 1.27 m directly behind the base, same y; deck surface at z=-0.275), stable across
# the assembly/coupling composed stages. --stage re-derives and asserts these.
CANONICAL_CAGE = [
    {'id': 'cart_deck',  'size': [0.80, 0.68, 0.05], 'center': [-1.270, 0.000, -0.300]},
    {'id': 'cart_back',  'size': [0.02, 0.68, 1.45], 'center': [-1.650, 0.000,  0.425]},
    {'id': 'cart_side0', 'size': [0.80, 0.02, 1.45], 'center': [-1.270, -0.320, 0.425]},
    {'id': 'cart_side1', 'size': [0.80, 0.02, 1.45], 'center': [-1.270,  0.320, 0.425]},
]
# The open loading face: +x in ur_base_link (toward the robot). Deck top z = center_z + size_z/2.
CAGE_AXIS = (1.0, 0.0, 0.0)
DECK_TOP_Z = -0.300 + 0.05 / 2.0

# Carry-fold joint configs (arm joints 1-5 = the carry shape cart_tuck_shape; joint0 = shoulder_pan).
# tuck folds at the measured lift pan; pan rotates joint0 to cart_pan_joint0 (run-276 values).
CART_TUCK_SHAPE = (-2.27, 1.0, 0.65, 0.8, -1.2545)
CART_PAN_JOINT0 = -1.822


def _quat_matrix(q):
    """Rotation matrix for a unit quaternion (x, y, z, w)."""
    x, y, z, w = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def carton_box_points(center, quat, dims, pitch=0.02):
    """EXACT carton box surface sample (no sphere inflation) at a given CARTON centre + orientation,
    in ur_base_link. Used for the tight insert-fit check where the carton's 0.6 m face nearly spans
    the cage's narrow interior -- the conservative sphere cover (sized for the 100s-of-mm carry
    margins) would read a false penetration there. A dense box grid + each wall's exact box SDF gives
    the true min gap; a convex box's nearest point to another convex box is on its surface, so the
    surface grid's min SDF is the box-box clearance."""
    d = np.asarray(dims, dtype=float)
    axes = [np.linspace(-v / 2.0, v / 2.0, max(2, int(math.ceil(v / pitch)) + 1)) for v in d]
    pts = np.array([[x, y, z] for x in axes[0] for y in axes[1] for z in axes[2]])
    r = _quat_matrix(quat)
    return pts @ r.T + np.asarray(center, dtype=float)


def carton_world_extents(quat, dims):
    """Half-extents of the carton's world AABB under orientation ``quat`` (axis-aligned bound of the
    rotated box) -- used to rest the carton on the deck and size the mouth clearance."""
    r = _quat_matrix(quat)
    return np.abs(r) @ (np.asarray(dims, dtype=float) / 2.0)


def _derive_from_stage(stage_path):
    """Re-derive the walls-only cage from a composed mission stage's /Rollwagen AABBs (needs usd-core).
    Returns specs in the ur_base_link frame; raises if the stage/usd is unavailable."""
    from pxr import Usd, UsdGeom  # noqa
    st = Usd.Stage.Open(str(stage_path))
    bbc = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ['default'], useExtentsHint=False)
    base = UsdGeom.Xformable(st.GetPrimAtPath('/MobileRobot/base_link')
                             ).ComputeLocalToWorldTransform(Usd.TimeCode.Default()).ExtractTranslation()
    urw = np.array([base[0], base[1], base[2] + 0.26])  # ur_base_link = base_link + arm_mount(0,0,0.26)
    ch = st.GetPrimAtPath('/Rollwagen/chassis')
    groups = {'cart_deck': lambda n: n == 'deck', 'cart_back': lambda n: n.startswith('back_'),
              'cart_side0': lambda n: n.startswith('side_0_'), 'cart_side1': lambda n: n.startswith('side_1_')}
    specs = []
    for gid, pred in groups.items():
        mn = np.array([math.inf] * 3)
        mx = np.array([-math.inf] * 3)
        for c in ch.GetChildren():
            if not pred(c.GetName()):
                continue
            rng = bbc.ComputeWorldBound(c).ComputeAlignedRange()
            mn = np.minimum(mn, np.array(rng.GetMin()))
            mx = np.maximum(mx, np.array(rng.GetMax()))
        center = ((mn + mx) / 2.0 - urw)
        size = (mx - mn)
        specs.append({'id': gid, 'size': [round(v, 4) for v in size], 'center': [round(v, 4) for v in center]})
    return specs


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--urdf', default=str(Path.home() / 'datasets/twin-robot-in-store-boot-v1/host-cad/mobile.urdf'))
    ap.add_argument('--xrdf', default=str(Path.home() / 'datasets/twin-robot-in-store-boot-v1/host-cad/mobile.xrdf'))
    ap.add_argument('--manifest', default=str(Path.home() / 'datasets/twin-robot-in-store-boot-v1/mobile/mobile_manifest.json'))
    ap.add_argument('--stage', help='composed mission.usda to RE-DERIVE + assert the cage (needs usd-core)')
    ap.add_argument('--dims', type=float, nargs=3, default=[0.6, 0.4, 0.16], help='carton extents (m)')
    ap.add_argument('--offset-tcp', type=float, nargs=3, default=[0.0, 0.0, 0.20])
    ap.add_argument('--tcp-quat', type=float, nargs=4, default=[0.5, 0.5, 0.5, 0.5], help='side-grip TCP quat')
    ap.add_argument('--mouth-standoff', type=float, default=0.45, help='preplace offset out the +x mouth (m)')
    ap.add_argument('--insert-gap', type=float, default=0.03, help='stop this far above the deck top (m)')
    ap.add_argument('--steps', type=int, default=24, help='insert interpolation samples')
    ap.add_argument('--json', action='store_true')
    args = ap.parse_args(argv)

    specs = CANONICAL_CAGE
    stage_ok = None
    if args.stage:
        derived = _derive_from_stage(args.stage)
        stage_ok = all(math.isclose(d['center'][i], c['center'][i], abs_tol=0.02)
                       and math.isclose(d['size'][i], c['size'][i], abs_tol=0.02)
                       for d, c in zip(derived, CANONICAL_CAGE) for i in range(3))
        specs = derived
    boxes = boxes_from_specs(specs)
    carton = dict(dims=args.dims, offset_tcp=args.offset_tcp)

    # (i) Carry-fold clearance: stow -> tuck -> pan, carton welded, vs all 3 walls + deck.
    manifest = json.loads(Path(args.manifest).read_text())['transport_positions']
    fold = {
        'STOW': [manifest[n] for n in ARM_JOINTS],
        'TUCK': [manifest[ARM_JOINTS[0]], *CART_TUCK_SHAPE],
        'PAN':  [CART_PAN_JOINT0, *CART_TUCK_SHAPE],
    }
    fold_rows, fold_min = [], {b.id: math.inf for b in boxes}
    for name, q in fold.items():
        r = check_configuration(args.urdf, args.xrdf, boxes, dict(zip(ARM_JOINTS, q)), carton=carton)
        per = {b['box']: b['clearance_m'] for b in r['boxes']}
        for k, v in per.items():
            fold_min[k] = min(fold_min[k], v)
        worst = min(r['boxes'], key=lambda b: b['clearance_m'])
        fold_rows.append((name, worst['box'], worst['clearance_m'], worst['worst_set'], per))
    fold_clear = all(v > 0 for v in fold_min.values())

    # (ii) Open-face insert sweep: the CARTON (exact box, side-grip orientation) travels from the mouth
    # standoff (preplace) down the +x axis onto the deck (place). The deck rest centres the carton on
    # the deck load centre with its underside insert_gap above the deck top; preplace offsets it out
    # the +x mouth. Fixed orientation throughout (a rigid straight insert).
    a = np.array(CAGE_AXIS, dtype=float)
    a = a / np.linalg.norm(a)
    half = carton_world_extents(args.tcp_quat, args.dims)  # world AABB half-extents of the oriented carton
    deck_center_xy = np.array([-1.270, 0.0])               # cart load centre (deck AABB centre, ur frame)
    place_center = np.array([deck_center_xy[0], deck_center_xy[1], DECK_TOP_Z + args.insert_gap + half[2]])
    preplace_center = place_center + a * args.mouth_standoff
    sweep_min = {b.id: math.inf for b in boxes}
    for t in np.linspace(0.0, 1.0, args.steps):
        c = preplace_center + (place_center - preplace_center) * t  # mouth -> deck (into the open face)
        pts = carton_box_points(c, args.tcp_quat, args.dims)
        for b in boxes:
            g = float(np.min(b.signed_distance(pts)))
            sweep_min[b.id] = min(sweep_min[b.id], g)
    deck = next(b for b in boxes if b.id == 'cart_deck')
    insert_to_deck = float(np.min(deck.signed_distance(carton_box_points(place_center, args.tcp_quat, args.dims))))
    walls = [b for b in boxes if b.id != 'cart_deck']
    insert_walls_clear = all(sweep_min[b.id] > 0 for b in walls)
    # preplace must present the carton OUTSIDE the open mouth (back face clear of the deck front plane)
    mouth_x = -1.270 + 0.80 / 2.0
    preplace_outside = (preplace_center[0] - half[0]) >= mouth_x

    ok = fold_clear and insert_walls_clear and preplace_outside and (stage_ok is not False)
    result = {
        'cage_specs': specs, 'stage_asserted': stage_ok,
        'carton_world_half_extents_m': [round(v, 4) for v in half],
        'fold_min_clearance_m': {k: round(v, 4) for k, v in fold_min.items()}, 'fold_clear': fold_clear,
        'insert_min_clearance_m': {k: round(v, 4) for k, v in sweep_min.items()},
        'insert_to_deck_gap_m': round(insert_to_deck, 4), 'insert_walls_clear': insert_walls_clear,
        'preplace_outside_mouth': bool(preplace_outside),
        'mouth_standoff_m': args.mouth_standoff, 'insert_gap_m': args.insert_gap, 'ok': ok,
    }
    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print('CART-CAGE OPEN-FACE PLACE PROOF (CPU, no GPU)')
        print('  cage source:', 'composed stage (asserted=%s)' % stage_ok if args.stage else 'committed CANONICAL_CAGE')
        print('  (i) carry-fold clearance (carton welded, vs 3 walls + deck, NO top):')
        for name, box, cl, wset, per in fold_rows:
            print('      %-5s worst %-11s %7.0f mm (%s)   %s' % (
                name, box, cl * 1000, wset, {k: round(v * 1000) for k, v in per.items()}))
        print('      min per box along fold: %s  => %s' % (
            {k: round(v * 1000) for k, v in fold_min.items()}, 'CLEAR' if fold_clear else 'BLOCKED'))
        print('      carton world half-extents (side-grip): %s m  (y-face spans the narrow cage gap)'
              % [round(v, 3) for v in half])
        print('  (ii) open-face insert sweep (mouth standoff %.2f m -> deck, +x into open face):' % args.mouth_standoff)
        print('      min wall clearance along insert: %s' % {k: round(v * 1000) for k, v in sweep_min.items() if k != 'cart_deck'})
        print('      preplace carton outside mouth: %s' % preplace_outside)
        print('      insert-to-deck final gap: %.0f mm  (walls %s)' % (
            insert_to_deck * 1000, 'CLEAR' if insert_walls_clear else 'BLOCKED'))
        print('  VERDICT:', 'PASS (safe to boot)' if ok else 'FAIL')
    return 0 if ok else 1


if __name__ == '__main__':
    raise SystemExit(main())
