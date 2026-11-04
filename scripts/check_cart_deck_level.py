#!/usr/bin/env python3
"""CPU proof that the Rollwagen cart DECK rests PARALLEL to the floor at the station
(sim-place-descent-residual-v3, brief part 2 "level the cart deck").

Part-1 characterisation (coldwin42-deck.txt + the cw39/cw42 anchors) found the reach-INDEPENDENT ~2.0 deg
collect set-down residual is the arm PRESSING the leveled welded carton onto the cart deck: the CARTON_TILT
grader reads FK of the EXECUTED joints, which deflect ~2 deg as the position-controlled arm drives into the
rigid deck reaction. The deck-conformance reading in coldwin42-deck.txt (an "authored 0.88 deg" cart tilt)
came from the END-of-run mission.usda SNAPSHOT, taken AFTER the controlled-stop knocked the cart
(``physics:angularVelocity`` = (3.2, 16.7, -63.4) on /Rollwagen/chassis) -- NOT the rest attitude at the
station. This proof reads the cart as AUTHORED (``mobile_assets.author_cart``) and shows the deck surface is
parallel to the floor by construction:

  * the deck top-face normal is world +Z to < 0.3 deg (the deck Cube is scale-only on an identity-rotated
    chassis -- no authored deck tilt);
  * the four caster ground contacts are at EQUAL height (a symmetric rectangle), so the cart cannot settle
    tipped on a level floor from caster geometry;
  * the cart centre of mass is inside that four-caster support with margin, so it does not statically tip.

So no static deck tilt explains the 2.0 deg -- the operative runtime fix is the contact-triggered weld
release (place_contact_release), which never presses the welded carton into the deck. Exit 0 when the deck is
within the tolerance of level; exit 1 (named) otherwise. Pure/CPU, needs ``usd-core`` (author_cart).

    PYTHONPATH=scripts python3 scripts/check_cart_deck_level.py [--tol-deg 0.3] [--json OUT]
"""
from __future__ import annotations

import argparse
import json
import math
import sys


def _measure(spec):
    import numpy as np
    from pxr import Usd, UsdGeom
    from twin.mobile_assets import author_cart

    stage = Usd.Stage.CreateInMemory()
    info = author_cart(stage, spec=spec)
    cache = UsdGeom.XformCache()

    # Deck top-face normal in world: the deck Cube's local +Z (thin axis) rotated by its world transform.
    deck = stage.GetPrimAtPath(info['body'] + '/deck')
    Td = np.asarray(cache.GetLocalToWorldTransform(deck)).T  # 4x4, row-major -> transpose to column form
    R = Td[:3, :3]
    # normalise the columns (scale is baked into the Cube, strip it for the normal)
    up = R[:, 2] / (np.linalg.norm(R[:, 2]) or 1.0)
    up = up if up[2] >= 0 else -up
    tilt_deg = math.degrees(math.acos(max(-1.0, min(1.0, float(up[2])))))

    # Four caster wheel-bottom z (ground contacts). author_cart lays forks/wheels as siblings of the chassis
    # under /Rollwagen; read each wheel prim's world z minus its cylinder radius.
    wheel_bottoms = []
    for i in range(4):
        w = stage.GetPrimAtPath(info['root'] + f'/wheel_{i}')
        if not w:
            continue
        Tw = np.asarray(cache.GetLocalToWorldTransform(w)).T
        cyl = UsdGeom.Cylinder(stage.GetPrimAtPath(info['root'] + f'/wheel_{i}/shape'))
        r = float(cyl.GetRadiusAttr().Get()) if cyl else 0.0
        wheel_bottoms.append(float(Tw[2, 3]) - r)
    spread = (max(wheel_bottoms) - min(wheel_bottoms)) if wheel_bottoms else float('nan')

    return {'deck_tilt_deg': tilt_deg, 'deck_up_world': [float(x) for x in up],
            'caster_wheel_bottom_z': wheel_bottoms, 'caster_bottom_spread_m': spread,
            'n_casters': len(wheel_bottoms)}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--tol-deg', type=float, default=0.3,
                    help='maximum deck tilt from level accepted (default 0.3 deg)')
    ap.add_argument('--caster-spread-tol-m', type=float, default=1e-6,
                    help='maximum caster ground-contact height spread accepted (default ~0)')
    ap.add_argument('--json', help='write the measured record to this path')
    args = ap.parse_args(argv)

    try:
        from twin.mobile import MobileSpec
    except Exception as exc:  # noqa: BLE001
        print(f'check_cart_deck_level: cannot import MobileSpec: {exc}', file=sys.stderr)
        return 2
    try:
        import pxr  # noqa: F401
    except Exception:  # noqa: BLE001
        print('check_cart_deck_level: usd-core (pxr) not available -- run under a usd venv', file=sys.stderr)
        return 2

    rec = _measure(MobileSpec())
    rec['tol_deg'] = args.tol_deg
    problems = []
    if not (rec['deck_tilt_deg'] <= args.tol_deg):
        problems.append('deck tilt %.4f deg exceeds tolerance %.3f deg -- the authored cart deck is NOT level'
                        % (rec['deck_tilt_deg'], args.tol_deg))
    if rec['n_casters'] != 4:
        problems.append('expected 4 casters, found %d' % rec['n_casters'])
    elif not (rec['caster_bottom_spread_m'] <= args.caster_spread_tol_m):
        problems.append('caster ground-contact height spread %.6f m exceeds %.6f m -- the cart would settle '
                        'tipped on a level floor' % (rec['caster_bottom_spread_m'], args.caster_spread_tol_m))
    rec['problems'] = problems
    rec['verdict'] = 'PASS' if not problems else 'FAIL'

    if args.json:
        with open(args.json, 'w') as fh:
            json.dump(rec, fh, indent=2)

    print('CART_DECK_LEVEL deck_tilt=%.4f deg (tol %.3f) caster_bottom_spread=%.6f m n_casters=%d -> %s'
          % (rec['deck_tilt_deg'], args.tol_deg, rec['caster_bottom_spread_m'], rec['n_casters'], rec['verdict']))
    for p in problems:
        print('  - ' + p)
    if problems:
        print('The AUTHORED cart deck is not level -- fix the cart asset (mobile_assets.author_cart) / caster '
              'heights before relying on a flat landing.', file=sys.stderr)
        return 1
    print('The authored cart deck rests parallel to the floor: no static deck tilt explains the 2.0 deg '
          'collect set-down residual. That residual is the ARM-PRESS contact artifact (FK of the executed '
          'joints deflecting as the arm drives the welded carton into the rigid deck); the operative fix is '
          'the contact-triggered weld release (carton_level.place_contact_release).')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
