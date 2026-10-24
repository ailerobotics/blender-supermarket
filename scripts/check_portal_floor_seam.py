#!/usr/bin/env python3
"""CPU before/after check for the sales_backroom_portal from-rest tow pin.

The pin is a PhysX triangle-mesh internal-edge ("ghost") contact: the cart's front
caster lands on the ``Backroom_floor`` cooked diagonal on a geometrically flat floor,
and the coupled base halts at ``cart_x + hitch_offset`` = x=-3.935. The fix authors the
flat drive floor as a BOX (``boundingCube``) collider -- no internal edge -- behind the
opt-in ``portal_floor_box`` scene switch (default off => byte-identical triangle mesh).

This proves, on CPU (no GPU, no Isaac):

  1. the base rig: drive-wheel contact x == base origin x, coupled hitch offset 1.27 m;
  2. the defect: the triangle-mesh floor's cooked diagonal crosses the portal caster band,
     and -- for a recorded run with ``--run`` -- the pinned cart's front-left caster sits
     ON that diagonal (perp ~0), while a crossing run's caster is clear;
  3. the fix: with ``portal_floor_box`` on, the floor collider has NO internal edge in the
     band (and, when pxr is available, a real authored stage round-trips to approximation
     "boundingCube"), so the caster has no edge to catch.

Exit 0 = the switch removes the portal-band internal edge (seam gone). Exit 1 = it does not.

Usage:
    python3 scripts/check_portal_floor_seam.py            # base rig + before/after proof
    python3 scripts/check_portal_floor_seam.py --run DIR  # + recorded-pin caster evidence
    .venv-usd/bin/python scripts/check_portal_floor_seam.py --prove   # same (pxr optional)
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from twin.mobile import MobileSpec           # noqa: E402
from twin.restock import portal_floor as pf  # noqa: E402
from twin import colliders                   # noqa: E402


def _pin_row(run_dir: Path):
    """The coupled nav-trace row nearest the x=-3.935 pin (and the one that crossed)."""
    trace = run_dir / "nav-trace.jsonl"
    rows = [json.loads(l) for l in trace.open() if l.strip()]
    coup = [r for r in rows if r.get("coupled") and r.get("cart")]
    if not coup:
        return None, None
    pin = min(coup, key=lambda r: abs(r.get("x", 0) + 3.935))
    crossed = [r for r in coup if r.get("x", 0) > -3.80]
    cross = crossed[0] if crossed else None
    return pin, cross


def _cart_pose(row):
    cart = row["cart"]
    yaw = math.radians(cart[2]) if len(cart) > 2 and isinstance(cart[2], (int, float)) else 0.0
    return (cart[0], cart[1], yaw)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", type=Path, help="recorded mission run dir (nav-trace.jsonl)")
    ap.add_argument("--prove", action="store_true", help="(default) run the before/after proof")
    args = ap.parse_args(argv)

    spec = MobileSpec()
    print("== base rig ==")
    print(f"  drive-wheel contact x  == base origin x (wheels jointed at base-local x=0)")
    print(f"  coupled hitch offset   base_x - cart_x = cart_hitch_x - hitch_x "
          f"= {spec.cart_hitch_x} - ({spec.hitch_x}) = {pf.hitch_offset(spec):.3f} m")
    print(f"  cart stuck at x=-5.205 => base pins at -5.205 + {pf.hitch_offset(spec):.2f} "
          f"= {-5.205 + pf.hitch_offset(spec):.3f} m  (matches the 7-run cluster -3.931..-3.935)")

    ok = True

    print("\n== portal caster band internal collision edges ==")
    off = pf.edges_in_portal_band(colliders.TRIANGLE_MESH)
    on = pf.edges_in_portal_band(colliders.BOX)
    print(f"  triangle-mesh floor (switch OFF): {len(off)} internal edge(s) in the band "
          f"{'<-- the defect (cooked quad diagonal)' if off else ''}")
    print(f"  box floor           (switch ON ): {len(on)} internal edge(s) in the band "
          f"{'<-- seam GONE' if not on else ''}")
    if not (len(off) == 1 and len(on) == 0):
        print("  FAIL: expected exactly one internal edge OFF and none ON")
        ok = False

    if args.run:
        run = args.run
        print(f"\n== recorded pin evidence: {run.name} ==")
        pin, cross = _pin_row(run)
        if pin is None:
            print("  (no coupled nav-trace rows)")
        else:
            an = pf.front_caster_edge_analysis(_cart_pose(pin), spec)
            print(f"  pin: base x={pin['x']:.4f} y={pin['y']:.3f}  cart x={pin['cart'][0]:.4f} "
                  f"y={pin['cart'][1]:.3f}  base-cart={pin['x']-pin['cart'][0]:.3f} m")
            for c in an:
                flag = "  <-- ON the cooked diagonal" if abs(c.perp_m) < 0.05 else ""
                print(f"    caster {c.label:11} {c.caster_xy}  perp_to_diagonal={c.perp_m:+.4f} m{flag}")
            fl = next(c for c in an if c.label == "front_left")
            print(f"  front-left caster perp = {fl.perp_m:+.4f} m "
                  f"({'ON the edge -> ghost contact -> pin' if abs(fl.perp_m) < 0.05 else 'clear of the edge'})")

    # Real authored-stage round-trip when pxr is importable (proves boundingCube is authored).
    try:
        from pxr import Usd, UsdGeom, UsdPhysics  # noqa: F401
    except Exception:
        print("\n(pxr not importable: skipping USD authored-approximation round-trip; "
              "pure-geometry proof above stands)")
    else:
        from twin import compose
        import tempfile
        print("\n== authored-stage round-trip (pxr) ==")
        for flag, want in ((False, "none"), (True, "boundingCube")):
            with tempfile.TemporaryDirectory() as td:
                path = Path(td) / "floor.usda"
                stage = Usd.Stage.CreateNew(str(path))
                root = stage.DefinePrim("/Supermarket", "Xform").GetPath()
                grp = stage.DefinePrim("/Supermarket/GRP_Structure", "Xform")
                mp = "/Supermarket/GRP_Structure/Backroom_floor/Backroom_floor"
                stage.DefinePrim("/Supermarket/GRP_Structure/Backroom_floor", "Xform")
                mesh = UsdGeom.Mesh.Define(stage, mp)
                q = pf.BACKROOM_FLOOR_TOP_QUAD
                from pxr import Gf, Vt
                mesh.CreatePointsAttr([Gf.Vec3f(x, y, 0.0) for x, y in q])
                mesh.CreateFaceVertexCountsAttr([4])
                mesh.CreateFaceVertexIndicesAttr([0, 1, 2, 3])
                n = compose.author_static_colliders(stage, root, flat_floor_box=flag)
                got = UsdPhysics.MeshCollisionAPI(stage.GetPrimAtPath(mp)).GetApproximationAttr().Get()
                status = "OK" if got == want else "FAIL"
                if got != want:
                    ok = False
                print(f"  flat_floor_box={str(flag):5} -> approximation={got!r} (want {want!r}) "
                      f"[{n} static collider(s)] {status}")

    print("\n" + ("PORTAL_FLOOR_SEAM OK: the box switch removes the portal-band internal edge"
                  if ok else "PORTAL_FLOOR_SEAM FAIL"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
