#!/usr/bin/env python3
"""CPU proof + regression guard: the sales_backroom_portal throat is collider-clear.

Disproves the latent-vertical-collider hypothesis for the coupled tow's fixed from-rest pin at
x=-3.935 (harvest item 5 / repeatability-ledger msg 628). It:

  1. enumerates the store's static collision obstacles in the throat region (world AABBs -- from a
     live ``usd-core`` scan of the store USD with ``--store``, else the committed fixture
     ``scripts/twin/tests/fixtures/portal_throat_obstacles.json``);
  2. builds the coupled train's swept envelope through the portal from the recorded from-rest pin
     pose and asserts NO obstacle intrudes any swept segment in the drive-height band
     (``twin.restock.portal_throat.assess``);
  3. confirms the load-bearing drive wheels are off every ``Backroom_floor`` cooked diagonal and
     the one caster over a diagonal is held airborne by ``caster_lift_m``;
  4. writes a before dump JSON and (if matplotlib is present) a top-down plot.

Exit 0 = the throat is collider-clear (the pin is NOT geometry). Exit 1 = a static collider
intrudes the corridor (a future store edit narrowed the portal / dropped a jamb footing / a
duplicated wall segment) -- the CPU guard fires before a GPU run.

Usage:
    python3 scripts/check_portal_throat_clear.py                      # fixture (no pxr needed)
    python3 scripts/check_portal_throat_clear.py --out DIR            # + dump + plot
    .venv-usd/bin/python scripts/check_portal_throat_clear.py --store <supermarket.usdc>
    .venv-usd/bin/python scripts/check_portal_throat_clear.py --store <usdc> --dump-fixture
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from twin.mobile import MobileSpec                 # noqa: E402
from twin.restock import portal_throat as pt       # noqa: E402

FIXTURE = Path(__file__).resolve().parents[1] / "scripts" / "twin" / "tests" / "fixtures" / "portal_throat_obstacles.json"
STATIC_GROUPS = ("GRP_Structure", "GRP_Fixtures", "GRP_Signage")
# The committed fixture region, so a live scan and the fixture cover the same window.
REGION_X = (-8.0, 0.5)
REGION_Y = (8.0, 20.0)


def scan_store(store_path: str) -> list:
    """World AABBs of every static collision-bearing mesh in the throat region (pxr/usd-core)."""
    from pxr import Usd, UsdGeom
    st = Usd.Stage.Open(store_path)
    if not st:
        raise RuntimeError(f"could not open store: {store_path}")
    dp = st.GetDefaultPrim()
    root = dp.GetPath()
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(),
                             [UsdGeom.Tokens.default_, UsdGeom.Tokens.render])
    obs = []
    for grp in STATIC_GROUPS:
        g = st.GetPrimAtPath(root.AppendChild(grp))
        if not g:
            continue
        for p in Usd.PrimRange(g):
            if p.GetTypeName() != "Mesh" or p.IsInstance() or p.IsInstanceProxy():
                continue
            r = cache.ComputeWorldBound(p).ComputeAlignedRange()
            mn, mx = r.GetMin(), r.GetMax()
            if (mx[0] >= REGION_X[0] and mn[0] <= REGION_X[1]
                    and mx[1] >= REGION_Y[0] and mn[1] <= REGION_Y[1]):
                obs.append(dict(id=p.GetName(),
                                min=[round(float(v), 4) for v in mn],
                                max=[round(float(v), 4) for v in mx]))
    obs.sort(key=lambda o: o["id"])
    return obs


def load_fixture() -> dict:
    return json.loads(FIXTURE.read_text())


def plot(obstacles, spec, clearance, out_png: Path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle, Polygon
    except Exception as exc:  # pragma: no cover - plotting is best-effort
        return f"plot skipped ({exc})"
    from twin.mobile import swept_envelope
    fig, ax = plt.subplots(figsize=(8, 9))
    band = pt.DRIVE_BAND_Z
    for o in obstacles:
        mn, mx = o["min"], o["max"]
        drive = mx[2] >= band[0] and mn[2] <= band[1]
        ax.add_patch(Rectangle((mn[0], mn[1]), mx[0] - mn[0], mx[1] - mn[1],
                               facecolor=("0.55" if drive else "0.85"),
                               edgecolor="0.3", linewidth=0.5, zorder=1))
    samples = pt.corner_samples(spec)
    for a, b in zip(samples, samples[1:]):
        poly = swept_envelope([a, b], spec, margin=0.05)
        ax.add_patch(Polygon(poly, closed=True, facecolor="tab:green", alpha=0.22,
                             edgecolor="tab:green", linewidth=1.0, zorder=2))
    px, py = pt.PIN_BASE_XY
    ax.plot([px], [py], marker="x", color="tab:red", markersize=12, markeredgewidth=3, zorder=5,
            label="from-rest pin x=-3.935")
    base = pt.Pose2(px, py, 0.0)
    for w in pt.drive_wheel_world(base, spec):
        ax.plot([w[0]], [w[1]], marker="o", color="tab:blue", markersize=6, zorder=5)
    for d in pt.cooked_diagonals():
        ax.plot([d[0][0], d[1][0]], [d[0][1], d[1][1]], color="tab:orange", linewidth=1.0,
                linestyle="--", zorder=3)
    ax.set_xlim(-8, 1.5)
    ax.set_ylim(8, 20)
    ax.set_aspect("equal")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title("Portal throat: coupled swept envelope (green) vs static colliders\n"
                 "nearest obstacle %s @ %.2f m; corridor %s"
                 % (clearance.nearest_obstacle_id, clearance.nearest_obstacle_m or 0.0,
                    "CLEAR" if clearance.corridor_clear else "BLOCKED"))
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_png, dpi=110)
    plt.close(fig)
    return str(out_png)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--store", help="store USD (usd-core) to scan; default: committed fixture")
    ap.add_argument("--out", help="directory for the before dump JSON + plot")
    ap.add_argument("--caster-lift", type=float, default=0.03,
                    help="caster_lift_m for the rig (default 0.03, the late-run traction config)")
    ap.add_argument("--dump-fixture", action="store_true",
                    help="(re)write the committed throat obstacle fixture from --store")
    args = ap.parse_args(argv)

    spec = MobileSpec(caster_lift_m=args.caster_lift)

    if args.store:
        obstacles = scan_store(args.store)
        source = args.store
        if args.dump_fixture:
            payload = dict(
                _comment=("Static collision-obstacle AABBs (world, metres) in the "
                          "sales_backroom_portal throat region, from the generator output of the "
                          "store USD below. See scripts/twin/restock/portal_throat.py."),
                source_dataset=Path(args.store).parent.name,
                obstacle_height_band=list(pt.DRIVE_BAND_Z),
                region=dict(x=list(REGION_X), y=list(REGION_Y)),
                pin_base_xy=list(pt.PIN_BASE_XY),
                n_obstacles=len(obstacles), obstacles=obstacles)
            FIXTURE.write_text(json.dumps(payload, indent=2) + "\n")
            print(f"wrote fixture {FIXTURE} ({len(obstacles)} obstacles)")
    else:
        fx = load_fixture()
        obstacles = fx["obstacles"]
        source = f"fixture:{FIXTURE.name} (dataset {fx.get('source_dataset')})"

    clearance = pt.assess(obstacles, spec)

    print(f"store obstacles (throat region): {len(obstacles)}  source: {source}")
    print(f"caster_lift_m: {args.caster_lift}")
    print()
    for k, v in dataclasses.asdict(clearance).items():
        print(f"  {k:30s} {v}")
    print()
    print(f"VERDICT: {clearance.verdict}")

    if args.out:
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        dump = dict(source=source, caster_lift_m=args.caster_lift,
                    n_obstacles=len(obstacles),
                    clearance=dataclasses.asdict(clearance),
                    drive_band_z=list(pt.DRIVE_BAND_Z),
                    pin_base_xy=list(pt.PIN_BASE_XY),
                    obstacles=obstacles)
        (out / "throat-collider-dump.json").write_text(json.dumps(dump, indent=2) + "\n")
        png = plot(obstacles, spec, clearance, out / "throat-clearance.png")
        print(f"\nwrote {out/'throat-collider-dump.json'}\nplot: {png}")

    ok = clearance.collider_free
    print(f"\n{'PASS' if ok else 'FAIL'}: throat {'collider-clear' if ok else 'has an intruding collider'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
