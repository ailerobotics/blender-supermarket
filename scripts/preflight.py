#!/usr/bin/env python3
"""ONE offline command for the whole robot-in-store chain — no Isaac Sim, no GPU.

Runs the three offline pre-flights the captain would otherwise run separately, and
gives a single pass/fail so it is unambiguous whether booting the simulator is worth it:

  1. STORE     — ``validate_usd.py``            : the exported store USD is Isaac-loadable
                                                   (Z-up, 1 m/unit, default prim, every
                                                   texture/reference resolves, composer paths).
  2. ROBOT     — ``validate_robot_urdf.py``     : the robot URDF is Isaac-ready (parses, one
                                                   tree, every mesh resolves, masses/inertias/
                                                   joint limits sane) AND fits the store aisle.
  3. COMPOSER  — ``twin.compose`` author-offline : the composed stage authored for real with
                                                   pure USD — physics, colliders, robot placed
                                                   on the floor inside the aisle gap.

Exit code: ``0`` all three pass (safe to boot), ``1`` at least one failed (do NOT boot —
fix first), ``2`` bad usage, ``3`` a required dependency (numpy / pxr) is unavailable.

Run under the CPU-only usd-core venv (needs numpy for the robot side, pxr for the store +
composer sides):

    .venv-usd/bin/python scripts/preflight.py

Safe beside the captain's live Blender session: this reads the store USD and authors only a
scratch composed stage; it never boots the simulator and never writes to the repo.
"""
from __future__ import annotations

import argparse
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _HERE)

HOME = os.path.expanduser("~")
DEFAULT_STORE = os.path.join(_ROOT, "export", "supermarket.usdc")
DEFAULT_URDF = os.path.join(HOME, "workspaces/shelfbot_newgripper/description/shelfbot_gripper_isaac.urdf")
DEFAULT_SAVE = os.path.join(os.environ.get("TMPDIR", "/tmp"), "twin_preflight_store_with_robot.usda")


def _section(title):
    print("\n" + "#" * 78)
    print(f"# {title}")
    print("#" * 78)


def run_store(store_path):
    import validate_usd as V
    rep = V.validate(store_path)
    print(rep.render())
    return not rep.failed


def run_robot(urdf_path, store_facts):
    import validate_robot_urdf as R
    rep = R.validate(urdf_path, store_facts_path=store_facts)
    print(rep.render())
    return not rep.failed


def run_composer(store_path, urdf_path, save_path):
    """Author the composed stage offline (pure USD) and validate it — the same core the
    real Isaac run uses, with a faithful pedestal+arm stub standing in for the URDF import."""
    from twin import usd_probe, compose
    facts, spawn, _plan, extras = usd_probe.probe(store_path)
    stub = os.path.splitext(save_path)[0] + "_stub_robot.usda"
    compose.build_stub_robot(stub)
    result = compose.build_composition(store_path=store_path, robot_usd=stub,
                                       out_path=save_path, spawn=spawn)
    rep = compose.validate_composition(
        save_path, spawn=spawn, floor_z=facts["floor_z"], extras=extras,
        expect_triangle=facts["collider_targets"]["triangle_mesh_meshes"])
    print(f"composed stage: {result.out_path}  "
          f"({result.triangle_mesh} tri + {result.convex_hull} convex colliders, "
          f"{result.prototypes_covered}/{result.prototypes_total} prototypes, "
          f"{result.prim_count} prims)")
    print(rep.render())
    return not rep.failed


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Offline pre-flight for the whole robot-in-store chain (store + robot + "
                    "composer). No Isaac, no GPU. Exit 0 = safe to boot.")
    ap.add_argument("--store", default=DEFAULT_STORE, help="Store USD (default: %(default)s)")
    ap.add_argument("--robot-urdf", default=DEFAULT_URDF, help="Robot URDF (default: %(default)s)")
    ap.add_argument("--save", default=DEFAULT_SAVE, help="Scratch composed stage (default: %(default)s)")
    ap.add_argument("--store-facts", default=os.path.join(_ROOT, "docs", "twin_usd_facts.json"),
                    help="Store facts JSON for the robot aisle-fit check (default: %(default)s)")
    ap.add_argument("--skip-composer", action="store_true",
                    help="Skip the (heavier) composer author-offline step.")
    args = ap.parse_args(argv)

    # numpy (robot) + pxr (store, composer). Fail early with the same code the siblings use.
    try:
        import numpy  # noqa: F401
    except ImportError:
        sys.stderr.write("ERROR: numpy unavailable — see validate_robot_urdf.py --help\n")
        return 3
    try:
        import validate_usd
        validate_usd.ensure_pxr()
    except SystemExit:
        raise
    except Exception as exc:
        sys.stderr.write(f"ERROR: USD (pxr) unavailable: {exc}\n")
        return 3

    results = {}
    _section("1/3  STORE — validate_usd.py")
    results["store"] = run_store(args.store)
    _section("2/3  ROBOT — validate_robot_urdf.py")
    results["robot"] = run_robot(args.robot_urdf, args.store_facts)
    if args.skip_composer:
        print("\n(skipping composer step: --skip-composer)")
    else:
        _section("3/3  COMPOSER — twin.compose author-offline")
        try:
            results["composer"] = run_composer(args.store, args.robot_urdf, args.save)
        except Exception as exc:
            print(f"[FAIL] composer author-offline raised: {type(exc).__name__}: {exc}")
            results["composer"] = False

    _section("ROBOT-IN-STORE PRE-FLIGHT — VERDICT")
    label = {"store": "STORE    (export USD Isaac-loadable)",
             "robot": "ROBOT    (URDF Isaac-ready + fits aisle)",
             "composer": "COMPOSER (offline composition sane)"}
    for k in ("store", "robot", "composer"):
        if k in results:
            print(f"  {'PASS' if results[k] else 'FAIL'}  {label[k]}")
    all_ok = all(results.values())
    print("-" * 78)
    if all_ok:
        print("VERDICT: PASS — store checked, robot checked, composer checked. The only things "
              "left are what\n         a simulator alone can test: URDF→USD import, PhysX loading "
              "the colliders, and\n         the robot settling under gravity. Safe to boot (close "
              "Blender first).")
    else:
        failed = [label[k] for k in results if not results[k]]
        print("VERDICT: FAIL — do NOT boot the simulator yet. Fix: " + "; ".join(failed))
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
