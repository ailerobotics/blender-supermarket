#!/usr/bin/env python3
"""Compose the fleet's UR10 (shelfbot_newgripper) into the supermarket twin for
Isaac Sim — standalone, modelled on ~/workspaces/shelfbot/tools/run_tray_scene.py.

Run under env_isaaclab (Isaac Sim 5.1). Booting Isaac Sim is a GPU session; on
this 12 GB / low-RAM box, beside the captain's live Blender GUI, it can OOM-kill
that window. So the modes split by whether they need the simulator:

  * ``--report-usd`` / ``--dry-run``  : read the store USD with pure USD bindings
    (NO SimulationApp, NO RTX, NO GPU) and print the measured facts, the
    geometry-derived spawn pose and the exact collider plan. Safe to run now.
  * ``--author-offline``              : ACTUALLY author the composed stage with pure
    USD (still NO GPU boot) — sublayer the store, author PhysicsScene + colliders,
    place the robot (a faithful stub unless ``--robot-usd`` is given), save it and
    VALIDATE it. This is the offline proof that everything except the simulator
    itself works. Exit 0 = sane, 1 = a check failed.
  * a real run (no mode flag)         : refuses via the GPU guard if a protected /
    heavy GPU consumer is present, then boots SimulationApp, imports the URDF to
    USD, and calls the SAME pure-USD composition core (``twin.compose``) that
    ``--author-offline`` exercises, then opens the composed stage, steps it and
    writes a screenshot.

The composition core (``twin/compose.py``) is fully exercised offline and unit
tested. The ONLY code that stays unverified until the captain's GPU boot is the
simulator itself — URDF→USD import, physics stepping and the screenshot. Those
blocks are marked ``# UNVERIFIED (needs a simulator boot)``.
"""
from __future__ import annotations

import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from twin import cli  # noqa: E402  (pure; no pxr/isaacsim)
from twin import compose  # noqa: E402  (pure USD; pxr imported lazily inside)
from twin import gpu_guard  # noqa: E402
from twin import usd_probe  # noqa: E402  (pxr imported lazily inside probe())


# --------------------------------------------------------------------------- #
# No-GPU paths: probe + dry-run                                               #
# --------------------------------------------------------------------------- #
def run_report_usd(args) -> int:
    facts, spawn, plan, extras = usd_probe.probe(
        args.store, spawn_aisle_cam=args.spawn_aisle,
        spawn_xy=tuple(args.spawn_xy) if args.spawn_xy else None,
        product_collider=args.product_collider, min_clearance=args.min_clearance)
    out = {
        "facts": facts,
        "spawn": vars(spawn),
        "collider_plan": vars(plan),
        "geometry": extras,
    }
    print(json.dumps(out, indent=2))
    return 0


def run_dry_run(args) -> int:
    facts, spawn, plan, extras = usd_probe.probe(
        args.store, spawn_aisle_cam=args.spawn_aisle,
        spawn_xy=tuple(args.spawn_xy) if args.spawn_xy else None,
        product_collider=args.product_collider, min_clearance=args.min_clearance)
    # Query the GPU read-only to show what a real run would decide right now.
    gpu = gpu_guard.query_and_assess(force=args.force,
                                     required_free_mb=args.vram_budget_mb,
                                     heavy_threshold_mb=args.heavy_threshold_mb)
    pland = {
        "store": args.store, "robot_urdf": args.robot_urdf, "robot_usd": args.robot_usd,
        "usd_facts": facts, "spawn": vars(spawn), "collider_plan": vars(plan),
        "product_collider": args.product_collider, "min_clearance": args.min_clearance,
        "yaw_deg": args.yaw_deg, "save": args.save, "screenshot": cli.resolve_screenshot(args),
        "gpu": {"ok": gpu.ok, "message": gpu.message},
    }
    print(cli.render_plan(pland))
    return 0


# --------------------------------------------------------------------------- #
# No-GPU path: actually author + validate the composed stage (pure USD).      #
# --------------------------------------------------------------------------- #
def run_author_offline(args) -> int:
    """Drive the pure-USD composition core for real and validate the result. No
    SimulationApp, no RTX, no GPU — safe beside the live Blender session."""
    facts, spawn, plan, extras = usd_probe.probe(
        args.store, spawn_aisle_cam=args.spawn_aisle,
        spawn_xy=tuple(args.spawn_xy) if args.spawn_xy else None,
        product_collider=args.product_collider, min_clearance=args.min_clearance)

    # Robot: a caller-supplied USD, else a faithful pedestal+arm stub so the
    # placement/transform maths is exercised for real (the true URDF→USD import
    # needs Isaac and is only done in a real run).
    if args.robot_usd:
        robot_usd = args.robot_usd
        robot_kind = "supplied --robot-usd"
    else:
        robot_usd = os.path.splitext(args.save)[0] + "_stub_robot.usda"
        compose.build_stub_robot(robot_usd)
        robot_kind = "faithful stub (no --robot-usd given)"

    result = compose.build_composition(
        store_path=args.store, robot_usd=robot_usd, out_path=args.save,
        spawn=spawn, yaw_deg=args.yaw_deg, product_collider=args.product_collider)

    rep = compose.validate_composition(
        args.save, spawn=spawn, floor_z=facts["floor_z"], extras=extras,
        product_collider=args.product_collider,
        expect_triangle=facts["collider_targets"]["triangle_mesh_meshes"])

    print("=" * 78)
    print("compose_robot_in_store — AUTHOR OFFLINE (pure USD, nothing booted)")
    print("=" * 78)
    print(f"store (sublayered, read-only): {result.store_sublayer}")
    print(f"robot referenced             : {robot_usd}  [{robot_kind}]")
    print(f"composed stage written       : {result.out_path}")
    print(f"  default prim   : {result.default_prim}")
    print(f"  robot prim     : {result.robot_prim}  translate {tuple(round(v,3) for v in result.robot_translate)}")
    print(f"  colliders      : {result.triangle_mesh} triangle-mesh (static) + "
          f"{result.convex_hull} convex-hull (products/tags)")
    print(f"  prototypes     : {result.prototypes_covered}/{result.prototypes_total} carry a collider")
    print(f"  prim count     : {result.prim_count} (instancing preserved)")
    print("-" * 78)
    print(rep.render())
    print("-" * 78)
    verdict = "FAIL" if rep.failed else "PASS"
    n_fail = sum(1 for s, _, _ in rep.checks if s == compose.ComposeReport.FAIL)
    print(f"VERDICT: {verdict}" + (f"  ({n_fail} check(s) failed)" if rep.failed else
                                   "  — offline composition is sane"))
    if args.json:
        print("-" * 78)
        print(json.dumps({"result": vars(result), "facts": rep.facts}, indent=2, default=str))
    return 1 if rep.failed else 0


# --------------------------------------------------------------------------- #
# Real run: boot the simulator, then reuse the pure-USD composition core.      #
# The boot, URDF→USD import, stepping and screenshot are UNVERIFIED in this    #
# build — the boot is the captain's decision.                                 #
# --------------------------------------------------------------------------- #
def run_compose(args) -> int:
    # 1. GPU guard FIRST — before importing/booting anything heavy.
    gpu = gpu_guard.query_and_assess(force=args.force,
                                     required_free_mb=args.vram_budget_mb,
                                     heavy_threshold_mb=args.heavy_threshold_mb)
    print(gpu.message, file=sys.stderr)
    if not gpu.ok:
        return 2

    # 2. Boot Isaac Sim. SimulationApp MUST be created before any omni.* import.
    from isaacsim import SimulationApp  # UNVERIFIED (needs a simulator boot)
    sim = SimulationApp({"headless": args.headless})
    try:
        import omni.usd  # noqa: E402

        # 3. URDF -> USD (Isaac-only; needs the booted Kit app). Deferred here
        #    behind the boot — there is no pure-Python path (see docstring).
        robot_usd = args.robot_usd or _import_urdf_to_usd(args.robot_urdf, args.save)

        # 4. Probe geometry (no GPU) for the spawn pose, then author the whole
        #    composed stage with the SAME pure-USD core the offline path uses.
        _facts, spawn, _plan, _extras = usd_probe.probe(
            args.store, spawn_aisle_cam=args.spawn_aisle,
            spawn_xy=tuple(args.spawn_xy) if args.spawn_xy else None,
            product_collider=args.product_collider, min_clearance=args.min_clearance)
        result = compose.build_composition(
            store_path=args.store, robot_usd=robot_usd, out_path=args.save,
            spawn=spawn, yaw_deg=args.yaw_deg, product_collider=args.product_collider)
        print(f"[compose] authored {result.out_path} "
              f"({result.triangle_mesh} tri + {result.convex_hull} convex colliders, "
              f"{result.prototypes_covered}/{result.prototypes_total} prototypes)", file=sys.stderr)

        # 5. Open the composed stage in the simulator, step physics, screenshot.
        omni.usd.get_context().open_stage(os.path.abspath(args.save))  # UNVERIFIED
        import omni.timeline  # noqa: E402
        omni.timeline.get_timeline_interface().play()
        for _ in range(args.steps):
            sim.update()
        _screenshot(sim, cli.resolve_screenshot(args))
        print(f"[compose] wrote {args.save}", file=sys.stderr)
    finally:
        sim.close()
    return 0


def _import_urdf_to_usd(urdf_path: str, near_save: str) -> str:
    """URDF -> USD via the isaacsim.asset.importer.urdf Kit extension.

    UNVERIFIED (needs a simulator boot). The extension depends on omni.kit.*, so
    it only runs inside the booted Kit app — it CANNOT be done without a GPU
    session, which is why the whole conversion is deferred to here rather than
    done ahead of time. package:// meshes must resolve: ur_description
    (/opt/ros/jazzy/share) and shelfbot (~/workspaces/shelfbot_newgripper — the repo
    that declares package 'shelfbot' AND carries the gripper meshes, NOT the older
    ~/workspaces/shelfbot) on the ROS package path (the wrapper builds a shim for this).
    Prove it offline first with scripts/validate_robot_urdf.py.
    """
    import omni.kit.commands  # UNVERIFIED
    from isaacsim.core.utils.extensions import enable_extension
    enable_extension("isaacsim.asset.importer.urdf")

    out_usd = os.path.splitext(near_save)[0] + "_robot.usd"
    _status, cfg = omni.kit.commands.execute("URDFCreateImportConfig")
    cfg.set_merge_fixed_joints(False)
    cfg.set_fix_base(True)                 # pedestal-mounted arm: base pinned to world
    cfg.set_make_default_prim(True)
    cfg.set_import_inertia_tensor(True)
    cfg.set_distance_scale(1.0)            # URDF is in metres, matches the store
    cfg.set_density(0.0)                   # keep URDF-declared masses
    _status, prim_path = omni.kit.commands.execute(
        "URDFParseAndImportFile", urdf_path=urdf_path, import_config=cfg, dest_path=out_usd)
    print(f"[compose] imported robot -> {out_usd} (prim {prim_path})", file=sys.stderr)
    return out_usd


def _screenshot(sim, path: str) -> None:
    """Capture a viewport screenshot. UNVERIFIED (needs a simulator boot)."""
    try:
        import omni.kit.viewport.utility as vp
        vp.capture_viewport_to_file(vp.get_active_viewport(), path)
        for _ in range(20):
            sim.update()
        print(f"[compose] screenshot -> {path}", file=sys.stderr)
    except Exception as exc:  # visual-only; never fail the run on it
        print(f"[compose] screenshot skipped: {exc!r}", file=sys.stderr)


def main(argv=None) -> int:
    args = cli.build_parser().parse_args(argv)
    if args.report_usd:
        return run_report_usd(args)
    if args.dry_run:
        return run_dry_run(args)
    if args.author_offline:
        return run_author_offline(args)
    return run_compose(args)


if __name__ == "__main__":
    raise SystemExit(main())
