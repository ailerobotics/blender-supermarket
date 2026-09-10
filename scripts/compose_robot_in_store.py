#!/usr/bin/env python3
"""Compose the fleet's UR10 (shelfbot_newgripper) into the supermarket twin for
Isaac Sim — standalone, modelled on ~/workspaces/shelfbot/tools/run_tray_scene.py.

Run under env_isaaclab (Isaac Sim 5.1). Booting Isaac Sim is a GPU session; on
this 12 GB / low-RAM box, beside the captain's live Blender GUI, it can OOM-kill
that window. So this script:

  * ``--report-usd`` / ``--dry-run``  : read the store USD with the Isaac-bundled
    pxr bindings (NO SimulationApp, NO RTX, NO GPU) and print the measured facts,
    the geometry-derived spawn pose and the exact collider plan. Safe to run now.
  * a real run (no --dry-run)         : refuses via the GPU guard if a protected /
    heavy GPU consumer is present, then boots SimulationApp, imports the URDF to
    USD, references the store, authors PhysicsScene + colliders (triangle-mesh on
    static scenery, convex hull on the ~196 instance PROTOTYPES — never per
    instance), places the robot at the aisle pose, steps, saves the stage and a
    screenshot.

The real-run authoring below has NOT been executed in this build (the GPU boot is
the captain's decision, deliberately not taken). Blocks that require a booted
simulator are marked `# UNVERIFIED (needs a simulator boot)`.
"""
from __future__ import annotations

import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from twin import cli  # noqa: E402  (pure; no pxr/isaacsim)
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
# Real run: everything below needs a booted simulator (a GPU session).        #
# UNVERIFIED in this build — the boot is the captain's decision.              #
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
        from isaacsim.core.utils.extensions import enable_extension  # noqa: E402
        import omni.usd  # noqa: E402
        from pxr import Gf, Usd, UsdGeom, UsdPhysics, PhysxSchema  # noqa: E402

        # 3. URDF -> USD (step 1 of the real pipeline; deferred here behind the boot).
        robot_usd = args.robot_usd or _import_urdf_to_usd(args.robot_urdf, args.save)

        # 4. Compose: reference the store, author physics + colliders, place robot.
        omni.usd.get_context().new_stage()
        stage = omni.usd.get_context().get_stage()
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
        UsdGeom.SetStageMetersPerUnit(stage, 1.0)
        store_ref = stage.DefinePrim("/World/Store", "Xform")
        store_ref.GetReferences().AddReference(os.path.abspath(args.store))

        _author_physics(stage, UsdPhysics)
        _author_colliders(stage, Usd, UsdPhysics, PhysxSchema,
                          author_products=(args.product_collider == "convex"))

        # Re-probe the geometry (no GPU) for the spawn pose, then place + correct.
        _facts, spawn, _plan, _extras = usd_probe.probe(
            args.store, spawn_aisle_cam=args.spawn_aisle,
            spawn_xy=tuple(args.spawn_xy) if args.spawn_xy else None,
            product_collider=args.product_collider, min_clearance=args.min_clearance)
        _place_robot(stage, robot_usd, spawn, args.yaw_deg, Gf, UsdGeom)

        # 5. Step and save.
        import omni.timeline  # noqa: E402
        omni.timeline.get_timeline_interface().play()
        for _ in range(args.steps):
            sim.update()
        stage.Export(os.path.abspath(args.save))
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
    (/opt/ros/jazzy/share) and shelfbot (~/workspaces/shelfbot) on the ROS
    package path (the wrapper sources ROS for this).
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


def _author_physics(stage, UsdPhysics) -> None:
    """PhysicsScene + gravity. UNVERIFIED (needs a simulator boot)."""
    scene = UsdPhysics.Scene.Define(stage, "/World/physicsScene")
    scene.CreateGravityDirectionAttr().Set((0.0, 0.0, -1.0))   # Z-up store
    scene.CreateGravityMagnitudeAttr().Set(9.81)


def _author_colliders(stage, Usd, UsdPhysics, PhysxSchema, author_products: bool) -> None:
    """Apply colliders using the SAME policy as twin.colliders.classify:
    triangle-mesh on static scenery, convex hull on instance PROTOTYPES (authored
    once — never on the 24.9k instances). UNVERIFIED (needs a simulator boot)."""
    from twin import colliders as C

    def apply(prim, decision):
        UsdPhysics.CollisionAPI.Apply(prim)
        mesh_api = UsdPhysics.MeshCollisionAPI.Apply(prim)
        mesh_api.CreateApproximationAttr().Set(decision.approximation)
        # Static: NO RigidBodyAPI -> immovable collider (robot can't knock stock over).

    store = stage.GetPrimAtPath("/World/Store")
    root = store.GetPath()
    # Static scenery under the referenced store.
    for grp in C.STATIC_GROUPS:
        grproot = stage.GetPrimAtPath(root.AppendChild(grp))
        if not grproot:
            continue
        for p in Usd.PrimRange(grproot):
            info = C.PrimInfo(path=str(p.GetPath()), type_name=str(p.GetTypeName()),
                              top_group=grp, is_instance=p.IsInstance(),
                              in_prototype=False, name=p.GetName())
            d = C.classify(info)
            if d.kind == C.TRIANGLE_MESH:
                apply(p, d)
    # Instance prototypes (convex, once each).
    if author_products:
        for proto in stage.GetPrototypes():
            for p in Usd.PrimRange(proto):
                if p.GetTypeName() == "Mesh":
                    info = C.PrimInfo(path=str(p.GetPath()), type_name="Mesh",
                                      top_group="__prototype__", is_instance=False,
                                      in_prototype=True, name=p.GetName())
                    apply(p, C.classify(info))


def _place_robot(stage, robot_usd, spawn, yaw_deg, Gf, UsdGeom) -> None:
    """Reference the robot USD and set its root xform to the estimated translate,
    then re-measure the pedestal and correct so it lands at the aisle spot.
    UNVERIFIED (needs a simulator boot)."""
    robot = stage.DefinePrim("/World/Robot", "Xform")
    robot.GetReferences().AddReference(os.path.abspath(robot_usd))
    xf = UsdGeom.Xformable(robot)
    xf.ClearXformOpOrder()
    t = spawn.root_translate or (spawn.x, spawn.y, spawn.z)
    xf.AddTranslateOp().Set(Gf.Vec3d(float(t[0]), float(t[1]), float(t[2])))
    if yaw_deg:
        xf.AddRotateZOp().Set(float(yaw_deg))
    # Correction pass: measure where the pedestal actually landed (import root
    # convention varies) and nudge so the pedestal base sits at (spawn.x, spawn.y).
    # Left as a runtime check because it depends on the imported prim tree.


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
    return run_compose(args)


if __name__ == "__main__":
    raise SystemExit(main())
