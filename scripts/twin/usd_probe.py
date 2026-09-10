"""Read the store USD WITHOUT booting Isaac Sim.

Uses the USD (pxr) bindings bundled inside Isaac Sim's extscache. This is a pure
read — no SimulationApp, no Kit, no RTX, no GPU session — so it is safe to run
beside the captain's Blender GUI, and it is how the measured facts in
docs/twin_usd_facts.json were produced.

``ensure_pxr_env`` injects the extscache lib paths and, because Linux reads
LD_LIBRARY_PATH only at exec time, re-execs this process once with them set if
``pxr`` cannot import yet. After that, plain ``from pxr import ...`` works.
"""
from __future__ import annotations

import glob
import os
import sys
from typing import Optional

# Import the sibling policy modules whether run as a package or a loose script.
try:
    from . import geometry, colliders
except ImportError:  # pragma: no cover - executed only as a top-level script
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import geometry  # type: ignore
    import colliders  # type: ignore

_REEXEC_FLAG = "_TWIN_PXR_REEXEC"


def ensure_pxr_env() -> None:
    """Make ``from pxr import Usd`` work without a simulator boot; re-exec once if
    the shared libs need LD_LIBRARY_PATH set."""
    try:
        from pxr import Usd  # noqa: F401
        return
    except ImportError:
        pass
    try:
        import isaacsim  # noqa: F401  (import only locates files; does not boot anything)
        base = os.path.join(os.path.dirname(isaacsim.__file__), "extscache")
    except ImportError:
        base = os.path.join(sys.prefix, "lib", f"python{sys.version_info.major}."
                            f"{sys.version_info.minor}", "site-packages", "isaacsim", "extscache")
    libdirs = sorted(glob.glob(os.path.join(base, "omni.usd.libs-*")))
    if not libdirs:
        raise ImportError(
            "Could not locate the Isaac-Sim-bundled USD libs (omni.usd.libs-* under "
            f"{base}). Run under env_isaaclab, or source the compose_robot_in_store.sh wrapper.")
    libdir = libdirs[-1]
    conda_lib = os.path.join(sys.prefix, "lib")
    if os.environ.get(_REEXEC_FLAG):
        # Already re-exec'd and still failing — surface the real error.
        from pxr import Usd  # noqa: F401
        return
    os.environ[_REEXEC_FLAG] = "1"
    os.environ["PYTHONPATH"] = os.pathsep.join(
        [libdir] + ([os.environ["PYTHONPATH"]] if os.environ.get("PYTHONPATH") else []))
    os.environ["LD_LIBRARY_PATH"] = os.pathsep.join(
        [os.path.join(libdir, "bin"), conda_lib]
        + ([os.environ["LD_LIBRARY_PATH"]] if os.environ.get("LD_LIBRARY_PATH") else []))
    os.execv(sys.executable, [sys.executable] + sys.argv)


# Gondola fixture groups form the parallel aisles.
GONDOLA_GROUPS = ["GRP_Fixtures_G1", "GRP_Fixtures_G2", "GRP_Fixtures_G3", "GRP_Fixtures_G4",
                  "GRP_Fixtures_G5", "GRP_Fixtures_G6", "GRP_Fixtures_G7"]
_SKIP_STRUCTURE = {"Floor", "Ceiling", "Ground_exterior"}


def probe(store_usd: str, spawn_aisle_cam: str = "Cam_Aisle",
          spawn_xy: Optional[tuple] = None, product_collider: str = "convex",
          min_clearance: float = 0.6):
    """Open the store USD and return (facts, spawn_candidate, collider_plan, extras)."""
    ensure_pxr_env()
    from pxr import Usd, UsdGeom

    stage = Usd.Stage.Open(store_usd)
    if not stage:
        raise RuntimeError(f"could not open store USD: {store_usd}")
    dp = stage.GetDefaultPrim()
    root = dp.GetPath()
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(),
                             [UsdGeom.Tokens.default_, UsdGeom.Tokens.render], useExtentsHint=True)

    def rng(prim):
        r = cache.ComputeWorldBound(prim).ComputeAlignedRange()
        return r.GetMin(), r.GetMax()

    def foot(prim):
        mn, mx = rng(prim)
        return (mn[0], mn[1], mx[0], mx[1])

    # --- floor plane ---
    floor = stage.GetPrimAtPath(root.AppendChild("GRP_Structure").AppendChild("Floor"))
    fmn, fmx = rng(floor)
    floor_z = round((fmn[2] + fmx[2]) / 2.0, 4)
    extent = (float(fmn[0]), float(fmx[0]), float(fmn[1]), float(fmx[1]))

    # --- collider targets + collider plan via the real policy ---
    prims = []
    static_counts = {"GRP_Structure": 0, "GRP_Fixtures": 0, "GRP_Signage": 0}
    for grp in ["GRP_Structure", "GRP_Fixtures", "GRP_Signage", "GRP_Products", "GRP_PriceTags"]:
        grproot = stage.GetPrimAtPath(root.AppendChild(grp))
        if not grproot:
            continue
        for p in Usd.PrimRange(grproot):
            is_inst = p.IsInstance()
            is_mesh = p.GetTypeName() == "Mesh"
            if not (is_inst or is_mesh):
                continue
            info = colliders.PrimInfo(path=str(p.GetPath()), type_name=str(p.GetTypeName()),
                                      top_group=grp, is_instance=is_inst,
                                      in_prototype=False, name=p.GetName())
            prims.append(info)
            if is_mesh and grp in static_counts:
                static_counts[grp] += 1
    # prototype meshes (instanced products/tags) — authored once each
    protos = stage.GetPrototypes()
    for proto in protos:
        for p in Usd.PrimRange(proto):
            if p.GetTypeName() == "Mesh":
                prims.append(colliders.PrimInfo(
                    path=str(p.GetPath()), type_name="Mesh", top_group="__prototype__",
                    is_instance=False, in_prototype=True, name=p.GetName()))
    n_instances = sum(1 for p in prims if p.is_instance)

    plan = colliders.plan_colliders(prims)
    if product_collider == "none":
        # Prototype convex hulls are opt-out; fixtures/ground keep triangle mesh.
        proto_convex = sum(1 for p in prims if p.in_prototype and p.type_name == "Mesh")
        plan.convex_hull -= proto_convex
        plan.skipped_other += proto_convex

    facts = {
        "store_usd": store_usd,
        "up_axis": str(UsdGeom.GetStageUpAxis(stage)),
        "meters_per_unit": float(UsdGeom.GetStageMetersPerUnit(stage)),
        "default_prim": str(root),
        "floor_z": floor_z,
        "floor_extent": {"x": [round(extent[0], 3), round(extent[1], 3)],
                         "y": [round(extent[2], 3), round(extent[3], 3)]},
        "collider_targets": {
            "structure": static_counts["GRP_Structure"],
            "fixtures": static_counts["GRP_Fixtures"],
            "signage": static_counts["GRP_Signage"],
            "triangle_mesh_meshes": plan.triangle_mesh,
            "prototypes": len(protos),
            "instances": n_instances,
        },
    }

    # --- clearance grid from real footprints (per fixture/signage MESH; gondola
    #     product runs by group; walls bound the domain, not stamped) ---
    footprints = []
    for grp in ["GRP_Fixtures", "GRP_Signage"]:
        for p in Usd.PrimRange(stage.GetPrimAtPath(root.AppendChild(grp))):
            if p.GetTypeName() == "Mesh":
                footprints.append(foot(p))
    for g in GONDOLA_GROUPS:
        gp = stage.GetPrimAtPath(root.AppendChild("GRP_Products").AppendChild(
            g.replace("GRP_Fixtures_", "GRP_Products_")))
        if gp:
            footprints.append(foot(gp))
    grid = geometry.build_clearance_grid(footprints, extent, res=0.05, wall_margin=0.30)

    # gondola X-ranges + Y-extent for the aisle geometry
    gondola_ranges = []
    gy_lo, gy_hi = None, None
    for g in GONDOLA_GROUPS:
        gp = stage.GetPrimAtPath(root.AppendChild("GRP_Fixtures").AppendChild(g))
        if not gp:
            continue
        mnx, mny, mxx, mxy = foot(gp)
        gondola_ranges.append((round(mnx, 3), round(mxx, 3)))
        gy_lo = mny if gy_lo is None else min(gy_lo, mny)
        gy_hi = mxy if gy_hi is None else max(gy_hi, mxy)
    aisles = geometry.aisle_centerlines(gondola_ranges)

    # pedestal XY
    if spawn_xy is not None:
        ped_x, ped_y = float(spawn_xy[0]), float(spawn_xy[1])
    else:
        cam = stage.GetPrimAtPath(root.AppendChild("GRP_Cameras").AppendChild(spawn_aisle_cam))
        if not cam:
            raise RuntimeError(f"spawn-aisle camera not found: {spawn_aisle_cam}")
        m = UsdGeom.Xformable(cam).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        cx = m.ExtractTranslation()[0]
        ped_x = round(float(cx), 3)
        ped_y = round((gy_lo + gy_hi) / 2.0, 3)  # mid gondola length, not the camera's Y

    spawn = geometry.resolve_spawn(
        grid, floor_z, (ped_x, ped_y), (float(gy_lo) + 0.5, float(gy_hi) - 0.5),
        min_clearance=min_clearance, label=f"{spawn_aisle_cam} aisle")

    extras = {
        "gondola_x_ranges": gondola_ranges,
        "gondola_y_extent": [round(float(gy_lo), 3), round(float(gy_hi), 3)],
        "aisle_centerlines": [{"center_x": c, "gap_width_m": w} for c, w in aisles],
        "global_max_clearance": vars(geometry.global_max_clearance(grid)),
    }
    return facts, spawn, plan, extras
