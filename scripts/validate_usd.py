#!/usr/bin/env python3
"""Validate the exported supermarket USD **offline** — no Isaac Sim, no GPU.

This is the pre-flight for standing the shelf robot in the store (see
``scripts/compose_robot_in_store.py`` and ``docs/isaac_robot_in_store.md``). It
opens the stage with the plain USD (``pxr``) bindings and proves the things that
otherwise only fail once Isaac tries to load the stage on the GPU:

  * the stage opens; up-axis is Z; metres-per-unit is 1.0; a default prim exists
    (the composer references the store by its default prim — no default prim means
    ``AddReference`` has nothing to target);
  * **every asset path resolves** — every texture, reference, payload and sublayer
    points at a file that exists, with no absolute build-machine paths and nothing
    escaping ``export/`` (the classic "worked on my machine" defect);
  * the prim paths the composer needs are present (floor, the group hierarchy, the
    gondola runs, the spawn camera);
  * a readable characterisation (prim counts, type histogram, world extents, store
    footprint) and a physics/collider census (the store is render-only by design;
    the composer authors colliders at compose time).

Exit code: ``0`` all hard checks pass, ``1`` a hard check failed, ``2`` bad usage,
``3`` USD (pxr) unavailable.

Run it under a CPU-only usd-core venv (no Isaac needed):

    python3 -m venv .venv-usd && .venv-usd/bin/pip install usd-core
    .venv-usd/bin/python scripts/validate_usd.py

or under ``env_isaaclab`` (it will fall back to the Isaac-bundled USD).

``--repair-textures`` additionally copies any authored-but-missing texture from the
versioned ``assets/textures/`` into ``export/textures/`` when a file of the exact
authored name exists there (deterministic, offline) — it makes the *current*
artifact fully resolvable without a Blender re-export. The durable fix for the
*next* export lives in ``scripts/export_usd.py``.
"""
from __future__ import annotations

import argparse
import collections
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)  # repo root (scripts/ is directly under it)
DEFAULT_STORE = os.path.join(_ROOT, "export", "supermarket.usdc")
ASSETS_TEXTURES = os.path.join(_ROOT, "assets", "textures")

# Expectations (Isaac + the composer). See scripts/twin/usd_probe.py.
EXPECT_UP_AXIS = "Z"
EXPECT_METERS_PER_UNIT = 1.0
# Prim paths (relative to the default prim) the composer walks.
COMPOSER_GROUPS = ["GRP_Structure", "GRP_Fixtures", "GRP_Signage",
                   "GRP_Products", "GRP_PriceTags", "GRP_Cameras"]
COMPOSER_FLOOR = "GRP_Structure/Floor"
COMPOSER_SPAWN_CAM = "GRP_Cameras/Cam_Aisle"
GONDOLA_FIXTURES = [f"GRP_Fixtures/GRP_Fixtures_G{i}" for i in range(1, 8)]
GONDOLA_PRODUCTS = [f"GRP_Products/GRP_Products_G{i}" for i in range(1, 8)]


# --------------------------------------------------------------------------- #
# pxr bootstrap                                                               #
# --------------------------------------------------------------------------- #
def ensure_pxr():
    """Import pxr, or explain how to get it. Prefer a plain (usd-core) install;
    fall back to the Isaac-bundled USD via the twin probe's env shim."""
    try:
        import pxr  # noqa: F401
        return
    except ImportError:
        pass
    try:
        sys.path.insert(0, _HERE)
        from twin import usd_probe  # type: ignore
        usd_probe.ensure_pxr_env()  # may re-exec this process once
        import pxr  # noqa: F401
        return
    except Exception:
        sys.stderr.write(
            "ERROR: USD (pxr) is not importable.\n"
            "  Offline (recommended): python3 -m venv .venv-usd && "
            ".venv-usd/bin/pip install usd-core\n"
            "                         .venv-usd/bin/python scripts/validate_usd.py\n"
            "  or run under env_isaaclab (Isaac-bundled USD).\n")
        raise SystemExit(3)


# --------------------------------------------------------------------------- #
# Result plumbing                                                            #
# --------------------------------------------------------------------------- #
class Report:
    PASS, FAIL, INFO, WARN = "PASS", "FAIL", "INFO", "WARN"

    def __init__(self):
        self.checks = []   # (status, name, detail)
        self.facts = {}

    def add(self, status, name, detail=""):
        self.checks.append((status, name, detail))

    def ok(self, name, detail=""):
        self.add(self.PASS, name, detail)

    def fail(self, name, detail=""):
        self.add(self.FAIL, name, detail)

    def info(self, name, detail=""):
        self.add(self.INFO, name, detail)

    def warn(self, name, detail=""):
        self.add(self.WARN, name, detail)

    @property
    def failed(self):
        return any(s == self.FAIL for s, _, _ in self.checks)

    def render(self):
        icon = {self.PASS: "PASS", self.FAIL: "FAIL", self.INFO: "····", self.WARN: "warn"}
        lines = []
        for status, name, detail in self.checks:
            lines.append(f"[{icon[status]}] {name}" + (f": {detail}" if detail else ""))
        return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Asset-path resolution                                                      #
# --------------------------------------------------------------------------- #
def collect_asset_attrs(stage):
    """Every ``Asset``-valued attribute in the stage: (attr_path, authored, resolved)."""
    from pxr import Sdf
    out = []
    for prim in stage.Traverse():
        for attr in prim.GetAttributes():
            if attr.GetTypeName() == Sdf.ValueTypeNames.Asset:
                val = attr.Get()
                if val is None or not val.path:
                    continue
                out.append((str(attr.GetPath()), val.path, val.resolvedPath))
    return out


def check_assets(stage, store_path, rep):
    """Prove every texture / reference / payload / sublayer resolves to a file.

    Combines a layer-level dependency walk (``UsdUtils.ComputeAllDependencies`` —
    the same traversal ``usdchecker`` uses for the asset graph) with a per-attribute
    sweep of every ``Asset``-valued input, which is where texture paths live.
    """
    from pxr import UsdUtils
    store_dir = os.path.realpath(os.path.dirname(os.path.abspath(store_path)))

    # 1) Layer graph: references / payloads / sublayers / clips + their assets.
    layers, assets, unresolved = UsdUtils.ComputeAllDependencies(store_path)
    ext_layers = [l.identifier for l in layers
                  if os.path.abspath(l.identifier) != os.path.abspath(store_path)]
    if ext_layers:
        rep.info("external layers (references/payloads/sublayers)",
                 f"{len(ext_layers)}: " + ", ".join(ext_layers))
    else:
        rep.ok("composition", "single self-contained layer (no external refs/payloads/sublayers)")

    # 2) Per-attribute asset sweep — classify each authored asset path.
    attrs = collect_asset_attrs(stage)
    absolute, missing, outside = [], [], []
    for apath, authored, resolved in attrs:
        if os.path.isabs(authored):
            absolute.append((apath, authored))
        rp = resolved
        if not rp:
            cand = (authored if os.path.isabs(authored)
                    else os.path.normpath(os.path.join(store_dir, authored)))
            rp = cand if os.path.exists(cand) else ""
        if not rp or not os.path.exists(rp):
            missing.append((apath, authored, resolved))
        else:
            real = os.path.realpath(rp)
            if not real.startswith(store_dir + os.sep) and real != store_dir:
                outside.append((apath, authored, real))

    rep.facts["asset_attrs"] = len(attrs)
    rep.facts["assets_referenced"] = len(assets)
    rep.info("asset-valued attributes", f"{len(attrs)} (textures + any file inputs)")

    if absolute:
        rep.fail("no absolute build-machine paths",
                 f"{len(absolute)} absolute authored path(s), e.g. {absolute[0][1]}")
    else:
        rep.ok("no absolute build-machine paths", "all asset paths are relative")

    if outside:
        rep.fail("all assets inside export/",
                 f"{len(outside)} asset(s) resolve outside export/, e.g. {outside[0][2]}")
    else:
        rep.ok("all assets inside export/", "no asset escapes the export directory")

    # Merge in ComputeAllDependencies' own unresolved list (belt and braces).
    unresolved_names = {os.path.basename(u) for u in unresolved}
    if missing or unresolved:
        detail_lines = sorted({os.path.basename(a) for _, a, _ in missing} | unresolved_names)
        rep.fail("every asset path resolves",
                 f"{len(detail_lines)} unresolved: " + ", ".join(detail_lines))
        rep.facts["unresolved"] = detail_lines
        rep.facts["unresolved_attrs"] = [
            {"attr": a, "authored": auth} for a, auth, _ in missing]
    else:
        rep.ok("every asset path resolves",
                f"all {len(attrs)} texture/asset paths point at existing files")
        rep.facts["unresolved"] = []

    # Orphan census (informational): texture files on disk that nothing references.
    texdir = os.path.join(store_dir, "textures")
    if os.path.isdir(texdir):
        on_disk = set(os.listdir(texdir))
        referenced = {os.path.basename(a) for _, a, _ in attrs}
        orphans = sorted(f for f in on_disk if f not in referenced)
        rep.facts["texture_files_on_disk"] = len(on_disk)
        if orphans:
            rep.info("orphan texture files",
                     f"{len(orphans)} file(s) on disk are unreferenced (e.g. {orphans[0]})")
    return missing


def repair_textures(store_path, missing, rep):
    """Deterministically copy authored-but-missing textures from the versioned
    ``assets/textures/`` into ``export/textures/`` when a file of the exact authored
    basename exists there. Offline, no Blender. Returns the number repaired."""
    import shutil
    store_dir = os.path.dirname(os.path.abspath(store_path))
    texdir = os.path.join(store_dir, "textures")
    os.makedirs(texdir, exist_ok=True)
    repaired, unrepairable = [], []
    for _apath, authored, _resolved in missing:
        base = os.path.basename(authored)
        src = os.path.join(ASSETS_TEXTURES, base)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(texdir, base))
            repaired.append(base)
        else:
            unrepairable.append(base)
    if repaired:
        rep.info("repair-textures",
                 f"copied {len(repaired)} texture(s) from assets/textures/ -> export/textures/: "
                 + ", ".join(sorted(set(repaired))))
    if unrepairable:
        rep.warn("repair-textures",
                 f"{len(unrepairable)} authored texture(s) have no match in assets/textures/: "
                 + ", ".join(sorted(set(unrepairable))))
    return len(repaired)


# --------------------------------------------------------------------------- #
# Stage characterisation + composer expectations                            #
# --------------------------------------------------------------------------- #
def characterise(stage, rep):
    from pxr import Usd, UsdGeom

    dp = stage.GetDefaultPrim()
    up = str(UsdGeom.GetStageUpAxis(stage))
    mpu = float(UsdGeom.GetStageMetersPerUnit(stage))

    rep.ok("stage opens", stage.GetRootLayer().identifier)

    if dp and dp.IsValid():
        rep.ok("default prim", str(dp.GetPath()))
    else:
        rep.fail("default prim", "NONE — composer AddReference has no target prim")

    rep.add(rep.PASS if up == EXPECT_UP_AXIS else rep.FAIL, "up-axis",
            f"{up} (Isaac wants {EXPECT_UP_AXIS})")
    rep.add(rep.PASS if mpu == EXPECT_METERS_PER_UNIT else rep.FAIL, "metres-per-unit",
            f"{mpu} (Isaac wants {EXPECT_METERS_PER_UNIT})")

    # counts + histogram
    types = collections.Counter()
    total = instances = 0
    for p in stage.Traverse():
        total += 1
        types[str(p.GetTypeName()) or "(untyped)"] += 1
        if p.IsInstance():
            instances += 1
    protos = stage.GetPrototypes()
    rep.facts.update(up_axis=up, meters_per_unit=mpu,
                     default_prim=str(dp.GetPath()) if dp else None,
                     prim_count=total, instances=instances, prototypes=len(protos),
                     type_histogram=dict(types))
    rep.info("prim count",
             f"{total} prims ({instances} instances, {len(protos)} prototypes)")
    rep.info("type histogram",
             ", ".join(f"{t}={c}" for t, c in types.most_common(8)))

    if not dp:
        return

    root = dp.GetPath()
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(),
                             [UsdGeom.Tokens.default_, UsdGeom.Tokens.render],
                             useExtentsHint=True)

    def aligned(prim):
        r = cache.ComputeWorldBound(prim).ComputeAlignedRange()
        return r.GetMin(), r.GetMax()

    wmn, wmx = aligned(dp)
    rep.facts["world_extent_m"] = {"min": [round(v, 3) for v in wmn],
                                   "max": [round(v, 3) for v in wmx]}
    rep.info("world extents (m)",
             f"min={tuple(round(v,2) for v in wmn)} max={tuple(round(v,2) for v in wmx)} "
             f"(includes the exterior ground plane)")

    # Floor: the surface the robot stands on.
    floor = stage.GetPrimAtPath(root.AppendPath(COMPOSER_FLOOR))
    if floor and floor.IsValid():
        fmn, fmx = aligned(floor)
        fz = round((fmn[2] + fmx[2]) / 2.0, 4)
        fx, fy = round(fmx[0] - fmn[0], 2), round(fmx[1] - fmn[1], 2)
        rep.facts["floor_z"] = fz
        rep.facts["store_footprint_m"] = {"x": [round(fmn[0], 2), round(fmx[0], 2)],
                                          "y": [round(fmn[1], 2), round(fmx[1], 2)]}
        rep.ok(f"floor present ({COMPOSER_FLOOR})",
               f"z={fz} m, footprint {fx}×{fy} m — robot has a surface to stand on")
    else:
        rep.fail(f"floor present ({COMPOSER_FLOOR})", "MISSING — nothing for the robot to stand on")


def check_composer_paths(stage, rep):
    dp = stage.GetDefaultPrim()
    if not dp:
        return
    root = dp.GetPath()

    def present(rel):
        pr = stage.GetPrimAtPath(root.AppendPath(rel))
        return bool(pr and pr.IsValid())

    missing = [g for g in COMPOSER_GROUPS if not present(g)]
    if missing:
        rep.fail("composer group hierarchy", "missing: " + ", ".join(missing))
    else:
        rep.ok("composer group hierarchy", ", ".join(COMPOSER_GROUPS))

    miss_g = [g for g in (GONDOLA_FIXTURES + GONDOLA_PRODUCTS) if not present(g)]
    if miss_g:
        rep.fail("gondola runs (aisle geometry)", "missing: " + ", ".join(miss_g))
    else:
        rep.ok("gondola runs (aisle geometry)", "7 fixture runs + 7 product runs (G1–G7)")

    if present(COMPOSER_SPAWN_CAM):
        rep.ok(f"spawn camera ({COMPOSER_SPAWN_CAM})", "marks the aisle centreline for the pedestal")
    else:
        rep.fail(f"spawn camera ({COMPOSER_SPAWN_CAM})", "MISSING — probe cannot derive the spawn pose")


def physics_census(stage, rep):
    """Report physics / collider presence. The store is render-only by design; the
    composer authors colliders at compose time, so absence is INFO, not FAIL."""
    from pxr import Usd
    scenes = sum(1 for p in stage.Traverse() if p.GetTypeName() == "PhysicsScene")
    schemas = set()
    for prim in Usd.PrimRange.Stage(stage, Usd.TraverseInstanceProxies()):
        for s in prim.GetAppliedSchemas():
            if "Physics" in s or "Collision" in s:
                schemas.add(s)
    rep.facts["physics_scenes"] = scenes
    rep.facts["physics_schemas"] = sorted(schemas)
    if scenes == 0 and not schemas:
        rep.info("physics / colliders",
                 "render-only geometry (no PhysicsScene, no collision APIs) — expected; "
                 "compose_robot_in_store.py authors colliders + gravity at compose time")
    else:
        rep.info("physics / colliders",
                 f"{scenes} PhysicsScene(s); schemas: {sorted(schemas)}")


# --------------------------------------------------------------------------- #
# Self-test (no artifact needed) — a synthetic stage exercising pass & fail   #
# --------------------------------------------------------------------------- #
def build_synthetic_stage(tmpdir, good=True):
    """A tiny in-memory-ish store stage for self-checking, written under tmpdir.
    ``good=True`` resolves cleanly; ``good=False`` seeds a dangling texture."""
    from pxr import Usd, UsdGeom, UsdShade, Sdf
    os.makedirs(os.path.join(tmpdir, "textures"), exist_ok=True)
    # a real texture file the good stage points at
    with open(os.path.join(tmpdir, "textures", "floor.png"), "wb") as f:
        f.write(b"\x89PNG\r\n\x1a\n")  # enough bytes to exist on disk
    path = os.path.join(tmpdir, "store.usda")
    stage = Usd.Stage.CreateNew(path)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    root = UsdGeom.Xform.Define(stage, "/Supermarket")
    stage.SetDefaultPrim(root.GetPrim())
    for grp in COMPOSER_GROUPS:
        UsdGeom.Xform.Define(stage, f"/Supermarket/{grp}")
    for g in GONDOLA_FIXTURES + GONDOLA_PRODUCTS:
        UsdGeom.Xform.Define(stage, f"/Supermarket/{g}")
    UsdGeom.Xform.Define(stage, "/Supermarket/GRP_Cameras/Cam_Aisle")
    floor = UsdGeom.Mesh.Define(stage, "/Supermarket/GRP_Structure/Floor")
    floor.CreateExtentAttr([(-13, -11, 0), (13, 11, 0)])
    mat = UsdShade.Material.Define(stage, "/Supermarket/_materials/floor")
    shd = UsdShade.Shader.Define(stage, "/Supermarket/_materials/floor/tex")
    shd.CreateIdAttr("UsdUVTexture")
    tex = "./textures/floor.png" if good else "./textures/MISSING_floor.png"
    shd.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(tex)
    stage.GetRootLayer().Save()
    return path


def run_selftest():
    import tempfile
    from pxr import Usd
    ok = True
    with tempfile.TemporaryDirectory() as td:
        good = build_synthetic_stage(os.path.join(td, "good"), good=True)
        rep = Report()
        stage = Usd.Stage.Open(good)
        characterise(stage, rep)
        check_composer_paths(stage, rep)
        check_assets(stage, good, rep)
        physics_census(stage, rep)
        if rep.failed:
            print("SELFTEST good-stage unexpectedly FAILED:\n" + rep.render())
            ok = False
        else:
            print("SELFTEST good-stage: PASS (clean stage validates)")

        bad = build_synthetic_stage(os.path.join(td, "bad"), good=False)
        rep2 = Report()
        stage2 = Usd.Stage.Open(bad)
        characterise(stage2, rep2)
        check_composer_paths(stage2, rep2)
        check_assets(stage2, bad, rep2)
        if not rep2.failed:
            print("SELFTEST bad-stage was NOT flagged (should have failed on the dangling texture)")
            ok = False
        else:
            print("SELFTEST bad-stage: correctly FAILED on the dangling texture")
    print("SELFTEST:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


# --------------------------------------------------------------------------- #
# Driver                                                                      #
# --------------------------------------------------------------------------- #
def validate(store_path, repair=False):
    from pxr import Usd
    rep = Report()
    if not os.path.exists(store_path):
        rep.fail("store USD exists", store_path)
        return rep
    stage = Usd.Stage.Open(store_path)
    if not stage:
        rep.fail("stage opens", f"Usd.Stage.Open returned null for {store_path}")
        return rep
    characterise(stage, rep)
    check_composer_paths(stage, rep)
    missing = check_assets(stage, store_path, rep)
    if repair and missing:
        n = repair_textures(store_path, missing, rep)
        if n:
            # Re-open + re-check assets so the exit code reflects the repair.
            stage = Usd.Stage.Open(store_path)
            # drop the stale asset checks, re-run
            rep.checks = [c for c in rep.checks
                          if c[1] not in ("every asset path resolves",
                                          "no absolute build-machine paths",
                                          "all assets inside export/", "composition",
                                          "asset-valued attributes", "orphan texture files")]
            check_assets(stage, store_path, rep)
    physics_census(stage, rep)
    return rep


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Validate the exported supermarket USD offline (no Isaac, no GPU).")
    ap.add_argument("--store", default=DEFAULT_STORE,
                    help="Store USD to validate (default: %(default)s)")
    ap.add_argument("--json", action="store_true", help="Emit the measured facts as JSON too.")
    ap.add_argument("--repair-textures", action="store_true",
                    help="Copy authored-but-missing textures from assets/textures/ into "
                         "export/textures/ (deterministic, offline) before the final verdict.")
    ap.add_argument("--selftest", action="store_true",
                    help="Validate synthetic good/bad stages (no artifact needed) and exit.")
    args = ap.parse_args(argv)

    ensure_pxr()

    if args.selftest:
        return run_selftest()

    rep = validate(args.store, repair=args.repair_textures)
    print("=" * 78)
    print(f"USD validation — {args.store}")
    print("=" * 78)
    print(rep.render())
    print("-" * 78)
    verdict = "FAIL" if rep.failed else "PASS"
    n_fail = sum(1 for s, _, _ in rep.checks if s == Report.FAIL)
    print(f"VERDICT: {verdict}" + (f"  ({n_fail} hard check(s) failed)" if rep.failed else ""))
    if args.json:
        import json
        print("-" * 78)
        print(json.dumps(rep.facts, indent=2, default=str))
    return 1 if rep.failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
