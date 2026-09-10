"""Pure-USD composition core for standing the robot in the store.

Everything here is plain USD (``pxr``) — no ``isaacsim``, no ``omni.*``, no GPU. It
is exactly the stage manipulation a real Isaac run performs *around* the simulator:
reference/sublayer the store, author the ``PhysicsScene``, author the colliders and
place the robot. Splitting it out means the whole composition can be driven for real
offline (``compose_robot_in_store.py --author-offline``) and unit tested
(``tests/test_compose.py``); the Isaac path calls the SAME functions so the only code
that stays unverified until a GPU boot is the simulator itself (URDF→USD import,
stepping, screenshot).

Why a sublayer, not a nested reference
--------------------------------------
The store instances its ~25k products from ~196 shared source meshes held under the
``/Supermarket/prototypes`` **class** scope; each product placement is an
``instanceable`` prim that references one of those class prims. USD forbids authoring
on the runtime ``/__Prototype_*`` prims ("authoring to an instancing prototype is not
allowed"), and an override authored on the class from *outside* the store's reference
layer stack does **not** reach the prototype either. So to give every instance a
shared collider we author the collision API on the **class source meshes**, and we
compose the store as a **sublayer** of the output so those overrides live inside the
store's own layer stack and propagate into all prototypes. The committed store file is
only ever read (sublayered by absolute path), never written.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

# Pure sibling policy modules (no pxr). Import whether run as a package or loosely.
try:
    from . import colliders, geometry, usd_probe
except ImportError:  # pragma: no cover - only when run as a top-level script
    import colliders  # type: ignore
    import geometry  # type: ignore
    import usd_probe  # type: ignore

DEFAULT_ROBOT_PRIM = "Robot"
PHYSICS_SCENE_PRIM = "physicsScene"
PROTOTYPES_SCOPE = "prototypes"   # the store's class scope holding shared source meshes
LOOSE_TAG_GROUP = "GRP_PriceTags"


# --------------------------------------------------------------------------- #
# A tiny pass/fail report, same shape/spirit as validate_usd.Report.          #
# --------------------------------------------------------------------------- #
class ComposeReport:
    PASS, FAIL, INFO = "PASS", "FAIL", "····"

    def __init__(self):
        self.checks: List[Tuple[str, str, str]] = []
        self.facts: dict = {}

    def _add(self, s, n, d=""):
        self.checks.append((s, n, d))

    def ok(self, n, d=""):
        self._add(self.PASS, n, d)

    def fail(self, n, d=""):
        self._add(self.FAIL, n, d)

    def info(self, n, d=""):
        self._add(self.INFO, n, d)

    def check(self, cond, n, d=""):
        self._add(self.PASS if cond else self.FAIL, n, d)

    @property
    def failed(self) -> bool:
        return any(s == self.FAIL for s, _, _ in self.checks)

    def render(self) -> str:
        return "\n".join(f"[{s}] {n}" + (f": {d}" if d else "") for s, n, d in self.checks)


@dataclass
class ComposeResult:
    out_path: str
    default_prim: str
    robot_prim: str
    robot_translate: Tuple[float, float, float]
    triangle_mesh: int = 0
    convex_hull: int = 0
    prototypes_covered: int = 0
    prototypes_total: int = 0
    prim_count: int = 0
    store_sublayer: str = ""


# --------------------------------------------------------------------------- #
# Stub robot (offline stand-in for the URDF→USD import, which needs Isaac).    #
# --------------------------------------------------------------------------- #
def build_stub_robot(path: str) -> str:
    """Write a faithful stand-in robot USD in the URDF's ``world`` frame: the
    0.2×0.4×0.7 pedestal box centred at the URDF ``pedestal_joint`` origin
    ``(0, -1.2, 0.35)`` (so its base is on the world Z=0 plane) plus a thin arm
    envelope above it. Placing this at the same root translate as the real robot
    lands its pedestal exactly where the real pedestal would, so the offline
    placement check exercises the real transform maths — not the mesh detail
    (which only the true URDF→USD import under Isaac can provide)."""
    usd_probe.ensure_pxr_env()
    from pxr import Usd, UsdGeom

    st = Usd.Stage.CreateNew(path)
    UsdGeom.SetStageUpAxis(st, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(st, 1.0)
    robot = UsdGeom.Xform.Define(st, "/shelfbot")
    st.SetDefaultPrim(robot.GetPrim())

    ox, oy, oz = geometry.PEDESTAL_OFFSET_FROM_ROOT[0], geometry.PEDESTAL_OFFSET_FROM_ROOT[1], 0.35
    ped = UsdGeom.Cube.Define(st, "/shelfbot/pedestal_link")
    ped.CreateSizeAttr(1.0)
    UsdGeom.XformCommonAPI(ped).SetTranslate((ox, oy, oz))
    UsdGeom.XformCommonAPI(ped).SetScale((0.2, 0.4, 0.7))  # -> spans Z [0, 0.7]
    arm = UsdGeom.Cube.Define(st, "/shelfbot/ur_arm_env")
    arm.CreateSizeAttr(1.0)
    UsdGeom.XformCommonAPI(arm).SetTranslate((ox, oy, 1.3))
    UsdGeom.XformCommonAPI(arm).SetScale((0.18, 0.18, 1.2))  # -> up to Z ~1.9
    st.GetRootLayer().Save()
    return path


# --------------------------------------------------------------------------- #
# Composition steps — each takes a live Usd.Stage; shared by offline + Isaac.  #
# --------------------------------------------------------------------------- #
def open_output_stage(out_path: str, store_path: str):
    """Create the output stage, sublayer the committed store by absolute path (so
    class-source collider overrides propagate into the prototypes), and inherit the
    store's default prim / up-axis / metres-per-unit. Returns (stage, default_prim_path)."""
    usd_probe.ensure_pxr_env()
    from pxr import Usd, UsdGeom, Sdf

    if os.path.exists(out_path):
        os.remove(out_path)
    d = os.path.dirname(os.path.abspath(out_path))
    if d:
        os.makedirs(d, exist_ok=True)
    # Read the store's default prim from its layer directly (cheap; avoids opening
    # and composing the whole 54k-prim store an extra time under memory pressure).
    src_layer = Sdf.Layer.FindOrOpen(os.path.abspath(store_path))
    if not src_layer:
        raise RuntimeError(f"could not open store layer: {store_path}")
    if not src_layer.defaultPrim:
        raise RuntimeError(f"store has no default prim: {store_path}")
    root = Sdf.Path("/" + src_layer.defaultPrim)
    stage = Usd.Stage.CreateNew(out_path)
    stage.GetRootLayer().subLayerPaths.append(os.path.abspath(store_path))
    stage.SetDefaultPrim(stage.GetPrimAtPath(root))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    return stage, root


def author_physics_scene(stage, root):
    """Author a Z-up gravity PhysicsScene under the store root."""
    from pxr import UsdPhysics, Gf
    scene = UsdPhysics.Scene.Define(stage, root.AppendChild(PHYSICS_SCENE_PRIM))
    scene.CreateGravityDirectionAttr().Set(Gf.Vec3f(0.0, 0.0, -1.0))
    scene.CreateGravityMagnitudeAttr().Set(9.81)
    return scene


def _apply_collider(prim, decision):
    from pxr import UsdPhysics
    UsdPhysics.CollisionAPI.Apply(prim)
    UsdPhysics.MeshCollisionAPI.Apply(prim).CreateApproximationAttr().Set(decision.approximation)


def author_static_colliders(stage, root) -> int:
    """Triangle-mesh colliders on the direct static meshes (structure/fixtures/
    signage). Static = immovable: no RigidBodyAPI, so the robot can't shove scenery."""
    from pxr import Usd
    n = 0
    for grp in sorted(colliders.STATIC_GROUPS):
        g = stage.GetPrimAtPath(root.AppendChild(grp))
        if not g:
            continue
        for p in Usd.PrimRange(g):
            if p.GetTypeName() != "Mesh":
                continue
            info = colliders.PrimInfo(path=str(p.GetPath()), type_name="Mesh", top_group=grp,
                                      is_instance=p.IsInstance(), in_prototype=False,
                                      name=p.GetName())
            d = colliders.classify(info)
            if d.kind == colliders.TRIANGLE_MESH:
                _apply_collider(p, d)
                n += 1
    return n


def author_product_colliders(stage, root, enable: bool = True) -> int:
    """Convex-hull colliders authored ONCE on the shared product/tag source meshes so
    every instance inherits — never on the runtime prototypes (USD forbids it) and
    never per-instance (would blow VRAM). Sources live under the ``prototypes`` class
    scope (abstract → walk with the all-prims predicate). Also covers the rare loose,
    non-instanced price-tag mesh. Returns the number authored (0 if ``enable`` is False,
    except the loose tag which is a normal small object)."""
    from pxr import Usd
    ALL = Usd.PrimAllPrimsPredicate
    n = 0
    # 1) shared source meshes under the class scope (the prototype sources).
    if enable:
        scope = stage.GetPrimAtPath(root.AppendChild(PROTOTYPES_SCOPE))
        if scope:
            for p in Usd.PrimRange(scope, ALL):
                if p.GetTypeName() != "Mesh":
                    continue
                info = colliders.PrimInfo(path=str(p.GetPath()), type_name="Mesh",
                                          top_group="__prototype__", is_instance=False,
                                          in_prototype=True, name=p.GetName())
                d = colliders.classify(info)
                if d.kind == colliders.CONVEX_HULL:
                    _apply_collider(p, d)
                    n += 1
    # 2) loose non-instanced price-tag mesh(es) sitting directly under GRP_PriceTags.
    tags = stage.GetPrimAtPath(root.AppendChild(LOOSE_TAG_GROUP))
    if tags:
        for p in Usd.PrimRange(tags):
            if p.GetTypeName() != "Mesh" or p.IsInstanceProxy() or p.IsInstance():
                continue
            info = colliders.PrimInfo(path=str(p.GetPath()), type_name="Mesh",
                                      top_group=LOOSE_TAG_GROUP, is_instance=False,
                                      in_prototype=False, name=p.GetName())
            d = colliders.classify(info)
            if d.kind == colliders.CONVEX_HULL:
                _apply_collider(p, d)
                n += 1
    return n


def place_robot(stage, root, robot_usd: str, translate: Sequence[float],
                yaw_deg: float = 0.0, robot_prim: str = DEFAULT_ROBOT_PRIM):
    """Reference the robot USD under the store root and set its root xform so the
    pedestal lands at the aisle spot (``translate`` already compensates the pedestal
    offset — see geometry.root_translate_for_pedestal).

    ``translate`` is the pre-import estimate assuming the referenced robot's root prim
    coincides with the URDF ``world`` link. The URDF importer's root convention can
    differ (fix_base / base_link); if the real run lands the pedestal off the floor,
    add a correction pass here that re-measures the imported pedestal prim and nudges
    the root xform. The offline stub is authored in the ``world`` frame, so it needs
    no correction — which is exactly why the correction is the one placement detail
    that stays a real-run concern."""
    from pxr import Usd, UsdGeom, Gf
    prim = stage.DefinePrim(root.AppendChild(robot_prim), "Xform")
    prim.GetReferences().AddReference(os.path.abspath(robot_usd))
    xf = UsdGeom.Xformable(prim)
    xf.ClearXformOpOrder()
    xf.AddTranslateOp().Set(Gf.Vec3d(float(translate[0]), float(translate[1]), float(translate[2])))
    if yaw_deg:
        xf.AddRotateZOp().Set(float(yaw_deg))
    return prim


def build_composition(store_path: str, robot_usd: str, out_path: str,
                      spawn, yaw_deg: float = 0.0, product_collider: str = "convex",
                      robot_prim: str = DEFAULT_ROBOT_PRIM) -> ComposeResult:
    """Author the whole composed stage with pure USD and save it to ``out_path``.

    ``spawn`` is a geometry.SpawnCandidate (from usd_probe.probe): its
    ``root_translate`` places the robot. Returns a ComposeResult with the counts.
    The committed store at ``store_path`` is only read (sublayered), never written.
    """
    usd_probe.ensure_pxr_env()
    from pxr import Usd

    stage, root = open_output_stage(out_path, store_path)
    author_physics_scene(stage, root)
    n_tri = author_static_colliders(stage, root)
    n_cvx = author_product_colliders(stage, root, enable=(product_collider != "none"))
    translate = spawn.root_translate or (spawn.x, spawn.y, spawn.z)
    place_robot(stage, root, robot_usd, translate, yaw_deg, robot_prim)
    stage.GetRootLayer().Save()

    # Count on the live stage (Save persisted exactly these overrides). The
    # independent re-open + full verification is validate_composition's job.
    protos = stage.GetPrototypes()
    covered = sum(1 for proto in protos for pp in Usd.PrimRange(proto)
                  if pp.GetTypeName() == "Mesh" and "PhysicsCollisionAPI" in pp.GetAppliedSchemas())
    return ComposeResult(
        out_path=out_path, default_prim=str(root),
        robot_prim=str(root.AppendChild(robot_prim)), robot_translate=tuple(translate),
        triangle_mesh=n_tri, convex_hull=n_cvx,
        prototypes_covered=covered, prototypes_total=len(protos),
        prim_count=sum(1 for _ in stage.Traverse()), store_sublayer=os.path.abspath(store_path))


# --------------------------------------------------------------------------- #
# Offline verification of a composed stage — the real "is it sane?" check.     #
# --------------------------------------------------------------------------- #
def _aligned_bbox(stage, prim):
    from pxr import Usd, UsdGeom
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(),
                             [UsdGeom.Tokens.default_, UsdGeom.Tokens.render])
    r = cache.ComputeWorldBound(prim).ComputeAlignedRange()
    return r.GetMin(), r.GetMax()


def validate_composition(out_path: str, spawn, floor_z: float, extras: dict,
                         product_collider: str = "convex",
                         expect_triangle: Optional[int] = None,
                         robot_prim: str = DEFAULT_ROBOT_PRIM,
                         floor_tol: float = 1e-4, xy_tol: float = 0.02) -> ComposeReport:
    """Open the composed stage and prove the robot landed sanely: right pose, on the
    floor (not in it or floating), inside an aisle (not inside a shelf), colliders in
    place, instancing preserved. Numbers, not claims."""
    usd_probe.ensure_pxr_env()
    from pxr import Usd, UsdGeom

    rep = ComposeReport()
    if not os.path.exists(out_path):
        rep.fail("composed stage exists", out_path)
        return rep
    stage = Usd.Stage.Open(out_path)
    if not stage:
        rep.fail("composed stage opens", out_path)
        return rep
    rep.ok("composed stage opens", out_path)

    dp = stage.GetDefaultPrim()
    rep.check(bool(dp and dp.IsValid()), "default prim present",
              str(dp.GetPath()) if dp else "NONE")
    rep.check(str(UsdGeom.GetStageUpAxis(stage)) == "Z", "up-axis Z",
              str(UsdGeom.GetStageUpAxis(stage)))
    rep.check(float(UsdGeom.GetStageMetersPerUnit(stage)) == 1.0, "metres-per-unit 1.0",
              str(UsdGeom.GetStageMetersPerUnit(stage)))
    if not dp:
        return rep
    root = dp.GetPath()

    # physics scene
    scene = stage.GetPrimAtPath(root.AppendChild(PHYSICS_SCENE_PRIM))
    rep.check(bool(scene and scene.GetTypeName() == "PhysicsScene"), "PhysicsScene authored",
              str(scene.GetPath()) if scene else "MISSING")

    # colliders
    protos = stage.GetPrototypes()
    covered = sum(1 for proto in protos for pp in Usd.PrimRange(proto)
                  if pp.GetTypeName() == "Mesh" and "PhysicsCollisionAPI" in pp.GetAppliedSchemas())
    if product_collider == "none":
        rep.ok("product prototypes: colliders skipped (--product-collider none)",
               f"{covered}/{len(protos)} prototypes carry a collider")
    else:
        rep.check(len(protos) > 0 and covered == len(protos),
                  "every product prototype inherits a convex collider",
                  f"{covered}/{len(protos)} (authored on the shared source, never per-instance)")
    n_static = 0
    for grp in sorted(colliders.STATIC_GROUPS):
        g = stage.GetPrimAtPath(root.AppendChild(grp))
        if not g:
            continue
        for p in Usd.PrimRange(g):
            if p.GetTypeName() == "Mesh" and "PhysicsCollisionAPI" in p.GetAppliedSchemas():
                n_static += 1
    if expect_triangle is not None:
        rep.check(n_static == expect_triangle, "static triangle-mesh colliders authored",
                  f"{n_static} (expected {expect_triangle})")
    else:
        rep.ok("static triangle-mesh colliders authored", str(n_static))
    rep.facts["triangle_mesh"] = n_static
    rep.facts["prototypes_covered"] = covered
    rep.facts["prototypes_total"] = len(protos)

    # robot placement — the load-bearing geometric check
    robot = stage.GetPrimAtPath(root.AppendChild(robot_prim))
    if not (robot and robot.IsValid()):
        rep.fail("robot present", "MISSING")
        return rep
    rep.ok("robot present", str(robot.GetPath()))
    mn, mx = _aligned_bbox(stage, robot)
    cx, cy = (mn[0] + mx[0]) / 2.0, (mn[1] + mx[1]) / 2.0
    rep.facts["robot_bbox"] = {"min": [round(v, 4) for v in mn], "max": [round(v, 4) for v in mx]}
    rep.facts["robot_footprint_centre"] = [round(cx, 4), round(cy, 4)]

    rep.check(abs(mn[2] - floor_z) <= floor_tol, "robot stands ON the floor",
              f"pedestal bottom Z={mn[2]:.4f} vs floor_z={floor_z} (tol {floor_tol})")
    rep.check(abs(cx - spawn.x) <= xy_tol and abs(cy - spawn.y) <= xy_tol,
              "robot footprint at the intended aisle spot",
              f"centre=({cx:.3f},{cy:.3f}) target=({spawn.x},{spawn.y})")

    # inside an aisle, not inside a shelf: robot X-span must fall in a gondola gap.
    ranges = extras.get("gondola_x_ranges", [])
    in_gap = _footprint_in_aisle_gap(mn[0], mx[0], ranges)
    rep.check(in_gap, "robot is in an aisle gap, not inside a gondola",
              f"robot X∈[{mn[0]:.3f},{mx[0]:.3f}]; gondola runs {ranges}")
    rep.info("robot clearance at spawn (from geometry)", f"{spawn.clearance_m} m")

    # instancing preserved (no accidental de-instancing blow-up)
    n = sum(1 for _ in stage.Traverse())
    rep.facts["prim_count"] = n
    rep.info("composed prim count (instancing preserved)", str(n))
    return rep


def _footprint_in_aisle_gap(x_lo: float, x_hi: float,
                            gondola_x_ranges: Sequence[Tuple[float, float]]) -> bool:
    """True if [x_lo, x_hi] overlaps no gondola X-run and sits between two runs (or
    outside all runs). Empty ranges → cannot prove, return False."""
    if not gondola_x_ranges:
        return False
    for (gx0, gx1) in gondola_x_ranges:
        if x_hi > gx0 and x_lo < gx1:      # overlaps a gondola run
            return False
    return True
