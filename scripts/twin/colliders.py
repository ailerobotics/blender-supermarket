"""Collider-selection policy for the store USD.

This is the load-bearing decision the scout report warns about: author colliders
on the ~196 instance PROTOTYPES, never on the 24.9k instances (instance-level
convex decomposition blows VRAM). Static fixtures get triangle-mesh (non-convex)
colliders; the small instanced products/tags get convex-hull colliders.

Pure logic (no pxr): a prim is described by a small ``PrimInfo`` and classified
into a ``ColliderDecision``. ``usd_probe`` builds ``PrimInfo`` from the live
stage; the tests build them by hand.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

# Collider kinds
TRIANGLE_MESH = "triangle_mesh"   # UsdPhysics.MeshCollisionAPI approximation "none"
CONVEX_HULL = "convex_hull"       # approximation "convexHull"
BOX = "box"                       # approximation "boundingCube"
SKIP = "skip"

# UsdPhysics MeshCollisionAPI approximation tokens
APPROX = {TRIANGLE_MESH: "none", CONVEX_HULL: "convexHull", BOX: "boundingCube"}

# Top-level groups whose direct meshes are large, static, non-convex scenery.
STATIC_GROUPS = {"GRP_Structure", "GRP_Fixtures", "GRP_Signage"}

# Flat, horizontal drive surfaces the coupled tow train (robot + cart) rolls across.
# A triangle-mesh collider on such a slab cooks an internal diagonal edge straight under
# the tow lane; PhysX reports an internal-edge ("ghost") contact with an in-plane normal
# when a wheel/caster comes to low speed ON that edge, arresting the train on a perfectly
# flat floor with NO z step. That is the from-rest sales_backroom_portal pin: the cart's
# front caster lands on Backroom_floor's cooked diagonal (CPU-measured perp <2 cm on every
# pinned run; >20 cm clear on the one that crossed) and the base halts at cart_x + hitch
# offset = x=-3.935 to the millimetre. A box ("boundingCube") approximation is the same
# flat slab with NO internal edge, so the casters roll across cleanly. See
# data/sim-robust-portal-floor-seam-v1/report.md. Opt-in only (``flat_floor_box``); off =>
# these stay triangle meshes, byte-identical.
DRIVE_FLOOR_MESHES = frozenset({"Backroom_floor"})

# Redundant sub-floor backdrop planes that sit a few mm BELOW the real drive floors.
# ``Ground_exterior`` is a 120x120 m plane with its top at z=-0.010, 10 mm under the
# ``Floor`` (sales) and ``Backroom_floor`` (backroom) colliders, which both top out at
# z=0 and together cover the whole drivable area. Under a ROLLING wheel the flat-floor
# box lets the wheel penetrate the box top (z=0) down to this stiffer triangle-mesh plane
# (measured wheel_z 0.150 vs 0.160 on the mesh = resting at z=-0.010), so the box floor is
# never actually exercised. Dropping this redundant sub-floor collider removes the 10 mm
# "sink trap" so the wheel rests on the box at z=0. Opt-in (paired with the flat-floor box);
# off => the collider stays, byte-identical. See data/sim-robust-portal-floor-seam-v1 and the
# sim-robust-portal-floor-box-scope evidence.
SUBFLOOR_MESHES = frozenset({"Ground_exterior"})

# Physics contact offsets authored on the flat drive-floor box so a rolling wheel's resting
# contact is exactly at the box top (z=0 => wheel_z = wheel_radius = 0.160). restOffset=0
# keeps the rest-contact surface at the box top; contactOffset is a small positive margin so
# the box generates contacts before the wheel can reach any lower plane. Authored as raw
# ``physxCollision:*`` attributes (readable by PhysX at runtime; no PhysxSchema needed offline).
DRIVE_FLOOR_BOX_REST_OFFSET = 0.0
DRIVE_FLOOR_BOX_CONTACT_OFFSET = 0.02


@dataclass(frozen=True)
class PrimInfo:
    path: str
    type_name: str                 # "Mesh", "Xform", ...
    top_group: str                 # first path segment under /Supermarket, e.g. "GRP_Fixtures"
    is_instance: bool = False      # prim.IsInstance() — an instanceable PLACEMENT
    in_prototype: bool = False     # lives under a UsdGeom prototype (/__Prototype_*)
    name: str = ""


@dataclass(frozen=True)
class ColliderDecision:
    kind: str        # TRIANGLE_MESH | CONVEX_HULL | SKIP
    approximation: str  # UsdPhysics approximation token, or "" for skip
    reason: str

    @property
    def authored(self) -> bool:
        return self.kind != SKIP


def classify(prim: PrimInfo, *, flat_floor_box: bool = False,
             drop_subfloor: bool = False) -> ColliderDecision:
    """Decide the collider for one prim. Order matters.

    ``flat_floor_box`` (opt-in, default off => byte-identical): when True, the flat
    horizontal drive-floor meshes in :data:`DRIVE_FLOOR_MESHES` get a box
    ("boundingCube") collider instead of a triangle mesh, removing the internal
    triangulation edge that ghost-collides the coupled tow casters at the portal.

    ``drop_subfloor`` (opt-in, default off => byte-identical): when True, the redundant
    sub-floor backdrop meshes in :data:`SUBFLOOR_MESHES` (e.g. ``Ground_exterior``) get NO
    collider, removing the z=-0.010 plane a rolling wheel otherwise sinks to through the
    flat-floor box. Paired with ``flat_floor_box``; see :data:`SUBFLOOR_MESHES`.
    """
    # 1. Never author on an instanceable placement — the collider belongs on the
    #    shared prototype so all instances inherit it once. This is THE guard.
    if prim.is_instance:
        return ColliderDecision(
            SKIP, "",
            "instanceable placement — author the collider on its prototype, not the instance")

    # 2. Only meshes carry collision geometry.
    if prim.type_name != "Mesh":
        return ColliderDecision(SKIP, "", "not a Mesh")

    if prim.top_group == "GRP_EnvironmentDecor":
        return ColliderDecision(SKIP, "", "nonphysical environment decoration")

    # 2a. Redundant sub-floor backdrop (Ground_exterior): drop its collider when asked, so a
    #     rolling wheel cannot sink through the flat-floor box to its z=-0.010 plane.
    if drop_subfloor and prim.name in SUBFLOOR_MESHES:
        return ColliderDecision(
            SKIP, "",
            f"redundant sub-floor backdrop {prim.name} — collider dropped (real floors cover "
            "the drivable area at z=0; removes the flat-floor-box sink trap)")

    # 3. Prototype meshes: the small, repeated products/price-tags. Convex hull is
    #    cheap and authored ONCE per prototype (~196), shared by every instance.
    if prim.in_prototype:
        return ColliderDecision(
            CONVEX_HULL, APPROX[CONVEX_HULL],
            "instance prototype mesh — convex hull authored once, shared by all instances")

    # 4. Large static scenery (structure/fixtures/signage): triangle mesh so the
    #    concave shelves/walls collide faithfully. Immovable, so no mass.
    if prim.top_group in STATIC_GROUPS:
        # 4a. A flat drive floor the tow train rolls across: box collider (no internal
        #     triangle edge for the casters to ghost-collide at the portal). Opt-in.
        if flat_floor_box and prim.name in DRIVE_FLOOR_MESHES:
            return ColliderDecision(
                BOX, APPROX[BOX],
                f"flat drive-floor {prim.name} — box collider (no internal triangle edge "
                "to ghost-collide the coupled tow casters at the portal)")
        return ColliderDecision(
            TRIANGLE_MESH, APPROX[TRIANGLE_MESH],
            f"static non-convex scenery in {prim.top_group} — triangle-mesh collider")

    # 5. A non-instanced mesh sitting under a product/tag group (rare — the export
    #    has exactly one such price-tag mesh): treat like a small object.
    return ColliderDecision(
        CONVEX_HULL, APPROX[CONVEX_HULL],
        "non-instanced small object — convex hull")


@dataclass
class ColliderPlan:
    triangle_mesh: int = 0
    convex_hull: int = 0
    box: int = 0
    skipped_instances: int = 0
    skipped_other: int = 0

    @property
    def authored(self) -> int:
        return self.triangle_mesh + self.convex_hull + self.box


def plan_colliders(prims: Iterable[PrimInfo], *, flat_floor_box: bool = False,
                   drop_subfloor: bool = False) -> ColliderPlan:
    """Aggregate classify() over prims into counts for the dry-run / logs."""
    plan = ColliderPlan()
    for p in prims:
        d = classify(p, flat_floor_box=flat_floor_box, drop_subfloor=drop_subfloor)
        if d.kind == TRIANGLE_MESH:
            plan.triangle_mesh += 1
        elif d.kind == CONVEX_HULL:
            plan.convex_hull += 1
        elif d.kind == BOX:
            plan.box += 1
        elif p.is_instance:
            plan.skipped_instances += 1
        else:
            plan.skipped_other += 1
    return plan
