"""Offline composition core — exercised against a SYNTHETIC instanced store.

Like test_validate_usd, this needs the USD (``pxr``) bindings and is skipped where
they are absent. It builds a tiny store that mirrors the real one's shape — a
``prototypes`` **class** scope with a shared source mesh, ``instanceable`` product
placements that reference it, static fixtures, a loose price tag, a floor and a
spawn camera — so the collider-propagation and placement logic is proven WITHOUT
opening the 54k-prim export (and without a GPU).

    .venv-usd/bin/python -m pytest scripts/twin/tests/test_compose.py
"""
import os
import sys

import pytest

pytest.importorskip("pxr")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from twin import compose, geometry  # noqa: E402
from pxr import Usd, UsdGeom, Sdf  # noqa: E402

DP = "Store"  # default prim name of the synthetic store


def build_synthetic_store(path, n_instances=5, floor_z=0.0):
    """A minimal instanced store: class-scope source mesh + instanceable placements,
    static fixtures, a loose price tag, a floor plane, a spawn camera."""
    st = Usd.Stage.CreateNew(path)
    UsdGeom.SetStageUpAxis(st, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(st, 1.0)
    root = UsdGeom.Xform.Define(st, f"/{DP}")
    st.SetDefaultPrim(root.GetPrim())

    # floor plane at floor_z, 6x6 m
    floor = UsdGeom.Mesh.Define(st, f"/{DP}/GRP_Structure/Floor")
    floor.CreatePointsAttr([(-3, -3, floor_z), (3, -3, floor_z), (3, 3, floor_z), (-3, 3, floor_z)])
    floor.CreateFaceVertexCountsAttr([4])
    floor.CreateFaceVertexIndicesAttr([0, 1, 2, 3])
    floor.CreateExtentAttr([(-3, -3, floor_z), (3, 3, floor_z)])
    # a wall (static structure) + a fixture shelf (static fixture)
    for p, grp in [(f"/{DP}/GRP_Structure/Wall", None), (f"/{DP}/GRP_Fixtures/Shelf", None)]:
        m = UsdGeom.Mesh.Define(st, p)
        m.CreatePointsAttr([(-1, -1, 0), (1, -1, 0), (1, 1, 2), (-1, 1, 2)])
        m.CreateFaceVertexCountsAttr([4])
        m.CreateFaceVertexIndicesAttr([0, 1, 2, 3])
        m.CreateExtentAttr([(-1, -1, 0), (1, 1, 2)])

    # Class scope holding the shared source mesh (mirrors the real export: only the
    # `prototypes` scope is a `class`; the proto holder + mesh are `def`, so they are
    # abstract only by ancestry and become concrete when an instanceable placement
    # references the holder — exactly the shape the composer's collider fix relies on).
    proto = st.OverridePrim(f"/{DP}/prototypes")
    proto.SetSpecifier(Sdf.SpecifierClass)
    st.DefinePrim(f"/{DP}/prototypes/proto_can_0", "Xform")  # def under the class scope
    src_mesh = UsdGeom.Mesh.Define(st, f"/{DP}/prototypes/proto_can_0/M_can")
    src_mesh.CreatePointsAttr([(-0.03, -0.03, 0), (0.03, -0.03, 0), (0.03, 0.03, 0.1), (-0.03, 0.03, 0.1)])
    src_mesh.CreateFaceVertexCountsAttr([4])
    src_mesh.CreateFaceVertexIndicesAttr([0, 1, 2, 3])

    # instanceable product placements referencing the class source
    UsdGeom.Xform.Define(st, f"/{DP}/GRP_Products")
    for i in range(n_instances):
        place = st.DefinePrim(f"/{DP}/GRP_Products/can_{i}", "Xform")
        place.GetReferences().AddInternalReference(Sdf.Path(f"/{DP}/prototypes/proto_can_0"))
        place.SetInstanceable(True)

    # a loose (non-instanced) price-tag mesh directly under GRP_PriceTags
    tag = UsdGeom.Mesh.Define(st, f"/{DP}/GRP_PriceTags/loose_tag")
    tag.CreatePointsAttr([(0, 0, 0), (0.05, 0, 0), (0.05, 0.03, 0), (0, 0.03, 0)])
    tag.CreateFaceVertexCountsAttr([4])
    tag.CreateFaceVertexIndicesAttr([0, 1, 2, 3])

    UsdGeom.Xform.Define(st, f"/{DP}/GRP_Cameras/Cam_Aisle")
    st.GetRootLayer().Save()
    return path


def _spawn(floor_z=0.0, x=0.0, y=0.0, clearance=0.9):
    sp = geometry.SpawnCandidate(x=x, y=y, z=floor_z, clearance_m=clearance, label="aisle")
    sp.root_translate = geometry.root_translate_for_pedestal((x, y), floor_z)
    return sp


# gondola runs either side of x=0 -> the robot at x=0 sits in the aisle gap.
EXTRAS = {"gondola_x_ranges": [(-2.0, -1.0), (1.0, 2.0)]}


def test_build_composition_propagates_collider_to_all_prototypes(tmp_path):
    store = build_synthetic_store(str(tmp_path / "store.usda"), n_instances=5)
    out = str(tmp_path / "store_with_robot.usda")
    stub = compose.build_stub_robot(str(tmp_path / "stub.usda"))
    res = compose.build_composition(store, stub, out, _spawn())
    # every runtime prototype inherits the convex collider (authored ONCE on source)
    assert res.prototypes_total >= 1
    assert res.prototypes_covered == res.prototypes_total
    # triangle-mesh on the 3 static meshes (Floor, Wall, Shelf); convex on source + loose tag
    assert res.triangle_mesh == 3
    assert res.convex_hull == 2  # 1 class source + 1 loose price tag


def test_authoring_never_touches_a_prototype_directly(tmp_path):
    # Regression guard for the bug this task found: authoring a collider on a USD
    # instancing prototype raises. build_composition must NOT do that — it authors on
    # the class source. If it regressed to touching prototypes, this would raise.
    store = build_synthetic_store(str(tmp_path / "store.usda"))
    out = str(tmp_path / "out.usda")
    stub = compose.build_stub_robot(str(tmp_path / "stub.usda"))
    res = compose.build_composition(store, stub, out, _spawn())  # must not raise
    assert res.prototypes_covered == res.prototypes_total


def test_validate_composition_passes_for_sane_placement(tmp_path):
    store = build_synthetic_store(str(tmp_path / "store.usda"))
    out = str(tmp_path / "out.usda")
    stub = compose.build_stub_robot(str(tmp_path / "stub.usda"))
    compose.build_composition(store, stub, out, _spawn())
    rep = compose.validate_composition(out, _spawn(), floor_z=0.0, extras=EXTRAS,
                                       expect_triangle=3)
    assert not rep.failed, rep.render()
    # the robot really is on the floor and at the aisle spot
    assert rep.facts["robot_bbox"]["min"][2] == pytest.approx(0.0, abs=1e-4)
    cx, cy = rep.facts["robot_footprint_centre"]
    assert (cx, cy) == pytest.approx((0.0, 0.0), abs=0.02)


def test_validate_flags_robot_floating_off_the_floor(tmp_path):
    store = build_synthetic_store(str(tmp_path / "store.usda"), floor_z=0.0)
    out = str(tmp_path / "out.usda")
    stub = compose.build_stub_robot(str(tmp_path / "stub.usda"))
    compose.build_composition(store, stub, out, _spawn())
    # validate against a WRONG floor (robot bottom is at 0.0, claim floor is 0.5)
    rep = compose.validate_composition(out, _spawn(floor_z=0.5), floor_z=0.5, extras=EXTRAS)
    assert rep.failed
    assert any("stands ON the floor" in n and s == compose.ComposeReport.FAIL
               for s, n, _ in rep.checks)


def test_validate_flags_robot_inside_a_gondola(tmp_path):
    store = build_synthetic_store(str(tmp_path / "store.usda"))
    out = str(tmp_path / "out.usda")
    stub = compose.build_stub_robot(str(tmp_path / "stub.usda"))
    # place the robot at x=-1.5 -> inside the left gondola run (-2,-1)
    sp = _spawn(x=-1.5)
    compose.build_composition(store, stub, out, sp)
    rep = compose.validate_composition(out, sp, floor_z=0.0, extras=EXTRAS)
    assert rep.failed
    assert any("aisle gap" in n and s == compose.ComposeReport.FAIL
               for s, n, _ in rep.checks)


def test_product_collider_none_skips_prototype_colliders(tmp_path):
    store = build_synthetic_store(str(tmp_path / "store.usda"))
    out = str(tmp_path / "out.usda")
    stub = compose.build_stub_robot(str(tmp_path / "stub.usda"))
    res = compose.build_composition(store, stub, out, _spawn(), product_collider="none")
    assert res.prototypes_covered == 0            # products skipped
    assert res.triangle_mesh == 3                 # static still authored
    assert res.convex_hull == 1                   # only the loose price tag


def test_committed_store_is_not_modified(tmp_path):
    import hashlib
    store = build_synthetic_store(str(tmp_path / "store.usda"))
    before = hashlib.sha256(open(store, "rb").read()).hexdigest()
    out = str(tmp_path / "out.usda")
    stub = compose.build_stub_robot(str(tmp_path / "stub.usda"))
    compose.build_composition(store, stub, out, _spawn())
    after = hashlib.sha256(open(store, "rb").read()).hexdigest()
    assert before == after  # the store is sublayered read-only, never edited


def test_default_prim_and_axes_inherited(tmp_path):
    store = build_synthetic_store(str(tmp_path / "store.usda"))
    out = str(tmp_path / "out.usda")
    stub = compose.build_stub_robot(str(tmp_path / "stub.usda"))
    res = compose.build_composition(store, stub, out, _spawn())
    assert res.default_prim == f"/{DP}"
    re = Usd.Stage.Open(out)
    assert re.GetDefaultPrim().GetPath() == Sdf.Path(f"/{DP}")
    assert str(UsdGeom.GetStageUpAxis(re)) == "Z"
    assert float(UsdGeom.GetStageMetersPerUnit(re)) == 1.0
