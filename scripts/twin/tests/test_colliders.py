"""Collider-selection policy — the prototype-vs-instance decision."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from twin import colliders as C  # noqa: E402


def _mesh(group, path="/p", type_name="Mesh", **kw):
    return C.PrimInfo(path=path, type_name=type_name, top_group=group, name="m", **kw)


def test_instance_placement_is_never_authored():
    # The load-bearing guard: an instanceable placement must be skipped so the
    # collider is authored on its prototype, not on all 24.9k instances.
    d = C.classify(_mesh("GRP_Products", is_instance=True, type_name="Xform"))
    assert d.kind == C.SKIP
    assert "prototype" in d.reason.lower()


def test_prototype_mesh_gets_convex_hull():
    d = C.classify(_mesh("__prototype__", in_prototype=True))
    assert d.kind == C.CONVEX_HULL
    assert d.approximation == "convexHull"


def test_fixture_mesh_gets_triangle_mesh():
    d = C.classify(_mesh("GRP_Fixtures"))
    assert d.kind == C.TRIANGLE_MESH
    assert d.approximation == "none"


def test_structure_and_signage_get_triangle_mesh():
    assert C.classify(_mesh("GRP_Structure")).kind == C.TRIANGLE_MESH
    assert C.classify(_mesh("GRP_Signage")).kind == C.TRIANGLE_MESH


def test_non_mesh_skipped():
    d = C.classify(C.PrimInfo(path="/x", type_name="Xform", top_group="GRP_Fixtures"))
    assert d.kind == C.SKIP


def test_loose_product_mesh_is_convex():
    # The single non-instanced price-tag mesh -> small object -> convex.
    d = C.classify(_mesh("GRP_PriceTags"))
    assert d.kind == C.CONVEX_HULL


def test_plan_matches_measured_store_shape():
    # Mirror the real export: 702 static meshes, 196 prototype meshes, 24927 instances.
    prims = []
    prims += [_mesh("GRP_Structure") for _ in range(10)]
    prims += [_mesh("GRP_Fixtures") for _ in range(674)]
    prims += [_mesh("GRP_Signage") for _ in range(18)]
    prims += [_mesh("__prototype__", in_prototype=True) for _ in range(196)]
    prims += [_mesh("GRP_Products", is_instance=True, type_name="Xform")
              for _ in range(24927)]
    plan = C.plan_colliders(prims)
    assert plan.triangle_mesh == 702        # structure + fixtures + signage
    assert plan.convex_hull == 196          # one per prototype, shared by all instances
    assert plan.skipped_instances == 24927  # NONE authored per-instance
    # 196 colliders cover 24927 instances -> the VRAM-safe ratio the report demands.
    assert plan.convex_hull < plan.skipped_instances / 100
