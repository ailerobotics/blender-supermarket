"""sim-robust-portal-floor-seam: the flat drive floor becomes a box collider (opt-in).

Root cause (CPU-established, no collider edge exists at the pin): the coupled tow pins
because the cart's front-left caster lands on the ``Backroom_floor`` triangle-mesh cooked
diagonal (a PhysX internal-edge ghost contact on a flat floor), and the base halts at
cart_x + hitch offset = x=-3.935. The fix authors the flat drive floor as a box
(``boundingCube``) collider -- no internal edge. Default off => byte-identical.

Pure-logic tiers need no pxr; the authored-stage round-trip skips without usd-core.
"""
import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from twin import colliders  # noqa: E402
from twin.mobile import MobileSpec  # noqa: E402
from twin.restock import portal_floor as pf  # noqa: E402


def _floor(name="Backroom_floor"):
    return colliders.PrimInfo(path=f"/Supermarket/GRP_Structure/{name}/{name}",
                              type_name="Mesh", top_group="GRP_Structure", name=name)


# --- collider policy -------------------------------------------------------

def test_floor_off_is_triangle_mesh_byte_identical():
    d_default = colliders.classify(_floor())
    d_off = colliders.classify(_floor(), flat_floor_box=False)
    assert d_default.kind == colliders.TRIANGLE_MESH
    assert d_default.approximation == "none"
    # the flag defaulting / explicit-off must be identical to the historical call
    assert (d_off.kind, d_off.approximation) == (d_default.kind, d_default.approximation)


def test_floor_on_is_box():
    d = colliders.classify(_floor(), flat_floor_box=True)
    assert d.kind == colliders.BOX
    assert d.approximation == "boundingCube"
    assert colliders.APPROX[colliders.BOX] == "boundingCube"


def test_non_drive_structure_unchanged_by_switch():
    # walls / ceiling / shelves must STAY triangle mesh even with the switch on.
    for name in ("Backroom_wall_east", "Backroom_ceiling", "Wall_back_west"):
        info = colliders.PrimInfo(path=f"/x/{name}", type_name="Mesh",
                                  top_group="GRP_Structure", name=name)
        assert colliders.classify(info, flat_floor_box=True).kind == colliders.TRIANGLE_MESH
    # a fixtures mesh is untouched too.
    fx = colliders.PrimInfo(path="/x", type_name="Mesh", top_group="GRP_Fixtures", name="Rack_3_deck_0")
    assert colliders.classify(fx, flat_floor_box=True).kind == colliders.TRIANGLE_MESH


def test_instance_and_prototype_unaffected_by_switch():
    inst = colliders.PrimInfo(path="/i", type_name="Mesh", top_group="GRP_Products",
                              is_instance=True, name="p")
    proto = colliders.PrimInfo(path="/p", type_name="Mesh", top_group="__prototype__",
                               in_prototype=True, name="p")
    for p in (inst, proto):
        assert colliders.classify(p).kind == colliders.classify(p, flat_floor_box=True).kind


def test_plan_colliders_default_has_no_box():
    prims = [_floor(), colliders.PrimInfo(path="/w", type_name="Mesh",
             top_group="GRP_Structure", name="Backroom_wall_east")]
    assert colliders.plan_colliders(prims).box == 0
    assert colliders.plan_colliders(prims, flat_floor_box=True).box == 1


# --- base rig numbers ------------------------------------------------------

def test_hitch_offset_and_pin_x():
    spec = MobileSpec()
    assert pf.hitch_offset(spec) == pytest.approx(1.27, abs=1e-9)
    # cart stuck at the measured -5.205 => base pins at -3.935 (the whole cluster).
    assert -5.205 + pf.hitch_offset(spec) == pytest.approx(-3.935, abs=1e-9)


def test_drive_wheel_contact_is_base_origin():
    assert pf.drive_wheel_contact_x(-3.935) == pytest.approx(-3.935)


def test_casters_airborne_under_lift():
    assert pf.caster_contact_z(MobileSpec(caster_lift_m=0.03)) == pytest.approx(0.03)


def test_cart_caster_local_geometry():
    locs = pf.cart_caster_local(MobileSpec())
    # four casters; wheel x offset -0.025 from the fork corner.
    assert len(locs) == 4
    xs = sorted({round(x, 3) for x, _ in locs})
    ys = sorted({round(y, 3) for _, y in locs})
    assert xs == [-0.325, 0.275]   # (-l/2+0.10)-0.025 ; (l/2-0.10)-0.025
    assert ys == [-0.26, 0.26]     # -w/2+0.08 ; w/2-0.08


# --- the defect / fix on the floor edges -----------------------------------

def test_triangle_floor_has_one_portal_band_edge_box_has_none():
    off = pf.edges_in_portal_band(colliders.TRIANGLE_MESH)
    on = pf.edges_in_portal_band(colliders.BOX)
    assert len(off) == 1          # cooked quad diagonal crosses the caster band
    assert on == []               # box floor: no internal edge in the band (seam gone)


def test_cooked_diagonal_is_v0_v2():
    # the fan-from-v0 split; the other diagonal (v1-v3) would NOT fit the measured casters.
    assert pf.cooked_diagonal() == (pf.BACKROOM_FLOOR_TOP_QUAD[0], pf.BACKROOM_FLOOR_TOP_QUAD[2])


# --- recorded-pin evidence (hardcoded cart poses; no external files) --------
# cart (x, y, yaw_deg) at the x=-3.935 pin, read from each run's nav-trace.
PINNED = {"launch1b": (-5.204, 15.223, 0.84),
          "coldwin18": (-5.205, 15.240, 0.48),
          "coldwin9": (-5.200, 15.145, 5.01)}
CROSSED = {"launch1a": (-5.206, 15.473, -1.18)}


def _front_left_perp(pose):
    an = pf.front_caster_edge_analysis((pose[0], pose[1], math.radians(pose[2])))
    return next(c for c in an if c.label == "front_left").perp_m


def test_pinned_runs_front_left_caster_on_diagonal():
    for run, pose in PINNED.items():
        perp = _front_left_perp(pose)
        assert abs(perp) < 0.05, f"{run}: front-left caster perp {perp} should be on the diagonal"


def test_crossing_run_front_left_caster_clear_of_diagonal():
    for run, pose in CROSSED.items():
        perp = _front_left_perp(pose)
        assert abs(perp) > 0.15, f"{run}: front-left caster perp {perp} should be clear of the diagonal"


# --- authored-stage round-trip (needs usd-core) ----------------------------

def test_authored_floor_approximation_round_trip():
    pytest.importorskip("pxr")
    from pxr import Usd, UsdGeom, UsdPhysics, Gf
    from twin import compose
    for flag, want in ((False, "none"), (True, "boundingCube")):
        stage = Usd.Stage.CreateInMemory()
        root = stage.DefinePrim("/Supermarket", "Xform").GetPath()
        stage.DefinePrim("/Supermarket/GRP_Structure", "Xform")
        stage.DefinePrim("/Supermarket/GRP_Structure/Backroom_floor", "Xform")
        mp = "/Supermarket/GRP_Structure/Backroom_floor/Backroom_floor"
        mesh = UsdGeom.Mesh.Define(stage, mp)
        mesh.CreatePointsAttr([Gf.Vec3f(x, y, 0.0) for x, y in pf.BACKROOM_FLOOR_TOP_QUAD])
        mesh.CreateFaceVertexCountsAttr([4])
        mesh.CreateFaceVertexIndicesAttr([0, 1, 2, 3])
        # a non-floor structure mesh must stay a triangle mesh under both settings.
        stage.DefinePrim("/Supermarket/GRP_Structure/Backroom_wall_east", "Xform")
        wp = "/Supermarket/GRP_Structure/Backroom_wall_east/Backroom_wall_east"
        wmesh = UsdGeom.Mesh.Define(stage, wp)
        wmesh.CreatePointsAttr([Gf.Vec3f(1, 11, 0), Gf.Vec3f(1.25, 11, 0), Gf.Vec3f(1.25, 19, 4)])
        wmesh.CreateFaceVertexCountsAttr([3]); wmesh.CreateFaceVertexIndicesAttr([0, 1, 2])
        compose.author_static_colliders(stage, root, flat_floor_box=flag)
        got = UsdPhysics.MeshCollisionAPI(stage.GetPrimAtPath(mp)).GetApproximationAttr().Get()
        assert got == want, f"flat_floor_box={flag}: floor approximation {got!r} != {want!r}"
        wgot = UsdPhysics.MeshCollisionAPI(stage.GetPrimAtPath(wp)).GetApproximationAttr().Get()
        assert wgot == "none", "wall must stay a triangle mesh regardless of the switch"
