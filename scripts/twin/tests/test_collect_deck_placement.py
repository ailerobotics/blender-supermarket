"""fm389: the collect placement Target must land on the cart DECK interior, derived from the live
support prim, NOT at the scenario ``support_z_world`` constant (which aimed the carton at the upper
shelf, 0.71 m above the deck and 0.08 m east of centre -- run-300e released the carton off the east
edge to the floor).

Builds a synthetic stage reproducing run-300e/mission.usda's MEASURED geometry (usd-core, no Isaac):
  cart /Rollwagen origin (-5.73, 15.25, 0), yaw 0
  deck  /Rollwagen/chassis/deck  world AABB x[-6.13,-5.33] y[14.91,15.59] z[0.1749,0.225]  (top 0.225)
  shelf /Rollwagen/chassis/shelf world AABB z top 0.9325   (the WRONG legacy target surface)
  arm base ur_base_link at (-5.73, 16.0, 0.5), yaw 0
and proves both the DECK-mode and the byte-identical legacy path through the real bridge code.
"""
import math

import numpy as np
import pytest

pytest.importorskip('pxr')
from pxr import Usd, UsdGeom, Gf  # noqa: E402

from scripts.twin.restock_manipulation.isaac_factory import CollectTargetBridge  # noqa: E402
from scripts.twin.restock_manipulation.types import ProductProfile  # noqa: E402


class NS:
    def __init__(self, **kw):
        self.__dict__.update(kw)


def _box(stage, path, center, half):
    """A Cube (world AABB = center +/- half) under an Xform chain, so BBoxCache sees real geometry."""
    cube = UsdGeom.Cube.Define(stage, path)
    cube.GetSizeAttr().Set(2.0)                       # spans [-1,1] before scale
    x = UsdGeom.Xformable(cube.GetPrim())
    x.AddTranslateOp().Set(Gf.Vec3d(*center))
    x.AddScaleOp().Set(Gf.Vec3f(*half))
    return cube


def _stage():
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    # cart root at the measured staged pose (yaw 0)
    cart = UsdGeom.Xform.Define(stage, '/Rollwagen')
    UsdGeom.Xformable(cart.GetPrim()).AddTranslateOp().Set(Gf.Vec3d(-5.73, 15.25, 0.0))
    UsdGeom.Xform.Define(stage, '/Rollwagen/chassis')
    # deck/shelf centres are CART-LOCAL (the cart root already carries the -5.73,15.25 translate): deck
    # local (0,0,0.19995) half (0.4,0.34,0.02505) -> world AABB x[-6.13,-5.33] y[14.91,15.59] top z 0.225.
    _box(stage, '/Rollwagen/chassis/deck', (0.0, 0.0, 0.19995), (0.4, 0.34, 0.02505))
    # shelf: the legacy target surface (world top 0.9325) -- present so the test is honest about the bug
    _box(stage, '/Rollwagen/chassis/shelf', (0.0, 0.0, 0.92), (0.38, 0.32, 0.0125))
    base = UsdGeom.Xform.Define(stage, '/World/ur_base_link')
    UsdGeom.Xformable(base.GetPrim()).AddTranslateOp().Set(Gf.Vec3d(-5.73, 16.0, 0.5))
    return stage


def _bridge(stage, cart_support_path):
    cp = {'location': 'rollwagen', 'load_center_body_xy': (0.08, 0.0),
          'support_z_world': 0.9325, 'free_dimensions': (0.65, 0.45, 0.3)}
    return CollectTargetBridge(
        stage=stage, arm_camera=NS(), arm_camera_path='/arm', base_path='/World/ur_base_link',
        profiles={'juice_orange': ProductProfile(sku='juice_orange', dimensions=(0.6, 0.4, 0.16),
                  tcp_quaternion=(0., 0., 0., 1.))},
        motion=NS(parked=lambda: True, collision_time=9.0), contact=NS(read=lambda: False),
        joints=lambda: (0.,) * 6, source_location='backstock_00',
        cart_placement=cp, cart_root='/Rollwagen', cart_support_path=cart_support_path)


def _world_target(stage, bridge):
    """The placement load-centre, returned in ur_base_link, mapped back to WORLD for the assertions."""
    res = bridge._cart_pose_station()
    assert res is not None
    (px, py, pz), _yaw = res
    cache = UsdGeom.XformCache()
    Tub = np.asarray(cache.GetLocalToWorldTransform(stage.GetPrimAtPath('/World/ur_base_link'))).T
    w = Tub @ np.array([px, py, pz, 1.0])
    return float(w[0]), float(w[1]), float(w[2])


def test_deck_mode_targets_the_deck_interior_not_the_shelf():
    stage = _stage()
    wx, wy, wz = _world_target(stage, _bridge(stage, '/Rollwagen/chassis/deck'))
    # z on the DECK top (0.225), NOT the shelf (0.9325): ~0.71 m lower than the legacy constant
    assert wz == pytest.approx(0.225, abs=1e-3)
    assert abs(wz - 0.9325) > 0.5
    # centred on the deck (no +0.08 east bias): the 0.65-wide carton fits inside the 0.8-wide deck
    assert wx == pytest.approx(-5.73, abs=1e-3)
    assert wy == pytest.approx(15.25, abs=1e-3)
    # carton footprint (0.65 x 0.45) strictly inside the deck AABB (x[-6.13,-5.33], y[14.91,15.59])
    assert -6.13 <= wx - 0.65 / 2 and wx + 0.65 / 2 <= -5.33
    assert 14.91 <= wy - 0.45 / 2 and wy + 0.45 / 2 <= 15.59


def test_legacy_constant_path_is_unchanged_byte_identical():
    stage = _stage()
    wx, wy, wz = _world_target(stage, _bridge(stage, None))   # cart_support_path None => legacy
    # cart origin + load_center_body_xy(0.08,0) at support_z_world(0.9325) -> the OLD run-300e target
    assert (wx, wy, wz) == pytest.approx((-5.65, 15.25, 0.9325), abs=1e-6)


def test_named_offset_legacy_minus_deck():
    """Documents the fm389 offset: legacy target was +0.08 east and +0.7075 above the deck interior."""
    stage = _stage()
    lx, ly, lz = _world_target(stage, _bridge(stage, None))
    dx, dy, dz = _world_target(stage, _bridge(stage, '/Rollwagen/chassis/deck'))
    assert (lx - dx, ly - dy, lz - dz) == pytest.approx((0.08, 0.0, 0.7075), abs=1e-3)
