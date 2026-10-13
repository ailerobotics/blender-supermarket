"""The diagnostic variant must change exactly one collision participation flag."""
import pytest
pxr=pytest.importorskip('pxr')
from pxr import Usd,UsdGeom,UsdPhysics
from twin.restock.cart_diagnostic import GROUND,FLOOR,select_floor


def test_only_lower_collision_changes():
    stage=Usd.Stage.CreateInMemory()
    for path in (GROUND,FLOOR):
        prim=UsdGeom.Cube.Define(stage,path).GetPrim()
        UsdPhysics.CollisionAPI.Apply(prim).CreateCollisionEnabledAttr(True)
    before=stage.GetRootLayer().ExportToString()
    select_floor(stage,'baseline')
    assert stage.GetRootLayer().ExportToString()==before
    select_floor(stage,'without_lower_ground')
    assert not UsdPhysics.CollisionAPI(stage.GetPrimAtPath(GROUND)).GetCollisionEnabledAttr().Get()
    assert UsdPhysics.CollisionAPI(stage.GetPrimAtPath(FLOOR)).GetCollisionEnabledAttr().Get()
    UsdPhysics.CollisionAPI(stage.GetPrimAtPath(GROUND)).CreateCollisionEnabledAttr(True)
    assert stage.GetRootLayer().ExportToString()==before


def test_absent_support_and_unknown_variant_refused():
    stage=Usd.Stage.CreateInMemory()
    with pytest.raises(ValueError,match='unknown'):
        select_floor(stage,'other')
    with pytest.raises(ValueError,match='absent'):
        select_floor(stage,'without_lower_ground')
