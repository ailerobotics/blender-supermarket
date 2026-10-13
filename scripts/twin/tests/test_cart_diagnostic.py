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


def test_passive_tensor_snapshot_copies_backend_buffers_without_writes():
    import numpy as np
    from twin.restock.cart_diagnostic import read_tensors
    class View:
        prim_paths=['/Rollwagen/chassis']
        transforms=np.array([[1.,2.,3.,0.,0.,0.,1.]])
        velocities=np.zeros((1,6))
        def get_transforms(self): return self.transforms
        def get_velocities(self): return self.velocities
        def __getattr__(self,name): raise AssertionError('unexpected tensor operation: '+name)
    view=View()
    snapshot=read_tensors(view)
    view.transforms[0,0]=9.
    view.velocities[0,0]=4.
    assert snapshot['/Rollwagen/chassis']['get_transforms'][0]==1.
    assert snapshot['/Rollwagen/chassis']['get_velocities'][0]==0.


def test_passive_snapshot_rejects_nonfinite_physics_state():
    import numpy as np
    from types import SimpleNamespace
    from twin.restock.cart_diagnostic import read_tensors
    view=SimpleNamespace(prim_paths=['/Rollwagen/chassis'],
        get_transforms=lambda:np.full((1,7),np.nan))
    with pytest.raises(ValueError,match='invalid passive'):
        read_tensors(view)
