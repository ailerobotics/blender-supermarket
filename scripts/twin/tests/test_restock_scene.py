"""Synthetic USD fixtures prove authoring and input immutability, not dynamics."""
from pathlib import Path
import hashlib
import sys
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
pytest.importorskip('pxr')
from pxr import Usd, UsdGeom, UsdPhysics
from twin.restock.scene import build, promote_stock


def test_promote_mesh_overrides_static_collider_and_requires_mass():
    stage=Usd.Stage.CreateInMemory()
    mesh=UsdGeom.Mesh.Define(stage,'/Case')
    UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())
    UsdPhysics.MeshCollisionAPI.Apply(mesh.GetPrim()).CreateApproximationAttr().Set('none')
    with pytest.raises(ValueError,match='mass'):
        promote_stock(stage,'/Case','milk',0)
    promote_stock(stage,'/Case','milk',2.0)
    assert UsdPhysics.RigidBodyAPI(mesh.GetPrim()).GetRigidBodyEnabledAttr().Get()
    assert UsdPhysics.MeshCollisionAPI(mesh.GetPrim()).GetApproximationAttr().Get()=='convexHull'
    assert UsdPhysics.MassAPI(mesh.GetPrim()).GetMassAttr().Get()==2.0


def test_scene_keeps_sources_read_only_and_uses_distinct_body_poses(tmp_path):
    source=tmp_path/'store.usda'
    robot=tmp_path/'fixture_robot.usda'
    out=tmp_path/'composed.usda'
    stage=Usd.Stage.CreateNew(str(source))
    root=UsdGeom.Xform.Define(stage,'/Supermarket')
    stage.SetDefaultPrim(root.GetPrim())
    UsdGeom.Mesh.Define(stage,'/Supermarket/GRP_Fixtures/Case')
    stage.GetRootLayer().Save()
    arm=Usd.Stage.CreateNew(str(robot))
    root=UsdGeom.Xform.Define(arm,'/FixtureRobot')
    arm.SetDefaultPrim(root.GetPrim())
    UsdPhysics.ArticulationRootAPI.Apply(root.GetPrim())
    arm.GetRootLayer().Save()
    before={p:hashlib.sha256(p.read_bytes()).hexdigest() for p in (source,robot)}
    result=build(source,robot,out,robot_pose=(0,0,0),cart_pose=(-1.27,0,0),
                 stock_path='/Supermarket/GRP_Fixtures/Case',sku='milk',stock_mass_kg=2.)
    assert result['physics_verified'] is False
    assert all(hashlib.sha256(p.read_bytes()).hexdigest()==digest for p,digest in before.items())
    composed=Usd.Stage.Open(str(out))
    assert UsdGeom.Xformable(composed.GetPrimAtPath('/MobileRobot')).GetOrderedXformOps()[0].Get()[2]==.24
    assert UsdGeom.Xformable(composed.GetPrimAtPath('/Rollwagen')).GetOrderedXformOps()[0].Get()[0]==-1.27
    with pytest.raises(ValueError,match='new scratch'):
        build(source,robot,out,robot_pose=(0,0,0),cart_pose=(-1.27,0,0),
              stock_path='/Supermarket/GRP_Fixtures/Case',sku='milk',stock_mass_kg=2.)


def test_physical_qr_label_decodes_without_semantic_metadata(tmp_path):
    pytest.importorskip('qrcode')
    cv2=pytest.importorskip('cv2')
    from twin.restock.scene import label_stock
    stage=Usd.Stage.CreateInMemory()
    mesh=UsdGeom.Mesh.Define(stage,'/Case')
    mesh.CreatePointsAttr([(x,y,z) for x in (-.3,.3) for y in (-.2,.2) for z in (0,.32)])
    texture=tmp_path/'label.png'
    path=label_stock(stage,'/Case','biscuit_choc',texture)
    decoded,_,_=cv2.QRCodeDetector().detectAndDecode(cv2.imread(str(texture)))
    assert decoded=='biscuit_choc'
    assert path=='/Case/RestockLabel'
    assert not stage.GetPrimAtPath(path).HasAPI(UsdPhysics.CollisionAPI)
    with pytest.raises(ValueError,match='must be new'):
        label_stock(stage,'/Case','biscuit_choc',texture)


def test_initial_scenario_depletes_only_exact_product_slot_and_refuses_replay(tmp_path):
    import json
    from pxr import Sdf
    from twin.restock.scene import author_initial_scenario
    scenario=json.loads((Path(__file__).resolve().parents[2]/'restock_scenario.json').read_text())
    source=tmp_path/'source.usda'
    original=Usd.Stage.CreateNew(str(source))
    paths=['/Supermarket/GRP_Products/Target','/Supermarket/GRP_PriceTags/Tag',
           '/Supermarket/GRP_Fixtures/Shelf','/Supermarket/GRP_Products/Other']
    for path in paths:
        prim=UsdGeom.Cube.Define(original,path).GetPrim()
        prim.CreateAttribute('userProperties:slot',Sdf.ValueTypeNames.String).Set(
            'other' if path.endswith('Other') else scenario['job']['shelf_id'])
    original.GetRootLayer().Save()
    before=source.read_bytes()
    stage=Usd.Stage.CreateInMemory();stage.GetRootLayer().subLayerPaths.append(str(source))
    result=author_initial_scenario(stage,scenario)
    assert result['depleted_product_prims']==[paths[0]]
    assert not stage.GetPrimAtPath(paths[0]).IsActive()
    assert all(stage.GetPrimAtPath(path).IsActive() for path in paths[1:])
    assert source.read_bytes()==before
    assert UsdPhysics.RigidBodyAPI(stage.GetPrimAtPath(result['stock'])).GetRigidBodyEnabledAttr().Get()
    assert not result['hidden_contents_counted']
    with pytest.raises(ValueError,match='already authored'):
        author_initial_scenario(stage,scenario)
