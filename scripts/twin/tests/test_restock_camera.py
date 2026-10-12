"""Behavioral CPU fixtures only: no rendered depth, inventory or physics claim."""
import math
import numpy as np
import pytest

from scripts.twin.restock_camera import (
    Intrinsics, backproject, project, pose_matrix, look_at, relative_camera,
    calibrate_support)
from scripts.twin.restock_camera.geometry import (transform_points, validate_roi,
                                                  plane_roi, usd_camera_transform)
from scripts.twin.restock_camera.visibility import box_surface, inspect_view, ray_hits, usd_surfaces


def setup_plane(height=.2):
    k = Intrinsics(80,60,60.,60.,39.5,29.5)
    camera = look_at([0,-.15,1.],[0,0,.2])
    v,u = np.mgrid[:k.height,:k.width]
    rays = np.stack([(u-k.cx)/k.fx,(v-k.cy)/k.fy,np.ones_like(u)],axis=-1)
    world_rays = rays @ camera[:3,:3].T
    depth = (height-camera[2,3])/world_rays[:,:,2]
    return k,camera,depth


def calibration(**overrides):
    k,camera,depth = setup_plane()
    args = dict(location='rollwagen',now=10.,stamp=9.9,frame_id='render:40',
                epoch='park:2',parked=True,expected_support_z=.2,transform_stamp=9.9)
    args.update(overrides)
    return calibrate_support(depth,k,camera,(10,10,70,50),**args)


def test_projection_backprojection_and_range():
    k = Intrinsics(640,480,320.,320.,319.5,239.5)
    points = np.array([[0,0,1],[.4,.2,2],[-.2,-.1,.5]])
    uv,valid = project(points,k)
    assert valid.all()
    np.testing.assert_allclose(backproject(uv,points[:,2],k),points)
    _,valid = project([[0,0,-1],[0,0,.1],[0,0,4],[10,0,1]],k)
    assert not valid.any()


def test_independent_actual_articulated_arrival():
    robot = pose_matrix([-2.971381378328302,5.994060950255253,-3.0944697697763393],.5)
    cart = pose_matrix([-1.9221936604120067,6.51081802014273,-2.293410330750298])
    local = look_at([.08,-.02,.88],[.08,0,.3])
    relative = relative_camera(robot,cart @ local)
    points = [[0,0,.5],[.1,.1,.8]]
    np.testing.assert_allclose(transform_points(robot,transform_points(relative,points)),
                               transform_points(cart @ local,points),atol=1e-12)
    wrong = pose_matrix([-1.9221936604120067,6.51081802014273,-3.0944697697763393])
    assert np.linalg.norm(relative-relative_camera(robot,wrong @ local)) > 1
    assert abs(math.degrees(-3.0944697697763393+2.293410330750298)) == pytest.approx(45.8973249953)


@pytest.mark.parametrize('roi',[(-1,0,30,20),(0,0,81,20),(0,20,30,20),(0.,0,30,20)])
def test_roi_refusal(roi):
    with pytest.raises(ValueError):
        validate_roi(roi,setup_plane()[0])


@pytest.mark.parametrize('changes',[
    {'stamp':8.}, {'stamp':10.1}, {'parked':False}, {'transform_stamp':9.7},
    {'transform_stamp':None}, {'frame_id':''}, {'epoch':''}, {'expected_support_z':.04}])
def test_stale_moving_uncalibrated_refusal(changes):
    with pytest.raises(ValueError):
        calibration(**changes)


@pytest.mark.parametrize('value',[np.nan,np.inf,0.,-.1,.1,4.])
def test_missing_depth_cannot_prove_empty(value):
    k,camera,depth = setup_plane()
    depth[30,40] = value
    with pytest.raises(ValueError):
        calibrate_support(depth,k,camera,(10,10,70,50),location='cart',now=1.,stamp=1.,
            frame_id='1',epoch='park1',parked=True,expected_support_z=.2,transform_stamp=1.)


def test_plane_dimensions_and_slot_contract():
    result = calibration()
    assert result.support_z == pytest.approx(.2)
    assert result.residual_m < 1e-12
    assert all(.1 < value < 1. for value in result.observed_dimensions_xy)
    slot = result.slot_view(now=10.,epoch='park:2',parked=True,clearance_m=.3,
                            clearance_stamp=9.9,clearance_epoch='park:2')
    assert slot.location == 'rollwagen'
    assert slot.support_z == pytest.approx(.2)
    assert slot.free_dimensions == (*result.observed_dimensions_xy,.3)
    for changes in ({'now':11.},{'epoch':'park:3'},{'parked':False},
                    {'clearance_stamp':8.},{'clearance_epoch':'park:1'},{'clearance_m':np.nan}):
        args = dict(now=10.,epoch='park:2',parked=True,clearance_m=.3,
                    clearance_stamp=9.9,clearance_epoch='park:2')
        args.update(changes)
        with pytest.raises(ValueError):
            result.slot_view(**args)


def test_carton_top_is_not_support():
    k,camera,depth = setup_plane(.36)
    with pytest.raises(ValueError,match='elevated'):
        calibrate_support(depth,k,camera,(10,10,70,50),location='cart',now=1.,stamp=1.,
            frame_id='1',epoch='park1',parked=True,expected_support_z=.2,transform_stamp=1.)


def test_rail_occlusion_and_endpoint_surface():
    rail = box_surface('rail',[0,0,.5],[.02,1,.02])
    assert ray_hits([0,0,1],[0,0,0],rail.triangles)
    assert not ray_hits([.1,0,1],[.1,0,0],rail.triangles)
    support = box_surface('support',[0,0,-.1],[1,1,.2])
    assert not ray_hits([0,0,1],[0,0,0],support.triangles)
    camera = look_at([0,-.01,1],[0,0,0])
    report = inspect_view(camera,[[0,0,0]],setup_plane()[0],[rail])
    assert report['in_fov_and_range'] == 1
    assert report['visible_samples'] == 0
    assert report['occluders'] == ['rail']


def test_inscribed_roi_and_usd_axes():
    k,camera,_ = setup_plane()
    corners = [[-.2,-.15,.2],[.2,-.15,.2],[.2,.15,.2],[-.2,.15,.2]]
    roi = plane_roi(corners,camera,k)
    assert 0 < roi[0] < roi[2] < k.width
    assert 0 < roi[1] < roi[3] < k.height
    usd = usd_camera_transform(camera)
    np.testing.assert_allclose(usd[:3,:3] @ [0,0,-1],camera[:3,2])
    np.testing.assert_allclose(usd[:3,3],camera[:3,3])
    with pytest.raises(ValueError):
        plane_roi([corners[i] for i in [0,2,1,3]],camera,k)
    with pytest.raises(ValueError):
        plane_roi(np.asarray(corners)*100,camera,k)


def test_tilt_and_nonrigid_transform_rejected():
    k,camera,depth = setup_plane()
    depth[:,40:] -= .025
    args = dict(location='cart',now=1.,stamp=1.,frame_id='1',epoch='park1',
                parked=True,expected_support_z=.2,transform_stamp=1.)
    with pytest.raises(ValueError,match='plane'):
        calibrate_support(depth,k,camera,(10,10,70,50),**args)
    camera[0,0] *= 2
    with pytest.raises(ValueError,match='rigid'):
        calibrate_support(depth,k,camera,(10,10,70,50),**args)


def test_usd_disconnected_rails_keep_real_opening(tmp_path):
    pytest.importorskip('pxr')
    from pxr import Usd, UsdGeom
    path = tmp_path/'rails.usda'
    stage = Usd.Stage.CreateNew(str(path))
    for index,x in enumerate((-.3,.3)):
        cube = UsdGeom.Cube.Define(stage,f'/rail{index}')
        cube.CreateSizeAttr(1)
        cube.AddTranslateOp().Set((x,0,.5))
        cube.AddScaleOp().Set((.025,.4,.025))
    stage.GetRootLayer().Save()
    surfaces,layers = usd_surfaces(path,([-1,-1,0],[1,1,1]))
    assert len(surfaces) == 2
    assert str(path) in layers
    assert not any(ray_hits([0,0,1],[0,0,0],s.triangles) for s in surfaces)
    assert any(ray_hits([.3,0,1],[.3,0,0],s.triangles) for s in surfaces)


def test_usd_unsupported_occluder_is_not_silently_dropped(tmp_path):
    pytest.importorskip('pxr')
    from pxr import Usd, UsdGeom
    path = tmp_path/'sphere.usda'
    stage = Usd.Stage.CreateNew(str(path))
    UsdGeom.Sphere.Define(stage,'/sphere')
    stage.GetRootLayer().Save()
    with pytest.raises(ValueError,match='unsupported occluder'):
        usd_surfaces(path,([-2,-2,-2],[2,2,2]))
