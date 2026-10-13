"""CPU light-rig design and decodability fixtures; never a render or decode claim."""
import math

import numpy as np
import pytest

from scripts.twin.restock_camera.lights import (RectFixture, LightRigTuning, store_fixtures,
    task_fill_fixture, light_rig, tonemapping_settings, author_light_rig)
from scripts.twin.restock_camera.geometry import Intrinsics, look_at
from scripts.twin.restock_camera.visibility import marker_decodability


def test_store_fixtures_match_blender_grids_and_face_down():
    fx = store_fixtures()
    panels = [f for f in fx if f.kind == 'sales_panel']
    tubes = [f for f in fx if f.kind == 'backroom_tube']
    # build_lighting(): 9 x 7 ceiling panels; add_service_environment(): 3 x 3 tubes.
    assert len(panels) == 63 and len(tubes) == 9
    assert {round(f.center[0], 3) for f in panels} == {-12, -9, -6, -3, 0, 3, 6, 9, 12}
    assert {round(f.center[1], 3) for f in panels} == {-9.5, -6.5, -3.5, -0.5, 2.5, 5.5, 8.5}
    for f in panels:
        assert (f.width, f.height, f.power_w) == (1.2, 0.6, 45.0)
        assert round(f.center[2], 3) == 3.97 and f.world_from_light is None  # down-facing
    for f in tubes:
        assert (f.width, f.height, f.power_w) == (1.2, 0.3, 100.0)
        assert round(f.center[2], 3) == 3.85
        assert 13 <= f.center[1] <= 18.5 and f.center[0] in (-9, -5, -1)


def test_intensity_preserves_blender_exitance_ratio_and_scales_linearly():
    panel = next(f for f in store_fixtures() if f.kind == 'sales_panel')
    tube = next(f for f in store_fixtures() if f.kind == 'backroom_tube')
    # Lambertian radiance = power / (area * pi); appearance ratio is the exitance ratio.
    assert panel.radiance_w() == pytest.approx(45.0 / (0.72 * math.pi))
    assert tube.radiance_w() == pytest.approx(100.0 / (0.36 * math.pi))
    t1, t2 = LightRigTuning(intensity_scale=1.0), LightRigTuning(intensity_scale=2.0)
    assert t2.intensity_nits(panel) == pytest.approx(2 * t1.intensity_nits(panel))
    assert (t1.intensity_nits(tube) / t1.intensity_nits(panel)
            == pytest.approx(tube.radiance_w() / panel.radiance_w()))


def test_task_fill_emits_toward_its_target():
    eye, target = np.array([-3.55, 4.826, 1.0]), np.array([-3.47, 4.826, 0.73])
    fill = task_fill_fixture('fill:shelf', eye, target)
    assert fill.kind == 'task_fill'
    pose = np.array(fill.world_from_light)
    emission = pose[:3, :3] @ [0, 0, -1]          # UsdLux emits along local -Z
    want = (target - eye) / np.linalg.norm(target - eye)
    np.testing.assert_allclose(emission, want, atol=1e-9)


def test_light_rig_is_design_only_with_dome_and_tonemapping():
    rig = light_rig(LightRigTuning())
    assert rig['runtime_ready'] is False
    assert rig['evidence'].startswith('physical_fixture_design')
    assert rig['counts'] == dict(sales_panel=63, backroom_tube=9, task_fill=0)
    assert rig['dome']['enabled'] is True and rig['dome']['intensity_nits'] > 0
    tm = rig['tonemapping']
    assert tm['blender_reference']['view_transform'] == 'AgX'
    assert tm['isaac_rtx_recommendation']['auto_exposure'] is False


def test_tonemapping_tracks_intensity_scale():
    assert tonemapping_settings(LightRigTuning(intensity_scale=3.0))['intensity_scale'] == 3.0


def test_authoring_places_down_facing_rectlights_and_dome(tmp_path):
    pytest.importorskip('pxr')
    from pxr import Usd, UsdGeom, UsdLux
    rig = light_rig(LightRigTuning(), fills=[task_fill_fixture(
        'fill:shelf', [-3.55, 4.826, 1.0], [-3.47, 4.826, 0.73])])
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.Xform.Define(stage, '/World')
    paths = author_light_rig(stage, rig, parent_path='/World')
    assert 'dome_fill' in paths and 'fill:shelf' in paths
    assert len(paths) == 63 + 9 + 1 + 1
    panel = next(f for f in store_fixtures() if f.kind == 'sales_panel')
    prim = stage.GetPrimAtPath(paths[panel.id])
    rect = UsdLux.RectLight(prim)
    assert rect.GetWidthAttr().Get() == pytest.approx(1.2)
    assert rect.GetIntensityAttr().Get() == pytest.approx(LightRigTuning().intensity_nits(panel))
    assert rect.GetNormalizeAttr().Get() is False
    matrix = np.array(UsdGeom.Xformable(prim).GetLocalTransformation()).T
    np.testing.assert_allclose(matrix[:3, :3], np.eye(3), atol=1e-9)   # emits world -Z
    np.testing.assert_allclose(matrix[:3, 3], panel.center, atol=1e-6)
    # Idempotent re-author does not duplicate prims.
    before = len(list(stage.Traverse()))
    author_light_rig(stage, rig, parent_path='/World')
    assert len(list(stage.Traverse())) == before


def test_authoring_requires_existing_parent():
    pytest.importorskip('pxr')
    from pxr import Usd, UsdGeom
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.Xform.Define(stage, '/World')
    with pytest.raises(ValueError):
        author_light_rig(stage, light_rig(), parent_path='/Missing')


def test_retune_overrides_authored_intensity(tmp_path):
    pytest.importorskip('pxr')
    from pxr import Usd, UsdGeom, UsdLux
    rig = light_rig(LightRigTuning(intensity_scale=1.0))
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.Xform.Define(stage, '/World')
    paths = author_light_rig(stage, rig, parent_path='/World',
                             tuning=LightRigTuning(intensity_scale=4.0))
    panel = next(f for f in store_fixtures() if f.kind == 'sales_panel')
    rect = UsdLux.RectLight(stage.GetPrimAtPath(paths[panel.id]))
    assert rect.GetIntensityAttr().Get() == pytest.approx(
        LightRigTuning(intensity_scale=4.0).intensity_nits(panel))


def _square(center, side):
    cx, cy, cz = center
    h = side / 2
    return np.array([[cx - h, cy - h, cz], [cx + h, cy - h, cz],
                     [cx + h, cy + h, cz], [cx - h, cy + h, cz]])


def test_decodability_pixel_size_scales_with_focal_length():
    k1 = Intrinsics(2560, 1920, 1280., 1280., 1280., 960., .15, 3.5)
    k2 = Intrinsics(2560, 1920, 2560., 2560., 1280., 960., .15, 3.5)
    cam = look_at([0.001, 0, 0.8], [0, 0, 0])   # tiny offset: a pure vertical look_at is rejected
    corners = _square([0, 0, 0], 0.06)
    d1 = marker_decodability(cam, corners, k1, 25)
    d2 = marker_decodability(cam, corners, k2, 25)
    assert d1['module_count'] == 25
    assert d2['min_edge_px'] == pytest.approx(2 * d1['min_edge_px'], rel=1e-6)
    assert d2['px_per_module'] == pytest.approx(d1['min_edge_px'] * 2 / 25, rel=1e-6)


def test_decodability_incidence_grows_with_obliquity():
    k = Intrinsics(2560, 1920, 1280., 1280., 1280., 960., .15, 3.5)
    corners = _square([0, 0, 0], 0.06)   # marker normal is +Z
    frontal = marker_decodability(look_at([0.001, 0, 0.8], [0, 0, 0]), corners, k, 25)
    oblique = marker_decodability(look_at([0.5, 0, 0.5], [0, 0, 0]), corners, k, 25)
    assert frontal['incidence_deg'] < 5
    assert oblique['incidence_deg'] > 40


def test_decodability_flags_out_of_frame_marker():
    k = Intrinsics(2560, 1920, 1280., 1280., 1280., 960., .15, 3.5)
    corners = _square([2.0, 0, 0], 0.06)   # far to the side, outside the frustum
    d = marker_decodability(look_at([0.001, 0, 0.8], [0, 0, 0]), corners, k, 25)
    assert d['in_fov_and_range'] is False


def test_decodability_rejects_bad_inputs():
    k = Intrinsics(2560, 1920, 1280., 1280., 1280., 960., .15, 3.5)
    cam = look_at([0.001, 0, 0.8], [0, 0, 0])
    with pytest.raises(ValueError):
        marker_decodability(cam, _square([0, 0, 0], 0.06)[:3], k, 25)
    with pytest.raises(ValueError):
        marker_decodability(cam, _square([0, 0, 0], 0.06), k, 0)


# --- camera-mounted lamp (recessed shelf support) ---------------------------------
from scripts.twin.restock_camera.lights import (camera_lamp, sphere_illuminance,
                                                rect_illuminance, aimed_illuminance,
                                                lamp_illuminance, author_camera_lamps)
from scripts.twin.restock_camera.config import camera_config
import math as _math


def _shelf_config():
    scenario = dict(handling_unit=dict(local_wall=3, grasp_mode='side', dimensions_m=[.6, .4, .16]),
        initial_staging=dict(support_center_xy=[-5.73, 17.1], unit_bottom_z=.6,
            target_surface_pose_world=[-3.05, 4.826, .73, 0], cart_load_center_body_xy=[.08, 0],
            cart_support_z_world=.225))
    return camera_config(scenario, dict(spec=dict(arm_mount_z=.5)))


def test_camera_lamp_is_an_aimed_disk_sharing_rig_tuning():
    aim = [-3.47, 4.826, 0.73]
    lamp1 = camera_lamp('shelf', 'map', [-3.55, 4.826, 0.90], aim_at=aim, radius=.06, power_w=8.,
                        tuning=LightRigTuning(intensity_scale=1.0))
    lamp2 = camera_lamp('shelf', 'map', [-3.55, 4.826, 0.90], aim_at=aim, radius=.06, power_w=8.,
                        tuning=LightRigTuning(intensity_scale=5.0))
    assert lamp1['kind'] == 'camera_lamp' and lamp1['type'] == 'disk'
    assert lamp1['radiance_w'] == pytest.approx(8. / (_math.pi * .06 ** 2 * _math.pi))
    assert lamp2['intensity_nits'] == pytest.approx(5 * lamp1['intensity_nits'])
    # emission axis (-Z of the transform, stored as emit_normal) points at the aim.
    want = np.array(aim) - np.array([-3.55, 4.826, 0.90]); want /= np.linalg.norm(want)
    np.testing.assert_allclose(lamp1['emit_normal'], want, atol=1e-9)


def test_aimed_disk_lamp_needs_an_aim():
    with pytest.raises(ValueError):
        camera_lamp('shelf', 'map', [-3.55, 4.826, 0.90], radius=.06, power_w=8.)


def test_sphere_illuminance_inverse_square_and_cosine():
    up = np.array([0, 0, 1.0])
    e_near = sphere_illuminance([0, 0, 1.0], .03, [0, 0, 0], up, 1000.)
    e_far = sphere_illuminance([0, 0, 2.0], .03, [0, 0, 0], up, 1000.)
    assert e_near == pytest.approx(4 * e_far, rel=1e-6)          # half distance -> 4x
    straight = sphere_illuminance([0, 0, 1.0], .03, [0, 0, 0], up, 1000.)
    tilted = sphere_illuminance([1.0, 0, 1.0], .03, [0, 0, 0], up, 1000.)
    assert tilted < straight                                     # off-normal gets less
    assert sphere_illuminance([0, 0, -1.0], .03, [0, 0, 0], up, 1000.) == 0.0  # below surface


def test_aimed_illuminance_inverse_square_and_emitter_cosine():
    up = np.array([0, 0, 1.0]); down = np.array([0, 0, -1.0])  # emitter faces -Z (down)
    e_near = aimed_illuminance([0, 0, 1.0], .01, down, [0, 0, 0], up, 1000.)
    e_far = aimed_illuminance([0, 0, 2.0], .01, down, [0, 0, 0], up, 1000.)
    assert e_near == pytest.approx(4 * e_far, rel=1e-6)          # half distance -> 4x
    # emitter aimed away from the point delivers nothing
    assert aimed_illuminance([0, 0, 1.0], .01, up, [0, 0, 0], up, 1000.) == 0.0


def test_rect_illuminance_zero_when_point_above_down_emitter():
    panel = next(f for f in store_fixtures() if f.kind == 'sales_panel')
    up = np.array([0, 0, 1.0])
    below = rect_illuminance(panel, [panel.center[0], panel.center[1], panel.center[2] - 1], up, 3000.)
    above = rect_illuminance(panel, [panel.center[0], panel.center[1], panel.center[2] + 1], up, 3000.)
    assert below > 0 and above == 0.0                           # -Z emitter lights only below


def test_config_carries_shelf_camera_lamp_and_collidable_housing():
    cfg = _shelf_config()
    assert len(cfg['lamps']) == 1
    lamp = cfg['lamps'][0]
    assert lamp['id'] == 'shelf' and lamp['parent'] == 'map' and lamp['type'] == 'disk'
    np.testing.assert_allclose(lamp['center'], [-3.55, 4.826, 0.90])
    assert lamp['center'][2] < 0.974                            # below the shelf tier plane
    housing = next(s for s in cfg['solids'] if s['id'] == 'shelf_lamp_housing')
    assert housing['collision'] is True


def test_shelf_lamp_outilluminates_the_source_reference():
    # Source markers decode 3/3 at ~1007 lux (scale 1) from the backroom tubes; the shelf
    # disk lamp alone must exceed that on the recessed shelf markers.
    cfg = _shelf_config()
    lamp = cfg['lamps'][0]
    up = np.array([0, 0, 1.0])
    for marker in cfg['supports']['shelf']['markers']:
        pt = np.array(marker['parent_from_marker'])[:3, 3]
        e = lamp_illuminance(lamp, pt, up)
        assert e > 1007 * 2                                      # >= 2x the decoding source level


def test_author_camera_lamps_places_aimed_disk_light(tmp_path):
    pytest.importorskip('pxr')
    from pxr import Usd, UsdGeom, UsdLux
    cfg = _shelf_config()
    lamp = cfg['lamps'][0]
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.Xform.Define(stage, '/World')
    paths = author_camera_lamps(stage, cfg, parent_paths={'map': '/World', 'cart_chassis': '/World'})
    assert set(paths) == {'shelf'}
    prim = stage.GetPrimAtPath(paths['shelf'])
    assert prim.GetTypeName() == 'DiskLight'
    dl = UsdLux.DiskLight(prim)
    assert dl.GetRadiusAttr().Get() == pytest.approx(lamp['radius'])
    assert dl.GetIntensityAttr().Get() == pytest.approx(lamp['intensity_nits'])
    matrix = np.array(UsdGeom.Xformable(prim).GetLocalTransformation()).T
    np.testing.assert_allclose(matrix[:3, 3], lamp['center'], atol=1e-6)
    emission = matrix[:3, :3] @ [0, 0, -1]                        # -Z points at the aim
    want = np.array(lamp['aim_at']) - np.array(lamp['center']); want /= np.linalg.norm(want)
    np.testing.assert_allclose(emission, want, atol=1e-6)
    before = len(list(stage.Traverse()))
    author_camera_lamps(stage, cfg, parent_paths={'map': '/World', 'cart_chassis': '/World'})
    assert len(list(stage.Traverse())) == before                # idempotent
    retuned = author_camera_lamps(stage, cfg, parent_paths={'map': '/World'},
                                  tuning=LightRigTuning(intensity_scale=3.0))
    dl2 = UsdLux.DiskLight(stage.GetPrimAtPath(retuned['shelf']))
    assert dl2.GetIntensityAttr().Get() == pytest.approx(3 * lamp['intensity_nits'])


from scripts.twin.restock_camera.lights import lamp_marker_occlusion
from scripts.twin.restock_camera.visibility import box_surface, Surface
from scripts.twin.restock_camera.geometry import pose_matrix as _pose


def test_lamp_marker_occlusion_flags_blocker_on_the_segment():
    emitter = np.array([0, 0, 1.0])
    marker = np.array([0, 0, 0.0])
    wall = box_surface('wall', [0, 0, 0.5], [1.0, 1.0, 0.02])     # straddles the segment
    beside = box_surface('beside', [0.5, 0, 0.5], [0.1, 0.1, 0.02])  # off to the side
    assert lamp_marker_occlusion(emitter, [marker], [wall]) == {0: ['wall']}
    assert lamp_marker_occlusion(emitter, [marker], [beside]) == {}


def test_configured_shelf_lamp_is_not_blocked_by_its_own_or_camera_housing():
    # Reproduces the native failure fix: the earlier emitter co-located with the lamp
    # housing was occluded by it; the configured emitter below the lens must clear both
    # the lamp housing and the camera housing (shelf tiers/products need USD and are
    # covered by plan_restock_lighting.py).
    cfg = _shelf_config()
    lamp = cfg['lamps'][0]
    housings = [box_surface(s['id'], [0, 0, 0], s['dimensions_m'],
                            transform=np.array(s['parent_from_solid']))
                for s in cfg['solids'] if s['id'].endswith('_housing')]
    markers = [np.array(m['parent_from_marker'])[:3, 3] + [0, 0, .001]
               for m in cfg['supports']['shelf']['markers']]
    assert lamp_marker_occlusion(np.array(lamp['center']), markers, housings) == {}


def test_emitter_at_old_above_tier_position_is_occluded_by_housing():
    # Guard against regressing to the native-failure placement (emitter inside its body).
    cfg = _shelf_config()
    body = next(s for s in cfg['solids'] if s['id'] == 'shelf_lamp_housing')
    bad_emitter = np.array(body['parent_from_solid'])[:3, 3]      # emitter inside the housing
    housing = box_surface('shelf_lamp_housing', [0, 0, 0], body['dimensions_m'],
                          transform=np.array(body['parent_from_solid']))
    markers = [np.array(m['parent_from_marker'])[:3, 3] + [0, 0, .001]
               for m in cfg['supports']['shelf']['markers']]
    assert lamp_marker_occlusion(bad_emitter, markers, [housing]) != {}
