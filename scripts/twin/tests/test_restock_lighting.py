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
