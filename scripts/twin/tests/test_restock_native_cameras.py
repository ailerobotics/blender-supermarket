from types import SimpleNamespace

import numpy as np
import pytest

from twin.restock.native_cameras import RenderedCamera,camera_spec


def test_intrinsics_preserve_actual_offcentre_opencv_model():
    camera=RenderedCamera(SimpleNamespace(get_opencv_pinhole_properties=lambda:(307.,231.,601.,605.,[0.]*12)))
    assert np.array_equal(camera.get_intrinsics_matrix(),[[601,0,307],[0,605,231],[0,0,1]])


def test_distorted_depth_not_silently_treated_as_registered_pinhole():
    camera=RenderedCamera(SimpleNamespace(get_opencv_pinhole_properties=lambda:(320.,240.,600.,600.,[.1])))
    with pytest.raises(ValueError): camera.get_intrinsics_matrix()


def test_camera_requires_real_mount_geometry():
    spec=dict(id='source',parent='map',parent_from_optical=np.eye(4).tolist(),
              intrinsics=dict(width=640,height=480,fx=600.,fy=600.,cx=320.,cy=240.))
    with pytest.raises(ValueError,match='housing'): camera_spec(spec)
    spec['collision_solids']=[dict(size_m=[.05,.04,.03],parent_from_solid=np.eye(4).tolist())]
    assert camera_spec(spec)[0]=='source'
    spec['parent_from_optical'][0][0]=-1
    with pytest.raises(ValueError,match='rigid'): camera_spec(spec)


def test_native_identity_adaptation_keeps_original_clock_and_never_restamps():
    raw=dict(rendering_frame={'referenceTimeNumerator':112,'referenceTimeDenominator':30},
             rendering_time=1.48)
    original=SimpleNamespace(get_current_frame=lambda:raw,
        _restock_acquisition=dict(native_id=raw['rendering_frame'],acquired_steady=100.,
                                  session_id='commission',steady_domain='process-clock'))
    camera=RenderedCamera(original)
    first=camera.get_current_frame();second=camera.get_current_frame()
    assert first==second and first['rendering_frame']==112
    assert first['acquired_steady']==100. and first['rendering_time']==1.48
    assert isinstance(raw['rendering_frame'],dict)
    raw['rendering_frame']={'referenceTimeNumerator':224,'referenceTimeDenominator':60}
    with pytest.raises(ValueError,match='basis changed'): camera.get_current_frame()


def test_recreate_destroyed_reauthors_only_the_torn_down_camera_prim():
    # sim-warm-reset-stage-clean-v1: CameraRig.recreate_destroyed re-authors the <root>/Camera prim of a
    # camera whose render product was destroyed (released wrapper), leaving a live camera untouched. Uses a
    # real in-memory USD stage where pxr is available (the usd-core venv); the mount root persists (the
    # teardown removes only the Camera child + its render product), so only the Camera prim is re-Defined.
    pytest.importorskip('pxr')
    from pxr import Usd, UsdGeom
    from twin.restock.native_cameras import CameraRig

    stage = Usd.Stage.CreateInMemory()
    spec = dict(id='source', parent='map', parent_from_optical=np.eye(4).tolist(),
                intrinsics=dict(width=640, height=480, fx=600., fy=600., cx=320., cy=240.),
                collision_solids=[dict(size_m=[.05, .04, .03], parent_from_solid=np.eye(4).tolist())])
    rig = object.__new__(CameraRig)         # bypass __init__ (it imports isaacsim via initialize_camera)
    rig.stage = stage
    rig.frequency = -1
    rig.render_scale = 1.0
    rig.camera_render_scales = {}
    rig.specs = {'source': spec, 'shelf': spec}
    rig.optical_paths = {'source': '/World/restock_camera_source', 'shelf': '/World/restock_camera_shelf'}
    for root in rig.optical_paths.values():
        UsdGeom.Xform.Define(stage, root)   # mount root persists through the teardown
    # 'source' was torn down (released wrapper, no Camera prim); 'shelf' is live (Camera prim present).
    rig.cameras = {'source': SimpleNamespace(_render_product_released=True),
                   'shelf': SimpleNamespace(_render_product_released=False)}
    UsdGeom.Camera.Define(stage, '/World/restock_camera_shelf/Camera')

    made = []
    def _fake_init(path, k, *, name, frequency, resolution):
        made.append((path, name, resolution))
        return SimpleNamespace(prim_path=path, _render_product_released=False)

    restored = rig.recreate_destroyed(_init=_fake_init)
    assert restored == ['source']                                   # only the torn-down camera
    assert stage.GetPrimAtPath('/World/restock_camera_source/Camera').IsValid()  # prim re-authored
    assert [m[0] for m in made] == ['/World/restock_camera_source/Camera']
    assert rig.cameras['source']._render_product_released is False   # wrapper replaced
