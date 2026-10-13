"""CPU design/authoring fixtures, not renderer or physics evidence."""
import numpy as np
import pytest

from scripts.twin.restock_camera.config import camera_config, author_camera_design
from scripts.twin.restock_camera.geometry import pose_matrix, transform_points, usd_camera_transform


def design():
    scenario = dict(handling_unit=dict(local_wall=3,grasp_mode='side',dimensions_m=[.6,.4,.16]),
        initial_staging=dict(support_center_xy=[-5.73,17.1],unit_bottom_z=.6,
            target_surface_pose_world=[-3.05,4.826,.73,0],cart_load_center_body_xy=[.08,0],
            cart_support_z_world=.225))
    return camera_config(scenario,dict(spec=dict(arm_mount_z=.5)))


def test_current_measured_margin_arrival_uses_independent_chassis_frame():
    config = design()
    intr = config['cameras']['cart']['intrinsics']
    assert (intr['cx'],intr['cy']) == (intr['width']/2,intr['height']/2)
    camera = np.array(config['cameras']['cart']['parent_from_optical'])
    np.testing.assert_allclose(camera[:3,3],[.08,-.02,.68])
    robot = pose_matrix([-2.971786444702803,5.992828646700736,-3.0963991557288555],.5)
    cart = pose_matrix([-1.9217970823989616,6.507731608263511,-2.294942380545212],.2)
    base_from_optical = np.linalg.inv(robot) @ cart @ camera
    np.testing.assert_allclose(robot @ base_from_optical,cart @ camera,atol=1e-12)
    aligned = pose_matrix([-1.9217970823989616,6.507731608263511,-3.0963991557288555],.2)
    assert np.linalg.norm(base_from_optical-np.linalg.inv(robot) @ aligned @ camera) > 1.
    assert all(s['collision'] for s in config['solids'])
    assert config['runtime_ready'] is False


def test_cart_markers_and_witness_are_outside_loaded_unit():
    support = design()['supports']['cart']
    # Unit spans x [-.12,.28], y [-.30,.30] in the ground/cart chassis XY.
    for marker in support['markers']:
        center = np.array(marker['parent_from_marker'])[:3,3]
        assert center[0]+marker['side_m']/2 < -.12
        assert center[0]-marker['side_m']/2 > -.4
    assert support['required_xy_m'] == [.62,.42]


def test_authoring_retains_physical_mounts_and_metric_printed_markers(tmp_path):
    pytest.importorskip('pxr')
    cv2 = pytest.importorskip('cv2')
    from pxr import Usd, UsdGeom, UsdPhysics
    from scripts.twin.restock_manipulation.isaac_sensors import write_carton_label
    config = design()
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.Xform.Define(stage,'/World')
    UsdGeom.Xform.Define(stage,'/Cart')
    textures = {}
    for support in config['supports'].values():
        for marker in support['markers']:
            path = tmp_path/(marker['id'].replace(':','_')+'.png')
            write_carton_label(marker['id'],path)
            textures[marker['id']] = path
    paths = author_camera_design(stage,config,parent_paths={'map':'/World','cart_chassis':'/Cart'},
                                 marker_textures=textures)
    assert set(paths) == {'source','cart','shelf'}
    for name,path in paths.items():
        matrix = np.array(UsdGeom.Xformable(stage.GetPrimAtPath(path)).GetLocalTransformation()).T
        np.testing.assert_allclose(matrix,usd_camera_transform(config['cameras'][name]['parent_from_optical']))
    for solid in config['solids']:
        parent = '/Cart' if solid['parent']=='cart_chassis' else '/World'
        assert stage.GetPrimAtPath(parent+'/RestockCamera_'+solid['id']).HasAPI(UsdPhysics.CollisionAPI)
    mesh = UsdGeom.Mesh(stage.GetPrimAtPath('/Cart/RestockCamera_cart_marker_0'))
    points = np.array(mesh.GetPointsAttr().Get())
    texture = cv2.imread(str(textures['support:cart:0']))
    code,corners,_ = cv2.QRCodeDetector().detectAndDecode(texture)
    qr_fraction = np.linalg.norm(corners[0,1]-corners[0,0])/texture.shape[0]
    assert code == 'support:cart:0'
    assert np.ptp(points[:,0])*qr_fraction == pytest.approx(.06,abs=1e-6)


def test_marker_material_is_rtx_robust_emissive_qr(tmp_path):
    pytest.importorskip('pxr')
    pytest.importorskip('cv2')
    from pxr import Usd, UsdGeom, UsdShade
    from scripts.twin.restock_manipulation.isaac_sensors import write_carton_label
    config = design()
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.Xform.Define(stage, '/World'); UsdGeom.Xform.Define(stage, '/Cart')
    textures = {}
    for support in config['supports'].values():
        for marker in support['markers']:
            path = tmp_path / (marker['id'].replace(':', '_') + '.png')
            write_carton_label(marker['id'], path)
            textures[marker['id']] = path
    author_camera_design(stage, config, parent_paths={'map': '/World', 'cart_chassis': '/Cart'},
                         marker_textures=textures)
    # Source and cart stay textured; the self-lit emissive + sRGB colour space hardens
    # them against a diffuse-only distillation quirk. (The shelf markers are geometry now.)
    for name in ('source', 'cart'):
        base = ('/Cart' if name == 'cart' else '/World') + f'/RestockCamera_{name}_marker_0'
        tex = UsdShade.Shader(stage.GetPrimAtPath(base + '/Material/Texture'))
        assert tex.GetInput('sourceColorSpace').Get() == 'sRGB'
        surf = UsdShade.Shader(stage.GetPrimAtPath(base + '/Material/Surface'))
        for channel in ('diffuseColor', 'emissiveColor'):
            srcs = surf.GetInput(channel).GetConnectedSources()
            assert srcs and srcs[0][0].source.GetPrim().GetName() == 'Texture'


def test_shelf_markers_are_textureless_geometry_that_decodes(tmp_path):
    pytest.importorskip('pxr')
    cv2 = pytest.importorskip('cv2')
    from pxr import Usd, UsdGeom, UsdShade
    from scripts.twin.restock_manipulation.isaac_sensors import write_carton_label
    from scripts.twin.restock_camera.config import decode_authored_marker
    from scripts.twin.restock_camera.geometry import Intrinsics
    config = design()
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.Xform.Define(stage, '/World'); UsdGeom.Xform.Define(stage, '/Cart')
    textures = {}
    for support in config['supports'].values():
        for marker in support['markers']:
            path = tmp_path / (marker['id'].replace(':', '_') + '.png')
            write_carton_label(marker['id'], path)
            textures[marker['id']] = path
    author_camera_design(stage, config, parent_paths={'map': '/World', 'cart_chassis': '/Cart'},
                         marker_textures=textures)
    # The native render never sampled the shelf markers' texture, so they are geometry:
    # a white base quad plus one black quad per module, constant-colour, no UsdUVTexture.
    for index in (0, 1):
        base = f'/World/RestockCamera_shelf_marker_{index}'
        assert not stage.GetPrimAtPath(base + '/Material/Texture')          # no texture node
        white = UsdGeom.Mesh(stage.GetPrimAtPath(base))
        assert len(white.GetFaceVertexCountsAttr().Get()) == 1             # single white quad
        black = UsdGeom.Mesh(stage.GetPrimAtPath(base + '/Modules'))
        assert len(black.GetFaceVertexCountsAttr().Get()) > 50             # many black modules
        white_surf = UsdShade.Shader(stage.GetPrimAtPath(base + '/Material/Surface'))
        assert tuple(white_surf.GetInput('diffuseColor').Get()) == (1, 1, 1)
        black_surf = UsdShade.Shader(stage.GetPrimAtPath(base + '/Modules/Material/Surface'))
        assert tuple(black_surf.GetInput('diffuseColor').Get()) == (0, 0, 0)

    def centroid(path):
        prim = stage.GetPrimAtPath(path)
        matrix = np.array(UsdGeom.XformCache().GetLocalToWorldTransform(prim)).T
        points = np.array(UsdGeom.Mesh(prim).GetPointsAttr().Get(), float)
        return ((matrix @ np.c_[points, np.ones(len(points))].T).T[:, :3]).mean(axis=0)

    # Decode from the AUTHORED world geometry (not the design model): this catches the
    # child Modules mesh double-applying its parent transform, which flung the QR pattern
    # off behind the camera and rendered only a blank white plane. The black modules must
    # share the white base plane.
    camera = config['cameras']['shelf']
    world = np.array(camera['parent_from_optical']); k = Intrinsics(**camera['intrinsics'])
    for index, marker in enumerate(config['supports']['shelf']['markers']):
        base = f'/World/RestockCamera_shelf_marker_{index}'
        assert decode_authored_marker(stage, base, world, k) == marker['id']
        base_c, modules_c = centroid(base), centroid(base + '/Modules')
        assert np.linalg.norm(modules_c[:2] - base_c[:2]) < .02
        assert abs(modules_c[2] - base_c[2]) < .002
