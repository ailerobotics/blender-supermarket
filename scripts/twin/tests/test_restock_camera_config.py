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


def test_source_and_cart_markers_are_plain_diffuse_texture(tmp_path):
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
    # Source and cart stay plain diffuse texture. An emissive copy of the QR over-exposed
    # the SOURCE markers natively and broke the working decode, so the material must carry
    # NO emissive connection (and no colour-space override). The shelf markers are geometry.
    for name in ('source', 'cart'):
        base = ('/Cart' if name == 'cart' else '/World') + f'/RestockCamera_{name}_marker_0'
        tex = UsdShade.Shader(stage.GetPrimAtPath(base + '/Material/Texture'))
        assert not tex.GetInput('sourceColorSpace')
        surf = UsdShade.Shader(stage.GetPrimAtPath(base + '/Material/Surface'))
        diffuse = surf.GetInput('diffuseColor').GetConnectedSources()
        assert diffuse and diffuse[0][0].source.GetPrim().GetName() == 'Texture'
        emissive = surf.GetInput('emissiveColor')
        assert not emissive or not emissive.GetConnectedSources()[0]


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
        assert not white_surf.GetInput('emissiveColor')   # passive white; emissive washed the pattern
        black_surf = UsdShade.Shader(stage.GetPrimAtPath(base + '/Modules/Material/Surface'))
        assert tuple(black_surf.GetInput('diffuseColor').Get()) == (0, 0, 0)
        # Truly matte so the black does not mirror the white base / lamp at the grazing
        # shelf angle (ior 1.0 -> zero dielectric specular; metallic 0; roughness 1).
        for surf in (black_surf, white_surf):
            assert surf.GetInput('ior').Get() == 1.0
            assert surf.GetInput('metallic').Get() == 0.0
            assert surf.GetInput('roughness').Get() == 1.0

    def centroid(path):
        prim = stage.GetPrimAtPath(path)
        matrix = np.array(UsdGeom.XformCache().GetLocalToWorldTransform(prim)).T
        points = np.array(UsdGeom.Mesh(prim).GetPointsAttr().Get(), float)
        return ((matrix @ np.c_[points, np.ones(len(points))].T).T[:, :3]).mean(axis=0)

    # Decode from the AUTHORED world geometry (not the design model): this catches the
    # child Modules mesh double-applying its parent transform, which flung the QR pattern
    # off behind the camera and rendered only a blank white plane.
    camera = config['cameras']['shelf']
    world = np.array(camera['parent_from_optical']); k = Intrinsics(**camera['intrinsics'])
    from scripts.twin.restock_camera.geometry import project, transform_points
    for index, marker in enumerate(config['supports']['shelf']['markers']):
        base = f'/World/RestockCamera_shelf_marker_{index}'
        assert decode_authored_marker(stage, base, world, k) == marker['id']
        assert UsdGeom.Mesh(stage.GetPrimAtPath(base + '/Modules')).GetDoubleSidedAttr().Get()
        # PR#44's "full white base in frame" guarantee is relaxed: the shelf camera is
        # tilted (config camera() roll/yaw/dz levers; shipped pose is a pure -6.75 deg
        # tilt) to clear both QR cores off the certified production tile seams, which
        # pushes each outer white border toward a frame edge
        # where it may clip. The decodable CORE (the black-module region) plus a >=4-module
        # quiet zone must still sit inside the frame.
        modules = UsdGeom.Mesh(stage.GetPrimAtPath(base + '/Modules'))
        mm = np.array(UsdGeom.XformCache().GetLocalToWorldTransform(modules.GetPrim())).T
        pts = np.array(modules.GetPointsAttr().Get(), float)
        core = (mm @ np.c_[pts, np.ones(len(pts))].T).T[:, :3]
        uv, in_fov = project(transform_points(np.linalg.inv(world), core), k)
        assert in_fov.all()
        quiet = 4 * np.ptp(uv[:, 0]) / 21          # >=4-module zone; 25-module v2, ~21 data span
        assert uv[:, 0].min() > quiet and uv[:, 0].max() < k.width - quiet
        assert uv[:, 1].min() > quiet and uv[:, 1].max() < k.height - quiet
        base_c, modules_c = centroid(base), centroid(base + '/Modules')
        assert np.linalg.norm(modules_c[:2] - base_c[:2]) < .02         # same marker location
        # The modules sit ~2 mm in front of the base along the marker normal, and strictly
        # nearer the camera than the base at the same pixel, so RTX does not let the base win
        # the depth test (at 0.4 mm it did and the modules never rendered).
        assert 0.0015 < modules_c[2] - base_c[2] < 0.0025
        base_depth = (np.linalg.inv(world) @ np.r_[base_c, 1])[2]
        modules_depth = (np.linalg.inv(world) @ np.r_[modules_c, 1])[2]
        assert modules_depth < base_depth - 0.001                      # nearer the camera

    # Authoritative mission proof: the PRODUCTION decoder (FixedTileQR, the mission
    # RenderedSupportContext decoder) reads BOTH shelf markers from a rendered-faithful
    # rasterisation of the authored geometry at this pose. Before the camera tilt both
    # cores straddled the 640 px tile seams and FixedTileQR decoded neither (a full-frame
    # cv2 pass caught only the nearer one). NOTE: a clean raster is necessary, NOT
    # sufficient — the oblique decode is pose/noise sensitive; integration confirms the
    # decode on the native render (see shelf-tile-seam-root-cause.md).
    from scripts.twin.restock_camera.config import render_authored_marker
    from scripts.twin.restock_camera.qr import FixedTileQR
    from scripts.twin.restock_camera.sensed import FrameEvidence
    canvas = np.full((k.height, k.width), 128, np.uint8)
    for index in (0, 1):
        tile, _ = render_authored_marker(stage, f'/World/RestockCamera_shelf_marker_{index}', world, k)
        canvas = np.where(tile != 128, tile, canvas).astype(np.uint8)
    rgb = np.repeat(canvas[:, :, None], 3, axis=2).astype(np.uint8)
    evidence = FrameEvidence('shelf', 1, 1., 1., 'park', 'lens', 'mission')
    detections = FixedTileQR('shelf').decode(rgb, evidence, now=1., parked=True).detections
    assert {d.text for d in detections} == {'support:shelf:0', 'support:shelf:1'}


def test_shelf_camera_sees_under_the_tier_to_the_carton_top_and_support():
    """The shelf lens must stay below the shelf tier so it still senses the station.

    The decode proof above guards the OTHER half of the shelf camera's job (reading the
    two markers). This gate guards the primary half: the carton-top label plane and the
    empty support must stay visible UNDER the next shelf tier. A steeper candidate (pose
    A: -9 deg tilt, -1 deg yaw, +4 cm eye-raise) scored a marginally better marker-decode
    noise margin but raised the lens to z=1.00 m, ABOVE the ~0.97 m tier, so the tier
    occluded the whole station (plan_restock_sensed_views reported top/support 0/0) -- it
    decoded markers while blinding restocking. The shipped pure -6.75 deg tilt keeps the
    lens under the tier. This reproduces the planner's ray-visibility audit with a
    synthetic tier so the trade-off can never regress silently without the store USD.
    """
    from scripts.twin.restock_camera.geometry import Intrinsics
    from scripts.twin.restock_camera.visibility import box_surface, inspect_view, sample_plane
    config = design()
    camera = config['cameras']['shelf']
    pose = np.array(camera['parent_from_optical']); k = Intrinsics(**camera['intrinsics'])
    support = config['supports']['shelf']
    center = np.array(support['nominal_center']); yaw = support['yaw']
    dims = np.array([.6, .4, .16])                       # audited wall-3 handling unit
    top = sample_plane(center + [0, 0, dims[2]], dims[:2] - .02, yaw)   # carton-top plane
    footprint = sample_plane(center, [.62, .42], yaw)                  # empty support
    # The next shelf tier: a horizontal plate whose underside sits ~0.24 m above the 0.73 m
    # support surface (~0.97 m), spanning the support footprint and out over the aisle lip.
    tier_z = center[2] + .24
    tier = box_surface('shelf_tier_above', [center[0] - .07, center[1], tier_z + .01],
                       [.80, .60, .02])
    assert pose[2, 3] < tier_z                           # lens stays under the tier
    for plane in (top, footprint):
        view = inspect_view(pose, plane, k, [tier])
        assert view['all_samples_visible']               # whole station seen under the tier
        assert not view['occluders']
    # Negative control: the gate must BITE. Raise only the lens by pose A's +4 cm (above
    # the tier) and the tier must now occlude the station -- proving this is a real audit,
    # not a pose-agnostic pass.
    raised = pose.copy(); raised[2, 3] += .04
    assert raised[2, 3] > tier_z
    blinded = inspect_view(raised, top, k, [tier])
    assert blinded['visible_samples'] == 0 and 'shelf_tier_above' in blinded['occluders']
