"""CPU tests for the eye-in-hand wrist RGB-D camera: mount, intrinsics, design, USD authoring.

No Isaac/GPU. The grasp-region projection is pose-independent (camera + gripper are both rigid on
ur_tool0), which is exactly why the eye-in-hand view frames the side-grasp workspace the fixed body
camera could not. Kinematic-chain coverage of the URDF frame is in test_mobile_assets.py.
"""
import math

import numpy as np
import pytest

from scripts.twin.restock_camera import wrist_camera as W
from scripts.twin.restock_camera.config import camera_config
from scripts.twin.restock_camera.geometry import usd_camera_transform


def _design():
    scenario = dict(handling_unit=dict(local_wall=3, grasp_mode='side', dimensions_m=[.6, .4, .16]),
        initial_staging=dict(support_center_xy=[-5.73, 17.1], unit_bottom_z=.6,
            target_surface_pose_world=[-3.05, 4.826, .73, 0], cart_load_center_body_xy=[.08, 0],
            cart_support_z_world=.225))
    return camera_config(scenario, dict(spec=dict(arm_mount_z=.5)))


# --- intrinsics: wide field, modest (non-decode-grade) resolution ------------

def test_wrist_intrinsics_are_wide_field_and_modest_resolution():
    k = W.wrist_camera_intrinsics()
    assert (k.width, k.height) == (424, 240)                       # modest depth-camera resolution
    hfov = 2 * math.degrees(math.atan((k.width / 2) / k.fx))
    assert 70 < hfov < 80                                          # wide field (~75 deg)
    # Not decode-grade: far coarser than the fixed source(fx=2000)/shelf(fx=1050) station cameras.
    design = _design()
    assert k.fx < design['cameras']['source']['intrinsics']['fx'] / 5
    assert k.fx < design['cameras']['shelf']['intrinsics']['fx'] / 3


# --- calibrated mount ---------------------------------------------------------

def test_wrist_parent_from_optical_is_a_valid_rotation_aimed_at_the_grasp():
    T = W.wrist_parent_from_optical()
    R = T[:3, :3]
    np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-9)       # orthonormal
    assert np.isclose(np.linalg.det(R), 1.0, atol=1e-9)            # right-handed
    # Optical +Z (view direction) points from the mount toward the grasp point, in ur_tool0.
    view = W.wrist_tool0_from_optical()[:3, :3] @ [0, 0, 1]
    to_grasp = np.asarray(W.GRASP_POINT_TOOL0) - np.asarray(W.WRIST_MOUNT_XYZ)
    to_grasp /= np.linalg.norm(to_grasp)
    assert view @ to_grasp > 0.999                                 # aimed straight at the grasp


def test_grasp_region_projects_in_front_and_inside_the_wrist_image():
    k = W.wrist_camera_intrinsics()
    # The grasp point, the pre-grasp approach and the lift all sit in front of the camera and inside
    # the frame -- independent of base/arm pose, because they are rigid on the tool with the camera.
    tcp = np.asarray(W.GRASP_POINT_TOOL0)
    for name, point in [('grasp', tcp), ('approach', tcp + [0, 0, 0.12]), ('lift', tcp + [0, 0, 0.03]),
                        ('neighbour', tcp + [0, -0.22, 0])]:
        u, v, z = W.grasp_region_in_image(point)
        assert z > 0, name                                         # in front of the wrist camera
        assert 0 <= u <= k.width and 0 <= v <= k.height, name      # inside the image
    # The grasp point itself is near the image centre (the camera is aimed at it).
    u, v, _ = W.grasp_region_in_image(tcp)
    assert abs(u - k.cx) < 2 and abs(v - k.cy) < 2


# --- design entry + collision-camera selection --------------------------------

def test_design_records_the_wrist_camera_as_the_collision_camera():
    design = _design()
    assert 'wrist' in design['cameras']
    wrist = design['cameras']['wrist']
    assert wrist['parent'] == W.WRIST_FRAME and wrist['role'] == 'collision'
    assert wrist['frame'] == 'restock_camera_wrist' and wrist['decode_grade'] is False
    # The wrist camera has NO support (it is not a station QR camera, it is the collision depth cam).
    assert not any(s['camera_id'] == 'wrist' for s in design['supports'].values())


def test_collision_camera_resolver_selects_exactly_the_wrist_camera():
    design = _design()
    cam = W.collision_camera(design)
    assert cam['id'] == 'wrist' and cam['frame'] == 'restock_camera_wrist'
    # Fail closed: refuse if the design declares none or more than one collision camera.
    with pytest.raises(ValueError, match='exactly one collision camera'):
        W.collision_camera({'cameras': {}})
    two = {'cameras': {'a': {'role': 'collision'}, 'b': {'role': 'collision'}}}
    with pytest.raises(ValueError, match='exactly one collision camera'):
        W.collision_camera(two)


# --- USD authoring ------------------------------------------------------------

def test_author_wrist_camera_prim_matches_the_calibrated_mount_and_fov():
    pytest.importorskip('pxr')
    from pxr import Usd, UsdGeom, Gf
    from scripts.twin.mobile_assets import author_wrist_camera
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.SetStageUpAxis(stage, 'Z'); UsdGeom.SetStageMetersPerUnit(stage, 1)
    frame = '/MobileRobot/ur_tool0/restock_camera_wrist'
    UsdGeom.Xform.Define(stage, frame)
    path = author_wrist_camera(stage, frame)
    assert path == frame + '/wrist_camera'
    cam = UsdGeom.Camera(stage.GetPrimAtPath(path))
    # The authored optical transform matches the calibrated parent_from_optical (USD convention).
    matrix = np.array(UsdGeom.Xformable(cam.GetPrim()).GetLocalTransformation()).T
    np.testing.assert_allclose(matrix, usd_camera_transform(W.wrist_parent_from_optical()), atol=1e-9)
    # The pinhole aperture encodes the ~75 deg collision HFOV, and the clip range is the depth range.
    focal = cam.GetFocalLengthAttr().Get()
    hap = cam.GetHorizontalApertureAttr().Get()
    hfov = 2 * math.degrees(math.atan(hap / (2 * focal)))
    assert abs(hfov - W.WRIST_HFOV_DEG) < 0.5
    clip = cam.GetClippingRangeAttr().Get()
    k = W.wrist_camera_intrinsics()
    assert abs(clip[0] - k.near) < 1e-6 and abs(clip[1] - k.far) < 1e-6
    # Authoring twice refuses rather than silently overwriting the mount.
    with pytest.raises(ValueError, match='already authored'):
        author_wrist_camera(stage, frame)


# --- native_cameras selection support (CPU-safe parts) ------------------------

def test_native_cameras_resolves_the_collision_frame_as_the_wrist():
    from scripts.twin.restock.native_cameras import collision_camera_frame
    assert collision_camera_frame(_design()) == 'restock_camera_wrist'


def test_grasp_region_depth_coverage_is_the_proof_assertion_core():
    # The native proof's assertion: a rendered wrist depth with valid returns around the projected
    # grasp point yields high coverage there; an out-of-frame point yields zero. Integration feeds
    # the real GPU-rendered depth at the collect pose; this is the pure CPU check.
    k = W.wrist_camera_intrinsics()
    depth = np.full((k.height, k.width), np.nan)               # nothing returned anywhere yet
    u, v, _ = W.grasp_region_in_image(W.GRASP_POINT_TOOL0)      # grasp point projects to the centre
    ui, vi = int(round(u)), int(round(v))
    depth[vi - 20:vi + 20, ui - 20:ui + 20] = 0.32             # a valid in-range depth patch at the grasp
    cover = W.grasp_region_depth_coverage(depth, [W.GRASP_POINT_TOOL0, (5.0, 0.0, 0.2)], window_px=16)
    assert cover[0] > 0.99                                     # grasp region is in the depth
    assert cover[1] == 0.0                                     # a point off to the far side is not framed
    # Out-of-range depth (beyond far clip) does not count as valid coverage.
    depth[vi - 20:vi + 20, ui - 20:ui + 20] = k.far + 1.0
    assert W.grasp_region_depth_coverage(depth, [W.GRASP_POINT_TOOL0], window_px=16)[0] == 0.0


def test_wrist_depth_proof_checker_passes_on_a_framed_render_and_fails_on_empty():
    from scripts.prove_wrist_camera_depth import proof
    k = W.wrist_camera_intrinsics()
    tcp = np.asarray(W.GRASP_POINT_TOOL0)
    depth = np.full((k.height, k.width), np.nan)
    # Fill valid depth around the projected grasp and approach windows (what a native render returns).
    for pt in (tcp, tcp + [0, 0, 0.12], tcp + [0, 0, 0.03]):
        u, v, z = W.grasp_region_in_image(pt)
        ui, vi = int(round(u)), int(round(v))
        depth[max(vi - 24, 0):vi + 24, max(ui - 24, 0):ui + 24] = float(z)
    record = proof(depth)
    assert record['status'] == 'passed'
    assert record['coverage']['grasp'] > 0.6 and record['coverage']['approach'] > 0.6
    # An empty render (nothing returned) fails -- never fabricated into a pass.
    assert proof(np.full((k.height, k.width), np.nan))['status'] == 'failed'


def test_initialize_collision_camera_refuses_when_the_wrist_prim_is_not_authored():
    pytest.importorskip('pxr')
    from pxr import Usd, UsdGeom
    from scripts.twin.restock.native_cameras import initialize_collision_camera
    stage = Usd.Stage.CreateInMemory()
    frame = '/MobileRobot/ur_tool0/restock_camera_wrist'
    UsdGeom.Xform.Define(stage, frame)                         # frame present, camera prim NOT authored
    # Refuses (before any Isaac import) because author_wrist_camera has not run on this stage.
    with pytest.raises(ValueError, match='not authored on the robot'):
        initialize_collision_camera(stage, _design(), frame)
