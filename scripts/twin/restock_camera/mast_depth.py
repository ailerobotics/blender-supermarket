"""Robot-mast depth camera that feeds the nvblox obstacle map while a carton is carried.

Problem (integration finding B'): the eye-in-hand ``wrist`` camera is nvblox's ONLY
collision/depth feed. From grasp on, the carried carton fills the wrist frame, the
rig-masked depth stream to nvblox stops and the ESDF freezes at the place re-park, so the
transfer/place approach plans against a stale collision world.

This module designs a SECOND, body-mounted RealSense-class depth camera on the robot mast.
It is fixed to ``base_link`` (never to the wrist/tool), so a carton held by the gripper
cannot fill its frame; it looks over the arm workspace toward the arm's working side
(``base`` +Y) and down onto the place/transfer approach. It feeds nvblox as ``camera_1``
ALONGSIDE the wrist (``camera_0``) with the SAME masking applied: the per-pixel instance-ID
rig self mask (the attached carton is a rig-instance subtree, so it is removed like any other
rig pixel) plus the ``object_segmenter`` grasp-target bbox mask -- never a raw unmasked feed.

Wiring lives in :mod:`twin.restock_host.launch`/:mod:`twin.restock_host.config`, gated by the
host-config ``mast_depth`` switch (off by default: existing cameras and the single-feed
default are byte-for-byte unchanged when off; the integration lane turns it on per run). The
camera prim is authored from :func:`mast_camera_entry` (added to
``restock_camera.config.camera_config(..., mast_depth=True)``) under ``base_link``.

Offline ray/visibility checks here are geometric design predictions, never rendered RGB-D
measurements (same disclaimer as :mod:`twin.restock_camera.visibility`).
"""
import math

import numpy as np

from .geometry import Intrinsics, pose_matrix

# --- RealSense D435-class depth mode (native authoring/calibration) ---------------------
# 848x480 is the D435 depth sweet spot: full stereo FOV at a modest pixel count. HFOV 87 deg
# is the published D435 depth horizontal field; fy is kept equal to fx (square pixels), which
# yields VFOV ~= 56 deg at 480 px -- within a degree of the published 58 deg. The principal
# point is centred (Isaac Camera.get_intrinsics_matrix / the native pinhole authoring both
# require cx,cy = width/2, height/2). near/far bracket the D435 usable range; the 0.02 m depth
# noise bound is the conservative stereo error used by the offline voxel model.
MAST_HFOV_DEG = 87.0
MAST_WIDTH = 848
MAST_HEIGHT = 480
MAST_NEAR_M = 0.30
MAST_FAR_M = 6.0
MAST_DEPTH_NOISE_M = 0.02

# Mount on base_link (metres, base frame: +X forward, +Y the arm's working side, +Z up).
# High on the mast and slightly behind the arm work line so it sees OVER the arm and a carton
# held in front of the gripper, pitched down toward the +Y workspace and the place approach.
MAST_MOUNT_XYZ = (0.0, -0.15, 1.55)
MAST_YAW_RAD = math.pi / 2       # optical forward from base +X to base +Y (the arm's side)
MAST_PITCH_RAD = -0.60           # tilt down ~34 deg onto the arm workspace / place approach

# nvblox second-camera feed. camera_0 stays the wrist (restock_host/launch.py); camera_1 is
# this mast feed. The Isaac side publishes rig+carton-masked depth on MAST_MASKED_DEPTH; a
# second object_segmenter masks the grasp target out and republishes MAST_WORLD_DEPTH, which
# nvblox reads as camera_1 (camera_info taken straight from the masked-depth stream).
MAST_CAMERA_INDEX = 1
MAST_FRAME_ID = 'restock_camera_mast'
MAST_MASKED_DEPTH = 'camera_mast/masked_depth'          # Isaac -> segmenter (NS-relative)
MAST_MASKED_DEPTH_INFO = 'camera_mast/masked_depth/camera_info'
MAST_WORLD_DEPTH = 'world_depth_mast'                    # segmenter -> nvblox camera_1


def mast_intrinsics():
    """Native RealSense-class :class:`Intrinsics` (centred pinhole, square pixels)."""
    f = (MAST_WIDTH / 2.0) / math.tan(math.radians(MAST_HFOV_DEG) / 2.0)
    return Intrinsics(MAST_WIDTH, MAST_HEIGHT, f, f, MAST_WIDTH / 2.0, MAST_HEIGHT / 2.0,
                      MAST_NEAR_M, MAST_FAR_M)


def _base_from_optical():
    """Optical frame (right/down/forward) expressed in base_link, from yaw+pitch+position.

    Same right/down/forward convention as ``observed_space.DepthMount.world_ray`` with a zero
    base heading: forward = (cos(yaw)cos(p), sin(yaw)cos(p), sin(p)); right = (sin(yaw),
    -cos(yaw), 0); down = forward x right-completion.
    """
    cy, sy = math.cos(MAST_YAW_RAD), math.sin(MAST_YAW_RAD)
    cp, sp = math.cos(MAST_PITCH_RAD), math.sin(MAST_PITCH_RAD)
    forward = np.array([cy * cp, sy * cp, sp])
    right = np.array([sy, -cy, 0.0])
    down = np.array([cy * sp, sy * sp, -cp])
    tf = np.eye(4)
    tf[:3, :3] = np.column_stack([right, down, forward])
    tf[:3, 3] = MAST_MOUNT_XYZ
    return tf


def mast_depth_mount(*, width=None, height=None):
    """A ``twin.navigation.observed_space.DepthMount`` for this mast camera.

    Defaults to a coarse (FOV-faithful) grid for cheap CPU ray/voxel proofs; pass the native
    ``width``/``height`` for a full-resolution model. The horizontal FOV, mount pose, range
    and noise bound match :func:`mast_intrinsics` and the mount constants, so the coarse proof
    grid predicts the same sightlines as the authored camera.
    """
    from twin.navigation.observed_space import DepthMount
    w = width or 96
    h = height or int(round(w * MAST_HEIGHT / MAST_WIDTH))
    return DepthMount('mast', 'base', MAST_MOUNT_XYZ, yaw=MAST_YAW_RAD, pitch=MAST_PITCH_RAD,
                      width=w, height=h, horizontal_fov=math.radians(MAST_HFOV_DEG),
                      near=MAST_NEAR_M, far=MAST_FAR_M, noise_bound=MAST_DEPTH_NOISE_M)


def mast_camera_entry():
    """``camera_config``-schema ``cameras``/``solids`` fragment for the mast depth camera.

    Returned as ``(camera, solids)`` in the same shape ``restock_camera.config.camera_config``
    emits, parented to ``robot_base`` (``camera_mount_obstacles`` accepts that parent, and the
    moving mount travels with the rig collision model). ``author_camera_design`` authors the
    camera prim and collision housing from this when a ``robot_base`` parent path is supplied.
    """
    k = mast_intrinsics()
    pose = _base_from_optical()
    housing = pose.copy()
    housing[:3, 3] += pose[:3, :3] @ [0, 0, -0.014]       # body behind the glass, like config.camera
    from dataclasses import asdict
    camera = dict(id='mast', parent='robot_base', parent_from_optical=pose.tolist(),
                  intrinsics=asdict(k), revision='mast-depth-realsense-v1', role='nvblox_depth')
    solids = [dict(id='mast_housing', parent='robot_base', parent_from_solid=housing.tolist(),
                   dimensions_m=[.09, .025, .025], collision=True),
              # Bracket tying the housing back to the mast column (collision geometry the
              # arm/cart planning must see); sits just behind the housing on the mast.
              dict(id='mast_bracket', parent='robot_base',
                   parent_from_solid=pose_matrix([*np.add(MAST_MOUNT_XYZ[:2], [0, -0.05]),
                                                  0.0], MAST_MOUNT_XYZ[2] - 0.05).tolist(),
                   dimensions_m=[.04, .06, .12], collision=True)]
    return camera, solids


def nvblox_feed():
    """The nvblox ``camera_1`` input contract for the host launch (NS-relative topic names)."""
    return dict(index=MAST_CAMERA_INDEX, frame_id=MAST_FRAME_ID,
                masked_depth=MAST_MASKED_DEPTH, masked_depth_info=MAST_MASKED_DEPTH_INFO,
                world_depth=MAST_WORLD_DEPTH)


# --- Offline design proofs (predictions, not sensor measurements) -----------------------

def world_from_base(base_pose):
    """4x4 world<-base from a planar base pose ``(x, y, yaw)`` (base_link on the ground)."""
    x, y, yaw = base_pose
    return pose_matrix((x, y, yaw), 0.0)


def mast_world_from_optical(base_pose):
    """World<-optical of the mast camera for a base pose ``(x, y, yaw)``."""
    return world_from_base(base_pose) @ _base_from_optical()


def proof_depth_frame(base_pose, stamp, *, scene_depth_m=2.0, carton_masked=True,
                      full_frame_masked=False, width=16):
    """A synthetic ``observed_space.DepthFrame`` from the coarse mast mount (offline ESDF proof).

    Models one rendered mast depth frame while a carton is carried: every pixel sees the scene
    at ``scene_depth_m`` EXCEPT a central block standing for the carton silhouette, which is
    masked (depth invalid, ``self_mask`` set) exactly as the rig+carton instance-ID mask removes
    it before the feed reaches nvblox. ``full_frame_masked`` masks the WHOLE frame, modelling the
    eye-in-hand wrist whose view the held carton fills end to end. Prediction, not a render.
    """
    import math as _math
    from twin.navigation.observed_space import DepthFrame
    mount = mast_depth_mount(width=width)
    w, h = mount.width, mount.height
    depth = np.full((h, w), float(scene_depth_m), np.float64)
    mask = np.zeros((h, w), bool)
    if full_frame_masked:
        mask[:, :] = True
        depth[:, :] = _math.nan
    elif carton_masked:
        # Central ~1/3 block: the carried carton's silhouette in the mast frame.
        r0, r1 = h // 3, h - h // 3
        c0, c1 = w // 3, w - w // 3
        mask[r0:r1, c0:c1] = True
        depth[r0:r1, c0:c1] = _math.nan
    k = mount.focal_pixels
    intrinsics = (k, 0., (w - 1) / 2., 0., k, (h - 1) / 2., 0., 0., 1.)
    x, y, yaw = base_pose
    mfo = mast_world_from_optical(base_pose)
    return DepthFrame(mount, float(stamp), float(stamp), (x, y, 0.0, yaw),
                      tuple(depth.ravel()), self_mask=tuple(map(bool, mask.ravel())),
                      intrinsics=intrinsics, map_from_optical=tuple(mfo.ravel()), rendering_frame=1)


def sees_points(base_pose, points, occluders):
    """``visibility.inspect_view`` report for the mast camera at ``base_pose``.

    ``occluders`` is a list of ``visibility.Surface`` (e.g. the held carton, cart, shelf).
    A point counts visible when it is inside the frustum/range AND no occluder segment lies
    between the lens and the point. ``visibility.ray_hits`` excludes the endpoint, so a
    carton surface sampled as a target is not treated as its own occluder. The returned dict
    carries ``visible_samples``, ``all_samples_visible``, ``in_fov_and_range``, ``occluders``
    and ``blocked_sample_indices``.
    """
    from .visibility import inspect_view
    return inspect_view(mast_world_from_optical(base_pose), np.asarray(points, float),
                        mast_intrinsics(), occluders)
