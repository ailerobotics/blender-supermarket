"""Carton-top QR yaw estimator (`IsaacCartonSensors._label_yaw`) — CPU, no GPU/Isaac/ROS.

Pins the fitted carton yaw error below 1 deg against the known truth over an analytic source
view, and locks the four-corner fit's robustness to a single mis-sampled/boundary corner — the
run-177 collect seal-alignment hardening (a single bad top-edge corner used to rotate the yaw by
several degrees; the four-corner fit averages corners and refuses a corner that straddles a depth
discontinuity). Analytic RGB-D fixture, never a sensor/physics demonstration.
"""
import math
from types import SimpleNamespace as NS

import numpy as np
import pytest

from scripts.twin.restock_camera.geometry import Intrinsics, project, transform_points
from scripts.twin.restock_manipulation.isaac_sensors import IsaacCartonSensors

# A realisable source-station view: fixed camera above the backstock, looking down with the
# config-derived ~14.56 deg pitch (parent_from_optical rotation), carton top plane at world z=0.
_R = np.array([[-1., 0., 0.],
               [0., 0.96788370, -0.25139836],
               [0., -0.25139836, -0.96788370]])
CAMERA = np.eye(4)
CAMERA[:3, :3] = _R
CAMERA[:3, 3] = [0., 0.20, 0.85]            # world_from_optical (self.transform = base_from_camera)
K = Intrinsics(1280, 960, 2500., 2500., 640., 480., near=.05, far=3.5)
INTR = (K.fx, K.fy, K.cx, K.cy)
QR_SIDE = 0.0873                            # decoded handling-unit label square (excludes quiet border)
LABEL = NS(transform=CAMERA)                # _label_yaw reads only self.transform


def _plane_depth_image(plane_z=0.):
    """Analytic distance-to-image-plane (optical z) of the flat top plane for every pixel."""
    vv, uu = np.mgrid[0:K.height, 0:K.width]
    dirs = np.stack([(uu - K.cx) / K.fx, (vv - K.cy) / K.fy, np.ones_like(uu, float)], -1)
    world_ray = dirs @ _R.T
    return (-CAMERA[2, 3] / world_ray[:, :, 2]).astype(float)


def _carton_corners_px(yaw_deg, center=(0., 0., 0.)):
    """Project the four QR corners (TL,TR,BR,BL) of a carton rotated yaw_deg about world +Z."""
    th = math.radians(yaw_deg)
    c, s = math.cos(th), math.sin(th)
    h = QR_SIDE / 2
    local = np.array([[-h, h], [h, h], [h, -h], [-h, -h]])        # TL,TR,BR,BL, +X right/+Y up
    world = np.array([[center[0] + c * lx - s * ly, center[1] + s * lx + c * ly, center[2]]
                      for lx, ly in local])
    uv, valid = project(transform_points(np.linalg.inv(CAMERA), world), K)
    assert valid.all(), 'fixture carton outside the analytic view'
    return uv, world


def _yaw_err_deg(estimated_rad, truth_deg):
    d = (math.degrees(estimated_rad) - truth_deg) % 90.0
    return d - 90 if d > 45 else d


@pytest.mark.parametrize('yaw_deg', [0., 0.6, 1.0, 2.0, 3.6, 5.0, 8.0, -4.0])
def test_label_yaw_matches_truth_within_one_degree(yaw_deg):
    depth = _plane_depth_image()
    poly, _ = _carton_corners_px(yaw_deg)
    fitted = IsaacCartonSensors._label_yaw(LABEL, poly, depth, INTR)
    assert abs(_yaw_err_deg(fitted, yaw_deg)) < 1.0


def test_axis_aligned_carton_reads_zero_yaw():
    depth = _plane_depth_image()
    poly, _ = _carton_corners_px(0.0)
    assert abs(_yaw_err_deg(IsaacCartonSensors._label_yaw(LABEL, poly, depth, INTR), 0.0)) < 0.1


def test_four_corner_fit_beats_a_single_biased_corner():
    """A sub-threshold depth bias on ONE corner: the four-corner fit's error stays well under the
    error a two-corner top edge would incur from that same corner (it averages four corners)."""
    depth = _plane_depth_image()
    yaw = 2.0
    poly, _ = _carton_corners_px(yaw)
    bad = depth.copy()
    ui, vi = int(round(poly[1, 0])), int(round(poly[1, 1]))      # bias the TR corner by 4 mm
    bad[vi - 1:vi + 2, ui - 1:ui + 2] -= 0.004
    four = IsaacCartonSensors._label_yaw(LABEL, poly, bad, INTR)
    # the former two-corner top edge (TL->TR) carries the biased corner at full weight
    pts = []
    for u, v in poly[:2]:
        z = bad[int(round(v)), int(round(u))]
        pts.append(CAMERA @ np.array([(u - K.cx) * z / K.fx, (v - K.cy) * z / K.fy, z, 1.]))
    axis = pts[1] - pts[0]
    two = math.atan2(axis[1], axis[0])
    assert abs(_yaw_err_deg(four, yaw)) < abs(_yaw_err_deg(two, yaw))
    assert abs(_yaw_err_deg(four, yaw)) < 1.0


def test_gross_corner_mis_sample_is_refused_not_mis_reported():
    """A corner backed by a ~2 cm depth discontinuity (carton rim / gap bleed) rotated the old
    two-corner yaw by ~3 deg; the four-corner fit refuses it (non-coplanar / discontinuous) so the
    collect path holds its last clean registration instead of emitting a wrong grasp pose."""
    depth = _plane_depth_image()
    yaw = 2.0
    poly, _ = _carton_corners_px(yaw)
    bad = depth.copy()
    ui, vi = int(round(poly[1, 0])), int(round(poly[1, 1]))
    bad[vi - 1:vi + 2, ui - 1:ui + 2] -= 0.02
    with pytest.raises(ValueError, match='uncertain/tilted|discontinuity'):
        IsaacCartonSensors._label_yaw(LABEL, poly, bad, INTR)


def test_tilted_top_is_rejected():
    """A non-horizontal top (corners spread in world Z) is uncertain and must refuse."""
    depth = _plane_depth_image()
    poly, _ = _carton_corners_px(0.0)
    tilted = depth.copy()
    # push the right half of the image nearer, tilting the recovered top plane
    tilted[:, K.width // 2:] -= 0.03
    with pytest.raises(ValueError, match='uncertain/tilted|discontinuity'):
        IsaacCartonSensors._label_yaw(LABEL, poly, tilted, INTR)


def test_production_decoder_to_yaw_end_to_end():
    """Drive the real FixedTileQR decode -> _label_yaw on a warped real QR label, truth yaw known."""
    cv2 = pytest.importorskip('cv2')
    from scripts.twin.restock_camera.qr import FixedTileQR
    from scripts.twin.restock_camera.sensed import FrameEvidence
    yaw = 3.6
    depth = _plane_depth_image()
    poly, _ = _carton_corners_px(yaw)
    qr = cv2.QRCodeEncoder_create().encode('juice_orange')
    qr = cv2.copyMakeBorder(qr, 4, 4, 4, 4, cv2.BORDER_CONSTANT, value=255)
    qr = cv2.resize(qr, (300, 300), interpolation=cv2.INTER_NEAREST)
    decoded, src, _ = cv2.QRCodeDetector().detectAndDecode(qr)
    assert decoded == 'juice_orange'
    # warp the printed label so its decoded corners land on the projected carton quad (TL,TR,BR,BL)
    homography = cv2.getPerspectiveTransform(src[0].astype(np.float32), poly.astype(np.float32))
    gray = cv2.warpPerspective(qr, homography, (K.width, K.height), flags=cv2.INTER_NEAREST, borderValue=255)
    rgb = np.repeat(gray[:, :, None], 3, axis=2).astype(np.uint8)
    ev = FrameEvidence('source', 1, 1., 1., 'park1', 'lens1', session_id='s1')
    frameqr = FixedTileQR('source').decode(rgb, ev, now=1., parked=True)
    corners = {d.text: np.array(d.corners, float) for d in frameqr.detections}
    assert 'juice_orange' in corners
    fitted = IsaacCartonSensors._label_yaw(LABEL, corners['juice_orange'], depth, INTR)
    assert abs(_yaw_err_deg(fitted, yaw)) < 1.0
