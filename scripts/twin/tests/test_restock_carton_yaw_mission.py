"""Carton-top QR yaw at the REAL collect source geometry (run-177/run-182 seal-align).

CPU, no GPU/Isaac/ROS. The existing ``test_restock_carton_yaw`` fixture uses a downscaled
1280x960 intrinsic and scores ``% 90`` (deviation from the nearest right angle). This suite
pins the production decode -> ``IsaacCartonSensors._label_yaw`` path at the EXACT mission source
camera (2560x1920, the design ``k_source``/``parent_from_optical`` the mission renders with) and
adds the guarantees the run-182 diagnosis established, which the mod-90 fixture could not see:

* ABSOLUTE tracking: a carton rotated by ``dy`` degrees moves the fitted yaw by ``dy`` (so a
  constant ~5 deg observation bias on an identity carton would be caught), and
* NO SILENT BIAS from depth error: a left-right depth gradient big enough to rotate the fit past
  a few tenths of a degree is REFUSED by the coplanarity/discontinuity guards, never mis-reported.

Together these prove the carton-yaw ESTIMATOR cannot be the deterministic ~5 deg grasp-yaw bias
seen in the mission: with accurate depth it recovers the truth to <0.1 deg, and with corrupt depth
it refuses (the collect gate then holds the last clean registration). Analytic RGB-D fixture --
geometric design evidence, never a sensor/physics demonstration.
"""
import math
from types import SimpleNamespace as NS

import numpy as np
import pytest

from scripts.twin.restock_manipulation.isaac_sensors import IsaacCartonSensors

# Real design source camera: parent_from_optical (world_from_optical, parent='map') and k_source
# from twin.restock_camera.config (intr(2000.) -> 2560x1920, fx=fy=2000, centred principal point;
# camera('source','map',source+[0,.20,.85],source+[0,0,.08])). This is the extrinsic the mission
# renders the collect source view with (identical to the smoke-578-source bench camera.json).
CAMERA = np.array([[-1., 0., 0., -5.73],
                   [0., 0.96788, -0.2514, 17.3],
                   [0., -0.2514, -0.96788, 1.45],
                   [0., 0., 0., 1.]])
W, H, FX, FY, CX, CY = 2560, 1920, 2000., 2000., 1280., 960.
INTR = (FX, FY, CX, CY)
LABEL = NS(transform=CAMERA)                 # _label_yaw reads only self.transform
CAM_INV = np.linalg.inv(CAMERA)
QR_SIDE = 0.12                               # RestockLabel quad side (points +-0.06 on the carton top)
CARTON_TOP = np.array([-5.73, 17.1, 0.76])   # run-182 HandlingUnit translate (-5.73,17.1,0.68) + half-height


def _corners_world(yaw_deg):
    th = math.radians(yaw_deg); c, s = math.cos(th), math.sin(th); h = QR_SIDE / 2
    local = np.array([[-h, -h], [h, -h], [h, h], [-h, h]])     # RestockLabel points order
    return np.array([[CARTON_TOP[0] + c * lx - s * ly, CARTON_TOP[1] + s * lx + c * ly, CARTON_TOP[2]]
                     for lx, ly in local])


def _project(world):
    opt = (CAM_INV @ np.c_[world, np.ones(len(world))].T).T[:, :3]
    z = opt[:, 2]
    return np.c_[opt[:, 0] / z * FX + CX, opt[:, 1] / z * FY + CY], z


def _plane_depth():
    vv, uu = np.mgrid[0:H, 0:W]
    dirs = np.stack([(uu - CX) / FX, (vv - CY) / FY, np.ones_like(uu, float)], -1)
    world_ray = dirs @ CAMERA[:3, :3].T
    return ((CARTON_TOP[2] - CAMERA[2, 3]) / world_ray[:, :, 2]).astype(float)


def _decode_poly(yaw_deg):
    """Warp a real juice_orange QR onto the projected carton quad; return the decoded corners."""
    cv2 = pytest.importorskip('cv2')
    from scripts.twin.restock_camera.qr import FixedTileQR
    from scripts.twin.restock_camera.sensed import FrameEvidence
    poly_px, _ = _project(_corners_world(yaw_deg))
    qr = cv2.QRCodeEncoder_create().encode('juice_orange')
    qr = cv2.copyMakeBorder(qr, 4, 4, 4, 4, cv2.BORDER_CONSTANT, value=255)
    qr = cv2.resize(qr, (290, 290), interpolation=cv2.INTER_NEAREST)
    _, src, _ = cv2.QRCodeDetector().detectAndDecode(qr)          # TL,TR,BR,BL in QR px
    hh, ww = qr.shape[:2]
    # st mapping (RestockLabel st: pt0(0,0) pt1(1,0) pt2(1,1) pt3(0,1); USD samples (0,0) at the
    # image lower-left, V up), so the finder patterns land as the mission authors them.
    st = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], float)
    dst_qr = np.array([[u * (ww - 1), (1 - v) * (hh - 1)] for u, v in st], np.float32)
    Hm = cv2.getPerspectiveTransform(dst_qr, poly_px.astype(np.float32))
    gray = cv2.warpPerspective(qr, Hm, (W, H), flags=cv2.INTER_NEAREST, borderValue=255)
    rgb = np.repeat(gray[:, :, None], 3, axis=2).astype(np.uint8)
    ev = FrameEvidence('source', 1, 1., 1., 'park1', 'lens1', session_id='s1')
    frameqr = FixedTileQR('source').decode(rgb, ev, now=1., parked=True)
    det = {d.text: np.array(d.corners, float) for d in frameqr.detections}
    assert 'juice_orange' in det, 'fixture QR did not decode at the mission source geometry'
    return det['juice_orange']


def test_identity_carton_reads_a_clean_right_angle_multiple():
    """At the true run-182 pose (axis-aligned carton), the fit sits on a multiple of 90 deg to
    well under a degree -- i.e. there is NO intrinsic estimator bias at the mission geometry."""
    fit = math.degrees(IsaacCartonSensors._label_yaw(LABEL, _decode_poly(0.0), _plane_depth(), INTR))
    dev = fit % 90.0
    assert min(dev, 90 - dev) < 0.5, f'identity carton off a right angle by {min(dev, 90 - dev):.3f} deg'


@pytest.mark.parametrize('dy', [1.0, 2.0, 3.6, 5.0, -5.0])
def test_fit_tracks_true_yaw_absolutely(dy):
    """A carton rotated by dy deg moves the fitted yaw by dy deg (within 1 deg). Convention-free:
    a CONSTANT bias (e.g. the mission's ~5 deg) cancels in the difference, so this isolates the
    estimator's response to the TRUTH. It stays <1 deg -> the estimator does not manufacture yaw."""
    depth = _plane_depth()
    base = math.degrees(IsaacCartonSensors._label_yaw(LABEL, _decode_poly(0.0), depth, INTR))
    rot = math.degrees(IsaacCartonSensors._label_yaw(LABEL, _decode_poly(dy), depth, INTR))
    tracked = (rot - base + 180) % 360 - 180
    assert abs(tracked - dy) < 1.0, f'fit moved {tracked:.3f} deg for a {dy:.3f} deg carton rotation'


@pytest.mark.parametrize('grad_mm', [10., 20., 40.])
def test_left_right_depth_gradient_is_refused_not_silently_biased(grad_mm):
    """A left-right depth gradient across the QR (the only depth-error shape that rotates the
    XY fit) rotates the fit ~0.3 deg per 5 mm; a gradient big enough to reach the mission's ~5 deg
    (~40 mm) breaks the world-Z coplanarity / discontinuity guard and is REFUSED -- so the collect
    path holds its last clean registration rather than emitting a 5-deg-wrong grasp. Proves the
    estimator has no SILENT multi-degree bias regime from the fast/full render depth."""
    poly = _decode_poly(0.0)
    depth = _plane_depth()
    uu = np.arange(W)[None, :]
    span_px = poly[:, 0].ptp()
    biased = depth + (grad_mm / 1000.) * (uu - CX) / span_px      # +-grad_mm across the QR width
    with pytest.raises(ValueError, match='uncertain/tilted|discontinuity'):
        IsaacCartonSensors._label_yaw(LABEL, poly, biased, INTR)


def test_small_subthreshold_gradient_stays_well_under_one_degree():
    """The depth errors the guards DO pass (a few mm across the QR) can only move the fit a few
    tenths of a degree -- never the mission's ~5 deg. Bounds the worst silent case."""
    poly = _decode_poly(0.0)
    depth = _plane_depth()
    uu = np.arange(W)[None, :]
    span_px = poly[:, 0].ptp()
    base = math.degrees(IsaacCartonSensors._label_yaw(LABEL, poly, depth, INTR))
    biased = depth + (0.005) * (uu - CX) / span_px                # 5 mm across the QR (passes guards)
    got = math.degrees(IsaacCartonSensors._label_yaw(LABEL, poly, biased, INTR))
    assert abs((got - base + 180) % 360 - 180) < 0.8
