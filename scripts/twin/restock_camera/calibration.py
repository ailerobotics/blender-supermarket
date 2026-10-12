"""Depth-only support inference. Never identifies SKU or certifies inventory."""
from dataclasses import dataclass
import numpy as np
from .geometry import backproject, rigid, transform_points, validate_roi


def fresh(now, stamp, parked):
    if not parked or not np.isfinite([now,stamp]).all() or not 0 <= now-stamp <= .5:
        raise ValueError('requires parked robot/cart and fresh synchronized depth/TF')


@dataclass(frozen=True)
class SupportCalibration:
    location: str
    roi: tuple
    support_z: float
    observed_dimensions_xy: tuple
    yaw_rad: float
    residual_m: float
    stamp: float
    frame_id: str
    epoch: str

    def slot_view(self, *, now, epoch, parked, clearance_m, clearance_stamp,
                  clearance_epoch):
        """Clearance is a separately sensor-measured unobstructed height, not CAD.

        Single plane depth cannot measure the volume above it. Integration must
        supply a fresh registered multi-view scan and its current epoch here.
        """
        fresh(now,self.stamp,parked)
        fresh(now,clearance_stamp,parked)
        if (epoch != self.epoch or clearance_epoch != epoch
                or not np.isfinite(clearance_m) or clearance_m <= 0):
            raise ValueError('frame changed or unmeasured vertical clearance')
        from ..restock_manipulation.isaac_sensors import SlotView
        return SlotView(self.location,self.roi,self.support_z,
                        (*self.observed_dimensions_xy,float(clearance_m)),self.yaw_rad)


def calibrate_support(depth, intrinsics, base_from_camera, roi, *, location,
                      now, stamp, frame_id, epoch, parked, expected_support_z,
                      yaw_rad=0., transform_stamp=None):
    """Reject missing pixels, non-horizontal planes and ambiguous elevated tops.

    expected_support_z is an independently calibrated support-height prior in
    the current base frame, used only to reject objects mistaken for the support.
    All ROI pixels must see the support; no inpainting or semantic masks.
    Caller owns frame-ID uniqueness and epoch invalidation, including motion
    away and back to an identical pose. A calibration expires after 0.5 s.
    """
    fresh(now,stamp,parked)
    if (transform_stamp is None or not np.isfinite(transform_stamp)
            or abs(transform_stamp-stamp) > .05):
        raise ValueError('missing or unsynchronized transform timestamp')
    if not location or not frame_id or not epoch or not np.isfinite([expected_support_z,yaw_rad]).all():
        raise ValueError('missing frame identity or support prior')
    k = intrinsics
    roi = validate_roi(roi,k)
    m = rigid(base_from_camera)
    depth = np.asarray(depth,float)
    if depth.shape != (k.height,k.width):
        raise ValueError('unregistered depth shape')
    x1,y1,x2,y2 = roi
    v,u = np.mgrid[y1:y2,x1:x2]
    if u.size < 64:
        raise ValueError('insufficient support pixels')
    cloud = transform_points(m,backproject(np.column_stack([u.ravel(),v.ravel()]),
                                           depth[y1:y2,x1:x2].ravel(),k))
    center = np.mean(cloud,axis=0)
    _,singular,vectors = np.linalg.svd(cloud-center,full_matrices=False)
    normal = vectors[-1]
    residual = float(np.max(np.abs((cloud-center) @ normal)))
    if (singular[1] < .01 or abs(normal[2]) < .999
            or residual > .003 or np.ptp(cloud[:,2]) > .006
            or abs(center[2]-expected_support_z) > .01):
        raise ValueError('support plane uncertain, tilted, occluded or elevated')
    # Infer an inscribed horizontal rectangle, not the larger trapezoid bounds.
    c,s = np.cos(yaw_rad),np.sin(yaw_rad)
    rotation = np.array([[c,-s],[s,c]])
    xy = (cloud[:,:2]-center[:2]) @ rotation
    corners = xy[[0,x2-x1-1,-1,-(x2-x1)]]
    half = np.minimum(-xy.min(axis=0),xy.max(axis=0))
    for _ in range(100):
        test = np.array([[-1,-1],[1,-1],[1,1],[-1,1]])*half
        edges = np.roll(corners,-1,axis=0)-corners
        delta = test[:,None,:]-corners
        cross = edges[None,:,0]*delta[:,:,1]-edges[None,:,1]*delta[:,:,0]
        if np.all(cross >= -1e-9) or np.all(cross <= 1e-9):
            break
        half *= .95
    else:
        raise ValueError('cannot establish observed support footprint')
    dimensions = 2*half-.004  # pixel/noise border allowance, metres
    if min(dimensions) <= .01:
        raise ValueError('observed support too small')
    return SupportCalibration(location,roi,float(center[2]),tuple(dimensions),
                              yaw_rad,residual,stamp,str(frame_id),str(epoch))
