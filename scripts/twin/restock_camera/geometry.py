"""Metres; column-vector parent_from_child; optical right/down/forward."""
from dataclasses import dataclass
import math
import numpy as np


@dataclass(frozen=True)
class Intrinsics:
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    near: float = .15
    far: float = 3.5

    def __post_init__(self):
        if (type(self.width) is not int or type(self.height) is not int
                or min(self.width, self.height) < 2
                or not np.isfinite([self.fx,self.fy,self.cx,self.cy,self.near,self.far]).all()
                or min(self.fx,self.fy,self.near) <= 0 or self.far <= self.near
                or not 0 <= self.cx < self.width or not 0 <= self.cy < self.height):
            raise ValueError('invalid camera intrinsics/range')


def rigid(matrix):
    m = np.asarray(matrix, float)
    if (m.shape != (4,4) or not np.isfinite(m).all()
            or not np.allclose(m[3], [0,0,0,1], atol=1e-8)
            or not np.allclose(m[:3,:3].T @ m[:3,:3], np.eye(3), atol=1e-8)
            or not np.isclose(np.linalg.det(m[:3,:3]), 1, atol=1e-8)):
        raise ValueError('expected rigid transform')
    return m


def pose_matrix(pose, z=0.):
    x,y,yaw = pose
    c,s = math.cos(yaw),math.sin(yaw)
    return rigid([[c,-s,0,x],[s,c,0,y],[0,0,1,z],[0,0,0,1]])


def look_at(eye, target):
    eye, target = np.asarray(eye,float), np.asarray(target,float)
    forward = target-eye
    if not np.isfinite(forward).all() or np.linalg.norm(forward) < 1e-8:
        raise ValueError('invalid look direction')
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, [0,0,1])
    if np.linalg.norm(right) < 1e-8:
        raise ValueError('vertical look direction needs explicit roll')
    right /= np.linalg.norm(right)
    result = np.eye(4)
    result[:3,:3] = np.column_stack([right, np.cross(forward,right), forward])
    result[:3,3] = eye
    return rigid(result)


def relative_camera(world_from_base, world_from_camera):
    return np.linalg.inv(rigid(world_from_base)) @ rigid(world_from_camera)


def usd_camera_transform(parent_from_optical):
    """USD camera looks along -Z with +Y up; result still column-vector form.

    Transpose when constructing a Gf.Matrix4d (USD uses row vectors).
    Do not apply this conversion to a sensor API already using optical axes.
    """
    return rigid(parent_from_optical) @ np.diag([1.,-1.,-1.,1.])


def transform_points(parent_from_child, points):
    m = rigid(parent_from_child)
    return np.asarray(points) @ m[:3,:3].T + m[:3,3]


def project(points, intrinsics):
    p = np.asarray(points,float)
    if p.ndim != 2 or p.shape[1] != 3 or not np.isfinite(p).all():
        raise ValueError('expected finite Nx3 points')
    k = intrinsics
    with np.errstate(divide='ignore', invalid='ignore'):
        uv = p[:,:2]/p[:,2,None] * [k.fx,k.fy] + [k.cx,k.cy]
    valid = ((p[:,2] >= k.near) & (p[:,2] <= k.far)
             & (uv[:,0] >= 0) & (uv[:,0] < k.width)
             & (uv[:,1] >= 0) & (uv[:,1] < k.height))
    return uv, valid


def backproject(uv, depth, intrinsics):
    uv, d = np.asarray(uv,float), np.asarray(depth,float)
    k = intrinsics
    if (uv.ndim != 2 or uv.shape[1] != 2 or d.shape != (len(uv),)
            or not np.isfinite(uv).all() or not np.isfinite(d).all()
            or np.any((d < k.near) | (d > k.far))
            or np.any((uv < 0) | (uv >= [k.width,k.height]))):
        raise ValueError('invalid depth/pixel coordinates')
    return np.column_stack([(uv-[k.cx,k.cy])/[k.fx,k.fy]*d[:,None],d])


def validate_roi(roi, k):
    if (len(roi) != 4 or any(type(x) is not int for x in roi)
            or not 0 <= roi[0] < roi[2] <= k.width
            or not 0 <= roi[1] < roi[3] <= k.height):
        raise ValueError('ROI must be integer, nonempty and inside image')
    return tuple(roi)


def plane_roi(corners, base_from_camera, k):
    """Inscribed integer ROI in an ordered convex, fully visible plane quad.

    Runtime corners must come from depth calibration, not scene semantics.
    Offline callers may use design corners, clearly labelled as predictions.
    """
    points = np.asarray(corners,float)
    if points.shape != (4,3):
        raise ValueError('four ordered support corners required')
    uv,valid = project(transform_points(np.linalg.inv(rigid(base_from_camera)),points),k)
    if not valid.all():
        raise ValueError('support quad truncated or out of range')
    edges = np.roll(uv,-1,axis=0)-uv
    turns = edges[:,0]*np.roll(edges,-1,axis=0)[:,1]-edges[:,1]*np.roll(edges,-1,axis=0)[:,0]
    if not (np.all(turns>1e-8) or np.all(turns < -1e-8)):
        raise ValueError('support quad must be convex and ordered')
    center = uv.mean(axis=0)
    half = np.minimum(center-uv.min(axis=0),uv.max(axis=0)-center)
    for _ in range(100):
        corners_uv = center+np.array([[-1,-1],[1,-1],[1,1],[-1,1]])*half
        delta = corners_uv[:,None,:]-uv
        cross = edges[None,:,0]*delta[:,:,1]-edges[None,:,1]*delta[:,:,0]
        if np.all(cross >= 0) or np.all(cross <= 0):
            lower,upper = np.ceil(center-half).astype(int),np.floor(center+half).astype(int)
            return validate_roi(tuple(map(int,[*lower,*upper])),k)
        half *= .95
    raise ValueError('cannot inscribe support ROI')
