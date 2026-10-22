"""Conservative free-volume checks from registered depth, not scene bounds.

Every small volume cell must project fully inside at least one fresh depth image
and lie in front of every depth sample in its padded image rectangle. Occluded,
invalid and unobserved cells refuse clearance. This is a bounded sensor-resolution
test, not a collision planner, surface reconstruction or ground-truth query.
"""
from dataclasses import dataclass
from itertools import product
import math

import numpy as np

from twin.restock_camera.geometry import Intrinsics,rigid


@dataclass(frozen=True)
class DepthView:
    camera_id: str
    frame_id: str
    stamp: float
    transform_stamp: float
    epoch: int
    depth: np.ndarray
    intrinsics: Intrinsics
    base_from_camera: np.ndarray


def measure_clearance(views, *, base_from_support, xy_bounds, required_height,
                      location, epoch, now, collision_time, parked,
                      support_plane_stamp, plane_tolerance=.002,
                      resolution=.02, max_cells=20000):
    """Test a measured rectangular support footprint above its contact band.

    xy_bounds are in the measured support frame. Caller must establish that this
    rectangle fits the full sensed support polygon; a label mask cannot supply it.
    The bottom 5 mm contact band requires the separately fresh support-plane fit.
    The candidate height is certified only when every queried cell is observed.
    """
    values=(*xy_bounds,required_height,now,collision_time,support_plane_stamp,plane_tolerance,resolution)
    if (not all(math.isfinite(x) for x in values) or len(xy_bounds)!=4
            or parked is not True or type(epoch) is not int or epoch<=0
            or not 0<=now-collision_time<=.5 or not 0<=now-support_plane_stamp<=.5
            or not 0<plane_tolerance<=.002 or not 0<resolution<=.02
            or required_height<=.005):
        raise ValueError('fresh measured support, parking and incorporated collision map required')
    a,b,c,d=xy_bounds
    if a>=c or b>=d or not location:
        raise ValueError('nonempty measured support footprint required')
    support=rigid(base_from_support)
    valid=[]
    for view in views:
        k=view.intrinsics
        if (view.epoch!=epoch or not view.frame_id or not view.camera_id
                or not all(math.isfinite(x) for x in (view.stamp,view.transform_stamp))
                or not 0<=now-view.stamp<=.5 or abs(view.stamp-view.transform_stamp)>.02):
            continue
        depth=np.asarray(view.depth,float)
        if depth.shape!=(k.height,k.width):
            raise ValueError('depth image differs from calibrated dimensions')
        valid.append((view,depth,np.linalg.inv(rigid(view.base_from_camera))@support))
    if not valid:
        raise ValueError('no fresh registered depth views for this parked epoch')
    low=np.array([a,b,.005]);high=np.array([c,d,required_height])
    counts=np.ceil((high-low)/resolution).astype(int)
    if int(np.prod(counts))>max_cells:
        raise ValueError('requested sensor volume exceeds bounded cell budget')
    steps=(high-low)/counts
    offsets=np.array(list(product((0.,1.),repeat=3)))
    used={}
    unobserved=0
    for index in product(*(range(int(n)) for n in counts)):
        corners=low+(np.asarray(index)+offsets)*steps
        covered=False
        for view,depth,camera_from_support in valid:
            cloud=corners@camera_from_support[:3,:3].T+camera_from_support[:3,3]
            k=view.intrinsics
            if np.any(cloud[:,2]<=k.near) or np.any(cloud[:,2]>=k.far):
                continue
            u=k.fx*cloud[:,0]/cloud[:,2]+k.cx
            v=k.fy*cloud[:,1]/cloud[:,2]+k.cy
            x1,y1=int(np.floor(u.min()))-1,int(np.floor(v.min()))-1
            x2,y2=int(np.ceil(u.max()))+2,int(np.ceil(v.max()))+2
            if x1<0 or y1<0 or x2>k.width or y2>k.height:
                continue
            patch=depth[y1:y2,x1:x2]
            if (not np.isfinite(patch).all() or np.any(patch<=k.near)
                    or np.any(patch>=k.far) or patch.min()<cloud[:,2].max()+plane_tolerance):
                continue
            used[view.camera_id]=(view.frame_id,view.stamp)
            covered=True
            break
        if not covered:
            unobserved+=1
    if unobserved:
        raise ValueError(f'{unobserved} clearance cells occluded, occupied or unobserved')
    stamp=min(v[1] for v in used.values())
    return dict(source='sensor',location=location,epoch=epoch,simulation_time=stamp,
        observation_id='clearance:'+str(epoch)+':'+','.join(k+':'+v[0] for k,v in sorted(used.items())),
        free_dimensions=(c-a,d-b,required_height),collision_time=collision_time,
        support_plane_stamp=support_plane_stamp,verified_above_support_m=.005,
        contact_band_evidence='separate fresh support-plane fit',
        cells=int(np.prod(counts)),cell_size_max_m=float(steps.max()),
        model='padded depth-ray free-volume bounds; not physics proof')
