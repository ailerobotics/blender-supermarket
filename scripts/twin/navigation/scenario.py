"""Build the same obstacle map for launch, command gating and CPU validation."""
import math
from .core import GridMap


def camera_mount_obstacles(camera_config):
    """Conservative XY bounds of every fixed camera collider, including booms.

    Moving parent mounts belong to the rig collision model, not this static map.
    Do not filter elevated solids: the transport arm/cart can strike them.
    """
    import numpy as np
    result=[]
    for solid in camera_config['solids']:
        if not solid.get('collision',False):
            continue
        if solid['parent'] not in ('map','robot_base','cart_chassis'):
            raise ValueError('unknown camera solid parent')
        matrix=np.asarray(solid['parent_from_solid'],float)
        size=np.asarray(solid['dimensions_m'],float)
        if (matrix.shape!=(4,4) or size.shape!=(3,) or not np.isfinite(matrix).all()
                or not np.isfinite(size).all() or np.any(size<=0)
                or not np.allclose(matrix[3],(0,0,0,1))
                or not np.allclose(matrix[:3,:3].T@matrix[:3,:3],np.eye(3),atol=1e-6)
                or not np.isclose(np.linalg.det(matrix[:3,:3]),1)):
            raise ValueError('invalid camera collision solid')
        if solid['parent']!='map':
            continue
        extent=np.abs(matrix[:3,:3])@(size/2)
        lo=matrix[:3,3]-extent;hi=matrix[:3,3]+extent
        result.append((float(lo[0]),float(lo[1]),float(hi[0]),float(hi[1])))
    return result


def environment_grid(manifest, scenario=None, camera_config=None):
    bounds = manifest['map_bounds']  # xmin,xmax,ymin,ymax
    obstacles = [(o['min'][0], o['min'][1], o['max'][0], o['max'][1])
                 for o in manifest['static_obstacles']]
    if scenario:
        staging = scenario['initial_staging']
        x, y = staging['support_center_xy']
        dx, dy, _ = staging['support_dimensions_m']
        if dx <= 0 or dy <= 0:
            raise ValueError('staging support dimensions must be positive')
        obstacles.append((x-dx/2, y-dy/2, x+dx/2, y+dy/2))
    if camera_config is not None:
        obstacles.extend(camera_mount_obstacles(camera_config))
    corridors=[]
    portal=manifest.get('portal')
    hints=manifest.get('route_hints',[])
    if portal and len(hints)>=3:
        x,y=portal['centre'][:2]
        for i in range(1,len(hints)-1):
            if math.hypot(hints[i][0]-x,hints[i][1]-y)<1e-6:
                dx=hints[i+1][0]-hints[i-1][0]
                dy=hints[i+1][1]-hints[i-1][1]
                if math.hypot(dx,dy)>1e-6:
                    corridors.append((x,y,math.atan2(dy,dx)))
                break
    return GridMap((bounds[0], bounds[2], bounds[1], bounds[3]), obstacles,
                   corridors=corridors)
