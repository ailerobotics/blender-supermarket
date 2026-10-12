"""Build the same obstacle map for launch, command gating and CPU validation."""
import math
from .core import GridMap


def environment_grid(manifest, scenario=None):
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
