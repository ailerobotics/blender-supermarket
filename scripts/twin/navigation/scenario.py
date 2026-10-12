"""Build the same obstacle map for launch, command gating and CPU validation."""
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
    return GridMap((bounds[0], bounds[2], bounds[1], bounds[3]), obstacles)
