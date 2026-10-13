"""Conservative solid-volume octrees, refined only near transport conflicts.

Every retained cell has a circumsphere. Cells are discarded only with an analytic
radial bound or a separating convex-hull plane, so the complete source volume
(including mesh interiors) remains covered. Splitting preserves that invariant.
"""
from dataclasses import replace
import itertools

import numpy as np
from scipy.spatial import ConvexHull, cKDTree

from ..urdf_probe import geometry_world_points, rpy_to_matrix

ALGORITHM = 'local-convex-volume-octree-v2'
TOLERANCE = 1e-9


class VolumeCover:
    def __init__(self, geometry, resolver, pitch):
        if not np.isfinite(pitch) or not .001 <= pitch <= .2:
            raise ValueError('finite collision-cover pitch between 1 mm and 20 cm required')
        self.geometry, self.pitch = geometry, pitch
        if not np.isfinite([geometry.origin_xyz, geometry.origin_rpy]).all():
            raise ValueError('finite collision origin required')
        dimensions = (geometry.box_size if geometry.kind == 'box' else
                      (geometry.radius, geometry.length) if geometry.kind == 'cylinder' else
                      (geometry.radius,) if geometry.kind == 'sphere' else ())
        if dimensions and not all(np.isfinite(v) and v > 0 for v in dimensions):
            raise ValueError('positive finite collision dimensions required')
        local = replace(geometry, origin_xyz=(0., 0., 0.), origin_rpy=(0., 0., 0.))
        points = geometry_world_points(local, np.eye(4), resolver)
        if points is None or not len(points) or not np.isfinite(points).all():
            raise ValueError('unreadable or nonfinite source collision geometry')
        self.rotation, self.translation = rpy_to_matrix(geometry.origin_rpy), np.asarray(geometry.origin_xyz)
        self.low, self.high = points.min(0), points.max(0)
        self.vertex_count = len(points)
        self.normal = self.offset = None
        self.residual = 0.
        if geometry.kind == 'mesh':
            hull = ConvexHull(points)
            self.normal, self.offset = hull.equations[:, :3], hull.equations[:, 3]
            for start in range(0, len(points), 512):
                self.residual = max(self.residual, float(np.max(points[start:start+512] @ self.normal.T + self.offset)))
        elif geometry.kind not in ('box', 'cylinder', 'sphere'):
            raise ValueError('unsupported collision geometry ' + geometry.kind)
        self.grid_shape = np.maximum(1, np.ceil((self.high-self.low)/pitch).astype(int))
        if np.prod(self.grid_shape) > 500000:
            raise ValueError('bounded collision cover exceeds 500000 cells per element')
        step = (self.high-self.low)/self.grid_shape
        self.leaves, self.splits, self.discarded = {}, [], 0
        if geometry.kind != 'sphere':
            cells = {}
            for index, coord in enumerate(itertools.product(*(range(n) for n in self.grid_shape))):
                low = self.low + np.asarray(coord)*step
                cells[(index,)] = (low, low+step)
            self._install(cells)

    def _install(self, cells):
        keys = sorted(cells)
        for start in range(0, len(keys), 512):
            batch = keys[start:start+512]
            low = np.array([cells[key][0] for key in batch])
            high = np.array([cells[key][1] for key in batch])
            centers, half = (low+high)/2, (high-low)/2
            keep = np.ones(len(batch), bool)
            if self.normal is not None:
                minimum = centers @ self.normal.T + self.offset - half @ np.abs(self.normal).T
                keep = ~np.any(minimum > self.residual + TOLERANCE, axis=1)
            elif self.geometry.kind == 'cylinder':
                nearest = np.maximum(np.abs(centers[:, :2]) - half[:, :2], 0.)
                keep = np.linalg.norm(nearest, axis=1) <= self.geometry.radius + TOLERANCE
            for key, retained in zip(batch, keep):
                if retained:
                    self.leaves[key] = cells[key]
                else:
                    self.discarded += 1

    def refine(self, indices, minimum):
        """Split selected leaf cells; indices refer to the current sphere order."""
        if self.geometry.kind == 'sphere':
            return 0
        keys = sorted(self.leaves)
        children, split_count = {}, 0
        for index in sorted(set(indices)):
            key = keys[index]
            low, high = self.leaves[key]
            if np.max(high-low) <= minimum:
                continue
            middle = (low+high)/2
            for child, bits in enumerate(itertools.product((0, 1), repeat=3)):
                bits = np.asarray(bits, bool)
                children[key+(child,)] = (np.where(bits, middle, low), np.where(bits, high, middle))
            del self.leaves[key]
            self.splits.append(list(key))
            split_count += 1
        if len(self.leaves) + len(children) > 500000:
            raise ValueError('bounded adaptive collision cover exceeds 500000 cells')
        self._install(children)
        return split_count

    def result(self):
        if self.geometry.kind == 'sphere':
            spheres = [{'center': self.translation.tolist(), 'radius': float(self.geometry.radius)}]
        else:
            spheres = []
            for key in sorted(self.leaves):
                low, high = self.leaves[key]
                center = self.rotation @ ((low+high)/2) + self.translation
                spheres.append({'center': center.tolist(), 'radius': float(np.linalg.norm(high-low)/2 + TOLERANCE)})
        return spheres, {'algorithm': ALGORITHM, 'kind': self.geometry.kind, 'pitch_m': self.pitch,
                         'local_min': self.low.tolist(), 'local_max': self.high.tolist(),
                         'grid_shape': self.grid_shape.tolist(), 'source_vertex_count': self.vertex_count,
                         'sphere_count': len(spheres), 'discarded_cells': self.discarded,
                         'refined_parent_cells': self.splits,
                         'proof': 'exact source sphere or complete local volume partition; only provably exterior cells discarded; retained leaves enclosed by circumspheres'}


def cover_geometry(geometry, resolver, pitch):
    return VolumeCover(geometry, resolver, pitch).result()


def transport_overlaps(spheres, transforms, excluded, *, with_conflicts=False):
    """Sphere-pair checks after KD-tree broad phase; optionally return leaf indices."""
    world = {}
    for name, entries in spheres.items():
        centers = np.array([s['center'] for s in entries])
        tf = transforms[name]
        centers = centers @ tf[:3, :3].T + tf[:3, 3]
        radii = np.array([s['radius'] for s in entries])
        world[name] = (centers, radii, cKDTree(centers))
    overlaps, conflicts = [], {name: set() for name in spheres}
    for a, b in itertools.combinations(sorted(world), 2):
        if (a, b) in excluded:
            continue
        ac, ar, _ = world[a]
        bc, br, tree = world[b]
        if np.any((ac-ar[:, None]).min(0) > (bc+br[:, None]).max(0)) or np.any(
                (bc-br[:, None]).min(0) > (ac+ar[:, None]).max(0)):
            continue
        neighbors = tree.query_ball_point(ac, ar + br.max())
        penetration = 0.
        for i, indices in enumerate(neighbors):
            if indices:
                value = ar[i] + br[indices] - np.linalg.norm(ac[i] - bc[indices], axis=1)
                penetration = max(penetration, float(value.max()))
                hits = np.array(indices)[value > TOLERANCE]
                if len(hits):
                    conflicts[a].add(i)
                    conflicts[b].update(map(int, hits))
        if penetration > TOLERANCE:
            overlaps.append({'links': [a, b], 'max_sphere_penetration_m': penetration})
    return (overlaps, conflicts) if with_conflicts else overlaps
