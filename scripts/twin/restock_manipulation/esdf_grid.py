"""Signed-distance ESDF snapshot + swept-volume clearance query (pure numpy, CPU).

An :class:`EsdfSnapshot` is a frozen read of an nvblox Euclidean Signed Distance Field
cropped to the arm base frame (``ur_base_link``): a 3-D grid of signed distances in metres
(positive = free space, the distance to the nearest surface; ``<= 0`` = at/inside an
obstacle), plus the grid origin, the isotropic voxel size, the ESDF incorporation stamp and
the source frame. It is the geometry/measurement core the perception collision gate
(:mod:`collision_gate`) consults to decide whether a planned arm leg's swept gripper +
welded-carton volume clears what the RGB-D cameras have mapped.

Conservative by construction, exactly like
:func:`twin.restock_manipulation.isaac_motion.IsaacCadMotion.update_esdf` and
:mod:`twin.restock.depth_clearance` ("absence is not free space"):

* **Unobserved voxels are NOT free space.** nvblox marks an unobserved voxel with a large
  negative sentinel (``esdf_and_gradients_unobserved_value``, typically ``-1000``); any value
  ``<= UNOBSERVED_FLOOR`` or non-finite, and any part of a query box that falls outside the
  grid, is reported as *unobserved* rather than clear. The gate decides what to do with an
  unobserved fraction (hold for a fresher map, refuse, or warn) — this module only measures.
* **Minimum, not mean.** A swept volume's clearance is the single nearest surface over all of
  its voxels, so one occupied voxel fails the leg.

No Isaac, ROS or GPU. The nvblox service response is turned into a snapshot by
:meth:`EsdfSnapshot.from_service_grid` (host path, see :mod:`esdf_source`); tests and the
offline collect replay synthesise a snapshot analytically (see
:class:`twin.restock_manipulation.esdf_source.AnalyticBoxEsdfSource`).
"""
from __future__ import annotations

from dataclasses import dataclass, field
import itertools
import math
from typing import Optional

import numpy as np

#: Any ESDF value at or below this (metres) is the nvblox unobserved sentinel, not a real
#: signed distance. nvblox's default ``esdf_and_gradients_unobserved_value`` is -1000; a safe
#: floor well below any physical clearance distinguishes it from a genuine deep-negative
#: (inside an obstacle) distance, which never reaches -999 m in a bounded workspace.
UNOBSERVED_FLOOR = -999.0


@dataclass(frozen=True)
class AabbQuery:
    """Result of querying one axis-aligned box against a snapshot.

    * ``min_observed`` -- the minimum signed distance over the OBSERVED voxels the box spans
      (``+inf`` when the box spanned no observed voxel at all).
    * ``unobserved`` / ``total`` -- unobserved vs total voxels the box spans (out-of-grid span
      counts as unobserved). ``unobserved_fraction`` is the ratio.
    * ``nearest_index`` -- grid ``(i, j, k)`` of the minimum-distance observed voxel, or
      ``None``; use :meth:`EsdfSnapshot.label_at` to name the nearest body when labels exist.
    """
    min_observed: float
    unobserved: int
    total: int
    nearest_index: Optional[tuple]

    @property
    def unobserved_fraction(self) -> float:
        return self.unobserved / self.total if self.total else 1.0

    @property
    def fully_unobserved(self) -> bool:
        return not math.isfinite(self.min_observed)


@dataclass(frozen=True)
class EsdfSnapshot:
    """Immutable signed-distance field cropped to ``frame`` (metres).

    :param distances: ``(nx, ny, nz)`` float array of signed distances; ``origin`` is the
        world position of voxel ``(0, 0, 0)``'s minimum corner and ``voxel_size`` the isotropic
        edge length, so voxel ``(i, j, k)`` covers
        ``origin + (i, j, k) * voxel_size .. origin + (i+1, j+1, k+1) * voxel_size``.
    :param labels: optional ``(nx, ny, nz)`` int array of body ids parallel to ``distances``
        (``0`` = none); ``labelmap`` maps an id to a body name. Lets a verdict report the
        nearest body "when the ESDF can say"; both default empty (nearest body = ``None``).
    """
    distances: np.ndarray
    origin: tuple
    voxel_size: float
    stamp: float
    frame: str = 'ur_base_link'
    unobserved_floor: float = UNOBSERVED_FLOOR
    labels: Optional[np.ndarray] = None
    labelmap: dict = field(default_factory=dict)

    def __post_init__(self):
        grid = np.asarray(self.distances, np.float32)
        if grid.ndim != 3 or grid.size == 0:
            raise ValueError('ESDF distances must be a non-empty 3-D grid')
        origin = tuple(float(x) for x in self.origin)
        if len(origin) != 3 or not all(math.isfinite(x) for x in origin):
            raise ValueError('ESDF origin must be a finite 3-vector')
        if not (math.isfinite(self.voxel_size) and self.voxel_size > 0):
            raise ValueError('ESDF voxel_size must be a positive finite metre length')
        if self.labels is not None:
            lab = np.asarray(self.labels)
            if lab.shape != grid.shape:
                raise ValueError('ESDF labels must match the distance grid shape')
            object.__setattr__(self, 'labels', lab)
        # Normalise the stored grid/origin to validated copies (frozen dataclass).
        object.__setattr__(self, 'distances', grid)
        object.__setattr__(self, 'origin', origin)
        object.__setattr__(self, 'voxel_size', float(self.voxel_size))
        object.__setattr__(self, 'stamp', float(self.stamp))

    @property
    def shape(self) -> tuple:
        return self.distances.shape

    @classmethod
    def from_service_grid(cls, grid, origin, voxel_size, stamp, *, frame='ur_base_link',
                          unobserved_floor=UNOBSERVED_FLOOR):
        """Build a snapshot from an already-reshaped nvblox ``get_esdf_and_gradient`` grid.

        Mirrors :class:`twin.restock_manipulation.ros_adapter.NvbloxWorldBridge.poll` reshape:
        ``grid`` is the ``(x, y, z)`` signed-distance array, ``origin`` the response
        ``origin_m`` and ``voxel_size`` the response voxel size (``aabb_size / dims``). nvblox's
        unobserved sentinel is preserved as-is and handled by the floor at query time.
        """
        return cls(np.asarray(grid, np.float32), tuple(origin), float(voxel_size), float(stamp),
                   frame=frame, unobserved_floor=float(unobserved_floor))

    def _observed(self, block: np.ndarray) -> np.ndarray:
        return np.isfinite(block) & (block > self.unobserved_floor)

    def clearance_over_box(self, low, high) -> AabbQuery:
        """Minimum observed signed distance + unobserved fraction over an AABB (metres).

        ``low``/``high`` are world-frame corners in ``self.frame``. Any span outside the grid,
        and any in-grid voxel that is non-finite or at/below ``unobserved_floor``, counts as
        unobserved (conservative). The returned ``min_observed`` is over the observed voxels
        only; combine it with ``unobserved_fraction`` in the gate.
        """
        low = np.asarray(low, float)
        high = np.asarray(high, float)
        if low.shape != (3,) or high.shape != (3,) or not np.isfinite([low, high]).all():
            raise ValueError('finite 3-vector AABB corners required')
        origin = np.asarray(self.origin, float)
        n = np.asarray(self.shape, int)
        lo = np.minimum(low, high)
        hi = np.maximum(low, high)
        i0 = np.floor((lo - origin) / self.voxel_size).astype(int)
        i1 = np.ceil((hi - origin) / self.voxel_size).astype(int)
        i1 = np.maximum(i1, i0 + 1)                       # at least one voxel thick
        total = int(np.prod(i1 - i0))
        c0 = np.clip(i0, 0, n)
        c1 = np.clip(i1, 0, n)
        if np.any(c1 <= c0):                              # box does not overlap the grid at all
            return AabbQuery(float('inf'), total, total, None)
        block = self.distances[c0[0]:c1[0], c0[1]:c1[1], c0[2]:c1[2]]
        observed = self._observed(block)
        n_observed = int(observed.sum())
        unobserved = total - n_observed
        if not n_observed:
            return AabbQuery(float('inf'), unobserved, total, None)
        masked = np.where(observed, block, np.inf)
        flat = int(np.argmin(masked))
        local = np.unravel_index(flat, masked.shape)
        nearest = tuple(int(c0[a] + local[a]) for a in range(3))
        return AabbQuery(float(block[local]), unobserved, total, nearest)

    def clearance_over_boxes(self, boxes) -> AabbQuery:
        """Aggregate :meth:`clearance_over_box` over a swept union of AABBs.

        The leg's clearance is the single nearest surface over the whole swept volume, so this
        is the minimum ``min_observed`` and the summed unobserved/total across all boxes.
        """
        min_obs = float('inf')
        unobserved = total = 0
        nearest = None
        any_box = False
        for low, high in boxes:
            any_box = True
            q = self.clearance_over_box(low, high)
            unobserved += q.unobserved
            total += q.total
            if q.min_observed < min_obs:
                min_obs = q.min_observed
                nearest = q.nearest_index
        if not any_box:
            raise ValueError('a swept volume requires at least one AABB')
        return AabbQuery(min_obs, unobserved, total, nearest)

    def label_at(self, index) -> Optional[str]:
        """Body name at grid index ``(i, j, k)`` from the optional label grid, or ``None``."""
        if index is None or self.labels is None:
            return None
        lid = int(self.labels[tuple(index)])
        return self.labelmap.get(lid)


# ------------------------------------------------------------------ swept geometry (pure) --
def quat_to_rotation(quat) -> np.ndarray:
    """3x3 rotation from a unit xyzw quaternion (same convention as :mod:`clearance`)."""
    q = np.asarray(quat, float)
    if q.shape != (4,) or not np.isfinite(q).all():
        raise ValueError('finite xyzw quaternion required')
    norm = np.linalg.norm(q)
    if not norm > 0:
        raise ValueError('non-zero quaternion required')
    x, y, z, w = q / norm
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def rotation_to_quat(rotation) -> np.ndarray:
    """Unit xyzw quaternion from a 3x3 rotation matrix (for FK -> pose)."""
    m = np.asarray(rotation, float)
    trace = np.trace(m)
    if trace > 0:
        s = math.sqrt(trace + 1.0) * 2
        w = 0.25 * s
        x = (m[2, 1] - m[1, 2]) / s
        y = (m[0, 2] - m[2, 0]) / s
        z = (m[1, 0] - m[0, 1]) / s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2
        w = (m[2, 1] - m[1, 2]) / s
        x = 0.25 * s
        y = (m[0, 1] + m[1, 0]) / s
        z = (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2
        w = (m[0, 2] - m[2, 0]) / s
        x = (m[0, 1] + m[1, 0]) / s
        y = 0.25 * s
        z = (m[1, 2] + m[2, 1]) / s
    else:
        s = math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2
        w = (m[1, 0] - m[0, 1]) / s
        x = (m[0, 2] + m[2, 0]) / s
        y = (m[1, 2] + m[2, 1]) / s
        z = 0.25 * s
    q = np.array([x, y, z, w])
    return q / np.linalg.norm(q)


def nlerp(qa, qb, t) -> np.ndarray:
    """Normalised linear interpolation between two xyzw quaternions (shortest arc)."""
    a = np.asarray(qa, float)
    b = np.asarray(qb, float)
    if np.dot(a, b) < 0:
        b = -b
    q = (1 - t) * a + t * b
    return q / np.linalg.norm(q)


def pose_aabb(pose7, pad_min, pad_max) -> tuple:
    """World AABB of an oriented box ``[pad_min, pad_max]`` placed at a 7-tuple pose.

    ``pose7`` is ``(x, y, z, qx, qy, qz, qw)``; the box is expressed in the pose's frame (e.g.
    the gripper ``gripper_pad_tcp`` envelope). Exact (all 8 rotated corners), like
    :func:`twin.restock_manipulation.clearance.linear_tool_sweep`.
    """
    pose = np.asarray(pose7, float)
    if pose.shape != (7,) or not np.isfinite(pose).all():
        raise ValueError('finite 7-tuple pose required')
    rotation = quat_to_rotation(pose[3:])
    corners = np.array(list(itertools.product(*zip(pad_min, pad_max))))
    world = corners @ rotation.T + pose[:3]
    return world.min(axis=0), world.max(axis=0)


def swept_pose_aabbs(poses, pad_min, pad_max):
    """List of per-pose AABBs for an oriented box swept through a sequence of poses.

    The caller (the gate) samples the leg start..goal into ``poses`` (interpolating rotation
    with :func:`nlerp`) and this returns one conservative AABB per sample; their union is the
    swept volume queried against the snapshot. A denser sampling only tightens the bound.
    """
    boxes = [pose_aabb(p, pad_min, pad_max) for p in poses]
    if not boxes:
        raise ValueError('at least one swept pose required')
    return boxes


def combined_envelope(pad_min, pad_max, carton_dims=None, *, grasp_axis=2, face_offset=0.0):
    """Fold a held carton box into the gripper envelope ``[pad_min, pad_max]`` (pad frame).

    With a top grasp the suction pad faces +``grasp_axis`` (default +Z, ``tcp_quaternion``
    ``(1,0,0,0)``) and the welded carton hangs from the pad face outward along +``grasp_axis``
    by its height, centred on the other two axes. The returned box is the element-wise union of
    the gripper envelope and that carton box, so the swept volume conservatively includes the
    carried unit. ``carton_dims`` ``None`` (not held) returns the bare envelope unchanged.
    """
    lo = np.asarray(pad_min, float).copy()
    hi = np.asarray(pad_max, float).copy()
    if carton_dims is None:
        return tuple(lo), tuple(hi)
    dims = np.asarray(carton_dims, float)
    if dims.shape != (3,) or not np.isfinite(dims).all() or np.any(dims <= 0):
        raise ValueError('finite positive carton dimensions required')
    half = dims / 2.0
    c_lo = -half.copy()
    c_hi = half.copy()
    # The carton extends from the pad face (+grasp_axis side) outward by its full height.
    c_lo[grasp_axis] = float(face_offset)
    c_hi[grasp_axis] = float(face_offset) + float(dims[grasp_axis])
    return tuple(np.minimum(lo, c_lo)), tuple(np.maximum(hi, c_hi))
