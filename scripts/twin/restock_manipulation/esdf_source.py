"""ESDF snapshot providers -- the seam between the collision gate and a distance field.

The gate (:mod:`collision_gate`) asks an :class:`EsdfSource` for the live
:class:`twin.restock_manipulation.esdf_grid.EsdfSnapshot` cropped to the arm base frame over
a workspace AABB. Three concrete providers:

* :class:`NvbloxEsdfSource` -- the real host path. A ``get_esdf_and_gradient`` client on the
  host rclpy node (same request/response as
  :class:`twin.restock_manipulation.ros_adapter.NvbloxWorldBridge`), so the gate measures the
  SAME nvblox ESDF cuMotion plans against, cropped to ``ur_base_link``. Lazy ROS import; never
  fabricates a field -- an unavailable mapper / unready service / failed call returns ``None``
  and the gate holds.
* :class:`AnalyticBoxEsdfSource` -- a CPU synthetic field: signed distance to a union of boxes
  voxelised over a workspace grid, fully observed. The CPU proof plants obstacles with it, the
  offline collect replay builds the known Rollwagen cage with it, and the collision-bench lane
  (sim-arm-collision-bench-v1) drives it with scripted static + MOVING obstacles (pass
  ``boxes`` as a ``boxes(now)`` callable -- the dynamic-layer seam).
* :class:`ReplayEsdfSource` -- replays a grid captured to ``.npz`` from a bounded GPU run, so a
  recorded ESDF can be re-queried offline.

None of these decide anything; they only measure what is mapped. "Unknown stays occupied" is
enforced in :class:`esdf_grid.EsdfSnapshot`, not here.
"""
from __future__ import annotations

import math
from typing import Callable, Optional, Protocol, Sequence, Union

import numpy as np

from . import esdf_grid


class EsdfSource(Protocol):
    """Provider of the live ESDF snapshot. ``snapshot`` returns ``None`` when unavailable."""

    def snapshot(self, now: float) -> Optional[esdf_grid.EsdfSnapshot]:
        ...


# ----------------------------------------------------------------------- host nvblox path --
#: The default workspace AABB cropped from nvblox for the gate, in ``ur_base_link`` metres.
#: Scoped to the reachable arm envelope WITHIN the measured mission ESDF grid (shape
#: [409, 217, 257] @ 0.01 m, origin [-2.00, -0.08, -0.48]; sim-perception-architecture-s1): the
#: reach target is at base +Y ~1 m, so this spans the forward/left workspace an arm leg sweeps
#: through. It deliberately does NOT extend to the rollwagen (base y ~ -0.8 m) -- that body is
#: outside the real grid (y_min -0.08) and is handled as a known cuMotion world obstacle
#: (:mod:`cart_world_obstacle`), not by the perceived ESDF. Config-overridable per run.
DEFAULT_WORKSPACE_ORIGIN_M = (-1.0, -0.08, -0.35)
DEFAULT_WORKSPACE_SIZE_M = (2.0, 1.7, 2.0)


class NvbloxEsdfSource:
    """``get_esdf_and_gradient`` client cropped to ``ur_base_link`` for the gate (host node).

    Constructed inside the host rclpy process next to the controller (see :mod:`ros_host`); it
    issues one bounded request per :meth:`snapshot`, waiting a short budget for the response.
    Requires the isolated sim domain/clock the rest of the host stack uses. ``map_stamp`` is the
    nvblox incorporation stamp supplied by the caller (the mirrored ``collision_time``), never
    wall-clock.
    """

    def __init__(self, node, *, origin=DEFAULT_WORKSPACE_ORIGIN_M, size=DEFAULT_WORKSPACE_SIZE_M,
                 frame='ur_base_link', spin_budget_s=0.2, require_sim_domain=True):
        self.node = node
        self.origin = tuple(float(x) for x in origin)
        self.size = tuple(float(x) for x in size)
        self.frame = frame
        self.spin_budget_s = float(spin_budget_s)
        if require_sim_domain:
            if node.context.get_domain_id() != 88 or not node.get_parameter('use_sim_time').value:
                raise RuntimeError('nvblox ESDF source must use the isolated simulation clock/domain')
        from nvblox_msgs.srv import EsdfAndGradients
        self._srv_type = EsdfAndGradients
        self.client = node.create_client(
            EsdfAndGradients, '/restock_sim/nvblox_node/get_esdf_and_gradient')

    def snapshot(self, now: float, map_stamp: Optional[float] = None
                 ) -> Optional[esdf_grid.EsdfSnapshot]:
        """One cropped ESDF read, or ``None`` when the mapper/service is not ready."""
        stamp = float(now if map_stamp is None else map_stamp)
        if not (math.isfinite(stamp) and stamp >= 0) or not self.client.service_is_ready():
            return None
        import rclpy
        from geometry_msgs.msg import Point, Vector3
        req = self._srv_type.Request()
        req.update_esdf, req.use_aabb = True, True
        req.frame_id = self.frame
        req.aabb_min_m = Point(x=self.origin[0], y=self.origin[1], z=self.origin[2])
        req.aabb_size_m = Vector3(x=self.size[0], y=self.size[1], z=self.size[2])
        future = self.client.call_async(req)
        rclpy.spin_until_future_complete(self.node, future, timeout_sec=self.spin_budget_s)
        if not future.done():
            return None
        response = future.result()
        if response is None:
            return None
        array = response.esdf_and_gradients
        dims = {d.label: int(d.size) for d in array.layout.dim}
        try:
            shape = tuple(dims[axis] for axis in ('x', 'y', 'z'))
        except KeyError:
            return None
        grid = np.asarray(array.data, np.float32).reshape(shape)
        voxel = self.size[0] / shape[0] if shape[0] else 0.0
        origin = (response.origin_m.x, response.origin_m.y, response.origin_m.z)
        return esdf_grid.EsdfSnapshot.from_service_grid(grid, origin, voxel, stamp, frame=self.frame)


# --------------------------------------------------------------------- CPU synthetic field --
def _box_sdf(points: np.ndarray, bmin: np.ndarray, bmax: np.ndarray) -> np.ndarray:
    """Exact signed distance (m) from each point to an AABB ``[bmin, bmax]`` (<0 inside)."""
    center = (bmin + bmax) / 2.0
    half = (bmax - bmin) / 2.0
    d = np.abs(points - center) - half
    outside = np.linalg.norm(np.maximum(d, 0.0), axis=-1)
    inside = np.minimum(np.max(d, axis=-1), 0.0)
    return outside + inside


class AnalyticBoxEsdfSource:
    """Fully-observed signed-distance field of a union of AABB boxes over a workspace grid.

    ``boxes`` is either a static sequence of ``(low, high)`` AABBs (``ur_base_link`` metres) or
    a callable ``boxes(now) -> sequence`` for moving obstacles (the dynamic-layer / bench seam).
    Every voxel is observed (synthetic), so the gate's unobserved path is never hit -- use this
    for planted-obstacle tests, the known-geometry collect replay and the bench. ``labels`` (a
    parallel list of body names per box) lets a verdict report the nearest body.
    """

    def __init__(self, boxes: Union[Sequence, Callable], *, origin=DEFAULT_WORKSPACE_ORIGIN_M,
                 size=DEFAULT_WORKSPACE_SIZE_M, voxel_size=0.03, labels=None, frame='ur_base_link',
                 clamp=5.0):
        self._boxes = boxes
        self.origin = tuple(float(x) for x in origin)
        self.size = tuple(float(x) for x in size)
        self.voxel_size = float(voxel_size)
        self.labels = list(labels) if labels is not None else None
        self.frame = frame
        self.clamp = float(clamp)
        n = [max(1, int(round(s / self.voxel_size))) for s in self.size]
        # Voxel CENTRE coordinates per axis.
        axes = [self.origin[a] + (np.arange(n[a]) + 0.5) * self.voxel_size for a in range(3)]
        gx, gy, gz = np.meshgrid(axes[0], axes[1], axes[2], indexing='ij')
        self._centres = np.stack([gx, gy, gz], axis=-1).reshape(-1, 3)
        self._shape = tuple(n)

    def _resolve_boxes(self, now):
        boxes = self._boxes(now) if callable(self._boxes) else self._boxes
        return list(boxes)

    def snapshot(self, now: float) -> Optional[esdf_grid.EsdfSnapshot]:
        boxes = self._resolve_boxes(now)
        pts = self._centres
        dist = np.full(pts.shape[0], self.clamp, np.float32)
        label_grid = np.zeros(pts.shape[0], np.int32) if self.labels is not None else None
        labelmap = {}
        for i, (low, high) in enumerate(boxes):
            sdf = _box_sdf(pts, np.asarray(low, float), np.asarray(high, float))
            closer = sdf < dist
            dist = np.where(closer, sdf, dist)
            if label_grid is not None:
                lid = i + 1
                name = self.labels[i] if i < len(self.labels) else 'box_%d' % i
                labelmap[lid] = name
                label_grid[closer] = lid
        dist = np.clip(dist, -self.clamp, self.clamp).reshape(self._shape).astype(np.float32)
        labels = label_grid.reshape(self._shape) if label_grid is not None else None
        return esdf_grid.EsdfSnapshot(dist, self.origin, self.voxel_size, float(now),
                                      frame=self.frame, labels=labels, labelmap=labelmap)


class ReplayEsdfSource:
    """Serve a single ESDF grid captured to ``.npz`` (keys ``distances``, ``origin``,
    ``voxel_size``) from a bounded GPU run, stamped to the query time for offline re-query."""

    def __init__(self, npz_path, *, frame='ur_base_link'):
        data = np.load(npz_path)
        self._grid = np.asarray(data['distances'], np.float32)
        self._origin = tuple(float(x) for x in data['origin'])
        self._voxel = float(data['voxel_size'])
        self.frame = frame

    def snapshot(self, now: float) -> Optional[esdf_grid.EsdfSnapshot]:
        return esdf_grid.EsdfSnapshot.from_service_grid(
            self._grid, self._origin, self._voxel, float(now), frame=self.frame)


class StaticSnapshotSource:
    """Return a fixed pre-built snapshot re-stamped to ``now`` (tests / fixed scenarios)."""

    def __init__(self, snapshot: esdf_grid.EsdfSnapshot):
        self._snapshot = snapshot

    def snapshot(self, now: float) -> Optional[esdf_grid.EsdfSnapshot]:
        from dataclasses import replace
        return replace(self._snapshot, stamp=float(now))
