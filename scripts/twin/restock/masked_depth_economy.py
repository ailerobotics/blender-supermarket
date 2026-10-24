"""Opt-in render-economy for the eye-in-hand masked-depth ARM FEED (sim-robust-arm-feed-economy-v1).

The honest live_pose transfer enables the arm (eye-in-hand) masked-depth feed: an extra RTX render
product (424x240 depth + instance-id segmentation) whose frame is rig-masked and shipped to the host
nvblox ESDF on EVERY manipulation tick. That extra render + the per-tick masking/IPC/integration is a
wall cost the simulator pays during collect/transfer, where the pacing drops to ~0.08 sim/wall
(~12x slower than wall) and the host's wall timers trip (the coldlive 4-sample whack-a-mole, 2026-10-23).

This module is the PURE, CPU-tested policy for an opt-in ``masked_depth.economy`` block that lowers the
feed's cost to what the collision/grasp solver actually needs at the recorded working distances, WITHOUT
touching the decode (QR) cameras, the walkthrough, or the default path. Two composable levers:

* ``resolution_scale`` (0 < s <= 1, default 1.0): render the arm camera's depth product at ``s`` of its
  native resolution. The camera APERTURES are unchanged, so the intrinsics (fx, fy, cx, cy) scale with
  the resolution and the field of view / reprojection geometry is preserved -- nvblox back-projects the
  coarser depth to the SAME 3-D rays, only sampled more sparsely. The only quality cost is pixel
  quantization: a target localizes to within half a (scaled) pixel, i.e. ``0.5 * Z / fx_scaled`` metres
  of lateral error at working distance ``Z``. :func:`reprojection_shift_m` states that bound so a CPU
  test can hold it under the ESDF voxel half-size (the solver's own spatial resolution).
* ``publish_interval_s`` (sim seconds, default None=every tick): the FRESHNESS-ON-SIM-CLOCK rate cap.
  The bridge publishes at most one masked-depth frame per ``publish_interval_s`` of SIM time, skipping
  (never shipping a stale buffer) between -- cutting the per-tick rig-mask + socket IPC + nvblox
  integration from ~60 Hz sim to ``1/interval`` Hz. It is bounded BELOW the nvblox depth-freshness
  window so the ESDF/parked-station map never lapses (the freshness principle, PR 201/204/205/226/228).

Absent (or an empty / ``{"enabled": false}`` block) => :class:`EconomySettings` inactive => the camera
renders native-resolution and the bridge publishes every tick => byte-identical to before. Pure: no
Isaac/USD/ROS import, so the policy + the pose-error acceptance are unit-tested off the live stage.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, fields

import numpy as np

# nvblox integrates a depth frame only while its stamp is within this window of the map clock; the
# per-feed publish interval is kept strictly below it so a rate-capped feed never starves the ESDF.
NVBLOX_DEPTH_WINDOW_S = 0.5


@dataclass(frozen=True)
class EconomySettings:
    """Resolved arm masked-depth economy levers. Defaults are the legacy (no-economy) behaviour."""
    resolution_scale: float = 1.0
    publish_interval_s: float | None = None

    @property
    def active(self) -> bool:
        """True when any lever differs from the legacy per-tick, native-resolution feed."""
        return self != EconomySettings()

    def describe(self) -> str:
        """One boot-log line naming every lever, so a wrongly-set switch is visible at boot."""
        return ('MASKED_DEPTH_ECONOMY active=%s resolution_scale=%g publish_interval_s=%s'
                % (self.active, self.resolution_scale, self.publish_interval_s))

    def to_summary(self) -> dict:
        """mission.json provenance: the levers as a plain dict (only when active)."""
        return {f.name: getattr(self, f.name) for f in fields(self)}


def from_block(block, *, nvblox_window_s: float = NVBLOX_DEPTH_WINDOW_S) -> EconomySettings:
    """Resolve a ``masked_depth.economy`` block (absent/empty/disabled => inactive defaults).

    Fails CLOSED on an unknown key or an invalid value (a typo refuses the boot rather than silently
    running the slow / wrong path). ``resolution_scale`` must be in (0, 1]; ``publish_interval_s`` must
    be positive and strictly below the nvblox depth-freshness window so a rate-capped feed stays fresh.
    """
    if block is None or block is False:
        return EconomySettings()
    if not isinstance(block, dict):
        raise ValueError('masked_depth.economy must be an object such as '
                         '{"resolution_scale": 0.5, "publish_interval_s": 0.1}')
    if block.get('enabled') is False:
        return EconomySettings()
    known = {f.name for f in fields(EconomySettings)} | {'enabled'}
    unknown = set(block) - known
    if unknown:
        raise ValueError('masked_depth.economy has unknown keys %s; known: %s'
                         % (sorted(unknown), sorted(known - {'enabled'})))

    scale = block.get('resolution_scale', 1.0)
    if isinstance(scale, bool) or not isinstance(scale, (int, float)) \
            or not math.isfinite(scale) or not 0.0 < scale <= 1.0:
        raise ValueError('masked_depth.economy.resolution_scale must be a number in (0, 1]')

    interval = block.get('publish_interval_s', None)
    if interval is not None:
        if isinstance(interval, bool) or not isinstance(interval, (int, float)) \
                or not math.isfinite(interval) or interval <= 0.0:
            raise ValueError('masked_depth.economy.publish_interval_s must be a positive number of '
                             'sim seconds, or null')
        if interval >= float(nvblox_window_s):
            raise ValueError('masked_depth.economy.publish_interval_s (%g s) must be below the nvblox '
                             'depth-freshness window (%g s) so the ESDF never lapses'
                             % (interval, nvblox_window_s))
        interval = float(interval)
    return EconomySettings(resolution_scale=float(scale), publish_interval_s=interval)


def scaled_resolution(width, height, scale) -> tuple[int, int]:
    """``(width, height)`` scaled by ``scale`` and rounded to EVEN pixels (>= 2 each).

    Even dimensions keep the render product well-formed; the floor-to-even matches the camera-rig
    convention (``native_cameras.CameraRig``) so a scaled arm feed sizes like every other product.
    ``scale == 1.0`` returns the native resolution unchanged (byte-identical camera authoring).
    """
    if not 0.0 < float(scale) <= 1.0:
        raise ValueError('scale must be in (0, 1]')
    w0, h0 = int(width), int(height)
    if w0 <= 0 or h0 <= 0:
        raise ValueError('native resolution must be positive')
    if float(scale) == 1.0:
        return w0, h0
    w = max(2, int(w0 * float(scale)) // 2 * 2)
    h = max(2, int(h0 * float(scale)) // 2 * 2)
    return w, h


def due_to_publish(now, last_publish_s, interval_s, *, tol_s: float = 1e-6) -> bool:
    """SIM-clock rate gate: should the bridge publish a masked-depth frame at sim time ``now``?

    ``interval_s is None`` (no economy) => always True (publish every tick, byte-identical). The first
    publish (``last_publish_s is None``) always fires. Otherwise fire only once ``interval_s`` of SIM
    time has elapsed since the last published frame. A tiny ``tol_s`` absorbs float dt accumulation so a
    frame exactly on the boundary is not deferred a whole extra tick. Monotonic-safe: a ``now`` that
    went backwards (a clock reset) also fires (treated as a fresh epoch), never wedging the feed off.
    """
    if interval_s is None:
        return True
    if last_publish_s is None:
        return True
    elapsed = float(now) - float(last_publish_s)
    if elapsed < 0.0:
        return True
    return elapsed >= float(interval_s) - float(tol_s)


# --- pose-error acceptance helpers (used by the CPU test on recorded working distances) ----------

def intrinsics_scaled(fx, fy, cx, cy, scale):
    """The arm camera's pinhole intrinsics after an economy ``resolution_scale`` (APERTURE preserved).

    Scaling the render-product resolution while holding the physical apertures scales every intrinsic
    uniformly by ``scale`` (``fx' = scale*fx`` ... ``cy' = scale*cy``), which preserves the field of
    view -- the reprojection rays are unchanged, only sampled more coarsely. This mirrors what Isaac
    computes for the authored camera, so the published ``camera_info`` K the host nvblox reads is this.
    """
    s = float(scale)
    return (float(fx) * s, float(fy) * s, float(cx) * s, float(cy) * s)


def reprojection_shift_m(fx, scale, working_distance_m) -> float:
    """Worst-case LATERAL reprojection error (metres) an economy ``resolution_scale`` adds at ``Z``.

    A point back-projects through its pixel CENTRE, so the localization error is bounded by half a
    (scaled) pixel: ``0.5 * Z / fx_scaled`` where ``fx_scaled = scale * fx``. This is the arm feed's
    contribution to the collision/grasp solver's spatial error; a CPU test holds it under the ESDF
    voxel half-size (the solver's own resolution) at the recorded working distances.
    """
    s = float(scale)
    if not 0.0 < s <= 1.0:
        raise ValueError('scale must be in (0, 1]')
    fx_scaled = s * float(fx)
    if fx_scaled <= 0.0:
        raise ValueError('fx must be positive')
    return 0.5 * float(working_distance_m) / fx_scaled


def downscale_depth(depth, scale):
    """Area-average downscale of a HxW metre depth image to the economy resolution (pure NumPy).

    Invalid returns (0 / non-finite, the nvblox no-return sentinel) are EXCLUDED from each output
    cell's average so a block straddling an object edge and free space is not pulled toward 0; a cell
    with no valid source pixel stays 0 (still no-return). This models what a native lower-resolution
    depth render yields: the surface depth is preserved, edges coarsen. ``scale == 1.0`` returns a copy.
    """
    d = np.asarray(depth, dtype=np.float64)
    if d.ndim != 2:
        raise ValueError('depth must be a 2-D image')
    h0, w0 = d.shape
    w, h = scaled_resolution(w0, h0, scale)
    if (w, h) == (w0, h0):
        return d.astype(np.float32).copy()
    out = np.zeros((h, w), dtype=np.float32)
    ys = (np.arange(h0) * h // h0)
    xs = (np.arange(w0) * w // w0)
    valid = np.isfinite(d) & (d > 0.0)
    for oy in range(h):
        rows = ys == oy
        if not rows.any():
            continue
        for ox in range(w):
            cols = xs == ox
            block = d[np.ix_(rows, cols)]
            bvalid = valid[np.ix_(rows, cols)]
            if bvalid.any():
                out[oy, ox] = float(block[bvalid].mean())
    return out
