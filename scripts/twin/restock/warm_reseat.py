"""Warm re-seed start-pose RESTORATION (sim-warm-reset-cart-reseat-v1).

A warm two-run's second attempt must begin from the SAME configured start pose the COLD start
establishes and verifies -- never from wherever attempt 1 left the train. The warm reset's
``reset_stage`` ``world.reset()`` does NOT reliably restore the maximal-coordinate cart body's
planar pose: the cart is a free rigid body (its casters are ``excludeFromArticulation`` maximal-
coordinate bodies), and attempt 1's host emergency stop at the pre-place step nudged it. warmacc4
measured the cart still at heading ``-99.1 deg`` after the reset, so PR 297's rest/heading verify
(``start_settle.heading_check``) correctly refused with ``START_STATE_SETTLE_MISMATCH`` (residual
9.1 deg > 5.0 deg) and the provider cold-booted (~215 s).

This module RESTORES the cart, the carton and the base to the configured start pose BEFORE the
settle + PR 297 verify runs, so the verify grades a train seated AT the dock rather than where
attempt 1 stopped. The restore is read-back verified: a body that cannot be moved back (no settable
handle, or a read-back still off the target) is refused LOUDLY by name (``WARM_RESEAT_UNRESTORED`` /
``START_STATE_BASE_UNRESTORED``) so the loop cold-boots with the reason named, never silently reusing
a mis-seated train.

Design:

* The CONFIGURED planar pose (x, y, yaw) of the cart and the base is read from the SAME source the
  cold build spawns from -- ``scenario['initial_staging']['cart_body_pose']`` / ``['robot_base_pose']``
  (``dock_align_scenario`` has already rewritten ``cart_body_pose`` to the configured dock heading for a
  dock-aligned run, so restoring to it is restoring to the exact pose the cold start verifies). The
  resting Z and the carton pose are CAPTURED at build time (right after the cold start-state is applied
  and verified, before any mission tick), because the backstock carton's spawn pose is composed from the
  planogram and is not an ``initial_staging`` field.
* The base policy is RESTORE, not refuse-on-any-drift: ``set_world_pose`` to the captured cold-start
  pose (which also un-flips an attitude flip) + zero velocity, reusing the same mechanism that restores
  the cart and carton and preserving the expensive ~205 s boot. We REFUSE (named) ONLY when the restore
  cannot be verified to have taken -- a body with no settable handle, or a read-back that is still off --
  which is the honest "cannot restore => cold boot" case the brief asks for.

PURE: every function takes injected body handles (``get_world_pose`` / ``set_world_pose`` /
``set_linear_velocity`` / ``set_angular_velocity``) and plain values, so capture, restore, the
read-back refusal and the drift record are all unit-tested on CPU with fakes. ``apply_start_state`` /
``verify_dock_seat`` keep ownership of the subsequent sim-time settle + verify on the Isaac side.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional


class WarmReseatRefused(RuntimeError):
    """A warm re-seed pose restore could not be verified to have taken (=> cold-boot fallback). Its
    message leads with a named code (``WARM_RESEAT_UNRESTORED`` / ``START_STATE_BASE_UNRESTORED``)."""


def wxyz_from_yaw(yaw):
    """Quaternion (w, x, y, z) for a rotation of ``yaw`` radians about world +Z."""
    return (math.cos(float(yaw) / 2.0), 0.0, 0.0, math.sin(float(yaw) / 2.0))


def yaw_from_wxyz(quat):
    """Yaw (radians) from a world quaternion (w, x, y, z), the only DoF a planar re-seat restores."""
    w, x, y, z = (float(v) for v in quat)
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def _wrap(a):
    return math.atan2(math.sin(a), math.cos(a))


def _as_tuple(seq, n):
    out = tuple(float(v) for v in seq)
    if len(out) != n or not all(math.isfinite(v) for v in out):
        raise ValueError(f'expected {n} finite components, got {seq!r}')
    return out


def read_pose(body):
    """``(position_xyz, quat_wxyz)`` of a live body via ``get_world_pose`` (Isaac returns wxyz).

    Works with numpy arrays or plain sequences; validated to finite 3-/4-vectors. Pure wrt the handle."""
    position, quaternion = body.get_world_pose()
    return _as_tuple(position, 3), _as_tuple(quaternion, 4)


@dataclass
class ReseatTarget:
    """One body to restore to a configured start pose. ``position`` / ``quat_wxyz`` are the TARGET world
    pose; ``zero_velocity`` zeroes linear+angular velocity after the move (best-effort on the handle)."""
    name: str
    body: object
    position: tuple                      # target world (x, y, z)
    quat_wxyz: tuple                     # target world quaternion (w, x, y, z)
    zero_velocity: bool = True
    xy_tol_m: float = 0.01               # read-back tolerance: the move must have taken
    yaw_tol_rad: float = math.radians(1.0)
    z_tol_m: float = 0.05


@dataclass
class ReseatRecord:
    """Per-body restore record for the console line + mission.json: the attempt-1 drift the restore
    corrected and the read-back residual proving it took."""
    name: str
    before_xy_yaw: tuple
    after_xy_yaw: tuple
    moved_xy_m: float
    moved_yaw_deg: float
    readback_ok: bool
    extra: dict = field(default_factory=dict)


def _zero_velocity(body):
    for setter, zero in (('set_linear_velocity', (0.0, 0.0, 0.0)),
                         ('set_angular_velocity', (0.0, 0.0, 0.0))):
        fn = getattr(body, setter, None)
        if callable(fn):
            try:
                fn(zero)
            except Exception:  # noqa: BLE001 -- a transient (unregistered) handle may refuse velocity;
                pass           # the pose set is what matters, and the settle re-zeroes residual motion
    return True


def restore_target(target: ReseatTarget) -> ReseatRecord:
    """Restore ONE body to its target pose + zero velocity, then READ BACK to confirm the move took.

    Raises :class:`WarmReseatRefused` (named ``WARM_RESEAT_UNRESTORED``) when the body has no
    ``set_world_pose`` or the read-back is still off the target beyond the tolerances -- the honest
    "cannot restore => cold boot" case. Returns the :class:`ReseatRecord` on success."""
    before_pos, before_quat = read_pose(target.body)
    before = (before_pos[0], before_pos[1], _wrap(yaw_from_wxyz(before_quat)))
    setter = getattr(target.body, 'set_world_pose', None)
    if not callable(setter):
        raise WarmReseatRefused(
            f'WARM_RESEAT_UNRESTORED: body {target.name!r} has no set_world_pose handle to restore it to '
            f'the configured start pose; cannot re-seat the warm run (cold boot required)')
    setter(position=list(target.position), orientation=list(target.quat_wxyz))
    if target.zero_velocity:
        _zero_velocity(target.body)
    after_pos, after_quat = read_pose(target.body)
    after = (after_pos[0], after_pos[1], _wrap(yaw_from_wxyz(after_quat)))
    target_yaw = _wrap(yaw_from_wxyz(target.quat_wxyz))
    dxy = math.hypot(after[0] - target.position[0], after[1] - target.position[1])
    dyaw = abs(_wrap(after[2] - target_yaw))
    dz = abs(after_pos[2] - target.position[2])
    readback_ok = dxy <= target.xy_tol_m and dyaw <= target.yaw_tol_rad and dz <= target.z_tol_m
    moved_xy = math.hypot(after[0] - before[0], after[1] - before[1])
    moved_yaw = abs(_wrap(after[2] - before[2]))
    rec = ReseatRecord(name=target.name, before_xy_yaw=before, after_xy_yaw=after,
                       moved_xy_m=round(moved_xy, 6), moved_yaw_deg=round(math.degrees(moved_yaw), 4),
                       readback_ok=readback_ok,
                       extra={'target_yaw_deg': round(math.degrees(target_yaw), 4),
                              'readback_dxy_m': round(dxy, 6),
                              'readback_dyaw_deg': round(math.degrees(dyaw), 4),
                              'readback_dz_m': round(dz, 6)})
    if not readback_ok:
        code = 'START_STATE_BASE_UNRESTORED' if target.name == 'base' else 'WARM_RESEAT_UNRESTORED'
        raise WarmReseatRefused(
            f'{code}: restoring {target.name!r} to the configured start pose did not take -- read-back '
            f'off target by (xy {dxy * 1000:.1f} mm, yaw {math.degrees(dyaw):.1f} deg, z {dz * 1000:.1f} '
            f'mm); the warm run would begin from a mis-seated train (cold boot required)')
    return rec


def restore_targets(targets) -> list:
    """Restore every target in order, read-back verified. Raises :class:`WarmReseatRefused` (named) at
    the first body that cannot be restored. Returns the list of :class:`ReseatRecord`."""
    return [restore_target(t) for t in targets]


def planar_target(body, configured_xy_yaw, *, rest_z, name,
                  xy_tol_m=0.01, yaw_tol_rad=math.radians(1.0), z_tol_m=0.05) -> ReseatTarget:
    """A :class:`ReseatTarget` for a body whose CONFIGURED planar pose (x, y, yaw) is known (cart / base,
    from ``initial_staging``) and whose resting Z is captured from the cold build."""
    x, y, yaw = _as_tuple(configured_xy_yaw, 3)
    return ReseatTarget(name=name, body=body, position=(x, y, float(rest_z)),
                        quat_wxyz=wxyz_from_yaw(yaw), xy_tol_m=xy_tol_m, yaw_tol_rad=yaw_tol_rad,
                        z_tol_m=z_tol_m)


def captured_target(body, captured_pose, *, name,
                    xy_tol_m=0.01, yaw_tol_rad=math.radians(1.0), z_tol_m=0.05) -> ReseatTarget:
    """A :class:`ReseatTarget` for a body whose configured pose is the one CAPTURED at the cold build
    (the backstock carton, whose spawn is composed from the planogram, not an ``initial_staging`` field)."""
    pos, quat = captured_pose
    return ReseatTarget(name=name, body=body, position=_as_tuple(pos, 3), quat_wxyz=_as_tuple(quat, 4),
                        xy_tol_m=xy_tol_m, yaw_tol_rad=yaw_tol_rad, z_tol_m=z_tol_m)


def configured_planar(scenario, key) -> Optional[tuple]:
    """``(x, y, yaw)`` from ``scenario['initial_staging'][key]`` (``cart_body_pose`` / ``robot_base_pose``),
    or ``None`` when absent/malformed. This is the SAME source the cold build spawns from -- for a
    dock-aligned run ``cart_body_pose`` is the configured dock heading ``dock_align_scenario`` wrote."""
    st = scenario.get('initial_staging') if isinstance(scenario, dict) else None
    pose = st.get(key) if isinstance(st, dict) else None
    if isinstance(pose, (list, tuple)) and len(pose) == 3 and all(
            isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) for v in pose):
        return (float(pose[0]), float(pose[1]), _wrap(float(pose[2])))
    return None


def brief(records) -> str:
    """Compact console summary of what the restore moved (for the ``WARM_RESEAT`` log line)."""
    return ' '.join(
        f'{r.name}[moved xy={r.moved_xy_m * 1000:.1f}mm yaw={r.moved_yaw_deg:.1f}deg '
        f'-> {r.after_xy_yaw[2] * 180.0 / math.pi:.1f}deg]'
        for r in records) or 'nothing-to-restore'
