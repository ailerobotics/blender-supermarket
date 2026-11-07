"""sim-place-release-carton-stays-v1: a robust, ORDERED sim-attach-fallback detach so the released carton
becomes a FREE DYNAMIC body supported by the shelf -- not a carton that stays rigidly coupled to the gripper.

WHY (coldwin105 -- the first run whose collect set-down reached the shelf seat):
  The descent lowered the WELDED carton to the modelled seat (carton centre z 0.6398 -> 0.5307,
  bottom_gap -0.3 mm). The release chain then acknowledged success end to end (gripper-open, cuMotion
  detach, ``SIM_ATTACH_FALLBACK release``, ``RELEASE_CONFIRMED held=False``), yet the carton stayed
  physically coupled to the gripper: through the 60 mm clearance lift the SEAL_DIAG pad->carton gap held
  **0.0 mm** and pad-vs-face **0.0 deg** while the measured carton CENTRE rose **0.1494 m** with the arm,
  and the PR 384 contact logger saw **zero** carton-vs-cart contacts for the whole run. The re-enable of the
  carton collider DID take (the seal raycast went from ``min_gap_mm=None`` while welded to ``0.48 mm`` the
  tick after release -- the collider is hit again), so the carton is NOT missing a collider; it is still
  rigidly held. The named mechanism: ``stage.RemovePrim(<gripper>/sim_attach_fallback)`` removes the USD
  prim (so the host ``read()`` reports ``held=False``) but the FixedJoint's PhysX constraint is NOT dropped
  from the running simulation -- the same runtime-USD-does-not-propagate class the PR 384 view-safe note
  warns about. ``RELEASE_FREE``'s "at rest, stable 5 ticks" measured the WELD holding the carton at the
  modelled seat, not the shelf supporting it (zero support contact), and the clearance lift carried the
  still-welded carton up.

THE FIX (opt-in; default off => byte-identical): drop the constraint EXPLICITLY, not only by prim removal.
This module is the PURE, CPU-testable core -- it owns the ORDER of the release operations (the names the
``cad_tool`` executor dispatches and a fake stage records) so the ordering can be proven without Isaac:

  robust OFF (legacy): enable carton collider -> (defer|restore tool-side) -> remove the weld prim
                       -> (optional proof-run kinematic flag)                    == byte-identical today

  robust ON:           enable carton collider -> DISABLE the weld joint (clear body0/body1, jointEnabled
                       False) -> (defer|restore tool-side) -> remove the weld prim -> make the carton a
                       FREE DYNAMIC body (clear any kinematic flag) -> zero its residual lin/ang velocity
                       and wake it

Disabling the joint (clearing its body relationships and ``physics:jointEnabled``) drops the PhysX
constraint even when a bare prim removal does not re-sync at runtime; zeroing the carton velocity removes
the coupling velocity/depenetration pop so the freed carton simply settles onto the shelf under gravity.
The companion host gate (:mod:`release_free`) then verifies a real SUPPORT CONTACT before the clearance
lift, so a detach that still fails is NAMED before the destructive lift instead of after it.
"""
from __future__ import annotations

from typing import Tuple

# Release operation names, in the order they must execute. The ``cad_tool`` executor dispatches on these
# and the fake-stage CPU test records them, so the order is provable without Isaac.
ENABLE_CARTON_COLLISION = 'enable_carton_collision'   # re-enable the welded carton's colliders (so it can rest)
DISABLE_WELD_JOINT = 'disable_weld_joint'             # robust: clear body0/body1 + jointEnabled=False (drop the constraint)
DEFER_TOOL_COLLISION = 'defer_tool_collision'         # leave the tool-side colliders OFF past the empty retreat
RESTORE_TOOL_COLLISION = 'restore_tool_collision'     # re-enable the tool-side colliders now (legacy non-defer)
REMOVE_WELD_JOINT = 'remove_weld_joint'               # stage.RemovePrim(<gripper>/sim_attach_fallback)
SET_CARTON_KINEMATIC = 'set_carton_kinematic'         # proof-run only (RESTOCK_RELEASE_KINEMATIC): no dynamic add
CLEAR_CARTON_KINEMATIC = 'clear_carton_kinematic'     # robust: ensure the freed carton is DYNAMIC, not kinematic
ZERO_CARTON_VELOCITY = 'zero_carton_velocity'         # robust: zero lin/ang velocity + wake (no residual pop)

ALL_OPS = (ENABLE_CARTON_COLLISION, DISABLE_WELD_JOINT, DEFER_TOOL_COLLISION, RESTORE_TOOL_COLLISION,
           REMOVE_WELD_JOINT, SET_CARTON_KINEMATIC, CLEAR_CARTON_KINEMATIC, ZERO_CARTON_VELOCITY)


def ordered_release_ops(*, robust: bool, defer_tool: bool,
                        release_kinematic: bool = False) -> Tuple[str, ...]:
    """The ORDERED list of release operations for one sim-attach-fallback detach. PURE.

    ``robust`` is the opt-in sim-place-release-carton-stays-v1 detach; ``defer_tool`` is the
    (default-on) tool-side collider deferral; ``release_kinematic`` is the ``RESTOCK_RELEASE_KINEMATIC``
    proof-run flag. The ``robust=False`` sequence is EXACTLY the legacy ``_release_fallback`` op order, so
    a default run is byte-identical; ``robust=True`` inserts the explicit joint disable before the prim
    removal and the free-dynamic + velocity-zero after it, and NEVER sets the kinematic flag (robust means
    a genuinely free dynamic body). The carton-collider re-enable is always FIRST (so the carton can rest),
    and the weld joint is always dropped (disable, then remove) before the carton is made free -- an order a
    CPU test can assert against the fake stage."""
    ops = [ENABLE_CARTON_COLLISION]
    if robust:
        ops.append(DISABLE_WELD_JOINT)
    ops.append(DEFER_TOOL_COLLISION if defer_tool else RESTORE_TOOL_COLLISION)
    ops.append(REMOVE_WELD_JOINT)
    if robust:
        # A robust detach always yields a free dynamic body (clear any kinematic flag) with no residual
        # velocity -- the kinematic proof-run shortcut is intentionally NOT honoured here.
        ops.append(CLEAR_CARTON_KINEMATIC)
        ops.append(ZERO_CARTON_VELOCITY)
    elif release_kinematic:
        ops.append(SET_CARTON_KINEMATIC)
    return tuple(ops)


def apply_release_ops(ops, adapter, *, body_path, joint_path, log=print):
    """Dispatch an :func:`ordered_release_ops` sequence to a physics-layer ``adapter`` IN ORDER. PURE
    dispatch (no Isaac) -- the ``adapter`` owns the actual USD/PhysX mutations, so a recording fake
    adapter proves the ordered release reaches the physics layer with no GPU.

    ``adapter`` is any object with the per-op methods (``enable_carton_collision(body)``,
    ``disable_weld_joint(joint)``, ``defer_tool_collision(body)``, ``restore_tool_collision(body)``,
    ``remove_weld_joint(joint)``, ``set_carton_kinematic(body)``, ``clear_carton_kinematic(body)``,
    ``zero_carton_velocity(body)``) -- in production this is the ``NativePanelContact`` itself. The
    carton-release ops (collider re-enable, tool-side defer/restore, kinematic, velocity) are skipped when
    ``body_path`` is None (no body was welded), matching the legacy ``if body is not None`` guard; the
    joint ops (disable/remove) always run. Returns the executed ``ops`` tuple."""
    executed = []
    for op in ops:
        if op == ENABLE_CARTON_COLLISION:
            if body_path is not None:
                adapter.enable_carton_collision(body_path)
        elif op == DISABLE_WELD_JOINT:
            adapter.disable_weld_joint(joint_path)
        elif op == DEFER_TOOL_COLLISION:
            if body_path is not None:
                adapter.defer_tool_collision(body_path)
        elif op == RESTORE_TOOL_COLLISION:
            if body_path is not None:
                adapter.restore_tool_collision(body_path)
        elif op == REMOVE_WELD_JOINT:
            adapter.remove_weld_joint(joint_path)
        elif op == SET_CARTON_KINEMATIC:
            if body_path is not None:
                adapter.set_carton_kinematic(body_path)
        elif op == CLEAR_CARTON_KINEMATIC:
            if body_path is not None:
                adapter.clear_carton_kinematic(body_path)
        elif op == ZERO_CARTON_VELOCITY:
            if body_path is not None:
                adapter.zero_carton_velocity(body_path)
        else:  # pragma: no cover -- ordered_release_ops only emits ALL_OPS
            raise ValueError('unknown release op %r' % (op,))
        executed.append(op)
    return tuple(executed)


# Config/env resolution for the opt-in robust detach, mirroring ``cad_tool.defer_tool_collision_enabled``.
RELEASE_DETACH_ENV = 'RESTOCK_RELEASE_DETACH_ROBUST'
_ON = ('1', 'true', 'on', 'yes')
_OFF = ('0', 'false', 'off', 'no')


def robust_detach_enabled(config=None, environ=None) -> bool:
    """The effective robust-detach setting (pure/CPU-testable). DEFAULT OFF => byte-identical.

    Resolution order (first that is set wins): the runtime-config key ``release_detach`` (a bool, or an
    on/off string, or a dict ``{"robust": bool}``), else the env ``RESTOCK_RELEASE_DETACH_ROBUST``, else
    DEFAULT FALSE. A malformed value raises by name rather than silently running the wrong path."""
    if config is not None and 'release_detach' in (config or {}):
        val = (config or {})['release_detach']
        if isinstance(val, dict):
            val = val.get('robust', False)
        if isinstance(val, bool):
            return val
        s = str(val).strip().lower()
        if s in _OFF:
            return False
        if s in _ON:
            return True
        raise ValueError('release_detach must be a bool, a dict {"robust": bool}, or one of '
                         'on/off/true/false/1/0/yes/no, got %r' % (val,))
    import os
    env = os.environ if environ is None else environ
    raw = str(env.get(RELEASE_DETACH_ENV, '')).strip().lower()
    if raw in _ON:
        return True
    return False
