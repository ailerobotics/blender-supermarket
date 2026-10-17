"""Start-phase shortcut (DEV ITERATION ONLY): synthesize the physical PRECONDITION of a
later mission stage so the real state machine can begin there instead of paying the full
upstream choreography every GPU run.

Pairs with ``Mission(start_at=...)`` (mission.py): the mission records the skipped stages as
honest ``<stage>_shortcut`` events; THIS module establishes the matching physical state in the
live stage before the first mission tick. Driven by mission-config ``config['start_at']``;
absent/None => this module is never called and the production path is byte-identical.

Design split, mirroring ``isaac_bundle.plan_bundle`` vs ``_build_isaac_runtime``:

* ``plan_start_state`` is PURE (no Isaac import) and CPU-tested. It derives, from the scenario's
  ``initial_staging`` geometry, exactly where each synthesized body must be, so the error-prone
  pose math is proven offline.
* ``apply_start_state`` runs ONLY inside the live SimulationApp: it moves the bodies named by the
  plan, zeroes their velocity and settles physics. It fails CLOSED on any start_at it cannot fully
  synthesize, so a shortcut never runs against a half-built physical state.

Supported today: ``start_at='couple'`` -- skip only ``collect`` (the backstock->cart grasp, the
expensive and least-reliable arm reach). The post-collect fact is "the carton is loaded on the
cart": the robot is already at the pickup pose, the cart at the dock and the arm stowed by the
assembly's spawn defaults, so the ONLY body to synthesize is the carton, placed resting at the
cart load centre exactly as the real collect would leave it (NOT welded -- the downstream transfer
stage must be able to re-grasp it off the cart). couple -> outbound -> park -> transfer -> verify ->
return then run for real, proving every downstream phase in one shortcut run.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from .mission import ACTIONS

# Stages whose physical end state this module knows how to synthesize, keyed by the single stage
# each one skips. Extending to 'transfer' (skip collect+couple+outbound+park) additionally needs
# the coupled-cart + parked-at-shelf state and is a deliberate follow-up, not a silent default.
_SUPPORTED_START_AT = ('couple',)


def _yaw_to_wxyz(yaw):
    """Quaternion (w, x, y, z) for a rotation of ``yaw`` radians about world +Z."""
    return (math.cos(yaw / 2.0), 0.0, 0.0, math.sin(yaw / 2.0))


def _finite(*values):
    return all(isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)
               for v in values)


@dataclass(frozen=True)
class StartState:
    start_at: str
    skipped: tuple               # stages the shortcut skips (in ACTIONS order)
    carton_position: tuple       # world (x, y, z) of the carton CENTRE
    carton_orientation_wxyz: tuple
    note: str


def plan_start_state(scenario, start_at, stages=None):
    """Pure start-state plan from the scenario's measured staging. No Isaac import.

    Raises ValueError (fail-closed) on an unsupported ``start_at`` or missing/malformed staging,
    so a misconfigured shortcut refuses to synthesize rather than place bodies at an invented pose.
    """
    stages = tuple(stages) if stages else ACTIONS
    if start_at not in stages:
        raise ValueError(f'start_at {start_at!r} not in mission stages {stages}')
    if start_at not in _SUPPORTED_START_AT:
        raise ValueError(
            f'start-phase shortcut does not yet synthesize start_at={start_at!r}; '
            f'supported: {_SUPPORTED_START_AT}')
    skipped = tuple(stages[:stages.index(start_at)])
    if skipped != ('collect',):
        raise ValueError(f'start_at={start_at!r} would skip {skipped}; only skipping '
                         "('collect',) is synthesized today")

    staging = scenario.get('initial_staging') if isinstance(scenario, dict) else None
    if not isinstance(staging, dict):
        raise ValueError('scenario.initial_staging is required to synthesize a start state')
    cart_pose = staging.get('cart_body_pose')
    load_xy = staging.get('cart_load_center_body_xy')
    support_z = staging.get('cart_support_z_world')
    handling = scenario.get('handling_unit', {})
    dims = handling.get('dimensions_m')
    yaws = staging.get('unit_yaws_world', {})
    cart_yaw_on_unit = yaws.get('cart')
    if not (isinstance(cart_pose, (list, tuple)) and len(cart_pose) == 3 and _finite(*cart_pose)):
        raise ValueError('initial_staging.cart_body_pose must be finite x/y/yaw')
    if not (isinstance(load_xy, (list, tuple)) and len(load_xy) == 2 and _finite(*load_xy)):
        raise ValueError('initial_staging.cart_load_center_body_xy must be finite x/y')
    if not _finite(support_z):
        raise ValueError('initial_staging.cart_support_z_world must be finite')
    if not (isinstance(dims, (list, tuple)) and len(dims) == 3 and _finite(*dims) and all(d > 0 for d in dims)):
        raise ValueError('handling_unit.dimensions_m must be three finite positive extents')
    if not _finite(cart_yaw_on_unit):
        raise ValueError('initial_staging.unit_yaws_world.cart must be a finite yaw')

    cx, cy, cyaw = (float(v) for v in cart_pose)
    lx, ly = (float(v) for v in load_xy)
    # Load centre is given in the cart BODY frame; rotate by the cart's world yaw into the map frame.
    cos_y, sin_y = math.cos(cyaw), math.sin(cyaw)
    wx = cx + lx * cos_y - ly * sin_y
    wy = cy + lx * sin_y + ly * cos_y
    wz = float(support_z) + float(dims[2]) / 2.0   # carton CENTRE sits half its height above the deck
    return StartState(
        start_at=start_at, skipped=skipped,
        carton_position=(wx, wy, wz),
        carton_orientation_wxyz=_yaw_to_wxyz(float(cart_yaw_on_unit)),
        note=('synthesized post-collect: carton resting at the cart load centre '
              '(not welded; transfer must re-grasp it)'))


def apply_start_state(assembly, config, scenario, *, settle_steps=30, _rigid_prim=None):
    """Establish the planned start state in the live stage. Isaac-only; never CPU-tested.

    ``_rigid_prim`` is a test seam (a SingleRigidPrim-like factory); production leaves it None and
    uses Isaac's SingleRigidPrim. Returns the StartState actually applied.
    """
    start_at = config.get('start_at')
    plan = plan_start_state(scenario, start_at, stages=config.get('stages'))

    stock_path = assembly.evidence['composition']['stock']
    if _rigid_prim is None:
        from isaacsim.core.prims import SingleRigidPrim
        _rigid_prim = SingleRigidPrim
    import numpy as np
    carton = assembly.world.scene.add(_rigid_prim(prim_path=stock_path, name='shortcut_carton'))
    carton.set_world_pose(position=np.asarray(plan.carton_position, dtype=float),
                          orientation=np.asarray(plan.carton_orientation_wxyz, dtype=float))
    carton.set_linear_velocity(np.zeros(3))
    carton.set_angular_velocity(np.zeros(3))
    # Let the carton settle onto the cart deck and the contact solver quiesce before the mission
    # ticks, so couple/outbound start from a stationary, resting payload.
    for _ in range(int(settle_steps)):
        assembly.world.step(render=False)
    print(f'START_PHASE_SHORTCUT applied start_at={plan.start_at} skipped={list(plan.skipped)} '
          f'carton_pos={tuple(round(v, 4) for v in plan.carton_position)} '
          f'carton_quat_wxyz={tuple(round(v, 4) for v in plan.carton_orientation_wxyz)}',
          flush=True)
    return plan
