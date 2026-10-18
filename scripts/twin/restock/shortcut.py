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

# Stages whose physical end state this module knows how to synthesize.
#   'couple'   -- skip only collect: carton placed resting on the cart at its DOCK spawn; the real
#                 couple->outbound->park then run (cart gets physically coupled/towed by the mission).
#   'transfer' -- skip collect+couple+outbound+park: carton on the cart AND the cart physically
#                 coupled behind the robot. The robot+cart are spawned already parked at the shelf
#                 (a variant scenario's initial_staging, the sanctioned build-time pose -- NOT a
#                 runtime teleport, which physics_setup forbids); this module places the carton and
#                 authors the hitch so transfer->verify run against a coupled, loaded, parked train.
_SUPPORTED_START_AT = ('couple', 'transfer')
# Exact in-order prefix each supported start_at must skip (a guard against a mis-ordered stages list).
_EXPECTED_SKIP = {'couple': ('collect',),
                  'transfer': ('collect', 'couple', 'outbound', 'park')}


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
    couple_cart: bool            # author the physical robot<->cart hitch at synthesis (transfer start)
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
    if skipped != _EXPECTED_SKIP[start_at]:
        raise ValueError(f'start_at={start_at!r} skips {skipped}; expected {_EXPECTED_SKIP[start_at]} '
                         '(mission stages must be the canonical in-order prefix)')

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
    couple_cart = start_at == 'transfer'
    return StartState(
        start_at=start_at, skipped=skipped,
        carton_position=(wx, wy, wz),
        carton_orientation_wxyz=_yaw_to_wxyz(float(cart_yaw_on_unit)),
        couple_cart=couple_cart,
        note=('synthesized carton resting at the cart load centre (not welded; the arm re-grasps it)'
              + ('; cart physically coupled behind the robot (hitch authored), parked at the shelf'
                 if couple_cart else '; cart at the dock, left uncoupled for the real couple stage')))


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
    # ticks, so the downstream phases start from a stationary, resting payload.
    for _ in range(int(settle_steps)):
        assembly.world.step(render=False)
    coupling = None
    if plan.couple_cart:
        # transfer start: the couple/outbound/park stages are skipped, so the cart must already be
        # physically coupled behind the robot (both spawned parked+aligned at the shelf by the
        # variant scenario). Author the real hitch through the SAME mobile.attach contract the couple
        # stage uses (fresh, stopped, aligned observation), then assert the latch is healthy -- a
        # silent uncoupled cart would let the arm's transfer reaction shove it.
        assembly.mobile.park()
        for _ in range(int(settle_steps)):
            assembly.hold()
            assembly.world.step(render=False)
        dock = _dock_observation(assembly)
        pre = assembly.mobile.coupling_state(dock)
        if pre['joint_present']:
            raise RuntimeError('shortcut: stage already coupled before the synthetic hitch request')
        coupling = assembly.mobile.attach(dock)
        if not coupling.get('healthy'):
            raise RuntimeError('shortcut: synthetic cart coupling is not healthy '
                               f"(residual {coupling.get('hitch_error_m')} m); check the variant "
                               'scenario cart_body_pose vs robot_base_pose hitch alignment')
        for _ in range(int(settle_steps)):
            assembly.hold()
            assembly.world.step(render=False)
    print(f'START_PHASE_SHORTCUT applied start_at={plan.start_at} skipped={list(plan.skipped)} '
          f'carton_pos={tuple(round(v, 4) for v in plan.carton_position)} '
          f'carton_quat_wxyz={tuple(round(v, 4) for v in plan.carton_orientation_wxyz)} '
          f'coupled={None if coupling is None else coupling.get("healthy")}',
          flush=True)
    return plan


def _dock_observation(assembly, *, age=0.0):
    """Measured DockObservation for the synthetic hitch (same shape as coupling_check.dock_observation)."""
    import numpy as np
    from twin.mobile import DockObservation
    from .isaac_port import body_measurement
    r, rz, rv, rw = body_measurement(assembly.robot_body)
    c, cz, cv, cw = body_measurement(assembly.cart_body)
    frames = assembly.mobile.spec.manifest()
    return DockObservation(r, c, rz + frames['robot_hitch_local'][2],
        cz + frames['cart_hitch_local'][2], float(np.linalg.norm(rv)), float(np.linalg.norm(cv)),
        float(rw[2]), float(cw[2]), age=age)
