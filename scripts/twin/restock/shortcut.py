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

``start_at='transfer'`` + ``config['start_staging']`` (see ``staged_scenario``) spawns the train at a
RECORDED park->transfer handover of a real tow instead of a hand-authored variant scenario.

Supported today: ``start_at='couple'`` -- skip only ``collect`` (the backstock->cart grasp, the
expensive and least-reliable arm reach). The post-collect fact is "the carton is loaded on the
cart": the robot is already at the pickup pose, the cart at the dock and the arm stowed by the
assembly's spawn defaults, so the ONLY body to synthesize is the carton, placed resting at the
cart load centre exactly as the real collect would leave it (NOT welded -- the downstream transfer
stage must be able to re-grasp it off the cart). couple -> outbound -> park -> transfer -> verify ->
return then run for real, proving every downstream phase in one shortcut run.
"""
from __future__ import annotations

import copy
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


# Opt-in RECORDED start staging for start_at='transfer' (sim-video-transfer-bench-v1): instead of a
# hand-authored variant scenario, spawn the train exactly where a REAL coupled tow leaves it at the
# park->transfer handover on the gentle backroom-bay route, so transfer -> verify -> return can be
# iterated in minutes. Selected by config['start_staging'] (absent => staged_scenario is the identity,
# byte-identical). Robot base pose = the nav-trace.jsonl row at the first 'transfer' tick of each run
# (park_succeeded, coupled=True, parked=True); the cart is coupled behind it on the hitch at the
# measured articulation (TRANSFER_CAPTURE put the deck centre at (-1.27, 0.00) in ur_base_link in all
# four runs == hitch_x - cart_hitch_x, i.e. articulation 0). Coupled: uncouple_on_return (captain #5)
# only releases the cart at the parking spot during return, so the train is still coupled here.
RECORDED_BAY_PARK_RUNS = {           # run -> robot base (x, y, yaw rad) at park -> transfer
    'run-351': (-2.079, 15.251, math.radians(-0.2)),
    'run-353': (-2.079, 15.248, math.radians(0.2)),
    'run-354': (-2.079, 15.248, math.radians(0.2)),
    'run-356': (-2.079, 15.254, math.radians(-0.5)),
}
RECORDED_BAY_PARK_CART_IN_BASE_XY = (-1.27, 0.0)   # deck centre in ur_base_link (TRANSFER_CAPTURE cart=)
# Rollwagen deck top in world z: chassis origin (deck centre) at z=0.20 + half the 0.05 m deck
# (mobile_assets.author_cart). The scenario's cart_support_z_world (0.9325) is the UPPER shelf.
DECK_TOP_Z_WORLD = 0.225
START_STAGING_PRESETS = ('recorded_bay_park',)


def _mean_pose(poses):
    xs, ys, yaws = zip(*poses)
    yaw = math.atan2(sum(math.sin(a) for a in yaws), sum(math.cos(a) for a in yaws))
    return (round(sum(xs) / len(xs), 4), round(sum(ys) / len(ys), 4), round(yaw, 4))


RECORDED_BAY_PARK_ROBOT_POSE = _mean_pose(RECORDED_BAY_PARK_RUNS.values())


def _wrap(a):
    return math.atan2(math.sin(a), math.cos(a))


def cart_pose_from_hitch(robot_pose, articulation=0.0, *, hitch_x=-0.62, cart_hitch_x=0.65):
    """Cart body (x, y, yaw) coupled behind ``robot_pose`` with the two hitch points coincident.

    articulation = robot yaw - cart yaw (the convention of MobileController/wheel_targets). The robot
    hitch is ``hitch_x`` along the robot heading, the cart drawbar ``cart_hitch_x`` along the cart's."""
    rx, ry, ryaw = (float(v) for v in robot_pose)
    cyaw = _wrap(ryaw - float(articulation))
    hx, hy = rx + hitch_x * math.cos(ryaw), ry + hitch_x * math.sin(ryaw)
    return (hx - cart_hitch_x * math.cos(cyaw), hy - cart_hitch_x * math.sin(cyaw), cyaw)


def resolve_start_staging(config):
    """Normalized ``config['start_staging']`` (a preset name or an object), or None when absent.

    Object keys (all optional; preset defaults): ``preset`` ('recorded_bay_park'), ``robot_base_pose``
    [x, y, yaw], ``articulation`` (rad), ``carton_load_center_body_xy`` [x, y] (default: the scenario's
    ``initial_staging.cart_load_center_body_xy``, the fm479 slot), ``carton_support_z_world`` (default
    the deck top). Fails closed unless ``start_at`` is 'transfer'."""
    spec = config.get('start_staging') if isinstance(config, dict) else None
    if spec is None:
        return None
    if isinstance(spec, str):
        spec = {'preset': spec}
    if not isinstance(spec, dict):
        raise ValueError('start_staging must be a preset name or an object')
    preset = spec.get('preset', 'recorded_bay_park')
    if preset not in START_STAGING_PRESETS:
        raise ValueError(f'start_staging preset {preset!r} unknown; supported: {START_STAGING_PRESETS}')
    if config.get('start_at') != 'transfer':
        raise ValueError("start_staging applies only to start_at='transfer' (the park->transfer handover)")
    if spec.get('coupled', True) is not True:
        raise ValueError('start_staging: the mission is still coupled at the transfer handover '
                         '(uncouple_on_return releases the cart at the parking spot during return)')
    robot = spec.get('robot_base_pose', RECORDED_BAY_PARK_ROBOT_POSE)
    articulation = spec.get('articulation', 0.0)
    load_xy = spec.get('carton_load_center_body_xy')
    support_z = spec.get('carton_support_z_world', DECK_TOP_Z_WORLD)
    if not (isinstance(robot, (list, tuple)) and len(robot) == 3 and _finite(*robot)):
        raise ValueError('start_staging.robot_base_pose must be finite x/y/yaw')
    if not _finite(articulation) or abs(articulation) > math.radians(5):
        raise ValueError('start_staging.articulation must be finite and within the 5 deg attach gate')
    if load_xy is not None and not (isinstance(load_xy, (list, tuple)) and len(load_xy) == 2
                                    and _finite(*load_xy)):
        raise ValueError('start_staging.carton_load_center_body_xy must be finite x/y')
    if not _finite(support_z):
        raise ValueError('start_staging.carton_support_z_world must be finite')
    return {'preset': preset, 'robot_base_pose': tuple(float(v) for v in robot),
            'articulation': float(articulation),
            'carton_load_center_body_xy': None if load_xy is None else tuple(float(v) for v in load_xy),
            'carton_support_z_world': float(support_z), 'coupled': True}


def staged_scenario(scenario, config, *, hitch_x=-0.62, cart_hitch_x=0.65):
    """``(scenario_for_build, record)``: the scenario whose ``initial_staging`` spawns the robot + cart
    at the configured start staging (a build-time pose, never a runtime teleport), plus the JSON record
    of what was staged for mission.json. Absent ``start_staging`` => ``(scenario, None)``, the SAME
    object (byte-identical). Pure; only the physical build and the shortcut synthesis consume it."""
    staging = resolve_start_staging(config)
    if staging is None:
        return scenario, None
    variant = copy.deepcopy(scenario)
    st = variant.get('initial_staging') if isinstance(variant, dict) else None
    if not isinstance(st, dict) or not _finite(*st.get('cart_body_pose', (math.nan,))[:3]):
        raise ValueError('scenario.initial_staging.cart_body_pose is required to stage the start')
    robot = staging['robot_base_pose']
    cart = cart_pose_from_hitch(robot, staging['articulation'], hitch_x=hitch_x, cart_hitch_x=cart_hitch_x)
    dyaw = _wrap(cart[2] - float(st['cart_body_pose'][2]))
    st['robot_base_pose'] = list(robot)
    st['cart_body_pose'] = list(cart)
    if staging['carton_load_center_body_xy'] is not None:
        st['cart_load_center_body_xy'] = list(staging['carton_load_center_body_xy'])
    st['cart_support_z_world'] = staging['carton_support_z_world']
    yaws = dict(st.get('unit_yaws_world', {}))
    if _finite(yaws.get('cart')):                      # the carton turns with the cart
        yaws['cart'] = _wrap(float(yaws['cart']) + dyaw)
    st['unit_yaws_world'] = yaws
    record = {'mode': f"start_phase_shortcut:{staging['preset']}", 'preset': staging['preset'],
              'robot_base_pose': [round(v, 4) for v in robot],
              'cart_body_pose': [round(v, 4) for v in cart],
              'articulation': staging['articulation'], 'coupled': True,
              'carton_load_center_body_xy': [round(float(v), 4) for v in st['cart_load_center_body_xy']],
              'carton_support_z_world': staging['carton_support_z_world'],
              'source_runs': sorted(RECORDED_BAY_PARK_RUNS),
              'note': ('robot+cart spawned at the recorded park->transfer handover of a real coupled tow '
                       '(not driven); carton synthesized resting on the cart at carton_support_z_world')}
    return variant, record


def walkthrough_caption(config):
    """The caption burned into a bench run's walkthrough (and stored in mission.json), so a bench video
    can never pass for the full seven-stage run. None unless ``config`` sets ``start_at`` (stages
    synthesized) or ``stop_after`` (stages cut)."""
    start_at = config.get('start_at') if isinstance(config, dict) else None
    stop_after = config.get('stop_after') if isinstance(config, dict) else None
    if not start_at and not stop_after:
        return None
    stages = tuple(config.get('stages') or ACTIONS)
    head, details = [], []
    if start_at:
        skipped = stages[:stages.index(start_at)] if start_at in stages else ()
        head.append(f'start_at={start_at}')
        details.append(', '.join(skipped) + ' SYNTHESIZED, not performed' if skipped else 'no stage skipped')
    if stop_after:
        not_run = ACTIONS[ACTIONS.index(stop_after) + 1:] if stop_after in ACTIONS else ()
        head.append(f'stop_after={stop_after}')
        details.append(', '.join(not_run) + ' NOT RUN' if not_run else 'no stage cut')
    text = f"BENCH RUN {' '.join(head)}: {'; '.join(details)}"
    staging = config.get('start_staging')
    if staging is not None:
        preset = staging if isinstance(staging, str) else staging.get('preset', 'recorded_bay_park')
        text += f' | staged: {preset}'
    return text


def bounded_summary(config):
    """mission.json fields for a ``stop_after`` targeted test ({} when absent): the stop stage, the
    stages driven, the stages NOT run and the walkthrough caption -- the run's 'complete' covers only
    the stages it drove."""
    stop_after = config.get('stop_after') if isinstance(config, dict) else None
    if not stop_after:
        return {}
    stages = list(config.get('stages') or ACTIONS)
    start_at = config.get('start_at')
    driven = stages[stages.index(start_at):] if start_at in stages else stages
    not_run = list(ACTIONS[ACTIONS.index(stop_after) + 1:])
    return {'stop_after': stop_after, 'stages_driven': driven, 'not_run': not_run,
            'walkthrough_caption': walkthrough_caption(config),
            'bounded_note': (f'BOUNDED RUN: stopped after {stop_after}; not run: '
                             f"{', '.join(not_run) or 'none'}; state covers only {', '.join(driven)}")}


def shortcut_summary(shortcut, config, staging=None):
    """mission.json fields for a start-phase shortcut run: top-level ``start_at``, the shortcut record
    (with the applied ``staging`` when the recorded start staging ran), the ``start_staging_mode``
    workaround label and the walkthrough caption."""
    out = {'start_at': shortcut['start_at'], 'shortcut': dict(shortcut),
           'walkthrough_caption': walkthrough_caption(config)}
    if staging:
        out['shortcut']['staging'] = staging
        out['start_staging_mode'] = staging['mode']
    return out


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
    # Transfer warm-up observe pose (firstmate decision B, sim-config driven, phase-local to transfer):
    # the mission warm pumps the station + arm masked-depth but does NOT move the arm, so at a stowed
    # spawn the eye-in-hand never sees the shelf/cart reach workspace and the host collision_time never
    # goes fresh (ds-009). Spawn the arm at a configured observe pose that aims the eye-in-hand at the
    # reach workspace, mirroring collect's observe-then-execute. Absent => arm stays at the transport
    # default (hook is a no-op). Joint-name keyed; unknown names are ignored by the articulation.
    observe = config.get('transfer_observe_joints') if plan.couple_cart else None
    if observe:
        robot = assembly.robot
        q = np.asarray(robot.get_joint_positions(), dtype=float).copy()
        applied = []
        for name, value in observe.items():
            try:
                idx = robot.get_dof_index(name)
            except Exception:
                idx = None
            if idx is not None:
                q[idx] = float(value); applied.append(name)
        robot.set_joint_positions(q)
        try:
            robot.set_joint_velocities(np.zeros_like(q))
        except Exception:
            pass
        print(f'START_PHASE_SHORTCUT transfer observe pose applied to {applied}', flush=True)
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
