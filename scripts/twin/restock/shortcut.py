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

Supported start points (dev iteration; each lets a LATER stage be proven in seconds-to-minutes
instead of a ~45-min cold run):

* ``start_at='couple'`` -- skip only ``collect`` (the backstock->cart grasp). The robot is already at
  the pickup pose, the cart at the dock and the arm stowed by the assembly's spawn defaults, so the
  ONLY body to synthesize is the carton, placed resting at the cart load centre exactly as the real
  collect would leave it (NOT welded -- the downstream transfer stage must re-grasp it off the cart);
  the cart is left UNCOUPLED for the real couple stage.
* ``start_at='outbound'`` / ``'park'`` / ``'transfer'`` -- skip the collect+couple(+outbound[+park])
  prefix: the carton rests level on the cart AND the cart is physically COUPLED behind the robot. The
  robot+cart are spawned at the sanctioned build-time pose for that stage (a variant scenario's
  ``initial_staging`` -- NOT a runtime teleport, which physics_setup forbids): ``outbound`` at the
  coupled tow lane START (the post-couple dock), ``park`` at the tow lane GOAL (the stationarise
  point -- bay, or the shelf-staging pose when ``shelf_staging`` is set), ``transfer`` at the
  recorded park->transfer handover (``recorded_bay_park``). This module places the carton and authors
  the real hitch so the chosen stage onward runs against a coupled, loaded, parked train.
* ``start_at='verify'`` -- skip collect..transfer: the carton is already PLACED level on the transfer
  bay top (``transfer_bay`` / the ``transfer_ground_truth.target``), the arm is empty and the cart is
  still coupled, exactly as a succeeded transfer leaves it. verify (the stock-commit gate) then runs
  for real against the synthesized placement.

``start_at='return'`` is NOT a Mission start point -- ``Mission`` refuses any ``start_at`` AFTER the
``verify`` stock-commit gate (verify is the only stage that commits stock, and skipping it would leave
inventory uncredited). ``return`` is therefore run in isolation via ``--start-at verify --stop-after
return``: the verify-entry state (carton placed, coupled) is synthesized, the one instant ground-truth
``verify`` runs (committing stock honestly) and ``return`` then tows home. ``stage_isolation`` encodes
exactly this. The verify-skip guard is a safety bound and is left intact (never widened for a bench).

Base poses for the coupled stages are DERIVED from the configured flow (``navigation.tow_lanes`` +
``mission_poses``), so the synthesis follows whichever of the dock-align / shelf-staging flows the run
is configured for; the verify/return carton placement follows the backroom-bay tail that
``check_mission_tail`` validates. The shelf-staging sales-floor transfer/verify/return (an UNCOUPLED
endcap service with the carton CARRIED, not on the cart) stays integration-owned per
``docs/examples/tow-lanes-mission-fragment.json``; see docs/restock_stages.md.
"""
from __future__ import annotations

import copy
import math
from dataclasses import dataclass

from .mission import ACTIONS

# Stages whose physical end state this module's start-phase shortcut can synthesize, in ACTIONS order.
#   'couple'   -- skip collect: carton resting on the cart at its DOCK spawn, left UNCOUPLED.
#   'outbound' -- skip collect+couple: carton on the cart, cart COUPLED, spawned at the tow lane start.
#   'park'     -- skip collect+couple+outbound: carton on the cart, cart COUPLED, spawned at the tow
#                 lane goal (the stationarise point).
#   'transfer' -- skip collect+couple+outbound+park: carton on the cart, cart COUPLED, spawned at the
#                 recorded park->transfer handover (recorded_bay_park staging).
#   'verify'   -- skip collect..transfer: carton PLACED on the bay top, arm empty, cart still COUPLED,
#                 exactly as a succeeded transfer leaves it; verify (the stock-commit gate) runs for real.
# ('return' is reached via start_at='verify' --stop-after return; Mission refuses a start_at past verify.)
_SUPPORTED_START_AT = ('couple', 'outbound', 'park', 'transfer', 'verify')
# Exact in-order prefix each supported start_at must skip (a guard against a mis-ordered stages list).
_EXPECTED_SKIP = {'couple': ('collect',),
                  'outbound': ('collect', 'couple'),
                  'park': ('collect', 'couple', 'outbound'),
                  'transfer': ('collect', 'couple', 'outbound', 'park'),
                  'verify': ('collect', 'couple', 'outbound', 'park', 'transfer')}
# start_at values that must author the physical robot<->cart hitch at synthesis (the skipped prefix
# already coupled the train). couple is excluded -- the real couple stage does the latching.
_COUPLE_AT_SYNTHESIS = ('outbound', 'park', 'transfer', 'verify')


# Opt-in RECORDED start staging for start_at='transfer' (sim-video-transfer-bench-v1): instead of a
# hand-authored variant scenario, spawn the train exactly where a REAL coupled tow leaves it at the
# park->transfer handover on the gentle backroom-bay route, so transfer -> verify -> return can be
# iterated in minutes. Selected by config['start_staging'] (absent => staged_scenario is the identity,
# byte-identical). The cart is coupled behind the robot on the hitch at the measured articulation
# (TRANSFER_CAPTURE put the deck centre at (-1.27, 0.00) in ur_base_link in all four runs == hitch_x -
# cart_hitch_x, i.e. articulation 0). Coupled: uncouple_on_return (captain #5) only releases the cart at
# the parking spot during return, so the train is still coupled here.
#
# HISTORICAL (v2 store, 0.6 m carton): the four runs below recorded the robot base pose at the first
# 'transfer' tick (park_succeeded, coupled=True, parked=True) on the v2-store / 0.6 m-carton build. That
# frozen pose went STALE on the v3-portal26 store with the 0.45 m carton (ledger stage-transfer2: the
# train spawned there read as misaligned at the dock gate before the transfer pick). It is kept ONLY as
# the documented 'before' of sim-rerecord-bay-park-v3-v1 and as the fallback when a scenario carries no
# ``transfer_bay`` to derive from -- the ``recorded_bay_park`` preset now DERIVES its pose from the live
# scenario geometry (``derive_bay_park``) so a store or carton change re-derives rather than silently
# replaying a stale pose. See ``bay_park_provenance`` / ``check_bay_park_provenance``.
RECORDED_BAY_PARK_RUNS = {           # run -> robot base (x, y, yaw rad) at park -> transfer (v2, historical)
    'run-351': (-2.079, 15.251, math.radians(-0.2)),
    'run-353': (-2.079, 15.248, math.radians(0.2)),
    'run-354': (-2.079, 15.248, math.radians(0.2)),
    'run-356': (-2.079, 15.254, math.radians(-0.5)),
}
RECORDED_BAY_PARK_CART_IN_BASE_XY = (-1.27, 0.0)   # deck centre in ur_base_link (TRANSFER_CAPTURE cart=)
# Carton spot (decided by sim-video-transfer-fallback-v1's live-pose reach check): the rollwagen UPPER
# shelf at the scenario's cart_load_center_body_xy (+0.18, 0) and cart_support_z_world (0.9325 shelf top
# -> carton centre z 1.0125), yaw -pi/2 cart-relative -- exactly where the warm start_at=couple shortcut
# leaves it (worst reach 1.040 m from the bay park, through the open +x mouth, clear of the drawbar). The
# deck (top z 0.225 = chassis origin 0.20 + half the 0.05 m deck, mobile_assets.author_cart) is
# collision-tight over the hitch; it stays available as an explicit carton_support_z_world override.
DECK_TOP_Z_WORLD = 0.225

# Start-staging presets. Each spawns the robot+cart at the sanctioned build-time pose for ONE start_at
# (the base pose is DERIVED from the configured flow for the lane-driven presets; recorded_bay_park is
# the hand-recorded backroom-bay pose). The preset<->start_at mapping is one-to-one so a mis-paired
# (preset, start_at) is refused, never silently applied to the wrong stage.
START_STAGING_PRESETS = ('recorded_bay_park', 'dock_coupled', 'bay_parked', 'bay_placed')
#   recorded_bay_park -> transfer : robot at the recorded park->transfer handover; carton on the cart.
#   dock_coupled      -> outbound : robot at the coupled tow lane START (post-couple dock); carton on cart.
#   bay_parked        -> park     : robot at the coupled tow lane GOAL (stationarise point); carton on cart.
#   bay_placed        -> verify   : robot at the transfer/verify park pose; carton PLACED on the bay top.
_PRESET_FOR_START_AT = {'transfer': 'recorded_bay_park', 'outbound': 'dock_coupled',
                        'park': 'bay_parked', 'verify': 'bay_placed'}
_START_AT_FOR_PRESET = {preset: stage for stage, preset in _PRESET_FOR_START_AT.items()}


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


# Default arm reach when the scenario does not state one (handling_unit.max_reach_m); 1.3 m is the
# UR10 value the mission scenario carries. A derived park pose must keep BOTH the carton on the cart
# behind the robot AND the transfer bay in front within this radius (planar, base frame).
_DEFAULT_MAX_REACH_M = 1.3


def derive_bay_park(scenario, *, hitch_x=-0.62, cart_hitch_x=0.65, articulation=0.0,
                    max_reach_m=None):
    """Re-derive the ``recorded_bay_park`` robot base pose from the CURRENT scenario geometry.

    This is the CPU twin of the pose the full run reaches at the transfer bay: the robot parks on the
    backroom tow corridor, laterally CENTRED under the transfer bay (so the place reach is centred),
    coupled to the cart it towed in; the transfer pick then reaches the carton resting on the cart's
    upper shelf BEHIND the robot and places it on the bay IN FRONT. Every number is read from the
    scenario -- the transfer bay centre, the corridor line + heading (``initial_staging.cart_body_pose``
    y/yaw), the cart hitch geometry (the mobile spec, passed in), the load centre and the carton height
    -- so a store or carton change RE-DERIVES the pose instead of replaying the frozen v2 record.

    Returns ``{'robot_base_pose': (x, y, yaw), 'reach': {...}, 'inputs': {...}}``. Raises ValueError,
    naming the offending leg with its measured reach vs the bound, when the derived pose cannot keep
    both the carton pick and the bay place within arm reach -- so an infeasible geometry fails LOUDLY
    here (CPU) rather than at the GPU bench. ``None`` is never returned; callers that want the legacy
    fallback must check for ``transfer_bay`` first (see ``staged_scenario``).
    """
    bay = scenario.get('transfer_bay') if isinstance(scenario, dict) else None
    st = scenario.get('initial_staging') if isinstance(scenario, dict) else None
    handling = scenario.get('handling_unit', {}) if isinstance(scenario, dict) else {}
    if not isinstance(bay, dict) or not isinstance(st, dict):
        raise ValueError('derive_bay_park needs scenario.transfer_bay and scenario.initial_staging')
    center = bay.get('center_xy')
    cart_pose = st.get('cart_body_pose')
    load_xy = st.get('cart_load_center_body_xy')
    support_z = st.get('cart_support_z_world')
    dims = handling.get('dimensions_m')
    reach = float(handling.get('max_reach_m', max_reach_m if max_reach_m is not None else _DEFAULT_MAX_REACH_M))
    if max_reach_m is not None:
        reach = float(max_reach_m)
    if not (isinstance(center, (list, tuple)) and len(center) == 2 and _finite(*center)):
        raise ValueError('transfer_bay.center_xy must be finite x/y')
    if not (isinstance(cart_pose, (list, tuple)) and len(cart_pose) == 3 and _finite(*cart_pose)):
        raise ValueError('initial_staging.cart_body_pose must be finite x/y/yaw')
    if not (isinstance(load_xy, (list, tuple)) and len(load_xy) == 2 and _finite(*load_xy)):
        raise ValueError('initial_staging.cart_load_center_body_xy must be finite x/y')
    if not _finite(support_z):
        raise ValueError('initial_staging.cart_support_z_world must be finite')
    if not (isinstance(dims, (list, tuple)) and len(dims) == 3 and _finite(*dims) and all(d > 0 for d in dims)):
        raise ValueError('handling_unit.dimensions_m must be three finite positive extents')
    if not _finite(reach) or reach <= 0:
        raise ValueError('arm max reach must be finite and positive')

    # Robot park pose: x under the bay centre (centred lateral place reach); y + yaw on the tow corridor
    # the cart was towed along (the scenario's own cart_body_pose), so the heading matches the outbound
    # lane and the dock articulation is 0.
    bx, _by = (float(v) for v in center)
    corridor_y, corridor_yaw = float(cart_pose[1]), float(cart_pose[2])
    robot = (bx, corridor_y, _wrap(corridor_yaw))

    # Where the pick/place must reach from this pose, with the given hitch geometry.
    cart = cart_pose_from_hitch(robot, articulation, hitch_x=hitch_x, cart_hitch_x=cart_hitch_x)
    cx, cy, cyaw = cart
    lx, ly = (float(v) for v in load_xy)
    cos_y, sin_y = math.cos(cyaw), math.sin(cyaw)
    carton_xy = (cx + lx * cos_y - ly * sin_y, cy + lx * sin_y + ly * cos_y)
    carton_reach = math.hypot(carton_xy[0] - robot[0], carton_xy[1] - robot[1])
    bay_reach = math.hypot(center[0] - robot[0], center[1] - robot[1])
    offenders = []
    if carton_reach > reach:
        offenders.append(f'carton pick reach {carton_reach:.3f} m > max {reach:.3f} m')
    if bay_reach > reach:
        offenders.append(f'bay place reach {bay_reach:.3f} m > max {reach:.3f} m')
    if offenders:
        raise ValueError('derive_bay_park: infeasible park pose -- ' + '; '.join(offenders))
    return {'robot_base_pose': (round(robot[0], 4), round(robot[1], 4), round(robot[2], 6)),
            'reach': {'carton_pick_m': round(carton_reach, 4), 'bay_place_m': round(bay_reach, 4),
                      'max_m': round(reach, 4)},
            'inputs': {'transfer_bay_center_xy': [round(v, 4) for v in center],
                       'corridor_y': round(corridor_y, 4), 'corridor_yaw': round(_wrap(corridor_yaw), 6),
                       'carton_dimensions_m': [round(float(v), 4) for v in dims],
                       'cart_load_center_body_xy': [round(lx, 4), round(ly, 4)],
                       'cart_support_z_world': round(float(support_z), 4),
                       'hitch_x': round(float(hitch_x), 4), 'cart_hitch_x': round(float(cart_hitch_x), 4)}}


def bay_park_provenance(scenario, *, hitch_x=-0.62, cart_hitch_x=0.65, dataset=None, git_head=None):
    """A provenance stamp for a derived ``recorded_bay_park`` pose: the geometry it was derived FROM
    (so a change invalidates it BY NAME) plus the environment it was captured in.

    ``dataset`` / ``git_head`` are environment facts the caller supplies (the dataset manifest id and
    the repo HEAD -- see ``scripts/check_transfer_bench_staging.py`` and the fragment writer); the pose
    + the geometry fields come from the scenario via :func:`derive_bay_park`. Pure."""
    d = derive_bay_park(scenario, hitch_x=hitch_x, cart_hitch_x=cart_hitch_x)
    prov = {'dataset': dataset, 'git_head': git_head,
            'carton_dimensions_m': d['inputs']['carton_dimensions_m'],
            'transfer_bay_center_xy': d['inputs']['transfer_bay_center_xy'],
            'cart_support_z_world': d['inputs']['cart_support_z_world'],
            'cart_load_center_body_xy': d['inputs']['cart_load_center_body_xy'],
            'derived_robot_base_pose': list(d['robot_base_pose']), 'reach': d['reach'],
            'source': 'derived_from_scenario (sim-rerecord-bay-park-v3-v1)'}
    return prov


def check_bay_park_provenance(provenance, scenario, *, hitch_x=-0.62, cart_hitch_x=0.65):
    """Raise (BY NAME) when a frozen ``recorded_bay_park`` provenance stamp no longer matches the live
    scenario geometry, so a dataset or carton change can never silently replay a stale staged pose.

    Compares the geometry the pose was derived FROM (carton dimensions, transfer-bay centre, cart load
    centre, support z) against the current scenario. A match returns the live provenance; any drift
    raises ValueError naming each field with old -> new. ``provenance`` None/empty => no stamp to check
    (returns the freshly derived provenance)."""
    live = bay_park_provenance(scenario, hitch_x=hitch_x, cart_hitch_x=cart_hitch_x,
                               dataset=(provenance or {}).get('dataset'),
                               git_head=(provenance or {}).get('git_head'))
    if not provenance:
        return live
    drift = []
    for field in ('carton_dimensions_m', 'transfer_bay_center_xy', 'cart_load_center_body_xy',
                  'cart_support_z_world'):
        if field in provenance and provenance[field] != live[field]:
            drift.append(f'{field}: {provenance[field]} -> {live[field]}')
    if drift:
        raise ValueError('recorded_bay_park staging is STALE for this scenario (re-derive with '
                         'derive_bay_park / regenerate the fragment): ' + '; '.join(drift))
    return live


def _mission_pose(config, name):
    poses = config.get('mission_poses') if isinstance(config, dict) else None
    pose = poses.get(name) if isinstance(poses, dict) else None
    if isinstance(pose, (list, tuple)) and len(pose) == 3 and _finite(*pose):
        return tuple(float(v) for v in pose)
    return None


def _park_pose_name(config):
    """The pose a coupled OUTBOUND tow ends at (where PARK stationarises the train): the shelf-staging
    pose when the sales-floor re-sequence is configured, else the backroom-bay park (``shelf_approach``)."""
    return 'shelf_staging' if (isinstance(config, dict) and config.get('shelf_staging')) else 'shelf_approach'


def _outbound_lane_cfg(config):
    """The configured coupled OUTBOUND tow lane (the one whose goal is the park pose), or None.

    Follows whichever flow is configured: the dock-align / shelf-staging ``navigation.tow_lanes`` are
    read straight from the mission runtime config, so the derived dock/park base poses match the lane
    geometry the live coupled drive actually follows (no second source of truth)."""
    nav = config.get('navigation') if isinstance(config, dict) else None
    lanes = nav.get('tow_lanes') if isinstance(nav, dict) else None
    if not isinstance(lanes, (list, tuple)) or not lanes:
        return None
    park = _mission_pose(config, _park_pose_name(config))
    if park is not None:
        for lane in lanes:
            goal = lane.get('goal') if isinstance(lane, dict) else None
            if (isinstance(goal, (list, tuple)) and len(goal) == 3 and _finite(*goal)
                    and math.dist(goal[:2], park[:2]) <= 0.30):
                return lane
    first = lanes[0]
    return first if isinstance(first, dict) else None


def _derive_stage_base_pose(config, start_at):
    """Derive the build-time base spawn pose + cart yaw for a lane-driven coupled start.

    outbound -> the outbound tow lane START (the post-couple dock, with the authored ``start_cart_yaw``);
    park     -> the outbound tow lane GOAL (the stationarise point -- bay / shelf-staging pose);
    verify   -> the transfer/verify park pose (``mission_poses.shelf_approach``), where the coupled train
                sits for the transfer and the placed carton is verified.
    Returns ``(robot_pose, cart_yaw)``. Raises (fail-closed) when the configured flow lacks the pose the
    stage needs, so a shortcut never spawns at an invented pose."""
    lane = _outbound_lane_cfg(config)
    if start_at in ('outbound', 'park'):
        if lane is None:
            raise ValueError(
                f"start_at={start_at!r} staging needs a coupled tow lane in navigation.tow_lanes whose "
                f"goal is the park pose (mission_poses.{_park_pose_name(config)}); none is configured")
        key = 'start' if start_at == 'outbound' else 'goal'
        pose = lane.get(key)
        if not (isinstance(pose, (list, tuple)) and len(pose) == 3 and _finite(*pose)):
            raise ValueError(f"tow lane {lane.get('name')!r} {key} must be a finite (x, y, yaw) for the "
                             f'start_at={start_at!r} staging')
        cart_yaw = lane.get('start_cart_yaw', pose[2]) if start_at == 'outbound' else pose[2]
        return tuple(float(v) for v in pose), float(cart_yaw)
    if start_at == 'verify':
        pose = _mission_pose(config, 'shelf_approach')
        if pose is None:
            raise ValueError("start_at='verify' staging needs mission_poses.shelf_approach (the "
                             'transfer/verify park pose where the placed carton is verified)')
        return pose, pose[2]
    raise ValueError(f'no lane-derived base pose for start_at={start_at!r}')


def _bay_target_world(config, scenario):
    """The transfer bay-top target (x, y, top_z) a verify-start carton is PLACED on: the configured
    ground-truth target when set, else the scenario ``transfer_bay`` top. Fails closed when absent."""
    gt = config.get('transfer_ground_truth') if isinstance(config, dict) else None
    if isinstance(gt, dict) and gt.get('target'):
        t = gt['target']
        if isinstance(t, (list, tuple)) and len(t) >= 3 and _finite(*t[:3]):
            return (float(t[0]), float(t[1]), float(t[2]))
    bay = scenario.get('transfer_bay') if isinstance(scenario, dict) else None
    if isinstance(bay, dict):
        c = bay.get('center_xy')
        top = bay.get('top_z')
        if (isinstance(c, (list, tuple)) and len(c) == 2 and _finite(*c) and _finite(top)):
            return (float(c[0]), float(c[1]), float(top))
    raise ValueError("start_at='verify' needs a bay target: config transfer_ground_truth.target or "
                     'scenario.transfer_bay (center_xy + top_z)')


def resolve_start_staging(config):
    """Normalized ``config['start_staging']`` (a preset name or an object), or None when absent.

    The preset selects WHICH stage is staged and how the base pose is sourced (one preset per start_at;
    a mis-paired (preset, start_at) is refused). ``recorded_bay_park`` uses the hand-recorded backroom
    pose; the lane-driven presets (``dock_coupled``/``bay_parked``/``bay_placed``) DERIVE the base pose
    from the configured flow (``navigation.tow_lanes`` + ``mission_poses``) so it matches whichever
    dock-align / shelf-staging lane the live coupled drive follows. Object keys (all optional; derived /
    preset defaults): ``preset``, ``robot_base_pose`` [x, y, yaw] (override the derivation),
    ``articulation`` (rad, default 0 -- spawn aligned so the synthetic hitch can latch),
    ``carton_load_center_body_xy`` [x, y], ``carton_support_z_world`` (``DECK_TOP_Z_WORLD`` puts it on
    the deck). Fails closed on an unknown preset or a (preset, start_at) mismatch."""
    spec = config.get('start_staging') if isinstance(config, dict) else None
    if spec is None:
        return None
    if isinstance(spec, str):
        spec = {'preset': spec}
    if not isinstance(spec, dict):
        raise ValueError('start_staging must be a preset name or an object')
    start_at = config.get('start_at')
    preset = spec.get('preset', _PRESET_FOR_START_AT.get(start_at, 'recorded_bay_park'))
    if preset not in START_STAGING_PRESETS:
        raise ValueError(f'start_staging preset {preset!r} unknown; supported: {START_STAGING_PRESETS}')
    expected = _START_AT_FOR_PRESET[preset]
    if start_at != expected:
        raise ValueError(f'start_staging preset {preset!r} applies only to start_at={expected!r}, '
                         f'not start_at={start_at!r}')
    if spec.get('coupled', True) is not True:
        raise ValueError('start_staging: the staged train is still coupled at a synthesized start '
                         '(uncouple_on_return releases the cart at the parking spot during return)')
    # None => derive in staged_scenario from the single configured source of truth: the live
    # transfer_bay geometry for recorded_bay_park, the configured tow lane for the lane-driven presets
    # (dock_coupled / bay_parked / bay_placed). An explicit robot_base_pose overrides either derivation.
    robot = spec.get('robot_base_pose')
    articulation = spec.get('articulation', 0.0)
    load_xy = spec.get('carton_load_center_body_xy')
    support_z = spec.get('carton_support_z_world')
    provenance = spec.get('provenance')
    if robot is not None and not (isinstance(robot, (list, tuple)) and len(robot) == 3 and _finite(*robot)):
        raise ValueError('start_staging.robot_base_pose must be finite x/y/yaw')
    if not _finite(articulation) or abs(articulation) > math.radians(5):
        raise ValueError('start_staging.articulation must be finite and within the 5 deg attach gate')
    if load_xy is not None and not (isinstance(load_xy, (list, tuple)) and len(load_xy) == 2
                                    and _finite(*load_xy)):
        raise ValueError('start_staging.carton_load_center_body_xy must be finite x/y')
    if support_z is not None and not _finite(support_z):
        raise ValueError('start_staging.carton_support_z_world must be finite')
    if provenance is not None and not isinstance(provenance, dict):
        raise ValueError('start_staging.provenance must be an object')
    return {'preset': preset, 'start_at': start_at,
            'carton_target': 'bay' if start_at == 'verify' else 'cart',
            'robot_base_pose': None if robot is None else tuple(float(v) for v in robot),
            'articulation': float(articulation),
            'carton_load_center_body_xy': None if load_xy is None else tuple(float(v) for v in load_xy),
            'carton_support_z_world': None if support_z is None else float(support_z), 'coupled': True,
            'provenance': provenance}


def couple_start_poses(config, *, hitch_x=-0.62, cart_hitch_x=0.65):
    """For a ``start_at='couple'`` bench: the dock-aligned couple-start spawn (robot seat + cart dock)
    DERIVED from the active outbound tow lane's ``couple_dock`` -- the SAME single source PR 282's
    ``navigation.lanes.couple_seat_from_cart`` uses to make the outbound lane start FOLLOW the configured
    dock (the legacy EAST dock and the dock-align SOUTH dock both fall out of the one formula).

    A ``start_at=couple --stop-after outbound`` bench must spawn the post-collect train at the dock the
    live outbound lane expects, so the real couple latch seats the base at the lane-start seat and the
    outbound tow ACQUIRES the lane -- instead of a stale scenario spawn seating the base ~1 m / 90 deg
    off the lane start (coldwin23-2: ``lane_not_acquired_timeout``). Returns
    ``{'robot_seat', 'cart_dock', 'articulation', 'lane'}`` or ``None`` when no active outbound lane
    declares a ``couple_dock`` -- then the couple shortcut is byte-identical and the backroom-bay bench
    is unaffected. Pure."""
    if not isinstance(config, dict) or config.get('start_at') != 'couple':
        return None
    lane = _outbound_lane_cfg(config)
    dock = lane.get('couple_dock') if isinstance(lane, dict) else None
    if not (isinstance(dock, (list, tuple)) and len(dock) == 3 and _finite(*dock)):
        return None
    art = lane.get('couple_dock_articulation', 0.0)
    if not _finite(art):
        art = 0.0
    from ..navigation.lanes import couple_seat_from_cart
    seat = couple_seat_from_cart(tuple(float(v) for v in dock), hitch_x=hitch_x,
                                 cart_hitch_x=cart_hitch_x, articulation=float(art))
    return {'robot_seat': tuple(round(float(v), 6) for v in seat),
            'cart_dock': tuple(float(v) for v in dock), 'articulation': float(art),
            'lane': lane.get('name')}


def _staged_couple_scenario(scenario, config, *, hitch_x, cart_hitch_x):
    """``staged_scenario`` for ``start_at='couple'``: spawn the robot at the dock-derived couple SEAT and
    the cart at the configured ``couple_dock`` (cart left UNCOUPLED -- the real couple stage latches), so
    the couple->outbound bench runs from the dock-aligned seat. No active outbound ``couple_dock`` =>
    ``(scenario, None)`` byte-identical."""
    poses = couple_start_poses(config, hitch_x=hitch_x, cart_hitch_x=cart_hitch_x)
    if poses is None:
        return scenario, None
    variant = copy.deepcopy(scenario)
    st = variant.get('initial_staging') if isinstance(variant, dict) else None
    if not isinstance(st, dict) or not _finite(*st.get('cart_body_pose', (math.nan,))[:3]):
        raise ValueError('scenario.initial_staging.cart_body_pose is required to stage the couple start')
    robot, cart = poses['robot_seat'], poses['cart_dock']
    dyaw = _wrap(float(cart[2]) - float(st['cart_body_pose'][2]))
    st['robot_base_pose'] = list(robot)
    st['cart_body_pose'] = list(cart)
    yaws = dict(st.get('unit_yaws_world', {}))
    if _finite(yaws.get('cart')):                      # the carton turns with the cart to the new dock yaw
        yaws['cart'] = _wrap(float(yaws['cart']) + dyaw)
    st['unit_yaws_world'] = yaws
    record = {'mode': 'start_phase_shortcut:couple_dock', 'preset': 'couple_dock', 'start_at': 'couple',
              'carton_target': 'cart',
              'robot_base_pose': [round(v, 4) for v in robot],
              'robot_base_pose_origin': 'derived_from_couple_dock',
              'cart_body_pose': [round(float(v), 4) for v in cart],
              'couple_seat': [round(v, 4) for v in robot],
              'couple_dock': [round(float(v), 4) for v in cart],
              'articulation': poses['articulation'], 'coupled': False, 'lane': poses['lane'],
              'note': ('robot+cart spawned at the dock-aligned couple start derived from the active '
                       'outbound lane couple_dock (PR282 couple_seat_from_cart); cart left UNCOUPLED for '
                       'the real couple stage, which latches at this dock-aligned seat so the outbound '
                       'tow acquires the lane')}
    return variant, record


def staged_scenario(scenario, config, *, hitch_x=-0.62, cart_hitch_x=0.65):
    """``(scenario_for_build, record)``: the scenario whose ``initial_staging`` spawns the robot + cart
    at the configured start staging (a build-time pose, never a runtime teleport), plus the JSON record
    of what was staged for mission.json. Absent ``start_staging`` => ``(scenario, None)``, the SAME
    object (byte-identical). Pure; only the physical build and the shortcut synthesis consume it."""
    if isinstance(config, dict) and config.get('start_at') == 'couple':
        # couple has no staging preset; its dock-aligned spawn is derived from the active lane couple_dock.
        return _staged_couple_scenario(scenario, config, hitch_x=hitch_x, cart_hitch_x=cart_hitch_x)
    staging = resolve_start_staging(config)
    if staging is None:
        return scenario, None
    variant = copy.deepcopy(scenario)
    st = variant.get('initial_staging') if isinstance(variant, dict) else None
    if not isinstance(st, dict) or not _finite(*st.get('cart_body_pose', (math.nan,))[:3]):
        raise ValueError('scenario.initial_staging.cart_body_pose is required to stage the start')
    # Robot base pose: an explicit config pose wins; otherwise DERIVE it from the live scenario geometry
    # (transfer_bay + corridor + hitch), so a store or carton change re-derives rather than replaying the
    # stale v2 record. A scenario without a transfer_bay falls back to the historical v2 pose.
    derived, provenance = None, None
    stamp = staging.get('provenance') or {}
    # A supplied provenance stamp (e.g. a frozen fragment) is checked BY NAME against the live geometry
    # first, so a dataset or carton change can never silently replay a pose for the wrong store/carton.
    if staging.get('provenance') is not None:
        check_bay_park_provenance(staging['provenance'], variant, hitch_x=hitch_x, cart_hitch_x=cart_hitch_x)
    # Robot base pose: an explicit config pose wins; otherwise DERIVE it from the single configured
    # source of truth for that preset -- the live transfer_bay + corridor geometry for recorded_bay_park
    # (so a store or carton change re-derives rather than replaying the stale v2 record), the configured
    # tow lane for the lane-driven presets (so the spawn follows the same dock-align lane the live coupled
    # drive does -- no second seat derivation). A scenario without a transfer_bay falls back to the v2 pose.
    robot = staging['robot_base_pose']
    pose_origin = 'explicit_config'
    if robot is None:
        if staging['preset'] == 'recorded_bay_park':
            if isinstance(variant.get('transfer_bay'), dict):
                derived = derive_bay_park(variant, hitch_x=hitch_x, cart_hitch_x=cart_hitch_x,
                                          articulation=staging['articulation'])
                robot = derived['robot_base_pose']
                provenance = bay_park_provenance(variant, hitch_x=hitch_x, cart_hitch_x=cart_hitch_x,
                                                 dataset=stamp.get('dataset'), git_head=stamp.get('git_head'))
                pose_origin = 'derived_from_scenario'
            else:
                robot = RECORDED_BAY_PARK_ROBOT_POSE
                pose_origin = 'v2_fallback'
        else:
            robot, _derived_cart_yaw = _derive_stage_base_pose(config, staging['start_at'])
            pose_origin = 'derived_from_lane'
    elif staging.get('provenance') is not None and isinstance(variant.get('transfer_bay'), dict):
        provenance = bay_park_provenance(variant, hitch_x=hitch_x, cart_hitch_x=cart_hitch_x,
                                         dataset=stamp.get('dataset'), git_head=stamp.get('git_head'))
    cart = cart_pose_from_hitch(robot, staging['articulation'], hitch_x=hitch_x, cart_hitch_x=cart_hitch_x)
    dyaw = _wrap(cart[2] - float(st['cart_body_pose'][2]))
    st['robot_base_pose'] = list(robot)
    st['cart_body_pose'] = list(cart)
    if staging['carton_load_center_body_xy'] is not None:
        st['cart_load_center_body_xy'] = list(staging['carton_load_center_body_xy'])
    if staging['carton_support_z_world'] is not None:
        st['cart_support_z_world'] = staging['carton_support_z_world']
    yaws = dict(st.get('unit_yaws_world', {}))
    if _finite(yaws.get('cart')):                      # the carton turns with the cart
        yaws['cart'] = _wrap(float(yaws['cart']) + dyaw)
    st['unit_yaws_world'] = yaws
    if not _finite(st.get('cart_support_z_world')):
        raise ValueError('scenario.initial_staging.cart_support_z_world (carton support z) is required')
    # The carton ends up ON THE CART for the coupled loaded starts, or PLACED on the bay for a verify
    # start (a succeeded transfer has already set it down); plan_start_state derives this from start_at,
    # and apply_start_state repositions the carton accordingly after the build spawn.
    if staging['carton_target'] == 'bay':
        note = ('robot+cart spawned coupled at the transfer/verify park pose; carton synthesized PLACED '
                'level on the bay top (arm empty), as a succeeded transfer leaves it')
    elif staging['preset'] == 'recorded_bay_park':
        note = ('robot+cart spawned at the park->transfer handover (derived from the live transfer_bay + '
                'corridor geometry, else the recorded v2 tow); carton synthesized resting on the cart at '
                'carton_support_z_world')
    else:
        note = (f"robot+cart spawned coupled at the {staging['preset']} pose derived from the configured "
                'tow lane; carton synthesized resting level on the cart at carton_support_z_world')
    record = {'mode': f"start_phase_shortcut:{staging['preset']}", 'preset': staging['preset'],
              'start_at': staging['start_at'], 'carton_target': staging['carton_target'],
              'robot_base_pose': [round(v, 4) for v in robot], 'robot_base_pose_origin': pose_origin,
              'cart_body_pose': [round(v, 4) for v in cart],
              'articulation': staging['articulation'], 'coupled': True,
              'carton_load_center_body_xy': [round(float(v), 4) for v in st['cart_load_center_body_xy']],
              'carton_support_z_world': float(st['cart_support_z_world']),
              'source_runs': sorted(RECORDED_BAY_PARK_RUNS) if staging['preset'] == 'recorded_bay_park' else [],
              'note': note}
    if derived is not None:
        record['reach'] = derived['reach']
    if provenance is not None:
        record['provenance'] = provenance
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
    from .staged_start import summary_fields
    mode = summary_fields(config).get('staged_start_mode')
    if mode and mode != 'off':
        text += f' | {mode}'
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


# ---------------- single-stage isolation (dev bench planning; CPU only) ----------------
# What physical state each stage needs AT ENTRY -- i.e. what its upstream stages leave behind
# (seeded cart pose, carried unit, coupled state). Used to plan "run this one stage alone" dev
# bench runs and documented in docs/restock_stages.md. Keyed on ACTIONS (a test asserts no drift).
_STAGE_PRECONDITIONS = {
    'collect':  ('spawn defaults only: robot at the backstock pickup, cart at the dock, arm stowed, '
                 'the carton in the backstock -- no carried unit, uncoupled'),
    'couple':   ('carton loaded on the cart (collect done); robot still at the pickup, cart at the '
                 'dock, uncoupled'),
    'outbound': 'carton loaded AND the cart physically coupled behind the robot at the dock (couple done)',
    'park':     'loaded + coupled, towed to the transfer bay (outbound done)',
    'transfer': 'loaded + coupled + parked at the transfer bay, arm stowed (park done)',
    'verify':   'the carton placed on the bay top, still coupled (transfer done)',
    'return':   'stock committed at verify, still coupled at the bay (verify done)',
}


def stage_isolation(stage, stages=None):
    """Pure facts for running ``stage`` ALONE on a dev bench (no GPU/Isaac here). Every stage isolates.

    ``stop_after`` bounds the END and is supported for every stage (``mission.bounded_stages``). The
    START needs the prefix's physical end state synthesized (``shortcut.apply_start_state``):

    * ``collect`` begins from the top (no start_at).
    * ``couple`` has a start-phase shortcut (carton on the cart, UNCOUPLED) with no staging.
    * ``outbound`` / ``park`` / ``transfer`` each spawn the coupled, loaded train at that stage's
      build-time pose via a staging preset (``dock_coupled`` / ``bay_parked`` / ``recorded_bay_park``);
      the lane-driven presets derive the base pose from the configured tow lane.
    * ``verify`` spawns the carton PLACED on the bay (``bay_placed`` staging) and runs the stock-commit
      gate for real.
    * ``return`` is reached via ``start_at=verify --stop-after return``: ``Mission`` refuses a start_at
      AFTER verify (the only stock-commit gate -- skipping it would leave inventory uncredited), so
      return runs the one instant ground-truth verify (committing stock honestly) then tows home. The
      verify-skip guard is a safety bound and is NOT widened for the bench.

    Returns ``{stage, precondition, stop_after, start_at, start_staging, can_isolate, recipe,
    blocker, note}``. ``start_at``/``start_staging`` are the flags to pass (for ``return`` they are the
    verify-start flags; the mission still stops after ``return``)."""
    stages = tuple(stages) if stages else ACTIONS
    if stage not in stages:
        raise ValueError(f'stage {stage!r} not in {stages}')
    idx = stages.index(stage)
    stop_after = stage                                  # bounded_stages supports every stage
    note = None
    if idx == 0:
        start_at, start_staging = None, None
        recipe = f'--stop-after {stage}'
    elif stage == 'return':
        # Mission forbids start_at past verify (the stock-commit gate); synthesize the verify-entry
        # state and run the one instant verify, then return. The synthesized START state is verify's.
        start_at, start_staging = 'verify', _PRESET_FOR_START_AT['verify']
        recipe = f'--start-at verify --start-staging {start_staging} --stop-after return'
        note = ('start_at may not skip the verify stock-commit gate, so return is reached by '
                'synthesising the verify-entry (placed + coupled) state and running the one instant '
                'ground-truth verify (which commits stock), then the real return tow')
    else:                                               # couple / outbound / park / transfer / verify
        start_at = stage
        start_staging = _PRESET_FOR_START_AT.get(stage)   # None for couple (no spawn needed)
        recipe = (f'--start-at {stage}'
                  + (f' --start-staging {start_staging}' if start_staging else '')
                  + f' --stop-after {stage}')
    return {'stage': stage, 'precondition': _STAGE_PRECONDITIONS[stage], 'stop_after': stop_after,
            'start_at': start_at, 'start_staging': start_staging, 'can_isolate': True,
            'recipe': recipe, 'blocker': None, 'note': note}


def isolation_matrix(stages=None):
    """The :func:`stage_isolation` record for every stage, in order."""
    return [stage_isolation(s, stages) for s in (tuple(stages) if stages else ACTIONS)]


def shortcut_summary(shortcut, config, staging=None):
    """mission.json fields for a start-phase shortcut run: top-level ``start_at``, the shortcut record
    (with the applied ``staging`` when the recorded start staging ran), the ``start_staging_mode``
    workaround label and the walkthrough caption."""
    from .staged_start import summary_fields
    out = {'start_at': shortcut['start_at'], 'shortcut': dict(shortcut),
           'walkthrough_caption': walkthrough_caption(config), **summary_fields(config)}
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
    couple_cart: bool            # author the physical robot<->cart hitch at synthesis (coupled starts)
    carton_target: str           # 'cart' (resting on the rollwagen) | 'bay' (placed on the bay top)
    note: str


def plan_start_state(scenario, start_at, stages=None, config=None):
    """Pure start-state plan from the scenario's measured staging. No Isaac import.

    Computes the carton CENTRE pose + whether the shortcut must author the robot<->cart hitch. The
    carton rests on the cart for the loaded starts (``couple``/``outbound``/``park``/``transfer``) and
    is PLACED on the bay top for ``verify`` (a succeeded transfer has set it down -- the bay target comes
    from ``config['transfer_ground_truth']['target']`` or ``scenario['transfer_bay']``). Raises
    ValueError (fail-closed) on an unsupported ``start_at`` or missing/malformed staging, so a
    misconfigured shortcut refuses to synthesize rather than place bodies at an invented pose.
    """
    stages = tuple(stages) if stages else ACTIONS
    if start_at not in stages:
        raise ValueError(f'start_at {start_at!r} not in mission stages {stages}')
    if start_at not in _SUPPORTED_START_AT:
        raise ValueError(
            f'start-phase shortcut does not synthesize start_at={start_at!r}; '
            f'supported: {_SUPPORTED_START_AT} (return is reached via start_at=verify --stop-after return)')
    skipped = tuple(stages[:stages.index(start_at)])
    if skipped != _EXPECTED_SKIP[start_at]:
        raise ValueError(f'start_at={start_at!r} skips {skipped}; expected {_EXPECTED_SKIP[start_at]} '
                         '(mission stages must be the canonical in-order prefix)')

    staging = scenario.get('initial_staging') if isinstance(scenario, dict) else None
    if not isinstance(staging, dict):
        raise ValueError('scenario.initial_staging is required to synthesize a start state')
    handling = scenario.get('handling_unit', {})
    dims = handling.get('dimensions_m')
    if not (isinstance(dims, (list, tuple)) and len(dims) == 3 and _finite(*dims) and all(d > 0 for d in dims)):
        raise ValueError('handling_unit.dimensions_m must be three finite positive extents')
    yaws = staging.get('unit_yaws_world', {})
    couple_cart = start_at in _COUPLE_AT_SYNTHESIS
    carton_target = 'bay' if start_at == 'verify' else 'cart'

    if carton_target == 'bay':
        # verify: the carton is already PLACED on the bay top (centre half its height above the top),
        # level, at the shelf placement yaw; the cart is empty but still coupled.
        bx, by, top_z = _bay_target_world(config or {}, scenario)
        carton_pos = (bx, by, float(top_z) + float(dims[2]) / 2.0)
        shelf_yaw = yaws.get('shelf', 0.0)
        if not _finite(shelf_yaw):
            shelf_yaw = 0.0
        carton_quat = _yaw_to_wxyz(float(shelf_yaw))
        note = ('synthesized carton placed level on the bay top (arm empty); cart physically coupled '
                'behind the robot (hitch authored), parked at the transfer/verify pose')
    else:
        cart_pose = staging.get('cart_body_pose')
        load_xy = staging.get('cart_load_center_body_xy')
        support_z = staging.get('cart_support_z_world')
        cart_yaw_on_unit = yaws.get('cart')
        if not (isinstance(cart_pose, (list, tuple)) and len(cart_pose) == 3 and _finite(*cart_pose)):
            raise ValueError('initial_staging.cart_body_pose must be finite x/y/yaw')
        if not (isinstance(load_xy, (list, tuple)) and len(load_xy) == 2 and _finite(*load_xy)):
            raise ValueError('initial_staging.cart_load_center_body_xy must be finite x/y')
        if not _finite(support_z):
            raise ValueError('initial_staging.cart_support_z_world must be finite')
        if not _finite(cart_yaw_on_unit):
            raise ValueError('initial_staging.unit_yaws_world.cart must be a finite yaw')
        cx, cy, cyaw = (float(v) for v in cart_pose)
        lx, ly = (float(v) for v in load_xy)
        # Load centre is given in the cart BODY frame; rotate by the cart's world yaw into the map frame.
        cos_y, sin_y = math.cos(cyaw), math.sin(cyaw)
        wx = cx + lx * cos_y - ly * sin_y
        wy = cy + lx * sin_y + ly * cos_y
        wz = float(support_z) + float(dims[2]) / 2.0   # carton CENTRE sits half its height above the deck
        carton_pos = (wx, wy, wz)
        carton_quat = _yaw_to_wxyz(float(cart_yaw_on_unit))
        note = ('synthesized carton resting level at the cart load centre (not welded; the arm re-grasps it)'
                + ('; cart physically coupled behind the robot (hitch authored)'
                   if couple_cart else '; cart at the dock, left uncoupled for the real couple stage'))
    return StartState(start_at=start_at, skipped=skipped, carton_position=carton_pos,
                      carton_orientation_wxyz=carton_quat, couple_cart=couple_cart,
                      carton_target=carton_target, note=note)


def apply_start_state(assembly, config, scenario, *, settle_steps=None, _rigid_prim=None):
    """Establish the planned start state in the live stage. Isaac-only for the stepping; the settle +
    rest/heading verify logic is pure (``start_settle``) and CPU-tested.

    The settle is SIM-TIME and convergence-verified (``start_settle.converge_settle`` +
    ``config['start_state_settle']``), NOT a fixed tick count -- ``settle_steps`` is accepted for
    backward compatibility only and is ignored. ``_rigid_prim`` is a test seam (a SingleRigidPrim-like
    factory); production leaves it None and uses Isaac's SingleRigidPrim. Returns the StartState applied.
    """
    del settle_steps  # legacy tick-count arg; the settle is now sim-time + convergence-verified
    start_at = config.get('start_at')
    plan = plan_start_state(scenario, start_at, stages=config.get('stages'), config=config)

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
    observe = config.get('transfer_observe_joints') if plan.start_at == 'transfer' else None
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
    # Settle in SIM TIME until the synthesized bodies actually stop moving, NOT for a fixed tick count
    # (sim-couple-shortcut-settle-dt-v1): a fixed count is a different amount of sim time at each physics
    # step and is unrelated to whether the payload has come to rest, so a coarser step or a faster pacing
    # profile silently changed the settle (coldwin25c/25d froze the couple-start cart mid-wobble on the
    # portal_floor_box flat-floor box at a tipped, mis-headed pose -> blocked_no_route). The step is read
    # from the LIVE world; the convergence window + verify tolerances are all in sim seconds.
    from .start_settle import resolve_settle_params, converge_settle, resting_check, heading_check
    params = resolve_settle_params(config)
    if hasattr(assembly.world, 'get_physics_dt'):
        dt = float(assembly.world.get_physics_dt())
    else:                                   # pure test seam / no live World: resolve the config step
        from .physics_step import resolve_dt
        dt = float(resolve_dt(config))

    cart_body = getattr(assembly, 'cart_body', None)
    mobile = getattr(assembly, 'mobile', None)
    track_cart = (not plan.couple_cart) and cart_body is not None
    names = ('carton', 'cart') if track_cart else ('carton',)

    def _sample():
        cpose, cz, _, _ = _measure(carton)
        out = {'carton': (cpose.x, cpose.y, cz, cpose.yaw)}
        if track_cart:
            kpose, kz, _, _ = _measure(cart_body)
            out['cart'] = (kpose.x, kpose.y, kz, kpose.yaw)
        return out

    # Couple start (cart UNCOUPLED): hold the base + cart at their authored dock seat THROUGH the settle
    # so the portal_floor_box flat-floor box (drop_subfloor removes the Ground_exterior sub-floor) cannot
    # tip or slide the free train before the couple reads the dock -- reproducing the stable towstaging3
    # seat at any floor/step. Released below before the mission drives; begin('couple') re-pins the cart.
    held = False
    cart_spawn_z = None
    if (not plan.couple_cart) and cart_body is not None:
        cart_spawn_z = _measure(cart_body)[1]
    if (not plan.couple_cart) and params['hold_during_settle'] and mobile is not None:
        if hasattr(mobile, 'engage_base_full_brake'):
            mobile.engage_base_full_brake()
        if hasattr(mobile, 'hold_cart_pin'):
            mobile.hold_cart_pin()
        held = True

    coupling = None
    if plan.couple_cart:
        # transfer start: the couple/outbound/park stages are skipped, so the cart must already be
        # physically coupled behind the robot (both spawned parked+aligned at the shelf by the
        # variant scenario). Settle the carton onto the cart (arm held) in sim time, author the real
        # hitch through the SAME mobile.attach contract the couple stage uses (fresh, stopped, aligned
        # observation), then assert the latch is healthy -- a silent uncoupled cart would let the arm's
        # transfer reaction shove it.
        def _held_step():
            assembly.hold()
            assembly.world.step(render=False)
        settle = converge_settle(step=_held_step, sample=_sample, dt=dt, params=params, names=names)
        dock = _dock_observation(assembly)
        pre = assembly.mobile.coupling_state(dock)
        if pre['joint_present']:
            raise RuntimeError('shortcut: stage already coupled before the synthetic hitch request')
        coupling = assembly.mobile.attach(dock)
        if not coupling.get('healthy'):
            raise RuntimeError('shortcut: synthetic cart coupling is not healthy '
                               f"(residual {coupling.get('hitch_error_m')} m); check the variant "
                               'scenario cart_body_pose vs robot_base_pose hitch alignment')
        settle_after = converge_settle(step=_held_step, sample=_sample, dt=dt, params=params, names=names)
        settle = {'place': settle, 'post_attach': settle_after}
        # The settle loops above held the arm via assembly.hold(), which LATCHES
        # command_gate.faulted=True (hold() is otherwise a terminal e-stop and is never reset on its
        # own). The mission that now runs MUST drive the arm (the executor/cuMotion transfer), so re-arm
        # the gate here -- otherwise every arm command is silently dropped at ``if self.faulted`` and the
        # arm stays frozen at the stow pose (execretest: the executor transfer faulted 1.3 rad on the
        # reference-vs-frozen-arm divergence, misread as a clock/self-collision fault). Only the
        # transfer-start shortcut holds the arm (couple start leaves command_gate untouched).
        assembly.command_gate.resume()
    else:
        settle = converge_settle(step=lambda: assembly.world.step(render=False),
                                 sample=_sample, dt=dt, params=params, names=names)
        # Verify the synthesized cart has come to REST on supported floor at the configured dock BEFORE
        # releasing the stage, so a mis-seated train is named (and refused) here instead of towing into a
        # no-route. Needs a measurable cart body; skipped on a minimal test seam (no cart_body).
        if cart_body is not None:
            kpose, kz, klin, kang = _measure(cart_body)
            roll_deg, pitch_deg = _roll_pitch_deg(cart_body)
            measured = dict(z=kz, roll_rad=math.radians(roll_deg), pitch_rad=math.radians(pitch_deg),
                            lin_speed=float((klin[0] ** 2 + klin[1] ** 2 + klin[2] ** 2) ** 0.5),
                            ang_speed=float((kang[0] ** 2 + kang[1] ** 2 + kang[2] ** 2) ** 0.5))
            expected_z = cart_spawn_z if cart_spawn_z is not None else kz
            ok, reason = resting_check(measured, {'z': expected_z}, params)
            if not ok:
                raise RuntimeError(reason)
            configured_yaw = float(scenario['initial_staging']['cart_body_pose'][2])
            ok, residual, reason = heading_check(kpose.yaw, configured_yaw, params['heading_tol_rad'])
            if not ok:
                raise RuntimeError(reason)
            print(f'START_STATE_VERIFY cart resting z={kz:.4f} tilt=({roll_deg:.2f},{pitch_deg:.2f})deg '
                  f'heading={math.degrees(kpose.yaw):.1f}deg vs dock {math.degrees(configured_yaw):.1f}deg '
                  f'(residual {math.degrees(residual):.1f}deg) -- OK', flush=True)
        if held:
            # Drop the settle holds so the couple nav can drive from the seated pose; begin('couple')
            # re-pins the cart (hold_cart_through_couple). command() also releases the base full-brake.
            if hasattr(mobile, 'release_cart_brake'):
                mobile.release_cart_brake(force=True)
            if hasattr(mobile, 'release_base_full_brake'):
                mobile.release_base_full_brake()

    print(f'START_PHASE_SHORTCUT applied start_at={plan.start_at} skipped={list(plan.skipped)} '
          f'carton_pos={tuple(round(v, 4) for v in plan.carton_position)} '
          f'carton_quat_wxyz={tuple(round(v, 4) for v in plan.carton_orientation_wxyz)} '
          f'coupled={None if coupling is None else coupling.get("healthy")} '
          f'settle={_settle_brief(settle)} dt={dt:.5f} held_at_seat={held}',
          flush=True)
    return plan


def _measure(body):
    """Isaac body measurement (Pose2, z, linear, angular) -- a thin wrapper so the shortcut's settle/
    verify can read any SingleRigidPrim-like body. Imported lazily (CPU-safe: no isaacsim import)."""
    from .isaac_port import body_measurement
    return body_measurement(body)


def _roll_pitch_deg(body):
    from .isaac_port import body_roll_pitch_deg
    return body_roll_pitch_deg(body)


def _settle_brief(settle):
    """Compact settle report for the console line (handles the single and coupled {place,post_attach})."""
    def one(s):
        return f"converged={s['converged']} sim={s['elapsed_s']:.2f}s steps={s['steps']}"
    if isinstance(settle, dict) and 'place' in settle:
        return f"place[{one(settle['place'])}] post_attach[{one(settle['post_attach'])}]"
    return one(settle)


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
