"""CPU DRY-RUN of the restock mission TAIL for the backroom-bay demo (sim-video-verify-return-v1, vr1).

Everything AFTER the transfer has never run live: the transfer's fm461 ground-truth placement result, the
mission-level ``verify``, the uncouple-at-parking-spot ``return`` (tow -> brake cart -> uncouple -> drive
home EMPTY) and the mission end (mission.json + the stage timeline the walkthrough cuts on). This drives
that tail through the REAL code -- the behaviour tree (``twin.restock.bt``, via ``run_once_bt`` so the
mission.json / bt_events.jsonl the run writes are the ones checked), the real ``Mission`` + sqlite
``Inventory``, the real ``RuntimeAdapter`` configured exactly as ``isaac_bundle.create`` configures it from
the runtime config, the real ``TransferController`` fm461 ground-truth VERIFY phase (so the transfer
evidence has the controller's true shape) and the real ``GroundTruthInventoryVerifier`` -- against a small
kinematic stand-in world (robot / cart / carton poses, coupling latch, parked flag) consistent with the bay
geometry. Navigation is a straight-line kinematic stub that honours the controller contract (coupled flag,
arrival, measured parking); the coupled return LANE is proven by ``check_tow_lanes.py`` (real controller),
and ``--environment`` adds the UNCOUPLED home leg with the real navigation controller + an emulated lidar
on the static map (+ the parked cart and the bay as live returns).

Checks (PASS/FAIL/WARN lines, exit 0 iff no FAIL):
  config_*            the tail's config keys (poses, return lane, goal articulation, ground-truth switches)
  tray_on_bay_*       the ground-truth verify tolerances against the bay top and the placed tray
  tail_*              BT dry-runs: nominal completion, an unsettled uncouple, a failed home leg retried, a
                      tray off the bay, and the default-off (no vr1 switches) behaviour
  home_leg_*          (with --environment) the real uncoupled controller drives cart_parking -> home
  return_tracking_*   (with --environment --return-tracking) the REAL COUPLED follower drives the return
                      from the recorded end-of-verify train pose (cart lagging at articulation ~0.2) to
                      cart_parking -- exercising the cart-yaw tracking law that the coupled return is
                      otherwise STUBBED for here (the warm3 stall; guards tow_lane_cart_yaw_rebase)

Usage (the integration lane's pre-boot gate; py_trees on the path for the BT checks):
  PYTHONPATH=scripts:$BTDIR python3 scripts/check_mission_tail.py \\
      --runtime-config <mission-runtime.json> --scenario <mission-scenario.json> \\
      [--environment <environment.json>] [--camera-config <camera-config-bay.json>] [--return-tracking]
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import sys
import tempfile
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace as NS

sys.path.insert(0, str(Path(__file__).resolve().parent))

from twin.restock.ground_truth_verify import (GroundTruthInventoryVerifier, config_block,  # noqa: E402
                                              from_config as gt_verifier_from_config,
                                              ground_truth_place_actions, mission_ground_truth_sources,
                                              mission_modes, stage_modes)
from twin.restock.inventory import Inventory, Job  # noqa: E402
from twin.restock.mission import Mission  # noqa: E402
from twin.restock.runtime import Packet, RuntimeAdapter  # noqa: E402

DT = 1 / 60
HITCH_TO_CART_M = 0.62 + 0.65        # MobileSpec |hitch_x| + cart_hitch_x: cart centre behind the base
TAIL_STAGES = ('transfer', 'verify', 'return')


def have_py_trees():
    try:
        import py_trees  # noqa: F401
        return True
    except ImportError:
        return False


# --------------------------------------------------------------------------------------------- world
class World:
    """Kinematic stand-in for the parked train at the bay: robot/cart poses, the coupling latch, the
    handling unit's world pose, plus injectable transients (unsettled ticks after the cart pin)."""

    def __init__(self, robot_pose, *, unit_offset=(0.0, 0.0, 0.0), unsettle_after_pin=0,
                 unit_falls_at_verify=False):
        self.robot = list(robot_pose)
        self.cart = self._towed_cart()
        self.coupled = True
        self.moving = False
        self.unit = None                 # (x, y, z, tilt) world centre once the transfer places it
        self.unit_offset = tuple(unit_offset)
        self.contact = False
        self.unsettle_after_pin = int(unsettle_after_pin)
        self.unsettled = 0
        self.unit_falls_at_verify = bool(unit_falls_at_verify)
        self.now = 0.0

    def _towed_cart(self):
        x, y, yaw = self.robot
        return [x - HITCH_TO_CART_M * math.cos(yaw), y - HITCH_TO_CART_M * math.sin(yaw), yaw]

    def move_robot(self, pose):
        self.robot = list(pose)
        if self.coupled:
            self.cart = self._towed_cart()


class PortStub:
    """The IsaacPort contract the RuntimeAdapter drives: one measured Packet per tick on the mission clock
    (tick*dt), begin_action -> on_action('verify') begins the verifier on the SAME clock (as the bundle)."""

    def __init__(self, world, verifier):
        self.world, self.verifier = world, verifier
        self.tick = 0
        self.actions = []
        self.stops = 0

    def mission_time(self):
        return self.tick * DT

    def begin_action(self, action, job):
        self.actions.append(action)
        if action == 'verify':
            if self.world.unit_falls_at_verify and self.world.unit is not None:
                x, y, z, _ = self.world.unit          # injected (once): the tray slid off the bay after release
                self.world.unit = (x, y, z - 0.73, 0.0)
                self.world.unit_falls_at_verify = False
            self.verifier.begin(job, self.mission_time())

    def advance(self, dt):
        self.tick += 1
        now = self.tick * DT
        w = self.world
        w.now = now
        unsettled = w.unsettled > 0
        w.unsettled = max(0, w.unsettled - 1)
        speed = 0.05 if unsettled else 0.0
        docking = NS(robot_speed=speed, cart_speed=0.0, robot_yaw_rate=0.0, cart_yaw_rate=0.0, age=0.0)
        parked = not w.moving and not unsettled
        evidence = self.verifier(now)
        return Packet(f'physics:{self.tick}', 'sensor', now, NS(pose=tuple(w.robot)), None, docking, None,
                      True, w.coupled, 0.0, parked, evidence)

    def controls(self, controls):
        pass

    def stop(self):
        self.stops += 1


class NavStub:
    """Straight-line kinematic navigation honouring the controller contract the adapter relies on: a
    measured coupled flag that disagrees with the goal's -> 'coupling_changed' until the blocked timeout;
    arrival -> 'succeeded' (repeated while parked); optional one-shot failure of a named goal."""

    def __init__(self, world, *, speed=0.25, blocked_timeout=40.0, fail_goal_once=None):
        self.world, self.speed, self.blocked_timeout = world, speed, blocked_timeout
        self.goals = []
        self.goal = None
        self.coupled = None
        self.fail_goal_once = tuple(fail_goal_once) if fail_goal_once else None
        self._blocked = 0.0

    def start(self, goal, coupled, **kwargs):
        self.goal, self.coupled = tuple(float(v) for v in goal), bool(coupled)
        self.goals.append({'goal': list(self.goal), 'coupled': self.coupled, 'kwargs': sorted(kwargs)})
        self._blocked = 0.0
        self._ticks = 0

    def step(self, dt, packet):
        w = self.world
        if packet.coupled != self.coupled:
            self._blocked += dt
            w.moving = False
            if self._blocked >= self.blocked_timeout:
                return NS(status='failed', reason='coupling_changed_timeout', linear=0.0, angular=0.0)
            return NS(status='running', reason='coupling_changed', linear=0.0, angular=0.0)
        self._ticks += 1
        if (self.fail_goal_once is not None and self._ticks > 30
                and math.dist(self.goal[:2], self.fail_goal_once[:2]) < 1e-6):
            self.fail_goal_once = None
            w.moving = False
            return NS(status='failed', reason='blocked_no_route_timeout', linear=0.0, angular=0.0)
        x, y, _ = w.robot
        gx, gy, gyaw = self.goal
        d = math.hypot(gx - x, gy - y)
        if d <= 1e-6:
            w.moving = False
            w.move_robot((gx, gy, gyaw))
            return NS(status='succeeded', reason='arrived_stopped', linear=0.0, angular=0.0)
        stride = min(d, self.speed * dt)
        heading = math.atan2(gy - y, gx - x)
        nx, ny = x + stride * math.cos(heading), y + stride * math.sin(heading)
        w.moving = True
        w.move_robot((nx, ny, gyaw if stride >= d else (heading if not self.coupled else w.robot[2])))
        return NS(status='running', reason='tracking', linear=stride / dt, angular=0.0)

    def cancel(self):
        self.world.moving = False


class MobileStub:
    """The MobileController calls the return legs make; uncouple() refuses a moving/stale train exactly
    like ``twin.mobile.Coupler.detach`` (before touching the joint)."""

    def __init__(self, world):
        self.world = world
        self.calls = []

    def park(self):
        pass

    def hold(self):
        pass

    def command(self, v, omega, docking, *, transport_ready=False):
        pass

    def hold_cart_pin(self):
        self.calls.append('hold_cart_pin')
        self.world.unsettled = self.world.unsettle_after_pin

    def release_cart_brake(self, *, force=False):
        self.calls.append('release_cart_brake')

    def uncouple(self, docking):
        vals = (docking.robot_speed, docking.cart_speed, docking.robot_yaw_rate, docking.cart_yaw_rate)
        if not self.world.coupled or max(abs(v) for v in vals) > 0.02 or not 0 <= docking.age <= 0.1:
            self.calls.append('uncouple_refused')
            raise ValueError('detach requires coupled, fresh, stopped bodies')
        self.calls.append('uncouple')
        self.world.coupled = False
        return {'requested_attached': False, 'joint_present': False, 'healthy': False}

    def coupling_state(self, docking):
        return {'healthy': self.world.coupled}


class _NoMotion:
    def stop(self):
        pass


class TransferStub:
    """Owns a REAL ``TransferController`` and runs only its fm461 GROUND-TRUTH VERIFY phase: the place
    itself (reach / release / stow) is the integration lane's and is out of this dry-run, so ``start``
    leaves the controller exactly where a released, stowed transfer stands -- the carton resting on the
    bay (the world model), the matched cart observation recorded, the bay target set -- and ``step``
    feeds it the sim-measured carton pose each tick, as ``isaac_factory._capture_transfer`` does."""

    cart_location = 'rollwagen'

    def __init__(self, world, profile, *, bay_target_world, transfer_ground_truth=True):
        from twin.restock_manipulation.controller import TransferController
        self.world, self.profile = world, profile
        self.bay_target_world = tuple(bay_target_world)
        self.ctl = TransferController(_NoMotion(), {profile.sku: profile},
                                      transfer_ground_truth=transfer_ground_truth)

    def _to_base(self, xyz):
        x, y, yaw = self.world.robot
        c, s = math.cos(yaw), math.sin(yaw)
        dx, dy = xyz[0] - x, xyz[1] - y
        return (c * dx + s * dy, -s * dx + c * dy, xyz[2])

    def start(self, operation, job):
        from twin.restock_manipulation.types import Observation
        self.ctl.start(operation, job)
        bx, by, bz = self.bay_target_world
        ox, oy, oz = self.world.unit_offset
        self.world.unit = (bx + ox, by + oy, bz + self.profile.dimensions[2] / 2.0 + oz, 0.0)
        now = self.world.now
        cart = self._to_base((self.world.cart[0], self.world.cart[1], 1.01))
        self.ctl.evidence.append(asdict(Observation(f'cart:{now:.3f}', 'sensor', now, job['sku'],
                                                    'rollwagen', (*cart, 0.0, 0.0, 0.0, 1.0), 1.0)))
        self.ctl.target_surface = (*self._to_base(self.bay_target_world), 0.0, 0.0, 0.0, 1.0)
        self.ctl.poses = {}           # the reach poses the (skipped) observe phase builds; verify uses none
        self.ctl.release_time = now
        self.ctl._transition('verify')

    def step(self, dt, _sensors):
        from twin.restock_manipulation.types import Sensors
        now = self.world.now
        placed = (*self._to_base(self.world.unit[:3]), 0.0, 0.0, 0.0, 1.0)
        return self.ctl.step(dt, Sensors(now, (), None, True, bool(self.world.contact), now, now,
                                         (0.0,) * 6, placed_pose=placed))

    def stop(self):
        pass


# ------------------------------------------------------------------------------------------ dry run
def _job(scenario):
    return Job(**scenario['job'])


def _profile(scenario, config):
    from twin.restock_manipulation.types import ProductProfile
    unit = scenario['handling_unit']
    # top grasp: the verify phase needs no CAD wall-pose policy (the side grasp is the place's business)
    return ProductProfile(unit['sku'], tuple(unit['dimensions_m']), tuple(unit['tcp_quaternion_xyzw']),
                          max_reach=float(unit.get('max_reach_m', 1.3)), grasp_mode='top')


def bay_target_world(scenario, config):
    gt = config.get('transfer_ground_truth') if isinstance(config.get('transfer_ground_truth'), dict) else {}
    if gt.get('target'):
        return tuple(float(v) for v in gt['target'][:3])
    bay = scenario['transfer_bay']
    return (float(bay['center_xy'][0]), float(bay['center_xy'][1]), float(bay['top_z']))


def build_tail(config, scenario, *, unit_offset=(0.0, 0.0, 0.0), unsettle_after_pin=0,
               fail_goal_once=None, unit_falls_at_verify=False):
    """Assemble world + stubs + the REAL RuntimeAdapter exactly as ``isaac_bundle.create`` wires it."""
    poses = config['mission_poses']
    world = World(poses['shelf_approach'], unit_offset=unit_offset, unsettle_after_pin=unsettle_after_pin,
                  unit_falls_at_verify=unit_falls_at_verify)
    verifier = gt_verifier_from_config(config, scenario, lambda: world.unit,
                                       read_contact=lambda: world.contact,
                                       source=config.get('verification_source', 'rollwagen'))
    if verifier is None:
        from twin.restock.verification import InventoryVerifier
        verifier = InventoryVerifier(lambda *a: None, source=config.get('verification_source', 'rollwagen'))
    port = PortStub(world, verifier)
    nav = NavStub(world, fail_goal_once=fail_goal_once)
    mobile = MobileStub(world)
    gt_cfg = config.get('transfer_ground_truth')
    gt_on = bool(gt_cfg.get('enabled') if isinstance(gt_cfg, dict) else gt_cfg)
    manip = TransferStub(world, _profile(scenario, config), bay_target_world=bay_target_world(scenario, config),
                         transfer_ground_truth=gt_on)
    adapter = RuntimeAdapter(nav, mobile, manip, port, poses, fixture_mode=False,
                             goal_articulation=config.get('goal_articulation'),
                             uncouple_on_return=bool(config.get('uncouple_on_return')),
                             ground_truth_place_actions=ground_truth_place_actions(config))
    adapter.stage_modes = stage_modes(config)
    return NS(world=world, port=port, nav=nav, mobile=mobile, manip=manip, adapter=adapter, verifier=verifier)


def run_tail(config, scenario, out_dir, **inject):
    """Drive transfer -> verify -> return -> finalize through ``run_once_bt`` (the live BT entry point,
    which also writes mission.json + bt_events.jsonl into ``out_dir``). Returns (mission, summary, rig)."""
    from twin.restock.bt.integration import run_once_bt
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rig = build_tail(config, scenario, **inject)
    db = Inventory(str(out_dir / 'inventory.sqlite'))
    for row in scenario.get('initial_inventory', []):
        db.seed(row['location'], row['sku'], int(row['quantity']))
    stages = config.get('stages') or None

    def factory():
        return Mission(db, _job(scenario), rig.adapter, stages=stages, start_at='transfer',
                       timeout_s=float(config.get('stage_timeout_s', 120.)),
                       ground_truth_sources=mission_ground_truth_sources(config))

    bundle = NS(adapter=rig.adapter)
    mission, summary = run_once_bt(db, scenario, bundle, dict(config, bt_ros_events=False), out_dir,
                                   mission_factory=factory)
    summary = dict(summary, events=db.events(scenario['job']['job_id']))
    rig.db = db
    return mission, summary, rig


# ---------------------------------------------------------------------------------------- checks
class Report:
    def __init__(self, stream=sys.stdout):
        self.rows, self.stream = [], stream

    def add(self, verdict, name, **detail):
        self.rows.append((verdict, name, detail))
        print(f'{verdict} {name} {json.dumps(detail, sort_keys=True, default=str)}', file=self.stream,
              flush=True)

    def check(self, ok, name, **detail):
        self.add('PASS' if ok else 'FAIL', name, **detail)
        return ok

    @property
    def failed(self):
        return [r for r in self.rows if r[0] == 'FAIL']


def check_config(config, scenario, rep):
    poses = config.get('mission_poses', {})
    job = scenario['job']
    stages = tuple(config.get('stages') or ())
    rep.check(all(k in poses for k in ('shelf_approach', 'cart_parking', 'home')), 'config_tail_poses',
              shelf_approach=poses.get('shelf_approach'), cart_parking=poses.get('cart_parking'),
              home=poses.get('home'), backaway=poses.get('cart_parking_backaway'))
    rep.check(not stages or stages[-3:] == TAIL_STAGES, 'config_stages_end_with_tail', stages=list(stages))
    rep.check(bool(config.get('uncouple_on_return')) and 'cart_parking' in poses,
              'config_uncouple_at_parking_enabled', uncouple_on_return=config.get('uncouple_on_return'))
    gt = config.get('transfer_ground_truth')
    gt = gt if isinstance(gt, dict) else {}
    rep.check(bool(gt.get('enabled')) and gt.get('location') == job['shelf_id'],
              'config_transfer_ground_truth_targets_job_shelf', location=gt.get('location'),
              shelf_id=job['shelf_id'])
    bay = scenario.get('transfer_bay') or {}
    target = bay_target_world(scenario, config)
    if bay:
        err = math.dist(target, (bay['center_xy'][0], bay['center_xy'][1], bay['top_z']))
        rep.check(err <= 0.01, 'config_ground_truth_target_is_bay_top', target=target, error_m=round(err, 4),
                  bay_center_xy=bay['center_xy'], bay_top_z=bay['top_z'])
    unit = scenario['handling_unit']['dimensions_m']
    free = gt.get('free_dimensions') or []
    rep.check(len(free) == 3 and all(f >= u for f, u in zip(sorted(free[:2]), sorted(unit[:2])))
              and free[2] >= unit[2], 'config_ground_truth_free_space_fits_tray', free=free, tray=unit)
    commissioned = set(config.get('commissioning_required_locations') or ())
    gt_verify = config_block(config) is not None
    camera_verify_possible = job['shelf_id'] in commissioned
    rep.check(gt_verify or camera_verify_possible, 'config_verify_stage_can_confirm',
              verify_ground_truth=gt_verify, shelf_camera_commissioned=camera_verify_possible,
              fix=None if (gt_verify or camera_verify_possible) else
              'set "verify_ground_truth": {"enabled": true} in the runtime config: the camera '
              'InventoryVerifier cannot confirm on an uncommissioned bay -> verify times out')
    # return lane: the coupled tow to cart_parking must resolve to an authored lane from the park pose,
    # and the lane's arrival articulation must not exceed the adapter's return goal articulation (the
    # adapter raises 'cart articulation exceeds ...' at cart_parking otherwise).
    lanes = (config.get('navigation') or {}).get('tow_lanes')
    lane = None
    if lanes and 'cart_parking' in poses:
        from twin.navigation.lanes import TowLaneGraph
        graph = TowLaneGraph.from_config(lanes)   # unexpanded: match the authored goal (no grid needed)
        lane = next((ln for ln in (graph.lanes if graph else ())
                     if ln.matches_goal(tuple(poses['cart_parking']))), None)
    if lane is None:
        rep.add('WARN', 'config_return_lane_resolves', note='no authored tow lane ends at cart_parking: the '
                'coupled return tow free-plans (check_tow_lanes covers authored lanes only)')
    else:
        start_err = math.dist(lane.start[:2], poses['shelf_approach'][:2])
        limit = (config.get('goal_articulation') or {}).get('return')
        rep.check(start_err <= 0.25, 'config_return_lane_resolves', lane=lane.name, start=list(lane.start),
                  shelf_approach=poses['shelf_approach'], start_err_m=round(start_err, 3))
        rep.check(limit is None or lane.goal_articulation <= limit,
                  'config_return_lane_articulation_within_adapter_limit',
                  lane_goal_articulation=lane.goal_articulation, adapter_return_limit=limit)


def check_tray_on_bay(config, scenario, rep):
    """The ground-truth verify tolerances vs the bay top and the placed tray (unit tests of the geometry)."""
    block = config_block(config) or {'enabled': True}
    v = gt_verifier_from_config(dict(config, verify_ground_truth=block), scenario, lambda: None)
    bx, by, top = bay_target_world(scenario, config)
    h = scenario['handling_unit']['dimensions_m'][2]
    rest = top + h / 2.0
    hx, hy = v.bay_half
    cases = {
        'centred': ((bx, by, rest, 0.0), True, False),
        'offset_5cm_full_support': ((bx + 0.05, by + 0.05, rest, 0.0), True, False),
        'com_inside_margin': ((bx, by + hy - v.xy_margin - 0.005, rest, 0.0), True, False),
        'com_on_edge_band': ((bx, by + hy - 0.005, rest, 0.0), False, False),
        'com_past_edge': ((bx, by + hy + 0.01, rest, 0.0), False, True),
        'held_5cm_above': ((bx, by, rest + 0.05, 0.0), False, False),
        'on_floor': ((bx, by, h / 2.0, 0.0), False, True),
        'tipped_25deg': ((bx, by, rest, math.radians(25)), False, True),
    }
    for name, (pose, on_bay, far) in cases.items():
        got = v.check(pose)
        rep.check(got['on_bay'] == on_bay and got['far'] == far, f'tray_on_bay_{name}', pose=pose,
                  on_bay=got['on_bay'], far=got['far'], offset_bay_xy=got['offset_bay_xy'],
                  bottom_gap_m=got['bottom_gap_m'])
    # The transfer's OWN fm461 place check (controller) uses a radial xy tolerance; flag when it is looser
    # than the bay half-depth (it would accept a tray whose centre is at/over the bay edge).
    import os
    xy_tol = float(os.environ.get('RESTOCK_PLACE_VERIFY_XY_TOL', 0.25))
    verdict = 'PASS' if xy_tol < min(hx, hy) else 'WARN'
    rep.add(verdict, 'tray_on_bay_transfer_controller_radial_tol', place_verify_xy_tol=xy_tol,
            bay_half_xy=[hx, hy], note=('the mission verify (vr1) still requires the tray centre inside '
                                        'the bay top by xy_margin, so an over-the-edge placement fails '
                                        'there, named' if verdict == 'WARN' else 'within the bay'))


def _types(summary):
    return [(e.get('type'), (e.get('data') or {}).get('stage'), (e.get('data') or {}).get('label'))
            for e in summary.get('stage_timeline', [])]


def check_tail(config, scenario, rep, workdir):
    if not have_py_trees():
        rep.add('SKIP', 'tail_dryruns', note='py_trees not importable: add the BT pythonpath (RESTOCK_BT_PYTHONPATH)')
        return
    poses = config['mission_poses']
    modes = mission_modes(config)

    # 1) nominal tail
    mission, summary, rig = run_tail(config, scenario, Path(workdir) / 'nominal')
    states = [e['state'] for e in summary['events']]
    rep.check(mission.state == 'complete', 'tail_nominal_mission_complete', state=mission.state,
              reason=mission.reason)
    by_state = {e['state']: e['evidence'] for e in summary['events']}
    tr = by_state.get('transfer_succeeded', {})
    rep.check(tr.get('released') is True and tr.get('location') == scenario['job']['shelf_id'],
              'tail_transfer_succeeded_on_ground_truth', source=tr.get('source'),
              verify_mode=tr.get('verify_mode'), observation_id=tr.get('observation_id'))
    vf = by_state.get('verify_succeeded', {})
    rep.check(vf.get('transfer_confirmed') is True and 'inventory_verified' in states
              and rig.db.count(scenario['job']['shelf_id'], scenario['job']['sku']) == scenario['job']['quantity']
              and rig.db.count(scenario['job']['source_bin'], scenario['job']['sku']) == 0,
              'tail_verify_commits_stock_once', source=vf.get('source'), verify_mode=vf.get('verify_mode'),
              shelf_count=rig.db.count(scenario['job']['shelf_id'], scenario['job']['sku']))
    goals = [(g['goal'], g['coupled']) for g in rig.nav.goals]
    want = [(list(map(float, poses['cart_parking'])), True)]
    if 'cart_parking_backaway' in poses:
        want.append((list(map(float, poses['cart_parking_backaway'])), False))
    want.append((list(map(float, poses['home'])), False))
    rep.check(goals == want, 'tail_return_legs_tow_then_uncoupled_home', nav_goals=goals)
    rep.check(rig.mobile.calls == ['hold_cart_pin', 'uncouple'], 'tail_uncouple_at_cart_parking_after_transfer',
              mobile_calls=rig.mobile.calls,
              where=f"uncouple at cart_parking {poses['cart_parking']} in RETURN, after transfer+verify at "
                    f"shelf_approach {poses['shelf_approach']}")
    rec = by_state.get('return_succeeded', {})
    rep.check(rec.get('home') is True and states[-1] == 'complete', 'tail_return_home_then_complete',
              final_robot_pose=[round(v, 3) for v in rig.world.robot], last_states=states[-4:])
    mj = json.loads((Path(workdir) / 'nominal' / 'mission.json').read_text())
    missing = {k: v for k, v in modes.items() if mj.get(k) != v}
    rep.check(mj.get('state') == 'complete' and not missing, 'tail_mission_json_success_fields',
              state=mj.get('state'), modes={k: mj.get(k) for k in modes}, missing=missing)
    kinds = _types(summary)
    begun = [s for t, s, _ in kinds if t == 'stage_begin']
    done = [s for t, s, _ in kinds if t == 'stage_succeeded']
    steps = [(s, lbl) for t, s, lbl in kinds if t == 'stage_step']
    shortcut = [s for t, s, _ in kinds if t == 'stage_shortcut']
    ok_events = (begun == list(TAIL_STAGES) and done == list(TAIL_STAGES)
                 and ('mission_complete', None, None) in kinds and shortcut == list(mission.shortcut['skipped'])
                 and all(any(st == s for st, _ in steps) for s in TAIL_STAGES))
    bt_lines = (Path(workdir) / 'nominal' / 'bt_events.jsonl').read_text().splitlines()
    rep.check(ok_events and len(bt_lines) >= len(kinds), 'tail_stage_events_for_walkthrough',
              begun=begun, succeeded=done, shortcut=shortcut, steps=steps)
    if 'verify_mode' in modes:
        rep.check(any('[ground truth]' in (lbl or '') for s, lbl in steps if s == 'verify'),
                  'tail_ground_truth_named_in_stage_label', verify_steps=[l for s, l in steps if s == 'verify'])

    # 2) the cart pin / hitch release perturbs the train for a few ticks: wait, never fail
    mission2, summary2, rig2 = run_tail(config, scenario, Path(workdir) / 'unsettled', unsettle_after_pin=45)
    rep.check(mission2.state == 'complete' and rig2.mobile.calls[-1] == 'uncouple',
              'tail_unsettled_uncouple_waits_then_releases', state=mission2.state, reason=mission2.reason,
              mobile_calls=rig2.mobile.calls)

    # 3) a failed home leg is retried by the tree from the LIVE (uncoupled) state, not by re-towing
    mission3, summary3, rig3 = run_tail(config, scenario, Path(workdir) / 'retry',
                                        fail_goal_once=poses['home'])
    goals3 = [(g['goal'], g['coupled']) for g in rig3.nav.goals]
    rep.check(mission3.state == 'complete' and goals3[-1] == (list(map(float, poses['home'])), False)
              and sum(1 for _, c in goals3 if c) == 1, 'tail_return_retry_resumes_empty_drive_home',
              state=mission3.state, reason=mission3.reason, nav_goals=goals3)

    # 4) a tray NOT on the bay never yields a false 'complete': released onto the floor, the transfer's own
    # fm461 place check refuses it (named); sliding off AFTER the transfer, the mission verify refuses it.
    mission4, _s4, _r4 = run_tail(config, scenario, Path(workdir) / 'floor',
                                  unit_offset=(0.0, 0.0, -scenario['transfer_bay']['top_z']))
    rep.check(mission4.state == 'failed' and 'not resting on deck' in mission4.reason,
              'tail_tray_on_floor_fails_transfer_named', state=mission4.state, reason=mission4.reason[:160])
    if config_block(config) is not None:
        mission5, _s5, _r5 = run_tail(config, scenario, Path(workdir) / 'slid_off', unit_falls_at_verify=True)
        rep.check(mission5.state == 'failed' and 'not resting on the bay' in mission5.reason,
                  'tail_tray_slid_off_fails_verify_named', state=mission5.state, reason=mission5.reason[:160])


def check_default_off(config, scenario, rep, workdir):
    """Without the vr1 switches the tail behaves as before: the ground-truth transfer result is refused."""
    if not have_py_trees():
        return
    base = {k: v for k, v in config.items() if k not in ('verify_ground_truth',)}
    base['transfer_ground_truth'] = dict(config.get('transfer_ground_truth') or {}, enabled=False)
    mission, _s, rig = run_tail(base, scenario, Path(workdir) / 'default_off')
    rep.check(mission.state == 'failed' and 'transfer evidence mismatch' in mission.reason
              and not rig.adapter.ground_truth_place_actions, 'tail_default_off_keeps_sensor_only_gate',
              state=mission.state, reason=mission.reason[:120])


def run_checks(config, scenario, *, environment=None, camera_config=None, workdir=None,
               return_tracking=False, stream=sys.stdout):
    rep = Report(stream)
    check_config(config, scenario, rep)
    check_tray_on_bay(config, scenario, rep)
    with tempfile.TemporaryDirectory(prefix='mission-tail-') as tmp:
        work = Path(workdir or tmp)
        check_tail(config, scenario, rep, work)
        check_default_off(config, scenario, rep, work)
    if environment is not None:
        from twin.restock.tail_nav import check_home_leg
        check_home_leg(config, scenario, environment, rep, camera_config=camera_config)
        if return_tracking:
            # Drive the REAL coupled follower for the return instead of the kinematic stub check_tail
            # uses, so the cart-yaw tracking law (the warm3 stall) is actually exercised on the grid.
            from twin.restock.tail_nav import check_return_tracking
            check_return_tracking(config, scenario, environment, rep, camera_config=camera_config)
    return rep


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--runtime-config', type=Path, required=True)
    ap.add_argument('--scenario', type=Path, required=True)
    ap.add_argument('--environment', type=Path, help='static environment manifest: adds the home-leg nav check')
    ap.add_argument('--camera-config', type=Path, help='camera config whose map solids join the home-leg map')
    ap.add_argument('--workdir', type=Path, help='keep the dry-run mission.json / bt_events.jsonl here')
    ap.add_argument('--return-tracking', action='store_true',
                    help='also drive the REAL coupled return follower (cart-yaw tracking from the recorded '
                         'end-of-verify pose) instead of the kinematic stub; needs --environment')
    args = ap.parse_args(argv)
    config = json.loads(args.runtime_config.read_text())
    scenario = json.loads(args.scenario.read_text())
    env = json.loads(args.environment.read_text()) if args.environment else None
    cam = json.loads(args.camera_config.read_text()) if args.camera_config else None
    if args.return_tracking and env is None:
        ap.error('--return-tracking requires --environment (the real nav grid)')
    rep = run_checks(copy.deepcopy(config), scenario, environment=env, camera_config=cam,
                     workdir=args.workdir, return_tracking=args.return_tracking)
    print('MISSION_TAIL_DRYRUN ' + ('PASS' if not rep.failed else f'FAIL ({len(rep.failed)} failed)'), flush=True)
    return 0 if not rep.failed else 1


if __name__ == '__main__':
    sys.exit(main())
