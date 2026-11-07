"""Host-Jazzy entry point for the existing ROS transfer/motion contract.

Run with host Python, never Isaac's interpreter. Starts only task-domain clients
and controller; GPU planner, mapper and trajectory servers are launched separately
by integration. No hardware launch or simulator boot occurs here.
"""
import argparse
from dataclasses import asdict
import json
import os
import sys
import time

from .ros_adapter import (RosMotionPort, RosTransferBridge, require_sim_domain,
                          sensors_from_json,planner_velocity_scale)
from . import carton_level

# Sim-clock pre-weld grasp-press budget (SIM seconds), now EFFECTIVE BY DEFAULT (firstmate
# sim-robust-weld-attach-v1). A press that never seals a weld keeps the host alive (trajectory_host
# still publishes /joint_states, so the pace-aware arm-command heartbeat does NOT trip -- it fires only
# on host SILENCE), so without this bound a non-forming grasp loops until the COARSE 120 s phase_timeout,
# burning the GPU wall cap with no named cause. This bounds the WHOLE unconfirmed press on the SIM clock
# (render-pacing-safe) and fails fast with a NAMED reason. The default is generous: the slowest observed
# LEGITIMATE grasp press (ground-truth collect, run collect1) was ~13.4 sim s, so 45 s is >3x that yet a
# third of the 120 s phase_timeout. Env RESTOCK_GRASP_PRESS_TIMEOUT_S overrides the default; the config
# key 'grasp_press_timeout_s' overrides both (set it to 0 to disable and reproduce a pre-fix run). The
# controller re-clamps to <= phase_timeout.
_DEFAULT_GRASP_PRESS_TIMEOUT_S = 45.0


def _default_grasp_press_timeout_s():
    try:
        v = float(os.environ.get('RESTOCK_GRASP_PRESS_TIMEOUT_S', _DEFAULT_GRASP_PRESS_TIMEOUT_S))
    except (TypeError, ValueError):
        return _DEFAULT_GRASP_PRESS_TIMEOUT_S
    return v if (v == v and v >= 0) else _DEFAULT_GRASP_PRESS_TIMEOUT_S


def contract_smoke():
    """CPU-only real ROS message codec check; no graph, action or inference."""
    from rclpy.serialization import serialize_message, deserialize_message
    from std_msgs.msg import String, Bool
    from .types import Result
    packet = json.load(sys.stdin)
    wire = String(data=json.dumps(packet,allow_nan=False))
    decoded = deserialize_message(serialize_message(wire),String)
    sensors = sensors_from_json(decoded.data)
    if not all(o.source == 'test_fixture' for o in sensors.observations):
        raise ValueError('smoke input must explicitly be a fixture')
    result = Result('running','contract_checked',
                    controls=({'kind':'gripper','enabled':False},),
                    evidence=tuple(asdict(o) for o in sensors.observations))
    response = String(data=json.dumps(asdict(result),allow_nan=False))
    assert deserialize_message(serialize_message(Bool(data=False)),Bool).data is False
    print(json.dumps({'host_python':sys.version.split()[0],
                      'result':json.loads(deserialize_message(serialize_message(response),String).data)}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config')
    parser.add_argument('--urdf')
    parser.add_argument('--shelfbot-root')
    parser.add_argument('--mobile-manifest')
    parser.add_argument('--contract-smoke',action='store_true')
    args = parser.parse_args()
    if args.contract_smoke:
        contract_smoke()
        return
    if not all((args.config,args.urdf,args.shelfbot_root,args.mobile_manifest)):
        parser.error('--config, --urdf, --shelfbot-root and --mobile-manifest are required')
    require_sim_domain()
    import rclpy
    from rclpy.parameter import Parameter
    from std_msgs.msg import Bool
    from .cad_tool import inspect_tool
    from .controller import TransferController
    from .types import ProductProfile
    from .wall_policy import WallCartonPolicy
    from .ros_payload import RosPayloadSync
    with open(args.config) as stream:
        config = json.load(stream)
    tool = inspect_tool(args.urdf)
    with open(args.mobile_manifest) as stream:
        mobile=json.load(stream)
    transport_joints=tuple(mobile['transport_positions'][name] for name in tool.arm_joints)
    profiles = {p['sku']:ProductProfile(**p) for p in config['profiles']}
    for profile in profiles.values():
        tool.validate_profile(profile)
    # SIM execution profile: a single configurable arm joint-speed gate (rad/s). Default 0.15 keeps
    # the conservative real-rig envelope; the sim mission raises it (0.3) so the render-paced grasp
    # completes within the wall/baseline budget. It bounds both the cuMotion plan speed (command_limit)
    # and the URDF-scale ceiling (scale_cap) so the planner actually commands up to it.
    arm_max_speed = float(config.get('arm_max_speed', .15))
    rclpy.init()
    node = rclpy.create_node('restock_manipulation',namespace='/restock_sim',
                            parameter_overrides=[Parameter('use_sim_time',value=True)])
    stop = node.create_publisher(Bool,'/restock_sim/manipulation/stop',10)
    # Payload-ack WALL timeout: real-rig 1 s; sim profile widens it (render pacing stalls the host
    # between spins at grasp time) via config, >= 3x the measured worst-case service round-trip.
    payload = RosPayloadSync(ack_wall_s=float(config.get('payload_ack_wall_s', 1.0)))
    # fm357: subscribe the measured arm /joint_states so the motion port can SETTLE the arm + RE-SEED
    # each ExecuteTrajectory's first waypoint to the LIVE state. The render-paced plan->validate gap
    # (~hundreds of sim seconds) lets a gravity-loaded joint drift off the planned start, so MoveIt's
    # start-state check aborts the move (run-280 pan, run-282 elbow). The provider returns the latest
    # arm positions/velocities by name + a sim-clock freshness flag; a stale/absent feed => the port
    # falls back to the original direct dispatch (byte-identical).
    from sensor_msgs.msg import JointState as _JointState
    _arm_set = set(tool.arm_joints)
    _live = {'pos': {}, 'vel': {}, 'stamp': None}
    def _on_arm_joint_states(msg):
        names = list(msg.name)
        pos = {n: msg.position[i] for i, n in enumerate(names)
               if n in _arm_set and i < len(msg.position)}
        if len(pos) != len(_arm_set):
            return
        vel = {n: msg.velocity[i] for i, n in enumerate(names)
               if n in _arm_set and i < len(msg.velocity)}
        _live['pos'], _live['vel'] = pos, vel
        _live['stamp'] = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
    node.create_subscription(_JointState, '/restock_sim/joint_states', _on_arm_joint_states, 10)
    def live_arm_state():
        if _live['stamp'] is None:
            return None
        now = node.get_clock().now().nanoseconds * 1e-9
        fresh = (now - _live['stamp']) <= 10.0   # generous sim-clock bound (render-paced gap ~1.7-3.5 s)
        return (dict(_live['pos']), dict(_live['vel']), bool(fresh))
    motion = RosMotionPort(node,joint_limits=tool.arm_limits,payload_sync=payload,
                           emergency_stop=lambda:stop.publish(Bool(data=True)),
                           velocity_scale=planner_velocity_scale(
                               args.urdf,command_limit=arm_max_speed,scale_cap=arm_max_speed),
                           max_velocity_scale=min(arm_max_speed,1.),
                           pose_goal=bool(config.get('pose_goal', False)),
                           stow_config=transport_joints,
                           plan_wall_cap_s=float(config.get('plan_wall_cap_s', 22.0)),
                           live_arm_state=live_arm_state)
    # Acquisition-to-use freshness (require_fresh on valid_until_steady) must be measured on the
    # SIM/node clock, not wall: the held sensing deadline is a SIM-time budget (world.current_time +
    # window remaining, sensing_sources.steady_clock) and under render pacing wall races far ahead of
    # sim, so a wall steady_clock trips 'acquisition-to-use freshness expired' instantly (run-143, once
    # PR 95 threaded the deadline onto the collect packet). use_sim_time is on this node, so its clock
    # is the shared sim time both the sensing deadline and this check live on.
    # The WallCartonPolicy applies ONLY to side-grip cartons (it raises for any other mode). A TOP
    # grip uses the controller's default vertical (TCP-down) poses instead (no wall standoff) -- the
    # reachable, cuMotion-feasible place the side-grip wall standoff could not achieve onto the cart
    # (run-222). Mirror isaac_runtime: policy only when every profile is side mode.
    _modes = {p.grasp_mode for p in profiles.values()}
    _policy = WallCartonPolicy(args.shelfbot_root, tool, walls=config['walls']) if _modes == {'side'} else None
    # SIM-ONLY cart place-pose policy (firstmate 282, key place-on-cart-strategy-2 opt 3): the wall
    # policy's outward place standoff is cuMotion-IK-infeasible onto the cart; wrap it so the PLACE
    # pre/post standoff offsets TOWARD the robot (CPU-verified reachable) while the proven side-grip
    # PICK and the real-rig wall policy are untouched. Gated on the config flag; off by default.
    if _policy is not None and config.get('sim_cart_place_policy'):
        from .wall_policy import SimCartPlacePolicy
        _policy = SimCartPlacePolicy(_policy)
    # Staged cart-swing (firstmate 292): route lift->preplace through body-clear joint-space via points
    # (tuck/pan) so the welded carton does not sweep the un-sphered chassis/arm-mount (run-236 joint0
    # stall). Needs a place policy + EITHER the sim cart-place flag (arm-only cage/mirror place) OR
    # fm390 reposition_before_place (the carry fold tucks the carton compact so the BASE can drive it to
    # place_approach; lift->tuck->TRANSIT, pan skipped). cart_pan_joint0 is CPU-verified body-clear for
    # the fixed cart dock (staged_swing_verify.py).
    staged = (bool(config.get('staged_cart_swing', False)) and _policy is not None
              and (bool(config.get('sim_cart_place_policy'))
                   or bool(config.get('reposition_before_place'))))
    # sim-preplace-leg2-level-v1 (part 2): the phase timeout. The two-leg wrist-first preplace legitimately
    # needs ~150 s at the caps -- over the legacy hard-coded 120 s (coldwin56 timed out at 58 % of leg 2).
    # When wrist-first is on, derive the budget from the planned leg durations at the active caps + a
    # configurable margin (HOST key ``preplace_budget_margin_s``), floored at the historical 120 s so a short
    # plan never shrinks it; an explicit HOST ``phase_timeout_s`` overrides. Non-wrist-first keeps 120 s
    # exactly (byte-identical). The computed budget is logged (PREPLACE_PHASE_BUDGET) and config_preflight
    # reports it + refuses a plan that exceeds it.
    phase_timeout = float(config.get('phase_timeout_s', 120.0))
    if bool(config.get('preplace_wrist_first', False)) and 'phase_timeout_s' not in config:
        _lc = carton_level.resolve_level_config(config.get('carton_level'))
        _carry = config.get('carry_lift_shape'); _standoff = config.get('collect_place_preplace_shape')
        if _lc is not None and _carry is not None and _standoff is not None:
            budget, plan_s, l1, l2 = carton_level.preplace_phase_budget(
                _carry, _standoff, speed=_lc.preplace_approach_speed, accel=_lc.preplace_approach_accel,
                max_segment_rad=_lc.preplace_max_segment_rad,
                margin_s=float(config.get('preplace_budget_margin_s', 90.0)))
            phase_timeout = budget
            print('PREPLACE_PHASE_BUDGET wrist_first plan=%.1f s (leg1=%.1f leg2=%.1f) + margin=%.1f '
                  '-> phase_timeout=%.1f s (was hard-coded 120 s)'
                  % (plan_s, l1, l2, float(config.get('preplace_budget_margin_s', 90.0)), phase_timeout),
                  flush=True)
    # sim-rgbd-arm-collision-aware-v1: perceived-clearance collision gate (opt-in HOST key
    # perception_collision_gate). Wrap the motion port so EVERY arm leg (pick approach, lift,
    # tuck, pre-place, place descent, withdraw, every transfer leg -- they all flow through
    # begin/begin_joint_goal/begin_joint_path) is queried against the live nvblox ESDF (the SAME
    # get_esdf_and_gradient map cuMotion plans against) for the swept gripper + welded-carton
    # volume BEFORE dispatch; a leg whose perceived clearance is below the margin is refused/held
    # by name (PERCEIVED_CLEARANCE / PERCEIVED_CLEARANCE_<ACTION>). Absent/disabled => the raw
    # RosMotionPort unchanged (byte-identical; the complete-cycle runs are untouched). The wrapper
    # reads the live phase/held/carton from the controller built just below via a late-bound box.
    from .gated_motion import build_gated_motion_port
    _gate_ctrl = {}
    motion = build_gated_motion_port(
        motion, config=config, tool=tool, urdf_path=args.urdf, node=node,
        get_phase=lambda: getattr(_gate_ctrl.get('c'), 'phase', 'idle'),
        get_held=lambda: bool(getattr(_gate_ctrl.get('c'), 'held', False)),
        get_carton_dims=lambda: (tuple(_gate_ctrl['c'].profile.dimensions)
                                 if _gate_ctrl.get('c') is not None
                                 and getattr(_gate_ctrl['c'], 'profile', None) is not None
                                 else None),
        steady_clock=lambda: node.get_clock().now().nanoseconds * 1e-9)
    controller = TransferController(motion,profiles,frame=tool.base_frame,
        phase_timeout=phase_timeout,pose_policy=_policy,
        transport_joints=transport_joints,
        contact_straight_line=bool(config.get('contact_straight_line', False)),
        contact_overpress_step_m=float(config.get('contact_overpress_step_m', 0.0)),
        contact_overpress_cap_m=float(config.get('contact_overpress_cap_m', 0.0)),
        staged_cart_swing=staged, cart_pan_joint0=float(config.get('cart_pan_joint0', 0.0)),
        cart_tuck_shape=config.get('cart_tuck_shape'),
        carry_wrist1_cap=config.get('carry_wrist1_cap'),
        carry_lift_shape=config.get('carry_lift_shape'),
        carry_wrist2_min=config.get('carry_wrist2_min'),
        # Cart-cage open-face place (firstmate 350): route the place THROUGH the cart's open +x face --
        # cuMotion plans PREPLACE around the published cage walls (config['world_obstacles']) to a mouth
        # standoff, then a straight-line insert drives onto the deck. Absent => the existing pose-IK
        # preplace/place (byte-identical). cage_axis/standoff/gap in ur_base_link.
        cart_cage_place=bool(config.get('cart_cage_place', False)),
        cage_axis=tuple(config.get('cage_axis', (1.0, 0.0, 0.0))),
        cage_mouth_standoff=float(config.get('cage_mouth_standoff', 0.55)),
        cage_insert_gap=float(config.get('cage_insert_gap', 0.03)),
        cage_preplace_shape=config.get('cage_preplace_shape'),
        # sim-robust-cart-place-goal-v1: the NON-cage COLLECT place-on-cart counterparts. A CPU-validated
        # UPRIGHT clear-region preplace joint shape (driven like cage_preplace_shape so the deck preplace
        # avoids the un-holdable deep fold / self-colliding branch), and the wrist-guard extended to the
        # collect place/retreat pose-IK (clear-branch re-seed). Both absent/False => byte-identical.
        collect_place_preplace_shape=config.get('collect_place_preplace_shape'),
        collect_place_wrist_guard=bool(config.get('collect_place_wrist_guard', False)),
        # sim-robust-collect-place-straightline-v1 (collect4): route the NON-cage collect deck PLACE
        # descent + empty RETREAT collision-off straight-line (begin(straight_line=True), the cart-cage /
        # live_pose place mechanism). The held carton is a cuMotion attached_object (34-sphere cuboid), and
        # the set-down fold brings it within the attached-object self-collision margin, so a collision-aware
        # plan to the place config is rejected ('self collision detected') even though the bare arm + wrist
        # guard are clear. The preplace standoff stays collision-aware. Absent/False => byte-identical.
        collect_place_straight_line=bool(config.get('collect_place_straight_line', False)),
        # sim-collect-retreat-abort-v2: lift the empty gripper straight UP before the collect deck-place
        # retreat folds to the clear preplace shape, so the enabled wrist links separate from the re-enabled
        # carton / shelf / cage first (the direct fold's horizontal first increment dragged them into a
        # contact that pinned the drive -- physcon1 / rtfcollect2). ON by default for this retreat (owner-
        # authorised; scoped to the non-cage collect deck place). dz = the straight-up rise. Set
        # collect_retreat_liftoff False to restore the single-fold retreat.
        collect_retreat_liftoff=bool(config.get('collect_retreat_liftoff', True)),
        collect_retreat_liftoff_dz=float(config.get('collect_retreat_liftoff_dz', 0.15)),
        # sim-robust-transfer-preplace-reach-v1 (coldwin6): steer the live_pose TRANSFER PREPLACE's
        # wrist-guarded KDL IK to the holdable upright clear branch and reject a clear-but-deep (un-reachable)
        # shoulder fold (coldwin6 accepted ur_shoulder_lift -4.003 and froze at -3.65). bool / mapping
        # {max_droop, upright_seed}; absent/False => the plain wrist-guard (byte-identical). See
        # transfer_carry.resolve_reach_guard.
        transfer_preplace_reach_guard=config.get('transfer_preplace_reach_guard'),
        # sim-robust-transfer-settle-oscillation-v1 (coldwin12): extend the clear-wrist-region guard to the
        # live_pose TRANSFER PICK (approach + contact), which the carry wrist_guard does not cover. The pick
        # intermittently IK'd the ur_wrist_2-vs-ur_forearm self-colliding family and the contact descent
        # limit-cycled 0.169 rad / ~1.6 rad/s for the whole 30 s settle window (PR239 envelope failed it);
        # coldwin11 took the clear family and settled instantly. On => the pick approach pose goal carries the
        # clear-region joint constraints and the contact straight-line IK is wrist-guarded. Absent/False =>
        # byte-identical. See controller._pick_wrist_guard.
        transfer_pick_wrist_guard=bool(config.get('transfer_pick_wrist_guard', False)),
        # fm390: drive the base in front of the rollwagen mouth between the carry fold and the place
        # (lift->tuck->TRANSIT->[base nav]->preplace), instead of the arm-only cage insert. Absent =>
        # byte-identical. Mutually exclusive in practice with cart_cage_place (base nav replaces it).
        reposition_before_place=bool(config.get('reposition_before_place', False)),
        # sim-service-leg-carry-v1: the shelf-staging TRANSFER service leg picks the carton off the parked
        # cart at the staging pose and CARRIES it to the endcap before placing. That needs the TRANSIT
        # carry-hold for a live_pose transfer (lift->tuck->TRANSIT->[base carry-drive]->preplace); the runtime
        # drives the carry (it is wired only in shelf_staging mode). Set this HOST key true together with the
        # mission runtime `shelf_staging: true`. Absent/False => a live_pose transfer stays tuck->preplace
        # (byte-identical). See controller._repositions_before_place + restock/runtime service-leg legs.
        transfer_service_carry=bool(config.get('transfer_service_carry', False)),
        # fm410 (run-316): physics grasp confirm -- the lift-probe carton-tracking threshold (m). The grasp
        # advances to carry only if the measured carton rose by >= this from the grasp-confirm baseline, so
        # the confirm reads physics, never the (arm-occluded) source camera. Absent => inert (byte-identical).
        grasp_track_m=config.get('grasp_track_m'),
        # Sim-clock pre-weld press budget (SIM seconds): fail the collect fast with a NAMED reason if the
        # grasp press never seals, before the opaque 90 s WALL arm-command heartbeat watchdog (runs
        # 321/327). Now EFFECTIVE BY DEFAULT (sim-robust-weld-attach-v1): absent config => the
        # _default_grasp_press_timeout_s backstop (45 s, env RESTOCK_GRASP_PRESS_TIMEOUT_S); the config
        # key still overrides, and setting it to 0 disables (byte-identical pre-fix). Clamped to <=
        # phase_timeout in the controller.
        grasp_press_timeout_s=float(config.get('grasp_press_timeout_s', _default_grasp_press_timeout_s())),
        # sim-robust-cart-place-descent-v1: SIM-clock release-detach budget (SIM seconds). After the place
        # descent, fail fast with a NAMED reason if the SIM_ATTACH weld / suction does not let go within
        # this long of the gripper-open command, instead of carrying the carton back up on retreat/stow or
        # stalling to the coarse phase_timeout. Absent/0 => off (rely on phase_timeout, byte-identical).
        release_detach_timeout_s=float(config.get('release_detach_timeout_s', 0.0)),
        # fm449 (run-332 cold collect): gate the observe->reach on a clean synchronized source frame + decoded
        # label at park (the Isaac-side verdict rides Sensors.collect_source_clean). Absent/False => the gate
        # never arms => byte-identical. No injected provider on the host path; the gate reads the packet field.
        require_collect_source_evidence=bool(config.get('require_collect_source_evidence', False)),
        # fm461 (demo fallback): accept the authored GROUND-TRUTH bay transfer target + ground-truth verify
        # when the shelf station is not camera-commissioned. Absent/False => byte-identical.
        transfer_ground_truth=bool(config.get('transfer_ground_truth', False)),
        # fm462 (demo, opt-in): label the (already controlled-asset) transfer pick ground-truth. Nested
        # under the transfer_ground_truth block; default off. Does NOT bypass the station/masked-depth pump.
        ground_truth_pick=bool(config.get('transfer_ground_truth', {}).get('ground_truth_pick', False))
                          if isinstance(config.get('transfer_ground_truth'), dict) else False,
        # sim-robust-esdf-boot-gate-v1: SIM-clock boot grace (seconds) for the first valid nvblox ESDF
        # incorporation ack; names a never-acknowledged ESDF at boot before the first arm motion, instead
        # of the opaque ~40 s collision_dt invariant trip (meshuris1). Absent/0 => off => byte-identical.
        esdf_boot_ack_grace_s=float(config.get('esdf_boot_ack_grace_s', 0.0)),
        # sim-preplace-per-joint-gains-v1: carried-preplace convergence guard + wrist-first sequencing
        # fallback. The guard names a frozen joint (PHASE_CONVERGENCE) when the preplace goal error stops
        # shrinking for this many SIM seconds, instead of the opaque 120 s phase timeout (coldwin53:
        # shoulder_pan froze 0.56 rad short). The sequence sweeps the wrists first at the tucked pose. Both
        # opt-in: stall 0 / flag False => inert => byte-identical. (The PRIMARY fix is the RUNTIME per-joint
        # phase_drive.carried_joint_profile; these are the HOST-side guard + fallback.)
        preplace_convergence_stall_s=float(config.get('preplace_convergence_stall_s', 0.0)),
        preplace_wrist_first=bool(config.get('preplace_wrist_first', False)),
        # sim-preplace-sweep-cage-clearance-v1: PAN-FIRST reorder (opt-in, needs preplace_wrist_first).
        # coldwin70: the wrist-first order EXTENDS the arm before the pan sweep, so leg-2 rotates the
        # extended gripper housing on an arc that rakes /Rollwagen/chassis/post_2 + the upper side bar at
        # shoulder_pan ~ 0.20 (the -48 mm strike that froze the run at 1632.7 N*m). Pan-first sweeps the pan
        # to the standoff while the arm stays TUCKED at the carry shape (CPU: cage clearance +252 mm vs
        # -48 mm), then extends. Absent/false => wrist-first => byte-identical.
        preplace_pan_first=bool(config.get('preplace_pan_first', False)),
        # sim-preplace-wrist2-hold-v1: opt-in DRIFT RE-CAPTURE on the convergence guard. When a driven
        # preplace joint drifts off its closest approach by more than preplace_drift_recapture_rad, the
        # guard re-issues the goal (re-driving the back-driven wrist) up to preplace_drift_recapture_max
        # times per joint and logs PREPLACE_DRIFT_RECAPTURE once, instead of running the stall to a fault
        # (cw66: the leg-2 pan sweep back-drove the arrived-but-soft wrist_2). Both <= 0 => inert =>
        # byte-identical. Needs preplace_convergence_stall_s > 0 (the guard must be armed).
        preplace_drift_recapture_rad=float(config.get('preplace_drift_recapture_rad', 0.0)),
        preplace_drift_recapture_max=int(config.get('preplace_drift_recapture_max', 0)),
        # sim-preplace-leg2-level-v1: the held chassis attitude during the collect place. The station
        # parking brake pins the base off level (the RUNTIME chassis_station_pitch_hold_pitch_deg /
        # _roll_deg; coldwin56 pitch -1.17), so ur_base_link +Z is off world +Z and the shoulder_pan YAW
        # axis is off vertical -- the carried carton then sweeps a cone as pan descends and the WORLD tilt
        # is not pan-invariant (leg 2 rose 1.64 -> 2.02). These HOST keys MIRROR the runtime values so the
        # carton-level FK levels / measures the preplace in the WORLD frame. Absent/false => None => the
        # base frame (byte-identical). They must match the runtime keys (named in the handoff fragment).
        station_pitch_hold=bool(config.get('station_pitch_hold', False)),
        station_hold_roll_deg=float(config.get('station_hold_roll_deg', 0.0)),
        station_hold_pitch_deg=float(config.get('station_hold_pitch_deg', 0.0)),
        # sim-carton-level-carry-v1: keep the welded carton parallel to the ground through every carried
        # move (level the fold goals within the clear wrist caps) and NAME + FAIL a carried carton that
        # leaves level by more than the bound (CARTON_TILT) rather than carry it visibly tilted. A mapping
        # {bound_deg, level_goals, trip, joints} / bool; absent/false => None => byte-identical. The FK
        # URDF is the arm the host already loaded (args.urdf).
        carton_level=carton_level.resolve_level_config(config.get('carton_level')),
        # sim-pick-contact-seal-fallback-v1 (coldwin63 pick never sealed): opt-in pick-contact guards. The
        # enable-reissue defeats the last-value /manipulation/status relay overwrite that dropped cw63's
        # one-shot close command; the suction-enabled guard fails fast / re-sends instead of pressing the
        # full 20 mm cap with the plugin never enabled; the seal-diag accept treats a dwell-stable flush
        # geometric seal as contact. All absent/false => byte-identical. See the handoff fragment.
        contact_reissue_suction_each_tick=bool(config.get('contact_reissue_suction_each_tick', False)),
        contact_require_suction_enabled=bool(config.get('contact_require_suction_enabled', False)),
        contact_reissue_close_once=bool(config.get('contact_reissue_close_once', False)),
        contact_accept_seal_diag=bool(config.get('contact_accept_seal_diag', False)),
        contact_seal_diag_dwell_ticks=int(config.get('contact_seal_diag_dwell_ticks', 5)),
        contact_seal_diag_max_gap_mm=float(config.get('contact_seal_diag_max_gap_mm', 0.5)),
        contact_seal_diag_max_angle_deg=float(config.get('contact_seal_diag_max_angle_deg', 1.0)),
        # sim-gripper-status-relay-v1: optional extra staleness bound (sim s) on the gripper status the
        # require_suction_enabled guard reads; 0 => presence-only. The relay that FILLS the status is the
        # Isaac-side RESTOCK_CONTACT_STATUS_FEEDBACK env (see docs/examples/gripper-status-relay-fragment.json).
        contact_status_max_age_s=float(config.get('contact_status_max_age_s', 0.0)),
        fk_urdf=args.urdf,
        # sim-arm-reactive-replan-v1 (P1): opt-in human-aware RUN/SLOW/STOP/BACKSTOP authority over the
        # free-space arm legs (ported humangate fed by the vendored scripts/twin/humans evaluator; the
        # e-stop-free motion.cancel preempt + a reversible ArmCommandGate pause). Absent/False => the
        # supervisor is never constructed => byte-identical. The detection provider + pause sink are wired
        # Isaac-side at boot (controller.set_human_detection_provider / set_arm_pause_sink) when enabled.
        reactive_arm_human=bool(config.get('reactive_arm_human', False)),
        reactive_arm_config=(config.get('reactive_arm') if isinstance(config.get('reactive_arm'), dict)
                             else None),
        # sim-arm-reactive-replan-v1 (PR-B): opt-in execution-time corridor-watch re-plan. While a
        # free-space leg EXECUTES, re-query the live ESDF along the REMAINING swept corridor; a NEW
        # obstacle below margin preempts (e-stop-free cancel) + re-plans the remaining leg (bounded;
        # HOLD when it cannot route clear). Consumes the sibling perceived-clearance seam (requires
        # perception_collision_gate on for the ESDF source). Absent/False => inert => byte-identical.
        reactive_arm_replan=bool(config.get('reactive_arm_replan', False)),
        reactive_arm_replan_config=((config.get('reactive_arm') or {}).get('replan')
                                    if isinstance(config.get('reactive_arm'), dict) else None),
        steady_clock=lambda: node.get_clock().now().nanoseconds*1e-9)
    payload.controller = controller
    # Late-bind the live controller into the perceived-clearance gate's phase/held/carton reads.
    _gate_ctrl['c'] = controller
    # sim-place-descent-reach-wire-v1: INJECT the cart cage-clearance providers (PR 366 carried pre-place
    # sweep + PR 376 collect place-DESCENT sweep) onto the controller from the HOST config. Both providers
    # (place_clearance.preplace_cage_clearance_provider / .place_descent_cage_clearance_provider) ship with
    # the controller hooks they feed (_preplace_cage_clearance / _place_descent_cage_gate) but had ZERO call
    # sites, so both hooks stayed None and the in-op gates never ran -- coldwin90 (every PR 376 switch ON)
    # repeated run 89/90 exactly (PLACE_SEAT_GATE_ARM / PLACE_OBSTRUCTED +87 mm / descent never started). The
    # blocks carry the live coupled cart-body -> ur_base_link pose + host XRDF; absent => the hooks stay None
    # => byte-identical. Loud at boot (PREPLACE_CAGE_GATE / PLACE_DESCENT_CAGE_GATE 'armed'); raises on a
    # malformed block so a configured gate can never again be left uncalled.
    from .place_cage_wiring import wire_place_cage_gates
    wire_place_cage_gates(controller, config, args.urdf)
    # sim-arm-reactive-replan-v1 (PR-B): wire the execution-time corridor watch to the sibling's
    # perceived-clearance ESDF source (the wrapped GatedMotionPort). Inert unless BOTH
    # reactive_arm_replan and perception_collision_gate are on (the watch needs the live ESDF).
    if bool(config.get('reactive_arm_replan', False)):
        from . import reactive_arm as _ra
        _esdf_src, _corridor_fn = _ra.build_corridor_fn(motion)
        if _esdf_src is not None:
            controller.set_arm_clearance_source(esdf_source=_esdf_src, corridor_fn=_corridor_fn)
            print('REACTIVE_REPLAN wired to the perceived-clearance ESDF source (execution-time '
                  'corridor watch active on the free-space legs)', flush=True)
        else:
            print('REACTIVE_REPLAN enabled but perception_collision_gate is OFF: the corridor watch '
                  'has no live ESDF source and stays inert -- enable perception_collision_gate',
                  flush=True)
    # collect-gt: the ground-truth collect switch is the shared RESTOCK_COLLECT_GROUND_TRUTH env (inherited from
    # the run env); log the host's view at boot so an Isaac/host activation split is visible immediately.
    print('COLLECT_GROUND_TRUTH host enabled=%s' % controller.collect_ground_truth, flush=True)
    # The first decode-grade frame + support registration under render pacing needs several
    # ~1 s RTX renders (~5-15 s WALL); the default 5 s wall startup grace flaked (run-108 failed
    # 'no initial sensor frame' where run-106/107 cleared it only by luck). Widen it for the sim
    # via config; the runtime.py sim-clock stall guard (PR 86) still catches a job that genuinely
    # never produces a frame, so the generous wall grace cannot hide a real stall.
    bridge = RosTransferBridge(node,controller,
        startup_grace_seconds=float(config.get('startup_grace_seconds', 5.0)),
        # sim-place-descent-watchdog-suspend-v1: suspend the per-frame /sensors SIM watchdog ONLY while the
        # controller is in the loaded place descent (the station/source decode stream legitimately stalls
        # under the full-six-axis base lock + deep render pacing -- coldwin64). The 180 s wall backstop
        # stays armed throughout. Default off => byte-identical.
        suspend_place_descent_watchdog=bool(config.get('place_descent_sensor_watchdog_suspend', False)),
        # sim-retreat-sensor-watchdog-scope-v1: broaden that suspension to EVERY capture-silent phase
        # (controller.in_capture_silent_window -- the post-place retreat/stow at least, where coldwin76/77/78
        # each died on a watchdog counting a legitimately quiet period). The wall backstop stays armed
        # throughout. Default off => byte-identical.
        suspend_capture_silent_watchdog=bool(config.get('capture_silent_sensor_watchdog_suspend', False)))
    # sim-host-status-watchdog-v1: the payload ack waits (RosPayloadSync._wait) block the main host
    # executor, silencing the controller-driven /manipulation/status stream for the whole bounded wait
    # (the grasp/weld attach, the post-move_group-respawn carton reconcile, the release detach). Pulse a
    # liveness heartbeat on the status topic each wait slice so the Isaac status watchdog sees the live
    # host and defers to the wait's own named sim-budget (collectretreat1 false trip).
    payload.heartbeat = bridge.heartbeat
    # sim-preplace-branch-select-bounded-v1: a SEPARATE daemon thread keeps /manipulation/status fresh
    # while the single rclpy executor is busy in a long controller.step()/planning call -- so no planner
    # that blocks the executor can ever starve the Isaac status watchdog again (coldwin58/59: the inline
    # reaction-branch solve ran >5 s at the preplace transition and tripped HOST_STATUS_WATCHDOG). The
    # filler publishes the SAME liveness message RosTransferBridge already emits (status='running',
    # heartbeat=True, no controls -> never clears active, never advances the sensor anchors), topping up
    # only genuine gaps older than the interval while a job is active. HOST key status_heartbeat_interval_s
    # (seconds); 0 disables => byte-identical (no filler thread). The reaction-branch solve is also now
    # cheap + time-boxed off the hot path; this is the belt-and-suspenders structural guarantee.
    _hb_interval = float(config.get('status_heartbeat_interval_s', 0.5))
    status_heartbeat = None
    if _hb_interval > 0:
        from .status_heartbeat import StatusHeartbeat
        status_heartbeat = StatusHeartbeat(
            publish=bridge._publish_liveness, now=time.monotonic,
            is_active=lambda: getattr(controller, 'phase', 'idle') not in ('idle', 'failed', 'succeeded'),
            interval=_hb_interval)
        bridge.status_heartbeat = status_heartbeat
        status_heartbeat.start()
        print('STATUS_HEARTBEAT filler thread started (interval=%.2f s; keeps /manipulation/status fresh '
              'while the executor is busy -- coldwin59)' % _hb_interval, flush=True)
    # Remember the synchronized native-contact packet before controller consumes it.
    original = bridge._sensors
    node.destroy_subscription(bridge.subscriptions[1])
    from std_msgs.msg import String
    def sensors(message):
        try:
            payload.sensors = sensors_from_json(message.data)
            original(message)
        except (ValueError,TypeError,KeyError) as exc:
            bridge._emit(controller._fail('invalid sensor packet: '+str(exc)))
    bridge.subscriptions[1] = node.create_subscription(String,'/restock_sim/sensors',sensors,10)
    # Static base obstacle (firstmate 295): publish the fixed mobile base as a world CollisionObject so
    # cuMotion plans the arm + attached carton clear of the chassis/arm-mount (the base is neither in the
    # arm self-collision model nor the ESDF -- run-236/240/241 carton swept through it). Sim-only; gated
    # on the config flag (real-rig path never publishes it). Done once at startup via the payload node's
    # spinning executor; wait_for_service rides out the host-stack/move_group warmup.
    if bool(config.get('base_collision_object', False)):
        payload.add_base_obstacle()
    # Static environment fixtures the forward eye-in-hand camera never maps into the nvblox ESDF
    # (the cart cage behind the robot, the overhead collider the carried carton swings into) must be
    # published as world CollisionObjects or cuMotion plans the carton straight through them (run-249/
    # 250). Each spec is {id, boxes:[{size,center,rpy?}]} in the ur_base_link frame; CPU-proven clear of
    # the arm's stow + grasp-start configs by twin.restock_host.world_obstacles before any GPU run.
    # sim-rgbd-arm-collision-aware-v1 (evidence-gated shortcut retirement): where the perceived
    # nvblox ESDF demonstrably covers a body the eye-in-hand camera DOES now map (a cart post /
    # shelf / deck the wrist+mast feeds see), the static CollisionObject for it can be withheld and
    # the perceived obstacle relied on instead. This is opt-in per-id and operator-asserted WITH GPU
    # coverage evidence: list the covered ids under world_obstacles_perceived.covered_ids. Absent =>
    # every static obstacle is published exactly as before (byte-identical). The ground-truth TARGET
    # is a separate decision and stays; only the perceived-vs-static OBSTACLE source changes here.
    _perceived_covered = set((config.get('world_obstacles_perceived') or {}).get('covered_ids', []))
    for spec in config.get('world_obstacles', []):
        if spec['id'] in _perceived_covered:
            print('WORLD_OBSTACLE_PERCEIVED_SKIP id=%s (perceived nvblox ESDF covers this body; '
                  'static CollisionObject withheld -- requires perception_collision_gate enabled '
                  'and GPU coverage evidence)' % spec['id'], flush=True)
            continue
        boxes = [(b['size'], b['center'], b.get('rpy', (0.0, 0.0, 0.0))) for b in spec['boxes']]
        payload.add_world_obstacle(spec['id'], boxes)
    # fm351 activation assertion: log the HOST's EFFECTIVE place/cage config as actually loaded, so a
    # console grep proves a batch key activated in the process that consumes it. run-277 lost a full GPU
    # run to a cage key authored into the wrong config file (silently inert); this line + the orchestrator
    # --require preflight (config_preflight) make that failure impossible to repeat unnoticed.
    print('HOST_EFFECTIVE_CONFIG ' + json.dumps({
        'cart_cage_place': controller.cart_cage_place,
        'cage_preplace_shape': controller.cage_preplace_shape,
        'cage_axis': controller.cage_axis, 'cage_mouth_standoff': controller.cage_mouth_standoff,
        'cage_insert_gap': controller.cage_insert_gap,
        'staged_cart_swing': controller.staged_cart_swing, 'cart_pan_joint0': controller.cart_pan_joint0,
        'cart_tuck_shape': controller.cart_tuck_shape,
        # fm402/fm403: log the EFFECTIVE loaded-lift self-collision guards so a console grep proves they
        # activated. carry_lift_shape routes the lift to a fixed clear-region joint goal; carry_wrist1_cap
        # + carry_wrist2_min bound any remaining task-space carry move below/above the wrist_2-vs-forearm
        # stop (None on any = that guard inactive -> the lift can jam, runs 300d/305/311).
        'carry_wrist1_cap': controller.carry_wrist1_cap,
        'carry_lift_shape': controller.carry_lift_shape,
        'carry_wrist2_min': controller.carry_wrist2_min,
        'grasp_track_m': controller.grasp_track_m,   # fm410: lift-probe carton-tracking confirm threshold (m)
        # sim-robust-cart-place-goal-v1: log the EFFECTIVE non-cage collect place goal guards so a console
        # grep proves they activated in the host that consumes them (host-vs-runtime-config-split trap).
        'collect_place_preplace_shape': controller.collect_place_preplace_shape,
        'collect_place_wrist_guard': controller.collect_place_wrist_guard,
        # sim-robust-collect-place-straightline-v1: collect deck place/retreat routed collision-off (False = off)
        'collect_place_straight_line': controller.collect_place_straight_line,
        # sim-collect-retreat-abort-v2: collect deck-place empty-retreat lift-off-then-fold (True = on) + rise
        'collect_retreat_liftoff': controller.collect_retreat_liftoff,
        'collect_retreat_liftoff_dz': controller.collect_retreat_liftoff_dz,
        # sim-robust-transfer-preplace-reach-v1: effective live_pose transfer preplace reach guard (None = off)
        'transfer_preplace_reach_guard': controller.transfer_preplace_reach_guard,
        # sim-robust-transfer-settle-oscillation-v1: effective live_pose transfer PICK wrist guard (False = off)
        'transfer_pick_wrist_guard': controller.transfer_pick_wrist_guard,
        # sim-robust-cart-place-descent-v1: effective SIM-clock release-detach budget (0 = off)
        'release_detach_timeout_s': controller.release_detach_timeout_s,
        'reposition_before_place': controller.reposition_before_place,
        # sim-service-leg-carry-v1: TRANSIT carry-hold for the shelf-staging live_pose transfer service leg
        'transfer_service_carry': controller.transfer_service_carry,
        'transfer_mode': controller.transfer_mode,   # env RESTOCK_TRANSFER_MODE (None = off)
        'transfer_carry': sorted(controller.transfer_carry),   # env RESTOCK_TRANSFER_CARRY ([] = off)
        'sim_cart_place_policy': _policy is not None and bool(config.get('sim_cart_place_policy')),
        'base_collision_object': bool(config.get('base_collision_object', False)),
        'world_obstacles': [s['id'] for s in config.get('world_obstacles', [])],
        # sim-robust-esdf-boot-gate-v1: boot ESDF-ack grace (sim s); 0.0 => gate off (byte-identical).
        'esdf_boot_ack_grace_s': controller.esdf_boot_ack_grace_s,
        # sim-preplace-per-joint-gains-v1: effective carried-preplace convergence guard (0 = off) + wrist-
        # first sequencing fallback (False = off). The PRIMARY per-joint drive override is a RUNTIME key
        # (phase_drive.carried_joint_profile), logged Isaac-side as PHASE_DRIVE_JOINT.
        'preplace_convergence_stall_s': controller.preplace_convergence_stall_s,
        'preplace_wrist_first': controller.preplace_wrist_first,
        # sim-preplace-sweep-cage-clearance-v1: pan-first reorder (True => sweep pan while tucked).
        'preplace_pan_first': controller.preplace_pan_first,
        # sim-preplace-wrist2-hold-v1: effective drift re-capture bound (rad) + per-joint budget (0 = off).
        'preplace_drift_recapture_rad': controller.preplace_drift_recapture_rad,
        'preplace_drift_recapture_max': controller.preplace_drift_recapture_max,
        # sim-place-descent-reach-wire-v1: whether the cart cage-clearance providers were INJECTED (True =
        # the in-op gate is live; False = the hook stayed None => byte-identical). The 'armed' log line above
        # carries the margin / xrdf / cart_pose when True.
        'preplace_cage_gate': controller._preplace_cage_clearance is not None,
        'place_descent_cage_gate': controller._place_descent_cage_gate is not None,
    }), flush=True)
    # sim-collision-dt-render-lead-floor-v1: log the DERIVED collision/feedback freshness band once at
    # host start, so every run dir's host-stack.log records the floor/ceiling in force (the future-lead
    # floor widening -- coldwin30's -0.5333 trip -- is never silent). The floor is derived from the
    # delivery render lead + the step-scaled ESDF incorporation window; the stale ceiling is unchanged.
    print(controller.freshness_band_summary(), flush=True)
    try:
        rclpy.spin(node)
    finally:
        # sim-robust-transfer-start-estop-v1: motion.stop() here publishes the global /manipulation/stop
        # (emergency_stop) on process exit. The Isaac side latches that bare Bool, so a host restart would
        # surface to a later op as a bare 'host emergency stop'. Name the publication so host-stack.log
        # always carries the trigger (PR 231 named only the in-op controller/port fault paths).
        print('HOST_EXIT_ESTOP_PUBLISH reason=%r' % 'motion host process exiting', flush=True)
        if status_heartbeat is not None:
            status_heartbeat.stop()
        motion.stop()
        payload.close()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
