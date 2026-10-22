"""Host-Jazzy entry point for the existing ROS transfer/motion contract.

Run with host Python, never Isaac's interpreter. Starts only task-domain clients
and controller; GPU planner, mapper and trajectory servers are launched separately
by integration. No hardware launch or simulator boot occurs here.
"""
import argparse
from dataclasses import asdict
import json
import sys

from .ros_adapter import (RosMotionPort, RosTransferBridge, require_sim_domain,
                          sensors_from_json,planner_velocity_scale)


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
    controller = TransferController(motion,profiles,frame=tool.base_frame,
        phase_timeout=120.,pose_policy=_policy,
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
        # fm390: drive the base in front of the rollwagen mouth between the carry fold and the place
        # (lift->tuck->TRANSIT->[base nav]->preplace), instead of the arm-only cage insert. Absent =>
        # byte-identical. Mutually exclusive in practice with cart_cage_place (base nav replaces it).
        reposition_before_place=bool(config.get('reposition_before_place', False)),
        # fm410 (run-316): physics grasp confirm -- the lift-probe carton-tracking threshold (m). The grasp
        # advances to carry only if the measured carton rose by >= this from the grasp-confirm baseline, so
        # the confirm reads physics, never the (arm-occluded) source camera. Absent => inert (byte-identical).
        grasp_track_m=config.get('grasp_track_m'),
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
        steady_clock=lambda: node.get_clock().now().nanoseconds*1e-9)
    payload.controller = controller
    # collect-gt: the ground-truth collect switch is the shared RESTOCK_COLLECT_GROUND_TRUTH env (inherited from
    # the run env); log the host's view at boot so an Isaac/host activation split is visible immediately.
    print('COLLECT_GROUND_TRUTH host enabled=%s' % controller.collect_ground_truth, flush=True)
    # The first decode-grade frame + support registration under render pacing needs several
    # ~1 s RTX renders (~5-15 s WALL); the default 5 s wall startup grace flaked (run-108 failed
    # 'no initial sensor frame' where run-106/107 cleared it only by luck). Widen it for the sim
    # via config; the runtime.py sim-clock stall guard (PR 86) still catches a job that genuinely
    # never produces a frame, so the generous wall grace cannot hide a real stall.
    bridge = RosTransferBridge(node,controller,
        startup_grace_seconds=float(config.get('startup_grace_seconds', 5.0)))
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
    for spec in config.get('world_obstacles', []):
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
        'reposition_before_place': controller.reposition_before_place,
        'transfer_mode': controller.transfer_mode,   # env RESTOCK_TRANSFER_MODE (None = off)
        'transfer_carry': sorted(controller.transfer_carry),   # env RESTOCK_TRANSFER_CARRY ([] = off)
        'sim_cart_place_policy': _policy is not None and bool(config.get('sim_cart_place_policy')),
        'base_collision_object': bool(config.get('base_collision_object', False)),
        'world_obstacles': [s['id'] for s in config.get('world_obstacles', [])],
    }), flush=True)
    try:
        rclpy.spin(node)
    finally:
        motion.stop()
        payload.close()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
