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
    motion = RosMotionPort(node,joint_limits=tool.arm_limits,payload_sync=payload,
                           emergency_stop=lambda:stop.publish(Bool(data=True)),
                           velocity_scale=planner_velocity_scale(
                               args.urdf,command_limit=arm_max_speed,scale_cap=arm_max_speed),
                           max_velocity_scale=min(arm_max_speed,1.),
                           pose_goal=bool(config.get('pose_goal', False)),
                           stow_config=transport_joints,
                           plan_wall_cap_s=float(config.get('plan_wall_cap_s', 22.0)))
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
    # stall). Only in the sim cart-place scenario (the policy + flag); cart_pan_joint0 is CPU-verified
    # body-clear for the fixed cart dock (staged_swing_verify.py).
    staged = bool(config.get('staged_cart_swing', False)) and _policy is not None and bool(config.get('sim_cart_place_policy'))
    controller = TransferController(motion,profiles,frame=tool.base_frame,
        phase_timeout=120.,pose_policy=_policy,
        transport_joints=transport_joints,
        contact_straight_line=bool(config.get('contact_straight_line', False)),
        contact_overpress_step_m=float(config.get('contact_overpress_step_m', 0.0)),
        contact_overpress_cap_m=float(config.get('contact_overpress_cap_m', 0.0)),
        staged_cart_swing=staged, cart_pan_joint0=float(config.get('cart_pan_joint0', 0.0)),
        cart_tuck_shape=config.get('cart_tuck_shape'),
        steady_clock=lambda: node.get_clock().now().nanoseconds*1e-9)
    payload.controller = controller
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
    try:
        rclpy.spin(node)
    finally:
        motion.stop()
        payload.close()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
