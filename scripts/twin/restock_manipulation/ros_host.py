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
    rclpy.init()
    node = rclpy.create_node('restock_manipulation',namespace='/restock_sim',
                            parameter_overrides=[Parameter('use_sim_time',value=True)])
    stop = node.create_publisher(Bool,'/restock_sim/manipulation/stop',10)
    payload = RosPayloadSync()
    motion = RosMotionPort(node,joint_limits=tool.arm_limits,payload_sync=payload,
                           emergency_stop=lambda:stop.publish(Bool(data=True)),
                           velocity_scale=planner_velocity_scale(args.urdf),
                           pose_goal=bool(config.get('pose_goal', False)))
    controller = TransferController(motion,profiles,frame=tool.base_frame,
        phase_timeout=120.,pose_policy=WallCartonPolicy(args.shelfbot_root,tool,walls=config['walls']),
        transport_joints=transport_joints)
    payload.controller = controller
    bridge = RosTransferBridge(node,controller)
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
    try:
        rclpy.spin(node)
    finally:
        motion.stop()
        payload.close()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
