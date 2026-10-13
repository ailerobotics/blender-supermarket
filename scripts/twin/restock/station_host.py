"""Task-local station supervisor; run only with the authorized simulator launch.

Starts the real host planner/mapper per parked epoch and reports only validated
NVIDIA ESDF incorporation timestamps. No dummy planner, map, payload or feedback.
"""
import argparse
import json
import os
from pathlib import Path
import sys
import time

from twin.restock_host.config import isolated_environment,launch_plan
from twin.restock_host.preflight import incorporated_time
from twin.restock_host.readiness import missing_endpoints
from .station import OwnedProcess,StationState


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    env=isolated_environment(os.environ)
    os.environ.update(env)
    config=json.loads(args.config.read_text())
    plan=launch_plan(config)
    if plan['static_errors']:
        raise ValueError('; '.join(plan['static_errors']))
    args.output.mkdir(parents=True,exist_ok=False)
    import rclpy
    from rclpy.parameter import Parameter
    from rclpy.action import get_action_server_names_and_types_by_node
    from std_msgs.msg import String,Bool
    from nvblox_msgs.srv import EsdfAndGradients
    rclpy.init()
    node=rclpy.create_node('restock_station_supervisor',namespace='/restock_sim',
        parameter_overrides=[Parameter('use_sim_time',value=True)])
    if node.context.get_domain_id()!=88:
        raise RuntimeError('station supervisor requires domain 88')
    state=StationState()
    publisher=node.create_publisher(String,'/restock_sim/station/status',10)
    stop=node.create_publisher(Bool,'/restock_sim/manipulation/stop',10)
    service=node.create_client(EsdfAndGradients,'/restock_sim/nvblox_node/get_esdf_and_gradient')
    child=None
    pending=None
    pending_epoch=None
    pending_wall=0.
    last_discovery=0.
    endpoints_ready=False
    now=lambda:node.get_clock().now().nanoseconds*1e-9

    def shutdown_station():
        nonlocal child,pending,endpoints_ready
        if pending is not None:
            pending.cancel()
            pending=None
        if child is not None:
            child.close()
            child=None
        endpoints_ready=False

    def command(message):
        nonlocal child,last_discovery
        try:
            action=state.accept(json.loads(message.data),now())
            if action in ('prepare','stop'):
                shutdown_station()
            if action=='prepare':
                script=Path(__file__).resolve().parents[2]/'run_restock_host.py'
                child=OwnedProcess([sys.executable,str(script),'--config',str(args.config),'--launch'],
                    args.output/f'station-{state.epoch}.log',env=env)
                child.start()
                last_discovery=0.
        except Exception as exc:
            state.fault=str(exc)
            shutdown_station()
            stop.publish(Bool(data=True))

    def tick():
        nonlocal pending,pending_epoch,pending_wall,last_discovery,endpoints_ready
        try:
            stamp=now()
            if state.active and not state.fault:
                if time.monotonic()-last_discovery>.5:
                    actions=[]
                    for name,namespace in node.get_node_names_and_namespaces():
                        actions.extend(get_action_server_names_and_types_by_node(node,name,namespace))
                    endpoints_ready=not missing_endpoints(node.get_service_names_and_types(),actions)
                    last_discovery=time.monotonic()
                if pending is not None and pending.done():
                    response=pending.result()
                    pending=None
                    # Warm-up responses can be empty/stale; they never advance readiness.
                    try:
                        incorporated=incorporated_time(response,now=stamp,parked_since=state.parked_since)
                        state.acknowledge(incorporated,pending_epoch)
                    except (ValueError,AttributeError):
                        pass
                if pending is not None and time.monotonic()-pending_wall>2.:
                    pending.cancel()
                    pending=None
                if pending is None and endpoints_ready and service.service_is_ready():
                    request=EsdfAndGradients.Request()
                    request.update_esdf=True
                    request.visualize_esdf=False
                    request.use_aabb=True
                    request.frame_id='ur_base_link'
                    lo,hi=config['workspace_min'],config['workspace_max']
                    for axis,a,b in zip(('x','y','z'),lo,hi):
                        setattr(request.aabb_min_m,axis,float(a))
                        setattr(request.aabb_size_m,axis,float(b-a))
                    pending=service.call_async(request)
                    pending_epoch=state.epoch
                    pending_wall=time.monotonic()
            result=state.snapshot(stamp,endpoints_ready=endpoints_ready,
                                  process_alive=bool(child and child.alive))
            if state.fault:
                shutdown_station()
                stop.publish(Bool(data=True))
            publisher.publish(String(data=json.dumps(result,allow_nan=False)))
        except Exception as exc:
            state.fault=str(exc)
            shutdown_station()
            stop.publish(Bool(data=True))

    node.create_subscription(String,'/restock_sim/station/command',command,10)
    node.create_timer(.05,tick)
    try:
        rclpy.spin(node)
    finally:
        shutdown_station()
        stop.publish(Bool(data=True))
        node.destroy_node()
        rclpy.shutdown()


if __name__=='__main__':
    main()
