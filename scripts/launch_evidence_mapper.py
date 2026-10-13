#!/usr/bin/env python3
"""Launch the navigation nvblox mapper as the restock_mapper_evidence producer.

This is the host-owned variant of scripts/launch_navigation_mapper.py: it loads
the restock_mapper_evidence::EvidenceProducerNode component (the NvbloxNode
subclass that binds each slice to the incorporated-acquisition ledger) instead
of the stock nvblox::NvbloxNode, with the evidence.* parameters the producer
needs. Node name, namespace, parameters and input remaps are taken verbatim from
the landed navigation mapper plan, so topics and the stable map frame are
unchanged. Construction/launch is CPU; --execute starts the mapper and must run
only under the GPU guard (see scripts/run_evidence_mapper_gpu.py). The base raw
static_map_slice is remapped to a dead internal topic so only the certified
EvidenceSnapshot carries geometry.
"""
import argparse
import json

from twin.navigation.mapper_launch import mapper_plan, NAMESPACE


def _evidence_params(args, plan):
    params = dict(plan['parameters'])
    slice_height = params.get('static_mapper.esdf_slice_height', 0.03)
    merged = {
        **params,
        'evidence.session': args.session,
        'evidence.scene': args.scene,
        'evidence.cameras': args.cameras,
        'evidence.localization': args.localization,
        'evidence.producer_nonce': args.generation,
        'evidence.steady_clock_domain': plan['evidence']['steady_clock_domain'],
        'evidence.slice_height': float(slice_height),
        'evidence.unknown_value': 1000.0,
        'evidence.sensors': list(args.sensors),
    }
    # Diagnostic/launch override: coerce k=v strings into the native param type.
    for item in getattr(args, 'mapper_param', None) or []:
        key, _, value = item.partition('=')
        key = key.strip()
        value = value.strip()
        if value.lower() in ('true', 'false'):
            merged[key] = (value.lower() == 'true')
        else:
            try:
                merged[key] = int(value) if value.lstrip('-').isdigit() else float(value)
            except ValueError:
                merged[key] = value
    return [merged]


def description(args):
    from ament_index_python.packages import get_package_share_directory
    from pathlib import Path
    from launch import LaunchDescription
    from launch.actions import EmitEvent, RegisterEventHandler
    from launch.event_handlers import OnProcessExit
    from launch.events import Shutdown
    from launch_ros.actions import ComposableNodeContainer
    from launch_ros.descriptions import ComposableNode
    from twin.navigation.nav2 import require_isolation

    require_isolation()
    plan = mapper_plan(generation=args.generation, camera_count=args.camera_count,
                       transport_height=args.transport_height)
    base = Path(get_package_share_directory('nvblox_examples_bringup')) / 'config/nvblox/nvblox_base.yaml'
    remaps = [('/tf', '/tf'), ('/tf_static', '/tf_static'), ('/clock', '/clock'),
              # certified geometry travels only in the EvidenceSnapshot envelope.
              ('~/static_map_slice', '~/raw_static_map_slice_internal')]
    for i, topics in enumerate(plan['inputs']):
        remaps.extend([(f'camera_{i}/depth/image', topics['depth']),
                       (f'camera_{i}/depth/camera_info', topics['camera_info'])])
    node = ComposableNode(
        package='restock_mapper_evidence',
        plugin='restock_mapper_evidence::EvidenceProducerNode',
        name=plan['node'], namespace=NAMESPACE,
        parameters=[str(base)] + _evidence_params(args, plan), remappings=remaps)
    container = ComposableNodeContainer(
        name=plan['container'] + '_evidence', namespace=NAMESPACE,
        package='rclcpp_components', executable='component_container_mt', output='screen',
        composable_node_descriptions=[node])
    guard = RegisterEventHandler(OnProcessExit(
        target_action=container, on_exit=[EmitEvent(event=Shutdown(reason='evidence mapper exited'))]))
    return LaunchDescription([guard, container]), plan


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--generation', required=True)
    p.add_argument('--camera-count', required=True, type=int)
    p.add_argument('--transport-height', required=True, type=float)
    p.add_argument('--session', default='mission_a')
    p.add_argument('--scene', default='supermarket')
    p.add_argument('--cameras', default='front')
    p.add_argument('--localization', default='amcl')
    p.add_argument('--sensors', nargs='+', default=['front=front_optical'])
    p.add_argument('--mapper-param', action='append', default=[],
                   help='override a native mapper parameter, KEY=VALUE (repeatable)')
    p.add_argument('--execute', action='store_true', help='start mapper; GPU guard required')
    args = p.parse_args()
    desc, plan = description(args) if args.execute else (None, mapper_plan(
        generation=args.generation, camera_count=args.camera_count, transport_height=args.transport_height))
    print(json.dumps(plan, indent=2), flush=True)
    if args.execute:
        from launch import LaunchService
        service = LaunchService()
        service.include_launch_description(desc)
        return service.run()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
