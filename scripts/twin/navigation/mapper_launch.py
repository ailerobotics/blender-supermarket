"""Station-independent NVIDIA mapper launch; never commissions a motion gate.

Launch construction is CPU-only. Only an explicit caller of LaunchService.run
starts the mapper. Integration owns that GPU execution and all evidence gates.
"""
from pathlib import Path
import re
import xml.etree.ElementTree as ET

from .mapper_config import navigation_mapper_parameters

NAMESPACE='/restock_sim/navigation'
SLICE_TOPIC=NAMESPACE+'/nvblox_node/static_map_slice'


def mapper_plan(*,generation,camera_count,transport_height):
    if not isinstance(generation,str) or not re.fullmatch(r'[A-Za-z][A-Za-z0-9_]{0,63}',generation):
        raise ValueError('generation must be a fresh ROS-safe identifier, at most 64 characters')
    params=navigation_mapper_parameters(camera_count=camera_count,transport_height=transport_height)
    params['input_qos']='SENSOR_DATA'
    # Distinct input names prevent a newly launched process from subscribing to
    # retained old-generation publishers. The producer must still hold admission
    # closed until emptiness is evidenced; names alone do not prove empty state.
    inputs=[{'depth':f'{NAMESPACE}/input/{generation}/camera_{i}/depth',
             'camera_info':f'{NAMESPACE}/input/{generation}/camera_{i}/camera_info'}
            for i in range(camera_count)]
    return {'namespace':NAMESPACE,'generation':generation,'frame':'map',
            'node':'nvblox_node','container':'navigation_mapper_'+generation,
            'component':'nvblox::NvbloxNode','parameters':params,'inputs':inputs,
            'slice_topic':SLICE_TOPIC,
            # Provenance comes from the typed EvidenceSnapshot envelope published by
            # the host restock_mapper_evidence producer subclass, NOT from an ESDF
            # service request. twin.navigation.temporal.accept_envelope consumes it.
            'evidence':{'package':'restock_mapper_evidence',
                        'snapshot_topic':NAMESPACE+'/nvblox_node/evidence_snapshot',
                        'empty_generation_service':NAMESPACE+'/nvblox_node/empty_generation',
                        'open_admission_service':NAMESPACE+'/nvblox_node/open_admission',
                        'register_acquisition_service':NAMESPACE+'/nvblox_node/register_acquisition',
                        'steady_clock_domain':'host_monotonic_shared',
                        # Provenance can never certify this; stays a closed gate.
                        'full_height_observation_verified':False},
            # Upstream service exits the node in 2D mode. Never use as a probe.
            'forbidden_probe_services':[NAMESPACE+'/nvblox_node/get_esdf_and_gradient'],
            'commissioned':False,'map_load':False,
            'required_evidence':['empty native map before sensor admission',
                                 'snapshot-bound native incorporation provenance',
                                 'full-height observed/unknown reduction semantics',
                                 'native departure and blocked-aisle stop/replan']}


def validate_exports(mapper_prefix,plugin_share):
    """Validate actual component/plugin exports without loading ROS or CUDA."""
    resource=Path(mapper_prefix)/'share/ament_index/resource_index/rclcpp_components/nvblox_ros'
    entries=[line.split(';') for line in resource.read_text().splitlines()]
    if not any(len(e)==2 and e[0]=='nvblox::NvbloxNode' and
               (Path(mapper_prefix)/e[1]).is_file() for e in entries):
        raise ValueError('installed nvblox component export/library missing')
    plugin=Path(plugin_share)
    if ET.parse(plugin/'package.xml').findtext('version')!='4.4.0':
        raise ValueError('matching nvblox_nav2 4.4.0 required')
    xml=ET.parse(plugin/'nvblox_costmap_layer.xml').getroot()
    if xml.get('path')!='nvblox_nav2' or not any(
            c.get('name')==c.get('type')=='nvblox::nav2::NvbloxCostmapLayer' and
            c.get('base_class_type')=='nav2_costmap_2d::Layer' for c in xml.findall('class')):
        raise ValueError('nvblox Nav2 plugin export mismatch')


def description(*,generation,camera_count,transport_height):
    from ament_index_python.packages import get_package_prefix,get_package_share_directory
    from launch import LaunchDescription
    from launch.actions import EmitEvent,RegisterEventHandler
    from launch.event_handlers import OnProcessExit
    from launch.events import Shutdown
    from launch_ros.actions import ComposableNodeContainer
    from launch_ros.descriptions import ComposableNode
    from .nav2 import require_isolation

    require_isolation()
    plan=mapper_plan(generation=generation,camera_count=camera_count,transport_height=transport_height)
    validate_exports(get_package_prefix('nvblox_ros'),get_package_share_directory('nvblox_nav2'))
    base=Path(get_package_share_directory('nvblox_examples_bringup'))/'config/nvblox/nvblox_base.yaml'
    remaps=[('/tf','/tf'),('/tf_static','/tf_static'),('/clock','/clock')]
    for i,topics in enumerate(plan['inputs']):
        remaps.extend([(f'camera_{i}/depth/image',topics['depth']),
                       (f'camera_{i}/depth/camera_info',topics['camera_info'])])
    node=ComposableNode(package='nvblox_ros',plugin=plan['component'],name=plan['node'],
                        namespace=NAMESPACE,parameters=[str(base),plan['parameters']],remappings=remaps)
    container=ComposableNodeContainer(name=plan['container'],namespace=NAMESPACE,
        package='rclcpp_components',executable='component_container_mt',output='screen',
        composable_node_descriptions=[node])
    guard=RegisterEventHandler(OnProcessExit(target_action=container,
        on_exit=[EmitEvent(event=Shutdown(reason='navigation mapper exited'))]))
    return LaunchDescription([guard,container])
