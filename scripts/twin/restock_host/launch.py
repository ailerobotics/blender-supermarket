"""Explicit launch of installed components, never vendor hardware launch files."""
from pathlib import Path
import os
import sys

from .config import FRAME, JOINTS, NS, launch_plan


def description(config):
    import yaml
    from ament_index_python.packages import get_package_share_directory as share
    from launch import LaunchDescription
    from launch.actions import ExecuteProcess, EmitEvent, RegisterEventHandler
    from launch.event_handlers import OnProcessExit
    from launch.events import Shutdown
    from launch_ros.actions import Node, ComposableNodeContainer
    from launch_ros.descriptions import ComposableNode

    plan = launch_plan(config)
    if plan['static_errors']:
        raise ValueError('; '.join(plan['static_errors']))
    robot = plan['robot']

    def params(package, relative):
        return yaml.safe_load((Path(share(package)) / relative).read_text())['/**']['ros__parameters']

    common = {'use_sim_time': True}
    pipeline = yaml.safe_load((Path(share('isaac_ros_cumotion_moveit')) /
                             'config/isaac_ros_cumotion_planning.yaml').read_text())
    pipeline['response_adapters'] = [
        'default_planning_response_adapters/AddTimeOptimalParameterization',
        'default_planning_response_adapters/ValidateSolution']
    xrdf = yaml.safe_load(Path(robot['xrdf']).read_text())
    limits = {j: {'has_acceleration_limits': True, 'max_acceleration': float(a)}
              for j, a in zip(JOINTS, xrdf['cspace']['acceleration_limits'])}
    moveit = Node(
        package='moveit_ros_move_group', executable='move_group', namespace=NS,
        name='move_group', output='screen', parameters=[{
            **common, 'robot_description': Path(robot['urdf']).read_text(),
            'robot_description_semantic': Path(robot['srdf']).read_text(),
            'robot_description_kinematics': {'ur_manipulator': {
                'kinematics_solver': 'kdl_kinematics_plugin/KDLKinematicsPlugin',
                'kinematics_solver_timeout': 0.1}},
            'robot_description_planning': {'joint_limits': limits},
            'planning_pipelines': ['isaac_ros_cumotion'],
            'default_planning_pipeline': 'isaac_ros_cumotion',
            'isaac_ros_cumotion': pipeline,
            'allow_trajectory_execution': True,
            'moveit_manage_controllers': False,
            'moveit_controller_manager': 'moveit_simple_controller_manager/MoveItSimpleControllerManager',
            'moveit_simple_controller_manager': {
                'controller_names': ['arm_controller'], 'arm_controller': {
                    'type': 'FollowJointTrajectory', 'action_ns': 'follow_joint_trajectory',
                    'default': True, 'joints': list(JOINTS)}},
        }], remappings=[('joint_states', NS + '/joint_states'),
                        ('/joint_states', NS + '/joint_states')])

    cp = params('isaac_ros_cumotion', 'params/isaac_ros_cumotion_params.yaml')
    cp.update(common, urdf_file_path=robot['urdf'], xrdf_file_path=robot['xrdf'],
              read_esdf_world=True, update_esdf_on_request=True,
              esdf_service_name=NS + '/nvblox_node/get_esdf_and_gradient',
              joint_states_topic=NS + '/joint_states', enable_cuda_mps=False,
              static_planning_scene_service_name=NS + '/publish_static_planning_scene')

    def component(package, plugin, name, parameters, remappings=()):
        return ComposableNode(package=package, plugin=plugin, name=name,
                              namespace=NS, parameters=parameters,
                              remappings=list(remappings))

    nodes = [component('isaac_ros_cumotion',
                       'nvidia::isaac_ros::cumotion::StaticPlanningSceneServer',
                       'static_planning_scene_server', [common, {'moveit_collision_objects_scene_file': ''}]),
             component('isaac_ros_cumotion', 'nvidia::isaac_ros::cumotion::CumotionPlanner',
                       'cumotion_planner', [cp]),
             component('isaac_ros_cumotion_object_attachment',
                       'nvidia::isaac_ros::cumotion::ObjectAttachmentNode',
                       'object_attachment', [common, {
                           'esdf_reference_frame': FRAME, 'clear_esdf_on_attach': True,
                           'get_robot_description_service': NS + '/cumotion/get_robot_description',
                           'set_robot_description_service': NS + '/cumotion/set_robot_description'}]),
             component('isaac_ros_cumotion_robot_segmenter',
                       'nvidia::isaac_ros::manipulator::RobotSegmenter',
                       'robot_segmenter', [common, {
                           'urdf_path': robot['urdf'], 'xrdf_path': robot['xrdf'],
                           'robot_base_frame': FRAME, 'input_qos': 'SENSOR_DATA',
                           'output_qos': 'SENSOR_DATA', 'additional_buffer_distance': 0.05,
                           'reload_robot_description_topic': NS + '/cumotion/robot_description_updated',
                           'robot_description_service_name': NS + '/cumotion/get_robot_description'}],
                       [('depth_image', NS + '/camera/depth'),
                        ('camera_info_depth', NS + '/camera/camera_info'),
                        ('joint_states', NS + '/joint_states'),
                        ('robot_mask', NS + '/robot_mask'),
                        ('robot_depth', NS + '/robot_world_depth')])]
    lo, hi = plan['workspace_min'], plan['workspace_max']
    world = {**common, 'global_frame': FRAME, 'map_clearing_frame_id': FRAME,
             'esdf_slice_bounds_visualization_attachment_frame_id': FRAME,
             'workspace_height_bounds_visualization_attachment_frame_id': FRAME,
             'use_depth': True, 'use_color': False, 'use_lidar': False, 'num_cameras': 1,
             'update_esdf_rate_hz': 0.0, 'static_mapper.workspace_bounds_type': 'bounding_box'}
    for key, val in zip(('min_corner_x_m', 'min_corner_y_m', 'min_height_m',
                         'max_corner_x_m', 'max_corner_y_m', 'max_height_m'), lo + hi):
        world['static_mapper.workspace_bounds_' + key] = float(val)
    nodes.append(component('nvblox_ros', 'nvblox::NvbloxNode', 'nvblox_node', [
        str(Path(share('nvblox_examples_bringup')) / 'config/nvblox/nvblox_base.yaml'),
        str(Path(share('isaac_ros_manipulation_bringup')) / 'config/nvblox/nvblox_manipulator_base.yaml'),
        world], [('camera_0/depth/image', NS + '/world_depth'),
                 ('camera_0/depth/camera_info', NS + '/camera/camera_info')]))
    container = ComposableNodeContainer(name='host_components', namespace=NS,
        package='rclcpp_components', executable='component_container_mt', output='screen',
        composable_node_descriptions=nodes)
    masker = ExecuteProcess(cmd=[sys.executable, '-m', 'shelfbot.perception.object_segmenter',
        '--ros-args', '-r', '__ns:=' + NS,
        '-p', 'use_sim_time:=true', '-p', 'depth_in_topic:=' + NS + '/robot_world_depth',
        '-p', 'depth_out_topic:=' + NS + '/world_depth',
        '-p', 'bbox_topic:=' + NS + '/target_bbox', '-p', 'bbox_timeout_s:=0.5',
        '-p', 'bbox_padding_px:=0'],
        additional_env={'PYTHONPATH': plan['shelfbot_root'] + os.pathsep + os.environ.get('PYTHONPATH', '')},
        output='screen')
    processes = [moveit, container, masker]
    # Losing mapping/masking must not leave an apparently healthy planner behind.
    guards = [RegisterEventHandler(OnProcessExit(target_action=process,
              on_exit=[EmitEvent(event=Shutdown(reason='restock host component exited'))]))
              for process in processes]
    return LaunchDescription(guards + processes)
