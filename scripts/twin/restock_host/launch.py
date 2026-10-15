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
    # Diagnostic-only, env-gated (default off): surface the nvblox input-queue / TF decisions
    # and the object_segmenter at debug so a depth frame that never resolves its transform is
    # explained in host-stack.log. Changes no mapping/masking behaviour. The mission enables it
    # (RESTOCK_NVBLOX_DIAG=1) for a diagnostic run; it is inert otherwise.
    nvblox_diag = bool(os.environ.get('RESTOCK_NVBLOX_DIAG'))
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
            # Disable MoveIt's WALL-clock execution-duration monitor: it cancels a trajectory that
            # the controller does not finish within ~expected_sim_duration*scaling WALL seconds, but
            # under render pacing the simulator executes each sim-second over many wall-seconds, so a
            # healthy render-paced grasp (17.6 s sim) is false-cancelled ('Controller is taking too
            # long ... Stopping trajectory' -> TIMED_OUT, run-111). trajectory_host owns sim-clocked
            # completion/tracking/settle, and the mission bounds each stage on the sim clock, so this
            # wall monitor is redundant here. Render-pacing watchdog (standing authority 212).
            'trajectory_execution': {'execution_duration_monitoring': False},
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
              # cuMotion emits the optimized trajectory resampled at interpolation_dt; MoveIt's
              # ValidateSolution densifies and FCL-checks the LINEARLY interpolated path between those
              # points. At the stock 0.05 s the straight segments between output points cut across a
              # curved reach and FCL found the gripper plate (surface_plate_1) clipping base_link at
              # 8/184 densified states (run-122 INVALID_MOTION_PLAN), even though cuMotion's own
              # (conservative-sphere) check passed. The sphere cover is sound; this is a planner-output
              # vs checker-densification discretization gap, so densify the output to hug the true path.
              interpolation_dt=0.02,
              static_planning_scene_service_name=NS + '/publish_static_planning_scene')

    def component(package, plugin, name, parameters, remappings=()):
        return ComposableNode(package=package, plugin=plugin, name=name,
                              namespace=NS, parameters=parameters,
                              remappings=list(remappings))

    lo, hi = plan['workspace_min'], plan['workspace_max']
    # input_qos must be SENSOR_DATA (best-effort): the only depth publisher into nvblox is
    # the shelfbot object_segmenter, which republishes on world_depth with SENSOR_DATA QoS.
    # nvblox's own default (SYSTEM_DEFAULT == RELIABLE) is incompatible with a best-effort
    # publisher, so depth would never reach the mapper and the ESDF would stay empty.
    world = {**common, 'global_frame': FRAME, 'map_clearing_frame_id': FRAME,
             'esdf_slice_bounds_visualization_attachment_frame_id': FRAME,
             'workspace_height_bounds_visualization_attachment_frame_id': FRAME,
             'use_depth': True, 'use_color': False, 'use_lidar': False, 'num_cameras': 1,
             'input_qos': 'SENSOR_DATA',
             'update_esdf_rate_hz': 0.0, 'static_mapper.workspace_bounds_type': 'bounding_box'}
    # nvblox's own input-queue-drop / per-message-delay counters to the stack log (see
    # nvblox_diag above); paired with the debug log levels below so queue decisions and the
    # tf2 lookup refusals that gate them both appear.
    if nvblox_diag:
        world.update(print_queue_drops_to_console=True, print_delays_to_console=True,
                     print_rates_to_console=True, print_timings_to_console=True,
                     print_statistics_on_console_period_ms=1000)
    for key, val in zip(('min_corner_x_m', 'min_corner_y_m', 'min_height_m',
                         'max_corner_x_m', 'max_corner_y_m', 'max_height_m'), lo + hi):
        world['static_mapper.workspace_bounds_' + key] = float(val)
    nvblox_node = component('nvblox_ros', 'nvblox::NvbloxNode', 'nvblox_node', [
        str(Path(share('nvblox_examples_bringup')) / 'config/nvblox/nvblox_base.yaml'),
        str(Path(share('isaac_ros_manipulation_bringup')) / 'config/nvblox/nvblox_manipulator_base.yaml'),
        world], [('camera_0/depth/image', NS + '/world_depth'),
                 ('camera_0/depth/camera_info', NS + '/camera/masked_depth/camera_info')])
    # Load order matters: composable nodes load sequentially in one container, and the
    # CumotionPlanner constructor blocks waiting for the nvblox get_esdf_and_gradient
    # service, so the mapper must load BEFORE the planner or the whole container
    # deadlocks (planner never returns, mapper never gets loaded). ObjectAttachment in
    # turn depends on the planner's robot-description service, so it follows it.
    # Order: scene server, mapper, planner, attachment.
    #
    # The host does NOT run the NVIDIA RobotSegmenter self-filter. It subscribed its Bool
    # reload trigger to cumotion/robot_description_updated, which the planner publishes as
    # std_msgs/UInt64 (cumotion_planner.hpp), so the container threw "could not create
    # subscription: invalid allocator" and the whole launch aborted. Rather than bridge an
    # incompatible reload signal, the self-filter is delegated to the Isaac side, which
    # produces exact per-pixel rig self masks from instance-ID segmentation every frame and
    # publishes robot-masked depth (rig pixels invalidated to 0) on camera/masked_depth
    # (Image 32FC1, metres) with camera/masked_depth/camera_info (frame restock_camera_<id>).
    # The object_segmenter below subscribes that masked depth, masks the grasp target out of
    # it, and republishes world_depth for nvblox (nvblox camera_info taken straight from the
    # masked_depth/camera_info stream above).
    nodes = [component('isaac_ros_cumotion',
                       'nvidia::isaac_ros::cumotion::StaticPlanningSceneServer',
                       'static_planning_scene_server', [common, {'moveit_collision_objects_scene_file': ''}]),
             nvblox_node,
             component('isaac_ros_cumotion', 'nvidia::isaac_ros::cumotion::CumotionPlanner',
                       'cumotion_planner', [cp]),
             component('isaac_ros_cumotion_object_attachment',
                       'nvidia::isaac_ros::cumotion::ObjectAttachmentNode',
                       'object_attachment', [common, {
                           'esdf_reference_frame': FRAME, 'clear_esdf_on_attach': True,
                           'get_robot_description_service': NS + '/cumotion/get_robot_description',
                           'set_robot_description_service': NS + '/cumotion/set_robot_description'}])]
    # Env-gated debug: raise only the nvblox_node logger (its TransformListener logs tf2
    # lookup refusals and the input-queue decisions under this name) -- scoped so the other
    # composable nodes in the container stay at their normal level.
    container = ComposableNodeContainer(name='host_components', namespace=NS,
        package='rclcpp_components', executable='component_container_mt', output='screen',
        composable_node_descriptions=nodes,
        ros_arguments=(['--log-level', NS.strip('/') + '.nvblox_node:=debug'] if nvblox_diag else []))
    masker = ExecuteProcess(cmd=[sys.executable, '-m', 'shelfbot.perception.object_segmenter',
        '--ros-args', '-r', '__ns:=' + NS,
        '-p', 'use_sim_time:=true', '-p', 'depth_in_topic:=' + NS + '/camera/masked_depth',
        '-p', 'depth_out_topic:=' + NS + '/world_depth',
        '-p', 'bbox_topic:=' + NS + '/target_bbox', '-p', 'bbox_timeout_s:=0.5',
        '-p', 'bbox_padding_px:=0']
        + (['--log-level', 'debug'] if nvblox_diag else []),
        additional_env={'PYTHONPATH': plan['shelfbot_root'] + os.pathsep + os.environ.get('PYTHONPATH', '')},
        output='screen')
    # Host manipulation bridge (ros_host / RosTransferBridge): the ONLY node that consumes
    # /restock_sim/manipulation/job + /sensors, drives the move_group/cuMotion action servers
    # launched above, and publishes /restock_sim/manipulation/status -- the reply the Isaac-side
    # IsaacRosGraph waits for. Without it the first manip job gets no status and dies on the Isaac
    # 5 s process/status watchdog. The verified {profiles, walls} fixture is the manipulation
    # lane's ros_host_config.json (walls cover source/cart/target so no regrasp); urdf, shelfbot
    # root and the mobile manifest come from the validated plan (integration_host_arguments).
    ros_host_config = Path(__file__).resolve().parents[1] / 'restock_manipulation' / 'ros_host_config.json'
    manipulation = ExecuteProcess(cmd=[sys.executable, '-m', 'twin.restock_manipulation.ros_host',
        '--config', str(ros_host_config), *plan['integration_host_arguments']],
        additional_env={'PYTHONPATH': plan['shelfbot_root'] + os.pathsep + os.environ.get('PYTHONPATH', '')},
        output='screen')
    # Simulation FollowJointTrajectory executor (twin.restock.trajectory_host): MoveIt above is
    # configured with MoveItSimpleControllerManager pointing at arm_controller/follow_joint_trajectory
    # (resolved in NS to /restock_sim/arm_controller/follow_joint_trajectory), but nothing advertised
    # that action server, so ExecuteTrajectory ABORTED with "action server not connected" and the host
    # emergency-stopped after the grasp plan succeeded (run-85). The host must launch this server itself,
    # in the same launch overlay + domain-88 FastDDS env, so move_group discovers it; integration must
    # NOT launch it separately (a second domain/DDS pin would not be discovered). It never plans -- it
    # interpolates move_group's plan, subscribing measured /restock_sim/joint_states and publishing
    # /restock_sim/arm_joint_command (the articulation command Isaac drives the UR10 with). --urdf is the
    # same generated CAD urdf the plan validates (host.json["urdf"]); it reads arm joint limits only.
    trajectory = ExecuteProcess(cmd=[sys.executable, '-m', 'twin.restock.trajectory_host',
        '--urdf', robot['urdf']], output='screen')
    processes = [moveit, container, masker, manipulation, trajectory]
    # Losing mapping/masking must not leave an apparently healthy planner behind.
    guards = [RegisterEventHandler(OnProcessExit(target_action=process,
              on_exit=[EmitEvent(event=Shutdown(reason='restock host component exited'))]))
              for process in processes]
    # Host-published nvblox mapper-incorporation acknowledgement. nvblox runs
    # update_esdf_rate_hz 0.0 (on-request only), so this node's ESDF service poll is what
    # integrates pending masked depth, and it publishes the real NVIDIA incorporation
    # stamp as collision_time on station/status -- the stamp the mission syncs onto
    # IsaacMeasuredFeedback so the first manip job answers inside the Isaac watchdog.
    # Auxiliary: it is deliberately NOT a shutdown guard (its loss must not tear down the
    # planner), but the guards above do tear it down with the stack, so a live ack node
    # implies a live planner/mapper -- which is why its snapshot reports process_alive.
    ack = ExecuteProcess(cmd=[sys.executable, '-m', 'twin.restock_host.incorporation_ack',
        '--workspace-min', *(str(v) for v in plan['workspace_min']),
        '--workspace-max', *(str(v) for v in plan['workspace_max'])], output='screen')
    return LaunchDescription(guards + processes + [ack])
