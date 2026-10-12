"""No ROS node creation, CUDA import, plugin loading or planner computation."""
import importlib
import subprocess
from pathlib import Path
import xml.etree.ElementTree as ET

PACKAGES = ('moveit_ros_move_group', 'moveit_kinematics',
            'moveit_simple_controller_manager', 'rmw_fastrtps_cpp',
            'isaac_ros_cumotion_moveit', 'isaac_ros_cumotion',
            'isaac_ros_cumotion_robot_segmenter',
            'isaac_ros_cumotion_object_attachment', 'nvblox_ros',
            'nvblox_examples_bringup', 'isaac_ros_manipulation_bringup',
            'rclcpp_components')
FILES = {
    'moveit_ros_move_group': ('../lib/moveit_ros_move_group/move_group',),
    'isaac_ros_cumotion': ('params/isaac_ros_cumotion_params.yaml',),
    'isaac_ros_cumotion_moveit': ('config/isaac_ros_cumotion_planning.yaml',),
    'nvblox_examples_bringup': ('config/nvblox/nvblox_base.yaml',),
    'isaac_ros_manipulation_bringup': ('config/nvblox/nvblox_manipulator_base.yaml',),
}
INTERFACES = {'moveit_msgs.srv': ('GetPositionIK', 'ApplyPlanningScene'),
              'moveit_msgs.action': ('MoveGroup', 'ExecuteTrajectory'),
              'control_msgs.action': ('FollowJointTrajectory',),
              'isaac_ros_cumotion_interfaces.action': ('AttachObject',),
              'nvblox_msgs.srv': ('EsdfAndGradients',)}
COMPONENTS = {'isaac_ros_cumotion': ('CumotionPlanner', 'StaticPlanningSceneServer'),
              'isaac_ros_cumotion_robot_segmenter': ('RobotSegmenter',),
              'isaac_ros_cumotion_object_attachment': ('ObjectAttachmentNode',),
              'nvblox_ros': ('NvbloxNode',)}


def link_errors(library):
    """ldd inspects trusted installed ELF dependencies; no plugin/node is loaded."""
    try:
        result = subprocess.run(['ldd', str(library)], capture_output=True, text=True,
                                timeout=10, check=False)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return [f'{library}: link inspection failed ({type(exc).__name__})']
    missing = [line.strip() for line in (result.stdout + result.stderr).splitlines()
               if 'not found' in line or 'version `' in line]
    if result.returncode and not missing:
        missing = [f'ldd exited {result.returncode}']
    return [f'{library}: {line}' for line in missing]


def dependencies(resolve=None):
    """Return exact missing packages/interfaces, never mistake an image for overlay."""
    errors, versions = [], {}
    if resolve is None:
        try:
            from ament_index_python.packages import get_package_share_directory
            resolve = get_package_share_directory
        except ImportError:
            return {'versions': {}, 'errors': ['ament_index_python unavailable: use /usr/bin/python3 after sourcing Jazzy and task-local NVIDIA overlay']}
    for package in PACKAGES:
        try:
            share = Path(resolve(package))
            versions[package] = ET.parse(share / 'package.xml').getroot().findtext('version')
            for relative in FILES.get(package, ()):
                path = (share.parent.parent / 'lib' / package / 'move_group'
                        if relative.startswith('../lib/') else share / relative)
                if not path.is_file():
                    errors.append(f'{package}: missing {path}')
            if package in COMPONENTS:
                prefix = share.parent.parent
                index = prefix / 'share/ament_index/resource_index/rclcpp_components' / package
                exports = dict(line.split(';', 1) for line in index.read_text().splitlines() if ';' in line)
                for required in COMPONENTS[package]:
                    matches = [library for plugin, library in exports.items()
                               if plugin.split('::')[-1] == required]
                    if len(matches) != 1:
                        errors.append(f'{package}: missing component export {required}')
                    else:
                        errors.extend(link_errors(prefix / matches[0]))
            if package == 'isaac_ros_cumotion_moveit':
                errors.extend(link_errors(share.parent.parent / 'lib/libisaac_ros_cumotion_moveit.so'))
        except Exception as exc:
            errors.append(f'{package}: unavailable ({type(exc).__name__}); provide matching Jazzy package in task-local overlay')
    for module, classes in INTERFACES.items():
        try:
            imported = importlib.import_module(module)
            for name in classes:
                getattr(imported, name)
            # Check compiled typesupport ABI, not merely generated Python stubs.
            from rosidl_generator_py import import_type_support
            import_type_support(module.split('.')[0])
        except Exception as exc:
            errors.append(f'{module}: message/typesupport import failed ({type(exc).__name__}); use matching Python 3.12 Jazzy overlay')
    for module in ('cv_bridge', 'numpy', 'rclpy', 'launch_ros'):
        try:
            importlib.import_module(module)
        except ImportError:
            errors.append(f'{module}: unavailable; required by host launch and reused shelfbot object segmenter')
    return {'versions': versions, 'errors': errors}


def incorporated_time(response, *, now, parked_since, max_age=0.5):
    """Only NVIDIA's successful sensor integration stamp can advance collision_time.

    Call after mapper restart at a measured parked station. This validates a real
    response; it does not publish timestamps or make input receipt an acknowledgement.
    """
    import math
    stamp = response.header.stamp.sec + response.header.stamp.nanosec * 1e-9
    data = response.esdf_and_gradients.data
    if (not response.success or response.header.frame_id != 'ur_base_link' or
            not data or not all(math.isfinite(v) for v in data) or
            not math.isfinite(response.voxel_size_m) or response.voxel_size_m <= 0 or
            not all(math.isfinite(v) for v in (now, parked_since, max_age, stamp)) or
            max_age <= 0 or not parked_since < stamp <= now or now-stamp > max_age):
        raise ValueError('ESDF absent, stale, empty, wrong frame or predates parked station')
    return stamp
