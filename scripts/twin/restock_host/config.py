"""CPU-only validation and a serializable launch contract for the real CAD robot."""
import math
import json
from pathlib import Path
import xml.etree.ElementTree as ET

NS = '/restock_sim'
FRAME = 'ur_base_link'
TCP = 'gripper_pad_tcp'
JOINTS = tuple('ur_' + n + '_joint' for n in (
    'shoulder_pan', 'shoulder_lift', 'elbow', 'wrist_1', 'wrist_2', 'wrist_3'))
ENV = {'ROS_DOMAIN_ID': '88', 'ROS_LOCALHOST_ONLY': '1',
       'ROS_AUTOMATIC_DISCOVERY_RANGE': 'LOCALHOST', 'ROS_STATIC_PEERS': '',
       'RMW_IMPLEMENTATION': 'rmw_fastrtps_cpp',
       'FASTDDS_BUILTIN_TRANSPORTS': 'UDPv4', 'SKIP_DEFAULT_XML': '1'}
ENDPOINTS = {
    'compute_ik': 'moveit_msgs/srv/GetPositionIK',
    'move_action': 'moveit_msgs/action/MoveGroup',
    'execute_trajectory': 'moveit_msgs/action/ExecuteTrajectory',
    'apply_planning_scene': 'moveit_msgs/srv/ApplyPlanningScene',
    'attach_object': 'isaac_ros_cumotion_interfaces/action/AttachObject',
    'nvblox_node/get_esdf_and_gradient': 'nvblox_msgs/srv/EsdfAndGradients',
    'arm_controller/follow_joint_trajectory': 'control_msgs/action/FollowJointTrajectory',
}


def isolated_environment(environ):
    """Reject inherited DDS transports/peers that could escape loopback."""
    result = dict(environ)
    for key in ('FASTRTPS_DEFAULT_PROFILES_FILE', 'FASTDDS_DEFAULT_PROFILES_FILE',
                'ROS_DISCOVERY_SERVER', 'CYCLONEDDS_URI', 'RMW_FASTRTPS_USE_QOS_FROM_XML'):
        if result.get(key):
            raise ValueError(f'{key} must be unset for task-only DDS')
    for key, value in ENV.items():
        if key in result and result[key] != value:
            raise ValueError(f'{key} must be {value!r}')
    result.update(ENV)
    return result


def validate_config(config):
    """Require expanded CAD URDF, matching SRDF and nonempty collision XRDF."""
    import yaml
    allowed = {'urdf', 'srdf', 'xrdf', 'workspace_min', 'workspace_max',
               'domain', 'namespace', 'frame', 'tcp', 'shelfbot_root', 'mobile_manifest', 'asset_report'}
    if set(config) - allowed:
        raise ValueError(f'unknown configuration keys: {sorted(set(config)-allowed)}')
    for key, expected in [('domain', 88), ('namespace', NS), ('frame', FRAME), ('tcp', TCP)]:
        if config.get(key, expected) != expected:
            raise ValueError(f'{key} must be {expected}')
    paths = {}
    for name in ('urdf', 'srdf', 'xrdf'):
        path = Path(config.get(name, ''))
        if not path.is_absolute() or not path.is_file():
            raise ValueError(f'{name}: absolute generated CAD file required: {path}')
        paths[name] = path
    source = Path(config.get('shelfbot_root', ''))
    if (not source.is_absolute() or
            not (source / 'shelfbot/perception/object_segmenter.py').is_file() or
            not (source / 'shelfbot/perception/depth_mask.py').is_file()):
        raise ValueError('shelfbot_root: read-only shelfbot_newgripper source with object_segmenter and depth_mask required')
    robot = ET.parse(paths['urdf']).getroot()
    links = {x.attrib['name'] for x in robot.findall('link')}
    joints = {x.attrib['name']: x for x in robot.findall('joint')}
    required = {FRAME, TCP, 'base_link', 'arm_mount', 'suction_cups'}
    if not required <= links or not set(JOINTS) <= joints.keys():
        raise ValueError('URDF must contain mobile CAD base, mount, suction_cups, TCP and six ur_* joints')
    for name in JOINTS:
        joint = joints[name]
        limit = joint.find('limit')
        if joint.get('type') != 'revolute' or limit is None:
            raise ValueError(f'{name}: bounded revolute joint required')
        lo, hi, vel = (float(limit.get(k, 'nan')) for k in ('lower', 'upper', 'velocity'))
        if not all(map(math.isfinite, (lo, hi, vel))) or lo >= hi or vel <= 0:
            raise ValueError(f'{name}: invalid limits')
    manifest_path = Path(config.get('mobile_manifest', ''))
    if not manifest_path.is_absolute() or not manifest_path.is_file():
        raise ValueError('mobile_manifest: absolute generated mobile manifest required for integration stow')
    transport = json.loads(manifest_path.read_text()).get('transport_positions', {})
    for name in JOINTS:
        q = transport.get(name, float('nan'))
        limit = joints[name].find('limit')
        if not math.isfinite(q) or not float(limit.get('lower')) <= q <= float(limit.get('upper')):
            raise ValueError(f'mobile_manifest: missing or out-of-bounds transport_positions.{name}')
    for mesh in robot.findall('.//mesh'):
        filename = mesh.get('filename', '')
        if filename.startswith('package://'):
            try:
                from ament_index_python.packages import get_package_share_directory
                package, relative = filename[10:].split('/', 1)
                mesh_path = Path(get_package_share_directory(package)) / relative
            except Exception as exc:
                raise ValueError(f'URDF mesh package unresolved: {filename}') from exc
        else:
            mesh_path = Path(filename.removeprefix('file://'))
            if not mesh_path.is_absolute():
                mesh_path = paths['urdf'].parent / mesh_path
        if not filename or not mesh_path.is_file():
            raise ValueError(f'URDF mesh missing: {filename}')
    srdf = ET.parse(paths['srdf']).getroot()
    if srdf.get('name') != robot.get('name'):
        raise ValueError('SRDF robot name must match URDF')
    group = srdf.find("group[@name='ur_manipulator']")
    chain = None if group is None else group.find('chain')
    if chain is None or chain.attrib != {'base_link': FRAME, 'tip_link': TCP}:
        raise ValueError('SRDF ur_manipulator must be chain ur_base_link -> gripper_pad_tcp')
    if len(group) != 1:
        raise ValueError('SRDF arm group must contain only its six-joint chain')
    # KDL requires a chain with exactly six active joints; the CAD fingers are branches.
    parents = {j.find('child').get('link'): j for j in joints.values()}
    active, cursor, seen = [], TCP, set()
    while cursor != FRAME:
        if cursor in seen or cursor not in parents:
            raise ValueError('broken CAD TCP chain')
        seen.add(cursor)
        joint = parents[cursor]
        if joint.get('type') != 'fixed':
            active.append(joint.get('name'))
        cursor = joint.find('parent').get('link')
    if tuple(reversed(active)) != JOINTS:
        raise ValueError('TCP chain must contain exactly the six ordered UR joints')
    for pair in srdf.findall('disable_collisions'):
        if pair.get('link1') not in links or pair.get('link2') not in links:
            raise ValueError('SRDF self-collision pair references unknown link')
    xrdf = yaml.safe_load(paths['xrdf'].read_text())
    if xrdf.get('format') != 'xrdf' or xrdf.get('format_version') != 1.0:
        raise ValueError('XRDF 1.0 required')
    if xrdf.get('cspace', {}).get('joint_names') != list(JOINTS):
        raise ValueError('XRDF cspace must be six ordered ur_* joints')
    if TCP not in xrdf.get('tool_frames', []):
        raise ValueError('XRDF must use CAD gripper_pad_tcp')
    if {'set_base_frame': FRAME} not in xrdf.get('modifiers', []):
        raise ValueError('XRDF must set base frame ur_base_link')
    for section in ('collision', 'self_collision'):
        geometry = xrdf.get(section, {}).get('geometry')
        spheres = xrdf.get('geometry', {}).get(geometry, {}).get('spheres', {}) if isinstance(geometry, str) else {}
        if not spheres or not any(spheres.values()):
            raise ValueError(f'XRDF {section}: generated CAD collision spheres required')
        for link, values in spheres.items():
            if link not in links and link != 'attached_object':
                raise ValueError(f'XRDF unknown collision link: {link}')
            for sphere in values:
                center, radius = sphere.get('center', []), sphere.get('radius', 0)
                if (len(center) != 3 or not all(math.isfinite(v) for v in center)
                        or not math.isfinite(radius) or radius <= 0):
                    raise ValueError('XRDF invalid collision sphere')
        # Do not accept an arm-only sphere model with an imaginary generic tool.
        if not spheres.get('suction_cups'):
            raise ValueError(f'XRDF {section}: suction_cups collision coverage required')
        for name in JOINTS:
            link = joints[name].find('child').get('link')
            if not spheres.get(link):
                raise ValueError(f'XRDF {section}: collision coverage missing for {link}')
    for key in ('acceleration_limits', 'jerk_limits'):
        values = xrdf['cspace'].get(key, [])
        if len(values) != 6 or not all(math.isfinite(v) and v > 0 for v in values):
            raise ValueError(f'XRDF cspace {key}: six positive finite values required')
    low, high = config.get('workspace_min', []), config.get('workspace_max', [])
    if (len(low) != 3 or len(high) != 3 or
            not all(math.isfinite(a) and math.isfinite(b) and a < b for a, b in zip(low, high))):
        raise ValueError('finite ordered workspace_min/max triples required in ur_base_link')
    return {**config, **{k: str(v.resolve()) for k, v in paths.items()}}


def launch_plan(config):
    """The runtime converts only these allowlisted nodes into ROS launch actions."""
    c = validate_config(config)
    static_errors = []
    if c.get('asset_report'):
        from .evidence import validate_evidence
        report = validate_evidence(c)
        overlaps = report['transport_sphere_overlaps']
        if overlaps:
            static_errors.append(f'CAD sphere model has {len(overlaps)} nonexcluded transport overlaps; '
                                 'refine conservative coverage or prove narrow exclusions; see ' + c['asset_report'])
    else:
        static_errors.append('asset_report: current recomputable coverage evidence required; rerun generate_assets')
    return {
        'environment': ENV, 'namespace': NS, 'frame': FRAME, 'tcp': TCP,
        'robot': {k: c[k] for k in ('urdf', 'srdf', 'xrdf')},
        'workspace_min': c['workspace_min'], 'workspace_max': c['workspace_max'],
        'shelfbot_root': str(Path(c['shelfbot_root']).resolve()),
        'integration_host_arguments': ['--urdf', c['urdf'], '--shelfbot-root', c['shelfbot_root'],
                                       '--mobile-manifest', c['mobile_manifest']],
        'endpoints': {NS + '/' + k: v for k, v in ENDPOINTS.items()},
        'nodes': ['moveit_ros_move_group/move_group',
                  'isaac_ros_cumotion/StaticPlanningSceneServer',
                  'isaac_ros_cumotion/CumotionPlanner',
                  'isaac_ros_cumotion_object_attachment/ObjectAttachmentNode',
                  'shelfbot.perception.object_segmenter',
                  'nvblox_ros/NvbloxNode'],
        'required_external': ['measured joint_states and TF including all movable CAD joints',
                              'rig-masked registered depth (Image 32FC1, metres, self pixels 0) on '
                              'camera/masked_depth from the Isaac per-pixel instance-ID self mask, with '
                              'camera/masked_depth/camera_info (frame restock_camera_<id>) and /clock',
                              'sensor-derived target_bbox Float32MultiArray [u1,v1,u2,v2], fresh for every masked frame',
                              'arm_controller/follow_joint_trajectory with articulation feedback',
                              'parked station lifecycle: stop host, restart empty map, fresh ESDF ack'],
        'readiness': 'configuration only; graph, sensing and GPU execution unverified',
        'static_errors': static_errors,
    }
