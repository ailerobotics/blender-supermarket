"""`manip_factory` for the mission bundle: build the production manipulation runtime
from the landed (retired-lane) restock_manipulation modules, unchanged.

The integration bundle (`twin.restock.isaac_bundle`) resolves this as the runtime-config
`manip_factory` ref and calls `create(app=, assembly=, plan=, config=, output=)`. It
returns a fully-configured `ManipulationBridge` (owns_sensor_io=True) wrapping the LANDED
HOST cuMotion path: a `twin.restock.ros_manipulation.RosManipulationRuntime` that bridges the
real arm-camera packets over domain-88 ROS (native Isaac `IsaacRosGraph`) to the host
MoveGroup/cuMotion planner, `TransferController` (CAD side-grip + support-contact gates) and
`trajectory_host`. No in-process cuRobo planner is constructed inside Isaac; no sensor data is
fabricated and the landed host-side modules are unchanged. This module only assembles the
Isaac-side camera/graph and prepares the arm station per mission action.

`parse_manipulation_config()` is the pure, CPU-tested contract: it validates the grasp
profiles (including the scenario SKU and a single calibrated grasp mode), the per-station
sensing slots + `base_from_camera` calibration, and the shelfbot package root from
`config['manipulation']`. The Isaac construction (arm camera + ROS graph +
`RosManipulationRuntime`) is isolated behind the `_build_runtime` seam so the wiring is proven
offline without a boot.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

from .types import ProductProfile

STATIONS = ('source', 'shelf')
# Mission action -> arm station whose ROI/calibration the sensors use.
ACTION_STATION = {'collect': 'source', 'transfer': 'shelf'}


def _finite_matrix4(m):
    return (isinstance(m, (list, tuple)) and len(m) == 4
            and all(isinstance(r, (list, tuple)) and len(r) == 4
                    and all(isinstance(v, (int, float)) and math.isfinite(v) for v in r) for r in m))


def _profile(entry):
    if not isinstance(entry, dict):
        raise ValueError('each grasp profile must be an object')
    try:
        sku = entry['sku']
        dims = tuple(float(v) for v in entry['dimensions'])
        quat = tuple(float(v) for v in entry['tcp_quaternion'])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f'grasp profile missing sku/dimensions/tcp_quaternion: {exc}')
    if len(dims) != 3 or len(quat) != 4:
        raise ValueError('profile dimensions must be 3 and tcp_quaternion 4 values')
    optional = {k: entry[k] for k in ('tcp_surface_offset', 'approach', 'max_reach', 'mass_kg', 'grasp_mode')
                if k in entry}
    return ProductProfile(sku=sku, dimensions=dims, tcp_quaternion=quat, **optional)


@dataclass(frozen=True)
class ManipConfig:
    profiles: dict               # sku -> ProductProfile (single grasp mode)
    stations: dict               # 'source'/'shelf' -> {'slots': tuple, 'base_from_camera': 4x4 list}
    shelfbot_root: str
    walls: dict
    voxel_dims: tuple
    voxel_size: float
    grasp_mode: str
    pose_bridge_root: str = ''    # filesystem root of the tray-perception pose bridge
    masked_depth: dict = field(default_factory=dict)  # arm masked-depth bridge settings
    collect_source_location: str = ''  # mission location of the collect pick (fixed-source target)


def parse_manipulation_config(config, job_sku):
    """Validate config['manipulation'] into a ManipConfig. Pure; no Isaac, fail-closed.

    `job_sku` is the scenario job SKU (carried on the BundlePlan so the frozen factory
    call signature need not change); the runtime must hold a grasp profile for it.
    """
    if not isinstance(config, dict):
        raise ValueError('config must be an object')
    if not isinstance(job_sku, str) or not job_sku:
        raise ValueError('job_sku required')
    mcfg = config.get('manipulation')
    if not isinstance(mcfg, dict):
        raise ValueError('config.manipulation object required (profiles, stations, shelfbot_root)')

    entries = mcfg.get('profiles')
    if not isinstance(entries, (list, tuple)) or not entries:
        raise ValueError('config.manipulation.profiles must be a non-empty list')
    profiles = {}
    for entry in entries:
        p = _profile(entry)
        if p.sku in profiles:
            raise ValueError(f'duplicate grasp profile for SKU {p.sku!r}')
        profiles[p.sku] = p
    modes = {p.grasp_mode for p in profiles.values()}
    if len(modes) != 1:
        raise ValueError('use exactly one calibrated grasp mode per runtime')
    if job_sku not in profiles:
        raise ValueError(f'scenario job SKU {job_sku!r} has no grasp profile')

    stations_in = mcfg.get('stations')
    if not isinstance(stations_in, dict) or set(stations_in) != set(STATIONS):
        raise ValueError(f'config.manipulation.stations must provide exactly {STATIONS}')
    stations = {}
    for name in STATIONS:
        st = stations_in[name]
        if not isinstance(st, dict) or 'slots' not in st or 'base_from_camera' not in st:
            raise ValueError(f'station {name!r} needs slots and base_from_camera')
        if not isinstance(st['slots'], (list, tuple)) or not st['slots']:
            raise ValueError(f'station {name!r} slots must be a non-empty list')
        if not _finite_matrix4(st['base_from_camera']):
            raise ValueError(f'station {name!r} base_from_camera must be a finite 4x4 matrix')
        stations[name] = {'slots': tuple(st['slots']),
                          'base_from_camera': [list(map(float, r)) for r in st['base_from_camera']]}

    shelfbot_root = mcfg.get('shelfbot_root')
    if not isinstance(shelfbot_root, str) or not shelfbot_root:
        raise ValueError('config.manipulation.shelfbot_root (package root) required')

    walls = mcfg.get('walls', {})
    if not isinstance(walls, dict):
        raise ValueError('config.manipulation.walls must be an object')
    vd = tuple(mcfg.get('voxel_dims', (2.4, 2.4, 2.4)))
    if len(vd) != 3 or not all(isinstance(v, (int, float)) and math.isfinite(v) and v > 0 for v in vd):
        raise ValueError('voxel_dims must be three positive finite values')
    vs = mcfg.get('voxel_size', 0.02)
    if not isinstance(vs, (int, float)) or not math.isfinite(vs) or not 0 < vs <= 0.05:
        raise ValueError('voxel_size must be in (0, 0.05]')

    # Top-level runtime-config field (not under 'manipulation'): the tray-perception
    # package root IsaacCartonSensors loads for depth geometry. Optional at parse (the
    # pure CPU contract); the live host-path builder requires it.
    pose_bridge_root = config.get('pose_bridge_root', '')
    # Arm eye-in-hand masked-depth bridge for the host nvblox (option B). Off unless the
    # runtime-config opts in; the pure settings are validated here, the publishers wired live.
    md = config.get('masked_depth', {})
    if not isinstance(md, dict):
        raise ValueError('config.masked_depth must be an object')
    masked_depth = {}
    if md.get('enabled'):
        prefixes = md.get('rig_prefixes', [])
        if prefixes and (not isinstance(prefixes, (list, tuple))
                         or not all(isinstance(p, str) and p for p in prefixes)):
            raise ValueError('masked_depth.rig_prefixes must be a list of USD prim-path strings')
        runtime_env = md.get('runtime_env')
        if not isinstance(runtime_env, str) or not runtime_env:
            raise ValueError('masked_depth.runtime_env (host overlay env.sh for the rclpy '
                             'publisher subprocess) is required when enabled')
        masked_depth = {'enabled': True, 'rig_prefixes': tuple(prefixes),
                        'frame_id': str(md.get('frame_id', 'restock_camera_arm')),
                        'domain': int(md.get('domain', 88)), 'runtime_env': runtime_env,
                        'python': str(md.get('python', 'python3'))}
    collect_source_location = config.get('collect_source_location', '')
    if collect_source_location and not isinstance(collect_source_location, str):
        raise ValueError('collect_source_location must be a mission-location string')
    return ManipConfig(profiles=profiles, stations=stations, shelfbot_root=shelfbot_root,
                       walls=walls, voxel_dims=tuple(float(v) for v in vd), voxel_size=float(vs),
                       grasp_mode=next(iter(modes)), pose_bridge_root=pose_bridge_root,
                       masked_depth=masked_depth, collect_source_location=collect_source_location)


class _ArticulationParked:
    """Base stationarity read from the real articulation. The host owns arm motion on
    the cuMotion path, so the Isaac side only certifies the base is parked (the view the
    station sensing reads); it never commands or teleports a joint."""
    def __init__(self, articulation):
        self._robot = articulation

    def parked(self):
        import numpy as np
        linear = np.asarray(self._robot.get_linear_velocity(), float)
        angular = np.asarray(self._robot.get_angular_velocity(), float)
        return bool(np.isfinite(linear).all() and np.isfinite(angular).all()
                    and np.linalg.norm(linear) < .005 and np.linalg.norm(angular) < .005)


class HostManipulationRuntime:
    """Isaac-side manipulation on the landed host cuMotion path.

    Wraps a `twin.restock.ros_manipulation.RosManipulationRuntime`, which bridges the
    real arm-camera packets to the host MoveGroup/cuMotion planner, TransferController
    (CAD side-grip + support-contact gates) and trajectory_host over domain-88 ROS; the
    Isaac process runs no in-process planner (no cuRobo). It also exposes the `.motion`
    (base parked) and `.controller.profiles` views the integration sensing lane reads.
    """
    cart_location = 'rollwagen'

    def __init__(self, *, runtime, sensors, motion, profiles, collect_bridge=None):
        from types import SimpleNamespace
        self._runtime = runtime
        self._sensors = sensors
        self._collect_bridge = collect_bridge
        self.motion = motion
        self.controller = SimpleNamespace(cart_location='rollwagen', profiles=profiles)

    def set_source_target_provider(self, provider):
        # Hand the fixed-source collect-target provider (built by the camera lane's build_sensing,
        # available only AFTER manip_factory) to the collect bridge, if this runtime uses one.
        bridge = getattr(self, '_collect_bridge', None)
        if bridge is not None:
            bridge.provider = provider

    def configure_station(self, *, slots, base_from_camera):
        # Re-point the arm camera's station sensing before the host plans the next grasp.
        import numpy as np
        self._sensors.slots = tuple(slots)
        self._sensors.transform = np.asarray(base_from_camera, float)
        self._sensors.last_frame = None

    def start(self, operation, job):
        return self._runtime.start(operation, job)

    def step(self, dt, simulation_time, parked=None):
        return self._runtime.step(dt, simulation_time, parked=parked)

    def stop(self):
        return self._runtime.stop()


class CollectTargetBridge:
    """Collect pick target from the FIXED source camera, reprojected for the moving arm.

    The grasp target no longer comes from the arm eye-in-hand capture. Each sensing tick,
    ``capture(now, source, destination)`` calls the fixed-source ``source_target_provider``
    (set after ``build_sensing`` via ``HostManipulationRuntime.set_source_target_provider``),
    re-validates freshness + parked at the moment of use, builds the host ``Sensors`` packet
    from the sensed ``SourceTarget`` (sku + base-frame pose), and reprojects that pose into the
    LIVE arm image as the masked-depth/nvblox target box. It exposes ``sensed_target_bboxes``
    (the only field the ``ArmDepthSubprocessBridge`` reads). Returns ``None`` when nothing is
    sensed or the capture is stale — never fabricates; the masked bridge then republishes its
    last good box so the host's 0.5 s bbox freshness holds.
    """
    def __init__(self, *, stage, arm_camera, arm_camera_path, base_path, profiles,
                 motion, contact, joints, source_location):
        self.stage, self.arm_camera, self.arm_camera_path = stage, arm_camera, arm_camera_path
        self.base_path, self.profiles = base_path, profiles
        self.motion, self.contact, self.joints = motion, contact, joints
        self.source_location = source_location
        self.provider = None
        self.sensed_target_bboxes = ()

    def _arm_base_from_camera(self):
        import numpy as np
        from pxr import UsdGeom
        cache = UsdGeom.XformCache()
        cam = np.asarray(cache.GetLocalToWorldTransform(self.stage.GetPrimAtPath(self.arm_camera_path))).T
        base = np.asarray(cache.GetLocalToWorldTransform(self.stage.GetPrimAtPath(self.base_path))).T
        # USD camera looks -Z/+Y-up; flip to the +Z/+Y-down optical convention of get_intrinsics_matrix.
        return np.linalg.inv(base) @ (cam @ np.diag([1., -1., -1., 1.]))

    def _reproject(self, target):
        import numpy as np
        from twin.restock.collect_target import reproject_carton_roi
        profile = self.profiles.get(target.sku)
        if profile is None:
            return None
        frame = self.arm_camera.get_current_frame()
        depth = frame.get('distance_to_image_plane') if isinstance(frame, dict) else None
        if depth is None:
            return None
        m = np.asarray(self.arm_camera.get_intrinsics_matrix(), float)
        intr = (m[0, 0], m[1, 1], m[0, 2], m[1, 2])
        return reproject_carton_roi(target.pose_station, profile.dimensions,
                                    self._arm_base_from_camera(), intr, np.asarray(depth).shape)

    def capture(self, now, source, destination):
        from twin.restock_manipulation.types import Sensors, Observation
        self.sensed_target_bboxes = ()
        self._calls = getattr(self, '_calls', 0) + 1
        if self.provider is None:
            if self._calls % 30 == 1:
                print(f'COLLECT_CAPTURE calls={self._calls} provider=None', flush=True)
            return None
        try:
            target = self.provider(now)             # SourceTarget | None (provider enforces its own gates)
        except Exception as exc:                     # a swallowed source-capture failure (decode/freshness/...)
            if self._calls % 10 == 1:
                print(f'COLLECT_CAPTURE calls={self._calls} provider_error={type(exc).__name__}:{exc}', flush=True)
            return None
        if target is None:
            if self._calls % 10 == 1:
                print(f'COLLECT_CAPTURE calls={self._calls} target=None parked={self.motion.parked()}', flush=True)
            return None
        parked = bool(self.motion.parked())
        fresh = -0.02 <= now - target.stamp <= .5
        if not parked or not fresh:
            print(f'COLLECT_CAPTURE calls={self._calls} sku={target.sku} REJECT parked={parked} '
                  f'fresh={fresh} now={now:.3f} stamp={target.stamp:.3f}', flush=True)
            return None
        roi = self._reproject(target)
        print(f'COLLECT_CAPTURE calls={self._calls} sku={target.sku} roi={roi} '
              f'pose={tuple(round(v,3) for v in target.pose_station)}', flush=True)
        if roi is not None:
            self.sensed_target_bboxes = (dict(roi=roi, location=self.source_location),)
        try:
            joints = tuple(float(j) for j in self.joints())
        except Exception:
            joints = (0.,) * 6
        obs = Observation(observation_id=f'source:{target.stamp}', source='sensor',
                          simulation_time=float(target.stamp), sku=target.sku,
                          location=self.source_location, pose=tuple(target.pose_station),
                          confidence=1.0, frame='ur_base_link')
        return Sensors(now, (obs,), None, parked, bool(self.contact.read()), now,
                       float(self.motion.collision_time), joints, (), float(target.stamp))


def _build_runtime(*, app, assembly, params, output):
    """Real Isaac construction of the host cuMotion manipulation path.

    Builds the arm camera + station sensing capture, the native Isaac ROS graph that
    bridges to the host planner/controller, and a RosManipulationRuntime over them. No
    in-process cuRobo planner is constructed. Only runs inside the live app; the CPU
    contract tests inject `_builder`.
    """
    import numpy as np
    from pxr import Gf, UsdGeom
    from isaacsim.sensors.camera import Camera
    from twin.restock.ros_manipulation import RosManipulationRuntime
    from .cad_tool import NativePanelContact
    from .isaac_ros_graph import IsaacRosGraph, IsaacMeasuredFeedback
    from .isaac_sensors import IsaacCartonSensors

    paths = assembly.paths
    stage = assembly.stage
    # front_camera_frame is a URDF body frame (+X forward, +Z up), not a Camera prim.
    # Author a child USD Camera (which looks -Z with +Y up) oriented to look forward
    # along the frame's +X with +Z up; Camera() then adopts this existing prim.
    camera_path = paths['front_camera_frame'] + '/arm_camera'
    if not stage.GetPrimAtPath(camera_path):
        cam = UsdGeom.Camera.Define(stage, camera_path)
        cam.AddTransformOp().Set(Gf.Matrix4d(0, -1, 0, 0, 0, 0, 1, 0, -1, 0, 0, 0, 0, 0, 0, 1))
    camera = Camera(prim_path=camera_path)
    camera.initialize()
    # The masked-depth bridge needs per-pixel rig identity to invalidate the robot/cart
    # self-returns; IsaacCartonSensors adds distance-to-image-plane, this adds the instance-ID
    # segmentation. Exact prim IDs only mark known rig pixels unknown; never product identity.
    if params.masked_depth.get('enabled'):
        camera.add_instance_id_segmentation_to_frame(init_params={'colorize': False})

    gripper_path = '/RestockPanelGripper'
    contact = NativePanelContact(stage, gripper_path)
    source = params.stations['source']
    if not params.pose_bridge_root:
        raise ValueError('runtime-config pose_bridge_root (tray-perception root) required')
    # Arm eye-in-hand camera is masked depth + fine alignment ONLY now; IsaacCartonSensors is
    # retained for its distance-to-image-plane annotator (the masked-depth bridge needs it) and
    # fine-alignment role, not as the grasp-target capture.
    sensors = IsaacCartonSensors(camera, pose_bridge_root=params.pose_bridge_root,
                                 profiles=params.profiles, slots=source['slots'],
                                 base_from_camera=np.asarray(source['base_from_camera'], float))
    # The full measured feedback (collision_time + joint_positions + observe_source + parked) the
    # sensing capture needs; collision_time is synced each step from the host mapper-incorporation
    # ack via IsaacRosGraph.refresh_incorporation (see RosManipulationRuntime.step). Replaces the
    # parked-only adapter that could not satisfy IsaacCartonSensors.capture's collision_time gate.
    class _PortParkedFeedback(IsaacMeasuredFeedback):
        # parked() reflects the port's robust net-pose-displacement parking (set each step via
        # set_parked), not the instantaneous articulation velocity, which flaps on sub-mm/rad
        # angular noise and would fatally trip StationClient's 'lost measured parking' gate.
        _port_parked = None
        def set_parked(self, value):
            self._port_parked = None if value is None else bool(value)
        def parked(self):
            return super().parked() if self._port_parked is None else self._port_parked
    feedback = _PortParkedFeedback(assembly.robot)

    # The collect grasp target comes from the FIXED source camera via the camera-lane provider,
    # reprojected into the live arm image. The provider is injected after build_sensing through
    # HostManipulationRuntime.set_source_target_provider.
    collect_bridge = CollectTargetBridge(
        stage=stage, arm_camera=camera, arm_camera_path=camera_path,
        base_path=assembly.paths['ur_base_link'], profiles=params.profiles,
        motion=feedback, contact=contact,
        joints=feedback.joint_positions,
        source_location=params.collect_source_location)

    sensing_bridge = None
    if params.masked_depth.get('enabled'):
        sensing_bridge = _build_masked_depth_bridge(
            stage=stage, camera=camera, camera_path=camera_path, gripper_path=gripper_path,
            assembly=assembly, sensors=collect_bridge, settings=params.masked_depth)

    # Native ROS bridge to the host MoveGroup/cuMotion + TransferController + trajectory_host.
    graph = IsaacRosGraph(contact=contact, hold=assembly.command_gate.hold)
    # Parked-station client to the host nvblox mapper-incorporation ack
    # (twin.restock_host.incorporation_ack, co-launched with the stack): drives prepare/heartbeat
    # on /restock_sim/station/command and reads collision_time from /restock_sim/station/status.
    # RosManipulationRuntime mirrors its collision_time onto the feedback each step. Only with the
    # masked-depth ESDF path enabled (the ack integrates that masked depth).
    station = None
    station_frame = params.masked_depth.get('frame_id', 'restock_camera_arm')
    if params.masked_depth.get('enabled'):
        from twin.restock.native_channel import JsonChannel, StationClient
        channel = JsonChannel(root='/RestockStationChannel',
                              publish_topic='/restock_sim/station/command',
                              subscribe_topic='/restock_sim/station/status')
        station = StationClient(channel)
    runtime = RosManipulationRuntime(graph, collect_bridge.capture,
                                     prepare_station=lambda operation, job: None,
                                     sensing_bridge=sensing_bridge, feedback=feedback,
                                     station=station, station_camera_frame=station_frame)
    return HostManipulationRuntime(runtime=runtime, sensors=sensors,
                                   motion=feedback, profiles=params.profiles,
                                   collect_bridge=collect_bridge)


def _rig_root(path):
    """Top-level USD prim of a path, e.g. '/mobile_ur10_rollwagen/ur_base_link' -> '/mobile_ur10_rollwagen'."""
    parts = str(path).strip('/').split('/')
    return '/' + parts[0] if parts and parts[0] else str(path)


def _build_masked_depth_bridge(*, stage, camera, camera_path, gripper_path,
                               assembly, sensors, settings):
    """Wire the arm eye-in-hand masked-depth/info/bbox OmniGraph publishers for host nvblox.

    Rig prefixes (robot + cart + gripper subtrees whose self-pixels are invalidated) come from
    the runtime-config when given, else are derived from the composed assembly roots. TF for the
    optical frame is published by the existing native_transport /tf tree (an optical child prim
    is added to it inside the bridge); publishing uses OmniGraph nodes, never rclpy.
    """
    from twin.restock.masked_depth import ArmDepthSubprocessBridge

    prefixes = settings.get('rig_prefixes')
    if not prefixes:
        cart_root = assembly.evidence['composition']['assets']['cart']['root']
        prefixes = tuple(dict.fromkeys(
            [_rig_root(assembly.paths['articulation']), _rig_root(cart_root), gripper_path]))
    return ArmDepthSubprocessBridge(
        camera=camera, camera_path=camera_path, stage=stage, transport=assembly.transport,
        sensors=sensors, rig_prefixes=prefixes,
        frame_id=settings.get('frame_id', 'restock_camera_arm'),
        domain=int(settings.get('domain', 88)),
        runtime_env=settings['runtime_env'], python=settings.get('python', 'python3'))


def create(*, app, assembly, plan, config, output, _builder=None):
    """Assemble the manipulation runtime and return a configured ManipulationBridge."""
    from twin.restock.runtime import ManipulationBridge

    params = parse_manipulation_config(config, plan.job_sku)
    build = _builder or _build_runtime
    runtime = build(app=app, assembly=assembly, params=params, output=output)

    def prepare_station(operation, job):
        station = ACTION_STATION.get(operation)
        if station is None:
            raise ValueError(f'manipulation does not support operation {operation!r}')
        st = params.stations[station]
        runtime.configure_station(slots=st['slots'], base_from_camera=_matrix(st['base_from_camera']))

    return ManipulationBridge(runtime, owns_sensor_io=True, prepare_station=prepare_station)


def _matrix(rows):
    try:
        import numpy as np
        return np.asarray(rows, float)
    except Exception:
        return rows
