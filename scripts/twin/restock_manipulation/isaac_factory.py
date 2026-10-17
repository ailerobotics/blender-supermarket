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
    cart_placement: dict = field(default_factory=dict)  # collect destination (rollwagen) placement, derived from the cart asset
    arm_camera_reaim: dict = field(default_factory=dict)  # {origin, aim_targets} in ur_base for the look-at mount (camera lane PR64)
    sim_attach_fallback: bool = False  # labelled sim attach-fallback (cad_tool); default OFF, see SIM_ATTACH_CONTACT_MM
    station_heartbeat_s: float = 10.0  # StationClient wall heartbeat-unavailable bound; real-rig 10 s, sim raises it


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
    cart_placement = _parse_cart_placement(mcfg.get('cart_placement'))
    arm_camera_reaim = _parse_arm_camera_reaim(mcfg.get('arm_camera_reaim'))
    return ManipConfig(profiles=profiles, stations=stations, shelfbot_root=shelfbot_root,
                       walls=walls, voxel_dims=tuple(float(v) for v in vd), voxel_size=float(vs),
                       grasp_mode=next(iter(modes)), pose_bridge_root=pose_bridge_root,
                       masked_depth=masked_depth, collect_source_location=collect_source_location,
                       cart_placement=cart_placement, arm_camera_reaim=arm_camera_reaim,
                       sim_attach_fallback=bool(config.get('sim_attach_fallback', False)),
                       station_heartbeat_s=float(config.get('station_heartbeat_s', 10.0)))


def _parse_arm_camera_reaim(entry):
    """Validate the arm-camera look-at re-aim spec (camera lane PR64) into a dict, or {}.

    `origin` is the mount frame origin in ur_base_link (front_camera_frame is rigid on base_link,
    so this is pose-independent); `aim_targets` is one or more (x,y,z) points in ur_base_link that
    the single fixed mount aims at (their centroid) -- the collect bin and transfer slot are only
    ~8 deg apart, so one re-aim frames both stages. Absent -> {} -> keep the run-75 forward mount.
    """
    if entry is None:
        return {}
    if not isinstance(entry, dict):
        raise ValueError('arm_camera_reaim must be an object')
    try:
        origin = tuple(float(v) for v in entry['origin'])
        targets = [tuple(float(v) for v in p) for p in entry['aim_targets']]
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f'arm_camera_reaim needs origin (3) and aim_targets (list of >=3-vectors): {exc}')
    if len(origin) != 3 or not targets or any(len(p) < 3 for p in targets):
        raise ValueError('arm_camera_reaim origin must be 3 and aim_targets a non-empty list of >=3-vectors')
    return {'origin': origin, 'aim_targets': [p[:3] for p in targets]}


def _parse_cart_placement(entry):
    """Validate the collect destination (rollwagen) placement spec into a plain dict, or {}.

    The carton is placed onto the robot's own staged cart; the Target POSE is derived at capture
    time from the cart asset's MEASURED staged pose (read from the stage) plus these scenario
    constants -- the cart is a controlled asset, so this is not a fabricated inventory observation.
    `load_center_body_xy` is a cart-body-frame xy offset to the load centre; `support_z_world` is
    the world z of the load surface; `free_dimensions` is the usable load volume (>= carton+margin).
    """
    if entry is None:
        return {}
    if not isinstance(entry, dict):
        raise ValueError('cart_placement must be an object')
    try:
        location = entry['location']
        lc = tuple(float(v) for v in entry['load_center_body_xy'])
        sz = float(entry['support_z_world'])
        fd = tuple(float(v) for v in entry['free_dimensions'])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f'cart_placement needs location/load_center_body_xy/support_z_world/free_dimensions: {exc}')
    if not isinstance(location, str) or not location or len(lc) != 2 or len(fd) != 3:
        raise ValueError('cart_placement location (str), load_center_body_xy (2), free_dimensions (3)')
    return {'location': location, 'load_center_body_xy': lc, 'support_z_world': sz, 'free_dimensions': fd}


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

    def set_shelf_target_provider(self, provider):
        # Hand the fixed-shelf transfer-destination provider (camera lane PR62) to the bridge for
        # the transfer stage (cart source -> shelf slot). Collect is unaffected.
        bridge = getattr(self, '_collect_bridge', None)
        if bridge is not None and hasattr(bridge, 'set_shelf_target_provider'):
            bridge.set_shelf_target_provider(provider)

    def configure_station(self, *, slots, base_from_camera):
        # Re-point the arm camera's station sensing before the host plans the next grasp.
        import numpy as np
        self._sensors.slots = tuple(slots)
        self._sensors.transform = np.asarray(base_from_camera, float)
        self._sensors.last_frame = None

    def start(self, operation, job):
        return self._runtime.start(operation, job)

    def warm(self, simulation_time, parked=None):
        # Approach/parked warm-up: build the host ESDF + drive collision_time fresh before the
        # collect job is dispatched. Delegates to the inner RosManipulationRuntime.
        return self._runtime.warm(simulation_time, parked=parked)

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
                 motion, contact, joints, source_location, cart_placement=None, cart_root=None,
                 mobile_base_path=None):
        self.stage, self.arm_camera, self.arm_camera_path = stage, arm_camera, arm_camera_path
        self.base_path, self.profiles = base_path, profiles
        # The arm executes relative to the MOBILE chassis (base_link), which carries the dock yaw
        # (run-186: chassis -6.1deg while the USD ur_base_link prim reads ~0); the executed pad yaw
        # tracks the chassis, so the carton grasp-yaw correction must re-express through the chassis
        # pose, NOT ur_base_link. base_path (arm base) stays for the eye-in-hand ROI reproject.
        self.mobile_base_path = mobile_base_path or base_path
        self.motion, self.contact, self.joints = motion, contact, joints
        self.source_location = source_location
        # Collect destination: place the carton onto the robot's own staged (not-yet-coupled) cart.
        # The placement Target POSE is derived at capture from the cart asset's MEASURED staged pose
        # (read live from the stage at cart_root) + the scenario load-centre/support constants. The
        # cart is a controlled asset, so this is not a fabricated inventory observation.
        self.cart_placement = dict(cart_placement or {})
        self.cart_root = cart_root
        self.provider = None
        # Transfer stage (cart is the SOURCE, shelf the DESTINATION): the shelf slot target comes
        # from the camera lane's fixed-shelf provider (PR62, set after build_sensing); the carton on
        # the cart is derived from the same cart asset pose as the collect placement (controlled
        # asset). Collect leaves this None and is unaffected.
        self.shelf_provider = None
        self.sensed_target_bboxes = ()

    def set_shelf_target_provider(self, provider):
        self.shelf_provider = provider

    def _arm_base_from_camera(self):
        import numpy as np
        from pxr import UsdGeom
        cache = UsdGeom.XformCache()
        cam = np.asarray(cache.GetLocalToWorldTransform(self.stage.GetPrimAtPath(self.arm_camera_path))).T
        base = np.asarray(cache.GetLocalToWorldTransform(self.stage.GetPrimAtPath(self.base_path))).T
        # USD camera looks -Z/+Y-up; flip to the +Z/+Y-down optical convention of get_intrinsics_matrix.
        return np.linalg.inv(base) @ (cam @ np.diag([1., -1., -1., 1.]))

    def _map_pose_to_base(self, pose):
        # Re-express a MAP-frame carton pose into the CURRENT (live) arm-base frame. source_target's
        # pose_station is in 'map' (fixed world), but the host grasp consumes the Observation as
        # 'ur_base_link'. When the base parks YAWED at the dock (run-182/183: world yaw -5deg; the
        # SOURCE_YAW_DIAG obs_yaw_base=+179.94 is the MAP marker yaw, not base-relative), passing the
        # map pose as the base pose drops the dock yaw -> the grasp is off by exactly it (4.99deg).
        # Read base_from_map LIVE off ur_base_link so observation and execution share the same base
        # at grasp time (firstmate finding). Returns (x,y,z, qx,qy,qz,qw) in the live base frame.
        import os, numpy as np
        try:
            from pxr import UsdGeom
        except ImportError:
            # CPU unit tests run capture() with no USD runtime; the map->base yaw re-expression is a
            # GPU-runtime concern, so pass the pose through unchanged there. The real run always has
            # pxr and performs the transform; a genuine stage-read failure below still raises.
            return tuple(float(v) for v in pose)
        cache = UsdGeom.XformCache()
        base = np.asarray(cache.GetLocalToWorldTransform(self.stage.GetPrimAtPath(self.mobile_base_path))).T
        bfm = np.linalg.inv(base)                       # chassis base_from_map (4x4)
        x, y, z, qx, qy, qz, qw = (float(v) for v in pose)
        # Keep the position (prior runs reach the carton at gap 0; the ROI/approach already use it);
        # only the ORIENTATION carries the dropped dock yaw. bfm's rotation is Rz(-base_yaw), so
        # Rb = Rz(-base_yaw) @ R re-expresses the carton yaw into the live base (180 -> 185 at b=-5).
        pos = np.array([x, y, z])
        n = math.sqrt(qx*qx + qy*qy + qz*qz + qw*qw) or 1.0
        qx, qy, qz, qw = qx/n, qy/n, qz/n, qw/n
        R = np.array([[1-2*(qy*qy+qz*qz), 2*(qx*qy-qz*qw), 2*(qx*qz+qy*qw)],
                      [2*(qx*qy+qz*qw), 1-2*(qx*qx+qz*qz), 2*(qy*qz-qx*qw)],
                      [2*(qx*qz-qy*qw), 2*(qy*qz+qx*qw), 1-2*(qx*qx+qy*qy)]])
        Rb = bfm[:3, :3] @ R
        t = float(np.trace(Rb))
        if t > 0:
            s = math.sqrt(t+1.0)*2; w2 = 0.25*s
            x2 = (Rb[2,1]-Rb[1,2])/s; y2 = (Rb[0,2]-Rb[2,0])/s; z2 = (Rb[1,0]-Rb[0,1])/s
        else:
            i = int(np.argmax([Rb[0,0], Rb[1,1], Rb[2,2]]))
            if i == 0:
                s = math.sqrt(1.0+Rb[0,0]-Rb[1,1]-Rb[2,2])*2
                w2 = (Rb[2,1]-Rb[1,2])/s; x2 = 0.25*s; y2 = (Rb[0,1]+Rb[1,0])/s; z2 = (Rb[0,2]+Rb[2,0])/s
            elif i == 1:
                s = math.sqrt(1.0-Rb[0,0]+Rb[1,1]-Rb[2,2])*2
                w2 = (Rb[0,2]-Rb[2,0])/s; x2 = (Rb[0,1]+Rb[1,0])/s; y2 = 0.25*s; z2 = (Rb[1,2]+Rb[2,1])/s
            else:
                s = math.sqrt(1.0-Rb[0,0]-Rb[1,1]+Rb[2,2])*2
                w2 = (Rb[1,0]-Rb[0,1])/s; x2 = (Rb[0,2]+Rb[2,0])/s; y2 = (Rb[1,2]+Rb[2,1])/s; z2 = 0.25*s
        out = (float(pos[0]), float(pos[1]), float(pos[2]), float(x2), float(y2), float(z2), float(w2))
        if os.environ.get('RESTOCK_SOURCE_YAW_DIAG', '0') != '0':
            print(f'MAP2BASE_YAW map_yaw={math.degrees(2*math.atan2(qz,qw)):.3f} '
                  f'base_yaw={math.degrees(2*math.atan2(out[5],out[6])):.3f}', flush=True)
        return out

    def _reproject(self, target):
        import numpy as np
        from twin.restock.collect_target import reproject_carton_roi
        self._roi_reason = None
        profile = self.profiles.get(target.sku)
        if profile is None:
            self._roi_reason = f'no-profile sku={target.sku!r} keys={sorted(self.profiles)}'
            return None
        frame = self.arm_camera.get_current_frame()
        depth = frame.get('distance_to_image_plane') if isinstance(frame, dict) else None
        if depth is None:
            self._roi_reason = f'no-depth frame_keys={sorted(frame) if isinstance(frame, dict) else type(frame).__name__}'
            return None
        m = np.asarray(self.arm_camera.get_intrinsics_matrix(), float)
        intr = (m[0, 0], m[1, 1], m[0, 2], m[1, 2])
        b2c = self._arm_base_from_camera()
        roi = reproject_carton_roi(target.pose_station, profile.dimensions,
                                   b2c, intr, np.asarray(depth).shape)
        if roi is None:
            # distinguish "behind camera" from "off image": re-run the projection summary for the
            # carton origin, reporting its camera-frame z and (if in front) its pixel coordinate.
            pose = np.asarray(target.pose_station, float)
            c2b = np.linalg.inv(np.asarray(b2c, float))
            pc = pose[:3] @ c2b[:3, :3].T + c2b[:3, 3]  # carton origin in camera frame
            h, w = np.asarray(depth).shape[:2]
            if pc[2] > 1e-6:
                u = pc[0] / pc[2] * intr[0] + intr[2]
                v = pc[1] / pc[2] * intr[1] + intr[3]
                where = f'off-image uv=({u:.0f},{v:.0f}) img=({w}x{h})'
            else:
                where = 'behind-camera'
            self._roi_reason = (f'reproject-None {where} carton_in_cam=({pc[0]:.3f},{pc[1]:.3f},{pc[2]:.3f}) '
                                f'intr=({intr[0]:.0f},{intr[1]:.0f},{intr[2]:.0f},{intr[3]:.0f})')
        return roi

    def _cart_target(self, now, src):
        """Derive the collect placement Target on the staged cart (controlled asset, not sensed stock).

        POSE = the cart's MEASURED staged world pose (read live at cart_root) with the scenario
        body-frame load-centre offset and world support z, expressed in the ur_base_link frame; the
        placement keeps the carried carton's yaw so the side-grip wall is unchanged pick->place.
        Returns None when no cart placement is configured (backward-compatible / non-collect paths).
        """
        cp = self.cart_placement
        pose = self._cart_pose_station()
        if pose is None:                          # no cart placement -> no destination target
            return None
        import math
        from twin.restock_manipulation.types import Target
        (x, y, z), _ = pose
        q = tuple(float(v) for v in src.pose_station[3:])             # carry-through carton yaw/quat
        return Target(observation_id=f'cart:{cp["location"]}:{now:.3f}', source='sensor',
                      simulation_time=float(now), location=cp['location'],
                      pose=(x, y, z, *q), clear=True,
                      free_dimensions=tuple(cp['free_dimensions']), frame='ur_base_link')

    def _cart_pose_station(self):
        """The cart load-centre as ((x,y,z) in ur_base_link, yaw-relative-to-ur_base), from the cart
        asset's live measured pose + scenario load-centre/support (controlled asset, not sensed
        stock). Shared by the collect placement Target and the transfer cart-source Observation.
        Returns None when no cart placement is configured (CPU-safe: no pxr on that path)."""
        cp = self.cart_placement
        if not cp or not self.cart_root:
            return None
        import numpy as np, math
        from pxr import UsdGeom
        if not self.stage.GetPrimAtPath(self.cart_root):
            return None
        cache = UsdGeom.XformCache()
        Tc = np.asarray(cache.GetLocalToWorldTransform(self.stage.GetPrimAtPath(self.cart_root))).T
        Tub = np.asarray(cache.GetLocalToWorldTransform(self.stage.GetPrimAtPath(self.base_path))).T
        cyaw = math.atan2(float(Tc[1, 0]), float(Tc[0, 0]))          # cart yaw in world
        lx, ly = cp['load_center_body_xy']
        world = np.array([float(Tc[0, 3]) + math.cos(cyaw) * lx - math.sin(cyaw) * ly,
                          float(Tc[1, 3]) + math.sin(cyaw) * lx + math.cos(cyaw) * ly,
                          float(cp['support_z_world']), 1.0])
        p = np.linalg.inv(Tub) @ world                                # load centre in ur_base_link
        ub_yaw = math.atan2(float(Tub[1, 0]), float(Tub[0, 0]))
        return (float(p[0]), float(p[1]), float(p[2])), float(cyaw - ub_yaw)

    def _capture_transfer(self, now):
        """Transfer sensing: the carton on the cart (derived cart pose, controlled asset) as the
        SOURCE observation + the camera-lane fixed-shelf slot as the DESTINATION target. Mirrors
        collect; never fabricates -- returns None until the base is parked and the shelf provider
        senses a fresh slot. NOTE: assumes the shelf provider's pose_station is ur_base_link (as the
        collect source provider's is); validate when the pipeline first reaches transfer."""
        from twin.restock_manipulation.types import Sensors, Observation, Target
        import math
        if self.shelf_provider is None:
            if self._calls % 30 == 1:
                print('TRANSFER_CAPTURE shelf_provider=None', flush=True)
            return None
        parked = bool(self.motion.parked())
        cart = self._cart_pose_station()
        if not parked or cart is None:
            if self._calls % 10 == 1:
                print(f'TRANSFER_CAPTURE parked={parked} cart={cart is not None}', flush=True)
            return None
        try:
            slot = self.shelf_provider(now)      # ShelfSlotTarget | None (provider enforces its gates)
        except Exception as exc:
            if self._calls % 10 == 1:
                print(f'TRANSFER_CAPTURE shelf_error={type(exc).__name__}:{exc}', flush=True)
            return None
        if slot is None or not (-0.02 <= now - slot.stamp <= .5):
            if self._calls % 10 == 1:
                print(f'TRANSFER_CAPTURE slot={slot is not None} fresh?', flush=True)
            return None
        (cx, cy, cz), yaw = cart
        sku = next(iter(self.profiles)) if self.profiles else ''
        cq = (0., 0., math.sin(yaw / 2.), math.cos(yaw / 2.))
        obs = Observation(observation_id=f'cart:{now:.3f}', source='sensor', simulation_time=float(now),
                          sku=sku, location=self.cart_placement['location'], pose=(cx, cy, cz, *cq),
                          confidence=1.0, frame='ur_base_link')
        tgt = Target(observation_id=f'shelf:{slot.stamp}', source='sensor', simulation_time=float(now),
                     location=slot.location, pose=tuple(slot.pose_station), clear=bool(slot.clear),
                     free_dimensions=tuple(slot.free_dimensions), frame='ur_base_link')
        try:
            joints = tuple(float(j) for j in self.joints())
        except Exception:
            joints = (0.,) * 6
        print(f'TRANSFER_CAPTURE calls={self._calls} sku={sku} cart=({cx:.2f},{cy:.2f},{cz:.2f}) '
              f'shelf={slot.location} clear={slot.clear}', flush=True)
        return Sensors(now, (obs,), tgt, parked, bool(self.contact.read()), now,
                       float(self.motion.collision_time), joints, (), float(now))

    def capture(self, now, source, destination):
        from twin.restock_manipulation.types import Sensors, Observation
        self.sensed_target_bboxes = ()
        self._calls = getattr(self, '_calls', 0) + 1
        # TRANSFER (cart is the SOURCE, shelf the DESTINATION): the mirror of collect. Routed here
        # when the mission source is the cart; collect (source = bin) falls through unchanged.
        cart_loc = self.cart_placement.get('location')
        if cart_loc and source == cart_loc:
            return self._capture_transfer(now)
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
        reason = '' if roi is not None else f' roi_reason=[{getattr(self, "_roi_reason", None)}]'
        print(f'COLLECT_CAPTURE calls={self._calls} sku={target.sku} roi={roi} '
              f'pose={tuple(round(v,3) for v in target.pose_station)}{reason}', flush=True)
        if roi is not None:
            self.sensed_target_bboxes = (dict(roi=roi, location=self.source_location),)
        try:
            joints = tuple(float(j) for j in self.joints())
        except Exception:
            joints = (0.,) * 6
        # Stamp the observation with the packet `now` (not the source render stamp): the host
        # controller's _provenance uses a STRICT _fresh (0 <= now - stamp <= max_age), but the
        # source render may lead `now` by up to the capture's 0.02 s render-lead tolerance, so a
        # target.stamp slightly ahead of `now` would fail _fresh (now - stamp < 0) and drop the
        # match every tick (observe never advances). `now` is the packet clock the host compares
        # against, so _fresh(now, now) = 0 passes; observation_id keeps the render stamp for identity.
        obs = Observation(observation_id=f'source:{target.stamp}', source='sensor',
                          simulation_time=float(now), sku=target.sku,
                          location=self.source_location, pose=tuple(self._map_pose_to_base(target.pose_station)),
                          confidence=1.0, frame='ur_base_link')
        try:                                      # a cart-placement derivation failure must NOT
            cart_target = self._cart_target(now, target)   # starve /sensors; degrade to target=None
        except Exception as exc:                            # (diagnosable observe stall, not host death)
            cart_target = None
            if self._calls % 10 == 1:
                print(f'COLLECT_CAPTURE calls={self._calls} cart_target_error={type(exc).__name__}:{exc}',
                      flush=True)
        # Carry the source target's acquisition-to-use deadline onto the host packet: under the
        # sensing gate the held republish provides a FRESH sim-clock valid_until_steady, which the
        # contact/grasp suction guard (require_fresh) needs to enable the gripper (run-140). None
        # when not suspended -- the rendered collect packet carried no deadline before, so gate-off
        # is byte-identical.
        return Sensors(now, (obs,), cart_target, parked, bool(self.contact.read()), now,
                       float(self.motion.collision_time), joints, (), float(target.stamp),
                       target.valid_until_steady)


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
    from .isaac_ros_graph import IsaacRosGraph, IsaacMeasuredFeedback, TOPICS
    from .isaac_sensors import IsaacCartonSensors

    paths = assembly.paths
    stage = assembly.stage
    # front_camera_frame is a URDF body frame (+X forward, +Z up), not a Camera prim.
    # Author a child USD Camera (which looks -Z with +Y up) oriented to look forward
    # along the frame's +X with +Z up; Camera() then adopts this existing prim.
    camera_path = paths['front_camera_frame'] + '/arm_camera'
    # Re-aim the fixed mount at the restock targets (camera lane PR64, arm_mount). Without it the
    # mount looks forward (+X) while both the collect bin and transfer slot sit to base +Y, so every
    # reprojection put the carton behind the optical plane (roi=None). The re-aim also widens the
    # field to the collision intrinsics (424x240 @ 75 deg HFOV) so the 0.6 m carton fits (the 128x128
    # forward field clamped the ROI to full width). Absent config keeps the run-75 forward mount.
    reaim = params.arm_camera_reaim
    resolution = None
    if reaim:
        from twin.restock_camera.arm_mount import arm_camera_look_at_mount, collision_camera_intrinsics
        mount = arm_camera_look_at_mount(reaim['origin'], reaim['aim_targets'])
        cam_xform = Gf.Matrix4d(*[float(v) for row in mount.usd_transform_op for v in row])
        k = collision_camera_intrinsics()
        resolution = (k.width, k.height)
    else:
        cam_xform = Gf.Matrix4d(0, -1, 0, 0, 0, 0, 1, 0, -1, 0, 0, 0, 0, 0, 0, 1)
        k = None
    if not stage.GetPrimAtPath(camera_path):
        cam = UsdGeom.Camera.Define(stage, camera_path)
        cam.AddTransformOp().Set(cam_xform)
    else:
        prim = UsdGeom.Camera(stage.GetPrimAtPath(camera_path))
        ops = prim.GetOrderedXformOps()
        (ops[0] if ops else prim.AddTransformOp()).Set(cam_xform)
    camera = Camera(prim_path=camera_path, resolution=resolution) if resolution else Camera(prim_path=camera_path)
    camera.initialize()
    if k is not None:                                   # collision/masked-depth intrinsics (wide field)
        camera.set_focal_length(.02)
        camera.set_horizontal_aperture(k.width * .02 / k.fx)
    # The masked-depth bridge needs per-pixel rig identity to invalidate the robot/cart
    # self-returns; IsaacCartonSensors adds distance-to-image-plane, this adds the instance-ID
    # segmentation. Exact prim IDs only mark known rig pixels unknown; never product identity.
    if params.masked_depth.get('enabled'):
        camera.add_instance_id_segmentation_to_frame(init_params={'colorize': False})

    gripper_path = '/RestockPanelGripper'
    contact = NativePanelContact(stage, gripper_path, sim_attach_fallback=params.sim_attach_fallback)
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
    # The collect PLACEMENT target (destination = staged cart) is derived from the cart asset's
    # measured pose; the cart's root prim comes from the composition evidence when configured.
    cart_root = None
    if params.cart_placement:
        cart_root = assembly.evidence['composition']['assets']['cart']['root']
    collect_bridge = CollectTargetBridge(
        stage=stage, arm_camera=camera, arm_camera_path=camera_path,
        base_path=assembly.paths['ur_base_link'], profiles=params.profiles,
        mobile_base_path=assembly.paths.get('base_link', assembly.paths['ur_base_link']),
        motion=feedback, contact=contact,
        joints=feedback.joint_positions,
        source_location=params.collect_source_location,
        cart_placement=params.cart_placement, cart_root=cart_root)

    sensing_bridge = None
    if params.masked_depth.get('enabled'):
        sensing_bridge = _build_masked_depth_bridge(
            stage=stage, camera=camera, camera_path=camera_path, gripper_path=gripper_path,
            assembly=assembly, sensors=collect_bridge, settings=params.masked_depth)

    # Isaac->host control-plane PUBLISHES (manipulation/job, sensors, station/command) go through
    # an out-of-Isaac host-DDS rclpy subprocess: the OmniGraph ROS2 publisher is invisible to the
    # host UDPv4 Fast DDS (run-59: job/command pub=0), so ros_host / the ack never received them.
    # The host->Isaac subscribers (status, stop, gripper, station/status) stay on OmniGraph.
    host_control = None
    if params.masked_depth.get('enabled'):
        from twin.restock.control_ipc import HostControlPublisher
        host_control = HostControlPublisher(
            topics=(TOPICS['job'], TOPICS['sensors'], '/restock_sim/station/command'),
            # The subprocess also SUBSCRIBES these and forwards them back over the socket, so the
            # StationClient (station/status) and IsaacRosGraph (manipulation/status) see host-side
            # latency instead of the slow OmniGraph subscribers that delivered the ack reply late.
            return_topics=('/restock_sim/station/status', TOPICS['status']),
            runtime_env=params.masked_depth['runtime_env'],
            python=params.masked_depth.get('python', 'python3'),
            domain=params.masked_depth.get('domain', 88))
    # Native ROS bridge to the host MoveGroup/cuMotion + TransferController + trajectory_host.
    graph = IsaacRosGraph(contact=contact, hold=assembly.command_gate.hold, host_control=host_control)
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
                              subscribe_topic='/restock_sim/station/status',
                              host_control=host_control)
        # Stamp station/command in the /clock (Isaac stage-time) domain the host ack validates
        # against -- read native_transport's last-published stage time -- NOT the mission clock
        # (Packet.simulation_time, zeroed at the first action), which the ack rejected as stale.
        # publish_lag past-dates the command stamp by ~6 clock periods (<< the ack's 0.5 s window)
        # so the ack -- whose /clock can lag the just-sampled stage time by ~one step when the fast
        # command outruns the matching /clock update -- never sees now() behind the stamp (run-66:
        # ack now()=0.20 vs stamp=0.233 tripped the strict 0<=now-stamp lower bound).
        station = StationClient(channel, stage_clock=lambda: assembly.transport.stage_time,
                                publish_lag=0.1, heartbeat_s=params.station_heartbeat_s)
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
