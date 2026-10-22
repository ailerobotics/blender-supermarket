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

from .transfer_carry import weld_attach as _weld_attach
from .transfer_mode import LIVE_POSE, transfer_mode
from . import ground_truth_collect as gtc
from .types import ProductProfile

STATIONS = ('source', 'shelf')
# Mission action -> arm station whose ROI/calibration the sensors use.
ACTION_STATION = {'collect': 'source', 'transfer': 'shelf'}


def _finite_matrix4(m):
    return (isinstance(m, (list, tuple)) and len(m) == 4
            and all(isinstance(r, (list, tuple)) and len(r) == 4
                    and all(isinstance(v, (int, float)) and math.isfinite(v) for v in r) for r in m))


def _wrap_angle(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def base_parked_by_displacement(poses, *, xy_tol=0.006, yaw_tol=0.02):
    """fm411 (run-317): base stationarity from MEASURED NET POSE DISPLACEMENT over a short window, NOT the
    TGS-reported instantaneous velocity. The PhysX solver carries a documented velocity bias (~14 mm/s on a
    provably frozen body) and SPIKES the reported base velocity when a parking-brake FixedJoint is created
    at a phase transition (run-317: the grasp->lift yaw-lock->full-lock switch tripped the controller's
    `parked` top-of-step invariant and e-stopped collect while base_dxy was ~0). `poses` is a window of
    (x, y, yaw) in metres/rad; parked when the xy spread and the YAW-WRAPPED yaw spread across the window
    are within tol (yaw wrap matters: the collect base sits at yaw ~= pi, so a raw max-min straddling the
    +/-pi branch would falsely read huge). < 2 samples -> True (just-arrived base, no window yet); any
    non-finite reading -> False (a real fault, never silently parked). Pure for CPU testing; mirrors
    settle_metrics / isaac_port's displacement gate."""
    pts = list(poses)
    if any(p is None or not all(math.isfinite(v) for v in p) for p in pts):
        return False
    if len(pts) < 2:
        return True
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    xy_spread = math.hypot(max(xs) - min(xs), max(ys) - min(ys))
    y0 = pts[0][2]
    dyaws = [_wrap_angle(p[2] - y0) for p in pts]
    yaw_spread = max(dyaws) - min(dyaws)
    return xy_spread <= xy_tol and yaw_spread <= yaw_tol


def hold_last_good_collision(read, last_good):
    """fm415 (run-319): the host packet must NEVER carry the uninitialized collision_time default (-1).
    The station/status->feedback mirror momentarily reads -1 for one tick at the weld/attach ESDF
    re-incorporation (run-319 died collision_dt=31.6 = now-(-1), while the station /status kept publishing
    a FRESH collision_time). A read < 0 is that mirror gap, NOT a real measured staleness, so emit the last
    genuinely-measured incorporation time instead -- honest (the most recent REAL value, never fabricated;
    if it has since genuinely gone stale the controller's freshness gate still catches it). A read >= 0
    updates and passes through. Returns (emit, new_last_good). Pure for CPU testing."""
    if read >= 0:
        return read, read
    return (last_good if last_good is not None else read), last_good


def survive_collect_occlusion(enabled, attached, parked):
    """fm410 (run-316): decide whether a SOURCE-OCCLUDED collect capture should still emit a packet.

    Once the carton is ATTACHED (weld/native grip), source localisation is done -- the arm occluding the
    fixed source camera during grasp/hold is the normal case, not a fault. Returning None then nulls
    /sensors and freezes the host controller mid-grasp/lift (run-316 looped COLLECT_CAPTURE 2663x). When
    enabled + attached + the base is PARKED, emit an honest arm-feedback packet instead (empty source
    observations -- no fresh localisation is claimed -- with LIVE contact/placed_pose/joints and no
    fabricated acquisition deadline). A moving base (not parked) is a real fault and is never survived.
    Off (enabled False) => never survives => byte-identical. Pure for CPU testing."""
    return bool(enabled) and bool(attached) and bool(parked)


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
    station_stage_unavailable_s: float = 3.0  # StationClient sim-clock (stage) supervisor-liveness bound (primary)
    gt_place_slot: dict = field(default_factory=dict)  # collect-gt: warm-shortcut place slot (ground-truth collect only)


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
                       sim_attach_fallback=_weld_attach(config),
                       station_heartbeat_s=float(config.get('station_heartbeat_s', 10.0)),
                       station_stage_unavailable_s=float(config.get('station_stage_unavailable_s', 3.0)))


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
    station sensing reads); it never commands or teleports a joint.

    fm411 (run-317): gated on MEASURED NET POSE DISPLACEMENT over a short window, not the TGS instantaneous
    velocity (`get_linear_velocity`), which spikes when the parking-brake FixedJoint is (re)created at a
    phase transition and false-tripped the controller's `parked` invariant mid-collect. See
    base_parked_by_displacement."""
    def __init__(self, articulation, *, xy_tol=0.006, yaw_tol=0.02, window=10):
        self._robot = articulation
        self._xy_tol, self._yaw_tol, self._window = float(xy_tol), float(yaw_tol), int(window)
        self._hist = []

    def parked(self):
        import numpy as np
        position, quaternion = self._robot.get_world_pose()  # Isaac quaternion is wxyz
        pos = np.asarray(position, float)
        quat = np.asarray(quaternion, float)
        if pos.shape != (3,) or quat.shape != (4,) or not np.isfinite(pos).all() or not np.isfinite(quat).all():
            return False
        w, x, y, z = quat
        yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
        self._hist.append((float(pos[0]), float(pos[1]), float(yaw)))
        if len(self._hist) > self._window:
            self._hist = self._hist[-self._window:]
        return base_parked_by_displacement(self._hist, xy_tol=self._xy_tol, yaw_tol=self._yaw_tol)


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

    def pause_station(self, simulation_time):
        # fm399/fm400: cleanly stop the parked station at reposition drive-start. This wrapper is what
        # the mission's ManipulationBridge probes with hasattr -- delegate so it is NOT a silent no-op
        # (the run-310 failure: the station stayed active through the drive and faulted -> emergency stop).
        return self._runtime.pause_station(simulation_time)

    def set_clock_sync(self, enabled=True):
        # Staged-start clock sync (twin.restock.staged_start): delegate to the inner runtime, which
        # stamps the host packets -- a wrapper-only setter would be a silent no-op (run-310 family).
        return self._runtime.set_clock_sync(enabled)

    def set_base_repositioned(self):
        # fm400: the base arrived at place_approach. Delegate so the inner runtime latches it onto the
        # published sensor packet, carrying arrival across the ROS boundary to the host controller's
        # TRANSIT gate (set_base_repositioned on this wrapper cannot reach the separate host process).
        return self._runtime.set_base_repositioned()

    @property
    def last_sensor_time(self):
        # fm400: the RuntimeAdapter's reach-stall watchdog probes getattr(core, 'last_sensor_time').
        # Expose the inner runtime's value so the watchdog is NOT silently inert on the mission path
        # (same silent-no-op family as the missing pause_station delegation).
        return self._runtime.last_sensor_time


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
                 mobile_base_path=None, cart_support_path=None, verify_place=False,
                 gt_place_slot=None):
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
        # fm389 deck-interior placement (opt-in via RESTOCK_PLACE_ON_DECK, plumbed from the composition's
        # stock_supports[0]): when set, the placement Target is the live support PRIM's measured world AABB
        # centre+top (the real deck interior, wall-clearance-centred) instead of the cart-origin +
        # load_center_body_xy offset at the scenario support_z_world constant -- which aimed the carton at
        # the UPPER shelf (z=0.9325), 0.71 m above the deck and 0.08 m east of centre (run-300e: carton
        # released off the east edge to the floor). None => legacy constants (byte-identical). Cached in
        # cart-root-local on first read (rigid w.r.t. the cart) so the per-tick cost stays one XformCache.
        self.cart_support_path = cart_support_path
        self._deck_local = None
        self._deck_extent = None
        # fm389 physical PLACE verification (opt-in via RESTOCK_VERIFY_PLACE): measure the carton's live
        # world pose (the fixed handling-unit prim) and thread its centre onto the host packet so the
        # collect `verify` phase can REFUSE an empty-gripper 'success' when the carton is not resting on
        # the deck. A controlled-asset ground-truth read for the PLACE physics gate only -- NOT inventory
        # (the mission `verify` stage still commits stock from sensor evidence). Off => placed_pose stays
        # None => byte-identical. Cached carton-local centre (rigid) keeps the per-tick cost at one XformCache.
        self.verify_place = bool(verify_place)
        self._handling_unit_path = '/RestockScenario/HandlingUnit'
        self._unit_local_center = None
        self.provider = None
        # Transfer stage (cart is the SOURCE, shelf the DESTINATION): the shelf slot target comes
        # from the camera lane's fixed-shelf provider (PR62, set after build_sensing); the carton on
        # the cart is derived from the same cart asset pose as the collect placement (controlled
        # asset). Collect leaves this None and is unaffected.
        self.shelf_provider = None
        self.sensed_target_bboxes = ()
        # Opt-in SIM transfer workaround (env RESTOCK_TRANSFER_MODE, see transfer_mode.py). 'live_pose': the
        # transfer SOURCE is the carton's LIVE top surface (handling-unit prim) instead of the deck-top AABB
        # centre. Unset => None => byte-identical. Logged at boot so the activation is visible in the console.
        self.transfer_mode = transfer_mode()
        if self.transfer_mode:
            print(f'TRANSFER_MODE {self.transfer_mode} isaac-capture: transfer source = live carton top pose '
                  f'({self._handling_unit_path})', flush=True)
        # GROUND-TRUTH COLLECT (collect-gt) (RESTOCK_COLLECT_GROUND_TRUTH=1, default off): serve the collect pick
        # from the controlled carton asset's simulated pose instead of the source-camera provider (which
        # refused every frame in cold runs 343-348, so /sensors never started). The source observation is
        # served only until the carton is first held; after that the packet keeps the cart target alive
        # (the place re-registration needs it) but claims no source localisation. Off => byte-identical.
        self.collect_ground_truth = gtc.enabled()
        self._gt_carton_held = False
        # The ONE carton spot (sim-video-transfer-fallback-v1 decision): the warm-shortcut slot from the
        # scenario initial_staging -- rollwagen UPPER shelf, cart body (+0.18, 0), yaw -90 deg cart-relative.
        # Ground-truth collect places there instead of the deck so cold and warm runs hand the transfer
        # the same carton pose. Empty => the existing _cart_target placement.
        self.gt_place_slot = dict(gt_place_slot or {})

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
        from pxr import Usd, UsdGeom
        cart_prim = self.stage.GetPrimAtPath(self.cart_root)
        if not cart_prim:
            return None
        cache = UsdGeom.XformCache()
        Tc = np.asarray(cache.GetLocalToWorldTransform(cart_prim)).T
        Tub = np.asarray(cache.GetLocalToWorldTransform(self.stage.GetPrimAtPath(self.base_path))).T
        cyaw = math.atan2(float(Tc[1, 0]), float(Tc[0, 0]))          # cart yaw in world
        if self.cart_support_path is not None:
            # DECK-interior load centre from the live support PRIM: its measured world AABB centre (xy) +
            # top (z). Cached in cart-root-local on first read (the deck is rigid w.r.t. the cart, so this
            # tracks the cart's live yaw+translation with one XformCache/tick instead of a per-tick mesh
            # BBox traversal). Centring on the deck AABB is the wall clearance (the deck footprint strictly
            # contains the carton -- validated by the controller's axis_fits gate on free_dimensions).
            deck_prim = self.stage.GetPrimAtPath(self.cart_support_path)
            if not deck_prim:
                return None
            if self._deck_local is None:
                bbox = UsdGeom.BBoxCache(Usd.TimeCode.Default(),
                                         [UsdGeom.Tokens.default_, UsdGeom.Tokens.render])
                rng = bbox.ComputeWorldBound(deck_prim).ComputeAlignedRange()
                mn, mx = rng.GetMin(), rng.GetMax()
                wc = np.array([(mn[0] + mx[0]) / 2.0, (mn[1] + mx[1]) / 2.0, float(mx[2]), 1.0])
                self._deck_local = np.linalg.inv(Tc) @ wc            # rigid in the cart frame
                self._deck_extent = (float(mx[0] - mn[0]), float(mx[1] - mn[1]))
            w = Tc @ self._deck_local
            world = np.array([float(w[0]), float(w[1]), float(w[2]), 1.0])
        else:
            lx, ly = cp['load_center_body_xy']
            world = np.array([float(Tc[0, 3]) + math.cos(cyaw) * lx - math.sin(cyaw) * ly,
                              float(Tc[1, 3]) + math.sin(cyaw) * lx + math.cos(cyaw) * ly,
                              float(cp['support_z_world']), 1.0])
        p = np.linalg.inv(Tub) @ world                                # load centre in ur_base_link
        ub_yaw = math.atan2(float(Tub[1, 0]), float(Tub[0, 0]))
        return (float(p[0]), float(p[1]), float(p[2])), float(cyaw - ub_yaw)

    def _carton_center_station(self):
        """The carton CENTRE in ur_base_link from the live handling-unit prim (controlled-asset ground
        truth, for the PLACE physics gate only). None when disabled / the prim is absent. Caches the
        carton-local centre offset on first read (rigid body) so the per-tick cost is one XformCache."""
        if not self.verify_place or self.stage is None:
            return None
        import numpy as np
        from pxr import Usd, UsdGeom
        unit = self.stage.GetPrimAtPath(self._handling_unit_path)
        if not unit:
            return None
        cache = UsdGeom.XformCache()
        Tu = np.asarray(cache.GetLocalToWorldTransform(unit)).T
        if self._unit_local_center is None:
            bbox = UsdGeom.BBoxCache(Usd.TimeCode.Default(),
                                     [UsdGeom.Tokens.default_, UsdGeom.Tokens.render])
            rng = bbox.ComputeWorldBound(unit).ComputeAlignedRange()
            mn, mx = rng.GetMin(), rng.GetMax()
            wc = np.array([(mn[0] + mx[0]) / 2.0, (mn[1] + mx[1]) / 2.0, (mn[2] + mx[2]) / 2.0, 1.0])
            self._unit_local_center = np.linalg.inv(Tu) @ wc
        w = Tu @ self._unit_local_center
        Tub = np.asarray(cache.GetLocalToWorldTransform(self.stage.GetPrimAtPath(self.base_path))).T
        p = np.linalg.inv(Tub) @ np.array([float(w[0]), float(w[1]), float(w[2]), 1.0])
        return (float(p[0]), float(p[1]), float(p[2]), 0.0, 0.0, 0.0, 1.0)

    def _carton_top_station(self):
        """live_pose transfer SOURCE: the carton's observed TOP contact surface (the types.Observation.pose
        contract) from the LIVE handling-unit prim -- its centre + half the profile height, and its yaw relative
        to ur_base_link. Controlled-asset ground truth like _cart_pose_station, but where the carton actually
        rests (a warm-start carton sits on the UPPER shelf, cart body +0.18 m -- not at the deck-top AABB
        centre). Returns ((x, y, z_top), yaw) in ur_base_link, or None when the prim is absent."""
        import numpy as np
        from pxr import Usd, UsdGeom
        unit = self.stage.GetPrimAtPath(self._handling_unit_path)
        if not unit:
            return None
        cache = UsdGeom.XformCache()
        Tu = np.asarray(cache.GetLocalToWorldTransform(unit)).T
        if self._unit_local_center is None:       # shared rigid-body cache with _carton_center_station
            bbox = UsdGeom.BBoxCache(Usd.TimeCode.Default(),
                                     [UsdGeom.Tokens.default_, UsdGeom.Tokens.render])
            rng = bbox.ComputeWorldBound(unit).ComputeAlignedRange()
            mn, mx = rng.GetMin(), rng.GetMax()
            wc = np.array([(mn[0] + mx[0]) / 2.0, (mn[1] + mx[1]) / 2.0, (mn[2] + mx[2]) / 2.0, 1.0])
            self._unit_local_center = np.linalg.inv(Tu) @ wc
        w = Tu @ self._unit_local_center
        Tub = np.asarray(cache.GetLocalToWorldTransform(self.stage.GetPrimAtPath(self.base_path))).T
        p = np.linalg.inv(Tub) @ np.array([float(w[0]), float(w[1]), float(w[2]), 1.0])
        height = float(next(iter(self.profiles.values())).dimensions[2])
        yaw = math.atan2(float(Tu[1, 0]), float(Tu[0, 0])) - math.atan2(float(Tub[1, 0]), float(Tub[0, 0]))
        return (float(p[0]), float(p[1]), float(p[2]) + height / 2.0), yaw

    def _ground_truth_target(self, now):
        """GROUND-TRUTH COLLECT (collect-gt): the pick target from the controlled carton asset's simulated pose.

        Same shape as the camera ``SourceTarget`` (top-face centre in ur_base_link + world/map yaw, see
        ground_truth_collect.carton_top_pose), so the packet, ROI reprojection and ``_map_pose_to_base``
        path downstream are unchanged. Provenance is 'ground_truth', never 'sensor'. None when the stage,
        carton prim or its grasp profile is missing -- nothing is filled in from scenario constants."""
        if self.stage is None:
            return None
        import numpy as np
        from pxr import Usd, UsdGeom
        from twin.restock_camera.source_target import SourceTarget
        unit = self.stage.GetPrimAtPath(self._handling_unit_path)
        base = self.stage.GetPrimAtPath(self.base_path)
        if not unit or not base:
            return None
        sku = unit.GetCustomDataByKey('restock:sku') or (next(iter(self.profiles)) if self.profiles else '')
        profile = self.profiles.get(sku)
        if profile is None:
            return None
        cache = UsdGeom.XformCache()
        Tu = np.asarray(cache.GetLocalToWorldTransform(unit)).T
        if self._unit_local_center is None:          # shared rigid carton-local centre (see above)
            bbox = UsdGeom.BBoxCache(Usd.TimeCode.Default(),
                                     [UsdGeom.Tokens.default_, UsdGeom.Tokens.render])
            rng = bbox.ComputeWorldBound(unit).ComputeAlignedRange()
            mn, mx = rng.GetMin(), rng.GetMax()
            wc = np.array([(mn[0] + mx[0]) / 2.0, (mn[1] + mx[1]) / 2.0, (mn[2] + mx[2]) / 2.0, 1.0])
            self._unit_local_center = np.linalg.inv(Tu) @ wc
        Tb = np.asarray(cache.GetLocalToWorldTransform(base)).T
        pose = gtc.carton_top_pose(Tu, Tb, self._unit_local_center[:3], profile.dimensions[2])
        return SourceTarget(sku=sku, pose_station=pose, stamp=float(now),
                            provenance={'source': gtc.SOURCE, 'asset': self._handling_unit_path,
                                        'note': 'collect-gt: simulated carton asset pose (sim ground truth)'},
                            location=self.source_location, station_frame='ur_base_link')

    def _gt_place_target(self, now):
        """collect-gt: the collect PLACE target at the warm-shortcut slot (gt_place_slot) on the cart's LIVE
        pose: the support point in ur_base_link + the slot's base-relative yaw (world yaw re-expressed
        through the chassis by _map_pose_to_base, like the pick), so the wall policy lands the carton at the
        slot yaw with the gripped face toward the hitch. Controlled-asset derived like _cart_target, hence
        the same 'sensor' label. None without a cart/stage."""
        cp = self.cart_placement
        if not cp or not self.cart_root or self.stage is None:
            return None
        import numpy as np
        from pxr import UsdGeom
        from twin.restock_manipulation.types import Target
        cart_prim = self.stage.GetPrimAtPath(self.cart_root)
        base = self.stage.GetPrimAtPath(self.base_path)
        if not cart_prim or not base:
            return None
        cache = UsdGeom.XformCache()
        pose = gtc.slot_pose(np.asarray(cache.GetLocalToWorldTransform(cart_prim)).T,
                             np.asarray(cache.GetLocalToWorldTransform(base)).T, self.gt_place_slot)
        return Target(observation_id=f'cart:{cp["location"]}:{now:.3f}', source='sensor',
                      simulation_time=float(now), location=cp['location'],
                      pose=tuple(self._map_pose_to_base(pose)), clear=True,
                      free_dimensions=tuple(cp['free_dimensions']), frame='ur_base_link')

    def _capture_ground_truth_collect(self, now):
        """collect-gt: the collect capture under RESTOCK_COLLECT_GROUND_TRUTH. Mirrors the camera path's
        gates (parked, fresh) and packet, but the source observation is the carton asset's pose labelled
        'ground_truth'. Once the carton has been held it is no longer AT the source, so later packets claim
        no source observation (like the fm410 survival packet) yet keep the cart placement target that
        the place re-registration and the preplace re-check need. None while the base is not parked."""
        from twin.restock_manipulation.types import Sensors, Observation
        parked = bool(self.motion.parked())
        if not parked:
            if self._calls % 10 == 1:
                print(f'COLLECT_GT calls={self._calls} parked=False', flush=True)
            return None
        held = self._carton_attached()
        if held:
            self._gt_carton_held = True
        try:
            target = self._ground_truth_target(now)
        except Exception as exc:
            target = None
            if self._calls % 10 == 1:
                print(f'COLLECT_GT calls={self._calls} target_error={type(exc).__name__}:{exc}', flush=True)
        if target is None:
            if self._calls % 10 == 1:
                print(f'COLLECT_GT calls={self._calls} target=None (no carton asset/profile)', flush=True)
            return None
        observations = ()
        if not self._gt_carton_held:
            roi = self._reproject(target)
            if roi is not None:
                self.sensed_target_bboxes = (dict(roi=roi, location=self.source_location),)
            observations = (Observation(observation_id=f'ground_truth:{now:.3f}', source=gtc.SOURCE,
                                        simulation_time=float(now), sku=target.sku,
                                        location=self.source_location,
                                        pose=tuple(self._map_pose_to_base(target.pose_station)),
                                        confidence=1.0, frame='ur_base_link'),)
        if self._calls % 10 == 1:
            print(f'COLLECT_GT calls={self._calls} sku={target.sku} held={self._gt_carton_held} '
                  f'pose={tuple(round(v, 3) for v in target.pose_station)} (ground-truth, not camera)',
                  flush=True)
        try:
            joints = tuple(float(j) for j in self.joints())
        except Exception:
            joints = (0.,) * 6
        try:
            cart_target = (self._gt_place_target(now) if self.gt_place_slot
                           else self._cart_target(now, target))
        except Exception as exc:
            cart_target = None
            if self._calls % 10 == 1:
                print(f'COLLECT_GT calls={self._calls} cart_target_error={type(exc).__name__}:{exc}', flush=True)
        try:
            placed_pose = self._carton_center_station()
        except Exception:
            placed_pose = None
        # fm488: GT collect has no camera framing the pick either; carry now+window so the collect grasp's
        # gripper command is not refused 'missing acquisition deadline' (the cold collect-only failure).
        return Sensors(now, observations, cart_target, parked, bool(self.contact.read()), now,
                       self._collision_time(), joints, (), float(now), now + 300.0, placed_pose)

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
        if self.transfer_mode == LIVE_POSE:
            # live_pose (transfer_mode.py): pick the carton where it actually rests -- its live TOP surface --
            # not the deck-top AABB centre (run-354/356 reach fail). No live carton => no packet (never fabricated).
            live = self._carton_top_station()
            if live is None:
                if self._calls % 10 == 1:
                    print('TRANSFER_CAPTURE live_pose: handling-unit prim absent', flush=True)
                return None
            (cx, cy, cz), yaw = live
        sku = next(iter(self.profiles)) if self.profiles else ''
        cq = (0., 0., math.sin(yaw / 2.), math.cos(yaw / 2.))
        obs = Observation(observation_id=f'cart:{now:.3f}', source='sensor', simulation_time=float(now),
                          sku=sku, location=self.cart_placement['location'], pose=(cx, cy, cz, *cq),
                          confidence=1.0, frame='ur_base_link')
        # fm461: honestly carry the provider's source. A ground-truth bay target (the demo fallback, when
        # the shelf is not camera-commissioned) is labelled source='ground_truth', NOT 'sensor', so the
        # controller gate (transfer_ground_truth) and the evidence never misrepresent it as a measurement.
        prov = getattr(slot, 'provenance', None)
        tgt_source = 'ground_truth' if isinstance(prov, dict) and prov.get('source') == 'ground_truth' else 'sensor'
        # fm477: the ground-truth bay target's pose_station is a WORLD/map pose (the authored bay surface),
        # but the Target/controller treat the shelf provider's pose_station as ur_base_link (the real
        # camera provider reprojects to base; the ground-truth provider has no stage). Reproject it to
        # ur_base_link with the live arm-base transform, or the controller reads a ~16 m world coordinate
        # as a base-frame reach and fails 'unreachable or undersized target' -- exactly the pose_station=
        # ur_base_link assumption this method's own NOTE flagged to validate at the first transfer (run-351).
        tgt_pose = tuple(slot.pose_station)
        if tgt_source == 'ground_truth' and self.stage is not None:
            import numpy as np
            from pxr import UsdGeom
            Tbu = np.linalg.inv(np.asarray(UsdGeom.XformCache().GetLocalToWorldTransform(
                self.stage.GetPrimAtPath(self.base_path))).T)      # base_from_world (4x4)
            bp = Tbu @ np.array([tgt_pose[0], tgt_pose[1], tgt_pose[2], 1.0])
            R = Tbu[:3, :3]                                         # the ground-truth world quat is identity
            tr = R[0, 0] + R[1, 1] + R[2, 2]
            if tr > 0:
                s = math.sqrt(tr + 1.0) * 2; qw = .25 * s; qx = (R[2,1]-R[1,2])/s; qy = (R[0,2]-R[2,0])/s; qz = (R[1,0]-R[0,1])/s
            elif R[0,0] >= R[1,1] and R[0,0] >= R[2,2]:
                s = math.sqrt(1.0+R[0,0]-R[1,1]-R[2,2]) * 2; qw = (R[2,1]-R[1,2])/s; qx = .25*s; qy = (R[0,1]+R[1,0])/s; qz = (R[0,2]+R[2,0])/s
            elif R[1,1] >= R[2,2]:
                s = math.sqrt(1.0+R[1,1]-R[0,0]-R[2,2]) * 2; qw = (R[0,2]-R[2,0])/s; qx = (R[0,1]+R[1,0])/s; qy = .25*s; qz = (R[1,2]+R[2,1])/s
            else:
                s = math.sqrt(1.0+R[2,2]-R[0,0]-R[1,1]) * 2; qw = (R[1,0]-R[0,1])/s; qx = (R[0,2]+R[2,0])/s; qy = (R[1,2]+R[2,1])/s; qz = .25*s
            tgt_pose = (float(bp[0]), float(bp[1]), float(bp[2]), float(qx), float(qy), float(qz), float(qw))
            if self._calls % 10 == 1:
                print('TRANSFER_GT_REPROJECT world=%s -> base=(%.3f,%.3f,%.3f) dist=%.3f' % (
                    tuple(round(c, 3) for c in slot.pose_station[:3]), bp[0], bp[1], bp[2],
                    float(np.linalg.norm(bp[:3]))), flush=True)
        tgt = Target(observation_id=f'shelf:{slot.stamp}', source=tgt_source, simulation_time=float(now),
                     location=slot.location, pose=tgt_pose, clear=bool(slot.clear),
                     free_dimensions=tuple(slot.free_dimensions), frame='ur_base_link')
        try:
            joints = tuple(float(j) for j in self.joints())
        except Exception:
            joints = (0.,) * 6
        # fm461: thread the sim-measured carton pose so the transfer verify can check placement from the
        # ground truth (carton vs bay target) -- the transfer destination has no fixed camera framing the
        # post-release slot in the ground-truth fallback. None (stage read failed) => camera verify path.
        try:
            placed_pose = self._carton_center_station()
        except Exception:
            placed_pose = None
        print(f'TRANSFER_CAPTURE calls={self._calls} sku={sku} cart=({cx:.2f},{cy:.2f},{cz:.2f}) '
              f'shelf={slot.location} clear={slot.clear} src={tgt_source}'
              + (f' transfer_mode={self.transfer_mode}' if self.transfer_mode else ''), flush=True)
        # fm488: carry a sim-clock acquisition-to-use deadline (now + window) so the gripper command at
        # CONTACT is not refused 'missing acquisition deadline' (isaac_ros_graph). The GT/live-pose transfer
        # has no camera framing the pick to stamp this (the rendered camera path sets now+300), so stamp it
        # here; the controller checks it on the sim clock and a 300 s budget never expires mid-grasp.
        return Sensors(now, (obs,), tgt, parked, bool(self.contact.read()), now,
                       self._collision_time(), joints, (), float(now), now + 300.0, placed_pose)

    def _collision_time(self):
        # fm415: hold last-good collision_time so a -1 mirror-gap tick at the weld never reaches the
        # controller freshness gate (run-319). See hold_last_good_collision.
        emit, self._last_good_collision = hold_last_good_collision(
            float(self.motion.collision_time), getattr(self, "_last_good_collision", None))
        return emit

    def _carton_attached(self):
        """True once the carton is physically held -- native grippedObjects OR the SIM_ATTACH_FALLBACK weld
        prim on the SHARED stage (reliable regardless of which NativePanelContact instance holds the fb
        flag, so it never mis-reads a sibling capture's instance)."""
        try:
            if bool(self.contact.read()):
                return True
        except Exception:
            pass
        stage = getattr(self, 'stage', None)
        path = getattr(self.contact, 'path', None)
        if stage is None or not path:
            return False
        try:
            return bool(stage.GetPrimAtPath(path + '/sim_attach_fallback'))
        except Exception:
            return False

    def _survival_packet(self, now):
        """fm410 (run-316): an honest arm-feedback /sensors packet when the SOURCE camera is occluded but the
        carton is attached and the base parked (see survive_collect_occlusion). Empty source observations
        (no fresh localisation claimed) + LIVE contact/placed_pose/joints/collision_time; valid_until_steady
        None (no source acquisition this frame -> the controller skips its acquisition freshness, driving on
        arm feedback). Returns None when survival is disabled or the carton is not attached/base not parked."""
        from twin.restock_manipulation.types import Sensors
        if getattr(self, '_survive_occlusion', None) is None:
            import os
            self._survive_occlusion = bool(os.environ.get('RESTOCK_COLLECT_SURVIVE_OCCLUSION'))
        parked = bool(self.motion.parked())
        if not survive_collect_occlusion(self._survive_occlusion, self._carton_attached(), parked):
            return None
        try:
            joints = tuple(float(j) for j in self.joints())
        except Exception:
            joints = (0.,) * 6
        try:
            placed_pose = self._carton_center_station()
        except Exception:
            placed_pose = None
        if self._calls % 10 == 1:
            print(f'COLLECT_SURVIVE calls={self._calls} source-occluded, carton attached + parked -> '
                  f'arm-feedback packet (contact={bool(self.contact.read())})', flush=True)
        return Sensors(now, (), None, parked, bool(self.contact.read()), now,
                       self._collision_time(), joints, (), float(now), None, placed_pose)

    def capture(self, now, source, destination):
        from twin.restock_manipulation.types import Sensors, Observation
        self.sensed_target_bboxes = ()
        self._calls = getattr(self, '_calls', 0) + 1
        # TRANSFER (cart is the SOURCE, shelf the DESTINATION): the mirror of collect. Routed here
        # when the mission source is the cart; collect (source = bin) falls through unchanged.
        cart_loc = self.cart_placement.get('location')
        if cart_loc and source == cart_loc:
            return self._capture_transfer(now)
        if self.collect_ground_truth:              # collect-gt: labelled ground-truth pick, never the camera
            return self._capture_ground_truth_collect(now)
        if self.provider is None:
            if self._calls % 30 == 1:
                print(f'COLLECT_CAPTURE calls={self._calls} provider=None', flush=True)
            return None
        try:
            target = self.provider(now)             # SourceTarget | None (provider enforces its own gates)
        except Exception as exc:                     # a swallowed source-capture failure (decode/freshness/...)
            if self._calls % 10 == 1:
                print(f'COLLECT_CAPTURE calls={self._calls} provider_error={type(exc).__name__}:{exc}', flush=True)
            return self._survival_packet(now)        # fm410: source occluded -> arm-feedback packet if attached
        if target is None:
            if self._calls % 10 == 1:
                print(f'COLLECT_CAPTURE calls={self._calls} target=None parked={self.motion.parked()}', flush=True)
            return self._survival_packet(now)        # fm410: source occluded -> arm-feedback packet if attached
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
        try:                                      # ground-truth PLACE-gate read must never starve /sensors
            placed_pose = self._carton_center_station()
        except Exception as exc:
            placed_pose = None
            if self._calls % 10 == 1:
                print(f'COLLECT_CAPTURE calls={self._calls} placed_pose_error={type(exc).__name__}:{exc}',
                      flush=True)
        return Sensors(now, (obs,), cart_target, parked, bool(self.contact.read()), now,
                       self._collision_time(), joints, (), float(target.stamp),
                       target.valid_until_steady, placed_pose,
                       # fm449: carry the source's clean-this-tick verdict to the host collect gate.
                       collect_source_clean=bool(getattr(target, 'clean', False)))


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
    cart_support_path = None
    import os
    verify_place = params.cart_placement and os.environ.get('RESTOCK_VERIFY_PLACE', '0') != '0'
    if params.cart_placement:
        cart_asset = assembly.evidence['composition']['assets']['cart']
        cart_root = cart_asset['root']
        # fm389: aim the collect placement at the DECK INTERIOR, derived from the live support prim, not
        # the scenario support_z_world constant (which targeted the UPPER shelf, 0.71 m too high). Opt-in
        # via RESTOCK_PLACE_ON_DECK so an unset env is byte-identical with the legacy constant path. The
        # prim path comes from the composition's own stock_supports (the deck is index 0), never a
        # hand-pinned path. Asserted at boot so a mis-set env is visible (host/runtime-split lesson).
        import os
        if os.environ.get('RESTOCK_PLACE_ON_DECK', '0') != '0':
            supports = cart_asset.get('stock_supports') or []
            cart_support_path = supports[0] if supports else None
        print('PLACE_ON_DECK env=%s support_prim=%s (None => legacy support_z_world=%s) VERIFY_PLACE=%s'
              % (os.environ.get('RESTOCK_PLACE_ON_DECK', '0'), cart_support_path,
                 params.cart_placement.get('support_z_world'), bool(verify_place)), flush=True)
    collect_bridge = CollectTargetBridge(
        stage=stage, arm_camera=camera, arm_camera_path=camera_path,
        base_path=assembly.paths['ur_base_link'], profiles=params.profiles,
        mobile_base_path=assembly.paths.get('base_link', assembly.paths['ur_base_link']),
        motion=feedback, contact=contact,
        joints=feedback.joint_positions,
        source_location=params.collect_source_location,
        cart_placement=params.cart_placement, cart_root=cart_root,
        cart_support_path=cart_support_path, verify_place=bool(verify_place),
        gt_place_slot=params.gt_place_slot)
    # collect-gt: name the workaround at boot so a wrongly-set switch is visible before the first stage.
    print('COLLECT_GROUND_TRUTH enabled=%s place_slot=%s (%s=1: collect pick from the simulated carton '
          'asset pose, labelled ground_truth; place at the warm-shortcut slot; default off)'
          % (collect_bridge.collect_ground_truth, params.gt_place_slot or None, gtc.ENV), flush=True)

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
                                publish_lag=0.1, heartbeat_s=params.station_heartbeat_s,
                                stage_unavailable_s=params.station_stage_unavailable_s)
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
    if gtc.enabled():
        # collect-gt: place at the warm-shortcut slot, read from the SAME scenario the shortcut uses
        # (fail closed at boot if it cannot be read -- never an invented spot).
        import json
        from dataclasses import replace
        from pathlib import Path
        scenario = config.get('scenario')
        if isinstance(scenario, (str, Path)):
            scenario = json.loads(Path(scenario).read_text())
        params = replace(params, gt_place_slot=gtc.warm_slot(scenario))
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
