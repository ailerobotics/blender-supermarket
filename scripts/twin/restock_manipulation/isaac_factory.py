"""`manip_factory` for the mission bundle: build the production manipulation runtime
from the landed (retired-lane) restock_manipulation modules, unchanged.

The integration bundle (`twin.restock.isaac_bundle`) resolves this as the runtime-config
`manip_factory` ref and calls `create(app=, assembly=, plan=, config=, output=)`. It
returns a fully-configured `ManipulationBridge` (owns_sensor_io=True — the CAD runtime owns
its camera/ESDF and control I/O) wrapping `CadRestockRuntime` (CAD side-grip planning +
`TransferController` + `IsaacCartonSensors`). No sensor data is fabricated and no behaviour
of the landed modules is modified; this module only assembles them and prepares the arm
station per mission action.

`parse_manipulation_config()` is the pure, CPU-tested contract: it validates the grasp
profiles (including the scenario SKU and a single calibrated grasp mode), the per-station
sensing slots + `base_from_camera` calibration, and the shelfbot package root from
`config['manipulation']`. The Isaac construction (arm camera + `CadRestockRuntime`) is
isolated behind the `_build_runtime` seam so the wiring is proven offline without a boot.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

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

    return ManipConfig(profiles=profiles, stations=stations, shelfbot_root=shelfbot_root,
                       walls=walls, voxel_dims=tuple(float(v) for v in vd), voxel_size=float(vs),
                       grasp_mode=next(iter(modes)))


def _build_runtime(*, app, assembly, params, output):
    """Real Isaac construction of CadRestockRuntime (arm camera + CAD runtime).

    Only runs inside the live app; the CPU contract tests inject `_builder`.
    """
    import numpy as np
    from pxr import Gf, UsdGeom
    from isaacsim.sensors.camera import Camera
    from .isaac_runtime import CadRestockRuntime

    paths = assembly.paths
    # front_camera_frame is a URDF body frame (+X forward, +Z up), not a Camera prim.
    # Author a child USD Camera (which looks -Z with +Y up) oriented to look forward
    # along the frame's +X with +Z up; Camera() then adopts this existing prim.
    stage = assembly.stage
    camera_path = paths['front_camera_frame'] + '/arm_camera'
    if not stage.GetPrimAtPath(camera_path):
        cam = UsdGeom.Camera.Define(stage, camera_path)
        cam.AddTransformOp().Set(Gf.Matrix4d(0, -1, 0, 0, 0, 0, 1, 0, -1, 0, 0, 0, 0, 0, 0, 1))
    camera = Camera(prim_path=camera_path)
    camera.initialize()
    source = params.stations['source']
    return CadRestockRuntime(
        stage=assembly.stage, articulation=assembly.robot, camera=camera,
        gripper_path='/RestockPanelGripper', urdf_path=str(output / 'mobile-import.urdf'),
        shelfbot_root=params.shelfbot_root, pose_bridge_root=paths['ur_base_link'],
        profiles=params.profiles, slots=source['slots'],
        base_from_camera=np.asarray(source['base_from_camera'], float),
        voxel_dims=params.voxel_dims, voxel_size=params.voxel_size, walls=params.walls)


class _DegradedMotion:
    """Reports the REAL base stationarity from the articulation; every manipulation
    operation is unavailable. Used only when the in-process cuRobo planner is absent.
    """
    def __init__(self, articulation):
        self._robot = articulation
        self.collision_time = -1.
        self.last_grid = None
        self.source = None

    def parked(self):
        import numpy as np
        linear = np.asarray(self._robot.get_linear_velocity(), float)
        angular = np.asarray(self._robot.get_angular_velocity(), float)
        return bool(np.isfinite(linear).all() and np.isfinite(angular).all()
                    and np.linalg.norm(linear) < .005 and np.linalg.norm(angular) < .005)

    def tick(self, dt):
        pass

    def stop(self):
        pass


class _UnavailableManipulation:
    """Honest stand-in when the in-process cuRobo planner is not provisioned.

    Construction succeeds so the mission still exercises and records every stage,
    but the manipulation stages (collect, transfer) report status ``'unavailable'``
    with the measured reason. It never fabricates a successful grasp or placement.
    This is the interim path until manip_factory is switched to the host cuMotion
    MoveGroup port; see data/mobile-restock-v2/report.md.
    """
    def __init__(self, reason, articulation):
        from types import SimpleNamespace
        self._reason = reason
        self.motion = _DegradedMotion(articulation)
        self.contact = None
        # ManipulationBridge(owns_sensor_io=True) reads controller.controller.cart_location;
        # sensing reads controller.controller.profiles (absent -> no product profiles).
        self.controller = SimpleNamespace(cart_location='rollwagen', profiles={})

    def start(self, operation, job):
        pass

    def configure_station(self, *, slots, base_from_camera):
        pass

    def step(self, dt, simulation_time):
        from .types import Result
        return Result('unavailable', 'unavailable', self._reason)

    def stop(self):
        pass


def create(*, app, assembly, plan, config, output, _builder=None):
    """Assemble the manipulation runtime and return a configured ManipulationBridge."""
    from twin.restock.runtime import ManipulationBridge

    params = parse_manipulation_config(config, plan.job_sku)
    build = _builder or _build_runtime
    try:
        runtime = build(app=app, assembly=assembly, params=params, output=output)
    except ModuleNotFoundError as exc:
        # The in-process cuRobo planner (shelfbot_newgripper) is not installed in the
        # Isaac interpreter. Degrade honestly instead of crashing the whole factory, so
        # the mission records the full navigate/couple/tow/park/verify/return loop with
        # the manipulation stages reported FAILED-unavailable (never a mock success).
        if exc.name != 'curobo':
            raise
        runtime = _UnavailableManipulation(
            f'manipulation construction unavailable: in-process cuRobo not provisioned ({exc})',
            assembly.robot)

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
