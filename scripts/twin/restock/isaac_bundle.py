"""`isaac_bundle:create` — the simulator bundle factory `run_mobile_restock.py --execute`
loads with `--factory twin.restock.isaac_bundle:create`.

The launcher boots the guarded `SimulationApp`, then calls

    bundle = create(app=app, manifest=..., scenario=..., config=..., output=...)

and drives `Mission(db, Job(**scenario['job']), bundle.adapter)`. This module owns the
INTEGRATION ASSEMBLY only: it composes the validated physics scene
(`physics_setup.assemble`, the same builder the coupling/towing/assembly native
diagnostics use), builds the measured physics->mission `IsaacPort`, and wraps the real
sub-runtimes in the CPU-tested `RuntimeAdapter` (`fixture_mode=False`, so the launcher's
production check passes). It fabricates no sensor data and installs no mock adapter.

The per-lane CONTROLLERS stay lane-owned and are resolved from the runtime-config as
`module:function` references (same delegation contract as the launcher's `--factory`):

    config['nav_factory']     -> navigation controller   (navigation lane)
    config['manip_factory']   -> manipulation controller (manipulation lane)
    config['sensing_factory'] -> inventory-capture callable + recording camera (camera lane)

Each factory is called as `fn(app=app, assembly=assembly, plan=plan, config=config,
output=output)` and returns the lane's validated runtime object; a missing/invalid
reference fails closed with the exact contract it must satisfy, never a stub. This keeps
integration from hard-coding another lane's constructor signatures and lets us consume
their validated follow-up commits unchanged.

`plan_bundle()` is the pure, CPU-tested contract the factory consumes: it derives and
validates the four mission poses, the per-action articulation tolerances, the navigation
mode and the verification/recording parameters from manifest+scenario+config. It imports
no Isaac and is where the error-prone config->mission-contract mapping is proven offline.
"""
from __future__ import annotations

import importlib
import math
from dataclasses import dataclass, field
from pathlib import Path

# RuntimeAdapter and its bridges are Isaac-free (pure mission logic); safe to import here.
from .runtime import RuntimeAdapter, NavigationBridge, ManipulationBridge

# The four navigation goals RuntimeAdapter.start() dispatches, and the actions that
# carry a cart to a destination under an articulation tolerance.
NAV_GOALS = ('pickup', 'cart_dock', 'shelf_approach', 'home')
ARTICULATED_ACTIONS = ('outbound', 'return')
NAV_MODES = ('reference', 'nav2', 'sensor')
MAX_ARTICULATION_RAD = math.radians(65)


def _finite_pose(pose):
    return (isinstance(pose, (list, tuple)) and len(pose) == 3
            and all(isinstance(v, (int, float)) and math.isfinite(v) for v in pose))


@dataclass(frozen=True)
class BundlePlan:
    poses: dict                 # name -> (x, y, yaw) for NAV_GOALS
    goal_articulation: dict     # action -> tolerance (rad), subset of ARTICULATED_ACTIONS
    nav_mode: str               # one of NAV_MODES
    verification_source: str    # e.g. 'rollwagen'
    recording_fps: int
    package_roots: tuple
    job_sku: str                # scenario job SKU (lane factories need it; call sig stays frozen)
    job_quantity: int


def require_transport_positions(mobile_manifest):
    """Fail closed unless the mobile manifest carries usable arm-stow `transport_positions`.

    The manipulation host derives `transport_joints` from `mobile_manifest['transport_positions']`
    (ros_host.py), and both the collect `stow` phase and the outbound/return `arm_not_in_transport`
    gate depend on the arm being stowed into the transport envelope. An absent or malformed mapping
    otherwise surfaces as an opaque KeyError in the host subprocess, far from the mission boot.
    Returns the validated mapping. Pure (no Isaac); CPU-tested.
    """
    positions = mobile_manifest.get('transport_positions') if isinstance(mobile_manifest, dict) else None
    if not isinstance(positions, dict) or not positions:
        raise ValueError(
            "mobile manifest is missing 'transport_positions' (arm-stow joint targets); the "
            'manipulation host needs it to build transport_joints and the collect-stow / '
            'outbound-return arm-transport gate cannot pass without it')
    for name, value in positions.items():
        if not isinstance(name, str) or not name or isinstance(value, bool) or \
                not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f'transport_positions[{name!r}] must be a finite joint angle')
    return positions


def plan_bundle(manifest, scenario, config):
    """Pure mission-bundle contract from manifest+scenario+config. No Isaac import.

    The four mission poses and the per-action articulation tolerances are REQUIRED
    explicit runtime-config inputs (`mission_poses`, `goal_articulation`); integration
    never guesses navigation geometry from staging. Raises ValueError (fail-closed) on
    any missing or malformed field so a misconfigured bundle refuses to build rather
    than drive the robot to an invented goal.
    """
    if not isinstance(config, dict) or not isinstance(scenario, dict):
        raise ValueError('config and scenario must be objects')

    poses_in = config.get('mission_poses')
    if not isinstance(poses_in, dict) or set(poses_in) != set(NAV_GOALS):
        raise ValueError(f'config.mission_poses must provide exactly {NAV_GOALS}')
    poses = {}
    for name in NAV_GOALS:
        if not _finite_pose(poses_in[name]):
            raise ValueError(f'mission pose {name!r} must be finite (x, y, yaw)')
        poses[name] = tuple(float(v) for v in poses_in[name])

    art_in = config.get('goal_articulation', {})
    if not isinstance(art_in, dict):
        raise ValueError('config.goal_articulation must be an object')
    goal_articulation = {}
    for action, tol in art_in.items():
        if action not in ARTICULATED_ACTIONS:
            raise ValueError(f'goal_articulation action {action!r} not in {ARTICULATED_ACTIONS}')
        if not isinstance(tol, (int, float)) or not math.isfinite(tol) or not 0 < tol <= MAX_ARTICULATION_RAD:
            raise ValueError(f'goal_articulation[{action!r}] must be in (0, {MAX_ARTICULATION_RAD:.4f}] rad')
        goal_articulation[action] = float(tol)

    # Navigation mode: explicit config override, else derived from the scenario model.
    nav_mode = config.get('nav_mode')
    if nav_mode is None:
        nav_mode = 'sensor' if scenario.get('navigation_model') == 'articulated' else 'reference'
    if nav_mode not in NAV_MODES:
        raise ValueError(f'nav_mode must be one of {NAV_MODES}')

    verification_source = config.get('verification_source', 'rollwagen')
    if not isinstance(verification_source, str) or not verification_source:
        raise ValueError('verification_source must be a non-empty string')

    fps = config.get('recording_fps', 30)
    if type(fps) is not int or not 1 <= fps <= 120:
        raise ValueError('recording_fps must be an int in [1, 120]')

    roots = config.get('package_roots')
    if not isinstance(roots, (list, tuple)) or not roots or not all(isinstance(r, str) and r for r in roots):
        raise ValueError('config.package_roots must be a non-empty list of paths')

    job = scenario.get('job')
    if not isinstance(job, dict) or not isinstance(job.get('sku'), str) or not job['sku']:
        raise ValueError('scenario.job.sku required')
    if type(job.get('quantity')) is not int or job['quantity'] < 1:
        raise ValueError('scenario.job.quantity must be a positive int')

    return BundlePlan(poses=poses, goal_articulation=goal_articulation, nav_mode=nav_mode,
                      verification_source=verification_source, recording_fps=int(fps),
                      package_roots=tuple(roots), job_sku=job['sku'], job_quantity=int(job['quantity']))


@dataclass
class Bundle:
    adapter: RuntimeAdapter
    recording: dict = field(default_factory=dict)
    _closers: tuple = ()

    def close(self):
        errors = []
        for closer in self._closers:
            try:
                closer()
            except Exception as exc:  # run every closer even if one raises
                errors.append(str(exc))
        if errors:
            raise RuntimeError('; '.join(errors))


def _resolve_factory(config, key):
    ref = config.get(key)
    if not isinstance(ref, str) or ':' not in ref:
        raise ValueError(
            f'runtime-config.{key} must name a lane-owned builder as "module:function" '
            f'(no stub is substituted); it is called fn(app, assembly, plan, config, output)')
    module, name = ref.split(':', 1)
    return getattr(importlib.import_module(module), name)


def _build_navigation(plan, controller):
    return NavigationBridge(controller, nav2=plan.nav_mode == 'nav2',
                            sensor_navigation=plan.nav_mode == 'sensor')


def create(app, manifest, scenario, config, output, *, _assemble=None, _runtime=None):
    """Assemble the production mission bundle inside the guarded SimulationApp.

    `_assemble`/`_runtime` are injection seams for CPU contract tests ONLY; production
    leaves them None and uses the real Isaac builders. The returned adapter is always
    `fixture_mode=False`, so the launcher's production check accepts it.
    """
    plan = plan_bundle(manifest, scenario, config)
    output = Path(output)

    assemble = _assemble or _real_assemble
    assembly = assemble(app=app, config=config, scenario=scenario, output=output)

    build_runtime = _runtime or _build_isaac_runtime
    parts = build_runtime(app=app, assembly=assembly, plan=plan, config=config, output=output,
                          scenario=scenario, manifest=manifest)
    navigation, mobile, manipulation, port, closers, recording = (
        parts['navigation'], parts['mobile'], parts['manipulation'], parts['port'],
        parts.get('closers', ()), parts.get('recording', {}))

    adapter = RuntimeAdapter(navigation, mobile, manipulation, port, plan.poses,
                             fixture_mode=False, goal_articulation=plan.goal_articulation)
    return Bundle(adapter=adapter, recording=recording, _closers=tuple(closers))


def _real_assemble(*, app, config, scenario, output):
    from .physics_setup import assemble
    return assemble(app=app, config=config, scenario=scenario, output=output)


def _build_isaac_runtime(*, app, assembly, plan, config, output, scenario, manifest):
    """Wire the measured IsaacPort + lane controllers around the validated assembly.

    Integration owns the port assembly and `assembly.sensing_sources`; the navigation,
    manipulation and inventory sensing controllers come from the lane-owned factories named
    in the runtime-config. Only runs inside the live app; never exercised by the CPU contract
    tests (which inject `_runtime`).
    """
    from .isaac_port import IsaacPort, TransportEnvelope
    from .recording import Recorder
    from twin.navigation.sensors import IsaacRangeSensor
    from .verification import InventoryVerifier
    from .sensing_sources import build_sensing_sources

    nav_factory = _resolve_factory(config, 'nav_factory')
    manip_factory = _resolve_factory(config, 'manip_factory')
    sensing_factory = _resolve_factory(config, 'sensing_factory')

    navigation = _build_navigation(plan, nav_factory(
        app=app, assembly=assembly, plan=plan, config=config, output=output))
    # manip_factory returns a fully-configured ManipulationBridge (it alone knows the
    # owns-sensor-io mode and per-station preparation); RuntimeAdapter leaves an existing
    # ManipulationBridge unwrapped, so do not re-wrap here.
    manipulation = manip_factory(
        app=app, assembly=assembly, plan=plan, config=config, output=output)
    # Build the integration-owned sensing bindings from the composed assembly + the camera
    # design + the manipulation motion port, and attach them so the camera-lane sensing_factory
    # resolves them. Must precede sensing_factory (which fails closed without them). The camera
    # design is sized against the MOBILE manifest (spec.arm_mount_z), not the environment manifest
    # run_mobile_restock validates against, so load it from the runtime-config.
    import json as _json
    mobile_manifest = _json.loads(Path(config['mobile_manifest']).read_text())
    # Transport (arm-stow) joint targets come from the mobile manifest's `transport_positions`
    # (the manip host builds `transport_joints` from it; the collect `stow` phase and the
    # outbound/return `arm_not_in_transport` gate both depend on it). Refuse closed here with a
    # clear message if it is absent, rather than letting the host subprocess fail with a cryptic
    # KeyError far from the mission boot. See docs/mobile_robot_rollwagen.md.
    require_transport_positions(mobile_manifest)
    # Every QR-decode support camera (source, shelf, cart) must render FULL-RES: the resolution
    # step-down does not survive a native RTX render for the decode-grade markers -- a native 0.6
    # source render (~4.3 px/module) failed to decode in 240 frames where 1.0 (~7.1 px/module)
    # decodes, and a resample is ~4.4x sharper than the native low-res render (camera-lane native
    # px/module floor; restock_camera.config.render_presets / DECODE_SUPPORTS). The opt-in `fast`
    # preset's reduced render_scale therefore applies ONLY to the non-decode cameras (mapper/nav/
    # arm); the decode support cameras stay at 1.0, and their baselines are commissioned at 1.0 too,
    # so commissioning._check_match stays satisfied at 1.0. build_sensing_sources filters these
    # mission locations to the cameras the design actually carries, so a superset is safe.
    _job = scenario.get('job', {})
    _decode_locations = tuple(dict.fromkeys(
        loc for loc in (config.get('collect_source_location') or _job.get('source_bin'),
                        config.get('transfer_shelf_location') or _job.get('shelf_id'),
                        'rollwagen') if loc))
    assembly.sensing_sources = build_sensing_sources(
        assembly=assembly, scenario=scenario, manifest=mobile_manifest,
        manip_runtime=manipulation.controller, pose_bridge_root=config['pose_bridge_root'],
        render_scale=float(config.get('render_scale', 1.0)),
        full_res_locations=_decode_locations)
    sensing = sensing_factory(app=app, assembly=assembly, plan=plan, config=config, output=output)
    capture = sensing['capture']
    recording_camera = sensing['recording_camera']
    # Hand the fixed-source collect-target provider (available only after build_sensing) to the
    # manipulation runtime, which reprojects it into the arm image for the grasp (collect stage).
    source_target_provider = sensing.get('source_target_provider')
    if source_target_provider is not None and hasattr(manipulation.controller, 'set_source_target_provider'):
        manipulation.controller.set_source_target_provider(source_target_provider)
    # Transfer destination: the camera lane's fixed-shelf slot provider (PR62), consumed by the
    # transfer stage (cart source -> shelf slot). Collect is unaffected if absent.
    shelf_target_provider = sensing.get('shelf_target_provider')
    if shelf_target_provider is not None and hasattr(manipulation.controller, 'set_shelf_target_provider'):
        manipulation.controller.set_shelf_target_provider(shelf_target_provider)

    paths = assembly.paths
    robot_root = paths['articulation'].rsplit('/', 1)[0] if '/' in paths['articulation'] else paths['articulation']
    cart_root = assembly.evidence['composition']['assets']['cart']['root']
    mobile_urdf = output / 'mobile-import.urdf'
    # Default the stow envelope radius to the robot's structural stow extent (measured 0.621 m at
    # the hitch, data/mobile-restock-v2/report.md): the old 0.55 m default could never contain the
    # arm's stowed binding, so the outbound/return `arm_not_in_transport` gate never cleared.
    transport = TransportEnvelope(mobile_urdf, plan.package_roots,
                                  radius=config.get('transport_radius_m', 0.65))
    # Mount the range sensor at the robot's lidar frame (forward of base_link), not the
    # base centre: a centre-mounted scan is self-occluded in every direction including
    # forward, and navigation refuses to drive on a blind forward sector. The lidar frame
    # is a fixed base_link child (rpy 0), so it shares the base yaw at the URDF offset.
    from twin.urdf_probe import parse_urdf as _parse_urdf
    _lidar_xyz = _parse_urdf(str(mobile_urdf)).joints_by_child['lidar_frame'].origin_xyz
    # The robot root is always self; the towed cart root is self ONLY while coupled (IsaacPort sets
    # range_sensor.coupled each tick from the measured latch). Otherwise rear beams off the coupled
    # cart read as a phantom obstacle behind the robot, blocking any turn-in-place/return maneuver.
    range_sensor = IsaacRangeSensor(
        get_sensor_pose=lambda: _lidar_pose(assembly.robot_body, _lidar_xyz[0], _lidar_xyz[1]),
        sensor_to_base=_lidar_xyz, self_paths=(robot_root,), coupled_self_paths=(cart_root,))
    recorder = Recorder(output / 'mission-frames', fps=plan.recording_fps)
    verifier = InventoryVerifier(capture, source=plan.verification_source)

    def on_action(action, job):
        if action == 'verify':
            # Begin verification on the SAME mission-relative clock the packets carry (tick*dt),
            # which InventoryVerifier.begin/__call__ compare scan stamps against. port.mission_time()
            # converts Isaac stage time back to mission time via the port's refreshed clock origin
            # (the former `port.origin` read assumed an attribute that never existed -> AttributeError
            # at the verify boundary, an immediate mission fail).
            verifier.begin(job, port.mission_time())

    # Physics ticks per recorded walkthrough frame. Lowering the cadence (default 2 ->
    # e.g. 6) both shrinks the video and, with the port's render gating, lets the
    # navigation-only stages skip most whole-scene renders -- the per-tick wall-clock cost
    # that otherwise blows the stage-pair cap. Decode-grade stages are unaffected (they
    # render every tick via SENSING_ACTIONS regardless of this stride).
    record_stride = int(config.get('record_stride', 2))
    # Physics ticks per rendered sensing frame (collect/transfer/verify). Rendering the whole scene
    # every tick drove collect/grasp to ~1:70 sim:wall (run-94); a modest stride keeps the sensing
    # consumers' sim-time freshness (controller max_age 0.5 s) while the render-paced grasp completes
    # inside the wall budget. Default 1 = legacy every-tick behaviour; the mission config sets it higher.
    sensing_stride = int(config.get('sensing_stride', 1))
    # Opt-in sensing execution gate (default off): pass the camera-lane set_render_suspended seam to
    # the port ONLY when config['sensing_execution_gate'] is on, so the port's phase-driven gate()
    # suspends the decode render during the reach. Off => seam is None and the port is byte-identical.
    gate_seam = sensing.get('set_render_suspended') if config.get('sensing_execution_gate') else None
    port = IsaacPort(
        world=assembly.world, robot=assembly.robot, robot_body=assembly.robot_body,
        cart_body=assembly.cart_body, mobile=assembly.mobile, range_sensor=range_sensor,
        transport=transport, contact=assembly.contact, hold_arm=assembly.command_gate.hold,
        verification=verifier, recording_camera=recording_camera, recorder=recorder,
        native_transport=assembly.transport, command_gate=assembly.command_gate,
        on_action=on_action, record_stride=record_stride,
        sensing_stride=sensing_stride, set_render_suspended=gate_seam,
        # Stationary-operation gate: real-rig defaults 5 mm / 10 mrad; the sim profile relaxes them
        # (firstmate msg 252 sim-only fidelity relaxation) to tolerate the harmless ~11 mrad contact
        # chassis-pivot impulse. Absent from config => real-rig defaults (real path untouched).
        stationary_dxy=float(config.get('base_stationary_dxy_m', 0.005)),
        stationary_dyaw=float(config.get('base_stationary_dyaw_rad', 0.01)))

    recording = {'frames_dir': str(output / 'mission-frames'), 'fps': plan.recording_fps}
    closers = (verifier.stop, recorder.close) if hasattr(recorder, 'close') else (verifier.stop,)
    return {'navigation': navigation, 'mobile': assembly.mobile, 'manipulation': manipulation,
            'port': port, 'closers': closers, 'recording': recording}


def _base_pose(body):
    from .isaac_port import body_measurement
    pose, _z, _v, _w = body_measurement(body)
    return (pose.x, pose.y, pose.yaw)


def _lidar_pose(body, ox, oy):
    """World (x, y, yaw) of a base_link-fixed sensor frame at base-frame offset (ox, oy)."""
    import math
    from .isaac_port import body_measurement
    pose, _z, _v, _w = body_measurement(body)
    c, s = math.cos(pose.yaw), math.sin(pose.yaw)
    return (pose.x + ox * c - oy * s, pose.y + ox * s + oy * c, pose.yaw)
