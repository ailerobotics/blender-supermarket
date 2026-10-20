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
import os
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


# Opt-in DEV render economy (devloop lane). A full-fidelity production run renders the
# walkthrough camera every ``record_stride`` ticks and the sensing camera every
# ``sensing_stride`` ticks; on a render-paced mission that whole-scene render is the
# dominant per-tick wall cost (run-248: 340 s walkthrough + 197 s sensing of an 800 s
# loop). During iteration the captain's deliverable video and perception fidelity are not
# the point -- reaching the arm phase fast is. This widens both strides for a dev run ONLY
# when the launch environment opts in, so the default (unset) behaviour is byte-identical.
#
#   SUPERMARKET_DEV_RENDER_ECONOMY unset / "" / "0" / "off" / "false"  -> no change
#   SUPERMARKET_DEV_RENDER_ECONOMY = "1" / "on" / "true"               -> dev defaults
#   SUPERMARKET_DEV_RENDER_ECONOMY = "record=20,sensing=8"             -> explicit per-knob
#
# Strides only ever WIDEN (max against the configured value) -- the knob never renders more
# often than the run already would. sensing_stride is capped so a stride-paced sensing frame
# still lands inside the controller's freshness window (max_age 0.5 s sim at the nominal
# 1/60 s step -> < 30 ticks); walkthrough has no freshness constraint but keeps a finite
# stride so the recorder still captures enough distinct frames to encode a (low-fps) clip.
_DEV_ECONOMY_RECORD = 30        # 2 fps walkthrough at 60 Hz physics (the configured base is ~6)
_DEV_ECONOMY_SENSING = 15       # 15/60 = 0.25 s sim between sensing renders < 0.5 s freshness
_DEV_ECONOMY_SENSING_CAP = 20   # never stride past the freshness window (0.33 s, margin under 30)


def _truthy(value):
    return str(value).strip().lower() in ('1', 'on', 'true', 'yes')


def resolve_render_strides(config, env):
    """(record_stride, sensing_stride, note): the render cadence the port should run at.

    Pure (no Isaac, no I/O): reads the configured strides and, iff the launch environment
    opts into DEV render economy, widens them for a faster dev iteration loop. ``note`` is a
    one-line marker string to log when the economy is active, else ``None``. Default path
    (no opt-in) returns exactly ``config``'s strides so a production run is unchanged.
    """
    base_record = int(config.get('record_stride', 2))
    base_sensing = int(config.get('sensing_stride', 1))
    spec = (env or {}).get('SUPERMARKET_DEV_RENDER_ECONOMY', '')
    spec = '' if spec is None else str(spec).strip()
    if not spec or spec.lower() in ('0', 'off', 'false', 'no'):
        return base_record, base_sensing, None

    dev_record, dev_sensing = _DEV_ECONOMY_RECORD, _DEV_ECONOMY_SENSING
    if not _truthy(spec):
        # explicit per-knob spec, e.g. "record=20,sensing=8"; unknown keys are ignored,
        # a malformed value fails closed to the dev defaults rather than the production rate.
        for item in spec.replace(';', ',').split(','):
            if '=' not in item:
                continue
            key, _, val = item.partition('=')
            try:
                num = int(val.strip())
            except ValueError:
                continue
            if num < 1:
                continue
            if key.strip().lower() in ('record', 'record_stride', 'walkthrough'):
                dev_record = num
            elif key.strip().lower() in ('sensing', 'sensing_stride'):
                dev_sensing = num

    record = max(base_record, dev_record)
    # The freshness cap bounds only the knob's own target, never the configured base: the knob
    # widens toward the cap but never narrows a run that already renders coarser than it.
    sensing = max(base_sensing, min(dev_sensing, _DEV_ECONOMY_SENSING_CAP))
    note = (f'DEV_RENDER_ECONOMY record_stride={record} sensing_stride={sensing} '
            f'(base record={base_record} sensing={base_sensing}; '
            f'SUPERMARKET_DEV_RENDER_ECONOMY={spec!r})')
    return record, sensing, note


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


def derive_place_approach(cart_xy, cart_yaw, standoff, *,
                          mouth_axis_body=(1.0, 0.0), deck_center_body_xy=(0.0, 0.0)):
    """fm390: a `place_approach` BASE pose (x, y, yaw) in front of the cart's open-face mouth, facing it.

    Derived from the cart's MEASURED pose (not a bench guess): the deck centre is
    ``cart_xy + R(cart_yaw)*deck_center_body_xy``; the base sits ``standoff`` metres out along the mouth
    axis (``R(cart_yaw)*mouth_axis_body``, the open-face direction in the cart body frame) and faces back
    toward the mouth. ``standoff`` is the arm-reach standoff proven by ``scripts/check_place_approach_reach.py``
    (deck reachable AND base clear of the drawbar). This is the single reproducible source for the value a
    mission's ``mission_poses['place_approach']`` carries; the cart is static pre-couple, so its staged
    pose equals the live /Rollwagen prim pose.
    """
    c, s = math.cos(cart_yaw), math.sin(cart_yaw)
    def rot(v):
        return (c * v[0] - s * v[1], s * v[0] + c * v[1])
    dcx, dcy = rot(deck_center_body_xy)
    deck = (cart_xy[0] + dcx, cart_xy[1] + dcy)
    mx, my = rot(mouth_axis_body)
    norm = math.hypot(mx, my) or 1.0
    mx, my = mx / norm, my / norm
    return (deck[0] + mx * standoff, deck[1] + my * standoff, math.atan2(-my, -mx))


def derive_place_approach_stage(place_approach, pickup):
    """fm390 nav proof #2: the staging waypoint the base drives to BEFORE place_approach.

    A DIRECT diagonal drive from the pick pose to place_approach clips the parked cart -- the base must
    translate out along the mouth axis (clear of the cart, keeping the pick-pose heading) BEFORE turning
    to face the mouth (``scripts/check_place_approach_nav.py``). The staging pose sits at place_approach's
    standoff distance (its x) but on the pick pose's lateral line, at the pick heading, so leg 1 is a
    straight translate that never swings a corner into the cart; leg 2 then turns + closes onto the mouth.
    """
    return (place_approach[0], pickup[1], pickup[2])


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
    # Required nav goals, plus OPTIONAL waypoints: the couple waypoints `couple_approach` (east of/clear
    # of the pinned cart) and `couple_standoff` (on the cart hitch axis, east of the dock) the couple
    # stage navigates to before the fine-dock insert; and `place_approach` (fm390: in front of the
    # rollwagen's open +x face, facing the mouth, at an arm-reach standoff) the collect stage drives to
    # WITH the carton carried before lowering it onto the deck, instead of reaching around from pickup.
    _optional_extra = {'couple_approach', 'couple_standoff', 'place_approach', 'place_approach_stage',
                       'cart_parking', 'cart_parking_backaway'}
    if not isinstance(poses_in, dict) or not set(NAV_GOALS) <= set(poses_in) \
            or not set(poses_in) <= set(NAV_GOALS) | _optional_extra:
        raise ValueError(f'config.mission_poses must provide {NAV_GOALS} '
                         f'(plus optional {sorted(_optional_extra)})')
    poses = {}
    for name in NAV_GOALS + tuple(n for n in sorted(_optional_extra) if n in poses_in):
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
    # Opt-in warm-session reset: a callable ``reset(attempt, now)`` that re-seats the live scene
    # to its commissioned initial state and re-validates the baseline (lifetime + session binding)
    # so the mission can be re-run in the SAME Isaac session without a re-boot/re-commission. None
    # unless config['warm_reset'] names a provider; the launcher runs a single attempt without it.
    warm_reset: object = None

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


def _build_navigation(plan, controller, detection_source=None):
    return NavigationBridge(controller, nav2=plan.nav_mode == 'nav2',
                            sensor_navigation=plan.nav_mode == 'sensor',
                            detection_source=detection_source)


def _build_detection_source(*, app, assembly, plan, config, output):
    """Build the opt-in moving-obstacle detection feed from the runtime-config.

    `navigation.dynamic_obstacle_source` names a lane-owned builder as
    `module:function`, called `fn(app, assembly, plan, config, output)` and expected to
    return a `callable(now) -> [navigation.dynamic.Detection]` (or None). Absent =>
    None => the sensor route is fed `detections=None`, byte-identical to a run with no
    moving-obstacle layer. Rejected for non-sensor nav modes (the nav2 route takes moving
    obstacles through its ROS costmap, not this in-process feed).
    """
    nav = config.get('navigation')
    ref = nav.get('dynamic_obstacle_source') if isinstance(nav, dict) else None
    if ref is None:
        return None
    if plan.nav_mode != 'sensor':
        raise ValueError('navigation.dynamic_obstacle_source applies only to nav_mode '
                         '"sensor"; the nav2 route uses its ROS costmap obstacle layer')
    if not isinstance(ref, str) or ':' not in ref:
        raise ValueError('navigation.dynamic_obstacle_source must name a lane-owned '
                         'builder as "module:function"')
    module, name = ref.split(':', 1)
    builder = getattr(importlib.import_module(module), name)
    return builder(app=app, assembly=assembly, plan=plan, config=config, output=output)


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

    # Start-phase shortcut (dev iteration only): synthesize the skipped prefix's physical end
    # state before any mission tick. Guarded by config['start_at']; absent => untouched.
    if config.get('start_at'):
        from .shortcut import apply_start_state
        apply_start_state(assembly, config, scenario)

    build_runtime = _runtime or _build_isaac_runtime
    parts = build_runtime(app=app, assembly=assembly, plan=plan, config=config, output=output,
                          scenario=scenario, manifest=manifest)
    navigation, mobile, manipulation, port, closers, recording = (
        parts['navigation'], parts['mobile'], parts['manipulation'], parts['port'],
        parts.get('closers', ()), parts.get('recording', {}))

    # Opt-in mobile store-sensor hook (twin.route_alerts). Built ONLY when the config
    # carries a `route_alerts` section; absent => route_alerts=None and the adapter is
    # byte-identical. The hook is self-disabling and never raises into the tick; the
    # frame_builder taps the drive camera and REQUIRES a people-mask source (GDPR), so
    # it no-ops until the humans lane's detections are wired via config['agents_factory'].
    route_alerts = None
    if config.get('route_alerts'):
        from twin.route_alerts.mission import build_from_config, recording_camera_frame_builder
        agents_provider = None
        if config.get('agents_factory'):
            agents_provider = _resolve_factory(config, 'agents_factory')(
                app=app, assembly=assembly, parts=parts, plan=plan, config=config, output=output)
        frame_builder = recording_camera_frame_builder(config, agents_provider=agents_provider)
        route_alerts = build_from_config(config, output_dir=output, frame_builder=frame_builder,
                                         robot_id=config.get('route_alerts_robot_id', 'shelfbot'))

    # Opt-in final fine-docking servo (twin.restock.fine_dock). Built ONLY when the config
    # carries a `fine_dock` block; absent => None and the couple stage attaches directly
    # (byte-identical). Sim and real-rig creep/gain/tolerance values live in separate blocks.
    from twin.restock.fine_dock import build as build_fine_dock
    fine_dock = build_fine_dock(config, spec=getattr(mobile, 'spec', None))

    adapter = RuntimeAdapter(navigation, mobile, manipulation, port, plan.poses,
                             fixture_mode=False, goal_articulation=plan.goal_articulation,
                             route_alerts=route_alerts, fine_dock=fine_dock,
                             hold_cart_through_couple=bool(config.get('hold_cart_through_couple')),
                             couple_static_drive=bool(config.get('couple_static_drive')),
                             uncouple_on_return=bool(config.get('uncouple_on_return')))

    # Opt-in warm-session reset provider (dev iteration): config['warm_reset'] names a
    # lane-owned builder `module:function` called once here with the live scene; it returns the
    # per-attempt `reset(attempt, now)` callable (re-seat + baseline re-validation), or None to
    # leave warm retry unavailable. The native World reset + robot re-seat + sensing-clock rebase
    # live where the stage/camera do (integration + camera/stations lanes), so the Isaac-side reset
    # is provided here rather than hard-coded into the launcher.
    warm_reset = None
    if config.get('warm_reset'):
        build_warm_reset = _resolve_factory(config, 'warm_reset')
        warm_reset = build_warm_reset(app=app, assembly=assembly, parts=parts, plan=plan,
                                      config=config, output=output)
    return Bundle(adapter=adapter, recording=recording, _closers=tuple(closers),
                  warm_reset=warm_reset)


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
    from ..restock_manipulation.cad_tool import ARM_JOINTS as _ARM_JOINTS   # fm347/348(1) phase-drive
    from .recording import Recorder
    from twin.navigation.sensors import IsaacRangeSensor
    from .verification import InventoryVerifier
    from .sensing_sources import build_sensing_sources

    nav_factory = _resolve_factory(config, 'nav_factory')
    manip_factory = _resolve_factory(config, 'manip_factory')
    sensing_factory = _resolve_factory(config, 'sensing_factory')

    detection_source = _build_detection_source(
        app=app, assembly=assembly, plan=plan, config=config, output=output)
    navigation = _build_navigation(plan, nav_factory(
        app=app, assembly=assembly, plan=plan, config=config, output=output),
        detection_source=detection_source)
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
    # The handling unit rides ON the cart deck once loaded. It is a separate rigid body (not under the
    # cart root), and the lidar's upper scan height (heights=(0.15,0.5,1.0)) strikes it on the deck
    # (~1.0 m), so without this it reads as a phantom nav obstacle sitting on/behind the cart whenever
    # the cart is masked as self -- blocking couple's dock approach (ds-003: 30 unmasked returns at the
    # cart's near face were the carton, not the cart, which IS under cart_root) and the coupled tow. Mask
    # the loaded carton together with the cart: it is cargo that moves with the train, not an obstacle.
    cart_self_paths = (cart_root, assembly.evidence['composition']['stock'])
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
    # fm393 carry self-mask radius: while the port flags the reposition carry, returns closer than this
    # are masked as self (the carry-fold arm/carton ring); beyond it stays a real obstacle. 0.0 => off =>
    # byte-identical (also inert unless the port sets range_sensor.carry_active during the carry).
    _carry_self_radius = float(config.get('carry_self_radius_m', 0.0))
    print('CARRY_SELF_MASK carry_self_radius_m=%.3f (0 => off; masks near-field self only while carrying)'
          % _carry_self_radius, flush=True)
    range_sensor = IsaacRangeSensor(
        get_sensor_pose=lambda: _lidar_pose(assembly.robot_body, _lidar_xyz[0], _lidar_xyz[1]),
        sensor_to_base=_lidar_xyz, self_paths=(robot_root,), coupled_self_paths=cart_self_paths,
        carry_self_radius=_carry_self_radius)
    # fm376: no walkthrough render product (recording_camera None) => no Recorder; the port then
    # self-disables the record cadence and the walkthrough is composed offline from sensing frames.
    recorder = Recorder(output / 'mission-frames', fps=plan.recording_fps) if recording_camera is not None else None
    verifier = InventoryVerifier(capture, source=plan.verification_source)

    def on_action(action, job):
        if action == 'verify':
            # Begin verification on the SAME mission-relative clock the packets carry (tick*dt),
            # which InventoryVerifier.begin/__call__ compare scan stamps against. port.mission_time()
            # converts Isaac stage time back to mission time via the port's refreshed clock origin
            # (the former `port.origin` read assumed an attribute that never existed -> AttributeError
            # at the verify boundary, an immediate mission fail).
            verifier.begin(job, port.mission_time())

    # Render cadence (both strides).
    #  - record_stride: physics ticks per recorded walkthrough frame. Lowering it (default 2
    #    -> e.g. 6) both shrinks the video and, with the port's render gating, lets the
    #    navigation-only stages skip most whole-scene renders -- the per-tick wall-clock cost
    #    that otherwise blows the stage-pair cap.
    #  - sensing_stride: physics ticks per rendered sensing frame (collect/transfer/verify).
    #    Rendering the whole scene every tick drove collect/grasp to ~1:70 sim:wall (run-94); a
    #    modest stride keeps the sensing consumers' sim-time freshness (controller max_age 0.5 s)
    #    while the render-paced grasp completes inside the wall budget. Default 1 = legacy
    #    every-tick behaviour; the mission config sets it higher.
    # SUPERMARKET_DEV_RENDER_ECONOMY (opt-in, dev runs only) widens BOTH to cut the dominant
    # whole-scene render cost; unset leaves the configured cadence byte-identical.
    record_stride, sensing_stride, _economy_note = resolve_render_strides(config, os.environ)
    if _economy_note:
        print(_economy_note, flush=True)
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
        stationary_dyaw=float(config.get('base_stationary_dyaw_rad', 0.01)),
        # Ideal simulated proprioception: a sim run may tighten the nav localization pad from the
        # conservative real-rig 0.05 m so it does not falsely fail free(start) against a trailing
        # obstacle. Absent => the real-rig default (untouched). See isaac_port.position_uncertainty.
        position_uncertainty=float(config.get('sim_odometry_position_uncertainty', 0.05)),
        # fm347/348 (1): opt-in phase-dependent arm drive (soft tuck-settle / stiff loaded-place).
        # Absent config['phase_drive'] => None => the assemble() drive stays (byte-identical).
        phase_drive=config.get('phase_drive'),
        # fm380: stage-scoped sensing -- destroy the collect-only station cameras' render products at
        # the collect->couple boundary to return their VRAM. SIM-ONLY, opt-in; unset => None => never
        # torn down => byte-identical. The cart camera + nav/nvblox + walkthrough recorder are untouched.
        stage_scoped_teardown=(assembly.sensing_sources.get('stage_scoped_teardown')
                               if os.environ.get('RESTOCK_STAGE_SCOPED_SENSING') else None),
        # fm384 phase-aware collect capture: at grasp-confirm, switch the capture to destination-only
        # (real fresh cart render) + destroy the source render product. None unless the knob is set.
        release_source_capture=(sensing.get('release_source_capture')
                                if os.environ.get('RESTOCK_PHASE_AWARE_COLLECT_CAPTURE') else None),
        arm_joints=_ARM_JOINTS)

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
