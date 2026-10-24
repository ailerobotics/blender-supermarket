"""Navigation lane factory for the mission bundle (`nav_factory`).

The integration bundle (`twin.restock.isaac_bundle:create`) resolves each lane's
controller from the runtime-config as a `module:function` reference and calls it

    nav_factory(app=app, assembly=assembly, plan=plan, config=config, output=output)

expecting the lane's validated navigation runtime object back. That object is then
wrapped by `twin.restock.runtime.NavigationBridge`, which drives it per
`plan.nav_mode`:

    reference -> step(dt, odometry, observation, now=...)
    nav2      -> publish_sensors(odometry, scan); step(dt, odometry, scan,
                 coupled=, arm_transport=, articulation=)
    sensor    -> step(dt, wheel_angles, scan, now=, coupled=, arm_transport=,
                 articulation=)

and for every mode `start(goal, coupled[, goal_articulation=])` and `cancel()`.

This factory builds the shared measured-state `ArticulatedController` from the
STATIC navigation manifest (the same conservative obstacle map used for command
gating and CPU validation; live obstacles are always sensed), then selects the
reference controller, the in-process sensor controller, or the domain-88 Nav2
overlay. It NEVER enables test fixtures: the returned controller only accepts
`wheel_odometry_localized`/`simulated_lidar` provenance, so no ground-truth or
fixture input can drive the base in production. Constructing the Nav2 overlay
requires ROS loopback isolation (domain 88) and is the only path that touches
ROS; it is exercised by integration inside the live app, never by CPU tests.
"""
from __future__ import annotations

import inspect
import os
from pathlib import Path

from .scenario import environment_grid
from .articulated import ArticulatedController

NAV_MODES = ('reference', 'nav2', 'sensor')


def _load(section, inline_key, path_key):
    """Return an inline object or the JSON at its path; None when neither is set."""
    if inline_key in section:
        value = section[inline_key]
        if not isinstance(value, dict):
            raise ValueError(f'navigation.{inline_key} must be an object')
        return value
    if path_key in section:
        import json
        return json.loads(Path(section[path_key]).read_text())
    return None


def _build_grid(nav):
    manifest = _load(nav, 'environment', 'environment_path')
    if manifest is None:
        raise ValueError('navigation config requires "environment" (manifest object) '
                         'or "environment_path"')
    scenario = _load(nav, 'scenario', 'scenario_path')
    camera_config = _load(nav, 'camera_config', 'camera_config_path')
    return environment_grid(manifest, scenario, camera_config)


def _build_spec(nav):
    from twin.mobile import MobileSpec
    spec_cfg = nav.get('mobile_spec')
    if spec_cfg is None:
        return MobileSpec()
    if not isinstance(spec_cfg, dict):
        raise ValueError('navigation.mobile_spec must be an object')
    return MobileSpec(**spec_cfg)


def _build_docking(nav):
    """Build the opt-in dock-approach speed cap from `navigation.docking`.

    Returns None when the key is absent (default-off, byte-identical run). The config
    object maps onto DockingApproach fields; it caps the commanded speed near a named
    dock goal (e.g. the cart hitch-meet pose at couple) so the braking disk shrinks
    honestly without touching the footprint, any margin, or the real-rig defaults.
    """
    cfg = nav.get('docking')
    if cfg is None:
        return None
    if not isinstance(cfg, dict):
        raise ValueError('navigation.docking must be an object')
    from .core import DockingApproach
    try:
        goal = tuple(float(v) for v in cfg['goal'])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError('navigation.docking needs goal (x, y, yaw)') from exc
    extra = {k: float(cfg[k]) for k in ('goal_tolerance',) if k in cfg}
    try:
        speed = float(cfg['speed'])
        approach_distance = float(cfg['approach_distance'])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError('navigation.docking needs speed and approach_distance') from exc
    return DockingApproach(goal=goal, speed=speed,
                           approach_distance=approach_distance, **extra)


# fm386: the SIM-clock nav bounds (couple/outbound/park). stale_after is the per-frame sensor-staleness
# (now-odom/obs stamp > stale_after -> 'stale_sensing'); on a render-paced tow/dock the stamp can lag
# past the tight 0.3 s default -- the sibling of the 1.0 s /sensors watchdog that tripped at retreat.
# blocked_timeout / goal_timeout are the no-progress / overall SIM-clock caps. Each SIM-only env-
# overridable; the env WINS over a config value (never shadowed -- the run-290 lesson); unset => the
# config value or the SensorNavigation ctor default, byte-identical. (max_speed/max_angular/
# brake_acceleration are NOT time bounds -> config-only, untouched.)
_NAV_SIM_BOUNDS = {'stale_after': ('RESTOCK_NAV_STALE_AFTER_S', 0.05, 30.0),
                   'blocked_timeout': ('RESTOCK_NAV_BLOCKED_TIMEOUT_S', 1.0, 600.0),
                   'goal_timeout': ('RESTOCK_NAV_GOAL_TIMEOUT_S', 10.0, 1800.0)}


def _nav_sim_env(var, lo, hi):
    """Clamped sim-clock bound from the env, or None when unset/malformed (=> caller keeps config/default)."""
    try:
        return max(lo, min(hi, float(os.environ[var])))
    except (KeyError, TypeError, ValueError):
        return None


def _strict_bool(nav, key):
    """A default-off boolean switch; refuse a non-boolean (e.g. the string "false") rather than coerce."""
    value = nav.get(key, False)
    if not isinstance(value, bool):
        raise ValueError(f'navigation.{key} must be a boolean')
    return value


def _controller_kwargs(nav):
    kwargs = {}
    for limit in ('goal_timeout', 'blocked_timeout', 'max_speed', 'max_angular',
                  'brake_acceleration', 'stale_after'):
        if limit in nav:
            # `blocked_timeout` is the avoidance patience: how long the base will
            # stop and wait on a blocked aisle (e.g. a crossing shopper) before
            # giving up the goal. Raising it lets the mission wait out a transient
            # dynamic block and resume rather than fail. Default is unchanged.
            kwargs[limit] = nav[limit]
    # fm386: SIM-clock bound env overrides win over config/default (unset => unchanged, byte-identical).
    for limit, (var, lo, hi) in _NAV_SIM_BOUNDS.items():
        override = _nav_sim_env(var, lo, hi)
        if override is not None:
            kwargs[limit] = override
    docking = _build_docking(nav)
    if docking is not None:
        kwargs['docking'] = docking
    # Direction-aware braking zone (default on). Explicit False restores the legacy
    # omnidirectional stop disk; the footprint radius/margins are identical either way.
    if 'directional_zone' in nav:
        if not isinstance(nav['directional_zone'], bool):
            raise ValueError('navigation.directional_zone must be a boolean')
        kwargs['directional_zone'] = nav['directional_zone']
    # fm482: opt-in arrival XY hysteresis for the uncoupled reference legs (the couple standoff
    # dither). Absent/0 => the legacy per-tick arrival test (byte-identical). See core.Controller.
    if 'arrival_xy_hysteresis_m' in nav:
        value = nav['arrival_xy_hysteresis_m']
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError('navigation.arrival_xy_hysteresis_m must be a number of metres')
        kwargs['arrival_xy_hysteresis'] = float(value)
    # Production never accepts fixtures: the knob is intentionally not exposed.
    if nav.get('allow_test_fixtures'):
        raise ValueError('allow_test_fixtures is forbidden in a production bundle')
    return kwargs


def _build_lane_graph(nav, grid, spec):
    """Build + expand the opt-in authored tow lanes from `navigation.tow_lanes`.

    Returns None when the key is absent (default-off, byte-identical run: coupled drives
    free-plan as before). When present, every lane is expanded ONCE against the static
    navigation map here, at build time (never under the per-tick budget); a lane that cannot
    be realised raises, so a broken lane fails the build rather than the mission.
    """
    cfg = nav.get('tow_lanes')
    if cfg is None:
        return None
    from .lanes import TowLaneGraph
    from .articulated import ArticulatedMap
    graph = TowLaneGraph.from_config(cfg)
    if graph is None or not graph.lanes:
        return None
    build = nav.get('tow_lane_build', {})
    if not isinstance(build, dict):
        raise ValueError('navigation.tow_lane_build must be an object')
    kw = {}
    if 'build_margin' in build:
        kw['build_margin'] = float(build['build_margin'])
    if 'build_timeout' in build:
        kw['build_timeout'] = float(build['build_timeout'])
    graph.expand_all(ArticulatedMap(grid, spec), **kw)
    return graph


def _build_dynamic_layer(nav):
    """Build the opt-in moving-obstacle layer from `navigation.dynamic_obstacles`.

    Returns None when the key is absent (default-off, byte-identical run). The
    config object maps directly onto DynamicObstacleConfig fields; the caller
    supplies detections per frame from the detection topic or scenario.
    """
    cfg = nav.get('dynamic_obstacles')
    if cfg is None:
        return None
    if not isinstance(cfg, dict):
        raise ValueError('navigation.dynamic_obstacles must be an object')
    from .dynamic import DynamicObstacleLayer, DynamicObstacleConfig
    return DynamicObstacleLayer(DynamicObstacleConfig(**cfg))


def _build_sensor(controller, grid, nav):
    from .runtime import SensorNavigation
    from .sensors import WheelOdometry, MapLocalizer
    cfg = nav.get('sensor')
    if not isinstance(cfg, dict):
        raise ValueError('nav_mode "sensor" requires a navigation.sensor calibration object')
    try:
        radius = float(cfg['wheel_radius'])
        separation = float(cfg['wheel_separation'])
        initial_pose = tuple(float(v) for v in cfg['initial_pose'])
        initial_map_pose = tuple(float(v) for v in cfg['initial_map_pose'])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError('navigation.sensor needs wheel_radius, wheel_separation, '
                         'initial_pose (x,y,yaw) and initial_map_pose (x,y,yaw)') from exc
    if len(initial_pose) != 3 or len(initial_map_pose) != 3:
        raise ValueError('navigation.sensor poses must each be (x, y, yaw)')
    wheels = WheelOdometry(radius, separation, initial_pose)
    localizer_kwargs = {}
    if 'localizer_max_error' in cfg:
        localizer_kwargs['max_error'] = float(cfg['localizer_max_error'])
    localizer = MapLocalizer(grid, initial_map_pose, **localizer_kwargs)
    return SensorNavigation(controller, wheels, localizer,
                            dynamic_layer=_build_dynamic_layer(nav))


def _build_nav2(controller, grid, output, nav, adapter_cls):
    from .nav2 import require_isolation, ArticulatedNav2Adapter
    # Domain-88 loopback isolation is mandatory before any ROS object is created.
    require_isolation()
    nav2_cfg = nav.get('nav2', {})
    if not isinstance(nav2_cfg, dict):
        raise ValueError('navigation.nav2 must be an object')
    if nav2_cfg.get('emit_assets'):
        from .nav2 import write_assets
        topic = nav2_cfg.get('nvblox_topic')
        write_assets(Path(output), grid, controller.footprint,
                     articulated_spec=controller.spec, nvblox_topic=topic,
                     template=nav2_cfg.get('params_template'))
    cls = adapter_cls or ArticulatedNav2Adapter
    return cls(controller)


# --- bridge-contract self-check -------------------------------------------------

_SENTINEL = object()


def _accepts(obj, name, args, kwargs):
    """True iff obj.name is callable and binds the given sample call."""
    func = getattr(obj, name, None)
    if not callable(func):
        return False
    try:
        inspect.signature(func).bind(*args, **kwargs)
        return True
    except TypeError:
        return False


def _require_bridge_contract(obj, mode):
    """Prove the returned object exposes EXACTLY the calls NavigationBridge makes.

    Signature-level verification only (no side effects); fail-closed so a wrong
    object is refused at build time rather than mid-mission.
    """
    s = _SENTINEL
    if not (_accepts(obj, 'start', (s, s), {}) and _accepts(obj, 'start', (s, s), {'goal_articulation': s})):
        raise ValueError('navigation object start() must accept (goal, coupled, goal_articulation=)')
    if not _accepts(obj, 'cancel', (), {}):
        raise ValueError('navigation object must expose cancel()')
    if mode == 'reference':
        if not _accepts(obj, 'step', (s, s, s), {'now': s}):
            raise ValueError('reference controller step() must accept (dt, odometry, observation, now=)')
    elif mode == 'sensor':
        if not _accepts(obj, 'step', (s, s, s),
                        {'now': s, 'coupled': s, 'arm_transport': s, 'articulation': s}):
            raise ValueError('sensor controller step() must accept '
                             '(dt, wheel_angles, scan, now=, coupled=, arm_transport=, articulation=)')
    elif mode == 'nav2':
        if not _accepts(obj, 'publish_sensors', (s, s), {}):
            raise ValueError('nav2 controller must expose publish_sensors(odometry, scan)')
        if not _accepts(obj, 'step', (s, s, s), {'coupled': s, 'arm_transport': s, 'articulation': s}):
            raise ValueError('nav2 controller step() must accept '
                             '(dt, odometry, scan, coupled=, arm_transport=, articulation=)')
    return obj


def nav_factory(*, app, assembly, plan, config, output, _nav2_adapter=None):
    """Build the validated navigation controller for the bundle's nav_mode.

    `_nav2_adapter` is a CPU-test injection seam for the ROS Nav2 overlay class;
    production leaves it None and uses the real `ArticulatedNav2Adapter`.
    """
    mode = getattr(plan, 'nav_mode', None)
    if mode not in NAV_MODES:
        raise ValueError(f'plan.nav_mode must be one of {NAV_MODES}')
    if assembly is None:
        raise ValueError('navigation factory requires the live assembly in production')
    if not isinstance(config, dict):
        raise ValueError('runtime-config must be an object')
    nav = config.get('navigation')
    if not isinstance(nav, dict):
        raise ValueError('runtime-config.navigation must be an object with the '
                         'static environment manifest and controller options')

    grid = _build_grid(nav)
    spec = _build_spec(nav)
    controller = ArticulatedController(grid, spec,
                                       require_temporal_map=bool(nav.get('require_temporal_map', False)),
                                       planner_timeout=nav.get('planner_timeout_s'),
                                       lane_graph=_build_lane_graph(nav, grid, spec),
                                       lane_static_tol=float(nav.get('tow_lane_static_tol_m', 0.0)),
                                       lane_trust_static=bool(nav.get('tow_lane_trust_static', False)),
                                       lane_reacquire_along_route=bool(nav.get('tow_lane_reacquire_along_route', False)),
                                       lane_cart_yaw_tol=float(nav.get('tow_lane_cart_yaw_tol', 0.20)),
                                       lane_curve_speed_floor=float(nav.get('tow_lane_curve_speed_floor', 0.25)),
                                       lane_max_curvature=(float(nav['tow_lane_max_curvature'])
                                                           if 'tow_lane_max_curvature' in nav else None),
                                       lane_coast_refusals=bool(nav.get('tow_lane_coast_refusals', False)),
                                       lane_trust_unknown=bool(nav.get('tow_lane_trust_unknown', False)),
                                       measured_turn_fault_s=nav.get('measured_turn_fault_s'),
                                       lane_track_tol=float(nav.get('tow_lane_track_tol_m', 0.25)),
                                       lane_track_envelope=bool(nav.get('tow_lane_track_envelope', False)),
                                       lane_cart_yaw_rebase=bool(nav.get('tow_lane_cart_yaw_rebase', False)),
                                       coupled_creep_escape=_strict_bool(nav, 'coupled_creep_escape'),
                                       coupled_creep_floor=float(nav.get('coupled_creep_floor_mps', 0.03)),
                                       coupled_progress_escape=_strict_bool(nav, 'coupled_progress_escape'),
                                       coupled_progress_min_m=float(nav.get('coupled_progress_min_m', 0.05)),
                                       coupled_progress_escape_s=float(nav.get('coupled_progress_escape_s', 3.0)),
                                       coupled_progress_fault_s=float(nav.get('coupled_progress_fault_s', 8.0)),
                                       coupled_progress_recovery=_strict_bool(nav, 'coupled_progress_recovery'),
                                       coupled_breakaway_speed_mps=float(nav.get('coupled_breakaway_speed_mps', 0.18)),
                                       coupled_breakaway_window_s=float(nav.get('coupled_breakaway_window_s', 2.0)),
                                       coupled_breakaway_pulse_s=float(nav.get('coupled_breakaway_pulse_s', 0.5)),
                                       coupled_breakaway_attempts=int(nav.get('coupled_breakaway_attempts', 3)),
                                       coupled_breakaway_rock=_strict_bool(nav, 'coupled_breakaway_rock'),
                                       coupled_breakaway_launch_mps=float(nav.get('coupled_breakaway_launch_mps', 0.30)),
                                       lane_acquire_timeout_s=float(nav.get('tow_lane_acquire_timeout_s', 0.0)),
                                       tow_lane_goal_stop=bool(nav.get('tow_lane_goal_stop', False)),
                                       coupled_heading_hold=_strict_bool(nav, 'coupled_heading_hold'),
                                       heading_hold_bound_deg=float(nav.get('heading_hold_bound_deg', 1.0)),
                                       heading_hold_creep_mps=float(nav.get('heading_hold_creep_mps', 0.14)),
                                       heading_hold_straight_kappa=float(nav.get('heading_hold_straight_kappa', 0.08)),
                                       **_controller_kwargs(nav))

    if 'dynamic_obstacles' in nav and mode != 'sensor':
        # The in-process layer is folded into the sensor route's Observation. The
        # Nav2 route takes moving obstacles through its ROS costmap obstacle/nvblox
        # layer instead, so refuse rather than silently ignore the config here.
        raise ValueError('navigation.dynamic_obstacles applies only to nav_mode "sensor"; '
                         'the nav2 route uses its ROS costmap obstacle/nvblox layer')

    if mode == 'reference':
        built = controller
    elif mode == 'sensor':
        built = _build_sensor(controller, grid, nav)
    else:
        built = _build_nav2(controller, grid, output, nav, _nav2_adapter)
    return _require_bridge_contract(built, mode)
