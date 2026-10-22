"""Integration adapter for navigation, physical coupling and sensor-led transfer.

No simulator imports or fabricated sensor data. A runtime port steps physics,
acquires observations and applies gripper controls. The controllers supplied by
the sibling modules are injected after a guarded simulator boot.
"""
from dataclasses import asdict, dataclass, replace
import math
import os
import time
from ..restock_camera.freshness import require_fresh
from ..restock_manipulation import ground_truth_collect as gtc
from typing import Protocol

from .mission import Result
from .staged_start import SettleHold

# Safety cap on the parked collision-map warm-up (RuntimeAdapter 'warm' phase): the warm-up
# normally exits as soon as the station reports a fresh collision_time (a few ticks once nvblox
# has incorporated the first depth), but if it never does, dispatch the grasp anyway so the host's
# startup grace -- not an unbounded warm-up -- bounds a genuinely stuck mapper.
_WARM_MAX_TICKS = 300

# fm399 (1): the reposition RE-warm at place_approach must NOT place on a stale/wrong-pose collision
# map. Unlike the pre-grasp 'warm' (which force-dispatches on the cap so the host startup grace bounds
# a stuck mapper), the re-warm REQUIRES a fresh incorporation at the NEW parked epoch before resuming
# place -- a generous bound (the fresh re-park + a few nvblox integration cycles, stretched by render
# pacing) after which it FAILS the collect honestly rather than placing blind.
_REWARM_MAX_TICKS = 900

# Grace for the parked-stationarity window to fill at the START of a warm phase. The port's `parked`
# needs ~0.25 s of pose history; when the mission BEGINS directly at a manipulation stage (a start-phase
# shortcut, e.g. --start-at transfer), that window is empty on the first ticks, so `parked` is False even
# though the held base is still. Hold and let it fill for this many ticks (~0.33 s at 60 Hz) before
# treating not-parked as a motion fault. In the normal flow the robot enters warm already parked (window
# filled by the preceding nav), so this grace never engages; a genuinely drifting base still fails after it.
_WARM_SETTLE_GRACE_TICKS = 20

# Fail-fast on an unrecoverable collect/transfer reach stall. The sensing capture publishes a
# fresh /sensors frame every tick under a healthy reach (incl. the PR#80 occlusion hold, which
# re-publishes the held registration each tick). If NO fresh /sensors frame is accepted for
# longer than the occlusion-hold window (sim seconds) while the controller is still in
# observe/approach, the reach cannot recover (lost support, permanent decode failure, ...), so
# the mission is failed in seconds instead of spinning 'running' until the slower host
# sensor-stream watchdog. The window matches the PR#80 hold (data/sim-restock-camera-v2);
# the host watchdog remains the backstop and is never bypassed.
_SENSOR_STALL_WINDOW_S = 20.
# collect-gt: manip stages whose final controller evidence may be the fm389 ground-truth place check
# (RESTOCK_VERIFY_PLACE), converted to labelled ground_truth stage evidence -- only while
# RESTOCK_COLLECT_GROUND_TRUTH is on (unset => the pre-existing 'transfer evidence mismatch', byte-identical).
# 'transfer' hits the same mismatch under fm461 ground-truth verify: handled, config-gated, by vr1's
# RuntimeAdapter(ground_truth_place_actions=('transfer',)) (transfer_ground_truth.enabled) + Mission opt-in.
_GT_PLACE_EVIDENCE_STAGES = ('collect',)

# vr1: bounded settle wait (sim seconds) before releasing the tow coupling at the cart parking spot.
# Coupler.detach REFUSES a moving/stale train (any base/cart speed or yaw rate > 0.02), and the cart pin
# engaged one tick earlier can leave a sub-second transient, so the uncouple waits for a measured
# stationary, parked train instead of failing the whole return stage on the first unsettled tick. A
# train that never settles inside this window still fails with a NAMED reason (never forced).
_UNCOUPLE_SETTLE_MAX_S = 30.
_UNCOUPLE_SPEED_TOL = 0.02          # == mobile.Coupler._stationary threshold (m/s and rad/s)


def _train_stationary(docking):
    """Mirror of ``twin.mobile.Coupler._stationary`` for the uncouple pre-check: fresh observation and
    every base/cart speed + yaw rate <= 0.02. ``None`` (a fixture packet without docking kinematics)
    returns True so the mobile adapter itself stays the authority (it re-checks and may refuse)."""
    if docking is None:
        return True
    try:
        vals = (docking.robot_speed, docking.cart_speed, docking.robot_yaw_rate, docking.cart_yaw_rate)
        age = docking.age
    except AttributeError:
        return True
    return (all(math.isfinite(v) for v in (*vals, age)) and 0 <= age <= 0.1
            and max(abs(v) for v in vals) <= _UNCOUPLE_SPEED_TOL)


def _profile_mark(port, name):
    """Attribute the wall since the last profiler mark to ``name`` (sim_speed ``tick_profile_s``).
    A no-op when the port has no TickProfiler (the default) => byte-identical."""
    profiler = getattr(port, '_profiler', None)
    if profiler is not None:
        profiler.mark(name)


@dataclass(frozen=True)
class Packet:
    observation_id: str
    source: str
    simulation_time: float  # mission-relative; never reset at an action boundary
    odometry: object
    navigation: object      # navigation.core.Observation or Nav2's raw scan
    docking: object         # mobile.DockObservation
    manipulation: object    # restock_manipulation.types.Sensors
    transport_ready: bool
    coupled: bool
    articulation: float
    parked: bool
    verification: dict | None = None  # independently observed post-transfer result


class Port(Protocol):
    def advance(self, dt: float) -> Packet: ...
    def controls(self, controls: tuple[dict, ...]) -> None: ...
    def stop(self) -> None: ...  # stop arm/base without releasing a held payload


class NavigationBridge:
    """Normalize reference and Nav2 APIs without changing measurement provenance."""
    def __init__(self, controller, *, nav2=False, sensor_navigation=False,
                 detection_source=None):
        self.controller, self.nav2 = controller, nav2
        self.sensor_navigation = sensor_navigation
        if nav2 and sensor_navigation:
            raise ValueError('select one navigation sensor adapter')
        # Opt-in moving-obstacle feed (twin.navigation.mission_source): a
        # callable(now) -> [navigation.dynamic.Detection] consumed ONLY by the sensor
        # route, which folds it into the sensed Observation add-only. None (default) =>
        # detections=None is passed, byte-identical to a run with no layer at all. The
        # nav2 route takes moving obstacles through its ROS costmap, so refuse the feed
        # there rather than silently dropping it.
        if detection_source is not None and not sensor_navigation:
            raise ValueError('detection_source requires nav_mode "sensor"')
        self.detection_source = detection_source

    def start(self, goal, coupled, *, goal_articulation=None, arrival_xy_tol=None,
              arrival_yaw_tol=None, static_drive=False, carry_drive=False,
              strict_xy_arrival=False):
        kw = {}
        if goal_articulation is not None:
            kw['goal_articulation'] = goal_articulation
        if arrival_xy_tol is not None:
            kw['arrival_xy_tol'] = arrival_xy_tol
        if arrival_yaw_tol is not None:
            kw['arrival_yaw_tol'] = arrival_yaw_tol
        if static_drive:
            kw['static_drive'] = True
        if carry_drive:
            kw['carry_drive'] = True
        if strict_xy_arrival:
            kw['strict_xy_arrival'] = True
        self.controller.start(goal, coupled, **kw)

    def step(self, dt, packet):
        if self.sensor_navigation:
            detections = (self.detection_source(packet.simulation_time)
                          if self.detection_source is not None else None)
            return self.controller.step(dt, packet.odometry, packet.navigation,
                now=packet.simulation_time, coupled=packet.coupled,
                arm_transport=packet.transport_ready, articulation=packet.articulation,
                detections=detections)
        if self.nav2:
            self.controller.publish_sensors(packet.odometry, packet.navigation)
            return self.controller.step(dt, packet.odometry, packet.navigation,
                coupled=packet.coupled, arm_transport=packet.transport_ready,
                articulation=packet.articulation)
        return self.controller.step(dt, packet.odometry, packet.navigation,
                                    now=packet.simulation_time)

    def cancel(self):
        self.controller.cancel()


class ManipulationBridge:
    """Normalize sensor packets versus a runtime that owns camera/control I/O.

    CAD runtime's motion port may cross a process boundary. This bridge neither
    imports ROS/cuRobo into Isaac nor duplicates the runtime's gripper commands.
    Station preparation must use calibrated frames and measured parked pose.
    """
    def __init__(self, controller, *, owns_sensor_io=False, prepare_station=None):
        self.controller = controller
        self.owns_sensor_io = owns_sensor_io
        self.prepare_station = prepare_station
        core = (controller.controller if owns_sensor_io and
                not hasattr(controller, 'cart_location') else controller)
        self.cart_location = core.cart_location

    def start(self, operation, job):
        if self.prepare_station is not None:
            self.prepare_station(operation, job)
        self.controller.start(operation, job)

    def warm(self, dt, packet):
        """Approach/parked warm-up tick before the collect job: let the sensor-owning runtime
        build the host ESDF + drive collision_time fresh so the first grasp /sensors frame lands
        within the host sensor watchdog. Returns True once the collision map is fresh (ready).
        A no-op (returns True) for runtimes that do not own sensor I/O."""
        if self.owns_sensor_io and hasattr(self.controller, 'warm'):
            return bool(self.controller.warm(packet.simulation_time, parked=packet.parked))
        return True

    def step(self, dt, packet):
        if self.owns_sensor_io:
            result = self.controller.step(dt, packet.simulation_time, parked=packet.parked)
            return replace(result, controls=())
        return self.controller.step(dt, packet.manipulation)

    def stop(self):
        if self.owns_sensor_io:
            if hasattr(self.controller, 'stop'):
                self.controller.stop()
            else:
                self.controller.motion.stop()

    def set_base_repositioned(self):
        """fm390/fm400: tell the arm controller the base has driven to place_approach, so its held
        TRANSIT phase proceeds to preplace. Two shapes: the HOST-cuMotion wrapper forwards arrival across
        the ROS boundary via set_base_repositioned() (the host controller is a separate process); the
        in-process fixture TransferController just sets its own base_repositioned attr."""
        core = (self.controller.controller if self.owns_sensor_io
                and not hasattr(self.controller, 'cart_location') else self.controller)
        if hasattr(core, 'set_base_repositioned'):
            core.set_base_repositioned()
        else:
            core.base_repositioned = True

    def set_clock_sync(self, enabled=True):
        """Staged-start clock sync (twin.restock.staged_start): enable the /clock-offset stamp on the
        host sensor packets. Returns True when the sensor-owning runtime accepted it, False for a runtime
        without the hook (the caller logs that the sync is unavailable rather than silently assuming it)."""
        if self.owns_sensor_io and hasattr(self.controller, 'set_clock_sync'):
            self.controller.set_clock_sync(enabled)
            return True
        return False

    def pause_station(self, simulation_time):
        """fm399 (1): cleanly STOP the parked station at the START of the reposition drive, while the
        base is still parked (so the 'stop' is accepted -> idle, NOT faulted). A moving base
        invalidates the parked mapper, and the host spams emergency-stop while a parked station is
        faulted/un-heartbeat-ed (the run-309 failure); an idle station publishes no stop. The station
        is re-begun fresh at place_approach in the re-warm. No-op for a runtime that does not own
        sensor I/O (byte-identical)."""
        if self.owns_sensor_io and hasattr(self.controller, 'pause_station'):
            self.controller.pause_station(simulation_time)


class RuntimeAdapter:
    def __init__(self, navigation, mobile, manipulation, port: Port, poses,
                 *, fixture_mode=False, goal_articulation=None,
                 stall_window_s=_SENSOR_STALL_WINDOW_S, route_alerts=None, fine_dock=None,
                 hold_cart_through_couple=False, couple_static_drive=False,
                 uncouple_on_return=False, wall_clock=time.monotonic, nav_stuck_wall_s=None,
                 carry_reposition_drive=False, couple_standoff_tolerance=None,
                 ground_truth_place_actions=(), staged_settle_s=0.0):
        self.nav, self.mobile, self.manip, self.port = navigation, mobile, manipulation, port
        # Staged-start settle hold (twin.restock.staged_start; the bundle passes
        # config['staged_start_settle_s'] only when start_at is set and not collect): the warm-up of the
        # mission's FIRST manipulation stage holds the dispatch until the simulation has run that long
        # and the station ack + collision-world stamps are steady. 0.0 (default) => no hold, byte-identical.
        self.settle_hold = SettleHold(staged_settle_s) if staged_settle_s and staged_settle_s > 0 else None
        # fm482 (config['couple_standoff_tolerance'], default None => xy 0.05 m + the nav default yaw
        # 0.08 rad, byte-identical): the arrival tolerance of the on-axis couple STANDOFF leg. The
        # standoff is not the latch -- the fine-dock servo closes the residual onto the measured drawbar
        # (attach needs hitch <= 25 mm / yaw <= 5 deg) -- so it may be looser than the latch, up to the
        # fine-dock capture bound proven offline by scripts/check_couple_speed.py.
        self.couple_standoff_tolerance = None
        if couple_standoff_tolerance is not None:
            tol = dict(couple_standoff_tolerance)
            unknown = set(tol) - {'xy_m', 'yaw_rad'}
            if unknown:
                raise ValueError(f'couple_standoff_tolerance: unknown keys {sorted(unknown)}')
            for key, cap in (('xy_m', 0.15), ('yaw_rad', 0.35)):
                if key in tol and not (isinstance(tol[key], (int, float)) and not isinstance(tol[key], bool)
                                       and math.isfinite(tol[key]) and 0 < tol[key] <= cap):
                    raise ValueError(f'couple_standoff_tolerance.{key} must be in (0, {cap}]')
            self.couple_standoff_tolerance = {k: float(v) for k, v in tol.items()}
        # vr1 (labelled sim workaround, default OFF): the stages whose FINAL manipulation evidence may be
        # the controller's ground-truth place check (fm461 'ground_truth_place_check', the sim-measured
        # carton pose vs the bay/deck target) instead of a camera observation. Without this the very first
        # live transfer that SUCCEEDS on the fm461 ground-truth verify is rejected here as 'transfer evidence
        # mismatch' (that evidence carries no sku/quantity and is not 'sensor'), the BT retries a transfer the
        # host refuses to re-begin, and the mission aborts. When listed, the stage succeeds with evidence
        # honestly tagged source='ground_truth' + verify_mode (never 'sensor'); the Mission must opt in to
        # that source (Mission(ground_truth_sources=...)). () (default) => byte-identical.
        self.ground_truth_place_actions = frozenset(ground_truth_place_actions or ())
        if not self.ground_truth_place_actions <= {'transfer'}:
            # collect's ground-truth place evidence is collect-gt's env-gated path below (gtc), not this one
            raise ValueError('ground_truth_place_actions must be a subset of {transfer}')
        # vr1: per-stage labelled evidence mode ({'verify': 'ground_truth', ...}) for the behaviour-tree
        # step labels / walkthrough captions; the bundle fills it from the runtime config. {} => no tags.
        self.stage_modes = {}
        # fm431 (B1) / run-325 cold TRANSIT: relax the nav arm-transport gate on the mid-collect
        # reposition legs so the base can drive to place_approach while the arm CARRIES the carton (the
        # stable braked carry, already lidar self-masked). False (default) => the empty-gripper gate
        # holds => byte-identical. SIM-only safety: valid only with the arm in the verified carry; the
        # in-sim drive + no-drop validation is a GPU check (run-326). See navigation.core carry_drive.
        self.carry_reposition_drive = bool(carry_reposition_drive)
        # Opt-in uncouple-at-parking-spot on return (captain 2026-10-20: "robot should uncouple at the
        # parking spot"): instead of towing the cart home coupled, the return stage tows it to a cart
        # parking spot (`cart_parking` pose, from the nav-lanes lane), brakes the cart, releases the tow
        # coupling (mobile.uncouple), optionally backs away (`cart_parking_backaway`) and drives home
        # EMPTY. False (default) => the legacy tow-home return, byte-identical. See _uncouple_return +
        # the return legs in start()/_step(), and data/mobile-restock-v2 for the deferred-item history.
        self.uncouple_on_return = bool(uncouple_on_return)
        # vr1: True once THIS mission released the tow coupling at the parking spot (the cart is left
        # parked + pinned). A re-begun return (BT StageRetry / fm363 STEP_RETRY) must then resume the EMPTY
        # drive home -- re-towing a cart that is no longer coupled can only loop 'coupling_changed' to the
        # blocked timeout. Also lets a re-begin release a cart pin left by an interrupted brake leg.
        self._cart_left_parked = False
        self._cart_pinned_for_uncouple = False
        self._uncouple_wait_s = 0.0
        # Set in start(); declared here so the opt-in behaviour-tree layer can tell at tree-build time
        # (before any stage begins) that this adapter exposes per-stage phases (vs a plain fixture).
        self.phase = None
        self._return_leg = None
        # Opt-in final fine-docking servo (twin.restock.fine_dock.FineDock). None (default) =>
        # the couple stage hands the parked base straight to attach, byte-identical. When present
        # it creeps the rear hitch onto the measured cart drawbar after nav parks, closing the
        # nav-arrival residual to inside the attach handshake before handover (motion only; the
        # physical latch stays in mobile.py). See fine_dock.py.
        self.fine_dock = fine_dock
        # Opt-in couple cart-hold: pin the cart to the world through the whole couple approach +
        # fine-dock so the base cannot shove the free cart (the cart stays at its known pose so the
        # hitch seats), released only at attach. False (default) => unchanged run. SIM-only; the
        # real cart brakes physically (real-rig semantics untouched).
        self.hold_cart_through_couple = bool(hold_cart_through_couple)
        # Opt-in: drive the couple nav legs against the STATIC map only (empty live obstacles),
        # the same TRUSTED-STATIC pattern as the mid-collect reposition legs (fm396). The couple
        # corridor is geometrically proven clear against the static map -- the backstock Support's
        # TRUE AABB (scenario staging) clears the footprint -- but the LIVE Support returns sit a
        # few cm proud of that surface (localisation/raycast-height bias), which erases the thin
        # margin and makes the planner refuse the route from the shortcut start pose directly south
        # of the Support (the real collect leaves the base at place_approach, EAST of the Support,
        # from which couple routes fine either way). Static only for the UNCOUPLED couple approach
        # legs; the standoff->drawbar insert is the motion-only fine-dock servo, not a nav plan.
        # False (default) => every couple leg keeps full live-scan obstacle avoidance
        # (byte-identical). SIM-only; the real rig keeps live avoidance (real-rig semantics
        # untouched). See core.Controller.static_drive and data/sim-restock-couple-nav-v1/report.md.
        self.couple_static_drive = bool(couple_static_drive)
        # Opt-in mobile store-sensor hook (twin.route_alerts). None (default) =>
        # byte-identical run. When attached it runs the read-only store-vision
        # detectors over the drive camera during navigate; it is self-disabling and
        # never raises into the tick, and never changes navigation/actuation here.
        self.route_alerts = route_alerts
        if not isinstance(self.manip, ManipulationBridge):
            self.manip = ManipulationBridge(self.manip)
        # Last manipulation controller phase (observe/approach/contact/...), fed to the port's opt-in
        # sensing gate each tick BEFORE advance() to suspend the decode render during the reach. None
        # (pre-grasp) and unknown phases fail safe to render-required (no suspend). A one-tick lag is
        # within the occlusion window the gate's hold already tolerates.
        self._last_manip_phase = None
        self.poses, self.fixture_mode = poses, fixture_mode
        # Fail-fast stall window (sim seconds); None disables (host watchdog still backstops).
        if stall_window_s is not None and (not math.isfinite(stall_window_s) or stall_window_s <= 0):
            raise ValueError('stall window must be positive and finite, or None')
        self.stall_window_s = stall_window_s
        self._reach_start = None
        self.goal_articulation = dict(goal_articulation or {})
        if any(not math.isfinite(v) or not 0 < v <= math.radians(65)
               for v in self.goal_articulation.values()):
            raise ValueError('invalid goal articulation tolerance')
        for name in ('pickup', 'cart_dock', 'shelf_approach', 'home'):
            pose = poses.get(name, ())
            if len(pose) != 3 or not all(math.isfinite(v) for v in pose):
                raise ValueError(f'missing/invalid pose: {name}')
        # Optional couple approach waypoint + on-axis standoff, and the fm390 place_approach waypoint
        # (in front of the rollwagen mouth) the collect stage drives to before placing; validated only
        # when supplied.
        for name in ('couple_approach', 'couple_standoff', 'place_approach', 'place_approach_stage',
                     'cart_parking', 'cart_parking_backaway'):
            if name in poses:
                wp = poses[name]
                if len(wp) != 3 or not all(math.isfinite(v) for v in wp):
                    raise ValueError(f'invalid pose: {name}')
        self.time = 0.
        # fm425 (c) (run-322b): WALL-clock stuck-stage fail-fast for navigate. The nav controller's own
        # blocked_timeout/goal_timeout are SIM-clock caps (factory.py) -- under render pacing the sim
        # advances far slower than wall, so run-322b's coupled OUTBOUND tow sat on reason='lane_blocked'
        # for ~50 min WALL (925 NAV_STEP) and NEITHER sim cap fired; only the 3600 s orchestrator wall cap
        # (SIGKILL) ended it, burning the whole GPU budget with no named stage failure. This is a WALL
        # budget: if navigate reports the SAME running reason with NO measured base-pose progress for
        # longer than nav_stuck_wall_s WALL, the stage FAILS FAST with that named reason. Reset whenever
        # the reason changes OR the base makes progress. 0 s (default) => off => byte-identical; a sane
        # value sits well under the orchestrator wall cap (GPU hours are the scarce resource). wall_clock
        # is injectable for tests (default monotonic).
        self.wall_clock = wall_clock
        if nav_stuck_wall_s is None:
            try:
                nav_stuck_wall_s = float(os.environ.get('RESTOCK_NAV_STUCK_WALL_S', 0.0))
            except (TypeError, ValueError):
                nav_stuck_wall_s = 0.0
        if not (math.isfinite(nav_stuck_wall_s) and 0.0 <= nav_stuck_wall_s <= 3000.0):
            raise ValueError('nav_stuck_wall_s must be a finite number of seconds in [0, 3000]')
        self.nav_stuck_wall_s = float(nav_stuck_wall_s)
        self._nav_stuck_reason = None     # the running reason currently being timed
        self._nav_stuck_wall = None       # wall time that reason/pose last changed (progress)
        self._nav_stuck_pose = None       # base pose at that progress mark
        self.action = None
        self.last_packet_id = None
        # SIM-ONLY scripted transfer fallback (RESTOCK_TRANSFER_MODE=scripted): a
        # twin.restock.scripted_transfer.ScriptedTransfer attached by the Isaac bundle. None (default)
        # => the transfer stage warms + dispatches the host manipulation job exactly as before.
        self.scripted_transfer = None

    def start(self, action, job):
        if action != 'return':
            # vr1: the uncouple memory spans only consecutive (re-)begins of THIS mission's return; any
            # other stage beginning (a new / warm-session attempt re-couples the cart) clears it.
            self._cart_left_parked = False
            self._cart_pinned_for_uncouple = False
        self.mobile.park()
        if hasattr(self.port,'begin_action'):
            self.port.begin_action(action,job)
        self.action, self.job = action, job
        self.phase = 'execute'
        self._return_leg = None           # set when the uncouple-on-return multi-leg return runs
        self._reach_start = None          # set on the first execute tick of a collect/transfer reach
        # fm390 mid-collect base reposition: driven once per collect when the arm reaches TRANSIT (carry
        # fold done) and a place_approach pose is configured. Reset here so a re-begun collect starts clean.
        self._collect_reposition_pending = False
        self._collect_repositioned = False
        self._reposition_seq = []          # fm390 nav proof #2: staged drive to place_approach
        self._reposition_seq_i = 0
        if action == 'couple':
            # Couple nav runs an in-order sequence of waypoints ending at the ON-AXIS standoff,
            # then the fine-dock servo inserts STRAIGHT along the cart axis to the drawbar:
            #   couple_approach (round the pinned cart, east of it) -> couple_standoff (on the cart
            #   hitch axis, ~0.4 m east of the dock, TIGHT arrival) -> fine-dock insert.
            # Approaching on-axis turns the lateral error into cross-track the servo nulls WHILE
            # reversing, instead of a residual nobody can close at zero speed. Absent both poses =>
            # the old single cart_dock nav + fine-dock from there.
            self._couple_seq = (['couple_approach'] if 'couple_approach' in self.poses else [])
            self._couple_seq.append('couple_standoff' if 'couple_standoff' in self.poses else 'cart_dock')
            self._couple_seq_i = 0
            self._couple_nav_goal = self._couple_seq[0]
            self._start_couple_nav(self._couple_nav_goal)
            self.phase = 'navigate'
            if self.hold_cart_through_couple:
                # Pin the cart NOW (at its known pose, before the base drives) so the approach
                # cannot shove the free cart; attach() releases it once coupled.
                self.mobile.hold_cart_pin()
        elif action in ('collect', 'outbound', 'return'):
            if action == 'return' and self._uncouple_return() and self._cart_left_parked:
                # vr1: a RE-BEGUN return after this mission already uncoupled (BT StageRetry / fm363
                # STEP_RETRY of a failed home leg): the cart is parked + pinned and NOT coupled, so resume
                # the empty drive home from the live pose instead of re-towing a cart that is gone.
                self._start_nav_leg(self.poses['home'], False)
                self.phase = 'navigate'
                self._return_leg = 'home'
            elif action == 'return' and self._uncouple_return():
                self._uncouple_wait_s = 0.0
                if self._cart_pinned_for_uncouple and hasattr(self.mobile, 'release_cart_brake'):
                    # vr1: an interrupted brake/uncouple leg left the STILL-COUPLED cart world-pinned; free
                    # it so the re-begun coupled tow is not dragging a pinned cart (re-pinned at 'brake').
                    self.mobile.release_cart_brake(force=True)
                    self._cart_pinned_for_uncouple = False
                # Uncouple-at-parking-spot return: tow the cart to its parking spot FIRST (still
                # coupled), then brake it, release the coupling and drive home empty. The first leg is a
                # coupled tow, so it keeps the return goal-articulation tolerance.
                kwargs = ({'goal_articulation': self.goal_articulation['return']}
                          if 'return' in self.goal_articulation else {})
                self._start_nav_leg(self.poses['cart_parking'], True, **kwargs)
                self.phase = 'navigate'
                self._return_leg = 'cart_parking'
            else:
                goal = {'collect': 'pickup', 'outbound': 'shelf_approach', 'return': 'home'}[action]
                kwargs = ({'goal_articulation':self.goal_articulation[action]}
                          if action in self.goal_articulation else {})
                if action == 'collect':
                    # Tolerance-aware final approach for the grasp-critical source park: never DECLARE
                    # arrived outside the real xy tolerance (so the pick starts within grasp reach), even
                    # when the global XY hysteresis latch (couple standoff, fm482) is configured. A no-op
                    # when the latch is off (the bare xy <= tol test is already strict) => byte-identical
                    # for a run with no hysteresis; it only ever TIGHTENS the collect arrival.
                    kwargs['strict_xy_arrival'] = True
                self._start_nav_leg(self.poses[goal], action in ('outbound', 'return'), **kwargs)
                self.phase = 'navigate'
        elif action == 'transfer' and self.scripted_transfer is not None:
            # SIM-ONLY scripted transfer (labelled workaround): no host manipulation job and no station
            # warm-up -- the scripted runner moves the carton (and, opted in, the arm through the host
            # trajectory executor) to the ground-truth bay target. Begun on the first parked tick.
            self.phase = 'scripted'
            self._warmn = 0
            self._scripted_begun = False
        elif action == 'transfer':
            # Mirror the collect warm-up: make the shelf station's host ESDF + collision_time fresh
            # BEFORE dispatching the place grasp, so the first transfer observe sees a fresh
            # collision world instead of racing a cold nvblox incorporation against the 30 s phase
            # timeout. The robot is already parked (from the park stage), so warm directly; the
            # grasp is dispatched (manip.start('transfer')) when the warm phase completes.
            self.phase = 'warm'
            self._warmn = 0
        elif action not in ('park', 'verify'):
            raise ValueError(f'unsupported action: {action}')

    def _start_couple_nav(self, goal_name):
        # Tight XY arrival for the on-axis standoff so the fine-dock insert starts near the cart
        # axis (its cross-track is what the servo nulls while reversing); normal arrival otherwise.
        kw = {'arrival_xy_tol': 0.05} if goal_name == 'couple_standoff' else {}
        if goal_name == 'couple_standoff' and self.couple_standoff_tolerance is not None:
            tol = self.couple_standoff_tolerance   # fm482: config override (see __init__)
            kw['arrival_xy_tol'] = tol.get('xy_m', 0.05)
            if 'yaw_rad' in tol:
                kw['arrival_yaw_tol'] = tol['yaw_rad']
        # Opt-in trusted-static couple approach: route against the static map so the thin live
        # margin at the backstock Support cannot false-block the leg (see self.couple_static_drive).
        if self.couple_static_drive:
            kw['static_drive'] = True
        self._start_nav_leg(self.poses[goal_name], False, **kw)

    def _uncouple_return(self):
        """True when the return stage leaves the cart parked and drives home empty, instead of towing
        it home coupled: opted in via config (``uncouple_on_return``) AND a ``cart_parking`` pose is
        configured. Default off (or no parking pose) => the legacy tow-home return, byte-identical."""
        return self.uncouple_on_return and 'cart_parking' in self.poses

    def stop(self):
        errors = []
        # Try every stop even when one adapter raises. Never command release.
        calls = [self.mobile.park, self.nav.cancel, self.manip.stop, self.port.stop]
        if self.route_alerts is not None:
            calls.append(self.route_alerts.close)
        for call in calls:
            try:
                call()
            except Exception as exc:
                errors.append(str(exc))
        if errors:
            raise RuntimeError('; '.join(errors))

    def _evidence(self, packet, **extra):
        return dict(observation_id=packet.observation_id, source=packet.source,
                    simulation_time=packet.simulation_time, **extra)

    def _nav_controller(self):
        """The navigation controller behind the bridge (unwrapping the sensor/Nav2 adapters' inner
        controller); None for a fixture without one."""
        core = getattr(self.nav, 'controller', None)
        for attr in ('controller', 'safe'):
            if core is not None and not hasattr(core, 'arrival_xy_hysteresis') and hasattr(core, attr):
                core = getattr(core, attr)
        return core

    def _nav_workarounds(self, action):
        """fm482: name the opt-in couple/outbound nav switches in the stage evidence (mission.json) as
        scalar ``*_mode`` fields, which the walkthrough cut (twin.demo.run_evidence.workaround_labels)
        puts on screen as that stage's caption label. {} when none is on => byte-identical events."""
        nav = self._nav_controller()
        if action == 'couple':
            parts = []
            hysteresis = getattr(nav, 'arrival_xy_hysteresis', 0.0) or 0.0
            if hysteresis > 0:
                parts.append(f'standoff xy-hysteresis {hysteresis:g} m')
            tol = self.couple_standoff_tolerance
            if tol is not None:
                parts.append('standoff tol ' + ' / '.join(
                    f'{k[:-2] if k.endswith("_m") else k[:-4]} {v:g}{" m" if k.endswith("_m") else " rad"}'
                    for k, v in sorted(tol.items())))
            return {'couple_arrival_mode': '; '.join(parts)} if parts else {}
        if action == 'outbound' and getattr(nav, 'coupled_creep_escape', False):
            ticks = int(getattr(nav, 'creep_escapes', 0))
            return {'outbound_brake_mode': f'coupled creep-escape ({ticks} noise-band ticks)'}
        return {}

    def _sensor_stall(self, result):
        """Return a failure reason if the collect/transfer reach has stalled, else None.

        Stall = no fresh /sensors frame accepted for longer than ``stall_window_s`` (sim
        seconds) while the controller is still in ``observe``/``approach``. A host trajectory
        fault (``TRAJ_HOST_FAULT``) that surfaces as a failed/rejected reply already fails via
        the normal status path above; this guard catches the quieter case where the reach keeps
        reporting 'running' but /sensors has gone dead. Active only when the controller exposes
        ``last_sensor_time`` (production sensing); other adapters are unaffected.
        """
        if self.stall_window_s is None:
            return None
        if getattr(result, 'phase', None) not in ('observe', 'approach'):
            return None
        core = getattr(self.manip, 'controller', None)
        last = getattr(core, 'last_sensor_time', None)
        if last is None:
            return None                                   # non-sensing adapter: nothing to time
        if self._reach_start is None:
            self._reach_start = self.time                 # first execute tick of this reach
        # Measure from the last accepted frame, or the reach start if none has landed yet.
        since = self.time - (last if last >= 0 else self._reach_start)
        if since > self.stall_window_s:
            return (f'{self.action} reach stalled: no accepted /sensors frame for {since:.1f}s '
                    f'(> {self.stall_window_s:.0f}s hold window) in {result.phase}; '
                    f'host sensor-stream watchdog is the backstop')
        return None

    def _nav_stuck_check(self, reason, odometry):
        """fm427 (C): WALL-clock NO-BASE-PROGRESS fail-fast for a running navigate stage.

        Returns a named fail reason when the measured base pose has NOT progressed (> ~2 cm / ~1 deg)
        for longer than nav_stuck_wall_s WALL seconds, else None. The timer resets ONLY on base progress
        -- NOT on a reason change -- so a stage that cycles reasons while frozen (run-323b: outbound
        alternating lane_not_acquired<->replan_changed_observations with ~0.37 m total drift then frozen,
        which the earlier reason-keyed check and every SIM-clock cap missed) is still caught. Off
        (nav_stuck_wall_s == 0) => always None (byte-identical); a missing odometry pose is treated as
        no-progress so a truly frozen stage still trips. The message names the CURRENT (last) reason."""
        if self.nav_stuck_wall_s <= 0:
            return None
        pose = getattr(odometry, 'pose', None)
        now_wall = self.wall_clock()
        progressed = False
        if pose is not None and self._nav_stuck_pose is not None:
            dyaw = pose[2] - self._nav_stuck_pose[2]
            dyaw = math.atan2(math.sin(dyaw), math.cos(dyaw))   # normalize to [-pi, pi]
            progressed = (math.hypot(pose[0] - self._nav_stuck_pose[0],
                                     pose[1] - self._nav_stuck_pose[1]) > 0.02
                          or abs(dyaw) > math.radians(1.0))
        if self._nav_stuck_wall is None or progressed:
            self._nav_stuck_wall = now_wall
            self._nav_stuck_pose = pose
            self._nav_stuck_reason = reason
            return None
        self._nav_stuck_reason = reason   # keep the latest reason for the failure message
        waited = now_wall - self._nav_stuck_wall
        if waited > self.nav_stuck_wall_s:
            return ('%s stuck with no base progress for %.0fs wall (> %.0fs budget; last reason %r); '
                    'failing the attempt fast instead of burning the GPU wall cap'
                    % (self.action, waited, self.nav_stuck_wall_s, reason))
        return None

    def _start_nav_leg(self, goal, coupled, **kw):
        """Dispatch a FRESH navigation leg and (re)start its no-base-progress WALL budget from THIS
        leg's start (fm-return-rebegin).

        The fm425 stuck timer (``_nav_stuck_wall``/``_nav_stuck_pose``) is only ever advanced inside
        the navigate phase (``_nav_stuck_check``); a NON-navigate phase never touches it. So the WALL
        seconds a stage legitimately spends stationary in a preceding non-navigate phase -- the long
        park + transfer + verify before the uncouple-return tow (warm7: ~207 s wall carried from the
        outbound tow's last progress through the whole transfer) -- would otherwise be charged against
        this leg's VERY FIRST navigate tick and trip the budget before the controller finishes its
        normal planning_stop -> route_ready sequence. That failed the return on tick one, and because
        the StageRetry re-begin also runs through here without clearing the baseline, every bounded
        re-attempt re-failed instantly (the 'per-tick re-begin', elapsed=0.02, route never persisted).
        The single-leg outbound never hits it: its nav leg follows the couple navigate/fine-dock, so
        its baseline is already fresh. Clearing the baseline on a genuinely new leg is identical to a
        progress reset, so the within-leg frozen-stage detection (run-323b) is unchanged -- no
        ``start`` ever occurs between a frozen leg's own ticks."""
        self._nav_stuck_wall = None
        self._nav_stuck_pose = None
        self._nav_stuck_reason = None
        self.nav.start(goal, coupled, **kw)

    def _step(self, dt):
        self.time += dt
        gate = getattr(self.port, 'gate', None)  # drive the opt-in sensing gate from last phase
        if gate is not None:
            gate(self._last_manip_phase)
        p = self.port.advance(dt)
        sources = ('sensor', 'test_fixture') if self.fixture_mode else ('sensor',)
        if (p.source not in sources or not p.observation_id or
                p.observation_id == self.last_packet_id or
                not math.isfinite(p.articulation) or
                not math.isfinite(p.simulation_time) or not -1e-6 <= self.time-p.simulation_time <= .1):
            raise ValueError(
                'missing, repeated, stale or unsupported sensor packet '
                f'[source={p.source!r} obs={p.observation_id!r} prev={self.last_packet_id!r} '
                f'articulation={p.articulation!r} sim_time={p.simulation_time!r} '
                f'adapter_time={self.time!r} age={self.time-p.simulation_time!r}]')
        self.last_packet_id = p.observation_id
        if self.phase == 'navigate':
            if self.route_alerts is not None:  # opt-in mobile store sensor; self-disabling
                self.route_alerts.on_navigate(self.port, p, self.time)
            r = self.nav.step(dt, p)
            _profile_mark(self.port, 'nav')     # sim_speed TICK_PROFILE split; no-op unless profiling
            self._navn = getattr(self, '_navn', 0) + 1
            # Throttled heartbeat every 30 ticks PLUS every tick whose reason is not plain 'tracking'
            # (de-duplicated on reason), so a stall's true per-tick refusal/replan reason is never hidden
            # by the 1/30 sampling (runs 327-329 stalled on an outbound 'planning_stop' the sampled log
            # never showed). RESTOCK_NAV_REASON_DIAG=0 silences the extra per-reason lines.
            _extra = (r.reason != 'tracking' and getattr(self, '_last_nav_reason', None) != r.reason
                      and os.environ.get('RESTOCK_NAV_REASON_DIAG', '1') != '0')
            self._last_nav_reason = r.reason
            if self._navn % 30 == 1 or _extra:           # diagnostic: why navigate does not complete
                print(f'NAV_STEP action={self.action} status={r.status} reason={r.reason!r} '
                      f'transport_ready={p.transport_ready} coupled={p.coupled} parked={p.parked}',
                      flush=True)
            if r.status == 'failed':
                return Result('failed', reason=r.reason)
            if r.status not in ('running', 'succeeded'):
                raise ValueError('invalid navigation status')
            if r.status == 'running':
                stuck = self._nav_stuck_check(r.reason, getattr(p, 'odometry', None))
                if stuck is not None:
                    return Result('failed', reason=stuck)
                self.mobile.command(r.linear, r.angular, p.docking,
                                    transport_ready=p.transport_ready)
                return Result('running', reason=r.reason)
            self.mobile.park()
            if not p.parked:
                return Result('running', reason='waiting for measured parking')
            if self.action == 'outbound':
                if not p.coupled:
                    raise ValueError('cart lost at destination')
                limit = self.goal_articulation.get('outbound')
                if limit is not None and abs(p.articulation) >= limit:
                    raise ValueError('cart articulation exceeds manipulation/return goal tolerance')
                return Result('succeeded', self._evidence(p, arrived=True, **self._nav_workarounds('outbound')))
            if self.action == 'return':
                if not self._uncouple_return():
                    # Legacy tow-home: the cart is still coupled at home (byte-identical path).
                    if not p.coupled:
                        raise ValueError('cart lost at destination')
                    limit = self.goal_articulation.get('return')
                    if limit is not None and abs(p.articulation) >= limit:
                        raise ValueError('cart articulation exceeds manipulation/return goal tolerance')
                    return Result('succeeded', self._evidence(p, home=True))
                # Uncouple-at-parking-spot legs reached by navigate completion:
                if self._return_leg == 'cart_parking':
                    # Towed to the parking spot: the cart must still be coupled here. Hand off to the
                    # execute phase, which brakes the cart and releases the coupling.
                    if not p.coupled:
                        raise ValueError('cart lost before the parking spot')
                    limit = self.goal_articulation.get('return')
                    if limit is not None and abs(p.articulation) >= limit:
                        raise ValueError('cart articulation exceeds manipulation/return goal tolerance')
                    self.phase = 'execute'
                    self._return_leg = 'brake'
                    return Result('running', reason='towed to cart parking spot; braking cart before uncouple')
                if self._return_leg == 'cart_parking_backaway':
                    # Backed clear of the parked cart (uncoupled): drive home empty.
                    self._start_nav_leg(self.poses['home'], False)
                    self.phase = 'navigate'
                    self._return_leg = 'home'
                    return Result('running', reason='backed away from parked cart; driving home empty')
                if self._return_leg == 'home':
                    # Home, empty: the cart was left parked, so no coupled requirement here.
                    return Result('succeeded', self._evidence(p, home=True))
                raise RuntimeError(f'invalid return leg at navigate: {self._return_leg!r}')
            if self.action == 'collect' and self._collect_reposition_pending:
                if self._reposition_seq_i < len(self._reposition_seq) - 1:
                    # Reached the staging waypoint (east of the cart); advance to place_approach.
                    self._reposition_seq_i += 1
                    self._start_nav_leg(self._reposition_seq[self._reposition_seq_i], False, static_drive=True,
                                        carry_drive=self.carry_reposition_drive,  # fm431 (B1): drive while carrying
                                        arrival_xy_tol=0.15, arrival_yaw_tol=0.20)  # fm408: cover the base static-drive residual (run-315 diag: xy~0.11, yaw~0.16)
                    self.phase = 'navigate'
                    return Result('running',
                                  reason=f'reposition advancing to {self._reposition_seq[self._reposition_seq_i]}')
                # fm390: arrived at place_approach with the carton carried. Resume the PAUSED arm (its
                # TRANSIT -> preplace) after re-warming the collision map at the new base pose; do NOT
                # re-begin the manipulation (the controller is mid-run, holding the carry pose).
                # fm399 (1): the station was cleanly STOPPED at drive-start; the re-warm below begins a
                # FRESH epoch at this new pose and re-incorporates a fresh ESDF before place (the
                # pickup-pose ESDF would be wrong here -- the run-309 stale-collision-world failure).
                self.manip.set_base_repositioned()
                self._collect_reposition_pending = False
                self._collect_repositioned = True
                self.phase = 'rewarm'
                self._warmn = 0
                return Result('running', reason='at place_approach; re-parking + re-warming before place')
            if self.action == 'collect':
                # Warm the host collision map (ESDF + a fresh collision_time) while parked BEFORE
                # dispatching the grasp, so the first /sensors frame lands within the host's sensor
                # watchdog instead of racing a cold ~1 s ESDF incorporation from job start.
                self.phase = 'warm'
                self._warmn = 0
                return Result('running', reason='warming collision map before grasp')
            if self.action == 'couple' and self._couple_seq_i < len(self._couple_seq) - 1:
                # Reached a couple nav waypoint; advance to the next (ending at the standoff).
                self._couple_seq_i += 1
                self._couple_nav_goal = self._couple_seq[self._couple_seq_i]
                self._start_couple_nav(self._couple_nav_goal)
                self.phase = 'navigate'
                return Result('running', reason=f'couple nav advancing to {self._couple_nav_goal}')
            if self.action == 'couple' and self.fine_dock is not None:
                # At the on-axis standoff (or cart_dock when no standoff): insert the rear hitch
                # STRAIGHT along the cart axis onto the measured drawbar before attach. Motion only.
                self.fine_dock.start(self.time)
                self._fdn = 0
                self.phase = 'fine_dock'
                return Result('running', reason='nav at standoff; starting fine-dock insert')
            self.phase = 'execute'
        if self.phase == 'scripted':
            self.mobile.hold()   # stationary manipulation: brake the base
            self._warmn += 1
            if not p.parked:
                # A start-phase shortcut begins AT this stage, so the stationarity window may not have
                # filled yet (same grace as warm); a base moving once the script runs is a real fault.
                if not self._scripted_begun and self._warmn <= _WARM_SETTLE_GRACE_TICKS:
                    return Result('running', reason='scripted transfer settling (stationarity window filling)')
                return Result('failed', reason='base moved during the scripted transfer')
            if not self._scripted_begun:
                self.scripted_transfer.begin(self.job, p.simulation_time)
                self._scripted_begun = True
            r = self.scripted_transfer.step(dt, p)
            if r.status != 'succeeded':
                return r
            ev = r.evidence
            # Checked against the SIM GROUND TRUTH, so labelled source='ground_truth' (vr1's convention;
            # the Mission accepts it only with ground_truth_sources) -- never relabelled 'sensor'.
            if (ev.get('source') != gtc.SOURCE or ev.get('ground_truth') is not True or
                    ev.get('sku') != self.job.sku or ev.get('location') != self.job.shelf_id or
                    ev.get('quantity') != self.job.quantity):
                raise ValueError('scripted transfer evidence mismatch')
            return Result('succeeded', dict(ev, released=True))
        if self.phase == 'warm':
            self.mobile.hold()   # stationary manipulation: brake the base against arm reaction
            self._warmn += 1
            # Pump the station (masked depth + heartbeat) EVERY warm tick, including the settle grace
            # below: the station host faults 'heartbeat lost' if it is not pumped within its wall/sim
            # heartbeat gap, and under per-tick rendering a few ticks of silence already blow the wall
            # gap. (Regression guard: the settle grace must never skip this pump -- ds-011.)
            ready = self.manip.warm(dt, p)
            if not p.parked:
                # A start-phase shortcut begins the mission AT this warm, so the port's stationarity
                # window has not filled yet and `parked` is False even with the base held still. Let it
                # fill before treating not-parked as a fault; a base still not parked after the grace is
                # a genuine motion fault. The normal flow enters warm already parked and skips this.
                if self._warmn <= _WARM_SETTLE_GRACE_TICKS:
                    return Result('running', reason='warm-up settling (stationarity window filling)')
                return Result('failed', reason='base moved during collision-map warm-up')
            hold = self.settle_hold
            if hold is not None and not hold.released:
                # Staged start: the FIRST manipulation dispatch waits for a settled simulation (see
                # staged_start.SettleHold); the station keeps being pumped above every held tick.
                if not hold.update(self.time, ready=ready, parked=p.parked):
                    return Result('running', reason=hold.status)
                print(hold.release_line + f' action={self.action}', flush=True)
                if hold.reason == 'cap':
                    self._warmn = max(self._warmn, _WARM_MAX_TICKS)   # bounded: dispatch like the warm cap
            if ready or self._warmn >= _WARM_MAX_TICKS:
                self.phase = 'execute'
                # Dispatch the current manipulation job (collect or transfer) now the collision map
                # is warm; the warmed station is kept (the runtime does not re-begin/re-cold it).
                self.manip.start(self.action, asdict(self.job))
                return Result('running', reason=('collision map warm; dispatching grasp' if ready
                                                 else 'warm-up tick cap reached; dispatching grasp'))
            return Result('running', reason='warming collision map')
        if self.phase == 'rewarm':
            # fm390: re-warm the collision map at place_approach, then RESUME the paused arm WITHOUT
            # re-beginning the manipulation (it is mid-run, holding the carry pose in TRANSIT). Mirrors
            # 'warm' except for the missing manip.start -- the base must stay parked here.
            self.mobile.hold()
            self._warmn += 1
            ready = self.manip.warm(dt, p)
            if not p.parked:
                if self._warmn <= _WARM_SETTLE_GRACE_TICKS:
                    return Result('running', reason='place_approach warm settling (stationarity filling)')
                return Result('failed', reason='base moved during place_approach re-warm')
            # fm399 (1): REQUIRE a fresh incorporation at the re-parked epoch before resuming place --
            # do NOT force-proceed on the cap (that would place on the stale pickup-pose map, the
            # run-309 failure). The incorporation-ack parked_since gate guarantees `ready` can only be
            # True once a fresh ESDF has integrated strictly after the new parked epoch at this pose.
            if ready:
                self.phase = 'execute'
                return Result('running', reason='re-warm done; fresh collision map; resuming place from place_approach')
            if self._warmn >= _REWARM_MAX_TICKS:
                return Result('failed', reason='no fresh collision map re-incorporated at place_approach')
            return Result('running', reason='re-warming collision map at place_approach')
        if self.phase == 'fine_dock':
            # Slow closed-loop creep onto the MEASURED cart drawbar until inside the attach
            # handshake, then hand over. Bounded + honest failure with residuals (motion only).
            r = self.fine_dock.step(p.docking, self.time)
            self._fdn = getattr(self, '_fdn', 0) + 1
            if self._fdn % 30 == 1 or r.status != 'creeping':
                print(f'FINE_DOCK_DIAG status={r.status} hitch_mm={r.hitch_residual_m*1000:.1f} '
                      f'yaw_deg={math.degrees(r.yaw_residual_rad):.2f} v={r.linear:.3f} '
                      f'w={r.angular:.3f} reason={r.reason!r}', flush=True)
            if r.status == 'failed':
                self.mobile.park()
                return Result('failed', reason=r.reason)
            if r.status == 'ready':
                self.mobile.park()
                if not p.parked:
                    return Result('running', reason='fine-dock settling; waiting for measured parking')
                self.phase = 'execute'   # settled inside tolerance: fall through to attach
            else:
                self.mobile.command(r.linear, r.angular, p.docking,
                                    transport_ready=p.transport_ready)
                return Result('running', reason=r.reason)
        self.mobile.hold()   # stationary manipulation: brake the base against stiff-arm reaction
        if self.action == 'couple':
            # fm419 (b): past fine-dock the couple stage is COMMITTED to the latch, so it is handled
            # ABOVE the stationary parked-guard below. Creating the drawbar joint snaps the cart onto
            # the hitch and briefly perturbs the cart/base (run-320b: a one-tick cart_dyaw ~26 mrad,
            # parked=False blip) -- fine_dock already gated parked before handover (above), so a
            # transient blip during the EXPECTED attach is not a fault. When it WAS a fault the mission
            # STEP_RETRY re-began couple and restarted the approach nav coupled=False against an
            # already-coupled cart, which loops 'coupling_changed' to coupling_changed_timeout (the
            # run-320b failure after a physically good 20 mm / 2.16 deg seat). A successful latch
            # (obs.coupled healthy) instead declares couple DONE and hands off to the coupled controller
            # for outbound; a genuinely bad base is still caught by coupling_state['healthy'].
            if self.phase != 'attached':
                self.mobile.attach(p.docking)
                self.phase = 'attached'
                return Result('running', reason='awaiting post-attach observation')
            state = self.mobile.coupling_state(p.docking)
            if p.coupled and state['healthy']:
                return Result('succeeded', self._evidence(p, coupled=True, **self._nav_workarounds('couple')))
            return Result('running', reason='awaiting healthy physical coupling')
        if self.action == 'return' and self._uncouple_return():
            # vr1: handled ABOVE the stationary parked-guard, mirroring couple's fm419(b): the brake pin
            # and the hitch release are EXPECTED to perturb the train for a tick or two, so an unsettled
            # tick WAITS (the base is held, nothing is commanded) instead of failing the whole stage -- a
            # failure here is unrecoverable once the coupling is released (see _cart_left_parked).
            return self._step_return_uncouple(p, dt)
        if not p.parked:
            return Result('failed', reason='base moved during stationary operation')
        if self.action == 'park':
            if not p.coupled:
                return Result('failed', reason='cart lost before transfer')
            return Result('succeeded', self._evidence(p, stationary=True))
        if self.action in ('collect', 'transfer'):
            r = self.manip.step(dt, p)
            _profile_mark(self.port, 'manip')   # sim_speed TICK_PROFILE split; no-op unless profiling
            self._last_manip_phase = getattr(r, 'phase', None)  # feeds next tick's port.gate()
            # fm390: the arm has folded to the carry pose and is HOLDING in TRANSIT -- drive the base to
            # place_approach (carton carried) before it places onto the deck. Pause stepping the arm (do
            # not dispatch its controls); nav takes over. One-shot per collect; absent place_approach =>
            # the arm never enters TRANSIT (reposition_before_place off) so this path is byte-identical.
            if (self.action == 'collect' and self._last_manip_phase == 'transit'
                    and 'place_approach' in self.poses and not self._collect_repositioned):
                # fm390 nav proof #2 (scripts/check_place_approach_nav.py): a DIRECT diagonal drive to
                # place_approach clips the parked cart -- the base must translate EAST clear of the cart
                # before turning to face the mouth. The parked cart is NOT a nav obstacle at these docking
                # distances (couple backs the base onto its hitch), so this SEQUENCING, not the planner,
                # keeps the base clear. Route via a staging waypoint (explicit place_approach_stage, else
                # the proven couple_approach) then place_approach; only the FINAL goal sets base_repositioned.
                stage = self.poses.get('place_approach_stage') or self.poses.get('couple_approach')
                self._reposition_seq = ([stage, self.poses['place_approach']] if stage is not None
                                        else [self.poses['place_approach']])
                self._reposition_seq_i = 0
                # fm399 (1): a moving base invalidates the parked mapper, and the host faults + spams
                # emergency-stop the instant a parked station stops being heartbeat-ed (run-309 died
                # here). CLEANLY stop the station NOW -- while still parked, so the 'stop' is accepted
                # (-> idle, NOT faulted, no emergency stop) -- before the drive. It is re-begun fresh at
                # place_approach in the re-warm, so place sees a correct new-pose ESDF by construction.
                self.manip.pause_station(self.time)
                # fm396 (B): trusted STATIC drive for the reposition -- the short move in front of the
                # parked cart is proven clear against the static map + footprint (check_place_approach_nav.py),
                # but the carry-fold self-geometry pollutes the live scan (run-304/307 ring the per-tick mask
                # did not clear). Static-only route + gates for these legs only; every other leg keeps full
                # live obstacle avoidance.
                self._start_nav_leg(self._reposition_seq[0], False, static_drive=True,
                                    carry_drive=self.carry_reposition_drive,  # fm431 (B1): drive while carrying
                                    arrival_xy_tol=0.15, arrival_yaw_tol=0.20)  # fm408: cover the static-drive yaw residual (run-315 diag)
                self.phase = 'navigate'
                self._collect_reposition_pending = True
                return Result('running', reason=f'pick done; staging base to {self._reposition_seq[0]} '
                                                 f'en route to place_approach')
            stalled = self._sensor_stall(r)
            if stalled is not None:                       # unrecoverable reach -> fail fast
                return Result('failed', reason=stalled)
            deadline=getattr(p.manipulation,'valid_until_steady',None)
            if deadline is not None and r.status not in ('failed','unavailable'):
                # fm346 / freshness-sim-clock-not-wall: valid_until_steady is a SIM-clock deadline, so
                # check it on the SIM clock (packet.simulation_time), NOT wall time.monotonic -- under
                # render pacing wall races far ahead of sim and (since the two clocks are different
                # epochs) a wall check trips the instant a place sets valid_until_steady (run-272 'adapter
                # failure: acquisition-to-use freshness expired', first run to reach the place with a
                # passing status). A genuinely hung sim is still caught by _sensor_stall (stall_window_s)
                # + the 30 s /sensors wall backstop, so no separate wall bound is needed here.
                require_fresh(deadline, lambda: p.simulation_time)
            # MotionPort owns trajectory commands; port must only dispatch
            # gripper/stop controls, never issue a second trajectory for `move`.
            self.port.controls(r.controls)
            if r.status == 'succeeded':
                if not r.evidence:
                    raise ValueError('transfer returned no observation evidence')
                last = r.evidence[-1]
                expected = self.job.shelf_id if self.action == 'transfer' else self.manip.cart_location
                key = 'released' if self.action == 'transfer' else 'loaded'
                ground_truth = self._ground_truth_place_evidence(last, expected, p, key)
                if ground_truth is not None:
                    return Result('succeeded', ground_truth)
                final_sources = sources
                if (gtc.enabled() and self.action in _GT_PLACE_EVIDENCE_STAGES and isinstance(last, dict)
                        and last.get('source') == gtc.PLACE_CHECK):
                    # collect-gt: under RESTOCK_VERIFY_PLACE the stage's final evidence is the controller's
                    # sim-measured ground-truth place check (no camera frames the cart), which has no
                    # sku/observation_id and is not 'sensor' -- so every completed collect would die here
                    # ('transfer evidence mismatch'). Convert it to honestly-labelled ground_truth evidence.
                    last = gtc.stage_evidence_from_place_check(
                        last, sku=self.job.sku, quantity=self.job.quantity, expected_location=expected)
                    final_sources = tuple(sources) + (gtc.SOURCE,)
                if (last.get('source') not in final_sources or last.get('sku') != self.job.sku or
                        last.get('location') != expected or last.get('quantity') != self.job.quantity):
                    raise ValueError('transfer evidence mismatch')
                # Keep the actual final observation identity and clock.
                return Result('succeeded', dict(last, **{key: True}))
            return Result(r.status, reason=r.reason)
        if self.action == 'verify':
            evidence = p.verification
            if evidence is None:
                return Result('running', reason='awaiting independent inventory observation')
            return Result('succeeded', evidence)
        raise RuntimeError('invalid runtime phase')

    def _ground_truth_place_evidence(self, last, expected, p, key):
        """vr1: the stage evidence for a manipulation that succeeded on the fm461 GROUND-TRUTH place
        check, or None when not applicable (action not opted in, or the final evidence is a camera
        observation -> the unchanged sensor path). The evidence is stamped by THIS accepting packet (its
        observation id + clock, so the mission's freshness gate measures the acceptance, not the host's
        reply lag) and is honestly labelled source='ground_truth' + the controller's verify_mode, with the
        raw place check kept verbatim -- never relabelled 'sensor'."""
        if self.action not in self.ground_truth_place_actions:
            return None
        if not isinstance(last, dict) or last.get('source') != gtc.PLACE_CHECK:
            return None
        if last.get('location') != expected:
            raise ValueError(f'ground-truth place check location mismatch: '
                             f'{last.get("location")!r} != {expected!r}')
        return {'observation_id': f'ground_truth:{self.action}:{p.observation_id}',
                'source': gtc.SOURCE, 'ground_truth': True, 'simulation_time': p.simulation_time,
                'sku': self.job.sku, 'quantity': self.job.quantity, 'location': expected,
                'verify_mode': last.get('verify_mode') or 'ground_truth',
                'ground_truth_place_check': dict(last), key: True}

    def _step_return_uncouple(self, p, dt):
        """Uncouple-at-parking-spot execute legs (brake the parked cart, release the tow coupling, then
        start the empty drive home), each a distinct observable step so the behaviour tree shows them in
        series. vr1: an unsettled train WAITS (bounded) rather than failing -- see the caller."""
        if self._return_leg == 'brake':
            if not p.parked:
                self._uncouple_wait_s += dt
                if self._uncouple_wait_s > _UNCOUPLE_SETTLE_MAX_S:
                    return Result('failed', reason=f'cart brake refused: base not parked for '
                                                   f'{self._uncouple_wait_s:.1f}s at the cart parking spot')
                return Result('running', reason='waiting for measured parking before braking the cart')
            self.mobile.hold_cart_pin()   # pin the parked cart to the world before releasing the joint
            self._cart_pinned_for_uncouple = True
            self._uncouple_wait_s = 0.0
            self._return_leg = 'uncouple'
            return Result('running', reason='cart braked at parking spot; releasing coupling next')
        if self._return_leg == 'uncouple':
            if not (p.parked and _train_stationary(p.docking)):
                self._uncouple_wait_s += dt
                if self._uncouple_wait_s > _UNCOUPLE_SETTLE_MAX_S:
                    return Result('failed', reason=f'uncouple refused: train not stationary/parked for '
                                                   f'{self._uncouple_wait_s:.1f}s at the cart parking spot')
                return Result('running', reason='waiting for a stationary train before releasing the coupling')
            try:
                state = self.mobile.uncouple(p.docking)   # release the tow coupling (once)
            except ValueError as exc:
                # Coupler.detach refuses a stale/moving observation BEFORE removing the joint (state is
                # unchanged), so retrying on a later, settled tick is safe; bounded like the settle wait.
                self._uncouple_wait_s += dt
                if self._uncouple_wait_s > _UNCOUPLE_SETTLE_MAX_S:
                    return Result('failed', reason=f'uncouple refused: {exc}')
                return Result('running', reason=f'uncouple deferred ({exc}); retrying on a settled tick')
            if state.get('joint_present'):
                return Result('failed', reason='uncouple failed: tow coupling still present')
            self._cart_left_parked = True
            self._cart_pinned_for_uncouple = False   # the pin is now the parked cart's own brake (kept)
            self._return_leg = 'uncouple_wait'
            return Result('running', reason='coupling released; awaiting measured uncoupled observation')
        if self._return_leg == 'uncouple_wait':
            if p.coupled:
                return Result('running', reason='awaiting measured uncoupled observation')
            backaway = self.poses.get('cart_parking_backaway')
            if backaway is not None:
                self._start_nav_leg(backaway, False)   # pull clear of the parked cart, uncoupled
                self.phase = 'navigate'
                self._return_leg = 'cart_parking_backaway'
                return Result('running', reason='uncoupled; backing away from the parked cart')
            self._start_nav_leg(self.poses['home'], False)   # no back-away waypoint: drive home empty
            self.phase = 'navigate'
            self._return_leg = 'home'
            return Result('running', reason='uncoupled; driving home empty')
        raise RuntimeError(f'invalid return leg at execute: {self._return_leg!r}')

    def step(self, dt):
        try:
            return self._step(dt)
        except Exception as exc:
            self.stop()
            # Keep the originating frame(s) in the reason: the mission records only this
            # string (SimulationApp.close() can hard-exit before any outer handler), so a
            # bare message leaves runtime-tick failures unlocatable.
            import traceback
            where = ' | '.join(
                f'{__import__("os").path.basename(fr.filename)}:{fr.lineno} {fr.name}'
                for fr in traceback.extract_tb(exc.__traceback__)[-3:])
            return Result('failed', reason=f'{type(exc).__name__}: {exc} [{where}]')
