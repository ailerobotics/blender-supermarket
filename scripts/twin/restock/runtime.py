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
from .native_transport import remap_arm_fault_reason
from .staged_start import SettleHold
from . import pace_watchdog

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

# sim-robust-cart-place-descent-v1: the collect manip phases that make up the reposition PLACE onto the
# rollwagen deck (the arm has folded to the carry pose and is now lowering/placing/retreating/stowing).
# During these the EXTENDED, carton-loaded arm's descent reaction transiently shoves the yaw-lock-only
# braked base a few cm in translation (collect2: base x -4.518 -> -4.559, ~4 cm, which flipped the 0.25 s
# windowed `parked` flag False for 3 ticks and tripped the stationary guard mid-descent). Scoped so the
# bounded base-settle tolerance below never relaxes the strict guard outside the loaded place.
_LOADED_PLACE_PHASES = frozenset(
    {'transit', 'preplace', 'place', 'release', 'retreat', 'retreat_tuck', 'stow'})

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


# sim-service-leg-carry-v1: the drive-with-payload base speed/accel profile for the TRANSFER carry→endcap
# leg. The base must not shove the held carton, so while carrying it drives slower and ramps gently. The
# defaults are conservative (well under the nav max 0.25 m/s / 0.35 rad/s): a calm carry the arm's level
# fold + the CARTON_TILT trip keep honest. A config mapping overrides any field; an unknown field raises so
# a typo fails loudly at boot.
_CARRY_DRIVE_DEFAULT = {'max_speed': 0.12, 'max_angular': 0.20, 'max_accel': 0.15, 'max_angular_accel': 0.40}


def _resolve_carry_drive_profile(spec):
    """Normalise a ``carry_drive_profile`` config value into a positive-finite mapping with keys
    ``max_speed`` / ``max_angular`` / ``max_accel`` / ``max_angular_accel`` (m/s, rad/s, m/s^2, rad/s^2).
    ``None`` / absent => the conservative built-in default; a mapping overrides any subset. Raises on a
    non-mapping, an unknown key, or a non-positive/non-finite value (a typo fails loudly, never silently)."""
    out = dict(_CARRY_DRIVE_DEFAULT)
    if spec is None:
        return out
    if not isinstance(spec, dict):
        raise ValueError('carry_drive_profile must be a mapping of {max_speed, max_angular, max_accel, '
                         'max_angular_accel}')
    bad = set(spec) - set(_CARRY_DRIVE_DEFAULT)
    if bad:
        raise ValueError(f'carry_drive_profile: unknown key(s) {sorted(bad)}; '
                         f'allowed {sorted(_CARRY_DRIVE_DEFAULT)}')
    for k, v in spec.items():
        if v is None:
            continue
        v = float(v)
        if not (math.isfinite(v) and v > 0):
            raise ValueError(f'carry_drive_profile.{k} must be a finite positive number')
        out[k] = v
    return out


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

    def pause_station(self, simulation_time, *, suspend_only=False):
        """fm399 (1): cleanly suspend the parked station while the base is about to leave it, so the host
        spams no emergency-stop while it cannot be heartbeat-ed (the run-309 failure). suspend_only=False
        (default, byte-identical): STOP (-> host idle) at the mid-collect reposition drive-start; re-begun
        fresh at place_approach. suspend_only=True (sim-robust-station-pause-on-transfer-v1): PAUSE (-> host
        'paused', epoch kept) when a manip op completes and the mission tows AWAY to a far op; the next op
        re-prepares fresh. No-op for a runtime that does not own sensor I/O (byte-identical)."""
        if self.owns_sensor_io and hasattr(self.controller, 'pause_station'):
            self.controller.pause_station(simulation_time, suspend_only=suspend_only)


class RuntimeAdapter:
    def __init__(self, navigation, mobile, manipulation, port: Port, poses,
                 *, fixture_mode=False, goal_articulation=None,
                 stall_window_s=_SENSOR_STALL_WINDOW_S, route_alerts=None, fine_dock=None,
                 hold_cart_through_couple=False, couple_static_drive=False,
                 uncouple_on_return=False, wall_clock=time.monotonic, nav_stuck_wall_s=None,
                 nav_stuck_sim_s=None,
                 carry_reposition_drive=False, couple_standoff_tolerance=None,
                 couple_cart_relative=None,
                 ground_truth_place_actions=(), staged_settle_s=0.0,
                 collect_place_base_settle_m=None,
                 hold_base_on_reposition_block=False,
                 shelf_staging=False, carry_drive_profile=None, cart_side_bay=False):
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
        # sim-robust-couple-approach-v1 (opt-in, default None => byte-identical): land the couple STANDOFF
        # against the MEASURED cart instead of the stale fixed map pose. The authored couple_standoff
        # carries yaw 0, but the pinned cart's measured yaw varies run to run (coldlivefinal -1.6deg,
        # coupledock +19.2deg), so the base arrives aligned to the MAP, not the CART -- an 8-27deg yaw /
        # 25-470mm hitch residual the fine-dock creep cannot null (it lands on its cross-vs-yaw
        # equilibrium). When set, the couple standoff leg navigates to fine_dock.standoff_from_cart(cart)
        # (on the cart axis, base yaw = measured cart yaw, hitch held back by standoff_offset_m), and on
        # arrival the cart-relative residual is checked tight in cross-track + yaw: inside the capture
        # region => hand to fine-dock from an aligned pose; outside => a bounded cart-measured re-approach;
        # exhausted => fail honestly with the residual. Requires a couple_standoff pose + DockObservation.
        self.couple_cart_relative = None
        if couple_cart_relative is not None:
            cr = dict(couple_cart_relative)
            # capture_cross_m/_yaw_rad gate ENTRY to the fine-dock (the fine-dock still closes to the
            # latch); they are sized to admit any LEGAL standoff arrival without a spurious re-approach.
            # The worst legal hitch cross-track is arrival_xy_tol + |hitch_x|*arrival_yaw_tol ~ 0.10 m
            # (the fine-dock converges from it in <14 s, verified offline), so 0.11 m clears it.
            defaults = {'standoff_offset_m': 0.40, 'capture_cross_m': 0.11, 'capture_yaw_rad': 0.10,
                        'max_reapproach': 3, 'arrival_xy_tol': 0.05, 'arrival_yaw_tol': 0.08}
            unknown = set(cr) - set(defaults)
            if unknown:
                raise ValueError(f'couple_cart_relative: unknown keys {sorted(unknown)}')
            merged = {**defaults, **cr}
            caps = {'standoff_offset_m': (0.05, 1.0), 'capture_cross_m': (0.0, 0.15),
                    'capture_yaw_rad': (0.0, 0.35), 'arrival_xy_tol': (0.0, 0.15),
                    'arrival_yaw_tol': (0.0, 0.35)}
            for key, (lo, hi) in caps.items():
                v = merged[key]
                if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) \
                        or not lo < v <= hi:
                    raise ValueError(f'couple_cart_relative.{key} must be in ({lo}, {hi}]')
            mr = merged['max_reapproach']
            if isinstance(mr, bool) or not isinstance(mr, int) or not 0 <= mr <= 10:
                raise ValueError('couple_cart_relative.max_reapproach must be an int in [0, 10]')
            self.couple_cart_relative = {k: (int(v) if k == 'max_reapproach' else float(v))
                                         for k, v in merged.items()}
        self._couple_reapproach_n = 0   # bounded cart-measured re-approaches this couple stage
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
        # sim-collect-reposition-block-v1 (opt-in, default off => byte-identical): HOLD the base pinned on
        # a non-driving tick (linear==0 AND angular==0) of the uncoupled mid-collect carry reposition,
        # instead of issuing mobile.command(0,0,...) -- which releases the base parking/full brake "to
        # resume driving" even when the planner has no route. Because port.advance() steps physics with
        # the PRIOR tick's brake state (it re-engages the full brake only AFTER world.step), that per-tick
        # release let the carried carton's cantilever moment roll the un-pinned base ~0.28 m NORTH into the
        # live backstock Support over a ~40 s blocked window (boxc1a), flipping free_start False and making
        # the short reposition permanently unroutable (blocked_no_route_timeout). Holding on the stopped
        # tick keeps the base on its proven-clear departure pose, where a route exists with OR without the
        # trusted-static drive. Off (default) keeps mobile.command(0,0,...) => byte-identical.
        self.hold_base_on_reposition_block = bool(hold_base_on_reposition_block)
        # sim-robust-cart-place-descent-v1 (opt-in, default off => byte-identical): the reposition collect
        # PLACE lowers the welded carton onto the rollwagen deck with the arm extended; its descent reaction
        # shoves the YAW-LOCK-ONLY braked base a few cm in TRANSLATION (isaac_port keeps translation free
        # while a carton is welded, to relieve the world->base->arm->weld->carton loop). In collect2 that
        # ~4 cm transient flipped the 0.25 s windowed `parked` flag False for 3 ticks, and the strict
        # stationary guard (`not p.parked`) aborted the stage mid-descent with 'base moved during stationary
        # operation' -- the carton stayed held, the arm never stowed, and every re-begin then failed
        # 'arm_not_in_transport_timeout'. The reaction is an EXPECTED transient that re-settles, and the
        # shove (~4 cm) is far inside the 0.25 m ground-truth place-verify xy tol, so when this is set we
        # tolerate a BOUNDED base excursion from the place_approach anchor through the loaded place phases
        # and let the place/release/stow sequence COMPLETE; a base pushed BEYOND the bound is a real fault
        # with a named reason (never a silent slide). The ground-truth carton-on-deck verify stays the
        # placement proof. None/0 => the strict guard everywhere (byte-identical). Collect-only (the
        # transfer place has its own carry/mode handling); SIM-only (the real base brakes rigidly).
        self.collect_place_base_settle_m = None
        if collect_place_base_settle_m is not None:
            v = collect_place_base_settle_m
            if not (isinstance(v, (int, float)) and not isinstance(v, bool)
                    and math.isfinite(v) and 0 < v <= 0.3):
                raise ValueError('collect_place_base_settle_m must be a finite value in (0, 0.3] m')
            self.collect_place_base_settle_m = float(v)
        # Base (x, y) captured at the place_approach re-park (end of rewarm); the loaded-place base-settle
        # tolerance measures the excursion from it. None until the reposition place begins / after a re-begin.
        self._place_base_anchor = None
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
        # sim-shelf-staging-v1 (captain follow-up, opt-in, default False => byte-identical): re-sequence
        # the sales-floor leg so the COUPLED train never has to make the ~46 deg in-aisle turn to the
        # externally-fixed west-facing endcap service pose (and never exits the doorway angled ~40 deg).
        # Instead the OUTBOUND stage tows the rollwagen STRAIGHT down the portal axis to a staging pose
        # (`shelf_staging`, south of the wall on the axis, coupled articulation ~0 deg the whole way,
        # cart crosses the doorway aligned), BRAKES the cart and RELEASES the tow coupling there (reusing
        # the uncouple mechanic), and leaves the base PARKED (uncoupled) at the staging pose beside the
        # parked cart. The TRANSFER stage then services the endcap ALONE (sim-service-leg-carry-v1): pick
        # the carton off the parked cart at the staging pose, carry it to the endcap held level, place it,
        # drive back -- the sharp turn to the endcap + the carry are done by the free base, not the coupled
        # train. The cart RETRIEVAL on return (re-couple) stays a separate captain decision (NOT here). See
        # docs/examples/service-leg-carry-mission-fragment.json and data/sim-service-leg-carry-v1/report.md.
        self.shelf_staging = bool(shelf_staging)
        # sim-cart-side-bay-v1 (captain option 1, opt-in extension of shelf_staging, default False =>
        # byte-identical). The sim-shelf-staging-v1 OUTBOUND parked the rollwagen ON the portal axis at the
        # staging pose, which BLOCKS the free base from threading the single 2.6 m doorway home (grid A* NO
        # PATH) and does not let a coupled re-couple + tow-home expand. Park the cart instead in a SIDE BAY
        # OFF the portal axis near the shelf: the OUTBOUND tows the COUPLED train straight down the axis
        # (dead-centre doorway crossing) then tucks it into the bay (`cart_side_bay` pose, ~14 deg coupled
        # articulation, cart parked off-axis), brakes + releases it there; the free base services the endcap
        # ALONE and then -- because the bay cleared the on-axis block -- drives HOME EMPTY through the freed
        # doorway on RETURN, leaving the cart parked (uncoupled) at the side bay. The COUPLED tow-home (cart
        # retrieved to the backroom) is a conserved-~180-deg-reorientation hard case left as a named,
        # config-selectable follow-up (docs/examples/cart-side-bay-mission-fragment.json; it is NOT wired
        # here). cart_side_bay EXTENDS shelf_staging, so it requires shelf_staging True + a `cart_side_bay`
        # pose; when active the outbound park goal is `cart_side_bay` (not `shelf_staging`) and the return
        # drives the base home empty. See data/sim-cart-side-bay-v1/report.md.
        self.cart_side_bay = bool(cart_side_bay)
        self._outbound_leg = None       # staging | brake | uncouple | uncouple_wait
        self._shelf_cart_parked = False  # True once the outbound released the cart at the staging pose
        # sim-service-leg-carry-v1 (captain 2026-11-03, opt-in with shelf_staging): the manipulation half
        # of the re-sequence. OUTBOUND leaves the base PARKED (uncoupled) at the staging pose beside the
        # parked cart (no empty drive to the endcap); the TRANSFER stage then services the endcap ALONE --
        # pick the carton off the parked cart at the staging pose, carry it to the endcap with the carton
        # held LEVEL (carton_level, CARTON_TILT trip) and a drive-with-payload base profile, place it, then
        # drive the empty base BACK to the staging pose. The carry-drive reuses fm390's TRANSIT carry-hold
        # handshake (the SAME mechanic as the collect place-on-cart reposition): the arm holds the level
        # fold while the base drives to `shelf_approach`, then re-registers the sensed endcap slot in the
        # new base frame and places. `_transfer_leg`: pick | carry | place | return_staging (None = legacy
        # single-pose transfer). The cart RETRIEVAL on return is a separate captain decision, NOT here.
        self._transfer_leg = None
        self._transfer_carry_pending = False   # carry-drive to the endcap in flight (base moving, carton held)
        self._transfer_carried = False         # True once the base has carried the carton to the endcap
        self._transfer_place_evidence = None   # the manip's place-success evidence, held across the drive back
        self._carry_last_v = 0.0               # last commanded carry linear/angular (for the accel ramp)
        self._carry_last_w = 0.0
        # Drive-with-payload base speed/accel caps for the carry-drive leg (the base must not shove the held
        # carton). A mapping {max_speed, max_angular} (m/s, rad/s); None => a conservative built-in default.
        # Only applied on the carry→endcap leg (carton held); every other leg keeps its normal profile.
        self.carry_drive_profile = _resolve_carry_drive_profile(carry_drive_profile)
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
                     'cart_parking', 'cart_parking_backaway', 'shelf_staging', 'cart_side_bay'):
            if name in poses:
                wp = poses[name]
                if len(wp) != 3 or not all(math.isfinite(v) for v in wp):
                    raise ValueError(f'invalid pose: {name}')
        # sim-shelf-staging-v1: the opt-in re-sequence needs a `shelf_staging` pose (the coupled tow goal /
        # cart parking on the portal axis) in addition to the existing `shelf_approach` service pose.
        if self.shelf_staging and 'shelf_staging' not in poses:
            raise ValueError('shelf_staging mode requires a "shelf_staging" pose')
        # sim-cart-side-bay-v1: the side bay extends shelf_staging and needs its own off-axis bay pose.
        if self.cart_side_bay and not (self.shelf_staging and 'cart_side_bay' in poses):
            raise ValueError('cart_side_bay mode requires shelf_staging and a "cart_side_bay" pose')
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
        # sim-robust-watchdogs: SIM-clock no-base-progress budget. A pure-WALL budget charges render-paced
        # wall seconds against the base: under the measured 0.17-0.31x sim-to-wall ratio a 120 s WALL budget
        # is only ~20-37 SIM seconds, so a legitimately slow (but progressing-in-sim) leg can be starved of
        # the SIM-seconds it was meant to get. When set, the base is given this many SIM seconds of no
        # progress before the stage fails -- immune to render pace -- while the WALL budget above stays as
        # the generous GPU-wall-cap backstop (sim-robust-audit rank 3/5 'sim-clock-gate the no-base-progress
        # watchdog'). A genuinely frozen base still trips: the sim clock keeps advancing past the sim budget.
        # None/0 => off (byte-identical); env RESTOCK_NAV_STUCK_SIM_S. See pace_watchdog.is_dead(mode='either').
        if nav_stuck_sim_s is None:
            try:
                nav_stuck_sim_s = float(os.environ.get('RESTOCK_NAV_STUCK_SIM_S', 0.0))
            except (TypeError, ValueError):
                nav_stuck_sim_s = 0.0
        if not (math.isfinite(nav_stuck_sim_s) and 0.0 <= nav_stuck_sim_s <= 3000.0):
            raise ValueError('nav_stuck_sim_s must be a finite number of seconds in [0, 3000]')
        self.nav_stuck_sim_s = float(nav_stuck_sim_s)
        self._nav_stuck_reason = None     # the running reason currently being timed
        self._nav_stuck_wall = None       # wall time that reason/pose last changed (progress)
        self._nav_stuck_sim = None        # sim time (self.time) at that progress mark
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
        self._outbound_leg = None         # set when the shelf-staging multi-leg outbound runs
        # sim-service-leg-carry-v1: reset the transfer service-leg carry state so a re-begun transfer
        # (BT StageRetry / fm363 STEP_RETRY) restarts the pick->carry->place->drive-back arc cleanly.
        self._transfer_leg = None
        self._transfer_carry_pending = False
        self._transfer_carried = False
        self._transfer_place_evidence = None
        self._carry_last_v = 0.0
        self._carry_last_w = 0.0
        if action in ('collect', 'couple'):
            # sim-shelf-staging-v1: a fresh collect or a (re-)couple re-attaches the cart, so forget any
            # prior shelf-staging park; it is set True again only when the outbound releases the cart.
            self._shelf_cart_parked = False
        self._reach_start = None          # set on the first execute tick of a collect/transfer reach
        # fm390 mid-collect base reposition: driven once per collect when the arm reaches TRANSIT (carry
        # fold done) and a place_approach pose is configured. Reset here so a re-begun collect starts clean.
        self._collect_reposition_pending = False
        self._collect_repositioned = False
        self._reposition_seq = []          # fm390 nav proof #2: staged drive to place_approach
        self._reposition_seq_i = 0
        self._place_base_anchor = None     # re-armed at the place_approach re-park (sim-robust-cart-place-descent-v1)
        if action == 'couple':
            # sim-robust-station-pause-on-transfer-v1: the collect op is COMPLETE and the base is about to tow
            # AWAY to the far transfer op. No op heartbeats the parked collect/backstock station across the
            # whole couple -> outbound -> park -> transfer-warm span, so without a pause its host-side liveness
            # watchdog accrues a heartbeat gap and false-trips the global /manipulation/stop on this idle-but-
            # ALIVE station (coldwin2: transfer sim gap 30.05 s, 0.05 s over budget -> the station recovered
            # one tick later, proving it was alive). Apply the fm399 pause_station pattern to the between-ops
            # nav: SUSPEND the station NOW, while still parked, so the 'pause' lands (host state 'paused', NOT
            # faulted -> no emergency stop); the transfer warm re-prepares a FRESH epoch at the bay. No-op when
            # no station was prepared (a start_at=couple/transfer staged start skips collect) or for a runtime
            # without sensor I/O (byte-identical). See docs/host_timers_audit.md.
            if hasattr(self.manip, 'pause_station'):
                self.manip.pause_station(self.time, suspend_only=True)
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
            self._couple_reapproach_n = 0   # fresh bounded re-approach budget for this couple attempt
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
            elif action == 'outbound' and self._shelf_staging_active():
                # sim-shelf-staging-v1: tow the COUPLED train STRAIGHT down the portal axis to the staging
                # pose (first leg of the re-sequenced sales-floor stage); the execute phase then brakes +
                # releases the cart there and leaves the base PARKED (uncoupled) at the staging pose. The
                # carton-carry service to the endcap is the transfer service leg (sim-service-leg-carry-v1).
                # sim-cart-side-bay-v1: when the side bay is active, the OUTBOUND tows the cart to the
                # OFF-AXIS `cart_side_bay` pose instead (a gentle straight-then-tuck lane; cart parks off
                # the portal axis so the free base can thread the doorway home); the brake/uncouple legs
                # are identical.
                park_goal = 'cart_side_bay' if self._cart_side_bay_active() else 'shelf_staging'
                kwargs = ({'goal_articulation': self.goal_articulation['outbound']}
                          if 'outbound' in self.goal_articulation else {})
                self._start_nav_leg(self.poses[park_goal], True, **kwargs)
                self.phase = 'navigate'
                self._outbound_leg = 'staging'
            elif action == 'return' and self._cart_side_bay_active() and self._shelf_cart_parked:
                # sim-cart-side-bay-v1: the cart is parked (uncoupled) in the OFF-AXIS side bay, so the
                # free base drives HOME EMPTY through the freed doorway (the side bay cleared the on-axis
                # block the on-axis staging left -- grid A* home now routes). The cart stays at the bay;
                # the coupled tow-home (cart retrieved to the backroom) is the named config follow-up.
                self._start_nav_leg(self.poses['home'], False)
                self.phase = 'navigate'
                self._return_leg = 'home'
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
            # sim-service-leg-carry-v1: in shelf-staging mode the transfer is the SERVICE LEG -- pick at the
            # staging pose, carry to the endcap, place, drive back. Mark the carry leg so the BT milestone
            # view decomposes it into its seven position-first steps (see stage_phases.TRANSFER_STAGING_PLAN).
            if self._shelf_staging_active():
                self._transfer_leg = 'pick'
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

    def _fine_dock_spec(self):
        """MobileSpec the couple geometry is expressed in: the fine-dock's own spec when present."""
        from twin.mobile import MobileSpec
        spec = getattr(self.fine_dock, 'spec', None)
        return spec if spec is not None else MobileSpec()

    def _start_cart_relative_standoff(self, p):
        """Navigate the couple standoff leg to a pose on the MEASURED cart axis (not the stale map pose).

        ``fine_dock.standoff_from_cart`` seats the base on the measured cart's hitch axis (base yaw =
        cart yaw, rear hitch held ``standoff_offset_m`` back), so the fine-dock insert starts aligned
        regardless of where the pinned cart settled. The arrival tolerance is tight in xy + yaw so the
        cart-relative residual check passes on arrival by construction."""
        from twin.restock.fine_dock import standoff_from_cart
        cr = self.couple_cart_relative
        cart = p.docking.cart
        pose = standoff_from_cart(cart, cr['standoff_offset_m'], self._fine_dock_spec())
        kw = {'arrival_xy_tol': cr['arrival_xy_tol'], 'arrival_yaw_tol': cr['arrival_yaw_tol']}
        if self.couple_static_drive:
            kw['static_drive'] = True
        self._couple_standoff_pose = pose
        self._start_nav_leg(pose, False, **kw)

    def _couple_capture_decision(self, p):
        """At the cart-relative standoff arrival: 'enter' | 'reapproach' | 'fail'.

        The capture gate is the OFF-AXIS alignment measured against the cart -- |cross-track| and |yaw|
        -- the pair that decides whether the fine-dock creep can converge (the along-axis hitch offset is
        the deliberate standoff hold-back the servo creeps in, not a miss). Inside the region => enter the
        fine-dock from an aligned pose; outside with re-approaches left => re-approach the cart-measured
        standoff; outside with none left => fail honestly with the residual (never loosen the latch)."""
        from twin.restock.fine_dock import cart_relative_residuals
        cr = self.couple_cart_relative
        hm, cross, yr = cart_relative_residuals(p.docking, self._fine_dock_spec())
        if abs(cross) <= cr['capture_cross_m'] and abs(yr) <= cr['capture_yaw_rad']:
            return 'enter'
        self._couple_capture_reason = (
            f'couple standoff outside capture after {self._couple_reapproach_n} re-approach(es): '
            f'hitch={hm*1000:.1f}mm cross={cross*1000:.1f}mm yaw={math.degrees(yr):.2f}deg '
            f'(capture cross<={cr["capture_cross_m"]*1000:.0f}mm yaw<={math.degrees(cr["capture_yaw_rad"]):.1f}deg)')
        if self._couple_reapproach_n < cr['max_reapproach']:
            return 'reapproach'
        return 'fail'

    def _uncouple_return(self):
        """True when the return stage leaves the cart parked and drives home empty, instead of towing
        it home coupled: opted in via config (``uncouple_on_return``) AND a ``cart_parking`` pose is
        configured. Default off (or no parking pose) => the legacy tow-home return, byte-identical."""
        return self.uncouple_on_return and 'cart_parking' in self.poses

    def _shelf_staging_active(self):
        """True when the opt-in sales-floor re-sequence is in effect: ``shelf_staging`` is enabled AND
        both the staging pose (coupled tow goal / cart parking on the portal axis) and the endcap
        ``shelf_approach`` service pose are configured. Default off (or either pose missing) => the legacy
        coupled tow straight to the endcap, byte-identical."""
        return self.shelf_staging and 'shelf_staging' in self.poses and 'shelf_approach' in self.poses

    def _cart_side_bay_active(self):
        """True when the opt-in side-bay re-sequence is in effect: ``cart_side_bay`` is enabled, the
        off-axis bay pose is configured, AND the shelf-staging re-sequence it extends is active. When
        active the OUTBOUND tows the cart to the OFF-AXIS ``cart_side_bay`` pose (not the on-axis
        ``shelf_staging`` pose) and the RETURN drives the base HOME EMPTY through the freed doorway
        (the cart stays parked, uncoupled, in the bay). Default off => the sim-shelf-staging-v1
        on-axis staging behaviour, byte-identical. See data/sim-cart-side-bay-v1/report.md."""
        return (self.cart_side_bay and 'cart_side_bay' in self.poses
                and self._shelf_staging_active())

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
            if self.couple_cart_relative is not None:
                cr = self.couple_cart_relative
                re = f' (+{self._couple_reapproach_n} re-approach)' if self._couple_reapproach_n else ''
                parts.append(f'cart-relative standoff offset {cr["standoff_offset_m"]:g} m{re}')
            fd = getattr(self.fine_dock, 'config', None)
            if fd is not None and getattr(fd, 'progress_cap', False):
                parts.append(f'fine-dock progress cap {fd.progress_max_time_cap_s:g} s')
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
        (both budgets == 0) => always None (byte-identical); a missing odometry pose is treated as
        no-progress so a truly frozen stage still trips. The message names the CURRENT (last) reason.

        sim-robust-watchdogs: the SIM budget (nav_stuck_sim_s) is the pace-aware PRIMARY -- it measures
        no-progress in SIM seconds (self.time), so render pacing cannot shorten the real budget the base
        was meant to get; the WALL budget (nav_stuck_wall_s) stays as the generous GPU-wall-cap backstop.
        is_dead(mode='either'): a stage frozen for either budget is stuck (a real nav stall trips on the
        sim leg because the sim clock keeps advancing past the sim budget while the base does not move;
        a wholly hung sim that cannot advance self.time trips on the wall leg)."""
        if self.nav_stuck_wall_s <= 0 and self.nav_stuck_sim_s <= 0:
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
            self._nav_stuck_sim = self.time
            self._nav_stuck_pose = pose
            self._nav_stuck_reason = reason
            return None
        self._nav_stuck_reason = reason   # keep the latest reason for the failure message
        waited = now_wall - self._nav_stuck_wall
        waited_sim = self.time - self._nav_stuck_sim if self._nav_stuck_sim is not None else None
        # When a SIM budget is set it GOVERNS and is pace-immune: a frozen base trips after
        # nav_stuck_sim_s SIM seconds however slowly the render runs, and the (pace-sensitive) WALL leg is
        # NOT also applied -- otherwise render pacing would shorten the real budget (120 s wall is only
        # ~20-37 s sim at the measured 0.17-0.31x). The legacy WALL budget is used alone only when no sim
        # budget is configured (byte-identical). The orchestrator's external wall SIGKILL cap remains the
        # ultimate GPU-hours backstop in both cases.
        sim_on = self.nav_stuck_sim_s > 0
        if pace_watchdog.is_dead(sim_elapsed=waited_sim, wall_elapsed=waited,
                                 sim_budget=self.nav_stuck_sim_s if sim_on else None,
                                 wall_backstop=self.nav_stuck_wall_s if (self.nav_stuck_wall_s > 0 and not sim_on) else None,
                                 mode='either'):
            if sim_on:
                measure = 'for %.0fs sim (> %.0fs sim budget)' % (waited_sim, self.nav_stuck_sim_s)
            else:
                measure = 'for %.0fs wall (> %.0fs budget)' % (waited, self.nav_stuck_wall_s)
            return ('%s stuck with no base progress %s; last reason %r; '
                    'failing the attempt fast instead of burning the GPU wall cap'
                    % (self.action, measure, reason))
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
        self._nav_stuck_sim = None
        self._nav_stuck_pose = None
        self._nav_stuck_reason = None
        self.nav.start(goal, coupled, **kw)

    def _carry_drive_clamp(self, linear, angular, dt):
        """sim-service-leg-carry-v1 drive-with-payload profile: clamp the commanded base (linear, angular)
        to the carry profile's speed caps AND ramp each toward the command by at most max_accel*dt (a gentle
        base so the held carton is not shoved; the arm keeps it level + the CARTON_TILT trip owns the attitude
        safety). Pure/stateful on ``_carry_last_{v,w}``; applied only while the carry→endcap leg is active."""
        prof = self.carry_drive_profile
        v = max(-prof['max_speed'], min(float(linear), prof['max_speed']))
        w = max(-prof['max_angular'], min(float(angular), prof['max_angular']))
        dv_cap = prof['max_accel'] * max(float(dt), 0.0)
        dw_cap = prof['max_angular_accel'] * max(float(dt), 0.0)
        v = self._carry_last_v + max(-dv_cap, min(v - self._carry_last_v, dv_cap))
        w = self._carry_last_w + max(-dw_cap, min(w - self._carry_last_w, dw_cap))
        self._carry_last_v, self._carry_last_w = v, w
        return v, w

    def _hold_base_on_nav_hold(self, r):
        """sim-collect-reposition-block-v1: should this navigate tick HOLD the base pinned instead of
        driving it? True only when the opt-in hold is enabled AND this is the uncoupled mid-collect carry
        reposition (``_collect_reposition_pending``) AND the navigation result is a full stop
        (linear==0 AND angular==0 -- a blocked/settling hold, not a turn-in-place or a forward creep).

        Keeping the base pinned on a stopped reposition tick stops the loaded chassis drifting off its
        proven-clear departure pose while the planner has no route (the boxc1a failure). A tick that
        commands ANY motion still drives through ``mobile.command`` as before. Off / any other stage /
        any nonzero command => False => byte-identical (``mobile.command`` on every tick)."""
        if not self.hold_base_on_reposition_block:
            return False
        if self.action != 'collect' or not self._collect_reposition_pending:
            return False
        return float(getattr(r, 'linear', 0.) or 0.) == 0. and float(getattr(r, 'angular', 0.) or 0.) == 0.

    def _loaded_place_base_excursion(self, p):
        """sim-robust-cart-place-descent-v1: the measured base xy excursion (m) from the place_approach
        anchor IFF the loaded-place base-settle tolerance applies this tick, else None (keep the strict
        stationary guard -- byte-identical when the key is off).

        Applies only when the opt-in ``collect_place_base_settle_m`` is set AND this is the reposition
        collect place (``_collect_repositioned``) AND the controller is in a loaded place phase AND the
        place_approach anchor + a usable base odometry pose are both present. Any gap => None => strict."""
        if (self.collect_place_base_settle_m is None or self.action != 'collect'
                or not self._collect_repositioned or self._place_base_anchor is None
                or self._last_manip_phase not in _LOADED_PLACE_PHASES):
            return None
        try:
            ox, oy = float(p.odometry.pose[0]), float(p.odometry.pose[1])
        except (AttributeError, TypeError, IndexError, ValueError):
            return None  # no usable base pose => fall back to the strict guard
        ax, ay = self._place_base_anchor
        return math.hypot(ox - ax, oy - ay)

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
                if self._hold_base_on_nav_hold(r):
                    # Blocked/stopped mid-collect carry reposition: HOLD the base pinned (do NOT release
                    # the full brake to a zero-velocity "drive"), so the carried load cannot roll the base
                    # off its proven-clear departure pose. mobile.hold() keeps the already-engaged full
                    # base brake (it never releases it) and locks the wheels. See _hold_base_on_nav_hold.
                    self.mobile.hold()
                else:
                    linear, angular = r.linear, r.angular
                    if self._transfer_carry_pending:
                        # sim-service-leg-carry-v1 drive-with-payload profile: while CARRYING the held carton
                        # to the endcap, cap the commanded base speed + yaw rate and ramp them gently so the
                        # base cannot shove the carton (the arm holds it level; CARTON_TILT is the trip). Only
                        # the carry→endcap leg is clamped; every other leg drives at its normal profile.
                        linear, angular = self._carry_drive_clamp(linear, angular, dt)
                    self.mobile.command(linear, angular, p.docking,
                                        transport_ready=p.transport_ready)
                return Result('running', reason=r.reason)
            self.mobile.park()
            if not p.parked:
                return Result('running', reason='waiting for measured parking')
            if self.action == 'outbound' and self._outbound_leg == 'staging':
                # sim-shelf-staging-v1: reached the staging pose still COUPLED (straight portal-axis tow).
                # Hand off to the execute phase, which brakes the cart and releases the tow coupling; the
                # base is then left PARKED (uncoupled) at the staging pose (the carton-carry service to the
                # endcap is the transfer service leg, sim-service-leg-carry-v1 -- not an empty drive here).
                if not p.coupled:
                    raise ValueError('cart lost before the staging pose')
                limit = self.goal_articulation.get('outbound')
                if limit is not None and abs(p.articulation) >= limit:
                    raise ValueError('cart articulation exceeds manipulation/return goal tolerance')
                self.phase = 'execute'
                self._outbound_leg = 'brake'
                return Result('running', reason='towed to shelf staging pose; braking cart before uncouple')
            if self.action == 'outbound':
                if not p.coupled:
                    raise ValueError('cart lost at destination')
                limit = self.goal_articulation.get('outbound')
                if limit is not None and abs(p.articulation) >= limit:
                    raise ValueError('cart articulation exceeds manipulation/return goal tolerance')
                return Result('succeeded', self._evidence(p, arrived=True, **self._nav_workarounds('outbound')))
            if self.action == 'return':
                if self._cart_side_bay_active() and self._return_leg == 'home':
                    # sim-cart-side-bay-v1: the base drove HOME EMPTY through the freed doorway; the cart
                    # stays parked (uncoupled) in the side bay. Arrival ends the mission -- no coupled cart.
                    return Result('succeeded', self._evidence(p, home=True))
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
            if self.action == 'transfer' and self._transfer_carry_pending:
                # sim-service-leg-carry-v1: carried the held carton to the endcap service pose. Resume the
                # PAUSED arm (its TRANSIT -> preplace) after re-warming the collision map at the endcap base
                # pose; do NOT re-begin the manipulation (the controller is mid-run, holding the level carry
                # fold). Mirrors the collect place_approach arrival (fm390/fm399): the station was cleanly
                # stopped at carry-drive start, so the re-warm begins a FRESH epoch at the endcap and re-reads
                # the sensed slot target in the NEW base frame (controller._reregister_place) before placing.
                self.manip.set_base_repositioned()
                self._transfer_carry_pending = False
                self._transfer_carried = True
                self._transfer_leg = 'place'
                self.phase = 'rewarm'
                self._warmn = 0
                return Result('running', reason='carried to the endcap; re-parking + re-warming before place')
            if self.action == 'transfer' and self._transfer_leg == 'return_staging':
                # sim-service-leg-carry-v1: the empty base drove BACK to the staging pose beside the parked
                # cart -- the service leg is complete (captain: end at staging). Return the held place-success
                # evidence re-stamped to THIS arrival packet (its observation id + sim clock), so the mission's
                # freshness gate measures the acceptance here, not the stale place instant. The `released`
                # fact + sku/quantity/location were validated at the place; only the clock is re-stamped.
                if self._transfer_place_evidence is None:
                    raise RuntimeError('transfer return-to-staging without held place evidence')
                evidence = dict(self._transfer_place_evidence,
                                observation_id=f'service_leg:transfer:{p.observation_id}',
                                simulation_time=p.simulation_time)
                return Result('succeeded', evidence)
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
                # sim-robust-couple-approach-v1: when cart-relative couple is on, the final standoff leg
                # targets the MEASURED cart axis (see _start_cart_relative_standoff) instead of the stale
                # fixed map pose, so the fine-dock starts aligned to wherever the pinned cart settled.
                if (self.couple_cart_relative is not None and self._couple_nav_goal == 'couple_standoff'
                        and getattr(p, 'docking', None) is not None):
                    self._start_cart_relative_standoff(p)
                else:
                    self._start_couple_nav(self._couple_nav_goal)
                self.phase = 'navigate'
                return Result('running', reason=f'couple nav advancing to {self._couple_nav_goal}')
            if self.action == 'couple':
                # sim-robust-couple-approach-v1: cart-relative standoff arrival gate. Measure the hitch's
                # alignment to the MEASURED cart (cross-track + yaw) and either enter the fine-dock from an
                # aligned pose, re-approach a fresh cart-measured standoff (bounded), or fail honestly.
                if (self.couple_cart_relative is not None and getattr(p, 'docking', None) is not None):
                    decision = self._couple_capture_decision(p)
                    if decision == 'reapproach':
                        self._couple_reapproach_n += 1
                        self._start_cart_relative_standoff(p)
                        self.phase = 'navigate'
                        return Result('running',
                                      reason=f'couple standoff outside capture; re-approaching cart-measured '
                                             f'standoff ({self._couple_reapproach_n})')
                    if decision == 'fail':
                        self.mobile.park()
                        return Result('failed', reason=self._couple_capture_reason)
                    # 'enter' => fall through to the fine-dock insert / direct attach.
                if self.fine_dock is not None:
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
                # sim-robust-cart-place-descent-v1: anchor the base pose here (parked at place_approach, a
                # fresh collision map just incorporated) so the loaded-place base-settle tolerance measures
                # the descent-reaction shove from the pose the place was planned against. Only used when
                # collect_place_base_settle_m is set; captured unconditionally (cheap, no behaviour change).
                try:
                    self._place_base_anchor = (float(p.odometry.pose[0]), float(p.odometry.pose[1]))
                except (AttributeError, TypeError, IndexError, ValueError):
                    self._place_base_anchor = None
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
        if self.action == 'outbound' and self._shelf_staging_active() and self._outbound_leg in (
                'brake', 'uncouple', 'uncouple_wait'):
            # sim-shelf-staging-v1: handled ABOVE the stationary parked-guard, mirroring the return uncouple
            # (the brake pin + hitch release perturb the train for a tick or two, so an unsettled tick WAITS
            # with the base held rather than failing). On release the base drives ALONE to the endcap.
            return self._step_outbound_uncouple(p, dt)
        place_shove_tolerated = False
        if not p.parked:
            tol = self._loaded_place_base_excursion(p)
            if tol is None:
                return Result('failed', reason='base moved during stationary operation')
            if tol > self.collect_place_base_settle_m:
                # sim-robust-cart-place-descent-v1: a real runaway (not the expected transient descent
                # reaction) -- fail with a NAMED reason, never a silent slide onto the wrong deck point.
                return Result('failed', reason=(
                    'base pushed %.3f m beyond the %.3f m collect-place settle tolerance by the loaded '
                    'place-descent reaction' % (tol, self.collect_place_base_settle_m)))
            # Expected, bounded place-descent reaction: the base is braked and re-settles; keep placing so
            # the place/release/stow sequence completes. The ground-truth carton-on-deck verify is the
            # placement proof, and the excursion bound above still fails a genuine runaway.
            place_shove_tolerated = True
        if self.action == 'park':
            if not p.coupled and not (self._shelf_staging_active() and self._shelf_cart_parked):
                # sim-shelf-staging-v1: in shelf-staging mode the cart is DELIBERATELY parked at the staging
                # pose and the base is parked UNCOUPLED at the endcap service pose, so an uncoupled park is
                # expected (not a lost cart). Every other mode still requires the cart here (byte-identical).
                return Result('failed', reason='cart lost before transfer')
            return Result('succeeded', self._evidence(p, stationary=True))
        if self.action in ('collect', 'transfer'):
            # sim-robust-place-descent-plan-v1: when the loaded-place base-settle tolerance above tolerated
            # the transient descent-reaction shove, present the station/host parking gate the SAME
            # effective-parked verdict the mission guard just reached. The sensor-owning host runtime feeds
            # this packet's `parked` straight into the station channel's own parking gate
            # (native_channel.StationClient.tick: `if not parked: raise 'station lost measured parking'`)
            # AND the capture's motion.parked() via set_parked. Without this the station gate raises on the
            # IDENTICAL ~cm transient the mission guard just tolerated -- collect3 died exactly here: the
            # place descent reached z=0.608 and was descending (gripper gap 31->15 mm) when a 25.7 mm base
            # shove flipped packet.parked False, the station gate raised, the step returned failed/'host',
            # the arm never placed/released/stowed, and every re-begin then failed arm_not_in_transport.
            # Scoped exactly to the already-tolerated loaded collect place (place_shove_tolerated is only
            # set inside the collect-scoped excursion tolerance); unset/off => p passes through unchanged
            # (byte-identical), so the scripted/still transfer path is untouched.
            manip_packet = replace(p, parked=True) if place_shove_tolerated else p
            r = self.manip.step(dt, manip_packet)
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
            # sim-service-leg-carry-v1: the transfer PICK arm has folded to the carry pose and is HOLDING in
            # TRANSIT -- carry the held carton to the endcap `shelf_approach` service pose BEFORE it places on
            # the endcap. This mirrors the collect place-on-cart reposition above (pause the arm, drive the
            # base, re-register + place at the new pose), re-targeted to the endcap and with a drive-with-
            # payload base profile (clamped in the navigate command below) + the carton held LEVEL
            # (carton_level, CARTON_TILT trip). One-shot per transfer; only fires in shelf-staging mode once
            # the arm reaches TRANSIT (the opt-in transfer_service_carry controller flag produces it).
            if (self.action == 'transfer' and self._shelf_staging_active()
                    and self._last_manip_phase == 'transit' and not self._transfer_carried):
                self.manip.pause_station(self.time)   # cleanly stop the parked station before the base moves
                self._carry_last_v = 0.0
                self._carry_last_w = 0.0
                self._start_nav_leg(self.poses['shelf_approach'], False, carry_drive=True)
                self.phase = 'navigate'
                self._transfer_carry_pending = True
                self._transfer_leg = 'carry'
                return Result('running', reason='transfer pick done; carrying the held carton to the endcap '
                                                'service pose (held level, payload speed profile)')
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
                    evidence = ground_truth
                else:
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
                    evidence = dict(last, **{key: True})
                if (self.action == 'transfer' and self._shelf_staging_active()
                        and self._transfer_place_evidence is None):
                    # sim-service-leg-carry-v1: the carton is placed + released at the endcap. Drive the EMPTY
                    # base BACK to the staging pose beside the parked cart before the stage succeeds (captain:
                    # end the leg at staging). Hold the place-success evidence; re-stamp it to the arrival
                    # packet on return (the navigate-completion return_staging handler above).
                    self._transfer_place_evidence = evidence
                    self._transfer_leg = 'return_staging'
                    self._start_nav_leg(self.poses['shelf_staging'], False)
                    self.phase = 'navigate'
                    return Result('running',
                                  reason='placed at the endcap; driving the empty base back to the staging pose')
                return Result('succeeded', evidence)
            if r.status == 'failed':
                # Cross-process host reason remap (sim-polish-bundle-v1, PR 262 follow-up): when the HOST
                # manipulation process reports a generic downstream symptom (heartbeat/abort/ack-stalled)
                # that the in-process arm-command gate CAUSED by holding the arm, surface the gate's NAMED
                # fault (controller_not_ready / the tracking-envelope trip) so the stage reason names the
                # real cause. No gate fault / an unrelated reason => r.reason unchanged (byte-identical).
                gate_reason = self.port.arm_fault_reason() if hasattr(self.port, 'arm_fault_reason') else None
                return Result('failed', reason=remap_arm_fault_reason(r.reason, gate_reason))
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

    def _step_outbound_uncouple(self, p, dt):
        """sim-shelf-staging-v1 outbound execute legs: brake the cart at the staging pose, release the tow
        coupling (reusing the SAME mechanic as the return uncouple), then start the UNCOUPLED drive to the
        endcap service pose -- each a distinct observable step so the behaviour tree shows them in series.
        An unsettled train WAITS (bounded) rather than failing (the brake/release perturb it for a tick)."""
        if self._outbound_leg == 'brake':
            if not p.parked:
                self._uncouple_wait_s += dt
                if self._uncouple_wait_s > _UNCOUPLE_SETTLE_MAX_S:
                    return Result('failed', reason=f'cart brake refused: base not parked for '
                                                   f'{self._uncouple_wait_s:.1f}s at the shelf staging pose')
                return Result('running', reason='waiting for measured parking before braking the cart')
            self.mobile.hold_cart_pin()   # pin the parked cart to the world before releasing the joint
            self._cart_pinned_for_uncouple = True
            self._uncouple_wait_s = 0.0
            self._outbound_leg = 'uncouple'
            return Result('running', reason='cart braked at shelf staging pose; releasing coupling next')
        if self._outbound_leg == 'uncouple':
            if not (p.parked and _train_stationary(p.docking)):
                self._uncouple_wait_s += dt
                if self._uncouple_wait_s > _UNCOUPLE_SETTLE_MAX_S:
                    return Result('failed', reason=f'uncouple refused: train not stationary/parked for '
                                                   f'{self._uncouple_wait_s:.1f}s at the shelf staging pose')
                return Result('running', reason='waiting for a stationary train before releasing the coupling')
            try:
                state = self.mobile.uncouple(p.docking)   # release the tow coupling (once)
            except ValueError as exc:
                self._uncouple_wait_s += dt
                if self._uncouple_wait_s > _UNCOUPLE_SETTLE_MAX_S:
                    return Result('failed', reason=f'uncouple refused: {exc}')
                return Result('running', reason=f'uncouple deferred ({exc}); retrying on a settled tick')
            if state.get('joint_present'):
                return Result('failed', reason='uncouple failed: tow coupling still present')
            self._shelf_cart_parked = True
            self._cart_pinned_for_uncouple = False   # the pin is now the parked cart's own brake (kept)
            self._outbound_leg = 'uncouple_wait'
            return Result('running', reason='coupling released at staging; awaiting measured uncoupled observation')
        if self._outbound_leg == 'uncouple_wait':
            if p.coupled:
                return Result('running', reason='awaiting measured uncoupled observation')
            # sim-service-leg-carry-v1: the cart is parked + uncoupled at the staging pose and the base is
            # left PARKED there beside it (uncoupled). Outbound is DONE -- the carton-carry service to the
            # endcap is the TRANSFER service leg (pick@staging -> carry -> place -> drive back), NOT an empty
            # drive here (which was PR 274's placeholder; the robot must pick the carton off the cart first).
            return Result('succeeded', self._evidence(p, arrived=True, shelf_staging=True,
                                                      **self._nav_workarounds('outbound')))
        raise RuntimeError(f'invalid outbound leg at execute: {self._outbound_leg!r}')

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
