"""Integration adapter for navigation, physical coupling and sensor-led transfer.

No simulator imports or fabricated sensor data. A runtime port steps physics,
acquires observations and applies gripper controls. The controllers supplied by
the sibling modules are injected after a guarded simulator boot.
"""
from dataclasses import asdict, dataclass, replace
import math
import time
from ..restock_camera.freshness import require_fresh
from typing import Protocol

from .mission import Result

# Safety cap on the parked collision-map warm-up (RuntimeAdapter 'warm' phase): the warm-up
# normally exits as soon as the station reports a fresh collision_time (a few ticks once nvblox
# has incorporated the first depth), but if it never does, dispatch the grasp anyway so the host's
# startup grace -- not an unbounded warm-up -- bounds a genuinely stuck mapper.
_WARM_MAX_TICKS = 300

# Fail-fast on an unrecoverable collect/transfer reach stall. The sensing capture publishes a
# fresh /sensors frame every tick under a healthy reach (incl. the PR#80 occlusion hold, which
# re-publishes the held registration each tick). If NO fresh /sensors frame is accepted for
# longer than the occlusion-hold window (sim seconds) while the controller is still in
# observe/approach, the reach cannot recover (lost support, permanent decode failure, ...), so
# the mission is failed in seconds instead of spinning 'running' until the slower host
# sensor-stream watchdog. The window matches the PR#80 hold (data/sim-restock-camera-v2);
# the host watchdog remains the backstop and is never bypassed.
_SENSOR_STALL_WINDOW_S = 20.


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
    def __init__(self, controller, *, nav2=False, sensor_navigation=False):
        self.controller, self.nav2 = controller, nav2
        self.sensor_navigation = sensor_navigation
        if nav2 and sensor_navigation:
            raise ValueError('select one navigation sensor adapter')

    def start(self, goal, coupled, *, goal_articulation=None):
        if goal_articulation is None:
            self.controller.start(goal, coupled)
        else:
            self.controller.start(goal, coupled, goal_articulation=goal_articulation)

    def step(self, dt, packet):
        if self.sensor_navigation:
            return self.controller.step(dt, packet.odometry, packet.navigation,
                now=packet.simulation_time, coupled=packet.coupled,
                arm_transport=packet.transport_ready, articulation=packet.articulation)
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


class RuntimeAdapter:
    def __init__(self, navigation, mobile, manipulation, port: Port, poses,
                 *, fixture_mode=False, goal_articulation=None,
                 stall_window_s=_SENSOR_STALL_WINDOW_S):
        self.nav, self.mobile, self.manip, self.port = navigation, mobile, manipulation, port
        if not isinstance(self.manip, ManipulationBridge):
            self.manip = ManipulationBridge(self.manip)
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
        self.time = 0.
        self.action = None
        self.last_packet_id = None

    def start(self, action, job):
        self.mobile.park()
        if hasattr(self.port,'begin_action'):
            self.port.begin_action(action,job)
        self.action, self.job = action, job
        self.phase = 'execute'
        self._reach_start = None          # set on the first execute tick of a collect/transfer reach
        if action in ('collect', 'couple', 'outbound', 'return'):
            goal = {'collect': 'pickup', 'couple': 'cart_dock',
                    'outbound': 'shelf_approach', 'return': 'home'}[action]
            kwargs = ({'goal_articulation':self.goal_articulation[action]}
                      if action in self.goal_articulation else {})
            self.nav.start(self.poses[goal], action in ('outbound', 'return'), **kwargs)
            self.phase = 'navigate'
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

    def stop(self):
        errors = []
        # Try every stop even when one adapter raises. Never command release.
        for call in (self.mobile.park, self.nav.cancel, self.manip.stop, self.port.stop):
            try:
                call()
            except Exception as exc:
                errors.append(str(exc))
        if errors:
            raise RuntimeError('; '.join(errors))

    def _evidence(self, packet, **extra):
        return dict(observation_id=packet.observation_id, source=packet.source,
                    simulation_time=packet.simulation_time, **extra)

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

    def _step(self, dt):
        self.time += dt
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
            r = self.nav.step(dt, p)
            self._navn = getattr(self, '_navn', 0) + 1
            if self._navn % 30 == 1:                     # diagnostic: why navigate does not complete
                print(f'NAV_STEP action={self.action} status={r.status} reason={r.reason!r} '
                      f'transport_ready={p.transport_ready} coupled={p.coupled} parked={p.parked}',
                      flush=True)
            if r.status == 'failed':
                return Result('failed', reason=r.reason)
            if r.status not in ('running', 'succeeded'):
                raise ValueError('invalid navigation status')
            if r.status == 'running':
                self.mobile.command(r.linear, r.angular, p.docking,
                                    transport_ready=p.transport_ready)
                return Result('running', reason=r.reason)
            self.mobile.park()
            if not p.parked:
                return Result('running', reason='waiting for measured parking')
            if self.action in ('outbound', 'return'):
                if not p.coupled:
                    raise ValueError('cart lost at destination')
                limit = self.goal_articulation.get(self.action)
                if limit is not None and abs(p.articulation) >= limit:
                    raise ValueError('cart articulation exceeds manipulation/return goal tolerance')
                key = 'arrived' if self.action == 'outbound' else 'home'
                return Result('succeeded', self._evidence(p, **{key: True}))
            if self.action == 'collect':
                # Warm the host collision map (ESDF + a fresh collision_time) while parked BEFORE
                # dispatching the grasp, so the first /sensors frame lands within the host's sensor
                # watchdog instead of racing a cold ~1 s ESDF incorporation from job start.
                self.phase = 'warm'
                self._warmn = 0
                return Result('running', reason='warming collision map before grasp')
            self.phase = 'execute'
        if self.phase == 'warm':
            self.mobile.park()
            if not p.parked:
                return Result('failed', reason='base moved during collision-map warm-up')
            ready = self.manip.warm(dt, p)
            self._warmn += 1
            if ready or self._warmn >= _WARM_MAX_TICKS:
                self.phase = 'execute'
                # Dispatch the current manipulation job (collect or transfer) now the collision map
                # is warm; the warmed station is kept (the runtime does not re-begin/re-cold it).
                self.manip.start(self.action, asdict(self.job))
                return Result('running', reason=('collision map warm; dispatching grasp' if ready
                                                 else 'warm-up tick cap reached; dispatching grasp'))
            return Result('running', reason='warming collision map')
        self.mobile.park()
        if not p.parked:
            return Result('failed', reason='base moved during stationary operation')
        if self.action == 'couple':
            if self.phase != 'attached':
                self.mobile.attach(p.docking)
                self.phase = 'attached'
                return Result('running', reason='awaiting post-attach observation')
            state = self.mobile.coupling_state(p.docking)
            if p.coupled and state['healthy']:
                return Result('succeeded', self._evidence(p, coupled=True))
            return Result('running', reason='awaiting healthy physical coupling')
        if self.action == 'park':
            if not p.coupled:
                return Result('failed', reason='cart lost before transfer')
            return Result('succeeded', self._evidence(p, stationary=True))
        if self.action in ('collect', 'transfer'):
            r = self.manip.step(dt, p)
            stalled = self._sensor_stall(r)
            if stalled is not None:                       # unrecoverable reach -> fail fast
                return Result('failed', reason=stalled)
            deadline=getattr(p.manipulation,'valid_until_steady',None)
            if deadline is not None and r.status not in ('failed','unavailable'):
                require_fresh(deadline,time.monotonic)
            # MotionPort owns trajectory commands; port must only dispatch
            # gripper/stop controls, never issue a second trajectory for `move`.
            self.port.controls(r.controls)
            if r.status == 'succeeded':
                if not r.evidence:
                    raise ValueError('transfer returned no observation evidence')
                last = r.evidence[-1]
                expected = self.job.shelf_id if self.action == 'transfer' else self.manip.cart_location
                if (last.get('source') not in sources or last.get('sku') != self.job.sku or
                        last.get('location') != expected or last.get('quantity') != self.job.quantity):
                    raise ValueError('transfer evidence mismatch')
                key = 'released' if self.action == 'transfer' else 'loaded'
                # Keep the actual final observation identity and clock.
                return Result('succeeded', dict(last, **{key: True}))
            return Result(r.status, reason=r.reason)
        if self.action == 'verify':
            evidence = p.verification
            if evidence is None:
                return Result('running', reason='awaiting independent inventory observation')
            return Result('succeeded', evidence)
        raise RuntimeError('invalid runtime phase')

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
