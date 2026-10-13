"""Integration adapter for navigation, physical coupling and sensor-led transfer.

No simulator imports or fabricated sensor data. A runtime port steps physics,
acquires observations and applies gripper controls. The controllers supplied by
the sibling modules are injected after a guarded simulator boot.
"""
from dataclasses import asdict, dataclass, replace
import math
from typing import Protocol

from .mission import Result


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

    def step(self, dt, packet):
        if self.owns_sensor_io:
            result = self.controller.step(dt, packet.simulation_time)
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
                 *, fixture_mode=False, goal_articulation=None):
        self.nav, self.mobile, self.manip, self.port = navigation, mobile, manipulation, port
        if not isinstance(self.manip, ManipulationBridge):
            self.manip = ManipulationBridge(self.manip)
        self.poses, self.fixture_mode = poses, fixture_mode
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
        if action in ('collect', 'couple', 'outbound', 'return'):
            goal = {'collect': 'pickup', 'couple': 'cart_dock',
                    'outbound': 'shelf_approach', 'return': 'home'}[action]
            kwargs = ({'goal_articulation':self.goal_articulation[action]}
                      if action in self.goal_articulation else {})
            self.nav.start(self.poses[goal], action in ('outbound', 'return'), **kwargs)
            self.phase = 'navigate'
        elif action == 'transfer':
            self.manip.start(action, asdict(job))
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

    def _step(self, dt):
        self.time += dt
        p = self.port.advance(dt)
        sources = ('sensor', 'test_fixture') if self.fixture_mode else ('sensor',)
        if (p.source not in sources or not p.observation_id or
                p.observation_id == self.last_packet_id or
                not math.isfinite(p.articulation) or
                not math.isfinite(p.simulation_time) or not 0 <= self.time-p.simulation_time <= .1):
            raise ValueError('missing, repeated, stale or unsupported sensor packet')
        self.last_packet_id = p.observation_id
        if self.phase == 'navigate':
            r = self.nav.step(dt, p)
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
            self.phase = 'execute'
            if self.action == 'collect':
                self.manip.start('collect', asdict(self.job))
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
            return Result('failed', reason=str(exc))
