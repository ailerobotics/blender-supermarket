"""Simulator-independent mobile platform controls and conservative geometry.

Coordinates: metres, radians, +X forward, +Y left, +Z up. Poses and velocities
must come from simulation sensors/state, never a route's desired pose. No method
here moves a body. The simulator applies wheel targets and creates/removes the
physical hitch joint via callbacks only after the coupling gate succeeds.
"""
from dataclasses import dataclass, asdict
import math


def wrap(a):
    return math.atan2(math.sin(a), math.cos(a))


@dataclass(frozen=True)
class MobileSpec:
    length: float = 1.0
    width: float = 0.60
    height: float = 0.20
    base_z: float = 0.24
    mass: float = 180.0
    wheel_radius: float = 0.16
    wheel_track: float = 0.70
    wheel_width: float = 0.08
    max_speed: float = 0.35
    max_yaw_rate: float = 0.35
    max_wheel_speed: float = 4.0
    arm_mount_z: float = 0.50
    # Conservative transport envelope; validate actual imported FK before driving.
    transport_radius: float = 0.65
    hitch_x: float = -0.62
    hitch_z: float = 0.28
    cart_length: float = 0.80
    cart_width: float = 0.68
    cart_height: float = 1.65
    cart_mass: float = 35.0
    cart_hitch_x: float = 0.65
    max_articulation: float = math.radians(65)

    def __post_init__(self):
        for key, value in asdict(self).items():
            if not math.isfinite(value) or (key != 'hitch_x' and value <= 0):
                raise ValueError(f'invalid {key}: {value}')
        if self.width + self.wheel_width > self.wheel_track:
            raise ValueError('drive wheels intersect chassis; increase track or reduce width')
        if self.length < 0.5 or self.cart_length < 0.4 or self.cart_width < 0.3 or self.cart_height < 1.0:
            raise ValueError('dimensions too small for caster/stock support geometry')
        if self.hitch_x >= -self.length / 2:
            raise ValueError('hitch must be behind chassis')
        if self.cart_hitch_x <= self.cart_length / 2:
            raise ValueError('cart hitch must extend beyond deck')
        if self.arm_mount_z <= self.base_z + self.height / 2:
            raise ValueError('arm mount must be above chassis')

    def manifest(self):
        return {**asdict(self), 'units': 'metres, radians, kg',
                'wheel_joints': ['left_wheel_joint', 'right_wheel_joint'],
                'frames': ['base_link', 'arm_mount', 'hitch', 'lidar_frame',
                           'front_camera_frame', 'imu_frame'],
                'robot_hitch_local': [self.hitch_x, 0, self.hitch_z-self.base_z],
                'cart_hitch_local': [self.cart_hitch_x, 0, self.hitch_z-0.20],
                'transport_requires_measured_envelope': True}


@dataclass(frozen=True)
class Pose2:
    x: float
    y: float
    yaw: float

    def __post_init__(self):
        if not all(math.isfinite(v) for v in (self.x, self.y, self.yaw)):
            raise ValueError('pose must be finite')

    def point(self, x, y=0):
        c, s = math.cos(self.yaw), math.sin(self.yaw)
        return (self.x+c*x-s*y, self.y+s*x+c*y)


def wheel_targets(v, omega, spec=MobileSpec(), *, transport_ready=False,
                  coupled=False, articulation=0.0):
    """Return left/right rad/s; arm interlock and jackknife guard fail closed.

    Reverse towing is deliberately disabled for initial bring-up. The caller
    must hold the upstream home joints and verify their measured errors/envelope.
    """
    if not all(math.isfinite(x) for x in (v, omega, articulation)):
        raise ValueError('nonfinite drive command')
    if not transport_ready or (coupled and abs(articulation) >= spec.max_articulation):
        return (0.0, 0.0)
    v = max(0.0 if coupled else -spec.max_speed, min(spec.max_speed, v))
    omega = max(-spec.max_yaw_rate, min(spec.max_yaw_rate, omega))
    # No pivot turns under tow; enforce a 1.5 m nominal hitch turning radius.
    if coupled:
        omega = max(-abs(v)/1.5, min(abs(v)/1.5, omega))
    left = (v-omega*spec.wheel_track/2)/spec.wheel_radius
    right = (v+omega*spec.wheel_track/2)/spec.wheel_radius
    scale = max(1, abs(left)/spec.max_wheel_speed, abs(right)/spec.max_wheel_speed)
    return left/scale, right/scale


def transport_ready(measured, target, envelope_radius, spec=MobileSpec(), tolerance=0.05):
    """All configured transport joints must be measured and within tolerance."""
    return (bool(target) and math.isfinite(envelope_radius)
            and 0 < envelope_radius <= spec.transport_radius
            and all(k in measured and math.isfinite(measured[k])
                    and abs(measured[k]-v) <= tolerance for k, v in target.items()))


@dataclass(frozen=True)
class DockObservation:
    robot: Pose2
    cart: Pose2
    robot_hitch_z: float
    cart_hitch_z: float
    robot_speed: float
    cart_speed: float
    robot_yaw_rate: float
    cart_yaw_rate: float
    # Observation age supplied from the simulator's monotonic clock.
    age: float = 0.0


class Coupler:
    """Validated physical latch state; callbacks must author/remove a real joint.

    State changes only if the callback succeeds. Call from the simulator thread
    while paused/between physics steps. No transform edits are permitted.
    """
    def __init__(self, spec=MobileSpec()):
        self.spec = spec
        self.attached = False

    def _stationary(self, obs):
        vals = (obs.robot_speed, obs.cart_speed, obs.robot_yaw_rate, obs.cart_yaw_rate,
                obs.robot_hitch_z, obs.cart_hitch_z, obs.age)
        return (all(math.isfinite(v) for v in vals) and 0 <= obs.age <= 0.1
                and max(abs(v) for v in vals[:4]) <= 0.02)

    def attach(self, obs, create_joint):
        if self.attached:
            raise ValueError('already coupled')
        a = obs.robot.point(self.spec.hitch_x)
        b = obs.cart.point(self.spec.cart_hitch_x)
        if (not self._stationary(obs) or math.dist(a, b) > 0.025
                or abs(obs.robot_hitch_z-obs.cart_hitch_z) > 0.015
                or abs(wrap(obs.robot.yaw-obs.cart.yaw)) > math.radians(5)):
            raise ValueError('dock rejected: stale, moving, distant, or misaligned')
        create_joint()
        self.attached = True

    def detach(self, obs, remove_joint):
        if not self.attached or not self._stationary(obs):
            raise ValueError('detach requires coupled, fresh, stopped bodies')
        remove_joint()
        self.attached = False


def convex_hull(points):
    points = sorted(set(points))
    def cross(o, a, b):
        return (a[0]-o[0])*(b[1]-o[1])-(a[1]-o[1])*(b[0]-o[0])
    def half(seq):
        out = []
        for p in seq:
            while len(out) >= 2 and cross(out[-2], out[-1], p) <= 0:
                out.pop()
            out.append(p)
        return out
    return half(points)[:-1]+half(reversed(points))[:-1]


def occupied_envelope(robot, cart=None, spec=MobileSpec(), margin=0.05):
    """Convex conservative outline includes arm, wheels, drawbar and separate cart.

    Cart pose must be measured/predicted independently, not assumed collinear.
    A circumscribed polygon contains the entire circular arm transport envelope.
    """
    if not math.isfinite(margin) or margin < 0:
        raise ValueError('invalid margin')
    radius = max(spec.transport_radius,
                 math.hypot(spec.length/2, (spec.wheel_track+spec.wheel_width)/2))
    radius = (radius+margin)/math.cos(math.pi/16)
    points = [robot.point(radius*math.cos(i*math.pi/8), radius*math.sin(i*math.pi/8))
              for i in range(16)]
    points.extend(robot.point(spec.hitch_x-margin, y) for y in (-margin, margin))
    if cart is not None:
        points.extend(cart.point(x, y)
                      for x in (-spec.cart_length/2-margin, spec.cart_hitch_x+margin)
                      for y in (-spec.cart_width/2-margin, spec.cart_width/2+margin))
    return convex_hull(points)


def swept_envelope(samples, spec=MobileSpec(), margin=0.05):
    """Conservative hull for piecewise-linear XY / shortest-yaw interpolation.

    Samples are (robot Pose2, cart Pose2 or None). Inflation bounds rotational
    arcs between samples; actual physics trajectories need sufficiently frequent
    measured samples. This is geometry for planning, not a dynamics integrator.
    """
    samples = list(samples)
    if not samples or any((c is None) != (samples[0][1] is None) for _, c in samples):
        raise ValueError('nonempty samples with consistent coupling required')
    inflation = 0.0
    for (r0, c0), (r1, c1) in zip(samples, samples[1:]):
        for a, b, radius in ((r0, r1, max(spec.transport_radius, abs(spec.hitch_x),
                                       math.hypot(spec.length/2, (spec.wheel_track+spec.wheel_width)/2))),
                             (c0, c1, math.hypot(spec.cart_hitch_x, spec.cart_width/2))):
            if a is not None:
                inflation = max(inflation, radius*(1-math.cos(abs(wrap(b.yaw-a.yaw))/2)))
    return convex_hull(p for r, c in samples
                       for p in occupied_envelope(r, c, spec, margin+inflation))
