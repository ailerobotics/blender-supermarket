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
    # Recess the passive casters this far ABOVE the drive-wheel ground plane so the two drive wheels
    # become the deterministic primary weight-bearing contacts (a real caster-stabilised diff-drive
    # keeps casters lightly loaded). 0.0 => all four contacts co-planar (legacy, byte-identical).
    caster_lift_m: float = 0.0
    # sim-chassis-pitch-stability-v1: opt-in COMPLIANT CASTER SUSPENSION. Absent/off => the URDF +
    # assets are byte-identical (the casters stay the rigid caster_lift_m recess). When
    # ``caster_suspension`` is True, mobile_assets inserts a sprung prismatic caster-fork joint per
    # caster and physics_setup authors its PhysX linear drive (stiffness ``caster_spring_n_per_m``,
    # damping ``caster_damping_n_s_per_m``, preload ``caster_preload_n`` holding both casters in light
    # ground contact, stroke ``caster_travel_m``). This gives the chassis real PITCH STIFFNESS so the
    # welded carton stays level at the collect set-down (the coldwin33-37 see-saw), while a LIGHT
    # preload keeps most of the weight on the drive wheels so the coupled tow keeps its traction. The
    # suspension is mutually exclusive with caster_lift_m (a sprung caster is not also recessed); set
    # caster_lift_m=0 when enabling it. Design: twin.traction (free_pitch_travel_deg, CasterSuspension,
    # sprung_chassis_pitch_deg, suspension_drive_wheel_share), proof scripts/check_chassis_pitch.py.
    caster_suspension: bool = False
    caster_spring_n_per_m: float = 0.0
    caster_damping_n_s_per_m: float = 0.0
    caster_preload_n: float = 0.0
    caster_travel_m: float = 0.0
    # sim-chassis-pitch-stability-v1 coldwin38 FIX: the sprung caster-link mass. The original 0.2 kg made
    # the stiff drive (k*dt^2/m ~= 83) EXPLODE at reset before any arm motion; the drive must sit in the
    # solver-stable band (see twin.traction.suspension_solver_stable). A physically plausible caster+fork+
    # spring assembly is ~3-5 kg. Required positive when caster_suspension is on.
    caster_suspender_mass_kg: float = 0.0
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

    # Fields that may be zero/negative/bool and so are skipped by the generic positivity sweep.
    _NONPOSITIVE_OK = ('hitch_x', 'caster_lift_m', 'caster_suspension', 'caster_spring_n_per_m',
                       'caster_damping_n_s_per_m', 'caster_preload_n', 'caster_travel_m',
                       'caster_suspender_mass_kg')

    def __post_init__(self):
        for key, value in asdict(self).items():
            if not math.isfinite(value) or (key not in self._NONPOSITIVE_OK and value <= 0):
                raise ValueError(f'invalid {key}: {value}')
        if not 0.0 <= self.caster_lift_m <= 0.03:
            raise ValueError(f'caster_lift_m must be in [0, 0.03] m: {self.caster_lift_m}')
        # sim-chassis-pitch-stability-v1: when the compliant suspension is enabled, require a positive
        # spring, a finite non-negative preload/damping and a positive stroke, and reject the
        # geometrically contradictory combination of a sprung caster that is ALSO recessed.
        if self.caster_suspension:
            if not (self.caster_spring_n_per_m > 0 and self.caster_travel_m > 0):
                raise ValueError('caster_suspension requires positive caster_spring_n_per_m and caster_travel_m')
            if not self.caster_suspender_mass_kg > 0:
                raise ValueError('caster_suspension requires positive caster_suspender_mass_kg (coldwin38: '
                                 'a too-light sprung mass makes the stiff drive explode at reset)')
            if self.caster_preload_n < 0 or self.caster_damping_n_s_per_m < 0:
                raise ValueError('caster_preload_n and caster_damping_n_s_per_m must be non-negative')
            if self.caster_lift_m != 0.0:
                raise ValueError('caster_suspension is mutually exclusive with caster_lift_m; set caster_lift_m=0')
            # coldwin38 FIX: refuse a sprung caster whose stiffness/mass/step ratio is numerically
            # unstable at the mission solver step (the ~83 that exploded the chassis at reset).
            from .traction import suspension_solver_stable, suspension_solver_stability_ratio
            if not suspension_solver_stable(self.caster_spring_n_per_m, self.caster_suspender_mass_kg):
                raise ValueError(
                    'caster suspension is solver-UNSTABLE at dt=1/60: k*dt^2/m_sprung='
                    f'{suspension_solver_stability_ratio(self.caster_spring_n_per_m, self.caster_suspender_mass_kg):.1f}'
                    ' exceeds the stable ceiling (coldwin38 exploded at ~83). Raise caster_suspender_mass_kg '
                    'or lower caster_spring_n_per_m (twin.traction.min_suspender_mass_kg sizes it).')
        elif (self.caster_spring_n_per_m or self.caster_preload_n or self.caster_damping_n_s_per_m
              or self.caster_travel_m or self.caster_suspender_mass_kg):
            raise ValueError('caster suspension params set but caster_suspension is False')
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

    def dock_reject_reasons(self, obs):
        """Which of the four attach gates fail, each with its measured value vs bound.

        Returns a list of ``(category, detail)`` for every FAILING gate (empty when the dock is
        acceptable). Categories are the four the legacy message named -- ``stale`` (observation age),
        ``moving`` (base/cart speed or yaw rate), ``distant`` (hitch-point gap in XY or Z) and
        ``misaligned`` (robot-vs-cart yaw) -- so a caller can report EXACTLY why it rejected instead
        of the undifferentiated 'stale, moving, distant, or misaligned'. Pure; no state change."""
        out = []
        finite = all(math.isfinite(v) for v in (obs.robot_speed, obs.cart_speed, obs.robot_yaw_rate,
                     obs.cart_yaw_rate, obs.robot_hitch_z, obs.cart_hitch_z, obs.age))
        if not finite:
            out.append(('stale', 'a dock measurement is non-finite'))
        if not (0 <= obs.age <= 0.1):
            out.append(('stale', f'age={obs.age:.3f} s not in [0, 0.100] s'))
        rate = max(abs(obs.robot_speed), abs(obs.cart_speed),
                   abs(obs.robot_yaw_rate), abs(obs.cart_yaw_rate)) if finite else float('nan')
        if not (finite and rate <= 0.02):
            out.append(('moving', f'max |speed/yaw-rate|={rate:.4f} > 0.0200 '
                        f'(robot_speed={obs.robot_speed:.4f} cart_speed={obs.cart_speed:.4f} '
                        f'robot_yaw_rate={obs.robot_yaw_rate:.4f} cart_yaw_rate={obs.cart_yaw_rate:.4f})'))
        gap = math.dist(obs.robot.point(self.spec.hitch_x), obs.cart.point(self.spec.cart_hitch_x))
        if gap > 0.025:
            out.append(('distant', f'hitch gap={gap*1e3:.1f} mm > 25.0 mm (XY)'))
        dz = abs(obs.robot_hitch_z - obs.cart_hitch_z)
        if dz > 0.015:
            out.append(('distant', f'hitch Z gap={dz*1e3:.1f} mm > 15.0 mm'))
        yaw = abs(wrap(obs.robot.yaw - obs.cart.yaw))
        if yaw > math.radians(5):
            out.append(('misaligned', f'robot-vs-cart yaw={math.degrees(yaw):.2f} deg > 5.00 deg'))
        return out

    def attach(self, obs, create_joint):
        if self.attached:
            raise ValueError('already coupled')
        reasons = self.dock_reject_reasons(obs)
        if reasons:
            raise ValueError('dock rejected: stale, moving, distant, or misaligned -- '
                             + '; '.join(f'{cat}: {detail}' for cat, detail in reasons))
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
