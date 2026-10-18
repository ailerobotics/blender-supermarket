"""CPU navigation reference and independent safety gate for Nav2 velocities.

A circular bound around the base contains every permitted cart articulation, not
merely its current rectangle. Unknown/outside-map space is occupied. This trades
access to narrow aisles for a simple conservative swept-clearance guarantee.
"""
from dataclasses import dataclass
import heapq
import math


def wrap(a):
    return math.atan2(math.sin(a), math.cos(a))


@dataclass(frozen=True)
class Footprint:
    base_radius: float
    hitch_offset: float
    cart_hitch_to_farthest_corner: float
    margin: float = 0.10
    min_turn_radius: float = 0.0
    max_articulation: float = math.pi

    def __post_init__(self):
        values = (self.base_radius, self.hitch_offset,
                  self.cart_hitch_to_farthest_corner, self.margin,
                  self.min_turn_radius, self.max_articulation)
        if not all(math.isfinite(v) and v >= 0 for v in values):
            raise ValueError('invalid footprint')
        if self.base_radius <= 0 or not 0 < self.max_articulation <= math.pi:
            raise ValueError('invalid base radius or articulation limit')

    def radius(self, coupled):
        return max(self.base_radius, self.hitch_offset +
                   self.cart_hitch_to_farthest_corner if coupled else 0) + self.margin


@dataclass(frozen=True)
class DockingApproach:
    """Config for the final-metre dock approach (e.g. seating the cart hitch at couple).

    Within `approach_distance` of the dock `goal`, the controller caps the commanded
    linear speed to `speed`. A slower command genuinely shortens the braking distance
    (`stop_distance` is a real function of speed), so the forward braking capsule — and
    the omnidirectional stop disk that bounds it — shrink honestly; the footprint radius
    and every safety margin are UNCHANGED. This lets a no-reverse base dock beside static
    geometry that sits in its *rear* hemisphere (which it can never drive into) without
    the full-speed braking reach spuriously tripping `obstacle_stop`. It does NOT mask any
    obstacle and does NOT widen a tolerance. Engaged only for a goal matching `goal`
    within `goal_tolerance`; absent from config -> None -> disabled (run byte-identical,
    real-rig defaults untouched).
    """
    goal: tuple  # (x, y, yaw) map pose of the dock this applies to
    speed: float  # commanded linear-speed cap inside the approach zone (m/s)
    approach_distance: float  # engage within this range of the dock goal (m)
    goal_tolerance: float = 0.25  # XY match radius binding the active goal to this dock

    def __post_init__(self):
        if len(self.goal) != 3 or not all(math.isfinite(v) for v in self.goal):
            raise ValueError('docking goal must be finite (x, y, yaw)')
        if not (math.isfinite(self.speed) and self.speed > 0):
            raise ValueError('docking speed must be positive')
        if not (math.isfinite(self.approach_distance) and self.approach_distance > 0):
            raise ValueError('docking approach_distance must be positive')
        if not (math.isfinite(self.goal_tolerance) and self.goal_tolerance >= 0):
            raise ValueError('docking goal_tolerance must be non-negative')

    def matches(self, goal):
        return math.dist(goal[:2], self.goal[:2]) <= self.goal_tolerance


@dataclass(frozen=True)
class Odometry:
    pose: tuple  # map x, y, yaw from localization, not a stage transform
    stamp: float
    linear: float = 0.0
    angular: float = 0.0
    source: str = 'wheel_odometry_localized'
    position_uncertainty: float = 0.05


@dataclass(frozen=True)
class Observation:
    stamp: float
    obstacles: tuple  # map-frame (x,y) sensor returns, excluding robot/cart
    coupled: bool
    arm_transport: bool
    articulation: float = 0.0
    source: str = 'simulated_lidar'
    coverage_radius: float = 0.0  # legacy full-circle inscribed free reach (diagnostic)
    observed_space: object = None  # optional depth-voxel evidence for articulated gate
    temporal_map: object = None  # controlled static-scene nvblox map, <=600s epoch
    coverage: object = None  # CoverageField: self-occlusion-aware directional certification


def sweep_extent(beta, radius, stop_distance):
    """Far reach of the forward braking capsule along relative bearing `beta`.

    The capsule is the circular footprint (disk of `radius`, which circumscribes the
    whole train including the towed cart) swept `stop_distance` forward along the
    heading. `beta` is measured from the heading. Rear and near-side bearings reach
    only the footprint edge (the capsule never extends behind the base: no-reverse),
    so they are never a swept sector; only forward bearings extend beyond `radius`.
    """
    c = math.cos(beta)
    if c <= 0.0:
        return radius
    s = abs(math.sin(beta))
    if stop_distance * s <= radius:
        return stop_distance * c + math.sqrt(radius * radius - (stop_distance * s) ** 2)
    return radius / s


class CoverageField:
    """Per-beam observed/blind record for direction-of-travel coverage certification.

    A self-occluded (UNKNOWN) beam hit the robot's own rig and certifies nothing
    beyond it (reach 0); an observed (non-self) beam certifies its bearing is not
    blind out to the sensor range (any real obstacle on it is reported separately in
    Observation.obstacles and gated by the obstacle check, so coverage only answers
    "did we observe here, or are we driving blind?"). Certification requires every
    bearing the forward braking sweep reaches BEYOND the footprint to be covered by a
    non-self beam; bearings whose sweep stays within the footprint are the robot's own
    body (self-returns expected there) and need no external observation. The rear is
    never swept, so rig self-occlusion behind the base never blocks motion.
    """
    __slots__ = ('range_max', 'beams', 'origin')

    def __init__(self, range_max, beams, origin=(0.0, 0.0)):
        self.range_max = float(range_max)
        # (map-frame bearing, certified not-blind reach); reach is 0.0 when blind.
        self.beams = tuple((float(a), float(r)) for a, r in beams)
        # Map-frame (x, y) the beams emanate from (the calibrated sensor origin).
        self.origin = (float(origin[0]), float(origin[1]))

    def certifies(self, heading, radius, stop_distance):
        """Circular-footprint certification (uncoupled base / scalar-radius probe).

        Requires every bearing the forward braking capsule enters beyond the circular
        footprint `radius` to be covered by a non-self beam. For the coupled train use
        `certifies_sweep`, which subtracts the actual (narrow) occupied envelope instead
        of this circumscribing circle so the cart's radius does not pull near-broadside
        bearings into the required set.
        """
        if not all(math.isfinite(v) for v in (heading, radius, stop_distance)):
            return False
        if stop_distance < 0 or radius <= 0:
            return False
        for angle, reach in self.beams:
            extent = sweep_extent(wrap(angle - heading), radius, stop_distance)
            if extent <= radius + 1e-9:
                continue  # within the footprint: robot's own body / not swept
            if reach < extent - 1e-9:
                return False  # a forward swept sector is blind (or past sensor range)
        return True

    def certifies_sweep(self, occupied, swept, travel, *, geometry_margin=0.05,
                        forward_cone=math.pi / 4):
        """Certify the braking sweep for a self-occluding robot by subtracting the body.

        `occupied` and `swept` are convex map-frame polygons (the current footprint — base,
        or base + cart — and its braking sweep at the commanded speed/curvature, both
        already carrying the geometry + localization margins); `travel` is the unit heading
        vector. A blind (self-occluded) beam is a refusal only when it enters a *newly swept
        crescent in the direction of travel*: its ray must reach beyond the current occupied
        envelope by more than the geometry margin (`sweep_far - body_far`), AND its bearing
        must lie within `forward_cone` of travel. Bearings outside that cone are the lateral
        and rear extent of the capsule — swept parallel to the body or not swept at all (no
        reverse) — and are covered by the footprint; this holds body-relative, so a rig that
        self-occludes its own sides and rear is exempt there whatever the heading. A
        genuinely unobserved sector in the forward cone still refuses. Observed beams are
        never blind and always pass. The bearing test is used rather than the exit-face
        normal because a front-offset sensor and the faceted hull make near-side rays leave
        the envelope through forward-tilted facets.
        """
        heading = math.atan2(travel[1], travel[0])
        ox, oy = self.origin
        for angle, reach in self.beams:
            if reach > 0.0:
                continue  # observed (non-self): the bearing is not blind
            if abs(wrap(angle - heading)) > forward_cone:
                continue  # lateral/rear of travel: covered by the footprint, never swept
            dx, dy = math.cos(angle), math.sin(angle)
            sweep_far, _ = _ray_convex_exit(ox, oy, dx, dy, swept)
            if sweep_far <= 0.0:
                continue  # ray never enters the braking sweep
            body_far, _ = _ray_convex_exit(ox, oy, dx, dy, occupied)
            if sweep_far - body_far <= geometry_margin:
                continue  # no real forward crescent beyond the body along this ray
            return False  # blind over a newly swept forward crescent the robot enters
        return True


def _ray_convex_exit(ox, oy, dx, dy, poly):
    """Farthest exit (t > 0) of a ray from a convex CCW polygon and that edge's outward
    unit normal. Returns (t, (nx, ny)); t is 0.0 when the ray does not exit forward."""
    best_t, normal = 0.0, (0.0, 0.0)
    n = len(poly)
    for i in range(n):
        ax, ay = poly[i]
        bx, by = poly[(i + 1) % n]
        ex, ey = bx - ax, by - ay
        nx, ny = ey, -ex  # outward normal for a counter-clockwise hull
        den = dx * nx + dy * ny
        if den <= 1e-12:  # parallel, or an entering (back-facing) edge
            continue
        t = ((ax - ox) * nx + (ay - oy) * ny) / den
        if t <= 1e-9 or t <= best_t:
            continue
        px, py = ox + t * dx - ax, oy + t * dy - ay
        s = (px * ex + py * ey) / (ex * ex + ey * ey)
        if -1e-9 <= s <= 1 + 1e-9:
            mag = math.hypot(nx, ny)
            best_t, normal = t, (nx / mag, ny / mag)
    return best_t, normal


def coverage_certifies(obs, heading, radius, stop_distance):
    """Shared coverage gate for both the reference and articulated braking sweeps.

    Uses the self-occlusion-aware directional field when the real pipeline supplied
    one; falls back to the legacy scalar reach for fixture observations that set only
    `coverage_radius`.
    """
    field = getattr(obs, 'coverage', None)
    if field is None:
        return obs.coverage_radius >= radius + stop_distance
    return field.certifies(heading, radius, stop_distance)


@dataclass(frozen=True)
class Result:
    linear: float = 0.0
    angular: float = 0.0
    status: str = 'running'
    reason: str = ''

    @property
    def velocity(self):
        return self.linear, self.angular


class GridMap:
    """Metric bounds [xmin,ymin,xmax,ymax] and occupied closed AABBs.

    Rectangles must include all non-traversable/unknown map regions. Static map
    geometry may come from the environment author; live obstacles must be sensed.
    """
    def __init__(self, bounds, obstacles=(), resolution=0.20, *, corridors=()):
        self.bounds = tuple(bounds)
        self.obstacles = tuple(tuple(r) for r in obstacles)
        self.resolution = resolution
        # Map-derived passage centres and crossing headings are search hints only.
        self.corridors = tuple(tuple(p) for p in corridors)
        if any(len(p)!=3 or not all(math.isfinite(v) for v in p) for p in self.corridors):
            raise ValueError('invalid corridor hint')
        if (len(self.bounds) != 4 or not math.isfinite(resolution) or resolution <= 0
                or not all(math.isfinite(v) for v in self.bounds)
                or self.bounds[2] <= self.bounds[0] or self.bounds[3] <= self.bounds[1]):
            raise ValueError('invalid map')
        for r in self.obstacles:
            if len(r) != 4 or not all(math.isfinite(v) for v in r) or r[2] < r[0] or r[3] < r[1]:
                raise ValueError('invalid obstacle')

    def free(self, p, radius, dynamic=()):
        x, y = p[:2]
        a, b, c, d = self.bounds
        if not a + radius < x < c - radius or not b + radius < y < d - radius:
            return False
        for xmin, ymin, xmax, ymax in self.obstacles:
            if math.hypot(max(xmin-x, 0, x-xmax), max(ymin-y, 0, y-ymax)) <= radius:
                return False
        return all(math.hypot(x-q[0], y-q[1]) > radius for q in dynamic)

    def segment_free(self, a, b, radius, dynamic=()):
        distance = math.dist(a[:2], b[:2])
        n = max(1, math.ceil(distance / (self.resolution / 2)))
        # The half-sample padding proves clearance between samples as well.
        padding = distance / n / 2
        return all(self.free((a[0]+(b[0]-a[0])*i/n, a[1]+(b[1]-a[1])*i/n),
                             radius+padding, dynamic) for i in range(n+1))

    def plan(self, start, goal, radius, dynamic=()):
        if not self.free(start, radius, dynamic) or not self.free(goal, radius, dynamic):
            return []
        if self.segment_free(start, goal, radius, dynamic):
            return [tuple(goal[:2])]
        res = self.resolution
        origin = self.bounds[:2]
        key = lambda p: (round((p[0]-origin[0])/res), round((p[1]-origin[1])/res))
        point = lambda k: (origin[0]+k[0]*res, origin[1]+k[1]*res)
        first = key(start)
        if not self.segment_free(start, point(first), radius, dynamic):
            return []
        queue = [(0.0, first)]
        cost, parent = {first: 0.0}, {first: None}
        closed = set()
        while queue:
            _, k = heapq.heappop(queue)
            if k in closed:
                continue
            closed.add(k)
            p = point(k)
            if math.dist(p, goal[:2]) <= 2*res and self.segment_free(p, goal, radius, dynamic):
                path = [tuple(goal[:2])]
                while k is not None:
                    path.append(point(k))
                    k = parent[k]
                return list(reversed(path))
            for dx, dy in ((1,0),(-1,0),(0,1),(0,-1),(1,1),(1,-1),(-1,1),(-1,-1)):
                nxt = k[0]+dx, k[1]+dy
                q = point(nxt)
                new = cost[k]+math.hypot(dx,dy)*res
                if new < cost.get(nxt, math.inf) and self.segment_free(p,q,radius,dynamic):
                    cost[nxt], parent[nxt] = new, k
                    heapq.heappush(queue, (new+math.dist(q,goal[:2]), nxt))
        return []


class Controller:
    """Stop/turn/drive reference controller for a zero-radius differential base.

    For a coupled cart requiring nonzero turning radius use Nav2 Smac Hybrid;
    this reference route explicitly refuses it. gate_velocity() is shared with
    that route. Coupling and arm state are measured on every tick.
    """
    def __init__(self, grid, footprint, *, max_speed=0.25, max_angular=0.35,
                 brake_acceleration=0.4, stale_after=0.3, blocked_timeout=8.0,
                 goal_timeout=180.0, allow_test_fixtures=False, docking=None):
        self.grid, self.footprint = grid, footprint
        self.max_speed, self.max_angular = max_speed, max_angular
        self.brake_acceleration, self.stale_after = brake_acceleration, stale_after
        self.blocked_timeout, self.goal_timeout = blocked_timeout, goal_timeout
        if not all(math.isfinite(v) and v > 0 for v in (max_speed,max_angular,
                   brake_acceleration,stale_after,blocked_timeout,goal_timeout)):
            raise ValueError('invalid controller limits')
        if docking is not None and not isinstance(docking, DockingApproach):
            raise ValueError('docking must be a DockingApproach or None')
        self.allow_test_fixtures = allow_test_fixtures
        # Optional config-driven dock-approach speed cap; None -> disabled (unchanged run).
        self.docking = docking
        self.goal = None
        self.coupled = False
        self.path = []
        self.elapsed = self.blocked = 0.0
        self.terminal = None
        self._docking_active = False

    def start(self, goal, coupled):
        if len(goal) != 3 or not all(math.isfinite(v) for v in goal):
            raise ValueError('goal must be finite map [x,y,yaw]')
        self.goal, self.coupled = tuple(goal), bool(coupled)
        self.path, self.elapsed, self.blocked, self.terminal = [], 0.0, 0.0, None
        # Engage the dock-approach cap only for the goal the config binds it to.
        self._docking_active = self.docking is not None and self.docking.matches(self.goal)

    def cancel(self):
        self.terminal = Result(status='failed', reason='cancelled')
        return self.terminal

    def _inputs(self, now, odom, obs):
        if odom is None or obs is None:
            return 'missing_sensing'
        vals = (*odom.pose, odom.stamp, odom.linear, odom.angular,
                odom.position_uncertainty, obs.stamp, obs.articulation, obs.coverage_radius)
        if len(odom.pose) != 3 or not all(math.isfinite(v) for v in vals):
            return 'invalid_sensing'
        if odom.position_uncertainty < 0 or obs.coverage_radius < 0:
            return 'invalid_sensing'
        if any(len(p) != 2 or not all(math.isfinite(v) for v in p) for p in obs.obstacles):
            return 'invalid_sensing'
        if any(now-s > self.stale_after or s > now+1e-6 for s in (odom.stamp,obs.stamp)):
            return 'stale_sensing'
        if odom.source != 'wheel_odometry_localized' or obs.source != 'simulated_lidar':
            if not self.allow_test_fixtures or odom.source != 'test_fixture' or obs.source != 'test_fixture':
                return 'unsupported_observation_source'
        if obs.coupled != self.coupled:
            return 'coupling_changed'
        if not obs.arm_transport:
            return 'arm_not_in_transport'
        if self.coupled and abs(obs.articulation) >= self.footprint.max_articulation:
            return 'articulation_limit'
        return ''

    def gate_velocity(self, now, odom, obs, linear, angular):
        """Final gate for BOTH reference and external Nav2 control outputs."""
        reason = self._inputs(now,odom,obs)
        if reason:
            return Result(reason=reason)
        if not all(math.isfinite(v) for v in (linear,angular)) or linear < 0:
            return Result(reason='reverse_or_invalid_command')
        v = min(linear,self.max_speed)
        # Final-metre dock approach: cap the commanded speed near the dock goal so the
        # braking distance (and the stop disk below) shrink honestly. Footprint and
        # margins are untouched; this never masks an obstacle.
        if (self._docking_active and self.goal is not None
                and math.dist(odom.pose[:2],self.goal[:2]) <= self.docking.approach_distance):
            v = min(v,self.docking.speed)
        w = max(-self.max_angular,min(angular,self.max_angular))
        if self.coupled and abs(w)*self.footprint.min_turn_radius > v+1e-9:
            return Result(reason='turn_radius')
        # Include measured velocity: braking cannot assume instantaneous stopping.
        speed = max(v,abs(odom.linear))
        stop_distance = speed*self.stale_after + speed*speed/(2*self.brake_acceleration)
        radius = self.footprint.radius(self.coupled)+odom.position_uncertainty
        if not self._coverage_ok(odom, obs, radius, stop_distance, v, w):
            return Result(reason='insufficient_sensor_coverage')
        # A disk expanded by stopping travel contains every possible braking arc.
        if not self.grid.free(odom.pose,radius+stop_distance,obs.obstacles):
            return Result(reason='obstacle_stop')
        return Result(v,w,reason='tracking')

    def _coverage_ok(self, odom, obs, radius, stop_distance, v, w):
        """Circular-footprint coverage for a plain differential base (overridden by the
        articulated train with occupied-envelope subtraction). `v`/`w` are the gated
        command, available to subclasses that model the swept envelope."""
        return coverage_certifies(obs, odom.pose[2], radius, stop_distance)

    def step(self, dt, odometry, observations, *, now=None):
        if self.terminal:
            return self.terminal
        if not math.isfinite(dt) or dt <= 0 or dt > self.stale_after:
            return Result(reason='invalid_control_period')
        self.elapsed += dt
        if now is None:
            now = self.elapsed
        if self.goal is None:
            return Result(status='failed',reason='no_goal')
        reason = self._inputs(now,odometry,observations)
        if reason:
            return self._wait(dt,reason)
        if self.elapsed > self.goal_timeout:
            self.terminal = Result(status='failed',reason='goal_timeout')
            return self.terminal
        if self.coupled and self.footprint.min_turn_radius > 0:
            self.terminal = Result(status='failed',reason='requires_nav2_curvature_planner')
            return self.terminal
        pose, obs = odometry.pose, observations
        radius = self.footprint.radius(self.coupled)+odometry.position_uncertainty
        if math.dist(pose[:2],self.goal[:2]) <= 0.08:
            error = wrap(self.goal[2]-pose[2])
            if abs(error) <= 0.08:
                if abs(odometry.linear) > 0.02 or abs(odometry.angular) > 0.03:
                    return Result(reason='settling_at_goal')
                clear = self.gate_velocity(now,odometry,obs,0.0,0.0)
                if clear.reason != 'tracking':
                    return self._wait(dt,clear.reason)
                self.terminal = Result(status='succeeded',reason='arrived_stopped')
                return self.terminal
            result = self.gate_velocity(now,odometry,obs,0.0,2*error)
        else:
            # Replan with current observations whenever any remaining segment is blocked.
            points = [pose[:2]]+self.path
            if not self.path or any(not self.grid.segment_free(a,b,radius,obs.obstacles)
                                   for a,b in zip(points,points[1:])):
                self.path = self.grid.plan(pose,self.goal,radius,obs.obstacles)
            if not self.path:
                return self._wait(dt,'blocked_no_route')
            while len(self.path) > 1 and math.dist(pose[:2],self.path[0]) < 0.10:
                self.path.pop(0)
            target = self.path[0]
            error = wrap(math.atan2(target[1]-pose[1],target[0]-pose[0])-pose[2])
            v = min(self.max_speed,math.dist(pose[:2],target)) if abs(error)<0.10 else 0.0
            result = self.gate_velocity(now,odometry,obs,v,2*error)
        if result.reason != 'tracking':
            return self._wait(dt,result.reason)
        self.blocked = 0.0
        return result

    def _wait(self,dt,reason):
        self.blocked += dt
        if self.blocked >= self.blocked_timeout:
            self.terminal = Result(status='failed',reason=reason+'_timeout')
            return self.terminal
        return Result(reason=reason)
