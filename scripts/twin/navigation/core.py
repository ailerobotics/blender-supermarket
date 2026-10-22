"""CPU navigation reference and independent safety gate for Nav2 velocities.

A circular bound around the base contains every permitted cart articulation, not
merely its current rectangle. Unknown/outside-map space is occupied. This trades
access to narrow aisles for a simple conservative swept-clearance guarantee.
"""
from dataclasses import dataclass
import heapq
import math
import os


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
                 goal_timeout=180.0, allow_test_fixtures=False, docking=None,
                 directional_zone=True, coupled_creep_floor=0.03):
        self.grid, self.footprint = grid, footprint
        self.max_speed, self.max_angular = max_speed, max_angular
        # fm481: forward creep floor (m/s) used by the COUPLED tow gate (ArticulatedController.gate_velocity)
        # when a yaw correction is wanted but forward speed is near zero, so the straight-lane start-yaw
        # error is absorbed over distance instead of tripping the turn-radius guard.
        self.coupled_creep_floor = float(coupled_creep_floor)
        self.brake_acceleration, self.stale_after = brake_acceleration, stale_after
        self.blocked_timeout, self.goal_timeout = blocked_timeout, goal_timeout
        if not all(math.isfinite(v) and v > 0 for v in (max_speed,max_angular,
                   brake_acceleration,stale_after,blocked_timeout,goal_timeout)):
            raise ValueError('invalid controller limits')
        if docking is not None and not isinstance(docking, DockingApproach):
            raise ValueError('docking must be a DockingApproach or None')
        # Direction-aware braking zone (default on): the circular footprint is checked all
        # around (never shrunk) and the braking-distance extension is applied ONLY along the
        # commanded forward motion (a capsule down the heading), so a no-reverse base never
        # trips obstacle_stop on static geometry in its rear hemisphere it can never drive
        # into. Set False to restore the legacy omnidirectional stop disk (radius+stop in all
        # directions). The footprint radius and every margin are identical either way; the
        # toggle only governs whether the braking inflation is omnidirectional or forward-only.
        self.directional_zone = bool(directional_zone)
        self.allow_test_fixtures = allow_test_fixtures
        # Optional config-driven dock-approach speed cap; None -> disabled (unchanged run).
        self.docking = docking
        self.goal = None
        self.coupled = False
        self.path = []
        self.elapsed = self.blocked = 0.0
        self.terminal = None
        self._docking_active = False
        self.arrival_xy_tol = 0.08
        self.arrival_yaw_tol = 0.08
        # fm396 (B): SIM-only TRUSTED STATIC drive for the mid-collect reposition legs. When set, the route
        # + the obstacle_stop/coverage gates use the STATIC map only (empty live-scan obstacles): the short
        # reposition in front of the parked cart is geometrically proven clear against the static map AND
        # the base/cart footprint (scripts/check_place_approach_nav.py), but the live lidar is polluted by
        # the carry-fold self-geometry (a ~0.78 m ring just outside the 0.75 m footprint -> no route;
        # run-304/307) that the per-tick carry mask did not clear live. Default False => every other leg
        # keeps full live-scan obstacle avoidance (byte-identical). Set only on the uncoupled reposition legs.
        self.static_drive = False
        # fm431 (B1) / run-325 cold TRANSIT: the mid-collect reposition drives the base to place_approach
        # while the arm CARRIES the carton (a stable braked carry fold, NOT the empty stow posture), so
        # obs.arm_transport is False and the nav refuses every drive with 'arm_not_in_transport' -> the
        # reposition never moves and the held arm starves the heartbeat. carry_drive relaxes ONLY that
        # arm-transport gate for the reposition leg (the carried load is in the verified braked carry and
        # already lidar self-masked via carry_self_radius); every other gate (coverage, obstacle, swept
        # envelope, coupling, articulation) is untouched. Default False => the empty-gripper gate holds
        # (byte-identical). Set only on the uncoupled reposition legs, and SAFE only when the arm is in
        # the stable carry -- the in-sim drive + no-drop validation is a GPU check (run-326).
        self.carry_drive = False

    def start(self, goal, coupled, *, arrival_xy_tol=None, arrival_yaw_tol=None, static_drive=False,
              carry_drive=False):
        self.static_drive = bool(static_drive)
        self.carry_drive = bool(carry_drive)
        if len(goal) != 3 or not all(math.isfinite(v) for v in goal):
            raise ValueError('goal must be finite map [x,y,yaw]')
        if arrival_xy_tol is not None and (not math.isfinite(arrival_xy_tol) or arrival_xy_tol <= 0):
            raise ValueError('arrival_xy_tol must be positive')
        if arrival_yaw_tol is not None and (not math.isfinite(arrival_yaw_tol) or arrival_yaw_tol <= 0):
            raise ValueError('arrival_yaw_tol must be positive')
        self.goal, self.coupled = tuple(goal), bool(coupled)
        self.path, self.elapsed, self.blocked, self.terminal = [], 0.0, 0.0, None
        # Per-goal XY arrival tolerance (default 0.08 m); a tighter value is used for the on-axis
        # couple standoff so the fine-dock insert starts close to the cart axis.
        self.arrival_xy_tol = 0.08 if arrival_xy_tol is None else float(arrival_xy_tol)
        # fm406: per-goal YAW arrival tolerance (default 0.08 rad). The fm390 reposition goal
        # place_approach is only a STANDOFF in front of the cart mouth (the deck target re-registers in
        # the new base frame, with the reach gate as the safety net), and the slow static-drive + soft
        # base settle ~0.078 rad / ~0.076 m short -- right at the 0.08 default, so arrival never cleanly
        # latches and the base sits at the goal until a watchdog e-stops (run-313 sat ~44 s). A relaxed
        # reposition tolerance lets the achievable park declare arrived_stopped -> advance to place.
        self.arrival_yaw_tol = 0.08 if arrival_yaw_tol is None else float(arrival_yaw_tol)
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
        if not obs.arm_transport and not self.carry_drive:
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
        # Include measured velocity: braking cannot assume instantaneous stopping.
        speed = max(v,abs(odom.linear))
        stop_distance = speed*self.stale_after + speed*speed/(2*self.brake_acceleration)
        radius = self.footprint.radius(self.coupled)+odom.position_uncertainty
        # fm396 (B): the trusted static reposition leg drives against the static map only -- the carry-fold
        # self-geometry pollutes the live scan (coverage + a phantom near-footprint ring the per-tick mask
        # did not clear), so skip the live coverage certification and use empty live obstacles for the
        # braking-disk stop check. Every other leg keeps the full live coverage + obstacle_stop.
        dyn = () if self.static_drive else obs.obstacles
        if not self.static_drive and not self._coverage_ok(odom, obs, radius, stop_distance, v, w):
            return Result(reason='insufficient_sensor_coverage')
        # fm396 (B)/run-308: on the trusted static reposition the braking inflation (stop_distance =
        # stale_after*speed, with the 3 s render-paced stale_after) overlapped the SOURCE SUPPORT 0.86 m
        # away and tripped obstacle_stop though the footprint radius clears it and the plan routes clear
        # against the static map. There the braking lookahead is dropped entirely (footprint RADIUS only).
        # Otherwise the braking zone bounds every possible braking arc: direction-aware (a forward capsule
        # down the heading) when self.directional_zone, else the legacy omnidirectional disk.
        brake = 0.0 if self.static_drive else stop_distance
        if not self._braking_zone_free(odom.pose, radius, brake, dyn):
            return Result(reason='obstacle_stop')
        return Result(v,w,reason='tracking')

    def _braking_zone_free(self, pose, radius, brake, dyn):
        """Is the braking zone clear of `dyn` obstacles and the static map?

        The circular footprint (`radius`, which circumscribes the whole train) is ALWAYS
        checked all around and is never shrunk -- any obstacle inside the body stops the
        robot regardless of heading. The braking-distance extension `brake` is then added:

        - directional (`self.directional_zone`, default): only along the forward motion -- the
          footprint disk swept `brake` metres down the heading (a capsule). `brake` is derived
          from the forward speed `max(commanded, measured)`, so a stop / rotate-in-place command
          with no residual forward motion has `brake == 0` and only the all-around footprint is
          checked (the circular footprint sweeps onto itself under rotation). A no-reverse base
          can never drive into geometry behind it, so a rear obstacle outside the footprint does
          NOT trip; an obstacle the same distance AHEAD (within the capsule) still does. A base
          still physically coasting forward (measured speed > 0) while commanded to stop keeps
          the forward capsule, because that travel is real.
        - legacy omnidirectional: `brake` is added in every direction (radius+brake disk),
          catching the rear hemisphere a forward-only base can never reach.
        """
        if not self.grid.free(pose, radius, dyn):
            return False  # footprint (all around, never shrunk)
        if brake <= 1e-9:
            return True
        if not self.directional_zone:
            return self.grid.free(pose, radius + brake, dyn)
        nose = (pose[0] + brake * math.cos(pose[2]), pose[1] + brake * math.sin(pose[2]))
        return self.grid.segment_free(pose, nose, radius, dyn)

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
        # fm406 diag (RESTOCK_NAV_DIAG, byte-identical off): why the reposition never latches arrival --
        # the nav's OWN measured xy/yaw error vs the active tolerances + the stop velocities, logged when
        # near the goal (runs 313/314 pinned the base ~0.08-0.12 m short with nav 'tracking' forever and
        # 0 settling_at_goal). Reveals whether the tol is effective at runtime and whether odom sees a
        # larger error than the ground-truth nav-trace. Throttled.
        if os.environ.get('RESTOCK_NAV_DIAG'):
            _xy = math.dist(pose[:2], self.goal[:2])
            if _xy < 0.4:
                self._adiag = getattr(self, '_adiag', 0) + 1
                if self._adiag % 20 == 1:
                    # fm407 pin-cause: also report the gate's verdict at a ZERO command (obstacle_stop /
                    # coverage vs tracking) and the heading error to the goal -- so a stall at the goal is
                    # attributable to the nav gate (footprint vs cart on the static map), a non-'tracking'
                    # gate, or simply the odom xy reading > tol (else-branch, no arrival branch entered).
                    _gate = self.gate_velocity(now, odometry, obs, 0.0, 0.0)
                    print('NAV_ARRIVAL_DIAG xy=%.4f xy_tol=%.3f yaw_err=%.4f yaw_tol=%.3f lin=%.4f ang=%.4f '
                          'gate0=%r in_xy=%s pose=%s goal=%s'
                          % (_xy, self.arrival_xy_tol, abs(wrap(self.goal[2]-pose[2])), self.arrival_yaw_tol,
                             odometry.linear, odometry.angular, _gate.reason, _xy <= self.arrival_xy_tol,
                             tuple(round(c,3) for c in pose), tuple(round(c,3) for c in self.goal)), flush=True)
        if math.dist(pose[:2],self.goal[:2]) <= self.arrival_xy_tol:
            error = wrap(self.goal[2]-pose[2])
            if abs(error) <= self.arrival_yaw_tol:
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
            # fm396 (B): the trusted static reposition leg plans against the static map only (empty live
            # obstacles) -- the short, proven-clear reposition must not be blocked by the carry-fold
            # self-return ring in the live scan (run-304/307). Every other leg uses the live obstacles.
            dyn = () if self.static_drive else obs.obstacles
            points = [pose[:2]]+self.path
            if not self.path or any(not self.grid.segment_free(a,b,radius,dyn)
                                   for a,b in zip(points,points[1:])):
                self.path = self.grid.plan(pose,self.goal,radius,dyn)
            if not self.path:
                # fm390 diagnostic (RESTOCK_NAV_DIAG, byte-identical off): why the plan is empty -- is the
                # START boxed in, the GOAL blocked, or the straight line crossed? Prints free(start)/
                # free(goal), the dynamic-obstacle count, the nearest few to the base, and how many sit
                # inside the footprint disk at start/goal. Throttled.
                if os.environ.get('RESTOCK_NAV_DIAG'):
                    self._ndiag = getattr(self, '_ndiag', 0) + 1
                    if self._ndiag % 20 == 1:
                        ob = obs.obstacles
                        d_start = sorted(math.hypot(pose[0]-q[0], pose[1]-q[1]) for q in ob)[:5]
                        in_s = sum(1 for q in ob if math.hypot(pose[0]-q[0], pose[1]-q[1]) <= radius)
                        in_g = sum(1 for q in ob if math.hypot(self.goal[0]-q[0], self.goal[1]-q[1]) <= radius)
                        print('NAV_DIAG blocked_no_route goal=%s pose=%s r=%.3f free_start=%s free_goal=%s '
                              'seg_free=%s n_obs=%d near5=%s in_start=%d in_goal=%d'
                              % (tuple(round(c, 2) for c in self.goal), tuple(round(c, 2) for c in pose),
                                 radius, self.grid.free(pose, radius, ob), self.grid.free(self.goal, radius, ob),
                                 self.grid.segment_free(pose, self.goal, radius, ob), len(ob),
                                 [round(d, 3) for d in d_start], in_s, in_g), flush=True)
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
