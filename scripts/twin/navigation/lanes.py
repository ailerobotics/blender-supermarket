"""Pre-planned tow lanes for coupled (robot + cart) drives.

A tow lane is an AUTHORED centreline -- a start pose, an ordered list of via-waypoints with
headings through the backroom/aisles, and a goal pose -- that a COUPLED drive follows instead
of free planning. Each lane is expanded ONCE, at build time, against the STATIC navigation map
into a kinematically valid, swept-envelope-clear train trajectory (the same measured-state
`ArticulatedPlanner` used for free planning, but run once offline at mission boot, never under
the render-paced per-tick budget). At runtime the coupled controller re-validates that stored
trajectory against the LIVE obstacle observation (a cheap swept-envelope replay) and tracks it
closed-loop; it does NOT fall through to the expensive hybrid-A* search, so a loaded host can
never starve a coupled drive into a `planning_timeout`, and the sanctioned authored lane -- not
a free detour -- is the path. A lane blocked by a live obstacle stops the tow (the lane is the
contract), rather than silently re-routing.

Uncoupled drives carry no lane and keep the full planner unchanged.

The geometry (`start`/`goal`/`waypoints`) is config-driven and documented in
`docs/restock_navigation.md`; the swept envelope is the shared `twin.mobile` train envelope
(base + towed cart), identical to the one the free planner and the braking gate use.
"""
import math

from .core import wrap


def _pose3(value, what):
    try:
        p = tuple(float(v) for v in value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f'tow lane {what} must be a finite (x, y, yaw)') from exc
    if len(p) != 3 or not all(math.isfinite(v) for v in p):
        raise ValueError(f'tow lane {what} must be a finite (x, y, yaw)')
    return p


class TowLane:
    """One authored coupled centreline, expanded to a validated trajectory at build time."""

    def __init__(self, name, start, goal, *, waypoints=(), start_cart_yaw=None,
                 goal_articulation=None, goal_tol_xy=0.30, goal_tol_yaw=0.26,
                 start_tol_xy=0.25, start_tol_yaw=0.15, start_tol_cart_yaw=0.20,
                 waypoint_tol=1.25, route_through_waypoints=False):
        if not isinstance(name, str) or not name:
            raise ValueError('tow lane needs a non-empty name')
        self.name = name
        self.start = _pose3(start, f'{name!r} start')
        self.goal = _pose3(goal, f'{name!r} goal')
        self.waypoints = tuple(_pose3(w, f'{name!r} waypoint') for w in waypoints)
        self.start_cart_yaw = (self.start[2] if start_cart_yaw is None
                               else float(start_cart_yaw))
        if not math.isfinite(self.start_cart_yaw):
            raise ValueError(f'tow lane {name!r} start_cart_yaw must be finite')
        self.goal_articulation = None if goal_articulation is None else float(goal_articulation)
        if self.goal_articulation is not None and not (
                math.isfinite(self.goal_articulation) and self.goal_articulation > 0):
            raise ValueError(f'tow lane {name!r} goal_articulation must be positive')
        tols = (goal_tol_xy, goal_tol_yaw, start_tol_xy, start_tol_yaw,
                start_tol_cart_yaw, waypoint_tol)
        if not all(math.isfinite(v) and v > 0 for v in tols):
            raise ValueError(f'tow lane {name!r} tolerances must be positive and finite')
        self.goal_tol_xy, self.goal_tol_yaw = goal_tol_xy, goal_tol_yaw
        self.start_tol_xy, self.start_tol_yaw = start_tol_xy, start_tol_yaw
        self.start_tol_cart_yaw, self.waypoint_tol = start_tol_cart_yaw, waypoint_tol
        # fm454 (option b): when set, expand() routes the train THROUGH the authored waypoints as
        # ordered sub-goals (concatenating per-segment articulated plans with cart_yaw continuity)
        # instead of free-planning start->goal and only validating the waypoints lie near the result.
        # This gives the author DIRECT control over the corner shape -- a gentle, large-radius portal
        # approach + a straight south lead-in (two collinear waypoints) so the towed cart lines up
        # behind the base BEFORE the doorway, replacing the single sharp ~90 deg reorientation loop.
        # Default False => free-plan-and-validate, byte-identical to the historical lane.
        self.route_through_waypoints = bool(route_through_waypoints)
        # Set by expand(): the validated train trajectory (an articulated.Route).
        self.route = None

    def matches_goal(self, goal):
        """True iff `goal` (the coupled nav goal) is this lane's destination."""
        if len(goal) != 3 or not all(math.isfinite(v) for v in goal):
            return False
        return (math.dist(goal[:2], self.goal[:2]) <= self.goal_tol_xy
                and abs(wrap(goal[2] - self.goal[2])) <= self.goal_tol_yaw)

    def acquirable(self, pose, cart_yaw=None, along_route=False):
        """True iff the measured base pose is close enough to the lane for the closed-loop follower to
        acquire the centreline (the controller tracks small offsets; a larger displacement means the
        robot is not on this lane).

        Default: proximity to the lane START only. With `along_route` (fm427/run-323b), proximity to the
        NEAREST authored route point instead -- the reference generalises from the fixed start to the whole
        centreline. run-323b froze 0.06 m from the route but 0.36 m from the start after the follower
        advanced a few waypoints and dropped the route; the start-only check then refused to re-acquire
        (> start_tol_xy) -> lane_not_acquired forever. A follower that has drifted a little ALONG its own
        lane must re-acquire at its progress, not be forced back to the start.

        fm448/run-331: the along-route gate is the SAME criteria the controller's stop_replan_tracking_error
        drop uses -- xy (start_tol_xy) + cart_yaw (start_tol_cart_yaw) -- and deliberately NOT the base
        heading. The start-only branch keeps the base-yaw gate because at the fixed start there is a single
        authored heading to line up with; mid-route the authored base yaw sweeps THROUGH a turn, so the
        nearest-by-xy route point's heading can differ from a base that drifted a little (> start_tol_yaw)
        even though xy AND cart_yaw are both in tolerance. Applying the start's heading gate there made
        acquire STRICTER than the drop that dropped the route, so after a mid-turn drift the follower could
        never re-acquire -> lane_not_acquired churned forever (846 occurrences). The closed-loop follower
        corrects residual heading via its tracking law, so re-acquiring on xy + cart_yaw is sufficient; a
        pose far from EVERY route point (> start_tol_xy from the nearest) still refuses."""
        if len(pose) != 3 or not all(math.isfinite(v) for v in pose):
            return False
        if along_route and self.route is not None and self.route.states:
            best = min(self.route.states, key=lambda s: math.hypot(pose[0] - s.x, pose[1] - s.y))
            ok = math.dist(pose[:2], (best.x, best.y)) <= self.start_tol_xy
            if cart_yaw is not None:
                ok = ok and abs(wrap(cart_yaw - best.cart_yaw)) <= self.start_tol_cart_yaw
            return ok
        ok = (math.dist(pose[:2], self.start[:2]) <= self.start_tol_xy
              and abs(wrap(pose[2] - self.start[2])) <= self.start_tol_yaw)
        if cart_yaw is not None:
            ok = ok and abs(wrap(cart_yaw - self.start_cart_yaw)) <= self.start_tol_cart_yaw
        return ok

    def start_state(self):
        from .articulated import TrainState
        return TrainState(self.start[0], self.start[1], self.start[2], self.start_cart_yaw)

    def expand(self, world, *, build_margin=0.02, build_timeout=30.0):
        """Expand the authored centreline into a validated train trajectory (build time).

        Runs the real planner once against the STATIC map. The authored via-waypoints are then
        asserted to lie on the resulting centreline (within `waypoint_tol`), so the human-
        readable lane and the executed trajectory cannot silently diverge. Raises if the lane
        cannot be realised -- a broken lane fails the build rather than the mission.
        """
        from .articulated import ArticulatedPlanner, Route, TrainState
        planner = ArticulatedPlanner(world, timeout=build_timeout)
        planner.seeds = []
        ga = self.goal_articulation if self.goal_articulation is not None else world.spec.max_articulation
        start = TrainState(self.start[0], self.start[1], self.start[2], self.start_cart_yaw)
        if self.route_through_waypoints and self.waypoints:
            route = self._expand_through_waypoints(planner, world, start, ga, build_margin)
        else:
            route = planner.plan(start, self.goal, (), build_margin, ga)
            if not route.states:
                raise ValueError(f'tow lane {self.name!r} does not expand to a valid route against '
                                 f'the static map: {route.reason}')
            for wp in self.waypoints:
                near = min(math.dist((s.x, s.y), wp[:2]) for s in route.states)
                if near > self.waypoint_tol:
                    raise ValueError(
                        f'tow lane {self.name!r} waypoint {tuple(round(c, 2) for c in wp)} is '
                        f'{near:.2f} m off the expanded centreline (> {self.waypoint_tol} m); '
                        f'the authored lane and the realised trajectory disagree')
        self.route = route
        self.goal_articulation = ga
        return route

    def _expand_through_waypoints(self, planner, world, start, ga, build_margin):
        """fm454 (option b): plan the coupled train THROUGH the authored waypoints as ordered sub-goals.

        Each leg is a full articulated plan from the running train state (x, y, yaw, cart_yaw) to the
        next via-pose, concatenated with cart_yaw carried continuously from one leg's terminal state
        into the next leg's start -- so the author shapes the corner directly (a gentle large-radius
        approach + a straight south lead-in before the doorway) and the realised trajectory IS the
        authored corner, not a free-plan that merely passes near it. Intermediate legs allow the full
        articulation range (the cart settles by the kinematics); only the final leg is held to the
        lane's goal_articulation. Raises if any leg cannot be realised against the static map."""
        from .articulated import Route
        legs = list(self.waypoints) + [self.goal]
        states, curvatures = [], []
        cur = start
        for i, tgt in enumerate(legs):
            leg_ga = ga if i == len(legs) - 1 else world.spec.max_articulation
            seg = planner.plan(cur, tgt, (), build_margin, leg_ga)
            if not seg.states:
                raise ValueError(
                    f'tow lane {self.name!r} leg {i} to {tuple(round(c, 2) for c in tgt)} does not '
                    f'expand against the static map: {seg.reason}')
            seg_states = seg.states
            if states:
                # the leg starts at `cur` (== the previous leg's terminal state); drop the duplicate
                # junction state but keep every curvature (curvatures[k] drives states[k]->states[k+1]).
                seg_states = seg_states[1:]
            states.extend(seg_states)
            curvatures.extend(seg.curvatures)
            cur = seg.states[-1]
        return Route(states, curvatures, 'validated_lane_via')

    @classmethod
    def from_config(cls, cfg):
        if not isinstance(cfg, dict):
            raise ValueError('each tow lane must be an object')
        known = {'name', 'start', 'goal', 'waypoints', 'start_cart_yaw', 'goal_articulation',
                 'goal_tol_xy', 'goal_tol_yaw', 'start_tol_xy', 'start_tol_yaw',
                 'start_tol_cart_yaw', 'waypoint_tol', 'route_through_waypoints'}
        unknown = set(cfg) - known
        if unknown:
            raise ValueError(f'unknown tow lane keys: {sorted(unknown)}')
        if 'name' not in cfg or 'start' not in cfg or 'goal' not in cfg:
            raise ValueError('a tow lane needs name, start and goal')
        tol_keys = ('goal_tol_xy', 'goal_tol_yaw', 'start_tol_xy', 'start_tol_yaw',
                    'start_tol_cart_yaw', 'waypoint_tol')
        return cls(cfg['name'], cfg['start'], cfg['goal'],
                   waypoints=cfg.get('waypoints', ()),
                   start_cart_yaw=cfg.get('start_cart_yaw'),
                   goal_articulation=cfg.get('goal_articulation'),
                   route_through_waypoints=bool(cfg.get('route_through_waypoints', False)),
                   **{k: float(cfg[k]) for k in tol_keys if k in cfg})


class TowLaneGraph:
    """The authored lane set; resolves the lane for a coupled drive by its goal."""

    def __init__(self, lanes):
        self.lanes = list(lanes)
        names = [ln.name for ln in self.lanes]
        if len(set(names)) != len(names):
            raise ValueError('tow lane names must be unique')

    def __bool__(self):
        return bool(self.lanes)

    def resolve_goal(self, goal):
        """The expanded lane whose destination is `goal`, or None (pose checked later)."""
        for lane in self.lanes:
            if lane.route is not None and lane.matches_goal(goal):
                return lane
        return None

    def expand_all(self, world, **kw):
        for lane in self.lanes:
            lane.expand(world, **kw)
        return self

    @classmethod
    def from_config(cls, cfg):
        if cfg is None:
            return None
        if not isinstance(cfg, (list, tuple)):
            raise ValueError('navigation.tow_lanes must be a list of lane objects')
        return cls([TowLane.from_config(c) for c in cfg])
