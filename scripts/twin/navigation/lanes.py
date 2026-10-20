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
                 waypoint_tol=1.25):
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
        # Set by expand(): the validated train trajectory (an articulated.Route).
        self.route = None

    def matches_goal(self, goal):
        """True iff `goal` (the coupled nav goal) is this lane's destination."""
        if len(goal) != 3 or not all(math.isfinite(v) for v in goal):
            return False
        return (math.dist(goal[:2], self.goal[:2]) <= self.goal_tol_xy
                and abs(wrap(goal[2] - self.goal[2])) <= self.goal_tol_yaw)

    def acquirable(self, pose, cart_yaw=None):
        """True iff the measured base pose is close enough to the lane start for the
        closed-loop follower to acquire the centreline (the controller tracks small offsets;
        a larger displacement means the robot is not on this lane)."""
        if len(pose) != 3 or not all(math.isfinite(v) for v in pose):
            return False
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
        from .articulated import ArticulatedPlanner, TrainState
        planner = ArticulatedPlanner(world, timeout=build_timeout)
        planner.seeds = []
        ga = self.goal_articulation if self.goal_articulation is not None else world.spec.max_articulation
        start = TrainState(self.start[0], self.start[1], self.start[2], self.start_cart_yaw)
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

    @classmethod
    def from_config(cls, cfg):
        if not isinstance(cfg, dict):
            raise ValueError('each tow lane must be an object')
        known = {'name', 'start', 'goal', 'waypoints', 'start_cart_yaw', 'goal_articulation',
                 'goal_tol_xy', 'goal_tol_yaw', 'start_tol_xy', 'start_tol_yaw',
                 'start_tol_cart_yaw', 'waypoint_tol'}
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
