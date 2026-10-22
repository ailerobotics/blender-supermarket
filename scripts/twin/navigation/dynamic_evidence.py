"""Closed-loop CPU evidence for dynamic-obstacle detection and avoidance.

Drives the coupled `ArticulatedController` along an aisle with a kinematic tow
model while a scripted dynamic obstacle acts, folding scenario detections through
the `DynamicObstacleLayer` into the gated/planned observation exactly as the
production sensor route does. It returns a deterministic per-tick trace so the
detect -> stop/re-plan -> resume -> arrive behaviour is provable without Blender,
Isaac or a GPU.

This is a quasi-static kinematic demonstration, NOT a PhysX, sensing or perception
run: the robot integrates the gated command through the same trailer model the
planner checks, and detections are scenario ground truth, not a detector output.
"""
from dataclasses import dataclass, field
import math

from twin.mobile import MobileSpec
from .core import Observation, Odometry
from .articulated import ArticulatedController, TrainState, advance
from .dynamic import DynamicObstacleLayer, DynamicObstacleConfig


@dataclass(frozen=True)
class Tick:
    t: float
    x: float
    y: float
    yaw: float
    cart_yaw: float
    linear: float
    angular: float
    reason: str
    status: str
    detections: int
    tracks: int
    nearest_m: float  # signed clearance to the nearest active detection (inf if none)


@dataclass
class EvidenceRun:
    scenario: str
    ticks: list = field(default_factory=list)
    arrived: bool = False
    arrival_t: float = None
    stopped_for_obstacle: bool = False
    resumed_after_clear: bool = False
    min_clearance: float = math.inf
    goal: tuple = None
    start: tuple = None

    def summary(self):
        moving = [k for k in self.ticks if k.linear > 0.02]
        return dict(scenario=self.scenario, ticks=len(self.ticks),
                    arrived=self.arrived, arrival_t=self.arrival_t,
                    stopped_for_obstacle=self.stopped_for_obstacle,
                    resumed_after_clear=self.resumed_after_clear,
                    min_clearance_m=None if self.min_clearance == math.inf else round(self.min_clearance, 3),
                    first_motion_t=moving[0].t if moving else None,
                    last_motion_t=moving[-1].t if moving else None)


def _nearest_clearance(scenario, x, y, t):
    best = math.inf
    for a in scenario.actors:
        d = a.detection_at(t)
        if d is None:
            continue
        best = min(best, math.hypot(x - d.position[0], y - d.position[1]) - d.radius)
    return best


# Gate reasons that mean "held because the way is not clear" (vs plain progress).
_BLOCKED = {'obstacle_stop', 'blocked_no_route', 'blocked_no_route_timeout',
            'stop_replan_tracking_error', 'replan_changed_observations',
            'occupied_navigation_cell', 'unknown_navigation_cell'}
_PLANNING = {'planning_stop', 'waiting_stopped_for_plan', 'route_ready'}


def simulate(scenario, *, grid, goal, spec=None, config=None, dt=0.1, horizon=90.0,
             start=(0.0, 0.0, 0.0), coverage_radius=50.0, blocked_timeout=30.0,
             goal_timeout=200.0, position_uncertainty=0.02):
    """Run the closed loop and return an EvidenceRun trace.

    `scenario` is a DynamicScenario; `grid` the static GridMap; `goal` the chassis
    (x, y, yaw). The robot starts coupled and at rest, aligned with the cart.
    """
    spec = spec or MobileSpec()
    layer = DynamicObstacleLayer(config or DynamicObstacleConfig())
    controller = ArticulatedController(grid, spec, allow_test_fixtures=True,
                                       blocked_timeout=blocked_timeout, goal_timeout=goal_timeout)
    controller.start(goal, True)
    state = TrainState(start[0], start[1], start[2], start[2])
    run = EvidenceRun(scenario.name, goal=tuple(goal), start=tuple(start))
    linear = angular = 0.0
    seen_obstacle_stop = False
    cleared_after_stop = False
    t = 0.0
    while t <= horizon:
        dets = scenario.detections_at(t)
        layer.update(dets, t)
        obs = Observation(t, (), True, True, state.articulation, 'test_fixture', coverage_radius)
        obs = layer.apply(obs)
        odom = Odometry(state.pose, t, linear, angular, source='test_fixture',
                        position_uncertainty=position_uncertainty)
        result = controller.step(dt, odom, obs, now=t)
        clearance = _nearest_clearance(scenario, state.x, state.y, t)
        run.min_clearance = min(run.min_clearance, clearance)
        run.ticks.append(Tick(round(t, 3), state.x, state.y, state.yaw, state.cart_yaw,
                              result.linear, result.angular, result.reason, result.status,
                              len(dets), len(layer.tracks),
                              math.inf if clearance == math.inf else clearance))
        # Record phase transitions: a hold while a detection is present (the path
        # is blocked), then a resumption to tracked motion once the PATH clears --
        # the obstacle may still be visible off to the side when the robot resumes.
        if dets and result.linear <= 0.02 and (result.reason in _BLOCKED or result.reason in _PLANNING):
            seen_obstacle_stop = True
        if seen_obstacle_stop and result.linear > 0.02 and result.reason == 'tracking':
            cleared_after_stop = True
        if result.status == 'succeeded':
            run.arrived = True
            run.arrival_t = round(t, 3)
            break
        if result.status == 'failed':
            break
        linear, angular = result.linear, result.angular
        if linear > 1e-6:
            state = advance(state, linear * dt, angular / linear, spec)
        t = round(t + dt, 6)
    run.stopped_for_obstacle = seen_obstacle_stop
    run.resumed_after_clear = cleared_after_stop
    return run
