"""In-process sensor-to-command adapter; integration owns physics and actuators."""
from .sensors import observation
from dataclasses import replace


class SensorNavigation:
    """Compose wheel measurements + range scan + localization + safe controller.

    wheel_angles are continuous encoder radians, never commanded wheel targets.
    The initial map pose belongs to MapLocalizer's declared deployment prior.
    Exceptions stop motion; a stale last-good pose is never reused as fresh data.
    """
    def __init__(self,controller,wheel_odometry,localizer):
        self.controller=controller
        self.wheels=wheel_odometry
        self.localizer=localizer
        self.last_localized=None

    def start(self,goal,coupled,*,goal_articulation=None):
        # Forward the per-goal articulation tolerance to the inner controller so the
        # NavigationBridge start(goal, coupled, goal_articulation=) contract holds.
        if goal_articulation is None:
            self.controller.start(goal,coupled)
        else:
            self.controller.start(goal,coupled,goal_articulation=goal_articulation)

    def cancel(self):
        return self.controller.cancel()

    def step(self,dt,wheel_angles,scan,*,now,coupled,arm_transport,articulation=0.0,
             observed_space=None,temporal_map=None):
        if self.controller.terminal:
            return self.controller.terminal
        try:
            raw=self.wheels.update(wheel_angles[0],wheel_angles[1],now)
            localized=self.localizer.update(raw,scan)
            sensed=observation(scan,localized,coupled=coupled,
                               arm_transport=arm_transport,articulation=articulation)
            if observed_space is not None:
                sensed=replace(sensed,observed_space=observed_space)
            if temporal_map is not None:
                sensed=replace(sensed,temporal_map=temporal_map)
        except (ValueError,TypeError,IndexError,AttributeError):
            self.last_localized=None
            return self.controller.step(dt,None,None,now=now)
        self.last_localized=localized
        return self.controller.step(dt,localized,sensed,now=now)
