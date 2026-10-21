"""In-process sensor-to-command adapter; integration owns physics and actuators."""
from .sensors import observation
from dataclasses import replace


class SensorNavigation:
    """Compose wheel measurements + range scan + localization + safe controller.

    wheel_angles are continuous encoder radians, never commanded wheel targets.
    The initial map pose belongs to MapLocalizer's declared deployment prior.
    Exceptions stop motion; a stale last-good pose is never reused as fresh data.
    """
    def __init__(self,controller,wheel_odometry,localizer,*,dynamic_layer=None):
        self.controller=controller
        self.wheels=wheel_odometry
        self.localizer=localizer
        # Optional moving-obstacle layer (DynamicObstacleLayer). When attached AND
        # the caller passes detections, its predicted swept cells are unioned into
        # the sensed observation before gating. Absent -> the observation is
        # untouched and the mission run is byte-identical.
        self.dynamic_layer=dynamic_layer
        self.last_localized=None

    def start(self,goal,coupled,*,goal_articulation=None,static_drive=False,carry_drive=False):
        # Forward the per-goal articulation tolerance + the reposition-leg drive flags to the inner
        # controller so the NavigationBridge start(..., static_drive=, carry_drive=) contract holds.
        kw={}
        if goal_articulation is not None:
            kw['goal_articulation']=goal_articulation
        if static_drive:
            kw['static_drive']=True
        if carry_drive:
            kw['carry_drive']=True
        self.controller.start(goal,coupled,**kw)

    def cancel(self):
        return self.controller.cancel()

    def step(self,dt,wheel_angles,scan,*,now,coupled,arm_transport,articulation=0.0,
             observed_space=None,temporal_map=None,detections=None):
        if self.controller.terminal:
            return self.controller.terminal
        try:
            raw=self.wheels.update(wheel_angles[0],wheel_angles[1],now)
            localized=self.localizer.update(raw,scan)
            sensed=observation(scan,localized,coupled=coupled,
                               arm_transport=arm_transport,articulation=articulation)
            # Fold tracked moving obstacles into the sensed returns (add-only: the
            # layer never clears a raw return). Skipped entirely when no layer is
            # attached or no detections are supplied, keeping the run byte-identical.
            if self.dynamic_layer is not None and detections is not None:
                self.dynamic_layer.update(detections,now)
                sensed=self.dynamic_layer.apply(sensed)
            if observed_space is not None:
                sensed=replace(sensed,observed_space=observed_space)
            if temporal_map is not None:
                sensed=replace(sensed,temporal_map=temporal_map)
        except (ValueError,TypeError,IndexError,AttributeError):
            self.last_localized=None
            return self.controller.step(dt,None,None,now=now)
        self.last_localized=localized
        return self.controller.step(dt,localized,sensed,now=now)
