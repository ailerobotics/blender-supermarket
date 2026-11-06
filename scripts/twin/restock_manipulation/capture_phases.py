"""Single-source definition of the capture-silent controller phases (pure, zero-dependency leaf).

A controller phase is *capture-silent* when the decode-camera render is unused by design -- observe
has already validated the registration and the arm is executing the reach / holding the carton, so the
fixed source/station camera is occluded and emits nothing useful. The mission's sensing gate suspends
the decode render for these phases, and -- because no fresh capture is produced while they run -- the
phase-scoped producer watchdogs (the per-frame ``/sensors`` SIM freshness bound and the parked-station
heartbeat SIM-gap bound) are legitimately quiet across them.

This leaf is the ONE authoritative home of that phase set so every consumer agrees:

  * ``isaac_sensors`` re-exports ``_SENSING_SUSPEND_PHASES``/``sensing_render_required`` from here (the
    sensing-render gate and ``controller.in_capture_silent_window``);
  * ``restock_host.incorporation_ack`` (the station-liveness node, a SEPARATE process) imports
    ``sensing_render_required`` directly to drive the capture-silent parked-heartbeat suspension
    (sim-retreat-station-heartbeat-scope-v1) -- so the station node and the sensor watchdog scope the
    SAME window with no drift and no heavy (numpy / restock_camera) import.

The phases (why each is capture-silent):
  * ``approach``/``contact``/``grasp``/``lift`` -- the pick reach/press/lift occludes the fixed camera;
  * ``preplace``/``place``/``retreat``/``release``/``stow`` -- the cart/shelf holding chain and the empty
    post-release retreat/stow; the placement target is DERIVED (not camera-sensed), so the held
    re-validation fabricates nothing and the fixed camera resumes only at the next capture-bearing phase;
  * ``couple`` -- the cart-coupling action's execution phase;
  * ``tuck``/``pan``/``retreat_tuck`` -- the staged cart-swing via points of the held carry chain.

A capture-BEARING phase (``observe``, the ``verify`` re-validation, and any other/unknown phase) is
NOT in the set -- fail safe: never suspend unless the phase is explicitly a validated-reach execution
phase.
"""

#: The authoritative capture-silent phase set (see module docstring).
CAPTURE_SILENT_PHASES = frozenset({'approach', 'contact', 'grasp', 'lift', 'preplace', 'place',
                                   'retreat', 'release', 'stow', 'couple',
                                   'tuck', 'pan', 'retreat_tuck'})


def sensing_render_required(phase):
    """Whether the decode camera must render for this controller phase.

    False only for the execution phases in :data:`CAPTURE_SILENT_PHASES` (observe already validated the
    registration and the rendered frame is unused while the arm occludes the fixed camera). True for
    observe and the verify re-validation, and for any other/unknown phase -- fail safe: never suspend
    unless the phase is explicitly a validated-reach execution phase. The integration runtime maps the
    controller phase through this each tick to drive ``set_render_suspended(not sensing_render_required
    (phase))``; the station-liveness node maps it to the capture-silent parked-heartbeat suspension."""
    return str(phase) not in CAPTURE_SILENT_PHASES
