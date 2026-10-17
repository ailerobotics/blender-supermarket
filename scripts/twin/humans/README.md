# Simulated people and human safety

This package is the opt-in simulated-people layer for mobile restocking.  With no
`humans` object in the runtime config, it imports nothing into the mission path
and changes neither the scene nor control commands.

## Configuration

Set `humans.enabled: true` in the runtime config.  Omit `people` to derive the
deterministic shopper/staff paths from the mission pickup, dock, shelf and home
poses.  The paths cross the outbound route and visit both the backstock dock and
shelf station.  `backend: teleport` is the supported deterministic Isaac backend;
`anim_people` requests `omni.anim.people` and reports a fallback when it is not
installed.

`detection.source` is mandatory when the safety layer is enabled:

* `detector` consumes the fleet detector's `/human_safety/state` output.  Use
  `params/human_safety.restock.yaml` to run the detector against the native
  restock camera topics.  It is deliberately fail-safe (a missing/stale state is
  STOP); it never substitutes ground truth.
* `ground_truth` is an explicit, labelled fallback that projects scripted actor
  poses into configured camera cones.  It drives the same vendored fleet safety
  evaluator, but is not a detector measurement.

All reaction values live in `humans.reaction`: base slow/stop scale, optional
dynamic-obstacle inflation, arm reach envelope, state timeout and clear-resume
dwell.  They are provisional configuration, not certified separation distances.

## CPU evidence

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=scripts \
  python3 -m pytest -q scripts/twin/humans/tests
python3 scripts/run_humans_safety_demo.py --out /tmp/sim-restock-humans-demo
```

The demo writes a labelled ground-truth trace, summary and top-down SVG.  It is
not pixel/detector evidence.  Run `test_isaac_people_usd.py` in the CPU-only
`usd-core` environment to verify USD actor authoring without booting Isaac.

## Integration seams

The scene owner calls `integration.author_people(stage, config, poses)` once and
`integration.step_people(people, simulation_time)` every physics step.  The
mission/control owner builds `integration.build_safety(...)`, applies
`ReactionOutput.base_scale` to navigation velocity, gates arm commands on
`ReactionOutput.arm_hold`, and supplies `MissionSafety.detections(now)` to the
dynamic-obstacle layer.  These calls remain absent when `humans.enabled` is false.

The detector bridge's `run_node()` is a separate ROS process by design; it uses
the real rig JSON messages and publishes reaction and obstacle events without
introducing a new custom ROS message package.
