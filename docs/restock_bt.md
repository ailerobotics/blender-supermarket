# Behaviour-tree mission layer (opt-in)

An **alternative orchestrator** for the Isaac supermarket restock mission that drives the *same*
`mission.Mission` state machine through a [py_trees](https://py-trees.readthedocs.io) behaviour
tree, adding **reactive event handling** and a **live, phone-friendly visualization** of the tree
and of how each event is handled. It is **default-off**: nothing here runs unless a caller opts in
(`"bt": true` in the runtime-config). The state machine (`Mission.tick`) stays the production path;
the tree reproduces it stage-for-stage and is byte-identical on the no-event path.

Modules live in `scripts/twin/restock/bt/`. Design, the full event contract, view paths, commands
and the proof record are in `data/sim-restock-bt-v1/report.md`.

## Why py_trees (not BehaviorTree.CPP/Groot2)

The codebase is Python and ROS 2 native. py_trees' core has **no ROS dependency**, so the identical
tree ticks in CPU tests *and* inside the Isaac process; `py_trees_ros` adds a live snapshot stream +
desktop viewers when ROS is present. nav2 keeps its internal BehaviorTree.CPP unchanged (it executes
the navigation trees); this layer sits *above* the mission, not inside navigation. BT.CPP+Groot2
would have forced a C++/XML tree and an IPC bridge for a Python mission with no offsetting benefit.

py_trees is installed into a **task-owned** venv only (never the shared `env_isaaclab`): see Commands.

## Tree shape

```
restock_root  (Sequence, memory=False)        -- re-evaluated every tick
  |- event_pump                               -- drain event bus -> blackboard, recompute control
  |- reactive  (Sequence, memory=False)
  |    |- health_guard      stuck / low_battery -> safe park (FAILS, tears the tree down)
  |    |- human_safety      slow -> stop + pause arm -> resume when clear
  |    |- obstacle_avoid    dynamic obstacle / blocked aisle -> wait + re-plan -> resume
  |    `- baseline_guard    baseline_expired -> re-commission via the stations lane's provider
  `- mission   (Sequence, memory=True)        -- resumes at the running stage
       |- stage:collect   bounded retry + re-observe recovery
       |- stage:couple / outbound / park / transfer / verify / return  (start_at-aware subset)
       `- finalize        mission.finalize once (stop adapter + record complete/failed)
```

`event_pump` and `reactive` run **before** `mission` every tick, so a safety event is observed and
the control flags/physical reactions are applied before the active stage decides whether to step.
The `Stage` leaf reads the `/control/pause` flag and **holds** (safety stop, payload preserved)
instead of stepping — so "the reactive subtree handles the event while the mission continues, then
resumes" is literally what the tree does. A `Stage` failure is classified onto the timeline and
retried within a bounded budget with a recovery reaction (re-observe / re-plan / wait); on
exhaustion it calls `mission.fail` honestly. `health_guard` is the only reactive leaf that FAILS, so
an abort short-circuits before the mission ticks.

### No duplicated stage logic

Stages call the mission seams `begin_stage()` / `step_stage()` / `finalize()` (added to
`mission.py`, behaviour-preserving: `Mission.tick` now wraps the same seams). Evidence validation,
inventory recording and the verify stock-commit gate live in exactly one place — the mission — so
inventory and the durable record are byte-identical to the state machine (proved by
`test_parity_with_state_machine`).

## Event contract

Canonical module: `scripts/twin/restock/bt/events.py` (pure Python; no py_trees/ROS, so
`scenario.py` validates scenario events without pulling py_trees). Full table: report §2.

- **ROS topic** `/restock_bt/events`, `std_msgs/msg/String`, JSON `{stamp_ns, type, data}`. The
  humans and dynamic-obstacle lanes publish detections here; `detection_to_event(Detection)`
  converts a `navigation.dynamic.Detection(position, radius, source)` into a `dynamic_obstacle`
  event so the dynobs lane need not restructure PR #114.
- **Blackboard** namespaces `/event/*` (active, log, last) and `/control/*` (pause, slow, replan,
  abort) + `/clock/now`. Reactive subtrees read the blackboard only (ROS-free, CPU-testable).
- **Scenario events** — an optional top-level `"events"` list in the mission scenario JSON, each
  `{"t": <sim seconds>, "type": <event type>, "data": {…}}`, validated by `scenario.validate`
  (additive, backward compatible). A `ScenarioEventSource` injects each once when the tick clock
  reaches `t`. This lets a scenario *declare* what happens so the tree *shows* how it is handled —
  the deterministic path used by CPU tests and the scripted GPU proof.

Closed event set: `human_detected`/`human_cleared`, `dynamic_obstacle`/`obstacle_cleared`,
`blocked_aisle`, `grasp_failed`/`seal_failed`, `tracking_miss`/`goal_miss`, `baseline_expired`,
`coupling_failed`, `stuck`, `low_battery`. Unknown types on the wire are logged and dropped.

## Live visualization

`SnapshotWriter` (`view.py`) writes into the run directory, every tick:

- `bt_view.html` — self-contained page (no external assets), polls `bt_snapshot.json` and renders
  the tree with node states coloured, the control flags, active events and the recent timeline,
  auto-refreshing (~0.7 s). This is the **phone view**; firstmate links it from the watch page
  (this lane does not edit the watch page).
- `bt_snapshot.json` — the current tree (nested nodes with name/status/feedback) + blackboard +
  mission state; overwritten atomically each tick.
- `bt_events.jsonl` — append-only event/action timeline (durable record).

Sample artifacts from a scripted human+obstacle run: `data/sim-restock-bt-v1/view-demo/`.

**Desktop viewer (optional, ROS):** with `py_trees_ros` installed and a `py_trees_ros.trees` wrapper
publishing the snapshot stream, run `py-trees-ros-viewer` or `rqt_py_trees`. The HTML view above is
the primary deliverable and needs neither ROS nor a viewer.

## Commands

```bash
# Task-owned CPU venv (gitignored; never the shared env_isaaclab):
python3 -m venv .venv-bt && .venv-bt/bin/pip install py_trees pytest

# CPU tests (tree ticks + event injection + snapshot + scenario events + state-machine parity):
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH= .venv-bt/bin/python -m pytest \
    scripts/twin/tests/test_restock_bt.py -q

# The pure event-contract tests also run under the repo interpreter (tree tests auto-skip):
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH= python3 -m pytest \
    scripts/twin/tests/test_restock_bt.py -q
```

**Opt-in GPU/live run:** add `"bt": true` to the runtime-config passed to
`scripts/run_mobile_restock.py` (compose with `start_at` and a scenario `events` list for a scripted
proof). The run drives the same mission through the tree and writes the live view into the run
output directory. py_trees must be importable by the sim interpreter — add the task-owned venv's
`site-packages` to `PYTHONPATH` (or `pip install --target` into a task dir) for that process; never
mutate `env_isaaclab`. GPU is lowest priority on `/tmp/aile-gpu.lock`, CPU-first; the VRAM guard
stays unchanged.
