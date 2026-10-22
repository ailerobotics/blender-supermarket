# Behaviour-tree mission layer (the mission driver)

The mission **run driver** for the Isaac supermarket restock mission: it drives the *same*
`mission.Mission` state machine through a [py_trees](https://py-trees.readthedocs.io) behaviour
tree, as a **subtree of explicit position-first leaves per stage** (navigate → perceive →
manipulate, step by step), plus **reactive event handling** and a **live, phone-friendly
visualization** of the tree and of how each event is handled.

Selected with `"bt": true` in the runtime-config (the integration lane's mission-runtime configs set
this); `"bt": false` keeps the sequential `Mission.tick` state machine as the unchanged fallback. The
tree reproduces the state machine stage-for-stage and **byte-identically** (same `begin_stage` /
`step_stage` / `finalize` calls, same inventory, evidence and durable record), proved on CPU by
`test_parity_with_state_machine` (single-leaf fixtures) and `test_restock_bt_phases.py` (the phased,
decomposed drive, step-for-step).

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
       |- stage:collect  (StageRetry) -> "collect steps" (Sequence, memory=True):
       |      drive->pick · perceive carton · grasp·lift·tuck · drive->rollwagen ·
       |      perceive deck · place · verify placed      (explicit position-first leaves)
       |- stage:couple / outbound / park / transfer / verify / return  (start_at-aware subset)
       `- finalize        mission.finalize once (stop adapter + record complete/failed)
```

Each stage is a **subtree of explicit position-first leaves** — one `PhaseGate` per step in the
stage's plan (`bt/stage_phases.py` `PLANS`), in the navigate → perceive → manipulate order the
runtime already follows (for `collect`: drive to the pick pose, perceive the carton, grasp/lift/tuck,
drive to `place_approach` in front of the rollwagen, perceive the deck, place, verify). **A new step
is a new entry in `PLANS` (a new leaf).** The whole seven-stage cycle runs this way by default.

`event_pump` and `reactive` run **before** `mission` every tick, so a safety event is observed and
the control flags/physical reactions are applied before the active leaf decides whether to step. The
active leaf reads the `/control/pause` flag and **holds** (safety stop, payload preserved) instead of
stepping — so "the reactive subtree handles the event while the mission continues, then resumes" is
literally what the tree does. A step failure is classified onto the timeline (naming the exact failed
field — the fm412 diagnostic) and retried within a bounded budget by the `StageRetry` node with a
recovery reaction (re-observe / re-plan / wait), re-entering the phase sequence from step 0; on
exhaustion it calls `mission.fail` honestly. `health_guard` is the only reactive leaf that FAILS, so
an abort short-circuits before the mission ticks.

### No duplicated stage logic / byte-identical parity

The leaves never issue a navigation/manipulation command themselves: one shared `StageRun` steps the
real `step_stage()` seam **at most once per tree tick**, and each leaf is a faithful *view* of that
one stepping, RUNNING while the adapter's observed phase is on its step and SUCCESS once it has
advanced past it (`stage_phases.milestone`). Because the `begin_stage` / `step_stage` / `finalize`
calls are identical in count and order to `Mission.tick`, evidence validation, inventory recording
and the verify stock-commit gate — all in the mission, one place — produce a byte-identical durable
record and stock. An adapter that exposes no `phase` (a plain mock/fixture) collapses each stage to a
single leaf (the former opaque behaviour), so the offline/mock harnesses are unchanged.

### Uncouple at the cart parking spot (return)

By default `return` tows the cart home coupled. With `"uncouple_on_return": true` **and** a
`cart_parking` mission pose (owned by the `sim-restock-nav-lanes-v1` lane; build against a clearly
named placeholder until it lands), the `return` subtree becomes five explicit leaves:
tow→cart park (coupled) · brake cart · uncouple · back away · drive→home (empty). The runtime
(`RuntimeAdapter`) tows to the parking spot, pins the parked cart (`mobile.hold_cart_pin`), releases
the tow coupling via the new `MobileController.uncouple(obs)` primitive (mirror of `attach`: pin the
cart, then remove the hitch; the `_cart_hold` latch keeps the cart braked as the robot drives away),
optionally backs away (`cart_parking_backaway`), then drives home with no coupled requirement. CPU
proof: `test_restock_runtime.py` (choreography + nav-leg coupled flags) and `test_mobile_assets.py`
(`test_uncouple_leaves_cart_parked_and_frees_the_robot`). Default off ⇒ byte-identical tow-home.

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

### ROS 2 domain hygiene

The CPU tests use **no ROS** (the event bus is driven by `QueueEventSource` / `ScenarioEventSource`,
not rclpy), so they need no domain setup. For ROS specifically:

- **A standalone `bt` ROS node/test** (one run *in isolation*, not inside the mission): export
  `ROS_DOMAIN_ID=106` (this lane's fleet assignment) and `ROS_LOCALHOST_ONLY=1`, stop every node you
  start, and afterwards remove only your **own** stale `/dev/shm/fastrtps_*` segments (never a live
  one, never another lane's) — FastDDS shared-memory port exhaustion from stale segments has killed
  runs fleet-wide.
- **The in-mission subscriber** (`integration.start_event_subscriber`, active only in a live run)
  deliberately does **not** set its own domain: it inherits the mission orchestrator's domain
  (**88**) so it hears the humans / dynamic-obstacle lanes' detections on `/restock_bt/events`.
  Those lanes must therefore publish on the mission domain (88) for the integrated run, not on their
  isolated lane-test domains.

The GPU is a single RTX 3060 under `/tmp/aile-gpu.lock`; the integration lane now holds the lock
across its whole commission-plus-mission sequence, so take the lock only when it is free and keep
sessions short (this lane is lowest priority, ≤15 min).

**ROS event subscriber is opt-in** (`"bt_ros_events": true`, default off): it creates a second rclpy
node inside the Isaac process, which already runs its own ROS2 bridge — doing so unconditionally
SIGABRTs the simulator (observed in the sim-restock-bt-v1 GPU proof). A scripted-scenario run needs
no ROS (events come from the `ScenarioEventSource`); the humans/dynobs lanes enable it only when they
publish live detections on `/restock_bt/events`.

**Opt-in GPU/live run:** add `"bt": true` to the runtime-config passed to
`scripts/run_mobile_restock.py` (compose with `start_at` and a scenario `events` list for a scripted
proof). The run drives the same mission through the tree and writes the live view into the run
output directory. py_trees must be importable by the sim interpreter — add the task-owned venv's
`site-packages` to `PYTHONPATH` (or `pip install --target` into a task dir) for that process; never
mutate `env_isaaclab`. GPU is lowest priority on `/tmp/aile-gpu.lock`, CPU-first; the VRAM guard
stays unchanged.
