# Behaviour tree — the map, how a run flows, and how to add a stage

This is the captain-facing map of the mission behaviour tree: a **diagram generated from the code**,
how a run flows through it, how each event is handled *today*, and a tested one-file recipe for
adding a stage. For the design rationale (why py_trees, byte-identical parity with the state machine,
the event contract, the live phone view) see [restock_bt.md](restock_bt.md); the modules live in
`scripts/twin/restock/bt/`.

## The generated diagram (SVG + HTML, aligned to a run)

`scripts/twin/restock/bt/diagram.py` **reconstructs the whole tree from the code** — from the
per-stage plans (`stage_phases.PLANS`) and the event contract (`events.py`) — never drawn by hand.
Every node carries its type (sequence / decorator / action / condition), its py_trees class, the
stage it belongs to, the `file:function` that implements it, and the events/guards it reacts to. A
unit test asserts the reconstruction matches the **real** serialised py_trees tree node-for-node
(`test_restock_bt_diagram.py::ParityWithRealTree`), so the diagram cannot drift from the tree.

Build it (CPU, no py_trees, no GPU) and align it to the banked `coldwin16` whole-run video:

```bash
python3 scripts/build_bt_diagram.py \
  --run   ~/datasets/twin-robot-in-store-boot-v1/mission/coldwin16 \
  --video bt.mp4 \
  --out   ~/firstmate/data/sim-bt-stage-map-v1 \
  --basename bt --title "Restock mission behaviour tree — aligned to coldwin16"
```

It writes a standalone **`bt.svg`** (static diagram) and a self-contained **`bt.html`** (the same tree
as an interactive DOM, the walkthrough video embedded, every stage/step node **clickable to seek the
video** to the moment it was active). The alignment replays the run's `bt_events.jsonl` through
`twin.demo.bt_timeline` and maps each node's sim-time window to a video timecode via the run's
`mission-frames/frames.json` (the whole-run mp4 is those frames at 20 fps, built by
`data/twin-robot-in-store-boot-v1/make_whole_run_video.py`). Nodes are coloured by their run status
(green success, amber running, red failure); reactive guards that never fired show as *armed · never
triggered* (honest — `coldwin16` raised no safety events).

Published on the phone watch page at **`http://<tailnet>:8787/bt`** (the `/bt`, `/bt.svg`, `/bt.mp4`
routes added to `data/isaac-watch/serve.py`; the page reads the generated files under
`~/firstmate/data/sim-bt-stage-map-v1/`). The route is additive — it takes effect on the watch
server's next restart (firstmate owns that process).

## How a run flows through the tree

The driver (`driver.run_mission_tree`) owns a tick clock and ticks the root **every tick** until the
mission is terminal. The root is a *non-memory* sequence, so all three children re-evaluate each tick
in this order:

```
restock_root  (Sequence, memory=False)        — re-evaluated every tick
  ├─ event_pump      drain the event bus → blackboard, then recompute the control flags
  ├─ reactive  (Sequence, memory=False)        — all guards evaluate every tick, before the mission
  │    ├─ health_guard    stuck / low_battery → brake to safe park + mission.fail  (FAILS → tears tree down)
  │    ├─ human_safety    slow-zone → slow;  stop-zone → safety hold + pause arm;  resume when clear
  │    ├─ obstacle_avoid  dynamic_obstacle / blocked_aisle → safety hold + request re-plan; resume when clear
  │    └─ baseline_guard  baseline_expired → hold + re-commission the station baseline via the provider
  └─ mission   (Sequence, memory=True)         — resumes at the running stage
       ├─ stage:collect  (StageRetry) → "collect steps" (Sequence, memory=True):
       │     drive→pick · perceive carton · grasp·lift·tuck · drive→rollwagen · perceive deck · place · verify placed
       ├─ stage:couple / outbound / park / transfer / verify / return   (one subtree each, start_at-aware)
       └─ finalize       mission.finalize once (stop adapter + record complete/failed)
```

1. **`event_pump`** drains the event bus (scenario `events` list and/or the ROS detection topic) into
   the blackboard and calls `MissionContext.recompute_control`, which derives the `pause` / `slow` /
   `replan` / `abort` flags **in one place** from the currently-active events. Always SUCCEEDs.
2. **`reactive`** runs its four guards *before* the mission every tick, so a safety condition is
   observed and the physical reaction fired **before** the active stage decides whether to step.
   `health_guard` is first and is the only guard that can FAIL (an abort short-circuits the root
   before the mission ticks). The other guards SUCCEED after reacting, leaving a control flag set.
3. **`mission`** is a memory sequence, so it resumes at the running stage. Each stage is a
   `StageRetry` decorator wrapping a memory sequence of **position-first `PhaseGate` leaves** (one per
   step in the stage's `PLANS` entry). One shared `StageRun` steps the real `Mission.step_stage`
   **once per tick**; the gates are faithful *views* of that one stepping gated on the adapter's
   observed phase (so the decomposition is byte-identical to the state machine — see
   `test_restock_bt_phases.py`). The active leaf reads `/control/pause` and **holds** (safety stop,
   payload preserved) instead of stepping while a reactive guard has paused the mission; it resumes
   when the flag clears. A step FAILURE is classified onto the timeline (naming the exact failed
   field), and `StageRetry` retries within a bounded budget with a recovery reaction
   (re-observe / re-plan / wait), re-entering the sequence from step 0; on exhaustion it calls
   `mission.fail` honestly.

A clean `coldwin16` run walks `mission` straight through: `collect` (7 steps, ~0–165 s sim) → `couple`
→ `outbound` → `park` → `transfer` (the long arm move, ~228–364 s) → `verify` → `return` (the
uncouple-at-parking plan: tow→cart park · brake cart · uncouple · back away · drive→home) → `finalize`
records `complete`. The reactive guards tick the whole time, all armed, none triggered.

## How each event is handled today

The event contract is `events.py` (`EVENT_TYPES`, the closed set). Two producers feed the same bus: a
scenario's deterministic `events` list (the path CPU tests and the scripted GPU proof use) and the
runtime detection topic `/restock_bt/events` (humans / dynamic-obstacle lanes), parsed by
`Event.from_json`. The live ROS subscriber is **opt-in** (`config['bt_ros_events']`, default off —
a second rclpy node in the Isaac process SIGABRTs the sim), so in a scripted run events arrive only
from the scenario list.

| Event | Who raises it | How the tree handles it |
|---|---|---|
| **blocked route** (`blocked_aisle`, `dynamic_obstacle`) | nav / dynobs lane, or a scenario event | `obstacle_avoid` sets a safety hold + requests a nav re-plan; the active stage holds (payload preserved) and resumes when cleared. A `blocked_aisle` that surfaces as a *step failure* instead is classified by `StageRetry` and retried with a `replan` recovery. |
| **coupling loss** (`coupling_failed`) | the `couple` stage's `step_stage` reason (`cart lost…`, `base moved during stationary operation`) | classified by `StageRetry`, retried with a `wait` recovery, re-entering the couple sequence from step 0; honest `mission.fail` on exhaustion. |
| human in the slow/stop zone (`human_detected` / `human_cleared`) | humans lane / scenario | `human_safety`: slow-zone → `slow`; stop-zone → safety hold + `pause_arm`; `resume` when cleared. |
| station baseline expiry (`baseline_expired`) | commissioning lane | `baseline_guard` holds and re-commissions via the stations provider; refusal keeps it held (never fabricated). |
| grasp/seal/goal/tracking miss (`grasp_failed`, `seal_failed`, `goal_miss`, `tracking_miss`) | a stage's `step_stage` reason | classified by `StageRetry` → `reobserve` (manip) or `replan` (nav) recovery, bounded retry, honest fail. |
| **stuck / low battery** (`stuck`, `low_battery`) | health lane / scenario | `health_guard` brakes to a safe park and `mission.fail`s, FAILS so the root tears the tree down. |

### Events the tree does **not** see today (handled by scripted code)

These are real run conditions that are handled **outside** the tree, by host/runtime/adapter code,
and surface to the tree only as a stage-step failure (if they fail the step) or not at all. They are
the follow-ups where a new reactive stage/handler would have to be wired:

* **Envelope trip** — `ArmCommandGate` (arm command envelope) and `trajectory_host` tracking
  e-stops are host-side; they latch/abort the arm and the mission reads a `('failed', …)` step
  reason at most. No `envelope_trip` event exists.
* **VRAM shed** — `vram_headroom.py` sheds optional load (capture rate → arm-feed economy →
  secondary cameras) on the runtime, logging `VRAM_SHED` by name. The tree never sees it.
* **move_group SIGABRT crash + respawn / scene re-sync** — recovered entirely inside
  `ros_adapter.RosMotionPort` (`MOVE_GROUP_CRASH_RETRY`); invisible to the tree.
* **Navigation stall recoveries** — coupled break-away, pre-throat heading hold, momentum launch,
  arrival-hysteresis, tow-lane re-centre — all inside the nav/adapter layer; `obstacle_avoid` only
  handles the `blocked_aisle` / `dynamic_obstacle` *events*, not these.
* **Host station-liveness e-stops** — `incorporation_ack` / `station_host` / `trajectory_host`
  publish the global `/manipulation/stop`; the mission surfaces a failure, not a tree event.
* **Start-phase shortcut** — `start_at` synthesizes the skipped stage prefix (`shortcut.py`); those
  stages are logged `*_shortcut` and shown *idle / synthesized* in the diagram, never driven by the
  tree.
* **Scripted transfer modes** — `RESTOCK_TRANSFER_MODE=live_pose|scripted` replace the transfer
  manipulation with scripted motion; the tree still drives `step_stage`, but the behaviour is scripted.

## Recipe: add a stage (one file, tested)

A stage is **data-driven**: the tree (`tree.build_stage`) and the diagram (`diagram.build_model`)
both build a stage's subtree from its ordered `(label, kind)` plan, and the timeline/diagram read
that same plan back. So adding a stage is three small local edits — no new leaf class, no tree
surgery. The worked template is `scripts/twin/restock/bt/stage_template.py` (a filled-in example
`inspect` stage), exercised by `scripts/twin/tests/test_restock_bt_diagram.py::AddStageRecipe` so it
cannot rot.

1. **Declare the plan** — add a `'<stage>': [(label, kind), …]` entry to `stage_phases.PLANS` (one
   tuple = one position-first leaf, navigate → perceive → manipulate). *This alone makes the stage and
   its steps appear in the diagram and the live view.*
2. **Classify its progress** (only if >1 step) — add a `_<stage>_milestone(adapter)` mapping the
   adapter's observed `phase` to the current step index, and register it in
   `stage_phases._CLASSIFIERS`. A single-step stage needs none.
3. **Drive it** — add the stage to `mission.Mission`'s stage tuple and implement its
   `begin_stage` / `step_stage`. `tree.active_stages` reads `mission.stages`, so the tree picks it up
   automatically.

The template shows the pattern without touching the real `PLANS` (its `register_plan` returns a new
dict). The test proves a dummy stage then appears **in the diagram model** (`build_model(plans=…)`)
and **in a replayed timeline** (`synthetic_bt_events` → `align_to_run` colours it with a window + a
video timecode). Run it:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH= python3 -m pytest \
  scripts/twin/tests/test_restock_bt_diagram.py -q            # pure tier, no py_trees
# with the task-owned py_trees for the parity tier:
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=~/datasets/sim-restock-bt-v1/pylib:scripts python3 -m pytest \
  scripts/twin/tests/test_restock_bt_diagram.py -q
```
