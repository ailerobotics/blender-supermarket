# Reactive arm layer — human-aware slow/stop + execution-time re-plan

`sim-arm-reactive-replan-v1`. Opt-in, **byte-identical when off**. Ported from the
real cell's Isaac-ROS stack (`~/ur10`: `humangate`, `dynamic_pick`). The nvblox map
is **never the stop** — the human gate owns it (scout `sim-perception-architecture-s1`
§4; captain decision: people are generic dynamic volumes for *routing*, the gate STOPs).

Shipped in two parts:

* **PR-A — human-aware RUN/SLOW/STOP/BACKSTOP authority** (next section). Live.
* **PR-B — execution-time corridor-watch preempt + cuMotion re-plan** against the
  perceived-clearance seams merged by `sim-rgbd-arm-collision-aware-v1`
  (`esdf_source` / `collision_gate` / `gated_motion`). Live (last section).

## PR-A: human-aware slow/stop

The ported humangate decision core (`arm_human_gate.ArmHumanGate`) is fed by the
**already vendored** `scripts/twin/humans` evaluator (an injected detection provider
returning a `SafetyState` with the CLEAR/SLOW/STOP zone + nearest distance) and drives
the free-space arm legs. It is the **single motion authority for people**.

Modules (all pure — stdlib only, all time passed in, unit-tested offline):

| File | Role |
|---|---|
| `arm_human_gate.py` | ported `HumanGate` + `GateConfig` + `CameraState`: RUN/SLOW/STOP/BACKSTOP, fail-toward-stop, self-observed resume dwell, lost-close resume refusal, STOP→BACKSTOP fall-through. `frame_from_safety_state()` adapts the vendored `SafetyState`. |
| `arm_backstop.py` | ported independent `BackstopWatcher` (its own heartbeats; latches). |
| `reactive_arm.py` | `ReactiveArmSupervisor`: the single controller seam. Translates the gate decision into a per-tick directive and realises it through the **e-stop-free** `motion.cancel()`/`cancel_outcome()` preempt + the reversible `ArmCommandGate.set_paused`. |

Decision → action mapping (the honest C1 segmented mapping, `human-dynamic-plan-s1` §2):

| Gate | Arm |
|---|---|
| RUN | run / issue the leg at full speed |
| SLOW | preempt a running re-plannable leg, re-issue it **slow**; a not-yet-issued leg is issued slow |
| STOP | preempt (e-stop-free cancel) a running leg, **pause** the arm command gate, **block** new legs; resume (re-plan from the live pose) only after the gate's **clear dwell** restores RUN |
| BACKSTOP | same terminal hold, additionally latched by the independent watcher until RUN is re-earned |

**Clean-start semantics** (`ArmGateConfig.require_initial_dwell=False`, default): a
mission that has never been restricted runs at full speed from tick one; the resume
dwell gates only RELEASE from a restriction, never the first RUN. Set `True` for the
real-cell behaviour (every RUN earned through a dwell).

**Scope:** the human layer acts only on the free-space re-plannable arm phases
(`approach/lift/retreat` + staged `tuck/pan/retreat_tuck/stow`). The delicate/frozen
phases (`contact/grasp/preplace/place/release`) and the pre-place planner (owned by
another lane) are untouched.

### Enabling it

HOST config (`ros_host_config.json`), read by `ros_host.py` → `TransferController`:

```json
{ "reactive_arm_human": true,
  "reactive_arm": { "clear_dwell_s": 2.0, "slow_speed_fraction": 0.1,
                    "backstop": { "startup_grace_s": 3.0 } } }
```

Absent / `false` ⇒ the supervisor is never constructed, `step()` skips the seam ⇒
byte-identical. Example: `docs/examples/arm-reactive-replan-fragment.json`.

The Isaac side wires the two injections at boot (behind `humans.enabled`):
`controller.set_human_detection_provider(provider)` (the vendored
`humans.perception` evaluator, ground-truth or detector-fed) and
`controller.set_arm_pause_sink(gate.set_paused)` (the reversible
`ArmCommandGate.set_paused`). With no sink wired, the cancel-hold is the sole stop
(still safe).

### CPU proof

`scripts/twin/tests/test_reactive_arm_human.py` (no ROS/Isaac/GPU): the ported gate
scenarios (clean-start / approach→SLOW→STOP / dwell-resume / lost-close / heartbeat /
stale / backstop), the supervisor driven by a scripted person distance (preempt +
pause + resume), the **real `TransferController.step` seam** (blocks + pauses on a
person, resumes on a clear dwell, byte-identical when off), and `set_paused`.

```
env -u PYTHONPATH PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest \
  scripts/twin/tests/test_reactive_arm_human.py -q
```

## PR-B: execution-time corridor-watch re-plan

The sibling `sim-rgbd-arm-collision-aware-v1` gate gates a leg at **dispatch** (before
`begin`). PR-B is the execution-time complement its `README_perception.md` names: while a
free-space re-plannable leg **executes**, re-query the live nvblox ESDF along the
gripper(+carton) swept corridor of the **remaining** leg; when a NEW obstacle enters it
below a margin, PREEMPT the running goal (the same e-stop-free `motion.cancel()` path),
re-plan the remaining leg with cuMotion against the current ESDF, and resume. Bounded
re-plan count with a self-recovering HOLD when it cannot route clear; rate-limited.
People re-route the plan (generic dynamic volumes); the human gate (PR-A) owns the STOP.

| File | Role |
|---|---|
| `reactive_replan.py` | `ReactiveReplanController` — the pure decide core (ported from `~/ur10/dynamic_pick`): continue / preempt / bounded / self-recovering HOLD, rate-limited. |
| `reactive_arm.py` | `ReactiveArmSupervisor._corridor_watch` — reuses the merged `collision_gate.PerceptionCollisionGate.evaluate_corridor` + `esdf_source` (so an execution intrusion uses the SAME margin logic as the dispatch gate); only a FINITE observed clearance below the margin counts (an unobserved/stale corridor is the dispatch gate's concern). `build_corridor_fn(gated_port)` builds the remaining-corridor FK from the sibling `GatedMotionPort`. |

Enable (HOST config; requires the sibling `perception_collision_gate` on for the ESDF source):

```json
{ "perception_collision_gate": { "margin_m": 0.05 },
  "reactive_arm_replan": true,
  "reactive_arm": { "replan": { "margin_m": 0.05, "max_replans": 3,
                                "min_replan_interval_s": 0.5 } } }
```

Absent / `false` ⇒ the corridor watch is inert ⇒ byte-identical. The mission wires the ESDF
source + corridor builder at boot (`ros_host` → `reactive_arm.build_corridor_fn(motion)` →
`controller.set_arm_clearance_source(...)`). With `perception_collision_gate` off the watch
has no live ESDF source and stays inert (logged).

### CPU proof

`scripts/twin/tests/test_reactive_arm_replan.py` (no ROS/Isaac/GPU): the decide core
(continue/preempt/bounded/HOLD/reset); the supervisor corridor watch against a **mock nvblox
ESDF** (`AnalyticBoxEsdfSource` with a moving obstacle) + the **real**
`PerceptionCollisionGate` — a planted obstacle entering the remaining corridor preempts +
re-plans, the arm continues once it clears, a persistent obstacle exhausts the bounded budget
into a self-recovering HOLD; the real `TransferController.step` seam; and a moving-obstacle
timeline replay showing the preempt → re-plan → continue timeline.

```
env -u PYTHONPATH PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest \
  scripts/twin/tests/test_reactive_arm_replan.py -q
```

## Bounded GPU S4 proof (PENDING — integration-lane-owned)

Bounded GPU S4 (person + moving obstacle, preempt + re-plan + human SLOW→STOP→resume). The full
HOST-key fragment + acceptance text lives in `docs/examples/arm-reactive-replan-fragment.json`;
the self-contained recipe is here so this file is the single hand-off.

**HOST config** (merge into `restock_manipulation/ros_host_config.json`, read by `ros_host.py`;
`config_preflight` registers them HOST):

```json
"reactive_arm_human": true,
"reactive_arm_replan": true,
"perception_collision_gate": { "margin_m": 0.05 },
"reactive_arm": {
  "clear_dwell_s": 2.0, "slow_speed_fraction": 0.1, "stop_speed_fraction": 0.0,
  "state_max_age_s": 0.2, "heartbeat_timeout_s": 0.3, "backstop_deadline_s": 0.5,
  "require_initial_dwell": false,
  "backstop": { "startup_grace_s": 3.0, "state_heartbeat_timeout_s": 0.5, "gate_heartbeat_timeout_s": 0.5 },
  "replan": { "margin_m": 0.05, "max_replans": 3, "min_replan_interval_s": 0.5, "samples": 6, "max_snapshot_age_s": 0.5 }
}
```

`reactive_arm_replan` REQUIRES `perception_collision_gate` on (it supplies the live nvblox ESDF
source + leg corridor via the sibling `GatedMotionPort`); with it off the corridor watch stays
inert (`ros_host` logs `REACTIVE_REPLAN enabled but perception_collision_gate is OFF`). Also set
`humans.enabled` with a scripted person crossing the arm's reach (`detection.source: ground_truth`
for the bench) and the Isaac boot wiring (`set_human_detection_provider` / `set_arm_pause_sink` /
`set_arm_clearance_source`, all named in the fragment `_isaac_wiring`).

**Run** (GPU discipline: take `/tmp/aile-gpu.lock` only when the integration lane releases it,
`HEAVY_MB=2500`, `renice +10`, ≤20 min, never `--force`, never touch `whispr`, do NOT edit the
`~/datasets` runtime configs from a worker PR):

```
scripts/run_mobile_restock.py ...   # a collect carry leg, with the HOST keys above + humans.enabled
```

**Acceptance** — in ONE bounded run:
1. **PR-B obstacle:** a person / sliding carton crosses the arm's REMAINING swept corridor on a
   free-space carry leg → console shows `REACTIVE_ARM_PREEMPT action=REPLAN`, the arm cancels
   (e-stop-free `motion.cancel()`, no `/manipulation/stop`), re-plans the remaining leg against the
   current ESDF (routes around it) and COMPLETES without contact; bounded by `max_replans` then a
   self-recovering HOLD.
2. **PR-A person:** a person entering reach drives the gate CLEAR→SLOW→STOP → `REACTIVE_ARM_HOLD
   action=SLOW` then `STOP`, the arm PAUSES (`set_paused True`), and after the person leaves THROUGH
   the SLOW ring (tracked departure) + the clear dwell the arm RESUMES (`set_paused False`, re-plan
   from the live pose).

No global `/manipulation/stop` (no host e-stop) in either path. Record VRAM peak + re-plan latency.
The box stays parallel to the ground (2.0° / 0.15 s, never loosened) throughout — the carton-level
gate is unchanged and runs before this seam each tick.

**CPU proof stands in until the S4 run:**

```
env -u PYTHONPATH PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest \
  scripts/twin/tests/test_reactive_arm_human.py \
  scripts/twin/tests/test_reactive_arm_replan.py -q
```
