# Reactive arm layer — human-aware slow/stop + execution-time re-plan

`sim-arm-reactive-replan-v1`. Opt-in, **byte-identical when off**. Ported from the
real cell's Isaac-ROS stack (`~/ur10`: `humangate`, `dynamic_pick`). The nvblox map
is **never the stop** — the human gate owns it (scout `sim-perception-architecture-s1`
§4; captain decision: people are generic dynamic volumes for *routing*, the gate STOPs).

Shipped in two parts:

* **PR-A — human-aware RUN/SLOW/STOP/BACKSTOP authority** (this section). Live.
* **PR-B — execution-time corridor-watch preempt + cuMotion re-plan** against the
  perceived-clearance seams merged by `sim-rgbd-arm-collision-aware-v1`
  (`esdf_source` / `collision_gate` / `gated_motion`). (follow-up)

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
