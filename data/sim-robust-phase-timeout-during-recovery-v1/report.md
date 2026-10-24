# sim-robust-phase-timeout-during-recovery-v1

Stop a **named `move_group` crash recovery from being charged against the per-phase timeout**, so a
recovered crash cannot convert into `phase timeout: <phase>`. PR #247 made the launch respawn bounded and
observable; PR #249 reconciles the surviving carton attach; PR #234/#242 fail the stage fast by name when
recovery does not complete. What was missing: nothing pauses or extends the controller's per-phase clock
while a crash recovery is in progress, so two recoveries inside one phase burned the whole budget.

- **Run that motivated it:** `~/datasets/twin-robot-in-store-boot-v1/mission/respawn2b` (head `bd06819`,
  PR #249; collect-only, fresh commission, HOST `movegroup_respawn={max_retries:2, delay:3.0}`, the v3
  injector killing `move_group` with SIGABRT twice inside the collect lift). `mission.json`
  `state=failed`, attempt-1 `reason='host emergency stop (host fault: phase timeout: lift)'`.
- **Evidence:** that run's `host-stack.log` and `mission.json` `stage_timeline`; this report's Step 1.
- **Base branch / PR target:** `fm/twin-robot-in-store-boot-v1`. **Out of scope:** the upstream SIGABRT
  heap corruption, the PR #247 respawn policy, the PR #249 re-attach logic (beyond confirming its log
  name), and any global `phase_timeout` increase.

## Step 1 — the finding (established on CPU from the respawn2b log, before any code change)

### Sim-time anchors (from `mission.json` `stage_timeline`, the mission packet clock)

| event | `sim_time` |
|---|---|
| collect attempt 1 begin | `0.0` |
| `grasp·lift·tuck` manip step begins | `0.6167` |
| collect attempt 1 **FAILED** `host fault: phase timeout: lift` | **`149.8667`** |

`TransferController.step` fails a phase when `self.elapsed > self.phase_timeout`, and `elapsed` is
per-phase (reset in `_transition`). The lift phase's budget is **120 sim-s**, and it tripped, so the lift
phase was **entered at sim ≈ 29.87** (`149.8667 − 120`). The grasp/contact/confirm steps consumed the
sim `0.6 → 29.9` window; the **lift phase then ran a full 120 sim-s to timeout**.

### Wall-clock recovery timeline (from `host-stack.log`, the ROS/unix clock)

| line | event | wall (unix s) |
|---|---|---|
| 3950 | post-attach ESDF kick → **lift armed** | `1792826714.9` |
| 4044 | `MOVE_GROUP_RESPAWN … pid=3994122 … attempt 1/2` (**death 1**) | `1792826718.0` |
| 11416–11417 | `MOVE_DEADLINE_FAULT clock=sim … state=execute_accept` → `MOVE_DEADLINE_RETRY phase=lift attempt=1/4` | `1792827461.4` |
| 11545 | `MOVE_GROUP_RESPAWN … pid=56634 … attempt 2/2` (**death 2**) | `1792827467.6` |
| 16229 | `HOST_FAULT_ESTOP phase=lift reason='phase timeout: lift'` | `1792827950.9` |

Lift wall span = `1236 s` (~20.6 min) for 120 sim-s ⇒ average pacing **≈ 0.097 sim/wall** (render-paced).
Mapping the wall events onto the lift sim clock (linear approximation, pacing is not perfectly uniform):

| lift sub-interval | wall Δ | **sim Δ (≈)** | what it is |
|---|---|---|---|
| lift entry → death 1 | 3.1 s | **0.3 s** | the actual lift plan/execute, before the first crash |
| death 1 → deadline-retry (cycle 1) | 743 s | **≈ 72 s** | orphaned lift goal waiting out its sim deadline, then re-issued |
| death 2 → phase timeout (cycle 2, truncated) | 483 s | **≈ 47 s** | second orphaned goal, cut short by the 120 s cap |

**So ~119 of the 120 sim-s lift budget was consumed by the two recovery cycles; only ~0.3 sim-s was real
lift motion.** The phase timed out purely because the recovery clock kept counting. A single real death
recovers within budget (hence this bit the double-injection proof), but any slow recovery — a long
respawn delay or a slow re-attach — converts a recovered crash into a timeout fault, and the triple-death
give-up proof (respawn3) is unreachable until this is fixed.

### The recovery path respawn2b actually took (this drives the fix)

The two deaths were handled entirely by the **launch respawn supervisor** (`MOVE_GROUP_RESPAWN 1/2`,
`2/2`, printed by the launch process, PR #247) plus the **controller's SIM-deadline `MOVE_DEADLINE_RETRY`
path** (fm367). The adapter's own crash-retry machinery (`_awaiting_server_respawn` /
`_resync_scene_after_respawn`) **never fired**: `grep` of that `host-stack.log` finds **zero**
`MOVE_GROUP_CRASH_RETRY`, `PAYLOAD_SYNC_DIAG`, `re-attached restock_carried_unit`, or `reconcile_detach`
lines. The launch respawned `move_group` inside the render-paced poll gap, so `_handle_server_crash`
(gated on `self.plan.server_is_ready() == False` for the current state) never observed the server down
during the state that mattered; the lost in-flight goal simply hung to its sim deadline and the controller
re-issued it.

Two consequences:

1. **The pause signal cannot rely on `_awaiting_server_respawn` alone** — it must also cover the
   launch-respawn + controller-deadline-retry path. The fix adds a `move_group` action-server **liveness
   latch**: a ready→not-ready transition on the plan (`move_group`) action server during an in-flight
   move is positive crash evidence, latched until the move completes. Under the measured ~0.17 s/poll
   cadence the 3 s respawn delay spans ~17 polls, so the latch catches the death even when the dedicated
   crash-retry path does not.
2. **The PR #249 `reconcile_detach_ok` log line was "not observed" simply because the reconcile path was
   never reached** in respawn2b — not because the string is wrong. The actual success gate is emitted by
   `ros_payload.RosPayloadSync._reconcile_detach` as
   `PAYLOAD_SYNC_DIAG gate=reconcile_detach_ok (surviving object_attachment attach cleared before re-ADD)`,
   which **already matches** `docs/examples/movegroup-reattach-after-respawn-mission-fragment.json`. No
   string change is needed; this task only adds a CPU test asserting that line is emitted on the reconcile
   path (previously only the boolean return was pinned).

## Step 2 — the fix (minimal, default-on, byte-identical with no crash)

**Controller (`restock_manipulation/controller.py`).** `step` credits the phase clock for the sim time
spent while the motion adapter reports it is inside a named `move_group` crash recovery
(`motion.crash_recovery_active()`), so the timeout comparison becomes
`self.elapsed - self._recovery_credit_s > self.phase_timeout`. With no recovery the probe is False, the
credit stays `0.0`, and the comparison is identical to the old `self.elapsed > self.phase_timeout` —
**byte-identical**. The credit is bounded by `_recovery_pause_cap_s` (env `RESTOCK_RECOVERY_PAUSE_CAP_S`,
default 300 sim-s, a belt-and-braces bound *above* the adapter's own faster-firing recovery bounds); if a
recovery outruns that cap the phase **fails fast by name** (`crash recovery exceeded its … sim pause
budget …`) rather than pausing forever. The paused total is logged once per phase
(`PHASE_CLOCK_RECOVERY_PAUSE phase=… paused=…s sim`). The probe is read through `getattr`, so a motion
stub without it ⇒ credit 0 ⇒ byte-identical.

**Adapter (`restock_manipulation/ros_adapter.py`).** `RosMotionPort.crash_recovery_active()` returns True
while `_awaiting_server_respawn` or `_reattach_active` (the explicit crash-retry hold / post-respawn
re-attach), **or** `_movegroup_death_seen` (the new liveness latch). The latch is set in `_poll` on a
`move_group` plan-server ready→not-ready transition during an active move, and cleared on a successful
move, on `stop()`, and at op start (`reset_crash_budget`). No death ⇒ latch never sets ⇒ byte-identical.

## Step 3 — GPU proof (handed to the integration lane, not run here)

`docs/examples/phase-timeout-during-recovery-mission-fragment.json` carries both bounded runs:
**respawn2c** (the respawn2b config unchanged — both deaths recover, no phase timeout, collect completes
carton-on-rollwagen) and **respawn3** (the triple-death variant — the third death logs
`MOVE_GROUP_RESPAWN_GIVE_UP`, no fourth `move_group` pid, the stage fails fast by name before any phase
timeout). PASS criteria are in the fragment. Do **not** run the GPU here; the integration lane holds
`/tmp/aile-gpu.lock`.

## CPU tests

- `scripts/twin/tests/test_restock_phase_timeout_recovery.py` — controller tier: a recorded two-recovery
  lift stays within budget with the pause; a hung recovery fails by name at the recovery bound; a
  no-recovery run has identical elapsed accounting and still times out by name; adapter-tier liveness
  latch + `crash_recovery_active()`.
- `scripts/twin/tests/test_restock_detach_ack.py` — adds an assertion that the reconcile success path
  emits `PAYLOAD_SYNC_DIAG gate=reconcile_detach_ok`.
