# sim-robust-plan-deadline-pacing-v1

The manipulation host's PLANNING-phase per-move deadline (`ik` / `plan_accept` / `plan_result`) is now
measured on the **SIM/node clock**, not WALL, so a render-paced slow-but-alive cuMotion plan is never
mis-read as a `move_group` crash; and the deadline-triggered crash-retry now requires **positive evidence
of a death** (the `move_group` action server gone) before treating a missed deadline as a crash. Branch
`fm/sim-robust-plan-deadline-pacing-v1` off `fm/twin-robot-in-store-boot-v1` (head `bbb5d02`, i.e. PR
224+225+226+227 merged). CPU tests only; the bounded GPU proof is handed to the integration lane (recipe
below, `docs/examples/plan-deadline-pacing-collect-fragment.json`).

## 1. What coldlive4 actually shows (root cause)

Evidence: `~/datasets/twin-robot-in-store-boot-v1/mission/coldlive/host-stack.log` (the sample-3 log, still
carrying the coldlive4 signature) and
`~/firstmate/data/twin-robot-in-store-boot-v1/coldlive-3sample-diagnosis.md`.

Three faults, all at **`state=plan_accept`**, each a hair over the deadline, with **ZERO `move_group`
deaths** (`grep -c 'process has died' = 0`):

```
MOVE_DEADLINE_FAULT clock=wall over=0.03s state=plan_accept straight=False
MOVE_GROUP_CRASH_RETRY plan-result wall-deadline (likely move_group crash+respawn between polls); resending plan goal (retry 1/3)
MOVE_DEADLINE_FAULT clock=wall over=0.06s state=plan_accept straight=False
MOVE_GROUP_CRASH_RETRY ... (retry 2/3)
MOVE_DEADLINE_FAULT clock=wall over=0.00s state=plan_accept straight=False
MOVE_GROUP_CRASH_RETRY ... (retry 3/3)
MOVE_DEADLINE_FAULT clock=wall over=0.10s state=plan_accept straight=False   # budget spent -> e-stop
```

### Measured pacing under the arm feed
The fully-honest `live_pose` transfer enables the `masked_depth` ARM FEED — a **second full-resolution
render** — which heavies the simulator's render pacing. From the run's own artifacts: nav-trace mission
time spanned **~32.3 s SIM** while the nvblox wall-epoch stamps spanned **~522 s WALL** → **~0.062x
sim-to-wall (≈16x wall per sim-second) overall**, and during the three crash-retry windows (~225 s wall for
~6 s sim of collect) **≈30x wall per sim-second**. So the legacy **25 s WALL** plan deadline corresponds to
only **~1.5 sim-seconds**: the single-threaded host executor, starved by the 2nd render, could not service
the plan-ACCEPT action callback within 25 s WALL even though `move_group` was alive and had barely advanced
a sim-second.

### The mis-read
The WALL deadline fired, and `_should_deadline_crash_retry` (which only checked `clock=='wall'` + plan
state + stored goal + budget) **assumed a crash+respawn-between-polls** and did a crash-retry. With no
actual death, it exhausted `MOVE_GROUP_CRASH_RETRY_MAX` (3) and the host raised `MANIP_HOST_FAILED ... 'host
emergency stop'` at ~sim 24 s. A slow plan was read as a crash. The scripted/still `coldhonest` path has no
arm feed, lighter pacing, and completes — confirming this is a render-pacing/wall-timer artifact, not a
planning fault.

## 2. Inventory of the host's planning/execution wall timers (before this lane)

| timer | clock | where | status after this lane |
|---|---|---|---|
| per-move deadline `self.timeout` (`MOVE_WALL_TIMEOUT_S`, 25 s) on `ik`/`plan_accept`/`plan_result` | **WALL** | `begin`, `_send_ik`, `begin_joint_goal`, replan reset, crash-retry re-sends | **→ SIM clock** (`PLAN_SIM_DEADLINE_S`, default 30 sim-s) |
| `_should_deadline_crash_retry` (deadline → crash guess) | WALL-gated | `_poll` deadline branch | **+ positive-death gate** (`not server_ready`); unreachable on a healthy plan now that the deadline is SIM |
| `planning_wall_exceeded` (`PLAN_TOTAL_WALL_CAP_S`, sim-widened) | WALL | bridge watchdog | **unchanged** — the genuine-process-HANG backstop stays on wall |
| execute/settle deadline (`sim_execute_budget`) | SIM | execute states | already SIM (fm367/PR226) — unchanged |
| respawn-wait hold (`RESPAWN_WAIT_CAP_S`) | WALL | `_handle_server_crash` / `_recover_movegroup_death_or_none` holds | unchanged — bounded hold, never checked while the server is down |
| server-down death detection (`_handle_server_crash`, `_recover_from_poll_fault`) | n/a | poll | unchanged (PR 225/227) — real deaths recover here with positive evidence |
| SIM-deadline → `_fail_hold` not `emergency_stop` | SIM | `_deadline_fail` | unchanged (PR 227) — a plan-state SIM miss now benefits from it too |

The two timers that tripped under the arm feed were the per-move WALL deadline and the crash-retry guess it
fed. Everything else was already sim-clock (PR 201/204/205/226/227) or is the legitimate wall hang/respawn
backstop.

## 3. Fix (`scripts/twin/restock_manipulation/ros_adapter.py`)

Default-on, no new **required** switch key, no-crash path byte-identical in outcome; one backward-compatible
env override.

1. **Planning deadline on the SIM clock.** New `PLAN_SIM_DEADLINE_S` (`RESTOCK_PLAN_SIM_DEADLINE_S`, default
   30 sim-s, clamped [5, 600]) and `RosMotionPort._arm_plan_deadline()` set
   `deadline_clock = self._sim_now; deadline = sim_now + PLAN_SIM_DEADLINE_S`. Used at every site that puts
   the port into an active planning wait: `begin` (pose goal), `_send_ik` (the IK service response is
   render-paced too), `begin_joint_goal`, the plan-reject re-plan reset, and the post-respawn crash-retry
   re-sends. A render-paced slow plan passes little SIM time (healthy planning ≈ 1-2 sim-s even at ~30x), so
   it waits instead of false-firing. A genuinely hung planner still fires after the bounded sim budget → a
   SIM-clock miss the controller re-issues (bounded `MOVE_DEADLINE_RETRY`, then an honest e-stop), and the
   **WALL** `planning_wall_exceeded`/`PLAN_TOTAL_WALL_CAP_S` backstop is untouched as the wall-side hang
   catch.

2. **Positive-death-evidence gate on the deadline crash-retry.** `_should_deadline_crash_retry` gains a
   `server_ready` parameter (default `True`, backward-compatible) and requires `not server_ready` — the
   `move_group` action server GONE — before treating a fired deadline as a crash+respawn. The call site
   passes `self.plan.server_is_ready()`. Because the planning deadline is now SIM, this WALL-gated path is a
   belt-and-braces catch for a crash+respawn that completed between polls (run-312) and fires ONLY with the
   server actually gone — never on a live-but-slow planner (coldlive4). Real deaths continue to recover via
   the server-down paths PR 225/227 provide (`_handle_server_crash`, `_recover_from_poll_fault`), which
   already require `not server_ready`.

### Why a slow plan waits while a real death still recovers
- **Slow plan (server alive):** the SIM budget tolerates the render-paced wait; the result arrives and the
  move proceeds. If it ever exceeds the budget it is a SIM miss → bounded re-issue, never a crash.
- **Real death (server down):** caught by `_handle_server_crash` (observed server-down, positive evidence)
  → hold for respawn → re-sync scene → re-plan. With the SIM deadline being generous, the port keeps polling
  across the respawn window instead of giving up at 25 s wall, so the down-window is more likely observed.
- **Crash that respawned fully between polls:** no result arrives → the SIM budget eventually fires → a
  bounded SIM re-issue re-plans to the healthy respawned server. Recovers without any crash guess.
- **Genuinely hung planner (server alive, no result):** fails by name after the bounded SIM budget (and the
  WALL `planning_wall_exceeded` backstop), via the controller's bounded `MOVE_DEADLINE_RETRY` → honest
  e-stop.

## 4. CPU tests

`scripts/twin/tests/test_ros_adapter_branch.py` — new class **`PlanDeadlineOnSimClockPacing`** (5 tests, a
fake wall/sim `_Clock` pair + a slow-but-alive planner, no ROS/GPU):
- `test_arm_plan_deadline_uses_the_sim_clock` — the deadline is armed on `_sim_now`, budget `PLAN_SIM_DEADLINE_S`.
- `test_slow_plan_under_3x_pacing_does_not_trip` — 12 sim-s delivery at 3x pacing (= 36 wall-s, PAST the old
  25 s WALL deadline) does NOT fire the SIM deadline, and is never a crash with the server alive.
- `test_hung_planner_fails_by_name_after_the_bounded_sim_budget` — sim past the budget fires, on the SIM
  clock, and is never a crash.
- `test_real_death_recovers_with_positive_evidence` — server gone → `_handle_server_crash` holds for respawn;
  the deadline crash-retry is permitted only with `server_ready=False`.
- `test_normal_path_is_byte_identical` — a healthy plan within the budget never fires and is never a crash.

Updated for the new `server_ready` gate: `test_deadline_crash_retry_covers_respawn_between_polls_fm404` and
the fm410 `begin_joint_goal` consequence assertion (a wall plan-deadline re-sends only with the server gone;
a live-but-slow server is not a crash). Fixtures that reach the sim-clock re-send were given a `_sim_now`
stub.

Suites green (`env -u PYTHONPATH PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv-usd/bin/python -m pytest`):
`test_ros_adapter_branch.py` (90), `test_fm367_sim_deadline.py`, `test_restock_manipulation.py`,
`test_ros_adapter_watchdog.py`, `test_restock_trajectory_host.py`, `test_restock_transfer_live_pose.py`,
`test_collect_ground_truth.py`, `test_restock_staged_start.py` — 325 passed / 15 skipped together. (The
full-dir run has 114 pre-existing failures confined to 5 unrelated files — `test_restock_sensing_gate/
sensing_stride/shelf_target/source_target/transfer_bench` — all a numpy-2.x `np.cross` 2D `ValueError` in
`.venv-usd`, none referencing this lane's symbols; unaffected by this change.)

## 5. Bounded GPU proof for the integration lane (PENDING — one flock'd run)

`docs/examples/plan-deadline-pacing-collect-fragment.json`. Collect-only (`--stop-after collect`),
fully-honest `live_pose` + `masked_depth` arm feed (the coldlive4 stress), **no injection**. GPU rule:
`flock /tmp/aile-gpu.lock`, never forced, never touch whispr (pid 1339, ~2.2 GB); the validator holds the
GPU for the proof — do not take it.

**PASS** = no false plan-result/plan-accept crash-retry (no `MOVE_GROUP_CRASH_RETRY plan-result
wall-deadline` without a preceding real `process has died ... exit code -6`; no `MOVE_DEADLINE_FAULT
clock=wall ... state=plan*`) **and** collect completes (`transport_ready=True`), with no terminal `host
emergency stop` / `arm command heartbeat lost`. Report PASS/FAIL with the run dir on the done line.

## Follow-ups (not in this PR)
- Default `PLAN_SIM_DEADLINE_S` (30 sim-s) is generous by design; if the integration lane observes a
  legitimately slower plan under heavier future pacing, raise via the env — no code change needed.
- The remaining WALL timer in the planning path is the deliberate `planning_wall_exceeded` hang backstop;
  it stays wall on purpose (a frozen sim clock must still be caught) and is already sim-widened for delivery.
