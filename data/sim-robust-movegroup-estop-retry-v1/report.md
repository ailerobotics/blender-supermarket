# sim-robust-movegroup-estop-retry-v1 — a mid-op move_group death retries instead of e-stopping

Make the Isaac restock mission less prone to failure from the intermittent upstream `move_group`
`exit code -6` (SIGABRT, `malloc(): unaligned tcache chunk detected`) death during a manipulation
op. The heap corruption itself is upstream (not patchable here; established across 20 run logs by
`sim-robust-movegroup-respawn-v1`). This lane closes the last recovery gap: a death that strikes
**mid-op — including mid-replan at the collect stow — must route through respawn + planning-scene
re-sync + a bounded retry of the interrupted plan, not an immediate `host emergency stop`.**

## 1. The gap (coldlive capstone, 2026-10-23 ~05:00–05:28, branch 9406181 = PR #223 + #224)

Evidence: `/home/manishsaini/firstmate/data/twin-robot-in-store-boot-v1/coldlive2-evidence.md`;
run dir `~/datasets/twin-robot-in-store-boot-v1/mission/coldlive/` (host-stack.log lines
11490–11535, console log lines 23380–23600 — note: the validator re-runs coldlive, so those exact
offsets are in the frozen evidence file, not necessarily the live dir).

The collect stage picked, carried, placed and **released** the carton cleanly (PR #223 validated —
no `payload collision detachment not acknowledged`). Then `move_group` died
(`process has died [pid 1770443, exit code -6]`, preceded by `malloc(): unaligned tcache chunk
detected` + a stack trace in thread 1770749) **during the collect retreat/stow re-plan**. The host
went **straight** to `MANIP_HOST_FAILED op=collect phase='host' reason='host emergency stop'` and
the arm heartbeat was lost. **No `MOVE_GROUP_CRASH_RETRY`, re-sync or re-attach line fired.**

### Why the existing recovery missed it

`sim-robust-movegroup-respawn-v1` (PR #224) + the earlier fm368/fm404/fm405 work give `move_group`
death recovery at exactly **two** trigger sites in `RosMotionPort._poll`:

1. `_handle_server_crash()` — fires at the top of `_poll` when the action server for the current
   state is observed **down** (`server_is_ready()` False). Holds the move open, waits for the launch
   respawn, re-syncs the scene (re-attach `restock_carried_unit`) and re-sends the SAME plan goal.
2. the **plan-result wall-deadline** path (`_should_deadline_crash_retry`) — fires when a WALL
   deadline elapses while waiting for a plan result (a respawn that completed *between* polls).

Both sites `return 'running'` and **never** call `emergency_stop()`. But a `move_group` process
death does not always present as "server reads down at the next poll" or "the plan request hangs to
its wall deadline." When the death strikes mid-request it can present as a **RAISE inside `_poll`**:
the in-flight action result future resolves with an exception on the lost server, or an **ABORTED
result arrives with no payload** so `result.status != 4 or result.result.error_code.val != 1`
dereferences `result.result` as `None` → `AttributeError`, or a service call fails. Every such raise
fell through to `poll()`'s generic handler:

```python
def poll(self):
    try:
        return self._poll()
    except Exception:
        self._last_fault_sim_deadline = False
        self.stop()          # <-- stop() calls emergency_stop() -> publishes the global /stop
        return 'failed'
```

`stop()` publishes `/restock_sim/manipulation/stop = True`, which the Isaac graph latches
(`IsaacRosGraph.check_health` → `faulted=True` → `RuntimeError('host emergency stop')`). That latch
is **terminal**: the controller's own bounded stow retry (`STAGED_MOVE_RETRY`) re-issues the move,
but the global e-stop has already killed the arm — the retry is moot. So a death that surfaced as a
raise during the stow re-plan e-stopped **immediately and irrecoverably**, which is exactly the
coldlive failure.

### Every path by which the host learns move_group died, and where each led (before this lane)

| Detection path | Mechanism | Before this lane |
|---|---|---|
| action-server disconnect at a poll | `server_is_ready()` False in `_handle_server_crash` | **recovers** (hold → respawn → re-sync → resend) |
| plan-result wall-deadline (respawn between polls) | `_should_deadline_crash_retry` | **recovers** |
| launch death → respawn | `launch.py respawn=True`; client re-sends | feeds the two paths above |
| **in-`_poll` RAISE** (lost-server result future, ABORTED result w/ no payload, service failure) | `poll()` `except` → `stop()` → `emergency_stop()` | **immediate e-stop (the gap)** |

The `cumotion::FatalException` container OOM / >1024-sphere death is a **separate** failure already
bounded by `PLAN_TOTAL_ATTEMPTS_CAP` + the sphere model — unchanged here.

## 2. The fix (`ros_adapter.py`, always-on, no-crash path byte-identical)

Route the in-`_poll` raise through the SAME bounded recovery as the other two paths, keeping the arm
holding its last commanded pose and the weld intact, and escalate to the e-stop only when the retry
budget is spent or the respawn never comes — with a named reason.

- **`poll()`** now passes the caught exception to `_recover_movegroup_death_or_none(exc)` **before**
  `stop()`. If it returns a status (`'running'`) the move is held; otherwise the honest e-stop path
  runs exactly as before.
- **`_recover_movegroup_death_or_none(exc)`** discriminates a recoverable upstream death from a
  genuine fault using the action server's readiness for the current state:
  - not in a plan/execute state (`_server_action_for_state()` is None) → genuine fault → e-stop.
  - the server is **still READY** → a live `move_group` delivered a real rejection/validation fault
    (IK failed, planning failed, invalid trajectory) → surface it honestly → e-stop. **This keeps
    every genuine failure byte-identical.**
  - the raise is the `RESPAWN_WAIT_CAP_S` `'did not respawn'` escalation → never masked → e-stop.
  - otherwise (server **gone**, a stored plan goal + crash-retry budget remain) → **hold**: arm
    `_awaiting_server_respawn`, invalidate the dead goal's late callback, reset the deadline to the
    respawn-wait budget, log one `MOVE_GROUP_CRASH_RETRY ... op-phase fault ... action server gone ->
    holding for respawn + scene re-sync instead of e-stop (retry N/3)` line, `return 'running'`. The
    next `poll()`'s `_handle_server_crash` then owns the respawn wait → PR #224 re-sync → re-send,
    and counts the retry.
- The decision is a pure helper **`_should_recover_from_poll_fault(server_ready, has_plan_goal,
  crash_retries, is_respawn_timeout)`** (CPU-testable), reusing `_should_crash_retry` for the budget.
- **`planning_wall_exceeded()`** is suspended while `_awaiting_server_respawn` is set, so the stale
  per-move `_plan_started_wall` can't let the bridge-watchdog planning-wall cap e-stop the hold; the
  dedicated `RESPAWN_WAIT_CAP_S` bounds the wait. This also hardens the pre-existing
  `_handle_server_crash` hold.

Boundedness is unchanged: `MOVE_GROUP_CRASH_RETRY_MAX` (3) full respawn cycles, each bounded by
`RESPAWN_WAIT_CAP_S` (45 s). After that, or if `move_group` never respawns, the move fails with a
named reason and e-stops. Switch keys are backward compatible (none added; the existing
`RESTOCK_RESPAWN_WAIT_CAP_S` / `RESTOCK_SENSOR_STREAM_WALL_BACKSTOP_S` still apply).

### Why no new emergency-stop suppression anywhere else
The controller-level stow/tuck/pan retries and the bridge-watchdog `retry_or_fail_move` path were
left untouched: they already avoid an e-stop on the paths they own (`failed_result_pending` →
`ack_failed_result` → re-issue, no `stop()`). The only unhandled death path was the in-`_poll` raise,
and it is now the sole change at the fault boundary.

## 3. CPU tests (`scripts/twin/tests/test_ros_adapter_branch.py`)

New class `MoveGroupDeathDuringOpRoutesToRetry` (drives the REAL `poll()` wrapper with a `_poll`
that raises, and the REAL `_handle_server_crash` hand-off, on a `__new__` port — no ROS/GPU):

- `test_pure_helper_recovers_only_a_gone_server_with_goal_and_budget` — the discriminator truth table.
- `test_death_raise_with_server_gone_holds_without_estop` — server gone → `'running'`, **no
  emergency_stop**, `_awaiting_server_respawn` armed, state preserved for the resend.
- `test_death_raise_at_each_op_move_recovers` — a death at each server-interaction state
  (`plan_accept` pick-reach, `plan_result` descent, `execute_accept` retreat, `execute_result` stow)
  holds instead of e-stopping (covers pick / descent / release-retreat / **stow**).
- `test_genuine_rejection_with_live_server_still_estops` — a live server's rejection raise still
  e-stops (byte-identical honest failure).
- `test_straight_line_descent_without_goal_estops` — the contact descent (no stored goal) e-stops as
  before.
- `test_retry_bound_exhausts_to_estop` — at `MOVE_GROUP_CRASH_RETRY_MAX` the recovery escalates.
- `test_respawn_timeout_raise_is_never_masked` — the `'did not respawn'` raise e-stops.
- `test_non_server_state_raise_estops` — an IK-service/terminal-state raise e-stops.
- `test_recovery_hands_off_to_handle_server_crash` — after the hold, the respawned server re-sends
  the SAME goal and counts one retry, no e-stop in between.
- `test_planning_wall_cap_suspended_during_respawn_wait` — a genuine hang still exhausts; a respawn
  hold is suspended.
- `test_no_crash_path_does_not_touch_recovery` — a clean `poll()` never reaches the recovery.

Run (no GPU, dodging the box's ROS pytest plugins):
```
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH= python3 -m pytest \
  scripts/twin/tests/test_ros_adapter_branch.py \
  scripts/twin/tests/test_restock_manipulation.py \
  scripts/twin/tests/test_restock_host.py \
  scripts/twin/tests/test_sim_robust_watchdogs.py -q
```
→ branch file 70 passed (11 new in this class); 268 passed + 15 skipped (usd/ROS-gated) across the
four files.

## 4. Bounded GPU proof (PENDING — handed to the integration lane)

A `move_group` SIGABRT is intermittent and cannot be forced from inside the mission, and the mission
process does **not** own the `move_group` pid (launch does), so **no repo code signals it**. The
bounded proof is a **collect-only run** (`--stop-after collect`; the stow is the last collect move)
that either carries + stows normally (fix inert, byte-identical) or, if a death occurs at/near the
stow, shows the recovery sequence in the log instead of a terminal `host emergency stop`. To force
the exact coldlive failure deterministically, the integration lane (which owns the single-GPU boot,
so only its `move_group` is up) sends `kill -ABRT` to the `move_group` pid when the console reaches
the stow. Config, run recipe, kill-injection step and the exact greps:
`docs/examples/movegroup-estop-retry-collect-fragment.json`. One GPU run at a time through
`/tmp/aile-gpu.lock`, never `--force`; whispr (pid 1339, ~2.2 GB) untouched. Report PASS/FAIL + run
dir on the done line.
