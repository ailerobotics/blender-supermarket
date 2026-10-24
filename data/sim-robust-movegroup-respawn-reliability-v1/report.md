# sim-robust-movegroup-respawn-reliability-v1

Make `move_group` respawn **bounded and observable** after its intermittent cuMotion
SIGABRT, and let the existing client fast-fail (PR 234 / PR 242) name the give-up — so a
collect-stage planner crash is either a logged, counted retry or a named stage failure,
never a lost 20-30 min GPU run.

Branch: `fm/sim-robust-movegroup-respawn-reliability-v1` → PR into
`fm/twin-robot-in-store-boot-v1`. CPU-only; one bounded GPU proof config is handed to the
integration lane (never run here).

## 1. Audit (written before the fix)

### How `move_group` is launched and supervised today

`scripts/twin/restock_host/launch.py` launches `move_group` as a `launch_ros.actions.Node`
with:

```
respawn=True, respawn_delay=3.0          # no respawn_max_retries => ROS 2 launch default -1 = INFINITE
```

and deliberately **excludes** it from the shutdown-guard set (`guarded`): the other
components (`container`, maskers, `manipulation`, `trajectory`) carry a
`RegisterEventHandler(OnProcessExit → EmitEvent(Shutdown))` so their death tears the whole
stack down; `move_group` does not, because its crash is meant to be a recoverable respawn
(fb9f1d1, run-266/268: before that, `move_group`'s death SIGINT'd the entire stack and cost
a full GPU run).

There is **no second supervisor** for `move_group`. The only other process supervisor in
the stack is `masked_depth_ipc.SubprocessSupervisor`, which watches the *arm masked-depth
publisher*, a different process. So today nothing can "fight" the launch respawn — and the
fix must preserve that (exactly one respawner).

The client side (`scripts/twin/restock_manipulation/ros_adapter.py`) already carries the
recovery that consumes the respawn:

- `_handle_server_crash` + `MOVE_GROUP_CRASH_RETRY_MAX=3`: when the action server for the
  current state vanishes, hold the move open for the respawn, then re-send the SAME plan
  goal (PR 242 second-death path, `_reanchor_plan_goal_to_live`).
- `RESPAWN_WAIT_CAP_S=45`: if the server does **not** come back within the cap, raise
  `move_group did not respawn within 45s` → the stage fails **fast by name** (PR 234).
- `_should_deadline_crash_retry` (fm404 B-lite): a SIGABRT+respawn that completes *between*
  `poll()`s is caught by the sim-deadline crash-retry.

### Why launch respawn fires for some deaths and not others

ROS 2 launch respawn is driven **purely by process EXIT**. In
`launch/actions/execute_local.py::__execute_process` the respawn branch is reached only
after `returncode = await self._subprocess_protocol.complete` resolves — i.e. only once the
OS has actually reaped the process and `ProcessExited` is emitted. The respawn branch then
honours `respawn_max_retries` (`-1` = infinite) and `respawn_delay`.

Consequence, confirmed against the three runs in
`~/datasets/twin-robot-in-store-boot-v1/mission/`:

| run | last `move_group` line | `process has died` | respawned? | outcome |
|-----|------------------------|--------------------|-----------|---------|
| **dblproof** (injected `kill -ABRT`) | — | `exit code -6` (line 6177) | **yes**, new pid 2897375 | clean exit → respawn works |
| **coldwin7** (first honest full cycle) | — | `exit code -6` (line 31137) | **yes**, new pid 2462644 | 1 real crash, respawned, cycle completed; client logged `MOVE_GROUP_CRASH_RETRY` |
| **coldliveapproach** (the open follow-up) | `malloc_consolidate(): invalid chunk size` (line 13869) | **NEVER logged** | **no** (single pid for the whole run) | `HOST_FAULT_ESTOP phase=stow reason='phase timeout: stow'` |

**Root cause of coldliveapproach's missing respawn:** `move_group` printed glibc's
`malloc_consolidate(): invalid chunk size` and then **hung in the abort path instead of
exiting** — the classic multithreaded-SIGABRT deadlock (the aborting thread re-enters the
already-corrupted allocator / backtrace while another thread holds the arena lock, so the
process neither exits nor services requests). Because it never exited, launch's
`ProcessExited` never fired — hence *no* `process has died` line, *no* respawn attempt, and
*no* shutdown. This rules out the brief's other candidates:

- **respawn not set** — ruled out: `respawn=True` landed 2026-10-18 (fb9f1d1), well before
  coldliveapproach (2026-10-23).
- **limit exhausted** — ruled out: the launch default is infinite and this was the first
  (and only) death.
- **process group killed** — ruled out: the rest of the stack stayed up and kept serving
  ESDF requests to the end.
- **respawn itself crashing** — ruled out: no second `process started with pid` was ever
  attempted.

So a clean-SIGABRT death (dblproof, coldwin7) **is** respawned reliably today; the hang
(coldliveapproach) is **structurally un-respawnable by launch** and must be owned by the
client fast-fail. Note that coldliveapproach ended on the *generic* `phase timeout: stow`
rather than the named `move_group did not respawn` — because a hung process still *advertises*
its action server in DDS discovery, so `server_is_ready()` reads True and the client never
enters `_handle_server_crash`. That naming gap is client-side and belongs to the
PR 234/242 recovery lane; it is called out as a follow-up below, not changed here (scope).

### Gaps this lane fixes

1. **Unbounded respawn.** A genuinely flapping planner (crash-on-every-plan) would respawn
   forever with a 3 s delay, never surfacing as a clean failure — it would instead be
   masked until the mission's overall wall cap or a GPU OOM ended the run opaquely.
2. **No observability.** The launch logs only the generic `process has died` / `process
   started with pid`; nothing names *move_group respawn attempt N of M*, and nothing names
   the give-up. The double-death proof needs each respawn and the give-up to be greppable
   by name with an attempt count.

## 2. Fix (minimal, launch-side, observe-only)

- **`scripts/twin/restock_host/movegroup_respawn.py`** (new, import-light, no ROS): the
  single source of truth for the policy and its by-name log lines.
  - `resolve_respawn(config)` → `(max_retries, delay)` from an optional
    `config['movegroup_respawn']` block (`max_retries`, `delay`), clamped. Absent ⇒ the
    bounded defaults.
  - `respawn_decision(death_count, max_retries)` → `('respawn', attempt, limit)` while the
    Nth exit is within budget, else `('give_up', limit, limit)`.
  - `respawn_log_line(...)` → the `MOVE_GROUP_RESPAWN attempt N/M` / `MOVE_GROUP_RESPAWN_GIVE_UP`
    strings.
  - `RespawnSupervisor` — **observe-only**: it counts exits and renders the log line. It has
    **no respawn/kill path**, so it cannot fight the launch respawn (the single respawner
    stays ROS 2 launch).
- **`scripts/twin/restock_host/config.py`**: `launch_plan` now resolves and surfaces
  `plan['movegroup_respawn'] = {'max_retries', 'delay'}` (so preflight can see it and it is
  CPU-tested the same way as the rest of the plan).
- **`scripts/twin/restock_host/launch.py`**: pass `respawn_max_retries=max_retries` and
  `respawn_delay=delay` to the `move_group` Node, and register one
  `OnProcessExit(target_action=move_group, on_exit=supervisor.on_exit)` handler that logs
  each respawn/give-up by name with the attempt count. The handler returns no launch
  actions (observe-only), so it adds **no** second teardown — the give-up leaves
  `move_group` dead and the existing client `RESPAWN_WAIT_CAP` fast-fail (PR 234) names the
  stage failure.

**Defaults:** `max_retries=3`, `delay=3.0` (env `RESTOCK_MOVEGROUP_RESPAWN_MAX` /
`RESTOCK_MOVEGROUP_RESPAWN_DELAY_S`). coldwin7 (a full honest cycle) crashed once, so 3
lifetime respawns is generous headroom for a healthy run while bounding a flapping planner
to ~3 respawns before the client fast-fails by name. **Byte-identical when the planner never
dies** (no exit ⇒ handler never fires ⇒ no respawn ⇒ no behaviour change); the only
behaviour change is in the pathological >3-crash case, which is exactly the "will not come
back" case we now fail by name instead of masking.

## 3. Bounded GPU proof (integration lane owns the boot)

`docs/examples/movegroup-respawn-reliability-mission-fragment.json` — the `movegroup_respawn`
block to merge into the host-stack config, plus the two variants driven by the validator's
v3 double-death injector (`~/datasets/twin-robot-in-store-boot-v1/mission/inject_dbl.sh`):

- **Double-death (PASS):** `max_retries: 2`, two `kill -ABRT` during collect → both deaths
  log `MOVE_GROUP_RESPAWN ... attempt 1/2` and `2/2`, `move_group` respawns twice and
  **collect completes**.
- **Triple-death (give up):** same config, a third `kill -ABRT` → the third exit logs
  `MOVE_GROUP_RESPAWN_GIVE_UP ... (2/2 respawns used)`, `move_group` is not relaunched, and
  the stage **fails fast by name** via the client `RESPAWN_WAIT_CAP` (`move_group did not
  respawn within 45s`).

## 4. Tests

`scripts/twin/tests/test_movegroup_respawn.py` (CPU, no ROS):

- respawn limit honoured (give-up exactly at `max_retries + 1`);
- give-up reason named (contains `move_group`, `GIVE_UP`, the count);
- delay + limit threaded from config into the resolved plan (`respawn_delay` respected);
- no double supervision (the supervisor exposes no respawn/kill, returns no launch actions;
  `move_group` stays out of the guarded shutdown set);
- byte-identical default when `movegroup_respawn` is absent.

## Out of scope / follow-ups

- The upstream cuMotion heap corruption itself.
- Making the client name a **hung** (advertised-but-wedged) `move_group` as the crash fast-fail
  instead of a generic `phase timeout` (coldliveapproach): that is a client-side liveness
  probe in the PR 234/242 recovery lane. A launch-level liveness watchdog that *kills* a
  wedged `move_group` so launch can respawn it would also address the hang, but it is a
  larger change (and a second supervisor) than this lane's minimal, observe-only fix.
