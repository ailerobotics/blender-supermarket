# sim-robust-collect-stow-stall-v1

The post-release collect **STOW** now survives a `move_group` planner **crash/hang**: a SIM-clock PLANNING
deadline is treated as positive evidence the planner is dead/hung, the port re-sends the plan a bounded
number of times (so a genuine crash+respawn recovers), and when it does not recover the op **fails fast
with a named cause** instead of looping to the opaque 120-sim-s `phase timeout: stow`. Branch
`fm/sim-robust-collect-stow-stall-v1` off `fm/twin-robot-in-store-boot-v1` (head `34b83b4`, i.e. PR 233).
CPU tests only; the bounded GPU proof is handed to the integration lane
(`docs/examples/collect-stow-stall-proof-fragment.json`).

## 1. What coldliveapproach actually shows (root cause)

Evidence: `~/datasets/twin-robot-in-store-boot-v1/mission/coldliveapproach/host-stack.log` and
`coldliveapproach.console.log`; validator write-up
`~/firstmate/data/twin-robot-in-store-boot-v1/coldliveapproach-evidence.md`.

The collect op **picked, carried, placed and released** the carton (`action=collect ... transport_ready`
reached release), then the STOW stalled for the full **120 simulated seconds** and e-stopped:

```
MOVE_DEADLINE_FAULT clock=sim over=0.00s state=plan_accept straight=False   (host-stack.log:17110)
STAGED_MOVE_RETRY phase=stow attempt=1/4
MOVE_DEADLINE_FAULT clock=sim over=0.00s state=plan_accept straight=False   (20272)  attempt=2/4
MOVE_DEADLINE_FAULT clock=sim over=0.00s state=plan_accept straight=False   (23491)  attempt=3/4
HOST_FAULT_ESTOP phase=stow reason='phase timeout: stow'                    (26732)
```

Each attempt burned **~255 WALL seconds** for **~30 SIM seconds** (sim ≈ 0.12× wall, render-paced *slow*),
i.e. the goal was never even **ACCEPTED** across the whole 30-sim-s PR-228 planning budget, 4× in a row.

The proximate cause is one line, at the retreat_tuck→stow boundary, right before the first fault:

```
[move_group-1] Execution completed: SUCCEEDED
[move_group-1] malloc_consolidate(): invalid chunk size      (host-stack.log:13869)
```

`move_group-1` produced **zero output after that line** — it SIGABRT'd on a heap corruption (the known
MoveIt/cuMotion octomap-clear data-race class, `malloc_consolidate` / `unaligned tcache chunk`) and **did
not respawn**. Every stow plan goal went to the dead `/move_action` server; goal **acceptance** — a
pacing-independent ms handshake — never returned, so the port sat in `plan_accept` until the SIM-clock
deadline. The passing sibling `hosttimers` (collect-only, no economy, no cart-relative couple) had **zero**
`move_group` crashes and completed the same stow — so the stall is this crash, **not** the `masked_depth`
economy publish interval (which never feeds the `plan_accept` budget).

## 2. The regression PR 228 left

Three recoveries existed for a `move_group` crash, and all three were inert here:

* `_handle_server_crash` (clean server-down poll) is gated on `not server_is_ready()`. A **SIGABRT-hung**
  (not cleanly exited) `move_group` keeps a **stale-true** `server_is_ready()` (no clean DDS
  deregistration), so it never fired — no `MOVE_GROUP_CRASH_RETRY`, no respawn-wait, no `did not respawn`.
* `_should_deadline_crash_retry` (the fm404 belt-and-braces for a crash+respawn whose lost plan hangs to
  the deadline) is gated on `clock == 'wall'`. **PR 228 moved every planning deadline to the SIM clock**,
  so that condition is now unreachable during planning — the catch is dead code for the real case.
* The poll-raise recovery (`_should_recover_from_poll_fault`) needs a result future that raised; a
  never-resolving acceptance future does not raise.

So the sim deadline flagged a plain `_last_fault_sim_deadline`, the controller's STOW `failed` branch did a
blind `STAGED_MOVE_RETRY`, re-sent to the still-dead planner, and looped — 4 × 30 sim-s = the 120-sim-s
phase timeout, reported only as the opaque `phase timeout: stow`. (Before PR 228 the WALL deadline was 25 s,
so the same crash failed ~20× faster in wall time, still opaquely.)

## 3. Fix

A **SIM-clock PLANNING-state deadline** (`plan_accept`/`plan_result`) is positive evidence of a dead/hung
`move_group` — PR 228's own rationale is that healthy planning consumes only ~1–2 sim-s, so the full sim
budget elapsing with the goal never accepted / plan never returned is a hang, not render-slow. In
`RosMotionPort._poll` (ros_adapter.py), such a deadline now:

1. re-sends the plan a bounded `PLAN_ACCEPT_CRASH_RETRY_MAX` times (default 1; env
   `RESTOCK_PLAN_ACCEPT_CRASH_RETRY_MAX`), re-anchored + scene-resynced exactly like the fm404 crash-retry,
   so a genuine **crash+respawn** whose server stale-reads ready still recovers — the belt-and-braces, now
   reachable on the sim clock; then
2. sets `last_fault_was_planner_unrecoverable()` and holds (`_deadline_fail('sim')`, no global stop).

The controller (`_planner_crash_failfast_or_none`, checked **before** the sim-deadline / staged retry at
each move `failed` branch) then fails the op fast with a named cause: *"<phase> aborted: the move_group
planner crashed or hung (a plan goal was never accepted / no plan returned within the sim planning budget)
and did not respawn — failing fast"*. Default budget ⇒ the named fail lands at ~60 sim-s, well under the
120-sim-s collect phase timeout.

Scope / backward compatibility:

* **No new mission key** — on by default; `RESTOCK_PLAN_ACCEPT_CRASH_RETRY_MAX` is the only tunable.
* Byte-identical for every passing path: a healthy run **never reaches a sim planning deadline**, so none
  of the new code executes. Execute/await_settle sim deadlines and goal-less straight-line moves fall
  through to the existing fm367 sim-deadline hold unchanged. The pure `_should_deadline_crash_retry`
  predicate is left wall-only (the new path is separate), so its existing callers/tests are unchanged.
* The `move_group` heap crash itself is **upstream** (MoveIt/cuMotion octomap handling) and not fixable in
  the host; this deliverable is the host-side robustness the spec asks for — reliable completion when the
  planner is healthy or recovers, else fail-fast-by-named-cause.

The spec's other stow waits were checked and already safe: a never-arriving sensing/ESDF-incorporation ack
during the (empty-handed) stow trips the per-tick `CONTROLLER_INVARIANT_TRIP` immediately with a named
collision-freshness reason (pinned by `test_stow_never_arriving_sensing_ack_fails_fast_named`); a true goal
miss (goal accepted, executed, missed) is unrelated to this crash and keeps PR 221's classifier.

## 4. CPU proof (no Isaac/ROS/GPU)

`scripts/twin/tests/test_restock_collect_stow_stall.py` (10 tests):

* **Port tier** — a port whose SIM-clock `plan_accept` deadline has fired with a stale-ready server:
  re-sends once as a crash-retry (`MOVE_GROUP_CRASH_RETRY`, state back to `plan_accept`, fresh budget, no
  e-stop); exhausted ⇒ `_planner_unrecoverable`, `last_fault_was_planner_unrecoverable()` True,
  `last_fault_was_sim_deadline()` False, no e-stop; a `plan_result` sim deadline routes the same; an
  **execute**-state sim deadline is NOT a planner crash (byte-identical fm367 hold); the fm404 predicate is
  unchanged; the named fail is bounded below the phase timeout.
* **Controller tier** — driven through the known-good staged-swing collect to the STOW: a terminal planner
  crash fails fast with the named reason (not a `STAGED_MOVE_RETRY` loop); a plain transient sim deadline
  still retries; a port without the probe is byte-identical; a never-arriving sensing ack fails named.

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest scripts/twin/tests/test_restock_collect_stow_stall.py -q`
→ 10 passed. The full `test_ros_adapter_branch.py` + `test_restock_manipulation.py` +
`test_host_timers_pacing.py` + `test_fm367_sim_deadline.py` suites pass (minus two **pre-existing**
environment failures unrelated to this change: `test_actual_ros_message_request_and_async_progression` and
`test_begin_joint_goal_straight_line_dispatches_executor_no_plan`, which also fail on the clean base).

## 5. Bounded GPU proof (handed to the integration lane)

`docs/examples/collect-stow-stall-proof-fragment.json`: a **collect-only** run (`--stop-after collect`)
under the full coldliveapproach keys with economy ON, defaults for the new switch, under
`/tmp/aile-gpu.lock` + `--heavy-threshold-mb 2500` (whispr ~2.2 GB, never touched), never `--force`. PASS =
the stow completes to `transport_ready` (planner healthy or crash+respawn recovered), OR the op ends fast
with `PLANNER_UNRECOVERABLE` + a named `MANIP_HOST_FAILED` reason instead of `phase timeout: stow`.
