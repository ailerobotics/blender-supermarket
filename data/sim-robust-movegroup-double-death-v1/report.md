# sim-robust-movegroup-double-death-v1

Close the **two remaining move_group-death recovery gaps** the canonical capstone **coldwin4**
exposed during collect, so a move_group death — or two — recovers without losing the carton from
the planning scene and without a stale trajectory_host goal rejecting the reseeded resend.

- **Run that motivated it:** `~/datasets/twin-robot-in-store-boot-v1/mission/coldwin4` (head `5611965`,
  PR #240, config `mission-runtime-capstone238.json`, `RESTOCK_STATION_SIM_GAP_S=30`, peak VRAM 10549 MiB).
  Evidence (the exact gap log lines): `~/firstmate/data/twin-robot-in-store-boot-v1/coldwin4-evidence.md`.
- **Base branch / PR target:** `fm/twin-robot-in-store-boot-v1` (recovery stack PR #224/#227/#228/#229/#234
  land here as #-numbered merges). The move_group SIGABRT itself (exit -6, `malloc(): unaligned tcache chunk
  detected` / `malloc_consolidate`) is **irreducible upstream** MoveIt/cuMotion instability — not fixed here.

## The two gaps

**Gap 1 — the post-respawn carton re-attach trusted a single attempt and then planned carton-less.**
coldwin4 death 1 (op 1/6): `MOVE_GROUP_CRASH_RETRY` engaged, but the re-attach of `restock_carried_unit`
to the respawned planning scene **failed** and the recovery proceeded without the carton (a resend would
plan straight through it). The validator's forensics (inbox correction) showed the failure was **not** a
stale/!contact sensor packet: `PAYLOAD_SYNC_DIAG` at the respawn read `held=True s_contact=True age=0.067
scene_ready=True attach_ready=True`, yet the gate was `attach_goal_rejected` — the freshly respawned
move_group **rejected the attach goal** because it had not finished re-initialising.

**Gap 2 — a mid-execution respawn+reseed left a stale ACTIVE goal that rejected the resend.**
coldwin4 death 2 (op 2/6): `MOVE_GROUP_RESPAWN_RESEED` re-anchored to live measured joints, then
`TRAJ_HOST_GOAL_REJECT a previous goal is still executing (active=True)` → `PORT_FAULT_ESTOP execution
failed` → `TRAJ_HOST_FAULT trajectory tracking error 1.314 exceeds 1.3 rad` → e-stop. move_group crashed
**mid-execution**, leaving its `FollowJointTrajectory` goal LIVE on the trajectory_host; PR #229's op-start
reset only clears a STALE active flag (`reset_clears` never touches a live goal), so the reseeded resend
was rejected and the abandoned goal then tracking-faulted.

## The fix

All of it is **always-on + self-disabling**: it runs only on a crash-retry resend, so a live-server
rejection and the whole no-crash path are byte-identical. No new mission runtime-config key.

### Gap 1 — bounded sim-clock re-attach; never resend carton-less (`ros_adapter.py`)
- Every crash-retry resend site now funnels through `_begin_post_respawn_recovery()`. When a carton is
  held (`_attached_profile` set — the authoritative object_attachment mirror), it starts a **poll-spanning**
  re-attach: `_step_reattach()` is stepped once per `poll()` (so the node `/clock` advances and the
  respawned attach server finishes init between attempts — a busy-wait in one poll would freeze the port's
  single-threaded-executor clock) and resends the plan **only once the attach is acknowledged**.
- Bounded on the SIM clock by `REATTACH_SIM_BUDGET_S` (env `RESTOCK_REATTACH_SIM_BUDGET_S`, default 30 sim-s,
  well under the 120-sim-s collect phase timeout). On exhaustion the port sets `_reattach_failed`, holds
  (`_fail_hold`, no global stop), and the controller (`_planner_crash_failfast_or_none`) **fails the op by
  name** — "the carried carton could not be re-attached to the respawned move_group planning scene …
  failing fast rather than planning without the carton" — rather than planning through the missing carton.
- `_resync_scene_after_respawn` is now a single attempt the retry loop calls; its pre-fix "the resend will
  plan WITHOUT it" log is removed (the caller never does that anymore).

### Gap 2 — force-clear the crash-abandoned live goal before resending
- New `ros_adapter` publisher `/restock_sim/arm_controller/abort_active`; `_publish_abort_active()` fires on
  every crash-retry resend (inside `_begin_post_respawn_recovery`).
- `trajectory_host.abort_active_signal` (new subscription) + the pure `abort_active_clears` helper: it
  REQUESTS the live goal's own `execute()` loop to self-terminate (`handle.abort()`, **no fault latched** —
  a crash-abandoned goal is not a tracking failure) and clears any latched fault, so the reseeded resend is
  accepted. A stale active flag is still cleared by the existing `active_is_stale` path; an idle, fault-free
  executor is a no-op (byte-identical). The resend already starts from the live measured joints
  (`_reanchor_plan_goal_to_live` + the `await_settle` reseed + the trajectory_host's first-feedback anchor),
  so tracking does not diverge from the reseeded reference.
- PR #227's per-op crash budget (`MOVE_GROUP_CRASH_RETRY_OP_MAX`) is unchanged: two deaths in one op each
  recover and accrue the op total.

## CPU tests (extend the existing recovery harness; no parallel harness)
- `tests/test_ros_adapter_branch.py::MoveGroupDoubleDeathRecovery` — gap 1: re-attach retries on the sim
  clock then resends when accepted; fails the op by name when the respawned planner never accepts (no
  carton-less resend, `_fail_hold` not a global stop); no-carry resend is immediate (byte-identical);
  `_handle_server_crash` defers the resend until the carton is re-attached. gap 2: every crash-retry resend
  publishes `abort_active` before resending (carry/no-carry × plan_result/execute_result = pick/descent/stow);
  two deaths in one op each recover and accrue the op budget; `_publish_abort_active` plumbing.
- `tests/test_restock_trajectory_host.py::test_abort_active_clears_forces_only_a_live_goal_and_clears_fault`
  — the pure gap-2 helper, contrasted with `reset_clears` (op-start reset keeps a live goal; abort clears it).
- Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=scripts python3 -m pytest
  scripts/twin/tests/test_ros_adapter_branch.py scripts/twin/tests/test_restock_trajectory_host.py -q`
  (143 pass). The 4 `test_restock_collect_stow_stall.py` failures that call `_poll()` directly are
  **pre-existing** in this box (no `moveit_msgs` installed) — they fail identically on the base commit.

## Bounded GPU proof (integration lane owns the boot)
`docs/examples/movegroup-double-death-estopretry-fragment.json` — the **estopretry** recipe: collect-only
under the full capstone238 keys (`--stop-after collect`) with a **double injected `kill -ABRT move_group`**,
one MID-PLAN (gap 1) and one MID-EXECUTION (gap 2). PASS = both recover to the post-release stow with the
carton attached and no `host emergency stop` / `arm command heartbeat lost`; a bounded NAMED fast fail
(re-attach budget, respawn cap, or planner-unrecoverable) is also an acceptable PASS when move_group
genuinely never respawns. One GPU run at a time through flock `/tmp/aile-gpu.lock`, never `--force`, never
touch whispr (pid 1339, ~2.2 GB). The integration lane must not have its `~/datasets` mission configs
edited by this worker — copy `mission-runtime-capstone238.json` and add the keys.
