# sim-robust-movegroup-respawn-v1 — move_group SIGABRT crash + respawn scene re-sync

Make the Isaac restock mission less prone to failure from the intermittent `move_group`
`exit code -6` (SIGABRT) death during manipulation: (1) establish the crash mechanism from the
run logs on CPU, (2) fix what the repository controls — re-synchronise the planning scene from the
authoritative object-attachment state on respawn before any plan is resent, and make the ordering
invariant explicit and tested. Upstream binaries are NOT patched.

## 1. Mechanism (established from the 19 run logs on CPU)

The `exit code -6` deaths under `~/datasets/twin-robot-in-store-boot-v1/mission/` are **two
distinct failures** that must not be conflated:

### A. `move_group` heap corruption — THIS task's target
`[ERROR] [move_group-1]: process has died ... exit code -6`, immediately preceded by
`[move_group-1] malloc(): unaligned tcache chunk detected`. Seen in **coldlive** (host-stack.log
line 6782, preceded by the tcache message at 6765) and **run-266** (death 4955, tcache 4954).

What `move_group` was doing at the death, every time: handling a `MoveGroupMoveAction` planning
request — forwarding it to the `isaac_ros_cumotion` pipeline (`move_action: Received request` →
`Sending goal`). The crash is in a `move_group` worker thread (coldlive: thread 3542828), **not**
the main action thread, and **not** in the cuMotion planner: the cuMotion component lives in a
*separate* process (`component_container_mt-2`), which in coldlive went on to plan successfully
(`MoveGroup planning succeeded in 1.236 seconds`) while `move_group-1` was already dead. So the
corruption is inside the upstream `moveit_ros_move_group` + cuMotion-plugin binary, in the request/
planning-scene handling path, concurrent with the continuous nvblox ESDF updates flowing into
`move_group`'s `PlanningSceneMonitor` (every death is immediately preceded by an nvblox
`Successfully wrote requested ESDF` line, never by a repository scene call — see below).

**Is it reachable from repository code?** No — not through a concurrent scene mutation during an
in-flight plan. The repository's planning-scene mutations are:
- the grasp **attach** / release **detach** (`ros_payload.RosPayloadSync.__call__`, one
  `ApplyPlanningScene` diff + one `AttachObject` action), issued by `controller.py` strictly
  **between** moves (grasp/release phases run after the previous move's `poll()` returned
  `succeeded`, so the motion port is in `succeeded`/`idle`), and
- the startup `add_base_obstacle` / `add_world_obstacle` publishes, issued **once before**
  manipulation while the base is parked.

The controller is single-threaded, so an attach/detach never overlaps an in-flight plan. Grepping
the 40 log lines before each `tcache` death for any repository scene marker
(`BASE_OBSTACLE` / `WORLD_OBSTACLE` / `PAYLOAD_SYNC` / `apply_planning_scene` / `AttachObject`)
returns **nothing** — the corruption is not triggered by a repository `ApplyPlanningScene`/attach
call. It is an upstream heap bug exercised when `move_group` serialises a planning request against a
scene that carries the attached carton + base/world obstacles while nvblox is updating the monitored
scene. We do not and cannot patch that binary.

### B. `cumotion::FatalException` container death — a SEPARATE, already-capped failure
`[ERROR] [component_container_mt-2]: process has died ... exit code -6`, preceded by
`terminate called after throwing an instance of 'cumotion::FatalException'`. Two sub-causes in the
logs: `CUDA error: out of memory` (collectsrc, run-223) and
`SelfCollisionDistanceKernel() only supports up to 1024 collision spheres` (run-244, run-245). This
is the cuMotion CUDA container, not `move_group`, and it is already bounded by
`PLAN_TOTAL_ATTEMPTS_CAP` (the OOM plan-storm cap) and the sphere-count model. Out of scope here; no
change.

> The other `move_group-1` deaths in the set (run-185/223/230/231/235/237/268/288/290/295/300d/
> 300e/312/316) are the same class A corruption (or, where the death line names
> `component_container_mt-2`, class B); all 19 were inspected to separate the two classes.

## 2. What the repository controls, and the fix

Since the corruption itself is upstream, the recurrence it causes is reduced by making the existing
**respawn recovery** correct, and by pinning the serialization invariant the single-threaded
controller already satisfies.

### 2a. Respawn planning-scene re-sync (the core fix)
`launch.py` already respawns `move_group` (`respawn=True`, excluded from the teardown guard), and
`ros_adapter.RosMotionPort` already holds the move open and re-sends the plan goal
(`MOVE_GROUP_CRASH_RETRY`, two sites: `_handle_server_crash` when the server is observed down, and
the fm404 deadline path when the respawn completed between polls). **But a respawned `move_group`
starts with a FRESH planning scene that no longer holds `restock_carried_unit`** — in coldlive the
respawn landed between the grasp attach and the detach, so any plan resent after it ignored the
attached carton and would plan straight through it. PR #223 (sim-robust-release-detach-ack-v1) made
the *detach* tolerate the lost object; this lane fixes the *planning* done after the respawn.

Fix (`ros_adapter.py`):
- The port now mirrors the authoritative object-attachment state in `self._attached_profile`: set to
  the `ProductProfile` on a confirmed `update_payload(held=True)`, cleared on detach. This is the
  same state the controller drives through its single `update_payload` channel.
- New `_resync_scene_after_respawn()` re-issues the attach (`payload_sync(True, profile)`) when a
  carton is held, restoring it in **both** MoveIt's and cuMotion's scene. It is a **no-op when
  nothing is held** (so every no-carry crash-retry, and the whole no-crash path, is byte-identical),
  and best-effort-but-loud on a re-attach raise (the resent plan's own start-state self-collision /
  result validation still gates correctness; we never mask the failure silently).
- Both resend sites call `_resync_scene_after_respawn()` **before** `_reanchor_plan_goal_to_live()`
  and `_send(...)`, so `MOVE_GROUP_CRASH_RETRY` only resends after the scene is re-synchronised.

### 2b. Serialization invariant (explicit + tested)
The mechanism analysis shows repository code does **not** mutate the scene during an in-flight plan,
so there is no concurrency bug to remove — but the invariant is now explicit:
`update_payload` raises if the port is in a `_PLAN_IN_FLIGHT_STATES` state
(`plan_accept`/`plan_result`/`execute_accept`/`execute_result`/`await_settle`). On the real path the
port is `succeeded`/`idle` when the controller attaches/detaches, so the guard is inert and
byte-identical; a future concurrent caller fails loudly on CPU instead of corrupting `move_group`'s
heap at GPU time.

### Backward compatibility
No new switch key. The re-sync activates only on the already-abnormal respawn path and only while a
carton is held; the guard is inert on the serialized real path. The no-crash path is byte-identical.

## 3. CPU tests (`scripts/twin/tests/test_ros_adapter_branch.py`, class `MoveGroupRespawnSceneResync`)
- `test_respawn_mid_carry_reattaches_before_resending_plan` — a mid-carry respawn re-attaches
  `restock_carried_unit` and the recorded call order is `['attach', 'plan_send']` (re-attach BEFORE
  resend; the SAME plan goal is resent).
- `test_respawn_without_carry_does_not_reattach_byte_identical` — no attach issued when nothing is
  held; only the plan is resent.
- `test_resync_helper_reattaches_held_carton` / `test_resync_helper_noop_when_nothing_held` /
  `test_resync_helper_does_not_mask_a_reattach_raise` — the helper directly.
- `test_update_payload_tracks_authoritative_attachment_state` /
  `test_update_payload_failed_attach_leaves_mirror_untouched` — the mirror is set on a confirmed
  attach, cleared on detach, untouched on a failed attach.
- `test_update_payload_refuses_mutation_during_in_flight_plan` /
  `test_update_payload_allowed_between_moves` — the serialization guard.

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH= python3 -m pytest
scripts/twin/tests/test_ros_adapter_branch.py scripts/twin/tests/test_restock_manipulation.py
scripts/twin/tests/test_restock_host.py -q` → all pass (branch file 59 passed; manipulation+host 172
passed, 13 usd-gated skips).

## 4. Bounded GPU proof (for the integration lane)
A `move_group` SIGABRT is intermittent and cannot be forced deterministically, so the GPU proof is a
**bounded collect run that exercises the carry** (attach → lift → carry) and, if the crash fires,
confirms the recovery is correct — i.e. the host log shows
`MOVE_GROUP_CRASH_RETRY re-attached restock_carried_unit ... before resend` **before** the
`resending plan goal` line, and the carton is still carried after the retry (no plan-through-carton,
no lost GPU run). Config + run recipe: `docs/examples/movegroup-respawn-collect-fragment.json`
(bounded with `--stop-after collect`, honest or ground-truth collect, one GPU run at a time through
`/tmp/aile-gpu.lock`, never forced, whispr untouched). PENDING — handed to the integration lane
(`twin-robot-in-store-boot-v1`), which owns GPU priority + commissioning + the host stack.
