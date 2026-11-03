# sim-robust-pick-plan-rejection-v1 — trajectory host first-goal reject race

## Problem

In the Isaac restock mission under the **live_pose + arm-feed** collect config
(`RESTOCK_TRANSFER_MODE=live_pose`, `RESTOCK_TRANSFER_CARRY=weld,wrist_guard`, masked_depth arm
feed on, mast off), the collect stage has a **non-deterministic failure at the very first arm
move**. Run `seconddeath` (2026-10-23, head PR 228): the first collect plan was delivered
(`PLAN_DELIVERY elapsed=2.74`, `AWAIT_SETTLE_EXIT path=settled`), MoveIt validated it and sent it
to `arm_controller`, `arm_controller started execution` logged, and in the **same millisecond** the
FollowJointTrajectory action server replied `Goal request rejected` / `Goal was rejected by server`
(`host-stack.log:878-879`). MoveIt aborted; the mission raised
`MANIP_HOST_FAILED op=collect phase=failed reason='collision planning or execution failed'`, then
lost the arm-command heartbeat across `StageRetry`. The identical config completed collect minutes
earlier in `pacingproof`, so it is a **timing race**.

## Root cause (established from the logs)

The action server is this repo's own `scripts/twin/restock/trajectory_host.py`. Its `goal()`
callback returned `GoalResponse.REJECT` **silently** on three startup/timing conditions
(`fault`, `active`, `feedback is None`) and on a `Plan` `ValueError`.

Which condition fired in `seconddeath`, reconstructed from the timeline:

- **No `TRAJ_HOST_*` output at all** from the executor process (`[python3-5]` printed only its two
  startup `rcl` warnings). `fault()` **always** logs `TRAJ_HOST_FAULT` — none appeared ⇒ **not the
  fault branch**.
- It was the **first** collect goal (only one `sending trajectory to arm_controller`, line 876; the
  earlier `Goal accepted by server` at line 830 was move_group accepting the mission's
  *ExecuteTrajectory*, not an arm goal) with no prior accept ⇒ `active` cannot be stuck ⇒ **not the
  active branch**.
- A `Plan` `ValueError` requires `feedback` to be present, and whenever feedback is present the
  symmetric accept path **always** prints `TRAJ_HOST_PLAN` (as `pacingproof` shows at its line 877,
  with `TRAJ_HOST_START anchored ... feedback 0.867`). No such line ⇒ feedback was **not** present.

⇒ The reject was **`feedback is None`**: move_group forwarded the first goal in the wall-gap after
the executor's `/joint_states` subscription went ready but **before its first sample callback
fired**. Under render pacing, frames are seconds apart in WALL time (though <0.1 s in SIM), and the
`/joint_states` relay only starts stepping at the collect stage — so whether a frame reached the
executor's callback before move_group forwarded the goal is a **non-deterministic race**.
`pacingproof` won it (feedback at sim 0.867 before the goal); `seconddeath` lost it.

### Why the mission reason was misleading

`ros_adapter._poll()` `execute_result` raises `RuntimeError('execution failed')` when move_group
aborts; `poll()`'s `except` swallows the message, calls `stop()`, and returns `'failed'`;
`controller.py:1102` then hardcoded `'collision planning or execution failed'`. A rejected goal is
not a planning collision.

## Fix (PR into `fm/twin-robot-in-store-boot-v1`)

All default-on and backward compatible; the normal path (feedback already present, no fault) is
byte-identical. New pure, importable helpers carry the logic so each race is CPU-tested.

`scripts/twin/restock/trajectory_host.py`:

1. **Never silent** — `goal()` logs every reject as `TRAJ_HOST_GOAL_REJECT <reason> (fault=...
   active=... feedback=... grace=...)` (helper `reject_reason`), plus the `Plan` `ValueError` path.
2. **Wait for first feedback** — `goal()` waits a bounded, env-tunable WALL grace
   (`RESTOCK_FIRST_FEEDBACK_GRACE_S`, default **8.0 s**, clamped [0,60]; `first_feedback_grace_s`)
   for the first `/joint_states` sample before deciding, releasing the lock so the feedback callback
   can latch it. A goal that already has feedback breaks out on the first poll with **zero added
   delay** (byte-identical). `0` restores the legacy immediate decision (used to reproduce the bug).
3. **Clear a stale `active` flag** when the previous goal handle has terminated (`active_is_stale` +
   `_handle_live`; the handle is stored in `execute()`), so a new goal is not rejected on a dead flag.
4. **Op-start re-arm** — a `/restock_sim/arm_controller/reset` subscription clears a *latched* fault
   (+ a stale `active`) but **never** a live execution (`reset_clears`), consistent with PR 210's
   `ArmCommandGate.resume`. The host's permanent latch otherwise bricks the arm against the mission's
   own bounded retries. A no-op when there is no fault.

`scripts/twin/restock_manipulation/ros_adapter.py`:

5. Publishes the reset at every move begin (`_publish_reset`, mirroring `_publish_goal_tol`), so a
   latched fault from a previous op is re-armed before the next goal arrives.
6. `execute_result` records a **distinct** abort reason (`take_exec_abort_reason`) naming a controller
   goal-rejection/execution-fault and pointing at the host log, instead of the generic string.

`scripts/twin/restock_manipulation/controller.py`:

7. Surfaces the port's distinct reason when present, else the generic
   `'collision planning or execution failed'` (so existing collision behaviour is unchanged).

### CPU tests

`scripts/twin/tests/test_restock_trajectory_host.py` — `reject_reason` (each race + accept-when-clear),
`active_is_stale`, `reset_clears` (re-arm latched fault, never a live goal, no-op when clean),
`_handle_live` (None/duck-type/raising), `first_feedback_grace_s` (default + env + clamp).
`scripts/twin/tests/test_restock_manipulation.py` — a rejected/aborted goal surfaces the distinct
reason (not `collision ...`) and is consumed once; the existing collision test still yields the
generic string. Full affected-suite after rebase onto `fm/twin-robot-in-store-boot-v1` (PR 221/227/228
landed): **383 passed, 18 skipped** (merge kept both sides — the shoulder-drive `pick_surface` feed +
goal-miss classifier, the execution-death recovery, the sim-clock planning deadline, and this fix).

### GPU proof (for the integration lane)

`docs/examples/pick-plan-rejection-proof-fragment.json`. Re-run the `seconddeath` config collect-only
(`--stop-after collect`) at the fixed head. **PASS** = first goal accepted (`TRAJ_HOST_PLAN` +
`Goal request accepted!`, never `rejected`), collect completes, and any reject is logged by reason.
One GPU run at a time via flock `/tmp/aile-gpu.lock`, never forced, whispr untouched. **PENDING** —
handed to the integration lane; the validator may hold the GPU.

## Separate latent defect (reported, NOT fixed here)

Every `move_group` start logs, for each UR link:
`Error retrieving file [/opt/ros/jazzy/share/ur_description/meshes/ur10/collision/*.stl]: URL
rejected: No host part in the URL` → `Link ur_* has visual geometry but no collision geometry`
(`seconddeath host-stack.log:45-60`). Cause: the resolved URDF (`isaac_bootstrap.resolve_meshes`,
line 32) writes **bare absolute** mesh paths (`/opt/ros/.../collision/base.stl`), and
`resource_retriever` (move_group's FCL collision loader) rejects a path with no `file://` scheme.
Visual `.dae` meshes are not retrieved by move_group, so only collision `.stl` fail. cuMotion's XRDF
spheres do the real arm collision checking, so planning still works.

**Why not fixed here:** the resolved URDF feeds **two** consumers — Isaac's URDF importer *and*
move_group's `robot_description`. Prefixing `file://` risks Isaac's importer, and — more importantly
— giving move_group real arm collision geometry would make its FCL `ValidateSolution`
PlanningResponseAdapter (which **runs** in the log) start checking self-collision, which could
**reject previously-accepted cuMotion plans**. That is a behavioural change only a GPU `move_group`
run can validate, so a blind one-line "fix" is unsafe. Recommended fix for the owning lane: a
launch-side transform of the `robot_description` string handed to move_group (prefix `file://` on
bare absolute mesh URIs), leaving the on-disk Isaac URDF untouched, then GPU-validate that
`ValidateSolution` does not regress planning.

## Files

- `scripts/twin/restock/trajectory_host.py`
- `scripts/twin/restock_manipulation/ros_adapter.py`
- `scripts/twin/restock_manipulation/controller.py`
- `scripts/twin/tests/test_restock_trajectory_host.py`, `test_restock_manipulation.py`
- `docs/examples/pick-plan-rejection-proof-fragment.json`

Evidence: `~/firstmate/data/twin-robot-in-store-boot-v1/proofs-pacing-and-seconddeath.md`;
runs `~/datasets/twin-robot-in-store-boot-v1/mission/{seconddeath,pacingproof}/`.
