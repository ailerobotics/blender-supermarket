# Scripted transfer, arm=executor: the arm visibly moves and completes (sim-video-executor-arm-v1)

Goal: the scripted transfer with `RESTOCK_TRANSFER_MODE=scripted` + `RESTOCK_SCRIPTED_TRANSFER_ARM=executor`
completes with the arm **visibly moving** (not the frozen `=still` arm the captain saw), proven offline
(CPU). The integration lane runs the GPU test. Switch keys are unchanged.

## 1. Evidence — what actually happened in warm5 (not a plain rejection)

Source: `~/datasets/twin-robot-in-store-boot-v1/mission/warm5.console.log`,
`warm5/host-stack.log`, `warm5/scripted-transfer/scripted-arm-pick.log`.

The brief's working hypothesis was that the host **rejected** the 81-point pick. The telemetry says
otherwise — the first pick was **accepted and executed**, then **faulted on a physical self-collision**:

1. `SCRIPTED_TRANSFER executor pick sent: 81 points 15.7s` → `host-stack.log`:
   `TRAJ_HOST_PLAN … npoints=81` (logged **only on `GoalResponse.ACCEPT`**, trajectory_host.py:402) — so
   the goal was **accepted**, not rejected.
2. `TRAJ_HOST_TRACK_MISS … worst_j=3 … meas=0.0122 ref=1.3194` ×3 → `TRAJ_HOST_FAULT trajectory tracking
   error 1.307 exceeds 1.3 rad`. Joint 3 is `ur_wrist_1`: the reference ramps to +1.32 rad while the
   measured joint is **stuck at 0.0122** — the arm physically could not follow it. The end-of-run
   `SEAL_DIAG … hit_bodies=['/MobileRobot/ur_upper_arm_link/…']` confirms the gripper was jammed against
   the arm. **The pick trajectory drove the arm into its own body (wrist-vs-forearm self-collision).**
3. The fault latches `state['fault']` in trajectory_host: `tick()` then **stops publishing**
   `arm_joint_command` (trajectory_host.py:298-300) and `goal()` **rejects every later goal**
   (`if state['fault'] … REJECT`, :375).
4. A retry `begin` sends a 58-point pick → `goal not accepted by the executor` (the latched fault) →
   the stage degraded silently to `still` → with the host stream dead since the fault, the Isaac
   arm-command heartbeat starved: `RuntimeError: arm command heartbeat lost`
   (native_transport.py:165-167) → mission failed.

**Why it was "sent twice":** the first (81-pt) pick was accepted and faulted; the second (58-pt) pick was
rejected because that fault had e-stopped the executor. The heartbeat death is a **downstream consequence
of the self-collision fault**, not of the fallback itself.

### Root-cause correction to the task premise
The scripted transfer has its **own** damped-least-squares IK (`scripted_transfer.ArmIK`), independent of
the live-pose cuMotion path and its `RESTOCK_TRANSFER_CARRY=wrist_guard` (PR 170). That IK seeded the
**self-colliding wrist branch** for the PICK half. The place half already resolved to the clear branch;
only the pick/carry half jammed.

## 2. The fix (offline, CPU; `scripts/twin/restock/scripted_transfer.py`)

**(a) Wrist-guard the scripted IK — default on.** `plan_transfer(..., wrist_guard=True)` makes every seeded
keyframe land in the bench-proven CLEAR wrist region (`wrist_1 <= 0.65`, `wrist_2 >= 0`; the same
`transfer_carry.in_clear_region` / `wrist_flip_seed` predicate the live-pose guard uses), re-seeding from
the wrist-flipped twin when the nearest branch is not clear. The clear branch is established at the
non-line `reach_high` keyframe and the continuation IK carries the rest of the pick onto it.
**TCP poses are unchanged**, so the `still`-mode carton path is byte-identical — only the arm's joint
branch changes.

**(b) No silent mid-transfer fallback; fail fast + keep-alive.** In executor mode a host **rejection**,
**abort**, pre-send **start-mismatch**, pre-send **infeasibility** or **settle-timeout** now call
`_fail_executor(...)`: it fails the stage FAST with a **named reason** and issues a keep-alive **hold of
the current joints** through the executor (`ExecutorArm.hold`), instead of degrading to the silent `still`
path that — once the executor has e-stopped — starves the heartbeat into the misleading
`arm command heartbeat lost` crash. Failing fast lets the mission report the true cause and tear down
before the wall-backstop watchdog can trip. (A genuine host fault is a terminal safety e-stop no caller can
revive; the wrist guard prevents that fault in the first place. The begin-time "no IK solution for every
keyframe" case keeps the labelled `still` fallback — a planning-time decision, not a mid-transfer degrade.)

The executor trajectory already starts from the live joint state (begin() plans from
`arm_joint_positions()`; the send re-seeds `points[0]` to the measured state, fm357) and is already timed
inside the host's speed/acceleration ceilings (`hermite_violation`) — both retained.

## 3. Proof (CPU, no GPU/Isaac/ROS)

`scripts/twin/tests/test_restock_scripted_transfer.py` (run with
`PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=`): **21 passed, 1 skipped** (the skip needs `usd-core`).

- `test_wrist_guard_puts_the_pick_on_the_clear_branch_without_moving_the_tcp` (committed, always runs):
  the guarded plan keeps every moving keyframe in the clear region; the unguarded pick half does not; the
  keyframe TCPs and `choice` are identical.
- `test_guard_is_self_collision_clear_where_the_unguarded_pick_penetrates` (dataset-gated): on the real
  asset URDF FK + XRDF collision spheres, the unguarded pick's min self-collision gap is **negative**
  (interpenetration, gripper vs forearm) and the guarded plan's is **positive** (clear).
- `test_warm5_recorded_pick_was_host_accepted_and_physically_self_collided` (dataset-gated): replays
  warm5's recorded 81-pt pick through the host validation path — `validate_trajectory` accepts it and the
  log shows `TRAJ_HOST_PLAN` (accepted), yet the sphere model shows it self-collides. Confirms §1.
- `test_a_rejected_goal_fails_fast_and_keeps_the_arm_held`,
  `test_an_aborted_goal_fails_the_stage_honestly_and_holds`,
  `test_an_infeasible_segment_fails_fast_not_silent_still`: the new fail-fast + keep-alive behaviour.

Measured sphere-model clearances (fixture geometry = run-356 transfer park; URDF + `host-cad/mobile.xrdf`):

| plan      | min self-collision gap over keyframes |
|-----------|----------------------------------------|
| unguarded | **−0.054 m** (pick half interpenetrates) |
| guarded   | **+0.015 m** (clear)                   |

STOW→reach_high transit (guarded): **+0.015 m**. Margins are thin (the sphere model is conservative); the
integration lane's GPU run is the physical confirmation.

Related suites unaffected: `test_restock_transfer_carry.py` + `test_restock_transfer_live_pose.py` +
`test_restock_manipulation.py` = **154 passed, 17 skipped**.

## 4. How to run it (switch keys unchanged)

`RESTOCK_TRANSFER_MODE=scripted` + `RESTOCK_SCRIPTED_TRANSFER_ARM=executor`
(optionally `RESTOCK_TRANSFER_CARRY=wrist_guard,weld` for the live-pose path labels). The wrist guard is
on by default in `plan_transfer`; nothing new to set.

## 5. Limits / for the integration lane (GPU)

- The self-collision model is the integration lane's conservative sphere set; offline it proves the
  guarded branch is clear where the unguarded one penetrates, but the physical completion (and the thin
  ~15 mm margin) must be confirmed on the GPU run. Do not take the GPU here.
- Once the wrist guard is proven live, the preferred transfer remains the real `live_pose` path
  (PR 160/170); this scripted-executor path is the guaranteed-moving arm for the first walkthrough video.
