# sim-robust-pick-wrist-guard-cspace-fix-v1 — PR254's guarded pick approach sends cuMotion a malformed c-space target

**Branch:** `fm/sim-robust-pick-wrist-guard-cspace-fix-v1` (off `fm/twin-robot-in-store-boot-v1` @ `8cd0411`,
i.e. after PR254 `#254` and PR255 `#255`).
**Symptom run:** pickguard1 (`~/datasets/twin-robot-in-store-boot-v1/mission/pickguard1`), staged transfer-only
from `recorded_bay_park`, fresh commission, HOST `transfer_pick_wrist_guard=true`, no injection, head `44c8c33`
(= PR254). `mission.json state=failed reason='arm command heartbeat lost'`, peak VRAM 11507 MiB.
Evidence: `~/firstmate/data/twin-robot-in-store-boot-v1/pickguard1-evidence.md`, that run's `host-stack.log`.

PR254's **intent held**: the coldwin12 pick-contact wrist limit cycle did NOT recur (`pose_not_bounded`=0,
`MOVE_RETRY phase=contact`=0). But its approach wiring sends cuMotion a malformed goal, which this PR fixes.

## 1. Mechanism (established on CPU, before any code change)

### 1.1 The malformed request
PR254 guards the **live_pose transfer PICK approach** (a cuMotion task-space pose goal) by passing
`wrist1_cap=0.65, wrist2_min=0.0` into `ros_adapter.RosMotionPort._pose_plan_goal`
(`controller.py` `_pick_wrist_guard` → `begin(..., wrist1_cap, wrist2_min)` → the pose-goal branch
`begin()` → `_pose_plan_goal`). That method builds a single `moveit_msgs/Constraints` carrying BOTH

* a `PositionConstraint` + `OrientationConstraint` (the TCP pose goal), AND
* two `JointConstraint`s — `ur_wrist_1_joint` and `ur_wrist_2_joint` — whose `position` is the band midpoint:
  * wrist_1: `(_low + cap)/2 = (-2.0 + 0.65)/2 = -0.675`
  * wrist_2: `(w2min + _high)/2 = (0.0 + 2.0)/2 = 1.0`

### 1.2 How the Isaac ROS cuMotion MoveIt planner treats it
The NVIDIA Isaac ROS cuMotion planner (`isaac_ros_cumotion` 4.4, runtime overlay
`~/datasets/sim-restock-host-v2/runtime-closure/overlay/jazzy`) interprets a `MotionPlanRequest`'s
`goal_constraints` by goal TYPE, not as a merged bounded-IK problem: **when `joint_constraints` is present it
takes the c-space (joint-target) path and builds `cspace_position` from exactly those `joint_constraints`
entries**, discarding the position/orientation constraints. Here that list has **2** entries, so it calls
`planToCSpaceTarget([[-0.675, 1]])` against a 6-DOF robot and rejects it:

```
Invalid c-space position: Number of c-space coordinates in 'cspace_position' [2] must equal the number of
  c-space coordinates of the robot [6].
Failed call to 'planToCSpaceTarget()': target c-space position [[-0.675      1]] is invalid.     (×7)
terminate called after throwing an instance of 'cumotion::FatalException'  what(): CUDA error: out of memory
  (/cumotion/src/cumotion/util/cuda_array.cu:66)
[ERROR] [component_container_mt-2]: process has died [pid 2178715, exit code -6, ...]
```

The rejected pair `[-0.675, 1]` is **exactly** the two band-midpoints above (confirmed arithmetically on CPU),
so the planner provably built `cspace_position` from the joint_constraints list. This confirms the plugin
behaviour empirically; the evidence is conclusive even though the goal-type decision lives in the compiled
planner (`libcumotion.so.1.1.0` `CSpacePathSpec`/`CreateCSpacePathSpec`; the MoveIt plugin
`libisaac_ros_cumotion_moveit.so` is only the MoveGroup client wrapper).

**Corollary — a full 6-joint list would be a PURE joint goal, discarding the pose.** The repo's own
seeded-KDL→joint-space plan path (`ros_adapter.py` ~L1850) proves the well-formed shape: it appends a
`JointConstraint` for **all six** joints and cuMotion accepts it as a valid c-space target. So candidate (b)
(joint_constraints as *bounds* on a pose IK) is **not available** — the plugin has no bounded-pose-IK mode;
any `joint_constraints` means a full c-space goal and the pose is dropped. A 6-entry list would reach a fixed
joint config, not the live carton TCP the approach must reach. Therefore the approach must send a **clean
pose goal (no joint_constraints)** and the wrist family must be enforced some other way.

### 1.3 Uniqueness and the same latent encoding elsewhere
`grep -c 'Invalid c-space position'` = 0 across coldwin11/12/13, meshuris0, armcameco, hold1, rock1; the line
appears ONLY with `transfer_pick_wrist_guard=true`. It is a PR254 regression.

The identical encoding also sits on the **fm327/fm403 task-space LIFT fallback** (`controller.py` ~L1391:
`carry_cap = self.carry_wrist1_cap if self.phase == 'lift'`, `carry_w2min = self.carry_wrist2_min ...`),
which flows into the same `_pose_plan_goal` whenever `carry_lift_shape` is unset. It is rarely hit (the lift
normally takes the `carry_lift_shape` joint goal), so the pick approach was the first path to exercise the
bug in anger — but it is the same 2-coordinate c-space target and is fixed here too.

### 1.4 Did the 7 rejected calls leak device memory?
Plausibly yes. Each `planToCSpaceTarget` was re-dispatched by the adapter's plan-rejection re-plan loop
(a rejected plan is a planning failure → re-plan the same `_plan_goal`). Seven rejections burst within ~1 s
(all between the same ESDF write and the FatalException) and then CUDA OOM'd at a peak of 11507 MiB — below
the 12 GB hard ceiling and with no successful plan freeing memory in between. That pattern (bounded repeat →
abrupt OOM, no large concurrent allocation) is more consistent with a **per-rejected-call device-memory leak**
(buffers allocated for the c-space path/solver before the length check throws, not freed on the error path)
than with the full-run VRAM ceiling; it is not certain. Either way the fix removes the malformed call
entirely, and the re-plan budget is already hard-capped at `PLAN_TOTAL_ATTEMPTS_CAP=10` per move.

## 2. Fix (below the unchanged opt-in key; byte-identical with the key off)

Chosen candidate: **(a)** — plan a CLEAN pose goal (no joint_constraints) and validate the planned
trajectory's terminal wrist configuration, re-planning (cuMotion's stochastic fresh seed draws a different IK
family, the same mechanism the existing collision re-plan relies on) when it lands on the self-colliding
family, bounded and failing by a named reason when exhausted. All in `ros_adapter.py`; the controller wiring
is unchanged (it still passes `wrist1_cap`/`wrist2_min`, now interpreted as a post-plan guard, not a goal
constraint).

1. `_pose_plan_goal` no longer appends any `joint_constraints`; it returns a clean position+orientation pose
   goal. When a wrist cap/min is supplied, `begin()` records it on `self._pose_wrist_guard`.
2. The `plan_result` SUCCESS path remaps the planned trajectory's terminal waypoint to ARM order and checks
   `transfer_carry.in_clear_region(q, cap, w2min)`. If it is in the clear family → accept (unchanged flow). If
   it is in the ur_wrist_2-vs-ur_forearm self-colliding family → log `POSE_WRIST_GUARD reject`, re-plan the
   same clean pose goal (fresh stochastic seed) up to a bounded budget; when exhausted, raise the named
   `pose goal could not reach the clear carry wrist region (wrist_1<=…, wrist_2>=…)`. Each re-plan is a plan
   dispatch, so the existing `PLAN_TOTAL_ATTEMPTS_CAP=10` GPU backstop still bounds the total.
3. Same clean-pose-goal path covers the fm327/fm403 LIFT fallback (it shares `_pose_plan_goal`), so it can no
   longer emit a 2-coordinate target either.

The straight-line CONTACT descent is already well-formed (a seeded-KDL → full 6-joint c-space target guarded
by `_guard_rejects`); it is unchanged.

## 3. Name the host-container death (step 3)
`restock_host/host_guard_exit.py` (pure, CPU-tested) builds a named reason from the dying process, its exit
code and its last error line, and bridges it across the process boundary via a small marker file
(`RESTOCK_HOST_GUARD_MARKER`). `launch.py`'s shutdown-guard `OnProcessExit` now captures the guarded
process's name + returncode (and, via an `OnProcessIO` last-stderr-line ring, its last error line — here
`cumotion::FatalException … CUDA error: out of memory`), writes the marker, and emits a `Shutdown` whose
reason names it. `trajectory_host.main()`'s last-resort `finally` reads the marker and faults with that named
reason instead of the bare `executor stopped`. No behaviour change otherwise (the stack still tears down; the
marker is absent on a clean run).

## 4. Bounded GPU proof (handed to the integration lane; not run here)
`docs/examples/pick-wrist-guard-cspace-fix-mission-fragment.json` — the one HOST key. PASS = the staged
transfer-only run (`recorded_bay_park`, fresh commission) with `transfer_pick_wrist_guard=true` shows **zero
`Invalid c-space position` lines**, the approach plan lands on the clear family (`POSE_WRIST_GUARD accept`
with terminal wrist_2 ≥ 0, no `pose goal could not reach the clear carry wrist region` fail), the pick contact
settles, transfer reaches verify, no container death; then the full cold capstone with the key on. GPU rules:
one run at a time under `flock /tmp/aile-gpu.lock`, never forced, `HEAVY_MB=2500` for the always-on whispr
(~2.2 GB), never touch whispr.

## 5. CPU tests
`scripts/twin/tests/test_restock_pick_wrist_guard_cspace.py`:
- the guarded approach goal (key on) carries **no** `joint_constraints` (and `in_clear_region` would reject a
  terminal config in the coldwin12 colliding family, accept the coldwin11 clear one);
- the lift-fallback goal is well-formed (no short joint_constraints list);
- key off / collect / non-live_pose ⇒ byte-identical (`_pose_plan_goal` emits the same clean goal it always
  would, with no wrist guard recorded);
- `host_guard_exit` reason/marker round-trips and `trajectory_host` names the container death.
