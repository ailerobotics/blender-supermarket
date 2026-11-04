# sim-preplace-accel-limit-reject-v1

Time-parametrise every arm plan against the simulation executor's interpolated-acceleration ceiling
**before** submission, so an unlevelled pre-place plan is slowed to fit (or refused by name at plan
time) instead of being rejected at the controller as a host fault with a lost heartbeat.

Branch `fm/sim-preplace-accel-limit-reject-v1` off `fm/twin-robot-in-store-boot-v1`. CPU-proved.

## 1. The bug, reproduced from warmacc4

warmacc4 (PR 299 warm two-run, collect-only, HOST `carton_level` **OFF**) welded the carton cleanly
and then, at the collect place **pre-place** step, the arm controller rejected the plan:

```
host-stack.log:7306 [python3-5] TRAJ_HOST_GOAL_REJECT plan rejected at goal: interpolated acceleration limit violation
host-stack.log:7313 [python3-4] PORT_FAULT_ESTOP state=execute_result exc=RuntimeError('execution failed')
host-stack.log:7314 [python3-4] HOST_FAULT_ESTOP phase=preplace reason='arm controller aborted trajectory (goal rejected or execution fault, moveit_err=-4; see host-stack.log TRAJ_HOST_GOAL_REJECT/TRAJ_HOST_FAULT)'
```

The mission's terminal reason in `mission.json` was the *downstream* symptom, not the cause:
`RuntimeError: arm command heartbeat lost` (native_transport.py watchdog) — the host stopped streaming
arm commands after the e-stop, so the heartbeat tripped and masked the real trip.

### The dispatch path (carton_level OFF vs ON)

The collect place pre-place is issued by `TransferController._issue_level_approach_move`
(`controller.py:1113-1127`):

* **carton_level ON (coldwin33):** `_level_approach_path_plan` returns a leveled `LevelPathPlan`, dispatched
  via `motion.begin_joint_path(...)` — an **explicit collision-off 12-/9-waypoint rest-to-rest path**
  (`ros_adapter.py:_waypoints_goal`, each segment sized to 0.08 rad/s / 0.15 rad/s² peak). No cuMotion,
  no MoveIt time-parametrisation. It also carries the PR 296 leveled-plan gate.
* **carton_level OFF (warmacc4):** `_level_approach_path_plan` short-circuits to `None`, so the move
  falls through to `motion.begin_joint_goal(leveled_goal=collect_place_preplace_shape, slow=True,
  goal_tol=0.2)` → a **cuMotion `MoveGroup` joint-space plan** (`ros_adapter.py:1245-1265`,
  `max_velocity_scaling_factor = velocity_scale*0.35`, `max_acceleration_scaling_factor = 0.02`),
  time-parametrised by MoveIt's `AddTimeOptimalParameterization` (TOTG) against `joint_limits.yaml`
  (3.0 rad/s² × 0.02 = **0.06 rad/s²**), then handed to the executor via `ExecuteTrajectory`.

The two log lines confirming it (warmacc4):

```
[component_container_mt-2] cumotion_planner: Using joint space goal ... MoveGroup planning succeeded in 1.267 seconds
[move_group-1]            Calling PlanningResponseAdapter 'AddTimeOptimalParameterization'
```

### Why the rejection: a resampling vs cubic-Hermite mismatch at the executor ceiling

The simulation FollowJointTrajectory executor (`trajectory_host.Plan`) does **not** trust the TOTG
timing. It rebuilds **every 0.1 s segment as a cubic Hermite** from the `(position, velocity)` pair and
checks the per-joint peak against `max_speed·(0.5/0.15)`. The host launches `trajectory_host` with the
default `--max-speed 0.3` (no override in `restock_host/launch.py`), so the ceiling is **1.0 rad/s²**.

A TOTG plan is timed against MoveIt's 0.06 rad/s² budget, but it is resampled onto a uniform 0.1 s grid
with a **sub-0.1 s terminal remainder**, and the executor's cubic-Hermite reconstruction of those
resampled `(position, velocity)` pairs sits **right at the 1.0 rad/s² ceiling** — the TOTG and the
cubic-Hermite representations disagree most at the coarse-sampled terminal remainder.

CPU reproduction from the recorded plans (`data/.../scratch/`, fixture
`scripts/twin/tests/fixtures/preplace_accel_limit_plans.json`), scored with the executor's own
cubic-Hermite reconstruction:

| plan (real, from host-stack.log) | config | peak cubic-Hermite accel | worst joint | executor verdict |
|---|---|---|---|---|
| warmacc4 lift, 157 pts, dur 15.51 s (accepted) | OFF (cuMotion+TOTG) | **0.936 rad/s²** (terminal remainder dt=0.011 s) | ur_wrist_2 | ACCEPT (no margin) |
| warmacc4 pre-place analog (lift, terminal remainder 0.011→0.006 s) | OFF | **1.887 rad/s²** | ur_wrist_2 | **REJECT: interpolated acceleration limit violation** |
| coldwin33 leveled pre-place, 9-pt `begin_joint_path`, dur 49.6 s | ON (explicit) | **0.116 rad/s²** (7× under) | ur_shoulder_pan | ACCEPT |

The recorded carton_level-OFF cuMotion+TOTG plans run with essentially **no margin** (0.936 of 1.0).
The pre-place is a larger carried move than the lift (joint deltas up to 1.25 rad), so its TOTG stop
landed a smaller terminal remainder off the grid and the executor's cubic-Hermite reconstruction of
that remainder segment crossed 1.0 on `ur_wrist_2` — the warmacc4 rejection, **speed stays in-profile
(0.270 < 0.3 rad/s), so it is an acceleration violation**. The leveled path never goes through
cuMotion/TOTG, so it is 7× under the ceiling and passes. Nothing upstream of the executor checked the
cubic-Hermite acceleration, so it surfaced only as the host fault.

## 2. The fix — one pre-submission gate for every arm plan

New pure module `scripts/twin/restock_manipulation/traj_limits.py` states the cubic-Hermite per-segment
extrema **once** (identical to `trajectory_host.Plan`; the `MAX_ACCEL_PER_SPEED = 0.5/0.15` ratio and the
`DEFAULT_EXEC_MAX_SPEED = 0.3` profile are now shared between the executor and the gate), plus:

* `trajectory_accel_peak(times, positions, velocities)` → the worst `(peak, joint, segment)`.
* `retime_to_accel_limit(...)` → a **uniform time reparameterisation** `t → s·t`, `v → v/s` (`s ≥ 1`)
  that brings every segment's interpolated acceleration within the executor ceiling. The geometric joint
  **path is preserved exactly** (positions untouched), acceleration scales `1/s²`, speed `1/s` (never
  worse), and the start-at-rest / end-at-rest / monotonic-time invariants hold. A plan already within the
  limit is returned **unchanged** (`s == 1` → byte-identical). A plan whose required stretch would push
  the duration past the bounded execution budget (`EXEC_DURATION_BUDGET_S = 110 s`) raises
  `PlanLimitError('plan exceeds acceleration limit: <joint> <value> > <limit>')`.

`RosMotionPort` (`ros_adapter.py`) gains `exec_max_speed` (default `DEFAULT_EXEC_MAX_SPEED`, matching the
executor's launch default) → `self._exec_max_accel = exec_max_speed · MAX_ACCEL_PER_SPEED`, and
`_retime_plan_for_exec_limits(trajectory)` applied at **both** dispatch seams:

* the cuMotion+TOTG joint/pose plans — `plan_result` in `_poll`, after `validate_trajectory` and the
  wrist guards, before the `ExecuteTrajectory` dispatch; a `PlanLimitError` sets `_exec_abort_reason` and
  fails the move with the named reason **before any arm command**;
* the explicit collision-off waypoint paths — `_begin_execute_trajectory` (the leveled `begin_joint_path`
  and the straight line); a no-op there (already under the ceiling), but it keeps **one gate for every
  submitted plan** — the same gate for levelled and unlevelled paths, as asked.

The executor's limit is **not raised** — the fix slows the plan to the existing limit, or refuses it.

A retimed plan logs `PLAN_RETIME_FOR_EXEC scale=… peak_accel X->Y (limit 1.000) …`; a refusal names the
joint and its interpolated acceleration.

## 3. A goal rejection is a named trip, heartbeat left alive (does not duplicate PR 316)

With the gate, a legitimate over-ceiling plan is retimed and never reaches the executor. A plan too
aggressive to fit is refused at plan time with `plan exceeds acceleration limit: …` — a clean,
**named** mission failure before any arm command.

Either way the terminal reason must survive the controlled stop's deliberate arm hold (which produces a
downstream `arm command heartbeat lost`). PR 316's controlled stop (`restock.controlled_stop`,
`isaac_port.enter_welded_controlled_stop` → `command_gate.suppress_heartbeat = True`) already holds the
base planted + arm held + keeps the heartbeat **alive** (holds, does not raise) while a carton is welded
— which the pre-place is. This lane only ensures the real cause **outranks** that symptom: both the
controller goal-rejection reason (`arm controller aborted trajectory … moveit_err=-4 …`) and the
plan-limit refusal are added to `controlled_stop.TRIP_MARKERS`, so `prefer_first_trip` keeps them over
`arm command heartbeat lost`. The controlled-stop mechanism itself is unchanged.

## 4. CPU proof

`scripts/twin/tests/test_restock_preplace_accel_limit.py` (11 tests, fixtures are **real recorded
plans**):

* the shared `segment_accel_peaks` equals the inline `trajectory_host.Plan` formula (2000 random segments);
* the recorded warmacc4 cuMotion plan sits at the ceiling (0.936) and is accepted;
* the warmacc4 **pre-place** analog is REJECTED by the real `trajectory_host.Plan` with exactly
  `interpolated acceleration limit violation`, then **passes the real Plan after `retime_to_accel_limit`**
  (positions preserved, speed never raised);
* the coldwin33 **leveled** plan passes and retiming is a **no-op** (`scale == 1`, byte-identical);
* a plan too aggressive to fit the budget **refuses** with `plan exceeds acceleration limit: <joint>
  <value> > <limit>`;
* the `RosMotionPort._retime_plan_for_exec_limits` ROS-msg wrapper retimes / refuses / is byte-identical;
* the goal-rejection and plan-limit reasons are preserved named trips over the heartbeat symptom
  (`controlled_stop`), verifying the heartbeat-alive path.

Regression: `test_restock_trajectory_host.py` (48), `test_sim_host_fault_controlled_stop.py`,
`test_ros_adapter_branch.py`, `test_restock_runtime.py` (183+1 skip) all pass unchanged.

Run: `env -u PYTHONPATH PYTHONPATH=scripts PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest
scripts/twin/tests/test_restock_preplace_accel_limit.py -q`.

## 5. Hand-off to the integration lane

The gate is **always-on** (no config key — a pure submission-path safety net). A run whose plans are
already within the ceiling is byte-identical apart from the (absent) `PLAN_RETIME_FOR_EXEC` line.

Bounded GPU acceptance = cold collect-only in BOTH configs, asserting the pre-place is accepted with no
goal rejection: `docs/examples/preplace-accel-limit-reject-fragment.json`.

### Follow-ups (out of scope here)
* The set-down **level** at phase=place (coldwin33 CARTON_TILT 2.06; separate lane).
* The warm re-seat cart pose (separate lane).
* Latent: `restock_host/launch.py` launches `trajectory_host` with the default `--max-speed`; the port's
  `exec_max_speed` defaults to the same `DEFAULT_EXEC_MAX_SPEED`, so they match in sim. If a future run
  overrides the executor `--max-speed`, plumb the same value to the port's `exec_max_speed` so the gate
  tracks the executor.
