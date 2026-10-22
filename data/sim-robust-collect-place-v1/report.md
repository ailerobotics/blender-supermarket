# Robust collect place-on-cart: bound the planner GPU + un-starve the station heartbeat (sim-robust-collect-place-v1)

Goal (captain, 2026-10-22): make the Isaac restock COLLECT stage's place-on-cart less prone to
failure. The real collect (pick the carton off the warehouse shelf, carry it to the rollwagen,
place it on the upper shelf) has never completed a recorded run; it dies **during place-on-cart**.

## 1. Root cause — one failure, two faces

Evidence: `~/datasets/twin-robot-in-store-boot-v1/mission/collectsrc/` (`host-stack.log`,
`bt_events.jsonl`, `mission.json`). GT collect (`RESTOCK_COLLECT_GROUND_TRUTH=1`), PR 187 transit
sensing pause, same profile as the earlier cold run.

Timeline (`bt_events.jsonl`): pick+weld (step 2) ✓, carry-drive to rollwagen (step 3) ✓, perceive
deck (step 4) ✓, **place (step 5)** begins `t=103.967` → `stage_failed t=108.417` reason
`station supervisor heartbeat unavailable: host stack went silent after last state=ready
(wall gap 30.1s)`.

`host-stack.log` shows what the place step actually did: a **tight cuMotion replan loop ~2 Hz**,
every attempt `PLANNING_FAILED` because the place goal c-space is self-colliding —

```
Invalid c-space position: self collision detected.
Failed call to 'planToCSpaceTarget()': target c-space position [[0.252 -1.839 1.964 -0.125 2.016 -4e-8]] is invalid.
```

— 13 `Received MoveGroup planning request` in the burst, then on the 13th:

```
terminate called after throwing an instance of 'cumotion::FatalException'
  what(): CUDA error: out of memory (/cumotion/src/cumotion/util/cuda_array.cu:66)
```

`component_container_mt` (cuMotion+nvblox) dies → the host launch's OnProcessExit guard SIGINTs the
whole stack → the host goes silent → the mission's station-supervisor wall guard trips at 30.1 s →
e-stop. The process deaths at the tail of `host-stack.log` are that teardown cascade, not the root.

**Both faces share one root: an unbounded replan of a deterministically-infeasible (self-colliding)
place goal.**
- The place pose IK landed on a self-colliding branch (ur_wrist_2-vs-forearm family — see the
  loaded-lift / transfer-preplace wrist-branch findings). Re-planning the **same** joint target
  with a fresh stochastic cuMotion seed cannot clear a self-colliding target; only a **different IK
  branch (next seed)** can. But the port burned its whole `REPLAN_MAX=12` same-goal budget on the
  unplannable target before it would ever reach the next seed.
- Each of those 12 plans re-requests the nvblox ESDF and allocates on-device. On the 12 GB card
  (the always-on `whispr` service holds ~2.2 GB, untouchable) the churn **OOMs cuMotion** (the
  earlier attempt died here too), and it **saturates the shared `get_esdf_and_gradient` service** so
  the host station supervisor (`incorporation_ack`) couldn't publish fresh acks → the 30 s gap.

The planner is **not** re-created per parked epoch here (`incorporation_ack` docstring: the
planner/mapper stack stays up, only the parked map is reset), so the OOM is per-plan churn, not
re-init — the "reuse rather than re-create" property already holds.

## 2. Fixes (both CPU-tested, all switches env-gated / backward compatible)

### 2a. Bound the planner GPU for this scene — `restock_manipulation/ros_adapter.py`

The compiled cuMotion ROS node exposes no trajopt-batch / collision-cache params (checked
`cumotion_src/.../cumotion_planner.cpp` — only visualization/world knobs), so the repo lever is the
**number of plan dispatches**, which is what drives on-device growth:

- **`PLAN_TOTAL_ATTEMPTS_CAP`** (env `RESTOCK_PLAN_ATTEMPTS_CAP`, default **10**): a hard count cap
  on cuMotion plan dispatches per move (re-plans + seed fallbacks + crash-retries all counted in
  `_send`). Below the ~13 OOM count with margin, far above any healthy move (plans in 1, legit
  stochastic retries 2–4). An unplannable goal now fails as a **clean bounded `planning failed`**
  before the card OOMs and before the ESDF service is starved. `PLAN_TOTAL_WALL_CAP_S` (22 s) was
  time-based and 12 plans tipped the card in ~6 s, well under it — a count cap is the direct bound.
- **`REPLAN_MAX_JOINT_SPACE`** (env `RESTOCK_REPLAN_MAX_JOINT_SPACE`, default **3**): tighter
  same-goal re-plan budget for a **joint-space** (cart PLACE / fixed-shape) goal, so a self-colliding
  joint target advances to the next IK seed/branch fast instead of hammering the GPU. Pose goals keep
  the full `REPLAN_MAX=12` (their rejections are genuinely stochastic seed-variant clips). This both
  bounds GPU churn **and gives the place a real chance** on an alternative branch — the 12-replan
  loop never even reached the next seed before.

Both only affect a move that is **already being rejected** (a healthy move plans in one), so healthy
runs are unchanged. Defaults keep the knobs active (captain: "less prone to failure in any case");
env overrides restore the old ceilings for a byte-identical comparison.

### 2b. Un-starve the station heartbeat during the press — `restock_host/incorporation_ack.py`

`tick()` now publishes the **station liveness ack BEFORE** its ESDF-service maintenance (graph
discovery + a fresh `get_esdf_and_gradient` request). Only a **done** ESDF result is consumed before
publishing (cheap, non-blocking, so `collision_time` stays current). A cuMotion replan storm
saturating that same shared service during a long press can no longer delay the heartbeat within a
tick. **Nothing is hidden**: readiness (`collision_time` fresh `> parked_since`) still advances only
through a real NVIDIA integration, and a genuinely dead process still publishes nothing →
`station process lost` / the wall-gap guard still fire.

This is the "keep the heartbeat alive during the press" half; 2a is the "bound the press" half — the
task's disjunction is satisfied on both sides.

## 3. CPU tests

- `tests/test_ros_adapter_branch.py` — `JointSpaceReplanBudget`, `PlanAttemptsCap`,
  `PressBoundUnderHeartbeat`, extended `ReplanRetry` (pose goal keeps full budget; `_send` counts
  only plan dispatches; press bounded under the 30 s heartbeat window).
- `tests/test_wall_gate_overrides.py` — `PlanCountCapOverrideTests` (defaults, env override, clamps).
- `tests/test_restock_host.py` — `IncorporationAckLivenessOrderingTest` (liveness without an
  integration, stale/contended ESDF keeps the heartbeat but not readiness, readiness only from a real
  integration, dead process still goes silent).

Full related-suite run (ros_adapter + host + station + manipulation + transfer-carry + wall sweeps):
all pass, no regressions.

## 4. Bounded GPU validation config (collect + couple) — hand to the integration lane

The mission runtime configs under `~/datasets` are owned by the integration lane and were not
edited; the new behaviour is on by default and the switches are env vars the launch sets. Validation
run = **collect + couple only**, GT collect allowed (same as `collectsrc`). Serialize the GPU through
`/tmp/aile-gpu.lock` (non-blocking, never forced); never touch `whispr`.

Env (same sim profile as `collectsrc`, plus the new caps — or leave the new caps unset to take the
on-by-default values):

```
RESTOCK_COLLECT_GROUND_TRUTH=1          # GT collect pick, as collectsrc
RESTOCK_HEARTBEAT_UNAVAILABLE_S=30      # unchanged sim profile
RESTOCK_PLAN_ATTEMPTS_CAP=10            # new: hard cuMotion plan-dispatch cap/move (default)
RESTOCK_REPLAN_MAX_JOINT_SPACE=3        # new: tighter joint-space same-goal replan (default)
# whispr margin: GPU guard HEAVY_MB=2500 opt-in only; never --force, never the hard Blender gate.
```

Expected: the place either (a) succeeds via an alternative IK branch now that the port reaches the
next seed fast, or (b) fails as a **clean bounded `planning failed (plan attempt cap)` /
`planning failed`** — never a cuMotion CUDA OOM and never a 30 s station-heartbeat silence. Report
PASS/FAIL from `bt_events.jsonl` (collect place step) + `host-stack.log` (no `FatalException`,
no 30 s gap).

Note: if the place goal is self-colliding for **every** IK branch, the place still cannot succeed —
that residual is the cart-place / wrist-branch IK pose problem tracked by the manipulation lane, not
this task. This task's contract is to make the place **fail safely and bounded** (no GPU crash, no
heartbeat silence) and to stop wasting the whole budget before trying other branches; a clean bounded
failure is strictly better than a lost GPU run.
