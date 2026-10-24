# sim-robust-transfer-preplace-reach-v1 — live_pose transfer preplace reaches a holdable goal from the full-chain park

**Status:** CPU fix + tests landed; one bounded GPU proof config handed to the integration lane (the full
capstone is the real proof — the staged start cannot reproduce the full-chain park pose). PR into
`fm/twin-robot-in-store-boot-v1`.

## The blocker (coldwin6, head 42e154c / PR #242)

The every-stage-honest live_pose cycle cleared its entire mid-chain in one run
(`~/datasets/twin-robot-in-store-boot-v1/mission/coldwin6`, 2026-10-24): collect ✓ (zero planner deaths),
couple ✓, outbound ✓ (PR 241 seat-independent arrival), parked station never tripped (PR 240), base parked
at the transfer bay. The transfer then failed at **phase=preplace**:

```
WRIST_GUARD reject q=[-1.047, -3.404, 2.383, 1.02, -1.128, 0.0] (w1<=0.65, w2>=0.00) -> wrist-flipped re-seed [...] (1/3)   # host-stack.log:26393
WRIST_GUARD accept q=[-1.047, -4.003, 2.62, -1.759, 1.128, -3.142] (w1<=0.65, w2>=0.00) seed=1 flips=1                      # :26394
TRAJ_HOST_GOAL_MISS pos_err=0.3531 per_joint_err=[~0, 0.3531, ~0, ~0, ~0, ~0]
  meas_q=[-1.0473, -3.65, ...]  goal_q=[-1.0473, -4.0031, ...]                                                             # :31700
TRAJ_HOST_FAULT pick physically blocked: ur_shoulder_lift_joint frozen 0.3531 rad short of goal ...                        # :31701
move_group: PATH_TOLERANCE_VIOLATED -> PORT_FAULT_ESTOP -> HOST_FAULT_ESTOP phase=preplace                                 # :31715
```

## Root cause — a BRANCH-SELECTION bug, not a reach limit

Diagnosed on CPU with the repo's own IK (`twin.restock.scripted_transfer.ArmIK` on the run's
`mobile-import.urdf`, the same tool the shoulder-drive / cart-place-goal lanes used). The authoritative
numbers (all reproduced by `scripts/twin/tests/test_restock_transfer_preplace_reach.py`):

- **The two accepted goals are the SAME preplace pose.** FK of the rejected KDL branch (ur_shoulder_lift
  −3.404) and the accepted flipped branch (−4.003) land at the identical TCP (position diff 0.3 mm, rotation
  0.6 mrad): `pos=(-0.033, 0.698, 0.340)` in `ur_base_link`, reach **0.777 m** — well inside the UR10's
  ~1.3 m. **The preplace is not out of reach.**
- **A HOLDABLE, wrist-clear branch EXISTS at the coldwin6 park pose.** A branch search over the preplace
  pose found a clean solution at **ur_shoulder_lift −1.953** (`fold_droop` 0.382), wrist_1 −0.43 ≤ 0.65,
  wrist_2 0.99 ≥ 0 — *the same upright family the collect place-on-cart lane targets*. Seeding `ArmIK` from
  a generic upright config converges straight to it.
- **The host chose the deep fold.** KDL's first (measured) seed resolved to the deep branch (−3.404,
  wrist NOT clear); the wrist-guard's wrist-flip re-seed — which is inexact for the real 0.116 m wrist
  offset — then re-converged to the deep-fold **clear** twin (−4.003, `fold_droop` 3.851). The plain
  wrist-guard accepts the first *wrist-clear* solution and never checks the fold, so it streamed a goal the
  arm physically cannot reach (it jams ~−3.65 under its own geometry).
- **The staged runs (`armcam4`/`armcameco`/`htsim`) got lucky, not fixed.** Their recorded_bay_park parks
  the base ~9 cm / 0.08 rad away; there KDL's measured seed lands directly in the upright branch
  (`WRIST_GUARD accept ... ur_shoulder_lift −1.735 seed=0 flips=0`). The full-chain park's slightly
  different base→bay geometry is the only difference — the staged start cannot reproduce it.

So the park pose is fine; the seeded-KDL branch selection is the bug.

## The fix — reach-guard the preplace seeder (opt-in, default-off ⇒ byte-identical)

A new HOST-config key `transfer_preplace_reach_guard` (`transfer_carry.resolve_reach_guard`) extends the
existing wrist-guard with a **holdable-fold** acceptance and an **upright seed**, scoped to the live_pose
transfer PREPLACE only (`controller._preplace_reach_guard`):

- `transfer_carry.fold_droop` / `reachable_fold` — the shared shoulder-fold predicate (upright ~0.38,
  deep ~3.85; default `max_droop` 2.0 separates them), the same `_droop` metric the cart-place-goal lane uses.
- `ros_adapter.RosMotionPort._guard_rejects` — a clear-wrist solution must ALSO be a holdable fold; a
  **clear-but-deep** goal is rejected and the IK advances to the **next seed** (never wrist-flipped — flipping
  a clear solution leaves the clear branch); if no clear+reachable branch remains it **raises by name**
  (`'transfer preplace: no wrist-clear IK branch within the reachable shoulder-fold envelope (fold_droop<=X)'`)
  rather than streaming the infeasible goal.
- `ros_adapter.begin` — when the reach guard is active, the upright seed is inserted as the **first
  fallback** (right after the measured seed, which stays seed 0), so an already-upright run is byte-identical
  and the full-chain park re-solves from the upright seed to the holdable branch.

Why the preplace seeder and not a transfer-park standoff: the diagnosis proves a reachable branch already
exists at the actual park pose, so moving the base is unnecessary; the robust, minimal fix is to make the
seeder select it. A park standoff derived from the measured bay (the PR 233 pattern) is documented here as
the follow-up only if a future park pose is found with NO clear+reachable branch (then the named fast-fail
above fires and points at it).

## Wiring / backward compatibility

- HOST key read by `ros_host.py` → `TransferController`; echoed in `HOST_EFFECTIVE_CONFIG`; registered in
  `config_preflight.KEY_CONSUMER` as HOST (the host-vs-runtime split trap).
- Default off (absent/`false`) ⇒ the plain wrist-guard ⇒ byte-identical, including the staged-start path.
- The mission runtime configs live outside this repo/worktree (integration-lane owned), so the capability +
  the example fragment `docs/examples/transfer-preplace-reach-mission-fragment.json` ship here; the
  integration lane merges the key into the HOST block for the GPU proof.

## CPU evidence — `scripts/twin/tests/test_restock_transfer_preplace_reach.py` (16 tests)

- the coldwin6 deep fold is clear but NOT reachable (`fold_droop` > 3);
- the coldwin6 park pose yields a reachable upright clear preplace branch (ArmIK, same URDF);
- the staged park preplace is already upright+clear+reachable (byte-identical outcome);
- the coldwin6 pick is reachable (pick_top within reach; the clean tuck config is clear + holdable);
- `_guard_rejects` accepts the upright branch, rejects the deep fold to the next seed (no flip consumed),
  raises by name when no reachable branch is left, and — with the guard off — accepts the deep fold
  (the coldwin6 bug, reproduced);
- `resolve_reach_guard` normalises bool/mapping/off and rejects malformed values.

Related suites regress clean: `test_restock_transfer_carry`, `test_restock_cart_place_goal`,
`test_restock_transfer_live_pose`, `test_restock_scripted_transfer`, `test_config_preflight`,
`test_restock_host`, `test_cad_restock`, `test_ros_adapter_branch`, `test_restock_manipulation`,
`test_restock_transfer_bench`, `test_restock_transfer_place_policy`, `test_restock_trajectory_host`.

## GPU proof (handed to the integration lane)

Re-run the coldwin6 capstone (capstone241 config) with `transfer_preplace_reach_guard` merged into the HOST
block, under `/tmp/aile-gpu.lock`, whispr untouched. Expected PASS: `HOST_EFFECTIVE_CONFIG` shows the guard
active; at preplace a `WRIST_GUARD reject-deepfold ... -> next seed` then a `WRIST_GUARD accept` with
ur_shoulder_lift ≈ −1.95; the preplace tracks to goal (no `PATH_TOLERANCE_VIOLATED`); transfer proceeds
through place → verify. Commands + the pass/fail criteria are in the example fragment's `_bounded_gpu_proof`.
