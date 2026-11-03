# sim-robust-cart-place-goal-v1 — place-on-cart plans + executes (goal/branch fix)

**Branch:** `fm/sim-robust-cart-place-goal-v1` (into `fm/twin-robot-in-store-boot-v1`).
**Date:** 2026-10-22. CPU-proven; one bounded GPU proof handed to the integration lane.

## Problem

The honest COLLECT now gets through the pick for real (weld at 0.48/0.50 mm, both `collect1` and
`collectsrc`), and both runs then failed at **place-on-cart**:

- `collectsrc` (PR 187): the deck place goal's IK landed in the `ur_wrist_2`-vs-`ur_forearm` self-collision
  branch → cuMotion `Invalid c-space position: self collision detected`, a 12-replan storm into GPU OOM +
  a host heartbeat gap (the storm itself fixed by PR 200's plan-attempt cap + heartbeat-before-ESDF).
- `collect1` (PR 200 merged): the cap fails fast, but the place goal still could not be executed. From its
  logs the preplace planned on a **clear** branch yet commanded `shoulder_lift = -3.596` (a DEEP fold); the
  soft arm sagged to -3.204 — **0.39 rad short** at steady state (settle 30 s, residual velocity) →
  `PATH_TOLERANCE_VIOLATED` → `ABORTED` → `arm command heartbeat lost`.

## Root cause (logs + CPU IK)

Runs under `~/datasets/twin-robot-in-store-boot-v1/mission/{collect1,collectsrc}`
(`host-stack.log`). The base drives in front of the rollwagen mouth (`reposition_before_place:true`,
`cart_cage_place:false`), then the deck place/preplace poses go through the host's seeded-KDL → cuMotion
nearest-branch IK (`Using joint space goal`). The recorded place target in `ur_base_link`
(`REPOSITION_REREGISTER`): deck `(1.088,0.122,0.419)`, place `(0.889,0.161,0.499)`, preplace
`(0.713,0.196,0.529)`.

CPU IK on the mission arm chain (pure-numpy `scripted_transfer.ArmIK`) shows each deck pose has **two**
collision-free (clear-wrist) IK families:

- an **UPRIGHT** branch (`shoulder_lift ≈ -1.9`, low gravitational droop), and
- a **DEEP** fold (`shoulder_lift ≈ -3.57`, near gravity-max under the 0.45 kg carton).

Seeded from the folded carry config the host picks the DEEP fold (un-holdable → collect1's sag) or, on a
different seed, the wrist self-colliding branch (`wrist_1>0.65, wrist_2<0` → collectsrc's storm). It is a
**goal/branch** problem, not a planner problem: a collision-free, trackable solution exists — the host just
isn't selecting it.

## Fix (the goal, not the planner)

Two opt-in HOST-config keys, default off ⇒ byte-identical; reuse the transfer path's wrist-guard predicate
(`transfer_carry.in_clear_region` / `wrist_flip_seed`) and the same `begin_joint_goal` fixed-shape pattern
as `cage_preplace_shape` / `carry_lift_shape`:

1. **`collect_place_preplace_shape`** — a CPU-validated UPRIGHT clear-region 6-joint preplace shape
   `[0.093, -1.910, 2.014, -0.103, 1.664, 0.0]`. For the non-cage COLLECT preplace it is driven as a fixed
   collision-planned `begin_joint_goal` (relaxed 0.2 rad gate), so the preplace avoids both the un-holdable
   deep fold and the self-colliding branch. FK TCP is ~0.6 mm from the sensed preplace standoff; reach 0.909 m
   (margin 0.391 m).
2. **`collect_place_wrist_guard`** — extends the transfer wrist-guard clear-branch re-seed to the COLLECT
   place + retreat pose-IK, so the SENSED-target place insert (seeded `near_start` from the upright preplace)
   stays on the upright clear branch instead of flipping to a self-colliding one.

The carton is still placed at the sensed deck centre with the profile yaw — **no off-centre / re-yawed
set-down**; only the IK branch changes. Full numbers + the preplace→place transition in
[`station-contract.md`](station-contract.md).

### Code

- `scripts/twin/restock_manipulation/controller.py` — new `collect_place_preplace_shape` /
  `collect_place_wrist_guard` ctor args + validation; collect-preplace `begin_joint_goal` branch (mirrors the
  cage branch); `_transfer_wrist_guard` generalised to `_place_wrist_guard` (transfer **or** guarded collect).
- `scripts/twin/restock_manipulation/ros_host.py` — wire both keys from the host config; log them in
  `HOST_EFFECTIVE_CONFIG`.
- `scripts/twin/restock_manipulation/config_preflight.py` — register both as HOST keys (mis-placement guard).

## Verification

- **CPU:** `scripts/twin/tests/test_restock_cart_place_goal.py` — 8 tests, all pass. Asserts: a collision-free
  in-reach IK solution exists for the deck preplace+place; a self-colliding branch exists (the guard does
  real work) and the wrist-flip recovers a clear one; the upright branch is clearly lower-droop than the deep
  fold; the shipped shape is clear / in-limits / FK-at-standoff; the place from the upright preplace stays
  upright+clear with a small move; defaults-off is byte-identical; bad shapes raise.
- Touched-area suites green: `test_restock_host.py`, `test_config_preflight.py`, `test_restock_transfer_carry.py`,
  `test_restock_cart_cage_place.py` (+ the full `test_restock_manipulation.py` / `test_restock_scripted_transfer.py`
  pass except a pre-existing missing-`PIL` case unrelated to this change).

## Bounded GPU proof — handed to the integration lane

`docs/examples/cart-place-goal-mission-fragment.json`: merge the two HOST keys into the `collect1` mission
runtime-config HOST block and run collect + couple (GT collect, `RESTOCK_COLLECT_GROUND_TRUTH=1`,
`visuals:false`) through `scripts/run_mobile_restock.py` under `/tmp/aile-gpu.lock`.

**Expected PASS:** `HOST_EFFECTIVE_CONFIG` shows both keys active; place-on-cart reaches preplace via the
upright `begin_joint_goal` (no `Invalid c-space position` storm, no deep-fold `PATH_TOLERANCE_VIOLATED`), the
place insert stays upright+clear and tracks the sensed deck target, and collect reports placed on the cart
deck; couple proceeds. If the upright preplace itself still settles beyond 0.2 rad, widen only that move's
gate; if the insert flips branch, confirm `collect_place_wrist_guard` is active in `HOST_EFFECTIVE_CONFIG`.
I do not run GPU (serialised through `/tmp/aile-gpu.lock`, integration-lane owned) and do not edit the
`~/datasets` runtime configs; this is the capability + fragment for that lane.
