# sim-robust-cart-place-descent-v1 — the collect place completes on the cart deck

**Branch:** `fm/sim-robust-cart-place-descent-v1` (into `fm/twin-robot-in-store-boot-v1`).
**Date:** 2026-10-22. CPU-proven; one bounded GPU proof handed to the integration lane.

## Problem

With the PR 209 goal/branch fix active (`collect_place_preplace_shape` + `collect_place_wrist_guard`,
both confirmed in `HOST_EFFECTIVE_CONFIG`), the validator's run **collect2** got through every earlier
collect failure: pick + weld (2.99 mm), carry fold, base reposition to `place_approach`, and the **UPRIGHT
preplace** over the rollwagen deck with no collision-planning failure, no deep fold and no wrist
self-collision. It then **FAILED** with `arm_not_in_transport_timeout`, the carton left held and the arm
never reaching the empty transport pose.

Run: `~/datasets/twin-robot-in-store-boot-v1/mission/collect2/` (`host-stack.log`, `collect2.console.log`,
`mission.json`, `nav-trace.jsonl`).

## Reconstructed place timeline (logs + nav-trace)

The nine `TRAJ_HOST_PLAN`/`TRAJ_HOST_START` moves in `host-stack.log` map to: approach, 4× contact-step
descent, lift→carry (`carry_lift_shape`), tuck, **preplace** (`collect_place_preplace_shape`, SIM start
93.417, upright `shoulder_lift=-1.91`), and **place** (pose-IK descent, SIM start 116.883). The place
descent trajectory **SUCCEEDED** in MoveIt (`Completed trajectory execution with status SUCCEEDED`). No
retreat / retreat_tuck / stow move was ever planned — the place was the last move dispatched.

`mission.json` stage_timeline: collect attempt 1 reached the `place` step (SIM 93.067) and failed at SIM
**117.317** with reason **`base moved during stationary operation`** (not a manip reason). Attempts 2 and 3
re-began at `drive→pick` and failed `arm_not_in_transport_timeout` (the mission's overall reason).

`nav-trace.jsonl` pins the cause exactly. During the place descent the braked base **physically drifts**:

| tick | SIM t | base x | base `lin` (m/s) | parked | carton z |
|------|-------|--------|------------------|--------|----------|
| 6997 | 116.62 | −4.518 | 0.000 | True | 1.093 |
| 7020 | 117.00 | −4.528 | 0.056 | True | 1.056 |
| 7039 | 117.317 | −4.557 | 0.139 | **False** | 0.941 |
| 7042 | 117.383 | −4.559 | 0.019 | True (re-settled) | 0.942 |

The extended, carton-loaded arm's place-descent reaction shoves the base ~**4 cm** in **x** (velocity
peaks 0.14 m/s), the 0.25 s windowed `parked` flag flips **False for 3 ticks**, and the mission-level
stationary guard `runtime.py` `if not p.parked:` fails the stage. Because the stage aborts mid-descent
(carton still welded, z≈0.94→held, arm never reaches `stow`→transport), every re-begin can't drive
(`arm_not_in_transport`) → `arm_not_in_transport_timeout`; the host also latches a station fault
(`station fault requires explicit recovery`).

**Why the base slides:** `isaac_port._apply_base_brake` deliberately uses a **yaw-lock-only** brake while a
carton is welded (translation left free to relieve the world→base→arm→weld→carton loop — a full lock
oscillated the arm in runs 208/209). The place descent reacts against that free translation. The shove is a
**transient** (re-settles by tick 7042) and ~4 cm is far inside the `_place_verify_xy_tol = 0.25 m`
ground-truth place tolerance — i.e. harmless to the placement, but fatal to the strict parked guard.

## Fix (opt-in, default-off ⇒ byte-identical)

1. **RUNTIME `collect_place_base_settle_m`** (`restock/runtime.py`, wired via `isaac_bundle.py`): during the
   loaded collect place phases (`transit/preplace/place/release/retreat/retreat_tuck/stow`) tolerate a
   **bounded** base excursion from the `place_approach` re-park anchor (captured at the rewarm→execute
   transition from `p.odometry.pose`) instead of failing on the expected, re-settling descent reaction. The
   `_loaded_place_base_excursion` predicate is scoped to a collect that is actually placing (repositioned,
   loaded place phase, anchor + base pose present) — everything else keeps the strict guard. A base pushed
   **beyond** the bound fails with a NAMED reason (`base pushed <x> m beyond the <tol> m collect-place settle
   tolerance…`), never a silent slide. The ground-truth carton-on-deck verify remains the placement proof.

2. **HOST `release_detach_timeout_s`** (`restock_manipulation/controller.py`, wired via `ros_host.py`): the
   place→`release` already emits a gripper-open and detaches the SIM_ATTACH weld, then retreats and stows.
   This SIM-clock budget makes a release that does **not** detach (contact/weld still engaged) fail FAST with
   a NAMED reason, instead of carrying the carton back up on retreat/stow or stalling to the coarse 120 s
   `phase_timeout`. (The stow's own `collision-planned transport stow failed` already gives the bounded,
   named stow-failure the brief asks for.)

Both keys are registered in `config_preflight.py` (RUNTIME / HOST) and echoed in `HOST_EFFECTIVE_CONFIG`
(release budget). Absent/0 ⇒ the strict guard + phase_timeout only (byte-identical).

## CPU proof

- `tests/test_restock_runtime.py`: `test_loaded_place_tolerates_a_bounded_base_shove_and_keeps_placing`
  (replays collect2's ~4 cm shove → the place keeps running), `…fails_named_when_the_base_is_pushed_beyond_
  tolerance`, `…strict_by_default_byte_identical`, `…scoped_to_collect_and_the_loaded_place`,
  `…off_key_leaves_excursion_predicate_none`, `test_collect_place_base_settle_m_validation`.
- `tests/test_restock_manipulation.py`: `test_collect_place_runs_descent_release_stow_to_the_transport_pose`
  (place→release(weld detach)→retreat→stow→**verify**/transport), `test_release_fails_fast_when_the_weld_
  does_not_detach`, `test_release_with_budget_off_does_not_fast_fail_a_stuck_weld`,
  `test_release_detach_timeout_off_by_default_and_rejects_bad_values`.
- `tests/test_restock_cart_place_goal.py`, `test_config_preflight.py` unchanged-green (goal keys + preflight).

## Bounded GPU proof (handed to the integration lane — PENDING)

Config fragment `docs/examples/cart-place-descent-mission-fragment.json`: merge
`collect_place_base_settle_m: 0.08` (RUNTIME) + `release_detach_timeout_s: 8.0` (HOST) into the **collect2**
config (GT collect + scripted transfer + couple) and re-run under `/tmp/aile-gpu.lock` (`HEAVY_MB=2500`).
Expected PASS: the place descent runs without `base moved during stationary operation`, the weld detaches,
the ground-truth verify shows the carton on the deck, the arm stows to transport (`transport_ready` True),
and collect reaches DONE so couple can begin. The integration lane owns the GPU boot.
