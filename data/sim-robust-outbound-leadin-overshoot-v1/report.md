# sim-robust-outbound-leadin-overshoot-v1 — the outbound tow arrives at its goal from any couple seat, no overshoot

**Branch:** `fm/sim-robust-outbound-leadin-overshoot-v1` (PR into `fm/twin-robot-in-store-boot-v1`).
**Scope:** CPU-only nav code + tests + one GPU proof config fragment for the integration lane.
No GPU used in this work (the validator holds the lock; not taken).

## Problem (from run `coldwin3`, 2026-10-24)

Run `~/datasets/twin-robot-in-store-boot-v1/mission/coldwin3-fail` completed collect ✓ and a clean
couple ✓, then **failed at the outbound tow** with `reason=lane_leadin_unreachable_timeout`. This is a
**PR 238 lead-in robustness gap** (couple-seat variance), not a PR 239/240 issue — attempt-2
(`coldwin2-fail2`) seated in-band at y=15.311 and the outbound arrived cleanly.

The three recorded post-couple seats vary ~0.5 m in y (the pinned cart's pose after the real collect
varies): `coldlivewin` 15.746, `coldwin2attempt2` 15.311, `coldwin3` 15.227. The lane start is
re-centred to y=15.5 (middle of the distribution). coldwin3 seated **(-4.487, 15.227)**, 0.27 m SOUTH
of the start — just over `start_tol_xy=0.25`, so `xy_ok=False` and the lead-in engaged.

### Root cause — a lateral offset larger than the hardcoded arrival ball

Reconstructed from the run's `nav-trace.jsonl` (outbound leg, 2830 rows):

- The lead-in correctly bridged the south seat ONTO the centreline
  (`NAV_PLAN_DIAG lane_leadin … states=33 converges_to_goal=(-2.0, 15.25)`).
- But the base **rode a steady-state ~0.09 m NORTH lateral offset** down the lead-in's descending final
  segment (lane y 15.5→15.25; base y stayed ~15.34). At the goal x it was at **(-2.016, 15.341)** —
  `dist = hypot(0.016, 0.091) = 0.092 m`, **just outside** the coupled follower's **hardcoded 0.08 m**
  arrival ball (`articulated.py` step() used a literal `0.08`, ignoring the lane's declared
  `goal_tol_xy=0.30`). So the arrival never latched.
- A forward-only (no-reverse) coupled train then **sailed 0.3 m past the goal** to (-1.706, 15.314) and
  stopped (`lin≈0.002`). Every re-plan's `build_lead_in` rolls FORWARD from past-the-goal, diverging
  forever → `lane_leadin_unreachable`, failed by name after `tow_lane_acquire_timeout_s=8` fired.

The lane author *declared* `goal_tol_xy=0.30` as the achievable arrival band precisely because a coupled
tow tracks a lane with a lateral offset bigger than 0.08 m — but the follower never used it.

## CPU reconstruction (navigation unit harness)

`scripts/twin/tests/test_navigation_lanes.py :: TowLaneGoalStopTests`, open-grid fixtures, the real
`ArticulatedController`/`ArticulatedPlanner`:

- **Exact integration** (`_drive`) nulls the lead-in's residual offset perfectly — all seats arrive even
  without the fix — so the overshoot needs the base's real **steering inertia**. `_drive_lagged` adds a
  first-order lag (a documented steering-inertia model, NOT a simulator) that reproduces the coldwin3
  regime: the follower rides a lateral offset to the goal x.
- **Reproduction:** with the lag and goal-stop OFF, the follower overshoots the goal x and fails
  `lane_leadin_unreachable[_timeout]` (`test_lagged_follower_overshoots_and_fails_without_goal_stop`,
  tau 1.5 → every seat fails, maxx past the goal).
- **Direct isolation (no dynamics):** a base AT the goal x, stopped, 0.12 m laterally off → OFF never
  latches (`arrived_stopped` not returned); ON latches (0.12 < goal_tol_xy 0.30).

## Fix — `navigation.tow_lane_goal_stop` (opt-in, default off ⇒ byte-identical)

In `ArticulatedController` (coupled path), when ON **and an authored lane is active**:

1. **Arrival capture = the lane's `goal_tol_xy` / `goal_tol_yaw`**, not the hardcoded 0.08 — the observed
   ~0.09 m lateral offset latches within the author's achievable band.
2. **Decelerate to rest at the route end** — speed is capped by `sqrt(2·brake_acceleration·remaining)`
   where `remaining` is distance measured **along the route** from the nearest route point to the end
   (NOT Euclidean distance-to-goal, which a laterally-offset base never drives to zero), and the 0.03 m/s
   floor is dropped there — so the train stops at the goal and **never passes it**. While still moving
   inside the (wide) ball with route left to run, the follower keeps the decelerating route-follow
   instead of slamming a stop at the ball edge, so it comes to rest NEAR the goal (CPU: gd ≈ 0.03–0.12 m).
3. **Past-goal fast-fail** — a seat left genuinely beyond the route end (forward-only unrecoverable)
   fails IMMEDIATELY by name `lane_leadin_past_goal` (CPU: 0.2 s) instead of churning to the 8 s acquire
   timeout.

Plumbed via `navigation/factory.py` from `navigation.tow_lane_goal_stop`.

### Validation (CPU)

- ON: arrives `arrived_stopped` from **all four recorded seats** (exact AND lagged tau 1.5), within
  `goal_tol_xy`, never sailing irrecoverably past the goal.
- Band: seats y 15.0…16.0 (re-centred start ±0.5 m) arrive; **0.5 m beyond the band** (y 14.5 / 16.5)
  fail FAST by name (`*_timeout`, ≤10 s), not the 120 s mission nav-stuck.
- Past-goal seat: `lane_leadin_past_goal` in 0.2 s with goal-stop; legacy `lane_leadin_unreachable_timeout`
  without it (unchanged).
- Default-off byte-identical: goal-stop off keeps the 0.08 ball; `tow_lane_goal_stop` defaults False.
- 61 `test_navigation_lanes.py` tests pass; `test_navigation_factory`, `test_navigation_coverage`,
  `test_restock_mission_tail`, `test_return_cartyaw_regression` unaffected.

### On widening `start_tol_xy` (the spec's PR-238-brief suggestion — a deliberate deviation)

CPU-proven HARMFUL as a standalone lever: `start_tol_xy=0.6` with the default `tow_lane_track_tol_m=0.25`
makes the 0.27 m-south seat acquire the AUTHORED route directly, immediately trip
`stop_replan_tracking_error` on that 0.27 m offset, and churn to `goal_timeout` (test regime: 120 s).
The **lead-in** (`lead_in_max_m`) is the correct merge mechanism for an off-centreline seat — it keeps
the tracked error ~0 — and the goal-stop fix makes the lane seat-independent across the observed band
**without** touching `start_tol_xy`. Widening it is safe only if `tow_lane_track_tol_m` is also raised to
the offset (within the swept-envelope slack); kept at the default here. `tow_lane_reacquire_along_route`
(nearest-route acquire) is already on in the capstone.

## GPU proof (handed to the integration lane)

`docs/examples/outbound-leadin-overshoot-fragment.json` — merge `navigation.tow_lane_goal_stop: true`
onto the sim-robust-outbound-lane-start-v1 capstone keys. Bounded proof: collect+couple+outbound under
the full capstone keys (or the full capstone). The integration lane owns the GPU run (its mission config
lives outside this repo); report PASS/FAIL with the run dir. The ~0.09 m offset + the 0.30 m band have
margin for the observed seat spread; if a future couple change moves the seat out of the ±0.5 m band the
named fast-fail makes it legible rather than a silent freeze.
