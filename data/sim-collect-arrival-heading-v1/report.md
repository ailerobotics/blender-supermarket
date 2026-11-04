# sim-collect-arrival-heading-v1 — the coldwin29 cold-collect place_approach stall

**Branch:** `fm/sim-collect-arrival-heading-v1` (base `fm/twin-robot-in-store-boot-v1`). CPU-only; no GPU
run from this lane. Saved run analysed: `~/datasets/twin-robot-in-store-boot-v1/mission/coldwin29-fail1-saved`.

## The failure, named by its real cause

coldwin29's cold **COLLECT** picked and carried the carton, drove to the `place_approach` standoff, then
**sat there ~160 s and never entered PREPLACE**; collect hit its 300 s stage timeout (`collect timed out`),
and the retries then died fast. The stage log advances to step 3 (`drive→rollwagen`, nav) at sim t≈92.7 and
never reaches step 4 (`perceive deck`).

The nav never latched arrival. From the run's own `NAV_ARRIVAL_DIAG`:

```
xy=0.1518 xy_tol=0.150 yaw_err=1.4377 yaw_tol=0.200 gate0='tracking'
pose=(-4.474, 15.55, -1.704) goal=(-4.5, 15.4, 3.142) latched=True
```

The base reaches the goal **position** (xy ~0.15–0.18 m, right at the 0.150 tol) but its **heading is frozen
at ~-1.71 rad (~-98°)** while the goal heading is **π (180°)** — a **1.44 rad error that never nulls**. Nav
returns `tracking` every tick (401×) and never `arrived_stopped`, so the collect stage never transitions
navigate → rewarm → place.

**Root cause (geometry, at the source):** `place_approach = (-4.5, 15.4, π)` was authored for the
**pre-304 EAST-facing cart** — its standoff faces the cart mouth at heading **π** (`derive_place_approach`
for an `+x`-open cart yields ±π). **PR 304 rotated the dock to the SOUTH-facing cart** `(-4.6, 16.0, -π/2)`,
whose mouth opens `-y`; its standoff must face **+π/2**. The `place_approach` pose was never updated. The
base drives the configured final leg (staging `(-4.5, 16, 0)` → `(-4.5, 15.4)`, a straight drive **south**)
and arrives facing **-π/2**, exactly `π/2` (≈1.57 rad) from the stale π goal.

**Mechanism of the stall (why it never recovers):** the arrival test needs xy **and** yaw within tolerance
at once (and stopped, gate clear). Nulling a ~1.43 rad heading in place drifts a differential base laterally
(~2.8 cm/rad; the fm482 pivot effect) off the xy circle; the XY-hysteresis latch releases and the drive
branch pulls the base back toward the goal *point*, so the heading **limit-cycles** and never accumulates.
`ang` oscillates ±0.3 rad/s but yaw stays pinned at ~-1.71. This is **not** PR 267 hold-vs-rotate contention
(`_hold_base_on_nav_hold` only holds when BOTH linear and angular are 0; the stall ticks command nonzero
jitter), **not** a planner loop, **not** the collect-capture gate (GT collect; the stage never leaves nav),
and **not** an unreachably tight tolerance (the demanded heading is 1.43 rad away — no sane yaw tol reaches
it). See `stall-diagnosis.png`.

## The fixes (at the source)

1. **Name it, in sim time.** `navigation/core.py` `Controller` gains `arrival_heading_stall_s` (**default
   20 s, SIM-clock**; `navigation.arrival_heading_stall_s` / `0`/`false` disables). When the base is XY-near
   the goal but the heading error exceeds tolerance **and stops improving** for the budget, it prints
   `ARRIVAL_HEADING_STALL yaw_err=… xy=… gate=… stalled_s=…` and fails the leg with the named reason
   `arrival_heading_stall`. The budget is charged **only** against ticks with no ≥0.05 rad heading
   improvement, so a slow-but-converging rotation never trips and a healthy arrival (latches in a few sim s)
   is byte-identical apart from no new output. On the recorded coldwin29 poses it fires at **sim ≈140.8 s**
   (leg start ~119 + 20 + settle) — ~159 s before the 300 s stage timeout, which was previously the *only*
   thing that fired.

2. **Arrive already facing the deck via a straight lead-in (no spin).** `derive_place_approach_stage` is
   re-authored: the staging waypoint sits `lead_in` m **behind** `place_approach` along its own heading, so
   the **final leg is a straight drive whose direction equals `place_approach`'s heading** — the base
   arrives aligned. For the PR 304 south dock the reach-correct standoff is **(-4.6, 14.92, +π/2)** (faces
   the deck; deck reach 1.037 m < 1.3 m gate) with a lead-in at **(-4.6, 14.32, -0.979)** south of it, so the
   base drives **north** into the standoff and arrives facing the deck. Works for any cart orientation.

3. **An offline gate for the mis-authored heading.** `scripts/check_collect_arrival_heading.py` derives the
   spin-free standoff + lead-in from the dock and asserts (A) the final leg arrives aligned, (B) the deck is
   in reach, (C — optional) an authored pose's heading matches its configured drive-in (the coldwin29 catch:
   the stale pose fails with `spin_deg=90.0`), and (D — optional `--environment`) the lead-in route footprint
   clears the real ~800-obstacle static map. `check_collect_dock_reach` validated a swept footprint but never
   the controller's heading convergence — that is the gap this closes.

## CPU proof

- `scripts/twin/tests/test_collect_arrival_heading.py` — **replays the recorded coldwin29 final-leg poses**
  (fixture `fixtures/coldwin29_collect_finalleg.json`, 438 ticks, sim 119→300) through the **real**
  `Controller` arrival logic: the stale π goal never latches `arrived_stopped` (legacy `stall_s=0`: silent
  `tracking` to the stage timeout); with the guard on it names itself (`arrival_heading_stall`) at sim < 160;
  an aligned goal heading never trips the guard; and a kinematic closed-loop drive to the corrected
  (drive-in) heading arrives. Plus the lead-in geometry + the CLI contract.
- `scripts/check_collect_arrival_heading.py` — against `docs/examples/cart-side-bay-mission-fragment.json`
  (the 304 dock) + the real `environment.json`: PASS for the corrected derivation and the lead-in route
  clearance (798 obstacles); FAIL (named 90° spin) for the authored coldwin29 pose. `check_collect_dock_reach`
  still PASS (reach 1.037 m).
- `scripts/twin/tests/test_isaac_bundle.py` — the re-authored `derive_place_approach_stage` (spin-free for
  both the east-cart and the south-dock).
- Regression: `test_navigation*`, `test_collect_arrival`, `test_articulated_navigation`, `test_restock_runtime`,
  `test_restock_fine_dock`, `test_navigation_lanes`, `test_mission_tail` — all green with the default-on guard.

## Hand-off to the integration lane

The mission `mission_poses` live in the integration config (outside this repo). To clear the collect
place-on-cart stall, set:

```json
"mission_poses": {
  "place_approach":        [-4.6, 14.92, 1.5708],
  "place_approach_stage":  [-4.6, 14.32, -0.9787]
}
```

(emitted by `check_collect_arrival_heading.py` as `CORRECTED_MISSION_POSES`; `lead_in` 0.6 m, standoff 1.0 m).
The `arrival_heading_stall` guard is **on by default** — no config needed; set
`navigation.arrival_heading_stall_s` to tune or `0`/`false` to disable. **Acceptance:** a cold collect-only
run (`--stop-after collect`, GT collect) enters PREPLACE and completes place-on-cart within the stage budget,
and any residual non-converging arrival names itself `ARRIVAL_HEADING_STALL` instead of running out the
300 s stage timeout.

**Out of scope / follow-ups:** the pre-304 reach-from-fixed-pose path is unchanged; whether to keep the arm
reaching the deck from the `+π/2` standoff vs the fixed collect pose is the integration lane's call (both are
in reach). The lead-in heading `derive_place_approach_stage` produces for leg 1 is the pickup→stage bearing;
a route that needs a different leg-1 heading should pass its own staging waypoint.
