# sim-robust-outbound-lane-start-v1 — outbound tow lane acquires from wherever the couple seats the base

**Branch:** `fm/sim-robust-outbound-lane-start-v1` (PR into `fm/twin-robot-in-store-boot-v1`).
**Scope:** CPU-only nav code + tests + one GPU proof config fragment for the integration lane.
No GPU used in this work (the validator may hold the lock; not taken).

## Problem (from run `coldlivewin`)

The every-stage-honest live_pose capstone now completes collect and a clean crosstrack couple
(`FINE_DOCK_DIAG status=ready hitch_mm=19.5 yaw_deg=0.92`) and then **deadlocks at the outbound
tow** — not on pacing or traction, on **lane acquisition**:

- couple seats the base at **(-4.584, 15.746, yaw 0.363)**, cart_yaw 0.362;
- the authored tow lane `outbound_dock_to_bay` has `start=(-4.46, 15.25, 0.0)`,
  `start_tol_cart_yaw=0.35`, and **no `start_tol_xy`** (default 0.25);
- the base is **+0.496 m north** of the lane start in y (Δx 0.124), so
  `dist = hypot(0.124, 0.496) = 0.51 m > 0.25 m` ⇒ `acquire(rebase) xy_ok=False` ⇒
  `plan_observed_route … states=0 reason='lane_not_acquired'`;
- the empty route cycles `lane_not_acquired` ↔ `replan_changed_observations` under the jittery live
  cloud, which **never reaches the base `_wait` 8 s `blocked_timeout`**, so the base froze for ~120 s
  WALL until the external mission nav-stuck budget failed the leg.

The lane start was authored for the **pre-crosstrack** couple end pose. The improved couple's seat
**varies run to run**: `coldliverecovery` seated at **(-4.464, 15.262, -0.013)** (essentially AT the
authored start, acquired fine), `coldlivewin` at **(-4.584, 15.746, 0.363)** (+0.5 m north, deadlocked).
A fixed authored start + a fine acquire tol cannot absorb that spread.

## Why a hand-authored lead-in to the exact start is NOT viable (kinematic finding)

The coupled base is **forward-only** (no reverse; the planner is "forward arcs only"). A forward-only
train 0.5 m **north** of a flat (yaw-0) lane cannot cheaply drop south: on CPU, the real
`ArticulatedPlanner` returns an **~11.5 m loop** for `seat → lane.start` and an **~12 m loop** for a
direct `seat → goal` re-plan (isolated to the 0.5 m **y**-offset, independent of the base yaw).
Re-planning from the seat, or an open-loop lead-in to the authored start, is therefore unusable.

What *does* work is a **closed-loop lateral merge**: the follower reduces lateral error while driving
forward along the lane (a gentle S-dip), which a 2.46 m flat lane can absorb for a **±0.25 m** offset
at min radius 1.5 m — but **not** a one-sided +0.5 m offset off a start at the band edge. Re-centring
the authored start to the **middle** of the seat distribution (y ≈ 15.5) puts both recorded seats at
±0.25 m, both absorbable.

## Fix (combination, all opt-in / default-off ⇒ byte-identical)

1. **Bounded coupled merge lead-in** (`TowLane.lead_in_max_m`, `scripts/twin/navigation/lanes.py`).
   When the base seats OFF the fine centreline, `build_lead_in` rolls a forward pure-pursuit coupled
   merge (curvature capped at the physical min turn radius, every step swept-validated against the
   LIVE map+cloud via `world.valid`) ONTO the authored centreline and splices the authored tail to the
   goal, returning a `Route` whose **first state is the measured seat** — so the follower acquires
   trivially from exactly where the couple left it. `_plan_lane` calls it only when the normal acquire
   fails; the verdict is `validated_lane_leadin`. `lead_in_max_m == 0` (default) ⇒ never synthesised.

2. **Fast named failure** (`tow_lane_acquire_timeout_s`, controller `_acquire_watchdog`).
   The empty-route churn is converted to a terminal failure by name in a few SIM seconds:
   `lane_not_acquired_timeout` / `lane_blocked_timeout` / `lane_leadin_unreachable_timeout`. Keyed on
   contiguous no-base-progress ticks with the active lane unacquired; reset on base progress or a
   non-empty route. SIM-clock, pace-immune. `0` (default) ⇒ the legacy churn (byte-identical).

3. **Re-centre the authored lane start** to the seat-distribution middle (y ≈ 15.5) — config, shipped
   in the example fragment, applied by the integration lane (mission configs are external). With the
   lead-in absorbing the residual spread around a centred start, a future couple-geometry change does
   not deadlock the outbound: it either merges, or fails fast by name.

`start_tol_xy` (already a per-lane key) + `tow_lane_reacquire_along_route` gate how far a seat may be
before the lead-in is attempted; `tow_lane_cart_yaw_rebase` + `tow_lane_cart_yaw_tol` let the lagging
cart arrive after the merge S-dip.

## CPU evidence

`scripts/twin/tests/test_navigation_lanes.py :: OutboundLeadInAndFastFailTests` (open-grid, the real
`ArticulatedController`/`ArticulatedPlanner`):

- `build_lead_in` bridges **each recorded seat** (route starts at the seat, ends at the goal);
- returns None when the seat is beyond `lead_in_max_m` (0.3 m cannot bridge 0.5 m) and when off (0.0);
- `plan_observed_route` ⇒ `validated_lane_leadin` for the drifted seat, `validated_lane` at the start;
- **full closed-loop drive arrives (`arrived_stopped`) from BOTH recorded seats** on the re-centred
  lane (coldlivewin ~30 s, coldliverecovery ~14 s sim);
- the unacquired lane fails fast by name (`lane_not_acquired_timeout`, `lane_leadin_unreachable_timeout`)
  in ≤ 6 s, not 120 s;
- timeout 0 + lead-in off ⇒ the legacy churn (no fast-fail), and the pre-existing straight-lane +
  trust-static + reacquire suites stay byte-identical (51 lane tests pass).

`test_navigation_factory.py` asserts `tow_lane_acquire_timeout_s` wires through (default 0.0).
All relevant nav suites pass (192 nav tests + 49 restock_runtime).

**Caveat:** the CPU drive is an open-grid kinematic integration, not a physics run. The real store
adds obstacles and PhysX cart dynamics (the lagging-cart terminal convergence in particular), so the
re-centred start must be confirmed to expand clear against the real static map, and arrival tuned, at
the GPU boot. That proof is the integration lane's.

## GPU proof config (for the integration lane)

`docs/examples/outbound-lane-leadin-fragment.json` — merge the `navigation.tow_lanes` +
`tow_lane_reacquire_along_route` / `tow_lane_cart_yaw_rebase` / `tow_lane_cart_yaw_tol` /
`tow_lane_acquire_timeout_s` keys into the capstone mission runtime-config (e.g.
`mission-runtime-coldlive-recovery.json`). Bounded proof: collect + couple + outbound under live_pose
with the cross-track couple (or the full capstone). Expected: the outbound ACQUIRES from the couple
seat (`validated_lane_leadin` when off the centre, `validated_lane` when on it) and tows to the bay;
a seat beyond band fails by name fast, never the 120 s nav-stuck.

## Deeper cause handed up

The couple seat spread (15.26 .. 15.75, ~0.5 m) is itself wide for a forward-only coupled outbound.
This change makes the outbound **tolerant** of that spread (centre + lead-in + fast-fail); tightening
the couple's crosstrack landing so it consistently seats near one point would further de-risk the
outbound and is the couple lane's (PR232/233 owner) call.
