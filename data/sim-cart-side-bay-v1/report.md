# sim-cart-side-bay-v1 — park the cart in a side bay off the portal axis

**Branch:** `fm/sim-cart-side-bay-v1` → PR against `fm/twin-robot-in-store-boot-v1`.
**Scope:** the captain's **option 1** follow-up to sim-shelf-staging-v1 (PR 274) / sim-service-leg-carry-v1
(PR 276): park the rollwagen in a **side bay off the portal axis** so both the free robot and the
re-coupled train can pass the single 2.6 m doorway. Opt-in extension of `shelf_staging`, default off ⇒
byte-identical. CPU-proven; the GPU full-cycle proof (coldwin25) is integration-owned.

## Captain's intent → what the side bay delivers

PR 274 re-sequenced the sales-floor leg so the coupled train never makes the ~46° turn to the endcap — but
it parked the rollwagen **on the portal axis** at the staging pose. That on-axis cart **blocks the free
base from threading the single 2.6 m doorway home** (grid A\* finds NO PATH) and a coupled re-couple +
tow-home does not expand. The captain picked **option 1**: park the cart in a **side bay off the portal
axis near the shelf**, so both the free robot and the re-coupled train can pass the doorway.

This PR delivers, CPU-proven, the parts that are **consistent with every resolution the captain may pick**:

| leg (each its own behaviour-tree step) | measured (CPU, real `twin.navigation` planner + 798-obstacle static map) |
|---|---|
| **tow→side bay** `outbound_dock_to_side_bay` | straight dead-centre doorway crossing (cart **5.4°** at the throat, PR 266 kept) then a west tuck into the bay at robot pose `[-4.3, 6.95, −π/2]`; cart parks centre `[-4.14, 8.2]` off-axis. **Whole-lane coupled articulation 14.4°** (< the 15° target), vs the ~46° of the old coupled-to-endcap lane. |
| **brake cart / uncouple** | pin + release the cart at the bay (reuse the uncouple mechanic). |
| **robot-only service** | the free base picks the carton off the parked cart, carries it to the endcap, places it, drives back (PR 276 service leg, re-targeted to the bay park). |
| **re-couple** (geometry proven, follow-up to execute) | the free base reverses **straight** along the cart's drawbar axis (`~[-4.45, 6.97]`, ~−104°) onto the hitch; swept base clears the static map. |
| **drive→home (empty)** | with the cart parked OFF the axis, the free base threads the **freed doorway** home — grid A\* now **routes** (the exact case that was NO PATH with the on-axis staging cart). The cart stays parked in the bay. |

`articulation-before-after.png`, `paths-top-down.png` (this dir).

## The hard wall (reported honestly): the coupled tow-home is a config-selectable follow-up

The brief asks for BOTH a gentle (<15°) outbound AND a coupled tow-home that returns the cart to the
backroom. These are **mutually infeasible** on this single-doorway cramped sales floor: a single doorway +
a forward-only rear-hitch drawbar forces the cart to cross the doorway **facing south out and facing north
home** — a **conserved ~180° cart reorientation** that must be paid at min radius 1.5 m with no room for a
large-radius turn. Measured either/or (CPU, real planner):

| design | tow-to-bay (outbound) | coupled tow-home (return) |
|---|---|---|
| **SOUTH-faced bay** (this baseline) | **14.4°** ✓ | the return must flip ~180° → authored route **loop_ratio 3.8** (fails the `check_tow_lanes` `loop ≤ 1.7` gate); guided U-turns **do not expand** (no room between the gondolas) |
| **WEST-faced bay** (dock-align shape) | **45.9°** ✗ (the captain's original complaint) | clean: **loop_ratio 1.09**, peak **45.6°**, crosses the doorway aligned (0.6°) |

So **gentle outbound ⟺ a looping return**, and **a clean return ⟺ a ~46° outbound** — the same wall PR 274
documented, re-confirmed from the bay with numbers. Option B (south bay, reorient in the aisle) needs a
wider aisle = **gondola moves = dataset regen, out of scope**. The captain is choosing the resolution
(firstmate relayed the either/or); this baseline is consistent with all of them and leaves the coupled
tow-home as a **named, config-selectable follow-up**.

### Which lanes/poses change under each captain option (so the follow-up is a small diff)

- **(A) The side bay IS the cart's parking spot** (this baseline, no further change): `cart_side_bay: true`,
  `mission_poses.cart_side_bay = [-4.3, 6.95, −π/2]`, the `outbound_dock_to_side_bay` lane; RETURN drives
  the base home empty. The cart stays uncoupled at the off-axis bay.
- **(B) South bay + reorient in the aisle** (not cleanly feasible as-is): keep the baseline poses/lane; add
  `cart_tow_home: true` and a wider-aisle return lane — **needs a planogram/dataset change**, so it is
  deferred, not authored here.
- **(C) West bay (pay the turn on the outbound)**: change `mission_poses.cart_side_bay` to the west-facing
  endcap pose `[-3.05, 6.0, π]`, set `cart_tow_home: true`, and add the `_followup_return_lane_west_bay`
  lane (in the fragment: `return_side_bay_to_cart_parking`, loop 1.09 / 45.6°). The runtime RETURN then
  re-couples at the bay (reusing the couple + `uncouple_on_return` mechanics) instead of driving home empty.

The BT retrieval steps (`RETURN_SIDE_BAY_RECOUPLE_PLAN`: re-couple@bay → tow→portal lead-in → tow→cart park
→ brake → uncouple → drive home) are already defined and the diagram renders them when `cart_tow_home` is
selected, so the follow-up is a small diff.

## Offline proof (CPU, no GPU lock)

- `scripts/check_cart_side_bay.py --environment <env.json> --scenario scripts/restock_scenario.json
  --lanes docs/examples/cart-side-bay-mission-fragment.json` → **CART_SIDE_BAY PASS** (6 gates): tow-to-bay
  14.49° (< 15°); cart crosses the doorway aligned (5.44°, swept within the 2.6 m opening); parked cart
  clears the y=6 service aisle; the on-axis staging cart BLOCKS the home route (regression reproduction);
  the side bay **frees** the robot-home-empty route; the straight-reverse re-couple clears. Plus NOTE lines
  quantifying both tow-home variants (west loop 1.09 / 45.6°, south loop 3.80).
- `scripts/check_tow_lanes.py … --lanes docs/examples/cart-side-bay-mission-fragment.json` →
  **TOW_LANES_PROOF PASS** (the `outbound_dock_to_side_bay` lane expands, is swept-envelope-clear live, the
  real `ArticulatedController` follows it to goal `arrived_stopped` loop_ratio 1.0, and the portal lead-in
  aligns the cart at the throat).
- `python3 -m pytest scripts/twin/tests/test_restock_runtime.py` — **72 pass** (incl. 4 new: outbound tows
  to the bay not the staging pose; return drives home empty uncoupled; cart_side_bay requires the pose +
  shelf_staging; flag-off is the PR 274 path).
- `… test_restock_bt_phases.py test_restock_bt_diagram.py` — **20 pass** (incl. the pure `stage_plan`
  dispatch to the side-bay plans, and the diagram showing `outbound: tow→side bay / brake / uncouple`,
  the transfer service leg, and `return: drive→home (empty)` — plus the `cart_tow_home` follow-up diagram
  showing the five retrieval steps). The py_trees tree-tier parity is skipped without the task `.venv-bt`.
- `… test_isaac_bundle.py test_service_leg_carry.py test_restock_mission.py test_navigation_lanes.py
  test_restock_mission_tail.py test_config_preflight.py` — **all pass** (no regression; default off is
  byte-identical).

## Hand-off for the GPU proof (integration-owned, coldwin25)

Merge `docs/examples/cart-side-bay-mission-fragment.json`: `navigation.tow_lanes`
(`outbound_dock_to_side_bay`), `mission_poses.cart_side_bay` (+ the existing `shelf_staging` /
`shelf_approach` / `cart_parking`), set the top-level `shelf_staging: true` AND `cart_side_bay: true`. Keep
the scene keys `portal_floor_box: true` and the `coupled_heading_hold` / `heading_hold_*` keys (unchanged
by this geometry change). **Acceptance (baseline):** the coupled train couples south-aligned on the portal
axis, tows STRAIGHT to the side bay (≤ 15° coupled articulation everywhere, cart crosses the doorway
aligned ≤ 10°, ≥ 0.5 m clearance each side), uncouples there; the free base services the endcap (carton
carried, PR 276) AND drives **home empty** through the freed doorway. For the coupled tow-home follow-up
(cart parked back in the backroom), apply option C (above) with `cart_tow_home: true` and GPU-validate the
45.6° aisle reorientation.

## Files

- `docs/examples/cart-side-bay-mission-fragment.json` — the bay pose + tow-to-bay lane + both tow-home
  variants + hand-off notes.
- `scripts/twin/restock/runtime.py` — the opt-in `cart_side_bay` outbound-to-bay goal + return home-empty.
- `scripts/twin/restock/bt/{stage_phases,diagram}.py` — the side-bay BT steps + diagram branch.
- `scripts/twin/restock/isaac_bundle.py`, `restock_manipulation/config_preflight.py` — config wiring.
- `scripts/check_cart_side_bay.py` — the CPU proof.
- `docs/restock_navigation.md` — the sequence ("Side bay for the cart").
- `data/sim-cart-side-bay-v1/{articulation-before-after,paths-top-down}.png` — before/after + top-down plots.
