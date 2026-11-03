# sim-shelf-staging-v1 — re-sequence the sales-floor leg so the coupled train never turns to the endcap

**Branch:** `fm/sim-shelf-staging-v1` → PR against `fm/twin-robot-in-store-boot-v1`.
**Scope:** tow-lane geometry + the opt-in sales-floor re-sequence (outbound → staging → uncouple →
solo service), verified offline on CPU. GPU proof + the uncoupled-service carton-carry (manipulation)
+ the cart-retrieval on return are integration-owned (handed off below). Out of scope (done /
separate): dock/portal geometry (sim-dock-align-v1 / sim-portal-widen-v1), carton handling, dataset
regen.

## Captain's intent → what the geometry says

The captain: make the warehouse→sales passage so the robot does **not** rotate the rollwagen going
through it. sim-portal-widen-v1 (PR 266) widened the portal to 2.6 m; sim-dock-align-v1 (PR 271)
re-headed the couple dock south on the portal axis so the **backroom** passage is straight (≤ 3.6°).
The **one remaining coupled rotation** was on the **sales floor**: the externally-fixed, west-facing
endcap service pose `shelf_approach [-3.05, 6.0, π]` is 0.825 m **east** of the portal axis, so
delivering the **coupled** cart there needed a ~46° in-aisle turn — and on the outbound that turn
began just south of the wall, so the cart's **tail exited the doorway angled up to ~40°** (dock-align
report's explicit residual + named follow-up).

The fix is **not** to re-author the endcap (store planogram, external): it is to **re-sequence** the
leg so the **free base**, not the coupled train, makes that turn.

## The re-sequence (opt-in, default off ⇒ byte-identical)

| step | body | coupling |
|---|---|---|
| **tow→staging** | tow the rollwagen STRAIGHT down the portal axis to `shelf_staging [-3.875, 8.0, -π/2]` | coupled |
| **brake cart** | pin the parked cart to the world | coupled |
| **uncouple** | release the tow coupling (the SAME mechanic as the return uncouple) | → uncoupled |
| **drive→service** | the FREE base drives ALONE to `shelf_approach [-3.05, 6.0, π]` | uncoupled |
| park / transfer / verify | service the endcap with the base parked at the service pose | uncoupled |

The coupled train only ever drives **straight**; the sharp turn to the west-facing endcap is done by
the uncoupled base (no cart to swing).

## Measured, before vs after (CPU, `twin.mobile` train envelope, 798-obstacle static map)

`articulation-before-after.png`, `paths-top-down.png` (this dir):

| sales-floor OUTBOUND | whole-lane coupled max | cart-in-doorway max | robot wall-cross |
|---|---|---|---|
| **BEFORE** (coupled tow → endcap, dock-align) | **45.9°** | **39.6°** | 0.0° |
| **AFTER** (straight coupled tow → staging) | **0.0°** ✓ | **0.0°** ✓ | 0.0° ✓ |
| AFTER uncoupled robot-only service → endcap | n/a (no cart) | n/a | — |

**The captain's win:** the coupled sales-floor turn (46°) and the cart-in-doorway angle (40°) both go
to **0.0°** — the coupled train tows straight down the portal axis and the cart crosses the doorway
aligned; the endcap turn is now an **uncoupled** free-base drive. Target "coupled < 10° anywhere": met
(0.0° everywhere on the coupled leg).

The parked rollwagen sits on the portal axis at centre `[-3.875, 9.27]` (body `y[8.6, 9.69]`), fully
south of the doorway (`y=11.125`) and north of the `y=6` service aisle, clear of the wall fixtures
(`y≥10.25`) — it does not block the aisle or the endcap camera view.

## Offline proof (CPU, no GPU lock)

- `scripts/check_tow_lanes.py --environment <env.json> --scenario scripts/restock_scenario.json
  --lanes docs/examples/tow-lanes-mission-fragment.json` → **TOW_LANES_PROOF PASS** (30 checks on the
  real 798-obstacle grid): `outbound_dock_to_staging` expands + swept-envelope-clear live; the real
  `ArticulatedController` follows it to goal (`arrived_stopped`, **loop_ratio 1.0, end articulation
  0.0°**); `portal_leadin_aligns_cart` straight span 3.84 m with cart articulation **0.0°** at the
  crossing.
- `scripts/check_shelf_staging_nav.py --environment <env.json> --scenario scripts/restock_scenario.json
  --lanes docs/examples/tow-lanes-mission-fragment.json` → **SHELF_STAGING_NAV PASS**: the straight
  coupled tow is ≤ 10° everywhere + cart crosses aligned; the parked cart clears the service aisle; the
  **uncoupled robot-only service path** staging→endcap clears the static map (0.20 m) and the parked
  cart (forward departure) with the base's PHYSICAL rectangle.
- `python3 -m pytest scripts/twin/tests/test_restock_runtime.py` — **64 pass** (incl. 5 new:
  outbound tows→staging→uncouple→service leg order; park allows an uncoupled cart-parked packet; the
  relax is scoped to `_shelf_cart_parked`; flag-off is byte-identical; the staging pose is required).
- `… test_restock_bt_phases.py test_restock_bt_diagram.py` (with py_trees) — **22 pass** (incl. the
  diagram shows the four new outbound nodes `tow→staging / brake cart / uncouple / drive→service`, and
  the real py_trees tree drives them in series with the mission completing — parity held).
- `… test_restock_mission_tail.py test_navigation_lanes.py test_articulated_navigation.py
  test_restock_mission.py test_isaac_bundle.py` — **all pass** (no regression; default off is
  byte-identical).

## The RETURN is a genuine kinematic hand-off (reported honestly)

A single-doorway store + a forward-only drawbar trailer means a **straight** outbound tow and a
**feasible** cart-home return cannot both hold on the sales floor. Proven on CPU this lane:

- The rollwagen parked **on the portal axis** blocks the free base from threading the single 2.6 m
  doorway to drive home empty — the real grid A* finds **NO PATH** (even at a 0.45 m robot radius).
- A coupled **re-couple + tow-home** from the portal-axis staging does **not expand** (planning
  timeout): the cart cannot reorient 180° in the ~5 m sales-floor corridor south of the wall (the
  dock-align return only reorients because it starts at the endcap, using the wide `y=6` aisle).
- Parking the cart **off** the portal axis to clear the doorway reintroduces 16–20° coupled
  articulation (a forward-only min-1.5 m-radius train cannot tuck 0.8 m laterally cleanly).

So the **return is integration-owned** (it was outside the captain's door-rotation complaint, which is
the outbound). The runtime reuses the existing couple/uncouple mechanics for either option; this lane
set deliberately authors **no** coupled return tow (it would not build-expand). See the fragment's
`_return_handoff_note`.

## Hand-off for the GPU proof (integration-owned)

1. Merge `docs/examples/tow-lanes-mission-fragment.json`: `navigation.tow_lanes`
   (`outbound_dock_to_staging`), `mission_poses.shelf_staging` (+ the existing `shelf_approach`), and
   set the top-level **`shelf_staging: true`** (the bundle reads `config['shelf_staging']`; the pose
   rides in `mission_poses`).
2. **Carton carry (manipulation — integration owns, "do not change manipulation" kept it out here):**
   the robot must CARRY the carton in the gripper through the uncoupled service leg, since the cart no
   longer accompanies it to the endcap. `RESTOCK_TRANSFER_MODE=live_pose` currently picks the carton
   OFF the cart at the service pose, which is incompatible with the cart parked at the staging — wire
   a carried-carton service + GPU-validate.
3. **Return:** choose (a) re-couple at the staging and tow home via the endcap-aisle reorientation the
   dock-align return proved feasible (45.6°), OR (b) leave the cart parked and treat it as a
   known-static for the final leg (the existing docking-distance pattern) + GPU-validate the empty
   return clearance. Both reuse the existing couple/uncouple runtime mechanics.
4. For the from-rest robustness keep `portal_floor_box=true` and the `coupled_heading_hold` /
   `heading_hold_*` keys (unchanged by this geometry change).

**Acceptance (couple + outbound):** the coupled train couples south-aligned on the portal axis and
tows **STRAIGHT** to `shelf_staging` (≤ 10° coupled articulation everywhere, cart crosses the doorway
aligned ≤ 3°, ≥ 0.5 m clearance each side), uncouples there, and the free base drives to
`shelf_approach`; the transfer still succeeds from the uncoupled service pose with the carton carried.
`nav_towing_check` validates the straight staging tow; the solo service drive is the uncoupled base
path.

## Files

- `docs/examples/tow-lanes-mission-fragment.json` — the staging lane + poses + hand-off notes.
- `scripts/twin/restock/runtime.py` — the opt-in `shelf_staging` outbound re-sequence (+ park relax).
- `scripts/twin/restock/bt/{stage_phases,diagram}.py` — the `OUTBOUND_STAGING_PLAN` BT steps + diagram.
- `scripts/twin/restock/isaac_bundle.py` — wires `config['shelf_staging']` + the `shelf_staging` pose.
- `scripts/check_shelf_staging_nav.py` — the uncoupled robot-only path CPU proof.
- `data/sim-shelf-staging-v1/{articulation-before-after,paths-top-down}.png` — before/after plots.
