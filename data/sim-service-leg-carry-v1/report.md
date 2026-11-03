# sim-service-leg-carry-v1 — the solo carton-carry service leg (manipulation half of shelf-staging)

**Branch:** `fm/sim-service-leg-carry-v1` → PR against `fm/twin-robot-in-store-boot-v1`.
**Scope:** the opt-in TRANSFER **service leg** that completes the sim-shelf-staging-v1 re-sequence —
pick the carton off the parked cart at the staging pose, **carry** it to the endcap held level, place
it, drive back to staging. Behaviour-tree steps + runtime legs + a narrow opt-in controller change,
**CPU-proved**. Opt-in under the existing `shelf_staging` switch (default off ⇒ byte-identical). GPU
proof (coldwin24) is integration-owned (hand-off below). **Out of scope** (captain): the cart RETURN to
the backroom (separate decision), portal/dock geometry, carton size.

## Captain's intent → what this adds

sim-shelf-staging-v1 (PR 274) re-sequenced the **nav** so the coupled train tows straight down the
portal axis to the `shelf_staging` pose and uncouples there — removing the last coupled rotation. What
PR 274 left wired as a placeholder was an **empty** `drive→service` leg to the endcap; it never picked
the carton off the cart. With the cart parked at staging (not at the endcap), the robot must **pick at
staging, carry the carton to the endcap, place it, and drive back** — the manipulation half.

The carton must stay **parallel to the ground** the whole carry (captain's standing rule): PR 270's
`carton_level` / `CARTON_TILT` trip stays on while driving, and the carry-drive uses a reduced
speed/accel profile so the base cannot shove the carton.

## The service leg (opt-in, default off ⇒ byte-identical)

The TRANSFER stage, when `shelf_staging` is active, decomposes into **seven position-first
behaviour-tree steps** (navigate → perceive → manipulate, one leaf each — the diagram shows them):

| # | step (BT node) | body | kind |
|---|---|---|---|
| 1 | `perceive carton` | warm the collision map + sense the carton on the parked cart at staging | perceive |
| 2 | `transfer-pick@staging` | grasp · lift · fold to the level carry-hold (reuse the transfer pick) | manip |
| 3 | `carry→endcap` | carry-drive the held carton to the endcap (`carton_level` + payload profile) | nav |
| 4 | `perceive slot` | re-warm + re-register the endcap slot target in the new base frame | perceive |
| 5 | `place@endcap` | lower + release (reuse the transfer place) | manip |
| 6 | `verify placed` | the controller's internal placement verify | verify |
| 7 | `drive→staging` | return the empty base to the staging pose beside the parked cart | nav |

OUTBOUND drops the empty `drive→service` leaf — it now ends `tow→staging · brake cart · uncouple`,
leaving the base **parked at staging** for the service leg (before/after diagram:
`bt-transfer-before.svg` / `bt-transfer-after.svg`).

### Reuse, not reimplementation

The carry-drive reuses fm390's **TRANSIT carry-hold handshake** — the exact mechanic the collect
place-on-cart already uses: the arm folds to `tuck` and HOLDS in `transit`; the runtime pauses the arm,
drives the base to the endcap, then signals `base_repositioned` so the controller re-registers the
sensed endcap slot in the new base frame (`_reregister_place`) and resumes `preplace → place`. The one
narrow controller change is opt-in: `transfer_service_carry` makes `_repositions_before_place()` True
for a **live_pose** transfer (which otherwise goes `tuck → preplace` with no carry-hold). The pick and
place are the existing live_pose transfer pick + place, re-targeted by the base move.

`carton_level`'s `CARTON_TILT` trip already covers `transit`/`preplace`/`place` for any op (keyed on
held + phase, not action), so the carried carton is level-governed through the whole carry with no
action gate change.

## Perception (the pick capture gate)

The pick at staging uses the **same** capture gate as the current transfer pick — the FIXED
`source`/`shelf` station cameras and the ground-truth / live-pose gate (`transfer_ground_truth.enabled`
/ `IsaacCartonSensors.capture`), NOT the arm eye-in-hand camera (masked-depth only, does not frame the
pick). `RESTOCK_TRANSFER_MODE=live_pose` picks from the **live sensed carton pose**, so it IKs to
wherever the cart is parked. The carton + its markers are in frame at the station camera per the PR 273
carton-0.45 audit (`shelf` camera: 9.9 px/module, 35.9° incidence, all markers in FOV; loaded-carton
all-in-FOV, 136 px edge margin; carton-top-on-shelf 0.91 m under the 0.97 m tier).

## CPU proof (no GPU lock)

`scripts/check_service_leg_carry.py --environment <env.json> --scenario scripts/restock_scenario.json
--lanes docs/examples/service-leg-carry-mission-fragment.json` → **SERVICE_LEG_CARRY PASS** (real
798-obstacle static map, real `twin.mobile` envelopes, real `carton_level` FK):

| check | result |
|---|---|
| **carry→endcap clears with the carton** | base ∪ 0.45×0.30 m carton swept staging→endcap: **0.15 m** static clear, **0.05 m** parked-cart clear |
| **drive→staging clears (empty)** | 0.44 m static, 0.05 m parked-cart clear (orients in the open aisle, not against the cart) |
| **pick@staging within reach** | **1.204 m** ≤ 1.30 m (margin 0.096) — the carton on the parked cart's upper shelf, the SAME base-relative geometry coldwin22's transfer picked at |
| **place@endcap within reach** | **1.217 m** ≤ 1.30 m (margin 0.083) — the proven coldwin22 endcap slot (G3_back_s2) |
| **carried carton stays level** | leveled max tilt **0.02°** ≤ 2.0° across every carried fold (vs up to 40° unleveled) — the real `carton_level_bench` on the mission URDF |

Plots: `service-leg-carry-path.png` (top-down swept base∪carton along the round trip),
`carton-tilt-before-after.png` (unleveled vs leveled carried tilt). Machine numbers:
`service-leg-carry.json`.

### The drive-with-payload profile

The carry→endcap base command is clamped to `carry_drive_profile` (default `max_speed 0.12 m/s`,
`max_angular 0.20 rad/s`, `max_accel 0.15 m/s²`, `max_angular_accel 0.40 rad/s²`) and ramped from rest,
so the base cannot shove the held carton; `CARTON_TILT` stays the attitude trip. Only the carry leg is
clamped — every other leg keeps its normal profile.

### Unit tests (all pass)

- `test_restock_bt_phases.py` / `test_restock_bt_diagram.py` — the 7-step service-leg plan + diagram +
  **byte-identical state-machine parity** (the tree drives the same `step_stage` sequence); OUTBOUND is
  now the 3-step staging plan; flag-off is byte-identical.
- `test_restock_runtime.py` — the runtime pick→carry→place→drive-back legs (ends at staging, evidence
  re-stamped to the arrival packet), the `carry_drive_profile` resolver + the speed/accel clamp, and
  flag-off single-pose transfer unchanged.
- `test_restock_transfer_live_pose.py` — `transfer_service_carry` produces the TRANSIT carry-hold for a
  live_pose transfer (and is off ⇒ `tuck → preplace`, byte-identical); collect unaffected.
- `test_service_leg_carry.py` + `test_carton_level_bench.py` — the carry/reach/tilt proof guards.
- No regression: `test_restock_mission*.py`, `test_isaac_bundle.py`, `test_restock_transfer_carry.py`,
  `test_restock_manipulation.py`, `test_config_preflight.py` all pass.

## Hand-off for the GPU proof (coldwin24, integration-owned)

Config: **`docs/examples/service-leg-carry-mission-fragment.json`** (the full re-sequenced-cycle config,
with the HOST / RUNTIME / ENV split made explicit so a key never lands in the wrong file). In short:

1. **RUNTIME** (mission runtime-config): `shelf_staging: true`, the `navigation.tow_lanes`
   `outbound_dock_to_staging` + `mission_poses` (`shelf_staging`, `shelf_approach`) from
   `docs/examples/tow-lanes-mission-fragment.json`, and (optional) `carry_drive_profile`.
2. **HOST** (`restock_manipulation/ros_host_config.json`): `transfer_service_carry: true` +
   `carton_level: true`. (`config_preflight` registers both sides; a split is the run-277 trap.)
3. **ENV** (both processes): `RESTOCK_TRANSFER_MODE=live_pose`, `RESTOCK_TRANSFER_CARRY=wrist_guard`.
4. Keep the sim-shelf-staging-v1 GPU keys (`portal_floor_box=true`, `coupled_heading_hold` /
   `heading_hold_*`) and the couple-approach station geometry unchanged.

**Acceptance:** pick at staging completes; the carton stays level through the drive (no `CARTON_TILT`);
place at the endcap completes; the robot ends back at the staging pose beside the parked cart. The BT
diagram shows the seven service-leg nodes.

**Follow-ups:** the cart RETURN from staging (captain decision — see tow-lanes fragment
`_return_handoff_note`); GPU-tuning the `carry_drive_profile` against the measured descent reaction; the
reach margins (~0.08–0.10 m) are tight-but-feasible and match coldwin22 — if a station re-measure shifts
the cart, a small `transfer_pick_approach` reposition pose is the lever (not needed by the CPU geometry).

## Files

- `scripts/twin/restock/bt/stage_phases.py` — `TRANSFER_STAGING_PLAN` + milestone; OUTBOUND drops `service`.
- `scripts/twin/restock/bt/diagram.py` — the transfer service-leg branch.
- `scripts/twin/restock/runtime.py` — the service-leg legs (pick→carry→place→drive-back) + the carry profile.
- `scripts/twin/restock_manipulation/controller.py` — opt-in `transfer_service_carry` (TRANSIT for live_pose).
- `scripts/twin/restock_manipulation/ros_host.py` / `config_preflight.py` — wire + register the keys.
- `scripts/twin/restock/isaac_bundle.py` — pass `carry_drive_profile` to the adapter.
- `scripts/check_service_leg_carry.py` — the CPU path/reach/tilt proof.
- `docs/examples/service-leg-carry-mission-fragment.json` — the coldwin24 config hand-off.
- `data/sim-service-leg-carry-v1/{report.md, bt-transfer-{before,after}.svg, service-leg-carry-path.png,
  carton-tilt-before-after.png, service-leg-carry.json}`.
