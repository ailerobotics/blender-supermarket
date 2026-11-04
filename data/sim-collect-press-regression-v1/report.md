# sim-collect-press-regression-v1 — coldwin32 collect-press stall

**Symptom.** coldwin32 (first complete side-bay cold cycle with the night's merges) failed at ~43 s of
mission time in COLLECT, at the pre-weld grasp press:
`MANIP_HOST_FAILED op=collect phase='failed' reason='contact not reached within capped over-press travel'`,
gripper ~35 mm from the carton surface when the 20 mm over-press cap was reached. The previous run
coldwin31 — same stack minus the night's merges — got through this press, welded and reached the place.

**Root cause (named): HOST carton-profile dimension drift in `ros_host_config.json`.**
`git diff 55dde7c..HEAD -- scripts/twin/restock_manipulation/ros_host_config.json` shows the juice_orange
profile `dimensions` changed `0.45, 0.3, 0.18 → 0.6, 0.4, 0.16` (and `"carton_level": true` was replaced
by `"transfer_service_carry": true`). The scenario single source of truth
(`handling_unit.dimensions_m`, PR 268) stayed `0.45, 0.3, 0.18`.

The HOST controller derives the side-grasp PICK pose from `profile.dimensions`: `wall_policy.WallCartonPolicy`
→ shelfbot `motion_planning.geometry.wall_grip` stands the pad off the carton **centre** by
`dimensions[wall-axis]/2 + contact_margin`. The backstock_00 source is an **odd** wall (3), so the standoff
uses `dimensions[1]`:

| dims source | dims[1] | standoff centre→pad (m) |
|---|---|---|
| scenario (correct) | 0.30 | 0.15275 |
| drifted host (HEAD) | 0.40 | 0.20275 |

Over-reach = **0.05 m** (half the 0.10 m dims[1] drift). The capped over-press recovers at most 0.02 m, so
the pad stalls **~0.03 m** short of the real face. The SIM_ATTACH raycast origin sits a few mm behind the
pad, so the measured gap reads a little higher: **GPU measured 34.96 mm (warmacc3), 35 mm (coldwin32)** —
bounded and explained by the ~30 mm standoff residual.

CPU reproduction: `bisect_repro.py` (uses the real `wall_grip`) → `bisect_proof.json`.

## Bisect — what was falsified

| hypothesis | verdict | evidence |
|---|---|---|
| PR 311 `carton_level.gravity_compensation` applied during the press | **FALSIFIED** | Absent from **every** config file in the dataset and from both run logs. Consumed only by `level_place_ik`/`level_preserve_tcp`, reached only **post-weld** (gated on `_carton_up_in_tool`, captured at the first carried pose) and only for place/preplace. The integration worker's **warmacc3** removed it from the HOST config and the press failed **identically** (gap 34.96 mm). |
| `ARM_GRAVITY_FF` (legacy drive FF) raised the arm ~35 mm | **FALSIFIED** | `ARM_GRAVITY_FF held=False ff=[0,33.09,-78.78,-13.66,0,0]` is **byte-identical** in the welding coldwin31 and the failing coldwin32. It is −G at the commanded pose (sign −1): it cancels gravity at the command, it does not offset the pose. Gated by RUNTIME `arm_gravity_ff` (default true), identical in both. |
| PR 312 observe-gate reordering touched the collect pick | **FALSIFIED** | The reordering only defers the **place**-target reach gate when `_repositions_before_place()`; for COLLECT that is `reposition_before_place` (True) in **both** runs. `transfer_service_carry` only changes it for a live_pose **transfer** (`_live_pose_transfer()` is False during collect). It changes no pose. |
| PR 313 warm-reset stage clean released a brake/weld on the cold run | **FALSIFIED** | The reset path runs only **between warm runs**; coldwin32 is a cold first attempt (no reset). No reset/stage-clean line in its log. |
| **HOST carton-profile dimension drift** `0.45/0.30/0.18 → 0.6/0.4/0.16` | **CONFIRMED** | The diff above + the standoff geometry + the GPU 34.96 mm match. Independent corroboration from the integration worker (fm inbox 002). |

**Commanded press-height comparison (CPU).** The commanded side-grasp standoff is **independent** of
every falsified hypothesis and depends only on the host dims: 0.15275 m (scenario) vs 0.20275 m (drifted).
coldwin31 (scenario dims) reached contact at over-press step 4 of 20 mm (min gap 3.46 mm → weld);
coldwin32 (drifted dims) ran all 6 steps to the 20 mm cap and stalled at 35 mm. Same GT pick target
`(-0.028, 1.1, 0.271)`, same base attitude, same `ARM_GRAVITY_FF`.

## Fix (at the source)

1. **`scripts/twin/restock_manipulation/ros_host_config.json`**: juice_orange `dimensions` restored to
   `[0.45, 0.3, 0.18]` (the scenario single source); `"carton_level": true` re-added alongside
   `"transfer_service_carry": true` (both kept). This lands the next cold press on contact and keeps the
   box level through place.
2. **Stop the drift class** — `config_preflight.check_carton_dims(host_cfg, scenario_cfg)` + a new optional
   `--scenario` flag: refuses the GPU boot (exit 2) when any HOST profile dimension disagrees with the
   scenario `handling_unit.dimensions_m`. Backward-compatible (absent `--scenario` = no-op). The
   integration lane adds `--scenario "$M/mission-scenario.json"` to the orchestrate preflight call
   (fragment documents it).

**Never widened:** the `contact_overpress_cap_m` (0.02), the contact threshold, the 2.0 deg carton-level
bound. Out of scope: navigation, towing, transfer.

## CPU proof / tests
- `bisect_repro.py` → `bisect_proof.json` (self-asserting; the real `wall_grip` geometry).
- `scripts/twin/tests/test_collect_press_dims_regression.py`: committed-config pin (dims == scenario,
  carton_level kept); `check_carton_dims` refuses the drift / passes on match / no-op without scenario;
  the side-grasp standoff reproduction (over-reach 0.05 m > 0.02 m cap → ~30 mm residual; 0 with correct dims).
- PR 311's `test_carton_level_southdock.py` (12 tests) stay green — the carton_level/gravity_compensation
  code is untouched (held-gating was explicitly **not** built, per fm inbox 002; noted as a non-hazard
  because gc is already structurally post-weld-only).

## Follow-up
- The place-level `gravity_compensation` tuning is owned by PR 311's fragment; the acceptance enables it.
- If the host is ever to own dims structurally (vs the preflight guard), read them from the scenario
  single source in `ros_host`.
