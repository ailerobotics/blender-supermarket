# Decision: `lidar_sensor_yaw_fix` — flip ON by default; retire `tow_lane_trust_static`

**Task:** sim-restock-lidar-yaw-fix-eval (vr1 follow-up). **Branch:** `fm/sim-restock-lidar-yaw-fix-eval`
cut from `fm/twin-robot-in-store-boot-v1` head. **Date:** 2026-10-22.

**Decision: the fix is correct — flipped ON by default in code.** The lidar `sensor_to_base` yaw is now 0
unless a config explicitly sets `lidar_sensor_yaw_fix: false` (backward-compatible escape hatch to
reproduce a pre-fix recorded run). The fix removes the sole source of the tow-lane "proud phantom"
returns, so the `tow_lane_trust_static` waiver is proposed for retirement (pending the bounded GPU run);
`tow_lane_static_tol_m` and `tow_lane_trust_unknown` stay (their jobs are independent of the yaw bug).

---

## 1. The bug (CPU, no GPU)

`twin.navigation.sensors.Scan.sensor_to_base` is `(x, y, YAW)` — the sensor origin x/y in `base_link`
plus the sensor's **yaw** offset about `base_link`. `isaac_bundle` (legacy wiring) passed the lidar URDF
`lidar_frame` origin `(0.52, 0.0, 0.15)` straight in, so the sensor's **Z = 0.15 m mount height** was
read as a **0.15 rad = 8.59° yaw**. The rays are cast at the true base yaw (`read()`), but
`observation()` reconstructs every return through `Scan.points` → `sensor_pose` = `yaw + da`, i.e. rotated
8.59° off its cast bearing.

The `lidar_frame` joint is **`fixed` with `rpy="0 0 0"`** (verified on the warm8 URDF and the generator
`mobile_assets.build_mobile_urdf`): the frame has **zero physical rotation**, so the correct offset yaw is
exactly 0. Reading the z as yaw is simply wrong — a pure correctness defect, not a tuning choice.

**Proud depth** (`scripts/check_lidar_yaw_fix.py`, synthetic wall, vr1 wiring verbatim):

| wall range | legacy x-span (reconstructed) | proud depth | fixed |
|---|---|---|---|
| 1.0 m | 0.31 … 1.68 | **0.69 m** | 1.00 (exact) |
| 2.0 m | 1.26 … 2.71 | **0.74 m** | 2.00 (exact) |
| 3.0 m | 2.25 … 3.70 | **0.76 m** | 3.00 (exact) |
| 5.0 m | 4.21 … 5.69 | **0.79 m** | 5.00 (exact) |

A flat surface reconstructs as a surface tilted 8.59° about the sensor; its near end swings ~0.15·r
**toward the robot**, straight into the forward braking capsule and the tow-lane swept envelope.

## 2. Effect on the recorded runs, and the change the fix makes (CPU)

`scripts/check_lidar_yaw_fix.py` replays the **recorded coupled-tow drive poses** (couple/outbound/return)
of the complete warm8 and warm11 baselines through the **same production `IsaacRangeSensor` +
`observation` wiring** over a box ray-caster of the mission's own static map (the 806-AABB nav grid lifted
to 3-D), legacy vs fixed. Because the ray-caster geometry is the AABB map `near_static` measures against, a
correctly-wired return lands on its source AABB (distance ≈ 0); the legacy rotation is what pushes it off.

| metric (per run) | warm8 legacy | warm8 fixed | warm11 legacy | warm11 fixed |
|---|---|---|---|---|
| returns reconstructed | 25,203 | 25,203 | 24,880 | 24,880 |
| **"kept" returns > 0.12 m from any mapped AABB** (the class that forces `trust_static`) | **11,956** | **0** | **11,781** | **0** |
| **forward-capsule `obstacle_stop` poses** (capsule-vs-planner deadlock) | **9 / 153** | **0 / 153** | **9 / 151** | **0 / 151** |
| outbound lane min clearance | 0.75 m | **1.02 m** | 0.79 m | **1.02 m** |
| return lane min clearance | 0.74 m | **0.90 m** | 0.77 m | **0.90 m** |

~47 % of all returns are phantoms under the legacy wiring; **0** under the fix. The ~0.83 m braking
capsule (`stale_after` 3.0 s at 0.25 m/s) deadlocks against a lane clearance eroded to ~0.75 m; the fix
restores the true ~1.0 m clearance, clear of the capsule — the 9 capsule-stop poses go to 0. Full JSON:
`replay-warm8.json`, `replay-warm11.json`.

## 3. Waiver decision (CPU, through the real `ArticulatedController`)

`check_lidar_yaw_fix.py::check_waivers` replays the real reconstructed clouds through
`ArticulatedController.plan_observed_route` on warm8's authored outbound lane:

- **LEGACY** cloud at the recorded return pose `(-0.63, 13.87)` → `lane_blocked` with `trust_static` OFF
  (66 rotated phantom returns seed-collide the lane) — reproduces the live run-321b/322b block.
- LEGACY + `tow_lane_trust_static: true` → `validated_lane` — the waiver's compensation.
- **FIXED** cloud at the same pose → `validated_lane` with `trust_static` **OFF** — the waiver is no
  longer needed. Every legacy-blocked pose validates under the fix without `trust_static`.
- SAFETY: fixed + a genuinely new obstacle on the lane centre → still `lane_blocked`; the live
  direction-aware braking gate is untouched and still stops the tow for a new obstacle ahead.

**`tow_lane_trust_static` was compensating for the yaw bug.** Propose **removing** it from the mission
runtime config once the bounded GPU run (fix on, `trust_static` off) confirms no residual block.

**Keep these — the yaw fix does not make them unnecessary:**
- `tow_lane_static_tol_m` (proud tolerance, 0.12): the yaw proud was ~0.75 m, far beyond 0.12, so the tol
  never covered it anyway (that is why `trust_static` existed). With the fix the residual proud is only a
  few cm of finer-than-AABB geometry — exactly the tol's original, small, safe job. Keep as-is.
- `tow_lane_trust_unknown` (the **coverage waiver**): waives the *unobserved-space* refusal during the
  turn-in-place, where a forward-facing lidar cannot pre-observe the ~90° arc (fm437/fm439/run-328 — a
  sensing-geometry deadlock). This is independent of the yaw rotation; keep it.

> **Model caveat (honest):** the box ray-caster carries only the 806-AABB map, so it cannot see
> finer-than-AABB sim geometry. It proves the fix removes the **rotation-induced** phantom class entirely
> (fixed kept = 0); it cannot rule out a small residual *unmapped-static* return that `trust_static` would
> still catch. Hence `trust_static` retirement is gated on the GPU run, not asserted from CPU alone.

## 4. What shipped on this branch

- Default flipped ON: `isaac_bundle.lidar_sensor_to_base` (new pure, test-locked resolver),
  `tail_nav.simulate_home_leg` / `check_home_leg`, `ground_truth_verify.mission_modes` (now records only
  the *exceptional* legacy wiring as provenance). `lidar_sensor_yaw_fix: false` still selects legacy.
- `scripts/check_lidar_yaw_fix.py` — the CPU proof above (runs in ~25 s on warm8).
- Regression lock: `test_restock_mission_tail.py::test_lidar_sensor_frame_orientation_locked_and_fix_is_default`
  asserts the generated `lidar_frame` joint is `fixed`/`rpy 0` with a nonzero z, and that the production
  resolver yields yaw 0 by default (explicit false → legacy).
- `docs/examples/mission-tail-fragment.json` — updated for the default + the `trust_static` retirement.
- `gpu-validation.md` — the one bounded GPU config for the integration lane.

Run command for §1–§3 is in `gpu-validation.md` and the script's header.
