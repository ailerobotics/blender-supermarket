# Collect source-observation carton yaw — diagnosis + hardening (run-177 seal-align)

CPU-only, read-only against native captures. Camera lane (`restock_camera` + the assigned
`restock_manipulation/isaac_sensors.py`). No GPU / Isaac / ROS / simulator runs.

## Symptom (relayed from integration + stations lanes)

run-177: the grasp reaches the carton (gap 0 mm) but the suction pad is **~4.2 deg off the
carton side-face normal**, above the gripper plugin's 2 deg seal floor, so it never latches.
Stations lane: the carton's **true** yaw in `run-177/mission.usda` is 0.00 deg (HandlingUnit
`xformOp:orient` ≈ identity), while the grasp yaw = `yaw(source.observation.pose)` came out
~4.2 deg (≈3.6 deg after the 0.6 deg base dock yaw). Candidate pointed at: the collect source
observation (`restock_camera` `source_target`) over-estimates the carton yaw.

## Where the source carton yaw actually comes from

`source_target.build_source_target_provider` passes through `obs.pose`. For the handling unit
that pose's yaw is set in `IsaacCartonSensors.capture` (isaac_sensors.py:380-381) from
`_label_yaw(polygon, depth, intr)` — i.e. the carton yaw is **entirely** the QR-top orientation
estimate; it is not composed with the support registration, and the support markers'
`grid_rot_quarters` (a **shelf-only** design input, config.py:515-539) never enters it. The
`RestockLabel` mesh is axis-aligned on the carton top (no mount rotation), so `_label_yaw`
measures the carton body yaw directly.

## What `_label_yaw` did (before this change)

It used **only the first two** QR corners (TL→TR top edge), each sampled at a **single
integer-rounded pixel's nearest depth**, and returned `atan2` of that one backprojected edge.
No neighbourhood averaging, no depth-discontinuity rejection, no multi-corner consistency check.

## Measured evidence (offline, this lane)

Only native source frames with depth that exist are the static optical bench
`datasets/.../smoke-578-source-01` (4 frames that decode `juice_orange` + both source supports,
same fixed source extrinsic `(-5.73, 17.3, 1.45)` as run-177). run-177 itself persists only the
walkthrough RGB, not the sensing camera depth, so it cannot be re-decoded directly.

1. **Real bench, production decoder + `_label_yaw` (old):** carton yaw deviation from the
   axis-aligned truth = **0.0000 deg** on all four frames. The top edge is image-horizontal with
   equal corner depths (0.7285 m); the dense top plane fit from the real depth is flat to
   **0.001 deg** over 55 647 points (ptp 0 m). So on a clean, well-resolved, flat top the old
   estimator is already exact.
2. **Controlled geometry at the real source view** (synthetic carton at known yaws 0–8 deg,
   analytic flat plane, exact depth + pixel int-rounding): both the old two-corner estimator and
   the new four-corner fit stay within **< 0.2 deg** of truth. Pixel quantisation on a ~240 px QR
   at 0.73 m is worth < 0.1 deg; it cannot, by itself, produce a 3.6 deg bias.

**Conclusion:** the 3.6 deg over-estimate is **not reproducible in the source observation
(map frame)** from any available native evidence, and is contradicted by both the real bench
decode and the controlled geometry. A clean source observation of an axis-aligned carton reads
~0 deg, not 3.6 deg.

### The one real-render mechanism that DOES reach the run-177 magnitude

A single QR corner whose depth is mis-sampled onto a discontinuity — the carton rim or a
gap/edge "bleed" in the RTX depth — rotates the **two-corner** edge estimate directly:

| one-corner depth error | old 2-corner yaw error |
|---|---|
| 2 mm | +0.30 deg |
| 4 mm | +0.59 deg |
| **20 mm (rim/gap bleed)** | **+2.94 deg** |

So the only estimator-side path that produces the observed ~3 deg is a **bad corner depth
sample**, which the old single-pixel two-corner method propagated silently.

## Fix (shipped, isaac_sensors.py `_label_yaw`)

Replace the two-corner single-pixel top edge with a **four-corner planar rigid fit** — the same
orthogonal-Procrustes fit `sensed.register_marker` already uses for supports:

- backproject **all four** decoded corners, each with a **3×3 depth patch** that **rejects a
  corner straddling a depth discontinuity** (`ptp > 15 mm`, as `_corner_points` does);
- require the four map-frame corners to be **coplanar in world Z** (`ptp ≤ 6 mm`, the old
  edge-tilt guard generalised) and to fit a rigid square (residual ≤ 10 mm);
- return the fitted rotation's yaw — the **same** marker-+X map-frame angle as before.

Result (new `_label_yaw`, verified):

- Real bench frames: **0.0018 deg** (was 0.0000) — unchanged in practice, well under 1 deg.
- Synthetic yaws 0–8 deg on a clean plane: **< 0.11 deg**.
- Sub-threshold biased corner (2–4 mm): error **halved** vs the two-corner edge (0.19 vs 0.30,
  0.24 vs 0.44, 0.28 vs 0.59 deg) — four-corner averaging.
- Gross rim/gap mis-sample (20 mm): the old method emits **+2.94 deg**; the new method
  **refuses** (`QR top orientation is uncertain/tilted`). Under the collect occlusion gate a
  refused frame holds the last clean registration rather than publishing a wrong grasp pose.

Identity, tamper, cross-frame/session, freshness (0.5 s), baseline-lifetime and clearance gates
are untouched; this changes only how the carton yaw is fit from the already-decoded, already-
validated corners. On a clean flat top the old and new angles agree to < 0.05 deg, so no existing
behaviour regresses (full twin suite: 861 passed, 0 failed; +13 new carton-yaw tests).

## Honest scope / what this does NOT prove

- I could **not reproduce** a 3.6 deg source-observation bias; the available native evidence says
  a clean source observation reads ~0 deg. This change **hardens** the estimator against the one
  real-render mechanism that reaches that magnitude (a bad corner depth sample) and refuses it
  instead of mis-reporting; it is **not** a confirmed fix for a measured 3.6 deg observation bias.
- If run-177's clean collect frames did **not** have a bad corner (the bench suggests clean, flat
  depth), then the 4.2 deg lives **downstream of the map-frame observation** — most plausibly the
  **map→arm reprojection / extrinsic composition in `isaac_factory`** (integration's lane, not
  this one), the base-dock yaw handling, or a genuine **carton settling** rotation between the
  source observation and contact (mission.usda is the initial composition, not the settled pose
  at observe time).

## Handoff — requested of integration (to localise the residual, next GPU run)

1. Log, for the collect source observation, the **raw map-frame yaw** returned by the provider
   (`yaw(SourceTarget.pose_station)`) **and** the **reprojected arm/world yaw** the grasp actually
   uses, as two separate numbers. If the map-frame yaw is ~0 but the arm yaw is ~3.6, the bias is
   in the reprojection/extrinsic, not this lane.
2. Persist (or log) the settled HandlingUnit world yaw at the observe tick vs at contact, to rule
   in/out carton settling between observation and grasp.
3. With this PR in, re-run: if the pad still sits ~3.6 deg off the side normal, (1)/(2) above
   pinpoint the real source; if the seal now latches, the run-177 clean frame carried a bad corner
   that the four-corner fit rejected/averaged out.

Reproduction scripts (CPU, task scratch): `datasets/sim-restock-camera-v2/yaw-bias/`.

---

# run-182 follow-up — the ~5 deg is the base dock yaw, not the QR fit (2026-10-16)

PR 104 (four-corner fit) did **not** change the seal failure: run-177 was 4.2 deg, run-182 is
**4.99 deg** off the +Y side-normal (SEAL_DIAG `forward=[0.087,0.997,-0.002]`). Integration ruled
out the extrinsic, `wall_grip` and the arm and routed it back to the camera lane. I reproduced the
collect carton-yaw **faithfully at the exact run-182 geometry** and localised it decisively.

## Faithful reproduction at the real mission source geometry (not the mod-90 bench)

`datasets/sim-restock-camera-v2/yaw-bias/repro_mission.py` drives the **production**
`FixedTileQR` decode -> `IsaacCartonSensors._label_yaw` at the real design source camera (2560x1920,
`k_source` fx=fy=2000, `parent_from_optical` with the X-flip + 14.56 deg pitch — identical to the
smoke-578 bench `camera.json`) with the real run-182 `handling-unit-qr.png`, warped onto the
projected carton quad, against an analytic flat-top depth. Reporting the **absolute** yaw (the bench
diagnosis scored `% 90`, which cannot see the design's 180 deg):

* Identity carton (run-182 truth), exact depth: fitted yaw **0.00 deg**; sweeping the true carton
  yaw 0..5 deg, the fit tracks it to **<0.06 deg**. With accurate depth the estimator has **no bias**
  at the mission geometry.
* Depth-error regimes (`repro_mission.py`): a left-right depth gradient — the only shape that rotates
  the XY fit — gives **0.13 deg @ 2 mm, 0.33 deg @ 5 mm, and is REFUSED at >=10 mm** (coplanarity /
  discontinuity guard). Uniform depth bias: **0 deg**. Per-corner +-4 mm noise: **all 30 refused**.
  So the four-corner fit **cannot silently emit a multi-degree yaw**: either it recovers the truth
  (<0.1 deg) or it refuses and the collect gate holds the last clean registration.

Pinned as regressions in `scripts/twin/tests/test_restock_carton_yaw_mission.py` (absolute tracking
<1 deg; >=10 mm gradient refused; identity on a right angle; sub-threshold gradient <0.8 deg).

## The extrinsic the collect capture uses IS live

`capture()` sets `self.transform = rigid(context.base_from_camera)` every frame from the context
provider (isaac_sensors.py:274-277); the collect `source_target` wraps the **same**
`station.adapters[camera_id]` (sensing.py:282). The rig's `base_from_camera` reads
`ur_base_link` **live** off the stage each call (`native_cameras.CameraRig.base_from_camera`,
`inv(world_from_base) @ world_from_optical`). So the observed carton yaw is expressed in the **live
arm-base frame at the capture tick** — the camera lane already accounts for the base pose.

## Decisive: the 4.99 deg equals the run-182 base dock yaw

run-182 `nav-trace.jsonl` shows the base **parked at world yaw -5.0 deg** throughout the collect
grasp window (x=-5.693, y=15.965, `yaw_deg=-5.0`, `parked=true`, sustained wall 485..1218 s) — it
first parked at 0 deg then turned in to -5.0 deg. **Integration's premise that the base dock yaw is
~0.6 deg (used to "rule the base out") is contradicted by run-182's own nav-trace.** The grasp-yaw
error (4.99 deg) equals this dock yaw in magnitude.

For an axis-aligned carton docked at base world-yaw `b`, a correct observation reads `-b` in the
base frame, and a grasp built in that frame and executed relative to the same base cancels `b` back
to world — the seal is on-normal. A residual equal to `b` means the observation base frame and the
execution base frame **disagree by one dock yaw**: the carton-top yaw the grasp consumes is not
expressed in the base pose the arm actually executes against (candidates, all downstream of the QR
fit: the observation held across the base's turn-in so its `base_from_camera` is frozen at the
earlier 0 deg park; or the grasp-pose construction / cuMotion execution base frame in `isaac_factory`).

## Verdict

The carton-yaw **estimator** (`_label_yaw`) is exonerated with direct mission-geometry evidence: it
recovers the true yaw to <0.1 deg and refuses any depth error large enough to bias it. The
deterministic ~5 deg is the **base dock yaw (-5.0 deg)** mishandled between the observation frame and
the grasp-execution frame — not the QR fit, the camera extrinsic (live), `wall_grip`, or the arm.

### Ask to integration (to close it next run)
1. Log, at the collect grasp tick: the **base world yaw** (nav-trace already has it) and the
   **observed carton yaw in the base frame** — set `RESTOCK_SOURCE_YAW_DIAG=1` (this PR;
   `SOURCE_YAW_DIAG obs_yaw_base_deg=...`). If `obs_yaw_base_deg == -base_world_yaw` the observation
   is correct and the bias is purely in grasp construction/execution; if it stays ~0 while the base
   is at -5 deg, the observation was **held across the base's turn-in** (stale base frame).
2. Confirm the base pose `isaac_factory` builds/executes the grasp against is the **same tick** as
   the `base_from_camera` the capture used (no observe-vs-execute base-pose skew while parked).
