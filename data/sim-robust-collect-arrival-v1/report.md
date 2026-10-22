# sim-robust-collect-arrival-v1 — honest collect: tolerant arrival + bounded grasp-press

Lane goal (firstmate): the honest (non-ground-truth) collect reaches a **tolerant, correct arrival**
at the source shelf and the **grasp completes without the host e-stop** — fix the arrival and the
grasp-press (bounded, clear fail reason, no infinite press) **without touching** the sibling lane
`sim-robust-collect-place-v1` (place-on-cart replans). Failure mode: `honest-collect-host-estop`
(rank 4 in `data/sim-robust-audit-v1/failure_modes.json`).

## Evidence (what actually fails, from the logs)

Runs under `~/datasets/twin-robot-in-store-boot-v1/mission/`.

1. **The honest source arrival is already tight.** `run-321` / `run-327` (the real-collect series) both
   park dead-on at the source: `NAV_ARRIVAL_DIAG xy=0.0005 xy_tol=0.080 yaw_err≈0 ... arrived_stopped`
   at `(-5.73,16.0,0)`. So the honest *source* park is not out of tolerance.

2. **The arrival-out-of-tolerance evidence the audit cites is the mid-collect reposition, not the
   source.** `cold.console.log:~20144` `MANIP_HOST_FAILED op=collect ... host emergency stop` is
   preceded by `NAV_ARRIVAL_DIAG xy=0.1691 xy_tol=0.150 yaw_err=0.1977 yaw_tol=0.200 ... latched=True
   pose=(-4.517,15.568,-2.944) goal=(-4.5,15.4,3.142)` with `carry_active=True` — i.e. the base is
   **carrying the carton and repositioning to `place_approach`**, then the press is at the **cart/place**
   (sibling territory). The mechanism is still instructive and generic: with the global XY hysteresis
   latch on (`arrival_xy_hysteresis`, added for the couple standoff in fm482), the base latches the XY
   capture once inside `arrival_xy_tol` and only releases past `arrival_xy_tol + hysteresis`. During the
   slow rotate-in-place the measured point **creeps** from `xy≈0.149` out to `0.169` but stays latched
   (< 0.150 + 0.03), so when the yaw finally aligns it **declares `arrived_stopped` ~19 mm outside the
   true tolerance**. For a grasp-critical park that is exactly the "arrives out of tolerance" defect.

3. **The grasp-press has no sim-clock bound, so it loops until the opaque wall heartbeat.** The honest
   press enables suction and over-presses toward the carton (`SIM_ATTACH_DIAG ... gap_mm` closing
   142→0.67 mm, square-on) but the native suction never seals and no `<=0.5 mm` weld forms. With
   `grasp_retry_max=0` (default) the `grasp` phase then returns **bare controls every tick** forever; the
   only backstops are the coarse `phase_timeout` (120 s SIM) or — under render pacing (wall ≫ sim) — the
   **90 s WALL arm-command heartbeat watchdog**, which fires first with the opaque
   `RuntimeError: arm command heartbeat lost` (`run-321`, `run-327` terminal reason). There is no named
   grasp-press failure: a stuck press is indistinguishable from a dozen other wall-heartbeat deaths.
   (`run-327` additionally never reached the press — the source station ack froze, `STATION_ACK_STALE
   stamp=1.65` — a separate cross-cutting root, rank 5/7, not this lane's.)

## Fixes (both opt-in / no-op by default → byte-identical when off)

### 1. Tolerance-aware ("strict") final approach — `scripts/twin/navigation/core.py`
`Controller.start(..., strict_xy_arrival=False)`. When set, the hysteresis latch can no longer **by
itself** declare arrival: the terminal `arrived_stopped` additionally requires the **true**
`xy <= arrival_xy_tol`, so a leg that has crept back outside tol keeps closing the residual (bounded by
`goal_timeout`) instead of latching "arrived" up to one hysteresis band out. The latch still prevents
dithering during the yaw rotate. No-op when no hysteresis is configured (the bare `xy <= tol` test is
already strict) ⇒ byte-identical.

Wired on the **collect source leg only** (`twin/restock/runtime.py` `_start_nav_leg('pickup',
strict_xy_arrival=True)`); threaded through `NavigationBridge.start` and `ArticulatedController.start`
(default False elsewhere, so the couple standoff / sibling legs are untouched). The uncoupled source leg
uses `core.Controller.step` arrival; coupled legs use the articulated path, which is left alone.

### 2. Sim-clock bounded grasp-press — `scripts/twin/restock_manipulation/controller.py`
`TransferController(..., grasp_press_timeout_s=0.0)`. When > 0, a **SIM-clock** budget bounds the WHOLE
unconfirmed pre-weld press (across the contact descent and the grasp confirm/retry loop): the clock
latches once the press begins and clears only on a **confirmed held carton** or when the phase leaves the
press, so retries cannot re-arm it. On exceeding it the collect **fails fast with a named reason**
(`grasp press exceeded <N>s sim without a confirmed held carton ... failing fast before the arm-command
heartbeat watchdog`) instead of burning wall budget to the opaque heartbeat. Sim-clock (not wall) per
"Freshness on sim clock, not wall" — the existing `contact_stuck_wall_s` is wall-based and only covers
the contact phase; this covers the whole press and is render-pacing-safe. Clamped to `<= phase_timeout`.
Wired from `config['grasp_press_timeout_s']` in `ros_host.py`. Default 0 ⇒ off ⇒ byte-identical.

## Tests (CPU, offline)
- `scripts/twin/tests/test_collect_arrival.py` — strict arrival refuses to declare arrival outside the
  true tol, still arrives within tol, is a no-op without the hysteresis latch, and defaults off.
- `scripts/twin/tests/test_restock_manipulation.py` — `test_grasp_press_sim_budget_fail_fast`,
  `..._default_off_is_byte_identical`, `..._is_not_rearmed_by_retries`.
- `scripts/twin/tests/test_restock_runtime.py` — `test_collect_source_leg_requests_tolerance_aware_strict_arrival`
  pins the wiring (collect requests it, outbound does not).

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH= python3 -m pytest scripts/twin/tests/test_collect_arrival.py
scripts/twin/tests/test_restock_manipulation.py scripts/twin/tests/test_restock_runtime.py -q`.

## GPU validation handoff (integration lane, through firstmate)
Bounded **collect-only** proof, no ground-truth shortcut:
`scripts/run_mobile_restock.py ... --stop-after collect` with `RESTOCK_COLLECT_GROUND_TRUTH` **unset** and
`grasp_press_timeout_s` merged into the mission runtime-config. `--stop-after collect` ends right after the
pick, so the sibling's place-on-cart replan / cuMotion OOM is never reached. Config + expected PASS/FAIL:
`docs/examples/collect-only-validation-fragment.json`. (The mission runtime-config under `~/datasets` is
owned by the integration lane and not edited here.)

PASS = honest collect reports SUCCEEDED, or fails with the named `grasp press exceeded ...` reason, with
**no** `arm command heartbeat lost` / `host emergency stop` terminal line and NAV `arrived_stopped` at the
source within the grasp xy/yaw tolerance.
