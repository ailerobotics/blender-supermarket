# Collect-segment source for the compiled demo (sim-video-collect-source-v1)

Goal: deliver one clean COLLECT segment (arm picks the carton off the warehouse shelf and
places it on the rollwagen upper shelf) for the compilation lane `sim-video-compile-v1`.

## 1. Step 1 (CPU scan): no pre-existing clean collect

Scanned all 372 mission runs under `~/datasets/twin-robot-in-store-boot-v1/mission/`
(mission.json + bt_events.jsonl). Across **every** run, bt_events carries **zero**
grasp/place/weld success events — 182 `grasp_failed`, 56 `seal_failed`, 0 success. The only
runs that pass collect do so via the **synthesized start-phase shortcut** (warm1..warm10 /
warm8), which never animates a shelf pick. This matches the audit lane's
`data/sim-video-stage-audit-v1/stage_sources.json` (`collect.clean_source: false`). So no
recorded clean collect existed before the cold run below.

## 2. The cold GT-collect run: what actually happened (diagnosis correction)

The integration lane's cold full run (`mission/cold`, `RESTOCK_COLLECT_GROUND_TRUTH=1`,
walkthrough camera on, 960×720 overhead chase cam = warm8 framing) FAILED at collect. The
field diagnosis relayed was "the grasp attach never engaged (`SIM_ATTACH_DIAG skip=suction_off`
every tick), carton never held". **The logs contradict that.** Evidence
(`cold.console.log`, `cold-manip.txt`):

| What | Evidence |
|------|----------|
| Weld DID form | `SIM_ATTACH_FALLBACK attach body=/RestockScenario/HandlingUnit gap_mm=0.5` (line 8142) |
| Carton WAS held | `COLLECT_GT ... held=True` from calls 1781 → 4111 (234 held=True vs 178 held=False) |
| Carton WAS lifted + tucked | held pose z 0.26 → **1.273** (COLLECT_GT pose) |
| Base carried it to the rollwagen | `NAV_STEP action=collect status=succeeded reason='arrived_stopped'` at place_approach, `transport_ready=True` throughout the carry |
| Then the run died | `MANIP_HOST_FAILED op=collect phase='host' reason='host emergency stop'` the moment the base arrived (sim 100, wall 1131); manip stream's terminal reason `sensor stream watchdog expired` |

The `skip=suction_off` lines are early, throttled approach ticks; the `arm_not_in_transport`
lines are the dying aftermath of the e-stop, not the cause.

The overhead frames (`cold_collect_grid.png`) visually confirm the arm grasping at the shelf,
lifting the carton clear (frames ~#130–160, sim ~44–54 s) and carrying it aloft toward the
rollwagen (~#200–290). **Only the final place onto the rollwagen deck is missing.**

### Real root cause

After the pick+lift, the base does the fm390 reposition carry-drive to `place_approach`
(`pause_station` → drive → re-warm → place). That drive is **render-paced**: ~31 s sim took
~740 s wall (~24×), with the arm station PAUSED (no arm-camera frames). `STATION_STATS` jumps
from tick 4131 (sim 69, epoch 1 `ready`) straight to tick 4141 (sim 100, **epoch 2 `warming`,
`collision_time=-1.0`**). On arrival the host re-warm starts with a stale/empty collision map
while the **wall-based** sensor-stream watchdog (`RESTOCK_SENSOR_STREAM_WALL_BACKSTOP_S=180`)
has long expired → host emergency stop before place. This is the
`freshness-on-sim-clock-not-wall` class (wall timers false-trip under render pacing), applied
to the paused reposition drive — **not** the attach.

The attach-engagement fix (weld at contact pose) is therefore unnecessary: the weld already
fires (gap reached 0.5 mm, pad flush 0.006°, hits on the HandlingUnit) and holds through the
lift and carry.

## 3. Tooling shipped here

`scripts/twin/demo/extract_stage_segment.py` (+ `tests/test_extract_stage_segment.py`):
carves one stage's recorded footage out of a full mission run into a self-contained segment
dir (frames renumbered + `frames.json` with original per-frame metadata, stage-scoped
`bt_events.jsonl` + `nav-trace.jsonl`, and a README naming the source run and original frame
range). Reuses `run_evidence` for the stage↔frame mapping and the frame/trace clock offset.
Validated on the cold run; preserved partial segment at
`mission/collect-src-cold-partial/` (329 frames, status=failed — pick+lift+carry visible,
no place).

## 4. Decision needed (escalated)

- **(A) Deliver the partial pick+lift+carry** from the cold run now (no GPU): visually strong
  shelf pick + lift + carry, but no on-rollwagen place.
- **(B) Fix the reposition-drive watchdog/pacing** (sim-clock-gate the sensor-stream/feedback
  watchdogs through the paused carry drive, and/or speed that render-paced leg) + re-run
  collect-only on the GPU for the full pick→place. Correct to spec; some knobs live in the
  integration lane's external mission config.

Decision (2026-10-22): BOTH, in order. (A) delivered as the interim `collect-src-001`; (B)
below.

## 5. Part (B) fix shipped: suspend the sensor watchdog through the TRANSIT sensing-pause

The per-frame `/sensors` watchdog is **already** sim-clock-gated (`ros_adapter.py` `_watchdog`,
node-clock bound) — so "sim-gate it" is not the fix. It fired because during the fm390
reposition carry-drive the station is **paused by design** (`pause_station`), so `/sensors`
legitimately stops for the whole render-paced drive: the cold run went **~31 s sim / ~740 s
wall** with no frame, which exceeds both the sim per-frame bound (15 s) and the wall backstop
(180 s). The arm's joint feedback stayed fresh (zero `measured joint feedback lost` in the
cold log) and the station was cleanly paused (no heartbeat fault), so the sensor-stream
watchdog was the **sole** trip.

Fix (`ros_adapter.py`): while the controller is in the `transit` phase, **suspend both**
the per-frame sim bound and the wall backstop and keep the packet timers fresh (so resume is
clean), under a generous env-overridable **transit-pause wall budget**
(`RESTOCK_TRANSIT_SENSING_PAUSE_WALL_S`, default 1200 s = `RESTOCK_NAV_GOAL_TIMEOUT_S`) that
still hard-stops a reposition that never arrives. Motion/planning faults above the branch
still surface. `phase=='transit'` only occurs with `reposition_before_place` on, so a default
run never enters the branch (byte-identical). The mission's own nav drive watchdogs
(`RESTOCK_NAV_STUCK_WALL_S`, `RESTOCK_NAV_GOAL_TIMEOUT_S`) remain the authority over the drive.

Tests: `scripts/twin/tests/test_ros_adapter_watchdog.py::TransitSensingPause` (6 new cases,
incl. the exact cold-run 31 s-sim / 740 s-wall trips now suspended, and the budget-exceeded
hard-stop). Full adapter/runtime/manipulation suites green (226 passed, 11 skipped).

### Re-run readiness (knobs for the integration lane)

No new external mission-config key is required: the fix is always-on for the `transit` pause,
the default budget (1200 s) covers the ~740 s drive, and `reposition_before_place` is already
active in the host config (the cold run reached `transit`). Optional: set
`RESTOCK_TRANSIT_SENSING_PAUSE_WALL_S` if a different budget is wanted. Schedule the
collect-only run (`stages=["collect","couple"]`, `RESTOCK_COLLECT_GROUND_TRUTH=1`, walkthrough
camera, warm8 framing) once this fix is merged.

## 6. Re-run outcome (collectsrc, with the PR #187 fix) — fix VALIDATED

The integration lane ran the collect-only re-run (`mission/collectsrc`,
`mission-runtime-collectonly.json`, stages collect+couple, `RESTOCK_COLLECT_GROUND_TRUTH=1`,
walkthrough cam) with the PR #187 TRANSIT-sensing-pause fix. **The fix held:** attempt 1
picked the carton, lifted + tucked, **carried it to the rollwagen** (the old transit
sensor-stream trip is GONE), perceived the deck, and reached the carton over the deck for the
place — i.e. it advanced from "dies in transit" all the way to the PLACE step.

It then failed during the place with a **new, unrelated** blocker: cuMotion place-stage
planning hit **CUDA out-of-memory** (`cumotion::FatalException: CUDA error: out of memory`,
`cuda_array.cu:66`), the host component container died (exit -6), and the station supervisor
heartbeat went silent (30.1 s) → mission failed. This is the known 12 GB VRAM ceiling with
`whispr` (~2.2 GB) holding the margin; render/GI is not the lever (fm373, see
[[mission-vram-ceiling-12gb]]). It is **not** the sensing-pause fix and **not** a pacing
watchdog.

Delivered `collect-src-002` (345 frames; clean cut 0–323: shelf pick → lift → carry → arrive →
reach over the cart deck; the set-down+release is not completed). Carton tilt: none physically
(GT pose ≈ 0°); the overhead-cam skew is perspective. To complete the place: free the VRAM
margin (stop `whispr`) and re-run — an infra/integration action, no code change on this lane.
