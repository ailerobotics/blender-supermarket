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
