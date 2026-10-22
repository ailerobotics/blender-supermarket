# sim-video-stage-audit-v1

Stage-by-stage source audit for the compiled restock demo (demo-006, owned by
`sim-demo-video-v1`). Goal: for each of the 7 mission stages — **collect, couple,
outbound, park, transfer, verify, return** — name the best available source run,
with frame ranges and faults. CPU only, no rendering, no GPU.

Machine-readable output: [`stage_sources.json`](stage_sources.json).

Mission root: `/home/manishsaini/datasets/twin-robot-in-store-boot-v1/mission/`.
Videos served at `http://100.100.47.65:8787/v/mission/<run>/walkthrough.mp4`.

## Headline findings

1. **No frame-recorded run ever completed COLLECT.** Every run that performed a
   *real* collect (not the start-phase shortcut) and recorded frames **failed
   mid-pick** — host e-stop (run-309/310/313–316), collision planning
   (run-311/312), **arm command heartbeat lost** (run-321→335, deepest: reached
   grasp/weld), or **StationAckStreamStalled** (run-336→348, source camera never
   decoded the carton). `collect_succeeded` appears in **zero** runs with frames.
   → The captain's "successful pick onto the rollwagen" clip **does not exist** and
   must be re-run or stitched from a partial pick with the failure masked.

2. **warm8 is the only COMPLETE cycle** (`state: complete`, real frames). It
   covers **couple, outbound, park, transfer, verify, return** cleanly — but with
   caveats: collect was a **start-phase shortcut** (synthesized, not performed);
   transfer was a **scripted still-arm** move with a SIM weld attach (not
   cuMotion-planned, not a physics grasp); verify was **ground-truth** (not a
   perception verify). park and verify are effectively 1-frame state transitions.

3. **OUTBOUND has no door passage.** warm8's outbound takes the **backroom-bay
   route** (straight east from the dock), which is clean but is *not* the door the
   captain wants shown. The choreographed demos (demo-004/005) clip **through the
   backroom wall**. The real planner needs a ~360° loop at the portal to pass the
   door, so no run drives cleanly through it — a pre-planned door pass must be
   authored/re-run as a separate clip and spliced in.

## Per-stage summary

| Stage | Clean source? | Best run | Frames | Note |
|-------|---------------|----------|--------|------|
| collect | **NO** | run-326 (partial) | 0–823 | all real-collect runs fail mid-pick; re-run needed |
| couple | yes | warm8 | 0–126 | drive→dock→attach, coupled=true |
| outbound | **NO (wrong route)** | warm8 | 126–164 | backroom-bay, not the door |
| park | yes | warm8 | 164 | 1-frame transition |
| transfer | yes (scripted) | warm8 | 164–223 | scripted still-arm, SIM weld |
| verify | yes (ground-truth) | warm8 | 223 | 1-frame transition |
| return | yes | warm8 | 223–358 | real return home, 136 frames |

## Method

- **Inventory** (`/tmp/inventory.json` build logic): every run dir under the
  mission root with `mission-frames/frames.json` (93 dirs). Per stage, outcome
  parsed from `mission.json` `stage_timeline` (warm*) or `events` job-state stream
  (run-3xx); sim-time windows mapped to nearest frame index via each frame's
  `simulation_time`.
- **Visual audit** (v2, below): OpenCV frame grids of demo-005, demo-004, warm8
  and the top collect candidates — wall/shelf clipping, pose jumps/teleports,
  floating/missing carton, arm-through-shelf, panel mismatch.

## Visual audit

_v2 — in progress; faults recorded per stage in `stage_sources.json`._
