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

## Visual audit (OpenCV frame grids; ffmpeg absent)

**warm8** (top-down walkthrough, 360 frames) — **visually CLEAN**. Across
couple (0–126) → outbound (126–164) → transfer (164–223) → return (223–358):
no wall or shelf clipping, no teleports/pose jumps, carton tracks the cart and
is set at the bay target by ~#250. This is the safe spine for the compiled video.
Collect was shortcut, so no pick is shown.

**demo-005** (2040 frames, with decision-tree panel) — two confirmed faults, both
matching the captain's notes:
- **Starts at placement** (#0–~60): opens already lowering the carton into the
  rollwagen; no warehouse-shelf pick is shown.
- **Outbound clips THROUGH the backroom wall** (~#820–900): frames whiteout as the
  camera/robot passes through solid wall geometry — backroom shelving before,
  shop-floor aisle after. It does not pass the door.

**demo-004** (2040 frames) — same choreography family, same two faults: starts at
placement; outbound wall clip-through (~#680–850).

**run-326** (real collect, 824 frames, later fails arm-heartbeat) — the arm reaches
the warehouse shelf and **grasps the carton off it** (#160–240), swings it down
(#320–400). This is the pick the captain wants — but after ~#420 the base
destabilizes and the carton is **never cleanly deposited in the rollwagen**; usable
as a **pick-only partial (≈#0–400)**.

## What the compilation lead (sim-video-compile-v1) must know

- **collect**: NO clean source. Best partial is `run-326` ≈#0–400 (pick off shelf
  only); the deposit-onto-rollwagen clip **does not exist** — re-run or author it.
- **outbound**: NO door passage anywhere. `warm8` 126–164 is a clean backroom-bay
  drive (wrong route); the demos clip the wall — **do not use the demos for
  outbound**. A pre-planned door pass must be authored/re-run as a separate clip.
- **couple / transfer / verify / return**: use `warm8` (ranges above); clean.
  park/verify are 1-frame transitions — fold them into the neighbouring beat.

Reproduce grids: `/tmp/grid.py` (png dir / mp4) and `/tmp/grid2.py` (panel-cropped).
