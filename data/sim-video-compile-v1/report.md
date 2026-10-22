# sim-video-compile-v1 — the demo-006 compile tool

Stitch one continuous seven-stage restock demo out of per-stage segments drawn from **different**
runs, so each stage is shown from a source where it worked, with the decision/reason-tree panel,
the mission clock and the stage ladder running **continuously across the source seams**.

## What it is

- **`scripts/compile_demo_video.py`** — CLI. Reads a compile spec JSON (ordered per-stage segments),
  writes `<out>.mp4` + `<out>.table.json` (the stage-to-source provenance table). CPU only; every
  source run dir is read-only.
- **`scripts/twin/demo/stitch.py`** — the engine. Reuses the demo panel
  (`twin.demo.panel.render_panel`) and the two per-frame panel paths already in the tree:
  - `real` source (a `run_mobile_restock.py` run dir): the panel entry is rebuilt per frame by
    `twin.demo.bt_timeline`; the scene is the recorded 960×720 overhead frame under a freshly drawn
    caption band.
  - `choreo` source (a choreographed demo dir, e.g. `demo-004/005`): the panel entry is the dir's
    own `decision_log.json` per-frame payload; the scene is the left region cropped out of the
    composited `walkthrough.mp4` (its band is already baked in).

  Both reduce to the same `entry = {tree, header, chips}`. The engine then **rewrites each entry for
  global continuity**: the ladder reads `done` for stages shown earlier in the compile and `idle`
  for later ones (no matter which run a frame came from), the active stage's own node / measured
  chips / scene pixels pass through untouched, the clock is cumulative and the mission reads running
  until the final frame. Short cross-fades are dropped **only at source boundaries** (consecutive
  segments from the same run stay seamless). Output 1920×1080 H.264 via the `imageio_ffmpeg` binary.

## The spec

`data/sim-video-compile-v1/compile-demo-006.json` — mission order, one segment per stage, each naming
`source` (run dir under `mission_root`), inclusive `frame_start`/`frame_end`, optional `kind`
(auto-detected), `min_seconds` (hold a sub-second stage), `labels` (honest provenance tags on the
band). Ranges come from `data/sim-video-stage-audit-v1/stage_sources.json`.

## v1 result (2026-10-22)

`mission/demo-006/walkthrough.mp4` — 32.3 s, 485 frames, 1920×1080, 15 fps; clock continuous
2.1→32.3 s, stage numbers 1→7, 0 dim/black frames. Story (per the captain's 2026-10-22 review):
warehouse pick onto the rollwagen → tow out → place → tow the rollwagen **back to the warehouse and
park**. **Cut rule:** every stage ends before any wrongly tilted carton / clip / jump / mid-way
uncouple.

| stage | source | note |
|-------|--------|------|
| collect | collect-src-001 f0–112 | real shelf pick; **cut at the clean grasp**, before the lift pitches the carton off level (tilt starts ~f120) |
| couple | warm8 f0–126 | real couple |
| outbound | warm8 f126–164 | **INTERIM** clean backroom drive (no wall clip) → door/front pass pending `door-src-001` |
| park | warm8 f164 | 1-frame transition, held 1.2 s |
| transfer | warm8 f164–223 | **INTERIM** scripted still-arm place at the backroom bay; front-shelf place pending (story-sources) |
| verify | warm8 f223 | ground-truth, held 1.2 s |
| return | demo-005 f1425–1800 (stride 3) | coupled tow from the front **back to the warehouse + park**; **cut before the uncouple** at f1830 |

Interim slots swap via a one-line spec edit as helpers land: `outbound ← door-src-001`,
`transfer` (front-shelf) + a native coupled return `← sim-video-story-sources-v1`, clean per-segment
cut ranges `← sim-video-cut-qa-v1`. Known interim discontinuity: transfer is still the backroom bay
while the return starts at the supermarket front (front-shelf transfer pending).

Per-segment knobs the compiler honours (all in the spec): `stride` (time-lapse a slow stretch),
`min_seconds` (hold a sub-second stage), `caption`/`title` (override the band wording, e.g. drop
"uncouple" on the return), `band` (draw a caption band; default real=yes, choreo=no).

`collect-src-001` is a carved single-stage clip (no `mission.json`); `stitch.load_run_lenient`
assembles it into an equivalent run (empty event log, panel from its own `bt_events`).

## Tests

`scripts/twin/demo/tests/test_stitch.py` (CPU, offline): continuity rewrite, spec parse/detect,
same-source-no-crossfade, min-seconds hold, and real + choreo end-to-end encodes. Run with
`PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=scripts` under an env that has ffmpeg
(`env_isaaclab`); encode tests skip when ffmpeg is absent.
