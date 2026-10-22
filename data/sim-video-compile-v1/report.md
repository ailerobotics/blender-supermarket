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

`mission/demo-006/walkthrough.mp4` — 45.9 s, 689 frames, 1920×1080, 15 fps; clock continuous
1.4→45.9 s, stage numbers 1→7, 0 dim/black frames. Spine:

| stage | source | note |
|-------|--------|------|
| collect | collect-src-001 f0–298 | real shelf pick + lift + carry to the rollwagen (cold GT run); **no place yet** — full pick→place re-run pending (part B of the collect lane) |
| couple | warm8 f0–126 | real couple |
| outbound | warm8 f126–164 | **INTERIM** clean backroom-bay drive (no wall clip, but not the door) → `mission/door-src-001` |
| park | warm8 f164 | 1-frame transition, held 1.2 s |
| transfer | warm8 f164–223 | scripted still-arm (sim weld) |
| verify | warm8 f223 | ground-truth, held 1.2 s |
| return | warm8 f223–358 | real return home |

One slot (`outbound`) is still interim; swapping it is a one-line edit of the spec once the door
helper run lands. Collect is a real pick/lift/carry but stops before the place (the cold run died on
a watchdog e-stop at arrival); the on-screen label says so. No run drives the door yet (see the
audit); those are honest, labelled gaps, not hidden.

`collect-src-001` is a carved single-stage clip (no `mission.json`); `stitch.load_run_lenient`
assembles it into an equivalent run (empty event log, panel from its own `bt_events`).

## Tests

`scripts/twin/demo/tests/test_stitch.py` (CPU, offline): continuity rewrite, spec parse/detect,
same-source-no-crossfade, min-seconds hold, and real + choreo end-to-end encodes. Run with
`PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=scripts` under an env that has ffmpeg
(`env_isaaclab`); encode tests skip when ffmpeg is absent.
