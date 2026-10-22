# sim-video-cut-qa-v1: frame-exact clean cut ranges for the compiled demo

The captain's ask after watching demo-006 (2026-10-22): cut before the carton tilts, never show the
robot unlatching the rollwagen part-way, and tell the story *warehouse pick → rollwagen → tow to the
store → place → robot + rollwagen back to the warehouse*. This lane only decides **what is safe to
show**. The compile lane does the stitching.

## Deliverable

- **`cut_ranges.json`** (schema `cut_ranges/v1`). For each source and stage it records the
  `stage_span`, the ordered `keep` pieces (inclusive frame indices into `mission-frames/frames.json`
  for real runs, or into `decision_log.json` frames, which equal walkthrough.mp4 frames, for choreo
  dirs), a `cuts` list with a reason for each cut, and every fault found, tagged by channel: `pose`,
  `image`, or `visual` (by-eye). Consecutive keep pieces join without a visible seam because only
  static dead air is removed between them.
- **`visual_faults.json`** lists the faults that no recorded pose can show. The traces carry the
  carton's **yaw only**, with no roll or pitch, so the carton tilt is a by-eye entry. It is frame-exact.
- **Checker**: `python3 scripts/check_cut_ranges.py --spec <compile spec | walkthrough.table.json>`.
  Exit 0 means every segment lies inside one clean keep run of its source. Exit 1 prints each
  violation together with the cut reason. It also validates `cut_ranges.json` itself.
- **Regenerate** (CPU, about 35 s, sources read-only): `python3 scripts/cut_qa_ranges.py --visual
  data/sim-video-cut-qa-v1/visual_faults.json --out data/sim-video-cut-qa-v1/cut_ranges.json`. Rerun
  it once `door-src-001` lands. That source is currently recorded as `missing`.

## Rules (`scripts/twin/demo/cut_qa.py`)

- An ending fault at frame f ends the usable range **8 frames (0.5 s of 15 fps output) before f**.
  A fault window at the start of a stage pushes the start past it instead.
- The **TERMINAL** faults are `carton_tilt` and `unlatch`. After one of them, nothing else in that
  stage is shown, because the carton stays tilted and the robot drives on alone.
- Detectors:
  - pose: base, cart or carton teleport (faster than 2, 2 or 3 m/s between frames); unlatch (the
    measured coupling drops, or the bt `uncouple` step); carton sliding or lifting relative to the
    rollwagen during couple, outbound and park; the scripted transfer moving the carton while the arm
    is still; choreo carton moving while the pad is more than 12 cm away and the carton is not riding
    the cart.
  - image: hard jump (more than 4× the local median); whiteout or black frame.
- A **stall** (no visible change for more than 2 s of output, measured as cumulative grey mean |Δ|
  < 1.0 against the stall's first frame) is spliced, not ended. 0.5 s is kept on each side and the
  join is verified seamless (|Δ| < 1.5).
- Frames map to mission time with the trace clock offset (`world_t − t`): 0.05 s for cold runs,
  **0.55 s for warm runs**. Without it, the warm unlatch lands on #239 instead of the correct #241.

## Ranges (keep pieces; ✗ = nothing safe to show)

| source | stage | span | keep | first fault |
|---|---|---|---|---|
| **collect-src-001** | collect | 0–328 | **0–34 + 78–125** | carton_tilt #134 (2nd face visible; ~90° by #154) |
| **warm11** | return | 225–358 | **225–232** | unlatch #241 (cart unlatched at the bay, robot drives home ALONE) |
| warm11 / warm8 | couple | 0–128 / 0–127 | whole | — |
| warm11 / warm8 | outbound | 129–164 / 128–165 | whole | — (backroom-bay route, not the door) |
| warm11 / warm8 / warm7 | transfer | 166–224 (warm7 –239) | ✗ | carton_floating #168/169 (scripted carton path, arm still) |
| warm8 | return | 225–359 | 225–232 | unlatch #241 |
| **demo-005** | park | 1110–1169 | whole | — |
| **demo-005** | transfer | 1170–1364 | **whole** (pad-held place on the store shelf) | — |
| **demo-005** | verify | 1365–1424 | whole | — |
| **demo-005** | return | 1425–2039 | **1425–1821** (coupled, through the door into the warehouse) | unlatch #1830 |
| demo-005 | outbound | 690–1109 | 690–787 | image_jump #796 → wall clip (#796–829 detected; audit #820–900) |
| demo-005 | couple | 510–689 | 510–636 | 8 cm carton snap #645 |
| demo-005 | collect | 0–509 | 11–61 + 83–456 | image_jump #1; carton teleport #465 (and the audit: it starts with the carton being set INTO the rollwagen, no shelf pick) |
| demo-004 | return | 1425–2039 | 1425–1821 | unlatch #1830 |
| demo-004 | transfer | 1170–1364 | 1180–1272 | image_jump #1171; carton teleport #1281 |
| demo-004 | outbound | 690–1109 | 859–903 | audit wall clip #680–850 (not reproduced here, kept conservatively) |
| run-326 / 325 / 322 | collect | 0–823 / 769 / 781 | about 12–210 in 3 pieces | image_jump #2; partial picks, superseded by collect-src-001 |
| door-src-001 | — | — | not rendered yet | — |

Checked against the current demo-006 table (16:42 build), 3 violations:

- **collect** collect-src-001 0–112 clears the tilt but keeps the 3.9 s static grasp hold. Use
  `0–34` + `78–112` instead.
- **transfer** warm8 164–223 is the floating scripted transfer. Use demo-005 1170–1364.
- **verify** warm8 223 falls within 0.5 s of the floating carton's last motion. Use demo-005
  1365–1424.

The **return** (demo-005 1425–1800) is clean.

## Follow-ups (proper fixes, not in this PR)

1. Record the carton's full quaternion in `nav-trace.jsonl` (`RESTOCK_TRACE_POSES`). That would let
   the tilt be detected from pose data instead of by eye.
2. Re-run collect with a wrist-level carry, so the carton stays level after the lift and the pick
   can continue to the rollwagen place.
3. Add a real (non-scripted) transfer, plus a coupled return to the warehouse with
   `return_mode` set so the robot does not uncouple at the bay. With that, warm runs could supply
   the whole story.
4. Re-run the scanner on `door-src-001` when it lands.
