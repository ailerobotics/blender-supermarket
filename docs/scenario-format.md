# Store-vision scenario dataset format (v1.0)

Labelled synthetic sequences rendered from the generated store for **store-vision**
development and evaluation — shelf depletion, floor spills/hazards, people movement,
misplaced products, and multi-view RGB-D shelf capture. Generated headless by
`scripts/scenarios/generate.py` in a **separate** Blender process; it never touches
the running Blender GUI, its MCP socket, or the repo's `scenes/`.

The generator and this spec are the same code: `scripts/scenarios/dataset.py` writes
the layout and `validate_dataset()` checks it, so this document describes exactly what
lands on disk. Large outputs live **outside git** (default `~/datasets/store-scenarios/`);
only generator code, this spec, a tiny smoke fixture and offline tests are committed.

## Directory layout

```
<root>/<scenario_name>/
  manifest.json                     # scenario manifest + the image index (below)
  ground_truth.json                 # scenario-level ground truth (per scenario kind)
  coco.json                         # COCO detection export (boxes + optional RLE masks)
  cameras/<camera>.json             # intrinsics + extrinsics for each camera
  frames/<camera>/f0000.png         # RGB frames (sRGB PNG, AgX view transform)
  depth/<camera>/f0000.png          # 16-bit depth, millimetres (only if with_depth)
  annotations/<camera>/f0000.json   # per-image ground truth (one file per rendered image)
```

Frame numbers are zero-padded to 4 digits (`f{frame:04d}`). A multi-view capture is a
single frame (`f0000`) under many cameras; a video-like scenario is many frames under
one or more cameras.

## Coordinates and conventions

* Units are **metres**, Blender world axes, **Z up**. The floor is z=0, store centred
  on the origin. This matches `scripts/build_supermarket.py` and the planogram, so
  store-floor positions in the ground truth are directly comparable to the real store.
* Cameras are **pin-hole**, looking down local **−Z**, local **+Y** up, **+X** right.
  `sensor_fit = HORIZONTAL`, `sensor_width = 36 mm`, so
  `fx = fy = lens_mm * width_px / 36`, principal point at the image centre.
* Image pixels: origin **top-left**, `u` right, `v` down.
* **Depth** is camera-space **planar Z** in metres (the Cycles `Depth` pass), stored as
  a 16-bit grayscale PNG where the stored integer **is the depth in millimetres**
  (`depth_mm = png_value`; `0` = no hit / background). Max range 65.5 m.

## `cameras/<camera>.json`

```json
{
  "name": "view_0",
  "intrinsics": {"fx": 497.8, "fy": 497.8, "cx": 320, "cy": 240,
                 "width": 640, "height": 480, "lens_mm": 28, "sensor_mm": 36,
                 "model": "pinhole"},
  "extrinsics": {"camera_to_world": [[...],[...],[...],[0,0,0,1]],
                 "eye": [...], "target": [...], "up": [0,0,1]},
  "convention": {"forward": "-Z", "up": "+Y", "right": "+X",
                 "image_origin": "top-left", "depth": "camera-space planar Z, metres"}
}
```

`camera_to_world` is a 4×4 row-major rigid transform. World→pixel:
`X_cam = R^T (X_world − t)`, `depth = −X_cam.z`, `u = cx + fx·X_cam.x/depth`,
`v = cy − fy·X_cam.y/depth` (see `scripts/scenarios/camera.py`).

## `annotations/<camera>/f####.json`

```json
{
  "camera": "shelf_G1R", "frame": 3, "time_s": 1.5, "width": 640, "height": 480,
  "file_name": "frames/shelf_G1R/f0003.png",
  "annotations": [
    {
      "id": "shelf_G1R:3:G1R_b2_s0_f0", "entity_id": "G1R_b2_s0_f0",
      "category": "cereal_choco", "kind": "product",
      "is_hazard": false, "is_distractor": false,
      "bbox": [x, y, w, h],              // occlusion-correct pixel box (from IndexOB)
      "bbox_amodal": [x, y, w, h],       // full projected box, ignoring occlusion
      "center_px": [u, v], "depth_m": 1.83,
      "world_center": [x, y, z], "world_size": [sx, sy, sz],
      "pass_index": 42,                  // IndexOB channel for this object
      "segmentation": {"size": [h, w], "counts": [...]},  // COCO uncompressed RLE, or null
      "area_px": 915, "visibility": 0.74,  // visible px, and visible/amodal area
      "meta": { ... }                    // scenario-specific (slot_id, session_id, ...)
    }
  ]
}
```

`bbox` is the tight box of the object's visible pixels in the `IndexOB` pass (so it is
occlusion-correct); `bbox_amodal` is the full projected bounding box. `segmentation` is
a COCO **uncompressed** RLE (column-major, `pycocotools`-readable) when a mask pass is
available, else `null` (box-only). Fully-occluded *products* are dropped; hazards and
distractors are kept with `visibility: 0` so onset timing stays exact.

## `coco.json`

Standard COCO detection: `images`, `annotations` (bbox in pixels, optional `segmentation`
RLE, plus `visibility`, `is_hazard`, `is_distractor`), and `categories` (each flagged
`is_hazard` / `is_distractor`). Ready for `pycocotools` / detector training.

## `ground_truth.json` (per scenario kind)

| kind | contents |
|------|----------|
| `depletion` | `events` (per-frame remove/restock with slot+SKU), `slots`, `per_frame` → per-slot per-SKU facing counts |
| `spills` | `hazards` (id, class, `onset_frame`, `onset_s`, store-floor `floor_xy`), `distractors` (must not trigger) |
| `people` | `sessions` (session-scoped tokens), `trajectories` → per-session `track` of `{frame, x, y, heading}` |
| `misplaced` | `misplacements` → `{slot, expected_sku, actual_sku, type}` (`wrong_slot` or `chilled_on_ambient`) |
| `multiview` | `slot_fill` → per-slot `{shelf_depth, stock_depth_m, fill_ratio, n_facings}` (volumetric fill GT) |

## `manifest.json`

Scenario metadata (`name`, `kind`, `seed`, `fps`, `n_frames`, `cameras`, `params`,
`samples`, `format_version`) plus `images` — the index the validator walks:

```json
{"file": "frames/c/f0000.png", "annotation": "annotations/c/f0000.json",
 "depth": "depth/c/f0000.png" | null, "width": 640, "height": 480,
 "frame": 0, "camera": "c"}
```

## Privacy

People are procedural mannequins; `session_id`s are **session-scoped random tokens**
(no names, no biometrics, no cross-visit re-identification). Trajectories/boxes are the
only people ground truth. This mirrors the store-vision GDPR/DSGVO constraints.

## Regenerating / validating

See **RUNBOOK §9**. Validate any scenario dataset offline (no Blender) with
`scripts/scenarios/dataset.py:validate_dataset(root, name)`; the committed smoke fixture
under `scripts/scenarios/tests/fixtures/` is checked by `tests/test_fixture.py`.
