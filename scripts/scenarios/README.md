# Store-vision scenario generator

Renders **labelled synthetic sequences** from the generated supermarket for
store-vision development and evaluation: shelf depletion, floor spills/hazards,
people movement, misplaced products, and multi-view RGB-D shelf capture. Output is a
documented dataset (frames, depth, camera intrinsics/extrinsics, per-frame annotation
JSON, COCO export, manifest) — format spec in [`docs/scenario-format.md`](../../docs/scenario-format.md).

Runs **headless in a separate Blender process**; it never uses the running Blender GUI
or its MCP socket (port 9876) and never writes to `scenes/`. Commands are in
**RUNBOOK §9**.

## The five scenarios

1. **depletion** — a fixed shelf-facing camera watches a gondola bay while facings are
   removed over time (with occasional restock) and a shopper occasionally occludes the
   shelf. GT: per-frame per-slot SKU + facing count, remove/restock events, occluder boxes.
2. **spills** — water (clear/specular), milk, juice, oil, debris and a fallen product
   appear at known times/places, plus distractors that must **not** trigger (shadow,
   reflection, floor marking, paper sign). GT: hazard masks/boxes, class, onset time,
   store-floor position.
3. **people** — mannequins walk entrance → aisle → dwell → checkout → exit. GT: per-frame
   boxes, session ids, store-floor trajectories.
4. **misplaced** — an item in the wrong slot and a chilled item on an ambient shelf.
   GT: slot, expected vs actual SKU.
5. **multiview** — several posed cameras across a bay with a **depth pass** and exact
   intrinsics/extrinsics, for 3D reconstruction. GT: per-slot remaining stock depth (fill).

## Design

The ground truth is **computed, not inferred**. `catalog_geom.py` places every product
facing in world space deterministically from the planogram (mirroring
`build_supermarket.py`); the Blender builder instantiates one object per facing at that
transform and the annotator projects the same world boxes — so the rendered pixels and
the labels agree by construction. Occlusion-correct masks/boxes come from the Cycles
`IndexOB` pass; depth from the `Depth` pass.

### Module map

| module | bpy? | role |
|--------|------|------|
| `catalog_geom.py` | no | planogram → per-facing world boxes (shared ground-truth geometry) |
| `camera.py` | no | pin-hole intrinsics/extrinsics, world→pixel projection |
| `state.py` | no | entity + timeline model; `state_at(frame)` |
| `people.py` | no | shopper trajectories (session-scoped tokens) |
| `scenario_defs.py` | no | the five parameterised, seeded scenario builders |
| `annotate.py` | no* | per-image GT, COCO, RLE; `refine_with_index` (numpy) |
| `common.py` | no | shared entity/geometry helpers |
| `dataset.py` | no | dataset layout writers + offline `validate_dataset` (Pillow for header probe) |
| `build_scene.py` | **yes** | realise a scenario as a Blender scene |
| `generate.py` | **yes** | headless render + annotate + write dataset (the entry point) |

The no-`bpy` modules are the offline-tested core (`tests/`, pure Python + numpy +
Pillow). `build_scene.py`/`generate.py` run only under `blender -b`.

## Tests

Offline and deterministic — no Blender, no network:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH= .venv-scenarios/bin/python -m pytest scripts/scenarios/tests/ -q
```

(`PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=` keeps the workstation's ROS pytest
plugins from being autoloaded.) The committed smoke fixture under `tests/fixtures/`
(rendered once, ~1 MB) is what `test_fixture.py` validates, standing in for Blender in CI.

## Environment

```bash
uv venv --python 3.12 .venv-scenarios
uv pip install --python .venv-scenarios numpy pillow pytest
```

Blender (4.5 LTS) runs the generator with its **bundled** Python (numpy included); the
`.venv-scenarios` above is only for the offline tests and the validator.

## Licences

All code here is first-party (same licence as this repository). Dependencies and assets
in **new code paths** are permissive, matching the task's licence policy:

| component | licence | note |
|-----------|---------|------|
| Blender | GPL (the *application*; our scripts are not linked/distributed with it) | render tool only |
| NumPy | BSD-3-Clause | |
| Pillow | MIT-CMU (HPND) | test/validator image-header probe |
| pytest | MIT | tests |
| People mannequins | — | **procedural primitives (own geometry), no third-party asset** |
| Product/fixture geometry | this repo | derived from the planogram + catalog |

No AGPL, no model weights, and no external human-figure assets are used. If a CC0 human
asset is ever substituted for the primitive mannequins, record its source and licence in
this table.
