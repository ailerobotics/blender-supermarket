# Regenerating the Isaac store dataset (the USD the mission actually loads)

## Why this exists

The GPU mission does **not** load the repo's `export/supermarket.usdc`. It loads an
**external dataset** pointed at by the runtime config key **`store_usd`** (read by
`scripts/twin/restock/physics_setup.py` and `view_scene.py`). Historically that is
`~/datasets/sim-store-realism-v2/supermarket.usdc`, produced by the earlier
`sim-store-realism-v2` lane on **2026-10-12** — i.e. **before** PR #266 widened the
portal to 2.6 m. So until the dataset is regenerated and the config repointed, GPU
runs keep driving through the **old 2.5 m portal** even though the branch's scene
code already says 2.6 m.

The mission-runtime JSON configs live **outside the repo** (owned by the
integration lane, e.g. `~/datasets/twin-robot-in-store-boot-v1/mission/*.json`); a
worker PR cannot flip them. This lane therefore ships: (1) the regenerator, (2) a
freshly built dataset, (3) the one-line config hand-off for integration.

## The pipeline that produced the dataset

Reverse-engineered from `~/datasets/sim-store-realism-v2/` build logs and
documented as RUNBOOK §10. All steps are **CPU / headless** — never the protected
live Blender GUI session, and never writing `scenes/*.blend`.

| step | tool | what it does | where it reads geometry |
|---|---|---|---|
| 1 | `python3 scripts/gen_labels.py` | deterministic PIL label/sign/price-tag PNGs | → `assets/labels/`, `assets/signs/` (gitignored) |
| 2 | `blender -b scenes/supermarket_base.blend --python scripts/build_supermarket.py -- --all --save <out>/store.blend --manifest <out>/environment.json` | builds the full store (structure, gondolas, products, **service backroom + portal**, cameras, lighting) and emits the semantic manifest | **portal width from `scripts/store_environment.py` `PORTAL_WIDTH`** via `build_environment` / `write_manifest` |
| 3 | `blender -b <out>/store.blend --python scripts/check_environment.py -- --manifest … --out <out>/geometry-check.json` | static circular-footprint clearance sanity (not dynamics) | the built scene + manifest |
| 4 | `blender -b <out>/store.blend --python scripts/export_usd.py -- --out <out>/supermarket.usdc` | Blender → USD (Z up, 1 m/unit, `/Supermarket` default prim, instanced, relative texture paths, `userProperties` custom props) + a `textures/` dir | the built scene |
| 5 | `.venv-usd/bin/python scripts/validate_usd.py --store <out>/supermarket.usdc [--repair-textures]` | strict offline Isaac-loadability check (CPU `usd-core`) | the exported stage |

- **Blender**: `Blender 4.5.13 LTS` (build 2026-08-25), at `~/.local/bin/blender`.
- **USD venv**: throwaway CPU `usd-core` venv `.venv-usd/` (gitignored):
  `python3 -m venv .venv-usd && .venv-usd/bin/pip install usd-core numpy`.
- **Realism assets**: versioned `assets/textures/` (Poly Haven floor/wall maps) +
  `report_assets/`; `scripts/planogram.json` drives product/slot layout. The base
  scene `scenes/supermarket_base.blend` carries the hand-built base geometry.
- **`SUPERMARKET_ROOT`**: `build_supermarket.py` / `export_usd.py` / `render_shots.py`
  derive `ROOT` from the script location (env `SUPERMARKET_ROOT` overrides), so they
  run in any worktree/clone.

### Texture repair is expected (not a defect introduced here)

A headless re-export still authors **6 dangling floor/wall texture references**
(`floor_tiles_08_{Diffuse,Rough,nor_gl}.jpg`, `white_plaster_02_{Diffuse,Rough,nor_gl}.jpg`).
Those maps were originally pulled into the scene by the live-GUI MCP Poly Haven tool
from `tmp*.jpg` temp files that do not exist in a headless session, so
`export_usd._stage_textures_for_export` cannot stage them ("staged 0 texture(s)").
`validate_usd` grades this **COSMETIC** (floor/walls load untextured; geometry
intact, stage loads, robot stands). `--repair-textures` copies the 6 names from the
versioned `assets/textures/` into the dataset's `textures/`. This is **identical to
the v2 baseline** (v2 was repaired the same way). The regenerator applies the repair
automatically and records `texture_repair_applied` in the manifest.

## One command

```bash
scripts/regen_store_dataset.py --out ~/datasets/sim-store-realism-v3-portal26
```

`scripts/regen_store_dataset.py` runs steps 1–5 above, then probes the exported
stage for the portal clear width and writes `manifest.json` next to the USD
recording: git HEAD/branch/dirty, Blender version, source-script SHA-256s,
`PORTAL_WIDTH`, the measured portal clear width (manifest **and** direct-from-USD),
jamb/wall edge X positions, backroom/map bounds, station/obstacle counts, the USD
validation verdict, and artifact SHA-256s. It **refuses** to write into
`~/datasets/sim-store-realism-v2` or anywhere inside the repo, and refuses to build
from the live `scenes/supermarket.blend`. A final gate aborts unless the source
constant, the manifest width and the measured-USD width all agree.

Flags: `--blender`, `--usd-python`, `--threads` (default 4), `--base-scene`,
`--skip-labels`. Reuse this whenever the scene/station geometry changes (e.g. the
`sim-dock-align-v1` lane).

## Delivered dataset

`~/datasets/sim-store-realism-v3-portal26/` — built from this branch
(scene-code HEAD `fb9ce8a`, PR #266 merged). Portal clear width **2.6 m**
(measured in the USD stage), jambs at `PORTAL_X ∓ 1.3` (were `∓ 1.25` at 2.5 m).
The old `sim-store-realism-v2/` dataset is left untouched. Before/after portal
dump: `firstmate/data/sim-dataset-usd-regen-v1/portal-before-after.md`.

## Integration hand-off (do NOT switch the running mission config here)

Repoint the mission's external runtime config key — a one-line change in each
`~/datasets/twin-robot-in-store-boot-v1/mission/mission-runtime-*.json` that sets it:

```json
"store_usd": "/home/manishsaini/datasets/sim-store-realism-v3-portal26/supermarket.usdc",
```

**Verify at boot**: the loaded stage's rear opening is 2.6 m — the gap between
`GRP_Structure/Wall_back_west` (east edge x ≈ -5.175) and
`GRP_Structure/Wall_back_east` (west edge x ≈ -2.575). Keep the dataset's adjacent
`textures/` directory beside the `.usdc`.
