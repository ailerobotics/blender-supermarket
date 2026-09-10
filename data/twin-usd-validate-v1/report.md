# Supermarket USD — offline Isaac pre-flight (twin-usd-validate-v1)

**Question:** when the captain opens the GPU session, will the store load and will the
robot be able to stand in it — and if not, exactly what stops it?

**Answer:** **Yes on both, with one caveat that does not block the robot.** The stage
opens cleanly, is Z-up / 1 unit = 1 m, has a valid default prim, and carries every
prim path and camera the composer needs. There is a 26 × 22 m floor at z = 0 for the
robot to stand on, and the pedestal fits its aisle spawn with room for the arm to
work. The **only** defect is cosmetic: **6 texture references dangle** — the floor and
all five walls — so in Isaac they load **untextured**. That is a material-fidelity
bug, not a load or placement blocker. It is fixed in the export script for the next
export and can be repaired on the current artifact **offline, without Blender**.

Verified entirely offline with `usd-core` **26.8** (`pip install usd-core`, CPU-only,
no Isaac, no GPU). Artifact validated: `export/supermarket.usdc`
(git blob `79603a91…`, 4.78 MB, 225 texture files) — the exact blob
`docs/twin_usd_facts.json` was derived from.

---

## 1. Stage characterisation (measured)

| Property | Value |
|---|---|
| Opens | yes (`Usd.Stage.Open`) |
| Up axis | **Z** (Isaac wants Z) ✓ |
| Metres per unit | **1.0** ✓ |
| Default prim | **`/Supermarket`** ✓ |
| Total prims | **54 150** (default traversal); 325 959 incl. instance proxies |
| Instances | **24 927** |
| Prototypes | **196** |
| Type histogram | Xform 50 738 · GeomSubset 1 377 · Shader 822 · Mesh 766 · Material 374 · RectLight 63 · Camera 8 · Scope 2 |
| World extents | min (−60, −60, −0.03) m, max (60, 60, 4.3) m — dominated by the 120 × 120 m exterior ground plane |
| **Store footprint** (Floor) | x [−13, 13], y [−11, 11] = **26 × 22 m** at z = 0 |

The store footprint **matches** the README ("26 × 22 m of shop floor") and RUNBOOK, and
the up-axis / scale / default-prim / floor-extent match `docs/twin_usd_facts.json`.

## 2. Reference resolution — **1 defect**

Every asset path was walked two ways: `UsdUtils.ComputeAllDependencies` (the
reference/payload/sublayer graph, as `usdchecker` does) and a per-attribute sweep of
all 226 `Asset`-valued inputs.

- **Composition:** single self-contained layer — no external references, payloads or
  sublayers. ✓
- **Absolute build-machine paths:** none — all 226 asset paths are relative
  (`./textures/…`). ✓
- **Escaping `export/`:** none. ✓
- **Unresolved: 6** — all texture maps, all on two materials:

  | Authored path | Bound to |
  |---|---|
  | `./textures/floor_tiles_08_Diffuse.jpg` | `GRP_Structure/Floor` |
  | `./textures/floor_tiles_08_Rough.jpg` | `GRP_Structure/Floor` |
  | `./textures/floor_tiles_08_nor_gl.jpg` | `GRP_Structure/Floor` |
  | `./textures/white_plaster_02_Diffuse.jpg` | all 5 walls |
  | `./textures/white_plaster_02_Rough.jpg` | all 5 walls |
  | `./textures/white_plaster_02_nor_gl.jpg` | all 5 walls |

**Root cause.** `scripts/build_supermarket.py:polyhaven_mat()` reuses the two Poly
Haven materials imported by the MCP `download_polyhaven_asset` tool. That tool loads
the maps from **temp files**, so the Blender image datablocks are *named*
`floor_tiles_08_Diffuse.jpg` … but *backed* by `tmp4mie4x9u.jpg` … . Blender's USD
exporter copies the **backing file** into `export/textures/` (so the six files land on
disk as `tmp*.jpg`, now orphaned) but authors the reference by **datablock name** — so
the six references point at names that were never written. The 219 product/sign PNGs
are unaffected: for those, name == backing basename, so they resolve. This is
deterministic — a headless re-export reproduces it (README decision log, 2026-09-09:
"4 783 096 bytes with 225 textures each").

**Impact.** Floor and walls load **untextured** in Isaac (missing-texture fallback).
Geometry, transforms, extents and instancing are intact. **Does not block loading or
robot placement.**

## 3. Composer expectations vs. the real stage

Checked against `scripts/compose_robot_in_store.py` and its probe
`scripts/twin/usd_probe.py`:

| Composer needs | Where | Verdict |
|---|---|---|
| `Usd.Stage.Open` succeeds | `usd_probe.py:77` | ✓ |
| Default prim to reference | `compose:104-105` `AddReference`; `usd_probe:80` | ✓ `/Supermarket` |
| Up-axis Z, metres | `compose:102-103` | ✓ (store already Z / 1.0) |
| Floor at `GRP_Structure/Floor` | `usd_probe.py:94` | ✓ z = 0 |
| Groups Structure/Fixtures/Signage/Products/PriceTags | `usd_probe.py:102` | ✓ |
| `GRP_Cameras/Cam_Aisle` with a transform | `usd_probe.py:183-186` | ✓ world xy (−1.525, −5.6) |
| Gondola fixtures `GRP_Fixtures_G1…G7` | `usd_probe.py:170` | ✓ |
| Product runs `GRP_Products_G1…G7` | `usd_probe.py:160-161` | ✓ |
| Prototypes to carry instance colliders | `usd_probe.py:118` | ✓ 196 |
| Static-group meshes for triangle-mesh colliders | `colliders.STATIC_GROUPS`; `compose:181-190` | ✓ |

**No composer line fails on this stage.** The probe already ran on this exact blob
(`docs/twin_usd_facts.json`). The dangling textures never reach the composer — it
references the store and authors colliders on **geometry**; it never touches material
inputs — so composition and physics authoring are unaffected. The consequence is
purely what the store *looks* like once loaded.

## 4. Robot-in-store readiness

The robot (`shelfbot_gripper_isaac.urdf`) is a **pedestal-mounted UR10 arm**, not a
mobile base: `world → pedestal_link` is a **fixed** joint at xyz `0 −1.2 0.35`, and the
composer sets `fix_base(True)`. It stands at one spot and reaches shelves; it does not
drive down aisles.

- **A floor to stand on?** **Yes.** `GRP_Structure/Floor` mesh, 26 × 22 m, z = 0,
  `purpose = default` (collidable). The pedestal base rests on it once a collider is
  authored.
- **Footprint vs. aisles.** Pedestal collision box **0.2 × 0.4 m** (URDF `world` link).
  Aisle gaps between gondola runs are **1.5 m** (`twin_usd_facts.json`
  `aisle_centerlines`, all six aisles). The geometry-derived spawn on `Cam_Aisle`
  (the G3–G4 aisle) is pedestal (−1.525, 0.75, 0) with **0.95 m planar clearance** to
  the nearest fixture/product — well above the composer's 0.6 m default bar, and far
  larger than the 0.2 m pedestal half-depth. The UR10's ~1.3 m reach comfortably
  reaches shelf faces ~0.75 m away. **The aisle is passable and workable for this
  footprint.**
- **Colliders / physics.** **None. The store is render-only geometry** — 0
  `PhysicsScene`, 0 collision/rigid-body APIs. **This is by design:**
  `compose_robot_in_store.py` authors physics at compose time — a `PhysicsScene` with
  gravity, **triangle-mesh** colliders on the 702 static structure/fixture/signage
  meshes (floor and walls included), and **convex-hull** colliders on the 196 instance
  prototypes (never on the 24 927 instances — that would blow VRAM). So the **minimum
  to place a robot meaningfully** is exactly what the composer already does; nothing
  extra needs adding to the export. Until the composer runs, a dropped rigid body
  would fall through the floor — expected for a render-only stage.

## 5. Validation tooling output (verbatim)

`usd-core` 26.8's wheel ships **no `usdchecker` binary and no
`UsdUtils.ComplianceChecker`**, so validation uses the equivalent Python API
(`Usd.Stage.Open`, `UsdGeom` metadata, `UsdUtils.ComputeAllDependencies`, per-attribute
`Sdf.ValueTypeNames.Asset` sweep), packaged as `scripts/validate_usd.py`. On the
as-shipped artifact:

```
[PASS] stage opens: export/supermarket.usdc
[PASS] default prim: /Supermarket
[PASS] up-axis: Z (Isaac wants Z)
[PASS] metres-per-unit: 1.0 (Isaac wants 1.0)
[····] prim count: 54150 prims (24927 instances, 196 prototypes)
[····] type histogram: Xform=50738, GeomSubset=1377, Shader=822, Mesh=766, Material=374, RectLight=63, Camera=8, Scope=2
[····] world extents (m): min=(-60.0, -60.0, -0.03) max=(60.0, 60.0, 4.3) (includes the exterior ground plane)
[PASS] floor present (GRP_Structure/Floor): z=0.0 m, footprint 26.0×22.0 m — robot has a surface to stand on
[PASS] composer group hierarchy: GRP_Structure, GRP_Fixtures, GRP_Signage, GRP_Products, GRP_PriceTags, GRP_Cameras
[PASS] gondola runs (aisle geometry): 7 fixture runs + 7 product runs (G1–G7)
[PASS] spawn camera (GRP_Cameras/Cam_Aisle): marks the aisle centreline for the pedestal
[PASS] composition: single self-contained layer (no external refs/payloads/sublayers)
[····] asset-valued attributes: 226 (textures + any file inputs)
[PASS] no absolute build-machine paths: all asset paths are relative
[PASS] all assets inside export/: no asset escapes the export directory
[FAIL] every asset path resolves: 6 unresolved: floor_tiles_08_Diffuse.jpg, floor_tiles_08_Rough.jpg, floor_tiles_08_nor_gl.jpg, white_plaster_02_Diffuse.jpg, white_plaster_02_Rough.jpg, white_plaster_02_nor_gl.jpg
[····] orphan texture files: 6 file(s) on disk are unreferenced (e.g. tmp4mie4x9u.jpg)
[····] physics / colliders: render-only geometry (no PhysicsScene, no collision APIs) — expected; compose_robot_in_store.py authors colliders + gravity at compose time
------------------------------------------------------------------------------
VERDICT: FAIL  (1 hard check(s) failed)
```

After `--repair-textures` (or a re-export with the fixed script), the same run reports
`[PASS] every asset path resolves: all 226 texture/asset paths point at existing files`
and **`VERDICT: PASS`** — confirming the textures are the *only* hard defect.

## 6. Fixes delivered

1. **`scripts/export_usd.py` — durable root-cause fix.** `_stage_textures_for_export()`
   repoints every Blender image so its backing-file basename equals its datablock name
   before `usd_export`, preferring the canonical copy in `assets/textures/`. So the next
   export authors resolvable paths for floor and walls. It runs in the headless export
   process (in-memory only; the `.blend` and the live GUI session are untouched) and is
   guarded so it can never break the export. The `TypeError` fallback export now also
   passes `relative_paths=True` so it can't regress to absolute paths.
   **⚠ Takes effect only on the next Blender re-export** — this task did not (and must
   not) run Blender.
2. **`scripts/validate_usd.py` — the repeatable offline check.** Runs without a GPU;
   exit `0` pass / `1` fail. `--json` for the facts, `--selftest` for a no-artifact
   sanity check, and `--repair-textures` to reconcile the **current** artifact offline
   (copies the 6 authored names from the versioned `assets/textures/` into
   `export/textures/` — deterministic, no Blender, no GPU).
3. **Tests:** `scripts/twin/tests/test_validate_usd.py` (7 tests) builds synthetic
   good/bad stages and asserts the validator's pass/fail behaviour. They need `pxr`, so
   they **skip cleanly** where USD is absent (plain `env_isaaclab`) and run under the
   usd-core venv. Full suite: **36 passed** (29 existing + 7 new).

## 7. What the captain should do in the GPU session

The store **will load** and the robot **will be able to stand and work** in it. Before
(or instead of) the GPU session:

- To get a **fully-textured** stage without re-running Blender:
  `python3 scripts/validate_usd.py --store export/supermarket.usdc --repair-textures`
  (offline; makes the current artifact `VERDICT: PASS`).
- For a clean regenerated export, the **next** `scripts/export_usd.py` run already
  carries the fix; verify it with `python3 scripts/validate_usd.py`.
- Then compose as documented: `scripts/compose_robot_in_store.py` authors the physics
  and colliders the robot needs; nothing else about the store must change.

The one thing that is **not** yet true of the committed pipeline artifact: the
`export/supermarket.usdc` currently in the primary checkout still has the 6 dangling
floor/wall textures. `export/` is gitignored (regenerated), so this is not committed;
run either remedy above once before relying on the store's floor/wall appearance in
Isaac.
