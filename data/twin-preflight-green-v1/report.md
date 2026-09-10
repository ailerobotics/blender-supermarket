# One-command pre-flight goes green (twin-preflight-green-v1)

**Question:** the captain's go/no-go for booting the robot-in-store simulator is one
command, `scripts/preflight.py`. It was coming back **red**. Is that red real, and can
the one command give a single unambiguous answer?

**Answer:** the red was **not** real in the go/no-go sense — it was the known cosmetic
floor/wall texture defect, which does not block the URDF import, the physics/colliders or
the robot placement. A red that does not mean "do not boot" is exactly how a person learns
to ignore a check. Two changes fix that **without suppressing anything**, and the one
command now exits **0** with all three sections green:

```
  PASS    STORE    (export USD Isaac-loadable)
  PASS    ROBOT    (URDF Isaac-ready + fits aisle)
  PASS    COMPOSER (offline composition sane)
VERDICT: PASS — ... Safe to boot (close Blender first).
```

Verified entirely offline (CPU-only `usd-core`, no Isaac, no GPU), safe beside the live
Blender session — no re-export, nothing written to `scenes/`, `renders/`, or committed.

---

## What changed

1. **The pre-flight repairs the current export in place before validating it.**
   `scripts/preflight.py` runs the store check with `repair=True`: any authored-but-missing
   floor/wall texture is copied from the versioned `assets/textures/` into the gitignored
   `export/textures/` (deterministic, offline, no Blender) before the verdict. So the one
   command is self-sufficient on a stale on-disk export — the actual defect is repaired,
   not hidden. `--no-repair` shows the raw on-disk state without touching `export/`.

2. **The store check grades a failure BLOCKING vs COSMETIC.** `scripts/validate_usd.py`
   now classifies every hard failure:
   - **COSMETIC** — a dangling *texture* (an `Asset`-valued shader input): the surface
     loads untextured but geometry, scale and placement are intact, so Isaac still loads
     the stage and the robot still stands. Renders as `[COSMETIC]`.
   - **BLOCKING** — a dangling *composition arc* (reference / payload / sublayer =
     missing geometry) or any structural defect (up-axis, metres-per-unit, default prim,
     floor, composer prim path, absolute build-machine path, asset escaping `export/`).

   The pre-flight boot gate keys off **blocking only**: a section is `PASS` (clean),
   `PASS*` (cosmetic — real defect, safe to boot) or `FAIL` (blocking — do NOT boot).
   Exit code is `0` when safe to boot, `1` only on a blocking defect. **A red pre-flight
   now always means "do not boot".** The standalone `validate_usd.py` stays strict (exit 1
   on any unresolved asset, so a re-export that reintroduces the defect can't slip
   through), and its verdict line says whether the failure is blocking or cosmetic.

## The artifact decision — repaired on demand, NOT committed

`export/` stays **gitignored** per the repository's file-policy rule ("ignore only what is
offline AND cheap to regenerate"). The texture repair is part of that on-demand
regeneration and is reproducible by anyone running the one command. Cost basis, measured on
this machine:

| Option | Git bytes added | Time | Verdict |
|---|---|---|---|
| **Repair on demand (chosen)** — `preflight.py` copies the 6 maps from `assets/textures/` | **0** (the 6 maps, 6.5 MB, are already versioned in `assets/textures/`) | **~7 s** (validate + repair, offline) | Respects the file policy; one command; no Blender. |
| Commit the repaired artifact | **+28 MB** (`export/` = 4.6 MB `.usdc` + ~23 MB textures) | — | Breaks the file policy (versions regenerable binary). Rejected. |
| Full re-export | 0 | **26.3 s** Blender, **and needs the GUI-held scene** | Exactly the GPU-adjacent work the pre-flight exists to avoid. Rejected for this task. |

The durable source-side fix for the *next* export already lives in
`scripts/export_usd.py` (`_stage_textures_for_export()`, from PR #2) — a fresh export
authors resolvable paths and needs no repair.

## Reproduction (the captain's one command, from a stale export)

From a checkout whose `export/supermarket.usdc` still has the 6 dangling textures:

```
$ .venv-usd/bin/python scripts/preflight.py
...
[····] repair-textures: copied 6 texture(s) from assets/textures/ -> export/textures/: floor_tiles_08_Diffuse.jpg, floor_tiles_08_Rough.jpg, floor_tiles_08_nor_gl.jpg, white_plaster_02_Diffuse.jpg, white_plaster_02_Rough.jpg, white_plaster_02_nor_gl.jpg
[PASS] every asset path resolves: all 226 texture/asset paths point at existing files
...
  PASS    STORE    (export USD Isaac-loadable)
  PASS    ROBOT    (URDF Isaac-ready + fits aisle)
  PASS    COMPOSER (offline composition sane)
VERDICT: PASS — ... Safe to boot (close Blender first).
$ echo $?
0
```

A truly bare `git clone` has no `export/` at all (gitignored) — regenerate it first via the
pipeline in README "Rebuilding from a fresh clone" (steps 1–4a; the fixed `export_usd.py`
then produces a clean USD directly), then run the pre-flight.

## Tests

`scripts/twin/tests/test_validate_usd.py` gains 3 tests (dangling texture is cosmetic not
blocking; a structural defect is blocking; a repaired texture leaves no failure). Full twin
suite: **67 passed** under the usd-core venv
(`env -u PYTHONPATH PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv-usd/bin/python -m pytest scripts/twin/tests/`).

## Not touched (hard constraints honoured)

Isaac Sim not booted; no GPU allocated. The live Blender session and `scenes/*.blend`
untouched; no re-export. `~/workspaces/shelfbot_newgripper` read-only — its placeholder-inertia
WARN (another repo, tracked separately) is left reported. No check suppressed or downgraded:
the actual texture defect is repaired from versioned sources; severity grading only
distinguishes what blocks booting from what does not.
