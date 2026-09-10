# Standing the robot in the store (Isaac Sim)

Prep and tooling to place the fleet's UR10 (`shelfbot_newgripper`) inside the
supermarket twin under **Isaac Sim 5.1** (`env_isaaclab`), following the fleet's
own `~/workspaces/shelfbot/tools/run_tray_scene.py` pattern.

> **The GPU boot is the captain's decision and has NOT been taken.** Everything
> here up to that boot is built and proven; the boot itself is deliberately left
> for the captain. See [What is verified vs not](#what-is-verified-vs-not).

## Why you must close Blender first

This is a tight box: RTX 3060 with **12 GB** VRAM, ~8 GB RAM free, ~0.5 GB swap,
and Isaac Sim has **OOM-crashed here at startup before**. The captain's live
**Blender GUI holds `scenes/supermarket.blend` open** (MCP socket on
`localhost:9876`) on the same GPU. Booting a 56k-prim RTX scene beside it risks an
out-of-memory kill of that window and its unsaved work.

So `compose_robot_in_store.py` **refuses to boot** when it detects the Blender
session (or any other heavy GPU consumer). The refusal for the protected Blender
session is **not** overridable by `--force` — close/save Blender first, then run.

## Command sequence

All commands run under `env_isaaclab`. The wrapper activates it and puts
`ur_description` (`/opt/ros/jazzy/share`) and `shelfbot` (`~/workspaces/shelfbot`)
on the ROS package path so the URDF's `package://` meshes resolve.

```bash
# 1. Read the store USD and print measured facts (NO GPU boot — safe any time,
#    even with Blender open). This is how docs/twin_usd_facts.json was produced.
bash scripts/compose_robot_in_store.sh --report-usd

# 2. Full plan: spawn pose, collider counts, and what the GPU guard decides right
#    now — authors nothing, boots nothing (safe with Blender open).
bash scripts/compose_robot_in_store.sh --dry-run

# 2b. ACTUALLY author + validate the composed stage with pure USD (still NO GPU boot).
#     Proves colliders/physics/placement offline; exit 0 = sane. Prefer the CPU-only
#     usd-core venv so it needs nothing from Isaac (RUNBOOK §6b):
.venv-usd/bin/python scripts/compose_robot_in_store.py --author-offline --save /tmp/store_with_robot.usda

# 3. REAL run — boots Isaac Sim (a GPU session). ONLY after the captain closes
#    Blender. The guard still refuses if Blender (or a heavy consumer) is present.
bash scripts/compose_robot_in_store.sh \
     --save /tmp/store_with_robot.usd --screenshot /tmp/store_with_robot.png
```

Useful flags (see `--help`): `--spawn-aisle Cam_Aisle` (default; the G3–G4
aisle), `--spawn-xy X Y` (explicit pedestal spot), `--product-collider none`
(fixtures/ground only), `--vram-budget-mb`, `--force` (override the *soft* gates —
low headroom / non-Blender heavy consumer — never the Blender gate).

The real run, in order: GPU guard → boot `SimulationApp({'headless': True})` →
**import the URDF to USD** (step 1, see below) → build the composed stage with the
**shared pure-USD core** `twin/compose.py` (sublayer the store, author `PhysicsScene`
+ colliders, place the robot at the aisle pose) → open that stage in the sim → step →
screenshot. The composition core is identical to what `--author-offline` runs, so it
is already proven; only the boot / import / step / screenshot are new at GPU time.

## URDF → USD conversion cannot be done without a boot

The URDF must become USD via `isaacsim.asset.importer.urdf`. That extension
depends on `omni.kit.*` (menu, notification, viewport, window) — it only runs
inside the booted Kit app, which starts the RTX renderer (the exact component
that crashed here). There is **no pure-Python path**; Isaac Lab's
`urdf_converter` wraps the same importer inside a `SimulationApp` too. So the
conversion **is a GPU/simulator boot** and, per this task's boundary, is **left as
the first step the compose script performs** (behind the same GPU guard), not done
ahead of time. `--robot-usd <path>` lets you skip it if a converted robot USD
already exists.

## Recorded evidence (measured, no GPU)

Full JSON: [`twin_usd_facts.json`](twin_usd_facts.json). Read from
`export/supermarket.usdc` with the USD bindings bundled in Isaac Sim (USD 0.24.5),
**without** booting the simulator.

| Property | Measured |
|---|---|
| Up axis | **Z** (matches Isaac + URDF) |
| Metres per unit | **1.0** (no rescale) |
| Default prim | `/Supermarket` |
| **Floor Z** | **0.0** — on the ground plane; no base-Z offset needed |
| Floor extent | X ∈ [−13, 13], Y ∈ [−11, 11] (26 m × 22 m) |
| Triangle-mesh collider targets | **702** direct static meshes (structure 10, fixtures 674, signage 18) |
| Convex collider targets | **196** instance prototypes → **24 927** instances inherit (authored once each, *never* per instance) |
| Aisles | 6 aisles between gondolas G1–G7, each **1.5 m** wide |

**Candidate spawn pose (pedestal ground contact, derived from geometry):**

| | Value |
|---|---|
| Aisle | Cam_Aisle → **G3–G4 centreline, X = −1.525** |
| Pedestal at | **(−1.525, 0.75, 0.0)** m |
| Planar clearance | **0.95 m** to the nearest fixture (pedestal is 0.2 × 0.4 m — wide margin) |
| Robot root xform translate | **(−1.525, 1.95, 0.0)** — compensates the pedestal's (0, −1.2, 0) offset from the URDF `world` link; re-measured and corrected after import |

The floor **is** at Z = 0, so the pedestal box (0.2 × 0.4 × 0.7, bottom on the
link origin) rests on the floor with the root at Z = 0. If a future export moves
the floor off Z = 0, the probe reports it and the script sets the base Z from the
measured floor plane.

Spawn the arm **tucked/home** (as `run_tray_scene.py` does) so the UR10's ~1.3 m
reach envelope doesn't penetrate the shelves it stands 0.95 m from — that reach is
intended for stocking, not for the rest pose.

Alternative wide-open spot (not an aisle): global-max clearance **2.5 m** at
(7.7, 7.45) — the open floor behind the gondolas, for a first fall-through test.

## Collider strategy (why prototypes, not instances)

- **Static scenery** (walls, ceiling, floor, 674 fixture meshes, signage):
  triangle-mesh colliders (`UsdPhysics.MeshCollisionAPI` approximation `none`) —
  concave geometry, immovable, no rigid body. 702 meshes.
- **Products & price tags**: convex-hull colliders authored on the **196 shared
  source meshes** (plus 1 loose price-tag mesh = 197), so every one of the 24 927
  instances inherits a collider *once*. Authoring convex decomposition on 24 927
  instances would blow VRAM — the guard rail in `twin/colliders.py` refuses to author
  on any instanceable placement.

### Where the shared collider is authored (the sublayer, not a nested reference)

The store instances its products from source meshes held under the
`/Supermarket/prototypes` **class** scope; each product placement is an `instanceable`
prim that references one of those class prims, and USD then builds one runtime
`/__Prototype_N` per unique source. Two facts drove the composer's design (both
verified offline with `usd-core`):

1. **You cannot author a collider on a runtime `/__Prototype_N` prim** — USD raises
   *"authoring to an instancing prototype is not allowed."* So the collider is authored
   on the **class source mesh** instead, and every prototype/instance inherits it.
2. **An override authored on that class from *outside* the store's reference layer
   stack does not reach the prototype either.** So the composer composes the store as a
   **sublayer** of the output (not a nested `AddReference`), which puts the collider
   overrides inside the store's own layer stack where they propagate into all 196
   prototypes. The committed `export/supermarket.usdc` is only ever **sublayered
   read-only** — never edited. (`--author-offline` asserts its SHA is unchanged.)

This is a script-only fix — no Blender re-export is needed; the class-scope structure
the export already produces is exactly what makes a single shared collider possible.

## What is verified vs not

**Verified now (no simulator booted):**
- The measured USD facts and spawn pose above — produced by a real run of
  `--report-usd` against the committed `export/supermarket.usdc`.
- **The whole composition authored for real, offline** (`--author-offline`): the store
  sublayered read-only, the `PhysicsScene`, all **702 triangle-mesh + 197 convex-hull**
  colliders, **196/196 product prototypes** inheriting a collider (authored once on the
  shared source — never per-instance, never on the forbidden runtime prototype), and the
  robot placed via the real transform maths — **standing on the floor** (pedestal bottom
  Z = 0.0), at the aisle spot **(−1.525, 0.75)**, **inside the aisle gap** (X ∈ ≈[−1.63,
  −1.43], between gondolas G3 and G4), with instancing preserved (≈54 154 prims, not
  de-instanced). The robot is a faithful pedestal+arm **stub** offline (the true mesh
  needs the URDF→USD import); the transform, colliders, physics and stage structure are
  the real thing Isaac will load.
- The GPU guard's refusal path — it correctly detects the Blender session and refuses.
- The pose maths, prototype-vs-instance collider selection, argument handling, the
  dry-run output, the refusal logic, and the composition core — unit tested
  (`scripts/twin/tests/`). The composition + validator tests need `usd-core`; the
  pure-logic tests need nothing:
  ```bash
  # offline (recommended): CPU-only usd-core venv (RUNBOOK §6a/§6b)
  env -u PYTHONPATH PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv-usd/bin/python -m pytest scripts/twin/tests/ -q
  # or under Isaac's env:
  conda activate env_isaaclab
  env -u PYTHONPATH PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest scripts/twin/tests/ -q
  ```

**NOT verified — needs the captain's GPU boot (the only untested links left):** the
URDF→USD import (`isaacsim.asset.importer.urdf`), PhysX actually loading the authored
colliders, the robot settling under gravity without penetration once stepped, and the
screenshot. Those blocks are marked `# UNVERIFIED (needs a simulator boot)` in
`compose_robot_in_store.py`; they have never executed, and whether Isaac Sim stays up on
this 12 GB box is still the open question the earlier boot crash left. Everything the
simulator *consumes* — the composed stage — is now proven offline; treat the first real
run as a bring-up of the simulator link, not a regression check of the composition.
