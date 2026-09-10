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
**import the URDF to USD** (step 1, see below) → reference the store USD → author
`PhysicsScene` + colliders → place the robot at the aisle pose → step → save the
stage + a screenshot.

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
  concave geometry, immovable, no rigid body.
- **Products & price tags**: convex-hull colliders authored on the **196 shared
  prototypes**, so every one of the 24 927 instances inherits a collider *once*.
  Authoring convex decomposition on 24 927 instances would blow VRAM — the guard
  rail in `twin/colliders.py` refuses to author on any instanceable placement.

## What is verified vs not

**Verified now (no simulator booted):**
- The measured USD facts and spawn pose above — produced by a real run of
  `--report-usd` against the committed `export/supermarket.usdc`.
- The GPU guard's refusal path — run live, it correctly detected the Blender
  session and refused.
- Pose maths, prototype-vs-instance collider selection, argument handling, the
  dry-run output, and the refusal logic — unit tested (`scripts/twin/tests/`,
  29 tests):
  ```bash
  conda activate env_isaaclab
  env -u PYTHONPATH PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
      python -m pytest scripts/twin/tests/ -q
  ```

**NOT verified (needs the captain's GPU boot):** the actual URDF→USD import, the
collider/physics authoring against the live stage, the robot resting on the floor
without penetration, and the screenshot. Those code paths are written and marked
`# UNVERIFIED (needs a simulator boot)` in `compose_robot_in_store.py`; they have
never executed, and whether Isaac Sim even stays up on this box is still the open
question the boot crash left. Treat the first real run as a bring-up, not a
regression check.
