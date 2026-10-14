# Blender Supermarket — RUNBOOK

Project: `~/workspaces/blender-supermarket` · Blender 4.5.13 LTS at `~/opt/blender` (symlink `~/.local/bin/blender`) · MCP: `blender-mcp` (ahujasid) via `uvx`.

## 1. Start / stop Blender with the scene (MCP socket auto-starts on :9876)
```bash
~/workspaces/blender-supermarket/scripts/launch_blender.sh              # GUI on DISPLAY=:0, tmux session "blender"
~/workspaces/blender-supermarket/scripts/launch_blender.sh scenes/supermarket_base.blend   # empty base (only textures + HDRI)
tmux attach -t blender      # console output;  Ctrl-b d  to detach
tmux kill-session -t blender
ss -ltnp | grep 9876        # is the addon socket up?
```

## 2. Claude Code ↔ Blender MCP
```bash
claude mcp list                                   # "blender: uvx blender-mcp" must be listed (user scope, ~/.claude.json)
claude mcp add -s user blender -- uvx blender-mcp # re-register if missing
BLENDERMCP_ADDONS_DIR=~/.config/blender/4.5/scripts/addons uvx blender-mcp install-addon   # re-install/upgrade the addon
```
Inside a Claude Code session the tools are `mcp__blender__get_scene_info`, `execute_blender_code`, `get_viewport_screenshot`, `download_polyhaven_asset`, … (28 tools). They only load at session start, so restart Claude after `claude mcp add`.

Stand-alone MCP test client (spawns the same `uvx blender-mcp` server over stdio):
```bash
cd ~/workspaces/blender-supermarket
uv run --with mcp scripts/mcpcall.py --schema                                    # list tools + args
uv run --with mcp scripts/mcpcall.py get_scene_info
uv run --with mcp scripts/mcpcall.py execute_blender_code '{"code":"import bpy; print(len(bpy.data.objects))"}'
uv run --with mcp scripts/mcpcall.py get_viewport_screenshot '{"max_size":1400}' --save renders/viewport.png
uv run --with mcp scripts/mcpcall.py download_polyhaven_asset '{"asset_id":"floor_tiles_08","asset_type":"textures","resolution":"2k"}'
```
Telemetry of the MCP server was switched OFF (`disable_telemetry`); re-enable only via Blender Preferences ▸ Add-ons ▸ MCP for Blender.

## 3. Regenerate the store (catalog → planogram → labels → scene)
```bash
cd ~/workspaces/blender-supermarket
python3 scripts/gen_planogram.py     # layout + per-bay shelves/facings  -> scripts/planogram.json  (edit GONDOLAS/WALL_UNITS/CHILLERS here)
python3 scripts/gen_labels.py        # PIL labels, price tags, signs     -> assets/labels, assets/signs
# headless full build (≈4 s) from the base file that holds the Poly Haven materials + HDRI:
blender -b scenes/supermarket_base.blend --python scripts/build_supermarket.py -- --all --save scenes/supermarket.blend
# or rebuild inside the running GUI through MCP, stage by stage:
uv run --with mcp scripts/mcpcall.py execute_blender_code '{"code":"import sys,importlib; sys.path.insert(0,\"/home/manishsaini/workspaces/blender-supermarket/scripts\"); import build_supermarket as B; importlib.reload(B); B.build([\"clear\",\"structure\",\"library\",\"gondolas\"])"}'
uv run --with mcp scripts/mcpcall.py execute_blender_code '{"code":"import sys; sys.path.insert(0,\"/home/manishsaini/workspaces/blender-supermarket/scripts\"); import build_supermarket as B; B.build([\"walls\",\"chillers\",\"front\",\"produce\",\"signage\",\"lighting\",\"cameras\",\"render\"], save=\"/home/manishsaini/workspaces/blender-supermarket/scenes/supermarket.blend\")"}'
```
Stages: `clear structure library gondolas walls chillers front produce signage lighting cameras render`. Products = `scripts/catalog.py` (93 SKUs, fictional brands, real dimensions).

## 4. Roam around in Blender
* Viewport is set to Material Preview (EEVEE) with the store lights + HDRI.
* **Walk mode:** hover the 3D view, press **Shift+`** (or View ▸ Navigation ▸ Walk). WASD to move, mouse to look, **Tab** toggles gravity (eye height 1.65 m preset), Shift = run, Esc/Enter to exit.
* **Cameras:** `Cam_Entrance`, `Cam_Aisle`, `Cam_Shelf`, `Cam_Checkout`, `Cam_Produce`, `Cam_Overview`, `Cam_Exterior`, `Cam_TopDown`, `Cam_Walk` (animated fly-through, frames 1–510 (24 fps, ~21 s)). Numpad 0 = look through the active camera; Space plays the walk.
* Collections: `Structure`, `Fixtures/*`, `Products/*` (per gondola/unit), `PriceTags/*`, `Signage`, `Lighting`, `Cameras` — hide `PriceTags` or `Lighting` for a faster viewport.

## 5. Renders
```bash
cd ~/workspaces/blender-supermarket
blender -b scenes/supermarket.blend --python scripts/render_shots.py -- --samples 200 --res 1920x1080            # all cameras, Cycles/OptiX
blender -b scenes/supermarket.blend --python scripts/render_shots.py -- --samples 64 --res 1280x720 --cams Cam_Aisle   # quick check (~15 s)
blender -b scenes/supermarket.blend --python scripts/render_shots.py -- --eevee --samples 32 --cams Cam_Overview       # EEVEE still
blender -b scenes/supermarket.blend --python scripts/render_walkthrough.py -- --res 1280x720 --samples 24            # renders/walkthrough.mp4
scripts/pipeline_bg.sh &  tail -f logs/pipeline.log      # USD export + all stills + walkthrough in one go
```

## 6. Export for Isaac Sim (USD)
```bash
blender -b scenes/supermarket.blend --python scripts/export_usd.py            # -> export/supermarket.usdc (+ export/textures/)
```
Z-up, metres, root prim `/Supermarket`, products instanced per SKU mesh, custom properties on every product prim (`userProperties:sku`, `category`, `brand`, `price`, `slot`, `facing`, `row`) and on fixtures (`fixture`, `gondola`, `side`, `bay`, `level`).
In Isaac Sim: File ▸ Import / add reference to the `.usdc`; add colliders (Physics ▸ Colliders preset "convex hull" on `Products/*`, "triangle mesh" on `Fixtures/*`), lights are exported as USD lights.

## 6a. Validate the USD offline (no Isaac, no GPU)
Prove the export is loadable / Isaac-ready **without a GPU** — run it after every export
and before any Isaac session. Needs only the CPU-only `usd-core` wheel:
```bash
cd ~/workspaces/blender-supermarket
python3 -m venv .venv-usd && .venv-usd/bin/pip install usd-core       # one-off, ~30 MB, offline
.venv-usd/bin/python scripts/validate_usd.py                          # exit 0 = pass, 1 = fail
.venv-usd/bin/python scripts/validate_usd.py --json                   # + measured facts as JSON
.venv-usd/bin/python scripts/validate_usd.py --selftest               # sanity check (no artifact needed)
.venv-usd/bin/python scripts/validate_usd.py --repair-textures        # reconcile the CURRENT artifact's
#   floor/wall textures from assets/textures/ into export/textures/ (offline, no Blender)
```
It checks: stage opens; up-axis Z; 1 m/unit; a default prim exists; **every texture /
reference / payload / sublayer resolves** (no absolute build-machine paths, nothing
outside `export/`); the prim paths + spawn camera `compose_robot_in_store.py` needs; and
reports the physics/collider census (render-only by design — the composer adds physics).
Tests: `env -u PYTHONPATH PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv-usd/bin/python -m pytest scripts/twin/tests/`
(the `env -u`/autoload-off avoids a broken ROS `launch_testing` pytest plugin on `PYTHONPATH`).

Standalone, `validate_usd.py` is a **strict** post-export check: it exits `1` on ANY
unresolved asset (so a re-export that reintroduces the defect can't slip through), and its
verdict line says whether the failure is **BLOCKING** (a missing reference/payload/sublayer
= missing geometry) or **COSMETIC** (a missing *texture* — floor/walls load untextured but
the stage still loads and the robot still stands). The boot go/no-go (`preflight.py`, §8)
keys off that: cosmetic never blocks booting.

**Known defect (fixed in `export_usd.py`, needs a re-export to clear at the source):** the
current `export/supermarket.usdc` authors 6 dangling floor/wall textures. It is **COSMETIC**
— does **not** block loading or robot placement — and the pre-flight (§8) now **repairs it
automatically** in place from `assets/textures/`, so the captain's one command is green
without a re-export. Clear it at the source with a re-export (the fixed `export_usd.py`
authors resolvable paths); or on the current artifact with `--repair-textures`. Full
write-up: `data/twin-usd-validate-v1/report.md`.

## 6b. Compose the robot into the store — OFFLINE (no Isaac, no GPU)
Prove the robot-in-store composer works **before** any GPU session — safe beside the live
Blender GUI. Uses the same CPU-only `usd-core` venv as §6a.
```bash
cd ~/workspaces/blender-supermarket
# read-only facts / spawn pose / collider plan (JSON):
.venv-usd/bin/python scripts/compose_robot_in_store.py --report-usd
# full plan + what the GPU guard would decide right now (authors nothing, boots nothing):
.venv-usd/bin/python scripts/compose_robot_in_store.py --dry-run
# ACTUALLY author the composed stage with pure USD (still no GPU) and VALIDATE it:
.venv-usd/bin/python scripts/compose_robot_in_store.py --author-offline --save /tmp/store_with_robot.usda
#   -> exit 0 = composition is sane; 1 = a check failed. Add --json for the numbers.
#   With no --robot-usd it references a faithful pedestal+arm STUB so the placement maths
#   is exercised for real; the true robot mesh only comes from the URDF->USD import (Isaac).
```
`--author-offline` sublayers the committed store (read-only — never written), authors the
`PhysicsScene`, the 702 triangle-mesh (static) + 197 convex-hull (product/tag) colliders, and
places the robot at the aisle pose, then re-opens the result and proves: default prim + Z-up +
1 m/unit, every product **prototype** inherits a collider (authored once on the shared source —
never per-instance, never on the forbidden runtime prototype), the robot stands **on** the floor
(pedestal bottom Z ≈ floor_z), at the intended aisle spot, inside an aisle gap (not inside a
gondola), with instancing preserved. Tests (needs `usd-core`; pure-logic tests need nothing):
```bash
env -u PYTHONPATH PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv-usd/bin/python -m pytest scripts/twin/tests/
```

## 6c. Validate the robot URDF offline (no Isaac, no GPU)
Prove the **robot asset itself** is Isaac-ready before any GPU session — the last of the three
offline-checkable links. Same CPU-only venv (this side needs only numpy).
```bash
cd ~/workspaces/blender-supermarket
.venv-usd/bin/python scripts/validate_robot_urdf.py                 # exit 0 = pass, 1 = fail
.venv-usd/bin/python scripts/validate_robot_urdf.py --json          # + measured facts as JSON
.venv-usd/bin/python scripts/validate_robot_urdf.py --selftest      # sanity check (no robot needed)
```
It reads `~/workspaces/shelfbot_newgripper/description/shelfbot_gripper_isaac.urdf` with pure
`xml`+`numpy` and checks: it parses into one connected kinematic tree; **every `package://`
mesh resolves** to a real file (no dangling meshes — the same defect class as the store
textures; Isaac imports *even when a mesh is missing*, so this is the only warning you get);
no absolute build-machine paths; every dynamic link has a positive mass + physically valid
inertia; every movable joint has real limits + effort/velocity; and **store consistency** —
the URDF's pedestal-mount XY offset equals the composer's `geometry.PEDESTAL_OFFSET_FROM_ROOT`,
and the arm's home-pose collision envelope (forward-kinematics, real numbers) fits the 1.5 m
aisle it stands in. Package `shelfbot` = `~/workspaces/shelfbot_newgripper` (the repo that
declares that ROS name AND carries the gripper meshes — **not** the older `~/workspaces/shelfbot`,
which has neither); the check flags the same-name ambiguity.

**One command for all three (store + composer + robot)** — this is the pre-flight the captain
runs before booting (see §8):
```bash
.venv-usd/bin/python scripts/preflight.py                          # exit 0 = whole chain OK
```

## 7. Where things are
```
scripts/catalog.py          product catalog (SKU dims, colors, prices)      scripts/planogram.json   generated layout + shelf facings
scripts/gen_planogram.py    layout generator                                 scripts/gen_labels.py    PIL label/tag/sign renderer
scripts/build_supermarket.py Blender builder (staged)                        scripts/render_shots.py  stills
scripts/render_walkthrough.py fly-through video                              scripts/export_usd.py    Isaac USD export
scripts/validate_usd.py     offline USD validator (no GPU; §6a)              scripts/compose_robot_in_store.py  Isaac robot-in-store composer
scripts/validate_robot_urdf.py robot URDF pre-flight (no GPU; §6c)           scripts/preflight.py     one-command chain pre-flight (store+robot+composer)
scripts/mcpcall.py          stdio MCP test client                            scripts/launch_blender.sh start GUI + MCP socket
scenes/supermarket.blend    THE scene (25.7k objects, 24.9k product instances, 93 shared meshes)
scenes/supermarket_base.blend  empty base: Poly Haven floor_tiles_08, white_plaster_02, HDRI wide_street_01 (fake-user pinned)
assets/textures, assets/labels, assets/signs, renders/, export/, logs/
scripts/twin/    compose.py (pure-USD composition core), usd_probe.py, urdf_probe.py (robot URDF read), geometry.py, colliders.py, gpu_guard.py, cli.py + tests/
```

## 8. GPU session runbook — stand the robot in the store (Isaac Sim)

The captain runs this **once**, in one sitting, on the RTX 3060 (12 GB) box. All three
offline-checkable links are now proven: **store checked** (§6a), **composer checked** (§6b),
**robot checked** (§6c). This session verifies only what a real simulator can: the URDF→USD
import, PhysX loading the colliders, and the robot resting under gravity. **Close/save Blender
first** — the guard refuses beside it and won't `--force` past that (see
[docs/isaac_robot_in_store.md](docs/isaac_robot_in_store.md)).

> **Status (2026-09-19): the one-command run is PREPARED and WAITS for the owner. No
> simulator boot has been performed.** The bounded run wrapper below, the base-height
> settle log, the offline pre-flight and tests are all in place and green, but the Isaac
> Sim boot is the repository owner's decision and was explicitly deferred ("not yet").
> When the owner approves, the whole run is the single command
> `bash scripts/run_robot_in_store_gpu.sh`. Prep write-up + PENDING boot sections:
> [`data/twin-robot-in-store-v1/report.md`](data/twin-robot-in-store-v1/report.md).

**The bounded one command (recommended) — `scripts/run_robot_in_store_gpu.sh`:** wraps the
real run with a readiness gate (no Blender, ≥8 GB free VRAM, ≥12 GB RAM), a **hard 20-min
wall-clock cap** (`TIMEOUT_S`), a **RAM watchdog** that stops only its own process group by
PID if available RAM drops below 4 GB (`RAM_FLOOR_GB`), and a peak GPU/RAM sampler
(`run_metrics.txt`). It passes `--store <this repo's export>` and never uses `--force`.
```bash
cd ~/workspaces/blender-supermarket
bash scripts/run_robot_in_store_gpu.sh                 # bounded real run (once approved)
HEAVY_MB=2500 bash scripts/run_robot_in_store_gpu.sh   # see the whispr note below
```

> **GPU-guard heavy-consumer OPTION (the guard default is unchanged):** the guard's default
> heavy-consumer threshold stays **1500 MiB**. On this box the always-on `whispr` service
> (`~/whispr/server.py`, ~2.2 GB VRAM, protected — never freed) exceeds that, so the guard
> will **correctly refuse** the boot as-is. That is the safe default. To accept whispr as a
> known-stable consumer, run with **`HEAVY_MB=2500`** (opt-in, raises `--heavy-threshold-mb`
> for that run only) — this is **not** `--force`, and the real OOM protection (free VRAM
> ≥ 8 GB) is unchanged. Do not bake 2500 in as the default.

**Pre-flight — ONE command checks the whole chain (safe with Blender still open, no GPU):**
```bash
cd ~/workspaces/blender-supermarket
.venv-usd/bin/python scripts/preflight.py     # store + robot + composer; exit 0 = safe to boot
```
It runs `validate_usd.py` (store), `validate_robot_urdf.py` (robot URDF + aisle fit) and the
`--author-offline` composition, and prints one verdict. Before validating the store it
**repairs the current export in place** (copies any authored-but-missing floor/wall texture
from the versioned `assets/textures/` into the gitignored `export/textures/` — offline, no
Blender), so the one command is self-sufficient on a stale on-disk export.

**This is expected to be GREEN.** The steady state is:
```
  PASS    STORE    (export USD Isaac-loadable)
  PASS    ROBOT    (URDF Isaac-ready + fits aisle)
  PASS    COMPOSER (offline composition sane)
VERDICT: PASS — ... Safe to boot (close Blender first).
```
The three still run standalone (`validate_usd.py`; `validate_robot_urdf.py`;
`compose_robot_in_store.py --author-offline`) if you want them one at a time.

**Reading a red (or amber) result — a section is one of three states:**

| Verdict / mark | Exit | Meaning | What to do |
|---|---|---|---|
| `PASS`  | 0 | clean | **Boot** (close Blender first). |
| `PASS*` = **COSMETIC** | 0 | a real but non-blocking defect remains — e.g. a floor/wall texture that could **not** be repaired (missing from `assets/textures/`). The surface loads untextured; geometry, scale and placement are intact. | **Boot** — this does not block the URDF import, physics or placement. Fix the texture at leisure (restore the map in `assets/textures/`, re-run, or re-export). |
| `FAIL` = **BLOCKING** | 1 | a structural defect — bad up-axis/scale, no default prim, no floor, a missing composer prim path, a dangling *reference/payload/sublayer* (missing geometry), or the robot/composer step failed. | **Do NOT boot.** Read which section failed; the per-check line names the cause. A store re-export or the named robot/composer fix is required first. |

So a red pre-flight now always means "do not boot"; an untextured-surface defect shows as
`PASS*`/exit 0, never as a boot-blocking red. `--no-repair` shows the raw on-disk state
(the known dangling textures then surface as a `[COSMETIC]` line) without touching `export/`.

**The GPU session, in order:**
```bash
# 1. Close/save Blender (frees the ~12 GB GPU + lifts the guard's hard refusal).
tmux kill-session -t blender          # or File ▸ Save in the GUI, then quit
nvidia-smi                            # confirm no blender / heavy process; ~11 GB free

# 2. Dry-run under Isaac's env to confirm the guard now PASSES (no boot yet):
bash scripts/compose_robot_in_store.sh --dry-run
#   -> the "GPU guard" block should read "GPU check passed: <N> MiB free". If it still names
#      blender, Blender is not fully closed — stop and close it.

# 3. THE real run — boots Isaac Sim, imports the URDF, authors colliders/physics, places the
#    robot, steps, and writes the stage + a screenshot:
bash scripts/compose_robot_in_store.sh \
     --save /tmp/store_with_robot.usd --screenshot /tmp/store_with_robot.png
```

**What a good run prints (to stderr, in order):**
- `GPU check passed: <N> MiB free of 12288.`
- `[compose] imported robot -> /tmp/store_with_robot_robot.usd (prim /shelfbot…)`
- `[compose] authored /tmp/store_with_robot.usd (702 tri + 197 convex colliders, 196/196 prototypes)`
- `[compose] screenshot -> /tmp/store_with_robot.png` then `[compose] wrote /tmp/store_with_robot.usd`

**What "good" looks like:** the screenshot shows the UR10 on its pedestal standing in the aisle
(≈ X −1.5, Y 0.75), upright, feet on the floor, ~0.95 m clear of the shelves either side; the
run exits 0. Re-open `/tmp/store_with_robot.usd` in the Isaac GUI to confirm nothing fell through
the floor or exploded.

**Most likely failures and the fix:**
| Symptom | Cause | Fix |
|---|---|---|
| `REFUSING … a protected GPU session … blender` | Blender still holds the GPU | Close/save Blender; re-check `nvidia-smi`. `--force` will NOT override this. |
| `REFUSING … only <N> MiB free` | another consumer (or a stale Isaac) holds VRAM | Free it; or `--force` to accept the OOM risk on a genuinely idle GPU. |
| URDF import errors on `package://ur_description/…` | ROS package path not sourced | Use the `.sh` wrapper (it sources `/opt/ros/jazzy` + puts `ur_description` on the path); confirm `/opt/ros/jazzy/share/ur_description` exists. |
| Robot imports but the **gripper is invisible** (arm only) | `package://shelfbot/…gripper meshes` didn't resolve — Isaac imports even when a mesh is missing | The gripper meshes live only in `~/workspaces/shelfbot_newgripper` (NOT `~/workspaces/shelfbot`). The `.sh` wrapper now builds a shim pointing package `shelfbot` there; `scripts/validate_robot_urdf.py` proves it offline **before** you boot. |
| Isaac OOM-kills / crashes at boot | 12 GB is tight; RTX scene + import together | Retry with Blender fully closed and nothing else on the GPU; `--no-headless` is heavier — keep the default headless. |
| Robot floats above / sinks into the floor after stepping | imported root convention differs from the pre-import estimate | The offline pose is measured (pedestal base at floor Z=0); if the imported root isn't the URDF `world` link, add a correction pass after `twin.compose.place_robot` (re-measure the pedestal prim and nudge) and re-run. Geometry is unchanged — no re-export needed. |

**If it goes wrong:** nothing here writes to the repo (`scenes/`, `export/` untouched; outputs go
to `/tmp`), so a failed run is safe to discard. Capture the stderr + `nvidia-smi`, `git stash`
nothing, and re-run the offline `--author-offline` to confirm the *stage* is still sane before
blaming the composer — a boot/OOM failure is an Isaac/VRAM problem, not a composition one.

## 12. Mobile restock navigation (CPU checks and isolated Nav2)

Contracts, footprint limits and unverified simulator links:
[docs/restock_navigation.md](docs/restock_navigation.md). No command here boots
Isaac or commands a physical robot. The integration runner owns the simulator.

```bash
# Deterministic CPU fixtures; no ROS/GPU required.
python3 -m unittest discover -s scripts/twin/tests -p test_navigation.py -v
python3 -m unittest discover -s scripts/twin/tests -p test_articulated_navigation.py -v
python3 -m unittest discover -s scripts/twin/tests -p test_navigation_corridor.py -v

# Optional host-Python Jazzy overlay, extracted only into task-owned storage.
# Uses existing trusted apt indexes; no sudo, package install, or service restart.
python3 scripts/setup_restock_nav2.py --output "$TASK_OUTPUT/nav2"
source "$TASK_OUTPUT/nav2/env.sh"

# Reproduce full-margin planning and feedback on the actual exported static map.
# This is a TEST_FIXTURE controller check, not an Isaac/PhysX demonstration.
python3 scripts/check_restock_navigation.py --environment "$ENVIRONMENT_MANIFEST" \
  --scenario "$RESTOCK_SCENARIO" --position-uncertainty 0.10 \
  --output "$TASK_OUTPUT/navigation-proof"

# In one isolated ROS shell, start the articulated FollowPath configuration.
python3 scripts/run_restock_nav2.py --environment "$ENVIRONMENT_MANIFEST" \
  --articulated --output "$TASK_OUTPUT/nav2-map" --launch

# In another shell with the same task-owned env.sh sourced, run the sensor bridge.
# The pose is a declared deployment prior, supplied by the integration scenario.
python3 scripts/run_articulated_nav2_bridge.py --environment "$ENVIRONMENT_MANIFEST" \
  --initial-pose "$INITIAL_X" "$INITIAL_Y" "$INITIAL_YAW"

# ROS-message handoff tests, after sourcing the task-owned Jazzy overlay.
python3 -m unittest discover -s scripts/twin/tests -p test_nav2_handoff.py -v

# With a compatible Jazzy/Nav2 environment already available, generate map/config.
# scenario.json supplies bounds, obstacles, resolution and the measured footprint.
# OUTPUT must be a task-owned scratch directory, outside scenes/export/renders.
python3 scripts/run_restock_nav2.py --scenario scenario.json --output "$OUTPUT"

# Start only the ROS navigation nodes; domain 88 must be reserved for this task.
ROS_DOMAIN_ID=88 ROS_LOCALHOST_ONLY=1 ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST \
  python3 scripts/run_restock_nav2.py --scenario scenario.json --output "$OUTPUT" --launch
```

Example **test-fixture** scenario schema (replace all dimensions with scene/rig
measurements before runtime):

```json
{
  "bounds": [-6, -6, 6, 6],
  "obstacles": [[2, 2, 3, 3]],
  "resolution": 0.2,
  "footprint": {
    "base_radius": 0.6,
    "hitch_offset": 0.5,
    "cart_hitch_to_farthest_corner": 1.2,
    "margin": 0.1,
    "min_turn_radius": 1.0,
    "max_articulation": 1.1
  }
}
```

Feed real wheel encoder and range samples through the adapter described in the
navigation document. Never connect raw Nav2 `cmd_vel` to the base: the adapter
returns the gated command. Missing Nav2 packages are a prerequisite failure;
this script intentionally does not install/upgrade shared ROS or Isaac.
For an integration scenario with staging support, add `--scenario "$RESTOCK_SCENARIO"`
to `check_restock_navigation.py` and `--integration-scenario "$RESTOCK_SCENARIO"`
to both `run_restock_nav2.py` and `run_articulated_nav2_bridge.py`.
This includes the support obstacle and checks the scenario’s actual chassis goal;
a successful environment-only route does not certify a different scenario.

## 11a. Navigation lane factory for the mission bundle

The mission bundle factory (`twin.restock.isaac_bundle:create`, integration lane)
resolves each lane's controller from the runtime-config as a `module:function`
reference and calls it `fn(app=, assembly=, plan=, config=, output=)`. The
navigation lane reference is **`twin.navigation.factory:nav_factory`**; set it as
the runtime-config key **`nav_factory`**. The returned object is wrapped by
`twin.restock.runtime.NavigationBridge` and driven per `plan.nav_mode`:

- `reference` → `ArticulatedController` (measured-state route following);
- `sensor` → `twin.navigation.runtime.SensorNavigation` (wheel odometry + map
  localizer + the articulated controller); packet odometry carries wheel encoder
  angles and packet navigation carries the raw scan;
- `nav2` → `twin.navigation.nav2.ArticulatedNav2Adapter` (domain-88 loopback Nav2
  FollowPath overlay). Constructing it REQUIRES `ROS_DOMAIN_ID=88`,
  `ROS_LOCALHOST_ONLY=1`, loopback discovery; it is the only mode that touches ROS.

The factory builds the shared controller from the STATIC navigation manifest in
`config['navigation']` (same conservative obstacle map used for gating/validation;
live obstacles are always sensed) and NEVER enables test fixtures — the returned
controller only accepts `wheel_odometry_localized`/`simulated_lidar` provenance,
so no ground-truth or fixture input can drive the base in production. It validates
at build time that the returned object exposes exactly the methods the bridge
calls (`start(goal, coupled, goal_articulation=)`, `cancel()`, the per-mode
`step(...)`, and `publish_sensors(...)` for nav2), failing closed otherwise.

`config['navigation']` keys: `environment` (manifest object) or
`environment_path`; optional `scenario`/`scenario_path`,
`camera_config`/`camera_config_path`, `mobile_spec`, `require_temporal_map`,
`goal_timeout`; and for `sensor` mode a `sensor` block with `wheel_radius`,
`wheel_separation`, `initial_pose`, `initial_map_pose`. For `nav2` mode an
optional `nav2` block may set `emit_assets`/`nvblox_topic`/`params_template`.

```bash
# Deterministic CPU contract tests (no ROS/GPU/Isaac, no fixtures).
PYTHONPATH=scripts PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest \
  scripts/twin/tests/test_navigation_factory.py -q
```

Integration owns the native run: the live `assembly`/`SimulationApp`, the Nav2
launch, and all GPU/physics validation. The factory performs no simulator boot
and is proven only at the CPU contract level here.

## 9. Mobile UR10 and rollwagen (CPU)

Interface and remaining simulator checks: [mobile_robot_rollwagen.md](docs/mobile_robot_rollwagen.md).
Use a Python environment with `numpy` and `usd-core`; no Isaac boot is needed.
`SOURCE_URDF`, `SHELFBOT_PACKAGE` and `TASK_OUTPUT` are caller-selected paths;
`TASK_OUTPUT` must be a scratch directory, outside `scenes/`, `export/`, `renders/`.

```bash
python scripts/build_mobile_assets.py \
  --source-urdf "$SOURCE_URDF" \
  --package-root /opt/ros/jazzy/share --package-root "$SHELFBOT_PACKAGE" \
  --output-dir "$TASK_OUTPUT"
# Optional --config mobile-config.json accepts MobileSpec keyword overrides.
PYTHONPATH=scripts PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest scripts/twin/tests/ -q
```

Import the generated mobile URDF with a free base and fixed frames preserved;
install sensors and position-hold the manifest's transport joints before enabling
wheel commands. Import and GPU verification belong to the integrated simulator
run and remain subject to the existing protected-Blender/resource guard.

## 10. Service backroom and environment contract

The `environment` build stage adds an illustrative German grocery service area
and keeps existing catalog/planogram assignments. Set `OUT` to an **absolute,
non-RAM scratch directory**, outside the repo's scenes/export/renders directories:

```bash
python3 scripts/gen_labels.py
blender -b scenes/supermarket_base.blend -t 4 --python-exit-code 1 \
  --python scripts/build_supermarket.py -- --all --save "$OUT/store.blend" \
  --manifest "$OUT/environment.json"
blender -b "$OUT/store.blend" -t 4 --python-exit-code 1 \
  --python scripts/check_environment.py -- --manifest "$OUT/environment.json" \
  --out "$OUT/geometry-check.json"
blender -b "$OUT/store.blend" -t 4 --python-exit-code 1 \
  --python scripts/export_usd.py -- --out "$OUT/supermarket.usdc"
.venv-usd/bin/python scripts/validate_usd.py --store "$OUT/supermarket.usdc" --repair-textures
blender -b "$OUT/store.blend" -t 4 --python-exit-code 1 \
  --python scripts/render_shots.py -- --cpu --samples 16 --res 960x540 \
  --cams Cam_Aisle,Cam_Backroom --out "$OUT/previews"
python3 -m unittest discover -s scripts/tests -v
```

The manifest uses metres, Z up, `map` world coordinates and planar `[x,y,yaw]`
poses. `stocking_stations` reference existing shelf `slot` IDs and SKU lists;
`backroom_stocks` list six case origins per source bin. `static_obstacles` are
conservative world AABBs measured from the generated structure and fixtures,
intersecting height band 0.02–2.6 m (floors and overhead structure excluded).
This is a static map and scenario ground truth, **not perception evidence**.
Schema 2 defines every `poses` entry's body through `pose_bodies`:
`cart_dock` is the cart body at [-7,16,0], while `robot_dock` and `pickup`
are the robot base at [-5.73,16,0] (1.27 m aligned separation). Planar poses
omit ground height: mobile base-link starts at Z=0.24 m, cart body at Z=0.20 m.
Each stocking station's `pose` is the **chassis** pose (`pose_body=robot_base`);
`arm_shelf_facing_yaw` is a separate desired world direction, not an arm joint
target or permission to rotate the chassis. Ordinary aisle chassis headings
are -pi/2, keeping a straight trailing cart along the aisle.

`first_cycle_station=stock_G3_back` selects an endcap on the rear cross aisle:
robot staging [-3.05,6,pi], arm facing -pi/2, with a westbound straight
pull-through from [-1.525,6,pi] to [-7,6,pi]. The corresponding slots are
`G3_back_s*`; `first_cycle_source_bin=backstock_00` carries
`first_cycle_sku=juice_orange`. `pull_through_robot_poses` describes only that
local forward segment, not a backroom-to-shelf trajectory. Navigation must
plan arrival and return using independently predicted cart poses.

To additionally check that segment with the actual mobile module (after it is
available in the checkout), add `--mobile-module scripts/twin/mobile.py` to
the geometry command. This tests its default-margin convex swept envelope
against the measured static AABBs. It proves static straight-segment clearance,
not turn feasibility, grasp reach, control tracking, or dynamics.
`layout(plan)` alone omits measured obstacles; request `--manifest` on a full
build to obtain them. `poses`, `zones`, `portal`, `map_bounds`, and `route_hints`
are available to scenario/navigation adapters. Route hints are not a planner.

The rear opening is 2.5 m wide × 2.8 m high and meets the existing fixture gap.
The backroom has a continuous floor at Z=0, racks, receiving staging, cart bays,
and a 3.2 m turning zone. `check_environment.py` samples a 0.65 m circular
footprint along the route to the G3/G4 aisle; it does **not** prove a coupled
cart's swept turns or dynamics. `--radius` can increase this check to include a
planner's safety margin; values below the configured 0.65 m transport envelope
are rejected. Shelf approaches require backing out to the rear cross aisle
instead of U-turning. The mobile controller disables reverse towing, so this
return guidance applies only when uncoupled. A feasible forward-only coupled
route or a separate cart parking/uncoupling procedure must be planned and checked
by integration; the route hints and 3.2 m zone do not establish one.

Source cases are static visual stock, with `sku`, `stock_id`, and `units` custom
properties. Scenario composition must explicitly promote/replace selected stock
with dynamic actors and populate live inventory from its own observations.
Movable robots/carts are authored by the robotics composer. Markers live under
`GRP_EnvironmentMarkers`; decoration under `GRP_EnvironmentDecor` is excluded by
the collider classifier. All solid service fixtures use the existing static
structure/fixture collision groups. Labels/styles are illustrative, not an exact
scan or an assertion of REWE store dimensions.

## 11. Mobile restocking mission (CPU integration checkpoint)

The mission/inventory core is independent of Isaac and ROS. Its fixtures test
transactional stock state, evidence checks and fail-stop requests; they are not
an autonomous robot demonstration. Current limitations and GPU diagnosis:
[data/mobile-restock-v2/report.md](data/mobile-restock-v2/report.md).

```bash
python3 -m unittest discover -s scripts/twin/tests -p test_restock_mission.py -v
python3 -m unittest discover -s scripts/twin/tests -p test_restock_runtime.py -v
```

Runtime adapters implement `start(action, job)`, `step(dt) -> Result`, `stop()`
from `scripts/twin/restock/mission.py`. Mission time starts at zero, so adapters
must express evidence timestamps relative to that origin. Production rejects
fixture and ground-truth provenance. `runtime.py` wires the controller ports;
`run_mobile_restock.sh` performs CPU checks unless `--execute` is explicit.
A reviewed simulator bundle factory providing actual sensors, controls, stage
construction and recording is still required; no mock fallback is installed.

The default preflight is CPU-only and never probes the GPU. Supply the environment
worker's final manifest; the checked-in scenario retains an explicit runtime
readiness gate. The articulated route check must use the actual 0.10 m or larger
localization uncertainty; earlier 0.02 m fixture evidence is insufficient.

```bash
bash scripts/run_mobile_restock.sh --manifest "$ENVIRONMENT_MANIFEST" \
  --scenario scripts/restock_scenario.json
```

The host arm executor runs in ROS host Python after sourcing the task-owned
overlay. Launch only with the authorized domain-88 simulator session. It consumes
planned trajectories, publishes stamped joint-drive targets, and checks measured
feedback; it never commands physical hardware or writes joint state:

```bash
ROS_DOMAIN_ID=88 ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST ROS_LOCALHOST_ONLY=1 \
  PYTHONPATH=scripts python3 -m twin.restock.trajectory_host --urdf "$MOBILE_URDF"
```

Endpoint: `/restock_sim/arm_controller/follow_joint_trajectory`. Bounds are
0.15 rad/s, 0.5 rad/s², 0.15 rad tracking error and 0.01 rad terminal error with
measured speed at most 0.02 rad/s. Plans require explicit waypoint velocities,
zero-time rest starts and rest ends. Faults/cancellation latch stop; recovery
requires reconciliation and restarting the task-owned executor.

The integration station supervisor is a separate host-Python process, started
only by the guarded runtime after GPU release. It listens on
`/restock_sim/station/command` and publishes `/restock_sim/station/status`.
`prepare`, `heartbeat` and `stop` carry monotonic parked epochs, sequence numbers
and simulation timestamps. Each new station starts a fresh task-owned host
planner/mapper session; stale/moving parking or process failure latches stop.
Readiness requires exact ROS endpoints and the real NVIDIA ESDF incorporated-depth
timestamp. The command is:

```bash
python3 -m twin.restock.station_host --config "$HOST_CONFIG" --output "$NEW_STATION_LOGS"
```

`restock.native_cameras` authors camera mounts and collision boxes from supplied
calibration, acquires RGB plus registered depth, and switches one native ROS
mapper stream at parked epochs. It reads actual OpenCV lens parameters because
Isaac5.1's aperture-based `get_intrinsics_matrix` assumes a centred principal
point. These runtime APIs have CPU contract tests but no rendered/DDS proof yet.
Version-matched interfaces: [Camera API](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/py/source/extensions/isaacsim.sensors.camera/docs/index.html)
and [native ROS camera helper](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/py/source/extensions/isaacsim.ros2.bridge/docs/ogn/OgnROS2CameraHelper.html).

After a complete simulator bundle exists and preflight is green, `--execute`
requires `--factory module:function` and a **new** `--output` scratch directory.
Use `RESTOCK_PYTHON` for the installed Isaac Python interpreter. The bundle receives
`app`, `manifest`, `scenario`, `config`, `output` and returns `adapter` plus `close()`; it owns
actual sensor acquisition, controller setup and recording. `--runtime-config`
supplies exact host/CAD/camera paths, checked before any GPU acquisition or run
directory creation; its scenario must match the mission scenario.
`recording.Recorder`
accepts real rendered RGB with unique render IDs and strictly increasing acquisition
timestamps; its H.264 encoder uses a caller-supplied local ffmpeg executable.
Variable frame intervals are preserved with a microsecond container timebase and
verified by decoding every presentation timestamp. Frames are never interpolated
or duplicated; cached acquisition IDs must be skipped by the capture caller.
No ready bundle is currently
shipped. The wrapper caps wall time at 20 minutes. GPU guard defaults stay intact;
`--heavy-threshold-mb 2500` is the existing explicit whispr opt-in, never `--force`.

The independent `scripts/run_restock_render_smoke.py --help` command validates
native RTX RGB-D acquisition and H.264 under a 300-second cap. Supply a verified
matching NVIDIA runtime with `--nvidia-runtime`, installed interpreter with
`--isaac-python`, `--ffmpeg` and a new `--output`. Its process-local aliases cover
Kit's unversioned NVML/CUDA requests without changing system libraries. Optional
`--runtime-config CONFIG --view source` composes a full-store **static optical
bench** with actual robot/cart CAD and physical camera mounts. This bench disables
dynamics at initial composition and is not evidence of robot motion or autonomy.
It preserves same-acquisition depth and instance-ID path mappings for verification;
those IDs may only identify known rig self pixels, never product inventory.

CPU assembled preview (URDF collision geometry, **not** a physics screenshot):

```bash
blender -b --factory-startup --python scripts/preview_mobile_restock.py -- \
  --manifest "$TASK_OUTPUT/mobile_manifest.json" \
  --package-root /opt/ros/jazzy/share --package-root "$SHELFBOT_PACKAGE" \
  --output "$TASK_OUTPUT/mobile-preview.png"
python3 scripts/build_restock_report.py --manifest "$ENVIRONMENT_MANIFEST" \
  --narrative data/mobile-restock-v2/report.md \
  --evidence data/mobile-restock-v2/evidence.json --output "$REPORT_HTML" \
  --image "Static mobile collision preview=$TASK_OUTPUT/mobile-preview.png"
```

Repeat `--image 'Caption=PNG path'` to embed actual before/after environment
renders. The resulting HTML includes all images, CSS and map geometry inline.
Generated HTML is a cheap CPU artifact kept outside git; report source, evidence
summary and generator are versioned. No MP4 exists until actual runtime capture.


Actual-URDF reach check (NumPy/SciPy environment, CPU only). This solves six
contact/approach endpoints but does **not** certify collision-free access:

```bash
python scripts/check_restock_reach.py --mobile-manifest "$TASK_OUTPUT/mobile_manifest.json" \
  --scenario scripts/restock_scenario.json --output "$TASK_OUTPUT/reach-check.json"
```

The current G3 endcap candidate has a measured gripper/upper-shelf obstruction;
see the report. A successful IK exit code must not be presented as task viability.
# CPU side-contact candidate audit

`scripts/restock_scenario.json` defines carton face/yaws and robot/cart poses.
`scripts/check_restock_reach.py` checks the actual mobile URDF at source, cart,
and shelf contact/approach/lift/preplace poses. Its JSON binds the scenario and
URDF hashes. `scripts/check_restock_sweep.py` consumes that JSON with `--reach`,
plus `--mobile-manifest`, `--scenario`, `--store`, repeated `--package-root`, and
`--output`. Use the composed initial-scenario USD as the store input.

The sweep is a conservative rejection test for joint-linear candidate segments,
including all link collision geometry and a rigidly carried handling unit.
Disconnected shelf-frame solids retain their openings. Bounds include at most
2 mm of between-sample displacement. An overlap requires investigation; a clear
result would still leave self collision, transit paths, actual planned trajectory,
sensor perception and contact physics unverified. This command never boots Isaac.

`restock.scene.author_initial_scenario` authors the shallow handling unit and
support and deactivates only exact-slot product prims in the output layer before
physics. It preserves fixtures, price tags and the input store. It refuses repeat
authoring and does not create runtime motion or inventory observations.

## 13. Mobile restock manipulation (CPU)

Contracts, dependencies and limits: [restock manipulation](docs/restock_manipulation.md).
No simulator is started by these tests:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest scripts/twin/tests/test_restock_manipulation.py scripts/twin/tests/test_cad_restock.py -q
```

Before integrating `RosTransferBridge` / `RosMotionPort` into an already running
simulation executor, configure the isolated ROS process (and all task-owned
planner/controller peers) and create its node with `use_sim_time=true`:

```bash
source /opt/ros/jazzy/setup.bash
export ROS_DOMAIN_ID=88
export ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST
unset ROS_STATIC_PEERS
```

The integration launch owns the RGB-D preprocessing, current mobile-arm TF,
collision world, simulated gripper/contact bridge and MoveIt/cuMotion servers.
Missing sensing or adapters are a failure, not a fixture fallback. See the linked
contract for topic schema and acknowledgement requirements. Full GPU execution
remains subject to §8's resource guard and integration ownership.

The CAD native-gripper/carton authoring tests additionally need `pxr`. Use the
CPU USD Python configured in §6a (or the existing shelfbot USD-only wrapper):

```bash
PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  bash "$SHELFBOT_ROOT/tools/usdpy" -m pytest scripts/twin/tests/test_cad_restock.py -q
```

Set `SHELFBOT_ROOT` to the read-only source checkout selected by integration.
This wrapper loads USD libraries only; it does not boot Isaac. The direct
`CadRestockRuntime` assembly/asset authoring order is documented in
[Actual CAD gripper and concrete Isaac path](docs/restock_manipulation.md#actual-cad-gripper-and-concrete-isaac-path).

For process-separated ROS, run the controller with **host Jazzy Python**, from
the repository root, after sourcing the task's MoveIt/cuMotion ROS overlay:

```bash
python3 -m scripts.twin.restock_manipulation.ros_host \
  --config scripts/twin/restock_manipulation/ros_host_config.json \
  --urdf "$ROBOT_URDF" --shelfbot-root "$SHELFBOT_ROOT" \
  --mobile-manifest "$MOBILE_MANIFEST"
```

This launches domain-88 clients/controller only. Integration enables
`isaacsim.ros2.bridge` in its existing guarded app and constructs `IsaacRosGraph`
with real native contact and a **latched** simulated-arm hold callback. Call
`start(operation, job)`, `publish_sensors(Sensors)`, and `poll()` each physics step.
Publish fresh sensor acquisition packets only; call `poll()` even without images.
The domain/loopback environment above also applies to Isaac. Start the host and
allow DDS discovery before submitting the first job.

Cross-interpreter CPU contract smoke (neither interpreter boots a ROS graph or GPU):

```bash
"$ISAAC_PYTHON" scripts/restock_ros_contract_smoke.py --ros-python "$ROS_PYTHON"
```

`ISAAC_PYTHON` selects the existing Isaac Python3.11 executable and `ROS_PYTHON`
the sourced host Jazzy Python3.12 executable. This verifies fixture-only schema
roundtrip and actual host ROS message serialization; it does not test DDS or
OmniGraph. Runtime server and attachment requirements are in the manipulation doc.

## 13. Restock inventory sensing factory (CPU contract)

The mission bundle (`twin.restock.isaac_bundle`) resolves the inventory-sensing lane from
the runtime-config:

```
config['sensing_factory'] = 'twin.restock_camera.sensing:build_sensing'
```

`build_sensing(app, assembly, plan, config, output)` returns `{'capture', 'recording_camera'}`.
`capture(now, source, destination)` returns a `restock_manipulation.types.Sensors` packet
(or `None`) from the multi-camera `StationSensors`, under the 500 ms freshness, exact-frame
identity, source-immutability/tamper and fresh-clearance gates (consuming the mapper
`collision_time`); it never fabricates an observation. The live cameras, the TF
`measure_context` and mapper `clearance_for_slot` callbacks, the motion port and the contact
reader are integration-owned and supplied via `assembly.sensing_sources` — the exact
contract is the `twin.restock_camera.sensing` module docstring and
[docs/restock_camera.md](docs/restock_camera.md). `support_slots(camera_config, camera_id,
...)` and `rendered_context(...)` build the per-camera slots/contexts from the camera design.
CPU contract tests (no Isaac/ROS, no GPU):

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest scripts/twin/tests/test_restock_sensing_factory.py -q
```

Commission the contexts from observed empty supports before the first capture; a mission
readiness claim still needs integration's live GPU run and the `InventoryVerifier`'s two
post-boundary confirmations.

### 13a. Commission empty-support baselines (CPU plan + persistence/loader)

`scripts/run_restock_commissioning.py` runs the commissioning choreography (camera lane).
Default `--plan` (CPU, no Isaac) prints the per-station plan (station → mission location →
camera → capture pose → required footprint / markers / lifetime) from the camera design +
mission scenario — the plan the native run executes; it captures nothing:

```bash
PYTHONPATH=scripts python3 scripts/run_restock_commissioning.py \
  --camera-config "$CAMERA_CONFIG" --scenario "$SCENARIO"      # covers backstock_00 + shelf
```

`--execute` runs the choreography: with each target station EMPTY and the robot at its
capture pose, it captures the eye-in-hand/station RGB-D frame, commissions the support via
the production `commission()` path, and persists the baseline with provenance (session id,
station, capture pose, stamp) under `--output`. All robot motion, empty staging, frame
capture and empty-polygon selection are integration/GPU-owned and supplied by a
`--sources module:function` hook (the script imports no Isaac/ROS). The mission picks the
baselines up at boot via `config['commissioning_dir']` (see `twin.restock_camera.sensing`);
a required station with no baseline is refused, never fabricated. Pure logic + persistence,
loader and refusal live in `twin.restock_camera.commissioning`. CPU tests (no Isaac/GPU):

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest scripts/twin/tests/test_restock_commissioning.py -q
```

## 14. Controlled static-scene navigation mapper commissioning

This is an uncommissioned integration handoff, not a moving-mission readiness
claim. Use the existing task plugin environment selected by `RESTOCK_PLUGIN_ENV`.
Set a unique ROS-safe `NAV_GENERATION`, actual camera count `NAV_CAMERA_COUNT`,
and measured complete transport height in metres `NAV_TRANSPORT_HEIGHT`.

CPU plan only (no ROS node or GPU):

```bash
source "${RESTOCK_PLUGIN_ENV:?existing task plugin environment required}"
python3 scripts/launch_navigation_mapper.py \
  --generation "${NAV_GENERATION:?}" --camera-count "${NAV_CAMERA_COUNT:?}" \
  --transport-height "${NAV_TRANSPORT_HEIGHT:?}"
```

Only integration may append `--execute`, inside its existing bounded GPU launch
with the resource checks and lock already held. This starts the navigation-only
mapper in domain 88/loopback, not Nav2 or robot motion. The launch refuses other
ROS domains/discovery settings. Keep generation-specific input publishers closed
until empty-map evidence is established. Terminate only the exact task-owned
process when reconstructing; never touch the parked manipulation mapper.

Canonical output is `/restock_sim/navigation/nvblox_node/static_map_slice`;
the plan prints generation-specific depth/CameraInfo inputs. **Never call
`get_esdf_and_gradient` on this 2D mapper: upstream exits the node in 2D mode.**
It is listed only under forbidden probes. Feed registered, self-filtered native depth without manipulation target
removal. Publish `/clock` and acquisition-time TF into stable `map`.

Provenance is consumed from the host typed `restock_mapper_evidence/EvidenceSnapshot`
envelope (the plan prints its `evidence` topic/service names), NOT from the ESDF
service. The navigation consumer is `twin.navigation.temporal.accept_envelope`; its
contract and the permanently closed full-height gate are documented in
[the temporal mapping contract](docs/navigation_temporal_mapping.md#consumed-typed-evidence-envelope).
CPU consumer tests:

```bash
PYTHONPATH=scripts PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest \
  scripts/twin/tests/test_navigation_evidence.py \
  scripts/twin/tests/test_navigation_temporal.py \
  scripts/twin/tests/test_navigation_mapper_launch.py -q
```

Before accepting any wheel command, satisfy the evidence table and unresolved
native API binding in [the temporal mapping contract](docs/navigation_temporal_mapping.md#executable-commissioning-handoff).
**This remains UNCOMMISSIONED:** no native producer populates `EvidenceSnapshot`
and the `EmptyGeneration`/`OpenAdmission`/`RegisterAcquisition` services are
unimplemented native hooks, so the launch does not prove an empty generation,
supply a real incorporation ack, commission the epoch, or bypass
unknown/stale/clearance checks. Integration must then demonstrate actual departure
and blocked-aisle stop/replan.
