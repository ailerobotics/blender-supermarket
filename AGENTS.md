# Project agent memory

This file is the project's committed home for project-intrinsic agent knowledge: build, test, release, architecture, and sharp-edge notes that should travel with the code.

## Do NOT touch the live Blender session

A Blender **GUI session has `scenes/supermarket.blend` open** (since 2026-09-06)
with its MCP socket listening on `localhost:9876`. Do **not** close, kill or save
over it, do **not** run `scripts/launch_blender.sh`, and do **not** write to
`scenes/*.blend`. Reading those files is fine. Do all headless verification with
`blender -b … --python …` writing to a **scratch output path**, never into the
repo's `scenes/`, `export/` or `renders/`.

## Orientation

- **[README.md](README.md)** — the design (why the store is generated), the four
  stages (catalog → planogram → labels → scene), what the base scene carries, the
  cameras, the USD export, the file policy, and a dated decision log.
- **[RUNBOOK.md](RUNBOOK.md)** — every command (launch, MCP, regenerate, render,
  export). Don't duplicate commands elsewhere; point here.
- **File policy**: an artifact is gitignored only when its regeneration is both
  offline AND cheap (seconds–minute) — built scene, labels/signs, `export/`.
  Versioned: the base scene + `assets/textures/` (network), `report_assets/`,
  `scripts/planogram.json`, and the GPU-hour render evidence (`renders/`,
  `walkthrough.mp4`, `report.html`). See README's file-policy table (with measured
  costs) before adding or committing any large file.
- **[docs/isaac_robot_in_store.md](docs/isaac_robot_in_store.md)** — standing the
  fleet's UR10 in the export under Isaac Sim (`scripts/compose_robot_in_store.py`).
  `--report-usd` / `--dry-run` / **`--author-offline`** all run with **no GPU boot**
  (safe beside Blender): `--author-offline` actually authors + validates the whole
  composed stage (physics, colliders, robot placement) with pure USD — the composition
  core is `scripts/twin/compose.py`, shared verbatim with the real run so only the
  boot / URDF→USD import / stepping / screenshot are new at GPU time. The GPU-session
  runbook is **RUNBOOK §8**. A real run boots Isaac Sim, gated by a GPU guard that
  hard-refuses beside the live Blender session. **The real boot is PREPARED but has NEVER
  been run — the owner deferred it ("not yet", 2026-09-19).** The whole run is one bounded
  command `scripts/run_robot_in_store_gpu.sh` (readiness gate + 20-min wall-clock cap + RAM
  watchdog + peak sampler; logs a base-height *settle* verdict via `twin/settle.py`). The
  GPU guard's heavy-consumer default (1500 MiB) is **unchanged**; the always-on `whispr`
  service (~2.2 GB VRAM) trips it, so accept it only as the **opt-in** `HEAVY_MB=2500`
  (never `--force`, never the hard Blender gate). Prep + PENDING boot sections:
  `data/twin-robot-in-store-v1/report.md`. Sharp edge the composer works around:
  **you cannot author a collider on a USD runtime prototype, nor reach it with an
  override from outside the store's layer stack** — so product colliders are authored on
  the `/Supermarket/prototypes` **class source** meshes with the store **sublayered**
  (read-only) into the output. Unit tested under `scripts/twin/tests/`; the compose +
  validator tests need `usd-core` (or `env_isaaclab`'s bundled `pxr`), the pure-logic
  tests need nothing.
- **Validate the USD offline (no GPU)**: `scripts/validate_usd.py` (RUNBOOK §6a) opens
  the export with a CPU-only `usd-core` venv and proves it is Isaac-loadable (up-axis,
  scale, default prim, every texture/reference resolves, composer prim paths). Run it
  after every `export_usd.py` — it's a **strict** check (exit 1 on any unresolved asset),
  but its verdict grades the failure **BLOCKING** (missing reference/payload/sublayer =
  missing geometry) vs **COSMETIC** (missing *texture* — untextured surface, geometry
  intact). Known standing defect it flags: the export authors 6 dangling floor/wall
  textures (MCP Poly Haven maps backed by `tmp*.jpg`) — COSMETIC. Fixed at source in
  `export_usd.py` (needs a re-export); on the current artifact clear it with
  `--repair-textures` (copies the 6 names from versioned `assets/textures/`). The
  pre-flight (§8) does that repair automatically, so the captain's one command is green
  without a re-export. Write-up: `data/twin-usd-validate-v1/report.md`; artifact decision +
  cost basis in README's decision log (2026-09-10). `export/` stays gitignored (repaired
  on demand, adds 0 git bytes) — do not commit it.
- **Validate the ROBOT URDF offline (no GPU)**: `scripts/validate_robot_urdf.py`
  (RUNBOOK §6c) reads `shelfbot_gripper_isaac.urdf` with pure `xml`+`numpy` (core in
  `scripts/twin/urdf_probe.py`) and proves it is Isaac-ready: parses to one kinematic
  tree, every `package://` mesh resolves, masses/inertias/joint-limits sane, and the
  home-pose FK envelope fits the aisle. `scripts/preflight.py` runs store + robot +
  composer as **one** command (exit 0 = safe to boot; the §8 pre-flight). **Package
  `shelfbot` = `~/workspaces/shelfbot_newgripper`** (declares that ROS name AND holds the
  gripper meshes) — NOT the older `~/workspaces/shelfbot`, which declares the same name
  but lacks the meshes; the wrapper builds a symlink shim to point there, since Isaac
  imports even when a mesh is missing. Write-up: `data/twin-robot-preflight-v1/report.md`.
  Standing upstream (URDF, another repo) WARN: placeholder inertias on the two gripper
  links — report, don't edit.
- **Path caveat**: `build_supermarket.py`, `export_usd.py`, `render_shots.py` derive
  `ROOT` from the script location (env `SUPERMARKET_ROOT` overrides; live-GUI `exec()`
  with no `__file__` falls back to `~/workspaces/blender-supermarket`), so they run in
  any worktree/clone. PIL scripts use script-relative paths. **`render_walkthrough.py`
  still hardcodes** `~/workspaces/blender-supermarket`.
- **Realism** of the generated look (floor reflections/wear, per-instance product &
  produce variation, LED bloom, AgX) lives in `build_supermarket.py`
  (`polish_floor`, `add_instance_variation`, `setup_compositor`, `setup_render`) — see
  README decision log 2026-09-14 and `data/twin-realism-v1/report.md`. Render evidence
  on CPU with `render_shots.py --cpu` when the GPU is held by the robot stack.
- **Mobile restock manipulation**: sensor/ROS contracts and supported handling units
  live in `docs/restock_manipulation.md`; CPU checks are in RUNBOOK §13.
  Runtime inputs must come from sensors; fixtures cannot certify inventory.
- **Empty-support commissioning**: a mission capture refuses until its station's empty
  baseline is commissioned (`binding.py` `'observed empty commissioning baseline is
  required'`). `scripts/twin/restock_camera/commissioning.py` runs `commission()` on an
  observed empty frame and persists the baseline with provenance; the mission loads it by
  station at boot via `build_sensing`'s `config['commissioning_dir']`, refusing a missing
  one (never fabricated). Choreography CLI + runbook: `scripts/run_restock_commissioning.py`
  and RUNBOOK §13a. The collect stage senses the pick carton with the FIXED `source` camera
  (not the arm eye-in-hand, which is masked-depth only): `restock_camera/source_target.py`
  `build_source_target_provider` → `source_target_provider(now)` gives the handling-unit pose
  in the station frame + baseline provenance (RUNBOOK §13b); the arm gets a reprojected target.
- **Mobile navigation**: `scripts/twin/navigation/` and
  [docs/restock_navigation.md](docs/restock_navigation.md) define sensor, footprint
  and goal contracts; RUNBOOK §12 has CPU checks and isolated Nav2 launch. Coupled
  clearance uses the independent base/cart swept envelope, so straight cart width alone does
  not prove aisle feasibility. Simulator/Nav2 execution still needs integration
  evidence; fixture tests must not be presented as a physics demonstration.
- Git delivery: the repository now has an `origin` remote; follow the current
  task delivery contract rather than the former local-only assumption.
- Mobile UR10/cart assets and control interfaces: [docs/mobile_robot_rollwagen.md](docs/mobile_robot_rollwagen.md), commands in RUNBOOK §9. The generated mobile URDF replaces the fixed pedestal; import with a free base. CPU FK/schema checks do not establish PhysX towing stability.
- **Service environment**: `scripts/store_environment.py` adds the connected backroom;
  RUNBOOK §10 documents scratch builds, CPU geometry checks and the semantic/static-map
  manifest. Source cases are static visual stock; scenario composition owns dynamic
  actors. `GRP_EnvironmentDecor` must remain nonphysical.
- Mobile restocking orchestration and transactional inventory live in
  `scripts/twin/restock/`; CPU checks and adapter contract are in RUNBOOK §11.
  `data/mobile-restock-v2/report.md` distinguishes fixture tests from the still
  unverified physical cycle and records the GPU driver diagnosis.
- **Bounded native restock diagnostics** (guarded GPU, no full mission) are driven by
  `scripts/run_restock_render_smoke.py` with one of `--assembly-only` (stationary
  physics hold → `restock/assembly_check.py`), `--coupling-only` (dock+latch+hold →
  `coupling_check.py`), `--towing-only` (forward crawl + VFR H.264 → `towing_check.py`),
  or a `--runtime-config … --view source|shelf` optical bench (`view_scene.py`). They
  run under the same guards as the mission (`--isaac-python` env_isaaclab, `--nvidia-runtime`
  the task-local matching userspace at `~/datasets/sim-gpu-userspace-recovery-s1/root`,
  `/tmp/aile-gpu.lock`, `--heavy-threshold-mb 2500` for the always-on whispr). ffmpeg is
  the `imageio_ffmpeg` binary under env_isaaclab. Stationarity is gated on **measured net
  pose displacement** (`restock/settle_metrics.py`), never the TGS-reported instantaneous
  velocity — that carries a documented solver bias (~14 mm/s while a body is provably
  frozen). Cart rest stability needs PhysX `enable_stablization(True)` applied through the
  World `PhysicsContext` before `reset()` (Isaac re-applies its `False` default at World
  init, clobbering the USD attr) **plus** rigid-body damping for caster/rolling resistance
  — `physxJoint:jointFriction` is inert on the maximal-coordinate cart casters
  (`excludeFromArticulation`); see `physics_setup.configure_scene_physics` /
  `configure_cart_bearings`. Evidence + the exact per-run commands are in
  `data/mobile-restock-v2/report.md`.
- Git hosting: inspect the configured remote and current task delivery contract
  before publishing; older local-only task notes predate the configured remote.
- **Store-vision scenario generator** (`scripts/scenarios/`): renders labelled
  synthetic sequences from the store (shelf depletion, floor spills, people movement,
  misplaced products, multi-view RGB-D) for the store-vision suite. Runs **headless in
  its own Blender process** (never the live GUI / `scenes/`); ground truth is *computed*
  from the planogram, not inferred, so pixels and labels agree by construction. Entry +
  one-command regenerate: **RUNBOOK §15**; design + module map: `scripts/scenarios/README.md`;
  dataset format: `docs/scenario-format.md`. Offline tests (no Blender) validate a
  committed smoke fixture — run with `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=` to
  dodge the box's ROS pytest plugins. Datasets write **outside git** (`~/datasets/store-scenarios/`).
- GitHub remote is `ailerobotics/blender-supermarket` (origin); changes land via PR
  (see the merged-PR history). Use **gh-axi** for GitHub operations; never push to or
  merge into the default branch yourself.

## Maintaining this file

Keep this file for knowledge useful to almost every future agent session in this project.
Do not repeat what the codebase already shows; point to the authoritative file or command instead.
Prefer rewriting or pruning existing entries over appending new ones.
When updating this file, preserve this bar for all agents and keep entries concise.
