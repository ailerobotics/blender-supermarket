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
  **Sharp edge**: a `/manipulation/status` carrying a gripper control MUST have a non-None
  `valid_until_steady` or the host gate (`restock_manipulation/isaac_ros_graph.py`) refuses it
  (`'missing acquisition deadline for gripper command'`) and the grasp aborts at CONTACT. The
  camera collect threads the source deadline; the GT-collect / transfer / live-pose capture paths
  have no camera framing the pick, so `CollectTargetBridge._capture_ground_truth_collect` /
  `_capture_transfer` stamp `now + 300` themselves (fm488). Pinned by
  `tests/test_collect_target_bridge.py`. Cold GT-collect proof + switch keys:
  `data/sim-video-collect-arm-v1/report.md`.
  **Place-on-cart GPU/heartbeat bound**: an unplannable cart-place goal (self-colliding IK branch,
  `'Invalid c-space position: self collision detected'`) used to churn the full `REPLAN_MAX`
  same-goal replans and OOM the 12 GB card (`cumotion::FatalException`, `cuda_array.cu`) / starve the
  shared `get_esdf_and_gradient` service into a 30 s station-heartbeat gap. `ros_adapter.py` now caps
  total cuMotion plan dispatches per move (`PLAN_TOTAL_ATTEMPTS_CAP`/`RESTOCK_PLAN_ATTEMPTS_CAP`=10)
  and shortens the joint-space same-goal replan budget (`REPLAN_MAX_JOINT_SPACE`/
  `RESTOCK_REPLAN_MAX_JOINT_SPACE`=3) so a self-colliding target advances to the next IK branch fast;
  `restock_host/incorporation_ack.py` publishes the station liveness ack BEFORE its ESDF maintenance
  so a plan storm can't starve the heartbeat (readiness still gated on a real integration). Report +
  the collect+couple validation env: `data/sim-robust-collect-place-v1/report.md`.
- **Tray/slot fit preflight** (opt-in, default-off): `restock/fit_preflight.py` refuses a
  handling unit that will not fit the destination slot BEFORE reserving stock or booting
  Isaac, instead of failing opaquely at place. `fit = ok | wont_fit | unknown` from a
  *measured* free-space record (mission depth capture in arm-base metres, or a store-vision
  `twin3d` export mapped via an EXPLICIT `axis_order` — never a silent identity); `unknown`
  (no/low-confidence/unmeasured record) falls back with a confidence flag, never silently.
  The axiswise size rule is the ONE shared predicate `restock_manipulation/fit.py::axis_fits`
  the place-phase gate (`controller.py`) also calls, so a preflight verdict matches the gate.
  Enable with `run_mobile_restock.py --fit-preflight` + a `config['fit_preflight']` record
  block; absent ⇒ byte-identical run. CPU tests: `tests/test_restock_fit_preflight.py`.
- **Empty-support commissioning**: a mission capture refuses until its station's empty
  baseline is commissioned (`binding.py` `'observed empty commissioning baseline is
  required'`). `scripts/twin/restock_camera/commissioning.py` runs `commission()` on an
  observed empty frame and persists the baseline with provenance; the mission loads it by
  station at boot via `build_sensing`'s `config['commissioning_dir']`, refusing a missing
  one (never fabricated). Choreography CLI + runbook: `scripts/run_restock_commissioning.py`
  and RUNBOOK §13a. One guarded boot commissions ALL THREE stations (source + cart + shelf)
  into one shared `commissioning_dir`: `scripts/run_restock_commissioning_gpu.py` with repeatable
  `--station` loops the renderer (one station per Isaac boot) and unions the baselines. Sharp edge:
  `camera_config.json` is a GENERATED artifact of `restock_camera.config` shared by the commissioning
  bench AND the mission — regenerate it from the current design with
  `scripts/write_restock_camera_config.py` before commissioning, or a stale file feeds OLD geometry
  (e.g. the pre-fix `.62×.42` cart support overshoots the rails → `_flat_patch` refuses the deck).
  For a warm-session re-run (`run_mobile_restock.py --warm-attempts`, PR 111), the `config['warm_reset']`
  provider refreshes an expired/unbound baseline IN-SESSION via
  `commissioning.recommission_in_session(...)` (re-commission all required stations through the
  unchanged gates, raising to refuse — never bypassing the lifetime guard; RUNBOOK §13a).
  The collect stage senses the pick carton with the FIXED `source` camera
  (not the arm eye-in-hand, which is masked-depth only): `restock_camera/source_target.py`
  `build_source_target_provider` → `source_target_provider(now)` gives the handling-unit pose
  in the station frame + baseline provenance (RUNBOOK §13b); the arm gets a reprojected target.
  The transfer stage mirrors this with the FIXED `shelf` camera + `G3_back_s2` baseline:
  `restock_camera/shelf_target.py` exposes the empty-slot placement target (`shelf_target_provider`)
  and the post-placement verification (`shelf_verification_provider`) when
  `config['transfer_shelf_location']` is set — contract `data/sim-restock-camera-v2/shelf-contract.md`,
  RUNBOOK §13c.
- **nvblox obstacle map while carrying**: the eye-in-hand `wrist` camera is nvblox's only depth
  feed, so a held carton freezes the ESDF at the place re-park (finding B'). The body-mounted
  `mast` depth camera (`restock_camera/mast_depth.py`, RealSense-class, fixed to `base_link`) is
  the second feed (`camera_1`), carton-masked like the wrist. Off by default; host switch
  `mast_depth: {enabled: true}` (`restock_host/config.py`/`launch.py` set `num_cameras: 2` + a
  second segmenter), and `camera_config(..., mast_depth=True)` adds the prim. Design + the
  named cold-collect cause (sequencing, not a camera): docs/restock_camera.md, proof
  `scripts/plan_restock_more_cameras.py`.
- **Mobile navigation**: `scripts/twin/navigation/` and
  [docs/restock_navigation.md](docs/restock_navigation.md) define sensor, footprint
  and goal contracts; RUNBOOK §12 has CPU checks and isolated Nav2 launch. Coupled
  clearance uses the independent base/cart swept envelope, so straight cart width alone does
  not prove aisle feasibility. Simulator/Nav2 execution still needs integration
  evidence; fixture tests must not be presented as a physics demonstration.
  **Couple dock approach** (`run_mobile_restock.py --start-at couple`): the backstock
  `/RestockScenario/Support` clears the `y=16.0` couple corridor by only ~0.11 m, and its live
  lidar returns sit a few cm proud of the true AABB — so from the shortcut start pose (directly
  south of the support) the PLANNER refuses the straight leg (`blocked_no_route`, speed-independent;
  the PR-142 `navigation.docking` cap only shrinks the braking disk, not the footprint). Fix:
  `config['couple_static_drive']` routes the couple legs against the static map (the fm396
  trusted-static pattern; default off, sim-only). **Coupled outbound** resolves cheaply via the
  `sales_backroom_portal` corridor route; a loaded host can starve the full-A\* fallback into
  `planning_timeout` — `navigation.planner_timeout_s` is the wall-budget backstop (default 30 s).
  Offline proof for both (CPU): `scripts/check_couple_nav.py`; design + numbers in
  `data/sim-restock-couple-nav-v1/report.md` (relaunch section) and its `handoff.md`.
  **Direction-aware braking zone** (`navigation.directional_zone`, default ON): the base
  `obstacle_stop` gate checks the footprint all around (never shrunk) + the braking extension
  ONLY along the forward heading (a capsule), so rear static geometry a no-reverse base can't
  reach no longer false-stops; `false` restores the legacy omnidirectional disk. Coupled gate
  was already forward-sweep direction-aware. **Pre-planned tow lanes** (`navigation.tow_lanes`,
  opt-in): authored centrelines in `navigation/lanes.py` expanded at build, followed +
  swept-validated closed-loop for every COUPLED drive (NO free search → no `planning_timeout`,
  and the east-facing `home` reorientation-loop return is replaced by the west-facing
  `mission_poses.cart_parking` `[-9,16,π]`); uncoupled drives unchanged. Example fragment
  `docs/examples/tow-lanes-mission-fragment.json`; offline proof `scripts/check_tow_lanes.py`;
  design in `data/sim-restock-nav-lanes-v1/report.md`. **Couple speed / outbound stall** (fm482, both
  opt-in): `navigation.arrival_xy_hysteresis_m` stops the couple-standoff arrival dither (~700 s wall per
  warm run) and `navigation.coupled_creep_escape` stops the outbound `measured_turn_radius` noise
  livelock; offline proof on the recorded runs `scripts/check_couple_speed.py`. **Sharp edge:
  `Scan.sensor_to_base` is (x, y, YAW)** but `isaac_bundle` passes the lidar URDF origin (x, y, z),
  rotating every live return 0.15 rad about the sensor (a likely root of "lidar-proud" phantoms);
  `lidar_sensor_yaw_fix: true` uses yaw 0 — default off, re-check couple/outbound before enabling
  (probe: `twin/restock/tail_nav.py` `lidar_wall_probe`).
- **Mission tail** (transfer → verify → uncouple return → end): the uncouple happens in RETURN at
  `cart_parking`, AFTER transfer + verify (tow → brake/pin → uncouple → back away → home uncoupled).
  CPU dry-run through the real BT/`Mission`/`RuntimeAdapter`: `scripts/check_mission_tail.py`
  (RUNBOOK §21, guarded by `test_restock_mission_tail.py`); the labelled ground-truth verify switch
  and the reachable return poses are in `docs/examples/mission-tail-fragment.json`. **The coupled RETURN
  tow is STUBBED kinematically by that dry-run** — it does not exercise the follower's cart-yaw tracking,
  which is where a real stall lives: the authored return lane has `start_cart_yaw` 0 but the towed cart
  lags at articulation ~0.2 after transfer, so without `navigation.tow_lane_cart_yaw_rebase` the follower
  drops its validated route every tick (`stop_replan_tracking_error`, `cart_err` > `tow_lane_cart_yaw_tol`)
  and never moves. Guards: `scripts/twin/tests/test_return_cartyaw_regression.py` (drives the REAL coupled
  follower via `nav_factory` from the recorded end-of-verify pose, fixture `tests/fixtures/return_cartyaw_nav.json`;
  a dropped rebase key at the default tol re-stalls and fails) and `check_mission_tail --return-tracking`
  (same drive on the real grid instead of the stub) — both via `tail_nav.py` `simulate_coupled_return`.
  **Sharp edge (fresh nav leg must reset the stuck timer):** the fm425 no-base-progress WALL fail-fast
  (`RuntimeAdapter._nav_stuck_check`) only advances inside the navigate phase, so wall-time spent
  legitimately stationary in a preceding NON-navigate phase (park + transfer + verify before the
  uncouple-return tow) would be charged against the return leg's very first tick and fail it before the
  controller plans — each StageRetry re-begin then re-failing instantly (warm7: `planning_stop`
  elapsed=0.02, "no base progress for 207s wall"). Every adapter nav dispatch goes through
  `RuntimeAdapter._start_nav_leg`, which clears the baseline so each leg's budget starts at its own
  start (outbound never hit it — its leg follows the couple drive). Pinned by
  `test_restock_runtime.py` (`*_return_rebegin`, `*_does_not_inherit_stuck_budget_from_transfer`);
  the lane itself plans + persists a route from the warm7 end pose (`scripts/check_tow_lanes.py`).
- **Transfer workaround modes** (env `RESTOCK_TRANSFER_MODE`, read by Isaac AND the host, never a config
  key; default off): `live_pose` = real-arm pick from the live carton pose + near-face yaw, no transfer
  reposition (`restock_manipulation/transfer_mode.py`); `scripted` = the labelled safety net
  (`restock/scripted_transfer.py`): no warm-up/host job, kinematic carton onto `transfer_ground_truth.target`
  (needs `enabled`; evidence `source='ground_truth'`), arm `RESTOCK_SCRIPTED_TRANSFER_ARM=still` (default) |
  `executor`. **Sharp edge: never drive the arm from Isaac (`ArmCommandGate.apply`) beside a live
  `trajectory_host`** — its idle hold trips the gate envelope + its tracking e-stop; send FollowJointTrajectory
  goals instead. From the bay park the side grip extracts only ~0.35 of 0.46 m out of the rollwagen mouth.
  Write-up: `~/firstmate/data/sim-video-transfer-fallback-v1/report.md`. **Sharp edge:** in `live_pose`
  the `TRANSFER_CAPTURE cart=(x,y,z)` console field is the LIVE carton top in `ur_base_link`, not the cart
  frame (warm2 was misread as "carton fell to the lower deck" while the weld still held it). Carry switch
  `RESTOCK_TRANSFER_CARRY=wrist_guard[,weld]` (env, both processes, default off; RUNBOOK §13e) keeps the
  transfer place-move KDL IK out of the `ur_wrist_2`-vs-forearm stop (`weld` labels the weld attach,
  `transfer_attach: weld`); replay a run with `scripts/check_transfer_carry_replay.py`. **`scripted` with
  `arm=executor` has its OWN DLS IK** (`scripted_transfer.ArmIK`, NOT cuMotion/KDL), so the live-pose
  `wrist_guard` does not apply to it — it carries a `plan_transfer(wrist_guard=True)` default that forces
  the PICK half onto the same clear branch (reusing `transfer_carry.in_clear_region`/`wrist_flip_seed`),
  else the pick seeds the gripper-vs-forearm self-collision and the host faults on the >1.3 rad tracking
  error (warm5). A mid-transfer executor refusal FAILS FAST with a named reason + a keep-alive hold (never
  the silent `still` degrade that starves the arm-command heartbeat). Offline proof (incl. the sphere model
  + warm5 host-validation replay): `data/sim-video-executor-arm-v1/report.md`,
  `tests/test_restock_scripted_transfer.py`. Shelf-slot place (sales-floor endcap G3_back_s2, s3 board
  0.25 m above): `RESTOCK_SCRIPTED_TRANSFER_PLACE=front` (env, scripted only, default off) slides the carton
  in under the board instead of descending through it — task-space plan, still arm only; staging keys +
  CPU proof `scripts/check_front_transfer_staging.py`, `data/sim-video-story-sources-v1/report.md`.
- **Rollwagen cage clearance** (sim-only workaround, RUNTIME config `cage_collision: "filtered"`, default
  off; `restock/cage_collision.py`): a FilteredPairsAPI on the carton against the 50 cage colliders (deck,
  upper shelf and drawbar still collide; rays and cuMotion untouched), labelled in mission.json and the
  caption. Why it is not a path: the 0.6 m carton side threads the 0.615 m front-post mouth with 7.5 mm
  per side, and leaving the cage takes 0.4125 m out through the mouth (rising 0.72 m over the top is out
  of reach), but a fixed-orientation straight line out stays IK-feasible for only ~0.41 m (collect, from
  `place_approach`) / ~0.36 m (transfer, coupled park) on every UR10 branch. **Sharp edge:
  `begin(straight_line=True)` is ONE 2-point joint-space smoothstep, not a Cartesian line.** Over a 0.4 m
  insert it bows 45–64 mm and strikes a post; it takes 5–9 segments to stay clear.
  CPU proof: `scripts/check_cage_clearance.py`.
- **Moving obstacles** (opt-in, default-off): `navigation/dynamic.py` tracks
  per-frame `Detection(position,radius,source)` into velocity-predicted occupancy
  points unioned (add-only) into `Observation.obstacles`, so the existing gates
  avoid them with the towed-cart envelope. Enable via `navigation.dynamic_obstacles`
  (sensor nav_mode only) + `SensorNavigation.step(detections=)`; absent ⇒
  byte-identical run. Nav2's existing obstacle layer receives transient nav-scan
  returns; installed Isaac ROS 4.4 supports dynamic/people nvblox maps, but this
  repo currently launches only the static evidence mapper (do not claim its dynamic
  modes are wired). `Detection` is the contract shared with `sim-restock-humans-v1`
  (detection topic) and `sim-restock-bt-v1` (event tree). Scenario + self-asserting
  CPU evidence: `navigation/dynamic_scenario.py`, `scripts/run_dynobs_evidence.py`
  (RUNBOOK §12); docs in [restock_navigation.md](docs/restock_navigation.md)
  "Moving obstacles". **Mission feed** (opt-in): the layer is inert until fed per
  frame — `navigation.dynamic_obstacle_source` names a `module:function` the bundle
  builds into a `source(now)->[Detection]` passed at the `NavigationBridge` sensor
  seam (`restock/runtime.py`) into `SensorNavigation.step(detections=)`; absent ⇒
  `detections=None` ⇒ byte-identical. Default builder
  `navigation/mission_source.py:build_dynamic_obstacle_source` reuses the
  `sim-restock-humans-v1` scripted person (`humans` block) as BOTH the rendered actor
  and the detection (`source='scenario_scripted_actor'`, positions from pure
  `PeopleStage.world_positions`); the humans perception path plugs into the same seam
  via `mission_source.humans_detection_source`. Example config fragment
  `docs/examples/dynobs-mission-fragment.json`. CPU-proven only
  (`test_navigation_mission_source.py`); the in-sim GPU evidence run rides the
  integration mission boot (owns GPU priority + commissioning + host-stack) and is
  PENDING (`start_at` supports `couple`/`transfer`; a bounded nav-only run still needs it
  extended to `outbound` — integration-owned).
- **Simulated people + human safety** (`scripts/twin/humans/`, opt-in, default
  OFF): an absent `humans` runtime-config block imports nothing and leaves the
  mission byte-identical. Three independent pieces — scripted shopper/staff actors
  authored into the Isaac stage (`isaac_people`/`people`/`paths`, `teleport` backend
  by default, `anim_people` when the extension is present); the fleet human-safety
  decision core vendored **byte-for-byte** from `ur10-setup/cams/human_safety`
  (`perception/`, parity-tested against that checkout), fed either by the real CUDA
  detector on the mission camera topics (`params/human_safety.restock.yaml` →
  `detector_bridge.py`) or, behind an EXPLICIT `detection.source: ground_truth`
  fallback, by projected known poses; and a configurable slow/stop-base +
  pause/resume-arm mechanism (`reaction.py`, fail-safe to STOP on stale state).
  Human detections use the dynobs `Detection(position, radius, source)` contract
  and are consumable by `navigation/dynamic.py` via `detection.to_nav_detections`;
  the mission detection feed + visible people authoring already land through the
  dynobs source (`navigation/mission_source.py`, PR #130). The reaction's base-scale
  + reversible arm-pause into the mission is the mechanism `mission_wiring.py`
  (`MissionReaction.step(now)`, `ReversibleArmPause`); its mission touch points
  (isaac_bundle scene author, isaac_port `after_step`, runtime base-velocity scale,
  a reversible `ArmCommandGate.set_paused`) are NAMED there for the owning lanes to
  wire behind `humans.enabled` at an integration boot, not edited here.
  Other touch points are documented seams only (`integration.py`, `README.md`),
  not wired into the lane-owned files. CPU tests + a ground-truth demo
  (`scripts/run_humans_safety_demo.py`). Guarded-GPU evidence is the standalone bench
  `gpu_bench.py` via `run_restock_render_smoke.py --humans-only` (people cross the
  robot on the real candidate camera, GT detection+reaction → CLEAR→SLOW→STOP→resume);
  one queued command `scripts/run_humans_gpu_bench.sh` (flock lock, ≤15 min; the GPU
  guard pins ROS_DOMAIN_ID=88 and the bench runs no ROS node; RUNBOOK render-smoke
  section). It is ground-truth, NOT a neural-detector run — that,
  plus mission wiring behind `humans.enabled`, is the follow-up. Report + open owner
  decisions: `data/sim-restock-humans-v1/report.md`.
- **Route alerts** (opt-in, default-off): the mission robot as a mobile store sensor.
  `scripts/twin/route_alerts/` bridges the drive camera to the READ-ONLY `store-vision`
  spill/obstruction detectors (classical CPU by default; open-vocab is the GPU path),
  publishes `robot.route_alert.*` events + de-duplicated staff tasks, and turns
  obstruction footprints into `navigation.dynamic.Detection` fed to nav ONLY through the
  dynobs costmap layer (no parallel layer). People-masked (GDPR); `--route-alerts` absent
  ⇒ byte-identical (`run_mobile_restock.py` → `isaac_bundle.create` → `RuntimeAdapter`
  `route_alerts` hook, all guarded/self-disabling). store-vision is imported lazily and
  lives OUTSIDE this repo (its `.venv` python3.11 runs the integration tests). CPU tests
  `scripts/twin/route_alerts/tests/`, evidence `scripts/run_route_alerts_evidence.py`,
  RUNBOOK §19, docs [docs/route_alerts.md](docs/route_alerts.md), write-up
  `data/sim-route-alerts-v1/report.md`. strays/coldchain need a products recognizer
  (GPU, off by default); store-vision has no per-frame chiller-door detector (proposal in
  the report). Live in-Isaac detection is proven through the IDENTICAL runner by
  `scripts/run_route_alerts_smoke.py` over real render-smoke RTX frames (a seeded
  cube-on-floor); the "on the route while driving" run needs a mobile floor-facing RGB-D
  camera the bundle lacks today (follow-up + touch points in the report). The dynobs
  mission feed (`navigation.dynamic_obstacle_source`) is the consumer seam a route-alerts
  `source(now)->latest_detections()` builder registers with — no parallel layer.
- Git delivery: the repository now has an `origin` remote; follow the current
  task delivery contract rather than the former local-only assumption.
- Mobile UR10/cart assets and control interfaces: [docs/mobile_robot_rollwagen.md](docs/mobile_robot_rollwagen.md), commands in RUNBOOK §9. The generated mobile URDF replaces the fixed pedestal; import with a free base. CPU FK/schema checks do not establish PhysX towing stability.
- **Mission scene visuals** (`scripts/twin/restock/visuals.py`): a config-switched,
  VISUAL-ONLY material pass run inside `restock/scene.build` (so only the mission path,
  not the commissioning `view_scene`). Binds a UR10 palette (silver link tubes, UR-blue
  joint caps, dark housings) onto the imported arm link prims, a neutral grey to the mobile
  base, galvanised steel + blue plastic to the rollwagen, and re-tones the backroom concrete/
  walls via read-only overrides on the sublayered store. Top-level runtime key `visuals`,
  **DEFAULT ON** (absent ⇒ enabled; restore the plain look with `false`/`{"enabled":false}`/
  `"off"`); example `docs/examples/visuals-mission-fragment.json`. It never touches
  geometry/colliders/physics/joints/cameras/lights or the carton QR + decode markers, so the
  sensing/decode suites are unchanged. CPU proof: `tests/test_restock_visuals.py`.
- **Service environment**: `scripts/store_environment.py` adds the connected backroom;
  RUNBOOK §10 documents scratch builds, CPU geometry checks and the semantic/static-map
  manifest. Source cases are static visual stock; scenario composition owns dynamic
  actors. `GRP_EnvironmentDecor` must remain nonphysical.
- Mobile restocking orchestration and transactional inventory live in
  `scripts/twin/restock/`; CPU checks and adapter contract are in RUNBOOK §11.
  `data/mobile-restock-v2/report.md` distinguishes fixture tests from the still
  unverified physical cycle and records the GPU driver diagnosis.
- **Smart post-restock verification** (CPU, read-only, opt-in): `restock/smart_verify.py`
  + CLI `scripts/run_smart_verify.py` turn a finished mission into an ADDITIONAL
  planogram/realogram compliance verdict + residual reset jobs (sim-smart-job-feed-v1
  `--job-from` shape), NEVER changing the mission's stock-commit gate. Evidence basis is
  labelled honestly (`mission_verify_sensor`/`bookkeeping`/`full_shelf_sensor`). Slot-id
  scheme mirrors `store_environment.layout` (a test asserts no drift). `restock/smart_verify_storevision.py`
  is a lazy, best-effort cross-check through store-vision's read-only `planoreset.verify`
  that degrades when store-vision/pydantic is absent. Also runs at mission end behind
  `run_mobile_restock.py --smart-verify` (byte-identical when absent). RUNBOOK §11b;
  write-up `data/sim-smart-verify-v1/report.md`.
- **Behaviour-tree mission DRIVER**: `scripts/twin/restock/bt/` drives the SAME `mission.Mission`
  (via the `begin_stage`/`step_stage`/`finalize` seams, so inventory + evidence stay byte-identical)
  with reactive event handling and a live phone view (port 8795). Selected by runtime-config
  `"bt": true`; `"bt": false`/absent keeps the sequential `Mission.tick` state machine as the
  unchanged fallback. **Each stage is a subtree of explicit position-first leaves** (navigate →
  perceive → manipulate; plans in `bt/stage_phases.py` `PLANS`): one shared `StageRun` steps once per
  tick and the `PhaseGate` leaves are faithful views of that one stepping gated on the adapter's
  observed `phase`, so the decomposed drive is byte-identical to the state machine; `StageRetry` holds
  bounded retry/recovery as one node; a non-phased fixture collapses a stage to one leaf. **A new step
  is a new `PLANS` entry (a new leaf).** Event contract (ROS topic `/restock_bt/events` + blackboard +
  scenario `events` list), tree shape, view paths, the uncouple-on-return plan and commands:
  [docs/restock_bt.md](docs/restock_bt.md); example runtime-config fragment
  `docs/examples/bt-driver-mission-fragment.json`. py_trees is task-venv-only (`.venv-bt`, gitignored);
  the pure `bt.events`/`bt.stage_phases` modules import with no py_trees/ROS so `scenario.py` validates
  scenario events. Tests: `test_restock_bt.py` + `test_restock_bt_phases.py` (phased decomposition +
  parity; tree tier needs py_trees). Design + proof: `data/sim-restock-bt-driver-v1/report.md`
  (superseding the original `data/sim-restock-bt-v1/report.md`).
- **Dev bench start/stop** (opt-in, default-off): `start_at` (`couple`|`transfer`) begins the mission
  at a stage with the skipped prefix synthesized (`restock/shortcut.py`, recorded as `*_shortcut`);
  `start_staging: recorded_bay_park` spawns the coupled train + upper-shelf carton at the recorded bay
  park->transfer handover; `stop_after` ends the run after a stage (`mission.bounded_stages`). Same keys
  as `run_mobile_restock.py --start-at/--start-staging/--stop-after`. Such runs are labelled in
  mission.json and by a burned-in walkthrough caption. RUNBOOK §11-bench; proof
  `scripts/check_transfer_bench_staging.py`. **Sharp edge: two sim clocks.** The shortcut's settle
  `world.step`s run before the first mission tick, so Isaac `/clock` (station acks, `collision_time`,
  the host node clock) leads the mission packet clock by a CONSTANT offset: 0.05 s cold, 0.55 s
  `couple`, 1.55 s `transfer`. A host check comparing `/clock` with a packet stamp sees that offset as
  age (the fm476 "slow render 0.55 s" and both bench1 transfer failures). Staged starts fix it via
  `restock/staged_start.py` (`Sensors.clock_offset`, default on; plus the first-manipulation settle
  hold `staged_start_settle_s`). Any new cross-clock host check must subtract `clock_offset`.
- **Uncouple-at-parking-spot on return** (opt-in, default-off): `return` tows the cart home coupled
  unless `config['uncouple_on_return']` + a `cart_parking` mission pose are set, when it tows the cart
  to the parking spot, brakes it, releases the tow coupling via `MobileController.uncouple(obs)` (new,
  beside `release_cart_brake`: pin cart then detach hitch; the `_cart_hold` latch keeps the cart braked
  as the robot drives home empty), optionally backs away (`cart_parking_backaway`) and drives home
  uncoupled. The parking poses + lane are owned by `sim-restock-nav-lanes-v1`. Absent ⇒ byte-identical
  tow-home. Runtime legs in `twin/restock/runtime.py`; CPU tests in `test_restock_runtime.py` +
  `test_mobile_assets.py`.
- **Demand-driven mission job** (opt-in, default-off): `--job-from <export.json>`
  on `run_mobile_restock.py` substitutes an approved store-vision restock plan
  (schema `storevision.restock.job_plan/v1`) for the scenario's hardcoded job.
  `twin/restock/job_from_plan.py` maps the plan to the mission `Job` and validates
  SKU + destination slot against `scripts/planogram.json` (via `scenarios.catalog_geom`,
  normalising `G3_R_b0_s1`→`G3R_b0_s1`), refusing unknown/mismatched slot or SKU.
  Absent ⇒ byte-identical run. Producer (store-vision read-only, its own venv):
  `scripts/integrations/export_restock_job.py`; example `scripts/restock_job_plan.example.json`;
  `--dry-run-job` prints the resolved job with no GPU. RUNBOOK §11a;
  write-up `data/sim-smart-job-feed-v1/report.md`.
- **Surface-gripper seal reader** (`restock_manipulation/cad_tool.py` `NativePanelContact`):
  `read()` is a PURE OBSERVER — never command the gripper from it (an `open_gripper()` on a
  not-yet-`Closed` grab aborted the seal on every poll of the slow contact descent), and key the
  hold on the plugin's `grippedObjects` list (exactly one body), NOT `status==Closed` — the status
  flickers `Closing↔Closed` tick to tick while a grip provably holds. `command()` owns open/close.
  The built-in `SEAL_DIAG`/`probe()` logs status, gripped count, per-seal pose + raycast gap every
  grasp tick (`RESTOCK_SEAL_DIAG=0` to silence). Evidence + the contact-gap GPU probe:
  `data/sim-restock-stations-v1/report.md`.
- **Dev-loop tooling for the Isaac mobile-restock lane** lives in `scripts/twin/tools/`
  (CPU, no GPU): `profile_mission_run.py` breaks a recorded run's wall clock into
  boot/commissioning/collect-observe/approach-reach/per-frame-render/ROS-idle/shutdown from
  the run's own artifacts (console log + `nav-trace.jsonl` + `mission.json` +
  commissioning `run.json`); `replay_reach.py` replays a run's recorded frame
  cadence+identity (`mission-frames/frames.json`) through the **production** capture+hold
  pipeline (`isaac_sensors`/`sensed`/`binding`/`qr`, imported unchanged) on a virtual clock
  so a camera-side change is provable in ~1-2 s instead of a 12-20 min GPU run. Caveat: a run
  persists only the walkthrough RGB (`restock/recording.py`), not the sensing camera's
  depth/context, so the replay drives the real gates with the recorded timing+identity against
  an analytic bench frame (labelled, not sensor evidence). Write-up + per-phase numbers:
  `data/sim-restock-devloop-v1/report.md`.
- **Sim speed profile** (opt-in, default off): runtime-config `"sim_speed": {"profile": "max"}`
  (`restock/sim_speed.py`) pauses the 2560×1920 station decode cameras wherever nothing reads them
  (nav stages; ground-truth transfer/verify), drops wall pacing outside collect/transfer/verify (a 1×
  cap stays there because host sim-time windows would tighten), labels `sim_speed_mode`, and prints
  per-stage `TICK_PROFILE` lines. **Sharp edge: never pause the arm/mast depth cameras.** The
  masked-depth bridge stamps frames with the publishing tick's sim time, so a pre-pause buffer would
  reach nvblox as current. The unrendered 40–72 ms tick caps the sim near 0.3×. Replay estimator
  `scripts/twin/tools/estimate_sim_speed.py`; guard audit `~/firstmate/data/sim-video-speedup-v1/report.md`.
- **Arm-carry bench** (`scripts/twin/tools/arm_carry_bench.py` + guarded GPU runner
  `scripts/run_arm_carry_bench.py`): boots Isaac ONCE, welds the carton at the tool via the
  mission's own `SIM_ATTACH_FALLBACK` weld, brakes the base in the airborne-carry yaw-lock
  state, then runs many trials of the joint-space carry fold (tuck → pan → …) in seconds each
  (physics-only stepping, no walkthrough). It drives the REAL `ArmCommandGate` and the REAL
  `trajectory_host` Plan + tracking/settle gates (imported, not copied), so drive gains,
  carry velocity/accel scaling and the sim-only tolerances are exercised identically; the
  only honest differences are the planning path (trajectories generated analytically, not by
  MoveIt/cuMotion) and the in-process executor loop (no ROS relay). Per-trial metrics include
  carton–obstacle **clearance** (AABB) and **PhysX contact** (which prim the carton hits, at
  what fold fraction / wrist_1) — built to diagnose the carry-fold stall (a rigid contact with
  the unmapped cart cage, gain-independent) and to sweep carry POSES for a cage/rack-clearing
  shape. CPU-tested under `scripts/twin/tests/test_arm_carry_bench.py` (no Isaac/ROS/GPU).
  Write-up: `data/sim-restock-armbench-v1/report.md`.
- **Arm MPC follow-and-avoid bench** (`scripts/twin/tools/arm_mpc_bench.py` pure core +
  `arm_mpc_run.py` container entry + guarded runner `scripts/run_arm_mpc_bench.py`):
  **EVALUATION-ONLY, simulation-only** — cuRobo V2 (Apache-2.0) reactive MPC for the mission
  arm, benched against the representative store restock-aisle colliders + a moving-person proxy
  (solve rate, clearance, gate behaviour), mirroring the armbench pattern (seconds per trial).
  MPC avoidance is a **SOFT cost, NOT a safety function**; the single motion authority stays the
  merged human-safety gate + measured joint-window guard, **reused verbatim** from the fleet
  `plan-mpc-curobo-v1` prototype's `ros_bridge` (`~/ur10/mpc_curobo`, mounted `/mpc`; override
  `MPC_CUROBO_DIR`) and re-keyed to the mission `ur_` joints. Reuses the mission arm's own
  `shelfbot_curobo.yml` and the existing `fm-plan-mpc-curobo-v1` container (NO Isaac boot, NO
  arm-stack rebuild; ~0.22 GB VRAM). ROS-free; lane `ROS_DOMAIN_ID=107` if ever wired. Opt-in
  everywhere: `compose_robot_in_store.py --arm-mpc` only annotates the dry-run plan — the
  composed stage and the mission cuMotion arm path are byte-identical when absent (asserted in
  `test_cli.py`). The guarded runner takes `/tmp/aile-gpu.lock` NON-BLOCKING (exits 3 when the
  integration lane holds it), hard-refuses beside live Blender, VRAM-guards, caps ≤15 min, never
  `--force`. CPU tests: `scripts/twin/tests/test_arm_mpc_bench.py`. Docs: `docs/arm_mpc_avoid.md`,
  RUNBOOK §18; write-up `data/sim-arm-mpc-avoid-v1/report.md`.
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
- **Fleet dashboard** (`scripts/twin/dashboard/`, launcher `scripts/run_fleet_dashboard.py`,
  RUNBOOK §17): phone-first, read-only CPU panel merging the live Isaac mission (pulse from
  the firstmate watch page `status.json`, pipeline from each run's `mission.json`, media
  linked back) with a store-vision store-state JSON overlay + an opt-in app trigger/monitor
  queue. Standalone — never imported or launched by the mission, so a run is byte-identical
  whether or not it is up; everything opt-in and default off. Pure merge in `aggregate.py`
  (unit-tested, no network/GPU/sim); binds the tailnet IP on a free port (default `8794`).
  Contract + store-vision exporter proposal: [docs/fleet_dashboard.md](docs/fleet_dashboard.md).
- GitHub remote is `ailerobotics/blender-supermarket` (origin); changes land via PR
  (see the merged-PR history). Use **gh-axi** for GitHub operations; never push to or
  merge into the default branch yourself.
- **Choreographed demo video** (`scripts/twin/demo/`, `scripts/run_demo_gpu.sh`): a SEPARATE
  demo mode that renders a scripted seven-stage restock walkthrough in the mission's Isaac
  scene with a live decision/reason-tree panel. Never changes the autonomous mission. Reuses
  `restock.scene.build` (same store/robot/cart/carton) WITHOUT the autonomy stack
  `physics_setup.assemble` adds (so it's light on VRAM and spawns no host/ROS); the base/cart
  are kinematic along the mission poses (demo-005: a stationary front-pick COLLECT, a
  straight-reverse COUPLE, one loop-free OUTBOUND arc, the PR154 return lane), the arm
  IK-tracks the scripted carton (`demo/arm_ik.py`), and every panel value is READ from the
  live stage each frame into `decision_log.json` (`driver.recompute` re-derives panel reasons
  on CPU from the logged reads — e.g. settled grasp/place readings — with no re-capture).
  Capture needs the mission chase-camera + fill-light recipe (plain Camera/viewport render
  black). Encodes with `restock.recording.Recorder` (or an in-memory rawvideo→ffmpeg stream
  when disk is tight). CPU tests: `scripts/twin/demo/tests/`; design + choreographed-vs-measured
  split: `~/firstmate/data/sim-demo-video-v1/report.md`.
  **Running `run_demo_gpu.py`**: use the `env_isaaclab` python (system `python3` has no
  `isaacsim`) under the ROS isolation the guard enforces (`ROS_DOMAIN_ID=88 ROS_LOCALHOST_ONLY=1
  ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST`; the box defaults to `SUBNET`). `--frame-start/--frame-end`
  renders ONE stage/segment as a compilation source (outbound = global frames 460–739 at fps 10).
  The OUTBOUND choreography must leave the backroom through the `sales_backroom_portal`
  (centre x=−3.875, 2.5 m opening at y=11.125; `store_environment.py`), NOT the solid wall —
  pinned by `scripts/twin/demo/tests/test_door_path.py` (base/cart footprint vs wall/door,
  articulation < 55°).
- **Real-mission demo cut + run checker** (CPU, only READS the run dir; RUNBOOK §20):
  `scripts/make_walkthrough_cut.py` turns a run's `mission-frames/` + `nav-trace.jsonl` +
  `mission.json` into the captioned, time-lapsed `walkthrough-demo.mp4`, labelling every
  `mission.json` workaround (`start_at`, `*_mode`, a non-sensor `source`) on screen.
  `scripts/check_walkthrough_run.py` is the pre-send gate. Carton-on-cart/on-bay evidence comes
  from the `RESTOCK_TRACE_POSES=1` nav-trace poses; without them it falls back to labelled
  frame analysis (chase-camera model). Frame `simulation_time` is Isaac stage time and trace
  `t` is mission time (offset 0.05 s cold, 0.55 s warm); `twin/demo/run_evidence.py` aligns them.
  Opt-in `--bt-panel` replaces the route-map panel with the demo-004 decision/reason tree (1920x1080,
  `walkthrough-bt.mp4`), driven entirely by the RECORDED evidence — `twin/demo/bt_timeline.py` replays
  `bt_events.jsonl`/`bt_snapshot.json` to per-frame node statuses + the authored step labels, and
  measures each reason (couple hitch gap, base→bay distance, lane progress, placed Δ, the failed step's
  recorded fault) from the traced poses via `twin/demo/measures.py`. Default (no flag) is byte-identical.
- **Multi-run compile** (CPU, sources read-only): `scripts/compile_demo_video.py` +
  `scripts/twin/demo/stitch.py` stitch ONE continuous seven-stage demo from per-stage segments of
  DIFFERENT runs (each stage from a run where it worked), re-rendering the demo panel so the ladder,
  clock and stage numbers run continuously across the source seams (earlier stages `done`, later
  `idle`; active stage + chips + scene pass through). Two source kinds, both reduced to the panel
  `entry`: `real` (bt_timeline over a run dir's mission-frames) and `choreo` (a demo dir's
  `decision_log.json` + scene cropped from its composited mp4). Cross-fades only at source boundaries.
  Spec + ranges: `data/sim-video-compile-v1/compile-demo-006.json` (ranges from
  `data/sim-video-stage-audit-v1/stage_sources.json`); design + the demo-006 stage→source table:
  `data/sim-video-compile-v1/report.md`. CPU tests: `scripts/twin/demo/tests/test_stitch.py`.
  **No real run has ever reached the sales floor** (every nav-trace stays in the backroom, y ≥ 13.75): the
  front place and the coupled tow back into the warehouse exist only in choreographed demo-005 — cut
  ranges + faults in `data/sim-video-story-sources-v1/story_sources.json`.
  **Before stitching, gate the spec on the clean cut ranges**: `scripts/check_cut_ranges.py --spec
  <spec or walkthrough.table.json>` (exit 1 = a segment shows a visual fault). Ranges:
  `data/sim-video-cut-qa-v1/cut_ranges.json`, regenerated by `scripts/cut_qa_ranges.py` (rules
  in `twin/demo/cut_qa.py`). Traces carry carton YAW only, so a carton tilt is a by-eye entry in
  `visual_faults.json`. Sharp edges: warm runs' scripted transfer floats the carton while the arm
  stays still, and their return unlatches the cart at the bay.

## Maintaining this file

Keep this file for knowledge useful to almost every future agent session in this project.
Do not repeat what the codebase already shows; point to the authoritative file or command instead.
Prefer rewriting or pruning existing entries over appending new ones.
When updating this file, preserve this bar for all agents and keep entries concise.
