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
  **move_group SIGABRT (`exit -6`, heap corruption) recovery** (`ros_adapter.py` `RosMotionPort`,
  default-on, no switch key): the upstream move_group dies intermittently mid-op; `poll()` holds the arm,
  waits for the launch respawn, re-syncs the planning scene (re-attach the carried carton) and re-plans
  from the live pose instead of e-stopping. Covers a death during PLANNING (`_handle_server_crash` /
  `_should_deadline_crash_retry`) AND during trajectory EXECUTION (`_server_action_for_state` watches
  move_group, not just the trajectory host, in execute states); the per-move crash budget
  (`MOVE_GROUP_CRASH_RETRY_MAX`) resets each move and a per-op cap (`MOVE_GROUP_CRASH_RETRY_OP_MAX`,
  `reset_crash_budget` at `controller.start`) bounds a storm, so a SECOND death in a later step still
  recovers. **Sharp edge:** a SIM-clock move-deadline miss is a render-pacing transient the controller
  RE-ISSUES (`MOVE_DEADLINE_RETRY`) — `poll()` must NOT fire `emergency_stop()` on it (`_fail_hold`, not
  `stop()`), because the global `/manipulation/stop` latches PERMANENTLY in the Isaac host
  (`isaac_ros_graph.check_health` → `'host emergency stop'`) and killed the run before the retry (fm225
  estopretry). CPU: `tests/test_ros_adapter_branch.py`; write-ups `data/sim-robust-movegroup-*/report.md`.
  **Sharp edge (planning deadline is SIM, not WALL):** the PLANNING-phase per-move deadline
  (`ik`/`plan_accept`/`plan_result`, `RosMotionPort._arm_plan_deadline`, `RESTOCK_PLAN_SIM_DEADLINE_S`
  default 30 sim-s) is on the SIM clock, NOT the legacy ~25 s WALL `self.timeout`. Under the `live_pose`
  `masked_depth` arm feed (a 2nd full-res render, ~16-30x wall/sim) the starved host executor could not
  service the plan-ACCEPT callback within 25 s WALL though move_group was alive, so the WALL deadline
  false-tripped `_should_deadline_crash_retry` — which read the slow plan as a crash and burned the whole
  crash-retry budget with ZERO deaths → e-stop (coldlive4). The crash-retry now also requires POSITIVE
  death evidence (`server_ready=False`, the move_group action server gone). Write-up + bounded GPU proof:
  `data/sim-robust-plan-deadline-pacing-v1/report.md`, `docs/examples/plan-deadline-pacing-collect-fragment.json`.
  **Sharp edge (the TOTAL planning budget is SIM too, not WALL — fixes the LAST pace-unsafe host timer):**
  `planning_wall_exceeded`/`PLAN_TOTAL_WALL_CAP_S` was the one planning timer PR 228 left on the wall; it
  tripped BARE at the live_pose place descent (coldlive sample-4: `PLAN_WALL_DIAG … future_done=False
  elapsed=120.10`, only ~4-8 SIM s passed) — a render-starved-but-alive plan read as a hang. It is now
  SIM-primary (`RESTOCK_PLAN_TOTAL_SIM_CAP_S`, default 22 sim-s) with a derived-generous frozen-sim WALL
  backstop (`frozen_sim_wall_backstop`, never pre-empts the sim leg); `RESTOCK_PLAN_WALL_CAP_S` is now only
  the backstop FLOOR (a stale sim-widen is harmless). The first-frame startup grace (`RESTOCK_STARTUP_GRACE_SIM_S`)
  and the payload/attach ack (`ros_payload._wait`, `RESTOCK_PAYLOAD_ACK_SIM_S`) got the same treatment.
  Every host e-stop now prints a NAMED trigger first (`HOST_FAULT_ESTOP`/`PORT_FAULT_ESTOP`; `PLAN_WALL_DIAG`
  names `leg=sim|wall`) so a bare `'host emergency stop'` can never recur. Full timer classification +
  switch keys: **`docs/host_timers_audit.md`**; CPU pacing proof `scripts/twin/restock/host_pacing_harness.py`
  + `tests/test_host_timers_pacing.py` (1x/16x/30x); bounded GPU proof
  `docs/examples/host-timers-pacing-proof-fragment.json`.
  That audit missed the **parked-STATION heartbeat** (`restock_host/incorporation_ack.py`
  `IncorporationAckState.snapshot` + its `restock/station.py` `StationState` twin): a flat wall-gap guard
  (`RESTOCK_STATION_WALL_GAP_S`) e-stopped coldliverecovery mid-live_pose-transfer at `STATION_ESTOP_PUBLISH …
  'parked heartbeat lost: wall gap 30.75 s'` while the SIM gap was healthy. sim-robust-station-heartbeat-simclock-v1
  converted it to the SAME shape: `RESTOCK_STATION_SIM_GAP_S` is the real bound; `RESTOCK_STATION_WALL_GAP_S` is
  now only the floor feeding the derived `pace_watchdog.frozen_sim_wall_backstop` (the formula moved to the leaf
  `pace_watchdog`; `ros_adapter.frozen_sim_wall_backstop` delegates there). The ESDF staleness reject
  (`preflight.incorporated_time`) was already sim-clock. Proof fragment
  `docs/examples/station-heartbeat-simclock-proof-fragment.json`; write-up
  `data/sim-robust-station-heartbeat-simclock-v1/report.md`.
  **The global `/manipulation/stop` wire has publishers BEYOND the motion controller** — the station-liveness
  nodes (`restock_host/incorporation_ack.py`, `restock/station_host.py`) and `restock/trajectory_host.py` —
  and they set NO `/manipulation/status`, so PR 231's naming did not cover them: a stop from them surfaced as
  a BARE `'host emergency stop'`. It latches PERMANENTLY on the Isaac OmniGraph subscriber (nothing publishes
  `False`). The full-chain transfer-START e-stop (sim-robust-transfer-start-estop-v1, coldlivefull): the
  collect station, left active through the couple→outbound→park nav gap (base moving, no parked heartbeat),
  tripped its wall-gap liveness guard → bare stop → latched; transfer's fresh `prepare` RECOVERED the host
  fault but could not un-latch the Isaac wire → transfer's first `check_health` read stale True. Fix: those
  publishers now print `STATION_ESTOP_PUBLISH`/`HOST_EXIT_ESTOP_PUBLISH`; `_host_fault_suffix` hints at the
  host-stack trigger when no controller fault matches; and `IsaacRosGraph.start()` RE-DERIVES the stop wire
  per op (`_clear_stale_global_stop`) so each op begins like a fresh staged-start graph (a genuine ongoing
  stop re-arms next tick; fresh-graph/staged-start path byte-identical, no new keys).
  Write-up `data/sim-robust-transfer-start-estop-v1/report.md`; CPU proof `tests/test_restock_native_stop.py`.
  **Pause the parked station on op completion before towing away (sim-robust-station-pause-on-transfer-v1).**
  The root sequencing fix for the above, after PR 237 moved the parked-heartbeat SIM-primary: coldwin2
  e-stopped the live_pose TRANSFER because the collect station, left ACTIVE through the whole
  couple→outbound→park→transfer-warm span, hit a 30.05 s SIM gap (0.05 s over the 30 s budget) and published
  the global stop, which latched even though the station recovered one tick later. `RuntimeAdapter.start('couple')`
  now calls `manip.pause_station(…, suspend_only=True)` → `StationClient.pause()` sends an explicit **`pause`**
  command; the host watchdog is **paused-aware** (`IncorporationAckState.paused` + `StationState` twin, state
  `'paused'`): a paused active epoch publishes NO stop for any op length/pacing, while an in-use station still
  faults by name. The next op re-prepares a fresh epoch (clears the pause). Belt-and-suspenders: a within-budget
  recovery now RELEASES the latched Isaac wire mid-op (`incorporation_ack` publishes `Bool(False)`,
  `STATION_ESTOP_RELEASE`; extends PR 235's op-START re-derive). Default-on, no new keys; staged-start/scripted
  byte-identical. Bounded GPU proof = the full capstone (`mission-runtime-capstone238.json`, `orchestrate-coldwin2.sh`)
  with `RESTOCK_STATION_SIM_GAP_S=30` (NOT the 60 s quick-lever). Write-up
  `data/sim-robust-station-pause-on-transfer-v1/report.md`; CPU proof `tests/test_host_timers_pacing.py`
  (`simulate_far_station_paused`), `test_restock_host.py`, `test_restock_station.py`.
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
  **Sharp edge — never put `joint_constraints` on a cuMotion POSE goal**: the Isaac ROS cuMotion MoveIt
  planner reads ANY `joint_constraints` in `goal_constraints` as a FULL c-space target (it builds
  `cspace_position` from exactly those entries and DROPS the pose), so a partial (1–2 joint) wrist-bound
  list made it call `planToCSpaceTarget([[-0.675, 1]])` against a 6-DOF robot → `'Invalid c-space position …
  [2] must equal [6]'` → `cumotion::FatalException` CUDA OOM → container death (pickguard1, the PR254
  regression). To bound a pose goal's wrist family, plan a CLEAN pose goal and validate the planned
  TERMINAL config post-plan (`ros_adapter.pose_guard_decision` + `transfer_carry.in_clear_region`),
  re-planning off the self-colliding family — never joint_constraints as bounds. The well-formed c-space
  shape is the seeded-KDL joint path's **six**-entry list. Fix + evidence:
  `data/sim-robust-pick-wrist-guard-cspace-fix-v1/report.md`.
  **Place-on-cart goal/branch fix** (opt-in, default-off ⇒ byte-identical): the non-cage reposition deck
  place poses have TWO collision-free IK families — UPRIGHT (`shoulder_lift≈-1.9`, holdable) and a DEEP
  fold (`≈-3.57`, the soft arm can't hold under load → collect1 sagged 0.39 rad short → heartbeat lost);
  the host's folded-carry seed also hits the self-colliding wrist branch (collectsrc's storm). Fix the
  GOAL, not the planner: host keys `collect_place_preplace_shape` (a fixed UPRIGHT clear-region preplace
  driven via `begin_joint_goal` like `cage_preplace_shape`) + `collect_place_wrist_guard` (the transfer
  wrist-guard clear-branch re-seed extended to the collect place/retreat pose-IK). CPU proof
  `tests/test_restock_cart_place_goal.py`; targets + reach margins `data/sim-robust-cart-place-goal-v1/`
  (`station-contract.md`, `report.md`); GPU fragment `docs/examples/cart-place-goal-mission-fragment.json`.
  **Place-on-cart descent/release completion** (opt-in, default-off ⇒ byte-identical): with the goal fix
  above active, collect2 reached the UPRIGHT preplace and the place-descent trajectory itself SUCCEEDED,
  but the mission stationary guard (`runtime.py` `not p.parked`) aborted the stage mid-descent — the
  extended carton-loaded arm's reaction shoves the YAW-LOCK-ONLY braked base ~4 cm in translation
  (isaac_port leaves translation free while a carton is welded), flipping the 0.25 s windowed `parked`
  flag False for a few ticks → `'base moved during stationary operation'`; the carton stayed held and the
  arm never stowed → re-begins failed `arm_not_in_transport_timeout`. Fix (two keys): RUNTIME
  `collect_place_base_settle_m` tolerates a BOUNDED base excursion from the place_approach anchor through
  the loaded place phases (the shove re-settles and is far inside the 0.25 m place-verify xy tol) so
  place→release→stow completes, failing with a NAMED reason only beyond the bound; HOST
  `release_detach_timeout_s` (SIM-clock) fails the RELEASE fast with a named reason if the SIM_ATTACH weld
  doesn't detach, instead of carrying the carton back up on stow or stalling to `phase_timeout`. CPU proof
  `tests/test_restock_runtime.py` (`test_loaded_place_*`) + `tests/test_restock_manipulation.py`
  (`test_release_*`, `test_collect_place_runs_descent_release_stow_*`); GPU fragment
  `docs/examples/cart-place-descent-mission-fragment.json`; write-up `data/sim-robust-cart-place-descent-v1/`.
  **Place-on-cart endpoint self-collision** (opt-in, default-off ⇒ byte-identical): with the goal + descent
  fixes above, collect4 reached the upright preplace and the host planned the place, but the deck PLACE
  config is rejected by cuMotion as an attached-object SELF-collision (`'Invalid c-space position: self
  collision detected'`, **0 world collisions**) — the held carton is a 34-sphere cuMotion `attached_object`
  (`isaac_motion.attach_object_world` at `gripper_tcp`) NOT self-collision-ignored vs forearm/wrist/upper-arm,
  so the set-down fold brings it inside the self-collision margin EVEN THOUGH the bare arm is self-clear
  (+0.0339 m at BOTH preplace and place) and PR #209's wrist-guard accepted the config. The cart deck is not
  a cuMotion WORLD object here (`world_obstacles:[]`, `base_collision_object:false`). Fix = port the cart-cage
  / live_pose (PR #215) mechanism: HOST key `collect_place_straight_line` routes the deck place descent + empty
  retreat as the collision-OFF straight-line seeded insert (`begin(straight_line=True)`, same `straight`
  predicate, no second mechanism); the preplace standoff stays collision-aware. Scoped to the non-cage collect
  place (`_collect_deck_place`), so transfer/live_pose/scripted-still are byte-identical. CPU proof
  `tests/test_restock_manipulation.py` (`test_collect_deck_place_*`); GPU fragment
  `docs/examples/collect-place-straightline-mission-fragment.json`; write-up
  `data/sim-robust-collect-place-straightline-v1/`.
  **Honest-collect robustness** (opt-in, default-off ⇒ byte-identical): the grasp-critical collect
  source park uses a tolerance-aware arrival (`navigation` `strict_xy_arrival`, set automatically on the
  collect leg — never declares `arrived` outside the true xy tol even under the fm482 hysteresis latch),
  and the host config key `grasp_press_timeout_s` (SIM seconds, `ros_host.py`) bounds the WHOLE
  unconfirmed pre-weld press so an unsealing grasp FAILS FAST with a named reason before the opaque 90 s
  WALL arm-command heartbeat (runs 321/327). Design + bounded collect-only validation config
  (`--stop-after collect`, no GT): `data/sim-robust-collect-arrival-v1/report.md`,
  `docs/examples/collect-only-validation-fragment.json`.
  **move_group SIGABRT crash + respawn scene re-sync**: the intermittent `move_group` `exit code -6`
  (`malloc(): unaligned tcache chunk detected`) during manipulation is UPSTREAM heap corruption in the
  `moveit_ros_move_group`+cuMotion-plugin binary (crashes handling a planning request concurrent with
  nvblox ESDF updates), NOT a repo concurrent scene mutation — the single-threaded controller issues
  attach/detach strictly BETWEEN moves — and NOT patchable; it is distinct from the
  `cumotion::FatalException` OOM / >1024-sphere CONTAINER death (that's `component_container_mt-2`, capped
  by `PLAN_TOTAL_ATTEMPTS_CAP` + the sphere model). `launch.py` respawns move_group and
  `ros_adapter.RosMotionPort` re-sends the plan (`MOVE_GROUP_CRASH_RETRY`), but a respawned move_group's
  FRESH planning scene drops `restock_carried_unit`. Fix (always-on, no-op when nothing held ⇒
  byte-identical): `RosMotionPort` mirrors the attach state in `_attached_profile` (set via
  `update_payload`) and `_resync_scene_after_respawn()` re-attaches the carton BEFORE either resend site
  resends the plan; `update_payload` now also raises if called in a `_PLAN_IN_FLIGHT_STATES` state
  (serialization invariant, inert on the real path). PR #223 handled only the detach tolerating the lost
  object. Report + bounded GPU proof: `data/sim-robust-movegroup-respawn-v1/report.md`,
  `docs/examples/movegroup-respawn-collect-fragment.json`.
  **A mid-op death must RETRY, not e-stop** (sim-robust-movegroup-estop-retry-v1): the respawn recovery
  above only fires from the server-down poll (`_handle_server_crash`) and the plan-result wall-deadline
  (`_should_deadline_crash_retry`). A death that strikes mid-replan (coldlive: the collect STOW) can
  instead surface as a RAISE inside `_poll` (a lost-server result future, an ABORTED result whose
  `result.result` is `None`, a service failure), which `poll()`'s generic handler sent straight to
  `stop()`→`emergency_stop()` (terminal `host emergency stop`, arm heartbeat lost) — no retry. Fix
  (always-on, byte-identical no-crash path): `poll()` routes the caught raise through
  `_recover_movegroup_death_or_none`, which HOLDS the move (arm keeps its last pose, weld intact) for the
  respawn+re-sync+resend ONLY when the action server for the current state is GONE and crash-retry budget
  remains; a LIVE server (genuine rejection) / spent budget / the `'did not respawn'` escalation still
  e-stops honestly (pure discriminator `_should_recover_from_poll_fault`). `planning_wall_exceeded()` is
  suspended while `_awaiting_server_respawn`. Report + bounded GPU proof (collect-only, `kill -ABRT` the
  move_group pid at the stow — integration-owned): `data/sim-robust-movegroup-estop-retry-v1/report.md`,
  `docs/examples/movegroup-estop-retry-collect-fragment.json`.
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
- **Arm masked-depth feed lifecycle**: the out-of-Isaac publisher is a STATION-level resource the
  host incorporation-ack consumes, NOT a per-op one. A per-op manip failure must hold the gripper but
  NOT close it — `ros_manipulation.stop(close_sensing=False)` (the two `step()` failure sites); only
  genuine mission teardown closes the feed. Closing it on a per-op `failed` froze the station ack
  stamp → `StationAckStreamStalled` masking the TRUE reason (every recorded arm-feed death was this
  `close()`, rc=0 writer_eof — never an OOM/signal kill). The bridge is supervised:
  `masked_depth_ipc.SubprocessSupervisor` respawns the publisher on any non-`close()` exit within the
  ack window (bounded `max_restarts`), and `FrameWriter` bounds its send buffer. Pre-boot VRAM gate
  `scripts/twin/restock/gpu_budget.py` (12 GB card, whispr ~2.2 GB resident): one arm feed fits, a 2nd
  full-res feed doesn't — keep `mast_depth` off for live_pose transfer. Report + bounded config
  fragment: `data/sim-robust-arm-cam-feed-v1/`. **Same rule for the SHARED control relay**
  (`control_ipc.HostControlPublisher`, `host_control`): it is one subprocess forwarding
  `/restock_sim/station/status` (StationClient) AND `/manipulation/status` (graph), so the per-op
  `ros_manipulation.stop(close_sensing=False)` must NOT tear it down either — `IsaacRosGraph.close`
  takes `close_control=` (gated on `close_sensing`), closing the relay only at genuine teardown. Closing
  it on a per-op `failed` was the OTHER half of the live_pose freeze: the StationClient read a frozen ack
  → `StationAckStreamStalled` at ~sim 31 s, masking the true `collision planning or execution failed`
  (armcam/transfer2/bench1). Report: `data/sim-robust-station-ack-v1/report.md`.
- **VRAM headroom monitor + crash-safe recording**: the full capstone runs within ~500 MiB of the
  12 GiB top (completes peak ~11.7 GB, whispr ~2.2 GB resident) and can SIGSEGV late (coldwin13, native,
  no `mission.json`). Two pieces (`data/sim-robust-vram-headroom-v1/report.md` has the component budget
  table + per-run peaks). (1) **Opt-in monitor** `scripts/twin/restock/vram_headroom.py` (runtime key
  `vram_headroom`, DEFAULT OFF ⇒ byte-identical): samples used VRAM on the SIM clock sub-second (NVML or
  a background nvidia-smi poller), and below `floor_mib` (default 800) sheds optional load in a fixed
  named ratchet — `capture_rate` (widen the walkthrough record stride) → `arm_feed_economy`
  (stage-blocked in collect/transfer/verify, no runtime actuator — set `masked_depth.economy` at boot
  instead) → `secondary_cameras` (the fm380/fm381 `stage_scoped_teardown`, frees ~0.95 GB+ of never-read
  decode products) — logging each `VRAM_SHED` by name, never shedding the live_pose feed/nvblox/planner,
  and naming `VRAM_CEILING_REACHED` if still below floor. Writes `vram-headroom.json` (last headroom +
  peak) for the crash record. **Render resolution + GI/DLSS are NOT the lever** (≤160 MiB; memory
  `mission-vram-ceiling-12gb`); the late creep is an internal RTX/Vulkan allocator event, so freeing
  render targets is the real mechanism. (2) **Crash-safe recording** (always on): the mission appends a
  durable `mission-progress.jsonl` per stage transition (`crash_record.ProgressWriter`), and
  `run_mobile_restock.sh` (the shell parent — NO longer `exec`, so it survives a native python SIGSEGV)
  runs `scripts/finalize_crashed_run.py` after the process exits: on a signal death / wall-cap timeout
  with NO `mission.json`, it writes `state=crashed` (crash signal, last stage, sim time, stage ladder,
  last VRAM) reconstructed from `mission-progress.jsonl` else `nav-trace.jsonl`+`bt_events.jsonl` (so it
  also recovers legacy runs like coldwin13). No-op when `mission.json` exists ⇒ success path
  byte-identical. CPU tests `test_restock_vram_headroom.py`, `test_restock_crash_record.py`; GPU proof
  `docs/examples/vram-headroom-mission-fragment.json`.
- **move_group respawn (bounded + observable)**: the host launch (`restock_host/launch.py`)
  respawns `move_group` on the intermittent cuMotion SIGABRT instead of tearing the stack down.
  ROS 2 launch respawn fires ONLY on a real process EXIT — a clean SIGABRT (exit -6) respawns,
  but a `move_group` that HANGS in glibc's abort path (`malloc_consolidate(): invalid chunk size`
  with NO following `process has died` line) never exits, so no launch respawn can fire; that hang
  is owned by the client `RESPAWN_WAIT_CAP` fast-fail (`restock_manipulation/ros_adapter.py`,
  PR 234/242), not the launch. The respawn is BOUNDED + logged by name with an attempt count via
  `restock_host/movegroup_respawn.py` (pure/CPU-tested): optional host-config block
  `movegroup_respawn: {max_retries, delay}` (default 3 / 3.0 s, surfaced in `launch_plan`), each
  respawn/give-up prints `MOVE_GROUP_RESPAWN[_GIVE_UP]`. The `OnProcessExit` logger is observe-only
  (never respawns/kills — launch stays the sole respawner). Report + GPU double/triple-death proof:
  `data/sim-robust-movegroup-respawn-reliability-v1/report.md`,
  `docs/examples/movegroup-respawn-reliability-mission-fragment.json`.
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
  design in `data/sim-restock-nav-lanes-v1/report.md`. **Outbound lane acquires from the actual couple
  seat** (sim-robust-outbound-lane-start-v1, opt-in default-off): the improved crosstrack couple seats
  the base a VARIABLE distance off the fixed authored dock start (coldlivewin +0.5 m north → the lane
  never acquired, froze to the 120 s nav-stuck). Per-lane `lead_in_max_m` synthesises a bounded, swept-
  validated forward pure-pursuit COUPLED merge from the measured seat ONTO the centreline (a lead-in to
  the EXACT start is NOT viable — a forward-only train 0.5 m lateral off a flat lane free-plans a ~12 m
  loop), and `navigation.tow_lane_acquire_timeout_s` fails a genuinely non-acquirable lane by NAME in a
  few sim seconds (`lane_not_acquired_timeout`/`lane_leadin_unreachable_timeout`) not the 120 s WALL
  budget; re-centre the authored start to the seat-distribution middle so the lead-in absorbs the spread.
  Example `docs/examples/outbound-lane-leadin-fragment.json`; design `data/sim-robust-outbound-lane-start-v1/report.md`.
  **Couple speed / outbound stall** (fm482, both
  opt-in): `navigation.arrival_xy_hysteresis_m` stops the couple-standoff arrival dither (~700 s wall per
  warm run) and `navigation.coupled_creep_escape` stops the outbound `measured_turn_radius` noise
  livelock; offline proof on the recorded runs `scripts/check_couple_speed.py`. A LATER residual outbound
  stall (warm9/warm10: base pinned x≈−3.9 still COMMANDING forward, jittering in place; frozen ~14 s SIM =
  ~285 s WALL under render pacing, so the 120 s WALL no-base-progress budget false-failed the stage on the
  wrong clock — creep-escape/`measured_turn_fault` both miss it) is caught by the opt-in SIM-clock
  net-progress monitor `navigation.coupled_progress_escape` (escalate→fault `coupled_no_progress`→replan in
  sim-seconds); see docs/restock_navigation.md, `scripts/check_outbound_progress.py`,
  `docs/examples/outbound-progress-mission-fragment.json`. **Couple fine-dock limit cycle**
  (`fine_dock.approach_mode`, opt-in, default `'hitch_jacobian'` ⇒ byte-identical): the legacy creep has a
  degenerate zero-command equilibrium LINE (`lat = gain_yaw·|hitch_x|/gain_lin·yaw_err`, one ω for both
  lateral-hitch AND yaw), so the couple approach can land OUTSIDE the latch box and freeze — coldlivefinal
  stuck ~26 mm/~4.8° (yaw > 4° tol) for the whole 30 s cap while the still-arm coldhonest landed ~2.7°
  (inside) and latched. `'crosstrack'` servos the rear hitch (leading point of the reverse approach →
  stable minimum-phase) onto the cart axis, holding back until aligned then seating straight in.
  Example `docs/examples/couple-fine-dock-crosstrack-fragment.json`; design + recorded evidence
  `data/sim-robust-couple-fine-dock-v1/report.md`. **Couple APPROACH landing consistency**
  (`couple_cart_relative` runtime block + `fine_dock.progress_cap`, both opt-in, default off ⇒
  byte-identical): PR 232 fixed the servo but the ENTRY stayed variable — the authored `couple_standoff`
  is a FIXED map pose (yaw 0) while the pinned cart's MEASURED yaw varies run to run (coldlivefinal
  −1.6°, coupledock +19.2°), so the base lands aligned to the MAP not the CART and the fine-dock enters
  25–470 mm / 8–27° off (the `nav-trace` `cart` row is `[x,y,yaw_deg,z]` — yaw is the DEGREES field, not
  the z). Fix: navigate the standoff leg to `fine_dock.standoff_from_cart(measured cart)` (on-axis, base
  yaw = cart yaw, hitch held `standoff_offset_m` back), gate entry on the cart-relative cross-track + yaw
  (re-approach a fresh cart-measured standoff when outside, bounded `max_reapproach`; fail honestly when
  exhausted), and `progress_cap` replaces the blunt 30 s `time_cap_s` (keep docking while the residual
  improves, hard `progress_max_time_cap_s`, fast-fail `progress_patience_s`). CPU proof
  `scripts/check_couple_approach.py`; example `docs/examples/couple-approach-mission-fragment.json`;
  design `data/sim-robust-couple-approach-v1/report.md`. Uncoupled drive-fidelity lever = `caster_lift_m`
  (fm434 bench `base-drive-bench-*`: 0.097→0.988). **The residual coupled stall is drive-authority, NOT
  geometry** (sim-robust-coupled-traction-v1): outbound_dock_to_bay is a STRAIGHT y=15.25 pull (the portal
  is at y=11.125 — the x≈−3.875 coincidence is a red herring), every recorded robot+cart footprint is
  clear of the 806-obstacle map and the real controller follows the lane to goal KINEMATICALLY, so the
  stall is PhysX re-breakaway-from-rest (warm8/warm11 pass only with entry momentum). Fix = raise BOTH
  `caster_lift_m`→0.03 AND `wheel_drive.max_force`→200 (at 80 N·m the per-wheel torque cap 1000 N binds
  below μ·N_drive, so caster lift ALONE can't help — the budget shows the sign; GPU bench is truth).
  One-command CPU proof + acceptance gate (`--gpu-run` grades a new trace) `scripts/check_tow_traction.py`;
  pure core `scripts/twin/traction.py`; bounded GPU config + build steps
  `docs/examples/coupled-traction-fragment.json`; write-up `data/sim-robust-coupled-traction-v1/report.md`. **Complementary recovery** (sim-robust-outbound-stall-recovery-v1):
  opt-in `navigation.coupled_progress_recovery` turns the `coupled_progress_escape` fault into a BOUNDED
  break-away maneuver (continuous pulsed committed-forward surge, `coupled_breakaway_*` keys; reverse is
  clamped off for a coupled train so it rocks forward, not back — `MobileController.command` already
  force-releases every brake each drive tick, so `parked=True` is a readout not a stale brake) that re-tows
  a bounded number of times before failing `coupled_no_progress` by name; default-off ⇒ byte-identical.
  Example `docs/examples/outbound-stall-recovery-mission-fragment.json`, write-up
  `data/sim-robust-outbound-stall-recovery-v1/report.md`. **Reinforced break-away ROCK**
  (sim-robust-coupled-breakaway-from-rest-v1): coldwin9 pinned at the throat with the base yawed ~5° off-axis —
  a base+cart envelope CATCHES THE THROAT LIP (a forward WEDGE, not a gate refusal; all 3 runs enter at
  ~0.4 m/s and dip to rest, so the discriminator is the YAW DRIFT in the slow from-rest creep, NOT momentum at
  the latch), and the forward-only pulse above can't free a forward wedge (9 surges → 0 advance). Opt-in
  `navigation.coupled_breakaway_rock` (+`coupled_breakaway_launch_mps`, requires `coupled_progress_recovery`,
  default-off ⇒ byte-identical) makes the recovery surge an ELEVATED forward launch (clamped to `max_speed`:
  more momentum + the higher coupled steering cap `wmax=v/1.5`) with the corrective turn SATURATED to ±wmax
  toward the route heading on the hi pulse, to rotate the yaw-wedged base back into alignment (no reverse — the
  coupled v≥0 interlock stands; a bounded reverse back-off is the documented next lever, owner sign-off to relax
  it). CPU proofs `tests/test_articulated_navigation.py` (`test_breakaway_rock_*`), `tests/test_coupled_breakaway_rock.py`
  (real coldwin7/9 recorded-trace replay + `coupled_tow_margin` gate); fragment
  `docs/examples/coupled-breakaway-rock-mission-fragment.json`; write-up
  `data/sim-robust-coupled-breakaway-from-rest-v1/report.md`. **Lidar sensor-frame yaw (vr1, now FIXED default-on):**
  `Scan.sensor_to_base` is (x, y, YAW) but the legacy `isaac_bundle` wiring passed the lidar URDF origin
  (x, y, z) — the `lidar_frame` is a fixed rpy-0 child of base_link, so its z (0.15 m) is a mount height,
  NOT a yaw; reading it as yaw rotated every return 8.59° and was the sole source of the tow-lane
  "lidar-proud" phantoms (warm8/warm11 replay: fixed 0 returns > 0.12 m from any mapped AABB vs ~12k legacy)
  and the ~0.83 m capsule-vs-planner deadlock. `isaac_bundle.lidar_sensor_to_base` now defaults yaw 0;
  `lidar_sensor_yaw_fix: false` restores the legacy rotated wiring (reproduce a pre-fix run only). Because
  the fix removes the phantom class that forced it, `navigation.tow_lane_trust_static` is proposed for
  retirement (pending the bounded GPU run) — but KEEP `tow_lane_static_tol_m` and the `tow_lane_trust_unknown`
  coverage waiver (independent of the yaw bug). CPU proof `scripts/check_lidar_yaw_fix.py`; decision + GPU
  config `data/sim-restock-lidar-yaw-fix-eval/` (orientation locked by `test_restock_mission_tail.py`).
  **Outbound throat yaw-align** (`navigation.coupled_heading_hold`, opt-in, default off ⇒ byte-identical):
  the UPSTREAM fix for the `sales_backroom_portal` throat wedge — keep the coupled base aligned to the lane
  (east) travel axis BEFORE the lip so it never enters yawed (PR 250's rock is the after-the-wedge backstop).
  Five recorded tows: pass runs cross at −0.2…+0.3°, fail runs drift CCW in the slow from-rest creep
  (coldwin9→+4.9°, coldwin10→+2.2°) and wedge (~2° margin). The follower's lateral term COMMANDS the CCW
  rotation and at the crawl `wmax=v/1.5` collapses; the throat is INVISIBLE to the nav static map (footprint
  valid at every yaw, slack 0.8 m — a PhysX door-frame contact, gate says `tracking`), so the drift on the
  lane is the only trigger. On a straight segment past `heading_hold_bound_deg` (1.0°) CCW of the goal
  heading: hold a fixed `heading_hold_creep_mps` (0.14, restores authority + caps speed) and saturate the
  corrective turn toward the axis; one-sided, forward-only (v≥0). Doc + the five-run numbers:
  [docs/restock_navigation.md](docs/restock_navigation.md) "Pre-throat heading hold",
  `data/sim-robust-outbound-throat-yaw-align-v1/report.md`; fragment
  `docs/examples/outbound-heading-hold-mission-fragment.json`; CPU proof `test_outbound_heading_hold.py`.
- **Pace-aware producer watchdogs**: the mission renders slower than wall (measured 0.17-0.31x
  sim-to-wall), so a wall-timed watchdog false-trips a healthy-but-slow pipeline. The shared predicate
  `twin/restock/pace_watchdog.py` `is_dead(mode='either'|'both')` states the rule once (fire only when
  truly dead: stamps frozen while sim advances, AND/OR enough wall elapsed that an alive producer would
  have recovered). Converted: the arm-command heartbeat (`native_transport.ArmCommandGate.watchdog`,
  SIM-clock primary + wall-since-sim-now-advanced backstop, `RESTOCK_ARM_HEARTBEAT_SIM_BACKSTOP_S`), the
  station-ack frozen-stamp `StationAckStreamStalled` (now needs stage AND wall,
  `RESTOCK_STATION_ACK_FROZEN_WALL_S`, killed the warm false trips of runs 341/346/348/350/354/356), and
  the no-base-progress nav-stuck (`RuntimeAdapter._nav_stuck_check` gained a governing SIM budget
  `RESTOCK_NAV_STUCK_SIM_S`; legacy wall path byte-identical when unset). Already-sim-clock watchdogs
  (station host-silence/stage-liveness, nav `blocked_timeout`/`goal_timeout`, `/manipulation/status`
  feedback) are unchanged + regression-locked. CPU tests `twin/tests/test_sim_robust_watchdogs.py`;
  inventory + integration-lane env switches `data/sim-robust-watchdogs-simtime-v1/report.md`. See the
  `freshness-sim-clock-not-wall` principle.
- **Teardown robustness (record-first + ordered teardown)**: a run that ends on a per-op failure
  must still leave `mission.json` and exit without a core dump. `SimulationApp.close()` can hard-exit
  in the renderer shutdown, so the launcher (`run_mobile_restock.run`) writes `mission.json` BEFORE
  any Isaac/ROS teardown on every exit path (the `except` + the top of the middle `finally`, via
  `twin.restock.shutdown.write_mission_record`), then runs an ordered teardown
  (`shutdown.ordered_teardown`: `close_feeds → encode_walkthrough → release_render → close_app`, each
  best-effort) so RTX render products (`Bundle.release_render`) + camera feeds are released before the
  app close. Unconditional robustness, no new switch keys; successful runs are byte-identical apart
  from the earlier record write. CPU: `twin/tests/test_restock_shutdown.py`. **Sharp edge:** the
  original armcam crash was a NATIVE SIGSEGV mid-`world.step` under 12 GB VRAM OOM (py-spy), NOT at
  `app.close()` — the mission never reached a Python terminal, so no `finally` can help it; that run
  additionally needs the per-op failure to TERMINATE before OOM (PR 215 collision + the
  `mission-vram-ceiling-12gb` lane). Report + bounded GPU proof:
  `data/sim-robust-teardown-crash-v1/report.md`, `docs/examples/teardown-crash-proof-fragment.json`.
- **Boot ESDF-incorporation gate** (opt-in HOST key `esdf_boot_ack_grace_s`, SIM seconds; default 0 ⇒
  off ⇒ byte-identical; registered HOST in `config_preflight`): names a NEVER-acknowledged nvblox ESDF
  incorporation at boot, before the first arm motion streams, instead of the opaque ~40 s mid-stage
  `collision_dt` invariant trip (meshuris1: mirrored station `collision_time` stuck at the `-1`
  not-ready sentinel from tick 1 → `CONTROLLER_INVARIANT_TRIP collision_dt=40` at the first observe
  step). The pure gate `restock_manipulation/esdf_boot_gate.py` (driven from `TransferController.step`)
  HOLDs the op (no motion) for a sim-clock grace awaiting the first valid ack (`collision_time` leaving
  `-1`), then FAILS FAST by name (grace elapsed, stuck stamp, likely cause: nvblox not incorporating /
  stale commission baseline); once a valid ack arrives it disarms and the `collision_dt` invariant is
  the sole owner (band untouched). Does NOT fix why a stale baseline fails to incorporate, or the
  `collision_dt`/station-liveness thresholds. CPU `tests/test_sim_robust_esdf_boot_gate.py`; GPU proof
  fragment `docs/examples/esdf-boot-gate-mission-fragment.json`; write-up
  `data/sim-robust-esdf-boot-gate-v1/report.md`.
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
  reposition (`restock_manipulation/transfer_mode.py`). **`live_pose` REQUIRES
  `RESTOCK_TRANSFER_CARRY=wrist_guard`** — it fixes only the PICK reach; its preplace/place/retreat stay
  seeded-KDL, so without the guard the controller FAILS FAST at observe with a named reason rather than
  silently running the `ur_wrist_2`-vs-forearm jam (audit `transfer-preplace-wrist-branch-jam`). **Sharp
  edge:** `RESTOCK_TRANSFER_CARRY=wrist_guard` covers only the PLACE half (`transfer_carry.GUARDED_PHASES`
  = preplace/place/retreat); the live_pose transfer PICK (approach cuMotion pose goal + contact straight-line
  KDL) was unguarded and intermittently IK'd the same `ur_wrist_2`-vs-forearm family, where the contact
  descent limit-cycles ~0.17 rad / ~1.6 rad/s and PR239's settle envelope fails it (coldwin12; coldwin11 took
  the clear branch). Opt-in HOST key `transfer_pick_wrist_guard` (default off ⇒ byte-identical) extends the
  clear-region bounds (`carry_wrist1_cap`/`carry_wrist2_min`) to the pick approach+contact; CPU
  `tests/test_restock_transfer_pick_wrist_guard.py`, fragment
  `docs/examples/transfer-pick-wrist-guard-mission-fragment.json`, write-up
  `data/sim-robust-transfer-settle-oscillation-v1/report.md`. The
  live_pose PLACE lowers the carton ONTO the bay, so its place config sits ~13 mm above the bay top (inside
  cuMotion's ESDF voxel margin) and a collision-aware plan is rejected (`world collision detected`); the
  place descent + empty retreat therefore route as the collision-OFF straight-line insert (like the
  cart-cage place), PREPLACE stays collision-aware — automatic within live_pose, no new key
  (`data/twin-robot-in-store-boot-v1/livepose-pick-collision/report.md`). **Sharp edge: the wrist-guard
  wrist-flip re-seed can land the PREPLACE in a clear-but-DEEP shoulder fold the arm can't reach**
  (coldwin6: it accepted `ur_shoulder_lift` −4.003, froze at −3.65 → PATH_TOLERANCE_VIOLATED). The preplace
  standoff has a holdable UPRIGHT clear branch (`ur_shoulder_lift` ~−1.95) AND a deep fold; the full-chain
  park (unlike the staged recorded_bay_park, which accepts upright at seed 0) seeds KDL into the deep one.
  Opt-in HOST key `transfer_preplace_reach_guard` (default off ⇒ byte-identical) requires a clear solution
  to also be a holdable fold (`transfer_carry.reachable_fold`) + seeds the upright branch, rejecting a deep
  fold by name before execution (`restock_manipulation/ros_adapter._guard_rejects`). CPU:
  `tests/test_restock_transfer_preplace_reach.py`; fragment `docs/examples/transfer-preplace-reach-mission-fragment.json`;
  write-up `data/sim-robust-transfer-preplace-reach-v1/report.md`. `scripted`
  = the labelled safety net
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
  error (warm5). **Sharp edge:** `in_clear_region` WRAPS wrist_1, so a far +2π IK branch (raw ~+4.1, wraps
  to −2.16 ≤ cap) reads "clear" yet makes the joint sweep UP through the ~0.69 stop (execretest jam) —
  `_solve_seeded` now re-expresses the commanded wrist_1 to the equivalent in-limit branch ≤ cap
  (`transfer_carry.clear_branch`; identical TCP, so the still carton path is byte-identical), and the plan
  carries `wrist1_within_cap()` so the executor FAILS FAST (`begin`→`_step_executor`) with a named reason
  when no clear branch exists, never streaming the self-colliding trajectory. The `live_pose` pick does NOT
  share this (cuMotion plans it under a hard `ur_wrist_1 ≤ cap` goal constraint). Bounded GPU proof fragment:
  `docs/examples/transfer-executor-mission-fragment.json`. A mid-transfer executor refusal FAILS FAST with a
  named reason + a keep-alive hold (never the silent `still` degrade that starves the arm-command heartbeat). Offline proof (incl. the sphere model
  + warm5 host-validation replay): `data/sim-video-executor-arm-v1/report.md`,
  `tests/test_restock_scripted_transfer.py`. **Sharp edge (execution-clock anchor):** `trajectory_host`
  anchors a goal's trajectory time origin to the FIRST post-install `/joint_states` stamp
  (`anchored_start`), NOT `now()` at the execute-callback — the mission can burst-advance the sim many
  seconds between accept and first feedback (transfer1: ~11 s, with `fb_age=0` proving same-clock, not a
  stale frame), and the old `start=now()` charged that as trajectory progress and e-stopped the still-parked
  arm on the first sample (misdiagnosed as the warm5 wrist self-collision). Applies to EVERY mission arm
  move. **The execute loop's own clock is pace-aware too (`execute_clock`, fm-sim-robust-trajectory-clock-load):**
  because `start` is a measured-feedback stamp that can LEAD the node `/clock` by up to `FEEDBACK_RENDER_LEAD_S`
  under render pacing, the first post-anchor `now()-start` can read slightly negative — the same delivery race
  `feedback_fresh`/`reference_time` absorb. The old hard `elapsed<0` fired `TRAJ_HOST_FAULT trajectory
  clock/timeout failure` (twice) on a VALID 2-point plan under the heavy live_pose load (coldlive sample 2,
  anchor_lag 0.017 s); it now clamps a within-lead negative to the start pose, failing by name only on a
  genuine regression (`trajectory clock regression`) or the frozen-sim WALL backstop. `begin()` reports+bounds the IK-plan latency (`RESTOCK_SCRIPTED_TRANSFER_IK_BUDGET_S`, default 120 s,
  loud-not-fatal). Proof + GPU re-run config: `~/firstmate/data/sim-robust-executor-clock-v1/report.md`,
  `tests/test_restock_trajectory_host.py`. Shelf-slot place (sales-floor endcap G3_back_s2, s3 board
  0.25 m above): `RESTOCK_SCRIPTED_TRANSFER_PLACE=front` (env, scripted only, default off) slides the carton
  in under the board instead of descending through it — task-space plan, still arm only; staging keys +
  CPU proof `scripts/check_front_transfer_staging.py`, `data/sim-video-story-sources-v1/report.md`.
- **Executor LOADED-pick `ur_shoulder_lift` ~0.10 rad miss is NOT a gravity sag** (sim-robust-executor-
  shoulder-drive-v1, CORRECTS the PR218 premise). CPU static balance on the run URDF: true gravity on
  shoulder_lift at the stall pose ≈125 Nm, far under the 400 Nm drive force limit (`Kp=3e5`, `max_effort=400`
  readback), so the drive has ample authority and a gravity droop would be ~4e-4 rad — the miss is
  feedforward-INDEPENDENT (execff 0.1002 with `arm_gravity_ff_loaded_carry` on vs 0.0981 off). The joint
  FREEZES because the scripted side-grip pick physically jams ~40 mm short of the carton (min pad→carton gap
  39.8 mm, 5.3° off flush vs the 3 mm/2° weld gate; `extract_m` 0.306 vs 0.46) — an unmodeled collision, so
  NO FF/stiffness/tolerance on shoulder_lift makes it grasp (a bounded tol = a false-green pick that cannot
  weld). `trajectory_host.classify_goal_miss` now NAMES this ("pick physically blocked: <joint> frozen … not
  gravity sag", + gap/flush when the optional `/restock_sim/arm_controller/pick_surface` feed supplies them)
  instead of the generic goal-miss. The real pick-reach fix is the transfer-fallback lane's (side-grip
  geometry) or `RESTOCK_TRANSFER_MODE=live_pose`. Report `data/sim-robust-executor-shoulder-drive-v1/report.md`,
  tests `tests/test_restock_trajectory_host.py`. The opt-in `arm_gravity_ff_loaded_carry` FF (measured-pose
  robot gravity + held carton via `gravity_ff_effort`/`build_payload_gravity`) stays available, default off ⇒
  byte-identical, but it does NOT close this miss; `tests/test_restock_shoulder_sag_ff.py` now asserts the
  honest "not gravity sag" balance (PR218's fitted `J*Kp≈10 Nm/rad` reproduction was removed).
- **trajectory_host goal-acceptance never silent** (`trajectory_host.py`): `goal()` used to
  `GoalResponse.REJECT` SILENTLY on a latched fault, an already-active goal, or no feedback yet — and
  move_group then aborts the ExecuteTrajectory so the mission surfaced the misleading
  `collision planning or execution failed` (NOT a planning collision). seconddeath's first-collect
  failure was the **feedback-None startup race**: the first goal arrived before the executor's first
  `/joint_states` callback fired (render pacing spaces frames seconds apart WALL). Now every reject logs
  `TRAJ_HOST_GOAL_REJECT <reason> (fault/active/feedback/grace)`; `goal()` WAITS a bounded WALL grace
  (`RESTOCK_FIRST_FEEDBACK_GRACE_S`, default 8 s; 0=legacy) for the first feedback before deciding
  (zero delay when feedback is already present ⇒ byte-identical); a stale `active` flag from a
  terminated handle is cleared; and a `/restock_sim/arm_controller/reset` (auto-published by
  `ros_adapter` `begin`/`begin_joint_goal`) re-arms a LATCHED fault at op start (PR 210 parity — the
  host's permanent latch otherwise bricks the arm against the mission's bounded retries; never clears a
  live goal, no-op when fault-free). The mission names a rejected/aborted goal distinctly via
  `take_exec_abort_reason` instead of the generic collision string. Pure helpers (`reject_reason`,
  `active_is_stale`, `reset_clears`, `first_feedback_grace_s`) + CPU tests in
  `tests/test_restock_trajectory_host.py`. Design + the SEPARATE latent move_group collision-mesh
  URL defect (bare absolute `.stl` paths need `file://`; report-only — enabling move_group arm
  collision geometry may tighten FCL `ValidateSolution`, GPU-validate before fixing):
  `data/sim-robust-pick-plan-rejection-v1/report.md`, proof
  `docs/examples/pick-plan-rejection-proof-fragment.json`.
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
  **Sharp edge: a transfer-start shortcut LATCHES the arm command gate.** The `start_at='transfer'`
  coupling settle holds the arm via `assembly.hold()` → `ArmCommandGate.hold()`, which sets `faulted=True`
  (a terminal e-stop, never self-reset), so a bench's executor/cuMotion/live_pose arm would otherwise sit
  silently dead-gated at the stow pose (no `ARM_CMD_ENVELOPE`; the frozen arm faults the host's trajectory
  reference at 1.3 rad — misread as a clock/self-collision fault). `apply_start_state` calls the new
  `ArmCommandGate.resume()` at the end of that block to re-arm the gate; any new shortcut that holds the arm
  must do the same.
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
  **SIM_ATTACH_FALLBACK weld robustness** (sim-robust-weld-attach-v1): the weld gate is the pure
  `_contact_weld_ready(gap, tilt, hits)`; `SIM_ATTACH_CONTACT_MM` now defaults to a TOLERANT **3 mm**
  (was 0.5 mm, which sat right at the descent endpoint so welding depended on sub-mm jitter — runs
  321/327 bottomed at 0.67 mm and never welded), env `RESTOCK_SIM_ATTACH_CONTACT_MM` (clamped to the
  real cup's 5 mm seal reach), tilt gate unchanged. PR 203's `grasp_press_timeout_s` is now EFFECTIVE
  BY DEFAULT in `ros_host.py` (45 s SIM, env `RESTOCK_GRASP_PRESS_TIMEOUT_S`, config `0` disables).
  Sharp edge: `collect1`/`collectsrc` grasps actually SUCCEED — both runs fail later at the PLACE-on-cart
  step (sibling `sim-robust-collect-place-v1`), not at the grasp. Write-up:
  `data/sim-robust-weld-attach-v1/report.md`.
- **Collect release-detach ack is two-step; tolerate a lost move_group scene** (`restock_manipulation/ros_payload.py`):
  the detach does a move_group `ApplyPlanningScene` REMOVE **then** the AUTHORITATIVE object_attachment
  `AttachObject` detach (clears the cuMotion spheres the planner uses, separate container). move_group is
  `respawn=True` and has an intermittent SIGABRT double-free (run-266/268); if it crashes+respawns BETWEEN
  the grasp attach and the release detach (coldlive: live_pose + the arm masked-depth feed loaded it enough
  to trip it; coldhonest never crashed), the fresh scene never held `restock_carried_unit` so the REMOVE
  returns `success=False` and the single-shot handshake used to abort the whole collect
  (`gate=scene_apply_unsuccessful` → `payload collision detachment not acknowledged`). `detach_scene_gate`
  (default on; `RESTOCK_DETACH_SCENE_TOLERANT=0` reproduces the abort) treats an unsuccessful REMOVE on a
  DETACH as already-removed and PROCEEDS to the authoritative action, which decides the ack; an ATTACH, or
  a genuinely failing authoritative detach, still fails fast by name. A successful REMOVE (coldhonest) is
  byte-identical. CPU proof `tests/test_restock_detach_ack.py` + the release-phase contract in
  `tests/test_restock_manipulation.py`; bounded GPU proof `docs/examples/collect-release-detach-ack-fragment.json`.
- **move_group mesh URIs** (opt-in HOST key `moveit_mesh_uris`, default off ⇒ byte-identical):
  the resolved URDF the host ships (`host-cad/mobile.urdf`) carries BARE ABSOLUTE `<mesh>` paths,
  which MoveIt's `resource_retriever` rejects (`URL rejected: No host part in the URL`), so
  move_group loads NO arm link geometry and FCL `ValidateSolution` runs on empty geometry (the
  cuMotion XRDF spheres do the real checking). `restock_host.launch` rewrites them to `file://`
  URIs when the key is on (`restock_host/mesh_uri.py`; registered HOST in `config_preflight`).
  Loading real (tighter-than-sphere) link meshes only makes self-collision MORE permissive; the
  one newly-checked family is the ATTACHED carton vs non-touch arm links (collect place /
  transfer preplace). CPU proof `scripts/check_movegroup_mesh_uris.py` (cross-checks the real
  retriever when ROS is sourced); write-up + FCL risk + GPU fragment:
  `data/sim-robust-movegroup-collision-meshes-v1/report.md`,
  `docs/examples/movegroup-mesh-uris-mission-fragment.json`.
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
  **Stage-scoped teardown vs a later camera reader** (`RESTOCK_STAGE_SCOPED_SENSING`): the fm381
  teardown DESTROYS (not pauses) the "unused" shelf camera at the pick lift to free VRAM, but a
  camera-mode transfer or the verify-stage `InventoryVerifier` reads the shelf camera AFTER that →
  the post-place verify read an expired prim (`Accessed invalid expired 'Camera' prim`). The teardown
  now keeps the shelf camera alive whenever `sensing_sources.shelf_camera_needed(active_stages, ...)`
  says an executed stage still reads it (`isaac_bundle` computes `keep_shelf_camera` from the config);
  collect-only / ground-truth-verify missions free it unchanged. A destroyed-then-read camera now
  raises the named `native_cameras.CameraRenderProductReleased`, not Isaac's raw expired-prim trace.
  Report: `data/sim-robust-verify-camera-prim-v1/report.md`.
- **Arm masked-depth feed economy** (opt-in, default-off): `masked_depth.economy`
  (`restock/masked_depth_economy.py`) lowers the live_pose arm feed's render/per-tick cost —
  `resolution_scale` (scales the arm camera product; apertures unchanged ⇒ K scales ⇒ FOV/reprojection
  preserved; quantization held under half an ESDF voxel at the recorded distances) and `publish_interval_s`
  (SIM-clock rate cap on `ArmDepthSubprocessBridge.publish`, skip-never-stale, validated below the 0.5 s
  nvblox window). Absent ⇒ byte-identical. NOT the dominant pacing lever — the shared render cadence
  (`record_stride`/`sensing_stride`/`sim_speed`) is; coldhonest completed the cold cycle WITH the arm feed
  by widening those (measured: arm-on vs arm-off pace identically during collect). Example
  `docs/examples/masked-depth-economy-mission-fragment.json`; CPU `tests/test_masked_depth_economy.py`;
  measurement + design `data/sim-robust-arm-feed-economy-v1/report.md`.
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
