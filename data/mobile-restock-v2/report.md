# Mobile restocking integration checkpoint

Status: incomplete but materially advanced. Guarded GPU use was authorized on October 13;
native RTX RGB-D rendering, VFR recording, stationary hold, coupling and forward towing all
pass. **The native collision-map incorporation pipeline is now GREEN end to end**: the
integration sensor+handshake driver drives EmptyGeneration→OpenAdmission→register→integrate→
EvidenceSnapshot→consume and `TemporalMap.accept_envelope` adopts the slice, yielding a
measured incorporation identity (`incorp-identity-03`, under the host producer fixes PR#32/#33).
The `isaac_bundle:create` mission factory is built + CPU-tested, and the `manip_factory` is
authored + CPU-tested (515 CPU tests). No autonomous moving-mission MP4 has been produced yet:
the full recorded mission is gated on the lane-owned `nav_factory`/`sensing_factory` refs; shelf
support decode stays RED (native-render marker visibility, routed to the camera worker). The
landed mobile/cart, schema-2 environment, articulated navigation and CAD manipulation components
are integrated from origin/main.

## Mission gate table (guarded native + CPU evidence)

Each row is a mission prerequisite. "Native" rows are bounded guarded-GPU runs
(peak <= 6.6 GiB VRAM, >= 15.6 GiB host RAM, `/tmp/aile-gpu.lock`, whispr preserved via
`--heavy-threshold-mb 2500`, services unchanged); "CPU" rows are deterministic offline
checks. Detail + exact commands are in the per-topic sections below.

| Gate | Status | Kind | Evidence |
|---|---|---|---|
| Stationary cart/base hold | GREEN | native | `native-assembly-stabilized-04`: cart net 0.000 m/s, base 0.00079 m/s over 2 s |
| Cart coupling (dock + latch + hold) | GREEN | native | `native-coupling-02`: hitch from fresh/stopped/aligned dock, 120-step hold, 0.18 mm residual, 0.045 deg articulation |
| Forward towing (through portal) | GREEN | native | `native-towing-02`: towed 0.378 m to portal centre, 0.39 deg articulation, no wheel lift, 270-frame VFR H.264 (`native-towing.mp4`) |
| Host self-collision model | GREEN | CPU | `host-preflight-01`: 1930 spheres, zero transport overlaps, 7 separation proofs |
| Navigation route preflight | GREEN | CPU fixture | `nav-preflight-01`: outbound + return succeed at 0.10 m uncertainty, peak ~46 deg; not a physics demo |
| Source-station sensing (decode) | GREEN | native | `view-source-lit-03` under merged PR#37: 3/3 decoded (juice_orange + support:source:0/1); the PR#31 emissive regression is fixed |
| Shelf-station sensing (decode) | PARTIAL 1/2 (production decoder, native) | native | `view-shelf-pr45-prod` under merged camera PR#45 (shelf camera tilted 6.25° about world-X, far marker grid rotated 180°) with the bench now using the **production** `FixedTileQR` (not full-frame cv2): `support:shelf:0` decodes on all 6 frames (~0.30 s, < 500 ms), `support:shelf:1` does not → **1/2**. PR#45's offline raster asserted both decode; the native RTX render reads only shelf:0 (clean raster necessary, not sufficient — the risk the camera worker flagged). Production decode improved 0/2 → 1/2 natively; shelf:1 native decode remains open (camera worker) |
| Mapper construction | CONFIRMED | native | `mapper-evidence-construct-01` (PR#29 runtime closure): EvidenceProducerNode instantiated + advertised services/topic + 5893 ticks, GPU reclaimed; incorporation = sensor+handshake next |
| Collision-map incorporation (mapper) | GREEN | native | `incorp-identity-03` under merged PR#33: full path EmptyGeneration ACCEPT → admission → 2 frames registered+integrated → EvidenceSnapshot (frame_id='map') → `accept_envelope` ADOPTED. Measured identity: gen mission_incorp_e, nonce+certificate echoed, index 1, rev 1, front→front_optical, slice_adopted true, full_height_certified false |
| Full autonomous restock cycle | BOOTS CLEAN, BLOCKED at transfer-manip construction | native+gated | runs 05-07 (guarded, env_isaaclab, --host-stack): GPU guard, host MoveIt/cuMotion warmup (6/7), Isaac boot, ROS2 bridge **'rclpy loaded'**, full scene composition and arm-camera authoring all PASS. Two defects found + fixed en route: Isaac-ROS-bridge env (system /opt/ros rclpy shadowed the bundled Jazzy one) and the arm camera (attached to a body frame, not a Camera prim). Then blocks: `ModuleNotFoundError: No module named 'curobo'` — `isaac_factory`→`CadRestockRuntime`→`build_cad_planner` imports shelfbot's IN-PROCESS cuRobo planner, which is not installed; the provisioned planner is the HOST cuMotion ROS stack. Factory raises before `mission.start()`, so no stages tick / no MP4 this attempt. See "Full assembled mission run (d)" below |

Source-station note: per the integration/camera contract the 3/3 native decode
(`view-source-lit-01`) is the accepted source-station sensing proof. The pre-capture
marker metric-fit registration (`restock_camera.sensed.register_marker`) is deferred
into the mission-bundle station sensing, where `base_from_camera`, `FrameEvidence` and
the support/auxiliary layout are wired from the camera config in one place, rather than
hand-assembled against camera-owned internals for a one-off run; the native vacancy
capture it feeds requires the mapper incorporation gate above regardless.

## Full assembled mission run (d): boots clean through every infrastructure gate, blocks at transfer-manipulation (measured)

The full assembled mission was attempted as one bounded guarded native run with the
host planner stack co-launched (`run_mobile_restock.sh --execute --host-stack`, env_isaaclab,
task-local NVIDIA runtime, domain 88, `--heavy-threshold-mb 2500`). It now boots cleanly
through **every** infrastructure gate and stops at a single, specific, measured blocker.

Measured progression (runs 05-07; console logs + `run-06/07/mission-error.txt` under the
task scratch dir):

1. **GPU guard** passes (>= 8 GiB VRAM free) once orphaned co-launch host-stack GPU
   processes are reclaimed. (Teardown note below.)
2. **Host MoveIt/cuMotion stack** warms up: 6/7 servers advertise; the `robot_segmenter`
   NITROS-allocator failure is the documented-expected one.
3. **Isaac Sim** boots and composes the full scene (store + mobile UR10 + rollwagen +
   gripper + physics).
4. **ROS2 bridge**: `rclpy loaded` — the bundled Jazzy rclpy initializes.
5. **Arm camera** is authored and initialized.
6. **Transfer-stage manipulation construction** raises
   `ModuleNotFoundError: No module named 'curobo'`.

Two integration defects were found and fixed to reach the blocker:

- **Isaac ROS2 bridge env.** The bridge first failed with
  `No module named 'rclpy._rclpy_pybind11'`: a login shell sources the system
  `/opt/ros/jazzy` overlay, putting its python3.12 `site-packages` on `PYTHONPATH` and
  `/opt/ros/jazzy` on `AMENT_PREFIX_PATH`, so Isaac imported that (wrong-ABI) rclpy instead
  of its bundled one. The fix lives in `scripts/run_mobile_restock.sh`: on `--execute` it
  clears the ROS prefix vars, pins `PYTHONPATH` to the scripts dir, selects
  `rmw_fastrtps_cpp`, and appends the bundled `isaacsim.ros2.bridge/jazzy/lib` to
  `LD_LIBRARY_PATH`. This must be set at process creation (the ELF loader and rclpy
  resolver are fixed before Python starts). Result: runs 05/06/07 log `rclpy loaded`.
- **Arm camera prim.** `twin/restock_manipulation/isaac_factory.py` constructed an Isaac
  `Camera(prim_path=paths['front_camera_frame'])`, but `front_camera_frame` is a URDF body
  frame (an Xform, `rpy="0 0 0"` off `base_link`), not a USD Camera, so Isaac raised
  "prim path does not correspond to a Camera prim." The factory now authors a child USD
  `Camera` under that frame, oriented so the USD camera (looks -Z, +Y up) faces forward
  along the frame's +X with +Z up. Run-07 clears this gate.

**Remaining blocker (measured).** `isaac_factory` → `CadRestockRuntime` →
`isaac_motion.build_cad_planner` imports `shelfbot.planning.curobo_planner`, which imports
`curobo` (NVIDIA cuRobo). `curobo` is not installed in any environment on this machine; the
only cuMotion present is the **host** stack's ROS2 `isaac_ros_cumotion` packages (MoveGroup
on domain 88), consumed by `twin/restock_manipulation/ros_adapter.py:RosMotionPort` in host
Python (it needs `moveit_msgs`, which the Isaac interpreter does not carry). So the
in-process cuRobo path that `CadRestockRuntime` uses is the retired manipulation lane's
mechanism, not the planning capability this environment actually provisions. Because the
factory raises before `mission.start()`, no mission stages tick and no mission MP4 is
recorded in this attempt. Wiring the Isaac mission's manipulation to the host cuMotion path
(or installing cuRobo into the Isaac interpreter) is the open integration decision; it
touches the retired manipulation lane's runtime (`isaac_runtime.py`/`isaac_motion.py`).

`run_mobile_restock.py` now persists the construction traceback to
`<output>/mission-error.txt` before `app.close()`, because Isaac's `SimulationApp.close()`
can terminate the process hard and otherwise the outer `finally` (which writes
`mission.json`) never runs and the traceback is lost.

**Shared-environment teardown note.** The co-launched host stack's
`run_restock_host.py --launch` calls `setsid`, so it escapes `run_mobile_restock.py`'s
process-group teardown and leaves orphaned GPU-holding `component_container_mt` processes
that block the next boot. They must be reclaimed by targeting the actual `/dev/nvidia*`
holders only — never by matching `run_restock_host.py` on the command line, because the
CPU-only host worker runs an identically-named process concurrently that must not be
touched. A robust in-run teardown remains to be implemented.

## Mapper launch check: nvblox inits under guards, node load blocked by missing host package

With nav PR #23 merged (mapper launch code + typed `restock_mapper_evidence` consumer),
a bounded guarded native launch check (`mapper-check/run01`, domain 88) was run:
`launch_navigation_mapper.py --execute` under the GPU lock with the host nav-plugin env
plus the task-local GPU userspace. Measured: the `component_container_mt` started, the
nvblox node printed its full static-TSDF config and reached `subscribeToTopics()`, and
GXF `std`+`cuda` extensions loaded with **no CUDA driver error** (so the task-local GPU
userspace works for nvblox; GPU peaked 3059 MiB and returned to the 2258 MiB whispr
baseline, process torn down cleanly). The nvblox node constructor then threw: **package
`gxf_isaac_gxf_helpers` not found** in the host plugin `AMENT_PREFIX_PATH`. Consequently
the node did not load — `static_map_slice` was not advertised and no `EvidenceSnapshot`
was produced. This is a host plugin-environment completeness gap (a missing Isaac GXF
dependency), distinct from dependency (b), the still-unimplemented native producer
incorporation binding (host PR #25). The CPU consumer tests (`test_navigation_evidence`,
`test_navigation_temporal`, `test_navigation_mapper_launch`) pass. This is a measured
launch result, not a readiness claim: slice publication and incorporation stay gated on
restoring the host GXF package and landing (b).

## Incorporation driver built + run: producer emptiness gate unsatisfiable at construction (measured)

The integration **sensor + handshake driver** that was missing (the mapper gate only
proved node construction) is now built, CPU-tested and exercised natively under the GPU
guard. Three parts, all integration/launch-owned:

- `scripts/twin/restock_host/incorporation_driver.py` — a pure-logic handshake
  **sequencer** mirroring `native/restock_mapper_evidence/state.hpp`: empty-certificate
  echo, single admission, acquisitions strictly after the cutoff and fresh in both clocks
  (0.3 s), strictly increasing per sensor. No ROS/GPU; deterministic under supplied clocks.
- `scripts/twin/restock_host/sensor_handshake.py` — a ROS 2 node that publishes `/clock`,
  the static `map->optical/base_link` TF and a synthetic depth + camera_info on the
  generation input topics, drives `EmptyGeneration -> OpenAdmission -> RegisterAcquisition`,
  then decodes the `EvidenceSnapshot` via `evidence_bridge` and feeds
  `twin.navigation.temporal.TemporalMap`, recording the adopted incorporation identity.
  The synthetic depth is a FIXTURE source (not Isaac perception); the Isaac depth source is
  a drop-in seam. A `--probe-empty-count` diagnostic mode polls EmptyGeneration only.
- `scripts/run_incorporation_identity_gpu.py` — guarded orchestrator launching the mapper
  (`launch_evidence_mapper.py --execute`) + the driver under `/tmp/aile-gpu.lock`, >=8 GiB
  VRAM / >=12 GiB RAM preflight, `--heavy-threshold-mb 2500`, bounded, torn down by PID group.

CPU: 13 new tests (`test_incorporation_driver`) cover the sequencer invariants and the
consumer float-identity the `accept_envelope` `ack in history` check depends on; 490 CPU
tests pass, 10 dependency skips.

Native (6 bounded guarded runs, `data/.../incorp-*`, GPU peaked ~0.8 GiB over the 2258 MiB
whispr baseline, reclaimed to 9650 MiB free, whispr preserved, lock released each time):

- **Service path works end to end** — a real advance over `construct-01`, whose external
  probe false-negatived node discovery because it lacked the sourced overlay. Here the
  driver sources the PR#29 runtime closure (`sim-restock-host-v2/nav-plugin/runtime-env.sh`)
  so it discovers and successfully CALLS `empty_generation`/`open_admission`/
  `register_acquisition`; the mapper responds with typed replies.
- **EmptyGeneration is refused `native layer not empty`** on the earliest single-shot call
  (`incorp-probe-03`, ~3 s after node construction) with **nothing published** — no `/clock`,
  no TF, no depth — and again (`incorp-probe-noesdf2`) with the ESDF-update, decay and
  map-clearing timers disabled (`--mapper-param update_esdf_rate_hz=0.0 decay_tsdf_rate_hz=0.0
  map_clearing_radius_m=-1.0`). So a static nvblox layer carries >=1 allocated block from
  construction, independent of sensor input and of every timer the driver or launch controls.
- **The producer is one-shot**: `State::fail()` is `[[noreturn]]` and sets `Phase::Fault`, so
  the first refusal poisons the generation — subsequent EmptyGeneration calls return
  `invalid empty challenge` (`incorp-probe-02`: 1 x `native layer not empty` then 19 x
  `invalid empty challenge`). A restart is a new producer instance that hits the same gate.

Consequence: the producer's `verify_empty` (all of tsdf/esdf/color/freespace/occupancy
`numAllocatedBlocks == 0`) is not satisfiable against the installed nvblox static mapper as
launched. The embedded slice is emitted only from `processEsdf`, so the gate cannot simply be
bypassed by disabling ESDF. This blocks incorporation identity -> source-station vacancy
capture -> the full mission. It is a **native-producer-lane** issue (`native/restock_mapper_evidence`,
host PR#25), not editable by integration concurrently. Minimal asks routed to that lane:
(1) return the per-layer `LayerCount[]` in the EmptyGeneration **failure** response (currently
populated only on success) so the exact non-empty layer/count is pinned without faulting the
node; (2) make `verify_empty` baseline-aware of nvblox's construction-time allocation (measure
the construction baseline, or exclude derived/preallocated layers from the sensor-emptiness
check). The driver, sequencer, orchestrator and consumer path are complete and will produce a
measured incorporation identity the moment the gate accepts an empty generation.

**UPDATE (both asks landed, host PR#32; pipeline now runs end to end).** PR#32 made
`verify_empty` construction-baseline-aware (a fresh map reports 2048 preallocated blocks/layer;
emptiness means zero sensor-observed growth over baseline) and made emptiness refusals
non-faulting with `LayerCount[]` on the failure response. Re-running the driver against it
(`incorp-identity-01/02`, bounded guarded GPU) the **full native path now succeeds**:
EmptyGeneration ACCEPTED on a fresh construction (certificate echoed, baseline 2048/layer) →
OpenAdmission opened → 2 synthetic depth acquisitions REGISTERED and INTEGRATED (nvblox ESDF
revision advanced) → the mapper **PUBLISHED an EvidenceSnapshot** (snapshot_index 1,
esdf_revision 1, admission OPEN, empty_state POPULATED, ledger `front`/`front_optical` seq 1,
slice 48×80 @0.05 m, `data_len 3840 == width*height`). The consumer decode then fails closed
with `invalid native distance slice` for one measured reason: the producer's `processEsdf`
fills `env->slice` via `distanceMapSliceMsgFromSliceImage` but **never sets
`env->slice.header.frame_id`** (measured `frame_id=""`), while the stock nvblox publish path sets
it to the global frame and the consumer requires the slice frame be `map`. This is a one-line
host-lane fix (`native/restock_mapper_evidence/src/producer_node.cpp` `processEsdf`: set
`env->slice.header.frame_id = "map"` — the mapper's `global_frame` — before publish; optionally
stamp the header too). Integration does not paper over it (an empty slice frame is correctly a
fail-closed envelope). A measured incorporation identity, then source-station vacancy capture and
the full mission, follow immediately once the producer stamps the slice frame.

## Mission bundle factory `isaac_bundle:create` built (CPU-tested, unrun — gated)

`run_mobile_restock.py --execute --factory twin.restock.isaac_bundle:create` now has a real
factory (`scripts/twin/restock/isaac_bundle.py`). It owns the **integration assembly only** and
fabricates no sensor data:

- `plan_bundle(manifest, scenario, config)` — pure, Isaac-free, fail-closed: derives and validates
  the four mission navigation poses (`pickup`/`cart_dock`/`shelf_approach`/`home`), the per-action
  articulation tolerances (`outbound`/`return`, 0 < tol ≤ 65°), the navigation mode
  (`reference`/`nav2`/`sensor`, defaulting to `sensor` for an `articulated` scenario), and the
  verification/recording parameters. Poses are **required explicit runtime-config inputs**, never
  guessed from staging geometry. 11 CPU tests.
- `create(app, manifest, scenario, config, output)` — calls the validated
  `physics_setup.assemble` scene builder (the same one the coupling/towing/assembly native
  diagnostics use), constructs the measured `IsaacPort` (wheel-encoder odometry + idealized range
  scan + native contact + rigid-body cart/base proprioception + rendered inventory verification +
  H.264 recorder), and wraps the real sub-runtimes in the CPU-tested `RuntimeAdapter`
  (`fixture_mode=False`, so the launcher's production check accepts it). Mission actions map to the
  adapter's existing collect→couple→outbound→park→transfer→verify→return logic, each gated on
  measured sensor evidence.
- The per-lane **controllers stay lane-owned** and are resolved from the runtime-config as
  `module:function` refs — `nav_factory` (navigation), `manip_factory` (manipulation),
  `sensing_factory` (camera/inventory sensing) — each called `fn(app, assembly, plan, config,
  output)`. A missing/invalid ref fails closed naming the exact contract; no stub is substituted.
  Source-station sensing capture is wired through `sensing_factory` into `InventoryVerifier`.

Status: **built and CPU-tested, NOT run.** The full physical mission is gated on (a) the one-line
producer slice `frame_id` fix (`incorp-slice-frame`) so the incorporated map is adoptable, and
(b) the lane factory refs landing. When both land, one bounded guarded run drives the full cycle
and records it as VFR H.264, with any failed stage reported as failed. 502 CPU tests pass.

## Camera lighting applied: SOURCE decodes 3/3, SHELF still dark 0/2 (measured)

The camera worker's store light rig (PR #26, `twin.restock_camera.lights`) landed on
origin/main and was merged and consumed in the native composition per
`data/sim-restock-camera-v2/lighting-handoff.md`. `view_scene` now authors the designed
rig (72 store `RectLight`s at correct nits + a `DomeLight` fill) and **disables the 73
dim exported store lights** (authored at `power/pi` ~14.3, ~200x too dim in RTX nits -
the original dark-capture cause) to avoid doubling. For the shelf view a task fill is
co-located with the shelf camera; its eye/target derived from the camera config
(-3.55, 4.826, 0.96 -> -3.29, 4.826, 0.73) match the handoff. RTX auto-exposure is
already off; the single knob `intensity_scale` is 1.0.

Measured native decode (6 frames each, 2560x1920):

- **Source view: PASSED 3/3** markers decoded (`juice_orange`, `support:source:0`,
  `support:source:1`), up from 1/6 before lighting. The rig fixes the backroom-lit views.
- **Shelf view: FAILED 0/2** (`support:shelf:0/1`). The recessed ~0.24 m shelf slot stays
  in deep shadow even with the co-located task fill and dome: the central 800x800 support
  region renders **max luminance 4/255 (median 2)** and **0 %** of pixels reach the
  ~180/255 a QR white module needs; the two marker instance-id regions peak at **0** and
  **21/255**. Pixel extent is ample (marker bbox ~500-800 px, ~20-32 px/module for 25
  modules), so the limiter is **illuminance/contrast, not resolution**. Guards: peak
  6636 MiB GPU, services unchanged.

Per the integration/camera contract and firstmate's instruction, this is reported as a
measured illuminance/pixel-size result and **routed back to the camera worker** rather
than blind-tuned: the ceiling panels have no line of sight into the slot and the
handoff's grazing-only `keep_fill` fill, applied as specified, does not lift the slot to
decodable contrast. The source-view decode confirms the rig itself is correct where line
of sight exists. Evidence:
`~/datasets/twin-robot-in-store-boot-v1/view-source-lit-01/` and `view-shelf-lit-01/`.

Camera ring lamp (PR #27) follow-up — `view-shelf-lit-02`: `view_scene` now consumes
`config['lamps']` via `author_camera_lamps` (the shelf `SphereLight` ring lamp, 101321
nits at (-3.58, 4.826, 1.05), sharing `intensity_scale`). The lamp is authored active and
correct, but the shelf support **still renders 0/255 (0/2 decode)** while the rest of the
frame is lit (whole-frame max 243). Two native causes observed (not blind-tuned): the
opaque `shelf_lamp_housing` cube is co-located at the **same centre** as the lamp and
occludes its emission toward the support, and the lamp at z=1.05 sits **above** the
~0.97 m shelf tier the camera (z=0.96) sees beneath, so its light cannot enter the slot.
This native render contradicts the offline 2088-lux proof (which did not model the housing
as an opaque occluder / the above-tier height). Routed to the camera worker; the same
wiring will decode once the lamp clears its housing / drops below the tier. Evidence:
`~/datasets/twin-robot-in-store-boot-v1/view-shelf-lit-02/`.

## Host/collision preflight: COLLISION GATE GREEN (CPU, vs origin/main)

The fourth lighting-independent priority-3 step passed on CPU. The strict host
collision preflight (`python -m twin.restock_host.generate_assets`, `restock_host`
byte-identical to current origin/main) regenerates the refined mobile CAD sphere model
and checks transport self-collision: **1930 spheres, `transport_collision_check_passed
= True` (zero non-excluded transport overlaps), 7 full-range separation proofs** and
sound kinematic exclusions (same-rigid / adjacent / proven-separation). Evidence:
`~/datasets/twin-robot-in-store-boot-v1/host-preflight-01/asset-report.json`.

`planning_ready` and `native_gpu_capacity_verified` remain `False` by design: offline
asset generation deliberately does not certify the native cuMotion planner capacity,
batching or trajectory/dynamics - that is the remaining planner/sensing gate, not a
collision-model failure. With the collision gate confirmed green and the camera light
rig now landed on origin/main, the remaining mission gates are **lighting and sensing**.

## Forward towing: TRAIN TOWS THROUGH THE PORTAL on native guarded GPU

The third lighting-independent priority-3 step passed. `native-towing-02` docks the
train at a portal-aligned pose, then commands a single constant forward crawl straight
down the sales/backroom portal lane (x = -3.875, the home/portal corridor). The lead
robot tows the coupled cart forward 0.378 m, reaching the portal centre (start
y = 11.500 -> end y = 11.122, portal centre y = 11.125). Throughout, the cart follows
through the physical hitch with **max articulation 0.39 deg** (well within the 46 deg
limit) and **no cart wheel lift** (max wheel-centre height 0.0770 m vs the 0.075 m
resting height - a 2 mm margin, under the 10 mm lift tolerance). The motion is recorded
as a 270-frame **variable-frame-rate H.264** clip (`native-towing.mp4`) from a fixed map
camera, with the same acquisition/encoding provenance as the render smoke (unique render
IDs, strictly increasing acquisition timestamps, every presentation timestamp decoded
and checked). Guards held: peak 4890 MiB GPU, 15630 MiB host RAM floor, services
unchanged.

Honest limits: the wheel drive is traction/force-limited, so the achieved crawl was
about 0.045 m/s despite a 0.35 m/s command - enough to prove the towing dynamics and
reach the portal plane, but a full multi-metre traverse, turning, and the jackknife
guard under steering are navigation-lane behaviour not shown here; no planner, steering
or sensing is exercised and a single forward velocity is commanded. The native validator
is `scripts/twin/restock/towing_check.py`. Reproduce (guarded):
`... run_restock_render_smoke.py --towing-only --runtime-config <towing config> ...`.
Evidence: `~/datasets/twin-robot-in-store-boot-v1/native-towing-02/`
(`native-result.json`, `towing-feedback.json`, `native-towing.mp4`, `run.json`).

## Cart coupling: LATCH HOLDS on native guarded GPU

The second lighting-independent priority-3 step passed. `native-coupling-02` settles
the uncoupled robot+cart at the authored docking pose (150 steps), then requests the
physical hitch through the same `MobileController`/`Coupler` contract the mission uses:
the latch is created only from a fresh, stopped, aligned observation (measured hitch
error 9.6 mm < the 25 mm gate, hitch-Z difference 0.06 mm, relative yaw ~0, reported
speeds within the 20 mm/s stationary bound). Over the following 120 coupled hold steps
the hitch stays present and healthy with **max hitch residual 0.18 mm** and **max
articulation 0.045 deg**, and the coupled train is stationary by measured displacement
(cart net 1.9e-5 m/s, base 1.1e-4 m/s). Guards held: peak 4832 MiB GPU, 16494 MiB host
RAM floor, services unchanged. This proves docking and a holding physical coupling only
- no towing, navigation, planner or sensing is exercised; drive targets stay zero.

Reproduce (guarded): `... run_restock_render_smoke.py --coupling-only --runtime-config
<config> ...`. The native validator is `scripts/twin/restock/coupling_check.py`; the
shared measured-displacement settle metric is `scripts/twin/restock/settle_metrics.py`
(CPU-tested). Evidence:
`~/datasets/twin-robot-in-store-boot-v1/native-coupling-02/` (`native-result.json`,
`coupling-feedback.json`, `run.json`).

## Stationary cart/base: SETTLED on native guarded GPU (a11510f)

The uncoupled cart no longer drifts out of the stationary native assembly diagnostic;
it is now provably motionless. Run `native-assembly-stabilized-04` completed all 120
steps (2 s) and PASSED: measured net cart displacement **0.000 m/s / 0.000 rad/s** and
base **0.00079 m/s / 0.00019 rad/s** over the final 0.5 s window. Guards held: peak
4764 MiB GPU, 16199 MiB host RAM floor, 7144 MiB GPU free, `/tmp/aile-gpu.lock`,
protected whispr accepted via the `--heavy-threshold-mb 2500` opt-in, and all live
services unchanged. This is a stationary physics hold, not an autonomous mission:
no coupling, planner, navigation, sensing or wheel motion is exercised.

Root cause and fix, established one offline hypothesis then one bounded native run
at a time:

- The backroom floor collider top is exactly z=0 and the wheels bottom at z=0, so
  there is no geometric initial penetration. The failure was unstabilized resting
  contact (a persistent ~55-70 mm/s reported +z velocity with ~0 z displacement)
  rolled sideways by the cart's frictionless casters.
- `enableStabilization` was authored on the scene but **Isaac's PhysicsContext
  re-applies its `False` default at World construction**, silently clobbering the
  USD value. Re-asserting it through `PhysicsContext.enable_stablization(True)`
  before the first reset (the only path PhysX parses) is what actually engages it,
  and alone cut drift from 50.6 mm to 11 mm (`native-assembly-stabilized-02`).
- `physxJoint:jointFriction` is inert here: the schema restricts it to articulation
  joints and the casters are maximal-coordinate (`excludeFromArticulation`). Bearing
  and rolling resistance are therefore authored as rigid-body damping instead -
  strong angular damping dominated by the low-inertia wheels (negligible tow torque)
  plus mild linear damping on the 35 kg cage - which froze the cart's measured
  position completely (`native-assembly-stabilized-03/04`).
- The stationary verdict is gated on **actual net measured pose displacement** over
  the final 0.5 s, not the TGS-reported instantaneous velocity. PhysX TGS reports a
  velocity carrying a position/velocity solve bias that does not integrate to
  displacement (the ~14 mm/s the cart reports while its measured position is frozen);
  that biased value is retained in the result as a transparency diagnostic, not a
  gate. Mass, inertia, collision geometry, gravity and the free revolute model are
  untouched; the damping is a fidelity correction, not a positional brake.

Reproduce (guarded): `PYTHONPATH=scripts python3 scripts/run_restock_render_smoke.py
--assembly-only --runtime-config <config> --output <new dir> --ffmpeg <ffmpeg>
--isaac-python <env_isaaclab python> --nvidia-runtime <matching runtime>
--heavy-threshold-mb 2500`. CPU invariants are covered in
`scripts/twin/tests/test_restock_physics_port.py` (391 CPU tests, 16 subtests pass).
Evidence: `~/datasets/twin-robot-in-store-boot-v1/native-assembly-stabilized-04/`
(`native-result.json`, `stationary-feedback.json`, `run.json`). Towing-phase damping
values may need revisiting once the coupled mission is exercised.

## Latest passive-observer and CPU checkpoint

The complete offline reduction finds nonzero reported cart contacts only at the
four wheels on Backroom_floor; lower-floor and hitch/drawbar impulses are zero.
Rolling joints show the largest anchor-velocity disagreement. PhysX documents
different position/velocity solve biases, so reported velocities need not integrate
to body displacements. The exact TGS >4 velocity-iteration warning is present.
One targeted64/1 cart solver control is proposed, not executed; original run03's
16/1 failure remains disconfirming evidence against blaming excessive velocity
iterations alone. Full per-body/contact/joint tables, CSV paths and vendor sources
are retained in the diagnosis below. GPU ownership remains released.

At65bd36e,390 CPU tests and16 subtests pass, with five optional skips, including the
confirmed merged camera latency update. Native latency still has no commissioning
proof; reported offline processing includes500ms overruns and the strict rejection
gate remains active.

The independent audit found that late diagnostic body wrappers altered state in
the earlier matched pair. The separately authorized passive baseline at5cc2d00
replaced only that read path, proved equality of exposed authored/physical state
on attachment without stepping, and still failed50mm at step119/time2.033333s.
Measured drift was50.0166mm. Raw tensor pose/velocity disagreement persists.
No cart, floor, mass, friction or backend fix is established. All historical failures
remain available with their limitations; exact audit paths are in the diagnosis.

This run used CPU PhysX(TGS60Hz), with GPU RTX rendering. It took53.5s, peaked at
4745MiB GPU, retained16042MiB host RAM, and preserved services. GPU ownership was
released; no further run was made. Navigation producer-side acquisition/generation
binding and a native empty-map acknowledgement remain unproven.

## Earlier matched-pair and CPU checkpoint

At a446380, 380 CPU tests and 16 subtests pass, with five optional skips. The
authorized native contact pair at d2441bb failed in both arms: baseline 50.269 mm
drift at 1.550 s; lower-ground-disabled variant 50.788 mm at 1.683 s. Removing
lower-ground contacts did not resolve drift. Those baseline contacts reported zero
impulse. This disconfirms the lower floor as a necessary cause in this setup;
cart stability remains unresolved. No production floor patch was made.

The pair is complete and GPU ownership was released. Both runs peaked at
4745 MiB GPU; minimum RAM was 15225/15075 MiB, and services were unchanged.
Pristine authored states, measured initial baselines, native per-body feedback,
joint residuals and actual contact provenance are retained for independent audit.
The new contact-pair plot is actual telemetry; all videos remain earlier optical
checks, not an autonomous mission recording.

The merged camera component is now integrated. Fresh monotonic readings enforce
the unchanged 500 ms acquisition-to-use deadline after processing and sequential
aggregation, before motion, and on queued native gripper dispatch. CPU tests
explicitly reject delayed processing and handoff. Native latency and the full
sensor/map/planner/actuator path remain uncommissioned. No additional GPU run is
authorized by this checkpoint.

## Earlier stationary checkpoint: FAILED, cause unresolved

At e9acc7f the CPU checkpoint is 369 passed, 16 subtests and five optional skips.
Native assembly-03 crossed the unchanged 50 mm stationary envelope at simulation
time 1.266667 s; assembly-04 with 64/16 cart solver iterations crossed it at
2.033333 s. These are failed stationary diagnostics, not a demonstrated mission.
Higher iterations did not establish or fix the cause. The full offline diagnosis
is retained below and in `data/mobile-restock-v2/cart-stability-diagnosis.md`.
The embedded comparison plot uses actual saved telemetry, not animated motion.
The videos remain the earlier optical/startup evidence from report entry2457.

Camera PR https://github.com/ailerobotics/blender-supermarket/pull/22 was reported
merged by firstmate; this e9acc7f diagnosis does not claim its code has been merged
into the integration branch. Camera runtime remains uncommissioned: a fresh clock
must enforce the unchanged 500 ms acquisition-to-use limit after all processing and
before action. A delayed-processing rejection test is still required. Navigation's
assigned mapper namespace is `/restock_sim/navigation`, with domain88 and stable
map frame; the separate manipulation mapper and native plugin build are not ready.

## Existing CPU integration

`scripts/twin/restock/inventory.py` provides SQLite reservations, persistent action
history, atomic source-to-shelf stock transfer and observation deduplication.
Interrupted/failed jobs retain stock reservations and cannot automatically replay.
There is deliberately no automatic recovery/release operation: physical stock and
actuator state must be reconciled before adding one.

`scripts/twin/restock/mission.py` drives collect, couple, outbound, park, transfer,
verify and return through a small adapter protocol. Failure, invalid evidence and
action timeout request stop. Shelf credit requires a matching observed transfer;
a failed return preserves previously verified shelf stock. Simulation timestamps
are relative to mission start. Adapter implementers must convert their sensor
clock to that origin. Adapters remain responsible for sensor provenance and
collision/control checks: labeling a dictionary `sensor` does not prove autonomy.

390 combined CPU tests and 16 subtests pass with five optional skips at65bd36e, with the installed Isaac USD libraries exposed
without booting Kit (including mission/runtime, recording, stock-scene and environment tests). These prove bookkeeping and failure handling, not
robot movement, perception accuracy, coupling physics or collision avoidance.
Reproduction is in RUNBOOK section 11. The shared firstmate interface contract was
published before implementation so sibling lanes can align.

## Bounded GPU diagnosis (2026-10-12)

Expected: existing resource guard can query GPU headroom before an authorized
Isaac run. Observed: `nvidia-smi` reports `Failed to initialize NVML:
Driver/library version mismatch`, library version `580.178`.

- Loaded `/proc/driver/nvidia/version`: open kernel module **580.173.02**.
- Selected x86_64 NVML and CUDA libraries: **580.178.04**.
- `/var/log/apt/history.log`: unattended upgrade on **2026-10-10,
  06:32:44–06:34:41** replaced NVIDIA 580.173.02 packages with 580.178.04.
- Read-only `ctypes` call to `libcuda.so.1` / `cuInit(0)` returned **804,
  CUDA_ERROR_COMPAT_NOT_SUPPORTED_ON_DEVICE**. No context or simulator created.
- Bounded file inventory of system x86_64 libraries, CUDA location, apt archive
  cache and the installed Isaac environment found no alternate 580.173 NVML/CUDA
  libraries. This is not an exhaustive whole-machine search.
- Initial available RAM: **23,356 MiB**. GPU free memory could not be measured.
  Initial process/socket checks found no Blender match or listener on 9876;
  those observations expire and must be checked again before any launch.
- Installed Isaac distribution metadata identifies **5.1.0.0**. No upgrade done.

Initiating trigger: the recorded driver-package upgrade. Exposing condition:
new clients select the replaced user-space libraries while the older kernel
module remains loaded. Visible symptom: telemetry refuses, so the GPU safety
gate cannot authorize a boot. CUDA initialization also fails: this is not merely
a broken telemetry executable. Already-running services may retain older mapped
libraries; their survival is not proof that new Isaac processes can start.

Firstmate subsequently ran the isolated counterfactual using task-local matching
580.173.02 libraries: NVML, CUDA initialization (`cuInit=0`, driver API 13000,
one RTX3060) and Vulkan probes passed without changing live services. These are
supervisor-reported diagnostics, not this worker's simulator evidence. They
supersede the earlier unavailable-counterfactual diagnosis. Isaac compatibility,
memory headroom and simultaneous planner operation remain unproven.

The approval hold was released on October 13. The native smoke below used the
matching process-local runtime and unchanged protected-Blender/resource guard;
no driver installation, reboot, service restart or guard bypass occurred.

## Native GPU render evidence (2026-10-13)

The subsequent `native-source-view-01` at code `6ecb5c5` passed bounded native
capture of the full store, 24 actual robot collision geometries, rollwagen rails,
loaded source support and all physical camera mounts. Six real 2560×1920 RGB-D
frames span simulation time 1.5166667458 to 1.6166667510 seconds. The verified VFR
H.264 SHA-256 is `89bd8d59904c5c41d25b8e965a5a3659149dbed3bf35c0aaa857c6ba8e7f7bd4`.
Peak total GPU usage was 6465 MiB; minimum available host RAM was 16196 MiB;
system/user service states and PIDs were unchanged. Every rigid body was static
at initial composition: this is an optical bench, not a dynamics run.

The SKU and both support QR markers decoded together in only **one of six**
frames. The other five missed one support marker. This proves the view can see
the labels, but does not prove reliable registration, empty-support commissioning,
depth-plane fit, uncertainty margins or manipulation readiness. The native image
is dark; no enhancement or synthetic replacement was used in the evidence.
Same-acquisition `instance_id_segmentation` supplied uint32 pixels and exact USD
paths: 687264 pixels per frame belonged to the known robot/cart subtrees. The
captured depth archives include those IDs; frame metadata preserves their path
mapping, actual native frame identity and acquisition steady clock. Such IDs are
authorized only for known rig self masking, with self returns and their shadows
remaining unknown. They do not supply product identity or free-space truth.

Current end-to-end gates:

| Stage | Actual evidence | Remaining gate |
| --- | --- | --- |
| Native startup/render/record | Passed cube and full-store source optical checks | Other station views and reliable calibrated support sensing |
| Stock selection/collection | CPU state machine and CAD checks | Real empty-support commissioning, planner and physical grasp |
| Cart coupling | Landed controller and CPU checks | Measured native latch/constraint validation |
| Travel/parking | Geometric route, 805 fixed obstacles including mounts | Actual observed-space coverage, dynamic cart mounts, wheel-driven run |
| Shelf transfer/verification | CAD reach and sensor-contract tests | Real planner execution and fresh post-transfer RGB-D inventory |
| Return | CPU orchestration | Actual loaded/unloaded navigation and docking |

The full simulator factory is still incomplete. No autonomous restocking cycle,
robot-motion video, successful planner GPU run or collision-free physics result
is claimed. The recordings below are genuine static native captures only.

`native-smoke-04`, code `7858f8f`, passed a guarded Isaac 5.1 startup and recorded
60 actual 640×360 RGB-D frames of a static cube/floor/camera scene. No robot or
planner was present. The embedded H.264 MP4 preserves variable acquisition
intervals; decoding verifies every presentation timestamp against the capture
manifest at microsecond precision. Frames span simulation time 1.4833334107 to
2.5166667979 seconds. The final frame has one container timebase tick of display
duration; no later acquisition is inferred. Video SHA-256:
`535fcc7b564e03c934e8d4cde2a3545594451c84e8b2601e65e5fc59334de690`.
Per-frame RGB/depth hashes and structured native identities are retained.

Preflight had 9,667 MiB free VRAM; peak sampled total GPU use was 3,989 MiB and
minimum available host RAM was 17,171 MiB. System/user service identity and state
snapshots were unchanged; free VRAM returned to 9,667 MiB. CUDA/NVML/Vulkan core
driver mappings came from staged 580.173.02 libraries. The optional system NGX
580.178.04 library was also mapped, so this is not an all-libraries compatibility
claim. A task-local unversioned NVML alias fixed the plugin's fallback to the
mismatched system library; system files were untouched.

Attempts 01–03 are preserved as partial/failed evidence. Attempt 01 produced one
RGB-D frame but mishandled a structured native frame ID; exit zero was initially
misclassified and the verdict was corrected. Attempts 02–03 correctly refused
irregular cadence under the former CFR-only contract. Firstmate authorized VFR;
no frames were fabricated, interpolated or restamped. Offline regression and
separate single-frame encodings preceded attempt 04. Rendering/recording success
does not prove stock perception, camera calibration, navigation or manipulation.

## Remaining delivery gates

Integrate corrected sibling modules and finish the simulator bundle, then run the
sensor-to-control loop under the existing protected-Blender and resource gates.
Prove stock collection, physical coupling, loaded turning/obstacle response,
parking, transfer verification and return against final code. Record H.264 video
and update the self-contained report from that actual run. No fixture animation
may be substituted for the missing evidence.

Vendor reference: NVIDIA's [Isaac Sim 5.1 standalone examples](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/introduction/examples.html)
provide the version-matched standalone entrypoint pattern. Runtime wiring is not
yet implemented or claimed compatible solely from this documentation.


## CPU integration checkpoint after mobile landing

- `runtime.py` adapts actual sibling interfaces: mission-relative clocks, navigation
  velocity gate, physical mobile latch, controller-owned arm motion and separate
  inventory observations. Post-latch sensing is required before coupling succeeds.
  Missing/repeated/stale packets stop all ports; held payloads are not released.
- `guard.py` / `run_mobile_restock.py` / shell wrapper implement exclusive GPU flock,
  protected Blender/process/socket checks, 8 GB VRAM and 12 GiB RAM preconditions,
  4 GiB RAM watchdog, isolated ROS domain 88 and 20-minute wall timeout. These
  safeguards are CPU-tested with mocked probes. The later October 13 native checks
  above exercised the real resource guard after explicit GPU authorization.
- A new scratch output directory is required; existing run/inventory directories
  are refused. Interrupted reservations persist and are not automatically replayed.
- `scene.py` composes input USD layers read-only and promotes the selected concrete
  stock mesh to a positive-mass rigid body with convex-hull collision. Scene setup
  poses and SKU metadata remain ground truth, never perception evidence.
- `isaac_bootstrap.py` imports the free-base URDF with fixed frames retained and
  discovers actual gripper/camera/lidar paths. Runtime-only code remains unexecuted.
- Environment USD validation: PASS, all 226 asset attributes resolve. Offline
  legacy store composition: PASS, 826 static colliders, 197 convex colliders,
  196/196 inherited prototype colliders. Its arm is explicitly a pedestal **stub**;
  that check does not validate the new mobile physics or gripping.
- Mobile URDF/cart rebuilt from landed source using actual upstream CAD and home
  FK. All 136 tests then passed including USD tests (none skipped).
- CPU preview uses real generated URDF collision geometry at home FK and imported
  cart USD; 24-sample Cycles render on four CPU threads took about 23 seconds.
  This is an assembled static preview, not a simulator frame or physics evidence.

The self-contained HTML embeds the environment worker's actual before/after aisle
and backroom renders, the assembled CPU mobile/cart preview and a static collision
map. It follows the repository report's paper/navy/yellow design without external
font/network dependencies. It contains no rendered or fabricated motion video.

## Cross-lane constraints still being corrected

The original navigation disk (3.847 m diameter) cannot pass a 2.5 m portal, but
this is an overly conservative orientation-independent bound. The actual straight
train polygon measures about 2.434 × 1.427 m; the mobile measured home radius is
0.620725 m and configured radius 0.65 m. Navigation tests oriented articulated
swept footprints without relaxing 1.5 m turning radius, 65-degree articulation
limit or forward-only towing. Both legs pass the CPU feedback fixture against
798 obstacles including the staging table, with 0.05 m geometry and 0.02 m
localization allowances. This is a quasistatic evaluation, not loaded PhysX behavior.
That earlier route used the old source location and a 0.02 m evaluation allowance.
The actual sensor localizer requires 0.10 m; the final route must be rerun at that
bound after the initial source/support move below. No covariance or collision margin is reduced.

Environment schema2 separates robot/cart body poses and chassis heading from arm
target orientation. Manipulation binds real `gripper_pad_tcp` / `gripper_tcp`, the
suction panel and six prismatic tines to a native contact-gated simulation actuator.
All four components are integrated from confirmed landed source. Contact physics
and native ROS graphs remain unexecuted.

Version-matched vendor references for runtime-only helpers:
[URDF importer](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/importer_exporter/import_urdf.html),
[camera sensors](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/sensors/isaacsim_sensors_camera.html).
Installed source confirms `set_fix_base` / `set_self_collision` importer methods.


## Representative handling-unit feasibility check

The approved initial slice is configured in `scripts/restock_scenario.json`:
0.60 × 0.40 × 0.16 m shallow sealed handling unit, 0.45 kg, one inventory handling
unit of `juice_orange`; hidden retail contents are not counted. Source support is centred at (-5.73,17.1), top Z=0.60;
robot loading pose (-5.73,16,0), cart body (-7,16,0). Cart support top Z=0.225,
load centre local X=+0.08. These are initial conditions, never a runtime relocation.

The measured target shelf is approximately 0.94 m wide × 0.476 m deep; the tray
fits its support and 0.28 m vertical shelf pitch. The rejected top-down grasp
intersected the upper shelf; the current candidate uses the actual side seal on
carton-local wall3, with source yaw pi, cart yaw -pi/2 and shelf yaw0. Sixteen
contact/approach/lift/preplace IK stations solve, including unloading the cart
at the actual CPU route arrival. Nominal shelf robot goal is (-3.05,6.0,pi).
The rejected Y5.85 pose is no longer used. Actual fixture robot pose is
(-2.9714,5.9941,-3.0945) and cart body (-1.9222,6.5108,-2.2934), about 46 degrees
of articulation, retained in unloading calculations. A bounded 5-degree
alignment-goal search found no route; aligned arrival is not assumed.

`check_restock_sweep.py` evaluates every URDF collision link and held payload
along joint-linear candidate insertion segments, retaining all store/cart solids
and bounding between-sample displacement by 2 mm. Disconnected frame and shelf
solids preserve their openings. There are no arm/tool overlaps in this bounded
check; wheel/floor and payload/support bounds remain recorded and the strict
candidate verdict remains false. These contact bounds are not physics proof.
Transit paths, self collision, actual planner trajectories and navigation
tolerance envelopes still require validation. No fixture was deleted or disabled.

`scene.author_initial_scenario` now authors the support, dynamic shallow unit and
exact-slot initial depletion in a scratch layer; 29 target product prims are
deactivated, while price tags and fixtures remain. A rendered QR material labels
the unit. Setup metadata is not an observation. `ManipulationBridge` supports
the camera-owning CAD runtime and a separate planner process without issuing
duplicate gripper commands. `restock.ros_manipulation` dispatches camera packets
to native ROS graph messages and latches hold on failure; reordered statuses
cannot actuate the gripper. The importer enables robot self collision. The actual
simulator bundle, camera calibration, mapper acknowledgements and motion-server
launch remain incomplete. No runtime fixture substitutes for those dependencies.

The initial source/table moved 0.10 m north to clear the actual localizer's 0.10 m
uncertainty envelope. The new source IK and whole-link insertion sweep pass for
arm/tool geometry; intended payload/support and wheel/floor bounds remain.
Camera and full-route validation must consume this new scenario hash.

The initial north-shifted scenario check at the actual 0.10 m uncertainty floor
failed with `blocked_no_route_timeout`. This is historical diagnosis: the merged
portal/corridor correction now passes outbound and return at 0.05+0.10 m margins
against all 798 obstacles for scenario file hash
`fbea2c7420ea7f4451cae27981fde897811f6ce7d6da1a09d1073ec3d1fc3a80`.
Evidence is the navigation lane's `measured-margin-scenario/summary.json`.
Integration reran IK against its exact independent robot/cart arrival: all
16 poses solve (`reach-measured-margin.json`). This remains a quasistatic CPU
fixture, not continuous sensing or PhysX proof. Earlier 0.02 m evidence remains
insufficient and is not substituted for this result.
The matching `sweep-measured-margin.json` checks all 16 insertion segments with
the exact revised arrival. Only wheel/floor and held-payload/support overlap bounds
remain; no arm/tool overlap was found. The strict `candidate_clear` is false,
and transit, self collision and camera mounting geometry remain unverified.

`IsaacPort` acquires articulation encoders, rigid-body docking feedback, physical
range scans, native seal, measured transport geometry and recording frames.
Docking feedback is ideal simulated proprioception, separate from localized
navigation and image-derived inventory. `NativeTransport` authors domain88 native
clock/joint/TF and stamped arm command nodes. Limit, tracking, stale-command and
heartbeat faults latch measured hold. These APIs have not executed in Isaac.
Post-transfer stow uses a collision-planned joint goal and measured completion
before two fresh inventory observations, preventing travel with the arm extended.
`preflight_mobile_restock.py` lists unresolved files/modules and seven exact ROS
service/action prerequisites without probing the held GPU.

`twin.restock.trajectory_host` supplies the host FollowJointTrajectory endpoint
with guarded stamped arm targets and measured goal completion. Its cubic
interpolation checks all position, speed and acceleration extrema, rather than
only waypoints. Six CPU numerical tests pass; DDS transport and physical tracking
remain untested under the simulator hold. Start it only in the isolated host
runtime with the generated mobile URDF, as documented in RUNBOOK §11.
Installed Jazzy action/message fields were checked without creating a ROS context.

Firstmate reports task-local NVML/CUDA/Vulkan probes succeeded with staged
matching libraries; these were not run by this worker and do not establish Isaac
compatibility. The explicit simulator hold remains in effect pending relay.

## Runnable checkpoint and exact remaining inputs

Draft review: https://github.com/ailerobotics/blender-supermarket/pull/15.
The committed implementation includes merged environment/mobile/navigation/
manipulation modules and integration mission/transport/execution code. CPU proofs
are separated from all runtime claims above. `physics_setup.inputs` checks the
actual generated CAD limits, initial-state flag and input hashes without importing
Isaac. `physics_setup.assemble` provides the native import/composition/default-state/
gripper/drive assembly for the guarded app; it is not yet simulator-tested.

Host worker owns CAD SRDF/XRDF generation and coverage/hash evidence. Its task-local
ROS overlay imports and component dependency checks passed; this does not prove
GPU planner/mapper startup. Host, camera and navigation corrections are now merged.
The October 13 host refinement is merged: 1,930 conservative CAD spheres have
zero nonexcluded transport overlaps, and seven full-motion separation proofs
justify narrow exclusions. Strict host `--check` passes against the current
generated inputs and installed ROS/NVIDIA overlay; GPU capacity and actual
trajectories remain unverified. The original ten-overlap report is historical.
The camera follow-up is now merged with an executable 2560×1920 candidate config,
separate measured support/label masks and paired physical markers. Its mounts
clear all 24 collision links plus held payload in 16 bounded local segments;
whole-mission transit, perception and native dynamics remain unproven. Actual
empty RGB-D commissioning and fresh marker/witness registration are required;
an occupied slot never becomes empty by reusing dated geometry. Next is one
candidate view's actual render/decode check before allocating multiple cameras.
Mapper reset/incorporation acknowledgements and the complete mission factory
still require native integration. Static mount geometry must also enter navigation.

October 13 integration adds explicit host-interpreter/setup validation and owned
ROS-process sessions, native camera optical TF registration, global host-stop
checking outside manipulation, and actual mesh resolution before URDF import.
Independent inventory verification requires two new post-transfer sensor scans,
including full cart vacancy and matching shelf identity. A depth-volume checker
refuses any occupied, invalid, occluded or unobserved cell; it requires separate
fresh support-plane evidence and a real mapper incorporation timestamp. These
modules pass CPU tests; native ROS/robot physics execution is still untested.

Two newly identified runtime constraints remain open. First, the original lidar
column starts inside the chassis. The navigation follow-up finds unknown sectors
even with corrected height or an outboard mast when actual robot/cart collision
bounds are included. Its multi-origin stopping-sweep coverage work is not yet
landed. The passing map-only route fixture proves geometric routing, not usable
physical range-sensor coverage; the runtime must stop with unobserved space.
Second, the camera follow-up corrected its 120-second baseline lifetime mismatch.
The recorded IK branches at the existing 0.15 rad/s command limit
need approximately 52.53 seconds for collection and 52.55 seconds for shelf
transfer; adding the 52.95-second outbound fixture gives 158.03 seconds before
commissioning, reset, acceleration or planning overhead. This is a timing
estimate, not a measured or globally optimal trajectory. The merged baseline now
defaults to 600 seconds, bounded to 900, while fresh evidence remains limited to
0.5 seconds. Original acquisition timestamps remain immutable across scene resets.

Resume CPU validation from the isolated worktree (runtime config remains in the
task artifact directory, with machine paths kept out of reusable source):

```bash
python3 scripts/preflight_mobile_restock.py --runtime-config "$RUNTIME_CONFIG" \
  --output "$TASK_OUTPUT/runtime-preflight.json"
python3 scripts/check_restock_navigation.py --environment "$ENVIRONMENT_MANIFEST" \
  --scenario scripts/restock_scenario.json --localization-allowance .10 \
  --output "$TASK_OUTPUT/navigation-current"
```

After missing components land, rerun reach/sweep/camera proofs against the exact
new route and scenario hashes, then rerun the CPU suite. The future execution
command is `scripts/run_mobile_restock.sh --manifest "$ENVIRONMENT_MANIFEST"
--scenario scripts/restock_scenario.json --runtime-config "$RUNTIME_CONFIG"
--execute --factory twin.restock.isaac_bundle:create
--output "$NEW_RUN_OUTPUT"`; that factory is still incomplete, the scenario
readiness flag remains false, and this command is **not runnable as a complete
mission yet**. Execute only after explicit GPU release and fresh guards. No MP4
has been fabricated to fill the missing physics evidence.
