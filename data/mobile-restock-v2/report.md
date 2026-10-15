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
| Full autonomous restock cycle | CONSTRUCTS+STARTS+TICKS; PAUSED at a native RTX render segfault on the first mission step | native+gated | runs 05-18 (guarded, env_isaaclab, --host-stack). The bundle now fully assembles and the mission state machine starts and ticks into `collect`; honest manip degradation reports manipulation unavailable (cuRobo not provisioned) without blocking construction. 14 previously-untested integration gaps were found + fixed along the way (bridge env, arm camera, manip degradation, camera collision_solids, /World anchor, tray-perception pose_bridge, nominal-dt sensor clock, reference-nav Odometry+Observation, recorder empty-frame guard — see the failure-chain section below). run-18 then **segfaulted (EXIT=139) inside the native RTX renderer** (`librtx.raytracing`/`libcarb.scenerenderer-rtx`/`libgpu.foundation`) at the first `world.step(render=True)` of the mission, under the full composed scene + 3 camera render products + recording. No mission.json / 0 frames. Per the run-17/18 bound this lane is PAUSED; blocking stage = **first rendered physics step of `collect` (native RTX renderer)**. See "Full assembled mission run (d)" below |

Source-station note: per the integration/camera contract the 3/3 native decode
(`view-source-lit-01`) is the accepted source-station sensing proof. The pre-capture
marker metric-fit registration (`restock_camera.sensed.register_marker`) is deferred
into the mission-bundle station sensing, where `base_from_camera`, `FrameEvidence` and
the support/auxiliary layout are wired from the camera config in one place, rather than
hand-assembled against camera-owned internals for a one-off run; the native vacancy
capture it feeds requires the mapper incorporation gate above regardless.

## Stage-by-stage split (Firstmate-approved): VRAM segfault fixed, mission drives to the nav coverage gate

Firstmate approved splitting the cycle into bounded stage pairs after the run-18 VRAM
segfault. The `collect`+`couple` pair (`run_mobile_restock.sh --execute` with
`mission-runtime-cc.json`, `render_scale=0.5`, `stages=[collect,couple]`, no host stack —
reference nav + degraded manip need none) now runs for ~4.5 minutes with **no crash**:

- **VRAM segfault (run-18) FIXED.** Root cause (read-only confirmation, Firstmate step 1):
  Vulkan `ERROR_OUT_OF_DEVICE_MEMORY` / crash key `gpuOutOfMemory='oom'`,
  `gpuVRAM_0≈13.1 GB` on a 12 GB card — three 2560×1920 render products + recording. NGX was
  ruled out (disabled in the log; driver 580.173, not the system 580.178.04). Fix: config
  `render_scale` (scales render-product resolution; 0.5 → 1280×960) threaded through
  `CameraRig`. runs 19-21 boot and render with no OOM.
- **Sensor-clock float drift FIXED.** `RuntimeAdapter._step`/`Mission._validate` now allow
  `age >= -1e-6` (the port's `tick*dt` single-multiply vs the mission's accumulated `+=dt`
  differ by one ULP).
- **Arm-transport gate FIXED.** Nav requires the arm stowed in the transport envelope
  (`navigation.core` `arm_not_in_transport`). At the stowed pose the binding extents are
  **structural** — `hitch` 0.621 m and `base_link` 0.583 m — so the default
  `transport_radius_m=0.55` can never pass; it is too small for this mobile+cart robot. Set
  `transport_radius_m=0.65` (above the hitch; the stowed arm's 0.591 m upper-arm fits). The
  default in `isaac_bundle` should be raised to match this robot.
- **BLOCKING: `insufficient_sensor_coverage_timeout`.** With arm-transport satisfied, the
  reference nav still refuses to drive: it zeroes `coverage_radius` when **any** lidar beam is
  self-occluded (`navigation.sensors.observation`: `coverage=0 if any(scan.unknown)`), and the
  range sensor is posed at the robot **centre** (`base_link`, heights 0.15/0.5/1.0 m) raycasting
  through the robot's own geometry (cube/surface-plates/stowed arm/mast), so many beams self-hit
  → coverage 0 → nav waits → 120 s timeout. The robot stays stationary (238 recorded frames are a
  static robot, not a driven path). This is a navigation-lane coverage-certification contract
  that the robot's self-geometry fundamentally violates: a mobile base with a mast/arm/cart can
  never present a fully-unoccluded 360° scan. Resolving it needs a navigation-lane decision
  (tolerate self-occlusion in the coverage certification, or certify coverage from a dedicated
  elevated lidar frame with a clear sweep) — escalated to Firstmate.

## Full assembled mission run (d): boots clean through every infrastructure gate, blocks at transfer-manipulation (measured)

The full assembled mission was attempted as one bounded guarded native run with the
host planner stack co-launched (`run_mobile_restock.sh --execute --host-stack`, env_isaaclab,
task-local NVIDIA runtime, domain 88, `--heavy-threshold-mb 2500`). The full bundle now
constructs and the mission starts and ticks into `collect`; the run is PAUSED at a native RTX
renderer segfault on the first rendered physics step (details and the full failure chain below).

Measured progression (runs 05-18; console logs + `run-06..17 mission.json/mission-error.txt`
and the run-18 breakpad crash + py-spy traceback under the task scratch dir):

1. **GPU guard** passes (>= 8 GiB VRAM free) once orphaned co-launch host-stack GPU
   processes are reclaimed. (Teardown note below.)
2. **Host MoveIt/cuMotion stack** warms up: 6/7 servers advertise; the `robot_segmenter`
   NITROS-allocator failure is the documented-expected one.
3. **Isaac Sim** boots and composes the full scene (store + mobile UR10 + rollwagen +
   gripper + physics).
4. **ROS2 bridge**: `rclpy loaded` — the bundled Jazzy rclpy initializes.
5. **Arm camera** is authored and initialized.
6. **Full bundle construction** succeeds: manipulation degrades honestly (cuRobo not
   provisioned), sensing/camera rig, port, recorder, verifier and the RuntimeAdapter all build.
7. **Mission starts and ticks** into `collect`.
8. run-18 **segfaults (EXIT=139) in the native RTX renderer** on the first rendered
   physics step.

### Measured failure chain (runs 05-18) and the blocking stage (PAUSED)

Firstmate authorized an honest interim run: the mission records the navigate/couple/tow/
park/verify/return loop with the manipulation (`transfer`, and in this scenario `collect`'s
pick) reported FAILED-unavailable — never a mocked success. Reaching a running mission
required fixing a long sequence of previously-untested seams; each guarded run advanced past
the prior gap (convergent), ending at a native renderer crash:

| run | stopped at | measured cause | fix |
|---|---|---|---|
| 05-06 | factory | arm camera attached to body-frame `front_camera_frame`, not a Camera prim | author a child USD Camera (forward-looking) |
| 07 | factory (transfer manip) | `ModuleNotFoundError: curobo` — in-process cuRobo planner not provisioned | honest `_UnavailableManipulation` degradation (Firstmate-approved option a) |
| 08 | GPU guard | wrapper `:-` default kept inherited `ROS_AUTOMATIC_DISCOVERY_RANGE=SUBNET` | set domain-88 isolation vars unconditionally |
| 09 | factory/sensing | `CameraRig.camera_spec` needs per-camera `collision_solids` | bridge design `solids` (dimensions_m) → per-camera `collision_solids` (size_m) |
| 10 | factory/sensing | `camera physical parent absent: /World` on the composed stage | author an identity `/World` anchor |
| 11 | factory/sensing | `pose_bridge.py` FileNotFound — USD prim path used as a filesystem root | thread the tray-perception filesystem `pose_bridge_root` from the runtime-config |
| 12-14 | collect (first tick) | `stale sensor packet` age = -8.7e-10 — Isaac physics-dt vs python 1/60 float drift | report the nominal `tick*dt` sensor clock |
| 15 | collect (nav) | `'tuple' object has no attribute 'pose'` — raw wheel tuple as odometry | port builds `core.Odometry` + obstacle `Observation` for reference nav |
| 16-17 | collect (nav→record) | `too many indices: 1-D array` — recorder sliced a not-yet-rendered 1-D RGB buffer | recorder skips empty frames until a real one arrives |
| **18** | **collect — first `world.step(render=True)`** | **native RTX renderer SIGSEGV (EXIT=139)** in `librtx.raytracing` / `libcarb.scenerenderer-rtx` / `libgpu.foundation`, under the full composed scene + 3 camera render products + recording | **OPEN — native Isaac/RTX stability under full-scene multi-camera render; not a Python-layer fix** |

**Currently blocking stage: the first rendered physics step of `collect`** — Isaac's RTX
renderer segfaults when stepping the full composed mission scene (store + mobile UR10 +
rollwagen + gripper + 3 camera render products + mission recording). The single-camera
optical bench (`view-shelf-pr45-prod`, `view-source-lit-03`) renders the same store without
crashing, so the trigger is the **full-scene, multi-render-product** load, not RTX itself.
No `mission.json` is written and no frames are recorded (0 frames).

**Lane PAUSED** per Firstmate's run-17/18 bound: no further whole-cycle runs are launched.
The likely path (for Firstmate to direct) is a **stage-by-stage split** — e.g. `collect`+
`couple` with their own smaller scene/fewer render products and their own video — or reduced
render settings (resolution, fewer simultaneous render products) to clear the RTX segfault.
The manipulation planning (`transfer`) remains the retired in-process cuRobo path and still
needs rewiring to the host cuMotion MoveGroup (`ros_adapter.RosMotionPort`, Firstmate step b).

`run_mobile_restock.py` persists the construction traceback to `<output>/mission-error.txt`
before `app.close()` (which can hard-exit), and `RuntimeAdapter.step` now carries the
originating frames in its failure reason; both were added so these runtime-tick failures
could be localized without a core dump.

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

## Runs 30-32: nav coverage fixed, manipulation host-stack blocker exposed

The uncoupled collect-navigate coverage timeout that blocked runs 05-31 is now
**resolved**. Navigation PR#49 extended the occupied-envelope crescent
certification to `core.Controller.gate_velocity`: the sub-margin crescent of a
`v=0` settle stop is exempt, side/rear self-occlusion is exempt at every heading,
and a genuine forward blind zone still refuses. A run-31 collect-only diagnostic,
instrumented with a new per-tick `nav-trace.jsonl`, localized the exact cause:
the pickup goal equals the robot start pose `(-5.73, 16, 0)`, so the controller
only ever issued the settle stop command, whose stopping distance is the residual
physics velocity alone (~1e-4 m). `sweep_extent` at near-broadside bearings is
non-monotonic in stop distance (equal to the footprint radius at stop 0, slightly
above it for a tiny stop, below it again once the quadratic footprint-shrink
dominates), so an infinitesimal residual velocity spuriously pulled the five
near-side self-occluded beams (−85.4° … −89.4°, the robot's own side/cart
geometry) into the required swept crescent and refused forever. The fix exempts
exactly that case.

Run-32 (PR#49 merged, full `collect+couple`, `--host-stack`) confirmed the fix:
collect-navigate **passed** — no `insufficient_sensor_coverage_timeout`, and the
robot held station at the goal for 2070 ticks / 34.5 s of simulation (the old
failure fired `blocked_timeout` at ~8 s). The mission then advanced into
collect-**execute**, where it **stalled**: the host-stack node `robot_segmenter`
(`nvidia::isaac_ros::manipulator::RobotSegmenter`) fails to load at host launch
with *"could not create subscription: invalid allocator"*, caused by a
duplicate/incompatible subscription on `rt/restock_sim/cumotion/robot_description_updated`
(`std_msgs/Bool`). Without it, nvblox receives no robot-masked depth, the ESDF /
planning scene stays empty, cuMotion produces no plan, and the arm never picks —
`inventory.sqlite` shows the job stuck at `collect` with 0 units loaded and 0
observations. The robot sat stationary until the wrapper's 1200 s wall-clock cap
sent `SIGTERM` (signal 15) mid-execute; `mission.json` was never written, so this
is an interrupted run, not a clean pass or fail. The `robot_segmenter` load
failure lives in the host-stack launch graph
(`sim-restock-host-v2` runtime closure, `librobot_segmenter_node.so`), was latent
in every prior run, and only became reachable once navigation stopped failing
first. A secondary constraint: stepping runs at ~0.48 s wall per tick (render plus
host ROS round-trips), so 34.5 s of simulation consumed ~1000 s of wall time —
even without the stall, the full pair may not fit the 1200 s cap. No MP4 was
produced for this incomplete mission. Artifacts:
`run-32-collect-couple/{host-stack.log, inventory.sqlite, nav-trace.jsonl,
mission-frames/}`.

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

## Masked-depth + mission-path seam chain (runs 32-40, 2026-10-14)

Each guarded run cleared one real seam. Two independent chains converged.

**Masked-depth transport (host nvblox) — PROVEN end to end.** (a) Host `robot_segmenter`
failed to load (duplicate NITROS subscription) — fixed by host PR#50 (drop the segmenter;
take Isaac masked depth), evidence run-32 host-stack.log. (b) An rclpy publisher created inside
the Isaac SimulationApp aborts (`parameter_event` typesupport) — run-33 crash. (c) OmniGraph
`ROS2PublishImage` is render-product-driven, not data-driven, so it delivered no depth.
(d) Decided (Firstmate): publish from a separate out-of-Isaac rclpy subprocess fed over a
length-prefixed Unix socket (`masked_depth_ipc.py`, `ArmDepthSubprocessBridge`); TF via an
optical `restock_camera_arm` prim on the existing transform tree. (e) The subprocess must use
the HOST DDS config (`twin.restock_host.config.ENV`: rmw_fastrtps_cpp, FASTDDS_BUILTIN_TRANSPORTS=
UDPv4, SKIP_DEFAULT_XML, localhost) or it discovers topics but delivers no data (UDPv4 vs default
SHM+UDP). PROOF: focused probe `scripts/restock_runtime/run_transport_probe.sh` (reusable
regression test) — host subscriber got masked_depth 1.63 Hz, depth/info one stamp, 32FC1,
frame restock_camera_arm; `tf2_echo` resolves; nvblox `get_esdf_and_gradient` -> **248,984
observed voxels** (run-tp-3).

**Mission path (why collect never reached execute, runs 32-40).** The mission was stuck in
collect-NAVIGATE the whole time, so the bridge was never driven. Seams: (f) collect-navigate
SUCCEEDS (arrived_stopped) but `parked=False` loops at "waiting for measured parking": the
resting cart reads ~12-14 mm/s, the documented PhysX TGS instantaneous-velocity bias, which the
0.005 m/s gate can't clear. FIX: `parked` from net POSE displacement over 0.25 s (bias-free,
isaac_port). Then execute reached + bridge publishes real masked depth (no_depth=False). (g)
arm-camera capture "stale acquisition"/"no new synchronized rendered camera frame": the render
reflects the just-stepped state, leading the dt-accumulated sim clock by ~1 physics step. FIX:
tolerate a -0.02 s lead in `frame_deadline` + the two duplicate now-stamp checks
(isaac_sensors, ros_manipulation). (h) A capture-validation failure stopped the bridge (manip
step -> stop -> fail). FIX (Firstmate structural): the bridge publishes every sensing tick
independent of capture; a capture failure is non-fatal (Result running/observe), the stage
timeout still bounds it. CURRENT GATE: capture raises `'dict' object has no attribute
support_view` (slot-config in IsaacCartonSensors) — now non-fatal, so the bridge sustains.

**run-41 (structural fix in):** the bridge now publishes sustained (published=10, nvblox
received depth) and the capture gate is non-fatal, but the mission fails "host process/status
watchdog": IsaacRosGraph.poll times out after 5 s because the host planner does not reply to the
manipulation job (it needs a built ESDF + a cuMotion plan, which takes longer than the 5 s
watchdog to warm). NEXT (next session): the host manip-reply / ESDF-warm handshake vs the
IsaacRosGraph watchdog, and the IsaacCartonSensors slot `support_view` config. The masked-depth
transport itself is proven (focused probe, 248,984 voxels) and now drives the live mission.

### The slot `support_view` gate is architectural, not a config typo (diagnosis)

Tracing the wiring: the manipulation grasp capture (`RosManipulationRuntime` → `capture` →
`IsaacCartonSensors.capture`, built in `isaac_factory._build_runtime`) runs on a **bespoke arm
eye-in-hand camera** (`front_camera_frame/arm_camera`) and is constructed with **no
`context_provider`** and with the runtime-config station `slots` passed verbatim — those slots
are placeholder id tags `{"id":"source"}` / `{"id":"shelf"}`, not `SlotView` objects, so
`capture` hits the `elif self.slots and all(s.support_view ...)` branch and raises
`'dict' object has no attribute support_view`.

The only **registered support views / context_providers** in the system are built by the sensing
lane (`sensing_sources.build_sensing_sources` → `rendered_context`), keyed by the **fixed design
station cameras** (`source`/`shelf`/`cart`, see `restock_camera/config.py`: distinct intrinsics
fx 2000/1500/1050, 2560×1920). Those feed the camera lane's `sensing_factory` capture, which in
`isaac_bundle.create` is wired **only into the `InventoryVerifier` (the `verify` action)** — not
into the manipulation grasp path. So:

- **verify** → station-camera capture + context_provider (correct, registered).
- **collect/transfer grasp** → arm-camera capture, no registration → the error.

The arm eye-in-hand camera's role is the **masked-depth bridge to the host nvblox** (proven);
it has no support design, and its intrinsics would not match a station camera's `support_view`
(capture line ~117 `np.allclose(intr,[k.fx,...])`), so it cannot simply borrow a station
context_provider. A correct fix wires the **station camera + its context_provider** into the
manip capture path (plumb `assembly.sensing_sources['context_providers']` into the manip runtime
and have `configure_station` select the station camera + provider), OR registers an arm-camera
support view. Both are multi-file and, because the CPU contract tests inject `_builder`/`_runtime`,
**not offline-validatable** — only a GPU boot proves the real wiring. Combined with the still-open
host 5 s manip-reply/ESDF-warm watchdog (measured in parallel by the host worker), a clean
collect+couple proof inside this task's window was not attainable; escalated to Firstmate.

### The narrowest fix (register the arm view as the source support_view) is blocked by a missing commissioning phase

Firstmate authorised Option A (bounded) and asked first for the narrowest fix: register the arm
eye-in-hand view as the source station `support_view` for the COLLECT capture only. Checking it:
a usable `support_view` is produced only by `RenderedSupportContext.__call__`, which requires a
**commissioned baseline** — `slot.tracker` set by `RenderedSupportContext.commission(frame, now,
empty_polygons)` from an observed **empty** support plane (support + witness quads selected from
real empty RGB-D). Two hard blockers:

1. **Commissioning is never wired into the mission.** `.commission(` has callers only in
   `tests/test_restock_sensed_views.py`; no mission path (bundle, sensing_sources, runtime) calls
   it, so every `RenderedSupportContext` would raise *"observed empty commissioning baseline is
   required"* — this is why the sensing-lane context providers are built but the capture path has
   never actually run in a mission (verify included).
2. **The collect source is occupied at t=0.** The scenario places the `juice_orange` unit at
   `backstock_00` (`unit_bottom_z`, `unit_yaws_world`); only the *destination* shelf is
   `target_initially_depleted`. A commissioning baseline needs the support plane empty, so the
   source cannot be commissioned mid-collect while the carton to be picked sits on it.

The code deliberately refuses a design-only/fabricated support view (`calibrate_rigid_support`
from empty RGB-D is mandatory; the design itself warns "nominal geometry is design only; select
support/witness quads from empty RGB-D"), consistent with the project's standing sensor-provenance
invariant — so synthesising a support_view to make capture "pass" is not an option. The real
Option-A path is therefore not a bounded plumb but a new **arm-camera commissioning choreography**:
a pre-mission empty-source baseline acquisition (observe the source empty before the carton is
staged, or a separate staged empty frame) plus per-frame arm-camera marker framing at the source.
That is multi-session and not provable by this task's 13:00 window. Re-escalated to Firstmate with
the recommendation to bank the proven masked-depth transport chain + the run-41/host-3.30 s gate,
and scope the commissioning build as its own task.

## Banked state (2026-10-14) — what is proven, the host figure, and the commissioning contract

Firstmate's decision (2026-10-14 11:42): **bank**. Fabricating a baseline is out. This is the
authoritative summary of where the mission stands and what the next task (camera-lane
commissioning) must deliver.

### Proven, banked

- **Masked-depth arm-camera transport, end to end.** Focused probe
  `scripts/restock_runtime/run_transport_probe.sh` (kept as a reusable regression) drives the
  out-of-Isaac rclpy subprocess bridge (`twin/restock/masked_depth.py` `ArmDepthSubprocessBridge`
  + `twin/restock/masked_depth_ipc.py` over a Unix socket, host DDS config) and asserts on the
  host side: nvblox integrated **248,984 observed voxels**; masked depth published at **~1.63 Hz**;
  depth + camera_info share one stamp (32FC1, frame `restock_camera_arm`); `tf2_echo` resolves the
  optical frame. The same bridge now runs inside the live mission (runs 38–41) and publishes every
  sensing tick, independent of carton verification (the Firstmate structural fix: a capture failure
  is non-fatal, `Result('running','observe')`; only a host emergency via `graph.poll` is fatal).
- **Mission path through collect-execute.** nav coverage (PR#49) and the parked gate
  (net-pose-displacement over 0.25 s, bias-free, replacing the TGS instantaneous-velocity bias that
  stalled runs 32–39) both land; collect navigates to goal, parks, and reaches `MANIP_START`.
- **Host planner latency fits the watchdog.** Cold first job on this GPU measured **3.30 s**
  (ESDF ready 1.04 s + first cuMotion reply 2.26 s) — inside the IsaacRosGraph 5 s watchdog by
  1.7 s **without the renderer co-resident** (host worker, PR#51 instrumentation-only, merged). No
  IsaacRosGraph timing change is needed unless the contended figure lands near 5 s.

### The one remaining gate: the grasp-perception support_view baseline (commissioning)

The mission's collect grasp needs the arm-camera `IsaacCartonSensors.capture` to emit a sensed
target (observations + `sensed_target_bboxes`) so the host object_segmenter can carve the pick
target from the ESDF and cuMotion can plan. Capture cannot run without a **commissioned
support_view**, which the mission never provides. The precise contract the next task must satisfy:

- **What a support_view baseline is.** A `restock_camera.sensed.SupportView` carried on a
  `RegisteredSupport` tracker: the metric support plane geometry (support quad + witness quad,
  `support_z`, required `.62×.42` footprint, calibration id, lifetime) calibrated by
  `calibrate_rigid_support(depth, support_quad_uv, witness_quad_uv, intrinsics, transform,
  registration, ...)` from an observed **empty** support in real RGB-D, together with the
  per-frame fresh marker registration (`register_marker`, decoded QR fiducials) that re-locates it.
  Design geometry is explicitly **not** a substitute (`restock_camera/config.py` warns "nominal
  geometry is design only; select support/witness quads from empty RGB-D"), per the standing
  sensor-provenance invariant.
- **Who produces it.** `restock_camera.binding.RenderedSupportContext`: `.commission(frame, now,
  empty_polygons, lifetime_s=...)` acquires the baseline and sets each `slot.tracker`; thereafter
  `RenderedSupportContext.__call__(frame, now)` returns a `CameraContext` of `SlotView`s (each with
  a live `support_view`). Until commissioned, `__call__` raises *"observed empty commissioning
  baseline is required"*. These providers are built by `twin/restock/sensing_sources.py`
  (`build_sensing_sources` → `rendered_context`), keyed by **design camera id**.
- **What the mission must load/do.** A commissioning phase that, per station camera, captures an
  **empty** support frame, selects the support + witness quads from that measured RGB-D, calls
  `.commission(...)`, and retains the trackers (respecting `lifetime_s`) into the mission. `commission`
  is currently invoked **only** in `tests/test_restock_sensed_views.py` — it is wired into no
  mission path, so this phase must be added to the bundle/runtime and its baselines loaded before
  the first capture of each station it serves.
- **Which stations need it, and when.**
  - **collect** → the **source** station (bin `backstock_00`). The arm eye-in-hand camera is the
    sensor on the collect grasp path; it has no registered support view and no design camera entry.
    Note the source is **occupied at t=0** (the `juice_orange` unit is staged there; only the
    destination shelf is `target_initially_depleted`), so its empty baseline must be acquired
    **before the carton is staged** (or from a separate staged-empty frame) — it cannot be
    commissioned mid-collect.
  - **transfer/place** → the **shelf** destination (`G3_back_s2`), which *is* initially depleted, so
    an empty shelf baseline is acquirable in-scene.
  - **verify** → the station cameras already wired into `InventoryVerifier` via the sensing-lane
    `capture` (fixed `source`/`shelf`/`cart` design cameras). These, too, are built but never
    commissioned, so the verify path has likewise never run live; the commissioning phase must cover
    whichever stations the verify action reads. Keep the fixed station views for verify untouched by
    any arm-camera change (Firstmate constraint).

### Native commissioning blocked: the acquisition-to-use freshness budget is exceeded on this box (2026-10-14)

Built the integration side for Option A: `scripts/twin/restock/commissioning_sources.py` (a `--sources`
hook that boots the composed store as a static bench via `view_scene.build`, empties the collect
source by deactivating `/RestockScenario/HandlingUnit`, renders the fixed `source` camera and selects
the empty support+witness polygons) and `scripts/run_restock_commissioning_gpu.py` (guarded GPU
launcher). Both import-clean and committed (`cfc952d`).

The native run (`commissioning/run-01`) reached the real `commission()` and failed with
**`ValueError: acquisition-to-use freshness expired after processing`** — the camera lane's
`restock_camera.freshness` gate. Measured on this box: `commission` on the full-res source frame takes
**~0.575 s** (FixedTileQR decode 0.262 s + `calibrate_rigid_support` ~0.30 s), over the **0.5 s**
acquisition-to-use budget (`frame_deadline` = `started + 0.5 - (now-stamp)`). The same failure repros in
the camera lane's **own CPU tests here** — `test_restock_commissioning.py` is 12/14 FAILED, every
failing case routing through `commission()` with the identical error (the 2 passing are the pure-JSON
ones). So PR52 commissioning has never run green on this hardware. (The trailing SIGSEGV is Isaac's known
`atexit`/`Py_FinalizeEx` omni.graph shutdown crash, after the error is printed.)

This gate is **shared with live mission capture** (`RenderedSupportContext.__call__` and
`IsaacCartonSensors.capture` use the same `require_fresh`/`frame_deadline`), and all three run only with
the robot **parked** (capture gates on `parked=True`). So widening the parked acquisition-to-use bound
(e.g. 0.5 s → 1.0 s) is low-risk and would unblock both commissioning and the mission's own station
capture — but it is a camera-lane safety parameter, not mine to change unilaterally. Alternatives: reduce
commission/capture cost (bounded-tile decode; calibrate on the support ROI only). Separately to confirm
with the sensing lane: the runtime-config `render_scale=0.375` downscales the station render product in
`sensing_sources._real_rig`, yet the context intrinsics stay the full 2560×1920 design — the capture
shape check (`rgb.shape[:2] == (intrinsics.height, intrinsics.width)`) would then mismatch, so either the
station cameras must render full-res or the intrinsics must scale with it. Escalated to Firstmate.

### Native commissioning PROVEN; but the collect grasp consumes an arm-camera target the fixed-camera baseline does not feed (2026-10-14)

Firstmate authorized a scoped freshness fix (`commission()` gets its own `commissioning_freshness_s`,
default 5 s, provenance-recorded; live sensing stays 0.5 s). The camera lane landed the identical
mechanism as PR53 (merged); my branch is merged to it (resolved in favour of main, dropped my duplicate
test). With the 5 s commissioning window, **native commissioning of the fixed `source` camera succeeded
end to end** (`commissioning/run-03`): `backstock_00.json` persists and reloads, provenance carries
`camera_id=source`, `calibration_id=default:source:113:source`, `commissioning_freshness_s=5.0`,
`source=commissioned_empty_support`, `capture_pose=[-5.73,16,0]`. The hook
(`twin.restock.commissioning_sources`) renders the empty source (HandlingUnit deactivated) and selects a
converged frame whose paired markers both decode under the production `FixedTileQR`. (The trailing
SIGSEGV is Isaac's known atexit/omni-graph shutdown crash; the runner now treats a complete report as
success.)

**But this does not unblock the mission collect grasp.** The grasp capture is built in
`isaac_factory._build_runtime` as `IsaacCartonSensors(camera=front_camera_frame/arm_camera, slots=
source['slots'])` with **no `context_provider`** and placeholder slots `{"id":"source"}` — it consumes no
commissioned baseline. The commissioned fixed-`source`-camera baseline is reloaded by `build_sensing`
into the **verify lane** only, which the `[collect, couple]` mission never invokes. So the collect grasp
is unchanged from run-41: the arm capture still hits `'dict' object has no attribute support_view`
(now non-fatal), produces no `sensed_target_bboxes`, the masked-depth bridge has no arm-frame target
box, and the host never gets a grasp target (watchdog).

Root design mismatch: grasp-target registration needs a **pose-stable, commissionable** camera, but the
arm eye-in-hand camera moves with the arm, and its support view at grasp time depends on the (dynamic,
cuMotion-planned) arm pose — it cannot be pre-commissioned from a fixed pose. The fixed `source` camera
(which I commissioned) is the pose-stable one. The correct path is therefore to have the collect grasp
take its target from the **fixed source camera's commissioned capture** (carton base-frame pose), and let
the arm masked-depth bridge **reproject** that pose into the arm image for the nvblox target box (reusing
`IsaacCartonSensors._object_bbox`) — a bounded cross-camera target bridge in `isaac_factory`/`masked_depth`,
but net-new, only GPU-boot-provable, and not landable by this task's 13:45 window. Also still gating any
live station capture on this box: the 0.5 s live budget vs ~0.575 s full-res decode+calibrate (the
render_scale↔intrinsics item above). Recommend banking the commissioning + freshness wins and scoping the
grasp target-source rewire (and the live-capture resolution reconciliation) as the next task.

### Collect sensing contract (architecture decided 2026-10-14 12:25)

Firstmate fixed the architecture so it is designed once: **the collect stage senses the pick target
with the FIXED `source` station camera against its commissioned baseline** (the established design —
fixed `source`/`cart`/`shelf` views are the calibrated, pose-stable, commissionable cameras; verify
already uses them). **The arm eye-in-hand camera is demoted to masked depth (nvblox collision mapping)
and fine alignment only** — it no longer registers a support view or produces the grasp target. The arm
receives a **reprojected** target computed from the fixed-camera result. This removes the arm
`IsaacCartonSensors` placeholder-slots `support_view` path from the grasp-target critical path.

**What the collect stage consumes — the fixed-source target provider.** The camera lane supplies a
`source_target_provider(now) -> SourceTarget | None` (built beside the sensing lane's commissioned
`source` context, i.e. from `build_sensing_sources`/`build_sensing` with the `backstock_00` baseline
loaded). It wraps the existing fixed-`source` `IsaacCartonSensors.capture(now, 'backstock_00', ...)`
against the commissioned baseline and, when the pick carton is sensed, returns a `SourceTarget`:

- **identity** — `sku: str`, the handling-unit QR (e.g. `juice_orange`) decoded from the carton top
  label by the fixed `source` camera (`FixedTileQR`, never a USD/semantic name).
- **pose** — `pose_station: (x,y,z, qx,qy,qz,qw)`, the sensed carton 6-DoF pose in the **`source`
  station/world (map) frame** (the `source` camera is `parent='map'`), anchored to the commissioned
  `support_view` registration center/yaw — pose-stable, independent of the robot base.
- **acquisition stamp** — `stamp: float`, the `simulation_time`/`rendering_time` of the frame the pose
  was measured from.
- **provenance** — `dict(camera_id='source', calibration_id=<baseline calibration_id>,
  baseline_session_id, frame_id, source='sensor', epoch)`; carries the commissioned baseline linkage so
  the consumer can assert sensor provenance (the standing invariant: no fabricated/fixture target).
- **freshness rule** — valid under the **live 0.5 s acquisition-to-use bound** (`require_fresh`); the
  provider stamps acquisition and the consumer re-validates freshness at the moment of use, every
  sensing tick, with the base **parked**. (Live full-res capture cost vs 0.5 s on this box is the
  separate render_scale↔intrinsics item above; the camera lane owns meeting the budget.)

**Where the arm reprojection consumes it — `isaac_factory`.** In
`twin.restock_manipulation.isaac_factory._build_runtime`, the manip runtime takes the
`source_target_provider` (threaded from `isaac_bundle` after `build_sensing_sources`, like the sensing
lane). Each collect sensing tick:
1. the grasp **target published to the host** (`RosManipulationRuntime` → `graph.publish_sensors`) is
   built from the provider's `SourceTarget` (pose + sku + provenance) — not from an arm capture;
2. the arm `ArmDepthSubprocessBridge` (`_build_masked_depth_bridge`) computes the **nvblox target box by
   reprojecting** `pose_station` into the **arm camera image**: transform `pose_station` into the arm
   optical frame via the live arm TF (`world_from_base` ∘ arm `base_from_camera`), then project the
   catalog carton volume with the arm intrinsics using `IsaacCartonSensors._object_bbox` (already exists)
   to yield `[u1,v1,u2,v2]` on the masked-depth image the host carves;
3. the arm `IsaacCartonSensors` is retained only for masked depth + fine alignment; its support-view /
   `sensed_target_bboxes` grasp role is removed.

This keeps every grasp input sensor-sourced, uses the one commissionable camera for identity+pose, and
confines the arm to geometry it can measure from a moving eye-in-hand view.

### Collect sensing wired + three gates cleared; blocked at the cross-session commissioning clock (2026-10-14)

The collect consumer is implemented and the mission now drives the fixed-source capture. Merged PR55
(scale_intrinsics + injectable clock) and PR56 (ROI-bounded decode, ~311 ms). Delivered on the branch:
per-camera render_scale (collect source renders full-res 1.0 for its ~6 px/module markers, mapper/others
0.375, matching scaled intrinsics); the `source` design support mapped to `job.source_bin` so baseline /
slot / `collect_source_location` agree; `CollectTargetBridge` (fixed-source `SourceTarget` → host `Sensors`
packet + live arm-image reprojection for the nvblox box via `reproject_carton_roi`); provider injected
after `build_sensing`; re-commissioned `backstock_00` at scale 1.0 with the ROI persisted (`run-04`).
CPU tests green throughout.

Three successive native shakedown/attempt runs each cleared a gate and surfaced the next (collect reaches
`MANIP_START`, masked depth reaches the host nvblox, host cuMotion/ESDF/MotionPlan all initialise):

1. **run-42** — host watchdog, no target. `COLLECT_CAPTURE` diagnostic added.
2. **run-43** — `FrameEvidence.validate` rejected the source frame: `now-stamp = -0.0167 s`, the render
   leading the dt-clock by one physics step, with no tolerance. **Fixed** (`de014f5`): `FrameEvidence.validate`
   now allows `-_RENDER_LEAD` on the stamp/tf checks (matches `frame_deadline`).
3. **run-44** — `fresh support marker(s) not decoded`. Root cause: `author_camera_design` (the QR support
   markers) is called only in the view bench / commissioning, never in the mission compose, so the mission
   scene had cameras (via `CameraRig`) but no registration markers. **Fixed** (`2cc106c`): author the collect
   source support's markers in the mission compose (markers-only; no duplicate cameras/mount colliders).
4. **run-45** — `expired or reversed rigid-support commissioning clock` (`sensed.RegisteredSupport.age_at`).
   The baseline (`run-04`) has `steady_domain=''`/`baseline_acquired_steady=None`, so `age_at` takes the
   non-persistent path `age = now - baseline_stamp = 0.2 - 1.52 < 0` → reversed. Commissioning ran in a
   SEPARATE process/sim session (sim stamp ≈ 1.52), the mission's sim clock restarts at 0, and the two
   are not comparable. The persistent (cross-session) path needs the baseline + the mission capture to
   share a `steady_domain` AND an acquisition clock comparable across processes — per-process
   `time.monotonic()` (what the camera's `AcquisitionCamera` records) is not comparable across the
   commissioning and mission processes.

**This gate is a camera-lane acquisition-clock-policy question.** For a baseline commissioned in a separate
process and consumed by the mission, `age_at` must gate on a clock that is continuous across processes.
Proposed fix (needs camera-lane confirmation of how `calibrate_rigid_support` populates the geometry's
`steady_domain`/`baseline_acquired_steady` and how `clock_context` threads through): give commissioning
and the mission sensing lane a shared `clock_context` using **wall time** (`time.time()`) under one fixed
`steady_domain`, and commission with a generous `lifetime_s`, so `age_at`'s persistent path measures real
elapsed wall seconds since commissioning (bounded by lifetime) instead of incomparable sim/monotonic clocks.
Alternatively, gate a pre-commissioned baseline by support identity (+ a wall-clock lifetime), not sim time.
Integration can wire the shared `clock_context` once the camera lane confirms the scheme; everything up to
this gate is proven in `run-45`.

### Gates 4-6 fixed; collect capture now validates to the host; blocked on GPU lock contention (2026-10-14)

Resolved the cross-session gate chain and drove the live collect capture through, one gate per native run:

- **Gate 4 (commissioning clock)** — shared WALL-clock `clock_context` under one fixed `steady_domain`
  across commissioning + mission (`sensing_sources`/`commissioning_sources`), re-commission with the 900 s
  (policy max) lifetime. Cleared (`762f583`).
- **Gate 5 (layout_id)** — `register_marker`'s `layout_id` hashed raw float64 model bytes, which differ by
  ULPs between the commissioning and mission processes despite the identical design (REGMARK diagnostic
  confirmed same `side_m`/`marker_ids`/shape/sum, different hash). Made it reproducible by quantizing the
  model to 1 nm before hashing (`ca0870f`). Cleared live in `run-50`.
- **Gate 6 (support witness)** — the mission compose authored the marker PLANES but not the
  `source_marker_tab` solid they sit on, so the off-slot witness patch read the floor and tripped
  "fresh witness disagrees". Author the tab solid in the mission (`2c3b49f`).

Live progression now: nav → collect-execute (`MANIP_START`) → masked depth at the host nvblox → source
markers decode → freshness OK → commissioning clock OK → layout OK → (witness fix pending its first run).
Each fix committed with CPU tests green.

**Blocked: GPU lock contention.** The collect+couple attempt needs a fresh commissioned baseline
(≤900 s old) and then the mission, two GPU sessions. Two shelf-twin workers share this 12 GiB GPU via
`/tmp/aile-gpu.lock`; the guard's `flock(LOCK_EX|LOCK_NB)` loses the race to a shelf-twin between my
sessions (run-09 refused on VRAM 6943<8000; run-11 refused `[Errno 11]` on the lock). Firstmate stated my
mission runs keep priority, but the non-blocking flock race does not enforce it, and the 900 s baseline
expiry races the contention. Needs Firstmate to briefly hold the shelf-twin workers so the priority
commission+attempt can run back-to-back. All code gates are fixed and pushed; only GPU access blocks the
final attempt.

### Ran under a reserved GPU; gates 5-6 cleared live; stops at gate 7 (motion/mapper port) (2026-10-14)

Firstmate held the shelf-twins off and granted the GPU. Commission + collect+couple attempt ran
back-to-back (`run-12` baseline + `run-52-attempt7`). Live result — the collect sensing/perception chain
now clears SIX support-validation gates in the running mission:

`nav(arrived_stopped) -> MANIP_START(collect) -> masked depth at the host nvblox -> source markers decode
-> freshness(render-lead) -> commissioning clock -> layout_id -> support witness`

each with the fixes above. The attempt then fails at **gate 7**: `IsaacCartonSensors.capture` calls
`motion.collision_time` (and `motion.joint_positions()`, `motion.observe_source()`), but the sensing lane
is wired with `manip_runtime.motion = _ArticulationParked`, a parked-only port — `AttributeError:
'_ArticulationParked' object has no attribute 'collision_time'`. `collision_time` is the nvblox
mapper-incorporation freshness stamp (`0 <= now-collision_time <= 0.5`), sourced in the mission from the
host reply (`native_channel.collision_time` / `station.collision_time`), NOT from the articulation. It
must be a real mapper-incorporation value (the sensor-provenance invariant forbids fabricating it as
`now`).

Remaining work to a full collect+couple (each likely its own gate, as the live sensing path was never
exercised before): (1) a composite sensing **motion port** exposing `collision_time` from the live nvblox
incorporation + `joint_positions` from the articulation + a no-op `observe_source`, which depends on the
host→Isaac mapper-incorporation feedback reaching the sensing lane; (2) the host cuMotion **grasp plan +
reply** within the IsaacRosGraph watchdog (host cold first-job was 3.30 s uncontended; contended figure
still to be measured once a target is produced); (3) the **couple** (cart dock/latch) stage. The collect
PERCEPTION chain (identity + pose + reprojected nvblox target box, all sensor-sourced) is proven to the
point of handing the host a grasp target; the gate-7 motion/mapper port is the next piece. GPU released
back to the shared pool after the run.

Actionable gate-7 path (the adapter already exists): `isaac_ros_graph.IsaacMeasuredFeedback` is the intended
full feedback/motion adapter — `collision_time`, `observe_source`, `joint_positions()` (the six `ARM_JOINTS`)
and `parked()`. `station_runtime` shows the intended update pattern (`feedback.collision_time =
station.collision_time`, reset to -1 on dock/frame change), where `station.collision_time` is stamped only on
a mapper-incorporation ack. The mission's `isaac_factory._build_runtime` instead wires
`HostManipulationRuntime.motion = _ArticulationParked` (parked-only), and the sensing lane reads it via
`manip_runtime.motion`. Fix: `isaac_factory` uses `IsaacMeasuredFeedback` as the motion and syncs its
`collision_time` from the live mapper-incorporation ack each tick (`IsaacRosGraph` already declares
`self.collision_time` "set only from mapper incorporation acknowledgement" — confirm the host publishes it and
`poll()` updates it, then mirror onto the feedback). Cross-lane: the host must publish the nvblox incorporation
ack carrying `collision_time`; integration mirrors it. No fabricated `now` — the gate proves the ESDF is current.

**Gate-7 consumer side DONE (`7e96b0f`), only the host publish remains.** `isaac_factory._build_runtime`
now wires `IsaacMeasuredFeedback(assembly.robot)` as the sensing/manipulation motion (replacing the
parked-only `_ArticulationParked`); `RosManipulationRuntime.step` refreshes its `collision_time` BEFORE each
capture from `IsaacRosGraph.refresh_incorporation(now)`, and the `CollectTargetBridge` host packet reports
that same real `collision_time` (not `now`). CPU-tested against a fake ack
(`test_mapper_incorporation_consumer.py`: `parse_incorporation` fresh/stale/invalid cases + the step mirror);
184 adjacent tests green. **Interface the host lane must publish** (ROS domain 88): topic
`/restock_sim/mapper/incorporation`, `std_msgs/String`, JSON `{"collision_time": <sim s>, "stamp": <sim s>}`,
emitted each nvblox incorporation (continuous, independent of a grasp job). `parse_incorporation` accepts it
only when `collision_time` is finite and `0 <= now - stamp <= 0.5` (sim clock), else -1 (fail-closed). Once
the host PR merges and the GPU returns, the collect capture clears gate 7 and hands the host a grasp target.

### PR57 merged; gate-7 consumer rebuilt to the REAL interface; two more gates cleared; stuck at collision_time delivery (2026-10-14)

The host ack (PR57, `twin.restock_host.incorporation_ack`) publishes on `/restock_sim/station/status` under
the parked-station prepare/heartbeat/stop protocol (`/restock_sim/station/command`) — NOT the
`/mapper/incorporation` topic I had guessed. Reverted that wrong consumer and wired the REAL one: the mission
drives the ack via the existing `StationClient` (JsonChannel over station/command|status) and mirrors
`station.collision_time` onto `IsaacMeasuredFeedback` each step before the capture; stop clears it. The ack
is co-launched by `twin.restock_host.launch.description` (so `run_restock_host.py --launch` → the mission's
`--host-stack` now starts it; launch-wiring test green). CPU-tested vs a fake channel.

Two more gates cleared live across attempts 8-9:

- **Gate 7 reached** — the capture now fails on the actual `collision_time` gate (`collision map has not
  incorporated current clearance evidence`) rather than an AttributeError; the feedback is `IsaacMeasuredFeedback`.
- **Gate 8 (station parked)** — attempt 8 died on `station lost measured parking`: `StationClient.tick` was
  fed the instantaneous `IsaacMeasuredFeedback.parked()` (velocity `< 5 mm/s`), which flaps on sub-mm/rad
  angular noise (`rw` hit 0.006). **Fixed**: thread the port's robust net-pose-displacement `Packet.parked`
  through `ManipulationBridge`/`HostManipulationRuntime`/`RosManipulationRuntime` into a `_PortParkedFeedback`
  whose `parked()` returns that robust value (drives both the station gate and the capture). CPU-tested.

Attempt 9 then runs the station alive, masked depth flows (published=6), but `collision_time` never goes
fresh, so the capture stays on the collision_time gate, no grasp target is produced, and the mission dies on
the 5 s host watchdog. The ack process launches (`python3-*` in host-stack.log) but logs nothing past rcl
startup — no prepare received / no `/station/status` published that the `StationClient` sees. Open question
for the next run (a `STATION_STATS` diagnostic is committed, `c3b5c8a`): is this a **connection** problem (the
Isaac-side JsonChannel prepare/status not crossing to the host ack over the mixed OmniGraph/rclpy DDS — note
the masked-depth path needed a specific host DDS config to deliver) or a **timing** problem (the ack readies
after the 5 s watchdog, since the collect flow must publish masked depth → nvblox integrate → ack ready →
capture → host plan 3.30 s, within 5 s of the manip job)? `STATION_STATS` prints the station snapshot +
mirrored collision_time per 10 steps to distinguish them. Likely cross-lane with the host ack (its prepare
reception / status QoS). GPU released after the run.

### Consumer wiring ready; collect grasp gated by the fixed-source live-capture budget on this box (2026-10-14)

PR54 (merged) added the `source_target_provider` to my contract. I built + CPU-tested the pure arm-image
reprojection core `twin.restock.collect_target.reproject_carton_roi` (commit `cdbafaf`; reuses
`_object_bbox`'s projection, clamps to the image, returns `None` rather than fabricating). The remaining
consumer wiring is fully mapped and ready to land:

- thread `sensing['source_target_provider']` into the manip runtime (new `HostManipulationRuntime.
  set_source_target_provider`, called from `isaac_bundle` right after `build_sensing` at line ~228);
- `RosManipulationRuntime` builds the host `Sensors` packet from the `SourceTarget` — one
  `Observation(source='sensor', frame='ur_base_link', pose=pose_station, sku=..., simulation_time=stamp)`,
  `target=None`, finite `joint_positions`/`collision_time` (publish serializes the whole dataclass with
  `allow_nan=False`); the `source=='sensor'` + `now-stamp∈[-0.02,0.5]` gates in `step()` apply;
- a duck-typed object exposing `sensed_target_bboxes` (updated each tick via `reproject_carton_roi` with
  the **live** arm `base_from_camera` from `UsdGeom.XformCache` × the USD-cam→optical flip
  `diag([1,-1,-1,1])`) feeds `ArmDepthSubprocessBridge` — its only coupling to `sensors` is that attribute;
- the arm `IsaacCartonSensors` is demoted to masked depth. `pose_station` is the `ur_base_link` frame
  (`Observation.frame='ur_base_link'`, enforced), the same base as the live arm `base_from_camera`, so the
  reprojection is frame-consistent (the `SourceTarget.station_frame='map'` label is cosmetic — to fix).

**Gating blocker (why the shakedown/attempt cannot succeed on this box yet).** The provider wraps the
fixed-`source` `IsaacCartonSensors.capture`, which runs the **live 0.5 s acquisition-to-use budget** on a
real clock. At the design full resolution (2560×1920) decode+validate measures **~0.5–0.6 s** here, over
the budget — so the provider raises `acquisition-to-use freshness expired` and yields **no target**. This
is triple-confirmed on this hardware: (1) my commissioning needed the 5 s window for the same reason;
(2) the camera lane's `test_apply_baselines_attaches_tracker_so_mission_capture_validates` (live
`__call__`) FAILS here; (3) **PR54's own** `test_provider_over_a_real_commissioned_source_returns_none_on_empty`
FAILS here with the identical error (its adapter `steady_clock` is not injected, so it also runs real
time — a camera-lane test-determinism gap). Dropping `render_scale` to 0.375 to cut the cost does not
help: `CameraRig` scales the render resolution but **not** the context intrinsics, so the 960×720 frame
fails the capture shape check against the full-res 2560×1920 intrinsics (and the full-res commissioned
baseline). The live 0.5 s bound is Firstmate's protected safety contract, so the fix is **cost reduction,
not budget widening**: the camera/sensing lane should scale the station-camera context intrinsics with
`render_scale` (render the station cameras at reduced resolution with matching scaled intrinsics so the
live capture fits 0.5 s) and re-commission the baseline at that resolution. Then the provider returns a
target, my consumer wiring lands, and the shakedown + attempt follow. Reprojection core committed;
consumer wiring held ready pending that reconciliation.


## 2026-10-14 — DECISIVE: the mission host stack never launches the manipulation bridge (ros_host)

Full-length attempt **run-58-full** (fresh commission run-17, 900 s baseline, 1400 s cap)
failed with `mission.json: state=failed, reason="host process/status watchdog"` after the
collect phase ran only ~6 s. The host planner stack (move_group, cumotion_planner,
nvblox_node, object_attachment, static_planning_scene_server) was **alive and healthy to
the end** (nvblox allocating at 16:20:05, teardown ~16:20:13). So the watchdog fired
despite a live host.

Root cause (confirmed, upstream of all station/collision_time work):
- The Isaac-side `IsaacRosGraph` publishes the collect job on `/restock_sim/manipulation/job`
  and fails after `timeout=5 s` if it receives no `/restock_sim/manipulation/status`
  (`isaac_ros_graph.py:161-165`).
- The ONLY host node that consumes `/manipulation/job`+`/sensors`, drives move_group/cuMotion,
  and publishes `/manipulation/status` is `ros_host.py` (`RosTransferBridge`+`RosMotionPort`+
  `TransferController`).
- The mission's `--host-stack` runs `launch_restock_host_stack.sh` -> `run_restock_host.py
  --launch` -> `twin.restock_host.launch.description`, which launches the planner stack + the
  incorporation ack but **NOT ros_host**. `host-stack.log` for run-58 shows no
  `restock_manipulation` node; `git log -S ros_host -- launch.py` is empty (ros_host was never
  in that launch). `launch_plan()` even computes `integration_host_arguments`
  (`--urdf/--shelfbot-root/--mobile-manifest`) but it is consumed only by a test.
  `HostSession` (host_session.py) is the only launcher that starts the `manipulation` process,
  and the mission does not use it.
- Therefore `/manipulation/status` is never published -> the 5 s watchdog fires every attempt.
  This — not the station transport — is why every prior attempt died on the host watchdog.

Secondary (sd2 topic-info, still relevant): the Isaac OmniGraph ROS2 **publisher** on
`/restock_sim/station/command` is not visible to the host UDPv4 Fast DDS (`Publisher count: 0`
across 14 probes) while the OmniGraph **subscriber** on `/station/status` did appear
(`sub=1` at probes 13-14). The same publisher asymmetry likely affects `/manipulation/job`,
so even once ros_host is launched the job may not reach it over OmniGraph — Isaac->host
publishes may need the masked-depth-style host-DDS subprocess.

Fix plan (incremental, evidence-driven):
1. Launch `ros_host` (manipulation bridge) in the mission host stack, wired with a ros_host-
   shaped config (`{profiles, walls}`; mission manip block has profiles but no `walls` — `walls={}`
   is accepted by the CPU tests), `--urdf/--shelfbot-root/--mobile-manifest` from the plan.
   Re-attempt; inspect ros_host's log for job receipt.
2. If ros_host does NOT receive the job -> the OmniGraph publisher->host-DDS transport is broken
   for the manip graph; route Isaac->host publishes (job/sensors/station-command) through a
   host-DDS subprocess like `masked_depth_ipc`.

### Supplement — launching ros_host alone is NOT sufficient (host 1s sensor watchdog + no pre-dispatch mapper warm-up)

Reading `RosTransferBridge` (`ros_adapter.py:269-334`) closes the loop on timing:
- `_job` (job receipt) calls `controller.start(...)` but emits NO status — status is only
  published by `_sensors` when a `/restock_sim/sensors` packet arrives.
- `RosTransferBridge` runs its OWN 1.0 s STEADY_TIME sensor watchdog (`watchdog_seconds=1.0`,
  lines 277, 331-334): active controller + no sensor packet for 1 s -> it emits
  `failed: sensor stream watchdog expired`.
- But the collect `/sensors` packet needs a fresh `collision_time` (source ESDF incorporated),
  which needs several seconds of masked-depth accumulation. Masked depth only starts flowing
  at collect dispatch (`RosManipulationRuntime.step` -> `sensing_bridge.publish`), with no
  pre-dispatch warm-up — `assembly.json` records `camera_mapper_ready: false`.

So the blocker is 3-part and entangled:
1. ros_host (manipulation bridge / status publisher) is not launched by the mission host stack.
2. Even launched, the collect cannot produce `/sensors` within the host's 1 s (or Isaac's 5 s)
   watchdog, because the source ESDF/`collision_time` is not warmed BEFORE collect dispatch.
   The real design needs the station/mapper prepared and the source ESDF incorporated during
   the approach so `collision_time` is already fresh at dispatch (sensors then flow immediately).
3. The Isaac OmniGraph ROS2 publisher does not reach the host UDPv4 DDS (sd2: `/station/command`
   pub=0), so Isaac->host publishes (job/sensors/station-command) may need a host-DDS subprocess.

This is a cross-lane architecture decision (host-launch composition + when the mapper is warmed
vs when collect dispatches + Isaac->host transport), not a station-client rewire. Escalated to
Firstmate for direction before spending more GPU.

### 2026-10-14 — fix #1 CONFIRMED working; fix #2 triggered (OmniGraph publisher->host DDS)

Attempt run-59-fix1 (fresh commission run-18) with ros_host now launched by the host stack.
Host-DDS probe of the manip control-plane across the collect window:
- `Node name: restock_manipulation` is present -> **ros_host launched** (fix #1 works). It
  subscribes `/restock_sim/manipulation/job` (**sub=1**) and publishes
  `/restock_sim/manipulation/status` (**pub=1**); the Isaac OmniGraph status subscriber
  (`_RestockManipulationROS_status_wire`) appears intermittently (STATUS sub flaps 0/1).
- `/restock_sim/manipulation/job`: **pub=0 for all 8 probes** -> the Isaac OmniGraph
  `/manipulation/job` PUBLISHER is NOT visible to the host UDPv4 Fast DDS, so ros_host never
  receives the collect job. `/station/command`: pub=0 (same). Mission still fails
  `host process/status watchdog` (no job -> no status).

Conclusion: Isaac OmniGraph ROS2 PUBLISHERS do not reach the host DDS (pub=0), while
SUBSCRIBERS do (status/station-status eventually appear). This is the fix #2 trigger:
route the Isaac->host PUBLISHES (`/manipulation/job`, `/sensors`, `/station/command`, plus
`/manipulation/stop`, `/gripper/enable`) through a host-DDS rclpy subprocess like masked depth,
keeping OmniGraph subscribers for the return topics (status, station/status).

### 2026-10-14 — fix #1/#2/#3 all working; new blocker: ack rejects prepare on a clock/stamp mismatch

run-64 (fix #3 active: approach warm-up + split sensor watchdog, commission run-21). The pipeline
now advances: nav completes -> RuntimeAdapter 'warm' phase runs (STATION_STATS tick 1..N, prepare +
heartbeats published via the control subprocess) -> MANIP_START dispatches after the 300-tick warm
cap. Mission fails `station supervisor heartbeat unavailable` (StationClient got no reply in 10 s).

Decisive: the host echo of /restock_sim/station/status shows the ack RECEIVES the prepare (fix #2
fully confirmed) but rejects it -- it goes idle -> `failed: "invalid station epoch, sequence or
timestamp"` -> `station fault requires explicit recovery`. That reason is the freshness/field guard
in IncorporationAckState.accept: `0 <= now - stamp <= 0.5`. Measured:
- StationClient prepare stamp (mission Packet.simulation_time) = 0.233 at warm tick 1;
- ack now() (snapshot `stamp`, its /clock-derived sim time) advances in 0.05 steps to 1.0+.
The ack's now() and the mission's prepare stamp fall outside the 0.5 s window, so accept() rejects
every prepare/heartbeat -> collision_time never advances -> warm never reports ready (runs to the
cap) -> the StationClient never gets a matching-epoch reply -> 10 s heartbeat failure.

Root cause is a clock-sync / stamp-domain mismatch at the station boundary: the StationClient stamps
with the mission's sim time while the ack validates against its own /clock-derived sim time, and the
two diverge beyond 0.5 s (sampling skew + discovery/relay latency, with the host /clock published by
native_transport). This is cross-lane (host incorporation_ack.py freshness contract + native_transport
/clock) -- escalated to Firstmate. Candidate fixes: tolerate a bounded skew on the ack's freshness
lower/upper bound (symmetric with the gate-1 render-lead tolerance), or align the StationClient stamp
to the same /clock domain the ack validates against. Prepare-to-first-/sensors gap: N/A -- no /sensors
was produced because the prepare never cleared the ack's freshness guard.

### 2026-10-14 — fix (clock domain, option b): station/command stamps are /clock, never mission time

CONTRACT: station/command stamps (prepare, heartbeat, stop) are /clock (Isaac stage time), never
mission time. Root cause of the ack's 'invalid station epoch, sequence or timestamp' rejection was
two clock epochs, not skew: the StationClient stamped with Packet.simulation_time (mission clock,
zeroed at the first mission action) while the host ack validates its 0.5 s freshness guard against
/clock (Isaac stage time, published by native_transport from the same process). The gap was ~origin
(the stage time elapsed before the first action). Fix (my branch, host ack untouched): NativeTransport
exposes its last-published stage time (stage_time, set each tick(now)); StationClient takes a
stage_clock seam and stamps + validates freshness in that /clock domain; the factory wires
stage_clock=lambda: assembly.transport.stage_time. CPU-tested: a stage-stamped prepare clears
IncorporationAckState.accept at 0 and 0.4 s skew and is refused at 0.6 s; the StationClient stamps
from the stage clock, not the passed mission time (628 passed).

### 2026-10-14 — residual trailing-clock skew (<0.3s): publish_lag past-dates station stamps

After the /clock alignment, run-66 measured stage_stamp=0.2333 == mission now=0.233 (origin~=0 in
this mission, so the domain alignment was a no-op here) yet the ack still rejected: its /clock
TRAILS the stamp at processing (ack now()=0.20 when stamp=0.233), tripping the strict 0<=now-stamp
lower bound on a ~0.03 s NEGATIVE skew -- the small station/command outruns the matching /clock
update on a separate topic. Residual skew ~0.03 s, well below Firstmate's 0.3 s host-escalation
threshold, so handled on this branch: StationClient.publish_lag=0.1 past-dates every command stamp
by ~6 clock periods (<< the 0.5 s window) to a /clock value the ack is sure to already hold, so
ack now() >= stamp. CPU-tested. Host ack 0.5 s contract untouched.

### 2026-10-14 — ack ACCEPTS prepare (warming) now; residual = /clock RATE mismatch (cross-lane)

run-68 (wall-paced heartbeats + publish_lag + /clock-domain stamps): the ack now ACCEPTS the
prepare -- it reaches state 'warming' -- a real milestone. It then faults 'station process or
parked heartbeat lost'. Measured root cause: a CLOCK-RATE mismatch at the station boundary.
- The ack's now() (its /clock) advances at ~1x realtime: station/status stamps 0.116 -> 5.2 over
  ~5.15 s wall (0.05 per 50 ms timer tick).
- The mission's stage clock (native_transport stage_time = tick*dt, which the heartbeats stamp
  with) advances at ~0.03x: dt=0.0167 per ~0.58 s render-bound mission step (warm ran 300 ticks
  over 173 s wall).
So the ack's SIM heartbeat guard `now - last_heartbeat > 0.5` trips ~0.6 s wall after the prepare:
the ack's now races ahead at 1x while each heartbeat only bumps last_heartbeat by the slow stage
step. Wall cadence is fine now (station/command every ~0.4-0.8 s wall, was ~4 s). The ack's /clock
is therefore NOT native_transport's render-paced stage time -- it advances at realtime, from a
source outside this lane. This is the /clock-source / heartbeat-bound contract call reserved for
Firstmate + the host lane. Options: stamp heartbeats in the ack's reported-clock domain (track the
ack's now from its replies), or the host bounds the sim heartbeat against a render-paced /clock,
or native_transport's /clock becomes the ack's single sim-time source.

### 2026-10-14 — /clock DIAGNOSTIC (Firstmate msg157): no 2nd publisher; host gets NO /clock

Both msg157 hypotheses refuted; the msg153 relay fallback is the fix.
(1) stage_time is NOT fictional. CLOCK_DIAG per mission tick: stage_time vs world.current_time
    tick1 0.0167/0.0667, tick21 0.350/0.400, tick61 1.017/1.067, tick181 3.017/3.067 -- a constant
    ~0.05 offset, BOTH render-paced (~0.033x: wall 3->105 s while sim 0->3). Isaac's own sim time
    also crawls; there is no 1x Isaac clock. So do NOT remove render-lead/publish_lag.
(2) No second /clock publisher. From the host overlay env, `ros2 topic info -v /clock`:
    Publisher count: 0; Subscription count: 18 -- ALL host nodes (move_group x10, cumotion_planner,
    nvblox_node, object_attachment, object_segmenter, restock_incorporation_ack, restock_manipulation,
    restock_payload, static_planning_scene_server). `ros2 topic hz /clock` and `echo --once` x3
    received ZERO Clock messages. So the host gets NO /clock at all.
Unified cause: native_transport's OmniGraph ROS2PublishClock is invisible to the host (0 publishers)
-- the same OmniGraph-publisher-to-host-DDS problem the control subprocess fixed for job/sensors/
station-command; the Isaac-process DDS pins did NOT fix /clock. With no /clock, the ack's
use_sim_time clock falls back to WALL time (~1x) -- the '1x ack now()' seen earlier -- while the
station stamps are render-paced stage_time -> the 0.5 s sim heartbeat guard trips. FIX = the msg153
fallback: relay /clock (native_transport stage_time) [+ /tf, /tf_static, /joint_states for the grasp]
through the host-DDS subprocess so the host shares one render-paced sim clock. (CLOCK_DIAG print left
in isaac_port.advance, gated %20, as a standing diagnostic.)

### 2026-10-14 — relay /clock + /tf + /joint_states through the host-DDS subprocess (msg158)

Firstmate go: OmniGraph /clock/tf/joints are invisible to the host (run-69: 0 /clock publishers),
so relay the render-paced sim-time backbone out-of-Isaac. control_ipc.serve is now type-aware
(TOPIC_TYPES): String for job/sensors/station-command; rosgraph_msgs/Clock for /clock;
sensor_msgs/JointState for /restock_sim/joint_states; tf2_msgs/TFMessage for /tf. NativeTransport
spawns its own HostControlPublisher (topics /clock,/tf,/restock_sim/joint_states) under the host
overlay runtime_env and, each tick, ships: /clock={t:stage_time}; /joint_states from the
articulation (dof_names+positions); /tf as each tf_path relative to ur_base_link via USD XformCache
(ExtractTranslation/ExtractRotationQuat, frame_ids = prim names, matching the old OmniGraph tree).
physics_setup passes robot + runtime_env. CPU-tested: typed (topic,JSON) framing round-trips,
TOPIC_TYPES + _secs_to_time (633 passed). WALL-vs-SIM note (msg158): with the host on the
render-paced /clock, host sim-time timeouts stretch ~30x in wall; the ack's 0.5 s guards become ~15 s
wall (fine), and its 1 s WALL heartbeat guard is still met by the wall-paced station heartbeats.

### 2026-10-14 — /clock relay WORKS; residual = slow RETURN path (station/status) -> seq stalls

run-71 (unique node names): host `ros2 topic info -v /clock` = Publisher count 1, node
restock_transport_pub -> the /clock(+tf+joints) relay now reaches the host. The ack ACCEPTS the
prepare: /station/status goes idle -> warming (epoch 1, seq 0, parked_since 0.133). It then faults
'station process or parked heartbeat lost' with sequence STILL 0 (host echo). Per Firstmate msg159
(b)/(c): sequence stalled => no heartbeat was ever ACCEPTED. The ack's process_alive is hardcoded
True, so the trip is the 1 s WALL heartbeat guard. Root cause: the StationClient receives the ack's
/station/status reply only via the OmniGraph subscriber, which delivers slowly/intermittently (it
DID get the final 'failed' snapshot -- the mission reason is the ack's reason -- but not the
'warming' reply in time). Until self.latest is set, StationClient.tick re-sends PREPARE (seq 0),
which the ack ignores (epoch==, seq<=); it never switches to heartbeats, so no accepted command
updates wall_heartbeat and the 1 s wall guard trips ~1.7 s after the prepare. The return-direction
OmniGraph SUBSCRIBERS are as unreliable as the publishers were. FIX = mirror the forward relay:
relay the RETURN topics host->Isaac (station/status, manipulation/status, manipulation/stop,
gripper/enable) through a SUBSCRIBING host-DDS subprocess so the StationClient/IsaacRosGraph read
them promptly (StationClient then switches to heartbeats before the ack's 1 s guard).

### 2026-10-14 — fix#1 WORKS (ack accepts heartbeats, sustained warming); residual = 1s WALL guard + collision_time

run-72 (merged PR58; fix#1 heartbeat discipline; /clock+tf+joints relay; unique node names). Big
progress: the ack ACCEPTS heartbeats now -- /station/status sequence advances 0->3->6->...->22 and
state holds 'warming' across many snapshots (0.25..0.40+ ack-sim), vs the single 'warming' before.
/tf reaches the host (topic info Publisher count 2, incl restock_transport_pub). PR58 names the
fault exactly: 'parked heartbeat lost: wall gap 1.85 s' -> the WALL guard (c), a single ~1.85 s
render tick (first collect capture at MANIP_START) exceeded the ack's 1 s wall gap between accepted
heartbeats; the StationClient cannot send a heartbeat mid-render-tick. This is the wall-bound host
timeout Firstmate flagged (msg158) -- it does NOT stretch with the render-paced /clock. collision_time
stayed -1: during the ~5 s-wall warming the ack never completed an ESDF poll (no 'incorp reject'
lines; nvblox only reached 'Allocating space for last depth view'), so warm hit its cap and the
capture saw no clearance evidence. Two residuals: (a) relax the ack's 1 s WALL guard for render
spikes (host lane) OR a mission-side background heartbeat thread (decoupled from mission ticks);
(b) why collision_time never advances (ack endpoints-readiness / ESDF poll; host lane).

### 2026-10-14 — one-clock stamps in + ack survives (warming); collision_time still -1 (host-lane b)

run-73 (merged PR59 ack wall guard 3.0 s; one-clock stamps; heartbeat bracketing). The ack now
SURVIVES: /station/status reaches 'warming' and holds -- NO more 'parked heartbeat lost' (the 3.0 s
wall guard + the before/after-capture heartbeats fixed msg162 (a)). One-clock stamp fix verified:
BRIDGE_STATS now=0.2833 (= station stage_clock = native_transport.stage_time = world.current_time)
distinct from rendering_time=0.2500, and the /tf relay + masked depth both stamp that stage value
for the tick, /tf emitted before the depth (in isaac_port.advance, ahead of _pump). BUT collision_time
stays -1: nvblox still does not integrate an ESDF the ack can read. host-stack.log is SIGKILL-
truncated at the first 'Allocating space for last depth view' (19:53:05), so the runtime nvblox/ack
logs are lost. Mission reason 'host process/status watchdog' (IsaacRosGraph 5 s manip watchdog) +
/manipulation/status 'no initial sensor frame' = ros_host got the job but no /sensors (collision_time
never fresh) and its status returned too slowly over OmniGraph. Candidate b cause (host lane):
nvblox's TF lookup may still drop on ARRIVAL ORDER -- /tf (transport relay subprocess) and depth
(masked-depth subprocess) are independent DDS participants, so a depth can reach nvblox before the
matching /tf for the same stamp, failing the lookup even though publish order + stamps are correct.
Residuals: (b) host lane re-diagnoses run-73 nvblox (does the lookup resolve + integrate now?); and
fix#2 (return relay for /manipulation/status) for the IsaacRosGraph 5 s watchdog once /sensors flow.

### 2026-10-14 — BREAKTHROUGH run-74: nvblox integrates, collision_time READY, carton captured

The one-shared-stamp fix (msg168) resolved the permanent non-integration. RESTOCK_NVBLOX_DIAG run-74
evidence: nvblox 'Received request for ESDF ... Successfully wrote requested ESDF to MultiArrayMsg'
(450.24, 451.98, 455.49 s). collision_time ADVANCES: STATION_STATS tick=51 collision_time=1.0167
ready=True, snapshot state 'ready' (the StationClient receives the ack reply via the return relay).
COLLECT_CAPTURE calls=1 sku=juice -> the fixed-source capture produced the carton target and
/sensors flowed; /manipulation/status reached 'running'/'observe'. So nav -> warm -> collision_time
fresh -> ack ready -> carton captured -> /sensors ALL WORK now.

Remaining blocker: ros_host's 1 s per-frame SENSOR watchdog (RosTransferBridge.watchdog_seconds=1.0,
armed after the first /sensors frame). The /sensors cadence is render-paced: the ESDF polls are
1.73 s then 3.51 s apart (full-res source render + masked depth + capture per collect tick), so the
1 s watchdog trips -> mission fails 'sensor stream watchdog expired'. The StationClient cannot emit
/sensors mid-render-tick (it is the carton packet, one per capture). Per msg155 this watchdog is the
host lane's to widen (do NOT relax on this branch); measured gap ~1.7-3.5 s, so it needs ~4-5 s (cf.
the ack wall guard already widened to 3 s). Everything up to the grasp dispatch now works.

### 2026-10-14 — run-75/76: sensor watchdog fixed; collect stalls in observe (ROOT CAUSE: no cart target)

PR61 (node-clock sensor watchdog) HELD: run-75/76 never tripped "sensor stream watchdog expired".
Host stack came up fully (cuMotion + ESDF service + move_group "You can start planning now!"). The
fixed-source capture produced the carton target on every collect tick (sku=juice_orange, pose
(0.0,1.101,0.261), dist 1.13 m). But both runs sat in host `phase=observe reason=""` for the whole
run and died on the Isaac 5 s `host process/status watchdog` — the mission's own
IsaacRosGraph.status watchdog, firing because the controller never advanced past observe so
`/manipulation/status` stopped progressing (a render-paced status gap eventually >5 s).

ROOT CAUSE (by inspection, reproduced from the URDF): the controller observe gate
(`controller.py:132-142`) requires BOTH a unique source carton observation AND `s.target` — the
destination placement `Target` (for collect, destination = the cart `rollwagen`).
`CollectTargetBridge.capture` returns `Sensors(..., target=None)`: the source-camera refactor
provides the source carton observation but DROPPED the destination Target that the old
`IsaacCartonSensors.capture` built (`isaac_sensors.py:194`). So the gate loops forever on the
missing cart target. All three observe early-returns are indistinguishable (`observe`/`""`) in the
status stream, so a throttled `OBSERVE_GATE` diagnostic was added (commit de6ce4a) to confirm
`target=None` vs a source-match drop on the next run.

SECONDARY (never reached): `roi=None` on every capture. The reprojected grasp-target bbox is used
ONLY by the host `object_segmenter` for depth-masking — NOT for planning (the planner plans from
the carton pose). And it is structural, not a math bug: the so-called "arm eye-in-hand" camera is
actually BASE-front-mounted — `front_camera_frame` is a fixed joint off `base_link` at (0.46,0,0.35)
looking +X (authored next to lidar/imu in `mobile_assets.py:118`). The grasp is a +Y SIDE grasp
(scenario `grasp_mode "side"`), so the carton sits ~1.1 m to the camera's side/behind
(`carton_in_cam=(-1.101,-0.171,-0.460)`, derived exactly from the URDF chain: ur_base_link =
base_link + (0,0,0.26), no rotation; carton at ur_base +Y 1.1 m = base +Y side). No wrist/side
camera exists. Diagnostic added as commit 21a0f9b.

OPEN DECISION (escalated to Firstmate — alters mission product behaviour, not taken unilaterally):
(Q1) collect placement model — place on the staged (uncoupled) cart, needing a cart placement
Target (sensed, or derived from the known cart asset pose: scenario `cart_load_center_body_xy` +
`cart_support_z_world`), OR collect = pick+hold (and the observe gate must not require a destination
target for the source-camera path). (Q2) depth-masking — the side carton isn't in the base camera's
forward depth so it isn't an obstacle; masking is arguably a no-op and the grasp can plan from the
pose (relax the collect roi/bbox requirement), vs adding a wrist/side camera or re-orienting the
approach (robot-asset / scenario lane). Transfer is the MIRROR case (cart = SOURCE, shelf =
destination via PR62 shelf_target), so the same cart-target provider question applies there.

### 2026-10-14 — collect reaches the grasp; seam = KDL/cuMotion self-collision model disagreement

Sequence of landed fixes drove collect all the way to the grasp plan:
- cart placement Target (derived from the cart asset pose) + obs stamped with packet `now` -> the
  host observe gate advances (matches=1, target=rollwagen). run-79: observe->APPROACH (first ever).
- PR64 arm-camera look-at re-aim + 424x240@75deg collision intrinsics wired into build_sensing ->
  the carton is framed, roi non-None (e.g. (124,107,292,159)); mask path self-re-enables.
- native_transport now also relays base_link<-ur_base_link each tick -> move_group (plans in the
  URDF root base_link) can transform the grasp collision object (run-80: "base_link ... does not
  exist" is gone).

CURRENT SEAM (run-80): the approach plan fails `collision planning or execution failed` ->
`host emergency stop`. host-stack.log: `MoveGroup motion planning failed` /
`Invalid c-space position: self collision detected. Failed call to 'planToCSpaceTarget()': target
c-space position [[5.1284 -4.44626 2.48316 -1.17849 1.1506 -3.14159]] is invalid.`
RosMotionPort.begin computes IK via move_group GetPositionIK (KDL, avoid_collisions=True, SRDF
model) then plans to that c-space with cuMotion. KDL returns an awkward wrapped branch (joint2 ~
-4.45 rad) that its SRDF model allows but cuMotion's XRDF self-collision model (host-cad/mobile.xrdf,
a provided asset, not git-tracked) flags -> cuMotion rejects the pre-IK'd target. The grasp POSE is
SANE/reachable (computed offline from WallCartonPolicy: approach (0,0.778,0.181) dist 0.80, contact
(0,0.898,0.181) dist 0.92 -- well inside max_reach 1.3). So the pose is fine; the two collision
models disagree on which self-collision pairs to ignore (cuMotion couldn't apply several SRDF
disable-masks -- warnings "frame X has no self-collision spheres specified").

CROSS-LANE CONTRACT (host/robot lane, resumes 11:00): either (a) let cuMotion do collision-aware
IK+planning to the POSE (switch RosMotionPort from GetPositionIK->planToCSpaceTarget to a pose-goal
MoveGroup request) so one consistent model is used, or (b) align the XRDF self_collision.ignore with
the SRDF disable_collisions so cuMotion ignores the same arm-base pairs MoveIt does. Repro: c-space
target above; grasp pose approach (0,0.778,0.181) quat (0,-0.71,-0.71,0) in ur_base_link.

### 2026-10-14 — pose-goal planning (a) WORKS; residual = XRDF over-conservative self-collision (b)

run-81 (pose_goal=True): the MoveGroup pose-goal path is correct architecture — cuMotion logs
`Using task space goal` and does its OWN collision-aware IK (one model, the XRDF), no KDL c-space
pre-IK. It fails `trajopt: INVERSE_KINEMATICS_FAILURE` on the approach pose (0,0.778,0.181) quat
(0,-0.71,-0.71,0), which is well inside reach. Root: cuMotion's XRDF self-collision model checks
STRUCTURAL arm-mounting pairs that are always in contact, so EVERY IK branch is flagged. The SRDF
disables them but cuMotion could not apply the masks (host-stack.log "...has no self-collision
spheres specified. Self-collision mask will not be added") for:
  ur_shoulder_link <-> arm_mount / base_link / hitch
  ur_base_link_inertia <-> arm_mount / base_link / front_caster_fork / hitch / left_wheel /
                           rear_caster_fork / right_wheel
Fix (b), host/robot lane: add these pairs to host-cad/mobile.xrdf `self_collision.ignore` (align it
with the SRDF disable_collisions). They are structural (arm bolted to the mount; base_link_inertia
coincident with the chassis), so ignoring them is correct, not a workaround. (a) stays either way.
Pose-goal fix is behind the ros_host_config `pose_goal` flag.

### 2026-10-14 — outbound traction investigation (CPU): prime suspect is missing wheel-ground friction

PR70 nav-towing probe: ~0.3 m in 8 min (~0.6 mm/s) = near-stall, "traction-limited". CPU audit of
the drive/physics config on this branch:
- Base drive (mobile_runtime.py MobileController): velocity drive on left/right_wheel_joint,
  stiffness 0, damping 100, **maxForce 80 N·m**, target velocity set from the controller. 80 N·m /
  0.16 m wheel radius = ~500 N tractive force PER wheel (~1000 N pair) — ample for 180 kg base +
  35 kg cart (mobile.py spec.mass 180, cart_mass 35), so torque is NOT the bottleneck.
- Cart body damping (configure_cart_bearings): linearDamping 0.5, angularDamping 3.0 on all 9 cart
  bodies. Linear damping force = 0.5 * 35 kg * v ≈ 1.75 N at 0.1 m/s — negligible. (Tuned for REST
  stability per AGENTS.md; not the tow bottleneck.)
- **No wheel-ground PhysicsMaterial is authored anywhere** (physics_setup / store_environment /
  mobile_assets) — the drive-wheel↔floor contact uses the PhysX DEFAULT friction (~0.5). For a
  215 kg robot+cart that is almost certainly below the slip threshold, so the drive wheels spin
  without advancing: exactly "traction-limited".

RECOMMENDED config-level tune (ready to include in the next full run): author a high-friction
PhysicsMaterial (static ~1.0, dynamic ~0.9, restitution 0, combine=max or multiply) on the two
DRIVE wheel colliders (paths left_wheel/right_wheel) in physics_setup, so wheel↔floor traction rises
above the tow demand; keep the caster wheels and cart damping as-is (rest stability). If still slow
after friction, raise MobileController maxForce (80 -> ~150 N·m). NOT applied blind: it changes tow
physics and should be validated in a run (and the navigation lane, resuming 11:00, owns deeper tow
dynamics). Masses/damping left unchanged (physically faithful).

### 2026-10-14 — run-82 (final tonight): seeded IK retry works; two residual seams

pose_goal=False + hardened KDL->cuMotion joint-space path. Furthest stage: COLLECT/approach. The
seeded-IK retry IS exercised — 5 MoveGroup goals attempted across seeds (vs run-80's single wrapped
branch), so the stow-first seeding + nearest-branch normalise + retry-next-seed all run. BUT cuMotion
rejected every seeded c-space target (>=1 "Invalid c-space position: self collision detected",
"MoveGroup planning completed with success: false"). So the grasp residual is NOT branch selection
(now handled) — it is cuMotion's self-collision model rejecting the arm-mount region for this XRDF,
i.e. the same cuMotion/XRDF issue the host verdict flagged for NVIDIA-side work at 11:00.

Two residual seams for the next session:
1. **Isaac status watchdog too tight for the retry.** Mission failed `host process/status watchdog`
   (IsaacRosGraph 5 s WALL) during the render-paced ~5-attempt IK/plan retry. This is the status
   analogue of the PR61 sensor-watchdog fix: widen / measure it in node-clock (sim) time so the
   retry has room. On this branch (scripts/twin/restock_manipulation/isaac_ros_graph.py).
2. **Wheel-friction material bound 0 colliders.** `WHEEL_TRACTION static=1.0 bound=0` — the
   name-search for 'left_wheel'/'right_wheel' collider prims found none (wheel colliders are named/
   nested differently in the composed USD). Fix the prim lookup so the traction tune actually binds;
   outbound was not reached this run so it was untested regardless. No tow speed, no video.

Grasp blocked on host-lane cuMotion/XRDF (11:00); parking per Firstmate (allowance gate).

### 2026-10-15 — three blockers cleared; integrated host XRDF/wrist/return PRs; full run-83 launched

All three run-82 residual seams and the grasp blocker are fixed and on the integration branch
(PR https://github.com/ailerobotics/blender-supermarket/pull/15):

1. **Status watchdog node-clocked** (`isaac_ros_graph.IsaacRosGraph.poll`). The
   `/manipulation/status` watchdog measured its 5 s bound in WALL time, so a render-paced cuMotion
   multi-seed IK retry (many wall seconds, tiny sim advance per render tick) false-tripped and killed
   the mission mid-grasp (run-82 seam 1). It now measures the 5 s bound in NODE-CLOCK (sim) time when
   the caller supplies it (`ros_manipulation.step` passes `simulation_time`), unchanged semantics on
   real hardware, with a generous wall backstop for a frozen sim clock. CPU-tested (4 new cases in
   `test_restock_ros_split.py`: render-paced no-false-trip, sim-time trip, wall backstop, legacy).

2. **Wheel-traction collider bind fixed** (`physics_setup.configure_wheel_traction`, run-82 seam 2,
   `WHEEL_TRACTION bound=0`). Two faults: (a) it searched from the articulation root (`base_link`) but
   the drive wheels are its **siblings** under the robot root `/MobileRobot`; (b) the Isaac URDF
   importer references each collider under `<wheel>/collisions` with `instanceable=True`, so the
   `UsdPhysics.CollisionAPI` cylinder is an **instance proxy** a default `Usd.PrimRange` skips and a
   material cannot be bound onto. Now searches from the robot root and de-instances each drive wheel's
   collisions subtree before binding; **verified against a real imported mobile USD** that both drive
   wheels bind. CPU regression test added (synthetic instance-proxy layout).

3. **cuMotion/XRDF phantom self-collision (grasp blocker) cleared by the host lane (PR 72)** — five
   arm-mount pairs are excluded via `MANUAL_EXCLUSIONS` in `generate_assets.py`. Regenerated
   `host-cad/mobile.xrdf` with the generator (1870 spheres, 0 transport overlaps); confirmed all five
   pairs now in `self_collision.ignore`. Switched the mission grasp back to **`pose_goal=true`**
   (`ros_host_config.json`, the config `restock_host/launch.py` feeds `ros_host`) — the host lane
   proved the exact run-81 collect approach pose now plans a 244-point cuMotion **task-space** trajectory.

Also integrated from the host/camera lanes (all into this branch): the eye-in-hand wrist RGB-D camera
(PR 71), and the VERIFY/RETURN release/detach fix + 65-point return plan (PR 69, clears the carried-unit
planning-scene persistence that failed the earlier return). Main is already an ancestor of the branch.

Secondary integration fix in my mission lane: `ros_manipulation.step` now treats a returned packet's
integrity (stale/repeated timestamp, non-sensor provenance) as **fail-closed** outside the soft capture
try — a fixture or replayed packet stops the mission instead of degrading to `running` (was a latent
branch-local test failure). After the camera lane landed PR 73 (threads the steady-clock
acquisition-to-use freshness gate through `IsaacCartonSensors.capture` + `StationSensors`) the two
former `restock_camera` freshness failures are fixed; combined with the sensing fix below the **full
CPU suite is green: 771 pass, 0 fail, 10 skipped, 28 subtests**.

**run-83** (first full `collect`+`couple` attempt with all fixes active, fresh commission run-38 of
`backstock_00` at 2560×1920 / 900 s baseline, `wheel_friction=1.0`): the mission **failed during bundle
construction**, before any physics step — a regression surfaced by integrating the wrist camera (PR 71):

```
ValueError: camera 'wrist' has no housing/bracket collision solid in the design
  (sensing_sources._real_rig, run_mobile_restock -> isaac_bundle.create -> build_sensing_sources)
```

Root cause: PR 71's `restock_camera.config.camera_config()` adds the eye-in-hand `wrist` collision
camera (frame `restock_camera_wrist`, `role='collision'`, parented to the ur_tool0 link) to
`design['cameras']`, but the fixed-station `CameraRig` builder in integration-owned
`sensing_sources._real_rig` required every design camera to carry a housing/bracket collision solid —
which an eye-in-hand camera authored by its own path (`mobile_assets.author_wrist_camera` /
`native_cameras.initialize_collision_camera`) does not have. Fix (integration lane): extracted a pure
`station_camera_specs(design, parent_paths)` that builds specs only for cameras parented to a fixed
station parent (`map`/`cart_chassis`) and **skips eye-in-hand cameras** — the sensing
supports/context/location maps never reference the wrist camera, so nothing downstream expects it in
the rig. A station camera still missing its solid fails closed. CPU-tested
(`test_sensing_sources.py::StationCameraSpecsTests`). The run reached a clean Isaac boot (RTX up, host
stack warmed, ROS bridge loaded, gripper + physics scene composed) in ~24 s before the construction
error, so the boot/compose path is healthy; re-running as run-84 after this fix + pulling the camera
lane's PR 73.
