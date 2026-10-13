# Mobile restocking integration checkpoint

Status: incomplete. Guarded GPU use was explicitly authorized on October 13;
native RTX RGB-D rendering and VFR recording now pass. No autonomous physics cycle or mission MP4 has
been produced. The current branch contains CPU mission/inventory infrastructure;
the landed mobile/cart, schema-2 environment, articulated navigation and CAD
manipulation components are integrated from origin/main.
A controller dispatcher and guarded launcher now exist; a complete simulator
bundle with actual sensor acquisition/planner/actuator wiring is still missing.

## Latest passive-observer and CPU checkpoint

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
