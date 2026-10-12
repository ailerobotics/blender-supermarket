# Mobile restocking integration checkpoint

Status: incomplete, simulator execution held pending explicit release. No autonomous physics cycle or MP4 has
been produced. The current branch contains CPU mission/inventory infrastructure;
the landed mobile/cart, schema-2 environment, articulated navigation and CAD
manipulation components are integrated from origin/main.
A controller dispatcher and guarded launcher now exist; a complete simulator
bundle with actual sensor acquisition/planner/actuator wiring is still missing.

## Implemented and tested

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

265 combined CPU tests pass with three dependency skips after the landed camera/navigation corrections, with the installed Isaac USD libraries exposed
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

The simulator hold remains explicit pending firstmate's release and matching
runtime environment. Preserve live services and the protected-Blender/resource
guard; no driver installation, reboot, service restart or guard bypass occurred.

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
  safeguards are CPU-tested with mocked probes. **No new GPU probe was run after
  firstmate's instruction to preserve the accepted driver diagnosis.**
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
GPU planner/mapper startup. Camera and navigation corrections are now merged;
host launch code is still awaiting landing. Remaining inputs are exact CAD
collision assets, a current-scenario camera configuration with sensed support
and clearance, camera housing collision validation, mapper reset/incorporation
acknowledgements, and the complete sensor-binding mission factory. Camera work
currently reports loaded-support occlusion and insufficient shelf support ROI;
these must be resolved with actual sensing, not enlarged or invented free space.

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
