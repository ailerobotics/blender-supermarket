# Isolated restock host planner and mapper

`scripts/run_restock_host.py` prepares the existing Jazzy MoveIt + NVIDIA stack
for the mobile CAD robot. The package under `scripts/twin/restock_host/` imports
without ROS and starts nothing. It does not launch the simulator or a hardware
driver. Execution still requires the owner's GPU decision and integration's
resource/Blender guards. CPU validation is not an autonomous physics demonstration.

## Configuration and commands

The launch consumes expanded, generated CAD files. Supply a JSON object with
absolute paths (these generated files now exist in the task dataset):

```json
{
  "urdf": "/home/manishsaini/datasets/sim-restock-host-v2/refined/mobile.urdf",
  "mobile_manifest": "/home/manishsaini/datasets/twin-robot-in-store-boot-v1/mobile/mobile_manifest.json",
  "srdf": "/home/manishsaini/datasets/sim-restock-host-v2/refined/mobile.srdf",
  "xrdf": "/home/manishsaini/datasets/sim-restock-host-v2/refined/mobile.xrdf",
  "asset_report": "/home/manishsaini/datasets/sim-restock-host-v2/refined/asset-report.json",
  "shelfbot_root": "/home/manishsaini/workspaces/shelfbot_newgripper",
  "workspace_min": [-2.0, -2.0, -1.0],
  "workspace_max": [2.0, 2.0, 2.0]
}
```

Bounds are parameterized metres in `ur_base_link`; integration must choose them
from the actual parked station's sensed workspace. These example bounds are not
clearance evidence. Domain 88, `/restock_sim`, `ur_base_link`, `gripper_pad_tcp`
and the six ordered `ur_*` arm joints are enforced. SRDF requires the
`ur_manipulator` chain from arm base to CAD TCP. XRDF must include generated
world/self-collision spheres, CAD suction-pad and arm-link coverage, acceleration
and jerk limits. Placeholder shelfbot files are rejected. Required schema-2
coverage evidence is recomputed against the current CAD, meshes and manifest;
syntax validation or output hashes alone cannot establish coverage.

From the repository root:

```bash
source /opt/ros/jazzy/setup.bash
source /home/manishsaini/datasets/sim-restock-host-v2/setup.bash
export ROS_DOMAIN_ID=88 ROS_LOCALHOST_ONLY=1
export ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST ROS_STATIC_PEERS=''
/usr/bin/python3 scripts/run_restock_host.py --config /absolute/host.json --plan
/usr/bin/python3 scripts/run_restock_host.py --config /absolute/host.json --check
PYTHONPATH=scripts /usr/bin/python3 -m pytest -q scripts/twin/tests/test_restock_host.py
```

`--plan` validates robot/configuration inputs and prints exact endpoint types,
node allowlist, environment and integration-host arguments. `--check` additionally
checks installed package versions, configuration files, component exports, ELF
dependencies with `ldd`, and compiled message typesupport imports. Exit 2
identifies missing dependencies; exit 0 means only
these static checks passed. No rclpy node, mapper or planner is started by either.
Preserve the sourced `PYTHONPATH` if combining ROS checks with a custom one.

After compute authorization, and only under integration's resource guard:

```bash
/usr/bin/python3 scripts/run_restock_host.py --config /absolute/host.json --launch
```

The foreground launch owns its children; stop it cleanly before moving the base.
It starts MoveGroup, real cuMotion/static-scene/attachment components, NVIDIA
robot segmentation, the existing shelfbot object segmenter and nvblox. No saved
map is loaded. It sets sim time and fixed task isolation, rejects conflicting DDS
domains/profiles/peers, disables automatic default FastDDS XML loading and uses
UDPv4 loopback discovery. It does not start a hardware driver, robot state
publisher, fake controller or simulator. Actual measured TF and joints come
from integration. No global ROS/Python configuration is changed.

The separately owned manipulation process must receive the same generated
manifest for collision-planned stow. After the same authorization, integration
launches its existing module with `--config` pointing to its manipulation JSON,
not the host-launch JSON:

```bash
ROS_DOMAIN_ID=88 ROS_LOCALHOST_ONLY=1 ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST ROS_STATIC_PEERS='' \
PYTHONPATH="scripts:$PYTHONPATH" /usr/bin/python3 -m twin.restock_manipulation.ros_host \
  --config /absolute/manipulation.json --urdf /absolute/mobile_ur10.urdf \
  --shelfbot-root /home/manishsaini/workspaces/shelfbot_newgripper \
  --mobile-manifest /absolute/mobile_manifest.json
```

This launch does not duplicate that process or stow implementation. The emitted
`integration_host_arguments` carries the exact paths; validation checks six
manifest transport positions against URDF limits.

## Endpoint and sensing contract

All endpoints are below `/restock_sim`:

| Name | Type | Server owner |
| --- | --- | --- |
| `compute_ik` | `moveit_msgs/srv/GetPositionIK` | MoveGroup/KDL |
| `move_action` | `moveit_msgs/action/MoveGroup` | MoveGroup/cuMotion pipeline |
| `execute_trajectory` | `moveit_msgs/action/ExecuteTrajectory` | MoveGroup |
| `apply_planning_scene` | `moveit_msgs/srv/ApplyPlanningScene` | MoveGroup |
| `attach_object` | `isaac_ros_cumotion_interfaces/action/AttachObject` | NVIDIA attachment component |
| `nvblox_node/get_esdf_and_gradient` | `nvblox_msgs/srv/EsdfAndGradients` | nvblox |
| `arm_controller/follow_joint_trajectory` | `control_msgs/action/FollowJointTrajectory` | integration's measured articulation bridge |

`--probe` performs at most five seconds of graph discovery, without requesting
IK, planning, execution, attachment or ESDF computation. It requires the exact
server types, including the externally supplied trajectory server. A matching
action client alone does not pass. It has not been run against a live graph.
Graph discovery is not motion readiness.

Integration provides `/clock`, `/tf`, `/tf_static`, measured
`/restock_sim/joint_states` (including movable CAD fingers/base as needed),
registered `/restock_sim/camera/depth` and `/restock_sim/camera/camera_info`, and
sensor-derived `/restock_sim/target_bbox` (`Float32MultiArray [u1,v1,u2,v2]`).
NVIDIA segmenter emits `robot_world_depth`; shelfbot's existing mask node emits
`world_depth`, the only nvblox depth input. Sensed target boxes must describe the
current grasp target/carried target, not a fixture or USD pose. The inherited
mask node uses receipt-aged, unstamped boxes and passes depth through when stale;
it does not prove mask synchronization. Integration must gate motion on current
sensing. Broad bounding-box removal can also hide background geometry; it is
not a validated safe mask for every scene. No automatic success is inferred.

At each measured parked station, stop the host launch, move/park, restart it with
an empty mapper, and acquire new sensor data. Mapping in a moving `ur_base_link`
without this reset would mix stations. Neither this launcher nor `--probe`
implements integration's base-motion interlock. Do not move while host planning
is available. Workspace bounds and all visualization/clearing frames use the
same arm frame. Moving between stations invalidates every prior map acknowledgement.

The installed NVIDIA ESDF response `header.stamp` is explicitly the latest depth
image incorporated into ESDF. Integration may call
`incorporated_time(response, now=sim_time, parked_since=station_epoch)` after a
real ESDF request. It rejects failed, empty, nonfinite, stale, future, wrong-frame
and pre-station responses. It returns the server timestamp, never receive time.
Feed that to `Sensors.collision_time`; also require that cuMotion's per-plan ESDF
update and planning action succeed. No acknowledgement publisher is fabricated.
MoveIt scene acknowledgement, NVIDIA attachment action result and physical CAD
seal remain independent requirements. Attachment success is not a physics seal.

## Installed evidence and exact remaining dependencies

Inspected locally without GPU or networking:
`nvcr.io/nvidia/isaac/ros@sha256:0ff40ee743fbd5532a0864e4a7270c958421c5f52e863d5413a2d2c127b163a1`.
The image has Jazzy MoveIt 2.12.4 and NVIDIA cuMotion, attachment, segmenter,
nvblox and interface packages 4.4.0. NVIDIA Python 3.12 message typesupport imports
and constructing the launch description pass in a bounded CPU-only container.
No plugin was loaded or node executed. The installed C++ 4.4 API supersedes older
Python attachment launch arguments in the source examples; the task uses current
robot-description services and `cumotion/robot_description_updated` for masks.

Host Jazzy contains MoveIt 2.12.4. Its default environment lacks the NVIDIA
runtime, mapper and message overlay. A task-local extraction now supplies the
required NVIDIA packages and measured library closure under
`/home/manishsaini/datasets/sim-restock-host-v2/image-root`, with an explicit
`setup.bash` beside it. Source it only in the task process after Jazzy. No global
install, shared environment change or host-network container is used.

The broad package dependency set measured 3,912,799,347 bytes and was not
extracted. The selected runtime files measured 2,666,638,504 bytes; host `ldd`
then identified additional negotiated/GXF/VPI/glog/gflags libraries, which were
also copied from the same image. Final task dataset usage was 2,680,977,910 bytes
(decimal, below 3 GB). Host NVIDIA message typesupport imports, component-library
link checks and all ten focused tests including launch-description construction
pass after sourcing this overlay. Nothing was loaded as a GPU plugin or run as
a planner/mapper. These checks do not establish CUDA driver compatibility,
dynamic GXF graph loading or DDS operation; those remain runtime gates.

The bounded extractor, source image provenance, selected file lists, actual
library errors resolved, reproduction commands and CPU results are recorded in
`/home/manishsaini/firstmate/data/sim-restock-host-v2/report.md` and the task dataset.
The source-only PR does not vendor 2.68 GB of machine-local dependencies.

The task's actual mobile CAD SRDF/XRDF are now generated by
`python3 -m twin.restock_host.generate_assets` from the integration URDF, meshes
and transport manifest. The normalized URDF only resolves mesh paths; it does
not alter geometry. Collision-local cells cover full primitive volumes and mesh
convex hulls, including interiors. Only cells provably outside the solid/hull are
discarded; retained cells have enclosing spheres. An octree splits cells involved
in transport overlaps, bounded by `--min-pitch` and a per-element cell cap. Source
spheres remain exact. NumPy, SciPy and PyYAML are required CPU dependencies.

The actual CAD now produces 1,930 spheres and zero nonexcluded transport overlaps.
Seven additional, individually proven exclusions cover two caster/chassis pairs
(at least 7.685 mm separation for all roll/swivel angles) and five neighboring
plate pairs (0.628 mm separation throughout independent full prismatic strokes).
The three arm pairs remain enabled. Existing fixed/adjacent assembly exclusions
are unchanged. No sampled or blanket extra exclusions are generated.

`asset-report.json` schema 2 records every source/output hash, per-element cover,
refinement history and full-motion proof. Preflight reconstructs the entire report
and output hashes from current sources, rejecting missing, stale or forged evidence,
changed geometry and true residual overlaps. Both CLI and direct launch-description
construction reject static errors. The original 1,886-sphere, ten-overlap checkpoint
is preserved in the dataset root; use `refined/host.json` for the current assets.

Reproduce the assets and check without starting any ROS node:

```bash
OPENBLAS_NUM_THREADS=1 PYTHONPATH=scripts /usr/bin/python3 -m twin.restock_host.generate_assets \
  --urdf /home/manishsaini/datasets/twin-robot-in-store-boot-v1/mobile/mobile_ur10.urdf \
  --mobile-manifest /home/manishsaini/datasets/twin-robot-in-store-boot-v1/mobile/mobile_manifest.json \
  --shelfbot-root /home/manishsaini/workspaces/shelfbot_newgripper \
  --output /home/manishsaini/datasets/sim-restock-host-v2/refined
# Source the two overlays and isolation variables documented above first.
OPENBLAS_NUM_THREADS=1 /usr/bin/python3 scripts/run_restock_host.py \
  --config /home/manishsaini/datasets/sim-restock-host-v2/refined/host.json --check
```

A uniform refinement trial required 19,114 spheres; adaptive refinement uses about
90% fewer. Installed NVIDIA headers expose a default 3,000-sphere **generation
candidate budget**, not a runtime solver limit. No installed public runtime sphere
capacity was found. Raw xyzw storage counts in the report exclude native batches,
acceleration structures and GPU workspaces. Allocation, latency, native collision
behavior and trajectory feasibility remain unverified; `planning_ready` and
`native_gpu_capacity_verified` remain false even when CPU preflight passes.

Regression tests in `test_restock_collision_refinement.py` cover solid-volume
coverage, true collisions versus removable false positives, full-motion proof
counterexamples and tampered evidence. `RESTOCK_HOST_ACTUAL_CAD=1` enables the local
CAD regression; `RESTOCK_HOST_VALIDATE_INSTALLED=1` enables installed-overlay
launch-description construction. Neither flag executes a planner or mapper.

Also required: sensor bbox producer, integration trajectory server, station
lifecycle interlock and actual TF/depth transport. These are reported
dependencies, not replaced with dummy servers. The source patterns reused are
ur10-setup's `cams/cellmap/planner.launch.py`, MoveIt pipeline setup, and
shelfbot_newgripper's `launch/perception_collision_world.launch.py` plus its
`shelfbot/perception/object_segmenter.py`. Their hardware launchers are never run.

## Navigation evidence producer binding (completes the slice/incorporation gate)

The separate navigation mapper (`/restock_sim/navigation/nvblox_node`) must prove
that every published map slice was produced by a known, empty-started generation
and carries the exact incorporated-acquisition identity. The installed
`nvblox_msgs/DistanceMapSlice` cannot carry that, so the native producer
`restock_mapper_evidence::EvidenceProducerNode`
(`native/restock_mapper_evidence/`, a subclass of the installed `nvblox::NvbloxNode`)
emits a typed `restock_mapper_evidence/EvidenceSnapshot` that embeds the copied
slice next to the full per-sensor ledger. Its CPU build/verification evidence is in
`data/sim-restock-host-v2/producer-build-evidence.md`.

How integration binds it at launch:

1. Run the navigation component as `EvidenceProducerNode` (not the stock
   `NvbloxNode`) in the `/restock_sim/navigation` namespace with node name
   `nvblox_node`, so the canonical topic names are unchanged. Set evidence
   parameters: `evidence.session/scene/cameras/localization` (the MapIdentity
   revisions), `evidence.producer_nonce` (the unique empty-generation id — this is
   the `generation` the consumer's `TemporalMap` is constructed with),
   `evidence.steady_clock_domain` (the shared `CLOCK_MONOTONIC` domain string),
   `evidence.slice_height`/`evidence.unknown_value`, and
   `evidence.sensors:=['front=front_optical','rear=rear_optical', ...]`.
2. Remap the base raw slice output (`static_map_slice`) to a dead internal topic
   so only the certified envelope on `~/evidence_snapshot` carries geometry.
3. Commission the generation over the services: `~/empty_generation` (measures all
   maintained native layers + queues + watermarks with admission CLOSED and
   returns the certificate), then `~/open_admission` (rechecks emptiness, echoes
   the accepted acquisition cutoffs, admission OPEN — one-shot), then
   `~/register_acquisition` per registered depth+CameraInfo before its frame is
   integrated.
4. The consumer decode is `twin.restock_host.evidence_bridge.envelope_from_message`
   (reference): it maps the native message to navigation's
   `twin.navigation.temporal.EvidenceEnvelope`, binding the embedded slice to the
   **newest** ledger acquisition and converting stamps to float seconds. Feed it to
   the landed `TemporalMap.accept_envelope`; the consumer republishes only the
   validated embedded slice to the canonical Nav2 topic.

`full_height_observation_verified` is always false here: provenance/emptiness only.
The runtime binding (real frames advancing only their own ledger, GPU-copy/TF
failures stopping, restart rejecting old output) is an integration GPU gate, not
proven by this CPU build. Contract tests against the landed consumer are in
`scripts/twin/tests/test_restock_host_evidence_binding.py`.

### Task-local runtime closure + guarded GPU construction

The stock task env resolves CUDA but lacks the Isaac GXF/nitros ament closure the
`NvbloxNode` constructor requests at runtime (`gxf_isaac_gxf_helpers` + its
transitive packages), so the node constructor fails. `scripts/restock_runtime/`
holds the recipe: `extract_runtime_overlay.sh` extracts the complete image
`/opt/ros/jazzy` into a task-local overlay, `offline_closure_proof.sh` proves the
closure offline (ament resolution + `ldd -r`), and `runtime-env.sh` layers the
overlay + the built producer onto `nav-plugin/env.sh`. `scripts/launch_evidence_mapper.py`
loads the mapper as `EvidenceProducerNode`; `scripts/run_evidence_mapper_gpu.py`
runs it under the exclusive GPU guard (flock `/tmp/aile-gpu.lock`, protected-Blender
refusal, ≥8 GiB VRAM / ≥12 GiB RAM, domain-88, ≤5 min, exact-PID teardown) and
measures construction + evidence-interface advertisement. Measured evidence and
honest limits (a bound-slice still needs the integration sensor+handshake driver):
`data/sim-restock-host-v2/runtime-closure-report.md`.

### Empty generation is construction-baseline-aware (not zero-allocated-blocks)

A freshly constructed nvblox layer is **non-empty from construction**: the producer
reports 2048 preallocated blocks in every maintained layer (tsdf/esdf/color/freespace/
occupancy) before any sensor input, because `numAllocatedBlocks()` reports nvblox's
block pool, not sensor occupancy. "Empty" therefore means **zero sensor-observed
voxels, never zero allocated blocks**. `EmptyGeneration` measures the per-layer
allocation once at construction as a baseline (`State::set_construction_baseline`) and
`verify_empty` refuses only on **growth beyond** that baseline (plus the non-empty
queue / integration watermark / ledger / pending checks). An emptiness refusal is
recoverable — it reports the exact per-layer `LayerCount[]` in the FAILURE response and
does NOT fault the one-shot producer (only genuine protocol violations do), so a
diagnostic probe never bricks the map. A genuine refusal after real depth integration
still refuses (the integration watermark trips `empty evidence incomplete`). Verified on
GPU in one bounded guarded job (`scripts/verify_empty_baseline_gpu.py` +
`scripts/restock_runtime/empty_baseline_probe.py`): accept on a fresh construction with
the 2048/layer baseline reported, refused after one synthetic depth frame actually
integrated. Report: `data/sim-restock-host-v2/empty-baseline-report.md`.

### Published slice carries the global frame (consumer acceptance)

`processEsdf` fills `env->slice` through `distanceMapSliceMsgFromSliceImage`, which sets
only the slice geometry and leaves the message header default. The producer therefore
sets `env->slice.header.frame_id` to the mapper `global_frame` (read from the base node
parameter, default `map`) and the stamp to the newest incorporated acquisition's sim
time — mirroring the stock nvblox publish path. Without the frame the landed consumer
rejects the snapshot as an `invalid native distance slice` (empty frame), and
`accept_envelope` requires `slice.frame == 'map'`. Covered by the contract tests in
`scripts/twin/tests/test_restock_host_evidence_binding.py` and verified end to end on GPU
(`scripts/verify_slice_consume_gpu.py` + `scripts/restock_runtime/consume_snapshot_probe.py`):
the full pipeline publishes an EvidenceSnapshot whose slice frame is `map` and the landed
`TemporalMap.accept_envelope` adopts it. Report: `data/sim-restock-host-v2/slice-frameid-report.md`.

### Task-isolated planner + mapper launch and VRAM fit

`scripts/restock_runtime/launch_restock_host_stack.sh` is the one task-isolated recipe
that starts the whole host manipulation graph — MoveIt `move_group`, the Isaac cuMotion
servers, the shelfbot `object_segmenter`, and the nvblox mapper co-launched — on DDS
domain 88 under the task-local runtime closure, never the rig's shared containers. Two
load-time constraints are encoded in `launch.py` and `generate_assets.py`: the generated
xrdf `default_joint_positions` names **only the six cspace arm joints** (cuMotion/cuRobo
prunes off-arm-path branches such as the base's two-joint caster chains, and a default
position for a pruned joint aborts the planner at load — `test_restock_host.py` locks
this), and the composable nodes load **mapper before planner** (the `CumotionPlanner`
constructor blocks on the mapper's `get_esdf_and_gradient` service, so the reverse order
deadlocks the container). Verified in one bounded guarded GPU run
(`scripts/verify_restock_host_stack_gpu.py` + `scripts/restock_runtime/stack_probe.py`):
6/7 servers up (the depth self-filter `RobotSegmenter` needs the real NITROS depth
input), all stack endpoints advertised, `compute_ik` succeeds against the refined CAD
config, the mapper serves an ESDF+gradient grid, and the stack's **measured VRAM ~1.16 GB
fits** beside the Isaac renderer (~6.6 GB) and whispr (~2.2 GB) on the 12 GB card. A full
cuMotion trajectory and `RobotSegmenter` need the sensor-populated ESDF world and live
depth (`required_external`), integration-owned. Report:
`data/sim-restock-host-v2/host-stack-launch-report.md`.
