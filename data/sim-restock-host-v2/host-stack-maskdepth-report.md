# Restock host stack: RobotSegmenter load crash fixed; Isaac masked-depth path verified

Branch `fm/sim-restock-segmenter-maskdepth-v1` from `origin/main` 208c1e9.

## Symptom (integration run-32, collect-execute stall)

The full mobile-restock mission finally passed `collect-navigate` (nav coverage PR #49),
then stalled in `collect-execute`: the arm never picked. Root cause was in the host
planning stack, not navigation. The host launch graph aborted at startup:

```
[component_container_mt-2] 'create_subscription() called for existing topic name
  rt/restock_sim/cumotion/robot_description_updated with incompatible type
  std_msgs::msg::dds_::Bool_, at ./src/rcl/subscription.c:112'
[ERROR] [restock_sim.host_components]: Component constructor threw an exception:
  could not create subscription: invalid allocator
[ERROR] [launch_ros.actions.load_composable_nodes]: Failed to load node 'robot_segmenter'
```

With `robot_segmenter` dead the container's `OnProcessExit` guard shut the whole launch
down, so no robot-masked depth reached nvblox, the ESDF stayed empty, cuMotion produced
no plan, and the arm never executed.

## Root cause

`launch.py` wired the NVIDIA `RobotSegmenter`'s `reload_robot_description_topic`
(a `std_msgs/Bool` trigger) onto `cumotion/robot_description_updated`, which the cuMotion
planner publishes as **`std_msgs/UInt64`** (`cumotion_planner.hpp:330`,
`robot_description_updated_pub_`). Two endpoints on one topic name with incompatible
message types — the second `create_subscription` fails, and in a composable container that
exception kills node construction and the launch.

A **second, latent** defect sat behind it: nvblox's `input_qos` defaulted to
`SYSTEM_DEFAULT` (RELIABLE), but the only depth publisher into nvblox is the shelfbot
`object_segmenter`, which republishes `world_depth` with best-effort SENSOR_DATA QoS. A
RELIABLE subscriber cannot match a best-effort publisher, so depth would never have
reached the mapper even once the segmenter loaded.

## Decision: Option B (delegate the self mask to the Isaac side)

Rather than bridge the incompatible reload signal (Option A, which also keeps the
`RobotSegmenter`'s NITROS depth / `joint_states` / TF coupling on the host), the simplest
path that actually works is to **remove `RobotSegmenter` from the host graph entirely** and
take robot-masked depth from the Isaac side, which produces exact per-pixel instance-ID rig
self masks every frame. This deletes the crash by deletion, drops the NITROS/TF coupling,
and uses a ground-truth mask. Agreed contract with the integration worker (bridge
`scripts/twin/restock/masked_depth.py`, their PR #15):

| field | value |
|---|---|
| depth topic | `/restock_sim/camera/masked_depth` (`sensor_msgs/Image`, `32FC1`, metres) |
| info topic | `/restock_sim/camera/masked_depth/camera_info` (`sensor_msgs/CameraInfo`) |
| frame_id | `restock_camera_<id>` (TF to `ur_base_link`) |
| invalid/self pixels | `0.0` (nvblox no-return) |
| stamp / domain | mission sim time matching RGB+info / domain 88 |

Host wiring (`launch.py`): the shelfbot `object_segmenter` subscribes `camera/masked_depth`,
masks the grasp-target bbox out of it, and republishes `world_depth`; nvblox takes depth
from `world_depth` and camera_info straight from `camera/masked_depth/camera_info`, with
`input_qos` set to `SENSOR_DATA`. Node count drops 7 → 6 (`config.py`, `stack_probe.py`,
`verify_restock_host_stack_gpu.py`).

## Bounded guarded GPU verification — PASS

`scripts/verify_restock_host_stack_gpu.py` (flock `/tmp/aile-gpu.lock`, protected-Blender
refusal, ≥8 GiB free VRAM / ≥12 GiB RAM, domain-88 isolation, CPU preflight, 10-minute cap,
exact-PID teardown) launched the task-isolated stack under the runtime closure and ran
`stack_probe.py`, which publishes a synthetic masked-depth frame on the agreed contract (a
wide-FOV down-looking camera over a flat floor, so nvblox carves the whole arm envelope
free and the floor is an observed obstacle) through the real `object_segmenter → nvblox`
path. Config: `refined/host.json` (refined assets regenerated from the current mobile
source, sphere_count 1930, 0 transport overlaps). Output:
`/home/manishsaini/datasets/sim-restock-host-v2/host-stack-maskdepth-verify/`.

| check | result |
|---|---|
| all 6 nodes present, **no load failure** | ✅ (move_group, static_planning_scene_server, nvblox_node, cumotion_planner, object_attachment, object_segmenter) |
| stack endpoints advertised | ✅ (external `follow_joint_trajectory` absent, expected) |
| `compute_ik` round-trip vs refined CAD | ✅ error_code 1 (SUCCESS) |
| nvblox integrates masked depth → **ESDF non-empty** | ✅ observed voxels 0 → **2,546,419** (focused AABB) |
| **one cuMotion plan returns a trajectory** | ✅ error_code 1, **16 trajectory points** (`MoveGroup planning completed with success: true`) |
| GPU reclaimed on teardown | ✅ free 9649 → 9649 MiB |

The `robot_segmenter` / `invalid allocator` error is **gone** from the launch log.

### VRAM (honest caveat)

Stack VRAM is map-dependent: ~0.9 GB for a small observed patch, up to **~3.4 GB when the
whole workspace is carved free** (the case that lets the folded arm's start pose be
collision-free, needed to exercise cuMotion). Beside the Isaac renderer (~6.6 GB) and whispr
(~2.2 GB) the 3.4 GB case fits on the 12 GB card only by a razor-thin margin (~7 MiB
nominal). The real per-station map observes a local shelf scene, not the full workspace, so
actual mission footprint should sit below the worst case — but the three-way renderer /
stack / whispr fit is tight and the integration worker already reduces renderer VRAM
(`render_scale`). This is a datapoint for station-map sizing, not a host-graph defect.

## Scope / honest limits

This proves the host launch graph loads cleanly and the masked-depth → ESDF → cuMotion path
works end to end against a synthetic frame of the agreed type. A full mission trajectory
still needs integration's live masked depth, measured `joint_states`/TF, and the
sensor-populated per-station ESDF (`required_external` in `config.py`); a correct instance-ID
self mask remains integration's to certify. The integration worker enables its bridge on
this confirmation and reruns `collect+couple`.
