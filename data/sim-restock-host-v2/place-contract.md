# Mission-facing contract: the PLACE (transfer) job on the restock host stack

The HOST side of the post-collect TRANSFER stage, proven on the GPU stack probe
(`scripts/restock_runtime/place_probe.py`, run via `scripts/run_place_probe_gpu.py`). This is
what the mission's `TransferController` drives against the host; integration consumes it to wire
the place without re-deriving the host endpoints. Place is **`operation='transfer'`** on the same
controller as collect (not a separate op); the host stack is identical for both — only the job's
source/destination and the sensed poses differ.

## 1. Job message (mission → host)

Published as `std_msgs/String` JSON on `/restock_sim/manipulation/job`:

```json
{"operation": "transfer",
 "job": {"job_id": "<non-empty str>", "sku": "juice_orange",
         "source_bin": "rollwagen", "shelf_id": "G3_back_s2", "quantity": 1}}
```

- `operation`: `"transfer"` (collect is `"collect"`). Any other value → rejected.
- `job.quantity`: must be `int == 1`.
- `job.job_id`, `job.sku`, `job.source_bin`, `job.shelf_id`: non-empty strings. For transfer the
  destination is `job.shelf_id` (= `G3_back_s2`), the source is the cart (`rollwagen`).
- The SKU must have a profile in the verified `ros_host_config.json` (`juice_orange`:
  `dimensions [0.6, 0.4, 0.16]`, `grasp_mode "side"`, `tcp_quaternion [0.5,0.5,0.5,0.5]`,
  `approach 0.12`, `tcp_surface_offset 0.00275`, `mass_kg 0.45`), and `shelf_id` must have a wall
  index in `ros_host_config.json` `walls` (`G3_back_s2: 3`) matching the cart's wall (no regrasp).

## 2. Status sequence (host → mission)

`std_msgs/String` JSON on `/restock_sim/manipulation/status` each step, fields
`{status, phase, reason, controls, evidence, sequence, job_id}`. `status` ∈
`{running, succeeded, failed, rejected}`; `phase` carries the detail:

```
observe → approach → contact → grasp → lift → preplace → place → release → retreat → stow → verify → succeeded
```

- `grasp`: emits gripper-enable; on sensed `contact` the carton is **attached** (see §4).
- `preplace → place`: the side-grip Cartesian **insertion** into the shelf slot (lower by the
  lift offset). `place → release`: gripper-open + **detach**; `retreat`: back out along the wall
  normal; `stow`: collision-planned return to the transport joint pose; `verify`: carton observed
  at the destination across two post-release frames + source observed empty.
- Terminal: `status:"succeeded"` (phase `succeeded`) or `status:"failed"` with a `reason`
  (e.g. `payload collision attachment/detachment not acknowledged`, `planning failed`,
  `sensor stream watchdog expired`). A malformed job → `{"status":"rejected","reason":...}`.

## 3. Station protocol at the shelf (prepare / heartbeat / ack)

Identical to collect; the workspace bounds are a **single ur_base_link-relative triple**
(`host.json` `workspace_min [-2,-2,-1]`, `workspace_max [2,2,2]`) that moves with the parked base,
so the SAME bounds apply at `G3_back_s2`. Only the camera frame differs. On
`/restock_sim/station/command` (`std_msgs/String` JSON):

- `prepare`: `{action, epoch (int>0), sequence (int>=0), stamp, parked:true, camera_frame:"restock_camera_shelf"}`
  — `camera_frame` must start with `restock_camera_`; the host validated and echoed
  `restock_camera_shelf`. `0 <= now - stamp <= 0.5` (node-clock).
- `heartbeat`: same without `camera_frame`, `parked:true`, increasing `sequence`, every ≤0.5 s of
  node-clock time (and the host wall guard is 3.0 s — bracket long render ticks with a heartbeat).
- Host publishes `/restock_sim/station/status` `{state, stamp, collision_time, epoch,
  parked_since, camera_frame, reason}`; `state:"ready"` with a fresh `collision_time`
  (`0 <= now - collision_time <= 0.5`) once the shelf masked depth integrates into the nvblox ESDF.
  **PROVEN at the shelf:** ack `ready`, `collision_time` fresh, `camera_frame` echo
  `restock_camera_shelf`.

## 4. Carton attachment (collision-aware planning)

Two steps, both acknowledged, at `grasp` (attach) and `release` (detach) — PROVEN on the host:

1. MoveIt `/restock_sim/apply_planning_scene` (`ApplyPlanningScene`): a diff with an
   `AttachedCollisionObject` — `link_name "gripper_pad_tcp"`, `touch_links ["suction_cups"]`,
   `object.id "restock_carried_unit"`, `object.header.frame_id "ur_base_link"`, a `BOX`
   SolidPrimitive of the SKU `dimensions`, `operation ADD` (attach) / `REMOVE` (detach).
2. cuMotion `/restock_sim/attach_object` (`isaac_ros_cumotion_interfaces/action/AttachObject`):
   `attach_object: true/false`, `object_config` marker (`header.frame_id "gripper_pad_tcp"`,
   `type CUBE`, `scale = dimensions`, pose relative to the TCP). SUCCEEDED (status 4) = acknowledged.

Attach requires a fresh `ur_base_link ← gripper_pad_tcp` TF and the sensed contact. `object_attachment`
runs `clear_esdf_on_attach`, so masked depth must keep streaming so the shelf walls stay in the ESDF
for the subsequent plan.

## 5. Shelf walls in the collision model

The host carries **no static wall CollisionObject** (StaticPlanningSceneServer is empty). Shelf
geometry enters the planning world as **nvblox ESDF voxels from the masked depth** — so the depth
camera must see the shelf at `G3_back_s2`. cuMotion queries `get_esdf_and_gradient` at plan time
(`read_esdf_world`, `update_esdf_on_request`). **PROVEN:** masked depth → ESDF 0 → 2,546,419
observed voxels.

## 6. Planning (insertion + return) — host pattern

Each pose phase is `RosMotionPort.begin(pose, measured_joints)`: seeded `compute_ik`
(`/restock_sim/compute_ik`, group `ur_manipulator`, link `gripper_pad_tcp`, `avoid_collisions`,
frame `ur_base_link`) → a cuMotion joint-space `move_action` plan (`pipeline isaac_ros_cumotion`,
`plan_only`, `max_velocity_scaling_factor 0.04`). Return-to-transport (`stow`) is
`begin_joint_goal(transport_joints)` — a direct collision-planned joint goal to the 6 `ur_*`
transport positions from `mobile_manifest.transport_positions`
(`[0.0, -2.5, 1.7, -0.7, -1.5708, 0.0]`). **PROVEN:** a cuMotion joint-space plan WITH the carton
attached against the populated shelf ESDF returns a valid trajectory.

Host gap for the Cartesian insertion pose (see the report): the URDF references the UR10 collision
meshes by bare absolute path (`/opt/ros/jazzy/share/ur_description/...`), which MoveIt's resource
retriever rejects (`No host part in the URL`), leaving the arm's MoveIt collision geometry empty.
`avoid_collisions` IK can then return a contorted, self-colliding branch that cuMotion rejects
(`planToCSpaceTarget … self collision detected`). The mission's `wall_policy` insertion poses yield
natural IK (collect works), but giving the URDF mesh filenames a `file://` scheme restores MoveIt
collision-aware IK and is recommended before the first live place.

## 7. Timing (measured on the stack probe, real-time clock)

- Shelf station ack ready after prepare: ~0.4 s once depth integrates (collision_time fresh).
- Masked depth → first non-empty ESDF: ~1.0 s cold.
- Carton attach (ApplyPlanningScene + AttachObject) acknowledged: < 1 s; detach similar.
- cuMotion joint-space plan against the shelf ESDF (65-point trajectory): ~2.6-2.8 s reply.
- cuMotion `allowed_planning_time` is 5 s of SIM time — measure the watchdog/budget in node-clock
  time (the mission's render-paced `/clock` makes wall time much longer per plan). Use a joint
  tolerance of ~0.05 rad (0.001 rad makes cuMotion trajopt non-converge: UNKNOWN_STATUS).
- The plan START must match the measured joint_states (cuMotion rejects a disagreeing start_state).

Evidence: `/home/manishsaini/datasets/sim-restock-host-v2/place-probe-*/`. Probe + runner:
`scripts/restock_runtime/place_probe.py`, `scripts/run_place_probe_gpu.py`.
