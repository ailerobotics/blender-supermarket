# Sensor-driven restock manipulation

`scripts/twin/restock_manipulation/` provides the manipulation lane for mobile
restocking. It boots no simulator and opens no hardware driver. CPU checks and
ROS environment commands are in [RUNBOOK §9](../RUNBOOK.md#9-mobile-restock-manipulation-cpu).

## Supported operation and evidence boundary

`TransferController.start(operation, job)` accepts `collect` (backroom source bin
→ rollwagen) or `transfer` (rollwagen → shelf). The job contains `job_id`, `sku`,
`quantity`, `source_bin`, and `shelf_id`; quantity must be exactly one **handling
unit**, not a claim about the number of products inside a tray.

The initial supported family is a sealed rigid box with a horizontal, exposed,
suctionable top, a registered SKU, measured dimensions and calibrated TCP
orientation/offset. The six poses are approach, contact, lift, pre-place, place,
and retreat. Every movement passes through the collision-aware motion port.
Reach radius is only an early rejection; IK and collision planning decide actual
reachability. Both destination dimensions and approach clearance are required.

**Open trays, wall grips, bottles, bags, tilted products, multipicks and counting
hidden contents are unsupported.** No packaged SKU/model is asserted ready for a
real rendered scene. Gallery enrollment, top-face segmentation and TCP calibration
are explicit runtime dependencies. Existing tray-trained detection/classification
models must be evaluated on the selected sealed unit; they are not presumed to
recognize it because the controller supports its geometry.

The CPU tests exercise orchestration with clearly labeled `test_fixture` inputs.
They do not prove physics, successful suction, neural detection quality, planner
execution, or autonomous supermarket operation. No GPU run was made in this lane.

## Entry points for integration

- `TransferController(motion, profiles, cart_location='rollwagen')`;
  `start(operation, job)`; `step(dt, Sensors) -> Result`.
- `DepthPerception(load_pose_bridge(tray_perception_root))`: `observe(...)`
  combines accepted SKU retrieval with masked, registered depth and camera-to-arm
  TF; `target(...)` checks a calibrated slot ROI against its depth-fitted support
  plane. No scene prim lookup or object-pose truth is used.
- `EnsembleObservationPipeline(detector, classify_crop, depth_perception,
  segment_top, location_for_box).observe(rgb, depth_m=..., frame_id=..., ...)`
  invokes loaded YOLO and the existing ensemble classifier on RGB crops. Bind
  `classify_crop` to its loaded gallery. Supply a real image/depth top-face
  segmenter and calibrated image ROI lookup. Rejected detections are returned
  separately; an empty detection list does **not** prove source vacancy.
- `RosMotionPort(node, joint_limits=..., payload_sync=..., emergency_stop=...)`
  performs collision-checking seeded `GetPositionIK`, plan-only `MoveGroup`, then
  `ExecuteTrajectory`. It checks every returned joint waypoint, initial state,
  increasing time and terminal velocity. Server execution success is required;
  elapsed wall time alone never completes a move. The node must be spun by the
  integration executor. It does not start servers.
- `RosTransferBridge(node, controller)` attaches jobs/sensing/status/gripper topics
  to that same externally spun node. Missing sensor packets trip a steady-clock
  watchdog even if simulation time stops. Rejected jobs do not replace active jobs.

`Result` contains `status` (`running`, `succeeded`, `failed`), `phase`, `reason`,
`controls`, and `evidence`. Motion controls are audit records: the motion port
already submitted those movements. Consume `gripper` controls exactly once in a
non-ROS adapter; the ROS bridge publishes them itself. `stop` means immediate
simulation joint hold via the required motion-port stop callback. On failure a
held payload remains held, and a new job is refused until explicit recovery.

All poses in these modules are **base_link metres and xyzw**, not the shared
mission's map frame. Integration must transform camera and slot observations into
the current arm frame using timestamped TF after parking the mobile base. Never
reuse fixed-pedestal world-to-arm calibration on the moving robot.

## ROS wire contract

Use domain 88, loopback discovery and `use_sim_time=true`; the adapter refuses
other domains, explicit static peers, or endpoints outside `/restock_sim`.

| Endpoint | Type | Meaning |
|---|---|---|
| `/restock_sim/manipulation/job` | `std_msgs/String` | JSON `{operation, job}` |
| `/restock_sim/sensors` | `std_msgs/String` | Complete `Sensors` dataclass JSON |
| `/restock_sim/manipulation/status` | `std_msgs/String` | `Result` JSON including evidence |
| `/restock_sim/gripper/enable` | `std_msgs/Bool` | Simulated suction command; no physical I/O |
| `/restock_sim/compute_ik` | `moveit_msgs/GetPositionIK` | Seeded collision-checking IK |
| `/restock_sim/move_action` | `moveit_msgs/MoveGroup` | Planning, default `isaac_ros_cumotion` pipeline |
| `/restock_sim/execute_trajectory` | `moveit_msgs/ExecuteTrajectory` | Simulation controller execution |

The authoritative schema is [types.py](../scripts/twin/restock_manipulation/types.py).
`Sensors` includes simulation timestamp, joint positions in canonical UR order,
parked state, measured suction/contact, feedback timestamp, collision-map timestamp,
observations, target and scan coverage. `feedback_time` must be the oldest of the
joint/contact/base measurements. `collision_time` must acknowledge the planner's
actual sensor-updated world, not merely receipt of a depth image. `scanned_locations`
is only valid after **complete, unoccluded** scans of those inventory ROIs;
`scan_time` timestamps those scans. An absent detection in a blind/occluded ROI is
unknown, never absence. Do not publish a new timestamp around old measurements.

Runtime observation provenance is `sensor`. `fixture_mode=True` also permits
`test_fixture` for tests; `ground_truth` is always rejected. Post-placement success
requires two distinct acquisition times/IDs with matching SKU and pose after
release, measured loss of gripper contact, and fresh complete source+destination
scans with no source object. Integration must only credit inventory from this
verified result. Ambiguous/missing observations retry within a bounded phase;
occupied/unreachable targets, lost grasp, stale feedback/map, moving base, planning
failure and verification timeout stop. Contact loss does not automatically retry
a pick, because the object may have fallen.

`payload_sync(held, profile)` must return true **only after both MoveIt and the
selected cuMotion collision representation acknowledge attachment/detachment**.
It must remove the picked object from world obstacles and represent it as a carried
volume, then restore the placed world obstacle on release using measured placement.
There is no default true callback. Physics attachment is a separate simulator
contact/suction implementation; collision attachment must never teleport an object.
`emergency_stop()` immediately holds the task-owned simulated arm, in addition to
ROS action cancellation. A late action acceptance after cancellation is canceled.
Use a mutually exclusive callback group for controller state and sensor callbacks.

## Reuse inventory and external work

Read-only sources inspected:

| Source | Reuse or finding |
|---|---|
| `tray_perception/labeling/pose_bridge.py` | Loaded directly by explicit root; masked depth backprojection and PCA plane fitting exercised on CPU |
| `perception_ensemble/tray_pipeline.py` | Existing YOLO + `classify_crop(crop, gvecs, glabels)` accepts only `OK`; adapter adds explicit similarity/margin gates |
| `shelfbot_newgripper/shelfbot/perception/pose_from_depth.py` | Existing tray slab/PCA pose path assumes a support plane; useful for later open-tray integration |
| `shelfbot_newgripper/shelfbot/motion_planning/pose_plan.py` | Existing tray wall-grip/extract/insert choreography; not claimed compatible with the new sealed-box path |
| `shelfbot_newgripper/shelfbot/planning/curobo_planner.py` | Existing GPU pose planner, `attach_object_world`, ESDF update and collision boxes; integration may supply a compatible motion port around it |
| `shelfbot_newgripper/shelfbot/scripts/tray_transfer_orchestrator.py` | Existing action/behavior-tree mission uses fixed scene configuration in parts, so it is not launched blindly for mobile world poses |
| `ur10-setup/dynamic_pick/cumotion_planner.py` | Reuses the seeded joint-goal, 5% scaling, plan-only and waypoint-validation pattern; new asynchronous adapter adds explicit measured start state and cancellation |
| `ur10-setup/dynamic_pick/planner_port.py` | Analytic and Cartesian surrogate paths are fixtures, never imported for runtime |
| `shelfbot_newgripper/shelfbot/perception/color_state.py` | Color-coded fullness is synthetic semantics; not used for live SKU/count verification |

Runtime dependencies are provided as objects or source-root arguments; no private
machine paths, weights or external source changes are included. A future open-tray
port needs mobile-frame calibration and dynamic source/place overrides in the
external shelfbot mission, plus physically validated wall-grip geometry. Those
changes belong in a separately assigned source task. The historical
`data/twin-isaac-plan-s1/report.md` was absent from this checkout; existing
`data/twin-robot-in-store-v1/report.md` and `docs/isaac_robot_in_store.md` were read.

Installed message API check: ROS Jazzy `moveit_msgs` 2.6.0 imports pass. No cuMotion
server/version or GPU execution was verified in this lane. NVIDIA documents the
[MoveIt integration](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_cumotion) and
[UR simulator pipeline configuration](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_cumotion/blob/main/isaac_ros_cumotion_examples/launch/ur_isaac_sim.launch.py).
The installed ROS message definitions, rather than an assumed newest release,
were checked while writing the adapter.
