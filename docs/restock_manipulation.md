# Sensor-driven restock manipulation

`scripts/twin/restock_manipulation/` provides the manipulation lane for mobile
restocking. It boots no simulator and opens no hardware driver. CPU checks and
ROS environment commands are in [RUNBOOK §13](../RUNBOOK.md#13-mobile-restock-manipulation-cpu).

## Supported operation and evidence boundary

`TransferController.start(operation, job)` accepts `collect` (backroom source bin
→ rollwagen) or `transfer` (rollwagen → shelf). The job contains `job_id`, `sku`,
`quantity`, `source_bin`, and `shelf_id`; quantity must be exactly one **handling
unit**, not a claim about the number of products inside a tray.

The supported shelf slice is one sealed rigid carton with a centred top QR label
and an exposed solid side wall. The actual 260.424 × 108.204 mm panel must fit that
wall with a 10 mm border; retracted tines must clear the support by at least 10 mm.
The integration candidate is **600 × 400 × 160 mm**, at no more than 0.5 kg for
this conservative suction-only slice. All three dimensions, the additional 30 mm
lift and reach must pass. A 320 mm tall case is rejected for a 280 mm shelf gap.
This is one **handling unit**; evidence sets `unit_kind=handling_unit` and
`retail_quantity=null`. Sealed contents are not counted or credited as retail units.
All six CAD tines remain retracted. Native latch and collision-aware IK must still
succeed in simulation before physical handling can be claimed.
The six poses are approach, contact, lift, pre-place, place,
and retreat. Every movement passes through the collision-aware motion port.
Reach radius is only an early rejection; IK and collision planning decide actual
reachability. Both destination dimensions and approach clearance are required.

**Open trays, bottles, bags, tilted products, multipicks and counting
hidden contents are unsupported.** The concrete runtime uses actual QR decoding and registered rendered depth;
`carton_asset.author_carton` authors a dynamic carton with a real printed texture.
Neural SKU/gallery recognition remains an optional, unvalidated path. Existing tray-trained detection/classification
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

All poses in these modules are **ur_base_link metres and xyzw**, not the shared
mission's map frame. Integration must transform camera and slot observations into
the current arm frame using timestamped TF after parking the mobile base. Never
reuse fixed-pedestal world-to-arm calibration on the moving robot. The actual CAD
arm joint names have the `ur_` prefix, and the planning TCP is `gripper_pad_tcp`;
the first generic adapter used incompatible names, now corrected.


## Actual CAD gripper and concrete Isaac path

The imported URDF has `suction_cups` (fixed by `suction_cups_joint` to `big_cube`)
and six `surface_plate_N` tines with `surface_plate_N_joint` prismatic axes +Y,
0–0.2 m travel. It is a suction-panel/tine tool, not a position-closing finger
gripper. The URDF contains no vacuum actuator or contact plugin: importing it
alone cannot respond to a suction topic. `cad_tool.prepare_cad_gripper` adds that
missing simulation component using Isaac's installed native SurfaceGripper.

Measured directly from the current URDF (not the older dimensions in the external
CAD documentation), relative to `ur_tool0`:

| Frame | Translation, metres | Meaning |
|---|---|---|
| `gripper_pad_tcp` | (-0.000859, -0.000827085, 0.272033616) | Cup panel sealing centre |
| `gripper_tcp` | (-0.000076997, -0.058637002, 0.203906998) | Retracted tine support frame |

Both TCP rotations are aligned with tool0 to numerical precision; +Z is approach.
For a shelf wall grip, tool +Y stays up and +Z points into the selected carton wall.
`WallCartonPolicy` directly reuses the external `wall_grip` and `approach_contact`
primitives, adapting them to current sensed source and target poses. There are no
fixed desk coordinates or external-source edits. It uses a 30 mm lift, low shelf
approach and horizontal retreat. The grasped carton face remains the same;
changing faces would need a verified regrasp.

The panel tilt requires 2.5–3 mm seal clearance and a maximum native ray distance
of 5 mm. Measured retracted CAD geometry extends 67.020 mm below the pad centre:
a mid-height grip on the 160 mm case leaves 12.980 mm above the support. Stage
source/carton front faces near the support edge; actual CAD collision spheres and
IK still determine feasibility.

Top pickup remains available only for open overhead stations. It **cannot** place
a 160 mm case inside a 280 mm shelf gap: the tool extends 272.034 mm behind the
pad, exceeding the 120 mm overhead room. The shelf runtime must set
`grasp_mode='side'` and pass per-location wall indices (0=-X, 1=-Y, 2=+X, 3=+Y).
For a 600 mm width × 400 mm depth case, wall 1/3 presents the 600 mm face.

`author_native_gripper` authors four attachment joints at the CAD panel the way the
installed Isaac surface-gripper plugin (`isaacsim.robot.surface_gripper`) requires, so
the plugin can form the latch — the plugin never creates or enables the joints itself,
it only fetches the already-instantiated runtime PhysX joint and reconfigures it. Each
seal is therefore an **enabled** maximal-coordinate (`excludeFromArticulation`) D6 joint
with `body0` = the panel, an **empty** `body1` (the plugin records the gripped body on
the gripper's own `grippedObjects` relationship, never on a per-seal `body1`) and both
local poses authored; the shear/tilt axes are locked and the forward/suction axis is a
small `0..maxGripDistance` compliant stroke with a drive; a positive `clearanceOffset`
and `retryInterval` are required (a disabled joint is never instantiated — the symptom
was the plugin logging `has no joint at attachment point` and never latching; a zero
retry abandons the close in the same step). `NativePanelContact.read` confirms the latch
from the plugin's authoritative signals — gripper status `Closed` holding exactly one
object — and releases any partial grab; there is no fallback fixed joint, target-pose
lookup or transform parenting. This is a native simulation approximation of panel
suction, not proof of vacuum performance or real hardware control. The native latch is
verified by an isolated guarded GPU test (`data/surface-gripper-latch-v1`).

The complete local adapter is `isaac_runtime.CadRestockRuntime`:

1. Before Play, enable `isaacsim.robot.surface_gripper` in the already booted app;
   call `prepare_cad_gripper(stage, robot_root=..., urdf_path=...)` on the task's
   output stage. It resolves actual imported tool/panel links and the physical body
   carrying the panel, refusing missing or ambiguous frames.
2. Author a supported carton with `carton_asset.author_carton`, giving its initial
   scene placement and the PNG produced by `isaac_sensors.write_carton_label`.
   Initial placement is scene construction only; runtime has no product prim path.
3. Initialize the actual mobile `SingleArticulation` and a rigidly mounted Isaac
   `Camera`. Supply calibrated optical-camera-to-`ur_base_link` transform and slot
   pixel ROIs/empty-support depths. Slot views must cover one dedicated carton each;
   stacked or occluded bins are unsupported. No semantic segmentation annotator is
   used. RGB/depth come from one stamped `get_current_frame` acquisition.
4. Construct `CadRestockRuntime` with those objects, explicit source roots,
   profile/slot/wall configuration, URDF path and authored gripper path. It builds the
   existing shelfbot CAD cuRobo planner with all body/panel/tine spheres retained,
   actual URDF path, `ur_base_link` root and `gripper_pad_tcp` tool. Source copies
   remain read-only; imports suppress source bytecode writes. After mobile docking,
   call `configure_station` with current calibrated views before a new job; this
   invalidates the old arm-frame collision map and requires fresh sensing.
5. Feed the task's sensor nvblox ESDF through `NvbloxWorldBridge`, or call
   `runtime.motion.update_esdf` with the mapper's actual signed grid and incorporated
   sensor timestamp. The mapper must use the arm frame while parked and handle
   robot/carried-object depth segmentation. Missing/unobserved voxels remain occupied.
   A changed collision world holds and replans active motion; the old trajectory
   cannot continue after a rejected replan.
6. `start(operation, job)` then `step(dt, simulation_time)` before each PhysX step.
   The adapter decodes QR pixels, checks the whole carton top against depth,
   observes slot vacancy, reads actual joint/base velocities and native latch state,
   sends bounded articulation drive targets and waits for encoder convergence.
   It never calls `set_joint_positions` or teleports a product.

The QR is a mechanically centred label, not a pose marker supplied by the stage.
Its observed corner direction supplies carton yaw; depth checks the registered
footprint and height above the previously measured empty support. The optional
ensemble adapter remains available but is not the labelled-carton sensor path.
Removing its pixels produces no SKU observation even when the product catalog and
scene still exist. Destination verification also checks that observed foreground
belongs to the one decoded carton; unknown clutter cannot certify a complete scan.
During final approach the tool can occlude the label: an already planned, bounded
approach may finish from its acquired pose while fresh collision/contact sensing
continues. A newly visible moved source aborts. Vacancy is checked before entering
the destination; once the held carton enters it, its own occupancy is expected.

`IsaacCadMotion` uses the external planner's real pose IK/trajectory output,
`attach_object_world` from the **observed** carton centre and native latch feedback,
and `detach_object` only after native release. These calls update collision
spheres, not physics. World refreshes verify that the planner's attachment survived.
The nvblox client targets `/restock_sim/nvblox_node/get_esdf_and_gradient` and does
not invent map freshness. The real `nvblox_msgs` service package, a running mapper,
robot/carried-object segmentation and its sensor-update acknowledgement are still
required from the integration launch. No free-space or mock-planner fallback exists.

CPU evidence covers native USD authoring (no preselected payload), actual QR/depth
algorithms, real URDF capability measurement, partial-latch refusal, measured-motion
completion, tine posture, lost/unknown map handling and changed-world replanning.
GPU solver, camera rendering, seal physics, reachability at the final station and
end-to-end stocking remain **unverified** under the integration GPU gate.

`clearance.linear_tool_sweep` is an evaluation-only conservative CAD tool-envelope
check over fixed-orientation linear segments; rotating segments are rejected.
It catches an obstruction between clear endpoints, but does not certify actual
joint trajectories, arm links, carried-object geometry or the full scene. Those
still require planner collision validation. Do not use fixture obstacle bounds
as a substitute for the live sensor collision map.

Runtime environment is an additional integration prerequisite: the inspected Isaac
Python 3.11 environment has neither `curobo` nor `nvblox_msgs`. Its visible `rclpy`
installation belongs to Python 3.12 and is not evidence of an ABI-compatible ROS
runtime. `CadRestockRuntime` requires the existing shelfbot planner's cuRobo stack
in its interpreter; alternatively integration must supply a process-separated
motion port. Merely adding the shelfbot source directory does not install cuRobo.
The existing planner node documents a separate cuRobo environment. No packages or
shared environments were changed by this lane. Use the process-separated path
below for this installation; the direct assembly cannot launch in that interpreter.

### Process-separated Isaac / Jazzy path

`isaac_ros_graph.IsaacRosGraph` uses Isaac's native `ROS2Publisher`,
`ROS2Subscriber` and domain-88 `ROS2Context` nodes. It imports no Python ROS,
nvblox or cuRobo. Native Isaac SurfaceGripper and `IsaacCartonSensors` stay with
the simulator. Feed that camera adapter a measured articulation/base feedback
object (`IsaacMeasuredFeedback(articulation)`) whose `collision_time` is an actual mapper acknowledgement; no direct
`IsaacCadMotion` or GPU planner construction is necessary on this path.

`ros_host` runs in separate host Jazzy Python and instantiates the existing
`RosMotionPort`, `RosTransferBridge`, controller and CAD wall policy using
`ros_host_config.json`. RUNBOOK §13 has the exact launch. Sensor/job/status JSON
is unchanged except status adds monotonic `sequence` and `job_id` for stale-message
filtering. Native contact feedback is in `Sensors`; returned gripper controls call
the native seal. `/restock_sim/manipulation/stop` is Bool and must latch arm hold
and inhibit further trajectory commands until explicit recovery. Lost host
heartbeat also latches hold. Never release a held unit as a watchdog fallback.

Integration's existing Isaac ROS graphs supply `/clock`, registered RGB-D, current
TF and joint-state/controller transport. Host-side MoveIt/cuMotion provides the
three planning/execution endpoints below, `/restock_sim/apply_planning_scene`
(`ApplyPlanningScene`) and `/restock_sim/attach_object` (`AttachObject`). These
servers, sensor-updated mapper and NVIDIA ROS message overlay remain required;
the adapter does not silently launch an external rig or substitute fixture servers.
The current host lacks the NVIDIA attachment message overlay too: source the
planner's compatible ROS overlay before payload operations. This is separate from
the now-removed requirement to install host ROS into Isaac Python.

`RosPayloadSync` runs its bounded acknowledgement clients on a separate executor:
it checks measured native contact, converts the sensor-derived unit centre into
the current TF-measured TCP frame, and waits for both MoveIt scene and cuMotion
attachment acknowledgements. A missing package/server, timeout or partial update
fails the transfer and retains the physical seal for recovery. It never moves a
physics object. Sensor-map removal/segmentation of the carried unit must be enabled
in the planner launch; neither acknowledgement alone proves that camera voxels
were updated. `Sensors.collision_time` must still reflect that actual update.

The CPU two-interpreter smoke runs Isaac Python3.11 → host Python3.12 → Isaac,
roundtrips existing `Sensors` and `Result` with real ROS String/Bool serialization,
and asserts fixture provenance and no retail count. It creates no DDS graph and
does not validate native OmniGraph, GPU inference, attachment or motion execution.

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
| `shelfbot_newgripper/shelfbot/motion_planning/geometry.py` and `primitives.py` | `wall_grip` and `approach_contact` reused directly by `wall_policy.py` with sensed poses, CAD pad TCP and station support heights |
| `shelfbot_newgripper/shelfbot/planning/curobo_planner.py` | Loaded by `isaac_motion.py`: existing GPU pose planner, `attach_object_world`, ESDF update and collision boxes |
| `shelfbot_newgripper/shelfbot/scripts/tray_transfer_orchestrator.py` | Existing action/behavior-tree mission uses fixed scene configuration in parts, so it is not launched blindly for mobile world poses |
| `ur10-setup/dynamic_pick/cumotion_planner.py` | Reuses the seeded joint-goal, 5% scaling, plan-only and waypoint-validation pattern; new asynchronous adapter adds explicit measured start state and cancellation |
| `ur10-setup/dynamic_pick/planner_port.py` | Analytic and Cartesian surrogate paths are fixtures, never imported for runtime |
| `shelfbot_newgripper/shelfbot/perception/color_state.py` | Color-coded fullness is synthetic semantics; not used for live SKU/count verification |

Runtime dependencies are provided as objects or source-root arguments; no private
machine paths, weights or external source changes are included. A future open-tray
port needs mobile-frame calibration, local adapters around the existing pose
primitives and physically validated wall-grip geometry. Such adapters can be
implemented here without modifying the external shelfbot copies. The historical
`data/twin-isaac-plan-s1/report.md` was absent from this checkout; existing
`data/twin-robot-in-store-v1/report.md` and `docs/isaac_robot_in_store.md` were read.

Installed message API check: ROS Jazzy `moveit_msgs` 2.6.0 imports pass. No cuMotion
server/version or GPU execution was verified in this lane. NVIDIA documents the
[MoveIt integration](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_cumotion) and
[UR simulator pipeline configuration](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_cumotion/blob/main/isaac_ros_cumotion_examples/launch/ur_isaac_sim.launch.py).
The installed ROS message definitions, rather than an assumed newest release,
were checked while writing the adapter.
