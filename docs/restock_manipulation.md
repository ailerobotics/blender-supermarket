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

**Contact wall-budget fail-fast** (`RESTOCK_CONTACT_STUCK_WALL_S`, opt-in, default `0` = off): the
contact descent terminates on measured contact (`Sensors.contact` = native grippedObjects OR the
`SIM_ATTACH_FALLBACK` weld, which welds at the measured ≤0.5 mm pad-to-carton gap), creeping
`CONTACT_STEP` toward the carton between freeze-settles. In run-322 cold the native suction never
latched and the weld never formed, so the creep held with `contact=False` and the arm-command stream
lapsed into the opaque 90 s host transport heartbeat watchdog (`arm command heartbeat lost`, no named
cause). The phase's own `phase_timeout` is **sim-clock**, so under render pacing the WALL heartbeat trips
first. When set, the controller fails the contact phase fast with a named reason once no contact
progress (entering contact / a `CONTACT_STEP` advance) occurs for this many WALL seconds — below the
heartbeat backstop — so a stuck contact ends the attempt legibly instead of burning the GPU wall cap.
CPU tests: `tests/test_restock_manipulation.py`.

**Pick-contact seal reliability + guards** (sim-pick-contact-seal-fallback-v1, HOST keys, all opt-in,
default off ⇒ byte-identical). coldwin63's pick never sealed even though the controller DID issue the
suction enable (`CONTACT_SUCTION enable`, host-stack.log): the gripper-enable control is a **one-shot**
carried only on the `/manipulation/status` `controls` field, and the host→Isaac relay
(`control_ipc.HostControlPublisher`) is **last-value** — a later heartbeat status overwrote it before
`isaac_ros_graph.IsaacRosGraph.poll` read it, so `cad_tool.command(close)` never ran (plugin stayed
`Open`), the gated `SIM_ATTACH_FALLBACK` stayed `skip=suction_off`, and the press hit the 20 mm cap
(`contact not reached within capped over-press travel`). cw62/cw61 won the race; cw63 lost it. Three keys
address it: `contact_reissue_suction_each_tick` re-publishes the enable every press tick so the relay
always carries it (the direct fix — behaviour-preserving for sealed runs); `contact_require_suction_enabled`
(+ `contact_reissue_close_once`) fails fast `CONTACT_SUCTION_NOT_ENABLED` / re-sends the close once at the
first over-press step if the plugin never reported enabled, instead of pressing the full cap with no chance
of a latch; `contact_accept_seal_diag` (+ `contact_seal_diag_dwell_ticks`/`_max_gap_mm`/`_max_angle_deg`)
treats a dwell-stable flush geometric `SEAL_DIAG` on a body under the pad (with suction enabled) as contact,
driving the same latch path the plugin flag drives. The guards read new opt-in `Sensors` feedback
(`gripper_enabled`, `seal_gap_mm`, `seal_pad_angle_deg`, `seal_hit_target`) the driver populates from the
`NativePanelContact` probe. Recommended ON values + the full reconstruction:
`docs/examples/pick-contact-seal-fallback-fragment.json`. CPU tests:
`tests/test_pick_contact_seal_fallback.py`.

**Gripper-status feedback to the host** (sim-gripper-status-relay-v1, coldwin65). The guards above read
the **host-facing** `/sensors` packet, but those feedback fields were only ever populated in
`isaac_sensors.Sensors.capture` — **not** in `isaac_factory.CollectTargetBridge`
(`_capture_ground_truth_collect`/`capture`/`_capture_transfer`/`_survival_packet`), which is what the live
mission publishes. So the host saw the **all-default** feedback (`enabled=False`, `gap=None`, `hit=False`)
on every tick regardless of the true Isaac state, and `contact_require_suction_enabled` e-stopped coldwin65's
pick while the plugin was actually `Closing`/`Closed` with the gap closing (coldwin64 sealed the identical
pick only because nothing read that starved view). **Not** the last-value `/manipulation/status` relay: the
enable landed, and `/sensors` has no heartbeat/last-value drop. Fix: the opt-in env
`RESTOCK_CONTACT_STATUS_FEEDBACK=1` (`contact_status_feedback.py`) makes the `CollectTargetBridge` capture
paths carry the gripper feedback + a `gripper_status_time` stamp every tick (off ⇒ byte-identical); and
`contact_require_suction_enabled` now reads an **authoritative** status — it aborts only on a **fresh**
not-enabled, and on an **absent** status logs `CONTACT_STATUS_STALE` and continues the bounded over-press
(never aborts on staleness alone; optional extra bound `contact_status_max_age_s`). A genuine lost-enable
(cw63's fresh `Open`) still fails fast. Recommended ON values for the next complete run:
`docs/examples/gripper-status-relay-fragment.json`. CPU tests: `tests/test_gripper_status_relay.py`.

**Collision-stale pause keep-alive + wall budget** (`RESTOCK_COLLISION_PAUSE_WALL_S`, opt-in, default
`0` = off ⇒ the legacy empty-control pause, byte-identical): the fm419(a) `COLLISION_STALE_PAUSE` holds
the post-attach lift while the (held) carton's collision stamp is briefly stale — the benign weld/ESDF
re-incorporation lag (run-323: `collision_dt` 0.52→0.65 s). The legacy pause returned EMPTY controls — a
SILENT hold that, under render pacing, starved the 90 s host arm-command heartbeat watchdog before the
sim-clock `collision_stale_tol_s` could trip. When set, the pause (1) re-commands a HOLD at the CURRENT
measured joints each tick (stationary — the goal IS the measured pose, so the arm never advances the held
move into the stale collision world) so the arm-command stream stays alive and the heartbeat is fed, and
(2) fails fast with a named reason once the pause persists beyond this WALL budget (below the 90 s
backstop). CPU tests: `tests/test_restock_manipulation.py`.

**Post-attach collision-stamp lag — two opt-in mitigations (fm430 (2a), both default-off ⇒ byte-identical):**
diagnosis — right after the `SIM_ATTACH` weld the host `object_attachment` clears the carton voxels and
the parked mapper must re-incorporate at the new state, but under render pacing the next nvblox ESDF
integration stamp lags `now` by ~0.5–0.8 s (run-323 `collision_dt` 0.52→0.65, progressive), so the
controller's 0.5 s collision freshness trips through the lift and the fm419(a) pause holds.
(1) **Host ESDF re-integration kick** — `RESTOCK_ESDF_POSTATTACH_KICK` makes `restock_host/incorporation_ack.py`
(the nvblox `get_esdf_and_gradient` poll node) subscribe to `/manipulation/status` and, on the phase
rising into `lift` (post-weld), DROP any in-flight pre-attach ESDF request and issue a fresh one so the
post-attach integration lands ~a cycle sooner instead of waiting out the old request (≤ 2 s). Pure
`IncorporationAckState.note_manip_phase`/`consume_kick` sequencing, unit-tested in `tests/test_restock_host.py`.
(2) **Controller held-lift collision grace** — `RESTOCK_LIFT_COLLISION_GRACE_WALL_S` lets ONLY the
post-attach lift to the fixed, bench-proven clear-region `carry_lift_shape` (which does NOT depend on the
live collision world) RIDE THROUGH a stale collision stamp (proceed, not pause) for a bounded WALL grace,
then FAIL FAST with a named reason. Scoped to held + phase `lift` + `carry_lift_shape` set +
collision-SOLE staleness (every other invariant field must be good). CPU tests: `tests/test_restock_manipulation.py`.

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

## Carried pre-place wrist hold (sim-preplace-wrist2-hold-v1)

The carried pre-place runs on the soft drive (the swing fix) with shoulder_pan force-stiffened
(`phase_drive.carried_joint_profile`) so it holds its goal through the 1.64 rad wrist_2 sweep. A
remaining failure is the wrist **give-way**: a wrist that has already reached its target (e.g. wrist_2
at 0.6678 after leg 1) can be **back-driven off it** by leg 2's shoulder_pan sweep while still on the
soft hold (cw66 collapsed wrist_2 to a -0.3264 limit cycle; cw64 ran the same trajectory and recovered
— a coin toss). The static hold is trivial (gravity on wrist_2 ≈ 0.045 N·m); the give-way is purely the
**dynamic** pan-sweep reaction on the soft **held** wrist.

Two opt-in, default-off (byte-identical) mechanisms remove the coin toss:

- **Hold-firm-on-arrival** (RUNTIME `phase_drive.hold_firm_on_arrival`, `preplace_hold.HoldFirmOnArrival`
  + `isaac_port`): once a carried-pre-place joint arrives at its *settled* commanded target it is
  swing-safe to stiffen (it is no longer moving), so its PhysX drive is switched to the firm hold profile
  and it logs `PREPLACE_HOLD_FIRM`. An arrived wrist then cannot be back-driven. shoulder_pan is excluded
  by default (already firm). This is the **prevention**.
- **Drift re-capture** (HOST `preplace_drift_recapture_rad` / `preplace_drift_recapture_max` on
  `convergence_guard.ConvergenceGuard`): if a driven joint still drifts off its closest approach by more
  than the bound, the guard emits a non-terminal `DriftRecapture`; the controller re-issues the armed
  pre-place goal (re-driving the joint, which holds once the disturbing leg settles) and logs
  `PREPLACE_DRIFT_RECAPTURE` once, instead of faulting at the 30 s stall. Bounded per joint — a joint that
  keeps re-drifting still falls through to the terminal `PHASE_CONVERGENCE` fail-by-name. This is the
  **backstop**.

The 2.0° carton bound is untouched (firming a held joint does not move it, so the predicted tilt is
unchanged). Recommended ON values + acceptance: `docs/examples/preplace-wrist2-hold-fragment.json`; CPU
proof (incl. the real cw66 wrist_2 replay): `scripts/twin/tests/test_preplace_wrist2_hold_v1.py`.

### Serialize every re-issued goal behind the active motion (sim-motion-active-guard-v1)

Both recovery paths above (the **drift re-capture** here, and the **place set-down re-approach** below)
*re-issue an arm goal*. The motion port guards `begin()`/`begin_joint_goal()` with
`raise RuntimeError('motion already active')` while a move is still in flight, and the controller's generic
adapter-failure handler turns that raise into a `HOST_FAULT_ESTOP`. Twice a re-issue fired while the prior
motion was still active and crashed the host instead of recovering: **coldwin73** (the drift re-capture
re-issued the pre-place joint goal mid-move) and **coldwin90** (the place re-approach, after the seat gate
correctly refused the obstructed descent, re-descended while the obstructed descent handle was still active).

The fix is one serialization helper, `TransferController._serialize_reissue`, beneath **both** (already
opt-in) paths — the flags are unchanged; this is the fix beneath them. Before a re-issued goal it brings the
active motion to a **terminal** (not-in-flight) state on the channel that actually terminates, confirms the
handle is terminal, then begins:

- a **driven descent** (the place set-down) is stopped on its own channel via the PR 368 contact-hold
  (`settle_in_place` + a `{'kind':'contact'}` control so the sim `trajectory_host` *succeeds* holding the
  current pose — terminal, no fault, no global `/manipulation/stop`, which an `ExecuteTrajectory` cancel
  would latch);
- an **action goal** (the pre-place cuMotion/KDL move) is stopped with `motion.cancel()` + a bounded
  wait on `cancel_outcome()` (PR 334).

It logs `REISSUE_SERIALIZE` / `REISSUE_SERIALIZE_SETTLED`. If the active motion does not reach a terminal
result within `reissue_settle_timeout_s` sim (internal, env `RESTOCK_REISSUE_SETTLE_S`; see
`host_timers_audit.md`) it **fails by name** — `REISSUE_MOTION_STILL_ACTIVE`, naming the handle state —
rather than re-issuing into an in-flight motion, so a recovery is a real second attempt and never a crash.
A port already terminal (the normal first pre-place entry from a completed lift) is begun immediately with
no stop issued, so a crash-free run is byte-identical. CPU proof (the helper's three outcomes, the cw73 and
cw90 replays, byte-identical-inert): `scripts/twin/tests/test_motion_active_guard_v1.py`.

### Lock the base from pre-place, not the descent (sim-station-lock-from-preplace-v1)

Firming the held wrist (above) removed the give-way but **re-routed** leg 2's shoulder_pan-sweep reaction:
instead of being absorbed by the soft wrist it now drives through the stiff arm **into the base**.
coldwin68/69 parked at the same spot as the no-shove reference coldwin64 but, during the firm-held carried
pre-place, their base was shoved ~0.33 m sideways (`BASE_YAW_TICK` `base_dxy` over sim t ~191–227), so the
carton came down ~0.5 m off its cart-relative target onto a cage post. The cause is timing: during pre-place
only the base **rotation** is held (`CHASSIS_STATION_PITCH_HOLD` leaves translation free); the
`station_place_full_brake` full six-axis lock engaged only once the place **descent** began (phase `place`).

The opt-in RUNTIME key **`station_full_brake_engage_phase`** (default `"place"` ⇒ byte-identical) moves that
engage earlier: `"preplace"` locks the base at the **start** of the carried pre-place (before leg 1, so both
legs run under the full lock), and `"hold_firm"` locks it the moment any carried joint is firmed (the
`PREPLACE_HOLD_FIRM` event, end of leg 1 — requires `phase_drive.hold_firm_on_arrival.enabled`). One engage
at the chosen trigger, one release at withdraw — no cycling. Extending the lock backward is weld-loop-safe:
the carton is airborne through pre-place, so no loop closes until contact at `place`, and the effort guard's
window follows the same predicate (`station_full_brake.full_lock_window`). Requires `station_place_full_brake`
+ `chassis_station_pitch_hold`; `config_preflight.check_station_full_brake` enforces that and refuses an
unknown value or `"hold_firm"` without the hold-firm config. The 2.0° carton bound is untouched. Recommended
trigger (`preplace`) + acceptance: `docs/examples/station-lock-from-preplace-fragment.json`; CPU proof:
`scripts/twin/tests/test_station_full_brake.py`.

### Clear the pre-place pan sweep of the cart cage (sim-preplace-sweep-cage-clearance-v1)

With the base fully locked (above) and the wrist held firm, coldwin70's leg-2 shoulder_pan sweep **stalled
hard** (a 1632.7 N·m effort spike, `STATION_FULL_BRAKE_ABORT`), froze at `shoulder_pan` 0.2012 (commanded
−0.4053) and failed `PHASE_CONVERGENCE`. CPU reconstruction of the executed sweep at the real park pose
(`scripts/check_preplace_sweep_clearance.py`, from the nav-trace poses + the `CARTON_TICK` joints) names the
contact: the wrist-first order **extends** the arm to the standoff shape *before* the pan sweep, so leg 2
rotates the already-extended, low **gripper end-effector housing** (the robot links `cube_without_surface_plates`
/ `surface_plate_1`) on an arc that rakes `/Rollwagen/chassis/post_2` (the +x mouth post, −48 mm) and the upper
side-wall bar `side_0_horizontal_6` at `shoulder_pan ≈ 0.20` — exactly the freeze. It is the **arm, not the
carton**: the carton is cage-filtered (`cage_collision: filtered` keeps deck/shelf/drawbar), so only an
*unfiltered* robot link can be the sim contact (cw69's `SEAL_DIAG hit_bodies=['/Rollwagen/chassis/post_2']`
`min_gap_mm 0.0` is the same contact).

Two opt-in pieces, both default off ⇒ byte-identical, compatible with the base lock (PR 365) and the firm
wrist hold (PR 359):

- **Pan-first reorder** (HOST `preplace_pan_first`, requires `preplace_wrist_first`): leg 1 sweeps
  `shoulder_pan` to the standoff while the wrists + lift + elbow **hold at the compact carry shape**
  (`PREPLACE_PAN_FIRST`), then leg 2 extends over the open deck. The tucked arc clears the mouth post/side bar
  by **+252 mm** (vs −48 mm); the carton is re-leveled in leg 2's extend and the 2.0° bound is untouched.
- **Cage-clearance gate** (injected `controller._preplace_cage_clearance`, built by
  `place_clearance.preplace_cage_clearance_provider` from the live coupled cart pose + host XRDF): refuses an
  **intersecting** leg-2 sweep by name before motion — `PREPLACE_SWEEP_CLEARANCE prim=… min_clearance=…mm` —
  instead of the blind jam. It sweeps the cuMotion **arm sphere** model (the real robot-link set incl. the
  housing) against the cage bodies (`place_clearance.cage_body_names`), refuse margin default 0.0 (an actual
  penetration — the CPU geometry carries a documented ~40–90 mm offset from the sim colliders, and the standoff
  itself sits an inherent +4.2 mm from the far wall, within that noise). The ≥ 0.04 m clearing margin is the
  design target the reorder meets on the raking bodies.

Recommended ON values + bounded-GPU acceptance: `docs/examples/preplace-sweep-cage-clearance-fragment.json`;
CPU proof: `scripts/check_preplace_sweep_clearance.py`, `scripts/twin/tests/test_preplace_sweep_clearance.py`;
write-up: `data/sim-preplace-sweep-cage-clearance-v1/report.md`.

### Carry-connected standoff for the pan-first leg 2 (sim-preplace-pan-first-seeded-leg2-v1)

coldwin72 — the first run with every physical fix stacked (pick sealed, the pan swept the cage clear with
the base locked + low arm effort, the wrist held) — was stopped **before any motion** by the carton-level
continuity gate in the **pan-first leg 2** (the extend above): the leg-2 `wrist_2` held ~0 for nine
waypoints then **jumped 0.300 rad** in one step (> the 0.15 rad continuity limit the soft wrist can track)
→ `CARTON_LEVEL_PLAN_REJECTED` at waypoint 10/18. The PR 349 seeded interior was **already applied** to
leg 2 (CPU-reproduced byte-for-byte); it cannot help. Root cause (CPU-proven): with `shoulder_pan` held at
the **standoff** value and the `carry_wrist1_cap` (0.65) self-collision cap in force, the carry/tucked
shape (`wrist_2` ~0) and the reach-valid reaction-natural standoff (`wrist_2` 0.668) sit on **disconnected
carton-level basins** — the standoff branch needs `wrist_1` up to ~1.30 near the start pose (far over the
cap), so no continuous monotonic ≤ 0.15 rad level path connects them (the crossing tilt ridge minimum is
~2.44° > the 2.0° bound, for *every* `wrist_1` cap).

- **Carry-connected branch** (HOST `carton_level.preplace_branch_prefer_carry_connected`, requires
  `preplace_reaction_branch`; default off ⇒ byte-identical): `select_carry_connected_standoff` enumerates
  the reach+level+self-clear standoff branches (seeded from the carry shape *and* the reach shape, `wrist_2`
  swept), ranks them by total wrist travel from the carry shape, and returns the **first** whose leg-2
  world-level plan the continuity gate **accepts** (step ≤ `preplace_max_wrist_step_rad` *and* tilt ≤ the
  bound) — the gate stays the authority. On cw72 it picks a `wrist_2` ~0.09 standoff whose leg 2 is a
  monotonic extension (max wrist step 0.143 rad, tilt 0.52°), and the place descent from it stays level
  (0.02°). Logs `PREPLACE_REACTION_BRANCH chosen wrist_2=… reason=carry-connected`. The PR 359 firm wrist
  hold now covers the soft-wrist drift that originally pushed PR 345 to the 0.668 settling branch, so the
  ~0 branch is admissible again. The search is bounded by `preplace_branch_select_budget_s` (measured
  ~3.8 s CPU, well under the budget and the render-paced wall of the 5 sim-s host-status watchdog).

Recommended ON values + bounded-GPU acceptance: `docs/examples/preplace-pan-first-seeded-leg2-fragment.json`;
CPU proof: `scripts/twin/tests/test_preplace_pan_first_seeded_leg2_v1.py`;
write-up: `data/sim-preplace-pan-first-seeded-leg2-v1/report.md`.

#### Discard descent-infeasible standoffs (sim-preplace-branch-descent-feasible-v1)

coldwin79 sealed the pick and converged the pre-place, then the carton-level **place-descent** gate
(`CARTON_LEVEL_PLACE_DESCENT`) refused the set-down **before motion** because the descent from the chosen
standoff would roll the carton **3.43°** (> the 2.0° bound) → `HOST_FAULT_ESTOP phase=place`. Runs 76/77/78
descended level from the *same* configuration. The carry-connected selector above ranked candidates only by
carry travel + the leg-2 continuity gate and returned the **first leg-2-accepted** branch — it never checked
the **place descent** from that standoff. The candidate set and each branch's leg-2 step depend on the run's
measured carry pose / sensed place target (a few mm/mrad of drift), so the greedy winner flipped from a
~0.59 `wrist_2` level branch (cw76/77/78) to the low `wrist_2`=0.052 branch whose descent is infeasible
(cw79) — a lottery.

- **Descent feasibility filter** (HOST `carton_level.preplace_descent_feasible`, requires
  `preplace_branch_prefer_carry_connected`; default off ⇒ byte-identical): the controller hands
  `select_carry_connected_standoff` a `descent_predictor` — the **same** `plan_compensated_place_descent` the
  place-descent gate runs, from each **candidate** standoff to the resolved support (same seed = the leveled
  `collect_place_preplace_shape`, base-frame up-axis, gravity compensation, `place_from_above` /
  `place_above_clearance_m` / `place_depth_bias_m` / `path_waypoints`). The selector **discards** any candidate
  whose predicted descent exceeds the bound and chooses **deterministically** by the stable key (least carry
  travel, then descent tilt, then leg-2 max wrist step, then `wrist_2` nearest the carry value) — independent
  of the search time / enumeration order, so the set-down is no longer a lottery. Logs
  `PREPLACE_BRANCH_CANDIDATES n=… chosen=… descent_tilt=… leg2_step=…` with both scores per candidate. The
  2.0° bound is **untouched**; the leg-2 continuity gate stays the approach authority. CPU-reproduced: the
  cw79 low-`wrist_2` standoff descent is ~3.5° (the logged 3.43° signature) while the mid-`wrist_2` branches
  are ~1.8°.

Recommended ON values + bounded-GPU acceptance: `docs/examples/preplace-branch-descent-feasible-fragment.json`;
CPU proof: `scripts/twin/tests/test_preplace_branch_descent_feasible_v1.py`;
write-up: `data/sim-preplace-branch-descent-feasible-v1/report.md`.

#### Reach the seat without the housing catching the cage (sim-place-descent-reach-cage-v1)

coldwin87 reached the pre-place standoff and pressed on, but the arm **jammed ~14 cm above the corrected
seat** (PR 375's 0.9325 world carton bottom) with the carton resting on nothing, at ~1013 N·m. CPU
reconstruction of the executed place descent (host-stack `TRAJ_HOST_PLAN`, 13 leveled waypoints, against the
cuMotion **arm-sphere** model at the real park/cart pose — `scripts/check_place_descent_reach.py`): the
descent plan starts from the standoff at the **carry** wrist orientation (`wrist_2` ≈ 0.28) and its first leg
reorients to the place **pad-down** orientation (`wrist_2` ≈ 1.63) **while the gripper housing sits inside the
+x cage mouth**, so the *unfiltered* housing link `cube_without_surface_plates` wedges against the upper side
bar `side_1_horizontal_6` (and the mouth posts) at the descent **start** — the **only** penetrating config of
the whole path (−1.9 mm cw87, −13.8 mm cw64, −33.8 mm cw88, −36.4 mm cw89; every other waypoint ≥ +0.10 m).
`wrist_2` then froze at 0.28 against the 1.63 rad command → `TRAJ_HOST_TRACK_MISS worst_j=4 err=1.30` →
`PATH_TOLERANCE_VIOLATED`. **Not a reach/joint limit** (the 1.63 rad target and the whole reoriented chain are
far inside limits and ≥ +0.10 m clear; only the standoff carry-wrist orientation penetrates).

- **Place-descent cage gate** (injected `controller._place_descent_cage_gate`, built by
  `place_clearance.place_descent_cage_clearance_provider` from the live coupled cart pose + host XRDF, exactly
  as `controller._preplace_cage_clearance` is injected for the pre-place gate; default `None` ⇒ byte-identical):
  the **same** cuMotion arm-sphere sweep (`place_clearance.place_descent_cage_clearance` =
  `preplace_cage_arm_clearance`) over the planned place-descent waypoints vs the cage bodies, at the **30 mm**
  design margin (the clear descent reads +0.10 m, far outside the CPU-vs-sim noise, so unlike the pre-place
  sweep's penetration-only 0 mm this gate uses the full margin). It does **two** things: (a) the carry-connected
  standoff selector's `descent_predictor` (`_preplace_descent_predictor`) now also treats a cage-**raking**
  descent as infeasible — not just one over the 2.0° tilt bound — so the selector **discards** the carry-wrist
  branch and advances to the **pad-down** branch whose descent holds the housing clear; and (b) the place-descent
  dispatch refuses an intersecting descent **by name** before motion (`PLACE_DESCENT_CAGE_CLEARANCE
  prim=side_1_horizontal_6 min_clearance=… cube_without_surface_plates …`) instead of the blind `wrist_2` jam.
- **The remedy descent** holds the pad-down branch (`wrist_2` ≈ 1.63) from the standoff to the corrected seat:
  CPU-proven ≥ +0.10 m from **every** cage body at every collect-place park pose, tilt **1.72°** (≤ the
  **untouched** 2.0° bound), landing the carton bottom at **0.9325** world. Requires `place_seat_gate` (PR 375)
  so the descent targets the corrected seat, not the +3.8 cm-too-high model support top.

**Open question — approach feasibility of the pad-down standoff.** The CPU proof certifies the **descent** from
the pad-down standoff; whether the pre-place **approach** can deliver the arm to that standoff continuously
(the PR 366/coldwin72 leg-2 tilt-ridge concern) is the pre-place lane's, validated in-sim at coldwin91. If the
gate still refuses with a residual wedge, the selector could not reach the pad-down branch — see the write-up.

Recommended ON values + bounded-GPU acceptance (coldwin91): `docs/examples/place-descent-reach-fragment.json`;
CPU proof: `scripts/check_place_descent_reach.py`, `scripts/twin/tests/test_place_descent_reach.py`;
write-up: `data/sim-place-descent-reach-cage-v1/report.md`.

#### Separate the approach from the reorientation (sim-place-pad-down-arrival-v1)

coldwin92 (honest seat gate + both cage gates live) exposed the last contradiction of the **single**-standoff
design: every pre-place standoff whose carried **approach** keeps the carton level is on the **carry-connected**
(low `wrist_2`) branch, whose straight descent **rakes the mouth posts** (correctly refused — `… all
descent-infeasible`); the only cage-clear descent holds the **pad-down** (high `wrist_2` ≈ 1.6) branch, which the
approach **cannot reach level** (the fallback tilted the carton **13.96°** → `CARTON_LEVEL_PLAN_REJECTED` →
`HOST_FAULT_ESTOP phase=preplace`). This resolves that **open question** above by **separating** the approach from
the reorientation.

- **CPU finding** (`scripts/check_place_pad_down_arrival.py`, controller `gripper_pad_tcp` frame): the reorientation
  from the carry-wrist arrival to the pad-down branch is a ~73° rotation of the carton **about the vertical axis**
  (yaw 155°→82°), which keeps it **level**. An in-place reorientation **directly over the seat** rakes `post_2`
  at every height (the post is tall, z[−0.196, 1.254]; raising to clear it degenerates the level IK above ~0.40 m).
  But reorienting in the **open approach corridor** — the standoff shifted `pad_down_arrival_shift_m` toward the arm
  base + `pad_down_arrival_height_m` above the seat — keeps the housing clear of the mouth posts. The shift **alone**
  is not enough (the carry-wrist arrival's own descent there is only +18.6 mm, below the 30 mm gate).
- **The three stages** (`carton_level.place_pad_down_arrival`; requires `place_from_above` + `level_place` + the wired
  `place_descent_cage_gate`; opt-in, default off ⇒ byte-identical): **(a)** carried approach to the **shifted**
  above-shelf standoff on the carry-connected arrival branch (`_preplace_above_place_goal`, `PREPLACE_ABOVE_PAD_DOWN`;
  the straight-descent feasibility filter is disabled in pad-down mode); **(b)** the in-place **reorientation** to the
  cage-clear pad-down branch (`_issue_pad_down_reorient` / `place_reorient.select_pad_down_branch` +
  `plan_level_world_approach(anchor='seeded')` — the carry-wrist and pad-down branches are distinct wrist-flip IK
  solutions, so a **branch-continuous** seeded ramp is required; a naive re-leveler jumps 1.568 rad), gated on tilt
  **and continuity** (wrist step) **and** the same descent cage sweep, a short slow `begin_joint_path`
  (`PLACE_REORIENT`); a reorientation the gates refuse `_fail`s by name (`PLACE_REORIENT_REJECTED`); **(c)** the PR 376
  `place_from_above` descent + descent cage gate + the PR 375 seat gate, from the pad-down pose.
- **CPU-proven** at every cw64–cw92 park (default `shift=0.20 m` / `height=0.10 m`): reorient **0.04°** / step
  **0.080 rad** / ≥ +66 mm cage, descent **1.25°** / ≥ +103 mm cage (the **untouched** 2.0° bound), carton on the
  **0.9325** world seat; the old single-standoff descent is **refused**.

Recommended ON values (coldwin94): `docs/examples/place-pad-down-arrival-fragment.json`; CPU proof:
`scripts/check_place_pad_down_arrival.py`, `scripts/twin/tests/test_place_reorient.py`,
`scripts/twin/tests/test_place_pad_down_arrival.py`; write-up: `data/sim-place-pad-down-arrival-v1/report.md`.

## Contact-gated place-descent release (sim-place-release-on-contact-v1)

The collect deck set-down releases the weld the instant the support is **actually touched**, not at a
fixed carton-z only. coldwin68 (the furthest run) jammed the carried carton's ~1.46° roll edge on cage
post `/Rollwagen/chassis/post_2` at the descent **start** (carton_z frozen 0.075 m above the computed
z threshold), so the PR 334 fixed-z gate never fired and the stiff place drive pressed to **757 N·m** past
the 600 N·m station full-brake guard; coldwin64, same config, descended fully and released cleanly at
125.8 N·m. The place drive gains are identical in both runs — the differentiator is the PR 359 pre-place
hold-firm, which altered the settle into the roll that fouled the post.

The fix (HOST `carton_level.place_release_on_contact`, opt-in, default off ⇒ byte-identical; the legacy
`place_contact_release` z gate is disabled when it is on, so no double-fire). `place_release.PlaceReleaseGate`
releases on whichever fires first:

- **effort** — the max arm joint effort (`sensors.arm_effort_max`) rising `place_release_effort_margin_nm`
  (default 100) over its pre-contact hold baseline for `place_release_effort_consecutive` ticks, so the
  press never reaches the guard;
- **stall** — the carton centre z (`sensors.placed_pose`) making no `place_release_stall_m` downward
  progress for `place_release_stall_ticks` ticks;
- **contact** — a measured seal gap ≤ `place_release_contact_gap_mm` (off by default) on a support body;
- **z** — the legacy carton-z threshold (fallback).

Every release is **gated on the touched body being the support**: a hit-prim path containing a
`place_support_tokens` substring (`shelf`/`deck`) is the support surface; a flush (≤
`place_obstruction_gap_mm`) contact on any other body (a cage `post_*`) is an **obstruction** — the
controller logs `PLACE_OBSTRUCTED prim=…` and does up to `place_reapproach_max` bounded lateral
re-approaches (shift the set-down `place_reapproach_shift_m`, re-descend via the standoff) before failing
by name, never a press or a drop against the post. The re-descend first brings the obstructed descent to a
terminal state (**sim-motion-active-guard-v1**, above) so the re-issue is never a `motion already active`
crash (coldwin90). The soft carried drive is restored for the descent via
`phase_drive.carried_soft_phases: ["place"]` (hold-firm is already preplace-scoped), bounding any residual
press to eff400 < the guard. The per-tick `PLACE_RELEASE_TICK` logs carton_z / effort / hit prim /
trigger. The 2.0° bound is untouched. Recommended ON values + acceptance:
`docs/examples/place-release-on-contact-fragment.json`; CPU proof (each trigger, the obstruction gate, the
coldwin68/coldwin64 replays): `scripts/twin/tests/test_place_release_on_contact_v1.py`.

### Completing the release when the place-cancel never terminates (sim-place-release-cancel-handshake-v1)

When the contact-gated release fires mid place-descent, the release phase must **stop** the in-flight place
motion and reach a **terminal, settled** result before it detaches (PR 334 serialization — never detach into
an in-flight planning scene). The default path **cancels** the place `ExecuteTrajectory` (`motion.cancel`).
coldwin74/75 both reached the set-down and fired `PLACE_CONTACT_RELEASE trigger=stall`, but the place descent
is a **driven trajectory on the sim `FollowJointTrajectory` executor**, and that action cancel is **both**
dropped in the goal-accept race (`self.handle is None` at the stall, so `cancel_goal_async` never goes out —
cw75 `host-stack.log` the cancel at sim 245.600 while the goal is still being accepted / `TRAJ_HOST_START` at
246.217) **and** unsafe (the sim `trajectory_host` treats a goal cancel as a **fault** that latches the global
`/manipulation/stop`). `cancel_outcome()` stayed `None` the whole budget (cw74 2.0 s, cw75 8.0 s) and
`HOST_FAULT_ESTOP phase=release` fired with the carton still welded on the shelf.

The fix (HOST `carton_level.release_cancel_handshake`, opt-in, default off ⇒ the legacy `motion.cancel` path,
byte-identical) stops the descent on its **own channel**: the release phase emits the **contact-hold** control
(`/restock_sim/arm_contact=True`) so `trajectory_host` **succeeds holding the current pose** (terminal, no
fault, no global stop) and marks the port cancel-requested via `settle_in_place()` (no `cancel_goal_async`),
so `cancel_outcome()` reads the resolved execute future → `succeeded` → gripper-open + detach from the settled
pose. As a belt-and-braces **stopped-arm fallback**, if the terminal still never arrives but the arm is
**provably stopped** — the measured max joint speed (finite difference of `sensors.joint_positions`) stays ≤
`release_cancel_stopped_vel` (default 0.02 rad/s) for `release_cancel_stopped_ticks` (default 3) consecutive
ticks — the release logs `RELEASE_CANCEL_STOPPED_FALLBACK`, forces the port terminal (`force_terminal_settled`,
no e-stop) and detaches. It **never detaches into a moving arm**: a joint still advancing above the threshold
resets the tick count, and if neither the terminal nor the stopped-arm proof lands within the budget the
release still fails fast by name. Recommended ON values + acceptance:
`docs/examples/place-release-cancel-handshake-fragment.json`; CPU proof (the adapter `settle_in_place` /
`force_terminal_settled` contract, the hold-and-settle, the stopped-arm fallback, a moving arm never
detaching, and a cw75-trace replay OFF/ON): `scripts/twin/tests/test_place_release_cancel_handshake_v1.py`.

### Re-issuing the gripper-open so the weld release cannot be dropped (sim-release-open-reissue-v1)

The gripper-open that releases the weld at the end of the release phase is a **one-shot** control.
coldwin84 — a **byte-identical re-run** of the cleanly-released coldwin83 — **hung** at the cart set-down:
the release fired and the release phase emitted the gripper-open **once**, but the Isaac side ran **zero**
`cad_tool.command(open)` — `coldwin84.console.log` has **no** `SEAL_DIAG command:open` lines (the last
`SEAL_DIAG` is still `command:close … enabled=True`) while coldwin83 has exactly **one**. The single
open-carrying `/manipulation/status` was overwritten on the **last-value** control relay (`control_ipc`
keeps only the latest status per topic) by the next, no-control release tick's higher `sequence` before
`IsaacRosGraph.poll` read it — the **same one-shot drop PR 356 fixed for the contact enable**. The
`SIM_ATTACH_FALLBACK` weld therefore never released and the run e-stopped: `HOST_FAULT_ESTOP phase=release
'release did not detach the held carton within 8.0s sim of the gripper-open command'`. Five earlier runs on
the identical config released cleanly, so the release was a **dice roll on relay timing**.

The fix (HOST `carton_level.release_reissue`, opt-in, default off ⇒ the single-shot open + the 8 s
`release_detach_timeout_s` path, byte-identical) **re-emits the gripper-open control** — which on the Isaac
side is **both** the open **and** the weld detach (`cad_tool.command(False)` → `open_gripper` +
`_release_fallback`) — on **every** release tick until the per-tick feedback (`RESTOCK_CONTACT_STATUS_FEEDBACK`
/ PR 358: `gripper_enabled` + `gripper_status_time`, and the `s.contact` latch) confirms the gripper **open**
and the carton **not held** (seal released), then proceeds to the cuMotion planning-scene detach
(`update_payload`, issued **once** from the confirmed-released state). Every release status now carries
`controls=[open]`, so whichever one the last-value relay hands `IsaacRosGraph.poll` lands the open — the
relay can no longer drop it. It logs `RELEASE_REISSUE n=..` once per `release_reissue_log_every` ticks and
`RELEASE_CONFIRMED` with the tick count; if the feedback never confirms within `release_confirm_timeout_s`
sim it **fails by name** (`RELEASE_NOT_CONFIRMED`, with the `held` / `gripper_open` / `status_fresh` fields)
rather than the opaque 8 s never-detach e-stop, **never** restarting the stage with the carton welded. The
confirmation uses `not s.contact` as the authoritative physical seal-released signal, corroborated by the
gripper-open feedback when a fresh status is present. Recommended ON values + acceptance:
`docs/examples/release-open-reissue-fragment.json`; CPU proof (the pure `ReleaseReissueGate`, the controller
re-issue/confirm/timeout branches, OFF byte-identical, and coldwin84 (hang OFF, confirmed ON) / coldwin83
(single-shot success still confirms) replays): `scripts/twin/tests/test_release_open_reissue_v1.py`.

### Releasing at the TRUE set-down, not the render-paced startup stall (sim-place-verify-rest-tolerance-v1)

The contact-gated release's **stall** trigger (above) fired on the **descent START** height in coldwin85
and coldwin83, ~0.06 m **above** the planned set-down, leaving the carton cradled above the shelf — and
the collect **place verify** (`controller.py` `verify` phase, `sensors.placed_pose` vs `target_surface`:
`carton not resting on deck (xy bottom_gap)`) then caught the consequence. coldwin85 was an **exact re-run**
of coldwin83: both set the carton down and released it, but coldwin85's verify failed `xy=1.219
bottom_gap=-0.954` (the carton came to rest ~1 m below the shelf, **on the floor**) while coldwin83 passed
the same check. nav-trace (`RESTOCK_TRACE_POSES`) shows the cart and base **did not move** in either run —
the verify frame is correct; the carton genuinely left the shelf. The differentiator is the release pose:
both `PLACE_RELEASE_TICK` traces have the measured carton centre z **frozen** at the standoff (cw85 0.6251,
cw83 0.6256) for the first few render-paced ticks because the descent trajectory had not produced motion
yet, so the "no downward progress for `place_release_stall_ticks`" rule read that **startup** freeze as
"support met" and released the weld at carton_z 0.625 while the planned set-down (`PLACE_CONTACT_RELEASE_ARM`
`release at carton_z<=…`) was 0.5628/0.5635 — **0.062 m lower**. Released high and still cradled in the side
grip, whether the carton slips free onto the shelf (cw83: dropped 0.125 m, settled world z=1.022, stable
through verify) or stays perched and is carried off by the retreat fold (cw85: rode up to world z=1.77 as the
arm folded, then fell to the floor at z=0.08) is a coin flip — the rest is **non-deterministic**.

The fix (HOST `carton_level.place_release_stall_height_margin_m`, opt-in, **default None ⇒ byte-identical**):
a descent **stall only counts as "support met" when the carton centre z is within `release_z + margin`** of
the planned set-down. A stall still above that is the render-paced **startup** stall and is **suppressed** —
the gate keeps pressing, the descent reaches the true set-down, and the weld releases on the **z / effort /
support-contact** trigger at the height where the carton actually rests on the shelf, so the rest is
deterministic in both runs. It is **inert when `release_z` is None** (no reference height), and a genuine
stall *inside* the window (real support contact) still fires. This does **not** touch the 2.0° carton tilt
bound and does **not** change the verify: it makes the carton actually rest on the support (the verify stays
the honest `carton not resting on deck` gate, never loosened). Recommended ON value + acceptance:
`docs/examples/place-verify-rest-fragment.json`; CPU proof (the startup-stall suppression, the release at the
true set-down, a real near-set-down stall still firing, OFF byte-identical, and the coldwin85/coldwin83
replays): `scripts/twin/tests/test_place_verify_rest_tolerance_v1.py`.

### Releasing at the carton-CENTRE frame + a free-and-resting retreat (sim-release-threshold-carton-frame-v1)

coldwin87 showed the previous fix was aimed at the wrong frame. `carton_z` / `sensors.placed_pose[2]` is the
carton **CENTRE** (`types.py`), but the geometric contact-release threshold `place_contact_release_z`
(`start_carton_z − drop + gap`) expressed the carton **BOTTOM** frame (support_top + gap), so it sat one
carton **half-height** too low: cw87 `PLACE_CONTACT_RELEASE_ARM start_carton_z=0.6422 descent_drop=0.0907
gap=0.0120 → release at carton_z<=0.5635`, while the welded carton rested on the shelf at carton_z **0.6422**
(the `PLACE_RELEASE_TICK` plateau — `half − gap = 0.09 − 0.012 = 0.078` m **above** 0.5635). With the
startup-stall suppression (`place_release_stall_height_margin_m=0.02`) on, the stall trigger could not fire
either, so the stiff place drive pressed the welded carton into the shelf → `STATION_FULL_BRAKE_EFFORT` →
`PATH_TOLERANCE_VIOLATED` → `HOST_FAULT_ESTOP`. The stall **was the real contact** all along.

**Fix 1 — threshold frame (automatic, no new key).** `carton_level.place_contact_release_z_centre` adds the
carton half-height (`profile.dimensions[2]/2 = 0.18/2 = 0.09`) so the threshold is in the same **carton-centre**
frame as the measured pose: cw87 `0.6422 − 0.0907 + 0.09 + 0.012 = 0.6535`, i.e. the geometric z trigger fires
at the measured rest (carton **bottom** at `support_top + gap`), **consistent with the stall trigger**, not
0.078 m below it. The `PLACE_CONTACT_RELEASE_ARM` log now prints `release at carton_z(centre)<=…`. This is
automatic inside the already-opt-in `place_contact_release` / `place_release_on_contact` mechanism (no config
key; byte-identical when that mechanism is off). **Keep PR 373's `place_release_stall_height_margin_m` OFF** —
with the frame corrected the stall trigger fires at the true contact the same place the z trigger does, and
leaving the margin on is what suppressed cw87's stall into the press.

**Fix 2 — free-and-resting retreat (`carton_level.release_free_check`, opt-in, default off ⇒ byte-identical).**
coldwin85/coldwin86 released at the set-down and the retreat's first motion swept the barely-rested carton off
the shelf (verify `carton not resting on deck`, cw85 `bottom_gap=-0.954`, cw86 `-0.815`); coldwin83 on the same
config happened to rest. The fall was a **retreat** problem — the open gripper still cradled the carton. With
`release_free_check` on, `release_free.ReleaseFreeGate` **holds** after the detach until the carton CENTRE z is
stable **and** the carton bottom is resting (`|bottom_gap| ≤ release_free_rest_gap_m`) for
`release_free_settle_ticks` ticks, **then** raises the gripper straight up by `release_free_lift_dz_m` (a pure
vertical clearance lift, reusing the collect retreat lift-off IK) **before any lateral/pan retreat**, and
verifies the carton did **not** follow the gripper up (rose `< release_free_follow_frac` of the lift). It
**fails by name `RELEASE_NOT_FREE`** when the carton never rests (`release_free_settle_window_ticks`) or follows
the gripper. Needs a measured `placed_pose` (`RESTOCK_VERIFY_PLACE`); absent ⇒ the check is skipped (real rig).
Logs `RELEASE_FREE_BEGIN` / `RELEASE_FREE_LIFT` / `RELEASE_FREE_CLEAR`. Neither fix touches the 2.0° carton tilt
bound. Recommended ON values + acceptance: `docs/examples/release-threshold-carton-frame-fragment.json`; CPU
proof (the centre-frame threshold at the cw87/83/85/86 numbers, the free-and-resting settle/lift, the
following-refusal, OFF byte-identical, controller wiring):
`scripts/twin/tests/test_release_threshold_carton_frame_v1.py`.

### The physical seat, and the honest seat-gated release (sim-place-seat-geometry-v1)

PR 374's premise above — *"the welded carton rested on the shelf at carton_z 0.6422"* — is **wrong**, and
that is what made it calibrate the threshold to the stall instead of the seat. Established from the composed
`mission.usda` (usd-core) **and** each run's `nav-trace.jsonl` carton WORLD poses (the test-backed CPU tool
`scripts/check_place_seat_geometry.py` + `scripts/twin/restock_manipulation/place_seat_geometry.py`):

- **The true seat** is the upper-shelf **FLAT TOP of the SETTLED cart**: carton **bottom 0.9325** world /
  centre 1.0225 — proven by **coldwin83** (released + rested + verify PASSED, final carton centre 1.0225).
- **The model support top `0.9707` is +38.2 mm too high**, in two parts: **+7.3 mm** because it resolves the
  slightly-**tilted** shelf's world-AABB max-z **CORNER** (0.9712) instead of the **flat face** the carton
  sits on (0.9639 at the composed pose), and **+31.4 mm** because it is read at the cart's **un-settled** build
  chassis (z 0.2315) while the dynamic cart **settles ~3.1 cm** (chassis → ~0.20) by set-down. The carton
  collider **half-height (0.09)** and the **carton-CENTRE frame** are **correct** — PR 374's `+half` fixed a
  term that was already right.
- **coldwin87's "solid stall" is carton bottom 1.0734 world = 14 cm ABOVE the true seat.** The carton is
  filtered (`PhysicsFilteredPairsAPI`) against the **whole** rollwagen cage, so it collides only with the deck
  (0.26) / shelf (0.9325) — both far below — and **rests on nothing**: it is held by the arm (effort peaked
  **1013 N·m**) with a side-grip robot link jammed on the cage (robot links are **not** in the carton's cage
  filter; the horizontal side-grip seal raycast sees nothing). So the stall is an **OBSTRUCTION**, not a shelf
  contact; releasing there drops the carton 14 cm (cw85/86 fell off during the retreat; cw83 happened to rest).

**Fix (`carton_level.place_seat_gate`, requires `place_release_on_contact`; opt-in, default off ⇒ byte-identical).**
The `place_release.PlaceReleaseGate` is armed with the corrected seat (support TOP + carton half, in the
measured `placed_pose` carton-centre / `ur_base_link` frame) and `place_seat_tol_m` (0.02 m). A measured
contact releases the weld **only** within tol of the seat; a contact well **above** the seat is
`PLACE_OBSTRUCTED` (bounded lateral re-approach, else fail by name) — never a 14 cm drop; a descent that
reaches/passes the seat with **no** contact is a named `PLACE_SEAT_FAULT`. The release threshold becomes the
**absolute** seat (`support_top + half`), **not** PR 374's relative `placed_pose − drop + gap + half`, so it
can never sit **above the descent start** — the controller asserts that by name (`PLACE_SEAT_THRESHOLD_ABOVE_START`,
the coldwin89 bug) at arm time, and `place_release_stall_height_margin_m` is retired. The support top is
resolved as the slab **flat top face** of the live cart with the env `RESTOCK_PLACE_SUPPORT_FLAT_TOP=1`
(`cart_support.slab_flat_top_z_world`; default off ⇒ the AABB corner, byte-identical). The shared predicate is
`place_seat_geometry.classify_set_down`; `release_free_check` / `release_reissue` still guard the retreat once a
true set-down occurs. ON values + the bounded GPU proof: `docs/examples/place-seat-geometry-fragment.json`;
CPU proof (the geometry derivation, the seat-gate triggers, the cw83/85/86/87/88/89 replays, OFF
byte-identical): `scripts/twin/tests/test_place_seat_geometry.py`.

## Place-on-cart between-posts target (sim-place-target-between-posts-v1)

The collect place-on-cart load-centre on the rollwagen **upper shelf** is moved from the old
`(0.135, 0.0)` to the symmetric cage centre **`(0.0, 0.0)`** (cart body frame). The honest carton+pad
sweep clears the mouth posts by **0.258 m** there (post_2/post_3) versus **0.185 m** at the old spot, the
carton sits visually centred in the cage with equal front/back wall margin, and the standoff is
reach-proven level (≤ the 2.0° bound) from the real park pose. Recommended values (nested
`manipulation.cart_placement` + the matching `initial_staging.cart_load_center_body_xy`):
`docs/examples/place-target-between-posts-fragment.json`. The CPU geometry + reach check and its verdict:
`scripts/twin/restock_manipulation/place_target_between_posts.py`
(`python3 -m twin.restock_manipulation.place_target_between_posts`), CI-pinned in
`scripts/twin/tests/test_place_target_between_posts.py`.

**The new spot is necessary, not sufficient — and the old spot was NOT statically marginal.** Measured
from the three runs (nav-trace + host-stack): all three **parked the base at the same world pose**
`(-4.61, 14.77)` through t≈195. coldwin64 kept it there and the carton landed at the cart-frame nominal
`(0.131, 0.015)`, clearing every post. coldwin68 / coldwin69 **shoved the base ~0.33 m west during the
loaded place phase** (cw68 −4.61→−4.93 at t≈227; cw69 →−4.96 at t≈233 — both still `parked=True`), so the
arm, executing in `ur_base_link`, set the welded carton down ~0.5 m off-nominal in cart-y (~`(0.33, −0.52)`)
straight onto `post_2`. The place **target** is already resolved cart-relative from the live cart prim
(`isaac_factory._gt_place_target` → `ground_truth_collect.slot_pose`); it is the base *moving between
target-resolve and set-down* that breaks the cart-relative landing. A ~0.5 m cart-y excursion cannot be
absorbed by any spot on the 0.64 m-wide shelf with a 0.45 m carton (required margin 0.04 + 0.516 m
measured spread ≫ the ~0.38 m max shelf post clearance), so `evaluate` FAILs `absorbs_drift` for **every**
spot and names the shove. The blocking fix — hold the base through the loaded set-down (full six-axis
station lock from the carried pre-place) or re-resolve the place target at descent time — is the collect
place-descent control, lane **sim-station-lock-from-preplace-v1**. Adopt this nominal together with it.

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
