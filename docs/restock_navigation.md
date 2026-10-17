# Mobile restock navigation

`scripts/twin/navigation/` adds two simulation navigation routes. Neither has
been run against a moving Isaac robot yet. CPU tests are deterministic fixtures,
not evidence of autonomous physics operation. Commands live in RUNBOOK §12.

The controlled static-scene temporal-map handoff, required-map mode, epoch
limits and remaining native checks are in [navigation_temporal_mapping.md](navigation_temporal_mapping.md).

## Articulated route

`articulated.ArticulatedController(grid, spec=MobileSpec())` replaces the
full-swing-circle planner for coupled towing. Its `start`/`step` interface matches
the reference controller. It uses the mobile module's actual `occupied_envelope`
and local `swept_envelope`, separate base/cart yaw, forward Dubins/hybrid-A*
primitives, minimum radius 1.5 m and the 65 degree articulation limit. The trailer
model predicts cart yaw through an off-axle hitch equation; actual measured
articulation initializes every runtime braking rollout. It is a quasi-static
no-side-slip approximation, not a claim about free-caster PhysX dynamics.

The 0.05 m geometry margin remains mandatory and localization uncertainty is
added separately. Straight passage through a 2.5 m portal now passes, while an
in-aisle turn still fails. Optional `start(..., goal_articulation=0.08)` requests
aligned final parking; default arrival permits any articulation below the rig
limit. Shelf-facing arm yaw is separate from the chassis goal yaw.

For the scenario goal `[-3.05, 6.0, pi]` and staging support at `[-5.73, 17.1]`,
planning and feedback fixtures complete `robot_dock -> stock_G3_back -> robot_dock`
against all 798 AABBs with **0.05 m geometry margin plus 0.10 m localization
allowance**, matching the calibrated MapLocalizer fixture output. The checker is
`scripts/check_restock_navigation.py`; commands are in RUNBOOK §12. Both legs
stop within the chassis pose tolerances, with peak articulation about 46 degrees.
The cart remains coupled and articulated at arrival: this is not an aligned
re-latch or exact restoration of the cart's initial body pose.

For an offset lidar, construct `Scan(..., sensor_to_base=(x, y, yaw))` using its
calibrated origin and heading in base coordinates. `Scan.points(base_pose)` and
`MapLocalizer` apply this transform before projecting into the map.
`IsaacRangeSensor` accepts the same keyword; its pose callback still supplies
the physical **sensor** pose, so do not apply the offset twice. Full-circle
coverage about the base is reduced by the offset distance. ROS consumers must
preserve the sensor frame and calibrated TF; displaced polar ranges cannot be
relabelled `base_link` or used with the base-origin ROS input contract unchanged.

Native closest-hit queries must also identify self collision geometry. Pass
`self_paths=(robot_root, cart_root)` to `IsaacRangeSensor`; these are exact USD
subtree roots, not name substrings or spatial masks. Missing identity remains
an external obstacle. Self hits terminate the ray: the adapter never queries
through a robot/cart to invent visibility of the space behind it.
The [PhysX closest-hit API](https://docs.omniverse.nvidia.com/kit/docs/omni_physics/107.3/extensions/runtime/source/omni.physx/docs/api/python.html)
supplies the collision and rigid-body paths used for this classification.

`Scan.unknown` marks angles where **any** sampled height was self-occluded.
Those polar ranges are NaN, not maximum range. Visible external hits at other
heights remain in `Scan.external_returns` as `(beam_index, range)` pairs;
`points()` and MapLocalizer retain them. Preserve both fields when adapting
scans. A raw polar transport loses those extra obstacles and must not be used
to certify coverage: NaNs without the explicit mask fail validation. Unknown
sectors cannot be relabelled free even if another height sees over an obstacle.

The gate certifies coverage **directionally**, against the forward braking sweep
in the direction of travel rather than the full circle: every bearing the sweep
reaches beyond the footprint must be covered by a non-self observed beam, while
bearings whose sweep stays within the footprint (and the never-swept rear) are the
robot's own body and need no external observation. A self-occluded ring behind the
base with a clear forward sweep therefore drives, a genuinely unobserved forward
sector still refuses, and a stationary robot is certified so it can settle at the
goal. `coverage_radius` stays only as the legacy full-circle diagnostic / fixture
fallback; the gate consumes the `Observation.coverage` field. The intended
lidar/coverage model is specified in
[docs/navigation_temporal_mapping.md](navigation_temporal_mapping.md); counterfactuals
are in `test_navigation_coverage.py`. All-self scans still fail localization for
insufficient external returns, and a scan blind in the forward sweep still stops.
Timestamp gates, the .10 m localizer floor and .05 m geometry margin are unchanged.

Sensor heights are absolute world Z values. The default x=.46 ray column at
z=.15 is inside the chassis (world z=.14..34), whereas the authored lidar_frame
is at world z=.39. CPU source-derived box tests show all 360 sectors hidden
with the default column. Using only z=.39 still leaves arm/cart occlusions;
raising the sensor does not prove low-obstacle visibility. Integration must
author and validate physically supported mounts and observed-space coverage
before motion; neither this CPU fix nor the earlier map-only route fixtures
certifies native navigation. `query=` injection is for deterministic CPU tests,
not a replacement sensor in an autonomous demonstration.

Calibration does not reduce localization uncertainty. The in-process localizer
retains its conservative 0.10 m floor, not a fitted covariance or a 0.02 m cap.
An actual-map ideal-ray counterfactual at a 0.46 m sensor offset recovered the
declared dock/shelf base poses with calibration; omitting calibration biased X
by 0.10 m. A staging support only 0.76 m north of the initial chassis intersects
the full-margin transport envelope and correctly stops motion. The revised
support leaves 0.86 m clearance. The earlier 0.02 m fixture cannot certify a
0.10 m pipeline; the revised scenario was checked separately at that larger
bound. CPU ray checks and quasistatic feedback are not PhysX evidence or a
continuous sensor-pipeline demonstration.

The integration scenario can specify a different shelf pose and adds a staging
support. Pass `--scenario` to the CPU checker and `--integration-scenario` to
both Nav2 launch and the host bridge so all three use the same obstacle map.
The earlier scenario pose `[-3.05, 5.85, pi]` intersects `G3_back_frame` with the
transport envelope; use the validated scenario rather than that old pose.
Checker reports include input hashes and independent robot/cart body poses;
use the observed cart yaw for manipulation, never the chassis goal yaw.

At the larger uncertainty, the old route really collides at portal/cabinet
corners. A timeout does not prove the passage itself is infeasible.
`environment_grid` derives passage direction from the manifest's portal centre
and neighbouring route hints. The planner tries a Dubins entry, aligned straight
passage and Dubins exit before unconstrained search, carrying independent cart
yaw continuously through all segments. These are search hints only: every arc
and swept envelope is validated at full margins against current obstacles.
Blocked hints fall back to search. There is no store-specific waypoint embedded
in the planner, no pose reset and no unchecked point-robot route.

The planner also tries cached or proposed routes before full search. A proposal
may omit localization inflation during
search, but **acceptance always replays and collision-checks every primitive at
the full requested allowance**, from the measured start. Teleports, stale start
poses, invalid curvature, collisions and wrong goals reject a seed. Dynamic
obstacles never inherit a cached clear verdict. The feedback tracker follows
checked curvature instead of cutting portal corners with a long lookahead.

These are kinematic test fixtures, not PhysX or localization evidence. Runtime
uses measured uncertainty, with a 2 cm minimum; it does not force uncertain
localization down to 2 cm. The simple scan matcher below still supplies a larger
uncertainty and can still reject a route when its measured residual increases.
The checker defaults to 0.10 m; `--position-uncertainty` selects the evidence-backed
fixture bound. Runtime always uses its current measured uncertainty. AMCL must
actually converge to sufficient accuracy or select a route with more clearance.

`scripts/setup_restock_nav2.py` now supplies a task-owned Jazzy overlay by
hash-verifying apt archives and extracting them without package installation or
maintainer scripts. The overlay resolved 208 packages and the CPU Nav2 launch
loaded successfully, then waited for missing sensor TF. No host upgrades or
GPU job occurred. `ArticulatedNav2Adapter` now sends the validated articulated
path directly to Nav2 `FollowPath`, publishes the measured combined footprint,
and applies the articulated velocity/braking gate to Nav2 proposals. Generate
its configuration with `--articulated`: Nav2's fixed-shape predictive collision
check is disabled for this mode because the final gate predicts the separate
cart yaw. Never send raw Nav2 commands to an actuator. The in-process articulated
controller remains available without ROS.

`scripts/run_articulated_nav2_bridge.py` runs with host Jazzy Python, avoiding
the Isaac 3.11/host ROS 3.12 ABI mismatch. In isolated namespace `restock`, it
subscribes to `encoder_odom` (`nav_msgs/Odometry`, odom/base_link), `scan_input`
(`sensor_msgs/LaserScan`, base_link), and `measured_rig_state`
(`Float64MultiArray`: simulation timestamp, coupled 0/1, arm-transport 0/1,
measured articulation). Publish goals to `navigation_goal` as JSON String
`{"pose":[x,y,chassis_yaw],"coupled":true}`. It publishes `safe_cmd_vel` and
`navigation_status`. Missing/stale inputs produce zero, including wall-clock
staleness when simulation timestamps stop advancing. The initial AMCL pose is
an explicit deployment prior, retried only while localization is absent.

## Integration contract

Import with the repository's `scripts` on the Python path. No module starts
SimulationApp, installs packages, drives hardware, or reads a robot world pose
as localization.

- `GridMap(bounds, obstacles, resolution=0.20)`: metres, map frame, AABBs
  `[xmin,ymin,xmax,ymax]`. Outside bounds is occupied; include all holes,
  fixtures, and unknown regions in obstacles. Static environment geometry is a
  legitimate prior. Do not regenerate the map from live ground-truth obstacles.
- `Footprint(base_radius, hitch_offset, cart_hitch_to_farthest_corner,
  margin=0.10, min_turn_radius=0.0, max_articulation=pi)`: actual rig measurements.
  `base_radius` includes the tucked arm, payload, bumpers and wheels. The cart
  distance includes drawbar, load and every corner, measured from its hitch.
- `Controller(grid, footprint).start([x,y,yaw], coupled)` then
  `step(dt, Odometry(...), Observation(...), now=simulation_time)` returns
  `Result(linear, angular, status, reason)`; `.velocity` is `(m/s, rad/s)`.
  Apply this command every physics control tick, including all zero commands.
  Status is `running`, `succeeded`, or latched `failed`. Cancel commands zero.
- Odometry is **map-localized wheel odometry**, with timestamp, measured linear
  and angular speed, and position uncertainty. Observation contains timestamp,
  map-frame range returns, measured coupling, measured arm transport state,
  measured articulation and usable scan coverage radius. Missing/stale sensing,
  unexpected coupling, arm movement, articulation limit, insufficient coverage,
  and obstacles all stop motion. A blocked condition fails after 8 seconds;
  a goal has a 180 second deadline. Integration must retain an independent
  actuator watchdog if this process or its control ticks stop.
- Goal success requires 8 cm XY, 0.08 rad yaw and measured speed below 2 cm/s,
  0.03 rad/s. A shelf/cart **approach** goal is the docking interface: this does
  not claim a hitch latched. The mobile/cart adapter owns physical coupling
  and its measurement. No reverse docking maneuver is supported.

Every orientation of the cart is enclosed by a circle around the base:
`max(base_radius, hitch_offset + cart_hitch_to_farthest_corner) + margin`.
Localization uncertainty and braking travel further enlarge the gate. Thus the
required corridor width is larger than twice this radius. A cart cannot be sent
through the existing 1.5 m aisles merely because its straight rectangle fits.
This is deliberately conservative, and goals at narrow shelves can be rejected.
The environment/mission must supply a reachable wide staging area, park the cart,
and use an uncoupled approach where feasible. The gate never shrinks a footprint
to make a route succeed.

## In-process route

`Controller` supplies A* over swept-clearance-checked grid edges, dynamic replan,
and a forward stop/turn/drive tracker. Coupled reference control requires a rig
that physically permits zero-radius base turns (swiveling cart/casters within
its measured hitch limit). Nonzero minimum turning radius **fails explicitly**
and requires the Nav2 route. No reverse is emitted.

`sensors.WheelOdometry` integrates measured continuous wheel joint positions.
`sensors.MapLocalizer` fits range endpoints to the static map from a declared
initial map pose, propagating each correction through wheel odometry. It is a
bounded local scan matcher, not global relocalization; repetitive corridors can
be ambiguous and wheel slip can exceed its capture range. Its uncertainty is a
fit-based allowance, not a calibrated probabilistic safety bound. Poor fits and
insufficient mapped returns fail closed. Use AMCL for the ROS route.

`runtime.SensorNavigation(controller, wheel_odometry, localizer)` joins these
pieces. Its `step(dt, wheel_angles, scan, now=..., coupled=...,
arm_transport=..., articulation=...)` returns actuator velocity/status directly.
It never integrates a commanded velocity to invent feedback.

`sensors.IsaacRangeSensor(get_sensor_pose, heights=..., beams=360)` calls Isaac
PhysX collision ray queries in an **already booted** simulation. Physical sensor
pose is used only to generate range data; navigation receives no true pose.
This idealized collision scanner does not model RTX intensity, transparent
materials, noise or all obstacles between sampled heights/rays. Returns from
the robot/cart are **not erased**: self-occlusion stops navigation. Integration
must place sensors with appropriate coverage and supply a base-frame scan (or
fuse calibrated multi-sensor scans); the simple built-in sensor assumes XY/yaw
coincident with the base frame. A scan produced from that mounting may be
unusable on a particular rig. That is a visible sensor integration failure,
not permission to substitute a clear-space fixture or hide cart returns.

## NVIDIA Isaac / ROS route

Read-only inspection found ROS Jazzy and Isaac Sim **5.1.0-rc.19** installed.
The existing shelfbot repositories have manipulation tooling but no Nav2/mobile
navigation packages. Nav2 itself is **not installed** in `/opt/ros/jazzy` at the
time of implementation; no system packages or shared environments were changed.
An integration-owned compatible ROS environment with `nav2_bringup`,
`nav2_smac_planner`, `nav2_regulated_pure_pursuit_controller`, `nav2_amcl`,
`nav2_msgs`, `rclpy`, `tf2_ros` and PyYAML is required. Isaac's Python ABI may
require its bundled ROS bridge libraries; do not mix system Python 3.12 ROS
binaries into Isaac's Python 3.11 environment.

The version-matched [NVIDIA Isaac 5.1 Nav2 tutorial](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/ros2_tutorials/tutorial_ros2_navigation.html)
provides the ROS bridge/Nav2 integration route. NVIDIA now labels that release
unsupported; this lane preserves the installed version rather than upgrading it.
The [Jazzy Smac Hybrid planner](https://docs.nav2.org/jazzy/configuration_and_development/configuration_guide/planners_plugins/smac/smac_hybrid/configuring_smac_hybrid/)
supports a minimum turning radius and forward-only Dubins planning. Generated
parameters select it with regulated pure pursuit, disable reversing and
rotate-to-heading, and use the **full coupled radius** for both costmaps.
A small behavior tree replans while following; it has no spin/backup recovery.
The reference [PhysX query API](https://docs.omniverse.nvidia.com/kit/docs/omni_physics/107.3/extensions/runtime/source/omni.physx/docs/api/python.html)
is the range sensor's simulator API.

`nav2.write_assets(output_dir, grid, footprint)` derives map PGM/YAML and Nav2
parameters from the installed Jazzy template; its optional `template` argument
supports CPU verification against the upstream Jazzy template. Generated maps
include occupied cells touching obstacles, preserving thin walls. Generate the
coupled configuration once; keep its conservative radius when uncoupled.

`nav2.Nav2Adapter(safety_controller)` runs under namespace `restock`, domain
**88**, loopback only. It publishes `/clock`, wheel `odom`, `odom -> base_link`
TF and base-frame `scan`; AMCL supplies `map -> odom`. TF is consistently
remapped to `/restock/tf` and `/restock/tf_static`. Call
`initialize_pose(declared_map_prior)` after AMCL activates; this is an explicit
initialization prior, never a stream of true world poses.

Call `publish_sensors(raw_wheel_odom, scan)` each tick, `start(goal,coupled)`, then
`step(dt, raw_wheel_odom, scan, coupled=..., arm_transport=..., articulation=...)`.
The adapter consumes Nav2's isolated `cmd_vel` and **returns** the safety-gated
velocity. Never connect Nav2's raw command directly to the wheels. AMCL
covariance/freshness, TF freshness, command freshness, swept clearance, braking,
turn radius, reverse prohibition and measured goal arrival all gate execution.
The adapter owns its own ROS context; `close()` cancels only its goal/context.
External Nav2 nodes are owned by the launch process/integration, not this module.

## Moving obstacles (dynamic-obstacle layer)

There are two deliberately separate inputs for moving obstacles. The isolated
Nav2 route's standard `ObstacleLayer` consumes the robot's navigation `scan`, so
transient depth/range returns mark the local Nav2 costmap just like static
returns. Its final actuator command still passes the articulated safety gate,
which uses the **full towed-cart swept envelope**. The sensor-navigation route
uses `Observation.obstacles` instead, so it can receive an explicit detection
stream before the same coupled gate and planner.

The controlled static-scene temporal/nvblox map is *not* a dynamic-intrusion
claim (see [navigation_temporal_mapping.md](navigation_temporal_mapping.md)).
The installed Isaac ROS 4.4 nvblox supports generic `mapping_type: dynamic` in
`nvblox_node` and people mapping through `nvblox_human_node` with a semantic
mask; both publish dynamic and combined map slices for `nvblox_nav2`. This
project's navigation mapper is intentionally configured as the static evidence
producer today, so it does **not** claim to have enabled either mode. Enabling
one requires a version-matched camera/mask or depth bridge and selecting the
combined/dynamic slice in the Nav2 layer; until then the supported live Nav2
input is its scan obstacle layer, and the supported sensor-route input is the
explicit detection contract below.

`navigation/dynamic.py` turns per-frame obstacle **detections** (map-frame centroid
+ physical radius + source — from the robot depth/nav cameras via an nvblox
dynamic/people layer, a detector node, or the scenario) into conservative
occupancy points unioned into `Observation.obstacles`.
`DynamicObstacleLayer.update(detections, stamp)` associates detections to tracks by
nearest predicted position, low-passes a velocity estimate, debounces
(`confirm_frames`), decays stale tracks (`drop_after`), and
`predicted_points`/`apply` block each confirmed track's **current extent** plus, for
a moving track, its **predicted swept path** over `prediction_horizon`. The layer is
**add-only**: it never clears a raw return, shortens a stop, or relabels unknown
free — removing its output can only turn a stop into motion. `DynamicObstacleConfig`
exposes association / moving-speed / horizon / step / gain / confirm / decay /
spacing / cap, all as configuration. `Detection(position, radius, source)` is the
contract shared with the detection topic (`sim-restock-humans-v1`) and the reactive
event tree (`sim-restock-bt-v1`).

**Default-off / opt-in.** Nothing consumes the layer unless
`navigation.dynamic_obstacles` is set (sensor nav_mode only; the Nav2 route takes
moving obstacles through its ROS costmap obstacle/nvblox layer instead). When
configured, `nav_factory` attaches the layer to the sensor route and the caller
passes `detections=` to `SensorNavigation.step`; absent, the sensed observation is
untouched and the mission run is byte-identical. The avoidance **patience**
`navigation.blocked_timeout` sets how long the base stops and waits on a blocked
aisle before giving up the goal (default unchanged).

**Mission detection feed (opt-in).** The layer is inert until something supplies
`detections` each navigate tick. In a live mission that feed is built by
`twin.navigation.mission_source` and wired at the `NavigationBridge` seam
(`twin.restock.runtime`): set `navigation.dynamic_obstacle_source` to a lane-owned
`module:function` and the bundle (`twin.restock.isaac_bundle`) builds a
`source(now) -> [Detection]` that `NavigationBridge.step` passes into
`SensorNavigation.step(detections=...)` on the sensor route only. Absent, the bridge
passes `detections=None` — byte-identical to a run with no layer; a source on a
non-sensor route is refused. The default builder
`twin.navigation.mission_source:build_dynamic_obstacle_source` **reuses the
`sim-restock-humans-v1` scripted person asset**: the `humans` config block authors one
or more deterministic scripted actors into the Isaac stage, and each actor's known
pose at `now` (pure `PeopleStage.world_positions`) becomes a map-frame `Detection`
(`source='scenario_scripted_actor'`). Because the detection position and the rendered
prim come from the same scripted path, pixels and the costmap agree by construction;
authoring/stepping the visible prim is best-effort (a missing stage degrades to
"detected but not rendered", never a broken tick). The perception path (the humans
lane's detector / ground-truth projection via `MissionSafety.detections`) plugs into
the **same** `source(now)` seam through `mission_source.humans_detection_source`
(`detection.to_nav_detections`) once that lane's detector GPU run lands. Example
runtime-config fragment: [`docs/examples/dynobs-mission-fragment.json`](examples/dynobs-mission-fragment.json).

This wiring is CPU-proven only (`test_navigation_mission_source.py`): byte-identical
off, deterministic scripted detections, bundle resolution. The in-sim GPU evidence run
rides the integration mission boot (it owns GPU priority, the station commissioning
baselines and the host planner stack); it is **pending**.

**Behaviours** (deterministic CPU evidence, `scripts/run_dynobs_evidence.py`,
RUNBOOK §12): a shopper crossing the aisle is detected, the towed base **stops and
waits**, then **resumes** and arrives once the path clears; a box dropped on a
narrow route and left is an honestly **unsupported** case — the base stops and
gives up after `blocked_timeout` rather than ever reporting arrival; and the coupled
planner **re-plans a route around** an off-centre cart (vs the straight shot). The
closed-loop follower reacts locally (its per-tick braking sweep, not a global
early re-plan), so a cart met head-on in a narrow aisle is a stop/wait, while the
planner-level detour evidence shows the re-plan-around capability. These are
quasi-static kinematic demonstrations, not PhysX, sensing or perception runs;
detections are scenario ground truth, not detector output.

## Evidence and limits

The navigation unit tests cover cart swing bounds, corridor feasibility,
obstacle detours, blocked timeout, dynamic replan, arrival orientation/stopping,
transport/coupling/sensing stops, braking travel, wheel odometry, map scan
matching, command curvature and ROS isolation. They use explicit test-fixture
provenance. Full-stack Nav2 launch, PhysX range acquisition, cart turn dynamics,
AMCL convergence and Isaac closed-loop movement still require integration
execution. No GPU job or physical robot command was issued by this lane.
