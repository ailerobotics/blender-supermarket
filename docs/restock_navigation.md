# Mobile restock navigation

`scripts/twin/navigation/` adds two simulation navigation routes. Neither has
been run against a moving Isaac robot yet. CPU tests are deterministic fixtures,
not evidence of autonomous physics operation. Commands live in RUNBOOK §9.

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

## Evidence and limits

The navigation unit tests cover cart swing bounds, corridor feasibility,
obstacle detours, blocked timeout, dynamic replan, arrival orientation/stopping,
transport/coupling/sensing stops, braking travel, wheel odometry, map scan
matching, command curvature and ROS isolation. They use explicit test-fixture
provenance. Full-stack Nav2 launch, PhysX range acquisition, cart turn dynamics,
AMCL convergence and Isaac closed-loop movement still require integration
execution. No GPU job or physical robot command was issued by this lane.
