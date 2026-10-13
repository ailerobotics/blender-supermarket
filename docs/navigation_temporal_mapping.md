# Controlled static-scene temporal navigation

This path is for the **first controlled static-scene simulation only**. It is
not deployment, real-robot or person-safety readiness. Historically observed
static free space may persist through self-occlusion in one mission map epoch,
for at most 600 seconds from commissioning. It does not guarantee detection of
arbitrary new obstacles entering a hidden region. Native departure and
blocked-aisle stop/replan have not yet been demonstrated.

The mapper is NVIDIA Isaac ROS 4.4 nvblox, separate from the parked manipulation
mapper. Navigation consumes its temporal distance map; there is no new Python
depth-fusion loop. Checkpoint `7739eae` preserves the failed instantaneous experiment.
Passing that experiment as `Observation.observed_space` now stops with
`instantaneous_coverage_retired`.

## Integration contract

Construct `ArticulatedController(grid, require_temporal_map=True)` for this path.
That flag prevents missing map data from falling back to scalar scan coverage.
The current temporal route supports the coupled transport train; uncoupled travel
is refused in this mode. Existing legacy callers retain their previous mode.

`twin.navigation.temporal` exposes:

| Type / operation | Meaning |
| --- | --- |
| `MapIdentity(session, scene, cameras, localization)` | Explicit configuration/session revisions |
| `TemporalMap(identity, commissioned=..., required_sensors=..., generation=..., steady_commissioned=..., measured_latency=...)` | One mission epoch with acquisition and monotonic-clock watchdogs |
| `Acquisition(sensor, identity, sequence, stamp, steady_stamp)` | Actual registered sensor acquisition, not receipt time |
| `observe(acquisition)` | Records required-source liveness; rejects held/out-of-order frames |
| `DistanceSlice.from_message(msg, generation=..., incorporated=acquisition)` | Decodes native `nvblox_msgs/DistanceMapSlice` plus incorporation provenance |
| `accept(snapshot)` | Requires the correct stable frame, empty-map generation and known acquisition |
| `check_identity(identity)` | Latches a stop on scene, camera, localization or session change |
| `invalidate_clearance(now=...)` | Stock/manipulation/payload event; stops travel pending reconstruction |
| `begin_empty_reconstruction(new_generation)` | Accepts a newly EMPTY navigation mapper, without extending the mission deadline |

Pass `temporal_map=gate` into `SensorNavigation.step` or
`ArticulatedNav2Adapter.step`. Direct controller callers use
`Observation.temporal_map`. Host and simulator must share the steady-clock domain
used by acquisition metadata and the gate. The 600-second bound is checked in
both simulation and steady time, so slower-than-real-time execution cannot
silently extend an epoch.
Any decrease in either checked clock latches epoch invalidation, even inside
the freshness window. Reconstruction cannot clear this invalidation.

Required acquisitions and mapper incorporation must each be no more than 300 ms
old in both clock domains. The mapper may legitimately lag a newer camera frame;
the gate retains bounded acquisition metadata to match its acknowledgement.
Neither a slice publication time nor a newer unrelated ESDF response proves
which acquisition produced a particular snapshot. The host must provide that
binding. The native ESDF service header describes the latest incorporated depth,
not the age of every stored cell. No stored cell is restamped as freshly seen.

## Mapping and command behavior

`navigation_mapper_parameters(camera_count=..., transport_height=...)` in
`twin.navigation.mapper_config` returns stable-`map` configuration for 1–4 cameras,
5 cm voxels, and a height range covering the declared complete transport height.
Its 3 cm floor exclusion assumes the documented finite-resolution obstacle class;
it is not a claim about arbitrary floor lips or thin objects. Supply a measured
height that includes arm, loaded cart and mounts. The function starts no node.
Configured 20 Hz depth integration and 10 Hz ESDF updates are targets, not measured
throughput. Depth/TF registration, same-acquisition self filtering and invalid
pixel handling remain mapper input requirements. Self shadows are not cleared.

`nav2.write_assets(..., nvblox_topic='/restock_sim/navigation/nvblox_node/static_map_slice')`
adds NVIDIA's `nvblox::nav2::NvbloxCostmapLayer` while retaining unknown planning
disabled and no reverse. A raw-slice actuator check remains necessary: maximum
merging of Nav2 layers must not let static free cells erase unknown evidence.
The selected host extraction did not contain the plugin package/library when
inspected; matching package availability is a host handoff item, not proof that
the configured plugin can load.

The gate treats sentinel/nonfinite, out-of-map and occupied cells as STOP. For
the 4.4 plugin's rounded lookup versus the message's documented corner origin,
cell support conservatively covers both conventions. Planner row runs preserve
unknown and occupied regions; they do not reconstruct another sensor map.
The existing independently articulated model checks the entire base/arm/hitch/
cart sweep, including interpolation padding, at 5 cm geometry plus at least 10 cm
measured localization allowance. Actual mounts, casters and payload must fit the
declared transport envelope; otherwise update and validate that envelope first.

Command and measured-curvature braking paths both remain checked. Measured
pipeline latency/current incorporation age adds to the existing 300 ms watchdog
stopping allowance. Excess latency, missing map, invalid transport/coupling,
reverse, excessive articulation and insufficient turning radius all stop.
Nav2 commands remain proposals. An occupied/unknown stop requests cancellation
of the old FollowPath action; the adapter waits for its result before submitting
a newly checked route. CPU fixtures exercise this handoff without ROS nodes.

For the first implementation, a stock/manipulation/payload event conservatively
invalidates the whole navigation map, rather than pretending a global timestamp
proves regional re-observation. Host restarts only its own navigation mapper with
an empty map and a new generation, then supplies fresh observations. Reusing a
frame stamped exactly at the clearance-change time is refused; post-change
observations must have strictly later acquisition timestamps. Reusing a
retained map under a renamed generation or clearing changed regions to free is
not allowed. Scene/camera/localization/session discontinuities invalidate the
epoch itself. Expiry remains absolute from original commissioning.

## Validation and remaining native work

`test_navigation_temporal.py` checks historical free-space persistence, 600-second
expiry across rebuilds, revision changes, held/stale data, independently moving
cart geometry, unknown/occupied boundaries, measured-latency stopping, initial
command acceptance and stop/replan after a changed map. `test_nav2_handoff.py`
uses actual Jazzy message types with fake action clients, including cancellation
ordering and generated nvblox plugin configuration. These are deterministic CPU
contract fixtures, not native mapping or physical braking evidence.

The next native handoff must establish navigation-scoped mapper startup and
package loading, slice/incorporation pairing, sensor-time TF including cart
motion, full-height map behavior, empty-generation reset, timing/resource bounds
and actual departure followed by blocked-aisle stop/replan. Integration owns
those GPU checks. The parked `/restock_sim` mapper remains independent of
`/restock_sim/navigation`. No system install, service change or GPU execution is performed
by these modules.

Primary references: [NVIDIA4.4 mapper inputs and outputs](https://nvidia-isaac-ros.github.io/v/release-4.4/repositories_and_packages/isaac_ros_nvblox/isaac_ros_nvblox/api/topics_and_services.html),
[NVIDIA4.4 Nav2 plugin](https://nvidia-isaac-ros.github.io/v/release-4.4/repositories_and_packages/isaac_ros_nvblox/nvblox_nav2/index.html),
and [version-matched plugin lookup/merge implementation](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_nvblox/blob/release-4.4/nvblox_nav2/src/nvblox_costmap_layer.cpp).
