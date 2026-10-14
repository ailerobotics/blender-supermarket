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
The matching official 4.4.0 plugin is now delivered by the host lane with CPU
symbol/loader checks. This removes the missing-package handoff; native plugin
instantiation and costmap behavior remain unproven.

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

## Instantaneous self-occlusion-aware coverage (scalar-scan path)

When no temporal map is configured, braking is gated by the instantaneous range
scan instead. The physical restock rig's range sensor is mounted on the moving
base and **always self-occludes on its own base, mast, stowed arm and towed cart**,
so a fully unoccluded 360 degree scan is never produced and repositioning the mount
cannot remove every self-return. Self-returns are reported as UNKNOWN (NaN range,
`Scan.unknown[i]`) by the sensor's instance-ID / self-path mask (integration supplies
the rig self paths; identity is USD collider subtree membership, never inferred from
point location). The earlier contract zeroed coverage whenever **any** beam was
UNKNOWN, which left the rig permanently unable to drive. The coverage model is now
directional and keyed to the actual motion:

- **Lidar model.** A beam is either a NON-self observation (it saw free space out to
  the sensor range, or struck an external return that is reported in
  `Observation.obstacles` and gated separately) or a self-occluded UNKNOWN beam that
  certifies nothing beyond the rig — even when another height on the same bearing
  retains an external return, the near field there is unobserved and never certifies
  clearing.
- **Braking sweep.** `sweep_extent(beta, radius, stop_distance)` is the far reach, per
  relative bearing `beta` from the heading, of the forward braking capsule: the circular
  footprint `radius` (which circumscribes the **whole train, including the towed cart**)
  swept `stop_distance` forward. Rear and near-broadside bearings reach only the
  footprint edge — the capsule never extends behind the base because reverse is forbidden
  — so they are never a swept sector.
- **Occupied-envelope subtraction (`CoverageField.certifies_sweep`, both gates).** A single
  circular footprint radius is the wrong reference: the coupled train's circle is ≥ 1.1 m
  (it circumscribes the **towed cart behind the base**) and even the uncoupled base's circle
  reaches ≈ 0.75 m, so requiring observation out to that radius pulls self-occluded
  **near-side** bearings (≈ ±85–89°) into the required set even though the robot never enters
  that forward-side space — which kept the real, always-side-self-occluding rig from driving
  (collect stage) and forced refusals as the heading changed. Instead, each `gate_velocity`
  builds the **actual convex occupied envelope** (base-only uncoupled, base + cart at the
  measured articulation coupled) and the **braking sweep** of that envelope at the commanded
  speed/curvature, then **subtracts the occupied envelope**: a blind (self-occluded) beam
  refuses only when it enters a **newly swept crescent in the direction of travel** — its ray
  must reach beyond the occupied envelope by more than the geometry margin
  (`sweep_far - body_far`) **and** its bearing must lie within the forward cone (45° of
  travel). Bearings outside that cone are the lateral and rear extent of the capsule, swept
  parallel to the body or not swept at all (no reverse); they are covered by the footprint.
  The forward cone is evaluated **body-relative** (bearing minus heading), so a rig that
  self-occludes its own sides and rear is exempt there at every heading, not only the one
  sampled at spawn. The bearing is used rather than the occupied-envelope exit-face normal
  because the front-offset lidar and the faceted hull make near-side rays leave the envelope
  through forward-tilted facets. Rays originate at the calibrated sensor origin
  (`CoverageField.origin`, the chassis-front lidar mount). `CoverageField.certifies` keeps
  the older circular-capsule check as a scalar-radius probe.

Consequently the base/train drives when the forward cone is observed even though its own
base/mast/arm/cart self-occlude every near-side and rear bearing, at any heading; a genuinely
unobserved forward (or forward-lateral) zone still refuses; and a stationary robot
(`stop_distance ≈ 0`, no crescent) certifies so it can settle at the goal despite full
self-occlusion. The 0.05 m geometry margin and 0.10 m localization inflation are retained as
the occupied/swept envelope margins, the full (cart) sweep is retained via the measured
envelope, and the no-reverse rule is unchanged. A differential base turns in place (no
forward crescent while rotating) then drives straight, so the uncoupled sweep is a forward
capsule; the coupled swept hull is taken at the commanded curvature. `Observation.coverage_radius`
remains only as the legacy full-circle inscribed-reach diagnostic (and the fixture fallback
for observations that carry no field); the gates consume `Observation.coverage`. This is
instantaneous-scan safety for the controlled static scene, not person-safety readiness — a
blind obstacle lying outside the forward cone but inside a wide swept turn is covered only by
the obstacle gate on **sensed** returns — and it does not
substitute for the native departure and blocked-aisle evidence below. Counterfactuals:
`test_navigation_coverage.py`.

## Validation and remaining native work

`test_navigation_temporal.py` checks historical free-space persistence, 600-second
expiry across rebuilds, revision changes, held/stale data, independently moving
cart geometry, unknown/occupied boundaries, measured-latency stopping, initial
command acceptance and stop/replan after a changed map. `test_navigation_evidence.py`
exercises the typed `EvidenceSnapshot` consumer: full-ledger completeness, per-sensor
freshness, admission/empty state, producer-identity one-shot binding, monotonic
index/revision, optical-frame binding immutability, and the permanently closed
full-height gate. `test_nav2_handoff.py` uses actual Jazzy message types with fake
action clients, including cancellation ordering and generated nvblox plugin
configuration. These are deterministic CPU contract fixtures, not native mapping
or physical braking evidence.

The next native handoff must establish navigation-scoped mapper startup and
package loading, slice/incorporation pairing, sensor-time TF including cart
motion, full-height map behavior, empty-generation reset, timing/resource bounds
and actual departure followed by blocked-aisle stop/replan. Integration owns
those GPU checks. The parked `/restock_sim` mapper remains independent of
`/restock_sim/navigation`. No system install, service change or GPU execution is performed
by these modules.

## Executable commissioning handoff

`scripts/launch_navigation_mapper.py` prints a plan by default; `--execute`
starts only a separate navigation mapper. Commands are in RUNBOOK §14. It uses
the installed base nvblox YAML and stable-map/full-transport-height overrides,
without station bounds, manipulation target removal or map loading. It checks
the installed component and matching Nav2 plugin exports before constructing
the real ROS launch description. Construction is CPU-only; execution belongs
to integration under the existing GPU/resource guards.

The generation argument names a requested new process, **not an empty-map
acknowledgement**. Each generation has distinct depth/CameraInfo admission
topics printed in the plan. Integration keeps these publishers closed until
it has native empty-map evidence, then publishes only post-generation data.
Use sensor-time registered depth and matching CameraInfo, SENSOR_DATA QoS,
acquisition-time camera TF into `map`, `/clock`, and measured base/cart state.
Remove only actual same-acquisition self hits; invalid pixels and occluded
space remain unknown. No manipulation target-box mask feeds this mapper.

The required host-to-gate evidence interface is:

| Evidence | Required binding |
| --- | --- |
| Process generation | Exact task-owned mapper process, no map load, old process exit, native empty-state evidence before input admission; discard queued old output |
| Required acquisition | Native sensor identity/sequence, simulation stamp, acquisition steady stamp, optical frame, shared steady-clock domain and configured MapIdentity; register through `observe` (one per `restock_mapper_evidence/RegisterAcquisition`) |
| Snapshot incorporation | A typed `restock_mapper_evidence/EvidenceSnapshot` binding the native slice to the FULL per-sensor incorporation ledger, producer nonce + certificate, monotonic `snapshot_index`/`esdf_revision` and explicit admission/empty state; consume through `EvidenceEnvelope` + `TemporalMap.accept_envelope` |
| Clearance event | Hold motion/input, record event time, prove a new empty generation (fresh producer nonce/certificate), admit strictly later acquisitions; preserve the original commissioning deadline |

### Consumed typed evidence envelope

The host authored the typed interface package `restock_mapper_evidence` (messages
`MapIdentity`, `Acquisition`, `LayerCount`, `SensorBinding`, `EvidenceSnapshot`;
services `EmptyGeneration`, `OpenAdmission`, `RegisterAcquisition`). The navigation
consumer `TemporalMap.accept_envelope` validates each `EvidenceSnapshot` against
the whole ledger before adopting the embedded slice and refuses it unless **all**
of the following hold:

- `admission_state == OPEN` and `empty_state == POPULATED` (an unknown/verified-empty
  snapshot never certifies free space for motion);
- producer `producer_nonce`/`certificate_token`/steady-clock domain are bound once
  per empty-started generation and never change within it — a new producer identity
  cannot reopen a retained map;
- `snapshot_index` strictly increases and `esdf_revision` never decreases (an
  unchanged ESDF revision may repeat under a new index);
- every ledger `Acquisition` names a configured sensor, carries an immutable
  optical-frame binding, is already registered through `observe`, and is unique;
- the embedded slice binds the newest ledger acquisition in the current generation
  and stable `map` frame, with no out-of-order incorporation;
- `readiness` additionally requires the ledger to represent every configured sensor
  and each incorporated acquisition to be `<=300 ms` old in both the simulation and
  steady domains. `republished_slice` returns the slice to Nav2 only while all gates pass.

`full_height_observation_verified` is recorded verbatim but NEVER opens the
full-height observed-column gate: `full_height_certified()` stays `False` on this
lane. Provenance certifies only that a bound 2D slice came from an empty-started
mapper; it cannot establish vertical-column coverage.

The native slice output topic stays canonical across processes and has no
generation field. Input topic names alone cannot reject late output: integration
must bind producer lifetime and flush/subscriber state before accepting a new
generation. The supplied launcher does not manufacture that binding.

**Actual upstream API status:** the installed `DistanceMapSlice` definition has
no generation or acquisition-incorporation field, and the installed
`EsdfAndGradients` service is **3D-only and exits the node in 2D mode** — never
invoke it on this navigation mapper. The host has now authored a typed
`restock_mapper_evidence` interface package whose `EvidenceSnapshot` supplies the
missing atomic slice-to-ledger binding, empty/admission state and producer
identity; the navigation consumer above is written against that frozen interface.

What remains strictly native and **UNCOMMISSIONED**: no producer yet populates
`EvidenceSnapshot` from synchronized native work, and the `EmptyGeneration`/
`OpenAdmission`/`RegisterAcquisition` services are unimplemented native hooks. The
CPU consumer therefore proves only that a *well-formed* envelope is validated or
refused; it does not observe a real empty map, a real incorporation, or real
admission. Matching publication times, FIFO assumptions or a later successful ESDF
call still cannot substitute for a populated envelope. Full-height unknown-column
reduction must also be validated natively; merely increasing the configured slice
height — or the producer flag — is not that evidence. A native producer plus
departure and blocked-aisle stop/replan must pass before the motion gate is ready.

CPU checks cover the station-independent plan, isolated generation input topics,
argument rejection, real installed export lookup and construction of the ROS
launch description. They do not instantiate the mapper or plugin, observe an
empty map, prove incorporation, or demonstrate motion.

Primary references: [NVIDIA4.4 mapper inputs and outputs](https://nvidia-isaac-ros.github.io/v/release-4.4/repositories_and_packages/isaac_ros_nvblox/isaac_ros_nvblox/api/topics_and_services.html),
[NVIDIA4.4 Nav2 plugin](https://nvidia-isaac-ros.github.io/v/release-4.4/repositories_and_packages/isaac_ros_nvblox/nvblox_nav2/index.html),
and [version-matched plugin lookup/merge implementation](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_nvblox/blob/release-4.4/nvblox_nav2/src/nvblox_costmap_layer.cpp).
