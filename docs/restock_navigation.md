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

### Direction-aware braking zone (config-driven, default-on)

The `obstacle_stop` gate always checks the **circular footprint all around the base** —
`footprint.radius` (which circumscribes the whole train) plus the localization uncertainty —
and that disk is **never shrunk**: any obstacle inside the body stops the robot regardless of
heading. It then adds the braking-distance extension (`stop_distance`, which grows with the
commanded/measured forward speed). The **shape of that extension** is the direction-aware zone:

- **Direction-aware** (`navigation.directional_zone: true`, the default): the extension is applied
  **only along the forward motion** — the footprint disk swept `stop_distance` metres down the
  heading, i.e. a forward **capsule** (`Controller._braking_zone_free` →
  `GridMap.segment_free(pose, pose + stop_distance·heading, radius)`). `stop_distance` is derived
  from the forward speed `max(commanded, measured)`, so a stop or a rotate-in-place command with no
  residual forward motion has no extension and only the all-around footprint is checked (a circular
  footprint sweeps onto itself under rotation). A base still physically coasting forward while
  commanded to stop keeps the capsule, because that travel is real. A no-reverse base can never drive
  into geometry in its **rear hemisphere**, so a rear obstacle outside the footprint does NOT trip;
  an obstacle the same distance AHEAD (inside the capsule) still does.
- **Legacy omnidirectional** (`navigation.directional_zone: false`): the extension is added in every
  direction (`radius + stop_distance` disk), so the rearward reach a forward-only base can never
  collide with still trips on static geometry behind the parked start pose.

This directly fixes the ds-007 / run-319b couple-dock false stop: the backstock staging support,
sensed ~0.81 m in the robot's **rear** hemisphere while it drives forward and away, no longer trips
`obstacle_stop` under the direction-aware zone (the full-speed omnidirectional disk ≈0.90 m used to
engulf it). The footprint radius and every margin are identical in both modes — this is **not** a
mask or a widened tolerance; it only governs whether the braking inflation is omnidirectional or
forward-only. The coupled gate (`ArticulatedController.gate_velocity`) is already direction-aware by
construction: it rolls the braking **sweep** forward along the commanded curvature
(`world.rollout`) rather than bounding the arc with an omnidirectional disk, so a rear obstacle
beyond the train never trips there either. Parsed by `navigation/factory.py`; CPU tests in
`tests/test_navigation.py` (forward / rear-clears / rotate / legacy) and
`tests/test_articulated_navigation.py` (coupled).

> Note — the trusted-static couple/reposition legs (`couple_static_drive`, `static_drive`) drop the
> braking lookahead entirely (footprint RADIUS only) and remain unchanged; they are orthogonal to the
> directional zone. Whether the merged `couple_static_drive` / dock speed-cap workarounds can be
> retired is discussed in `data/sim-restock-nav-lanes-v1/report.md` (the direction-aware zone retires
> the GATE false-stop, but the `blocked_no_route` PLANNER margin that `couple_static_drive` actually
> addresses is separate and is NOT retired here).

### Dock-approach speed cap (opt-in, default-off)

With the direction-aware zone the dock-approach speed cap is no longer needed to clear a **rear**
obstacle at couple; it remains available for shrinking the **forward** braking reach near a dock
(e.g. a genuine forward blocker close to a dock goal). A no-reverse base braking forward can never
drive into geometry in its *rear* hemisphere.

`navigation.docking` caps the commanded linear speed to `speed` within
`approach_distance` of a named dock `goal` (bound to the active goal within
`goal_tolerance`, default 0.25 m). A slower command genuinely shortens
`stop_distance`, so the braking disk shrinks **honestly**:

```json
"docking": {"goal": [-4.46, 15.25, 0.0], "speed": 0.08,
            "approach_distance": 1.5, "goal_tolerance": 0.25}
```

It is **not** a mask and **not** a widened tolerance: the footprint radius and every
margin are unchanged (an obstacle inside the bare footprint still stops), and no
obstacle is hidden — a real forward blocker still refuses. The `goal` must equal the
couple-phase `mission_poses.cart_dock`. Absent from config ⇒ disabled, run
byte-identical, real-rig defaults untouched. Parsed by `navigation/factory.py`
(`DockingApproach` in `navigation/core.py`); CPU tests in `tests/test_navigation.py`
and `tests/test_navigation_factory.py`.

### Couple cart-hold + final fine-docking servo (opt-in, default-off)

Navigation parks at `cart_dock` within its arrival tolerance (0.08 m / 0.08 rad), but the
physical coupling `attach` (`twin.mobile.Coupler.attach`) is far tighter — hitch-point distance
≤ 25 mm, hitch-z ≤ 15 mm, yaw ≤ 5°, fully stationary — and those tolerances model a real hitch
and are **not** loosened.

**Intermediate couple approach waypoint** (`mission_poses['couple_approach']`, optional). The robot
ends collect directly NORTH of the cart, while the drawbar/dock is EAST of it. With the cart pinned
(below) the robot cannot shove it aside, so driving straight at `cart_dock` wedges against the cart.
When `couple_approach` is set, the couple stage navigates to it FIRST (east of / clear of the pinned
cart) and then to `cart_dock`, so the robot rounds the cart. Phase-local to couple; absent ⇒ couple
goes straight to `cart_dock` (unchanged). Like `cart_dock`, the integration mission config must carry
the same waypoint value.

**Cart-hold through the approach** (`config['hold_cart_through_couple']`, SIM-only). The cart is a
free body on bearing-damped casters. `mobile_runtime.command()` used to release the cart's
fixed-to-world sim brake on the first base motion, so the cart was free for the whole couple
approach; an angled nav approach then swept the rear hitch (0.62 m behind the base) into the cart
and shoved it hundreds of mm before docking began. With the flag set, `mobile_runtime.hold_cart_pin()`
pins the cart at couple start and `release_cart_brake()` **no-ops while the hold is latched** (so
neither `command()` nor the `isaac_port` per-tick action logic can drop it); `attach()` authors the
robot↔cart hitch THEN force-releases the world pin. The cart therefore stays at its known pose
through the approach, so the hitch seats with a small residual. The real cart brakes physically, so
real-rig semantics are untouched; absent flag ⇒ unchanged run.

**Fine-docking servo** `twin.restock.fine_dock.FineDock` spans the final residual: once the couple
nav goal is reached it runs a slow closed-loop creep that seats the rear hitch on the **measured**
cart drawbar (per-tick), then hands to `attach`. The rear hitch's planar velocity is a non-singular
linear map of (v, ω) (the inverse-hitch Jacobian, determinant `hitch_x` ≠ 0), so a proportional law
on the measured hitch error gives a **translation-dominant** command: `v` along heading
(REVERSE-capable) + a small `ω` for lateral+yaw. It **never rotates in place** — a real differential
base does not pivot about its measured centre, so an in-place spin drifts the base and never
converges (observed on cn-002). Motion only: it drives the base through the same
`mobile.command(v, omega, dock, ...)` seam navigation uses; the `mobile.py` latch stays the
coupling-joint side's. Bounded by `time_cap_s`; on expiry it fails HONESTLY with the residuals.

Config `config['fine_dock']` (top-level, NOT under `navigation`; consumed by `isaac_bundle` →
`RuntimeAdapter`): `creep_speed`, `creep_yaw_rate`, `gain_lin`, `gain_yaw`, `hitch_tol_m` (< 25 mm),
`yaw_tol_rad` (< 5°), `settle_speed`, `time_cap_s`. Absent ⇒ no servo ⇒ couple attaches directly off
the nav arrival, byte-identical. **Sim and real-rig values live in SEPARATE config blocks** (the real
rig carries its own, more conservative tuning). CPU tests: `tests/test_restock_fine_dock.py`
(servo + drift-augmented closed-loop convergence + reverse + honest time-cap) and
`tests/test_restock_runtime.py` (couple nav→fine_dock→attach handover + cart pin at start /
release at attach).

**Cross-track terminal mode** (`approach_mode: 'crosstrack'`, opt-in; default `'hitch_jacobian'` ⇒
byte-identical). The legacy translation-dominant law has a degenerate zero-command equilibrium **line**
— wherever `lat = (gain_yaw·|hitch_x|/gain_lin)·yaw_err`, the single `ω` cancels (lateral vs yaw) with
NEITHER hitch nor yaw seated. The couple approach lands on it at whatever yaw the standoff cross-track
implies: `coldlivefinal` (real live_pose collect) landed at ~4.8° yaw / ~26 mm — just outside the 4° /
20 mm latch box — and limit-cycled the full 30 s cap (`FINE_DOCK_DIAG`, run dir
`twin-robot-in-store-boot-v1/mission/coldlivefinal`), while the still-arm `coldhonest` landed at ~2.7°
(inside) and latched. `'crosstrack'` (`fine_dock._crosstrack`) instead servos the **rear hitch** — the
LEADING point of the reverse approach, so steering it onto the cart axis is a stable minimum-phase
cross-track problem whose only equilibrium is the seated pose. It drives the hitch along the cart axis
to the drawbar (`ct_gain_approach`), holding back a distance `ct_align_hold_m·(|cross|+0.3|yaw|)` until
cross-track + heading null (`ct_gain_cross`, `ct_gain_yaw`), with a `ct_min_creep` floor so the base
keeps moving (steering stays live). Still motion-only; latch tolerances + the physical `attach` are
unchanged. Example: `docs/examples/couple-fine-dock-crosstrack-fragment.json`; design + the recorded
limit-cycle evidence: `data/sim-robust-couple-fine-dock-v1/report.md`. CPU tests cover the recorded
`coldlivefinal`/`coldhonest` standoffs (converge+latch where the legacy law stalls under a drive
dead-band), the degenerate-equilibrium property, and the byte-identical default.

### Pre-planned tow lanes for coupled drives (opt-in, default-off)

Free hybrid-A* planning of the long, wide coupled robot+cart train is where the tow stalls: the
outbound tow blocked earlier, and the coupled RETURN to the east-facing `home` **times out** — the
planner finds a valid route but it is a 23.9 m loop for a 10.4 m direct, because the goal yaw forces
a reorientation from the southbound portal arrival (a loaded host then starves the search inside its
wall budget). `navigation.tow_lanes` replaces free planning for coupled drives with **authored lanes**.

A tow lane (`navigation/lanes.py` `TowLane`) is an authored centreline — a `start` pose, an ordered
list of `waypoints` (x, y, yaw through the backroom/aisles) and a `goal` pose — plus `start_cart_yaw`
and `goal_articulation`. The **lane graph is expanded once, at build time**, against the static map
by `navigation/factory.py` (`ArticulatedPlanner`, run at mission boot, never under the per-tick
budget): each lane becomes a kinematically valid, swept-envelope-clear train trajectory, and the
authored waypoints are asserted to lie on the realised centreline (a broken lane fails the build).

At runtime, when a COUPLED drive's goal matches a lane (and the measured base is close enough to the
lane start to acquire it), `ArticulatedController` follows that lane's trajectory
(`_plan_lane`): it re-validates the stored trajectory's swept robot+cart envelope against the **live**
obstacle observation (cheap `validate_seed` replay, ~0.1 s) and tracks it closed-loop, and it
**never falls through to the expensive hybrid-A\* search** — so a loaded host cannot starve a coupled
drive into `planning_timeout`, and the authored lane (not a free detour) is the sanctioned path. A
live obstacle on the lane **blocks** the tow (`lane_blocked`, bounded) rather than silently
re-routing; a base too far from the lane start waits (`lane_not_acquired`). When a lane is active, the
**arrival articulation follows the lane's `goal_articulation`** — the lane is the authority on its own
kinematically-achievable terminal cart pose. Uncoupled drives carry no lane and keep the full planner
unchanged; absent `navigation.tow_lanes` ⇒ every coupled drive free-plans exactly as before
(byte-identical).

The authored mission lanes (example fragment `docs/examples/tow-lanes-mission-fragment.json`):

| lane | start → goal | authored arrival | route | notes |
|------|--------------|------------------|-------|-------|
| `outbound_dock_to_shelf` | couple dock `[-4.46, 15.25, 0]` → shelf park `[-3.05, 6.0, π]` | `goal_articulation 0.9` | 22.2 m via the `sales_backroom_portal` corridor | the coupled train can only curve into this pull-through shelf, arriving at ~0.8 rad articulation — aligning to a tighter arrival is not kinematically feasible here (see the report; a transfer/station concern) |
| `return_shelf_to_cart_parking` | shelf park `[-3.05, 6.0, π]` → **cart parking** `[-9.0, 16.0, π]` | `goal_articulation 0.2` | 15.1 m, arrival articulation 0.18 | west-facing park: no reorientation loop (ratio 1.3 vs the 2.3 of the east-facing `home` that timed out) |

**Cart parking spot** (`mission_poses.cart_parking = [-9.0, 16.0, π]`, robot base pose): the coupled
return ends here (in the backroom `rollwagen_parking` zone, near the dock); the cart trails to
~`[-7.73, 16.0]` and the robot, facing west, has a clear forward escape to drive away empty. The
behaviour-tree lane (`sim-restock-bt-driver-v1`) owns the uncouple step on this pose and the empty
drive home. The integration/orchestrator mission config wires `mission_poses.cart_parking` and the
return nav goal (like `cart_dock`); absent ⇒ the coupled return uses today's goal.

**Lidar-proud tolerance for lane re-validation** (`navigation.tow_lane_static_tol_m`, opt-in,
default `0.0` = off ⇒ full live cloud, byte-identical): a lane's swept envelope is re-validated each
tick against the LIVE obstacle cloud, but live lidar/ESDF returns sit a few cm **proud** of the true
static AABB (toward free space). Where the authored centreline hugs a shelf/support (the outbound
lane's tightest clearance is ~0.089 m, at a shelf near the pull-through goal), those proud returns
intrude the swept envelope and the lane re-validation returns `lane_blocked` even though the lane
clears the static map at build — the run-321b outbound stall (`lane_blocked` for ~56 min, never
advancing). When set, a live return within this distance of KNOWN static geometry is discounted as a
proud return of that geometry during lane re-validation **only** (`ArticulatedMap.near_static`); a
genuinely NEW return — not within the tolerance of any static geometry — is kept, so it still blocks
the lane at plan time AND the live directional braking gate (unchanged) still stops the tow for it.
This touches the lane plan-validation alone, never the braking gate, mirroring `couple_static_drive`'s
config-driven trusted-static style but narrower. Staged value `0.12 m` (covers the measured ~0.089 m
clearance + the ~0.10 m proud offset that reproduces the live block). The opaque bare `lane_blocked`
now also logs `LANE_BLOCKED_DIAG` with the specific `validate_seed` rejection + discounted/kept cloud
sizes AND a capped sample of the surviving (kept) returns with each one's distance to the nearest
mapped static geometry (`ArticulatedMap.nearest_static`), so a future stall is classifiable from the
console: a small distance is a proud-of-mapped return, a distance at/over the search cap is a return
off finer sim geometry than the AABB static map carries (unmapped static) or open space.

**Trusted-static lane re-validation** (`navigation.tow_lane_trust_static`, opt-in, default `false` =
off ⇒ byte-identical): the superset of the proud tolerance, for when the tol cannot reach the kept
returns. run-322b showed the outbound tow STILL `lane_blocked` with `tow_lane_static_tol_m=0.12`
active — the `LANE_BLOCKED_DIAG` named 128 kept returns sitting **> 0.12 m** from any mapped AABB. With
no dynamic-obstacle source every live return is static, so those are proud-beyond-tol OR off finer sim
geometry than the ~800-AABB static map carries; `near_static` can never discount an unmapped-static
return, and the dense map makes raising the tol unsafe (it would swallow a genuinely new obstacle near
any of hundreds of shelves). When set, the authored lane is re-validated against the **static map
alone** (`dyn=()`) — the geometry it was expanded and swept-validated clear of at build — so proud or
unmapped static cannot false-block it. It is scoped to lane plan-validation ONLY and NEVER sets the
base `static_drive`, so the live direction-aware braking gate (`gate_velocity`) keeps the full live
obstacle cloud and **still stops the tow for a genuinely new obstacle ahead** — the new-obstacle safety
net moves to the live gate (direction/time-aware), exactly as `couple_static_drive` does for the couple
legs. Staged `true` for the mission tow lanes.

**Re-acquire along the route** (`navigation.tow_lane_reacquire_along_route`, opt-in, default `false` =
off ⇒ byte-identical, start-only acquire): run-323b exposed a follower trap once `tow_lane_trust_static`
let the lane validate — the follower drove ~0.36 m along the lane, dropped the route on a tracking
transient, then could **not** re-acquire because `TowLane.acquirable` checked proximity to the fixed lane
**start** only (0.36 m away, > `start_tol_xy` 0.25) even though the train was **0.06 m from the route
centreline** → `lane_not_acquired` forever, train frozen. When set, `_plan_lane` checks acquirability
against the **nearest authored route point** (the SAME tolerances — only the reference generalises from
the fixed start to the whole centreline), so a closed-loop follower that has drifted a little along its
own lane re-acquires at its progress instead of being trapped. An off-route pose (far from every route
point) is still rejected. Staged `true`. (The measured run-323b post-couple base delta to the start was
0.023 m / 0.007 rad with articulation ≈0 — i.e. the pose was ON the lane start; the trap was the
start-only acquire check, not a bad start pose, so no pre-lane alignment leg or tolerance widening was
needed.) CPU tests: `tests/test_navigation_lanes.py`; offline proof `check_tow_lanes.check_reacquire_along_route`.

**Coupled-tow tracking knobs** (fm429, all default to the prior hardcoded values ⇒ byte-identical):
`navigation.tow_lane_cart_yaw_tol` (default 0.20 rad) — the `stop_replan_tracking_error` cart-yaw
replan tolerance; `navigation.tow_lane_curve_speed_floor` (default 0.25) — the curve-slowdown speed
floor; `navigation.tow_lane_max_curvature` (default = the physical 1/1.5 cap) — caps the follower's
COMMANDED curvature below the physical min-turn-radius for gentler corrections. run-324b/c measured: the
cart yaw-**rate** tracks the base well (|rw−cw| max 0.097, mean 0.015 rad/s) — the stall is a sim
base/cart-dynamics transient (the under-powered base over-rotating at the curve-slowed crawl, measured
arc κ up to 7.2 vs the 0.667 cap), **not** cart articulation lag and **not** reproducible in the
kinematic model (offline the follower tracks the lane with zero cart lag). The one lever the evidence +
an offline swept-envelope proof support is relaxing the cart-yaw tolerance: `check_tow_lanes.check_cart_yaw_tol_envelope`
sweeps a ± cart-yaw offset on every authored route state and finds the largest offset whose
occupied-envelope stays clear of the static map — **0.33 rad (outbound) / 0.338 rad (return)**; the
staged `tow_lane_cart_yaw_tol=0.30` sits under that bound (absorbs the measured transient, cuts the
route-drop churn) and an offset past the bound provably breaches the envelope (so the tolerance can never
be widened to mask a cart that has actually left the swept envelope). The speed-floor and curvature-cap
knobs ship tunable but are NOT staged lower (slowing further raises the measured κ at crawl; the
commanded curvature is already clamped). CPU tests: `tests/test_navigation_lanes.py`,
`tests/test_navigation_factory.py`.

Offline proof (CPU, the mission's real inputs): `scripts/check_tow_lanes.py`
(`--environment … --scenario … --camera-config … --lanes docs/examples/tow-lanes-mission-fragment.json`)
drives the real controller/planner and asserts the direction-aware zone + both lanes end to end, plus
(fm423) reproduces the proud-cloud `lane_blocked` and proves the tolerance reopens the lane while a new
on-lane obstacle still blocks the tow; and (fm425) reproduces the run-322b block with an unmapped-static
return the tol cannot reach, proves `tow_lane_trust_static` reopens the lane, and proves the live braking
gate still stops the tow for a new obstacle with trust-static ON (`check_trust_static`). Design and
numbers in `data/sim-restock-nav-lanes-v1/report.md`. CPU unit tests: `tests/test_navigation_lanes.py`,
`tests/test_navigation_factory.py`.

**Arrival XY hysteresis** (`navigation.arrival_xy_hysteresis_m`, opt-in, default `0` ⇒ the legacy
per-tick arrival test, byte-identical; fm482): the reference arrival test was a bare `xy <= tol` every
tick, so a base captured just inside the circle that then turns in place to the goal yaw is pushed back
out by the turn itself (the measured point sits ~8 mm off the turning pivot), and the drive branch turns
it back toward the goal point — a limit cycle. At the couple standoff (tol 0.05 m) warm runs 347–356
swept ~41 rad of yaw for a net 1.0 rad, 162–188 s sim / 680–790 s wall per run. With hysteresis the XY
capture latches and releases only past `tol + hysteresis`; arrival still needs the yaw tolerance, a
stopped base and a clear gate. Staged `0.03` (inside the fine-dock capture bound). The standoff leg's own
tolerance can be overridden with the top-level `couple_standoff_tolerance` `{xy_m, yaw_rad}`.

**Coupled creep escape** (`navigation.coupled_creep_escape`, opt-in, default `false` ⇒ byte-identical;
fm481/fm482; floor `navigation.coupled_creep_floor_mps`, default 0.03): the coupled measured-curvature
brake divides `odom.angular` by `odom.linear`; at a crawl that ratio is noise. runs 352/355 stopped
abruptly ~0.4–0.5 m into the outbound, then tripped `measured_turn_radius` on ~700 noise ticks each (every
refusal commands a stop, so the base never left the band) → 121 s livelock. With the escape, inside
`|odom.linear| < max(0.06, 2·floor)` a forward command keeps a creep (≥ floor) with the arc clamped to
the min turn radius, and the coverage/footprint/braking rollouts check that command; above the band the
brake is unchanged. Both switches are named in the stage evidence (`couple_arrival_mode`,
`outbound_brake_mode`). Example: `docs/examples/couple-speed-mission-fragment.json`; offline proof
`scripts/check_couple_speed.py` (RUNBOOK §12); CPU tests `tests/test_couple_speed.py`.

**Coupled net-progress escape + fault** (`navigation.coupled_progress_escape`, opt-in, default `false` ⇒
byte-identical; sim-robust-outbound-drive-v1; `coupled_progress_min_m` 0.05, `coupled_progress_escape_s`
3.0, `coupled_progress_fault_s` 8.0). A **SIM-clock** commanded-vs-measured NET-progress monitor for the
coupled follower, keyed on net base displacement rather than instantaneous velocity. warm9/warm10 outbound
pinned at x≈−3.9 while the follower kept **commanding forward** (`reason='tracking'` on most ticks) and the
base jittered in place (measured `|rv|` noise to ~0.23 m/s, ~0 net displacement); under ~20× render pacing
the frozen window was ~14 s SIM but ~285 s WALL, so the WALL budget below failed the stage on the wrong
clock. `coupled_creep_escape` misses it (its trigger needs `|odom.linear|` inside the noise band the jitter
spikes leave) and `measured_turn_fault_s` misses it (it accrues only on `measured_turn_radius` coasts, not
the dominant `tracking` ticks). With this on: while the base has not moved ≥ `coupled_progress_min_m` from
its anchor for `coupled_progress_escape_s` SIM seconds it **forces the committed creep** (the same
arc-clamped forward command, so it never thrashes to a measured-curvature stop); after
`coupled_progress_fault_s` SIM seconds of no net progress it **faults `coupled_no_progress`** so the stage
StageRetry-replans — in a few SIM seconds, before the WALL budget. A healthy (even slow curve-floor) tow
re-anchors well within the window, so no false trip. Example:
`docs/examples/outbound-progress-mission-fragment.json`; offline proof on the recorded runs
`scripts/check_outbound_progress.py`; CPU tests (REAL controller: reproduce the stall + prove the
escalate/fault recovery) `tests/test_articulated_navigation.py`. The UNCOUPLED drive-fidelity lever behind
the crawl is `mobile_spec.caster_lift_m` (fm434 bench `base-drive-bench-*`: 0.097→0.988 at 0.005 m); the
warm runs already set it, so this monitor covers the residual **coupled** limit cycle.

**Coupled break-away stall recovery** (`navigation.coupled_progress_recovery`, opt-in, default `false` ⇒
byte-identical to the escalate→fault ladder above; sim-robust-outbound-stall-recovery-v1;
`coupled_breakaway_speed_mps` 0.18, `coupled_breakaway_window_s` 2.0, `coupled_breakaway_pulse_s` 0.5,
`coupled_breakaway_attempts` 3). The coldlivefull outbound showed the coupled tow **fail to break away from
rest** at the portal throat (NAV_STEP cycled `measured_turn_radius`↔`tracking` with `parked=True`, the base
jittering in place with ~0 net displacement) — a non-deterministic re-break-away deficit. Two own-code
contributors: a `measured_turn_radius` **coast** tick returns a `(0,0)` command (`_wait`), so a stuck tow
**thrashes stop↔forward** every few ticks and never sustains drive effort, and the forced creep only engages
after `escape_s` and is a weak 0.03 m/s static crawl. With recovery on, once the base has been pinned for
`coupled_progress_escape_s` SIM seconds the controller runs a **bounded break-away maneuver** instead of the
single creep-then-fail: up to `coupled_breakaway_attempts` windows of `coupled_breakaway_window_s` SIM
seconds, each commanding a **continuous pulsed committed-forward surge** (`coupled_breakaway_speed_mps`,
square-waved every `coupled_breakaway_pulse_s` between the high level and a relaxed 0.4× to **rock** the
contact). The surge **bypasses** the `measured_turn_radius` coast (so the command is never a v=0 stop) while
the obstacle/coverage/map safety gates stay **unchanged** — a real obstacle still stops the surge. The
instant net displacement reaches `coupled_progress_min_m` the break-away **succeeded** and normal tracking
resumes; exhausting every attempt **faults `coupled_no_progress`** (same name + StageRetry contract), fast
on the SIM clock. **Reverse** towing is clamped off for a coupled train (`mobile.wheel_targets`: coupled
`v≥0`), so the maneuver is a forward surge/relax pulse, **not** a reverse back-off; and all brakes are
already force-released every drive tick in `MobileController.command`, so no stale park/cart brake can hold
the base. Example `docs/examples/outbound-stall-recovery-mission-fragment.json`; CPU tests (REAL controller:
stall → break-away success, attempt exhaustion → named fault, continuous-forward, off-is-byte-identical)
`tests/test_articulated_navigation.py` (`test_coupled_breakaway_*`). It makes the stall **rarer** and
**fail-fast-by-name** when it cannot recover; the underlying coupled traction asset levers (caster lift,
drive force) remain owned by sim-robust-coupled-traction-v1 (PR 207).

**Wall-clock stuck-stage fail-fast** (`RESTOCK_NAV_STUCK_WALL_S`, opt-in, default `0` = off): the nav
`blocked_timeout`/`goal_timeout` are **sim-clock** caps, which under render pacing (sim advances far
slower than wall) cannot fire — run-322b's outbound sat on `lane_blocked` for ~50 min WALL and only the
3600 s orchestrator cap ended it, burning the whole GPU budget with no named stage failure. When set,
the runtime fails a navigate stage fast with its named reason once the SAME running reason persists with
NO measured base-pose progress for longer than this many WALL seconds (reset on a reason change or base
progress). Staged `120`. CPU tests: `tests/test_restock_runtime.py`.

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
