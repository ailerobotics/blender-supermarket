# Mobile UR10 and rollwagen

The mobile asset replaces the existing UR10 pedestal with a simulated differential
platform. It preserves the source arm/gripper links, mesh references, inertias,
joint limits, and transport targets. No hardware purchase or physical robot driver
is involved. The rollwagen is a three-sided metal cage with an open loading face,
a blue stock deck, an intermediate shelf, four passive swivel/rolling casters and
a front drawbar. Build and test commands are in [RUNBOOK §9](../RUNBOOK.md#9-mobile-ur10-and-rollwagen-cpu).

## Interface for integration

`build_mobile_assets.py` produces `mobile_ur10.urdf`, `rollwagen.usda` and
`mobile_manifest.json` in a caller-selected scratch directory. JSON includes
physical dimensions, wheel names, hitch offsets, transport positions, and measured
FK/COM evidence. The package roots remain caller arguments; the upstream robot
repository and its meshes are read-only. Generator removes `ros2_control`, Gazebo
and transmission blocks so importing this asset cannot load the real-rig plugin.

Import **the generated mobile URDF**, with `fix_base=False` and
`merge_fixed_joints=False`. Disable self-collision for initial bring-up, configure
arm position drives and keep passive caster joints undriven. Preserve the
`base_link` frame: its origin is chassis centre, initially 0.24 m above the floor.
`mobile_assets.author(stage, robot_usd, robot_path='/MobileRobot',
cart_path='/Rollwagen', spec=MobileSpec())` references that imported USD and authors
the independent cart. It does not boot Isaac or create a robot substitute.
Use initial root placement only **before** stepping physics; never write robot or
cart root poses to follow a route. The imported articulation/body/joint paths must
be discovered from the actual import, not guessed from a previous pedestal USD.

Defaults (all dimensions configurable through `MobileSpec`/JSON):

| Item | Default |
|---|---|
| Chassis | 1.00 × 0.60 × 0.20 m; 180 kg |
| Drive wheels | radius 0.16 m, track 0.70 m, width 0.08 m; 4 kg each |
| Arm mount | (0, 0, 0.50) m above ground; short 12 kg support column |
| Transport | source home arm pose, six gripper tines retracted; 0.65 m radius |
| Cart | 0.80 × 0.68 × 1.65 m; cage 35 kg plus 3.4 kg casters |
| Cart body origin | deck centre, initial ground Z 0.20 m |
| Robot hitch | base-link local (-0.62, 0, 0.04) m |
| Cart hitch | chassis local (0.65, 0, 0.08) m |
| Straight dock centre separation | 1.27 m; cart behind robot, same yaw |
| Drive limit | 0.35 m/s, 0.35 rad/s; wheel cap 4 rad/s |
| Tow limit | forward only, no pivot turns, nominal minimum radius 1.5 m |
| Hitch joint | yaw revolute, ±65° mechanical limits; passive, no drive |

Sensor **mount frames** are `lidar_frame`, `front_camera_frame`, `imu_frame`;
`arm_mount` and `hitch` are also preserved. Frames use +X forward/+Y left/+Z up.
The camera mount is not a ROS optical frame. Integration must instantiate actual
simulated lidar/camera/IMU sensors and supply their extrinsics; these assets do not
claim that a named empty frame produces observations.

`MobileController(stage, robot_body, cart_body, [left_joint_path,
right_joint_path], spec=..., hitch_path='/MobileHitch')` configures wheel velocity
drives. `command(v, omega, observation, transport_ready=False)` sets angular drive
targets and returns left/right rad/s; `park()` commands zero. Alternatively use
`mobile.wheel_targets` with Isaac's articulation velocity API (rad/s). Do not apply
both control paths. USD angular target velocities use degrees/s, converted in the
adapter per [OpenUSD DriveAPI](https://openusd.org/release/api/class_usd_physics_drive_a_p_i.html).
Call every control tick; a missing/stale observation parks immediately. The outer
simulation loop must park on adapter errors/timeouts and before stopping ticks.

The controller assumes wheel axes are the generated URDF's +Y; positive wheel
rotation drives +X. Before motion, hold `transport_positions`, acquire measured
joint feedback and compute the current collision envelope; use
`mobile.transport_ready(measured, targets, measured_envelope_radius)` as the
interlock. Do not set this flag merely because targets were sent. Static FK proves
an unloaded home COM margin, not stability of an extended moving arm or loaded cart.

## Physical docking and envelopes

`DockObservation(robot: Pose2, cart: Pose2, robot_hitch_z, cart_hitch_z,
robot_speed, cart_speed, robot_yaw_rate, cart_yaw_rate, age)` contains measured
planar body poses and world hitch heights, speed magnitudes and observation age in
seconds. Source it from the running simulator; label ground-truth PhysX state as
such in integration evidence. The controller does not synthesize odometry.

`attach(observation)` parks and requires age ≤0.1 s, both bodies ≤0.02 m/s and
≤0.02 rad/s, hitch XY error ≤0.025 m, height error ≤0.015 m, yaw alignment ≤5°.
It then authors a real `UsdPhysics.RevoluteJoint` between the rigid bodies, excluded
from the reduced-coordinate robot articulation. It changes no transforms. A
callback failure leaves the logical state unchanged. `detach(observation)` requires
fresh stopped bodies and removes the joint. `uncouple(observation)` is the mission-level
leave-the-cart-parked primitive on `return` (mirror of `attach`): it pins the parked cart
to the world (`hold_cart_pin`, a latched cart brake) *before* releasing the tow joint
(`detach`), so the cart stays braked as the robot drives home empty — every later
`release_cart_brake`, including the one `command()` issues on drive-away, no-ops while the
latch is set. Run these calls on the simulator thread between steps, with the stage's
writable edit target selected.

`coupling_state(observation)` reports requested state, enabled joint presence and
measured hitch residual. Residual >0.04 m, stale state or excessive articulation
makes it unhealthy and stops drive commands. **An authored joint is not proof of
PhysX contact or solver health**; integration must step and inspect this state.
The cart rolls through constraint/contact forces, not route pose updates or visual
parenting. Stock on the deck/shelf must be independent rigid bodies with contact;
this asset lane does not parent stock to the cart or implement grasping.

`occupied_envelope(robot, cart=None, spec=..., margin=.05)` returns a conservative
convex polygon covering chassis, wheels, transport arm, drawbars and independently
oriented cart. `swept_envelope([(robot, cart), ...], ...)` includes rotational-arc
inflation between pose samples for piecewise-linear XY/shortest-yaw trajectories.
This is planning geometry, **not a cart motion simulator**. Navigation must predict
or measure the independent cart trajectory and check the resulting full sweep.
The conservative hull fills the gap between cart and robot; narrow turn rejection
is intentional. Reverse towing is disabled pending validated jackknife control.

## Evidence and remaining checks

CPU unit tests cover free-base URDF structure, wheel geometry/axes, positive
inertias, passive cart joint anchors, docking rejection and transactional state,
wheel command units/interlocks, stale-feedback stops and sampled swept-envelope
containment. The real-source generator resolves every collision mesh and measures
home collision radius and static COM support margin; missing meshes fail closed.
The upstream gripper placeholder inertias are preserved (see the existing URDF
validator warning), not silently repaired.

Isaac import/drive response, material friction, caster settling, loaded towing,
joint creation/removal while stepping, arm holds, contact and tip resistance are
**not proven by CPU tests**. Integration owns that initial GPU execution. Start
unloaded in open space, inspect solver/joint errors and measured hitch residuals,
then add a representative tray and test straight travel, gentle turns, stopping,
detachment and return. No autonomous restocking demonstration is claimed by this
asset module alone.
