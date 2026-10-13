# Stationary cart diagnosis — offline, 2026-10-13

The native cart is not stable within the existing 50 mm envelope. No autonomous
restocking cycle has run. This diagnosis uses saved source, USD and telemetry only;
it does not authorize a new GPU run or establish a production fix.

## Reproduction and comparison

The closest executed path imports the mobile URDF, cart and full store, initializes
physics, holds the robot's arm and zero wheel targets, then observes an uncoupled,
unbraked cart for up to 120 steps. It is a prerequisite diagnostic, not the user's
collect/couple/tow/transfer/return path. The intended stationary acceptance remains
50 mm XY drift from measured initial position, 80 mm Z drift, 0.1 rad tilt, and
last-ten-step speeds below 5 mm/s and 5 mrad/s. No acceptance change is proposed.

| Native run | Code | Cart solver position/velocity | Recorded steps | Failure simulation time | Final x minus nominal x |
|---|---|---|---|---|---|
| assembly-03 | 361e50e | 16/1 | 73 | 1.266667 s | −50.801 mm |
| assembly-04 | e9acc7f | 64/16 on all nine bodies | 119 | 2.033333 s | −50.217 mm |

Both failed the measured-initial XY envelope. Nominal x above is a plotting
reference, not the gate's exact baseline. Initial baselines were not serialized;
first samples occur after a physics step. First-to-last recorded x changes are
−50.174 and −50.007 mm respectively. Raising iterations changed the trajectory
and delayed failure in these single runs; it did not fix or identify the cause.
There are no repeated seeds/runs from which to infer statistical significance.

Assembly-01 failed ROS bridge node initialization. Assembly-02 failed with a native
segmentation fault during stage reopening. Deferring bridge initialization until
after the final scene reset allowed 03/04 to reach measurement; this exposes the
cart failure, not evidence that the bridge caused mechanical drift.

Initiating transition: physics initialization and gravity/contact/joint stepping
of the free cart. The exact first erroneous impulse is unrecorded. Exposing
conditions: dynamic bodies and joints are enabled, the cart is uncoupled, and the
stationary loop is now reachable. Visible symptom: sustained negative-X travel
beyond 50 mm without a cart command. The earlier static optical bench disabled
dynamics and therefore masked this class of failure. It is not a successful
stationary-cart control. The robot stays within its envelope under active holding,
but its driven articulation is not a mechanically equivalent control either.
There is no proven successful native free-cart case in the evidence.

## Geometry, joint frames and actuation

`scripts/twin/mobile_assets.py:author_cart` came from 24d364a; its caster authoring
has not changed in subsequent integration history. Four wheels have radius 75 mm,
nominal centers at z=75 mm and Y rolling axes. All four trails point −X by 25 mm.
Nominal swivel and rolling anchors coincide algebraically with identity joint
frames. Adjacent joint collisions are disabled. The fork box and deck overlap by
5 mm in Z, but their directly joined pair is filtered; overlap alone is not proof
of an active contact error. No cart drive, brake or commanded velocity is authored.
`assembly.hold()` commands the robot wheels and arm only. No hitch is attached.

The saved assembly-04 USD is a post-reset snapshot, not the pristine initial
condition. Its analytical cylinder minima are −0.00009, −0.00276, +0.00014 and
+0.00158 mm relative to the z=0 floor. Rotated box bounds had suggested up to
0.824 mm penetration; that is a bounding-box artifact, not cylinder penetration.
Post-reset swivel anchor separations are 0.014–0.061 mm; rolling anchors are
0.387–0.700 mm apart. These are measured residuals after solving, not proof of
incorrect authored frames or a measured cause of the drift.

The chassis is 35 kg, each fork 0.25 kg and each wheel 0.6 kg. The 140:1
chassis/fork ratio and small fork inertia can make constraint convergence
sensitive. The failed 64/16 experiment disconfirms the claim that simply increasing
these iterations suffices. It does not distinguish contact, inertia, initialization
or solver effects. Other physics parameters were not intentionally changed.

## Contact and settling uncertainty

Two static slabs occupy the area beneath the cart: Backroom_floor spans z=−0.12
to 0 m; Ground_exterior spans z=−0.03 to −0.01 m. The lower top surface is 10 mm
below the load-bearing floor. No nearby wall or fixture overlaps the cart envelope
in the saved spatial bounds. This establishes overlapping geometry, not that both
surfaces generate solver contacts. Installed PhysxSchema contactOffset defaults to
−inf, meaning simulation-selected by shape extent; its numerical runtime value
was not logged. The schema describes the rigid-body default restOffset as zero.
Actual contact pairs, normals, impulses and effective offsets remain unmeasured.

The free, ideal cart on a level floor has no specified horizontal propulsive force.
A short transient during constraint/contact initialization is physically plausible;
an unbraked cart also cannot be assumed to arrest arbitrary initial momentum.
Continual drift must be measured and explained before calling the assembly stable.
This is not grounds to add a hidden brake or relax the 50 mm safety envelope.

Recorded chassis Z spans only 1.048 mm (03) and 0.294 mm (04), yet sampled Z
velocity is always positive: 41.9–108.9 and 46.6–70.6 mm/s. Last-ten-sample mean
Z velocity is 70.3 and 56.2 mm/s. These velocities do not integrate to the saved
Z displacement. Position correction, sampling phase, or state-access semantics
could contribute; the recordings cannot distinguish them. Calling this simply
physical bouncing or energy growth would overstate the evidence. Position and
velocity both come from Isaac body APIs after `world.step(render=True)`.

## One proposed counterfactual — not executed

Test whether the lower overlapping ground contributes spurious contact impulses.
In a disposable diagnostic scene only, disable collision on Ground_exterior while
retaining the complete Backroom_floor support and all visible geometry. Change
only that collision participation; retain assembly-04's bodies, masses, joints,
64/16 settings, initial placement, unbraked state, guards and acceptance thresholds.
The bounded experiment is a matched baseline and single-condition variant, each
at most 120 steps, with the same early-abort rules and resource cap. It requires
firstmate's next authorization; no runtime change has been made here.

For both arms, log before-reset authored geometry, measured initial state, actual
per-wheel contact pairs/normals/impulses and effective offsets, joint anchor errors,
and same-step PhysX pose/velocity observations. If that instrumentation cannot
identify contact provenance, report the diagnostic limitation rather than claim
a causal result. A baseline with no lower-slab contacts already weakens this
hypothesis; unchanged drift after removing lower-slab collision falsifies its
necessity in that setup. Disappearance of drift with matching baseline contacts
would support a contact-interaction cause but still needs repeatability before
production remediation. Do not simultaneously alter caster orientation, material,
time step, inertia or solver parameters. The caster-trail asymmetry and unresolved
telemetry consistency remain alternative explanations, not additional proposed tests.

## Evidence and remaining gates

Task artifacts: `native-assembly-03/stationary-feedback.json` SHA256
`d5c53bd50436c80d946c124c12d00a210a40d1f8dedc9331620feb16476ca1a0`;
04 SHA256 `99a75da8ef08d76bcf349b2a2addd366aa2bef34ea9acf5ab1b4f6776e8e2e22`.
`cart-offline-analysis.json`, `analyze_cart.py` and
`cart-stationary-comparison.png` preserve the CPU reduction and plot in the task
artifact directory. Original native USD, logs and recordings remain unchanged.
Peak GPU usage for 03/04 was 4594/4820 MiB; minimum available RAM was
15741/15700 MiB; both wrappers reported services unchanged.

CPU checkpoint at e9acc7f: 369 tests passed, 16 subtests, five optional skips.
This does not certify native dynamics. Gripper attachment warnings, imported
reference completeness, native map coverage, manipulation and the full mission
remain unresolved. Camera processing is also uncommissioned: acquisition-to-use
must be checked with a fresh clock after decode, hashing, registration, clearance
and aggregation and before action. Decoder-only duration and component sums do
not establish the unchanged 500 ms native freshness limit. Delayed-processing
rejection remains an integration implementation gate after this offline step.
