# Stationary cart investigation record — 2026-10-13

## Complete passive reduction and one proposed control

All nine bodies and eight joints are reduced in
`/home/manishsaini/firstmate/data/twin-robot-in-store-boot-v1/passive-complete-reduction.json`.
The same directory holds `reduce_passive_all.py`, `passive-all-body-timesteps.csv`,
`passive-joint-timesteps.csv`, and `passive-contact-timesteps.csv`. These preserve
phase-labelled per-step values and every recorded contact pair. Sampling is after
each completed `world.step(render=True)`; contacts distinguish initialization and
step-N. All recorded intervals are0.016666667536s. This reduction includes the
serialized measured initial sample; earlier first-to-last-recorded-row figures
below intentionally use a slightly shorter interval.

| Body | Measured COM ΔX mm | Integrated reported VX mm | Measured COM ΔZ mm | Integrated reported VZ mm |
|---|---:|---:|---:|---:|
| chassis | −50.073 | −65.212 | −0.110 | 111.740 |
| fork0 | −50.029 | −77.940 | −0.114 | 119.260 |
| wheel0 | −50.159 | 113.221 | 0.005 | −2.794 |
| fork1 | −49.981 | −78.111 | −0.182 | 119.315 |
| wheel1 | −50.211 | 112.964 | −0.007 | −3.181 |
| fork2 | −50.031 | −78.405 | −0.039 | 104.687 |
| wheel2 | −50.031 | 92.516 | 0.003 | −4.092 |
| fork3 | −49.986 | −78.334 | −0.106 | 104.649 |
| wheel3 | −50.061 | 94.044 | −0.000 | −4.886 |

COM positions use measured body rotation and native local COM, rather than
equating chassis origin with its offset COM. Trapezoidal integration is a
diagnostic comparison, not an asserted reconstruction of PhysX position updates.

All contact pairs with nonzero reported impulses are listed here. Vectors sum
recorded contact impulses across initialization and measured steps, in N·s;
they are not instantaneous forces or complete joint-force measurements.

| Pair (wheel shape → Backroom_floor) | Nonzero points | Σ impulse X | Σ impulse Y | Σ impulse Z |
|---|---:|---:|---:|---:|
| wheel0 | 249 | −0.315625 | 0.379948 | 179.658492 |
| wheel1 | 262 | −0.590828 | 0.380045 | 181.162842 |
| wheel2 | 257 | −0.497435 | 0.702965 | 212.184219 |
| wheel3 | 256 | −0.996118 | 0.553361 | 210.713623 |

Every reported hitch/drawbar and lower-ground contact has zero impulse. No other
cart self/store contact pair is recorded. All nine cart bodies have threshold-zero
contact reporting and callbacks report no errors. This disconfirms an observed
unexpected contact push as the leading explanation; it does not expose every
solver-internal constraint impulse or guarantee reporting of uninstrumented bodies.

| Joint | Max anchor separation mm | Max reported anchor-velocity difference m/s |
|---|---:|---:|
| swivel0 | 0.0513 | 0.000000249 |
| swivel1 | 0.0695 | 0.000326318 |
| swivel2 | 0.0489 | 0.000343503 |
| swivel3 | 0.0673 | 0.000452230 |
| roll0 | 0.9381 | 0.136906 |
| roll1 | 0.9952 | 0.141004 |
| roll2 | 1.1641 | 0.120605 |
| roll3 | 1.2173 | 0.133779 |

Anchor velocities include angular velocity crossed with the anchor's offset from
the measured COM. The mismatch is concentrated at rolling constraints, while
swivel anchor velocities nearly agree. This supports investigating the interaction
of position correction and the velocity solve; it does not prove position
correction causes or dominates the50mm drift. We did not record separate
pre/post-position-solve velocities or joint impulses, so that contribution cannot
be quantified from these callbacks alone.

PhysX documents split-impulse solving: position and velocity iterations use
different bias terms, and reported final velocities need not integrate to the
measured positions. Thus the disagreement alone is not evidence of a bad read
path or physical energy gain. [NVIDIA PhysX simulation documentation](https://nvidia-omniverse.github.io/PhysX/physx/5.7.0/docs/Simulation.html).
The installed107.3-family documentation specifically discusses TGS velocity
iterations above four and suggests zero or few velocity iterations. This replay
logs that exact >4 warning for the chassis. [NVIDIA Omni Physics107.3 limitations](https://docs.omniverse.nvidia.com/kit/docs/omni_physics/107.3/dev_guide/guides/current_limitations.html).

The strongest next single-variable control proposed from this reduction is cart
velocity iterations16→1, holding position iterations64 and every physical model,
floor, material, timestep, CPU/TGS backend, passive observer and threshold fixed.
It is a test of the documented position/velocity-solve interaction, not a general
parameter sweep or a claimed remedy. Keep at most120steps/300s and the original
early stop/resource gates. No such run or production setting change has occurred.
Compare rolling-anchor velocity disagreement, COM displacement/integration gap,
actual drift and final settle speed with the preserved passive baseline.
Unchanged gaps/drift would disconfirm16 velocity iterations as the relevant
interaction in this setup. Improved velocity consistency without acceptable
drift/settling would be a telemetry/solver-semantics result, not an operational fix.
Original run03 already failed with16/1: excessive velocity iterations are therefore
not established as necessary for the original failure. The proposed64/1 control
isolates them at the current fixed position-iteration setting.

## Latest passive-observer baseline — FAILED

The independent contact-observer audit found that late `SingleRigidPrim`
construction in the matched pair changed transforms/sleep-related state. That
limits interpretation of those runs. Original stationary runs03/04 predate these
observers and remain failed evidence. No production cart or floor fix is established.

Firstmate authorized one subsequent baseline at 5cc2d00, replacing only the late
diagnostic wrappers with raw PhysX tensor views/getters. It failed at step119,
simulation time2.033333s, with50.0166mm XY drift from the serialized initial state.
The 50mm envelope, original floors, mass, friction, timestep,64/16 solver settings,
physical joints and unbraked state were retained. No variant or sweep followed.

`observer-attachment.json` records equality before/after attachment without a
physics step: authored layer hashes, independent native rigid-body poses and sleep
states, simulation time and chassis velocity. A second raw view also reproduced
all copied tensor transforms, velocities, masses, COMs, inertias, gravity/simulation
flags, materials and contact/rest offsets. Both guards passed before stepping.
The installed vendor implementation was inspected: these raw getters bind backend
handles and copy buffers; they do not invoke high-level wrapper initialization.
This is a check of exposed state, not a proof about every hidden solver variable.

Raw tensor inconsistency remains: wheel0 moved−49.718mm in X while integrating its
reported X velocity gives+112.261mm. Its approximate Y rotation changed−0.664rad
versus integrated angular velocity+1.721rad. Chassis Z changed+0.212mm versus
integrated reported COM Z velocity+111.040mm. Integrals use consecutive sample
trapezoids; COM velocity and body-origin position are explicitly distinguished.
These observations do not justify claims of bouncing, energy gain, or a root cause.
The wheel0 tensor shape slot0 reports contact offset0.005450001m, rest offset0,
static/dynamic friction0.5/0.5 and restitution0. Other padded shape slots are not
treated as real shapes. Ground effective offsets and friction-anchor impulses
remain unmeasured; contact callbacks again reported no errors.

The backend is explicitly **CPU PhysX**, TGS at60Hz, enableGPUDynamics=False and
override−1; **GPU RTX rendering** is separate. No backend setting was changed.
The replay took approximately53.5s, peaked at4745MiB GPU, and retained at least
16042MiB RAM. Services were unchanged. Afterwards9669MiB GPU and20647MiB host RAM
were available and the lock was confirmed free. GPU ownership was released.

Exact replay evidence:
`/home/manishsaini/datasets/twin-robot-in-store-boot-v1/native-passive-baseline-01/`
contains `observer-attachment.json`, `pristine-authored.usda`, `measured-initial.json`,
`stationary-feedback.json`, `contact-events.json`, `contact-provenance.json`,
`native-result.json`, `run.json` and `isaac.log`. CPU reduction artifacts are
`/home/manishsaini/firstmate/data/twin-robot-in-store-boot-v1/analyze_passive.py`,
`passive-baseline-summary.json` and `passive-baseline.png` in that same task artifact
directory. The earlier independent audit is `contact-observer-audit-2026-10-13.md`.

Final CPU checkpoint65bd36e:390 tests passed,16 subtests,five optional skips after
the confirmed camera latency merge. Its reported offline timing still has500ms
overruns. The acquisition-to-use rejection gate remains unchanged; no native
camera/map/planner/actuator commissioning or autonomous cycle is claimed.

## Authorized matched native follow-up — both failed

Historical evidence warning: this pair used the subsequently identified mutating
diagnostic wrappers. Read it with the passive replay and audit above.

The original offline diagnosis below is retained as the pre-experiment record.
Firstmate subsequently authorized exactly one matched pair at code d2441bb.
That authorization is consumed. No further native run was made; the GPU slot was
released for shelf reconstruction. No production floor change was applied.

| Diagnostic arm | Steps | Final simulation time | XY drift from serialized measured initial | Final cart speed |
|---|---|---|---|---|
| baseline | 90 | 1.550000 s | 50.269 mm | 80.969 mm/s |
| lower-ground collision disabled | 98 | 1.683333 s | 50.788 mm | 81.674 mm/s |

Both failed the unchanged 50 mm gate. The variant removed all reported lower-ground
contacts but did not remove the failure. Lower-ground collision participation is
therefore not necessary for this drift in the tested setup. It is not an established
root cause and the experiment does not justify a production floor patch.

Actual contact callbacks recorded 885 lower-ground contact points in the baseline,
all with zero reported impulse. The baseline also recorded 168 hitch/drawbar
points, all with zero reported impulse; the variant recorded 188, also zero.
Backroom-floor wheel contacts carried nonzero impulses in both arms. No callback
errors were recorded. Contact presence alone must not be described as a pushing
force. Effective simulation-selected contact offsets and friction-anchor impulses
remain unavailable through the selected callback. Pose/velocity inconsistency and
joint convergence remain unresolved; no alternative cause is certified.

The diagnostic saved pristine pre-reset authored layers, actual measured initial
states, per-step native body pose/velocity and joint-anchor residuals, and contact
collider identities/normals/impulses/separations. Both arms used identical diagnostic
instrumentation. Only the variant's lower-ground collision flag changed; visible
geometry, the load-bearing floor, 64/16 solver settings, physical joints, masses,
placement, free casters and original acceptance limits were retained.

Task dataset directories are `native-contact-baseline-01/` and
`native-contact-variant-01/`, each containing `pristine-authored.usda`,
`measured-initial.json`, `contact-events.json`, `contact-provenance.json`,
`stationary-feedback.json`, `native-result.json`, `run.json`, and `isaac.log`.
Exact audit locations on the workstation:

- `/home/manishsaini/datasets/twin-robot-in-store-boot-v1/native-contact-baseline-01/`
- `/home/manishsaini/datasets/twin-robot-in-store-boot-v1/native-contact-variant-01/`
- `/home/manishsaini/firstmate/data/twin-robot-in-store-boot-v1/contact-pair.json`

The task artifact directory contains the reproducible CPU reduction
`analyze_contact_pair.py`, hashes and summary `contact-pair.json`, and
`contact-pair.png`. The exact local dataset paths were sent to firstmate for
independent dynamics audit; no files were removed or overwritten.

Each guarded run took approximately 68 s; the matched pair completed in under
three minutes. Peak GPU usage was 4745 MiB in each arm. Minimum available RAM was
15225/15075 MiB. Services were unchanged in both. After the pair, 9669 MiB GPU and
19606 MiB host RAM were available; the GPU lock was confirmed free.

Subsequent CPU integration at a446380 consumes the merged camera component and
enforces acquisition-to-use expiration after processing/aggregation and before
motion or queued native gripper dispatch. Validation: 380 tests passed, 16 subtests,
five optional skips. No GPU validation of that later CPU change was attempted.
Camera runtime and the autonomous cycle remain uncommissioned/incomplete.

## Original offline record (before the authorized pair)

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
