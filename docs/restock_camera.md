# Restocking camera geometry and depth calibration

`scripts/twin/restock_camera/` supplies bounded CPU geometry and support-calibration
helpers and a fresh rendered-pixel binding for `IsaacCartonSensors` and `SlotView`.
It does not boot Isaac or credit inventory from scene geometry. Offline ray checks are
**geometric design predictions, not rendered RGB-D measurements**.

## Current registered-support binding

`restock_camera.qr.FixedTileQR(camera_id).decode(rgb, evidence, now=..., parked=...)`
searches original RGB pixels with single-code detection on fixed overlapping
640×640 tiles, stride 480 with the last tile anchored at the image edge. The
2560×1920 view uses 20 tiles; larger images refuse. There is no initial full-frame
multicoder, resizing, expected-identity input or prior successful-frame crop.

The immutable `FrameQR` contains decoded text, full-image corners, duplicate
statistics, unresolved candidates, exact `FrameEvidence`, RGB hash and processing
time. `CameraContext.qr_observation` carries the SAME result used by support
registration into `IsaacCartonSensors`. The inventory adapter validates the pixels,
frame/session/epoch and reuses it instead of decoding the acquisition twice.
Custom providers that decode support markers must supply this field; providers
without it use one fixed-tile inventory decode. The diagnostic `_registrations`
return shape remains unchanged, with its result on `context.qr_observation`.

Spatial duplicates require identical decoded text, centers within 5 px and
corresponding-corner RMS <=2 px; first actual corners are retained, not averaged.
Conflicting text/corners, overlapping labels and the same identity at separate
locations refuse. Unreadable candidates remain explicit. Unknown/undecoded labels
inside an inventory search mask refuse the packet; support IDs never become SKUs.

The decoder enforces a **500 ms wall-processing ceiling**, discarding partial
results on overrun/error. Failed attempts consume their frame: replay cannot retry
old pixels into success. This ceiling does not extend acquisition freshness or
commissioning TTL, and processing completion never replaces capture time.

CPU replay of actual `native-source-view-01` PNGs reproduced the original 1/6
full-frame success. The implementation recovers all three codes in 6/6 frames,
with zero unresolved/conflicting labels, 0.95–1.03 mm maximum marker-fit residual
and 335–357 ms decode/hash/merge time. These six frames are the discovery set,
**not held-out proof**. Shared-result validation, depth/witness fitting, clearance,
native readback/queueing and multiple cameras still need end-to-end timing on an
independent integration capture. Runtime readiness remains false.
Shared-result validation took another 58–107 ms in this replay; decoding plus
validation plus diagnostic depth archive I/O/registration reached 516 ms on one
frame. That diagnostic I/O is not the live acquisition path, but the result does
not establish a complete 500 ms pipeline. No budget extension or further tuning
was applied; integration must refuse stale end-to-end observations.

Focused tests in `test_restock_qr.py` cover fixed coverage, pixel/frame binding,
negative/unreadable cases, duplicate/conflict rejection, timeout and replay.
The support/inventory test forbids a second decode on the shared acquisition.
No camera scene, physical layout or published configuration changes accompany
this decoder update.

RGB hashing snapshots all pixels into a contiguous C-order array on every
decode and shared-result validation, then hashes its buffer directly. This
preserves the original byte digest and mutation detection without the slower
strided-array `tobytes()` serialization. No hash/validation success is cached;
even contiguous inputs are copied to preserve independent snapshot semantics.
The API, frame/session checks and 500 ms gates are unchanged.

A bounded two-thread CPU comparison on the same six discovery frames (three
paired repetitions per frame) measured median shared validation 57.3 → 32.6 ms
and decode + validation + depth archive I/O + registration 470.2 → 428.7 ms.
The corrected chain still reached 551.0 ms (p95 513.6 ms): **the complete 500 ms
pipeline remains unproven**. All paired hashes, corners, identities and metric
registrations matched. Sensor aggregation was not measurable from this loaded
source capture alone: it lacks a genuine empty commissioning baseline. These
component measurements exclude native acquisition, witness/clearance checks,
sensor aggregation and action handoff; integration's final fresh-clock gate
must still reject expired observations. No further tuning follows this pass.

`restock_camera.sensed` separates dated rigid geometry from current vacancy.
An actual empty commissioning RGB-D frame measures a full support quadrilateral
and an independent visible plane witness. Paired physical QR markers establish
support identity and pose. Every later frame must decode those same markers,
fit their depth-backed metric layout, and reobserve the independent plane patch.
The loaded carton may hide the original footprint; that never refreshes the
empty scan or proves the hidden space empty. Required .62×.42 m dimensions are
checked inside the measured quad with uncertainty margins, never inflated from
a smaller inscribed image rectangle.

`restock_camera.binding.RenderedSupportContext` provides the executable adapter:

```python
context = RenderedSupportContext(
    camera_id="cart", intrinsics=k, slots=support_slots,
    measure_context=measured_camera_context,
    clearance_for_slot=independent_clearance,
    clock_context=persistent_acquisition_clock,
    depth_error_m=validated_depth_error, camera_revision="mount-revision")
geometry = context.commission(empty_rgbd_frame, commissioning_now,
                             observed_empty_quads, lifetime_s=600.)
sensor = IsaacCartonSensors(camera, profiles=profiles,
    pose_bridge_root=pose_bridge_root, camera_id="cart", context_provider=context)
```

`SupportSlot` supplies location, support identity/revision, primary and auxiliary
marker identities, metric printed side and rigid offset, separate label search
mask, and required dimensions. The label mask must be selected from the current
sensed view; it is not evidence of support or clearance. The full support mask is
projected from the measured quad registered into the current camera frame. Raw
legacy `SlotView` constants alone now fail closed in the runtime sensor adapter.
The original `calibrate_support` rectangular helper remains an offline/diagnostic
API; its legacy `slot_view` is insufficient for the current runtime proof gates.

Callbacks are deliberately native-runtime-free:

- `measure_context(frame, now)` returns `(base_from_camera, transform_stamp,
  epoch, parked)`. Stop both robot/cart and withdraw the arm for the capture.
- `clock_context(frame, now)` returns `(session_id, steady_domain,
  acquired_steady)`. Record acquisition time in one persistent monotonic clock
  across commissioning and mission resets; never timestamp retrieval as capture.
  Without it, a different simulation session cannot reuse commissioning geometry.
- `clearance_for_slot(location, support_view, frame, now)` returns independently
  measured `ClearanceEvidence`, or `None` for occupied/unknown volume. It runs
  after the fresh witness fit, so occupied source recognition does not require
  falsely claiming its volume free. Without clearance, target.clear is false
  and certified free height is zero. Use the measured support pose/footprint,
  uncertainty, current plane stamp and actual incorporated collision-map stamp.

`CameraContext` adds optional session/acquisition-clock fields after its original
five fields. `StationSensors` aggregates separately assigned cameras into the
existing `Sensors` packet and refuses unsynchronized station images. Its
`sensed_target_bboxes` are catalog volumes anchored to accepted current QR/depth
poses, separate from label/support masks. Integration chooses the active target
for mapper masking; multiple inventory cameras do not certify mapper coverage.

Baseline identity, original render/session/frame and acquisition time are immutable.
The 600 s default lifetime accommodates the current 158.03 s CPU timing estimate
plus bounded commissioning/reset/planning overhead. Configuration is restricted
to 1–900 s at commissioning. It cannot extend an existing record. Live RGB-D,
marker, witness, TF and clearance freshness remain <=0.5 s (TF skew <=0.05 s).
The unchanged 10 micrometre/s drift allowance, marker/depth uncertainty, 10 mm
combined limit and full-footprint containment can reject earlier. These are
bounded design policies requiring deployment validation, not measured accuracy.
Unbolting/deformation, changed marker layout, lens/mount revision or calibration
requires explicit invalidation and a new actual empty scan. Motion/localization
changes need a new parking epoch and fresh measured transforms. Session/frame
replay, missing/invalid depth, hidden witnesses and unknown identities refuse.

## Executable physical design and CPU audit

`restock_camera.config.camera_config(scenario, manifest)` returns camera IDs,
map/cart-chassis parents, optical transforms, pinhole intrinsics, collidable
housing/bracket solids and paired physical fiducial layouts. `author_camera_design`
authors those solids/cameras and actual printed marker textures in a supplied USD
stage; it never boots an app. Marker side refers to decoded QR corners, with the
texture quiet border accounted for in the physical plane size. Integration owns
native camera initialization, rigid attachment and actual rendered commissioning.

Current design sampling is 2560×1920, fx=fy=1280, cx=1280, cy=960;
this matches the installed Isaac camera's width/2, height/2 principal-point API.
the field of view and .15–3.5 m optical range are unchanged. This increases
sampling for the small paired support QR markers; actual decoding still needs
render validation. The revised shelf view is target+(-.50,.30,.22), looking at
target+(-.20,0,-.06) to retain both the top label and exposed support markers.
Source and cart optical poses remain as in the historical table below. The source
adds a rigid coplanar marker/witness tab behind the existing table; cart markers
and witness occupy its exposed left strip; shelf markers use the exposed left
strip of the actual shelf. These additions remain collision geometry and must
be included in navigation/planning. No rail, lip, handle/post or shelf is removed.

`scripts/plan_restock_sensed_views.py --help` is the current optional CPU helper.
Supply current scenario, measured-margin navigation summary, matching IK report,
mobile manifest, composed store, package roots and scratch output. It verifies
scenario/arrival identity, composes the cart independently of the robot, checks
sampled top/support/marker/witness rays against actual meshes plus new mounts,
and checks every URDF collision link and held payload against the added solids
over bounded joint-linear approach/contact/lift/preplace segments. Output hashes
include inputs, composed USD layers, collision meshes and helper sources.

An overlap rejects a candidate; a clear mount audit does not certify existing
store/cart/self-collision, transit, physical towing or an actual planner trajectory.
Geometric visibility does not certify QR decoding, error bounds or free volume.
Use the reported rejections; `runtime_ready` remains false until integration has
real sensor/physics evidence. CPU tests: `python3 -m pytest -q
scripts/twin/tests/test_restock_sensed_views.py scripts/twin/tests/test_restock_camera_config.py
scripts/twin/tests/test_restock_camera.py scripts/twin/tests/test_cad_restock.py`.
USD authoring tests require the CPU USD environment described in RUNBOOK §6a.

Current CPU design prediction (2026-10-13, no rendered sensor capture):

| View | Top samples | Empty support | Loaded support | Loaded witness | Each support marker |
|---|---:|---:|---:|---:|---:|
| Source | 81/81 | 81/81 | 9/81 | 25/25 | 25/25 |
| Cart pickup | 81/81 | 81/81 | 0/81 | 25/25 | 25/25 |
| Cart articulated arrival | 81/81 | 81/81 | 0/81 | 25/25 | 25/25 |
| Shelf | 81/81 | 81/81 | 17/81 | 25/25 | 25/25 |

The added solids have no conservative overlap with 24 URDF collision links or
held payload in 7,498 bounded samples over 16 local segments. This is only the
new-mount audit, with <=2 mm intersample displacement padding; it excludes
transit, self-collision and existing store/cart collisions. Shelf fit still needs
a measured full-footprint quad with uncertainty margin; nominal placement is only
4 mm from the flat shelf's front margin, so a sensed feasible center may shift
rearward. The helper does not change the scenario or manufacture a larger shelf.

Exact main input SHA-256 values for this prediction:

- Current scenario: `fbea2c7420ea7f4451cae27981fde897811f6ce7d6da1a09d1073ec3d1fc3a80`
- Measured-margin navigation: `0ccfb5d83f25b715fc031b9d6d6dda8f3377e4de34646b27b030bef4a08a35a5`
- Matching IK: `e57419969ce2496d9eb95c071e18605e023ae701636732c90b597b1b9abb50ad`
- Mobile URDF: `ac45b9f71360e0631ff79100f1a3c09772c773397cbee86c74fc26524d33d0e4`

The complete layer/mesh/helper hash manifest is emitted by the offline command.
Predicted marker image bounds are about 105×103 px at source, 121×120 and
118×115 px on cart, and 254×131 and 413×352 px at shelf. These are projected
quadrilateral bounding boxes, not a measured minimum module resolution.
All current IDs fit the minimum 21-module QR grid. Shorter explicit decoded IDs
are supported, but do not substitute a configured identity when decoding fails.
An 80 mm QR needs about 126 mm substrate with this writer's quiet border. It is
a possible lower-resolution fallback after moving source marker centers inward
to +/-280 mm and shelf centers to X=targetX-405 mm; cart left-strip space already
permits it. This alternative needs a new layout revision, geometric audit and
actual rendered commissioning. Freshness/latency thresholds remain unchanged.

## Historical single-ROI design (superseded runtime handoff)

The following records the original checkpoint and its refusals. Use the current
binding above for runtime integration, not the historical six-step handoff.

All lengths are metres. Matrices use column vectors and `parent_from_child`.
Optical axes are right/down/forward. Depth must be registered **distance to the
image plane**, not Euclidean distance along a ray. The proposed pinhole settings
are 1280×960, fx=fy=640, cx=639.5, cy=479.5, near=.15, far=3.5. These are simulator
design requirements; they are not a claim about an installed physical camera's
precision, distortion, baseline, exposure or QR decoding performance.

| Station | Optical centre and mounting proposal | Look-at target |
|---|---|---|
| Source | source support centre + (0,.20,.85); fixed mast behind the staging table, short overhead boom | support centre + (0,0,.08) |
| Cart lower deck | (.08,-.02,.88) in ground-aligned cart coordinates; fixed to underside of upper shelf | (.08,0,.30) in the same cart frame |
| Shelf | target support centre + (0,.55,.22); fixed bracket beneath upper shelf projecting into aisle | support centre + (0,0,.12) |

The cart chassis origin is .20 m above ground, so its camera translation in the
actual chassis frame is (.08,-.02,.68). The cart is .80×.68 m, deck top .225 m,
upper shelf underside .9075 m; the proposed .06×.025×.025 m housing fits beneath
that shelf. The camera and a bracket must be authored as physical geometry by
integration. No rails, shelf, back cage or handle-like post geometry is removed.
The exported cart has no separate handle solid; all available cage solids are
included. Do not invent a camera above the upper shelf to observe through it.

For the revised source at (-5.73,17.1), the proposed lens is (-5.73,17.30,1.45),
with a mast behind the table at approximately (-5.73,17.55). For the historical
source at (-5.73,17.0), a mast behind the table at approximately
(-5.73,17.45) can support the lens at (-5.73,17.20,1.45). The shelf lens is
(-3.05,5.376,.95), on a bracket extending approximately .33 m beyond the shelf
front. These additions need navigation and all-link arm collision validation;
the original navigation margins remain unchanged. Camera/body/bracket placement
is a candidate mechanical design, not an installed and collision-certified rig.

The actual arrival has independent robot pose
(-2.9713813783,5.9940609503,-3.0944697698) and cart pose
(-1.9221936604,6.5108180201,-2.2934103308), about 45.897° articulation. The helper
composes the cart camera through that independent cart transform. `ur_base_link`
is .50 m above ground for the inspected mobile URDF; it is not the chassis origin
at .24 m. Integration must use **measured** transforms at runtime.

The .60×.40×.16 m unit uses local wall3 throughout: its outward normal is local
+Y; source yaw π, cart yaw relative to cart −π/2, shelf yaw 0. The helper reads the
actual CAD panel (.260424×.108204×.02558 m) and tool envelope from the URDF and
reports conservative camera-housing separation from the contact/insertion/lift
tool envelope. This narrow check excludes other arm links, transit, self-collision
and the brackets; it cannot replace the manipulation planner's collision checks.

## Runtime handoff

1. Author/init actual cameras and their collision geometry in the already booted
   simulator. A fixed station camera is allowed; use its measured extrinsics.
   `geometry.usd_camera_transform(parent_from_optical)` converts to USD −Z-forward,
   +Y-up camera axes; transpose its matrix for `Gf.Matrix4d`. Do not apply this
   conversion again when an API already accepts optical axes.
2. Stop robot **and independently articulated cart**, withdraw the arm/tool from
   the view, and synchronize RGB, image-plane depth, camera intrinsics and TF.
   Recompute `relative_camera(world_from_arm_base, world_from_camera)` at parking.
3. `geometry.plane_roi(corners, base_from_camera, intrinsics)` inscribes an integer
   image rectangle inside a visible convex plane quadrilateral. Runtime corners
   must come from sensing. A projected bounding box can contain pixels outside
   the support; it is not a valid empty-support ROI. Distortion/registration must
   be corrected upstream.
4. Call `calibrate_support(depth, intrinsics, base_from_camera, roi, location=...,
   now=..., stamp=..., transform_stamp=..., frame_id=..., epoch=..., parked=True,
   expected_support_z=..., yaw_rad=...)`. The independent commissioned support
   height is a rejection prior only. The returned height, residual and conservative
   inscribed XY footprint come from the supplied depth. Every ROI pixel must be
   finite and in range. Missing pixels, elevated carton tops, tilted or irregular
   surfaces, small/truncated ROIs and stale/unsynchronized data are rejected.
5. Call `calibration.slot_view(now=..., epoch=..., parked=True, clearance_m=...,
   clearance_stamp=..., clearance_epoch=...)`. The vertical clearance must be
   measured by a separate registered sensor scan of the free volume; a plane fit
   does not establish it. Never fill this value from USD product semantics.
   An insufficient observed footprint is retained as insufficient; do not enlarge
   it to the catalog dimensions merely to pass manipulation's fit check.
6. Construct `IsaacCartonSensors(camera, pose_bridge_root=..., profiles=...,
   slots=[slot], base_from_camera=...)`. Only its **fresh rendered RGB-D/QR pixels**
   may establish SKU/count/empty support. Pass the measured transform in arm-base
   coordinates, not the USD camera-axis transform.

The integration caller owns frame-ID uniqueness, strictly monotonic simulation
time and the calibration epoch. Use a separate epoch per camera/station. Change
it on any motion (even moving away and returning to the same pose), localization
reset, cart articulation, parking transition, mount/ROI/intrinsics/extrinsics
change. Discard old `SlotView`/sensor adapters on invalidation. The helper refuses
different epochs, motion and calibration/clearance age above .5 s; the existing
sensor adapter separately refuses duplicate render frames. A wrist camera would
need a new measured transform and calibration after every arm movement; these
presets use fixed station/cart mounts instead.

A loaded carton hides the support, especially on the lower cart deck. Do not
recalibrate through it. An old empty scan expires; transferring its support plane
through newly measured cart geometry requires a separately validated temporal
registration path. That path is not implemented here. This is an explicit runtime
limitation, not permission to substitute the offline model.

## Offline helper and evidence

`scripts/plan_restock_camera.py --help` lists its explicit input/output arguments:
scenario JSON, embedded arrival/IK scenario, navigation summary, composed store
USD, rollwagen USD, mobile URDF and manifest. Use the CPU USD environment described
in RUNBOOK §6a/§12; no `SimulationApp`, renderer or extensions are imported.
Write output to scratch. The helper hashes input files, all used USD layers and
helper source. Mesh/cube solids (including instance proxies and disconnected
frame parts) are ray tested, without replacing a hollow cage by its overall AABB.
Unsupported visible geometry fails explicitly. Mesh faces must be convex for
fan triangulation. This is a sampled two-sided geometric check, not rasterization.

The output gives 9×9 top/support samples, FOV/range checks, blocker paths, optical
transforms, wall3 TCP/approach positions and the loaded carton's support occlusion.
It preserves a negative result; no collision margin or occluder is weakened.
The shelf front lip blocks part of the requested support margin even where its
carton top is visible. A valid smaller ROI cannot establish a larger placement
footprint. Integration must resolve the physical/support-scan limitation.

The inspected historical scene produced these visible-sample counts (out of 81):

| Station | Carton top | Empty support | Support behind loaded carton |
|---|---:|---:|---:|
| Source | 81 | 81 | 9 |
| Cart at pickup | 81 | 81 | 0 |
| Cart at articulated arrival | 81 | 81 | 0 |
| Shelf | 81 | 72 | 0 |

All top samples are within optical range (.365–.766 m across stations). The
shelf result rejects complete support visibility. These counts do not prove QR
decoding, full continuous visibility between samples, or sensor/physics success.

The actual flat shelf surface is **0.730000019 m**, matching the .730 m prior;
the front lip rises to **0.736000019 m** over Y≈5.040–5.046. The enclosing mesh
maximum is the lip height, not the placement plane. Calibration must fit the
observed flat .730 m patch and reject an ROI crossing the lip; raising the prior
to .736 would invent a floating placement surface. A .62×.38 m inner-plane
candidate avoids the lip but is smaller than the required .62×.42 m placement
footprint. Additionally, the top QR centre's ray intersects the empty plane
behind the usable shelf patch. This view cannot satisfy the existing single
`SlotView` ROI for both support and QR verification. Multi-view sensing or an
integration-owned adapter change remains necessary; no observation gate is relaxed.

The integration scenario changed its source Y from 17.0 to 17.1 while the supplied
arrival and composed USD still represented 17.0. By default the helper refuses
such a mismatch. `--historical-arrival` explicitly replays the embedded arrival
scenario and records differences from the current scenario. Those predictions
must not be presented as clearance evidence for the updated scenario.
`--current-source-only` instead inspects the revised source against a matching
source-support USD and makes no arrival/route assertion. Both modes check the
actual support centre, dimensions and top against the selected scenario. The
task's current-source report uses a scratch USD overlay translating the original
support/unit by .10 m; it is a geometric design proposal pending integration's
artifact regeneration, not an edited original scene or a runtime result.

Focused tests live in `scripts/twin/tests/test_restock_camera.py`; they use analytic
depth fixtures labelled as CPU tests. No fixtures establish sensor or physics
success. The task's exact-input visibility JSON and report are handed to
integration separately; final recorded demonstration remains integration-owned.
# Acquisition-to-use integration gate

`restock_camera/freshness.py` keeps the 500 ms acquisition deadline across
processing. Native frames carry their original monotonic acquisition timestamp;
registration, commissioning, carton sensing and sequential station aggregation
read a fresh clock after their work. They refuse expired evidence even while
simulation time is paused. CPU functional pixel fixtures use explicit fake clocks;
they do not establish native latency.

`Sensors.valid_until_steady` carries the earliest deadline through the same-host
ROS handoff. The transfer controller rechecks it before motion starts; native
gripper dispatch also refuses an expired queued command. The host and native
process must share the workstation monotonic clock. This is not a cross-machine
clock synchronization protocol. Runtime remains UNCOMMISSIONED until held-out
native acquisition, all processing, mapping and actuator handoff pass the limit.
