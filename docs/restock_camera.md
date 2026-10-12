# Restocking camera geometry and depth calibration

`scripts/twin/restock_camera/` supplies bounded CPU geometry and support-calibration
helpers for the existing `IsaacCartonSensors` and `SlotView` adapter. It does not
boot Isaac, perform perception, or credit inventory. Offline ray checks are
**geometric design predictions, not rendered RGB-D measurements**.

## Frames and physical camera candidates

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
