# Observed-space implementation checkpoint

**Retired from motion acceptance.** The controlled static-scene replacement is
[temporal nvblox navigation](navigation_temporal_mapping.md). Checkpoint7739eae
and this report preserve the failed investigation; these candidate rigs must not
be deployed. The controller rejects `Observation.observed_space`.

This implementation is **not ready for the autonomous mission**. It provides a
CPU depth binding and occupancy prototype, with fail-closed tests. Useful motion
under complete observed stopping-sweep coverage has not been demonstrated.

`twin.navigation.observed_space.RenderedDepthReader` accepts integration's native
Isaac 5.1 `RenderedCamera`. It reads actual OpenCV intrinsics, registered float32
`distance_to_image_plane`, `rendering_frame`, and `rendering_time`. The caller
supplies acquisition-time optical TF and the independently measured base/cart
body pose. Held frames, invalid calibration and unsynchronized transforms fail.
With explicit exact `self_paths`, the adapter reads `instance_id_segmentation`
and its `info.idToLabels` from the same acquisition. Only exact rig subtrees
mark own endpoints; names sharing a prefix remain external. Missing segmentation
fails closed when this mode is requested. Shadows remain unknown. A CPU replay
of integration's native source-view recording classified 687,264 own pixels
among 4,915,200 depth pixels. This validates data binding, not motion coverage.

`ObservedSpace` uses 10 cm voxels by default. Finite valid depth rays clear cells
only before an endpoint band that accounts for configured depth error and voxel
radius. Invalid/background pixels and shadows remain unknown; external endpoints
take precedence over free and own-solid evidence. This is a discrete sampling
model, not a guarantee for subpixel or subvoxel obstacles. No PhysX volume query
is represented as a camera measurement.

The articulated controller accepts optional `Observation.observed_space`;
`SensorNavigation.step(..., observed_space=...)` forwards it. Static collision,
transport, coupling, articulation, forward-only motion and braking gates remain
mandatory. Depth coverage requires at least the measured 10 cm localization
bound plus the existing 5 cm geometry margin. Both measured and commanded
curvature rollouts are checked. The prototype pads between sampled body poses.

`RigVoxels` requires actual collision-solid samples, including arm transport
geometry, camera mounts and carried payload. **Production generation and runtime
validation of this model are unfinished.** A whole cage bounding box or arm
transport disk must never be substituted for current solid matter: their voids
cannot exempt unknown space. Moving caster geometry and payload changes also
need an explicit model. Do not enable this optional path with an approximate
own-solid exemption.

The CPU mount audit tried 12 base cameras and 24 base/cart cameras at the actual
staging pose. With conservative source-collider occluders, the layouts leave
35 and 22 of 161 exterior probe cells unknown respectively. Even this necessary
subset does not pass. Candidate housing/mast geometry is design evidence only;
complete mounting collision/attachment, solid voxel generation, acquisition
timing and useful forward progress remain open.

A smaller four-camera investigation (front/rear RGB-D on each body, height
0.60 m) leaves 79/161 probes unknown. Upper cart clearance is outside near-field
vertical FOV; other gaps are occluded by the articulated rig. Its 80x60 images
at 110-degree horizontal FOV sample approximately 7.1 cm per pixel at 2 m and
21.4 cm at 6 m. A two-pixel target assumption therefore needs roughly 14 cm or
43 cm apparent width at those distances, in addition to the 10 cm voxel model;
this is not a detection guarantee. The model cannot claim thin-object coverage.

Four-image Python fusion took 2.80 s in the CPU fixture, so the current full-grid
prototype cannot support a 10 Hz loop with 300 ms freshness. RTX render latency
is unmeasured and CPU OBB rendering is not a proxy. At 10 Hz, acquisition plus
fusion must fit 100 ms total; four views leave at most 25 ms per view before
shared processing, and 24 views only 4.2 ms. Neither budget has been demonstrated.
Future work should assess a small practical sensor arrangement, indexed/local
fusion and timestamped temporal observations; this checkpoint does not silently
persist old free space or increase freshness limits.

Fixed perception-camera mounts are a separate completed geometry correction:
`environment_grid(environment, scenario, camera_config)` includes their rotated
XY bounds. Use `scripts/check_restock_navigation.py --camera-config FILE` when
checking the actual mounted scenario. Both route legs pass with 805 obstacles
and unchanged 5 cm geometry plus 10 cm localization margin. This result uses
ideal CPU feedback and is not observed-space or native simulation evidence.

Focused checks: `python3 -m unittest discover -s scripts/twin/tests -p
'test_navigation_observed_space.py' -v`. The tests cover calibrated projection,
held/missing/stale frames, unknown shadows, external occupied precedence,
independent cart transforms, and fixed mount bounds.
