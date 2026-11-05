# PARKED: sim-rgbd-nav-obstacle-avoidance-v1

**Status:** Lane cancelled by captain decision (2026-11-05): navigation stays
2D-lidar based. No PR was opened. This branch is parked so no work is lost; it is
not intended to merge. Delete this file if the lane is ever revived.

## What was done

Research/design phase only — **no implementation code was written** before the
cancel arrived. The study below is recorded so a future revival starts informed.

### Findings (navigation stack on `fm/twin-robot-in-store-boot-v1`)

- The complete-cycle `coldwin*` runs use **`nav_mode: "reference"`**, not `sensor`
  or `nav2`. The reference `ArticulatedController` is stepped with
  `step(dt, odometry, observation, now=)`; its `Observation.obstacles` come from the
  integration-owned `IsaacRangeSensor` (simulated lidar collision raycasts).
- The most CPU-complete obstacle-detection-to-costmap path is **`nav_mode: "sensor"`**:
  `SensorNavigation` + `DynamicObstacleLayer` (`navigation/dynamic.py`) fed by a
  `source(now) -> [Detection]` resolved from `navigation.dynamic_obstacle_source`
  (`isaac_bundle._build_detection_source`, consumed only in sensor mode via
  `NavigationBridge.step`). An RGB-D nav camera would plug a depth→detections builder
  into that same seam.
- `navigation/observed_space.py` already fuses depth-camera `DepthFrame`s into
  free/occupied voxels (`ObservedSpace`), and `Observation.observed_space` is wired
  through the sensor route — the natural home for a front-camera "ESDF slice at base
  height".
- `scripts/twin/mobile_assets.py` already authors a `front_camera_frame` link at
  `(0.46, 0, 0.35)` on `base_link` (no USD camera prim on it yet). `mast_depth.py` is
  the model for authoring a masked-depth camera + nvblox feed; `native_cameras.py`
  (`initialize_camera`, `set_camera_render_products_enabled`) is the seam for
  authoring + phase-sharing (pausing) a render product.
- `scripts/twin/restock/gpu_budget.py`: the 12 GB card sits at ~9.6 GB with one arm
  depth feed (26 MiB free at peak). A nav camera must be budgeted and time-shared
  (live during drive phases, off during arm phases).
- `nav2.py` already wires an `nvblox::nav2::NvbloxCostmapLayer` into the Nav2
  global/local costmaps via `write_assets(nvblox_topic=...)`.
- The real-robot nvblox reference (`/home/manishsaini/ur10/cams/cellmap/`) is a
  **cuMotion 3D ESDF-via-service** path, not a Nav2 2D-costmap path: reusable shapes
  are voxel 0.02, `global_frame`, the `static_mapper` workspace bounds + integrator
  params, the `dynamic_mapper` occupancy/decay probabilities, `mapping_type: "dynamic"`
  for moving obstacles (depth-only, no detection messages), and the
  `/camera_0/{depth,color}/{image,camera_info}` remap convention in a
  `component_container_mt` composable container. For a Nav2 mobile-base path it would
  have needed `esdf_mode: "2d"` + `publish_esdf_distance_slice: true` + slice params,
  which are absent there and would be authored fresh.

No files under the navigation/restock packages were modified.
