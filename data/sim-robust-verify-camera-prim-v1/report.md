# sim-robust-verify-camera-prim-v1: shelf-camera prim lifecycle at the post-place verify

## Symptom

In the armcam re-run 3 (2026-10-23, branch at PR 215 + 216, honest `live_pose` transfer) the arm
picked, carried, placed and released the carton at the cart, `mission.json` was written and the app
shut down cleanly (peak VRAM 10323 MiB, no core dump). The one remaining failure was at **verify**:

```
state=failed
reason=RuntimeError: Accessed invalid expired 'Camera' prim </World/restock_camera_shelf/Camera>
       [native_cameras.py:65 get_intrinsics_matrix | camera.py:2155 get_opencv_pinhole_properties
        | camera.py:1907 _get_lens_distortion_properties]
```

Run dir: `~/datasets/twin-robot-in-store-boot-v1/mission/armcam/` + `armcam.console.log`.

## Root cause (established from the console log, no GPU)

Two console lines bracket it:

```
9391 DESTROY_RENDER_PRODUCT {... 'prims_removed': ['/World/restock_camera_shelf/Camera']}
9392 STAGE_SCOPED_SENSING torn down: ['shelf']
...
12977 SIM_ATTACH_FALLBACK release body=/RestockScenario/HandlingUnit   (the place, LATER)
```

- The fm381 **stage-scoped sensing teardown** (`sensing_sources.stage_scoped_teardown`, wired only
  when `RESTOCK_STAGE_SCOPED_SENSING` is set) frees the "unused" **shelf** camera at the pick lift to
  return VRAM before the release peak. It picks the shelf camera via `_unused_sensing_camera_ids`,
  whose premise is *"the shelf camera serves only the transfer stage, which this mission does not
  run."*
- But the armcam mission runs `start_at=transfer stop_after=verify`, i.e. it **does** run transfer
  and verify. The teardown fires at the `live_pose` transfer pick lift, destroying the shelf camera's
  render product **and** `RemovePrim`-ing `/World/restock_camera_shelf/Camera`.
- After the place, the **verify** stage runs the camera `InventoryVerifier`
  (`verification.py`), which calls `capture(now, 'rollwagen', job.shelf_id)` →
  `StationSensors.capture` reads the destination (shelf) camera → `IsaacCartonSensors.capture` calls
  `camera.get_intrinsics_matrix()` on the now-expired shelf prim → the raw Isaac `RuntimeError`.

Why scripted/still (warm11, visualsfull, lidar2) never hit it: they run the SIM ground-truth verify
(`verify_ground_truth`), which reads the handling-unit pose, never the shelf camera.

### What verify expects alive, by mode
- **camera verify** (`InventoryVerifier`, no `verify_ground_truth`): the fixed `shelf` station camera
  (+ its commissioned baseline) must be live at verify. The armcam bay shelf is NOT commissioned
  (`commissioning_required_locations = backstock_00, rollwagen`), so camera verify can't confirm there
  anyway — but it must at least not read an expired prim.
- **ground-truth verify** (`verify_ground_truth.enabled`): reads the handling-unit prim pose; the
  shelf camera is never read, so freeing it is fine.

## Fix (CPU-reproduced first)

1. **Keep the camera alive until its last reader.** `build_sensing_sources` gains
   `keep_shelf_camera`; the teardown EXCLUDES the shelf camera when it is set. `isaac_bundle` computes
   it from the config via the new pure predicate `sensing_sources.shelf_camera_needed(active_stages,
   transfer_ground_truth, verify_ground_truth)`: True when an executed stage still reads the shelf
   camera (camera-mode transfer, or the verify-stage `InventoryVerifier`). Collect-only / fully
   ground-truth missions return False → the shelf camera is freed exactly as before (byte-identical;
   the teardown is only ever wired under `RESTOCK_STAGE_SCOPED_SENSING`).
2. **Defensive named failure.** `destroy_camera_render_product` latches the `RenderedCamera` as
   released; a later `get_intrinsics_matrix` / `get_current_frame` raises the named
   `CameraRenderProductReleased` (naming the camera prim) instead of Isaac's opaque expired-prim
   `RuntimeError`. A regression now fails by name, not a bare prim trace.

Scripted/still path untouched (GT verify never reads the shelf camera). PR 216's at-exit teardown
ordering is untouched — this only changes what the mid-mission fm381 teardown frees.

## CPU tests

`scripts/twin/tests/test_sim_robust_verify_camera.py` (10 tests): the `shelf_camera_needed` stage
predicate (incl. the armcam transfer-GT + camera-verify case), the teardown keep-vs-free core
(`_run_stage_scoped_teardown`), and the `CameraRenderProductReleased` named failure after a per-op
release. Regressions green: `test_fm380_stage_scoped`, `test_fm379_sensing_drop`,
`test_restock_native_cameras`, `test_sensing_sources`, `test_restock_sensing_factory`,
`test_restock_shelf_target`, `test_isaac_bundle`.

## Bounded GPU proof (for the integration lane)

`docs/examples/verify-camera-prim-mission-fragment.json`: the armcam staging UNCHANGED plus the one
new key `verify_ground_truth: {enabled: true}`, so the honest `live_pose` transfer COMPLETES through
verify at the uncommissioned bay (GT verify, consistent with the already-GT transfer; no shelf-camera
read; no expired-prim crash). The code fix also removes the crash on the camera-verify path (shelf
kept alive → live read, or a named `CameraRenderProductReleased` if ever destroyed-then-read). One
bounded run under `/tmp/aile-gpu.lock`, never forced, never beside live Blender, whispr untouched.
PENDING a GPU slot.
