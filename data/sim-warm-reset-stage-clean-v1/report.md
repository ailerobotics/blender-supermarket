# sim-warm-reset-stage-clean-v1 — reset the stage clean so the warm two-run re-seats

## Problem

`warmtwo1` (PR 299 warm two-run, `--warm-attempts 2`, collect-only; saved
`/home/manishsaini/datasets/twin-robot-in-store-boot-v1/mission/warmtwo1-fail1-saved/`) ended attempt 1
and logged a **cold boot** between runs instead of the warm re-seat PR 309 was meant to deliver
(`orch-console.log:24107`):

```
WARM_RESET decision=cold_boot cold boot: stage_clean: leaked prims: /CartParkingBrake, /MobileGraspBrake,
/RestockPanelGripper/sim_attach_fallback; missing prims:
/Render/OmniverseKit/HydraTextures/Replicator_01[…], …/Replicator_03[…],
/World/restock_camera_shelf/Camera, /World/restock_camera_source/Camera
```

`config_host_stable` and `vram_returned` passed; only `stage_clean` failed. So every second attempt still
paid the ~215 s cold boot.

## 1. Who creates each leaked prim, who tears down each missing prim, and why the reset missed them

### The ordering bug (why `reset_host_state` never ran)

The between-runs loop read the census **before** the reset, so the reset never got the chance to clean
the stage. `run_mobile_restock.py` `_between` computed the census first, then called
`plan_and_run_reset(…, stage_state=<pre-reset census>)`; `warm_session.plan_and_run_reset` /
`decide_reset` gate `stage_clean` up front and **skip the reset sequence entirely on a cold decision**
(`warm_session.py` `plan_and_run_reset`: "On any cold-boot decision the sequence is NOT run"). The saved
log confirms it: the only `WARM_RESET …` line is the `decision=cold_boot`; **no** `WARM_RESET step=…` or
`WARM_RESET_HOOK …` line appears — the reset hooks were never entered.

### Leaked prims — brakes + weld, created mid-collect, never released on the fail teardown

| Prim | Creator (owner) | Normal removal | Why it leaked |
|---|---|---|---|
| `/CartParkingBrake` | `mobile_runtime.py:176` `engage_cart_brake` (`CART_BRAKE_PATH`) | `release_cart_brake(force=True)` (`:205` `RemovePrim`) | engaged during the grasp press and never released on failure |
| `/MobileGraspBrake` | `mobile_runtime.py:230` `engage_base_full_brake` (`BASE_FULL_BRAKE_PATH`) | `release_base_full_brake` (`:242` `RemovePrim`) | engaged for the pre-weld press and never released on failure |
| `/RestockPanelGripper/sim_attach_fallback` | `cad_tool.py:530`/`:543` `_fallback_weld` (`joint_path = self.path + '/sim_attach_fallback'`) | `_release_fallback` (`:616` `RemovePrim`) on suction-off | the weld held (`SIM_ATTACH_FALLBACK attach body=/RestockScenario/HandlingUnit`, `orch-console.log:8360`) and was never released |

The warmtwo1 collect attempt **failed** (`MANIP_HOST_FAILED op=collect … collision_dt=-0.5000`,
`:21974` → `mission.fail`, `:24094`). The last brake events before teardown are
`BASE_FULL_BRAKE engaged` + `CART_BRAKE engaged` (`:24052–24053`) — both left engaged. The owners all
have release methods, but the normal release points (driving in `MobileController.command`, `attach`, the
suction-off `command(False)`) are never reached on a mid-collect failure, so the prims survived into the
census.

### Missing prims — the station cameras + their RTX render products, torn down for VRAM

Both station cameras and their render products were destroyed by
`native_cameras.destroy_camera_render_product` (`:131`), driven by `sensing_sources.stage_scoped_teardown`
(`:131`) / the VRAM-headroom `secondary_cameras` shed — it calls the render product's `destroy()` **and**
`stage.RemovePrim`s the camera prim + render-product prim (`:169-181`). The saved log pins the two calls:

```
DESTROY_RENDER_PRODUCT {…'rp_paths': ['/Render/OmniverseKit/HydraTextures/Replicator_03'],
                        'prims_removed': ['/World/restock_camera_shelf/Camera']}   # :11330
DESTROY_RENDER_PRODUCT {…'rp_paths': ['/Render/OmniverseKit/HydraTextures/Replicator_01'],
                        'prims_removed': ['/World/restock_camera_source/Camera']}  # :11332
```

So `Replicator_03` ↔ shelf, `Replicator_01` ↔ source; the `…/PostRender/SDGPipeline/*` entries are those
two render products' children. A destroyed camera prim is not re-created on the next `world.step` (the
teardown removes it precisely so it cannot), and a later read raises `CameraRenderProductReleased` — so
nothing restored them and the next attempt would have read expired prims even if the census had let it
start.

## 2. The fix (at the source, before the census runs)

**Reset before census.** `plan_and_run_reset` gained a `census` callback: the non-stage gates
(`config_host_stable`, `vram_returned`, caller doubts) run first with the stage check *deferred*
(`warm_session.STAGE_CHECK_DEFERRED`); a failure there still cold-boots **without touching the stage**.
Otherwise the reset sequence runs, then the census is read and the authoritative `stage_clean` gate is
applied to that **post-reset** state. `_between` now passes the census as a callback (reset-then-census).
The legacy `stage_state=` path is unchanged (byte-identical; all prior tests pass).

**Owners own removal** — `reset_host_state` (`warm_reset_hooks.py`) grew two stage-cleanup sub-steps that
run first, so the stage is clean before the census judges it:
* `release_sim_attachments` — `mobile.release_cart_brake(force=True)` / `release_base_full_brake()` /
  `release_brake()` and `contact.release_sim_attach()` (a new public wrapper on `NativePanelContact`
  over `_release_fallback`). Calling the owner clears its engaged flag too (`_cart_braked`,
  `_base_full_braked`, `_fallback_body`), so a future brake does not no-op on a stale flag and leak again.
* `restore_sensing` — `CameraRig.recreate_destroyed` (new) re-authors the `<root>/Camera` prim + its render
  product for any camera whose render product was released, replacing the released wrapper. Exposed to the
  hook as `sensing_sources['restore_cameras']`.

Each is best-effort by handle presence; the **post-reset census is the authority** and still cold-boots
loudly on any leftover, so a missing owner is never a silent pass.

**Volatile render prims excluded (proven re-created on demand).** The census now excludes the
`/Render/OmniverseKit/HydraTextures/` RTX render-target subtree (`warm_reset_provider._census_prims`,
applied to **both** the baseline and the present set). Those prims are Isaac-managed: allocated when a
camera starts rendering, destroyed to free VRAM, and re-created with a fresh `Replicator_<NN>` index when
the render product re-initializes (which `restore_sensing` does). Matching their churning paths against the
baseline is both brittle and redundant — the **stable** `/World/restock_camera_*/Camera` prim they hang off
IS matched and restored. The exclusion is scoped tightly to that namespace, so a leaked `/World` prim still
fails the census (asserted). **No physics joint is excluded** — the brakes/weld are removed, not tolerated.

## 3. Loud fallback preserved

Every cold branch of the new path still prints the same `WARM_RESET decision=cold_boot <reason>` line (a
refused reset step, a stage still dirty after the reset, or a config/VRAM doubt). Proven by
`test_dirty_stage_the_reset_cannot_clean_still_fires_the_named_cold_boot_line`.

## 4. CPU proof

All CPU-only (no Isaac/ROS/GPU); run with `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=scripts`.

* `scripts/twin/tests/test_sim_warm_reset_stage_clean.py` (8 tests) — a fake stage seeded with the exact
  warmtwo1 dirty state:
  * each leaked brake/weld released at its owner (+ owner flags cleared); each torn-down camera restored;
  * `test_full_provider_reset_then_census_is_clean_on_the_warmtwo1_stage` + `…_warm_reseats_after_the_fixed_reset`
    — the full provider reset then census is clean and `plan_and_run_reset` re-seats (`reused=True`), with the
    `WARM_RESET step=…` lines warmtwo1 never reached and no `decision=cold_boot`;
  * `…_still_fires_the_named_cold_boot_line` — a torn-down non-camera prim the reset can't restore still
    cold-boots loudly;
  * `test_volatile_render_prims_are_excluded_from_the_census` — the HydraTextures subtree is excluded while a
    leaked `/World` prim is still caught;
  * `test_host_key_change_cold_boots_without_running_the_reset` — the non-stage gate runs first.
* `scripts/twin/tests/test_restock_native_cameras.py::test_recreate_destroyed_reauthors_only_the_torn_down_camera_prim`
  — `CameraRig.recreate_destroyed` on a **real in-memory USD stage** (`pytest.importorskip('pxr')`, runs under
  `.venv-usd`) re-authors only the released camera's `Camera` prim and leaves a live camera untouched.
* `scripts/twin/tests/test_restock_warm_reset_hooks.py` — updated for the two new ordered sub-steps; the loud
  fallback + per-step refusal naming still proven.

Full suite green: `test_warm_session.py`, `test_restock_warm_session.py`, `test_restock_warm_reset_provider.py`,
`test_restock_warm_reset_hooks.py`, `test_sim_warm_reset_stage_clean.py`, `test_sensing_sources.py`,
`test_cad_tool_weld_latch.py`, `test_restock_native_cameras.py` — 103 pass (+1 pxr test under `.venv-usd`).

## 5. Hand-off — bounded GPU acceptance (integration lane owns the GPU)

No new config keys. Fragment: `docs/examples/warm-reset-stage-clean-mission-fragment.json`. Run the warm
two-run collect-only as before (`--consecutive-runs 2 --start-at collect --stop-after collect --host-stack`,
`--heavy-threshold-mb 2500` for whispr). **Accept** when attempt 2 logs a warm re-seat, not `cold_boot`:
the between-run console shows `WARM_RESET step=reset_stage_and_reseed` → `WARM_RESET_HOOK step=release_sim_attachments ok`
(`applied=[release_cart_brake, release_base_full_brake, release_brake, release_sim_attach]`) →
`WARM_RESET_HOOK step=restore_sensing ok` (`restored=[…]`) → the arm/latch/heartbeat steps → sequence
complete, and `warm-session.json` run-1 `warm.reused=true`. **Report the warm relaunch time**
(`warm-session.json` `sequence.wall_seconds`) — the hand-off number, a stage re-seat + preflight, not a
~215 s boot.

## Out of scope / follow-ups

The carton-level re-solve, the drive-to-shelf leg and the collision-freshness band are separate lanes.
A `start_at` warm run whose reseat deliberately holds a brake at the start state would need that brake in
the census baseline (today's baseline has none — correct for the collect-only cold-dock acceptance); fold
that in if/when a staged-start warm loop is accepted.
