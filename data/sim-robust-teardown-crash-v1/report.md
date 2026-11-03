# sim-robust-teardown-crash-v1: record-first + ordered teardown

**Goal.** A mission run that ends on a per-op manipulation failure must still leave its
`mission.json` and exit without a core dump. Tonight's validator run `armcam` (branch at PR 212)
failed a live_pose transfer (`MANIP_HOST_FAILED op=transfer`), then the Isaac process core-dumped
and wrote **no** `mission.json` — the run's structured outcome was lost, leaving only the console
and `host-stack.log`.

## 1. What actually happened (CPU investigation, no GPU)

Evidence: `~/datasets/twin-robot-in-store-boot-v1/mission/armcam.console.log`, the crash-time
py-spy dump `…/isaacsim/kit/data/Kit/Isaac-Sim Python/5.1/2c4e6661-….py.txt`, and
`~/firstmate/data/twin-robot-in-store-boot-v1/armcam2-pr212-evidence.md`.

The brief described the crash as "at teardown". The py-spy dump **corrects** that — the SIGSEGV was
**mid-tick, inside a render `world.step`**, not in `SimulationApp.close()`:

```
run_once (run_mobile_restock.py) -> run_once_bt -> run_mission_tree (bt/driver.py)
  -> py_trees tick -> leaves.pump -> mission.step_stage
    -> runtime.step -> isaac_port.advance -> world.step -> simulation_context.step  <-- CRASH
```

Sequence on the console:
1. `MANIP_HOST_FAILED op=transfer phase='failed' reason='collision planning or execution failed'`
   (the live_pose pick c-space target is in world collision — that root cause is
   `sim-robust-livepose-pick-collision-v1` / PR 215's lane, not this one).
2. The mission did **not** terminate on that per-op failure — the BT transfer subtree kept
   ticking (grasp re-press: `BASE_FULL_BRAKE engaged for pre-weld grasp press`, more `SEAL_DIAG`),
   and every tick kept calling `world.step(render=True)`.
3. VRAM climbed to the known 12 GB ceiling (`[carb.graphics-vulkan.plugin] … VkResult:
   ERROR_OUT_OF_DEVICE_MEMORY`, repeated `HydraEngine::render failed to end the compute graph`),
   and a render `world.step` finally SIGSEGV'd (`rtx.raytracing` / `omni.hydra`).
4. The native crash killed the process **before** any Python terminal / `run_once_bt` return, so
   neither `run_once_bt`'s early write nor the launcher's outer `finally` ran — hence no
   `mission.json`.

**Shutdown order, failure vs success.** On a *clean* terminal failure (`Mission.fail` /
`run_once` returns a failed summary) the launcher already writes `mission.json` early (sequential:
`run_mobile_restock.py` ~L521; BT: `bt/integration.py` L130) **before** teardown. The gap was the
**exception path**: an exception out of `run_once` wrote only `mission-error.txt`, and the only
`mission.json` write was the *outer* `finally` — which runs **after** `app.close()`. If
`app.close()` hard-exits, that write never happens. The teardown order itself (adapter.stop →
bundle.close → encode walkthrough → app.close) was the same on both paths, but **nothing released
the RTX render products before the app close** (the station decode cameras alone are the dominant
2560×1920 buffers).

## 2. Fix (two layers; switch-key compatible; success runs byte-identical apart from the earlier write)

New pure, CPU-only module `scripts/twin/restock/shutdown.py`:

- **(a) Record first — `write_mission_record(output, summary, events)`.** Called on *every* exit
  path **before any Isaac/ROS teardown begins**: in the launcher's middle `except` (before re-raise)
  and at the top of the middle `finally`. Writes a copy of `summary` with `events` merged (never
  mutates the live `summary`), never raises. So a run that reaches a Python terminal/exception now
  leaves `mission.json` even if `app.close()` later core-dumps. Clean runs still re-write the fuller
  record (events + smart-verify) in the outer `finally`, so their final `mission.json` is
  byte-identical.
- **(b) Ordered teardown — `ordered_teardown(steps)`** + `Bundle.release_render()`. The teardown
  now runs `close_feeds → encode_walkthrough → release_render → close_app`, each best-effort (a
  raising step never skips a later one, and never overrides the mission outcome). `release_render()`
  pauses every station/location camera's RTX render product (reusing the existing best-effort
  `native_cameras.set_camera_render_products_enabled`) right before `app.close()`, so the renderer
  shuts down with the VRAM headroom a clean exit has. Camera *feeds* (the masked-depth ROS bridges,
  recorder) are stopped in `close_feeds` as before, ahead of the app close.

`_mission_events(db, scenario)` reads the inventory event log robustly for the record-first write
(degrades to `[]` rather than raising at teardown).

### Honest scope boundary

The record-first + ordered-teardown fix fully covers the **teardown-crash class**: any per-op
failure that terminates at the Python level (or raises `simulation app stopped`) now persists the
record before teardown and releases render products before the app close. It does **not**, by
itself, save the *specific* `armcam` run, because that crash was a **native SIGSEGV mid-`world.step`
under VRAM OOM before any Python terminal** — no Python `finally` can run after a native crash. For
that run to end on the per-op failure (and thus benefit here), the failure must terminate the
mission before VRAM OOM, which belongs to:
- `sim-robust-livepose-pick-collision-v1` (PR 215): make the live_pose transfer pick not fail /
  fail fast rather than re-press under retry; and
- the 12 GB VRAM-ceiling lane (`mission-vram-ceiling-12gb`): the run must not keep render-stepping
  up to OOM.

Recommended follow-up (not done here, to stay in lane and avoid GPU-coupled, untestable code): a
bounded post-per-op-failure abort budget so a stalled retry terminates the mission cleanly instead
of render-stepping to OOM.

## 3. CPU tests

`scripts/twin/tests/test_restock_shutdown.py` (7 tests, `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
PYTHONPATH=scripts`):
- `write_mission_record` merges events without mutating summary; never raises on a bad record.
- `ordered_teardown` runs in order, skips `None`, and a raising step never skips the app close.
- render products + feeds are released before the app close (the shutdown-sequence invariant).
- composed as the launcher does it: a failing mission leaves `mission.json` **before** a
  core-dumping `app.close()` is attempted (SystemExit proxy).
- `Bundle.release_render()` is best-effort and never raises; `_render_closers` collects one closer
  per station/location camera and is a no-op on a headless/partial assembly.

## 4. Bounded GPU proof (for the integration lane)

`docs/examples/teardown-crash-proof-fragment.json`. Hand it to `twin-robot-in-store-boot-v1` for one
bounded run through `/tmp/aile-gpu.lock` (never forced; never touch whispr). **PASS** = a bounded
staging that ends on a per-op failure now writes `mission.json` and exits with code 0 or a named
non-zero, **no core dump**; the console shows `release_render` ran before the app close. Report
PASS/FAIL with the run dir.
