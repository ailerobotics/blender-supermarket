# Robust arm eye-in-hand masked-depth feed (sim-robust-arm-cam-feed-v1)

Branch `fm/sim-robust-arm-cam-feed-v1` off `fm/twin-robot-in-store-boot-v1`. Goal: the eye-in-hand
masked-depth feed survives a whole transfer and the run stays clear of GPU OOM. Captain's intent
(2026-10-22): "keep working on the simulation on the side to make less prone to failure in any case."

## 1. Root cause — the feed is NEVER killed by OOM; it is torn down by a per-op `stop()`

Every recorded arm masked-depth publisher death — **36/36 `MASKED_DEPTH_SERVE_EXIT` across all
console logs** — is `reason=writer_eof` with `MASKED_DEPTH_BRIDGE_CLOSED ... returncode=0`. Not one
is a signal kill (negative rc = OOM/SIGKILL), an `idle_timeout`, or an `exception`. The publisher
subprocess never crashes on its own.

`writer_eof` means the Isaac-side `FrameWriter` socket closed — i.e. the bridge's `close()` ran. In
both named failures the `close()` caller stack is identical:

- **transfer2** (`transfer2.console.log:3527-3555`):
  `MANIP_HOST_FAILED op=transfer reason='collision planning or execution failed'`
  → `ros_manipulation.py step() → self.stop() → self.sensing_bridge.close()`
  → `MASKED_DEPTH_BRIDGE_CLOSE ... alive=True` → `MASKED_DEPTH_SERVE_EXIT reason=writer_eof published=1747`
  → `BRIDGE_CLOSED returncode=0`.
- **bench1** (`bench1.console.log:1969-1988`): same chain, host reason
  `'base moving or stale/missing feedback/collision world [collision_dt=-1.3333]'`.

So the sequence is: a **per-op transfer manip failure** fires `stop()`, `stop()` closes the
station-level masked-depth feed, the host `incorporation_ack` loses its depth input and stops
advancing its ack stamp, and `StationClient` raises `StationAckStreamStalled: stamp frozen ... for
5.0s (publisher stopped)` — which overwrites `mission.json`'s reason, masking the TRUE stage reason.
This exact pathology is already noted piecemeal in the code (run-347 at `ros_manipulation.py:213`,
and the fm474 `MANIP_HOST_FAILED` print added precisely because the stall masks the reason).

The transfer2 VRAM note (11882/12288 MiB used, **26 MiB free**) is a real adjacent near-OOM risk,
but it is **not** what killed the publisher in any run (rc=0 proves a clean close, not a kill).

## 2. Fix

Three coordinated, default-safe changes:

1. **Decouple the station feed from a per-op failure** (`ros_manipulation.py`).
   `stop(*, close_sensing=True)`. The two per-op failure handlers in `step()` (host `failed`, and the
   exception path) now call `stop(close_sensing=False)`: the gripper/graph is still held and the
   station stopped, but the masked-depth publisher stays alive so the station ack stays fresh and the
   mission reports the TRUE reason instead of `StationAckStreamStalled`. The feed is still torn down
   once, at genuine mission teardown (`MissionRuntime.stop() → manip.stop()`, `close_sensing=True`).

2. **Supervise + restart the bridge** (`masked_depth_ipc.SubprocessSupervisor`, used by
   `ArmDepthSubprocessBridge`). If the publisher ever exits for any reason OTHER than our `close()`
   (a genuine crash / VRAM pressure / broken pipe — defense-in-depth for the near-OOM risk), the next
   `publish()` detects the dead subprocess, respawns it, and reconnects the writer on the same
   per-feed socket — WITHIN the station ack's freshness window (stage-clock seconds; under render
   pacing a sub-second wall respawn is well inside it). Bounded by `max_restarts` (default 20) so a
   hard-crashing publisher can't spin forever. A death from our own `close()` never restarts.

3. **Bound the writer buffers** (`FrameWriter`). Opt-in `send_buffer_bytes` (SO_SNDBUF cap, default
   8 MiB from the bridge) + `send_timeout_s` (10 s) so a stalled/dead publisher cannot back the Isaac
   physics loop up without limit — a blocked `sendall` becomes a prompt error the supervisor restarts
   on. Defaults are `None` (legacy unbounded) so every other caller/test is byte-identical.

All switch keys are backward compatible and default-safe; a config that sets none of the new knobs
runs byte-identically except that a per-op failure now keeps the feed alive (the intended fix).

## 3. GPU memory budget (`twin.restock.gpu_budget`)

A pre-boot, CPU-only gate anchored to the transfer2 measurement (card 12288 MiB; whispr ~2252 MiB
resident and untouchable; one full-res arm feed → 11882 MiB used, 26 MiB free). `check_budget(feeds)`
estimates peak VRAM from the enabled feeds and refuses an over-budget config before Isaac boots:

- the current single arm feed (mast OFF) **fits** (estimate 11882 MiB, matches the measured peak);
- a second full-res feed (the opt-in mast camera_1) **does not fit** (gate returns 1);
- lowering a feed's render resolution cuts the estimate (ties the gate to the real config lever).

CLI: `python3 scripts/twin/restock/gpu_budget.py <mission-config.json>` (reads
`masked_depth`/`mast_depth`), or `--feed WxH` explicitly; exit 0 = fits, 1 = over budget.

Because every recorded death is a lifecycle close (never OOM), the memory fix is: **keep the mast
feed OFF for live_pose transfer (one feed, the gate confirms it fits) and gate any feed-config change
through `gpu_budget` first.** A resolution/rate cut to the arm feed changes ESDF fidelity (what the
mission measures) and needs on-card validation, so it is handed to the integration lane rather than
applied blind here — see `example-config-fragment.json`.

## 4. Tests (CPU, offline)

Run with `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=scripts python3 -m pytest <files>`:

- `test_masked_depth_ipc.py` — `SubprocessSupervisor`: **transfer2 EOF replay** (publisher exits rc=0
  mid-stream → next send respawns + ships, restart-before-stall), broken-pipe retry, no-restart after
  `close()`, bounded restart budget.
- `test_restock_ros_runtime.py` — a per-op host `failed` holds the gripper + stops the station but
  does NOT close the sensing feed; mission teardown `stop()` closes it exactly once.
- `test_gpu_budget.py` — single arm feed fits / two feeds don't / resolution lever cuts the estimate /
  config reader / CLI exit codes.

## 5. Bounded GPU proof — handed to the integration lane

The one bounded GPU proof is the live_pose transfer-only staging run (as transfer2). The GPU is
serialized through `/tmp/aile-gpu.lock` and this lane (twin-robot-in-store-boot-v1) owns the mission
runtime configs (external, under `~/datasets`), so the bounded config fragment + the request to run
the staging proof are handed to the integration lane through firstmate rather than booted here beside
the live Blender session. Expected result: the transfer either completes, or fails with its TRUE
reason — no `StationAckStreamStalled` masking the arm-camera lifecycle.
