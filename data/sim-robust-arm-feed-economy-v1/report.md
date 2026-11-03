# sim-robust-arm-feed-economy-v1 — opt-in render economy for the eye-in-hand masked-depth ARM FEED

**Goal (captain's intent, 2026-10-22):** make the honest `live_pose` cold cycle less failure-prone by
cutting the arm feed's render/pacing load so the simulator paces closer to the still-arm run, without
losing the live-pose quality the transfer needs. Opt-in; default byte-identical.

## 1. Measurement — what the arm feed actually costs, and what paces the sim

Pacing from the two 2026-10-23 capstone runs' `CLOCK_DIAG` lines (`world_current_time` vs `wall`), run
dirs `~/datasets/twin-robot-in-store-boot-v1/mission/{coldlive,coldhonest}`:

| phase | coldlive (live_pose, FAILED @ collect-place) | coldhonest (scripted-still, COMPLETE) |
|-------|----------------------------------------------|----------------------------------------|
| collect manipulation (sim 0–55) | ~0.073–0.085 sim/wall (≈12–13×) | ~0.073–0.085 sim/wall (≈12–13×) |
| nav (drive→rollwagen) | ~0.11–0.13 sim/wall (≈8–9×) | ~0.11–0.13 sim/wall |
| collect place / transfer region | ~0.10–0.11 | ~0.10–0.12 |
| whole-run average | 0.097 sim/wall | 0.121 sim/wall |

**Key correction to the brief's premise.** coldhonest did **NOT** run without the arm feed — its console
has 698 `BRIDGE_STATS` lines (masked_depth enabled, `mission-runtime-cold.json`). The real difference:
coldhonest ran `SUPERMARKET_DEV_RENDER_ECONOMY='record=20,sensing=8'` (3× fewer render ticks); coldlive
ran at the production `record_stride=sensing_stride=6` with no economy env. **So the proven pacing lever
is the shared render cadence, and the arm feed's mere presence is not what separates the completing run
from the failing one.** During the phases both runs share (collect), the arm feed is present in BOTH and
they pace identically (~0.08) — consistent with fm373 ("render resolution saves only ~160 MiB"; the arm
product is small, 424×240).

**Render model (from `sim_speed.py` + `isaac_port.advance`).** A `world.step(render=True)` tick costs
~0.66 s wall, a non-render tick ~0.04 s. During collect/transfer, renders fire on the
`sensing_stride`/`record_stride` cadence; the arm camera render product (424×240 depth + instance-id seg)
piggybacks on those render ticks (it does **not** force its own renders — only the opt-in mast does, via
`_masked_depth_due`). The arm bridge's `publish()` runs every manipulation tick (`ros_manipulation.step`),
each doing a 424×240 rig-mask + `camera_info` + bbox + ~400 KB socket send + nvblox integration.

**Feed parameters exposed:** resolution (camera render product), publish rate (per-tick today), rig/carton
masking (instance-id), domain/topics, and the advisory `gpu_budget` gate (PR 206, `feeds_from_config` reads
`resolution`). The solver consumes the masked depth → nvblox ESDF (`voxel_size 0.02`) + the camera `K`;
its spatial resolution is the voxel, so a feed is "good enough" when its reprojected surface stays within
half a voxel at the working distances (collect ~1.2 m, wrist ~0.32 m).

## 2. Deliverable — `masked_depth.economy` (opt-in, default byte-identical)

`scripts/twin/restock/masked_depth_economy.py` (pure, CPU-tested) + wiring:

- **`resolution_scale`** (`0<s≤1`): `isaac_factory` renders the arm camera at `scaled_resolution`; apertures
  unchanged → `fx/fy/cx/cy` and the published `camera_info` scale with it → FOV/reprojection preserved. The
  only cost is pixel quantization, bounded `0.5·Z/fx_scaled` (`reprojection_shift_m`), held under half a
  voxel (0.01 m) at both recorded distances for `s ≥ 0.25` (proved in the CPU test on synthetic carton-face
  fixtures). This is the GPU-render lever (fewer pixels + smaller annotator readback + smaller IPC payload).
- **`publish_interval_s`** (SIM s): `ArmDepthSubprocessBridge.publish` skips on the SIM clock
  (`due_to_publish`) until the interval elapses — never ships a stale buffer, TF still streams every tick —
  cutting the per-tick rig-mask + IPC + integration from ~60 Hz sim to `1/interval` Hz. Validated **below**
  the 0.5 s nvblox window so the ESDF never lapses (freshness-on-sim-clock, PR 201/204/205/226/228).

Both levers are composable, validated at boot (fail-closed on a typo), logged (`MASKED_DEPTH_ECONOMY …`),
and reflected in the pre-boot VRAM gate. Mast (camera_1) reuses the block's rate cap. **Absent/empty/
`{"enabled": false}` ⇒ inactive ⇒ native-resolution camera + publish-every-tick ⇒ byte-identical.** The
dominant shared render-cadence lever (`record_stride`/`sensing_stride`/`sim_speed`) is separate and
composable — stack them for the biggest pacing win.

Example: `docs/examples/masked-depth-economy-mission-fragment.json` (`resolution_scale 0.5`,
`publish_interval_s 0.1`). CPU tests: `scripts/twin/tests/test_masked_depth_economy.py` (34),
`test_isaac_factory.py` (parse wiring), `test_gpu_budget.py` (scale in the gate).

## 3. Honesty / scope

The economy reduces the arm feed's OWN render + per-tick + IPC + integration load. Because the arm product
is small and piggybacks on the shared render ticks, the measured pacing benefit of `resolution_scale` alone
is modest; `publish_interval_s` cuts real per-tick CPU on the mission's critical loop. The **measured**
pacing delta under the live_pose load is for the integration lane's one bounded GPU proof to record (the
armcam staging with this block on; PASS = live_pose transfer completes through verify with the pacing
improvement recorded). This lane ships the capability + example + CPU proof; it does not take the GPU.
