# Host-published nvblox mapper-incorporation acknowledgement (gate 7 / collision_time)

Branch `fm/sim-restock-incorporation-ack-v1` from `origin/main` 2005d37.

## Why

The integration mission (run-52, PR #15) clears six sensing gates live and stops before
the grasp at gate 7: the sensing motion runs through `IsaacMeasuredFeedback`, which needs
`collision_time` — the nvblox mapper-incorporation freshness stamp
(`0 <= now - collision_time <= 0.5`) — synced from a **mapper-incorporation acknowledgement
the host never published**, so the first manip job has no fresh ESDF stamp and dies on the
Isaac process/status watchdog.

The gap is specific to the mission's `--host-stack` path: it launches only the 6-node
planner/mapper stack (`launch_restock_host_stack.sh` → `run_restock_host.py --launch`),
not the full station supervisor, so nothing published `/restock_sim/station/status`. The
mission-side `StationClient` therefore never saw a reply and held `collision_time` at `-1`.

## What (host side only; the integration side mirrors it)

A new host node `twin.restock_host.incorporation_ack` (`restock_incorporation_ack`),
co-launched with the stack in `twin.restock_host.launch`:

- speaks the parked-station command protocol on `/restock_sim/station/command`
  (prepare / heartbeat / stop; epoch + sequence validated) — the existing wire contract the
  mission's `StationClient` already drives;
- polls the live nvblox `get_esdf_and_gradient` service. nvblox runs
  `update_esdf_rate_hz: 0.0` (on-request only), so this poll is what integrates pending
  masked depth and reads NVIDIA's real incorporation stamp;
- advances `collision_time` **only** through `preflight.incorporated_time` (rejects failed,
  empty, nonfinite, stale, future, wrong-frame and pre-station responses) — warm-up / empty
  / stale responses never advance it, and input receipt is never an acknowledgement;
- publishes the snapshot (`{state, stamp, collision_time, epoch, parked_since,
  camera_frame}`, `std_msgs/String` JSON) on `/restock_sim/station/status`.

It launches **no planner child** (the stack is already up — unlike the full station
supervisor), and it is deliberately **not** a launch shutdown guard (its loss must not fell
the planner); the core guards do tear it down with the stack, so a live ack implies a live
mapper (hence its snapshot reports `process_alive`). The acknowledgement state machine is a
self-contained copy of the parked-mapper lifecycle contract vendored into this lane (the
integration lane keeps the mirror-side copy; the two agree on the JSON wire fields, not a
shared import — the integration lane is unmerged, so it cannot be imported from main).

Node readiness excludes the external `arm_controller/follow_joint_trajectory` (provided
outside this launch): the ESDF-freshness ack depends on the planner/mapper graph, not the
downstream arm controller.

## Proof — bounded guarded GPU verification, PASS

`scripts/verify_restock_host_stack_gpu.py` launches the task-isolated stack (now including
the ack node) and runs `stack_probe.py`, which drives the station command protocol (prepare
+ heartbeats, waiting for the subscription to match so prepare lands reliably), publishes
synthetic masked depth on the agreed contract, and reads `/restock_sim/station/status`.
Output: `/home/manishsaini/datasets/sim-restock-host-v2/incorporation-ack-verify-6/`.

| check | result |
|---|---|
| all 6 core nodes load, no failure | ✅ |
| `compute_ik` round-trip | ✅ error_code 1 (SUCCESS) |
| ESDF non-empty | ✅ 0 → 2,546,419 observed voxels |
| cuMotion plan | ✅ 16-point trajectory |
| **ack becomes ready after first integration** | ✅ `first_ready_s` **0.42 s** after prepare |
| **collision_time is fresh** (`0 <= stamp - collision_time <= 0.5`) | ✅ (node emits `ready` only when fresh) |
| **collision_time is the integrated frame stamp** (matches a published depth stamp, never a fabricated `now`) | ✅ |
| **ack fires after each integration** (collision_time advances) | ✅ distinct stamps `8.54, 8.82, 9.18, 9.42, 9.72, 9.98, 10.22, 10.48, …` |
| warm-up / pre-station responses rejected | ✅ (e.g. a `resp_stamp 5.48 < parked_since 8.14` response correctly refused) |
| GPU fit beside renderer + whispr | ✅ stack ~3.4 GB, headroom ≥ 0 |

`fires_after_integration_with_right_stamp = True`. CPU: `test_restock_host.py` 22 passed /
1 skipped (6 new acknowledgement-state tests + 1 node-allowlist test).

## Scope / honest limits

This proves the host publishes a fresh, correctly-stamped incorporation ack driven by real
nvblox integration, against synthetic masked depth of the agreed type. The mission still
owns the mirror onto `IsaacMeasuredFeedback` and the live masked-depth / measured
joint_states / per-station ESDF; a correct instance-ID self mask remains integration's to
certify. Integration enables its mirror on this and reruns `collect` to the grasp.
