# Restock host: cold first-job latency vs the IsaacRosGraph 5 s watchdog

Branch `fm/sim-restock-first-job-latency-v1` from `origin/main` 2005d37 (PR #50).

## Why

Integration run-41 reaches `collect-execute` with real arm masked depth on the host, and
the current gate is whether the host planner answers the first manip job inside the
IsaacRosGraph **5 s process/status watchdog** window. A cold first job has to (1) build an
ESDF from depth that has only just started flowing and (2) return a cuMotion plan. This
measures both latencies on **this GPU with whispr resident** and compares the cold total
to the 5 s budget.

## How it was measured

`scripts/restock_runtime/stack_probe.py`, run under the bounded guarded GPU harness
(`scripts/verify_restock_host_stack_gpu.py`), now times two cold-start latencies against the
**real** `object_segmenter → nvblox → cuMotion` host path:

- **first ESDF ready after depth starts** — from a genuinely cold mapper (baseline ESDF
  query returns **0** observed voxels), begin publishing masked depth at 10 Hz and poll the
  `get_esdf_and_gradient` service (which drives `update_esdf=True`, exactly as the cuMotion
  world query does) every 0.3 s until it first reports observed voxels. Elapsed from the
  first depth frame to that first non-empty read is the latency.
- **first cuMotion plan reply after a manip job request** — wall time from sending the
  `MoveGroup` (plan-only) job to receiving the plan result. The cuMotion node queries the
  nvblox ESDF as part of answering, so this includes its own world-sync cost.
- **cold first-job total** — the conservative worst case where the manip request lands
  before any depth has integrated, so the host must build the ESDF *and then* plan before
  replying: `first_esdf_ready + first_cumotion_reply`.

The masked-depth frame is the synthetic Option-B fixture (a wide-FOV down-looking camera
over a flat floor). It stands in for integration's live Isaac bridge: the ESDF-integrate and
cuMotion-solve **host compute** time — what the watchdog gates — is driven by the stack on
this GPU, not by whether the depth pixels are synthetic or live.

## Result — within the 5 s watchdog (two reproducible cold runs)

| latency | run 1 | run 2 |
|---|---|---|
| first ESDF ready after depth starts | **1.038 s** | **1.038 s** |
| first cuMotion plan reply after manip job | **2.260 s** | **2.259 s** |
| **cold first-job total (ESDF build + plan)** | **3.298 s** | **3.297 s** |
| IsaacRosGraph watchdog budget | 5.0 s | 5.0 s |
| within watchdog | ✅ (~1.7 s margin) | ✅ (~1.7 s margin) |

Both runs also PASS the full stack verification: 6 nodes load with no failure,
`compute_ik` SUCCESS, ESDF 0 → 2,546,419 observed voxels, cuMotion returns a 16-point
trajectory (`success: true`). Stack VRAM 3.0–3.4 GB; whispr resident 2258 MiB.
Outputs: `/home/manishsaini/datasets/sim-restock-host-v2/latency-measure-{1,2}/`.

**No host-side change is required.** The cold first job answers in ~3.3 s, ~1.7 s inside
the 5 s window. No warm-up/preload, planner-parameter, or ESDF-cadence change is warranted.

## Honest caveat — renderer GPU-compute contention

These numbers are measured with **whispr resident** (as asked) but with the **Isaac
renderer NOT co-resident**: the guarded probe launches only the host stack, not the full
Isaac boot. In the live mission the renderer (~6.6 GB, active GPU compute) runs alongside
the stack and will contend for GPU compute, which can lengthen both nvblox ESDF integration
and the cuMotion solve. So **3.3 s is a lower bound** on the real cold first-job latency;
the ~1.7 s margin to the 5 s watchdog is what absorbs that contention. Only a full-boot
measurement (integration's `run_transport_probe.sh`, which boots Isaac + renderer) captures
the contended figure. If the integration-side contended measurement comes back above ~5 s,
the host has headroom to shave the cuMotion reply (the 2.3 s dominant term) via a warm-up
plan at launch — flagged, not pre-emptively applied.
