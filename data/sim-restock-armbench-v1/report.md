# sim-restock-armbench-v1 — arm-carry bench

_Worker lane `sim-restock-armbench-v1`, branch `fm/sim-restock-armbench-v1` (cut from
`origin/fm/twin-robot-in-store-boot-v1` @ 08e7e0d). As of 2026-10-17._

## Purpose

A boot-once bench that proves the carton-carrying arm MOVES (tuck fold → pan → …) in
**seconds per trial** instead of a 15-minute mission run, and diagnoses/sweeps the carry
fold so the integration lane can fix it. After the 2026-10-17 redirect (inbox 003), the
primary question is **contact**, not drive gains: does the carried carton swing into an
unmapped static obstacle (the cart cage) during the lift→tuck fold?

## What the bench exercises identically vs what differs (honest)

Identical to the live mission: the real Isaac arm position drive (gains via the same
`articulation_controller.set_gains`/`set_max_efforts` path `physics_setup` uses), the real
`ArmCommandGate.command` envelope, the real `trajectory_host` Plan + `tracking_fault` +
terminal settle gate (imported, not copied), the carton welded at the tool through the
mission's own `SIM_ATTACH_FALLBACK` fixed joint, and the base in the airborne-carry yaw-lock
brake.

Differs (stated with every result): **planning path** — carry trajectories are generated
analytically (rest-to-rest cubic per joint goal) rather than by MoveIt/cuMotion; the
joint-space goals, carton load and drive are the mission's. **Timing** — the executor loop
runs in-process on the sim clock, not over the ROS relay round-trip (sim-clock tolerances,
which decide pass/fail, are preserved). **Clearance** is carton-AABB vs obstacle-AABB
(over-approximate); **contact** is PhysX (hollow-geometry-aware) when available — the two
together settle the "hollow cage" question.

## Files

- `scripts/twin/tools/arm_carry_bench.py` — pure, CPU-tested logic: analytic carry
  trajectory, the executor loop (reusing `trajectory_host`), per-move metrics (per-joint
  peak tracking error, settle time, peak velocity, carton drift, base displacement,
  **min clearance + first contact prim/fold-fraction/wrist1**), the sweep grid and the
  robustness ranking (robust ⇒ settles every move AND no contact, ranked by clearance).
- `scripts/run_arm_carry_bench.py` — guarded GPU runner + `IsaacCarryStage` glue (weld,
  brake, clearance, PhysX contact reporting, per-trial reset).
- `scripts/twin/tests/test_arm_carry_bench.py` — 17 CPU tests (no Isaac/ROS/GPU).

## Commands

CPU tests (no GPU):
```
cd <worktree>
PYTHONPATH=scripts PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest scripts/twin/tests/test_arm_carry_bench.py -q
```

GPU diagnosis (trial-1; under `/tmp/aile-gpu.lock`, short session, integration lane has
priority). Runtime root / plugin env / lock etiquette per the integration handoff §(d):
```
RESTOCK_PYTHON=/home/manishsaini/miniforge3/envs/env_isaaclab/bin/python \
RESTOCK_NVIDIA_RUNTIME=/home/manishsaini/datasets/twin-robot-in-store-boot-v1/gpu-runtime-578/root \
  scripts/run_mobile_restock.sh --execute   # (shape reference; the bench is launched directly:)

RESTOCK_PYTHON=/home/manishsaini/miniforge3/envs/env_isaaclab/bin/python \
  <env pins from run_mobile_restock.sh --execute> \
  "$RESTOCK_PYTHON" scripts/run_arm_carry_bench.py \
    --runtime-config <mission-runtime-cc-collect.json> \
    --output <scratch>/armbench-diag --diagnose --expect reproduce \
    --heavy-threshold-mb 2500 --lift-joints <6 measured lift joints>
```

Carry-pose sweep (ranked by cart/rack clearance):
```
  … scripts/run_arm_carry_bench.py --runtime-config <cfg> --output <scratch>/armbench-pose \
    --pose-sweep --lift-joints <6>
```

## Diagnosis (run-248/run-249 analysis, pre-GPU)

The mission fold stalls on **wrist_1** (ARM_JOINTS index 3) at BOTH gain extremes:
- run-248 (Kp 1e6 / Kd 1e4 / effort 4000): `GOAL_MISS` wrist_1 1.23 rad off, joints still
  2.76/3.4 rad/s after the 20 s settle window.
- run-249 (Kp 1e5 / Kd 2000 / effort 400): `TRACK_MISS` wrist_1 stuck at 0.71 vs commanded
  2.0, error 1.306 > tol 1.3.

Per the integration lane's env-aware CPU check (inbox 003), this is **gain-independent** —
a rigid contact of the carried carton against the unmapped cart cage, not a drive limit
cycle. The bench confirms this in the full scene.

## Results

_(pending the guarded GPU session — trial-1 diagnosis then the carry-pose sweep)_

| trial | outcome | wrist1_track | min_clear_mm | hit prim | note |
|---|---|---|---|---|---|
| _pending_ | | | | | |

## Recommendation

_(pending — a carry shape that clears the cage + rack with margin, for the integration lane
which owns the carry reshaping in the mission)_
