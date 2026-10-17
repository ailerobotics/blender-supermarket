# sim-restock-armbench-v1 — arm-carry bench + carry-fold diagnosis

_Worker lane `sim-restock-armbench-v1`, branch `fm/sim-restock-armbench-v1` (cut from
`origin/fm/twin-robot-in-store-boot-v1`). 2026-10-17 → 2026-10-18._

## TL;DR

A boot-once **arm-carry bench** was built (merged: PR #113, PR #116; follow-up PR pending) that
boots Isaac once, welds the carton at the tool via the mission's own `SIM_ATTACH_FALLBACK`, brakes
the base, and runs **many carry-fold trials in seconds each** driving the real `ArmCommandGate` +
the real `trajectory_host` Plan/tracking/settle gates. Using it (plus the integration lane's CPU
sphere model) the carry-fold failure was diagnosed and a fix validated:

1. **The fold stall is NOT a drive-gain tradeoff** (the initial read). It is **gain-, drive-type-
   and carton-weld-independent**: the bench reproduces run-250's exact signature — wrist_1 pinned at
   **0.6965 rad** while shoulder/elbow/wrist_3 track — because the **carried carton collides with the
   robot's own `ur_upper_arm_link`** as wrist_1 rotates toward the tuck goal (2.1066), and **cuMotion
   does not self-collision-check the attached carton against the arm**, so it plans that colliding
   tuck "collision-free" and PhysX/the drive then cannot complete it.
2. **The reshape fixes the collision.** CPU carton-vs-`ur_upper_arm_link` clearance along the fold:
   old (wrist1=2.1066) **−23 mm (COLLIDES)**; reshaped (wrist1=0.9) **+180 mm (clears)**.
3. **After the reshape, a residual DRIVE oscillation remains** at the terminal settle (run-254): the
   stiff drive **1e6/1e4/4000 rings wrist_1 at 3.14 rad/s** in the now-collision-free pose. The bench
   sweep shows softer gains damp it below the 0.70 rad/s goal watchdog: **1e5/2000/400 → 0.48 rad/s**,
   **kp3e5/kd8000/eff400 → 0.337 rad/s**.

**Recommendation:** carry shape `cart_tuck_shape` wrist_1 = **0.9** (the collision fix; owned by the
integration lane) **+** the carry drive **arm_stiffness_abs 1e5 / arm_damping_abs 2000 /
arm_max_effort 400** (damps the terminal ring; `kp3e5/kd8000/eff400` if more velocity margin is
wanted). This is what integration's run-255 is validating.

## The bench

- `scripts/twin/tools/arm_carry_bench.py` — pure, CPU-tested: analytic rest-to-rest carry trajectory,
  an executor loop that **reuses `trajectory_host`'s real Plan + `tracking_fault` + terminal settle
  gate** (imported, not copied), per-move metrics (per-joint peak tracking error, settle time, peak
  velocity, carton drift, base displacement, carton–obstacle clearance, first-contact prim/fold-
  fraction/wrist1), drive-type (FORCE/ACCELERATION) + UR10 force-unit gains, the sweep grid and the
  robustness ranking.
- `scripts/run_arm_carry_bench.py` + `run_arm_carry_bench.sh` — guarded GPU runner + `IsaacCarryStage`
  glue: assemble, weld (carton at `pad + R_tcp·(0,0,0.20)`, the `carton_spheres_world` model), yaw-
  brake, drive the real gate, per-trial gain/drive-type reset, incremental result writes, blocking-
  `flock` (`--no-gpu-lock`), fastrtps shm hygiene, `--cart-tuck-shape` / `--start-at-lift` / `--no-carton`.
- `scripts/twin/tools/carton_arm_fold_clearance.py` — **CPU (no GPU)** carton-vs-arm-link min
  clearance along the fold, reusing `twin.restock_host.world_obstacles` sphere primitives.
- `scripts/twin/tests/test_arm_carry_bench.py` — 23 CPU tests (no Isaac/ROS/GPU).

## What is exercised identically vs what differs (honest)

Identical: the real Isaac arm position drive (gains via the same `articulation_controller` path
`physics_setup` uses), the real `ArmCommandGate.command` envelope, the real `trajectory_host` Plan +
`tracking_fault` + terminal settle gate + tolerances, the carton welded via the mission's own
`SIM_ATTACH_FALLBACK`, the airborne-carry yaw-brake.

Differs: **planning path** — carry trajectories are generated analytically (rest-to-rest per joint
goal), not by MoveIt/cuMotion; **timing** — the executor loop runs in-process on the sim clock, not
over the ROS relay. Both are acceptable here because the stall is a *geometry/terminal-settle*
property both paths share (the bench reproduces run-250's wrist_1=0.6965 pin and run-254's 3.14 rad/s
ring). One artifact: the straight-line grasp→lift→tuck swings the carton lower than cuMotion's path,
so a long-fold run adds a spurious carton-vs-floor contact; `--start-at-lift` isolates the terminal
settle to avoid it, and the CPU clearance tool is path-controlled.

## Evidence

### Carton-vs-`ur_upper_arm_link` clearance along the fold (CPU, faithful sphere model)
```
PYTHONPATH=scripts python3 -m twin.tools.carton_arm_fold_clearance \
  --urdf …/host-cad/mobile.urdf --xrdf …/host-cad/mobile.xrdf \
  --lift 1.4051,-1.0499,0.894,-1.4149,-1.5708,-0.1657 \
  --tuck-shape old=-2.3,1.0,2.1066,0.8,-1.2545 --tuck-shape new=-2.3,1.0,0.9,0.8,-1.2545
```
| fold | wrist1 goal | min clearance to ur_upper_arm_link | verdict |
|---|---|---|---|
| old | 2.1066 | **−23.1 mm** | COLLIDES |
| new (reshaped) | 0.9 | **+180.0 mm** | clears |

### Drive-oscillation sweep at the reshaped collision-free tuck (GPU, carton welded)
`run_arm_carry_bench.sh … --cart-tuck-shape -2.3 1.0 0.9 0.8 -1.2545 --diagnose --sweep …`
| drive gains (accel) | terminal residual velocity | note |
|---|---|---|
| 1e6 / 1e4 / 4000 (stiff) | **3.14 rad/s** | reproduces run-254/248 ring |
| 1e5 / 2000 / 400 (soft) | **0.48 rad/s** | under the 0.70 goal watchdog |
| 3e5 / 8000 / 400 | **0.337 rad/s** | lowest residual |
| 1e5 / 8000 / 400 | 0.98 rad/s | |
| 1e5 / 20000 / 400 | 1.08 rad/s | damping non-monotonic (accel drive is mass-normalised) |
| 3e5 / 20000 / 400 | 1.02 rad/s | |

(The long-fold sweep also showed a ~0.198 rad wrist_1 position miss from the straight-line carton-vs-
floor artifact; a `--start-at-lift` clean terminal-settle confirmation is queued behind run-255.)

## Run history (all this lane's GPU sessions respected the lock + VRAM guard; carton-only PhysX
contact so the articulation tensor view / get_gains stays valid)

diag/diag2/diag3 — validated boot/weld/brake/fold/contact/clearance/reset + reproduced the wrist_1
pin. pinned-config probe at run-250's jam config. drive comparison (no-carton: wrist_1 pins even
empty → geometry, not drive). drivecmp (accel vs force same gains). settle2 (the sweep above).

## Status / ownership

The carry-shape reshape (wrist1=0.9) is owned by the integration lane (it owns the mission carry);
this lane recommends it with the clearance evidence and recommends the damped carry drive. The robust
fix for the class of bug (cuMotion self-collision-checking the attached carton) is the
planner-obstacles lane's. The bench is reusable for the downstream lane's pan/place moves
(`extra_goals`, `--cart-tuck-shape`, `--start-at-lift`).
