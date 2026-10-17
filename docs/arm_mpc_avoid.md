# Reactive arm MPC follow-and-avoid (evaluation-only)

cuRobo V2 (Apache-2.0) Model-Predictive-Control reactive follow-and-avoid for the
mission UR10, **benched** against the store twin's colliders and a moving-person
proxy. This is the store-twin realisation of the fleet's `plan-mpc-curobo-v1`
prototype (the cell UR10); it measures the same thing for the mission arm.

## What this is — and is not

- **Evaluation-grade, simulation-only.** It produces the measured evidence the
  owner-held cuRobo eval-vs-ship decision needs
  (`firstmate/data/plan-mpc-curobo-v1/decision.md`). It does not pre-empt that
  decision and is **not built into the mission chain**.
- **MPC collision avoidance is a SOFT optimiser cost, NOT a certified safety
  function.** The prototype measured a fast person transiently intruded by ~3 cm.
  The single motion authority remains the **merged human-safety gate**
  (RUN/SLOW/STOP/BACKSTOP, fail-toward-stop) and the **measured joint-window guard**
  — both reused *verbatim* here (`scripts/twin/tools/arm_mpc_bench.py:load_gate`),
  never re-implemented.
- **Licence.** Pinned to cuRobo **V2 (≥ v0.8.0, Apache-2.0)**, which still exposes
  MPC. A `v0.7.x` tag (NVIDIA non-commercial) is never used. The reused container
  `fm-plan-mpc-curobo-v1` builds V2 for exactly this reason.

## Opt-in and default-off (chain byte-identical when absent)

MPC lives entirely in an opt-in bench and is **never imported by the mission
compose/boot path**, so a mission run contains zero MPC code and is byte-identical
whether or not MPC exists.

- `compose_robot_in_store.py --arm-mpc` (default off) only **annotates the dry-run
  plan** with an evaluation-only note. It does not change the composed stage or the
  cuMotion arm path (`scripts/twin/tests/test_cli.py` asserts the rendered plan is
  byte-identical when the flag is absent or `False`).
- The measured evidence comes from the bench runner, not the mission:
  `scripts/run_arm_mpc_bench.py`.

## Architecture

| Piece | File | Needs |
|---|---|---|
| Pure core: store world, scenarios, clearance, gate trial, metrics | `scripts/twin/tools/arm_mpc_bench.py` | stdlib only (CPU-testable) |
| cuRobo MPC solver loop (drives the mission arm) | `scripts/twin/tools/arm_mpc_run.py` | cuRobo V2 + torch + GPU (container) |
| Guarded GPU runner (lock + VRAM + Blender guard + container) | `scripts/run_arm_mpc_bench.py` | docker, the GPU lock |
| CPU logic tests | `scripts/twin/tests/test_arm_mpc_bench.py` | stdlib + the reused gate |

The pure core is imported by both the CPU tests and the GPU loop, so the world,
scenarios and metrics are identical offline and on-GPU. The only GPU/cuRobo-dependent
code is the solver loop.

### Reuse, not rebuild

- **Gate + window guard**: imported from the `plan-mpc-curobo-v1` prototype's
  `ros_bridge` (`~/ur10/mpc_curobo`, mounted at `/mpc` in the container; override with
  `MPC_CUROBO_DIR`). `GatedStreamController` and `JointWindows` are pure stdlib, so the
  gate trial runs on CPU too. The window guard is re-expressed under the mission's
  `ur_`-prefixed joint names from the arm's own URDF limits
  (`cad_tool.inspect_tool(urdf).arm_limits`) — the same measured envelope the
  trajectory host enforces.
- **Robot config**: the mission arm's own `shelfbot_curobo.yml`
  (`~/workspaces/shelfbot_newgripper/description`, mounted at
  `/ros_ws/src/shelfbot`) — the same collision spheres / joint windows the mission
  cuMotion path uses. The tool frame is `platform_tcp`.
- **Container**: `fm-plan-mpc-curobo-v1:latest` (scout build spec (a): do not rebuild
  the arm stack). No Isaac Sim boots — cuRobo MPC is pure cuRobo + torch, the
  cheapest GPU consumer in the fleet (~0.22 GB VRAM).

## Representative store world

`store_world()` is a REPRESENTATIVE restock-aisle model (ground + the target shelf
plate + the shelf back wall + the opposite aisle shelf), exactly as the prototype's
`cell_world` is a representative shelf. At GPU time the obstacles are rebased onto the
arm's measured FK reach centre (`shift_world`) so the shelf sits just in front of
wherever the mission arm reaches. Feeding the real composed USD / nvblox ESDF to the
optimiser is a separate (option-3-style) follow-up, not this bench.

The moving-person proxy is a named, runtime-updatable cuboid (cuRobo only exposes
cuboids as movable obstacles). Its plan-view radius + moving centroid match the
`Detection(position, radius, source)` contract shared with `sim-restock-humans-v1`
and `sim-restock-dynobs-v1`; when the humans lane merges, its people assets /
`/human_safety/state` detector can drive the proxy in place of the scripted sweep.

## Scenarios and metrics

`follow` (sweeping restock target, shelves present), `person` (proxy walks the
aisle), `blocked` (proxy parked on the straight-line path), `unreachable` (target
across the aisle). Per scenario the run records: control rate (Hz), warm solve time
(ms, mean/p95/max), estimated process VRAM, steady-state tracking error (m), minimum
obstacle and person clearance (m), joint-window violations, and — against the LIVE
MPC command stream — the gate behaviour (travel while RUN, zero travel while STOP,
commands dropped during STOP, out-of-window refusals, resume-to-freshest).

## Commands

See **RUNBOOK §18**. CPU tests are ROS-free; the guarded GPU run reuses the container
and takes `/tmp/aile-gpu.lock` non-blocking (the mission lanes hold priority). If this
bench ever grows a ROS touch point, its fleet lane domain is `ROS_DOMAIN_ID=107`
with `ROS_LOCALHOST_ONLY=1`.
