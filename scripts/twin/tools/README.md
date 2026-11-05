# `scripts/twin/tools/` — offline dev-loop benches

CPU-first benches and profilers for the Isaac mobile-restock lane. None of these are on the mission
default path — a mission run is byte-identical whether or not they exist.

| Tool | What it does |
|---|---|
| `arm_collision_bench.py` + `arm_collision_scene.py` | **Arm collision-avoidance evaluation bench** (below) |
| `arm_carry_bench.py` | boot-once carry-fold trials (drive gains, carton weld, cage contact) |
| `arm_mpc_bench.py` | cuRobo MPC follow-and-avoid evaluation (eval-only, not shipped) |
| `base_drive_bench.py` | base drive-authority / traction bench |
| `preplace_drive_bench.py` | pre-place drive bench |
| `profile_mission_run.py` | break a recorded run's wall clock into phases |
| `replay_reach.py` | replay a run's frame cadence through the production capture pipeline |
| `estimate_sim_speed.py` | sim-to-wall pace estimator |

---

## Arm collision-avoidance evaluation bench (`sim-arm-collision-bench-v1`)

Measurable, not asserted. It answers the captain's question — *how often does the arm avoid an
unexpected obstacle, with what clearance, at what cost in time, across the mission's arm legs* — for a
declarative set of planted-obstacle scenarios, and gives a pass/fail against thresholds.

**Out of scope** (per the brief): navigation, the planner internals, target perception, the carton
tilt bound + grace, carton dimensions. The bench *reports* carton tilt but does not enforce the bound
(that is the mission's own invariant).

### Two modes, one schema

* **CPU mock-ESDF mode (default, no Isaac/ROS/GPU)** — `run_arm_collision_bench.py`. The arm's curobo
  collision spheres are forward-kinematic at each waypoint of a leg's nominal swept trajectory (plus
  the welded carton on carry legs); obstacles are rasterised into a voxel signed-distance field at the
  mission voxel size (0.01 m) and queried by trilinear interpolation. Pure numpy + the pure URDF /
  carton-level FK.
* **Bounded GPU mode** — `run_arm_collision_bench_gpu.py` extends the boot-once `arm_carry_bench`
  stage: it authors the same obstacles into the assembled Isaac stage as world colliders, drives each
  leg, and reads the LIVE AABB clearance + PhysX contact. Same per-scenario schema → the CPU and GPU
  numbers drop into the same report/CSV.

### The gate is pluggable

A **Gate** maps (scenario, per-frame clearances) → a verdict
(`avoid` | `preempt_replan` | `slow_stop_resume` | `refuse` | `clear` | `none`). Two ship:

* `baseline` — the **current stack**: no perceived-ESDF gate on these arm legs (ground-truth collect,
  CAGE pair-filter, collision-OFF straight-line inserts, wrist-only ESDF, cart outside the grid). It
  never avoids/refuses → the **BEFORE** numbers.
* `reference` — a transparent collision-aware reference instrument modelling what the sibling lanes
  (`sim-rgbd-arm-collision-aware-v1`, `sim-arm-reactive-replan-v1`) target and the captain's section-9
  decisions: **people are a routing aid, the human gate owns the STOP**; the **cart is a known world
  obstacle**. It reacts ONLY to what it PERCEIVES.

Plug a sibling lane's real gate in with `--gate module:function` — the same scenarios + schema measure
it identically (the AFTER column).

### Modeling the current stack's blindness

Each obstacle carries `observed` (does a current camera frame it?). An `observed=False` obstacle — a
thin source-shelf cage bar behind the carton, the whole rollwagen cart (outside the arm ESDF grid), a
person during the wrist-blind carry — is present in the TRUE clearance but **absent from the ESDF**
(the scout's *unobserved == free* finding). So a perceiving gate is blind to it, which is exactly how
the bench measures the current gap. `--observe-all` flips every obstacle observed (models the coverage
the mast + cart-as-world-obstacle add) → the **AFTER-coverage** column.

### Metrics (per scenario)

`collision_free` (nominal) · `min_true_clearance_m` · `min_perceived_clearance_m` (through the mock
ESDF) · `esdf_error_m` (perceived − true; sub-mm for observed, metres for the blind cases) ·
`n_replans` · `extra_cycle_time_s` + overhead frac · `tilt_peak_deg` (carry legs) · gate verdict vs
expected. **Pass/fail**: no contact on the executed path (a refuse/stop/replan that prevents the strike
is credited); executed min true clearance ≥ margin; cycle-time overhead ≤ bound; verdict matched.

### Run it

```bash
# BEFORE — the raw current stack (no arm-collision gate)
PYTHONPATH=scripts python3 scripts/run_arm_collision_bench.py --gate baseline \
    --out <scratch>/before-baseline
# the collision-aware reference instrument at CURRENT camera coverage
PYTHONPATH=scripts python3 scripts/run_arm_collision_bench.py --gate reference \
    --out <scratch>/before-reference
# AFTER-coverage — once the stack SEES the obstacles (mast + cart world obstacle)
PYTHONPATH=scripts python3 scripts/run_arm_collision_bench.py --gate reference --observe-all \
    --out <scratch>/after-reference-coverage
# a sibling lane's real gate (the AFTER column), once importable
PYTHONPATH=scripts python3 scripts/run_arm_collision_bench.py \
    --gate twin.restock_manipulation.<their_module>:<their_gate_fn>
```

Exit code is 0 iff every scenario passed → CI / a pre-send gate uses it. The regression test
`scripts/twin/tests/test_arm_collision_bench.py` runs the CPU mode on every shipped scenario.

### BEFORE numbers (CPU, against the current stack — this branch)

8 scenarios (6 static per arm leg, 2 moving: a person crossing during carry, a hand reaching at
transfer). Margin 30 mm.

| Gate | Passed | Verdict match | Nominal-path collisions | Executed collisions |
|---|--:|--:|--:|--:|
| `baseline` (raw current stack) | 0 / 8 | 0% | 8 | 8 |
| `reference`, current coverage | 3 / 8 | 38% | 8 | 5 |
| `reference`, `--observe-all` (AFTER coverage) | 8 / 8 | 100% | 8 | 0 |

Reading: the current stack has **no arm-collision gate** (baseline 0/8). A collision-aware gate already
works on the **3 scenarios the wrist frames pre-grasp** (a misplaced carton / box on the store shelf, a
hand reaching at the slot). The **5 failures are the measured blind spots** — the source-shelf cage
context, the whole rollwagen cart at set-down/pre-place, the person during the wrist-blind carry —
each closed by a named sibling feature (mast depth at collect/carry; cart as a cuMotion world
obstacle; the straight-line-insert ESDF gate). With those (`--observe-all`), the reference gate makes
the right call on **all 8** with **zero executed contacts** (S5: no contact on the static set,
STOP-not-contact on the moving set).

### Bounded GPU proof (deliverable 4 — follow-up)

The live boot is bounded (≤ 20 min), gated by the canonical GPU lock + `gpu_budget` + the Blender
hard-refuse, never `--force`; the integration lane (`coldwin*`) has priority. Collect-only, two static
+ one moving:

```bash
renice +10 -p $$ >/dev/null
flock -w 5 /tmp/aile-gpu.lock env_isaaclab/bin/python scripts/run_arm_collision_bench_gpu.py \
    --runtime-config <mission-runtime.json> \
    --scenarios docs/examples/arm-collision-scenarios --legs collect_pick \
    --heavy-threshold-mb 2500 --no-gpu-lock \
    --out /home/manishsaini/firstmate/data/sim-arm-collision-bench-v1/gpu-collect
```

VRAM (scout §7): collect/carry/set-down has ~2.9 GB free (the mast +0.45 GB and the dynamic layer
+1.1 GB both fit); the `live_pose` transfer stage has 26 MiB free — bench nothing extra in that phase.

### Scenarios & hand-off

- Scenario set: `docs/examples/arm-collision-scenarios/*.json` (regenerated by the scratch
  `gen_scenarios.py` under the task data dir; obstacle placements are computed relative to each leg's
  swept volume, not magic numbers).
- Keys, scenario list, thresholds, before/after numbers: `docs/examples/arm-collision-bench-fragment.json`.
