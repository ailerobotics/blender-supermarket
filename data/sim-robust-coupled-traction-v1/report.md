# sim-robust-coupled-traction-v1 — the coupled outbound tow stall is drive-authority, not geometry

**Branch:** `fm/sim-robust-coupled-traction-v1` → PR into `fm/twin-robot-in-store-boot-v1`
**Captain's intent (2026-10-22):** "keep working on the simulation on the side to make less prone to
failure in any case." The coupled outbound tow (outbound_dock_to_bay) stalled at x≈−3.93 in the
validator's outbound1/warm9/warm10 runs while warm8/warm11 got through.

## Verdict: TRACTION (a PhysX drive-authority deficit), not geometry

Decided from the recorded evidence on CPU (no GPU, no Isaac). The proof is one command:

```
python3 scripts/check_tow_traction.py \
  --environment ~/datasets/sim-store-realism-v2/environment.json \
  --scenario   ~/datasets/twin-robot-in-store-boot-v1/mission/mission-scenario.json \
  --camera-config ~/datasets/twin-robot-in-store-boot-v1/mission/camera-config-bay.json \
  --fragment docs/examples/coupled-traction-fragment.json \
  --runs-dir ~/datasets/twin-robot-in-store-boot-v1/mission \
  --stalled outbound1 warm9 warm10 --clean warm8 warm11
# -> TOW_TRACTION_PROOF PASS (14/14 checks)
```

### Geometry is categorically ruled out
- **outbound_dock_to_bay is a straight y=15.25 pull**, start (−4.46, 15.25, 0) → goal (−2.0, 15.25, 0),
  heading +x. It is **not** a portal turn. The `sales_backroom_portal` is at y=11.125 on the south wall;
  the x=−3.875 portal-centre coincidence with the x≈−3.93 stall is a red herring (the lane never nears
  y=11.125). The "portal throat" framing in the prior evidence is a mislabel.
- **0 obstacles** in the lane corridor; the lane expands against the static map and its swept robot+cart
  envelope re-validates clear against the full **806-obstacle** live set (32 states).
- **Every recorded robot+cart footprint** of all five runs — stalled and clean, including the exact
  x≈−3.93 stall pose — is collision-free against the static map with the mandatory 0.05 m margin
  (outbound1 1598 ticks, warm9 13695, warm10 987, warm8 749, warm11 738 → **0 collisions each**).
- The **real ArticulatedController follows the lane closed-loop to the goal in the kinematic (no-slip)
  model** (arrived_stopped at (−2.08, 15.26), loop_ratio 1.0). Reaching the goal there while the GPU run
  stalls isolates the stall to sim physics — not geometry, not the controller logic.

### The stall is a drive-authority (re-breakaway-from-rest) deficit
- At the stall the train is **nearly straight** (articulation ≈ 0.3°, not a jackknife/cart swing).
- **Commanded ≈ 0.35 m/s, achieved ≈ 0.05** then 0. The base drove 0.52 m at a healthy cruise (~0.215 m/s)
  then collapsed into the **`measured_turn_radius` low-speed latch**: under-translating from a near-stop,
  measured |ω/v| exceeds the 1/1.5 arc threshold → the gate refuses → commands stop → cannot re-accelerate
  → PR 201's net-progress monitor correctly faults `coupled_no_progress` (console: 119 `tracking`,
  11 `measured_turn_radius`, 3 `coupled_no_progress`).
- **warm8/warm11 passed only by carrying entry momentum** across the same clear spot (cruise identical
  ~0.215 m/s, but 643/629 driving ticks vs 159 for the stalled runs). The forced-progress creep (PR 201)
  commands a clean forward v but **PhysX still won't translate the base from rest** → the deficit is
  physical, provable only on GPU.

The **loaded-tow traction margin** gate (`twin.traction.coupled_tow_margin`): a tow PASSES iff it reaches
the goal with no multi-second sim-clock pin. Measured on the real traces:

| run       | reached goal | longest pin (sim s) | max advance (m) | verdict |
|-----------|:---:|---:|---:|:---:|
| warm8     | yes | 0.95  | 2.37 | PASS |
| warm11    | yes | 0.93  | 2.37 | PASS |
| outbound1 | no  | 24.1  | 0.52 | FAIL |
| warm9     | no  | 87.3  | 0.52 | FAIL |
| warm10    | no  | 13.9  | 0.52 | FAIL |

## Fix: raise BOTH drive-wheel normal force (caster lift) AND drive force

`docs/examples/coupled-traction-fragment.json` is the bounded GPU proof config. Current stuck state:
`wheel_friction=1.0`, `wheel_drive=None` (80 N·m / 100), `caster_lift_m=0.005`.

- **caster_lift_m 0.005 → 0.03** (MobileSpec cap; baked into the URDF by `build_mobile_assets.py`): fully
  unloads the fore/aft casters so ~all robot weight bears on the two drive wheels, maximising μ·N_drive.
  0.005 fixed the UNCOUPLED crawl (bench tracking 0.097 → 0.988) but the heavier coupled re-breakaway
  still starves.
- **wheel_drive.max_force 80 → 200 N·m** (runtime `config['wheel_drive']`). **Key finding** from the
  quasi-static budget (`twin.traction.static_traction_budget`, unit-pinned): at 80 N·m the per-wheel drive
  **torque cap binds** (2·80/0.16 = 1000 N available) **below** the friction limit even at μ=1.0, so
  **caster lift alone cannot help** — raising the force to 200 N·m (cap 2500 N) unbinds it so caster lift's
  higher μ·N_drive then applies. This refines the validator's "caster lift toward 0.03" to "0.03 **and**
  the drive-force bump." The recommended config strictly dominates the stuck config in the budget
  (margin 6.93 → 16.1; share 0.61 → 0.95); the absolute numbers are **not** an efficacy claim — the real
  failure is dynamic re-breakaway, GPU-only. `wheel_friction` stays 1.0; the progress monitor stays ON.

Every switch is opt-in and backward compatible (absent ⇒ byte-identical).

## Handoff to the integration lane (the bounded GPU proof)

The 4 steps are in the fragment's `_integration_lane_steps`: (1) regenerate the mobile assets at
`caster_lift_m=0.03` offline with `build_mobile_assets.py` (no GPU), (2) point the mission config's
`mobile_manifest` at it and merge the fragment's top-level + navigation keys, (3) run the bounded
couple+outbound on the warm8 lanes (`--start-at couple --stop-after outbound`, serialized through
`/tmp/aile-gpu.lock`, `HEAVY_MB=2500` if whispr holds VRAM, never `--force`), (4) feed the run's
nav-trace to `scripts/check_tow_traction.py --gpu-run <dir>` → the SAME acceptance gate warm8/warm11 clear.
The uncoupled drive-force lever can be benched first with `run_base_drive_bench.py
--wheel-drive-max-force 200`.

## Artifacts in this PR
- `scripts/twin/traction.py` — pure core: the recorded-trace acceptance grader + the lever-direction budget.
- `scripts/check_tow_traction.py` — the one-command CPU proof (geometry clear + footprints clear +
  margin gate + lever direction); `--gpu-run` grades a single new trace.
- `scripts/twin/tests/test_tow_traction.py` — CPU unit tests (grader + budget monotonicity + torque-cap).
- `scripts/run_base_drive_bench.py` — adds the `--wheel-drive-max-force/--wheel-drive-damping` lever.
- `docs/examples/coupled-traction-fragment.json` — the bounded GPU proof config + build steps.
