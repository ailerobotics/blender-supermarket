# sim-carton-level-southdock-v1 — south-dock collect place carton-level

**Branch:** `fm/sim-carton-level-southdock-v1` (base `fm/twin-robot-in-store-boot-v1`), CPU-proved.
**Trip fixed:** coldwin31 `CARTON_TILT ... roll=-0.04 pitch=+2.07 deg ... sustained 0.167s > grace 0.150s`,
at `op=collect`. The captain-fixed 2.0° bound and 0.15 s grace are **unchanged**.

## 1. What actually tripped (corrects the brief's read)

The brief said the guard tripped "in the pre-place phase" at a PR-307 place **endpoint** tilt, and that a
`path_waypoints` tweak cannot shave it. The host-stack log (which the integration worker did not have — see
§4) shows otherwise:

| host-stack.log line | what it says |
|---|---|
| `CARTON_LEVEL_PATH phase=preplace n=8 predicted_max roll=0.43 pitch=0.11 tilt=0.43` | pre-place approach planned level |
| `CARTON_LEVEL_PLACE_DESCENT phase=place predicted_max roll=0.01 pitch=0.05` | **place set-down** planned level |
| `CARTON_TILT ... roll=-0.04 pitch=+2.07 ... phase=place` (`HOST_FAULT_ESTOP phase=place`) | trip in the **place** set-down |

So the trip is the **PLACE set-down**, not pre-place; and **both the planned endpoints and the commanded
straight joint line are level to < 0.05°** (CPU-confirmed by FK on the reconstructed configs). The +2.07° is an
**execution** effect, not a planned-pose tilt. A `path_waypoints` tweak cannot shave it because there is no
planned tilt to shave.

## 2. The mechanism (named)

The welded carton's attitude is the arm's attitude. At the PR-307 south dock the sensed deck set-down sits
far forward in the arm base frame (`REPOSITION_REREGISTER place=(1.043, 0.055, 0.435)` in `ur_base_link`), so
the arm is **far-extended and loaded** at the set-down. The dominant driver is the **shoulder_lift gravity
torque**, which the finite servo cannot fully reject, so the physical joints **droop below** the commanded
geometrically-level config and the carton pitches off level:

| geometry | place x (base) | shoulder_lift gravity torque | executed set-down pitch (model) | in-sim |
|---|---|---|---|---|
| **south dock** (coldwin31) | 1.043 m | **58.1 N·m** | **2.03°** | tripped at **2.07°** (phase=place) |
| east cart (collectlevel4) | 0.970 m | 49.2 N·m | 1.86° | sub-grace transient, **passed** |

This is a **quasi-static** effect (set by torque/stiffness), **not** velocity lag — which is why PR 302's
0.02 rad/s gentling could not save it: coldwin31 ran the gentled 0.02 rad/s descent (40.8 s plan duration) and
**still** pitched +2.07°. The velocity-lag model PR 302 was built on is falsified by this one data point (a
velocity-lag model calibrated to collectlevel3's 2.22° at 0.08 rad/s predicts ≈0.5° at 0.02 — coldwin31 saw
2.07). The extra ~7 cm of forward reach between the east cart (0.970) and the south dock (1.043) raises the
shoulder_lift torque 49→58 N·m and pushes the marginal 1.86° droop past the 2.0° bound.

**Model.** `carton_level.arm_gravity_torques` computes the static gravity torque on each arm joint from the
real mission-URDF link masses/COMs plus the 0.45 kg welded carton. The one free parameter is an **effective
droop stiffness** `K = 2850 N·m/rad`, pinned by matching coldwin31's observed +2.07° (model → 2.03°); the same
model then gives the east cart 1.86° (which collectlevel4 saw as a 0.017 s sub-grace transient and passed) —
an independent check on one calibrated constant, exactly the PR 302 "one GPU data point" discipline.

## 3. The fix — loaded gravity-droop feedforward on the level place IK (opt-in, default off ⇒ byte-identical)

`carton_level.gravity_compensation` makes the collect-deck level place/preplace IK solve for the **COMMANDED**
config whose **EXECUTED** (drooped) pose is level — i.e. pre-tilt the command against the expected droop so the
physical welded carton lands parallel to the ground. Applied to both the preplace standoff and the set-down, so
the whole carried descent executes level (pre-place + place + release):

| geometry | current executed set-down tilt | **compensated executed** | command (geometric) tilt | descent exec max |
|---|---|---|---|---|
| south dock | 2.04° (> bound) | **0.00°** | 2.05° (intentional pre-tilt) | ≤ bound |
| east cart | 1.86° | **0.00°** | 1.86° (intentional pre-tilt) | ≤ bound |

The clear carry wrist region (`carry_wrist1_cap` / `carry_wrist2_min`) is preserved, the executed set-down
reaches the sensed deck within 5 mm, and the plan-time predictor is made droop-aware
(`predict_path_tilt_executed`) so the gate reads what the physical carton (and the `CARTON_TILT` trip) sees
rather than the intentional pre-tilt — otherwise the compensated plan would falsely self-reject.

**Why not "planned ≤ 1.0°" as the brief proposed:** the planned path was already level (< 0.05°); the ~2°
is a systematic droop present *at a perfectly level plan*, so lowering the planned tilt cannot help. The
physically meaningful quantity (the captain's real carton, and the GPU trip) is the **executed** attitude,
which the feedforward drives to ~0.00° — well inside the 1.0° margin.

Bound and grace untouched; off (the repo default `carton_level` is still absent ⇒ None) ⇒ byte-identical.

## 4. The diagnostic that was missing (brief point 3)

`CARTON_SEGMENT_TILT` (the PR 302 planned-vs-executed per-segment line) was absent from coldwin31 **everywhere,
including host-stack.log** — it never emitted in any CARTON run. Cause: `_transition` keyed the emit off
`self.phase` *after* it had been reassigned to the NEW phase, so the executed sample stashed under the
DEPARTING phase never matched. Fixed to key off the departing phase, and additionally relayed through the host
reply `evidence` to the orchestrator console so it lands in **orch-console.log** (the integration worker's log)
each cold run — not only host-stack.log.

## 5. CPU proof

- `scripts/twin/tests/test_carton_level_southdock.py` — pins the 307 geometry; the model reproduces south 2.03
  / east 1.86; the feedforward lands both executed ~0.00°; speed-independence vs PR 302; config resolution;
  `level_place_ik` pre-tilt; droop-aware predict; the `CARTON_SEGMENT_TILT` emit fix; controller wiring.
- `scripts/twin/tools/carton_level_bench.py::run_southdock_place` + `southdock_table` — the before/after bench.
- Plots + numbers (CPU proof artifacts, regenerate with `make_proof.py`):
  `setdown_tilt_before_after.png`, `south_descent_pitch.png`, `proof.json` (south 2.04→0.00°, east 1.86→0.00°,
  shoulder_lift torque 58.1/49.2 N·m) — under `/home/manishsaini/firstmate/data/sim-carton-level-southdock-v1/`.
- Existing carton-level suite still green (`test_carton_level*.py`); one pre-existing failure on the base
  (`test_end_to_end_collect_place_levels_the_preplace_standoff_and_reaches_verify`) is **not** introduced by
  this change (reproduced on the base with these files reverted).

## 6. Integration hand-off + bounded GPU acceptance

Enable on the collect-deck place (HOST `carton_level`), add the feedforward block:

```json
"carton_level": { "gravity_compensation": { "stiffness_nm_per_rad": 2850, "carton_mass_kg": 0.45 } }
```

(Keep the rest of `carton_level` as the lane runs it. `carton_level` is a HOST key — it must be in the host
config the manip host reads, not a runtime-only file, or it silently never activates.)

**Bounded GPU acceptance (cold collect-only, HOST carton_level on, 307 geometry):** no `CARTON_TILT` through
pre-place, place and release; expect the new `CARTON_SEGMENT_TILT phase=place executed_max ...` line in
orch-console.log — tune `stiffness_nm_per_rad` from its executed vs planned if the in-sim residual differs
from the 2850 N·m/rad calibration (lower stiffness ⇒ more pre-tilt).

Fragment for the lane: `docs/examples/carton-level-southdock-mission-fragment.json`.

## Out of scope / follow-ups
- Navigation / dock placement (moving the base closer would lower the reach torque) — out of scope.
- The preplace-standoff droop (~1.2° executed, under bound, already passing) is also compensated here; if a
  future geometry pushes the standoff near the bound the same feedforward covers it.
