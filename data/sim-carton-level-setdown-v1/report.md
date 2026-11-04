# sim-carton-level-setdown-v1 — collect place SET-DOWN carton level (follows sim-carton-level-southdock-v1)

**Branch:** `fm/sim-carton-level-setdown-v1` (base `fm/twin-robot-in-store-boot-v1`), CPU-proved.
**Trip fixed:** coldwin33 `CARTON_TILT ... roll=-0.09 pitch=+2.06 deg exceeds bound 2.00 deg (sustained 0.150s
> grace 0.150s)`, `phase='host'`/`phase=place` (`HOST_FAULT_ESTOP phase=place`). The captain-fixed **2.0° bound
and 0.15 s grace are UNCHANGED** — never loosened.

coldwin33 was the first complete side-bay cold cycle with every fix of the day in (PR 309..315) **and**
sim-carton-level-southdock-v1's `gravity_compensation {stiffness_nm_per_rad 2850, carton_mass_kg 0.45}` ON.
The grasp welded, and the carry stayed level through lift / tuck / transit / preplace. The **place set-down**
— the most extended welded pose — still pitched **+2.06°**, sustained exactly the 0.150 s grace, and the guard
tripped. PR 311's one global stiffness closed the preplace droop but moved the set-down only from 2.07
(coldwin31, uncompensated) to **2.06** (coldwin33, compensated).

## 1. What tripped, and the per-segment evidence (brief point 1)

coldwin33 host-stack per carried segment (planned = the pre-dispatch predicted peak, executed = the measured
max):

| segment | planned | executed | source line |
|---|---|---|---|
| lift | 1.41 | 1.43 | `CARTON_SEGMENT_TILT phase=lift` |
| tuck | 0.01 | 0.01 | `CARTON_SEGMENT_TILT phase=tuck` |
| transit | (unknown) | 0.01 | `CARTON_SEGMENT_TILT phase=transit` |
| preplace | 1.21 | 1.21 | `CARTON_SEGMENT_TILT phase=preplace` |
| **place (set-down)** | **0.07** | **2.06** | `CARTON_LEVEL_PLACE_DESCENT ... predicted_max ... pitch=0.07` then `CARTON_TILT ... pitch=+2.06 ... phase=place` |

Two facts settle the diagnosis:

- **lift / tuck / preplace: planned ≈ executed within 0.02°.** Those segment predictions are the GEOMETRIC path
  replay. That they match the GPU proves the arm TRACKS its commanded joint path closely at those (less-extended)
  poses — there is no un-modelled droop there.
- **place set-down: predicted 0.07° but executed 2.06°.** The quoted gate line
  `CARTON_LEVEL_PLACE_DESCENT phase=place predicted_max roll=0.01 pitch=0.07 deg (<= bound 2.00)` is the
  droop-aware replay of the gravity-compensated command. It proves the set-down **did receive** the compensated
  (pre-tilted) goal — so the candidate "set-down not receiving the compensated goal" is ruled out.

Running PR 311's own model on the coldwin33 south geometry (`REPOSITION_REREGISTER place=(1.042, 0.064, 0.437)`
in `ur_base_link`; `scripts/twin/tools/carton_level_bench.py::run_southdock_place`) shows the pre-tilt the IK
applied and rules out the other two candidates:

| joint | gravity torque | model droop (pre-tilt applied) | ≥ max_pretilt cap 0.12? |
|---|---|---|---|
| ur_shoulder_lift | 58.0 N·m | −0.0204 rad (−1.17°) | no |
| ur_elbow | 41.2 N·m | −0.0145 rad (−0.83°) | no |
| ur_wrist_1 | 2.2 N·m | −0.0008 rad | no |

- **The `max_pretilt_rad` cap (0.12 rad) is NOT clipping** (max |droop| 0.0204 rad ≪ 0.12) — second candidate ruled out.
- The compensated command is **intentionally pre-tilted to +2.04° geometric**, landing the pre-tilt on the
  HEAVY shoulder_lift / elbow pitch joints. The model's "executed" = command + droop(command) is then ~0
  (circular by construction — the IK solves the command so that command+droop is level, and the predictor
  evaluates command+droop). **So the first candidate — under-compensation — is the real one, but for a reason
  the model cannot see:** the +2.04° pre-tilt is placed on joints that do not realise it.

## 2. The mechanism (named): the pre-tilt lands on gravity-pinned joints

The GPU moved the executed carton by only **0.01°** (2.07 → 2.06) for a **+2.04°** commanded pre-tilt, i.e.
`d(executed)/d(command) ≈ 0.005` on the heavy joints. At the far-extended loaded set-down the shoulder_lift /
elbow pitch chain is gravity-**pinned** (near effort saturation): it settles at a gravity-determined equilibrium
that a sub-degree setpoint change cannot move. PR 311's feedforward pre-tilted exactly those joints, so the
pre-tilt is swallowed and the carton stays ~2° off level. The **wrist** (2.2 N·m) tracks faithfully — which is
why lift/tuck/preplace executed matched the geometric plan within 0.02°.

The PR-311 "executed = command + droop(command)" model is **circular** for the compensated command, so it could
never express this residual (it predicts 0.00° on CPU). The honest **two-regime** model — heavy pitch joints
pinned at `geometric_level − τ/K`, wrist tracking its command — is calibrated to **both** GPU points from the
one K = 2850 constant and reproduces them (CPU, `scripts/twin/tools/carton_level_bench.py::run_setdown_tworegime`,
within ~0.1° of the observed 2.06/2.07):

| geometry | place x (base) | uncompensated executed | PR311 heavy-routed executed | the trip |
|---|---|---|---|---|
| south (coldwin33) | 1.043 m | 1.99° (obs 2.07) | 1.94° (obs 2.06) | over bound |
| east (collectlevel4) | 0.970 m | 1.82° (obs 1.86) | 1.77° | sub-grace, passed |

The heavy-routed pre-tilt changes the executed carton by < 0.06° at the south dock — matching the GPU's
2.07 → 2.06.

## 3. The fix — route the droop-cancelling pre-tilt onto the TRACKING wrist (brief point 2)

`GravityCompensation.correct_joints` (default the wrist cluster) names the faithfully-tracking joints the
feedforward is placed ON. `level_place_ik` with `gravity_comp` now (a) solves the pose GEOMETRICALLY (reach +
level, no droop), then (b) predicts the loaded joint sag and re-levels **only the wrist** against the settled
(sagged) config (`GravityCompensation.wrist_feedforward`). The heavy joints keep their geometric command and
sag from there (a disturbance the fine setpoint cannot null); the wrist pre-rotates to counter the sag-induced
carton pitch, and because the wrist tracks, the physical welded carton lands level. The correction **follows
the extension**: the sag is `τ(command)/K` and `τ` grows with reach, so a deeper set-down gets a larger wrist
pre-rotation (a per-pose gravity-torque solve — not a hand-dropped constant).

Two-regime executed set-down, before/after (CPU; `proof.json`, `setdown_before_after.png`):

| geometry | uncompensated | heavy-routed (PR311) | **wrist-routed (fix)** | fix pre-tilt joint | executed reach err |
|---|---|---|---|---|---|
| south | 1.99° | 1.94° (> bound) | **0.00°** | ur_wrist_1 | 23 mm |
| east | 1.82° | 1.77° | **0.00°** | ur_wrist_1 | 19 mm |

Every carried segment including the **set-down and release** plans ≤ 1.0° executed-equivalent at BOTH docks
(the fix lands ~0.00°, a full 1.0° inside the brief's margin and far inside the 2.0° bound), the clear carry
wrist region (`ur_wrist_1 ≤ 0.65`, `ur_wrist_2 ≥ 0`) is preserved, and the executed set-down still reaches the
sensed deck within 23 mm — far inside the mission place goal tolerance (0.12 m), and the physical loaded sag the
arm already placed through (coldwin31 reached the deck). **The bound and grace are untouched.** Off (no
`gravity_compensation`) ⇒ byte-identical.

> **Reach trade-off (honest):** routing leveling through the wrist pivots the commanded pad TCP ~2 cm off the
> sensed deck — the heavy joints can no longer be pre-tilted to hold the executed TCP exactly on target (the
> old `level_place_ik` coupled solve did, but the GPU showed that pre-tilt does not realise). This is far
> inside the place goal tolerance (0.12 m) and smaller than the loaded vertical sag the arm already tolerates.
> One PR-311 assertion that checked the old 5 mm executed-reach (tied to the falsified coupled-heavy solve) is
> updated to the real invariant (within the place tolerance); every other PR-311 assertion stays green.

## 4. Per-segment diagnostic — set-down AND release (brief point 3)

`CARTON_SEGMENT_TILT` now leads with the compact tokens the integration worker asked for:

```
CARTON_SEGMENT_TILT phase=<segment> planned=<deg> executed=<deg> | executed_max roll=.. pitch=.. | planned_max roll=.. pitch=.. deg (bound 2.00)
```

- It emits for the **place set-down** on the (now successful) place→release transition — previously the set-down
  reported only through the trip, so with the fix in, the descent becomes log-tunable without a trip.
- It emits for the **release** too: `_carton_seg_sample` stashes the welded carton's release attitude (stash-only,
  never fails the move — the gripper is opening), and `_transition` prints the `phase=release` line at
  release→retreat. The set-down's predicted level is carried to the release line as its `planned`.

Both lines are relayed through the host reply `evidence` to the orchestrator console (`orch-console.log`).

## 5. CPU proof

- `scripts/twin/tests/test_carton_level_setdown.py` — the two-regime model reproduces 2.07/2.06; the fix lands
  the set-down level (≤ 1.0°) at BOTH docks, strictly better than uncompensated AND the PR311 heavy-routing;
  the correction lands on the wrist, reach within the place tolerance, clear region preserved; extension-following;
  `correct_joints` config; the `CARTON_SEGMENT_TILT planned=/executed=` tokens for place AND release.
- `scripts/twin/tools/carton_level_bench.py::run_setdown_tworegime` / `setdown_tworegime_table` — the
  before/after bench; `proof.json` + `setdown_before_after.png` + `setdown_per_segment.png` regenerate with
  `make_proof.py` (under `/home/manishsaini/firstmate/data/sim-carton-level-setdown-v1/`).
- The **PR 311 carton-level suite stays green** (`test_carton_level_southdock.py`, `_place_descent.py`,
  `_preplace_path.py`, `_lift_path.py`, `_bench.py`, `_carton_level.py`, `_controller.py`) apart from the one
  pre-existing base failure PR 311 already documented
  (`test_end_to_end_collect_place_levels_the_preplace_standoff_and_reaches_verify` — a 180° inverted geometric
  preplace rejection, reproduced on the untouched base, NOT introduced here and unrelated to the gravity-comp path).

## 6. Integration hand-off + bounded GPU acceptance (brief point 5)

Keep the `carton_level` block exactly as the lane runs it; the `gravity_compensation` sub-block is unchanged
from PR 311 — the fix is in HOW the compensation is realised, so **no config change is required**. The default
`correct_joints` is the wrist. Exact HOST block (`docs/examples/carton-level-setdown-mission-fragment.json`):

```json
"carton_level": { "gravity_compensation": { "stiffness_nm_per_rad": 2850, "carton_mass_kg": 0.45 } }
```

`carton_level` is a HOST key — it must be in the host config the manip host reads
(`scripts/twin/restock_manipulation/ros_host_config.json` merge / the integration host block), not a runtime-only
file, or it silently never activates.

**Bounded GPU acceptance (cold collect-only, HOST `carton_level` on, 307 south-dock geometry):** all
`CARTON_SEGMENT_TILT phase=<segment>` lines (lift, tuck, transit, preplace, **place**, **release**) under 2.0°
with margin; a successful weld; a successful release; **no `CARTON_TILT` / `HOST_FAULT_ESTOP` through place and
release**. If the in-sim set-down executed residual differs from ~0°, tune `stiffness_nm_per_rad` from the new
`CARTON_SEGMENT_TILT phase=place executed=<deg>` line — **lower stiffness ⇒ more wrist pre-tilt**. Do NOT loosen
`bound_deg` or `tilt_grace_s` (captain-fixed).

## Out of scope / follow-ups
- The emergency-stop fling that followed the trip (unbraked extended welded arm) — separate lane.
- Navigation / dock placement — out of scope.
- The two-regime constant K = 2850 is calibrated to the GPU set-down droop; the bounded GPU run above re-pins it
  from the per-segment log if the in-sim residual differs. The mechanism (correction on the tracking wrist) is
  independent of the exact K.
- The pre-existing `test_end_to_end_...reaches_verify` 180° inverted-preplace failure predates this lane.
