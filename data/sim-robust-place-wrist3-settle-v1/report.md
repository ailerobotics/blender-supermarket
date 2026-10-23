# sim-robust-place-wrist3-settle-v1 — loaded collect-place wrist_3 terminal settle

**Branch:** `fm/sim-robust-place-wrist3-settle-v1` (off `fm/twin-robot-in-store-boot-v1`).
**Touches:** `scripts/twin/restock/trajectory_host.py`, `scripts/twin/tests/test_restock_trajectory_host.py`,
`docs/examples/place-wrist3-settle-validation-fragment.json`.

## Symptom (coldwin2-fail1, 2026-10-23)

The honest collect reached base-arrival, pick and weld, then the place-on-cart trajectory REACHED its
position (`TRAJ_HOST_GOAL_MISS pos_err=0.0034`, every per-joint error ≤ 3.4 mm, the upright place
family) but `ur_wrist_3` kept oscillating at **0.72 rad/s** (just over the 0.70 vel tol) and never
damped within the 30 s settle window. `TRAJ_HOST_FAULT "measured goal tolerance not reached"` →
`PATH_TOLERANCE_VIOLATED` → `HOST_FAULT_ESTOP phase=place` → "arm command heartbeat lost". No planner
crash, no respawn, no planning rejection. The identical place passed in `coldlivewin`/`coupledock90`.
Evidence: `~/firstmate/data/twin-robot-in-store-boot-v1/coldwin2-fail1-evidence.md` and the run dir
`~/datasets/twin-robot-in-store-boot-v1/mission/coldwin2-fail1/`.

## Diagnosis (CPU, analytic — the saturation limit cycle, NOT underdamped gains)

The place plan (`host-stack.log:9903`) is a 2-point move `[2.014,-1.91,0.093,-0.103,1.664,-0.0]` →
`[1.8392,-1.5036,0.0973,-0.3356,1.8604,-0.0]` over 7.62 s, `goal_pos_tol=0.120`. **wrist_3's commanded
trajectory is 0 → 0** — the joint that is NOT supposed to move is the one that rang. The ring is the
plant's response to the terminal reaction of the other joints coupled through the welded carton.

Drive gains at the place pose: `kp=3e5 kd=8000 max_effort=400`.

1. **The linear drive is heavily OVERDAMPED**: `ζ = kd/(2·√(kp·I)) = 7.3 … 32.7` for any plausible
   carton inertia about the wrist_3 roll axis (`I = 0.05 … 1.0 kg·m²`). A linear drive at that ζ does
   not ring — which is why `coldlivewin` settled crisply.
2. **The effort SATURATES**: at the observed terminal state the PD wants `kp·err + kd·v = 570 + 5760 =
   6330 N·m` against the **400 N·m cap — 16× saturated**. The damping term alone (`kd·v`) exceeds the
   cap for any `v > 0.05 rad/s`. So above 0.05 rad/s the drive delivers a bang-bang 400 N·m, not the
   computed linear damping, and under a persistent terminal disturbance (the weld reaction / PhysX
   weld + deck contact) a residual **limit cycle** can sustain. Whether it decays below the 0.05 rad/s
   desaturation velocity within 30 s is intermittent — exactly "damped in some runs, not others".

**Consequences for the fix.** Retuning gains is contraindicated: raising `kd` worsens the saturation,
lowering `kp/kd` wrecks the tracking envelope the earlier stiffening fixed (the existing code comment
warns of this). The commanded terminal velocity is already 0, so a terminal brake/ramp in the
interpolator has nothing to brake. **The lever is the ACCEPTANCE criterion**, which is also the
deliverable's second clause ("report a damped-but-nonzero wrist as a successful place by name").

**Why the instantaneous gate is structurally unable to pass this.** The old terminal gate needs ONE
sample with `pos_err ≤ tol` AND `vel ≤ vel_tol` simultaneously. A sinusoidal ring never has both near
zero at one instant: `pos ≈ 0` only at a zero-crossing, which is exactly where `|vel|` peaks; `vel ≈ 0`
only at a turning point, where `pos = amplitude`. So the gate can fire only if the amplitude ≤ tol (it
then passes at a turning point). coldwin2 never fired over ~300 samples ⇒ the wrist_3 amplitude was
**above the 0.12 rad box** (a > ±7° roll ring), so no instant fit the box.

**Reconstruction honesty.** The existing log records only the single `GOAL_MISS` sample — the settle
window was never instrumented, so the full wrist_3 envelope (sign/amplitude/frequency over 30 s) is
**not reconstructable from coldwin2-fail1**. This fix adds the per-sample `TRAJ_HOST_SETTLE` trace so
the next run records it. The CPU damped-oscillator characterisation above is analytic, not a replay.

## Fix — integrated terminal-settle envelope (`settle_decision`, pure)

`scripts/twin/restock/trajectory_host.py`:

- A new pure, importable `settle_decision(samples, *, pos_tol, vel_tol, final)` + `_envelope_peaks`.
  `samples` is the stream of `(t_sim, pos_err, vel_mag)` collected once per DISTINCT post-plan-end
  feedback sample. It returns:
  - `('settled', reason)` when the position stayed within `goal_pos_tol` across the **recent** window
    AND the velocity-peak envelope is **decaying** (recent peak ≤ `DECAY_RATIO`×earliest) below a
    bounded ceiling `ACCEPT_VEL` — a damped-but-nonzero wrist at a tight pose is a successful place.
    **Position is never loosened.**
  - `('diverging', reason)` when the recent velocity peak exceeds `DIVERGE_RATIO`×earlier and the vel
    tol — fails BY NAME, may fire before the budget ends.
  - `('miss', reason)` at budget expiry, NAMING why: `pose not bounded` (amplitude > pos tol) /
    `sustained residual oscillation (limit cycle, not damping)` / `decaying but not converged`.
  - `('pending', None)` until there is enough window (`MIN_WINDOW_S` 3 s) and samples (`MIN_SAMPLES` 8).
- `execute()` keeps the **instantaneous fast path first and byte-identical**, so a crisply-settling
  move (every currently-passing run) returns before the envelope is consulted. The envelope only runs
  when the fast path would have missed, so it can only convert a false-fail into a named success or
  name a genuine non-settle — it never changes a passing outcome.
- A throttled `TRAJ_HOST_SETTLE` trace (≈ every 1 s SIM) logs `pos_err`, `vel_mag` and per-joint
  velocity/err over the settle window, so a future run's wrist envelope is reconstructable.
- All thresholds are env-tunable; `RESTOCK_SETTLE_ENVELOPE=0/false/off/no` restores the legacy
  instantaneous-only gate **byte-for-byte** (same `GOAL_MISS` print + `measured goal tolerance not
  reached`). Default ON. Switch keys are backward compatible (absent ⇒ default behaviour).

### What this does and does not promise

- **Robustness gain (what it fixes):** every place where the wrist ring is bounded and *decaying* but
  the instantaneous both-small gate rejected it (render-paced aliasing, or a slightly slower
  coldlivewin) now settles as a named success instead of e-stopping the mission.
- **Honest limit:** if coldwin2's wrist_3 ring was a *sustained* > ±7° limit cycle (did not decay in
  the window), the envelope correctly fails it BY NAME — that is a genuine effort-saturation/weld
  physics issue for the integration/arm-stack owner, not a trajectory_host tolerance. The new
  `TRAJ_HOST_SETTLE` trace is what tells the two apart on the next run.

## Tests (CPU, no GPU/ROS)

`scripts/twin/tests/test_restock_trajectory_host.py` (all 22 pass, `.venv-usd`):
`_ring()` generates damped/sustained/diverging residual-oscillation sample streams and asserts:

- bounded **decaying** oscillation (amplitude starts above tol, decays below) → `settled` by name;
- **sustained** limit cycle above tol → `miss` "not bounded"/"sustained" by name;
- bounded-but-sustained (flat envelope within tol) → `miss` "sustained" by name;
- **diverging** (growing) → `diverging` by name and early;
- `pending` until enough window/samples (no lucky-dip accept);
- the coldwin2 phase-space fact (an above-tol ring never fits the instantaneous box, so the envelope
  is the only classifier) and its decaying counterpart accepted;
- `SETTLE_ACCEPT_VEL ≥ GOAL_VELOCITY_TOL` (envelope never undercuts the instantaneous velocity bound);
- a decaying ring that never falls below the position tol is **not** accepted (position not loosened);
- the `RESTOCK_SETTLE_ENVELOPE` switch parses backward-compatibly (default ON).

Related suites still green: `test_restock_trajectory_host`, `test_fm367_sim_deadline`,
`test_feedback_bounds`, `test_wall_bound_sweep`, `test_restock_manipulation`,
`test_sim_robust_watchdogs`, `test_restock_arm_reseed`, `test_restock_scripted_transfer`
(183 passed, 11 skipped). `test_restock_host.py` has a **pre-existing** collection error
(numpy 1.26 vs scipy 1.18 in `.venv-usd`, via `restock_host/coverage.py`), untouched by this change.

## Bounded GPU proof (for the integration lane)

`docs/examples/place-wrist3-settle-validation-fragment.json`: a collect-only run
(`--stop-after collect`) under the full coldwin2 capstone keys. PASS = the place-on-cart settles
(existing instantaneous success OR a new `TRAJ_HOST_GOAL_SETTLE` line) and the collect completes
through stow, with no `measured goal tolerance not reached` / `PATH_TOLERANCE_VIOLATED` /
`HOST_FAULT_ESTOP phase=place`. One GPU run at a time via `/tmp/aile-gpu.lock`, never forced, never
touching whispr. The fix is default-ON, so no config change is needed to exercise it;
`RESTOCK_SETTLE_ENVELOPE=0` reproduces the legacy false-fail for an A/B.
