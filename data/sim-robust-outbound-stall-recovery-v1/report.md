# sim-robust-outbound-stall-recovery-v1 — coupled outbound tow: bounded break-away recovery

Lane goal (firstmate): the coupled outbound tow **recovers from a no-progress stall instead of failing
the mission**, and the stall itself becomes **rarer** — as a PR into `fm/twin-robot-in-store-boot-v1`
with CPU tests and one bounded GPU proof config for the integration lane. Do not touch the mission
runtime configs under `~/datasets`; ship capability + an example fragment.

## Evidence (what holds the base when it is commanded forward while coupled)

Failing run: `~/datasets/twin-robot-in-store-boot-v1/mission/coldlivefull` (2026-10-23, full live_pose,
keys PR 227–235, economy on, heartbeat 120; peak VRAM 10548 MiB, **no OOM**). Collect succeeded, couple
latched, then the outbound **stalled** and `coupled_no_progress` fired (console
`coldlivefull.console.log` ~31520 / 31657 / 31793 — three times: StageRetry re-planned twice, each
re-stalled, then terminal).

From `coldlivefull.console.log` 31700–31800 (the stall window):

- The base is pinned at `at=(-3.90,15.25)` for the whole window — right at the `sales_backroom_portal`
  throat (centre x=−3.875). `TRACK_DIAG err` drifts 0.071→0.096 (well inside `tol=0.300`), `cart_err`
  0.006→0.025 — the controller thinks tracking is fine and keeps commanding forward.
- `NAV_STEP action=outbound status=running` alternates `reason='measured_turn_radius'` and
  `reason='tracking'`, with **`parked=True` on every tick**. `BASE_ATT` is level, `z=0.24` constant.

What `parked=True` means, and what actually holds the base (the task's first question):

1. **`parked=True` is a stationarity READOUT, not a brake our code left on.** It is the sim/station
   `parked` packet flag (the base's own transport/stationary window), reported True because the base is
   not moving — a **consequence** of the stall, not its cause.
2. **The runtime is commanding the tow, and `command()` releases every brake each tick.** On a
   `status=='running'` nav tick the runtime calls `MobileController.command(r.linear, r.angular, …)`
   (`restock/runtime.py`), and `command()` force-releases the parking brake, the pre-weld full base pin
   and the cart brake, then drops the stationary wheel-hold, **before** setting any wheel target
   (`mobile_runtime.py`). `mobile.park()` only runs when nav returns `succeeded`. So **no stale
   park/cart brake can hold the base during the tow** — there is no brake-left-asserted own-code bug on
   the running path. The commanded forward `v` is also substantial (`min(max_speed, distance*0.5)` ≈
   `max_speed`, not a below-break-away trickle). The stall is the **physics break-away from rest** —
   the drive cannot overcome the coupled static resistance at the throat ramp, and the base jitters in
   place with ~0 net displacement (forensics: measured `|rv|` noise to ~0.23 m/s).
3. **Two own-code contributors make the break-away fail.** (a) A `measured_turn_radius` **coast** tick
   returns `_wait(...) == velocity (0,0)` — so a stuck tow **thrashes stop↔forward** every few ticks and
   never sustains the drive effort needed to break static friction from rest. (b) The existing forced
   creep (`coupled_progress_escape`, sim-robust-outbound-drive-v1) only engages after `escape_s` and is a
   weak 0.03 m/s static crawl — "another retry of the same weak command", not a real break-away.

Non-determinism (validator, fm581): the **prior** coldlivefull (PR234, peak 10549 MiB), `lidar2` and
`visualsfull` cleared the SAME outbound `coupled=True` on the SAME config; PR 207 (caster_lift 0.03 +
wheel_drive 200) improved the break-away but did not eliminate it. VRAM/economy/keys identical between
pass and fail ⇒ the stall is genuine non-determinism in the coupled re-break-away from rest, not a
resource or config difference.

## Fix (two layers, opt-in / no-op by default ⇒ byte-identical when off)

`navigation.coupled_progress_recovery` (default `false`): turns the no-progress ESCALATE→FAULT ladder
into a **bounded break-away RECOVERY** in `ArticulatedController` (`scripts/twin/navigation/articulated.py`).

- **(a) Recovery path on `coupled_no_progress`, bounded on the SIM clock.** Once the base has been
  pinned below `coupled_progress_min_m` for `coupled_progress_escape_s` SIM seconds, run up to
  `coupled_breakaway_attempts` (default 3) windows of `coupled_breakaway_window_s` (2.0 s) each,
  commanding a **continuous pulsed committed-forward surge** — `coupled_breakaway_speed_mps` (0.18),
  square-waved every `coupled_breakaway_pulse_s` (0.5 s) between the high level and a relaxed 0.4× to
  **rock** the stuck contact. The surge is consumed by `gate_velocity` (new `_breakaway_cmd`), which
  **bypasses the `measured_turn_radius` coast** (so the command is never a v=0 stop) while keeping the
  obstacle / coverage / map safety gates **unchanged** — a real positive obstacle, a coverage gap or a
  stale map still refuses the surge. The instant net displacement from a window's anchor reaches
  `coupled_progress_min_m`, the break-away **succeeded** and normal tracking resumes; exhausting every
  attempt **faults `coupled_no_progress`** terminal (same name + StageRetry/BT contract), fast on the
  SIM clock — not the 120 s WALL nav-stuck.
- **(b) Removing the own-code cause.** The recovery's continuous forward command eliminates the
  stop↔forward thrash during the stuck state, and the surge is a real re-tow (0.18 m/s pulsed) rather
  than the 0.03 m/s creep. **"Back off a few cm" is deliberately NOT a reverse:** reverse towing is
  clamped off for a coupled train (`mobile.wheel_targets`: coupled `v≥0`) as a bring-up safety, so the
  implementable, safety-preserving equivalent is the forward surge/relax PULSE (a rock), documented in
  the code. The brake audit above confirmed there is no stale-brake own-code bug to remove on the
  running path.

Switch keys are backward compatible: absent ⇒ the sim-robust-outbound-drive-v1 escape→fault behaviour
is byte-identical (asserted by `test_coupled_breakaway_recovery_off_is_byte_identical` and the unchanged
pre-existing `coupled_progress_*` tests).

## Validation

CPU (real `ArticulatedController`, no GPU/ROS/Isaac):
`scripts/twin/tests/test_articulated_navigation.py`
- `test_coupled_breakaway_recovery_breaks_a_stuck_tow_free` — a pinned tow enters recovery and the surge
  breaks it free; never faults, records a break-away success.
- `test_coupled_breakaway_commands_continuous_pulsed_forward` — every recovery command is forward (> 0,
  no v=0 thrash) and pulses between the high and the 0.4× relaxed level.
- `test_coupled_breakaway_exhausts_attempts_then_faults_by_name` — a base that never breaks free faults
  `coupled_no_progress` after the bounded attempts, in ≤ `escape_s + attempts*window_s` SIM seconds.
- `test_coupled_breakaway_recovery_off_is_byte_identical` — recovery off keeps the escape→fault ladder
  and leaves the break-away counters at 0.
- `test_coupled_breakaway_rejects_invalid_params` — parameter validation (speed/window/pulse ranges,
  attempts int in [1,20], bools rejected).

Factory wiring: `scripts/twin/tests/test_navigation_factory.py`
`test_coupled_breakaway_recovery_flows_through` + `test_coupled_progress_recovery_rejects_non_bool`.

Full nav-adjacent sweep (articulated / factory / mission-tail / return-cartyaw / lidar-yaw):
304 passed, 18 skipped.

## GPU proof (handed to the integration lane via firstmate)

`docs/examples/outbound-stall-recovery-mission-fragment.json` — merge into the mission runtime config
(default-off keys; absent ⇒ byte-identical). Bounded couple+outbound: `run_mobile_restock.py
--start-at couple --stop-after outbound` with `--start-staging recorded_bay_park` if available, else the
full `coldlivefull` config; enable `RESTOCK_NAV_PLAN_DIAG=1` so the `NAV_BREAKAWAY`
enter/next/success/exhausted lines and commanded-vs-measured velocity are captured (the validator noted
these were NOT logged in coldlivefull because NAV_PLAN_DIAG was off). Expected PASS: the outbound either
breaks free (`NAV_BREAKAWAY success` → outbound `coupled=True`) or, if the physics deficit is total,
fails fast with a NAMED `coupled_no_progress` after the bounded attempts (no WALL hang). GPU rules: one
run at a time via flock `/tmp/aile-gpu.lock`, never forced; never touch the whispr service. The worker
does not take the GPU (the validator holds it free); this is an integration-lane run.

## Scope / ownership

The recovery makes the stall rarer (re-break-away retries) and fail-fast-by-name when it cannot recover;
it does **not** change the underlying drive physics. The coupled traction asset levers (caster lift,
wheel drive force) remain owned by sim-robust-coupled-traction-v1 (PR 207); the uncoupled drive-fidelity
lever is `navigation.mobile_spec.caster_lift_m` (fm434 bench).
