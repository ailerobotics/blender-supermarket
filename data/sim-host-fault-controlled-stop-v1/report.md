# sim-host-fault-controlled-stop-v1 — a controlled stop on a host fault while welded

## Symptom (coldwin33)

A complete side-bay cold cycle (PR 309..315 in). At the collect **place** set-down the carried-carton
level guard tripped by **0.06 deg**:

```
MANIP_HOST_FAILED op=collect phase='host' reason='host emergency stop (host fault: CARTON_TILT
carried carton off level: roll=-0.09 pitch=+2.06 deg exceeds bound 2.00 deg
(sustained 0.150s > grace 0.150s, peak 2.06 deg))'
```

The 2.0 deg bound + 0.15 s grace are captain-fixed and never loosened (the set-down tilt itself is a
separate lane). What this lane fixes is what the emergency stop then did to the scene:

- the arm was **welded** to the carton and fully extended over the cart deck, with the carton
  **interpenetrating** the deck;
- the host emergency stop **released the brakes** and the collect stage **re-began** and drove the
  base, so PhysX resolved the penetration by **flinging** the base and the cart
  (`BASE_ATT roll_deg=138 pitch_deg=47 z_m=2.11`);
- collect re-began **twice** on the flipped robot until the retry cap, and the run ended with the
  generic `RuntimeError: arm command heartbeat lost` — masking the true first trip.

Saved: `/home/manishsaini/datasets/twin-robot-in-store-boot-v1/mission/coldwin33-fail1-saved/`
(`mission.json`, `coldwin33.console.log`, `host-stack.log`, `nav-trace.jsonl`). The same fault path
was seen earlier at the far-extended dock place; it is reached whenever a guard trips while welded.

## Root cause — the exact order of operations (quoted)

1. **The host controller trips and publishes the global stop** — `restock_manipulation/controller.py`
   `_fail()`:

   ```
   print('HOST_FAULT_ESTOP phase=%s reason=%r' % (self.phase, reason), flush=True)
   self.reason, self.phase = reason, 'failed'
   try:
       self.motion.stop()            # publishes the global /manipulation/stop (emergency stop)
   ...
   # Never drop a held object on failure. Recovery is an explicit new decision.
   return self._result(({'kind': 'stop'}, {'kind': 'gripper', 'enabled': self.held}))
   ```

   The controller keeps the weld (`gripper enabled=self.held`) and stops — this half is correct. But
   the keep-held control lives in the **host** subprocess; the mission-side `ros_manipulation.step`
   builds its `Result` with `controls=()`, so that intent never reaches the Isaac side.

2. **The mission reads the stop and the stage fails.** `isaac_ros_graph.check_health()` raises
   `'host emergency stop' + <host-fault suffix>`; the suffix carries the CARTON_TILT reason. Collect
   attempt 1 fails with that reason.

3. **The behaviour tree RE-BEGINS collect** (coldwin33 console, immediately after MANIP_HOST_FAILED):

   ```
   NAV_PLAN_DIAG start goal=(-5.73, 16.0, 0.0) coupled=False active_lane=None ...
   MOBILE_BRAKE released                         <-- the yaw-lock loop-relief brake let go
   BASE_FULL_BRAKE engaged ... / BASE_FULL_BRAKE released / CART_BRAKE released   <-- thrash, every tick
   ```

   `classify_stage_failure('collect', reason)` returns `'coupling_failed'` (the reason contains
   "**cart**on"), which is in `STAGE_FAILURE_TYPES` — so `StageRetry` treated the CARTON_TILT trip as
   a recoverable failure and re-began the stage.

4. **The re-begin drive releases every brake** — `mobile_runtime.MobileController.command()` (pre-fix
   top):

   ```
   def command(self, v, omega, observation, *, transport_ready=False):
       self.release_brake()            # drops the yaw-lock D6 brake
       self.release_base_full_brake()  # drops the full fixed brake
       self.release_cart_brake()
       self.release()                  # drops the wheel position-hold
   ```

   The step-0 nav leg (`drive->pick`) calls `command()`, which unpinned the chassis **while the arm
   was still welded to the carton and interpenetrating the deck**. The per-tick thrash above is
   `advance()`'s `_apply_base_brake` re-engaging the brake and `command()` dropping it again each
   tick, so the FixedJoint is created-and-removed every tick and never actually holds.

5. **PhysX flings the base.** With the base free and the welded carton inside the deck, the
   penetration impulse rotated the chassis through the arm/weld loop:
   `BASE_ATT roll_deg=-0.354 pitch_deg=5.163` (sim 197.07) → … → roll 138, pitch 47, z 2.11.

6. **Collect re-began twice more on the flipped base** and each attempt died with
   `arm command heartbeat lost`, which `mission.json` recorded as the **terminal reason** — a
   marginal, recoverable 2.06 deg trip turned into a wrecked scene and a misleading reason.

The root defect: **a host fault while a carton is welded is not a controlled stop.** The base brake
is released and the stage is driven again before the welded, interpenetrating body is cleared.

## Fix — a controlled stop at the source (default-on; inert on every success)

New pure decision module `scripts/twin/restock/controlled_stop.py` + minimal wiring in the base/weld
owners and the BT retry decorator. On a manipulation failure **while a carton is welded**:

- **Keep the base brake engaged, never drive.** `mobile_runtime.MobileController` gets a
  `_welded_hold` latch; `command()` returns a zero-velocity `park()` **without releasing any brake**
  while it is set. `isaac_port._apply_base_brake` forces the **FULL** fixed brake (never the yaw-lock
  loop-relief brake, never released) for every tick the controlled stop is latched, so a re-begin
  drive can no longer unpin a welded arm.
- **Hold the arm energised, not slack.** `isaac_port.enter_welded_controlled_stop()` calls
  `hold_arm()` (the gate position-holds the measured joints with the configured drive) every tick,
  and sets `ArmCommandGate.suppress_heartbeat` so the deliberate host-silent hold does **not** raise
  `arm command heartbeat lost` over the true trip.
- **Keep the weld until the carton is supported or clear.** The controlled stop never opens the
  gripper; the pure `decide_welded_stop()` releases the weld only when `supported or not
  interpenetrating`, and offers a retreat (lift/back-off) only when a clear retreat exists. The
  terminal stop holds in place (the safe default) — never cuts the drive while a welded body
  interpenetrates.
- **Retry gate.** `RuntimeAdapter.manip_retry_gate()` refuses a collect/transfer re-begin while the
  controlled stop is active (the base must stay planted) **and** when the chassis attitude is outside
  a sane band derived from the robot's nominal (level) pose, captured + logged once at the first
  stage start (`CONTROLLED_STOP_BAND nominal ...`; band = nominal ± 20 deg roll/pitch, ± 0.15 m z —
  passes a level `(0, -1.17, 0.24)` chassis, rejects the `(138, 47, 2.11)` fling). `StageRetry`
  consults the gate before re-beginning.
- **Name the first trip.** `StageRetry` preserves the first real trip and
  `controlled_stop.prefer_first_trip()` keeps it (CARTON_TILT 2.06) over a later downstream symptom
  (`arm command heartbeat lost`); normal non-trip retry chains are unchanged.

Net on coldwin33: attempt 1's CARTON_TILT trip → the base/weld owners enter the controlled stop
(`WELDED_CONTROLLED_STOP engaged: base full-braked, arm held energised, weld kept`); `StageRetry`'s
gate refuses the re-begin → the mission fails **immediately** with the named **CARTON_TILT 2.06**
reason. No re-begin drive, no brake release, no fling, no heartbeat cascade.

### Files

| File | Change |
|---|---|
| `scripts/twin/restock/controlled_stop.py` | **new** pure module: attitude band, first-trip preference, welded-stop decision |
| `scripts/twin/mobile_runtime.py` | `_welded_hold` latch; `command()` refuses to release brakes / drive while latched; `latch/release_welded_hold` |
| `scripts/twin/restock/native_transport.py` | `ArmCommandGate.suppress_heartbeat` — the deliberate controlled-stop hold does not raise the heartbeat loss |
| `scripts/twin/restock/isaac_port.py` | `carton_welded()`, `base_attitude()`, `enter_welded_controlled_stop()`; `_apply_base_brake` + `advance` force the full brake and re-hold the arm while latched |
| `scripts/twin/restock/runtime.py` | `_note_manip_failure` (enter controlled stop + record first trip on a welded fault), `manip_retry_gate`, `first_trip_reason`, one-time band capture |
| `scripts/twin/restock/bt/leaves.py` | `StageRetry` consults the retry gate and names the first trip on abort |

All changes are default-on but **inert on success**: the latch engages only on a failure while
welded; the retry gate returns None on a level base with nothing welded; the first-trip preference
returns the latest reason for every non-trip chain. The one added always-on line is the one-time
`CONTROLLED_STOP_BAND` log.

## CPU proof

`scripts/twin/tests/test_sim_host_fault_controlled_stop.py` (22 tests, no Isaac/ROS/GPU):

- **pure logic** — attitude band passes level / rejects the fling / rejects non-finite; `prefer_first_trip`
  keeps CARTON_TILT over heartbeat and leaves normal chains unchanged; `decide_welded_stop` holds +
  never drives while interpenetrating, releases only when supported/clear;
- **brake-hold (fake host)** — the real `MobileController.command` releases **no** brake and parks
  while latched, and releases every brake when unlatched (the baseline that proves the latch is the
  only thing holding them);
- **no slack while welded** — `ArmCommandGate.watchdog` with `suppress_heartbeat` still HOLDS the arm
  (applies the hold target) but does **not** raise; fires normally when not suppressed; and
  `enter_welded_controlled_stop` pins the base full + holds the arm + suppresses the heartbeat;
  `_apply_base_brake` forces the full lock instead of the yaw-lock brake at `place` while latched;
- **retry refusal on a flipped base** — `manip_retry_gate` refuses while the controlled stop is
  active and when attitude is out of band, allows a level base, inert without a band;
- **named reason + coldwin33 replay** — through the real `StageRetry`: the CARTON_TILT-while-welded
  trip aborts **immediately** (zero re-begins), `mission.fail` is called with the **CARTON_TILT**
  reason, and a later heartbeat symptom still names the tilt; a normal grasp failure still retries.

No regressions: `test_restock_bt.py`, `test_restock_bt_phases.py`, `test_restock_runtime.py`,
`test_restock_manipulation.py`, `test_restock_collect_stow_stall.py`, `test_ros_adapter_branch.py`,
`test_restock_transfer_carry.py`, `test_sim_robust_watchdogs.py`, `test_restock_trajectory_host.py`,
`test_restock_staged_start.py`, `test_arm_carry_bench.py`, `test_mobile_assets.py` all pass.

## Bounded GPU acceptance (integration lane)

`docs/examples/host-fault-controlled-stop-fragment.json`. Force a guard trip at the welded place (the
captain-fixed 2.0 deg bound trips naturally at a marginal set-down, as in coldwin33; a `start_at=transfer`
bench reaches the same far-dock place quickly). Assert, after the trip:

- console shows `WELDED_CONTROLLED_STOP engaged: base full-braked, arm held energised, weld kept`;
- `BASE_ATT` stays inside the band after the trip (roll/pitch near level, z near 0.24) — **no fling**;
- the carton stays held/supported (not launched); the base does **not** re-begin-drive;
- `mission.json` terminal reason **names the first trip** (`CARTON_TILT ... 2.06 deg`), never
  `arm command heartbeat lost`.

## Out of scope (follow-ups)

- The set-down tilt itself (the 2.06 vs 2.00 deg) — a separate lane; this lane never loosens the
  captain-fixed bound or grace.
- Navigation and towing.
- An ACTIVE retreat (lift/back-off along the approach) is decided by `decide_welded_stop` but the
  controlled stop currently holds in place (the safe default); wiring a measured `retreat_clear`
  signal from the composer into `enter_welded_controlled_stop` is a follow-up.
