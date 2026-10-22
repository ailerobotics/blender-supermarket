# sim-robust-watchdogs-simtime-v1

Make no mission watchdog trip merely because the simulation renders slower than wall time,
while real host deaths are still caught. Lane for `sim-robust-audit-v1` failure mode
**rank 5 `wall-clock-watchdog-vs-render-pacing`** (and the related rank-7 station-ack stall and
rank-3 no-base-progress clock basis). Sibling lanes own the *behaviours* under these watchdogs
(`sim-robust-collect-place-v1` the place-on-cart cuMotion OOM, `sim-robust-outbound-drive-v1` the
coupled-tow no-progress stall); this lane owns only the **watchdog clock basis**.

## Measured sim-to-wall ratio

The mission runs render-paced: wall time advances far ahead of sim/stage time. Across the audited
runs the realized ratio is **0.17–0.31x** (sim-seconds per wall-second). A watchdog whose budget is
expressed in WALL seconds therefore gives the producer only `budget × ratio` sim-seconds of grace —
e.g. a 120 s WALL budget is only ~20–37 s of SIM, and a 30 s WALL command gap is a perfectly healthy
render-paced gap. That mismatch is the rank-5 root cause: healthy-but-slow pipelines false-trip, and
the single terminal line (`arm command heartbeat lost`) then masks very different upstream causes.

## Inventory — every wall-time watchdog the audit names

| # | Watchdog | Location | Clock it measured (before) | Verdict |
|---|----------|----------|----------------------------|---------|
| 1 | Arm-command heartbeat | `native_transport.py` `ArmCommandGate.watchdog` | **WALL only** (`time.monotonic() - last_receive > 30 s`) | pure-wall → **converted** |
| 2 | Station-ack frozen stamp (`StationAckStreamStalled`) | `native_channel.py` `StationClient.tick` | STAGE only (`now - ack_stamp_stage > 2 s`) — but fired on an **alive, still-delivering** publisher whose stamp was render-lagged | mis-firing → **hardened** |
| 3 | Station host-silence gate (`RESTOCK_HEARTBEAT_UNAVAILABLE_S`, 30 s widened) | `native_channel.py` `StationClient.tick` | WALL **with a STAGE primary** (`stage_unavailable_s`) already wired | already sim-primary + wall-backstop — **no change** |
| 4 | No-base-progress nav-stuck (120 s) | `runtime.py` `RuntimeAdapter._nav_stuck_check` | **WALL only** (`wall - nav_stuck_wall > 120 s`) | pure-wall → **converted** |
| 5 | Nav stage timeouts (`blocked_timeout` / `goal_timeout` / `stale_after`) | `navigation/factory.py` `_NAV_SIM_BOUNDS` | SIM (stage) clock | already sim — **no change** (regression-asserted by `test_fm386_sim_bounds.py`) |
| 6 | Collect/transfer sensor stall (`stall_window_s`) | `runtime.py` `_sensor_stall` | SIM clock | already sim — **no change** |
| 7 | Host manipulation `/manipulation/status` feedback | `restock_manipulation/isaac_ros_graph.py` | SIM primary (`last_receive_sim`) + WALL backstop (fm385 family) | already pace-aware — **no change** |

Converting #1, #2 and #4 closes the pure-wall / mis-firing cases; #3,#5,#6,#7 were already on the
sim clock and are locked by regression tests so a future change cannot quietly re-pin them to wall.

## The shared liveness rule — `pace_watchdog.is_dead`

`scripts/twin/restock/pace_watchdog.py` states the audit's rule once as pure arithmetic:

- `mode='either'` — dead if the SIM budget OR the WALL backstop is exceeded. Use when the sim clock
  advancing past the budget is itself proof of death (the producer stopped while the sim kept
  stepping); the wall backstop then only guards the frozen-sim / wholly-hung-process case.
- `mode='both'` — dead only when BOTH are exceeded. Use when a sim-advancing-but-producer-render-
  lagged transient must not trip: real wall time must pass so an alive-but-lagged producer can
  recover first. This is the audit's "**stamps frozen AND process gone**".

## What changed

**#1 Arm-command heartbeat** (`ARM_HEARTBEAT_SIM_BACKSTOP_S`, default 8 s, `RESTOCK_ARM_HEARTBEAT_SIM_BACKSTOP_S`).
The host trajectory executor is alive iff its command stamps keep advancing. The PRIMARY check is now
SIM: when the host dies (cuMotion CUDA OOM taking `move_group`+`trajectory_host` down) the greatest
received stamp freezes while Isaac keeps stepping, so `sim now − _host_now` grows past the budget
within a few SIM seconds — independent of render pace. The budget (8 s) sits well above the worst
host↔ISAAC `/clock` relay lag (~2.5 s) the host-domain freshness gate already cancels, so a
healthy-but-slow stream never false-trips. The WALL backstop (`ARM_HEARTBEAT_WALL_BACKSTOP_S`, 30 s)
now measures **wall-since-SIM-NOW-advanced** (reset whenever `now` moves), so a slow-but-advancing
render never arms it — only a sim whose clock has stopped while the watchdog keeps ticking.

**#2 Station-ack frozen stamp** (`_STATION_ACK_FROZEN_WALL_S`, default 45 s, `RESTOCK_STATION_ACK_FROZEN_WALL_S`).
Runs 341/346/348/350/354/356 tripped `StationAckStreamStalled` while the publisher was **still
delivering acks for our epoch** — only its `/clock`-derived stamp was render-lagged at warm start
(e.g. run-341: stamp frozen at 0.95 s, stage clock 6.017 s, frozen ~5 stage-s ≈ 16–30 wall-s). The
stage budget alone could not tell that lag from a dead publisher. The frozen detector now needs BOTH
the stage budget AND this WALL backstop (`is_dead(mode='both')`): a lagged-but-alive publisher
advances its stamp (or the host-silence/stage-liveness guards catch a truly gone process) well inside
45 s wall, while the genuine run-335 wedge still trips once real wall time confirms the publisher is
gone. Sim-profile only (unchanged on the real rig, `stage_clock=None`).

**#4 No-base-progress nav-stuck** (`nav_stuck_sim_s`, default off, `RESTOCK_NAV_STUCK_SIM_S`).
Added a SIM-clock no-progress budget. When set it GOVERNS and is pace-immune — a frozen base trips
after `nav_stuck_sim_s` SIM seconds however slowly the render runs — and the legacy (pace-sensitive)
WALL budget is NOT also applied, so render pacing can no longer shorten the real budget. When no sim
budget is set, `nav_stuck_wall_s` keeps its exact legacy behaviour (byte-identical). The
orchestrator's external wall SIGKILL cap remains the ultimate GPU-hours backstop in both cases.

## Backward compatibility / default-on

All new env/config keys are backward compatible. Default-on only where the audit shows false trips:

- **#1** and **#2** are sim-profile paths (engaged only when a stage clock / sim transport is wired);
  their new behaviour is default-on there, with generous sim-only budgets. The real rig is untouched.
- **#4** stays default-off (`nav_stuck_sim_s` unset ⇒ legacy wall path, byte-identical), because the
  budget is opt-in and owned by the integration lane's external runtime config. See below.

## Integration-lane switches (external mission runtime config)

The mission runtime JSON / sim-env block is owned by the integration lane (`twin-robot-in-store-boot-v1`).
To adopt the pace-aware behaviour, add to the sim-only env block (alongside the existing
`RESTOCK_*` widens in `test_wall_bound_sweep.py::SIM_ENV`):

```
RESTOCK_NAV_STUCK_SIM_S=120        # SIM-clock no-base-progress budget (governs; pace-immune)
# optional overrides (sane defaults already apply):
RESTOCK_ARM_HEARTBEAT_SIM_BACKSTOP_S=8     # arm liveness SIM budget
RESTOCK_STATION_ACK_FROZEN_WALL_S=45       # station frozen-stamp WALL backstop
```

`RESTOCK_NAV_STUCK_SIM_S` should replace (not accompany) a `RESTOCK_NAV_STUCK_WALL_S` chosen only to
bound GPU wall; keep a wall value only if a separate hard GPU-wall cut is still wanted — but note it
will no longer fire while a sim budget is set.

## Tests (CPU-only, no GPU)

`scripts/twin/tests/test_sim_robust_watchdogs.py` replays the named false-trip signatures and asserts
no trip, plus genuine deaths that must trip:

- `pace_watchdog.is_dead` either/both/None-leg semantics.
- Arm heartbeat: a 40 s-wall render tick with advancing stamps → no trip; a true host death
  (stamps frozen, sim advancing) → trip; a frozen sim (now not advancing, wall backstop) → trip.
- Station-ack: runs 341/346/348/350/354/356 (parametrized by their logged stamp/stage) → no
  `StationAckStreamStalled`; a forever-frozen publisher past both budgets → trip.
- Nav-stuck: a base frozen 30 s SIM / 400 s WALL under a 120 s sim budget → running; frozen past the
  sim budget → trip; the legacy wall-only path unchanged; progress resets both timers.

Existing `test_restock_station.py::test_frozen_ack_stamp_fails_fast_with_named_cause` was updated to
exercise the new stage-AND-wall contract. The integration lane validates on its next full GPU run.
