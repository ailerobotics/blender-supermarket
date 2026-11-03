# sim-collect-host-estop-v1 — the intermittent collect host e-stop

## Symptom

`coldwin23` attempt 1 (full cycle) FAILED at the **collect** stage at **sim 175.7 s** with
`host emergency stop (station-liveness/trajectory stop)` and **no matching controller fault**.
The same day `collect2` (collect-only) and `coldwin22` (full cycle, the captain's video) both
PASSED. The integration worker relaunched `coldwin23` from the couple stage.

## Root cause

**A valid parked source-station heartbeat read LATE by a loaded host, rejected as "stale", faults
the station and latches the bare global e-stop.**

During collect the base is parked and the mission heart-beats the parked **source** station
(`restock_camera_arm`/collect epoch) every tick so its nvblox ESDF stays fresh. The host-side
acknowledger (`restock_host/incorporation_ack.py`, `IncorporationAckState.accept`) validated every
command — including the recurring **heartbeat** — against a hard freshness window
`0 <= now - stamp <= 0.5` (sim/`/clock` seconds). On a rejection, `command()` sets `state.fault`
and publishes the global `/manipulation/stop` (`STATION_ESTOP_PUBLISH`), which the Isaac
`check_health` latches as a bare "host emergency stop".

The collect stage is long (coldwin22: sim **0 → 173.87 s**, ~174 s). Near the collect→couple
boundary the host executor sees a load spike (cuMotion stow replan + couple-nav kickoff + nvblox
integration). Under the live_pose render pacing (measured 16–30× wall-to-sim) a spike that starves
the host component executor lets the simulator's `/clock` advance while a heartbeat sits queued; the
host drains the backlog **oldest-first**, so the first heartbeat it processes after the spike was
stamped **> 0.5 sim-s earlier** — a *valid* heartbeat from an *alive* mission, read late. `accept()`
rejected it as `invalid station epoch, sequence or timestamp` → `state.fault` → bare e-stop.

This matches every observed fact:

- **collect stage, sim 175.7 s** — right at the collect→couple load spike (coldwin22 crossed the
  same boundary cleanly at 173.87 s; attempt-1's spike was ~2 s worse).
- **no controller fault** — it is a *station command rejection* in the host ack node, not a motion
  controller fault. The reason lives in `host-stack.log` (`STATION_ESTOP_PUBLISH`), while the mission
  side only sees the latched bare Bool wire → "host emergency stop".
- **intermittent** — it only fires when a spike pushes the read-lag past 0.5 sim-s; `coldwin22` and
  `collect2` did not hit it hard enough.

It is the captain's case (a): **a watchdog too tight for a loaded host.** The snapshot sim-gap guard
was already made pace-aware (PR 237); the `accept()` command-stamp freshness was the *last*
pace-unsafe liveness timer in this node — a late READ of a live heartbeat.

> Note on evidence: `coldwin23` reuses its run name, so the relaunch (booting at 18:08) **overwrote
> attempt-1's `host-stack.log`/`console.log`** in `~/datasets/twin-robot-in-store-boot-v1/mission/coldwin23/`
> before this task could read them. The root cause is reconstructed from the shipped code paths and
> the **passing** `coldwin22`/`stage-collect2` timing (collect station `parked_since=101.67`, snapshot
> `stamp` tracking `now`, couple begins 173.87 s). **Follow-up for the integration lane: copy a failing
> run dir aside before relaunching under the same name.**

## Fix (at the source, pace-aware, no bound widened)

`IncorporationAckState.accept` and its mirror `restock/station.py::StationState.accept`:

1. The `now - stamp <= 0.5` **upper** freshness bound is kept **only for `prepare`** (a new measured
   parked epoch must be fresh for the nvblox-incorporation contract). For the recurring
   liveness-relaxing/asserting commands **`heartbeat` / `pause` / `stop`**, a late read is **accepted**.
   Strictly-increasing `(epoch, sequence)` already rejects a replay as `ignored`, so a newer-sequence
   liveness command is proof the mission is alive regardless of host read latency. A **future** stamp
   (`now - stamp < 0`) and non-finite/malformed fields are still invalid for every action.
2. The liveness gap reference `last_heartbeat` is set to the host's **own acceptance clock `now`**
   (not the command `stamp`), so a late-but-accepted heartbeat resets the gap and the next snapshot
   tick does not fault on a stale reference.

**Nothing is widened.** The 0.5 s SIM-gap death guard (`snapshot`) is unchanged: a genuinely silent
station — one sending **no newer-sequence command** — still faults by name
(`parked heartbeat lost: sim gap X.XX s`). The collision_time/ESDF freshness safety
(`acknowledge` + `ready`'s `0 <= now - collision_time <= 0.5`) is untouched. The two couple-start
quiesce commands (`pause` at `runtime.start('couple')`, `stop` at the mid-collect reposition) get the
same late-delivery tolerance, closing the sibling stage-boundary trip.

On the real rig (`sim == wall`, no pacing) reads are fresh and `now ≈ stamp`, so the path is
byte-identical.

## CPU proof

- `scripts/twin/tests/test_restock_host.py` —
  `test_late_read_liveness_command_accepted_not_faulted` (heartbeat/pause/stop read 0.7 s late are
  accepted, no fault), `test_late_read_heartbeat_resets_the_liveness_gap`,
  `test_prepare_still_rejects_a_stale_stamp` (prepare stale + future stamp still invalid). Each FAILS
  on the pre-fix code (the late read raised) and passes after.
- `scripts/twin/tests/test_restock_station.py` — `test_late_read_liveness_command_accepted_and_gap_reset`,
  `test_prepare_still_rejects_stale_stamp_and_future_stamp` (the `StationState` mirror).
- `scripts/twin/tests/test_host_timers_pacing.py::LoadedCollectHeartbeatReadLag` — drives the SHIPPED
  `IncorporationAckState` through a ~174 s collect at 1×/16×/30× pacing with every heartbeat read
  0.6 s late (`host_pacing_harness.simulate_loaded_collect_heartbeat_read_lag`): survives, all
  heartbeats accepted; and `test_silent_station_still_faults_by_name` confirms a real death is still
  caught at ~0.5 s SIM.
- Full suites green: `test_restock_host.py test_restock_station.py test_host_timers_pacing.py
  test_sim_robust_watchdogs.py` — 119 passed, 2 skipped.
- `make_timeline.py` → `before-after-timeline.png` / `.txt` (this dir): the read-lag crossing 0.5 s at
  sim ~175.8 s (BEFORE trips → bare e-stop; AFTER absorbs it → collect runs to the 173.87 s couple
  handoff).

## Hand-off to the integration lane

- **What to watch in the next collect run:** the fix removes the false trip. If the parked station is
  *genuinely* lost, `host-stack.log` now only e-stops via the unchanged snapshot guard and names it:
  `STATION_ESTOP_PUBLISH site=tick reason='parked heartbeat lost: sim gap X.XX s'` (or
  `... wall gap ...` for a frozen sim). A bare "host emergency stop" at collect with no such named
  line should no longer occur.
- **Config:** none required — default-on, byte-identical on the real rig and when no spike occurs.
  The existing `RESTOCK_STATION_SIM_GAP_S` lever is unchanged and still available.
- **Preserve failing run dirs:** copy a failing run aside before relaunching under the same name, so
  the next host e-stop can be read directly from its `host-stack.log`.

## Out of scope (per brief)

dock-align re-sequence, navigation lanes, carton geometry.
