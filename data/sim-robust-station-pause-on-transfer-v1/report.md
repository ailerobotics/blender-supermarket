# sim-robust-station-pause-on-transfer-v1 — a far parked station never trips the global stop

## The failure (coldwin2 attempt-2, PR #238 head)

Run dir `~/datasets/twin-robot-in-store-boot-v1/mission/coldwin2-fail2/`, config
`mission-runtime-capstone238.json`, full honest live_pose cold cycle. It cleared **collect → couple →
OUTBOUND** (PR 238 validated in-sim) and then e-stopped at the **transfer** stage:

```
host-stack.log:
  STATION_ESTOP_PUBLISH site=tick reason='parked heartbeat lost: sim gap 30.05 s'
  restock_incorporation_ack: station recovered: epoch=3 from='parked heartbeat lost: sim gap 30.05 s' recoveries=1/3
mission.json final: RuntimeError: arm command heartbeat lost
```

The parked **collect/backstock** station was left ACTIVE after the collect op and was NOT heartbeat-ed through
the whole `couple → outbound → park → transfer-warm` span. Its liveness SIM gap reached **30.05 s — 0.05 s over
the 30 s `RESTOCK_STATION_SIM_GAP_S` budget** (PR 237 made this heartbeat SIM-primary; the earlier
coldliverecovery `wall gap 30.75 s` was the same class on the wall leg). The station published the global
`/manipulation/stop`, which **latched** on the bare-Bool Isaac wire. The station itself **recovered one tick
later** (epoch 3, recoveries 1/3) — proving it was alive — but the latch held, the arm (held by the latched
stop) lost its command heartbeat, and the transfer failed. A **pace/duration false trip of a live-but-idle far
station**, not a dead one. htsim never saw it: its staged transfer (`start_at=transfer`) reaches transfer with
no far parked collect station accruing a gap.

## The fix (two parts, both default-on, no new switch keys)

### 1. Pause the parked station when the op completes and the mission tows away (the root fix)

The documented proper fix (host_timers_audit.md root-sequencing note): apply the fm399 `pause_station` pattern
to the between-ops nav. `RuntimeAdapter.start('couple')` — the first tow-away stage after collect — calls
`manip.pause_station(…, suspend_only=True)`, which sends the station an explicit **`pause`** command (distinct
from the mid-collect reposition's `stop`).

The host-side liveness watchdog is now **explicitly paused-aware** (`IncorporationAckState.paused` and the
station-supervisor twin `StationState.paused`, state `'paused'`): a paused ACTIVE epoch skips the heartbeat-gap
check entirely, so it publishes **no stop** for any op length at any pacing, while an in-use (active, not
paused) station still faults **by name** on a real gap. The epoch is kept (not torn down like `stop`→idle);
the next op re-prepares a fresh epoch, which clears the pause. `StationClient.pause()` stamps the command fresh
off the stage clock (a stage-boundary call) and frees the client to re-prepare; it falls back to `stop()` for a
station predating the capability.

Byte-identical where it must be: a no-op when no station was prepared (a `start_at=couple/transfer` staged
start skips collect) or for a runtime without sensor I/O, and the reposition `pause_station` default
(`suspend_only=False` → `stop`) is unchanged. Default-on (no new key) so the capstone exercises it.

### 2. Release the Isaac stop wire mid-op on recovery (belt-and-suspenders)

PR 235 (`IsaacRosGraph._clear_stale_global_stop`) only re-derives the wire at a NEW op's START; a stop that
fires mid-op (`site=tick`) and whose station recovers one tick later was never released for the CURRENT op.
`incorporation_ack` now arms a one-shot `consume_release()` on a within-budget recovery and publishes
`Bool(False)` on `/manipulation/stop` (`STATION_ESTOP_RELEASE site=recovery`) — a new message the Isaac
ROS2Subscriber takes, un-latching the wire. A genuinely ongoing stop re-arms `True` on the next faulted tick,
so this only ever clears a now-recovered stale latch. With part 1 the trip no longer fires in the fixed flow;
part 2 guarantees a borderline trip from any path cannot kill a live op.

## Files

- `scripts/twin/restock_host/incorporation_ack.py` — `paused` state + `pause` action; snapshot pause-aware;
  `consume_release()` one-shot + `release()` publisher (`STATION_ESTOP_RELEASE`) on recovery.
- `scripts/twin/restock/station.py` — `StationState` parity (`paused` + `pause` action; snapshot pause-aware).
- `scripts/twin/restock/native_channel.py` — `StationClient.pause()` (sends `pause`, frees to re-prepare).
- `scripts/twin/restock/ros_manipulation.py`, `scripts/twin/restock_manipulation/isaac_factory.py`,
  `scripts/twin/restock/runtime.py` — `pause_station(…, suspend_only=…)` plumbed through the wrapper chain;
  couple-start pause wiring.
- `scripts/twin/restock/host_pacing_harness.py` + `scripts/twin/tests/test_host_timers_pacing.py` — the
  far-station pacing scenario (`simulate_far_station_paused`, `FarStationPausedUnderPacing`) at 1×/16×/30×.
- `scripts/twin/tests/test_restock_host.py`, `test_restock_station.py`, `test_restock_runtime.py` — state +
  client + wiring guards.
- `docs/host_timers_audit.md` — the coldwin2 section + timer-table rows.

## CPU proof

```
env -u PYTHONPATH PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 scripts python3 -m pytest \
  scripts/twin/tests/test_host_timers_pacing.py scripts/twin/tests/test_restock_host.py \
  scripts/twin/tests/test_restock_station.py scripts/twin/tests/test_restock_native_stop.py \
  scripts/twin/tests/test_restock_ros_runtime.py scripts/twin/tests/test_restock_runtime.py
```

Asserts: a parked-and-paused station accrues no gap over a 300 s SIM far op (≈9000 s WALL at 30×) and publishes
no stop; an in-use station with a real gap still stops by name; a recovery within budget arms the stop-wire
release one-shot; pause cleared by a fresh prepare / by stop; `StationClient.pause` sends `pause` and frees the
client to re-prepare; the couple-start pause wiring fires.

## Bounded GPU proof (for the integration lane)

A STAGED transfer cannot reproduce the far-station gap, so the real proof is the **full capstone** on this
branch head:

- Config: `~/datasets/twin-robot-in-store-boot-v1/mission/mission-runtime-capstone238.json` (UNCHANGED — the
  fix is default-on, no new keys).
- Orchestrate: `orchestrate-coldwin2.sh`, with its existing env **`RESTOCK_STATION_SIM_GAP_S=30`
  `RESTOCK_STATION_WALL_GAP_S=30`** (the DEFAULT budget — NOT the 60 s quick-lever). The 30 s budget is what
  tripped coldwin2, so leaving it at 30 is the honest proof that the pause (not a widen) prevents the trip.

**PASS criterion:** no `STATION_ESTOP_PUBLISH … parked heartbeat lost` during the couple→outbound→park→transfer
span, and the transfer op is not `MANIP_HOST_FAILED` on a stale latched stop. One GPU run at a time through
`flock /tmp/aile-gpu.lock`, never forced; never touch the `whispr` service.
