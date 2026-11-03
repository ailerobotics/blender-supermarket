# sim-robust-station-heartbeat-simclock-v1

Sim-clock the parked-station heartbeat and the incorporation-ack liveness so render pacing can no longer
false-trip a station stop. Closes the last pace-unsafe wall timer on the live_pose honest cycle.

## The blocker (coldliverecovery, PR #236 head)

Run dir `~/datasets/twin-robot-in-store-boot-v1/mission/coldliverecovery/` (2026-10-23, full live_pose, all
keys PR 227–236, outbound recovery + `RESTOCK_NAV_PLAN_DIAG` on, `masked_depth.economy` on,
`RESTOCK_HEARTBEAT_UNAVAILABLE_S=120`, peak VRAM 10767 MiB — no OOM). Furthest the honest live_pose chain
has reached: **collect → couple → outbound (coupled) all cleared**, and the transfer started clean
(`STALE_OP_START_STOP_CLEARED op=transfer`, PR #235) and ran **9** live_pose captures. It then e-stopped:

```
MANIP_HOST_FAILED op=transfer phase='host' reason='host emergency stop (... see the named trigger ...)'
host-stack.log: STATION_ESTOP_PUBLISH site=tick reason='parked heartbeat lost: wall gap 30.75 s'
  preceded by: restock_incorporation_ack: incorp reject: ESDF absent/stale
```

Source: `restock_host/incorporation_ack.py` `IncorporationAckState.snapshot` — the parked-heartbeat **WALL**
guard, a flat bound (`RESTOCK_STATION_WALL_GAP_S`, the validator's env = 30 s) checked independently of the
SIM gap. Under the live_pose transfer render pacing the mission ran a multi-capture manip op without
heart-beating the parked station; the SIM gap stayed within the 30 s budget while WALL stretched to 30.75 s,
so **only the WALL leg tripped**, the station published the global `/manipulation/stop`, it re-latched on the
Isaac wire, and the transfer e-stopped. `incorporation_ack.py:227` already commented this exact edge from the
earlier `coldlivefull` run. This is the same wall-backstop-under-render-pacing class PR 231
(sim-robust-host-timers-audit-fix-v1) converted elsewhere — **this specific STATION timer was missed by that
audit** (its timer table had a `native_channel.py` StationClient section but no `incorporation_ack.py` row).

## Measured render pacing (from the run's own logs)

The `incorp reject` lines carry both the ROS wall stamp and the sim `now=`:

| span (host-stack.log) | Δwall (s) | Δsim (s) | wall-to-sim |
|---|---|---|---|
| collect → early transfer (line 6994 → 15285) | 640.2 | 75.45 | **≈8.5×** avg |
| transfer-window sample (line 15285 → 15321) | 1.99 | 0.40 | **≈5×** |

Consistent with the documented worst-case live_pose economy of 16–30×. At those ratios the 30.75 s WALL gap
is a SIM gap of only ~2–6 s — far under the 30 s SIM budget — which is why the SIM leg was healthy and only
the pace-unsafe WALL leg fired. **Check that tripped:** the parked-heartbeat WALL gap in
`IncorporationAckState.snapshot` (`STATION_ESTOP_PUBLISH site=tick`, `wall gap 30.75 s`).

## Fix

Make the parked heartbeat SIM-clock primary with a DERIVED frozen-sim WALL backstop — the identical shape
PR 231 shipped for the motion/payload/startup timers.

- **`restock/pace_watchdog.py`** gains the pure `frozen_sim_wall_backstop(sim_budget, wall_floor,
  pacing_factor=50)` = `max(wall_floor, sim_budget × 50)` (the leaf home of the derivation).
  `ros_adapter.frozen_sim_wall_backstop` now delegates there with its env-tunable factor default
  (byte-identical; its public name + tests unchanged).
- **`restock_host/incorporation_ack.py`** `IncorporationAckState`: `RESTOCK_STATION_SIM_GAP_S` (default 0.5)
  stays the real SIM liveness bound; `RESTOCK_STATION_WALL_GAP_S` (default 3) becomes the wall **FLOOR**
  feeding `self.wall_backstop_s = frozen_sim_wall_backstop(sim_gap_s, wall_gap_s)`; `snapshot` routes both
  gaps through `pace_watchdog.is_dead(mode='either')`. Named reasons (`sim gap` / `wall gap`) kept. Default
  derived backstop = `max(3, 0.5×50) = 25 s`.
- **`restock/station.py`** `StationState.snapshot` (the station-supervisor twin): the flat 15 s wall backstop
  → derived `_WALL_HEARTBEAT_BACKSTOP_DERIVED_S = 25 s`, routed through the same predicate. The combined
  message `'station process or parked heartbeat lost'` is **byte-identical**.
- **ESDF staleness reject** (`restock_host/preflight.py` `incorporated_time`, `now - stamp > max_age`) was
  already SIM-clock — both the node `now` and the ESDF header stamp ride `/clock`. Verified, unchanged. The
  in-flight ESDF request cancel (`time.monotonic() - pending_wall > 2`) is a benign request-retry cadence
  (reissues, never faults), classified in the audit.

On the real rig (sim == wall) the SIM leg fires first, so the wall value is inert and the **no-pacing path is
byte-identical**. Switch keys are backward compatible: a stale `RESTOCK_STATION_WALL_GAP_S` widen only ever
widens the backstop.

## Why the widen is now unnecessary

With the proof config (`RESTOCK_STATION_SIM_GAP_S=30`, `RESTOCK_STATION_WALL_GAP_S` left at the default 30),
the derived backstop is `max(30, 30×50) = 1500 s`, so coldliverecovery's 30.75 s render-paced gap can never
trip it while the 30 s SIM bound is the real guard. No 120 s quick-lever needed.

## CPU proof

- `scripts/twin/restock/host_pacing_harness.py` `simulate_station_heartbeat` drives the **real**
  `IncorporationAckState.snapshot` against the fake wall/sim clock pair.
- `scripts/twin/tests/test_host_timers_pacing.py` `StationHeartbeatUnderPacing`: at 1×/16×/30× — a healthy
  station whose manip gap stretches past the flat 30 s wall NEVER false-stops (the coldliverecovery shape),
  a dead station is caught on the SIM leg in bounded SIM time, a frozen sim is caught on the derived WALL
  backstop, and 1× default budgets are byte-identical.
- `scripts/twin/tests/test_restock_host.py` / `test_restock_station.py`: per-state regression guards (the
  derived backstop value, no false-trip at a 3.4 s render-paced gap, the frozen-sim catch at the derived
  backstop, the proof-config `sim=30 / wall=30 → 1500 s` shape on the exact 30.75 s gap).

Run (CPU, no ROS/Isaac/GPU):

```
cd scripts && PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=. python3 -m pytest \
  twin/tests/test_host_timers_pacing.py twin/tests/test_restock_host.py twin/tests/test_restock_station.py -q
```

## Bounded GPU proof (integration lane owns the boot)

`docs/examples/station-heartbeat-simclock-proof-fragment.json`: the armcam transfer staging
(`--start-at transfer --start-staging recorded_bay_park --stop-after verify`) under live_pose + the
`masked_depth` economy arm feed, with `RESTOCK_STATION_SIM_GAP_S=30` and **`RESTOCK_STATION_WALL_GAP_S` left
at its default 30** (the 120 s widen deliberately NOT set — the point is the fix makes it unnecessary).
**PASS** = transfer through verify with no station stop (`host-stack.log` shows no `STATION_ESTOP_PUBLISH …
parked heartbeat lost: wall gap` during an advancing sim; no `MANIP_HOST_FAILED op=transfer … host emergency
stop`). One GPU run at a time through `flock /tmp/aile-gpu.lock`, never `--force`; never touch the whispr
service (pid 1339, ~2.2 GB VRAM). **PENDING** — handed to the integration lane via firstmate.

## Audit

Full timer table (now with `restock_host/incorporation_ack.py` + `restock/station.py` rows), the coldlive-
recovery section, and the switch keys: `docs/host_timers_audit.md`.
