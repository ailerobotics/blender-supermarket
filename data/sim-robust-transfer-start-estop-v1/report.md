# sim-robust-transfer-start-estop-v1 — the full-chain transfer-START bare e-stop, named and fixed

## Symptom

In the every-stage-honest live_pose capstone run **coldlivefull** (branch
`fm/twin-robot-in-store-boot-v1`, head PR 234; run dir
`~/datasets/twin-robot-in-store-boot-v1/mission/coldlivefull/`) the mission cleared **collect →
couple → outbound → park** (the furthest any live_pose run has reached) and reached **transfer**,
which then e-stopped at its very first op:

```
console 31237  MANIP_START operation=transfer
console 31238  TRANSFER_GT_REPROJECT world=(-2.0,16.424,0.73) -> base=(0.091,1.178,0.230)
console 31239  TRANSFER_CAPTURE calls=9841 ... transfer_mode=live_pose
console 31240  MANIP_HOST_FAILED op=transfer phase='host' reason='host emergency stop'
mission.json  state=failed reason='arm command heartbeat lost' (the downstream consequence)
```

The stop was **bare**: zero `HOST_FAULT_ESTOP` / `PORT_FAULT_ESTOP` / `PLANNER_UNRECOVERABLE` /
`PLAN_WALL_DIAG` in `host-stack.log`, so PR 231's e-stop naming and PR 234's recovery did not cover
it. The same live_pose transfer completed through verify from a **staged start** (`start_at=transfer`,
runs armcam4 / armcameco). So the full-chain entry differs from the staged start in a way that trips a
stop at `MANIP_START`.

## Root cause (confirmed from host-stack.log)

`reason='host emergency stop'` is raised by `IsaacRosGraph.check_health()` when the Isaac-side global
`/manipulation/stop` wire reads True. That wire is a bare `Bool` on an OmniGraph subscriber and
**latches permanently** (nothing ever publishes `False`; `check_health` then sets `faulted=True`).
`phase='host'` is the Isaac poll reply default — there is no host-controller fault behind it.

PR 231 named only the **motion controller / port** stop paths (they set a failed `/manipulation/status`
that the Isaac-side `_host_fault_suffix` reads). But the shared stop topic has **other** publishers that
set **no** `/manipulation/status`:

- `restock_host/incorporation_ack.py` (the live station-liveness node, `python3-6`) — 3 sites;
- `restock/station_host.py` (the alternative station supervisor) — 4 sites;
- `restock/trajectory_host.py` — fault + republish (already logged `TRAJ_HOST_FAULT`);
- `restock_manipulation/ros_host.py` `finally: motion.stop()` (motion-host process exit).

The decisive line in coldlivefull's `host-stack.log`:

```
19310 [python3-6] [restock_incorporation_ack]: station recovered: epoch=3
      from='parked heartbeat lost: wall gap 30.64 s' recoveries=1/3
```

So the mechanism is:

1. Collect left a parked-station epoch active (the `place_approach` re-warm, epoch 2).
2. During the **between-ops nav gap** (couple → outbound → park, base moving so the mission cannot
   parked-heartbeat the station), that station tripped its **wall-gap liveness guard**
   (`parked heartbeat lost: wall gap 30.64 s`, `IncorporationAckState.snapshot`), set `state.fault`, and
   published a **bare** `stop=True` (unnamed) — which latched on the Isaac wire. `check_health` is not
   called during these nav stages, so it never consumed it then (nav succeeded).
3. Transfer's fresh `prepare` (epoch 3) **recovered** the host-side station fault — but nothing
   un-latches the Isaac wire.
4. Transfer's first poll ran `check_health`, read the stale latched True → bare `host emergency stop`.

The staged start never hits this: `start_at=transfer` begins with a **fresh** `IsaacRosGraph` whose stop
wire was never True, and skips the collect station + nav gap entirely. The full chain carries the latched
wire on the SAME long-lived graph — the exact "what the full chain carries over vs what the staged start
initializes" diff.

## Fix

Two complementary changes; **no new config/env keys** (the fresh-graph path is byte-identical, so every
staged start and every collect op start are unchanged).

1. **Name every unnamed stop publisher** (so a bare op-start e-stop can never recur). A named,
   reason-throttled `STATION_ESTOP_PUBLISH site=… reason=…` line is printed before each
   `stop.publish(Bool(data=True))` in `incorporation_ack.py` and `station_host.py`; a
   `HOST_EXIT_ESTOP_PUBLISH` before the motion-host `finally` stop. The Isaac-side `_host_fault_suffix`
   now appends a hint (`see the named trigger in host-stack.log: HOST_FAULT_ESTOP / PORT_FAULT_ESTOP /
   *_ESTOP_PUBLISH`) whenever `/manipulation/status` carries no matching controller fault, so the mission
   reason is never bare-with-no-direction.

2. **Re-derive the global-stop wire per op** (`IsaacRosGraph.start()` → `_clear_stale_global_stop`,
   called after the existing explicit-recovery guard). A stop latched True AT a new op's start is a stale
   carry-over — a genuine op-fatal stop during the PRIOR op would have failed the mission before a new op
   began — so it is cleared (writing the same `Controller.attribute` path `check_health` reads). This
   makes the full chain begin each op like a fresh staged-start graph. A genuinely ongoing stop (the host
   republishes True every tick) is re-taken by the ROS2Subscriber on the next tick, so `check_health`
   still fires within the op — now with a named trigger and the suffix hint. For a fresh graph (stop wire
   already False) `start()` does no write and prints nothing → byte-identical.

Changed files: `scripts/twin/restock_manipulation/isaac_ros_graph.py`,
`scripts/twin/restock_host/incorporation_ack.py`, `scripts/twin/restock/station_host.py`,
`scripts/twin/restock_manipulation/ros_host.py`, `docs/host_timers_audit.md`.

### Root sequencing note (follow-up, not this lane)

The station faulting during the between-ops nav at all is the fm399 pattern (`pause_station`) that already
stops the collect station cleanly (→ idle, not faulted) before the **reposition** drive. Extending that to
the **between-ops** nav (stop the station when a manip op completes, before the mission tows away) would
stop the spurious stop being published in the first place. That is a mission-runtime change with its own
test burden; this lane ships the localized, directly-testable op-start re-derivation + naming, which makes
the symptom impossible regardless.

## CPU tests

`scripts/twin/tests/test_restock_native_stop.py` (all pass; `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=.`
from `scripts/`):

- `test_full_chain_transfer_start_clears_a_stale_latched_global_stop` — a transfer op started with a stale
  latched stop (the collect→couple→outbound→park carry-over) clears it, accepts its first goal (job
  dispatched), logs `STALE_OP_START_STOP_CLEARED op=transfer`, and its first `check_health` does not e-stop.
- `test_fresh_graph_op_start_is_byte_identical` — a fresh graph (stop wire False, the staged start + every
  collect op start) does no write and prints nothing.
- `test_a_genuine_ongoing_stop_still_fires_within_the_op` — a host still asserting stop re-arms the wire →
  `check_health` still e-stops within the op.
- `test_non_controller_stop_names_the_host_stack_trigger` / `test_controller_fault_status_still_names_the_reason`
  — the `_host_fault_suffix` hint vs the matching-controller-fault reason.

Regression: 171 affected CPU tests pass (native_stop, ros_split, host, station, station_ack_relay,
collision_time_mirror, staged_start, stop_after, trajectory_host, control_ipc).

## Bounded GPU proof (for the integration lane)

The entry **cannot be isolated to a staged start**: the bug requires a real collect op (which leaves a
station active) followed by the couple→outbound→park nav gap (which faults that station and latches the
stop) on the SAME long-lived Isaac graph. A `start_at=transfer` staged start begins with a fresh graph and
no prior station, so it does not reproduce it. Therefore the proof is the **full coldlivefull config,
unchanged**, run on this branch:

```
# integration lane, GPU, flock /tmp/aile-gpu.lock, never --force, never touch whispr (pid 1339, ~2.2 GB)
# same config as the coldlivefull run (mission-runtime-coldlivefull.json under ~/datasets, owned by the
# integration lane — do NOT edit): full chain, live_pose transfer, masked_depth.economy on,
# RESTOCK_HEARTBEAT_UNAVAILABLE_S=120. No new keys needed for this fix.
```

PASS criteria (grep the console + host-stack):

- transfer reaches `MANIP_START operation=transfer` and does NOT emit `MANIP_HOST_FAILED … reason='host
  emergency stop'` at its first op;
- `host-stack.log` shows `station recovered: … wall gap …` at the transfer prepare (the fault still occurs)
  **followed by** `STALE_OP_START_STOP_CLEARED op=transfer` on the console at `MANIP_START`;
- the transfer runs through `TRANSFER_MODE live_pose observe → PLAN_DELIVERY → … → verify` as it does from
  the staged start (armcam4);
- if any stop DOES fire during transfer, it now carries a named trigger (`*_ESTOP_PUBLISH` /
  `HOST_FAULT_ESTOP` / `PORT_FAULT_ESTOP`) in `host-stack.log` and a non-bare mission reason.

Optional cheaper bound: add `stop_after: transfer` (already supported) to end the run once transfer
completes, skipping the return leg.
