# Manipulation-host timer audit (sim-robust-host-timers-audit-fix-v1)

Exhaustive inventory of every timer, deadline, timeout, freshness window and retry budget in the
manipulation host, trajectory host, station client, payload/attach handshakes and the mission runtime's
manipulation watchdogs, with its clock basis and this lane's disposition.

## Principle

Under the honest live_pose cold cycle the second full-resolution render (the `masked_depth` arm feed)
slows the simulator to a measured **16–30x wall-to-sim** during manipulation. A bound timed on the WALL
clock then false-trips a healthy-but-render-slow producer, while the sim clock stays a faithful measure of
mission progress. The rule (shared by [`pace_watchdog.is_dead`](../scripts/twin/restock/pace_watchdog.py)
and the `freshness-sim-clock-not-wall` principle): **measure every deadline/freshness window on the SIM
clock (or scale it by the measured render economy); keep a WALL leg only as a bounded backstop for a
genuinely dead process with positive evidence** (a frozen /clock the SIM leg can never advance, a vanished
action server, a specific never-resolving future). The canonical frozen-sim wall backstop is derived
generous by [`frozen_sim_wall_backstop`](../scripts/twin/restock_manipulation/ros_adapter.py)
(`max(wall_floor, sim_budget × PLAN_WALL_BACKSTOP_PACING_FACTOR)`, factor default 50 > the worst observed
pacing), so a wall backstop can never pre-empt its sim leg for an alive producer, yet on the real rig
(sim == wall) the sim leg fires first and the no-pacing path is byte-identical.

## Sample-4 e-stop: the trigger, found and fixed

coldlive sample 4 (run dir `~/datasets/twin-robot-in-store-boot-v1/mission/coldlive/`, head PR #227+228+229)
e-stopped **bare** `MANIP_HOST_FAILED reason="host emergency stop"` at the live_pose place descent. The host
side showed only `PLAN_WALL_DIAG state=ik future_done=False result_status=None elapsed=120.10 replans=0`:
the **total-move planning WALL cap** (`planning_wall_exceeded` / `PLAN_TOTAL_WALL_CAP_S`, env-widened to
120 s) tripped while the cuMotion plan future had not resolved — the arm feed render-starved the planner, so
only ~4–8 **SIM** seconds had passed. The plan was alive-but-starved, not hung. PR #228 had put the
per-dispatch deadline on the sim clock but *explicitly left this total cap on the wall*; it was the last
pace-unsafe timer in the planning path. Fixed here (SIM-primary total budget + derived frozen-sim wall
backstop). The e-stop is no longer reportable bare: `controller._fail` now logs `HOST_FAULT_ESTOP
reason=... phase=...` and the port logs `PORT_FAULT_ESTOP state=... exc=...` before any stop publication,
and `PLAN_WALL_DIAG` now names which budget leg fired (`leg=sim|wall`).

## Named-trigger instrumentation (so a bare emergency stop can never recur)

The Isaac-side stop wire is a bare `Bool`, so the mission can only ever report `"host emergency stop"` with
no cause. Every host-side fault path now prints a distinct named trigger to `host-stack.log` BEFORE the
stop publication:

| site | line emitted | file |
|---|---|---|
| controller fault chokepoint | `HOST_FAULT_ESTOP phase=<p> reason=<r>` | `restock_manipulation/controller.py` `_fail` |
| port poll() raise → stop() | `PORT_FAULT_ESTOP state=<s> exc=<e>` | `restock_manipulation/ros_adapter.py` `RosMotionPort.poll` |
| planning-budget fire | `PLAN_WALL_DIAG … leg=sim|wall` | `ros_adapter.py` `plan_wall_diag` / `_watchdog` |
| station-liveness stop (incorporation-ack) | `STATION_ESTOP_PUBLISH site=<command\|tick\|tick-exception> reason=<r>` | `restock_host/incorporation_ack.py` |
| station-supervisor stop | `STATION_ESTOP_PUBLISH site=<command\|tick\|tick-exception\|process-exit> reason=<r>` | `restock/station_host.py` |
| trajectory-host stop | `TRAJ_HOST_FAULT <reason>` (pre-existing) | `restock/trajectory_host.py` `fault` |
| motion-host process exit | `HOST_EXIT_ESTOP_PUBLISH reason=<r>` | `restock_manipulation/ros_host.py` `finally` |

**sim-robust-transfer-start-estop-v1 (the full-chain transfer-START e-stop).** PR 231's naming + the
Isaac-side `_host_fault_suffix` covered only the MOTION controller/port (they set a failed
`/manipulation/status`). But the **station-liveness** guard (`incorporation_ack`/`station_host`) and the
**trajectory host** also publish the shared global `/manipulation/stop`, with NO `/manipulation/status`,
so a stop from them surfaced as a **bare** `"host emergency stop"` with no cause. coldlivefull died this
way: `incorporation_ack` tripped its **wall-gap liveness guard** (`parked heartbeat lost: wall gap
30.64 s`) during the collect→couple→outbound→park nav gap (the station stayed active while the mission
drove away and could not parked-heartbeat it), published the bare stop, which **latched permanently** on
the Isaac OmniGraph subscriber (nothing publishes `False`). The station then **recovered** at transfer's
fresh `prepare` (host-stack.log `station recovered: epoch=3 from='parked heartbeat lost: wall gap 30.64 s'`)
— but that could not un-latch the Isaac wire, so transfer's first `check_health` read the stale True. The
staged start (`start_at=transfer`) never hit it: it begins with a FRESH graph.

Two changes close it: (1) the named `STATION_ESTOP_PUBLISH` / `HOST_EXIT_ESTOP_PUBLISH` triggers above, plus
`_host_fault_suffix` now appends a hint (`see the named trigger in host-stack.log`) whenever
`/manipulation/status` carries no matching controller fault, so no op-start stop is ever bare-with-no-
direction; (2) `IsaacRosGraph.start()` **re-derives the global-stop wire per op** (`_clear_stale_global_stop`):
a stop latched True AT a new op's start is a stale carry-over (a genuine op-fatal stop during the prior op
would have failed the mission before a new op began), so it is cleared — the full chain now begins each op
like a fresh staged-start graph. A genuinely ongoing stop is re-taken by the ROS2Subscriber on the next
tick, so `check_health` still fires within the op (now named). No new switch keys; the fresh-graph path
(every collect op start, every staged start) does no write and is byte-identical. Root sequencing note: the
station should ideally be stopped (→ idle, not faulted) when a manip op completes before the mission tows
away — the fm399 `pause_station` pattern applied to the between-ops nav — which would stop the spurious
stop being published at all; the op-start re-derivation is the localized, test-anchored fix for this lane.
CPU proof: `scripts/twin/tests/test_restock_native_stop.py`.

## Timer table

Legend — **clock**: `SIM` (node/sim clock, pace-safe), `WALL` (monotonic), `SIM+WALL` (sim primary +
wall backstop). **status**: `sim-clocked` (already pace-safe before this lane: PR 201/204/205/226/228 or
earlier), `converted` (this lane), `wall-backstop` (kept wall by design — bounded, positive-death), `count`
(retry/attempt count, not a clock).

### `restock_manipulation/ros_adapter.py` — RosMotionPort / RosTransferBridge

| symbol / env | bounds | clock | status |
|---|---|---|---|
| `planning_wall_exceeded` / `PLAN_TOTAL_SIM_CAP_S` (env `RESTOCK_PLAN_TOTAL_SIM_CAP_S`, default 22) | total-move planning budget across all ik/plan dispatches, re-plans, seeds | **SIM**+WALL | **converted** (sample-4 fix); SIM primary, `plan_wall_backstop_s` = `frozen_sim_wall_backstop` derived generous |
| `PLAN_TOTAL_WALL_CAP_S` (env `RESTOCK_PLAN_WALL_CAP_S`, default 22) | now the *wall floor* feeding the derived frozen-sim backstop (kept for back-compat; a stale sim-widen only widens the backstop) | WALL (backstop) | **converted** (role changed from primary cap → backstop floor) |
| inline total-cap in `_poll` plan_result | same budget at the re-plan fail-fast | **SIM**+WALL | **converted** (now calls `planning_wall_exceeded()`) |
| `PLAN_SIM_DEADLINE_S` (env, default 30) | per-dispatch ik/plan_accept/plan_result deadline (`_arm_plan_deadline`) | SIM | sim-clocked (PR 228) |
| `sim_execute_budget` / `EXECUTE_SETTLE_MARGIN_S` / `MOVE_SIM_DEADLINE_SCALE` | per-trajectory execution deadline | SIM | sim-clocked |
| `SENSOR_WATCHDOG_S` (env, default 1) | per-frame /sensors freshness | SIM | sim-clocked; paired with the wall backstop below |
| `SENSOR_STREAM_WALL_BACKSTOP_S` (env, default 30) | /sensors frozen-sim backstop | WALL (backstop) | wall-backstop (frozen sim can't advance the SIM per-frame bound) |
| `STARTUP_GRACE_SIM_S` (env `RESTOCK_STARTUP_GRACE_SIM_S`, default 5) | first-frame "no initial sensor frame" grace | **SIM**+WALL | **converted**; `startup_grace_wall_backstop_s` derived generous |
| `STARTUP_GRACE_S` (env `RESTOCK_STARTUP_GRACE_S`, default 5) | wall floor feeding the startup backstop (kept for back-compat) | WALL (backstop) | **converted** (role → backstop floor) |
| `MOVE_WALL_TIMEOUT_S` / `self.timeout` (env, default 25) | legacy per-dispatch wall deadline; now used only to hold the deadline AHEAD during a respawn wait | WALL | sim-clocked for active waits (superseded by `PLAN_SIM_DEADLINE_S`); residual use bounded by `RESPAWN_WAIT_CAP_S` |
| `RESPAWN_WAIT_CAP_S` (env, default 45) | wait for a move_group PROCESS to respawn + re-advertise | WALL (backstop) | wall-backstop — **positive evidence**: the action server vanished (SIGABRT); respawn is OS/GPU-init bound, not render-paced |
| `TRANSIT_SENSING_PAUSE_WALL_S` (env, default 1200) | reposition carry-drive sensing-pause bound (transit only, default-off path) | WALL (backstop) | wall-backstop — nav drive watchdogs own the drive; generous; not on the cold cycle |
| `PLAN_TOTAL_ATTEMPTS_CAP` (env, default 10) | GPU-OOM count cap on cuMotion dispatches/move | count | count (direct GPU bound; time-independent) |
| `REPLAN_MAX` / `REPLAN_MAX_JOINT_SPACE` / `WRIST_FLIP_MAX` / `MOVE_GROUP_CRASH_RETRY_MAX` | re-plan / seed / crash-retry budgets | count | count |

### `restock_manipulation/controller.py`

| symbol / env | bounds | clock | status |
|---|---|---|---|
| `phase_timeout` (host passes 120) | per-phase sim budget (`self.elapsed` accrues `dt`) | SIM | sim-clocked |
| `max_age` / `RESTOCK_SENSOR_MAX_AGE_S` + `render_lead` | sensor-packet freshness window | SIM | sim-clocked |
| `grasp_press_timeout_s` | pre-weld press budget | SIM | sim-clocked (PR 203) |
| `collision_stale_tol_s` | held-carton collision-stamp staleness | SIM | sim-clocked |
| `contact_stuck_wall_s` / `collision_pause_wall_s` / `lift_collision_grace_wall_s` (all default 0 = off) | contact/collision progress bounds | **SIM** | sim-clocked — named `_wall` but `ros_host.py` injects `steady_clock = node.get_clock().now()` (the sim clock); documented clock-name caveat |
| `grasp_confirm_ticks` / `grasp_wait_ticks` / `collect_source_evidence_wait_ticks` | tick-count budgets | count | count |

### `restock/trajectory_host.py`

| symbol / env | bounds | clock | status |
|---|---|---|---|
| `FEEDBACK_SIM_STALE_S` (env, default 0.2) | measured-feedback staleness | SIM | sim-clocked; paired with the backstop below |
| `FEEDBACK_WALL_BACKSTOP_S` (env, default 30) | frozen-sim feedback backstop | WALL (backstop) | wall-backstop |
| `GOAL_SETTLE_MARGIN_S` / `goal_settle_budget` / `MOVE_SIM_DEADLINE_SCALE` | terminal-settle window past trajectory end | SIM | sim-clocked |
| execute-loop completion/overrun (`elapsed` vs `plan.times[-1]`) | execution completion | SIM | sim-clocked (PR 226) |
| `EXECUTE_WALL_BACKSTOP_S` (1800) | frozen-sim hard ceiling on the execute loop | WALL (backstop) | wall-backstop — 10x the worst render-paced execution; paired with the SIM terminal branch |
| `_EXECUTOR_TRANSIENT_BUDGET` | executor re-entry count | count | count |

### `restock/native_transport.py` (arm-command heartbeat)

| symbol / env | bounds | clock | status |
|---|---|---|---|
| `ARM_HEARTBEAT_SIM_S` / `ARM_HEARTBEAT_SIM_BACKSTOP_S` (env, default 8) | arm-command sim freshness + sim liveness | SIM | sim-clocked (PR 204); `pace_watchdog.is_dead(mode='either')` |
| `ARM_HEARTBEAT_WALL_BACKSTOP_S` (env, default 30) | frozen-sim arm-heartbeat backstop | WALL (backstop) | wall-backstop |

### `restock/native_channel.py` (StationClient — station ack / parking gate)

| symbol / env | bounds | clock | status |
|---|---|---|---|
| `_STAGE_UNAVAILABLE_S` (env, default 3) | supervisor-liveness PRIMARY (stage/sim clock) | SIM | sim-clocked |
| `_STATION_ACK_FROZEN_S` (env, default 2) + `_STATION_ACK_FROZEN_WALL_S` (env, default 45) | frozen-ack-stamp watchdog | SIM+WALL (`mode='both'`) | sim-clocked (needs BOTH; PR 204 family) |
| ack-freshness band (`now - ack.stamp` in the stage clock) | parked-station ack freshness | SIM (stage) | sim-clocked |
| `_HEARTBEAT_UNAVAILABLE_S` (env `RESTOCK_HEARTBEAT_UNAVAILABLE_S`, default 10) | no-reply hard catch | WALL (backstop) | wall-backstop — paired with the SIM-stage primary above; **default 10 s is real-rig-tight, MUST be widened under pacing** (set in the proof fragment). Left unchanged: load-bearing for the parking gate, no evidence of false-trip in coldlive 1–4 |
| `_HEARTBEAT_WALL_S` (0.3) | heartbeat SEND cadence (not a liveness bound) | WALL | cadence (a send rate, not a timeout) |

### `restock_manipulation/ros_payload.py` (payload / attach handshake)

| symbol / env | bounds | clock | status |
|---|---|---|---|
| `_wait` ack budget — `PAYLOAD_ACK_SIM_S` (env `RESTOCK_PAYLOAD_ACK_SIM_S`, default = wall budget) | grasp/detach + obstacle-service future ack | **SIM**+WALL | **converted**; SIM-primary poll loop, derived frozen-sim wall backstop |
| `PAYLOAD_ACK_WALL_S` (env, config `payload_ack_wall_s`, default 1 / sim 15) | wall floor feeding the ack backstop | WALL (backstop) | **converted** (role → backstop floor) |
| `RENDER_LEAD_S` (0.02) | tf freshness future-lead tolerance | SIM | sim-clocked |
| `add_base_obstacle` / `add_world_obstacle` `wait_s` (30) | ROS service-availability + call wait | WALL | service-availability wait at setup (the `_wait` future poll is now SIM-primary; the `wait_for_service` leg is infra readiness, not a mission-pacing timer) |
| executor `shutdown(timeout_sec=2)` / `thread.join(2)` | teardown | WALL | teardown (not a mission timer) |

### `restock_manipulation/isaac_ros_graph.py` (Isaac-side status watchdog)

| symbol | bounds | clock | status |
|---|---|---|---|
| `self.timeout` (5) via `simulation_time` | host process/status liveness | SIM | sim-clocked |
| `wall_backstop` (180) | frozen-sim status backstop | WALL (backstop) | wall-backstop |
| gripper acquisition `valid_until_steady` | gripper-command acquisition freshness | SIM | sim-clocked |

### `restock/runtime.py` (mission-runtime manipulation/nav watchdogs)

| symbol / env | bounds | clock | status |
|---|---|---|---|
| `nav_stuck_sim_s` (env) + `nav_stuck_wall_s` (env) | no-base-progress nav-stuck | SIM+WALL | sim-clocked (`pace_watchdog.is_dead`) |
| `_SENSOR_STALL_WINDOW_S` (20) | reach-stall fail-fast | SIM | sim-clocked |
| `_UNCOUPLE_SETTLE_MAX_S` (30) | tow-uncouple settle | SIM | sim-clocked |
| `staged_settle_s` → `SettleHold` | staged-start settle hold | SIM | sim-clocked |
| `_WARM_MAX_TICKS` / `_REWARM_MAX_TICKS` / `_WARM_SETTLE_GRACE_TICKS` | collision-map warm-up caps | count | count |

## Switch keys (all SIM-only env; unset ⇒ real-rig default ⇒ byte-identical)

| key | default | role |
|---|---|---|
| `RESTOCK_PLAN_TOTAL_SIM_CAP_S` | = `PLAN_TOTAL_WALL_CAP_S` (22) | total-move planning SIM budget (new) |
| `RESTOCK_PLAN_WALL_CAP_S` | 22 | wall floor for the planning frozen-sim backstop (back-compat; a stale sim-widen now only widens the backstop) |
| `RESTOCK_PLAN_WALL_BACKSTOP_PACING_FACTOR` | 50 | derives the frozen-sim wall backstop (> worst pacing) |
| `RESTOCK_STARTUP_GRACE_SIM_S` | = `STARTUP_GRACE_S` (5) | first-frame SIM grace (new) |
| `RESTOCK_PAYLOAD_ACK_SIM_S` | = `payload_ack_wall_s` | attach-ack SIM budget (new) |
| `RESTOCK_HEARTBEAT_UNAVAILABLE_S` | 10 | station no-reply wall backstop — widen under pacing |

## CPU proof

[`scripts/twin/restock/host_pacing_harness.py`](../scripts/twin/restock/host_pacing_harness.py) drives the
shipped SIM-primary + frozen-sim-wall-backstop logic (and the real `RosMotionPort.planning_wall_exceeded`)
against a fake wall/sim clock pair at 1x, 16x and 30x. [`test_host_timers_pacing.py`](../scripts/twin/tests/test_host_timers_pacing.py)
asserts: no false-trip of a healthy producer at any pacing (incl. the exact coldlive sample-4 shape), a
genuine hang caught on the SIM leg in bounded SIM time, a frozen sim caught on the WALL backstop, and
byte-identical 1x behaviour. The updated `test_ros_adapter_watchdog.py` / `test_ros_adapter_branch.py`
carry the per-timer regression guards.
