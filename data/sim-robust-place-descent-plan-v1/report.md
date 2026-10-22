# sim-robust-place-descent-plan-v1 — the collect place descent completes past the station parking gate

**Branch:** `fm/sim-robust-place-descent-plan-v1` (into `fm/twin-robot-in-store-boot-v1`).
**Date:** 2026-10-22. CPU-proven; one bounded GPU proof handed to the integration lane (collect3 staging unchanged).

## Problem

With PR 209 (upright preplace branch) and PR 211 (`collect_place_base_settle_m` base-settle tolerance)
active, the validator's run **collect3** reached the UPRIGHT preplace over the rollwagen deck with the
carton held (z≈0.608) and began the place descent — but the descent never completed; no
place/release/retreat/stow phase followed and the run ended `arm command heartbeat lost`.

Run: `~/datasets/twin-robot-in-store-boot-v1/mission/collect3/` (`host-stack.log`, `collect3.console.log`,
`mission.json`, `nav-trace.jsonl`). Evidence: `data/twin-robot-in-store-boot-v1/collect3-pr211-evidence.md`.

## Root cause (reconstructed from the console + mission.json)

`mission.json`: collect attempt 1 reached step 5 `place` (SIM 94.75) and failed at SIM **118.45** with
reason **`station lost measured parking`** — NOT collect2's `base moved during stationary operation`
(the reason PR 211 fixed). `collect3.console.log` pins the exact tick:

```
COLLECT_GT ... held=True pose=(0.916, 0.198, 0.608, ...)   # carton at the upright preplace, descending
SEAL_DIAG ... min_gap_mm=31.6 ... hit_bodies=['/Rollwagen/chassis/shelf']   # gripper gap closing
SEAL_DIAG ... min_gap_mm=22.98 ...                                          # 31 -> 23 -> 18 -> 16 mm
BASE_DISP base_dxy=25.7mm base_dyaw=0.01mrad ... now=118.45                 # the descent-reaction shove
MANIP_STEP_EXCEPTION op=collect ValueError: station lost measured parking
PHASE_DRIVE phase=host -> ...                                               # the FAILED result's phase label
```

So the place descent WAS running (gripper gap 31→15 mm) when the extended, carton-loaded arm's reaction
shoved the yaw-lock-only braked base **25.7 mm** — well inside PR 211's 0.08 m (80 mm) tolerance. The
`station lost measured parking` ValueError is raised at `scripts/twin/restock/native_channel.py:280`
(`StationClient.tick`: `if not parked: self.stop(); raise ValueError('station lost measured parking')`),
caught in `ros_manipulation.py` (printed as `MANIP_STEP_EXCEPTION`, returns `Result('failed','host',…)` —
**that `'host'` phase label is the "phase=host" in the evidence; the host controller never took over**),
then `stop(close_sensing=False)` halts the stage. The arm never places/releases/stows → every re-begin
fails `arm_not_in_transport` → `arm command heartbeat lost`.

**Why PR 211 did not catch it.** `packet.parked` has TWO independent consumers:

1. the mission-level stationary guard in `runtime.py` (`if not p.parked:`) — PR 211 relaxed this with
   `_loaded_place_base_excursion` + `collect_place_base_settle_m`;
2. the **station channel's own parking gate** — the sensor-owning host runtime feeds `packet.parked`
   straight into `StationClient.tick(parked=…)` (`ros_manipulation._pump` → `feedback.set_parked` →
   `station.tick`), which raises on `not parked`.

PR 211 fixed (1) but left (2) getting the raw `packet.parked`. On the tolerated tick the mission guard
falls through (line ~1000) and then `self.manip.step(dt, p)` passes the raw `parked=False` to the station
gate, which raises on the **identical transient** the mission guard just tolerated.

## Fix (no new key; reuses `collect_place_base_settle_m`; default-off ⇒ byte-identical)

`scripts/twin/restock/runtime.py`, in `_step`'s stationary-manipulation branch: when the loaded-place
base-settle tolerance accepts the shove (`place_shove_tolerated`), present `self.manip.step` the SAME
effective-parked verdict the mission guard reached — `replace(p, parked=True)` — so the station/host
parking gate (and the capture's `motion.parked()` via `set_parked`, the designed override seam) sees the
tolerated value instead of raising. The override is scoped exactly to the already-tolerated loaded collect
place (`_loaded_place_base_excursion` is collect-scoped, repositioned, loaded-place-phase, anchor + base
pose present, within bound); everything else passes `p` through unchanged (byte-identical). A runaway shove
still fails with PR 211's NAMED reason BEFORE `manip.step`, so the station gate never sees a runaway.

No new config key: the fix is wired to the existing PR 211 `collect_place_base_settle_m`. The scripted/still
transfer path is untouched (the tolerance is collect-scoped).

## CPU proof

`tests/test_restock_runtime.py` (new, on the real `owns_sensor_io=True` bridge path that actually feeds
`parked` into the controller):

- `test_loaded_place_feeds_effective_parked_to_the_station_gate` — a within-bound shove (raw
  `parked=False`) reaches a controller that models `native_channel`'s gate as `parked=True`; the gate does
  NOT raise and the place keeps running.
- `test_loaded_place_without_tolerance_station_gate_never_fed_the_shove` — key off: the strict mission
  guard fails `base moved…` first, the gate is never fed the bad parked (byte-identical).
- `test_loaded_place_beyond_tolerance_does_not_reach_the_station_gate` — a runaway shove fails with the
  PR 211 named settle-tolerance reason before `manip.step`.
- `test_loaded_place_parked_true_passes_through_unchanged` — a genuinely parked base passes `parked=True`
  straight through (common case, byte-identical).

Plus the PR 211 `test_loaded_place_*` and the full `test_restock_runtime.py` / `test_restock_cart_place_goal.py`
/ `test_restock_manipulation.py` / `test_restock_ros_runtime.py` / `test_restock_staged_start.py` suites
stay green.

## Bounded GPU proof (handed to the integration lane)

`docs/examples/cart-place-descent-plan-mission-fragment.json` — the **collect3 staging unchanged** (no new
key). Expected vs collect3: the place descent completes onto the deck without `station lost measured
parking`, the gripper-open detaches the weld, the ground-truth verify shows the carton on the deck, the arm
stows to transport, `transport_ready` becomes True, couple can begin.
