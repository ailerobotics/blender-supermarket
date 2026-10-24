# sim-robust-esdf-boot-gate-v1

> **CORRECTED by `data/sim-robust-esdf-boot-gate-v2/report.md` (2026-10-24).** The gate code in this
> PR is kept, but §1's claim that the stale `cc-bench1` baseline (meshuris1) reproduces a
> *never-acknowledged* ESDF is **wrong**. The GPU proof `esdfgate1` (same stale baseline, gate on)
> held exactly ONE tick then ran to completion at sim 174 with zero `collision_dt` trips — i.e. the
> stale baseline incorporates **one tick late**, not never; meshuris1 died one controller step before
> it could land. So this PR's Run A proved the gate's one-tick-hold boot-race absorb, **not** the
> fail-fast TRIP. The genuine never-incorporation TRIP proof, the once-per-op `ESDF_BOOT_GATE_READY`
> transition log, and the `RESTOCK_ESDF_FORCE_NEVER_ACK` reproduction are in v2. Read v2 for the
> current facts and the `docs/examples/esdf-boot-gate-v2-mission-fragment.json` hand-off.

Name a **never-acknowledged nvblox ESDF incorporation at boot**, on the sim clock, **before the
first arm motion streams** — instead of letting it surface ~40 s into a staged run as an opaque
mid-stage controller invariant trip. The same failure would happen on the real robot if nvblox
never incorporated at boot, so it must fail fast and *by name*.

Out of scope (owned elsewhere, per the brief): *why* a stale/uncommissioned baseline fails to
incorporate, the `collision_dt` invariant threshold, and the station-liveness gaps
(`RESTOCK_STATION_SIM_GAP_S` / `RESTOCK_STATION_WALL_GAP_S`).

## 1. The distinguishing signal (established on CPU from the two host-stack logs, before any code)

Two staged transfer-only runs at head `ccc479e`, same bench
(`--start-at transfer --start-staging recorded_bay_park --stop-after verify`, live_pose):

- **meshuris1** (`~/datasets/twin-robot-in-store-boot-v1/mission/meshuris1/host-stack.log`) — FAILED.
  Reused the stale `cc-bench1` commission baseline.
- **meshuris0** (`.../mission/meshuris0/host-stack.log`) — PASSED (transfer + verify complete).
  Fresh `cc-meshuris0` commission.

The signal that separates them in the first seconds is the **incorporation-ack `resp_stamp`**
published by `restock_incorporation_ack` (and mirrored to the arm host as the station
`collision_time`):

| run | first ticks (now≈1.85–1.95) | later | arm host `collision_time` | outcome |
|-----|------------------------------|-------|---------------------------|---------|
| meshuris1 | `resp_stamp=0.000` (3 throttled `incorp reject` WARNs) | **never leaves 0.000** | stays `-1` (not-ready sentinel) | `CONTROLLER_INVARIANT_TRIP phase=observe now=39.000 collision_dt=40.0000` → `HOST_FAULT_ESTOP 'stale collision world'` → arm heartbeat lost |
| meshuris0 | `resp_stamp=0.000` (same early rejects) | **`resp_stamp 0.000 → 102.650`** (first valid `dt=0.600` by `now=103.250`) | advances to a real stamp (`> parked_since 1.667`) | transfer + verify complete, no `collision_dt` trip |

Both runs log the *identical* early reject (`resp_stamp=0.000 now≈1.85–1.95`). The divergence is
purely **whether `resp_stamp` ever advances past `0.000`**: meshuris0's does (→102.650,
`dt=0.600`), meshuris1's never does.

**Which component sees it first.** `restock_incorporation_ack` sees the raw nvblox response
`resp_stamp` first (its `incorp reject` WARN at `now≈1.85`), but it only *rejects* — it never
advances `collision_time` and nothing fails on the stuck stamp; it is a pure observer/publisher.
The **station liveness** (`restock.station.snapshot`) is a heartbeat/freshness guard, not an
"ever-incorporated" guard — it reports `warming` while unready, never a boot fault. The first
component to actually *fail the run* on the stuck stamp is the **controller `collision_dt`
invariant** (`restock_manipulation/controller.py`), and only at the first observe step of the
transfer op (`now=39`): with `collision_time` frozen at `-1`, `collision_dt = now - (-1) = now+1`
is `40.0` at `now=39`, outside the `[-render_lead, max_age]` band. So the only thing that ever
names the never-incorporated ESDF is a mid-stage invariant with an opaque reason, ~37 s after
boot (the op does not begin until `now≈39` after the staged-transfer warm-up/settle; the arm host
simply runs on until then).

**Conclusion.** The boot signal the gate must key on, from the arm host's own view, is
`sensors.collision_time` **leaving the `-1` not-ready sentinel** for a real integration stamp
(`>= 0`; the station only publishes a finite stamp once the ESDF has incorporated fresh and
strictly after the parked epoch, so "ever left `-1`" is a faithful "first valid incorporation
ack"). Freshness *after* that first valid ack stays the `collision_dt` invariant's job.

This is consistent with the mesh-URI isolation already in `meshuris1-evidence.md` /
`repeatability-ledger.md`: PR246/247 only rewrites move_group's `robot_description`, not nvblox,
so the stuck `resp_stamp` is an ESDF-incorporation-freshness failure, not a mesh-URI regression.

## 2. The gate (opt-in HOST key `esdf_boot_ack_grace_s`)

Pure state machine `restock_manipulation/esdf_boot_gate.py::EsdfBootGate`, driven from
`TransferController.step` at the top of each op (before the `collision_dt` invariant, before any
motion is streamed). Wired through the HOST config key **`esdf_boot_ack_grace_s`** (sim seconds;
registered `HOST` in `config_preflight.KEY_CONSUMER`; read by `ros_host.py` and logged in
`HOST_EFFECTIVE_CONFIG`). **`<= 0` (absent) ⇒ disabled ⇒ byte-identical.** A positive value is the
grace (recommended ≈15 s).

Semantics, from the first controller step of an operation, on the **sim clock** (`now =
sensors.simulation_time`):

- a valid incorporation ack has arrived (`collision_time >= 0`, finite) ⇒ **disarm permanently**,
  PROCEED; the `collision_dt` invariant is thereafter the sole owner (band untouched);
- no valid ack yet and `now - op_start <= grace` ⇒ **HOLD** — return a running result that streams
  no motion and does not advance the phase (the arm stays parked; incorporation proceeds
  independently because it is driven by the mapper/masked-depth feed, not by the controller);
- no valid ack and `now - op_start > grace` ⇒ **FAIL FAST** by name, the reason naming the elapsed
  grace, the stuck `collision_time`, and the likely causes (nvblox not incorporating; stale /
  uncommissioned baseline).

Because every currently-passing run already has a fresh `collision_time` at the first stepped
phase (otherwise the `collision_dt` invariant would already trip it), the gate **disarms on the
first step** in all healthy runs — so enabling it does not change their behaviour. It changes
behaviour only in the failure case: it pre-empts the opaque `collision_dt=40` invariant trip with
a HOLD + a named boot fault before any arm motion.

## 3. CPU tests (`scripts/twin/tests/test_sim_robust_esdf_boot_gate.py`)

Both traces are modelled from the logged `collision_time` signal above and driven through the real
`EsdfBootGate` and the real `TransferController.step` seam (no Isaac/ROS/GPU):

- **meshuris1 trace** (`collision_time` stuck at `-1` from op start) fails by name, bounded by the
  grace (at ≈`op_start + grace`, not the ~40 s invariant), the reason naming the grace + stuck stamp;
- **meshuris0 trace** (`collision_time` already valid at op start) passes — the gate disarms and
  never fires, the normal `collision_dt` invariant owning freshness;
- **key off** (`esdf_boot_ack_grace_s` absent / `<= 0`) is byte-identical: the controller takes the
  exact legacy path (the opaque `collision_dt` invariant still trips the stuck-stamp trace);
- the grace is **sim-clock, not wall-clock**: advancing wall time with the sim clock frozen never
  trips; advancing the sim clock past the grace trips.

## 4. Bounded GPU proof (handed to the integration lane; not run here)

`docs/examples/esdf-boot-gate-mission-fragment.json` — the HOST key fragment to merge into the
staged transfer-only mission config for two bounded runs (the validator holds the GPU):

1. **Stale-baseline (deliberately reuse `cc-bench1`), key on** — PASS = the named boot fault fires
   **before any arm motion** and **no 40 s `collision_dt` invariant trip** (the gate HOLDs then
   fails by name instead).
2. **Fresh-commission staged transfer, key on** — PASS = unchanged completion with the gate silent
   (it disarms on the first valid ack).

GPU rule honoured: one run at a time via `flock /tmp/aile-gpu.lock`, never forced; the always-on
`whispr` service (pid 1339, ~2.2 GB VRAM) is untouched; the mission runtime configs under
`~/datasets` are not edited here — the fragment is the hand-off.
