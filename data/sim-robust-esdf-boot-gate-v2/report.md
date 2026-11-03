# sim-robust-esdf-boot-gate-v2

Establish **what the boot ESDF-incorporation gate actually saw** on its 2026-10-24 GPU proof,
make its validity predicate provably mean "the ESDF really incorporated", add a once-per-op log so
the gate's PROCEED decision is never again invisible, and ship a proof config that genuinely
exercises **both** the one-tick hold and the named fail-fast TRIP.

Supersedes the facts/proof sections of `data/sim-robust-esdf-boot-gate-v1/report.md` (the gate code
from PR #248 is kept; only its claims about what the stale-baseline run reproduced were wrong). Out
of scope (owned elsewhere, per the brief): the `collision_dt` band, the station-liveness gaps, and
*why* a stale/uncommissioned baseline incorporates late.

## 1. The facts — where `collision_time` comes from, and what the gate saw

### 1a. The source chain (traced on CPU, no fabricated `>= 0`)

The arm host's `sensors.collision_time` (the only value the gate and the `collision_dt` invariant
read) is produced by this chain, every hop verified in the tree at head `bd06819`:

1. **`restock_host/incorporation_ack.py::IncorporationAckState.acknowledge`** — the ONLY writer that
   advances `collision_time` past `-1`. It takes `preflight.incorporated_time(nvblox response)`, a
   stamp that must be a non-empty, fresh ESDF integration strictly after the parked epoch. Input
   receipt, warm-up/empty/stale responses never advance it.
2. **`IncorporationAckState.snapshot`** publishes `collision_time = self.collision_time if ready else
   -1.`, where `ready` additionally requires `collision_time > parked_since` and `0 <= now -
   collision_time <= 0.5`. So the wire (`/restock_sim/station/status`) carries `-1` unless a genuine
   fresh integration exists.
3. **`restock/native_channel.py::StationClient.collision_time`** returns `latest['collision_time']`
   only when `ready(now)` (state `ready` + both stamps fresh), else `-1`.
4. **`restock/ros_manipulation.py::RosManipulationRuntime._pump`** mirrors it:
   `feedback.collision_time = self.station.collision_time`. Reset to `-1` on stop/pause.
5. **`restock_manipulation/isaac_factory.py::hold_last_good_collision`** — `read < 0` emits the last
   genuinely-measured `>= 0` (never fabricated; `None` until a real one is seen, so it emits `-1`
   until the first real stamp), `read >= 0` passes through.
6. **`restock/staged_start.py::sync_packet_clock`** (staged runs only) — a negative `collision_time`
   passes through **untouched**; a real `>= 0` is only *shifted* into the packet clock
   (`collision - offset`). It never turns `-1` into `>= 0`.

**Conclusion.** In every current deployment the arm-host path produces a finite `collision_time >= 0`
**only** from a real nvblox integration stamp. There is no default, wall stamp, or fabricated `>= 0`
source. (The one value that is `now`-stamped, `isaac_motion.IsaacCadMotion.collision_time =
simulation_time`, is the **retired** in-process CAD path — it is not used in the host-stack live_pose
runs, and even there it is only set after a successful GPU ESDF update, i.e. still "incorporated at
least once".) So the gate's predicate — `collision_time` leaving the `-1` sentinel for a finite
`>= 0` — is a faithful "first valid incorporation ack", and `>= 0` is exactly the handoff point to
the `collision_dt` freshness invariant. **This is reading (i); reading (ii) is ruled out.**

### 1b. What the three run dirs actually show

`~/datasets/twin-robot-in-store-boot-v1/mission/{meshuris1,meshuris0,esdfgate1}/host-stack.log`,
all staged transfer-only (`--start-at transfer --start-staging recorded_bay_park --stop-after
verify`, live_pose, `staged_start_clock_sync=True`):

| run | baseline | gate | ack-node `incorp reject` lines | controller | outcome |
|-----|----------|------|-------------------------------|------------|---------|
| meshuris1 | stale `cc-bench1` | OFF | 3 × `resp_stamp=0.000 now≈1.85–1.95` (early warm) | `CONTROLLER_INVARIANT_TRIP phase=observe now=39.000 collision_dt=40.0000` → `HOST_FAULT_ESTOP` | FAILED at the first observe step |
| esdfgate1 | **same** stale `cc-bench1` | ON (15 s) | 3 × `resp_stamp=0.000 now≈1.85–1.95` (same) | **one** `ESDF_BOOT_GATE_HOLD phase=observe now=39.000 collision_time=-1.000`, then PROCEED; the only `phase=` the controller logged is that single `observe`; no HOLD/TRIP/INVARIANT after | **completed at sim 174** |
| meshuris0 | fresh `cc-meshuris0` | OFF | 3 early + 1 × `resp_stamp=102.650 now=103.250 dt=0.600` (a late stale poll) | none | completed |

Three facts settle the reading:

- **`incorp reject` only logs *rejects*** (budget `diag_left=6`); `acknowledge` success is silent. All
  three runs logged ≤ 4 rejects, so "no more reject lines" means nothing — it does NOT prove "never
  incorporated". The v1 report read meshuris1's absence-of-later-rejects as "never incorporated"; that
  inference is unsound.
- **esdfgate1 ran to sim 174 with ZERO `collision_dt` trips** (no `CONTROLLER_INVARIANT_TRIP`, no
  `COLLISION_STALE_PAUSE`). The `collision_dt` invariant fails within `max_age` (1.5 s) of any frozen
  stamp, so surviving ~135 s of real cuMotion planning is only possible if `collision_time` **advanced
  continuously with `now`** — i.e. the ESDF genuinely and repeatedly incorporated. The stale baseline
  **does** incorporate.
- **meshuris1 and esdfgate1 use the identical stale baseline** and both read `-1` at the first observe
  step (now=39). The only difference is the gate: meshuris1's invariant killed the run one controller
  step before the (one-tick-late) incorporation could land; esdfgate1's gate HELD exactly that one
  tick and then saw the fresh stamp. meshuris1 was never shown to "never incorporate" — it died too
  early to find out.

**Verdict — reading (i).** The stale baseline incorporates **one tick late** (right at/after the op's
first observe step), not *never*. The gate's bounded HOLD absorbs that one-tick boot race — which is
real and valuable (it converts meshuris1's fatal early trip into esdfgate1's completed run). But the
stale-baseline run is **not** a never-incorporation reproduction, so PR #248's Run A did **not**
exercise the fail-fast TRIP path. That path was unproven.

## 2. The fix (reading (i): keep the gate, make the facts honest + visible)

- **Predicate unchanged, now documented as correct.** `EsdfBootGate._ack_valid` stays "finite and
  `>= 0`": §1a proves that is exactly "incorporated at least once" and the correct handoff to the
  `collision_dt` invariant. No `collision_dt`-invariant input change is needed (reading (ii) is ruled
  out). Byte-identical when the key is off (`grace_s <= 0` ⇒ always PROCEED, one-shot never set).
- **Once-per-op transition log** (`controller.py`): on the step the gate first sees a valid ack it
  prints `ESDF_BOOT_GATE_READY phase=… now=… collision_time=… (first valid nvblox incorporation ack
  … source=sensors.collision_time packet mirror; collision_dt invariant now owns freshness)`. A new
  one-shot `EsdfBootGate.consume_just_satisfied()` (set inside `observe` on the `False→True` flip)
  drives it, so it fires exactly once whether or not a HOLD preceded it. Scoped to the gate being
  **enabled**, so a gate-off run stays byte-identical on the console too. This makes the PROCEED
  decision — the value, the sim `now`, the source — never again invisible (esdfgate1 logged only the
  HOLD; the proceed was silent).
- **Docstrings corrected** in `esdf_boot_gate.py` to describe both cases the gate serves (the one-tick
  boot-race absorb that the GPU proof actually exercised, and the genuine never-incorporation TRIP),
  with the measured meshuris1/esdfgate1 evidence.

## 3. A genuine never-incorporation reproduction for the TRIP proof

Added an opt-in, default-off **proof switch** `RESTOCK_ESDF_FORCE_NEVER_ACK` read by
`IncorporationAckState` (`incorporation_ack.py`). When set, `acknowledge()` refuses every stamp
through the **real** reject path, so `collision_time` never leaves the `-1` sentinel for the whole run
— a faithful "nvblox genuinely never incorporates" from the arm host's view (exactly what the gate
keys on), without touching cuMotion/nvblox or the boot path. Unset ⇒ byte-identical; a boot line
`ESDF_FORCE_NEVER_ACK enabled: …` names it when on (the host/runtime-split lesson). This is the
reproduction the stale baseline is NOT. (Equivalent physical repros — a disabled masked-depth feed or
a dead mapper — reach the same stuck `-1`; this switch is the surgical, CPU-testable one.)

Hand-off fragment: `docs/examples/esdf-boot-gate-v2-mission-fragment.json` (integration-lane GPU
runs, not run here):

- **Run A (TRIP)** — stale *or* fresh baseline, `esdf_boot_ack_grace_s=15.0`, env
  `RESTOCK_ESDF_FORCE_NEVER_ACK=1`. PASS = `ESDF_BOOT_GATE_HOLD` lines through the grace, then the
  named `ESDF_BOOT_GATE_TRIP … nvblox ESDF incorporation never acknowledged within ~15s sim boot
  grace …` BEFORE any arm motion, and NO `CONTROLLER_INVARIANT_TRIP … collision_dt=40.0`.
- **Run B (healthy)** — fresh commission, `esdf_boot_ack_grace_s=15.0`, switch unset. PASS = at most a
  one-tick `ESDF_BOOT_GATE_HOLD`, then exactly one `ESDF_BOOT_GATE_READY`, then unchanged
  transfer+verify completion, no `collision_dt` trip.

## 4. CPU proof

`scripts/twin/tests/test_sim_robust_esdf_boot_gate.py` (16 tests, `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
PYTHONPATH=. python3 -m pytest …`): disabled byte-identical; the stuck-stamp HOLD→named-fault; the
one-tick-late hold-then-proceed with the `collision_dt` invariant silent; the once-per-op
`ESDF_BOOT_GATE_READY` transition (and silence when off); any finite `>= 0` disarms + a transient `-1`
never re-arms; NaN keeps holding; and the REAL `IncorporationAckState` under
`RESTOCK_ESDF_FORCE_NEVER_ACK` leaving `collision_time` stuck at `-1` (`state=warming`) driven through
the real controller to the named TRIP. No Isaac / ROS / GPU.

## 5. GPU

Not run here (the integration/validator lane holds the GPU). Both runs ride that lane through
`/tmp/aile-gpu.lock`, never forced, never touching `whispr`. Fragment §3 is the hand-off.
