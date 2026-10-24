# sim-robust-vram-headroom-v1 — real VRAM headroom + crash-safe mission recording

**Lane:** make the Isaac restock capstone less prone to the two failures the repeatability ledger
exposes near the 12 GiB ceiling: (1) a late native crash when VRAM creeps within ~500 MiB of the
card top, and (2) the loss of `mission.json` when that crash is a native SIGSEGV (the record-first
teardown never runs). Base: `fm/twin-robot-in-store-boot-v1` @ `44c8c33` (after PRs 243–254).

Captain's intent (2026-10-22): "keep working on the simulation on the side to make [it] less prone
to failure in any case." Trigger: **coldwin13** — byte-identical to the passing coldwin11, fresh
commission, no injection — died in a native SIGSEGV (exit 139, `carb.crashreporter` Fatal, core
dumped) during the **transfer** approach at sim t≈241.7, base x=−2.026, with sampled VRAM climbing
to 10470 MiB used / 1438 free. Because the fault was native, the run has **no `mission.json`**, only
`host-stack.log`, `nav-trace.jsonl`, `bt_events.jsonl`, and a re-encoded `walkthrough.mp4`.

Validator steering (inbox 001, 2026-10-24): do **not** assume a strict ceiling OOM. The 20 s
orchestrator sampler can miss a transient allocation spike, so treat **three live hypotheses** —
(a) a sustained ceiling creep, (b) a transient spike above the sampler cadence, (c) a native fault
unrelated to memory — and make the in-process monitor sample **sub-second** so it can see a spike,
and put the **last sampled headroom** into the crash record.

---

## 1. VRAM footprint accounting (CPU, from the recorded evidence + the code)

Card: NVIDIA RTX 3060, **12288 MiB** total, shared with the always-on **whispr** service (pid 1339,
~**2252 MiB** resident — never touched). All figures are MiB *used*.

### Component budget

Anchored to the `gpu_budget.py` transfer2 measurement (2026-10-22: 11882 used incl. whispr ⇒
mission ≈ 9630) and the `mission-vram-ceiling-12gb` forensics (runs 297/299, fm373–fm380).

| Component | ~MiB | Where | Grows over a mission? |
|---|---|---|---|
| whispr (resident, external) | 2252 | captain service, pid 1339 | no (constant; must never be touched) |
| Isaac RTX render + kit core | part of base | `SimulationApp`, 960×540 app buffer (`RESTOCK_RENDER_WIDTH/HEIGHT`) | **yes — the dominant creep** |
| Station decode cameras (source/cart/shelf, 2560×1920 for QR) | ~650 each; shelf ~948 | `sensing_sources`/`native_cameras` render products | no (created at boot; constant) |
| nvblox ESDF + TSDF | part of base | host stack; aabb [1.6,1.6,1.6] | **no** — fm375: bounded + constant, no second map |
| cuMotion planner | part of base | host stack (move_group / curobo) | no (plan buffers transient, bounded) |
| ROS host stack | part of base | separate processes, mostly host RAM | no |
| Arm eye-in-hand masked-depth feed (424×240 depth+seg) | ~436 | `masked_depth.py` wrist bridge → nvblox camera_0 | no (constant per-tick) |
| Mast masked-depth feed (848×480, opt-in, off) | ~455 | `mast_depth` → camera_1 | n/a (off; the 12 GB card can't fit a 2nd feed) |
| mission-frames walkthrough capture | ~0 GPU | `recording.py`; RGB copied to CPU, PNGs on disk | no (CPU/disk, not VRAM) |
| walkthrough H.264 encoder | 0 GPU | CPU ffmpeg at teardown | no |
| **RTX/Vulkan allocator pool (GI realloc + fragmentation)** | **+700…+2680** | internal Omniverse, after ~17 min continuous render | **YES — the sole unbounded term** |

`gpu_budget.py` model (advisory pre-boot gate): `BASE_MISSION ≈ 9144` (Isaac render + host + nvblox +
cuMotion + the resident station cameras, *no* eye-in-hand feed) `+ 1 arm feed ~486 (@1280×720 ref) / ~436
(@424×240 actual)` ⇒ mission ≈ 9.6 GB `+ whispr 2.25` ⇒ **~11.9 GB**, i.e. the card top with a sliver.

**Key forensic finding (memory `mission-vram-ceiling-12gb`, carried forward):** render *resolution*
is NOT the lever (480×270 ≈ 640×360); disabling RTX GI/DLSS (`RESTOCK_LOW_VRAM_RENDER`) saves only
~160 MiB; nvblox/cuMotion/cameras are all bounded + constant. The late growth is an **internal RTX
allocator event** (GI reallocation / pool fragmentation after a long continuous render) — not
attributable to any discrete mission allocation, so it is not bounded by mission code. This is why the
right mechanism is **freeing render-target VRAM proactively to give the allocator headroom**, not
hunting a mission-side leak.

### Per-stage / per-run peaks (repeatability ledger, fixed head)

| run | result | failing stage | VRAM peak (sampled, MiB) |
|---|---|---|---|
| coldwin7 | PASS (complete) | — (reached return) | 11795 |
| coldwin8 | PASS (complete) | — (reached return) | 11651 |
| coldwin9 | FAIL @ outbound | outbound (never transfer) | 10549 |
| coldwin10 | FAIL @ outbound | outbound | 10549 |
| coldwin13 | **native SIGSEGV @ transfer** | transfer approach | **10549 peak / 10470 at crash** |
| meshuris0/1 | (queue proofs) | transfer/observe | 10288 / 10289 |
| respawn2b/2c | (collect-only) | — | 10727 / 10725 |

Reading: the **completes peak highest (~11.7 GB)** because they render all the way through
transfer→verify→return (the longest continuous-render exposure → the most allocator creep); the
outbound-fails and staged proofs die earlier and peak ~10.3–10.7 GB. coldwin13 crashed mid-transfer at
10.5 GB sampled — but the 20 s sampler can miss a transient spike, so the **instantaneous** peak at the
fault may have brushed the ceiling. Headroom at the completes' peak is ≈ **490–640 MiB** of 12288; the
run genuinely operates within ~500 MiB of the top.

### coldwin13 crash timeline (from `coldwin13-evidence.md`)

```
used=6471 free=5437   (early)
used=9991 free=1917
used=10549 free=1359   <- peak sampled (transfer)
used=9214 free=2694    (dip)
used=10470 free=1438   <- last sample before SIGSEGV
used=2326 free=9582    (post-crash: whispr only — the mission freed everything)
```
Crashing process = the embedded Isaac mission **python** (breakpad stack frames 042–050 are the python
entry stack), exit 139, core dumped; the host-stack ROS nodes logged clean exits *after* the kit died,
so they are not the crasher. Stage at crash: **transfer**, sim t≈241.7, base x=−2.026, right after a
transfer pick-approach fault.

---

## 2. Headroom monitor (opt-in, `config["vram_headroom"]`, default OFF, byte-identical off)

`scripts/twin/restock/vram_headroom.py`. Registered like `sim_speed` / `masked_depth.economy`: a
dataclass with `from_config` (fails closed on a typo), `describe()` boot line, and `summary_fields()`
provenance merged into `mission.json` via `ground_truth_verify.mission_modes`.

- **Sampling** is on the **sim clock** (cadence in sim seconds, default `sample_period_s=0.2`, i.e.
  sub-second — honoring inbox 001). The reading itself comes from a cheap in-process sampler: **NVML**
  (`pynvml`) if importable, else a background `nvidia-smi` poller thread so the per-tick cost stays off
  the hot path. The sampler is injectable, so the policy is CPU-tested with a recorded timeline.
- **Shed order** (fixed, named; `shed_order` default `["capture_rate", "arm_feed_economy",
  "secondary_cameras"]`): when headroom drops below the configurable **floor** (`floor_mib`, default
  800) the monitor sheds the **next** still-available lever (a ratchet — never un-sheds, so it cannot
  thrash), logging each shed by name with the reading (`VRAM_SHED name=… headroom=… used=… sim=…`).
  1. `capture_rate` — widen the walkthrough record stride (recording, never an honest-stage input).
  2. `arm_feed_economy` — cap the arm masked-depth publish rate (`masked_depth_economy`) **only outside
     the nvblox-critical stages** (collect/transfer/verify); stage-blocked there, so it is skipped and
     the monitor escalates.
  3. `secondary_cameras` — destroy the station decode render products this run never reads
     (reuses the fm380/fm381 `stage_scoped_teardown`; ~948 MiB+ returned, VRAM-measured).
- **Never shed** a load-bearing input of an honest stage: the live_pose transfer arm feed, nvblox, or
  the planner. Stage-safety is enforced per shed; the transfer stage (where coldwin13 died) can only
  ever fire `capture_rate` and `secondary_cameras`.
- **Ceiling named by name:** if headroom stays below the floor after every available shed has fired,
  the monitor logs `VRAM_CEILING_REACHED …` once (so the cause is named, not silent) and records it.
- **Last reading is durable:** the monitor atomically writes `vram-headroom.json` (latest used/free/
  headroom + sim time + peak + fired sheds + ceiling flag) throttled to ~1 sim-s, so the **crash record
  can include the last sampled headroom** even on a native crash.

Off (absent/`false`) ⇒ no monitor is constructed and every path is byte-identical.

## 3. Footprint trim (accounting-exposed)

The accounting confirms the only safe, provably-invisible runtime trims are the already-committed,
default-off knobs — this lane does **not** change any default (the completes pass today; a default
flip would need its own GPU proof):

- `RESTOCK_DROP_UNUSED_SENSING` drops the **shelf** station camera when `transfer_shelf_location` is
  unset — ~**948 MiB** (fm379, measured), invisible because that camera is never read. Documented;
  default unchanged.
- `RESTOCK_STAGE_SCOPED_SENSING` destroys the collect-only station render products at the
  collect→couple boundary — ~1598 MiB for shelf+source (fm380, measured). The monitor's
  `secondary_cameras` shed reuses this exact teardown, so the trim becomes **pressure-triggered**
  rather than requiring the operator to pre-set the env knob.
- Render resolution / GI-DLSS (`RESTOCK_RENDER_WIDTH/HEIGHT`, `RESTOCK_LOW_VRAM_RENDER`): confirmed
  NOT material (≤160 MiB) — documented as *not* a lever so future lanes don't re-chase it.

No new default is flipped; the net new capability is turning the measured teardown into an automatic,
pressure-gated shed.

## 4. Crash-safe recording

A native SIGSEGV kills the in-process Isaac python before the `finally` that writes `mission.json`
runs. Two pieces make a crashed run still leave a verdict:

- **Incremental, durable progress** (`crash_record.ProgressWriter`): the mission appends one
  fsync'd line per stage transition to `mission-progress.jsonl` (begin/complete/fail + sim time +
  stage ladder). Append-only + fsync ⇒ the already-written lines survive a native crash; ~7 lines/run,
  never raises, and `mission.json` on the success path is **byte-identical** (a new sidecar only).
- **Out-of-process finalizer** (`scripts/finalize_crashed_run.py` → `crash_record.write_if_missing`):
  runs from the **shell parent** (`run_mobile_restock.sh`, which survives the python SIGSEGV) *after*
  the mission process exits. If the process died by signal (or wall-cap timeout) **and** `mission.json`
  is absent, it reconstructs the verdict from the durable artifacts — preferring `mission-progress.jsonl`,
  falling back to `nav-trace.jsonl` (last `action`,`t`) + `bt_events.jsonl` (ladder + last reason) so it
  also recovers **legacy** runs like coldwin13 that predate the progress writer — and writes
  `mission.json` with `state="crashed"`, `crash_signal`, `last_stage`, `sim_time`, the stage ladder
  (done/current/pending), and `last_vram` (from `vram-headroom.json`). The record-first teardown and the
  evidence tooling (`check_walkthrough_run.py`, the ledger) then run on it.

If `mission.json` already exists (clean success or clean failure), the finalizer is a no-op ⇒ the
normal path is untouched.

### CPU tests
- `test_restock_vram_headroom.py` — the policy's shed order + floor logic on a recorded VRAM timeline
  (incl. coldwin13's), stage-safety blocking, ceiling naming, and `from_config`/`summary_fields`
  (key off ⇒ byte-identical).
- `test_restock_crash_record.py` — `ProgressWriter` + `scan_progress` produce a valid `state=crashed`
  record from a **truncated** `mission-progress.jsonl`; recovery from `nav-trace.jsonl`+`bt_events.jsonl`
  alone (the coldwin13 shape); `write_if_missing` is a no-op when `mission.json` exists.

## 5. Bounded GPU proof (handed to the integration lane; do NOT run here)

`docs/examples/vram-headroom-mission-fragment.json`:
- **Run A** — full capstone with the monitor ON. PASS = `state=complete`, headroom never below the
  floor OR every shed logged by name, peak VRAM recorded in `vram-headroom.json`.
- **Run B** — staged transfer with the monitor ON, the Isaac process deliberately `kill -SIGSEGV`'d
  mid-stage. PASS = `mission.json` exists with `state=crashed` + the stage ladder + `crash_signal=11`,
  and the teardown/finalizer run clean.

GPU rule: one run at a time via `flock /tmp/aile-gpu.lock`, never forced, whispr never touched.

## 6. Scope

In: CPU accounting, the opt-in monitor + sheds, the pressure-gated reuse of the measured teardown,
crash-safe recording, CPU tests, the GPU proof fragment. Out: which stages are honest, planner/nvblox
settings beyond memory, the upstream Isaac crash itself, editing `~/datasets` runtime configs.
