# sim-robust-weld-attach-v1 — reliable simulated grasp weld + default-on fail-fast

Lane goal (firstmate / captain intent "make the simulation less prone to failure in any case"):
the simulated collect grasp (the labelled `SIM_ATTACH_FALLBACK` weld, used because native suction never
seals from the arm articulation) must **attach reliably whenever the pad is square on the carton within
a sensible gap**, and a **non-forming grasp must fail fast with a named reason** instead of looping into
the opaque `arm command heartbeat lost` e-stop.

## What the runs actually show (and a correction to the brief's premise)

Runs under `~/datasets/twin-robot-in-store-boot-v1/mission/` (`collect1`, `collectsrc`), console logs
beside them. The authoritative `bt_events.jsonl` timelines:

| run | grasp·lift·tuck (step 2) | drive→rollwagen (step 3) | place (step 5) | terminal |
|-----|--------------------------|--------------------------|----------------|----------|
| `collectsrc` | ✅ weld @ **0.50 mm** | ✅ t=72.8 | ❌ t=108.4 | host heartbeat silence (30.1 s) at **place** |
| `collect1`   | ✅ weld @ **0.48 mm** | ✅ t=66.2 | ❌ t=163.9 | `collision planning or execution failed` (cuMotion) at **place**, then `arm command heartbeat lost` on the place retries |

**Correction:** the brief's premise — that `collect1` *failed at the collect grasp-press* and *the weld
never formed* — does not match the logs. In BOTH runs the grasp weld **did** form and the arm lifted,
tucked and drove the carton to the rollwagen. Both runs then **failed at the PLACE-on-cart step**
(`collect1`: a cuMotion `collision planning or execution failed`; `collectsrc`: the host stack going
silent). Place-on-cart (the replan storm / cuMotion OOM and the host heartbeat) is the sibling lane
`sim-robust-collect-place-v1` / PR 200 — **not** this lane and not touched here.

**But the grasp *is* genuinely fragile, and the brief's class of failure is real** — it just happened in
a different run series. The sibling `data/sim-robust-collect-arrival-v1/report.md` records the honest
(non-ground-truth) runs **321 / 327**: the contact descent bottomed at `SIM_ATTACH_DIAG ... gap_mm` **0.67 mm,
square-on**, which is *above* the old 0.5 mm gate, so **no weld ever formed** and the press looped to the
opaque `arm command heartbeat lost`. That is exactly the brief's symptom.

### Root cause: the weld gate sat AT the descent endpoint

The cuMotion contact descent settles the pad ~0.5 mm off the carton surface. The weld gate was
`SIM_ATTACH_CONTACT_MM = 0.5 mm` — right at that endpoint — so whether a weld formed was a coin-flip on
sub-mm pose/probe jitter:

- `collectsrc`: endpoint landed at 0.50 mm → welded.
- `collect1`: measured **0.56 mm** on the tick it would have welded (`SIM_ATTACH_DIAG skip gap_mm=0.56
  pad_vs_surface_deg=0.009`) → skipped; only welded a few ticks later at 0.48 mm by jitter.
- runs 321/327 (honest): endpoint at 0.67 mm → **never** crossed 0.5 mm → never welded → heartbeat loss.

The pad was square-on in every case (`pad_vs_surface_deg ≈ 0.009`, tilt gate is 2°). The only thing
keeping the weld from forming was the sub-millimetre gap threshold.

## Fix

Two levers, both **effective by default** (the captain asked to harden the sim "in any case"), both
backward-compatible (env/config can restore the old behaviour for a pre-fix repro).

1. **Tolerant weld gap** (`scripts/twin/restock_manipulation/cad_tool.py`). `SIM_ATTACH_CONTACT_MM`
   default `0.5 → 3.0 mm`, env-overridable (`RESTOCK_SIM_ATTACH_CONTACT_MM`), clamped to
   `(0.05, REAL_MAX_GRIP_DISTANCE*1000 = 5.0]` mm. The ceiling is principled: 5 mm is the **real** cup's
   physical seal reach (`REAL_MAX_GRIP_DISTANCE`), so a sim weld within 3 mm is a contact the real cup
   would also seal — not a fabricated hold. The **pad-tilt gate is unchanged** (`SIM_ATTACH_FLUSH_DEG =
   2°`): a tolerant gap must still be square-on. The gate decision is extracted into a pure, CPU-testable
   `_contact_weld_ready(gap_mm, pad_vs_surface_deg, hits)` (mirrors the existing `_release_held_on_open`)
   so the descent can be replayed without Isaac. With a 3 mm gate the honest-collect descent welds at
   0.67 mm, and the ground-truth descent welds earlier in the descent with margin instead of on a jitter
   coin-flip.

2. **Grasp-press budget effective by default** (`scripts/twin/restock_manipulation/ros_host.py`).
   PR 203's SIM-clock pre-weld press budget (`grasp_press_timeout_s`) was wired as `config.get(...,
   0.0)` — off unless the integration lane added the key (it was NOT set for `collect1`). It now defaults
   to `_default_grasp_press_timeout_s()` = **45 s SIM** (env `RESTOCK_GRASP_PRESS_TIMEOUT_S`); the config
   key still overrides, and `grasp_press_timeout_s: 0` disables it (byte-identical pre-fix). A non-forming
   grasp keeps the host **alive** (trajectory_host still publishes `/joint_states`, so the pace-aware
   arm-command heartbeat — which fires only on host *silence* — does NOT trip), so without this bound the
   press loops to the coarse 120 s `phase_timeout`. The budget fails it fast with
   `grasp press exceeded <N>s sim ... failing fast before the arm-command heartbeat watchdog`.
   **Why 45 s, not the sibling fragment's 10 s:** the slowest observed *legitimate* press (ground-truth
   `collect1`, suction-on at sim 16.0 → weld at sim 29.4) was ~13.4 sim s, so 10 s would false-fail a
   healthy-but-slow descent. 45 s is >3× the observed legit press and well under the 120 s phase_timeout.

The `TransferController` class default stays `grasp_press_timeout_s=0.0`, so direct unit constructions are
byte-identical; only the production host wiring is default-on.

## CPU tests (no Isaac/GPU)

- `scripts/twin/tests/test_cad_restock.py`
  - `test_contact_weld_ready_tolerates_the_descent_endpoint_gap` — the exact 0.56 mm that failed
    `collect1` under the old gate now attaches; tilt still gates; None/NaN/empty probe never ready.
  - `test_contact_weld_ready_replays_collect1_descent_must_attach` — replays `collect1`'s verbatim
    `SIM_ATTACH_DIAG` gap sequence (143 → 0.48 mm) and asserts the tolerant gate attaches **earlier** than
    the old 0.5 mm gate (not on the last sub-mm reading), monotone, square-on.
  - `test_contact_weld_ready_never_forming_press_stays_not_ready` — a pad parked 5 cm off, or badly
    tilted, for the whole press is NEVER ready (so the grasp-press backstop ends it, not a fake weld).
  - `test_sim_attach_contact_mm_env_override` — `RESTOCK_SIM_ATTACH_CONTACT_MM=0.5` restores the pre-fix
    gate; clamped to 5 mm; bad value → default.
- `scripts/twin/tests/test_restock_manipulation.py::test_grasp_press_budget_effective_by_default_in_host_wiring`
  — the host default is > 0; absent config key → default-on; `grasp_press_timeout_s: 0` → disabled;
  config and env overrides respected. (The existing fail-fast + not-rearmed-by-retries tests at
  `grasp_press_timeout_s=3.0` are unchanged and still pass.)

All of the above pass (`157 passed, 4 skipped`). One pre-existing unrelated failure
(`test_actual_ros_message_request_and_async_progression`, an incomplete rclpy node stub in
`ros_adapter.py` — a file untouched here) fails identically on clean `HEAD`.

## Bounded GPU proof (handed to the integration lane)

Both levers ship default-on in code, so the existing `collect1` config needs **no edit** to get them —
re-run it on `fm/sim-robust-weld-attach-v1` under `/tmp/aile-gpu.lock` (ground-truth collect). Switch
keys + env overrides and the expected result are documented in
`docs/examples/weld-attach-mission-fragment.json`.

**Expected:** the collect GRASP seals deterministically (`SIM_ATTACH_FALLBACK attach gap_mm <= 3`) with
no `grasp press exceeded` and no `arm command heartbeat lost` at the grasp.

**Caveat:** a full **collect + couple** proof additionally needs the **place-on-cart** blocker in effect
(sibling `sim-robust-collect-place-v1` / PR 200) — otherwise both runs fail at PLACE before couple. To
validate *this* lane's grasp fix in isolation, scope the proof to the pick, or run with the place-on-cart
fixes also present.
