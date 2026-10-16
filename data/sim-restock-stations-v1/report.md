# Gripper seal diagnostic + root cause (key `gripper-seal-fail`)

Task: the mobile-UR10 restock mission reaches GRASP every run (run-170/173: approach → contact
→ grasp, base gate + drive + sensor watchdog all clear) but the native surface gripper **never
seals** — `NativePanelContact.read()` stays `False` (status != Closed or gripped-object count != 1)
in every run, so the mission times out in grasp. Deterministic, not variance. No gripper state was
logged anywhere. Branch `fm/gripper-seal-diagnostic-v1` off integration head `6331b46`; PR targets
`fm/twin-robot-in-store-boot-v1`.

## (a) Diagnostic — shipped

`scripts/twin/restock_manipulation/cad_tool.py` `NativePanelContact` now instruments every grasp
tick. `probe()` returns a JSON-safe measured snapshot and `read()`/`command()` emit a throttled
`SEAL_DIAG` log line (on a command, on any status/gripped/gap change, else a sparse heartbeat):

- gripper **status** (`Open`/`Closing`/`Closed`) straight from the plugin,
- **gripped-object count + prim paths** (the contact prim pair),
- each **seal's live world pose** and the **suction-axis world direction** + its **tilt from
  world-down**,
- a **PhysX raycast gap (mm)** from each seal along the suction axis to the nearest body
  (`min_gap_mm`) — the measured contact gap and which prim is hit.

Self-contained in `cad_tool.py`: `read()`'s latch contract is unchanged, so the diagnostic lands
in the mission and any harness without touching the controller, sensing or port. Default on;
`RESTOCK_SEAL_DIAG=0` disables the logging. CPU-covered by
`test_cad_restock.py::test_seal_probe_measures_status_grip_and_geometry` (11 cad_tool tests pass).

## (b) Root cause — measured, not geometry

`NativePanelContact.read()` was a **side-effecting sensor**: it called `open_gripper()` whenever it
sampled a gripped object while the status was not yet `Closed` (meant to reject a "partial grab").
But the installed surface gripper's normal `Closing → Closed` suction transition transiently
reports its object *before* it reaches `Closed`. The mission enables suction **during** the slow,
time-dilated contact descent (`controller.py` contact phase) and polls `read()` every sensing
frame — so `read()` **aborted the seal on essentially every poll**, and it never completed.
`s.contact` was `False` in every run.

Measured on guarded GPU with the REAL `author_native_gripper` + `NativePanelContact` + the new
`probe()` (`~/datasets/sim-restock-stations-v1/contactgap-02`, mission handling-unit carton
0.6×0.4×0.16, descent to the real `tcp_surface_offset` = 2.75 mm):

| recipe | suction enabled | latched | final status | gripped | lift |
|---|---|---|---|---|---|
| `mission_slow` (max_gap 20, clear 10) | during descent | 1 step only, then lost | `Closed` | `[]` | carton 0.0 mm (not held) |
| `tight_slow` (max_gap 6, clear 3) | during descent | 1 step only, then lost | `Closed` | `[]` | carton 0.0 mm (not held) |
| `mission_direct` (max_gap 20, clear 10) | after contact | yes | `Closed` | `['/World/Carton']` | carton **+66.9 mm** |
| `tight_direct` (max_gap 6, clear 3) | after contact | yes | `Closed` | `['/World/Carton']` | carton **+61.5 mm** |

The per-step trace (`mission_slow.json`) shows the gripper oscillating `Closing↔Closed` with
`gripped_n=0`, hitting `Closed+1` for a **single step** (gap 19.9 mm) then immediately dropping
back — that single step is the one poll where `read()` sampled the completed latch before its own
`open_gripper()` on the next disturbed poll tore it down. The **clearanceOffset (10 mm vs 3 mm)
made no difference**; the distinguishing factor is purely the enable-timing / the read() abort.
This was never caught before because PR100's latch probe stopped the descent the instant it first
latched (so `read()` only ran once in the latched state) and never re-polled a moving seal.

## The fix

`scripts/twin/restock_manipulation/cad_tool.py` — `NativePanelContact.read()` is now a **pure
observer** that never mutates the gripper, and it keys the hold on the plugin's **grippedObjects
list (exactly one attached body)**, not on `status == Closed`. Two measured facts forced both parts:

1. **No abort.** The old `read()` called `open_gripper()` whenever it saw an object while the
   status was not yet `Closed`. The normal `Closing→Closed` suction transition reports its object
   *before* reaching `Closed`, so that abort tore the seal down on every poll of the slow descent →
   it never completed. Removing it let every recipe seal and lift the carton (~60 mm) —
   `contactgap-03`.
2. **Key on grippedObjects, not status.** With the pad frozen and the carton provably gripped
   (`grippedObjects == ['/World/Carton']` *every* step), the plugin's status still flickers
   `Closing↔Closed` tick to tick (`contactgap-03` trace). Keying on `Closed` made `read()` flicker
   `False` and would false-trip the controller's per-tick lost-grasp gate. The grippedObjects list
   is the stable physical-attachment signal (it lifts the carton in both states); status is used
   only to exclude `Open`.

Safety preserved without mutating state: a partial/multi-object or released grab reports *not-held*
(`False`) — no fabricated hold, no weld. Sim- and real-rig-safe (the real rig polls this exact
reader). Covered by `test_cad_restock.py::test_native_read_is_pure_observer_keyed_on_gripped_objects`.

### Proof — guarded GPU, real `author_native_gripper` + `NativePanelContact`

`~/datasets/sim-restock-stations-v1/contactgap-04` (`/tmp/aile-gpu.lock`, heavy=2500, runtime
`gpu-runtime-578/root`), all four recipes:

| recipe | suction enabled | latched | persisted (30 steps) | gripped | lift | still held |
|---|---|---|---|---|---|---|
| `mission_slow` (max_gap 20, clear 10) | **during descent** | yes | **yes** | `['/World/Carton']` | +59.8 mm | **yes** |
| `tight_slow` (max_gap 6, clear 3) | during descent | yes | yes | `['/World/Carton']` | +59.8 mm | yes |
| `mission_direct` (max_gap 20, clear 10) | after contact | yes | yes | `['/World/Carton']` | +60.2 mm | yes |
| `tight_direct` (max_gap 6, clear 3) | after contact | yes | yes | `['/World/Carton']` | +60.0 mm | yes |

`mission_slow` reproduces the mission's exact choreography (suction on during the contact descent):
it now latches at the window, the latch **persists** for ≥30 frozen steps, and the carton stays
held through a 60 mm lift. Before the fix (`contactgap-02`) this recipe latched for a single step
then ended `Closed`-but-empty with no lift.

Final authority is a full mission run: the shipped `SEAL_DIAG`/`probe()` will log the grasp numbers
there, and the integration lane confirms the sealed grasp → lift → couple → tow → park in-mission.

## Definition of done

- Diagnostic: `SEAL_DIAG`/`probe()` in `cad_tool.py`, CPU test, logged every grasp tick in-mission.
- Root cause + fix: measured on guarded GPU (`/tmp/aile-gpu.lock`, heavy=2500, runtime
  `gpu-runtime-578/root`) under `~/datasets/sim-restock-stations-v1/`.
- Harness: `evidence/probe_contactgap.py` (side-grasp contact-gap + enable-timing sweep).

## Evidence

- Root cause: `~/datasets/sim-restock-stations-v1/contactgap-02/` (old `read()`: slow recipes
  latch one step then lost, end `Closed`+empty, no lift).
- Fix proof: `~/datasets/sim-restock-stations-v1/contactgap-04/` (fixed `read()`: all recipes latch,
  persist ≥30 steps, lift the carton ~60 mm, stay held). Intermediate `contactgap-03/` shows the
  status-flicker that motivated keying on grippedObjects.
- Probe: `data/sim-restock-stations-v1/evidence/probe_contactgap.py` (side-grasp contact-gap +
  enable-timing sweep; same GPU guards as the mission).
