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

---

# Grasp-yaw seal alignment (key `gripper-seal-align`)

With PR 102 (the pure-observer `read()`) merged, run-177 reaches the grasp but still never seals.
`SEAL_DIAG` pinpoints why: the side-grasp pad reaches the carton (`min_gap_mm 0`, vertical tilt
89.875° correct) but `gripped_n = 0`, and the pad forward axis is `[0.073, 0.997, -0.002]` =
**4.19° off world +Y** — above the plugin's ~2° seal floor. This section measures where that 4.2°
comes from (Task 2) and whether the seal gate is world-frame or surface-relative (Task 1).

## (Task 2) The 4.2° is the collect **source observation** yaw, not the base or the gate

- **True carton yaw ≈ 0.00°.** `run-177/mission.usda` `def Mesh "HandlingUnit"` authors
  `quatf xformOp:orient = (-5.56e-9, -8.86e-10, 1.81e-8, 1)` — identity to ~1e-8 rad; the pick
  carton is axis-aligned (flush to the backstock wall).
- **Grasp yaw tracks the observed source yaw.** `wall_policy.py:56` sets
  `placement.yaw = yaw(source.observation.pose)`, and upstream `shelfbot…geometry.wall_grip`
  (verbatim) computes `angle = placement.yaw + wall·90°`, `quat = yaw_quat(angle)·grasp_quat`.
  The `wall·90°` term is a fixed quadrant, so the pad-forward's deviation from the wall-nominal
  face normal **equals the observed source yaw**.
- Measured pad forward `[0.073, 0.997, -0.002]` → `atan2(0.073, 0.997) = 4.19°` off +Y; the base
  dock yaw is ~0.6° and the arm tracked within 0.57° of command (integration SEAL_DIAG), so the
  commanded grasp orientation carries ~3.6–4.2°.

⇒ **The collect source observation estimates the carton yaw ~4.2° (≈3.6° net of the base dock) off
its true ~0° pose.** That bias originates in the camera lane's `restock_camera` source_target, not
in any file this lane owns. (Firstmate relaunched the camera lane on this bias; inbox 011.)

## (Task 1) The native seal gate is **surface-relative**, already keyed to the gripped face

A static GPU harness (`probe_surface.py`) drives the REAL `author_native_gripper` +
`NativePanelContact`: the panel body (dynamic, pinned to a kinematic carrier) is held `GAP = 3 mm`
from a carton face, suction enabled, and latch measured. It separates a top grasp from a side grasp
and, for side grasps, a pad misaligned from a flat face vs a pad held **flush** to a yawed carton.
`~/datasets/sim-restock-stations-v1/surface-20261016-183950` (`/tmp/aile-gpu.lock`, heavy=2500,
runtime `gpu-runtime-578/root`):

| recipe | grasp | pad vs TRUE face | latched | gripped |
|---|---|---|---|---|
| `top_flat` | top-down | 0° | **yes** | `['/World/Carton']` (sanity — harness latches) |
| `side_flat` | side | 0° | **yes** | `['/World/Carton']` (a flush side grasp seals) |
| `side_pad4` | side | **4.0°** | **no** | `[]` (reproduces run-177) |
| `side_both4` | side | 0° (pad ∥ **4°-yawed** carton) | **yes** | `['/World/Carton']` |
| `side_both8` | side | 0° (pad ∥ **8°-yawed** carton) | **yes** | `['/World/Carton']` |

**Verdict: the gate is surface-relative, not world-frame.** A flush pad latches at *any* carton
yaw (`side_both4`/`side_both8` seal with the carton yawed 4° and 8°); the gate rejects only when the
pad is misaligned from the *real* contacted face (`side_pad4`, 4° off a flat face → no grip — the
exact run-177 condition). So the authored forward axis + per-seal raycast already key the ~2°
tolerance to the gripped surface normal — which is precisely what Task 1 asked the gate to be.

Consequences for the fix:
- There is **no world-frame bug in `cad_tool.py`** to correct; the gate already does the
  surface-relative thing and is robust to the carton's own yaw.
- The plugin's ~2° *pad-vs-true-surface* tolerance is a compiled limit of
  `isaacsim.robot.surface_gripper` (no angle attribute among `maxGripDistance / clearanceOffset /
  forces / retryInterval`; rotational joint compliance was already proven not to widen it,
  `data/surface-gripper-latch-v1`). **No `author_native_gripper` lever widens that angle gate.**
- The mission's seal therefore needs the pad presented within ~2° of the *true* face. The only
  in-this-lane lever for that is the **grasp orientation** (`wall_policy.py:56`): source the pad
  rotation from the design-known wall normal instead of the noisy observed yaw. That overlaps the
  camera lane's relaunched observation fix and changes behaviour for a genuinely-tilted carton, so
  it is raised to firstmate rather than taken unilaterally.

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
