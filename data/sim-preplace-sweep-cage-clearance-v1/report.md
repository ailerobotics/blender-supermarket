# sim-preplace-sweep-cage-clearance-v1 — name & clear the pre-place pan-sweep cage contact

## Problem

With the base fully locked (PR 365, `sim-station-lock-from-preplace-v1`) and the held wrist firm
(PR 359, `sim-preplace-wrist2-hold-v1`), coldwin70's carried pre-place **leg-2 shoulder_pan sweep stalled
hard**: a `STATION_FULL_BRAKE_EFFORT max=1632.7` spike → `STATION_FULL_BRAKE_ABORT` → `WELDED_CONTROLLED_STOP`
(console lines 28907–28910), `shoulder_pan` froze at executed **0.2012** (commanded −0.4053) and the op failed
`MANIP_HOST_FAILED … PHASE_CONVERGENCE … shoulder_pan commanded=-0.4053 executed=0.2012` (line 33564). In the
two runs before (cw68/69) the same sweep shoved the unlocked base ~0.33 m west; in cw64 the soft wrist_2 gave
way during the sweep. All three point to a contact during the sweep.

## What the contact is (named, with numbers)

CPU reconstruction of the executed sweep at the real park pose — nav-trace poses (base `(-4.614, 14.769)`
yaw 84.2°, cart `(-4.607, 15.972)` yaw −93.61°) + the executed `CARTON_TICK phase=preplace q=[…]` joints +
the authored cart colliders (`place_clearance.rollwagen_collider_boxes`, PR 350) + the mobile URDF/XRDF —
reproduces the jam and names the body (`scripts/check_preplace_sweep_clearance.py`, pinned by
`scripts/twin/tests/test_preplace_sweep_clearance.py`):

- The leg-2 plan (host-stack `move hold joints to standoff`) holds elbow 1.4788 / lift −1.4856 and sweeps
  `shoulder_pan` **1.2678 → −0.4053**. The wrist-first order **extends** the arm to the standoff shape
  (leg 1: elbow 1.0 → 1.4788, lift −2.27 → −1.4856) *before* the pan sweep, so leg 2 rotates the already-
  extended, low **gripper end-effector housing** on an arc that crosses the +x cage mouth.
- Sweeping the **cuMotion arm sphere model** (the real robot-link collision set, incl. the gripper housing
  links `cube_without_surface_plates` / `surface_plate_1`) against the cage bodies: the **minimum clearance
  over the whole leg-2 is −48.0 mm into `/Rollwagen/chassis/post_2`** (the +x mouth post), with
  `side_0_horizontal_6` (upper side-wall bar, z 1.064–1.076) at −28.8 mm — **at `shoulder_pan ≈ 0.20`,
  exactly the executed freeze (0.2012)**.
- It is the **arm, not the carton**. The carton is cage-filtered (`CAGE_COLLISION mode=filtered`
  `kept=['deck','shelf','drawbar']`), so the carton cannot be the sim contact — only an *unfiltered* robot
  link can. The pad-only honest sweep reads +11 mm clear at the same spot, while the full housing penetrates
  post_2 by 48 mm. This matches cw69's `SEAL_DIAG … hit_bodies=['/Rollwagen/chassis/post_2'] min_gap_mm=0.0`
  (the gripper pad flush on post_2) and cw68's 757 N·m press — the same post, the same contact.

Cross-checks all agree: the independent carton-lumped sweep, the arm-sphere sweep, the sim's own freeze pan
(0.2012), and cw69's named `post_2` contact all land at `shoulder_pan ≈ 0.20` on the +x mouth post / upper
side bar.

## Fix

Two opt-in pieces (default off ⇒ byte-identical), compatible with PR 365 (base lock) and PR 359 (firm wrist):

1. **Pan-first reorder** — HOST `preplace_pan_first` (requires `preplace_wrist_first`). Leg 1 sweeps
   `shoulder_pan` to the standoff while the wrists + lift + elbow **hold at the compact carry shape**, then
   leg 2 extends over the open deck. The tucked arc clears the mouth post/side bar by **+252 mm** (vs −48 mm).
   The carton is re-leveled in leg 2's extend by the existing world-level leveller; the 2.0° tilt bound is
   untouched. (`preplace_sequence.pan_first_subgoals`, the inverse of `wrist_first_subgoals`.)
2. **Cage-clearance gate** — `place_clearance.preplace_cage_arm_clearance` / `preplace_sweep_reject` /
   `preplace_cage_clearance_provider`, injected as `controller._preplace_cage_clearance`. Refuses an
   intersecting leg-2 sweep **by name before motion**: `PREPLACE_SWEEP_CLEARANCE prim=post_2
   min_clearance=-48.0mm …`, instead of the blind 1632 N·m jam + opaque 120 s `PHASE_CONVERGENCE`. Checks the
   arm-sphere model against the cage bodies (`cage_body_names`: posts + side/back bars).

### The refuse margin, honestly

The gate's default refuse margin is **0.0 (an actually penetrating sweep)**, not 0.04, for two reasons:
(1) the CPU geometry carries a documented ~40–90 mm offset from the sim colliders (the same calibration caveat
the place-clearance block notes), so only a clear negative reading is a certain strike; (2) the pre-place
standoff itself sits an **inherent +4.2 mm** from the far side wall (`side_1_horizontal_6`) — a cage-width
tightness within that noise, **not** the cw70 blocker — so a positive margin would block even a clearing path.
The **≥ 0.04 m** figure is the *design target* the reorder meets on the raking bodies (+252 mm), asserted by
the check.

## Proof

- `scripts/check_preplace_sweep_clearance.py` (CPU, exit 0): OLD wrist-first leg-2 **REFUSED**
  `post_2=-48.0mm`; NEW pan-first reorder **CLEAR**, raking-sweep **+252 mm** ≥ 40 mm target.
- `scripts/twin/tests/test_preplace_sweep_clearance.py` (13 tests): reject naming/support-exclusion/margin,
  `pan_first_subgoals`, provider inert-off, controller defaults inert, and the guarded cw70 proof.
- Byte-identical off confirmed: `preplace_pan_first` default False, `_preplace_cage_clearance` default None;
  related preplace / carton_level / place_clearance / cart_world_obstacle suites green. (Two **pre-existing**
  failures on the base branch are unrelated: `test_carton_level_controller` e2e and the `test_restock_host`
  fixture `reactive_arm_*` keys from the open PRs 361/362.)

## Scope / follow-up

Out of scope (unchanged): the pick, the weld release, the transfer place, the perception lanes, the 2.0° bound.
If the standoff's inherent +4.2 mm far-wall clearance proves too tight on GPU once the reorder lets the arm
reach it, the follow-up is a place-target geometry change (lift/centre the standoff) — PR 364's lane.

Bounded-GPU acceptance recipe + recommended ON values:
`docs/examples/preplace-sweep-cage-clearance-fragment.json`.
