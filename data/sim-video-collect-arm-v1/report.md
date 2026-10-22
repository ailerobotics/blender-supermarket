# Cold ground-truth collect: real arm picks + places on the rollwagen (sim-video-collect-arm-v1)

Goal: the cold ground-truth collect completes with the real arm visibly picking the carton and
placing it on the rollwagen upper shelf, proven offline (CPU). The integration lane runs the GPU test.

## 1. Evidence — the exact collect-only failure and what it is

Source: `~/datasets/twin-robot-in-store-boot-v1/mission/collectonly.console.log`
(run-356, `--stop-after collect`, before any valid_until_steady fix).

Failure sequence:

1. `COLLECT_GT calls=931 sku=juice_orange held=False pose=(0.008, 1.1, 0.26, …) (ground-truth, not camera)`
   — ground-truth collect (`RESTOCK_COLLECT_GROUND_TRUTH=1`, PR 163) **activates**: the carton pose is
   read from the sim, not the camera.
2. The arm approaches and reaches CONTACT (manip stream: 928 `approach` ticks then 7 `contact`; the
   gripper control `{"kind":"gripper","enabled":true}` is emitted).
3. **`MANIP_HOST_FAILED op=collect phase='host' reason='missing acquisition deadline for gripper command'`**
   — the host OmniGraph gate (`restock_manipulation/isaac_ros_graph.py:196`) refuses the gripper command
   because the `/manipulation/status` result carrying it has `valid_until_steady = None`.
4. The host `stop()`s → the arm feed publisher dies (`MASKED_DEPTH_BRIDGE_CLOSE` / `writer_eof`), and the
   next `NAV_STEP` reports `arm_not_in_transport transport_ready=False parked=True`.

Root cause: `arm_not_in_transport` is a **downstream consequence** of the aborted collect, not an
independent fault. The one real blocker is the missing acquisition deadline on the gripper command — the
collect aborts at CONTACT, which is exactly the captain's symptom "the robot is picking but the arm
doesn't seem to move".

## 2. Root-cause correction to the task premise

The brief stated "PR 170 (fm488 valid_until_steady=now+300, commit 555823b) landed and may already fix the
deadline". **This was inaccurate.** PR 170 (`fm/sim-video-transfer-carry-v1`, merge 8946733) is the
transfer `wrist_guard`/`weld` work and does **not** touch the capture packets. Commit **555823b (fm488)**
lived only on the unmerged local branch `transfer-valid-until-fix` with **no open PR**, so the boot head
(`2e1b08d`) still had `_capture_ground_truth_collect` / `_capture_transfer` sending `valid_until_steady=None`.
The deadline blocker was therefore **unfixed** in the deliverable base.

(The GT-collect replay still "passed" against the old code because it is a kinematic + reach probe; it does
not exercise the host `isaac_ros_graph` deadline gate.)

## 3. Fix shipped on this branch

- **Cherry-picked fm488 (555823b)** onto the boot head: `_capture_ground_truth_collect` and
  `_capture_transfer` now stamp `valid_until_steady = now + 300.0` (a sim-clock budget the controller
  checks against the node clock; 300 s never expires mid-grasp). This propagates
  `Sensors.valid_until_steady` → `controller.step` (`self.valid_until_steady`) → `_result()` →
  `/manipulation/status` → the `isaac_ros_graph` gate accepts the gripper command.
  Scoped to the GT-collect and transfer capture paths only (the camera collect threads its own source
  deadline); byte-identical to the camera collect.
- **Added bridge-level regression guards** (`tests/test_collect_target_bridge.py`):
  `test_ground_truth_collect_capture_stamps_the_acquisition_deadline` (pre-grasp and after-held) and
  `test_transfer_capture_stamps_the_acquisition_deadline`. fm488's own guard
  (`test_gripper_command_result_carries_the_acquisition_deadline`) feeds the controller a synthetic fresh
  deadline, so it cannot catch a revert of the stamp at the source — these pin it at the capture.

Place-through-cage and the carry-reposition transport gate were **never reached** in the collect-only run
(it died at CONTACT). Their capabilities already exist and are tested; see §5 switch keys.

## 4. Offline proof (CPU, no GPU)

- Manipulation + collect suites: `test_restock_manipulation`, `test_collect_target_bridge`,
  `test_collect_ground_truth`, `test_collect_deck_placement`, `test_collect_capture_gate`,
  `test_collect_target_reproject`, `test_transport_relay`, `test_replay_reach`,
  `test_restock_sensing_gate`, `test_restock_runtime` → **229 passed, 15 skipped** (plain CPU);
  `test_restock_cage_collision`, `test_restock_cart_cage_place`, `test_collect_deck_placement` →
  **31 passed** (`.venv-usd`).
- GT collect bridge replay on recorded cold runs 343–348:
  `scripts/check_collect_ground_truth.py run-343 … run-348` → **all 6 PASS** (exit 0): ground-truth pick
  packet parked at pickup + slot place target re-registered at `place_approach`, every station pose
  (approach/contact/lift/preplace/place/retreat) within `max_reach`.
- Actual-URDF reach probe: `scripts/check_restock_reach.py` → **`ok: true`, all poses reachable** (exit 0).

Run commands: see RUNBOOK §13 (manip suite), §13b / the script header (GT replay), §12 (reach).

## 5. READY — switch keys for the GPU run (integration lane owns these configs)

The cold ground-truth collect requires, in the mission runtime config / env (outside this repo, under
`~/datasets/…`):

- `RESTOCK_COLLECT_GROUND_TRUTH=1` (env) — ground-truth collect pick (PR 163).
- `config['gt_place_slot']` set — place on the rollwagen **upper shelf** (cart body +0.18, z≈1.0125,
  yaw −90°), so cold and warm hand the transfer the same carton pose.
- `config['cage_collision'] = 'filtered'` — sim-only carton-vs-cage contact filter (PR 167) so the carton
  clears the ~7.5 mm-per-side cage front posts at place. Labelled; default off.
- `config['carry_reposition_drive'] = true` — relaxes the empty-gripper transport gate on the mid-collect
  carry reposition leg (fm431 B1; nav `carry_drive` + carry-phase lidar self-mask). Default off.

The `valid_until_steady` stamp (this PR) needs **no switch** — it is always-on for the GT-collect and
transfer capture paths that were otherwise refused at CONTACT, and does not alter the camera collect.
