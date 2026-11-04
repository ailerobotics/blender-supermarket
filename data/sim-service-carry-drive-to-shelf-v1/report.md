# sim-service-carry-drive-to-shelf-v1 — run the side-bay TRANSFER as separate BT steps: pick → NAVIGATE (carry) → place

**Branch:** `fm/sim-service-carry-drive-to-shelf-v1` → PR against `fm/twin-robot-in-store-boot-v1`.
**Scope:** make the shelf-staging / side-bay (option 1) TRANSFER actually execute its **service-drive
leg** — the base carries the picked carton from the side-bay park to the endcap `shelf_approach` service
pose **before** placing — as separate behaviour-tree steps (uncouple@side-bay → pick → **carry→endcap
(NAVIGATE)** → perceive slot → place@endcap). **CPU-proved.** GPU acceptance (coldwin24) is
integration-owned (hand-off below). Opt-in flow unchanged (default off ⇒ byte-identical). **Out of scope**
(captain): the collect carton-level re-solve (separate live lane `sim-carton-level-southdock-v1`), tow
lanes, the cart's turn home.

## The blocker (transferobs1)

The integration isolated the transfer stall in `transferobs1` (`start_at=couple`, PR 308 launch,
`stop_after=transfer`; saved `~/datasets/twin-robot-in-store-boot-v1/mission/transferobs1-fail1-saved/`).
Couple, the 50.3 s outbound tow and the side-bay park were all reliable (stage audit:
`data/.../transferobs1-audit/`). TRANSFER then failed:

* mission.json reason: `RuntimeError: arm command heartbeat lost` (the Isaac-side symptom);
* host-stack.log line 2462: **`HOST_FAULT_ESTOP phase=observe reason='unreachable or undersized target'`**
  — the real cause, i.e. `'host emergency stop (host fault: unreachable or undersized target)'`;
* nav-trace: the base is pinned at **(-4.348, 6.989)** for **all 1458 transfer rows** (1 distinct xy) —
  the service-drive leg never ran.

## Root cause — a cross-config-layer split (spec #1, quote the line)

`transferobs1`'s `HOST_EFFECTIVE_CONFIG` (host-stack.log:151) shows **`"transfer_service_carry": false`**
while the RUNTIME side had `shelf_staging` / `cart_side_bay` on. The service carry needs BOTH halves,
and they are read from two different config files:

1. The RUNTIME `shelf_staging` re-sequence (`isaac_bundle` → `RuntimeAdapter`) **drives** the base carry
   to `shelf_approach` — but only once the arm has folded to **TRANSIT** (runtime.py the carry leg fires
   on `self._last_manip_phase == 'transit'`).
2. The arm folds to TRANSIT only when the HOST `transfer_service_carry` makes
   `controller._repositions_before_place()` true:

   ```python
   # scripts/twin/restock_manipulation/controller.py:865
   if self.transfer_service_carry and self._live_pose_transfer():
       return True
   ```

   With `transfer_service_carry` **false**, this returns false, so the live_pose transfer goes
   `tuck → preplace` (no TRANSIT hold), the base **never** carries the carton, and the controller gates
   the endcap place reach from the **side-bay park** at observe:

   ```python
   # scripts/twin/restock_manipulation/controller.py (observe gate, pre-fix)
   if (not self._pose_ok(target.pose)                       # endcap 2.545 m from the side-bay park
           or not axis_fits(target.free_dimensions, ...)):   # > the 1.30 m UR10 reach
       return self._fail('unreachable or undersized target') # -> motion.stop() -> host emergency stop
   ```

That observe gate checks the raw endcap `target.pose` reach **from the current pose, before the pick** —
an ordering bug for the service-carry flow, where the place is reached only **after** the base drives to
`shelf_approach`. `coldwinint4`'s `'phase timeout: observe'` was the looped form of the same unreachable
place.

## Fix at the source

1. **Controller observe gate ordering** (`controller.py`): when `_repositions_before_place()` (the fm390
   collect place-on-cart **and** the shelf-staging live_pose transfer service leg), **defer** the raw
   `target.pose` reach — the place is reach-gated at plan time (from the service pose) and again
   post-drive in `_reregister_place`. The frame-independent size-fit (`axis_fits`) and the `clear` check
   still gate the target now. Non-repositioning transfers are byte-identical (the reach gate still fires).
2. **Plan-time trip named** (`transfer_place_target.resolve_transfer_place_target`): an endcap beyond the
   UR10 reach **from `shelf_approach`** now trips, at boot before any arm command,
   **`unreachable place target: <dist> m > <reach> m`** (spec #3) instead of a mid-transfer emergency stop
   or an observe phase timeout. `shelf_approach` is reachable here (1.217 m ≤ 1.30 m) — it is **not**
   moved; the reach bound is never moved.
3. **Config-layer consistency gate** (`config_preflight.check_flow_consistency`): runtime `shelf_staging`
   ON now **requires** host `transfer_service_carry` ON, else the preflight **refuses the boot at plan
   time** with a named reason — the `shelf_staging`/`transfer_service_carry` split can no longer reach a
   GPU run silently.
4. **Host default** (`ros_host_config.json`): `transfer_service_carry: true` (the service leg is the
   integration's current goal; blast radius is live_pose transfers only — `_repositions_before_place`
   needs `_live_pose_transfer()` too, so collect / legacy / non-live-pose transfers are unaffected).

The BT already decomposes the side-bay transfer into the right nodes (`stage_phases.TRANSFER_STAGING_PLAN`
+ `OUTBOUND_SIDE_BAY_PLAN`, rendered by `diagram.build_model(cart_side_bay=True)`); the break was purely
in execution. With the fix the `carry→endcap` NAVIGATE node now runs.

## CPU proof (spec #4)

`scripts/check_service_carry_drive.py` (artifact `check_service_carry_drive.json`), all PASS:

* **BT node order** — `OUTBOUND: tow→side bay → brake cart → uncouple`; `TRANSFER: perceive carton →
  transfer-pick@staging → carry→endcap [nav] → perceive slot → place@endcap → verify placed →
  drive→staging` (the carry is its own NAVIGATE node, index 2, before place@endcap, index 4).
  Diagram SVG: `bt-side-bay-diagram.svg`.
* **Reach check** — endcap reachable from `shelf_approach` (1.217 m ≤ 1.30 m), unreachable from the
  side-bay park (2.545 m > 1.30 m, the transferobs1 distance); the plan-time resolver accepts the
  reachable endcap and names an unreachable one `unreachable place target: 2.025 m > 1.30 m`.
* **Replay** of transferobs1's start state through the fixed controller observe gate — service carry ON:
  observe proceeds past the far endcap (no emergency stop); OFF: still fails fast
  `unreachable or undersized target` (byte-identical).
* **Preflight** — the split is refused at plan time; the paired config (and the committed repo host
  default) passes.

Unit tests (`env -u PYTHONPATH PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv-usd/bin/python -m pytest`):

* `test_restock_runtime.py::...test_side_bay_transfer_carries_the_carton_to_shelf_approach` — nav-trace
  regression: the carry base trace starts at the side-bay park and the base ARRIVES at `shelf_approach`
  to place (transfer base start ≠ its place end); the carry nav leg targets `shelf_approach`; the full
  service leg succeeds.
* `test_restock_transfer_live_pose.py::test_service_carry_defers_far_place_reach_at_observe_and_reaches_transit`
  and `::test_without_service_carry_far_place_still_fails_at_observe_transferobs1`.
* `test_transfer_place_target.py::test_unreachable_endcap_fails_loudly` (new trip phrasing).
* `test_config_preflight.py::FlowConsistencyTests`.

## GPU acceptance hand-off (spec #5)

Integration-owned (the GPU lane holds `/tmp/aile-gpu.lock`; this lane never boots a GPU mission).
Fragment: **`docs/examples/service-carry-drive-to-shelf-mission-fragment.json`** (merges onto the
existing `cart-side-bay-mission-fragment.json` + `service-leg-carry-mission-fragment.json`). The one
change the integration must carry is `transfer_service_carry: true` in the HOST config — now the repo
default, and enforced by the config-preflight gate so a split cannot recur.

**coldwin24 acceptance** (`start_at=couple` + `stop_after=transfer`): couple → 50 s outbound tow →
side-bay park → **uncouple at the side bay → pick the carton off the cart (level) → the base DRIVES to
`shelf_approach` carrying the carton level (nav-trace base x,y changes from (-4.35, 6.99) and reaches
(-3.05, 6.0)) → perceive slot → place at the endcap**. No `'unreachable or undersized target'` / no
`'phase timeout: observe'`.

## Files

* `scripts/twin/restock_manipulation/controller.py` — observe-gate reach deferral when repositioning.
* `scripts/twin/restock/transfer_place_target.py` — `unreachable place target: <d> m > <r> m` trip.
* `scripts/twin/restock_manipulation/config_preflight.py` — `check_flow_consistency` + wired into `main`.
* `scripts/twin/restock_manipulation/ros_host_config.json` — `transfer_service_carry: true`.
* `scripts/check_service_carry_drive.py` — the CPU proof.
* tests: `test_restock_runtime.py`, `test_restock_transfer_live_pose.py`, `test_transfer_place_target.py`,
  `test_config_preflight.py`, `test_restock_host.py` (key-set guard updated).
* `docs/examples/service-carry-drive-to-shelf-mission-fragment.json` — the integration hand-off.
