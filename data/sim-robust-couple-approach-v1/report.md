# sim-robust-couple-approach-v1 — the couple APPROACH lands the base consistently on the cart axis

Lane goal (firstmate): the couple approach lands the base consistently close to the hitch alignment so
the fine-dock starts inside (or near) its capture box, and the fine-dock window is sized to the
convergence actually needed. Follow-on to `sim-robust-couple-fine-dock-v1` (PR 232, the cross-track
servo): that fixed the *servo* but left the *entry* variable.

Failure observed. Under `live_pose` the couple standoff ARRIVAL residual at fine-dock entry varied
wildly: `coldlivefinal` entered ~25 mm / 4.8°, `coupledock` entered 469 mm / 354 mm cross / −23.6° and
the fine-dock was cut by the fixed 30 s cap mid-convergence
(`state=failed reason='fine_dock time cap 30s: residual hitch=88.3mm yaw=-27.26deg'`). Run dirs
`~/datasets/twin-robot-in-store-boot-v1/mission/{coldlivefinal,coupledock}`.

## Root cause — a FIXED map standoff against a run-varying pinned cart yaw

Reconstructed from the two runs' `nav-trace.jsonl` (the `cart` row is `[x, y, yaw_deg, z]` — the yaw is
the **degrees** field, not the z height; an early misread of the z as yaw sent the first analysis the
wrong way):

| run | measured cart yaw | authored standoff yaw | fine-dock entry/stuck | result |
|---|---|---|---|---|
| coldlivefinal | **−1.58°** | 0° | ~26 mm / 4.78° (near miss) | limit cycle, 30 s cap |
| coupledock | **+19.17°** | 0° | 88 mm / −27° (big miss) | 30 s cap mid-converge |

The couple standoff is a **fixed map pose** `couple_standoff = [-4.06, 15.25, 0.0]` (base yaw 0). The
cart is pinned at wherever it settled after the `live_pose` collect, and that yaw **varies run to run**
(−1.6° vs +19.2°). Navigating to the fixed pose lands the base aligned to the MAP, not the CART, so the
base-vs-cart yaw residual *is* the cart yaw, and the hitch residual grows with it (the rear hitch is a
0.62 m lever behind the base: a 19° base-vs-cart yaw alone throws the hitch ~0.2 m off-axis, and the
fixed pose's xy offset adds the rest to ~0.47 m). The fine-dock creep — legacy OR cross-track — cannot
null a large off-cart yaw: both have an equilibrium (`ct_gain_cross·cross == ct_gain_yaw·yaw_err`, w≈0)
the base settles on OUTSIDE the latch box. `coldlivefinal` nearly latched only because its cart happened
to settle near the authored yaw 0; `coupledock` did not. The variability IS the root cause.

Verified on CPU: `scripts/check_couple_approach.py --runs-dir …/mission --runs coldlivefinal coupledock`
reads both carts straight from the traces and reproduces the fixed-standoff failure of the 19°-off cart
through the REAL `fine_dock` servo.

## Fix — cart-relative standoff + bounded re-approach + progress-based cap (all opt-in, default off)

1. **Cart-relative standoff** (`fine_dock.standoff_from_cart`): seat the standoff on the MEASURED cart
   axis — base yaw = cart yaw, rear hitch held `standoff_offset_m` back along the axis so the servo
   creeps STRAIGHT in. The fine-dock then starts on-axis + yaw-aligned regardless of where the cart
   settled (entry cross ≈ 0, yaw ≈ 0 for BOTH recorded carts).
2. **Tight cart-relative arrival + bounded re-approach** (`runtime.RuntimeAdapter`, config
   `couple_cart_relative`): the couple standoff leg navigates to `standoff_from_cart(measured cart)`; on
   arrival the hitch's **cross-track + yaw** against the cart (not just map xy) gate entry — inside the
   capture region → hand to the fine-dock from an aligned pose; outside → re-approach a fresh
   cart-measured standoff (bounded `max_reapproach`); exhausted → fail honestly with the residual (never
   loosen the latch). The capture region (cross ≤ 0.11 m, yaw ≤ 0.10 rad) admits every LEGAL standoff
   arrival (worst legal hitch cross ≈ arrival_xy_tol + |hitch_x|·arrival_yaw_tol ≈ 0.10 m) so a good
   arrival never triggers a spurious re-approach.
3. **Progress-based fine-dock cap** (`fine_dock`, `progress_cap`): replace the blunt fixed `time_cap_s`
   with "keep docking while the hitch-equivalent residual (`hitch + |yaw|·|hitch_x|`) keeps improving,
   bounded by `progress_max_time_cap_s`, fail FAST at `progress_patience_s` on a limit cycle". This
   latches `coupledock`'s slow monotone close (≈13 mm/s, which the fixed 30 s cut off) and still
   fast-fails a true limit cycle. SIM-clock throughout (`freshness-sim-clock-not-wall`).

Backward compatible: absent `couple_cart_relative` ⇒ the authored fixed standoff (byte-identical);
absent `progress_cap` ⇒ the fixed `time_cap_s` (byte-identical, 400-sample fuzz-asserted). The scripted
warm couple path is untouched. Both switches are named in the couple stage evidence
(`couple_arrival_mode`) and so in the walkthrough caption.

## Evidence

- CPU proof (reads the recorded runs): `scripts/check_couple_approach.py` → `COUPLE_APPROACH_PROOF PASS`
  (both carts seat from the cart-measured standoff in ~11 s; the 19°-off cart fails from the fixed pose;
  the progress cap latches a 34.7 s monotone close the fixed cap cut and fails a limit cycle at 8.1 s).
- CPU tests: `scripts/twin/tests/test_restock_couple_approach.py` (16: geometry, closed-loop latch from
  both recorded carts, progress cap latch/fast-fail/byte-identical, runtime re-target/​re-approach/​fail/
  validation) + the PR 232 `test_restock_fine_dock.py` + `test_restock_runtime.py` still green.

## Bounded GPU proof — handed to the integration lane (PENDING)

Config fragment `docs/examples/couple-approach-mission-fragment.json` (merge the opt-in
`couple_cart_relative` + `fine_dock.progress_cap` keys). Run: collect+couple under `live_pose`,
cross-track + `masked_depth.economy` on, `--stop-after couple`, `RESTOCK_HEARTBEAT_UNAVAILABLE_S=120`.
**PASS = couple reaches a physical latch** (`action=couple status=succeeded coupled=True`). The
integration mission owns GPU priority + the host stack; this worker does not take the GPU (the validator
may be running it).
