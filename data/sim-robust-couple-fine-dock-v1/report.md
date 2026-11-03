# sim-robust-couple-fine-dock-v1 — couple fine-dock converges+latches from the live_pose-collect standoff

Lane goal (firstmate): the couple **fine-dock** converges and latches from the standoff pose the real
`live_pose` collect leaves the robot in. Failure: run `coldlivefinal` (combined head PR 227–231, full
honest `live_pose` chain) cleared COLLECT and reached the couple standoff
(`action=couple status=succeeded reason=arrived_stopped transport_ready=True`), then the fine-dock
**limit-cycled** and timed out: `state=failed reason='fine_dock time cap 30s: residual hitch=26.7mm
yaw=4.78deg'`. Run dir `~/datasets/twin-robot-in-store-boot-v1/mission/coldlivefinal`
(`coldlivefinal.console.log` 31552–33822); the still-arm `coldhonest` latched the SAME dock
(156→28 mm). Deliverable: a PR into `fm/twin-robot-in-store-boot-v1` with CPU tests + one bounded GPU
proof config.

## Root cause — a degenerate zero-command equilibrium LINE (analytic, model-independent)

The fine-dock creep (`fine_dock.FineDock.step`, legacy `hitch_jacobian` law) commands, on the measured
hitch error:

```
v = -gain_lin · fwd                                   # along the base heading
w = -(gain_lin/hitch_x) · lat - gain_yaw · yaw_err    # lateral-hitch  +  yaw alignment, ONE ω
```

The single `ω` serves BOTH the lateral-hitch nulling and the yaw alignment. It is zero wherever

```
lat = (gain_yaw · |hitch_x| / gain_lin) · yaw_err  = 0.31 · yaw_err   (shipped gains 2.0/1.0, hitch_x=-0.62)
```

with `fwd = 0`. That is a whole **equilibrium line**, not a point: the controller is "content"
(commands ≈ 0) at any pose on it, including poses OUTSIDE the latch box. On the line the hitch residual
is `|lat| = 0.31·yaw_err`, so at `yaw_err = 4.78°` the hitch is `25.9 mm` — i.e. the recorded failure
`hitch=26.7mm yaw=4.78deg` sits exactly on this line. Measured directly from the shipped law at that
pose: `v = -0.00000, w = +0.00000` — frozen, with yaw `4.78° > 4.0° tol`. Nonholonomically it cannot be
escaped reactively: the rear hitch is 0.62 m behind the axle, so the ω needed to null the 4.78° yaw
swings the hitch ~51 mm sideways, and a diff-drive base cannot translate that back at a standstill.

**Why `coldhonest` latched and `coldlivefinal` did not:** the equilibrium yaw the approach lands on is
set by the standoff cross-track error. `coldhonest` (still-arm, undisturbed cart) landed at ~2.68°
(inside the 4° tol) → hitch crept to 18.5 mm → latched. `coldlivefinal` (real `live_pose` collect
disturbed the base↔cart relative pose) landed at ~4.78° (outside) → stuck. The shipped law is right at
the margin; which side of it a run lands on is luck.

Reproduced on CPU with the shipped law + a kinematic base (drive dead-band modelling the real
near-zero-command motion floor — the documented ~14 mm/s PhysX-TGS velocity bias): from the recorded
standoff the legacy law hits the 30 s time cap stuck ~20–26 mm / ~4.8°, matching the run.
`test_legacy_law_stalls_on_its_degenerate_equilibrium`.

## Fix — opt-in cross-track terminal docking (`approach_mode: 'crosstrack'`)

Default `approach_mode='hitch_jacobian'` ⇒ the legacy law, **byte-identical**
(`test_absent_approach_mode_is_byte_identical_to_the_legacy_law`, 400 poses). `'crosstrack'`
(`fine_dock._crosstrack`) servos the **rear hitch** onto the cart's drawbar axis. The hitch is the
LEADING point of the reverse approach, so steering it onto the axis is a stable **minimum-phase**
cross-track problem — forward motion provides the steering authority, so the only equilibrium is the
seated pose (no degenerate line):

```
along = (hitch - drawbar)·cart_axis ;  cross = (hitch - drawbar)·cart_normal ;  yaw_err = base - cart
along_sp = min(along, ct_align_hold_m·(|cross| + 0.3|yaw_err|))      # hold back until aligned, then seat
v = -ct_gain_approach·(along - along_sp)   (floored to ct_min_creep so steering stays live)
w =  ct_gain_cross·cross - ct_gain_yaw·yaw_err
```

Pure motion: latch tolerances and the physical `attach` in `mobile.py` are unchanged. Gains
(`ct_gain_approach=2.0, ct_gain_cross=8.0, ct_gain_yaw=1.5, ct_align_hold_m=3.0, ct_min_creep=0.015`).

## CPU validation (`scripts/twin/tests/test_restock_fine_dock.py`, 18 tests)

- `test_crosstrack_seats_the_recorded_coldlivefinal_standoff` — from the reconstructed recorded standoff
  (base ODOMETRY arrival `(-4.101, 15.298)` from the last `NAV_ARRIVAL_DIAG`, yaw = cart yaw + the first
  `FINE_DOCK_DIAG` residual −2.93°, measured cart `(-5.745, 15.237, 0.20°)` from the `RESTOCK_TRACE_POSES`
  nav-trace), under the drive dead-band the legacy law hits the 30 s time cap, the cross-track mode
  reaches `ready` ≤ 20 mm / ≤ 4° in ~13 s.
- `test_crosstrack_still_seats_the_coldhonest_standoff` — no regression on the standoff the still-arm run
  already latched.
- `test_crosstrack_converges_across_pivot_drift` — both recorded standoffs latch across pivot-drift
  `0.0–0.20` (the base's rotation↔translation coupling, whose true value is unknown).
- `test_legacy_law_stalls_on_its_degenerate_equilibrium` / `…commands_real_motion_at_the_legacy_equilibrium`
  — the model-independent property: at the recorded stuck residual the legacy ω is frozen (< 0.005), the
  cross-track mode steers (> 0.02, above any plausible drive floor).
- `test_absent_approach_mode_is_byte_identical_to_the_legacy_law`, `test_crosstrack_config_validation`,
  `test_crosstrack_builds_from_config`, plus the pre-existing 10.

All 18 pass; `test_restock_runtime.py` + `test_restock_mission_tail.py` unchanged (81 pass / 6 skip).

**Honesty / limits.** The CPU model (kinematic diff-drive + pivot drift + a drive dead-band) reproduces
the recorded failure and proves the cross-track mode converges from the recorded standoffs; it is
approximate (the real PhysX base dynamics differ). Across a grid of realistic far standoffs (cross
≤ 60 mm, yaw ≤ 5°, pivot 0–0.2) it converges ~81%; the failures are extreme-cross corners beyond the
nav arrival envelope. The real proof is the bounded GPU run below.

## Bounded GPU proof (for the integration lane — I did NOT take the GPU)

Config fragment `docs/examples/couple-fine-dock-crosstrack-fragment.json` — merge the `fine_dock` block
(only `approach_mode` + the `ct_*` gains differ from the shipped block) into the mission runtime config.
Preferred bounded staging: a collect-plus-couple run from the `live_pose` collect
(`--start-at`/`start_at` not available for couple-from-collect without the pick, so the full `coldlive`
config is the robust choice if the stage cannot be isolated) — watch `FINE_DOCK_DIAG` for
`status=ready` and the couple `action=couple status=succeeded reason=arrived_stopped` → attach. PASS =
the couple fine-dock latches (no `fine_dock time cap`); report the run dir in the done line.
