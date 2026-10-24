# sim-robust-throat-momentum-launch-v1

Make the aligned coupled outbound tow cross the `sales_backroom_portal` throat (x≈−3.875) at
y≈15.25 deterministically, after PR 258 removed the **yaw** cause but coldwin18 still pinned
**from rest while aligned** at x=−3.935.

Branch `fm/sim-robust-throat-momentum-launch-v1` off `fm/twin-robot-in-store-boot-v1` @ 479572a.

## 1. What the base strikes at x=−3.935 — CPU scene finding (do this BEFORE assuming)

**There is no collider at the pin.** A CPU scan of the composed mission stage
(`mission.usda` + its sublayer `supermarket.usdc`, read with the `.venv-usd` `usd-core`) over the
region around the pin (x ∈ [−4.6, 1.0], y ∈ [12.5, 17.5]) returns, for **every** `UsdGeom.Gprim`
AND every `UsdPhysics.CollisionAPI` prim:

| prim | world AABB | note |
|---|---|---|
| `GRP_Structure/Backroom_floor` | x[−12, 1] y[11, 19.25] **z[−0.12, 0.0]** | single flat plane, top z=0 |
| `GRP_Structure/Floor` (sales) | x[−13, 13] y[−11, 11] z[0, 0] | meets the backroom floor at y=11, same top |
| `GRP_Structure/Ground_exterior` | x[−60,60] y[−60,60] z[−0.03,−0.01] | underlay, 1 cm below, no interference |
| `GRP_Structure/Backroom_ceiling` | … z[4.0, 4.05] | overhead |
| `GRP_Structure/Backroom_wall_east` | **x[1.0, 1.25]** y[11, 19.25] z[0, 4.3] | 3 m east of the pin |
| `RestockScenario/TransferBay` | x[−2.35,−1.65] y[16.17,16.67] z[0, 0.73] | NORTH + east, off the y=15.25 path |
| `RestockScenario/Support`, `HandlingUnit` | x[−6.07,−5.39] y[16.86,17.34] | the source, behind/NW of the cart |
| `backroom_turning_line_*` | z[0.002, 0.004] | painted floor **decals**, non-physical |

There is **no door threshold, sill, jamb, wall, step, or floor seam** anywhere the base, its wheels,
its casters, or the swept base+cart envelope pass through at the pin. The `sales_backroom_portal`
jambs/lintel the store authors (`store_environment.py`) are the real opening at **y=11.125** — ~4 m
south of the y=15.25 outbound lane; the "throat" label on the heading-hold config is geometric
shorthand for the x the from-rest tow stalls at, not a modelled doorway there.

### Base rig at the impact x (sketch; `MobileSpec`: length 1.0, width 0.6, wheel_track 0.7, wheel_radius 0.16, two trailing casters at local x = ±0.38, `caster_lift_m`=0.03 baked in coldwin18)

```
   travel  →  +x (east)                     throat x=-3.875
   y≈15.25                                        |
                 rear caster      drive axle      |   front caster     base front
   world x:        -4.315           -3.935        |     -3.555           -3.435
                     o========[ base_link ]========o        o              |
                   (caster)   L_wheel  R_wheel   (caster)  (caster)      (face)
                      ^                              ^         ^
                 trails, Z-swivel          two driven wheels   trails, Z-swivel
   floor:  ================= flat plane, top z = 0.000, NO step / sill / seam =================
   cart (towed, 1.27 m behind via drawbar hitch):  x≈-5.20, pushing +x into the base
```

At the pin the drive axle is **on** x=−3.935, the throat label sits 0.06 m ahead between the axle and
the front caster, and nothing in the table above intersects any of these x. The strike is therefore a
**PhysX contact of the wheels/casters on the floor**, not a modelled obstacle.

## 2. The pin signature — one-tick collapse, not a brake (coldwin18 nav-trace)

`/home/manishsaini/datasets/twin-robot-in-store-boot-v1/mission/coldwin18`, outbound rows:

```
tick 13282  t=221.367  x=-3.9360  rv=0.1830  cart_v=0.2013  hitch=0.1996   (rolling, aligned yaw +0.5°)
tick 13283  t=221.383  x=-3.9350  rv=0.0063  cart_v=0.0238  hitch=0.1987   ← ONE 16.7 ms tick
… then x pinned at -3.935 for ~27 s SIM (t 221.4 → 248.3), rv jitter < 0.09, net < 0.01 m, → coupled_no_progress
```

`rv 0.183 → 0.006 m/s` in one 16.7 ms tick is a ≈10 m/s² deceleration — far beyond anything the 200 N
drive can brake (200 N / ~250 kg ≈ 0.8 m/s²), so it is an **impulsive contact**, and the cart keeps
pushing (`cart_x` stays ≈−5.205; hitch articulation drops 0.1996→0.1987 as the cart loads the stopped
base). The approach speed (rv≈0.15–0.19 over x −4.3…−3.94), the lateral y (15.25) and the base yaw
(≈0°, aligned) do **not** separate coldwin18 from the passers — confirmed across coldwin16/11/hold2
(pass) and coldwin18/17/9/10 (pin). The only systematic difference the brief identifies is
straight-travel-into-the-spot (passers) vs just-corrected (pinners): a trailing caster left swiveled
off-axle by the pre-throat alignment turn **scrubs instead of rolls** from the near-stop, so the drive
wheels + the cart push cannot re-break-away. This is consistent with the stochasticity (contact
history, not geometry) and the repeatable x (the lane speed + the gate bring the train to its
lowest-momentum state right at the throat).

### Note on the brief's trace reading

The brief read `vel.cv` as "commanded linear velocity (stays positive)". In the trace `cv` is the
**cart body's measured** linear speed, not a command (the controller command is not recorded); it
collapses with `rv` at the pin. So "commanded stays positive while measured drops" is not supported by
the trace — both train bodies stop together, which is what an impulsive wheel/caster contact (or a
scrub stall) does.

## 3. The fix — opt-in momentum launch (forward-only, default off ⇒ byte-identical)

`navigation.heading_hold_launch_m` (+ `heading_hold_launch_mps`), layered on the PR 258 pre-throat
gate. Within the **launch window** (`|ahead| ≤ launch_m` of the throat along the travel axis, either
side of the lip) and **once aligned** (`|ae| ≤ heading_hold_bound`), the follower command is overridden
to a **dead-straight forward cruise** — `curvature = 0` (no steering corrections, so the casters trail
and straighten) at `max(follower_v, launch cruise)` (carry momentum, never crawl), forward only. The
alignment gate is scoped to `(launch_m, prethroat_m]` so alignment **completes before** the launch
window; a base still misaligned on entry keeps the bounded PR 258 gate (named `throat_misaligned`), so
the launch never fires misaligned and there is never a silent wedge. Construction enforces
`0 < launch_m < prethroat_m` and `prethroat_m > 0`.

Backstops untouched: the launch only acts while PR 250's break-away is idle (`_breakaway_cmd is None`),
so PR 201's monitor / PR 236's bound / PR 250's rock own any residual pin exactly as before.

`heading_hold_launch_m = 0.0` ⇒ `in_launch` always False ⇒ the branch reduces **exactly** to the PR 258
gate (asserted byte-identical by test + the 55 existing heading-hold/factory tests).

### Trace discriminator (step 2)

`RESTOCK_TRACE_CASTERS=1` (default off ⇒ no extra fields, no stage reads, byte-identical) adds to each
nav-trace row `casters: {swivel_deg:{front,rear}, wheel_z:{…}, contact:{…}}` — the two trailing-caster
swivel angles (the hypothesised discriminator) and a geometric wheel-ground-contact proxy (each
wheel/caster link's world-z vs its rolling radius above the flat floor). Wired in `isaac_port.py`
(`_caster_trace`) + `isaac_bundle.py`; the swivel angles come from the measured encoder dict (reliable),
the contact proxy from cached prim world transforms (best-effort, never raises).

## 4. Acceptance gate + bounded GPU proof (for the integration lane; not run here)

`scripts/check_tow_traction.py --gpu-run <run>` now also reports `single_tick_stops`
(`twin/traction.throat_single_tick_stops`: a one-tick `≥0.12 → ≤0.03 m/s` collapse west of x=−3.90 — the
pin signature) and `trace_has_casters`, and requires `single_tick_stops == 0` for PASS on top of
`reached_goal` + no multi-second pin. Validated on the recorded runs:

```
coldwin18  GPU_RUN FAIL  reached_goal=false longest_pin_s=26.95 single_tick_stops=3 first_stop=[-3.935,15.252]
coldwin16  GPU_RUN PASS  reached_goal=true  longest_pin_s=1.58  single_tick_stops=0
coldwin11  GPU_RUN PASS  reached_goal=true  longest_pin_s=1.53  single_tick_stops=0
```

GPU proof config: `docs/examples/throat-momentum-launch-mission-fragment.json` (the PR 258
heading-hold-bound block + `heading_hold_launch_m=0.25`, `heading_hold_launch_mps=0.18`). Recipe:
`run_mobile_restock.py --start-at couple --stop-after outbound` from the coldwin18 seat with
`RESTOCK_TRACE_CASTERS=1`, serialized through flock `/tmp/aile-gpu.lock`, `HEAVY_MB=2500` if whispr
holds VRAM, **never `--force`, never touch whispr (pid 1339)**, repeated ≥2× (the pin was stochastic).
PASS = `GPU_RUN PASS` (reached the bay, `single_tick_stops=0`, `trace_has_casters=true`) + no
`NAV_BREAKAWAY enter` / `throat_misaligned` in the console; then the full capstone.

## 5. Asset-artifact note (no asset change here)

Because the pin is a PhysX wheel/caster contact on a **flat, collider-free** floor and the throat has no
modelled door threshold, the robust fix is the momentum launch, not an asset edit. If a future review
wants a *realistic* backroom-to-sales-floor sill, it would be a modelled step of a few mm at the true
portal (y=11.125) on `store_environment.py` — a separate asset change, out of scope here, and it would
make the throat traction gate stricter, not laxer.

## Scope

Out of scope (untouched): any reverse motion / the coupled-reverse interlock (`mobile.wheel_targets`
`v≥0`, `gate_velocity` `linear≥0`), PR 241's lead-in, PR 201's thresholds, the couple fine-dock, and any
cart/caster friction or mass change.

## Files

- `scripts/twin/navigation/articulated.py` — `heading_hold_launch_m`/`_mps` + the launch-window command.
- `scripts/twin/navigation/factory.py` — config wiring.
- `scripts/twin/restock/isaac_port.py`, `isaac_bundle.py` — `RESTOCK_TRACE_CASTERS` trace fields.
- `scripts/twin/traction.py`, `scripts/check_tow_traction.py` — `throat_single_tick_stops` + `--gpu-run` gate.
- `docs/examples/throat-momentum-launch-mission-fragment.json`, `docs/restock_navigation.md`.
- CPU tests: `scripts/twin/tests/test_outbound_throat_launch.py` (+ fixture
  `fixtures/outbound_throat_launch_nav.json`), `test_navigation_factory.py`, `test_tow_traction.py`.
