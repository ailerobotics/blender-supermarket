# Outbound / park / return navigation contract (coupled tow)

Mission-facing navigation contract for the stages **after `couple`**: towing the
coupled rollwagen from the backroom collect/couple pose to the `G3_back_s2` shelf
station, parking there for the transfer, and returning to the backroom. It fixes
the goal poses, success criteria, timing and the towing-geometry numbers that the
integration `isaac_bundle` defaults must match.

All feedback here is **TEST_FIXTURE quasi-static** (the `advance()` no-side-slip
trailer model), not a PhysX or localization demonstration. It certifies geometry
and controller feasibility at full margins; dynamics/settle stay integration-owned.

- Base: `main` @ `71ffffc`; controller `twin.navigation.articulated.ArticulatedController`
  built by `twin.navigation.factory:nav_factory` for every nav mode.
- Store manifest sha256 `e79d4c7e…` (798 grid obstacles = 797 measured static AABBs
  + 1 scenario staging support); scenario sha256 `fbea2c74…`
  (`scripts/restock_scenario.json`, job `restock-one`, `juice_orange`,
  `backstock_00` → `G3_back_s2`).

## Stage sequence and goal poses (map frame, `[x, y, yaw]`, metres/rad, Z up)

The mission (`twin.restock.mission.ACTIONS`) runs
`collect → couple → outbound → park → transfer → verify → return`. The navigation
lane owns `outbound` and `return`; `park` is a measured-settle wait, no motion.

| Stage | nav goal key | pose | coupled | body |
|---|---|---|---|---|
| couple / outbound **start** | `robot_dock` (= `robot_base_pose`) | `[-5.73, 16, 0]` | robot parks, then couples cart at `cart_dock` `[-7, 16, 0]` | robot base |
| **outbound** goal | `shelf_approach` | `[-3.05, 6.0, π]` | **yes** | robot base |
| **park** | — | hold at `shelf_approach` | yes | measured settle (base + cart) |
| **return** goal | `home` | `[-5.73, 16, 0]` | **yes** | robot base |

`home` resolves through `twin.restock.scenario.poses_for_job(..., return_pose=…)`
to `poses[return_pose]`, and `restock_scenario.json` sets `return_pose: robot_dock`,
so the return goal **is** `robot_dock` `[-5.73, 16, 0]`. (The static manifest's own
`poses['home']` `[-3.875, 14, 0]` is unused by the runtime, and is **not** reachable
coupled from the shelf — `blocked_no_route` — so do not route to it.)

Planar poses omit ground height: base-link Z = 0.24 m, cart body Z = 0.20 m.

## Success criteria (per coupled leg)

The controller declares `succeeded` / `arrived_stopped` only when **all** hold:

- position error `≤ 0.08 m`, heading error `≤ 0.08 rad` (4.58°);
- cart articulation `|yaw − cart_yaw| < goal_articulation` (per-leg tolerance below);
- base settled: `|linear| ≤ 0.02 m/s` **and** `|angular| ≤ 0.03 rad/s`;
- the zero-velocity command passes the full coverage + obstacle gate (`tracking`).

The mission runtime (`RuntimeAdapter`) then additionally requires the **measured
parked** flag (`packet.parked`, a base+cart pose-displacement check it owns) and,
for `outbound`/`return`, re-checks `|articulation| < goal_articulation[action]`,
raising `cart articulation exceeds manipulation/return goal tolerance` otherwise.

### Docking articulation — the cart arrives swung ~46°, not aligned

The hybrid-A\* planner can reach `shelf_approach` **only** with a cart articulation
of ≈ **46°** (outbound final 45.9°, return final 46.0°). A tighter, aligned dock
(≤ 45°) is **not achievable**: the planner times out with no route (confirmed at
45°/30°/20°/15°/10°/5° targets, ~36 s, ~3000 expansions each). The designed
straight G3_back pull-through (`[-1.525,6,π]→[-7,6,π]`) is a *local* clearance
proof, not a reachable aligned approach from the backroom.

Consequences for the mission:

- **`goal_articulation` for `outbound` and `return` must be `> 0.805 rad` (46.1°).**
  Recommended: **`0.90 rad` (51.6°)** — above the achievable 46° with tracking
  margin, below the 65° (`1.1345 rad`) structural `max_articulation` cap. The
  runtime rejects any value `≤ 0` or `> 65°`.
- The transfer/manipulation must use the **nav-produced** cart pose
  (`shelf_cart_body_pose`, derived from the outbound arrival by
  `check_restock_reach.py`), i.e. cart body ≈ `[-1.92, 6.51, −2.30]` for this
  scenario — **not** a collinear cart. Reach/sweep must be validated at that pose.

### Cart settle after towing

In the no-side-slip tow model cart yaw-rate is bounded by the base settle
(`cart_rate = (v·sinβ + hitch_x·w·cosβ)/cart_hitch_x`); at the settle thresholds
this is `≤ ~0.06 rad/s`, and the fixture measures **0.032 rad/s** on both legs at
`arrived_stopped`. The physical trailer **pendulum** settle after the base stops is
an integration-verified `parked` concern (the runtime's pose-displacement check),
not something the quasi-static model proves.

## Towing geometry (source of the `isaac_bundle` defaults)

All from `twin.mobile.MobileSpec`; `check_restock_navigation.py` now records these
in `summary.json → towing_geometry` so the defaults trace to one source.

| Quantity | Value | Note |
|---|---|---|
| **`transport_radius_m`** | **0.65** | Uncoupled transport envelope. Must enclose the hitch extent **0.621 m** and base rear corner **0.583 m**; the `isaac_bundle` default **0.55 is too small** (clips the hitch → arm-transport gate can never pass). |
| Min turn radius | **1.5 m** | Enforced in `gate_velocity` (`|w|·1.5 ≤ v`) and the rollout curvature cap (`|k| ≤ 1/1.5`). |
| Max articulation | **65°** (1.1345 rad) | Structural (hitch + stowed arm). |
| Cart body | 0.8 × 0.68 × 1.65 m | Centre **1.27 m** behind base-link; rearmost **1.67 m**; far corner **1.70 m** radial. |
| Coupled occupied envelope (aligned) | x ∈ [−1.67, +0.66] m, y ∈ [±0.66] m | Subtracted by the coupled coverage gate; the braking sweep adds the forward crescent. |

**Cart self-geometry / phantom rear obstacle.** The range sensor `self_paths` must
include the **cart root prim and its children** so lidar returns off the towed cart
are masked as self (UNKNOWN), not read as a rear obstacle. The cart swings ±65°
about the hitch, so the masked region is the **articulated cart body** (it moves
with the cart joint) — a fixed rear box is wrong. The coupled coverage gate already
exempts self-occluded side/rear beams via the occupied-envelope subtraction; the
self-mask is what prevents the cart from entering `obs.obstacles` as a phantom.

**Arm stow (`transport_joints`).** Navigation requires `obs.arm_transport` true —
the arm stowed inside the 0.65 m transport envelope — before it will drive
(`_inputs` → `arm_not_in_transport`). The joint targets are manipulation-owned; the
envelope radius they must satisfy is the 0.65 m above.

## Timing (fixture, 0.05 s cadence, 0.10 m localization allowance)

| Leg | from → to | status | peak / final articulation | cart settle | sim time |
|---|---|---|---|---|---|
| outbound | `[-5.73,16,0]` → `[-3.05,6,π]` | succeeded | 45.9° / 45.9° | 0.032 rad/s | **53.0 s** |
| return | `[-3.05,6,π]` → `[-5.73,16,0]` | succeeded | 46.1° / 46.0° | 0.032 rad/s | **97.7 s** |

Sim times are kinematic path-time bounds at `max_speed` 0.35 m/s, **not** physics
dynamics. Both legs are forward-only coupled (no reverse towing) and collision-free
at the full 0.05 m geometry margin + 0.10 m localization inflation.

## Reproduce (CPU, no ROS/GPU/Isaac)

```bash
# Build the current store manifest (headless, scratch output):
blender -b scenes/supermarket_base.blend -t 4 --python-exit-code 1 \
  --python scripts/build_supermarket.py -- --all --save "$OUT/store.blend" \
  --manifest "$OUT/environment.json"
# Certify the coupled outbound + return routes at the mission's articulation tolerance:
PYTHONPATH=scripts python3 scripts/check_restock_navigation.py \
  --environment "$OUT/environment.json" --scenario scripts/restock_scenario.json \
  --position-uncertainty 0.10 --goal-articulation 0.90 0.90 \
  --output "$OUT/navigation-proof"
```

`passed: true` requires both legs `succeeded` **and** `within_goal_articulation`.

## Native confirmation (planner-steered coupled towing)

A bounded guarded native probe (`run_nav_towing_probe.py` + `twin.restock.nav_towing_check`,
merged PR #70) drove **this** controller in Isaac at the couple pose: the mobile base +
rollwagen assemble at `[-5.73,16,0]`/`[-7,16,0]`, settle (robot ~1 mm/s), the hitch latches
(11 mm < 25 mm gate), the planner routes to the shelf and the controller **steers the coupled
train forward** with bounded articulation and **no cart wheel lift** — confirming the towing
geometry under physics.

- `nav-towing-probe-03` (default friction): forward **0.314 m**, cmd ω ≤ 0.167 rad/s, articulation 0.51°.
- `nav-towing-probe-04` (**wheel_friction=1.0**, traction fix 9988f6b, `WHEEL_TRACTION bound=2`):
  forward **0.559 m (+78 %)**, max measured speed 0.202 m/s (0.073 avg while driving), no wheel lift.

The traverse is partial because the 8-minute probe window advances only ~17 s of sim at
render-paced stepping (a probe-harness limit, not a mission limit — the integration mission runs
are uncapped). Evidence: `~/datasets/sim-restock-navigation-v2/nav-towing-probe-0{3,4}/`.

### Controller fix: keep the route on a transient `measured_turn_radius`

probe-04 showed ~36 % of tow steps were stop/replan: when the under-powered base is momentarily
under commanded speed on a tight turn, its **measured** arc is tighter than the 1.5 m min radius,
tripping the braking-curvature gate; the controller was discarding the whole planned route and
forcing a full stop + replan each trip, which cycled and stalled the tow. `ArticulatedController.step`
now (a) **slows the commanded speed for tight turns** (0.25·max_speed at the curvature cap, scaling
to full speed when straight) so the base does not over-rotate the measured arc, and (b) on a
transient `measured_turn_radius` **keeps the planned route and coasts** (resuming the same route once
the measured curvature recovers) instead of dropping it; other refusals (obstacle, coverage, map,
tracking error) still replan, and a persistent trip still fails via the blocked timeout. CPU-tested
(`test_transient_measured_turn_radius_keeps_route`, `test_tight_turn_commands_reduced_speed`).

## Open items

- `transport_radius_m` 0.65, `transport_joints`, and the cart-root `self_paths`
  entries are `isaac_bundle` default changes owned by the integration branch; this
  contract confirms the numbers they must use.
