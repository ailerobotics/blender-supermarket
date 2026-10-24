# sim-robust-coupled-breakaway-from-rest-v1

Make the coupled outbound tow break away reliably at the `sales_backroom_portal` throat
(x=-3.875). The full every-stage-honest cycle passed twice (coldwin7, coldwin8) then failed
once at the outbound (coldwin9, 2026-10-24): collect + couple clean, then the coupled tow
pinned at x≈-3.93 and faulted `coupled_no_progress` on all three BT re-begins.

Branch: `fm/sim-robust-coupled-breakaway-from-rest-v1` (base `fm/twin-robot-in-store-boot-v1`).
GPU held by the integration/validator lane — **no GPU run was taken for this task**; the GPU
proof config is handed to that lane.

## 1. Diff of the tow start: coldwin9 (FAIL) vs coldwin7/coldwin8 (PASS)

All numbers are read from the runs' own `nav-trace.jsonl` (SI; `world_t` is the sim/node clock)
and `*.console.log` (the nav controller `NAV_STEP`/`NAV_BREAKAWAY` prints; these are **not** in
`host-stack.log`, which is the ROS host stack). Runs under
`~/datasets/twin-robot-in-store-boot-v1/mission/`, config `mission-runtime-capstone241.json`
(`coupled_progress_recovery: true`, `coupled_breakaway_speed_mps: 0.18`, `wheel_drive.max_force
200`, `wheel_friction 1.0`, caster/traction as landed by PR 207).

### The handoff is identical in all three runs

| quantity (couple→outbound) | coldwin7 PASS | coldwin8 PASS | coldwin9 FAIL |
|---|---|---|---|
| couple-latch base speed `rv` | 0.555 | 0.574 | 0.565 |
| first outbound-tick `rv` | 0.388 | 0.413 | 0.400 |
| cart articulation at latch | 0.195 | 0.195 | 0.195 |
| min `rv` in the first second (plan-wait dip) | **0.018** | **0.007** | **0.022** |

The base enters `outbound` still rolling (~0.4 m/s) in **every** run, then the authored-lane
route is not ready for a few ticks (`NAV_STEP reason='planning_stop'` →
`'waiting_stopped_for_plan'`) so the controller brakes the base to **near rest** (rv 0.007–0.022)
before `route_ready`. So the brief's framing — "coldwin7/8 began while still rolling, coldwin9
began from a dead stop" — is **not** what the traces show: all three dip to near-rest at the
handoff, all re-accelerate. There is **no** static-vs-rolling or drive-authority difference *at
the latch*.

### The real discriminator: a yaw-heading drift during the slow from-rest creep

After the plan-wait dip, the base re-accelerates and creeps east toward the throat. What differs
run-to-run is the **base heading** it carries into the 2.5 m portal opening:

| x band (throat approach/crossing) | coldwin7 | coldwin8 | coldwin9 |
|---|---|---|---|
| base `yaw` entering x≈-4.1 | -1.2° | -0.4° | **+3.1°** |
| base `yaw` at x≈-3.93 (throat lip) | ~-0.6° | ~0.0° | **+4.8°** |
| cruise `rv` across the throat | 0.17–0.21 | 0.18–0.21 | 0.21 → **0.035** (hard stop) |
| outcome | advance 3.86 m to bay | advance 3.82 m | **pinned at x=-3.931** |

coldwin9's heading drifts **monotonically** through the creep: -1.7°(t=214.1, x=-4.50) → 0°(215.7)
→ +1.2°(216.4) → +3.5°(217.4) → **+4.8°**(218.0, at the throat). coldwin7/8 hold their heading
within ±1.2° across the same band and sail through. The drift is a slow CCW rotation the lane
follower does not correct at the crawl speed (poor steering authority `wmax = v/1.5` ≈ 0.12 rad/s
at v=0.18, plus the base's known under-tracking drive).

### The stall is a forward wedge on the throat lip, not a gate refusal

At t=218.12 coldwin9's `rv` collapses 0.222 → 0.035 in one tick at x=-3.932, y=15.254,
**yaw=+4.8°**, and from there the base is pinned at x=-3.931 ± 0.001 for ~27 s: `rv` flickers
0.003–0.09 (it *is* commanding and the wheels turn a little) but **net Δx = 0**, `parked=1`,
cart articulation steady 0.199. A base+cart envelope yawed ~5° off the lane axis catches the
throat lip; the base commands forward and the gate returns `tracking` (not `obstacle_stop`), so
this is a **physical contact wedge**, not a controller/coverage refusal.

### Why PR 236's recovery and the three BT re-begins all failed

`coupled_progress_recovery` *was* on and *did* fire — `coldwin9.console.log` has 12
`NAV_BREAKAWAY` lines: 3 BT outbound attempts × (enter attempt=0 / next=1 / next=2 / exhausted),
all at x=-3.931/-3.932. PR 236's maneuver is a **forward-only** square-wave surge (0.18 m/s, with
reverse deliberately clamped off for a coupled train — `mobile.wheel_targets`). Nine forward
surges across three re-begins produced **zero net advance**: pushing forward only drives the
wedged corner harder into the lip. A re-begin from rest re-commands the same forward surge into
the same wedge, so it repeats the same result — exactly the failure the brief describes.

### Summary (why "momentum succeeds, rest fails")

The protective factor in the passing runs is not entry momentum at the latch (identical) but that
they crossed the throat **before** a heading error could accumulate and glanced off the lip while
moving nearly axis-aligned. The failing run crept slowly, accumulated a ~5° yaw error, arrived
slow enough to wedge, and then a forward-only recovery could not free a forward wedge.

Gate reproduction with the shipped acceptance grader `twin.traction.coupled_tow_margin` on the
recorded outbound rows:

```
coldwin7: reached_goal=True,  longest_pin_s=1.25,  max_advance=2.459 m, passed=True
coldwin9: reached_goal=False, longest_pin_s=27.10, max_advance=0.537 m, passed=False
```

## 2. The fix — opt-in bounded break-away ROCK (default off, byte-identical off)

Honest constraint: the physically correct unwedge of a forward wedge is a short **reverse**
back-off, which the brief explicitly sanctions ("a short reverse-then-forward rock"). But reverse
towing is a **deliberate safety interlock** — `mobile.wheel_targets` hard-clamps a coupled train
to `v >= 0` ("Reverse towing is deliberately disabled for initial bring-up") and
`gate_velocity` rejects `linear < 0` as `reverse_or_invalid_command`. Flipping that interlock
spans three layers (gate, wheel layer, runtime) plus a backward swept-envelope collision proof
for a *pushed* trailer, and is safety-sensitive. That is out of proportion for a default-off
robustness key and is recorded here as the next lever if the GPU proof shows the rock
insufficient (owner sign-off required to relax the interlock).

The shipped fix stays within `v >= 0` and makes the break-away attempts **differ from a plain
forward restart** by giving the recovery more of the two things the measured mechanism needs —
momentum and steering authority:

- **`navigation.coupled_breakaway_rock`** (bool, default **false** ⇒ byte-identical). Master
  switch; requires `coupled_progress_recovery` true (fails loud at construction otherwise — no
  silent no-op).
- **`navigation.coupled_breakaway_launch_mps`** (float, default 0.30). The elevated forward
  surge speed used during recovery when the rock is on, replacing the 0.18 m/s pulse. It is
  clamped to the controller `max_speed` at the gate, so it is bounded in effort; the lo phase
  stays 0.4× to "rock" the contact. A higher v also raises the coupled steering cap
  `wmax = v / 1.5`, so the wedged base gets ~1.7× more turn authority to rotate back toward the
  lane heading.
- On the **hi** phase of each pulse the commanded turn is **saturated** to the full `±wmax` in
  the follower's corrective direction (toward the route heading), actively rotating a
  yaw-wedged base toward alignment; the lo phase relaxes to the clamped tracking turn. This is
  the "rock to align the casters" with the only authority a coupled train has (forward + yaw).

All of it lives inside the existing `coupled_progress_recovery` path, which fires **only** when
the tow is pinned (net < `coupled_progress_min_m` for `coupled_progress_escape_s` sim seconds) —
coldwin9's condition at each re-begin, never coldwin7's clean tow. PR 201's net-progress monitor
and PR 236's three-attempt bound are untouched and remain the backstop: if the rock does not free
the base within `coupled_breakaway_attempts` windows it still faults `coupled_no_progress` by
name, fast on the sim clock.

Boundedness: effort ≤ `max_speed` (and ≤ `coupled_breakaway_launch_mps`), turn ≤ `v/1.5`;
distance/time bounded by the unchanged `coupled_breakaway_window_s × coupled_breakaway_attempts`.

### CPU tests

`scripts/twin/tests/test_articulated_navigation.py` (rock suite) and
`scripts/twin/tests/test_coupled_breakaway_rock.py` (recorded-trace suite, fixture
`fixtures/outbound_breakaway_nav.json` — the real coldwin7/coldwin9 outbound windows):

- the recorded **coldwin9** rest/stall start trips the acceptance grader (pinned
  `longest_pin_s` ≫ `escape_s`) and, driven through the real controller, **enters** the rock
  recovery and commands the elevated launch surge with a saturated corrective turn;
- the recorded **coldwin7** momentum start never pins (`longest_pin_s` < `escape_s`) and never
  enters the recovery — `breakaway_recoveries == 0`;
- the rock is bounded in effort (≤ max_speed), turn (≤ v/1.5) and attempts/time;
- **key off ⇒ byte-identical**: with `coupled_breakaway_rock` false the surge speed, pulse and
  fault ladder are exactly PR 236's (the legacy recovery tests still pass unchanged).

## 3. Bounded GPU proof (handed to the integration lane — do not run here)

`docs/examples/coupled-breakaway-rock-mission-fragment.json`: couple + outbound, deliberately
stopping the base fully after the latch before the tow (the coldwin9 condition) with the rock on.
PASS = break away on the first attempt and reach the transfer bay with **no** `coupled_no_progress`.
Grade the resulting nav-trace with the existing gate:
`python3 scripts/check_tow_traction.py --gpu-run <run-dir> …` (coldwin9's trace scores
`reached_goal=False, longest_pin_s=27.1 → FAIL`; a passing run scores `reached_goal=True,
longest_pin_s < escape_s → PASS`). Then the full capstone.

GPU rule honored: one run at a time via flock on `/tmp/aile-gpu.lock`, never forced, never
touching `whispr` (pid 1339, ~2.2 GB). The validator lane owns GPU priority.

## Out of scope (per brief)

PR 241's lead-in overshoot, PR 201's monitor thresholds, the couple fine-dock, and any change to
cart mass or to cart/caster friction values (none were changed). The couple-end heading variance
that seeds the yaw drift is a fine-dock property and is left to that lane; this task makes the
tow robust to it at the throat.
