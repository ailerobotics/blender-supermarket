# sim-robust-outbound-throat-yaw-align-v1

Keep the coupled outbound tow **aligned to the lane axis before the `sales_backroom_portal`
throat** (x≈-3.875), so the base+cart envelope never enters the 2.5 m opening yawed and wedges
on the lip. This is the UPSTREAM fix the PR 250 analysis called for: "the robust fix is upstream:
do not arrive at the lip yawed."

Branch: `fm/sim-robust-outbound-throat-yaw-align-v1` (base `fm/twin-robot-in-store-boot-v1`,
head after PRs 243–252). GPU held by the integration/validator lane — **no GPU run taken here**;
the bounded GPU proof config is handed to that lane (§3).

All numbers below are read on CPU from the runs' own `nav-trace.jsonl`
(`~/datasets/twin-robot-in-store-boot-v1/mission/<run>/`, config
`mission-runtime-capstone241.json`) and the real authored lane expanded against the real
environment manifest. The outbound lane `outbound_dock_to_bay`:
`start [-4.46, 15.5, 0.0]`, `goal [-2.0, 15.25, 0.0]` — **both endpoint headings 0° (due east)**;
the base tows due east through the portal at ~constant y≈15.25.

## 1. Quantified: why coldwin9/coldwin10 drifted and coldwin7/coldwin8/rock1 did not

### Frame
The authored centreline carries a local heading of **−6.21°** near the throat (an artefact of
its initial southward dip from y=15.5 to y≈15.45 — the base actually rides ~0.2 m south of it).
The physically meaningful reference for the throat crossing is the **travel axis = the lane goal
heading = 0° (due east)**: the direction the train must travel to pass the portal, and the
heading the passing runs held. `yaw_err = base_yaw − 0° = base_yaw`. (The validator framed the
same thing against a −5.80° axis; +4.8° "rel" there = −1.0° absolute = aligned east. The
absolute/east frame is used here because it is the throat-lip's own frame and makes the pass/fail
boundary a clean ~1° band.)

### The discriminator (five recorded outbound tows)

| run | seat yaw | post-dip yaw | yaw @ x=−4.1 | **yaw @ throat** | min rv in approach | outcome |
|---|---|---|---|---|---|---|
| coldwin7 | +2.4° | −0.0° | −1.2° | **−0.2°** | 0.152 | adv 2.46 m → bay ✓ |
| coldwin8 | +2.4° | +1.0° | −0.4° | **+0.2°** | 0.140 | adv 2.42 m → bay ✓ |
| rock1 | +2.5° | +0.2° | −0.5° | **+0.3°** | 0.006 | adv 2.45 m → bay ✓ |
| coldwin9 | +1.3° | −1.7° | **+3.1°** | **+4.9°** | 0.001 | adv 0.54 m, pinned x=−3.931 ✗ |
| coldwin10 | +2.5° | +1.2° | **+1.6°** | **+2.2°** | 0.001 | adv 0.56 m, pinned x=−3.935 ✗ |

**The seat is identical.** Every run begins the tow at ~(−4.48, 15.26, +2.5°) and, as PR 250
found, dips to near-rest during a brief authored-lane plan-wait (`planning_stop` →
`waiting_stopped_for_plan`) then re-accelerates. The drift is **not** a seat effect (the
validator confirmed this: coldwin7, rock1 and coldwin10 share the seat).

**The drift accumulates during the slow from-rest creep**, between the plan-wait dip (x≈−4.5)
and x≈−4.1:
- coldwin9 exits the dip *well aligned* (−1.7°) but re-accelerates to only ~0.11 m/s and drifts
  CCW monotonically: −1.7°(dip) → +3.1°(x=−4.1) → **+4.9°(throat)**.
- coldwin10 never aligns in the dip (+1.2°) and carries it: +1.2° → +1.6° → **+2.2°(throat)**.
- coldwin7/8/rock1 re-accelerate to a healthy ~0.15–0.19 m/s cruise and hold within ±1.2° of
  east across the whole approach, crossing the throat at **−0.2…+0.3°**.

**The wedge direction is CCW (nose north of east).** PASS runs cross at −0.2…+0.3°; FAIL runs at
+2.2…+4.9°. A yaw bound of ~1° cleanly separates the two populations; **coldwin10 wedged at only
+2.2°**, so the margin is ~2° (consistent with the validator: "about 2 deg of residual yaw at the
lip is the wedge margin").

### What the controller commanded in yaw rate, and why it could not correct

The coupled steering authority is capped at `wmax = v / 1.5` (the 1.5 m coupled minimum turn
radius — a real kinematic limit: a trailer cannot pivot in place). Averaged over the throat
approach:

| run | mean \|rw\| commanded | mean steering cap v/1.5 |
|---|---|---|
| coldwin7 | 0.142 rad/s | 0.120 rad/s |
| rock1 | 0.138 rad/s | 0.116 rad/s |
| coldwin9 | 0.070 rad/s | **0.032 rad/s** |
| coldwin10 | 0.071 rad/s | **0.033 rad/s** |

The failing runs crept ~3× slower, so their steering cap `v/1.5` was ~**0.03 rad/s** — roughly a
quarter of the passing runs'. The follower's pure-pursuit law *wanted* to correct the heading,
but at the crawl speed the authority was throttled to near zero, so the CCW drift (driven by the
base's known drive under-tracking) went uncorrected. **The authority deficit is the direct
consequence of the slow creep, because authority ∝ forward speed.**

### The throat is invisible to the nav controller

A decisive CPU check: at x=−3.90, y=15.26 the base+cart footprint is collision-free against the
static navigation map for **every** base yaw from −8° to +6° (0.29 m extra margin at each), and
the lateral slack along the whole approach is the probe ceiling (0.8 m) — the pinch only appears
at the bay (x≈−2.2). So the wedge is **PhysX contact against door-frame geometry finer than the
AABB nav map**: the gate returns `tracking` (never `obstacle_stop`) while the base physically
wedges. Consequences for the fix:
- The controller **cannot localize the throat** from its own occupancy — a slack-triggered
  pre-throat guard is impossible. The robust trigger is the **CCW yaw drift on the lane itself**.
- The physical un-wedge cannot be proved on CPU (no PhysX). CPU proves the controller *commands*
  the corrective heading and *converges* the yaw under the bound before the throat; the physical
  throat crossing is the bounded GPU proof (§3).

### Why the existing backstops don't prevent this

PR 201's net-progress monitor, PR 236's bounded attempts and PR 250's rock all fire **only after
the base is already pinned** at the lip — a forward-only recovery cannot free a *forward* wedge
(nine surges across coldwin9's three BT re-begins netted zero advance). They are the correct
backstop but act too late. The fix must act **upstream**, during the free creep, before the
envelope reaches the lip.

## 2. The fix — opt-in heading hold + pre-throat alignment (default off, byte-identical off)

One opt-in master key `navigation.coupled_heading_hold` (bool, default **false** ⇒
byte-identical), registered in `factory.py` exactly like the other switches, scoped to the
**coupled authored-lane follower only**. Reference axis = the lane **goal heading**
(`self.goal[2]`, the portal-crossing/travel axis; 0° = east for the outbound lane).
`ae = wrap(base_yaw − goal_heading)` is positive on the CCW / throat-lip (wedge) side.

**Engage** (in the coupled follow path, only while a route is being tracked):
`coupled_heading_hold` ON **and** an active authored lane **and** the local authored segment is
straight (`|route.curvatures[j]| ≤ heading_hold_straight_kappa`, so a legitimate authored turn is
never fought) **and** `ae > heading_hold_bound` (drifted beyond the bound toward the wedge side).
The guard is **one-sided**: a base on the safe (CW / south) side is never force-turned — turning
it would push it *toward* the lip. This is why it fires for coldwin9/coldwin10 (which drift CCW
past the bound) and **never for coldwin7/coldwin8/rock1** (which hold ≤ +0.3° east, below the
bound) — no creep-speed gate is needed; the CCW misalignment itself is the discriminator.

**When engaged** (the heading hold = the pre-throat alignment guard in one mechanism):
- `v = min(max_speed, heading_hold_creep_mps)` — a fixed, modest creep. This (a) **restores
  steering authority that does not depend on the tracking speed schedule** (`wmax = v/1.5` is
  pinned to the creep, not to whatever the collapsed schedule wanted), and (b) **caps the forward
  speed** so a misaligned base cannot accelerate into the throat — it *slows and corrects first*.
- `curvature = −(1 / min_turn_radius)` — the corrective turn **saturated** toward the axis
  (reduces `ae`; CW because `ae>0`). The command `(v, v·curvature) = (creep, −creep/1.5)` passes
  through the unchanged `gate_velocity`, so it is bounded by the coupled cap and
  braking-/footprint-swept-checked exactly like any command. **Forward (`v>0`) always — the guard
  never commands a negative linear velocity**, honouring the coupled-reverse interlock.

Because the correction rate (~5°/s at the creep) dominates the measured drift rate (~0.4°/s), the
hold drives `ae` down toward the axis far faster than it accumulates, so the base reaches the
throat aligned instead of yawed. The hold keeps the base creeping forward, so the net-progress
monitor never trips and PR 250's rock stays dormant — **PR 201's monitor, PR 236's bound and PR
250's rock are untouched and remain the backstop** if a wedge still occurs.

Keys (one master bool + two numerics, `factory.py`, all defaulted so absent ⇒ byte-identical):
- `coupled_heading_hold` (bool, default `false`)
- `heading_hold_bound_deg` (float, default `1.0`) — the CCW align-error firing threshold.
  1.0° sits safely below coldwin10's failing +2.2° (≈1.2° margin) and above the passing runs'
  ≤ +0.3°.
- `heading_hold_creep_mps` (float, default `0.14`) — the guaranteed authority creep (restores
  `wmax = 0.14/1.5 ≈ 0.093 rad/s`) and the pre-throat speed cap.
- `heading_hold_straight_kappa` (float, default `0.08` 1/m) — only hold heading on a straight
  authored segment (the portal crossing has curvature ≈ 0; a real authored turn is never fought).

### CPU tests
`scripts/twin/tests/test_outbound_heading_hold.py` (recorded-trace, fixture
`fixtures/outbound_throat_yaw_nav.json` — the real coldwin9/coldwin10/coldwin7/rock1 outbound
creeps) + cases in `test_articulated_navigation.py`:
- the recorded **coldwin9** and **coldwin10** creeps, replayed through the REAL controller,
  command a corrective (CW, negative) yaw rate before the throat and the closed-loop base
  converges `ae` under the bound before x=−3.875 — *arriving aligned*;
- **coldwin7** and **rock1** never engage the hold (they stay within the bound) ⇒ unchanged;
- **key off ⇒ byte-identical** (same `(v, w)` sequence as the legacy follower on every run);
- the guard **never commands a negative linear velocity** on any replay.

## 3. Bounded GPU proof (handed to the integration lane — do not run here)

`docs/examples/outbound-heading-hold-mission-fragment.json`: couple + outbound from the
coldwin9/coldwin10 seat with `coupled_heading_hold: true` (and the PR 250 rock on, as the
backstop). PASS = **yaw at throat entry within the bound, no `NAV_BREAKAWAY enter`, arrival at the
transfer bay**. Grade the resulting nav-trace with the shipped gate (full arg set):
`python3 scripts/check_tow_traction.py --environment <env.json> --scenario <mission-scenario.json>
--camera-config <camera-config-bay.json> --fragment docs/examples/coupled-traction-fragment.json
--runs-dir <mission-dir> --gpu-run <run-dir>` → `GPU_RUN PASS` (reached_goal=True, no multi-second
pin; coldwin9 scores `GPU_RUN FAIL` reached_goal=False longest_pin_s=27.2) — then the full capstone.

GPU rules honoured: one run at a time via `flock /tmp/aile-gpu.lock`, never forced, `whispr`
untouched. One GPU run at a time; the validator owns the card.
