# sim-robust-heading-hold-bound-v1 — two-sided pre-throat heading alignment

**Goal.** Make the coupled outbound tow enter the `sales_backroom_portal` throat (x≈−3.875)
**aligned to the travel axis regardless of its lateral offset or seat yaw**, so the base+cart
envelope never wedges on the lip. The coldwin17 FAIL (2026-10-24, byte-identical to the COMPLETE
coldwin16) proved PR 253's pre-throat heading hold was not enough; this closes the last stochastic
blocker of the deterministic cold cycle.

Branch `fm/sim-robust-heading-hold-bound-v1` off `caa3fcd` (twin head, incl. PR 253). **No GPU run
taken here** — the validator/integration lane owns the card; the bounded GPU proof config is handed
off (§3).

---

## 1. Finding — reproduced on CPU through the real controller, BEFORE any code change

Evidence: `coldwin17`, `coldwin16`, `coldwin11` run dirs under
`~/datasets/twin-robot-in-store-boot-v1/mission/`; reproduction scripts
`data/sim-robust-heading-hold-bound-v1/analyze_throat_yaw.py` (+ `build_fixture.py`).

### 1a. The three runs at the throat

| run | outcome | approach yaw (map frame, 0°=east) | at the lip | note |
|-----|---------|-----------------------------------|-----------|------|
| coldwin16 | **COMPLETE** | −0.3 → **+3.3°** peak then back to +2.0 | +2.0° @ x=−3.88, y converging north 15.23→15.26 | threaded easily |
| coldwin11 | **PASS** | +0.1 → **+5.5°** | +5.2° @ x=−3.86 | threaded on momentum (y converging) |
| coldwin17 | **FAIL** | −1.2 → **+5.18°** @ x=−4.079 | **pinned at x=−3.931, +4.4°, 31 s** | wedged; NAV_BREAKAWAY exhausted ×2 → `coupled_no_progress` |

All three share the **same CCW yaw drift on the approach**. coldwin17 just pinned — a yawed entry is
a **stochastic wedge**, exactly what the captain asked to remove ("less prone to failure in any
case").

### 1b. The drift is follower-lateral-term driven (reproduced through the real law)

The authored outbound lane `outbound_dock_to_bay` runs `start [-4.46, 15.5, 0°]` → `goal [-2.0,
15.25, 0°]`; the centreline **descends** y 15.5→15.25, so near the throat its position is y≈15.45
while the base creeps at **y≈15.24 — ~0.25 m south of the centreline**. The coupled follower law
(`articulated.py`: `curvature = authored_curv + 1.5·(ref_yaw−yaw) + 2·lateral`) therefore carries a
large lateral term `2·lateral ≈ +0.52` that commands a **CCW** (positive) turn to chase the
centreline north.

Applying that **real law** to the recorded coldwin17 poses over the approach window x∈[−4.35,−3.95]
(`analyze_throat_yaw.py`):

```
APPROACH coldwin17: rows=154  yaw 0.8->4.7 deg  lateral +0.256->+0.187 m
    recorded cw>0 (CCW) ticks=154/154  | follower-law turn: CCW=138 CW=16
coldwin16: rows=160  yaw 1.1->2.7 deg  recorded cw>0=160/160  follower CCW=148 CW=12
coldwin11: rows=160  yaw 0.1->5.3 deg  recorded cw>0=160/160  follower CCW=147 CW=13
```

The **recorded** commanded angular velocity (`vel.cw`) is positive (CCW) on **100 %** of approach
ticks while the yaw grows; the **follower law replayed through the real code** reproduces this (CCW
on ~90 % of ticks). The drift is not noise — it is the follower's own cross-track correction, and
the base never translates north fast enough (a coupled train can't slew laterally), so the yaw grows
instead.

### 1c. Why PR 253's hold did not stop it — three gaps

1. **Disabled where the drift builds.** The authored **merge lead-in** (which bridges the ~0.25 m
   seat offset) is a *curved* segment. PR 253's hold only fires on a *straight* segment
   (`|curv|≤heading_hold_straight_kappa`), so it is **off across the whole lead-in**. In the real
   run the hold first fired at **x=−4.079 with base_yaw already +5.18°** (`coldwin17.console.log`:
   `NAV_HEADING_HOLD n=1 at=(-4.079,15.253) base_yaw=5.18`) — ~0.2 m from the lip, far too late. (The
   ledger's "engaged 20×" is log-lines; the hold logs every 30th engagement, so it actually engaged
   ~570+ ticks — almost all *at the pin*, after the wedge.)
2. **Too little authority at the lip.** Once pinned at the crawl the creep's `wmax=creep/1.5` can't
   rotate a wedged base out of the lip in the remaining runway.
3. **One-sided, no throat gate.** It only corrected the CCW side (a CW seat drift would be ignored),
   bounded against the lane *goal* heading rather than being a true alignment gate, and a base that
   failed to align simply wedged with **no named failure**.

**Conclusion.** The robust fix must (a) act across the whole pre-throat approach (curved lead-in
included), (b) cap/override the lateral term so it can never drive the yaw further off-axis, (c) be
two-sided, and (d) gate the throat entry with a bounded, *named* failure.

---

## 2. The fix — two-sided pre-throat alignment gate (opt-in, default off ⇒ byte-identical)

One master sub-key `navigation.heading_hold_prethroat_m` (float, **default 0.0** ⇒ the new path is
off and PR 253's one-sided hold is **byte-identical**), under the existing `coupled_heading_hold`.
When `> 0` (requires `coupled_heading_hold` **and** a `heading_hold_throat` `[x,y]`), the hold
becomes a **pre-throat alignment gate** scoped to the window within `heading_hold_prethroat_m` of the
throat measured **along the travel axis** (engages only while the throat is still *ahead*):

- **Reference axis** = `heading_hold_axis_deg` if set, else the lane **goal heading** (the
  portal-crossing/travel direction; 0° east for this lane). This is the **min-clearance heading
  through the x-gap** — the heading the passing runs held — **not** the descending centreline tangent
  (−5.8°), which would widen the base+cart envelope through the opening. (PR 253's physical argument,
  kept.)
- **Two-sided**: fires when `|yaw − axis| > heading_hold_bound_deg` on **either** side.
- **Slow to the creep** `heading_hold_creep_mps` (restores `wmax=creep/1.5` independent of the
  collapsed speed schedule; caps forward speed so a misaligned base aligns rather than accelerating
  into the throat).
- **Saturate the corrective turn toward the axis**, two-sided: `curvature =
  −sign(ae)·(1/min_turn_radius)`. Fully **overriding** the follower curvature here subsumes the
  lateral-offset correction, so the lateral term **can never command a turn that increases the axis
  yaw inside the window** (brief clause B, strongest form) — on the curved lead-in and the straight
  segment alike.
- **Forward-only** (`v ≥ 0`; the coupled-reverse interlock is untouched).
- **Bounded, named failure**: if the base cannot reach within the bound inside
  `heading_hold_align_timeout_s` SIM seconds (default 20), fail by `throat_misaligned` (→ StageRetry)
  rather than wedging silently.
- Skipped while PR 250's break-away owns the command. **PR 201's monitor, PR 236's bound and PR 250's
  rock stay the untouched backstop.**

Size `heading_hold_prethroat_m` to the **straight run-up** to the throat — the window is for aligning
to the travel axis, not for spanning a genuine authored turn.

New keys (all neutral/off by default, byte-identical off): `heading_hold_prethroat_m` (0.0),
`heading_hold_throat` (null), `heading_hold_align_timeout_s` (20.0), `heading_hold_axis_deg` (null).
Wired in `scripts/twin/navigation/factory.py`; validated in `__init__`.

### 2a. CPU proof — all acceptance criteria

`scripts/twin/tests/test_outbound_heading_hold_bound.py` (fixture
`fixtures/outbound_throat_yaw_bound_nav.json` — the real coldwin17/16/11 approach creeps on the real
descending lane):

- **coldwin17**: replayed through the real controller with the gate ON, commands a **CW (negative w)
  turn that reduces the axis yaw from the first corrective tick** and drives the base **within the
  bound before x=−3.95**. Closed-loop, all three runs converge to **~+0.9° at the lip** (vs the
  recorded +5.5° wedge).
- **coldwin16 / coldwin11** (the passers): not degraded — forward progress maintained, **every**
  gate correction turns *toward* the axis (`w·ae ≤ 0`), still reach the crossing, **no negative
  linear velocity**.
- **Two-sided**: a mirror CW-seat drift (yaw → −yaw) is corrected with a **CCW (+w)** turn (PR 253's
  one-sided hold would never fire).
- **Key off byte-identical**: `heading_hold_prethroat_m = 0` (even with a throat set) reproduces the
  legacy one-sided PR 253 command stream exactly, on every run.

Regression: PR 253's `test_outbound_heading_hold.py` (7), `test_navigation_factory.py` (42, incl. the
new wiring + validation), and the articulated/lanes/traction/breakaway/couple-speed/mission-tail/
return-cartyaw suites all pass unchanged.

---

## 3. Bounded GPU proof (handed to the integration lane — do not run here)

`docs/examples/heading-hold-bound-mission-fragment.json`: MERGE into the mission runtime-config
navigation block **on top of** the PR 253 / breakaway / traction keys. Couple + outbound from the
coldwin17 seat (or the coldwin9 seat), `heading_hold_prethroat_m: 0.8`, `heading_hold_throat:
[-3.875, 15.26]`, align-timeout 20 s, with PR 250's rock on as the backstop.

Recipe: `run_mobile_restock.py --start-at couple --stop-after outbound` (or the full coldwin17/
capstone recipe), `RESTOCK_NAV_PLAN_DIAG=1`, serialized through `flock /tmp/aile-gpu.lock`,
`HEAVY_MB=2500` if whispr holds VRAM, **never `--force`, never touch whispr (pid 1339, ~2.2 GB)**.

**PASS** = base `|yaw|` about the east travel axis **within the bound at throat entry**, **no
`NAV_BREAKAWAY enter`** (never pins), **no `throat_misaligned`**, and arrival at the transfer bay.
Grade the nav-trace with the shipped gate:
`python3 scripts/check_tow_traction.py --environment <env.json> --scenario <mission-scenario.json>
--camera-config <camera-config-bay.json> --fragment docs/examples/coupled-traction-fragment.json
--runs-dir <mission-dir> --gpu-run <run-dir>` → `GPU_RUN PASS` (reached_goal=True, longest_pin_s <
escape_s). Then the full capstone. To reproduce a pre-fix run set `heading_hold_prethroat_m: 0.0`.

---

## 4. Scope

Only tightens the coupled heading hold to enter the throat aligned regardless of seat offset/drift.
Does **not** touch PR 241's lead-in, PR 201's thresholds, PR 236's bound, PR 250's rock, the couple
fine-dock, or any cart/caster friction or mass value, and **never commands reverse** (the
coupled-reverse interlock is a separate captain decision, out of scope).
