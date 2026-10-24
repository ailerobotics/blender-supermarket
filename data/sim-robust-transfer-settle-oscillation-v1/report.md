# sim-robust-transfer-settle-oscillation-v1 — the live_pose transfer PICK contact wrist limit cycle

**Branch:** `fm/sim-robust-transfer-settle-oscillation-v1` (off `fm/twin-robot-in-store-boot-v1` @ `25511f5`).
**Symptom run:** coldwin12 (`~/datasets/twin-robot-in-store-boot-v1/mission/coldwin12`), byte-identical to the
PASSING coldwin11, fresh commission, no injection. `state=failed reason='arm command heartbeat lost'`, peak
VRAM 10909 MiB. collect ✓ → couple ✓ → **outbound ✓** (PR253 heading-hold held) → FAIL at the transfer.

## 1. Mechanism (established on CPU from the two logs, before any code change)

### 1.1 The faulting move and its signature
The fault is at the live_pose transfer **PICK CONTACT descent** (`TRANSFER_CAPTURE cart=(-0.99,-0.02,0.59)`,
`transfer_mode=live_pose`). The carton is **NOT yet welded** (`SIM_ATTACH_DIAG … hits=[]`) — this is the
pre-grasp contact descent, a bare arm, not a loaded carry (validator `coldwin12-evidence.md`).

`host-stack.log:25864–25865`:
```
TRAJ_HOST_GOAL_MISS pos_err=0.1690 vel=1.6432 settle=30.01
  per_joint_err=[-0.0009, -0.00002, 0.00046, -0.16904(wrist_1), 0.16252(wrist_2), -0.0008]
  per_joint_vel=[-0.099, -0.092, 0.299, 1.4295(wrist_1), -0.455, -1.6432(wrist_3)]
  meas_q=[0.501, -3.183, 2.012, 1.002, -0.870, -0.0008]  goal_q=[0.502, -3.183, 2.012, 1.171, -1.033, 0]
TRAJ_HOST_FAULT pose not bounded within settle window: position peak 0.1690 > goal tol 0.010 rad
```
- shoulder_pan, shoulder_lift, elbow, wrist_3 reach goal (err ~1e-4..1e-3). **Only wrist_1 (−0.169 short:
  goal 1.171, meas 1.002) and wrist_2 (+0.163 short: goal −1.033, meas −0.870) miss**, by equal-and-opposite
  amounts; wrist_1/wrist_3 carry ~1.43 / ~1.64 rad/s.
- The `TRAJ_HOST_SETTLE` trace holds `pos_err=0.1690` / `vel≈1.64` **CONSTANT** from `settle=18.02` to `29.06`
  (n=1234→1978) — a **sustained wrist limit cycle**, neither decaying (vs PR239 `decay_ratio 0.6`) nor
  diverging (`diverge_ratio 1.25`), so it ran to the 30 s settle cap.

### 1.2 The settle classification is CORRECT (not a misread bounce)
`trajectory_host.settle_decision` returns `('miss', 'pose not bounded …')` because the recent-window
position peak (0.169) exceeds `pos_tol` (0.010). The arm genuinely sat 0.169 rad short for 30 s. PR239's
envelope did its job — **this is not a false-fail of a decaying ring, so the fix is NOT a wider tolerance or a
classifier change.** `classify_goal_miss` stays GENERIC here (two joints over tol, both non-frozen), which is
also correct — it is not the single-frozen-joint executor-pick signature.

### 1.3 Root cause: the pick landed on the wrist_2-vs-forearm self-colliding IK family
The jam joints and amounts are the **documented `ur_wrist_2`-vs-`ur_forearm` self-collision stop**
(`transfer_carry.py`, warm2 / runs 300d/305/311 / collect-two-ik-families): wrist_1 above the ~0.69 stop
(jammed at 1.002), wrist_2 below 0 (jammed at −0.870). That pair is invisible to cuMotion's sphere model and
MoveIt's SRDF, so neither the IK nor the planner avoids it; the position drive fights the hard contact in a
steady limit cycle (the high reported wrist velocity against a frozen position error is the contact
buzz / TGS bias of a body held against a constraint).

**coldwin11 vs coldwin12 — the intermittency is the IK family, confirmed from the recorded plans:**

| run | transfer pick approach→contact (moveit order e,sl,sp,w1,w2,w3) | wrist_2 | result |
|-----|---------------------------------------------------------------|---------|--------|
| coldwin11 | approach goal `[1.918,-1.740,**2.658**,-0.178,**+0.928**,0]`; contact `…+0.932…` | **positive (clear)** | settled instantly (`AWAIT_SETTLE_EXIT maxvel 0.001`, zero `TRAJ_HOST_SETTLE`) |
| coldwin12 | approach end `[2.208,-2.957,**0.937**,0.747,**−0.598**,…]`; contact goal `[2.012,-3.183,**0.502**,**1.171**,**−1.033**,0]` | **negative (colliding)** | limit-cycle 0.169 rad / 1.6 rad/s, 30 s, FAULT |

Both reach (approximately) the same live carton TCP but via **different whole-arm IK families** (shoulder_pan
2.66 vs 0.50). coldwin12's family is the wrist_2-negative self-colliding one.

### 1.4 Why the existing guards did NOT catch it
`RESTOCK_TRANSFER_CARRY=wrist_guard` (and PR243's preplace reach guard) is **scoped to the place half** —
`transfer_carry.GUARDED_PHASES = ('preplace','place','retreat')`. The comment is explicit: *"The pick half
(approach/contact …) is untouched."* So:
- The **approach** is a cuMotion **task-space pose goal** (`ros_adapter._pose_plan_goal`) with
  `wrist1_cap=None, wrist2_min=None` — free to IK the colliding family. It chose it in coldwin12.
- The **contact** is a **straight-line seeded-KDL IK** (`contact_straight_line: true` in the repo host
  config) seeded ONLY from the measured approach config and branched `_nearest_to(self.start)` — so it
  **inherits the approach's (colliding) branch** and descends deeper into the stop. No wrist guard on this
  path either.

So the pick commits to the colliding family at the *approach*, and the contact drives it into the jam.

## 2. Fix (minimal, behind one opt-in key; default off ⇒ byte-identical)

**Key:** `transfer_pick_wrist_guard` (HOST config, bool; default `False`). Active only for a **live_pose
transfer PICK** (`RESTOCK_TRANSFER_MODE=live_pose`, operation `transfer`, phase ∈ `{approach, contact}`).
Collect and every non-live_pose / non-pick move are untouched. It reuses the **already-proven clear-region
machinery** (fm327/fm403 `_pose_plan_goal` wrist joint-constraints; `_guard_rejects` wrist-flip re-seed) —
no new thresholds, no gain change, PR239's envelope untouched.

1. **Approach (root fix).** The approach's cuMotion pose goal gets the clear-region joint constraints
   `ur_wrist_1 ≤ carry_wrist1_cap` (0.65) and `ur_wrist_2 ≥ carry_wrist2_min` (0.0) — the exact
   `_pose_plan_goal(wrist1_cap, wrist2_min)` bounds the LIFT already uses. cuMotion then IKs the pick on the
   **clear wrist_2-positive family** (the coldwin11 branch, which provably reaches this TCP). The colliding
   family cannot be selected.
2. **Contact (belt-and-braces + legible retry).** The straight-line contact descent is handed the same
   `wrist_guard=(cap,w2min)`; the straight-line IK path now runs the existing `_guard_rejects` check, so a
   resolved config outside the clear region is re-seeded from its wrist-flipped twin (or fails by the named
   `IK outside the carry wrist region` reason) instead of silently executing into the stop. With the approach
   clear (step 1) the contact start is clear and this accepts on the first try (byte-identical outcome to a
   clean run). On a `MOVE_RETRY phase=contact`, `begin()` re-IKs from the **current** measured pose with the
   guard active, so the retry re-plans onto the clear branch rather than replaying the colliding goal.

PR239's terminal-settle envelope is left intact: a genuinely sustained oscillation still fails by name. This
fix removes the *source* of the oscillation (the colliding branch) rather than widening tolerance.

### Why not just cap wrist_1 / only wrist_2
The jam engages BOTH stops (wrist_1 above ~0.69 AND wrist_2 below 0), and fm403 already recorded that the
wrist_1 cap alone leaves wrist_2 jamming. The clear region is the pair; the coldwin11 clear family satisfies
both (w1 −0.18 ≤ 0.65, w2 +0.93 ≥ 0), so applying both selects it and over-constrains nothing. A straight-line
contact cannot escape a colliding *approach* branch on its own (`_nearest_to(self.start)` keeps it near the
colliding start and the collision-off descent would otherwise be a large swing), so guarding the approach is
necessary, not just the contact.

## 3. CPU tests
`scripts/twin/restock_manipulation/tests/test_restock_transfer_pick_wrist_guard.py`:
- the coldwin12 recorded contact goal `[0.502,-3.183,2.012,1.171,-1.033,0]` is OUTSIDE the clear region
  (`in_clear_region` False) and is rejected/steered by name when the guard is on;
- the coldwin11 recorded clear family `[…,-0.178,+0.928,0]` is INSIDE (`in_clear_region` True) and is
  accepted unchanged;
- `_pick_wrist_guard()` returns `(0.65, 0.0)` only for a live_pose transfer pick approach/contact, and `None`
  (byte-identical) otherwise (key off, collect, non-live_pose, other phases);
- the approach `begin()` carries the wrist joint-constraints only when the key is on (off ⇒ byte-identical);
- the straight-line contact IK path re-seeds via `_guard_rejects` on a colliding resolved config and accepts a
  clear one.

## 4. Bounded GPU proof (handed to the integration lane; not run here)
`docs/examples/transfer-pick-wrist-guard-mission-fragment.json` — merge the one HOST key into the mission
HOST block. PASS = across a bounded repeat of the staged transfer-only run (`recorded_bay_park`) every pick
contact move settles within tolerance (no `TRAJ_HOST_FAULT pose not bounded`, no `MOVE_RETRY phase=contact`
storm), with `HOST_EFFECTIVE_CONFIG` showing `transfer_pick_wrist_guard` active and a `WRIST_GUARD accept`
(or a clear cuMotion approach plan) landing wrist_2 ≥ 0; then the full cold capstone. The staged start cannot
by itself reproduce the intermittent colliding family (it is a full-chain live-carton-pose dependency), so the
capstone repeat is the real evidence. GPU rules: one run at a time under `flock /tmp/aile-gpu.lock`, never
forced, `HEAVY_MB=2500` for the always-on whispr (~2.2 GB), never touch whispr.
