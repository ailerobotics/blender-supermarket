# Cart-deck place-on-cart station contract (collect place target)

All poses in the arm base frame **`ur_base_link`** at the reposition `place_approach` park (base driven in
front of the rollwagen open `+x` mouth; `reposition_before_place: true`, `cart_cage_place: false`). Carton
SKU `juice_orange`, dims `[0.6, 0.4, 0.16]` m, fixed TCP quaternion `(0.5, 0.5, 0.5, 0.5)` (x, y, z, w).
Profile `max_reach` gate = 1.3 m. Values are the live `REPOSITION_REREGISTER` reads from the PR-200 runs
`~/datasets/twin-robot-in-store-boot-v1/mission/collect1` and `collectsrc` (within ~8 mm of each other, so a
FIXED preplace shape is robust).

| target    | position (ur_base_link)       | reach (m) | margin to 1.3 m |
|-----------|-------------------------------|-----------|-----------------|
| deck      | `(1.088, 0.122, 0.419)`       | 1.172     | 0.128           |
| place     | `(0.889, 0.161, 0.499)`       | 1.032     | 0.268           |
| preplace  | `(0.713, 0.196, 0.529)`       | 0.909     | 0.391           |

`collectsrc` logged `(1.085,0.115,0.419)/(0.886,0.154,0.499)/(0.709,0.189,0.529)`.

## The selected collision-free, trackable goal branch

Each deck pose admits **two** collision-free (clear-wrist) IK families on the UR10:

| branch  | shoulder_lift | droop proxy `|lift-(-pi/2)|` | holdable under load? |
|---------|---------------|------------------------------|----------------------|
| UPRIGHT | ~ -1.9 rad    | ~0.34 rad                    | yes (selected)       |
| DEEP    | ~ -3.57 rad   | ~2.0 rad                     | no (collect1 sag)    |

The host's seeded-KDL -> cuMotion nearest-branch IK, seeded from the folded carry config, lands in the DEEP
fold (un-holdable: collect1 shoulder_lift sagged **0.39 rad short** of goal -3.596 at steady state ->
`PATH_TOLERANCE_VIOLATED` -> `arm command heartbeat lost`) or in a wrist self-colliding branch
(`wrist_1 > 0.65, wrist_2 < 0`: collectsrc's repeated `Invalid c-space position: self collision detected`,
the 12-replan storm). The GOAL fix selects the UPRIGHT branch.

### Recorded preplace shape (the standoff, upright branch)

`collect_place_preplace_shape = [0.093, -1.910, 2.014, -0.103, 1.664, 0.0]`
(ARM order: `ur_shoulder_pan, ur_shoulder_lift, ur_elbow, ur_wrist_1, ur_wrist_2, ur_wrist_3`, radians).

- FK TCP = `(0.7129, 0.1962, 0.5281)` m — **~0.6 mm** from the recorded preplace standoff `(0.713,0.196,0.529)`.
- Reach 0.909 m (margin 0.391 m to the 1.3 m gate).
- `wrist_1 = -0.103 ≤ 0.65`, `wrist_2 = 1.664 ≥ 0` → **clear** of the `ur_wrist_2`-vs-`ur_forearm` stop.
- `shoulder_lift = -1.910` (droop proxy 0.34) → holdable under the 0.45 kg carton.
- Driven as a fixed collision-planned `begin_joint_goal` with a relaxed **0.2 rad** terminal goal gate
  (the run-279 / fm403 pattern), so a small settle offset is accepted; the place insert then tracks the
  SENSED deck target.

### Place insert (sensed, from the upright preplace, wrist-guarded)

From the upright preplace shape as the IK seed (`near_start`), the SENSED place target
`(0.889, 0.161, 0.499)` solves to the upright clear branch:
`q = [-0.004, -1.628, 1.844, -0.216, 1.567, 0.0]` (perr 0.02 mm, clear, droop 0.06), a **small** joint move
from the preplace (max |Δq| ≈ 0.28 rad). `collect_place_wrist_guard` keeps this (and the empty-handed
retreat) in the clear wrist branch on the host KDL path.

## No off-centre / re-yawed set-down needed

The carton is placed at the sensed deck target with the profile's fixed TCP yaw — **no** off-centre or
re-yawed placement is required; both the preplace standoff and the place insert reach the deck centre on the
upright clear branch within the reach gate. The only change versus the failing runs is the IK **branch**
(upright + clear), not the deck position or yaw.

## Verification

CPU: `scripts/twin/tests/test_restock_cart_place_goal.py` (pure numpy ArmIK on the committed mission arm
chain; the `in_clear_region` / `wrist_flip_seed` clear-region predicate shared with the transfer path).
Asserts the collision-free in-reach solution exists, that a self-colliding branch exists (so the guard does
real work), that the upright branch is clearly lower-droop than the deep fold, and that the shipped shape is
clear / in-limits / FK-at-standoff. Real-geometry XRDF sphere self-collision is covered dataset-gated by the
existing `test_restock_scripted_transfer.py` sphere tests.

GPU proof (collect + couple, GT collect, the collect1 config + the two keys) is handed to the integration
lane via `docs/examples/cart-place-goal-mission-fragment.json`; see `report.md`.
