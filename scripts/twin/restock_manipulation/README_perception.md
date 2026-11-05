# Collision-aware arm perception path

**Lane:** `sim-rgbd-arm-collision-aware-v1`. Makes every arm leg collision-aware against the
RGB-D nvblox ESDF, additively and **opt-in behind keys, byte-identical when off** — the
complete-cycle video run carries no new perception. See the config fragment
[`docs/examples/rgbd-arm-collision-fragment.json`](../../../docs/examples/rgbd-arm-collision-fragment.json)
and, for the cart, [`docs/examples/cart-world-obstacle-fragment.json`](../../../docs/examples/cart-world-obstacle-fragment.json).

## Why the arm was blind, and where

The `live_pose` transfer already plans with cuMotion against the perceived nvblox ESDF (wrist
depth = camera_0). The blind spots (sim-perception-architecture-s1):

- the ESDF is a single forward wrist carve, so **the rollwagen cart is outside the grid**
  (measured grid `y_min -0.08 m`; cart at base `y ~ -0.8 m`) — the planner never sees the cart at
  set-down;
- **unobserved voxels read free**, so thin cage bars and anything the one camera does not frame
  are invisible;
- from grasp on, the **carried carton fills the wrist frame** and the ESDF freezes;
- the **collect** stage sidesteps all of it with a ground-truth pick target + a `cage_collision`
  pair-filter + collision-OFF straight-line inserts, and `world_obstacles: []`.

## The two mechanisms this lane adds

### 1. The known-pose cart as a cuMotion world obstacle (`cart_world_obstacle.py`, PR-A, merged)

A camera is the wrong tool for a known rigid body. `rollwagen_world_obstacle_specs(cart_pose)`
turns the coupled cart's authored collider geometry into the existing `world_obstacles` host-key
box spec (cage posts grouped; deck/shelf/drawbar separate), so cuMotion plans the arm + attached
carton around the real deck/cage with **zero VRAM**. Opt-in: only active when the spec is placed
in `config['world_obstacles']`. This is the fix for the set-down blind spot — not perception.

### 2. The perceived-clearance gate on every leg (this PR)

Modules, all pure/CPU except the ROS service client:

| Module | Role |
|---|---|
| `esdf_grid.py` | `EsdfSnapshot`: a signed-distance grid cropped to `ur_base_link`; `clearance_over_box(es)` returns the min **observed** clearance + the unobserved fraction. **Unknown stays occupied.** Swept geometry: `pose_aabb`, `swept_pose_aabbs`, `combined_envelope` (folds a held carton into the gripper box). |
| `collision_gate.py` | `PerceptionCollisionGate.evaluate` / `evaluate_corridor`: per-leg verdict (`proceed`/`warn`/`hold`/`refuse`) from the min perceived clearance vs a margin, the unobserved fraction and snapshot age; logs `PERCEIVED_CLEARANCE`. `swept_corridor_aabbs` is the swept gripper(+carton) corridor. |
| `esdf_source.py` | The provider seam. `NvbloxEsdfSource` queries the SAME `get_esdf_and_gradient` map cuMotion plans against (mirrors `ros_adapter.NvbloxWorldBridge`), cropped to `ur_base_link`. `AnalyticBoxEsdfSource` (CPU, takes a static list or a `boxes(now)` callable), `ReplayEsdfSource` (a captured `.npz`). |
| `gated_motion.py` | `GatedMotionPort` wraps the motion port; before `begin`/`begin_joint_goal`/`begin_joint_path` it FKs the leg (`twin.urdf_probe`), builds the swept corridor, queries the source, and **raises `GateRefusal` before dispatch** when the gate blocks — the controller's `step` catch-all turns it into a named `_fail` and the mission's StageRetry re-begins against a fresher map. Everything else delegates. |

**Installed** in `ros_host.py` behind the HOST key `perception_collision_gate` (absent/disabled =>
the raw `RosMotionPort`, byte-identical). It gates EVERY leg — including the collision-OFF
straight-line KDL inserts (`contact_straight_line` / `collect_place_straight_line`) cuMotion does
not check — with zero controller edits, so it cannot disturb the pre-place planner another lane is
changing.

### Phase-shared mast feed (`arm_feed_schedule.py`, `gpu_budget.check_collect_budget`)

The wrist feeds the ESDF during arm phases; the **mast** (camera_1) is the only live feed while a
carried carton fills the wrist frame. The mast fits during collect/carry/set-down (~2.9 GB free)
but NOT at the `live_pose` transfer peak (26 MiB free), so it is phase-time-shared:
`feeds_active_in_phase(phase, mast_enabled, live_pose_transfer)` states which feeds are live, and
`gpu_budget.check_collect_budget` confirms wrist+mast fits the collect stage before boot (while the
transfer check refuses it). The Isaac render-lifecycle
(`restock.native_cameras.set_render_products_enabled`) consumes the schedule.

## Set-down + the gate

The rollwagen is handled by the **world obstacle** (mechanism 1), not the perceived ESDF — the
cart is outside the grid. So on the place/retreat legs the gate will read the cart region as
*unobserved* and `hold`. For a set-down run either exclude `place`/`retreat` from the gate's
`phases` or set `unknown_action: warn` (the cart is already collision-checked by cuMotion against
the world obstacle). The gate's value on set-down is catching a **new/unmapped** obstacle the
world obstacle does not model.

## Proof

- **CPU:** `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest scripts/twin/tests/test_perception_collision_gate.py scripts/twin/tests/test_arm_feed_schedule.py scripts/twin/tests/test_replay_collect_gate.py -q` — planted obstacle refused / clear accepted / absent-stale-unobserved held / inactive-phase & disabled byte-identical; per-phase feed switching + budget; a representative collect replay showing per-leg verdicts (`tools/replay_collect_gate.py`).
- **GPU (bounded, ≤20 min, `gpu_budget` fits=True, lock only when free, never `--force`):** S0 cart obstacle (set-down plans around the cart, no CAGE filter / straight-line insert), S1 mast-live carry (ESDF keeps integrating with the carton held), S2 gate on inserts (an insert violating the live ESDF refused by name before motion). S3 perceived collect only after S0+S1+S2.

## Hook points for the sibling lanes

- **`sim-arm-reactive-replan-v1`** (execution-time preempt + re-plan, human-aware slow/stop):
  watch `GatedMotionPort.leg_corridor` / `collision_gate.swept_corridor_aabbs` for a NEW obstacle
  entering a dispatched leg's corridor; subscribe a preempt hook on `GatedMotionPort`. This gate
  gates at **dispatch**; execution-time preempt is that lane. (People are generic dynamic volumes
  for routing; the human gate owns the STOP — never segment people into the arm ESDF.)
- **`sim-arm-collision-bench-v1`** (planted static + moving obstacles): read `GateVerdict` to score
  collision-free rate / min clearance / re-plans / cycle time; drive `AnalyticBoxEsdfSource` with a
  `boxes(now)` callable (static + moving); replay any leg list via
  `tools/replay_collect_gate.replay_gate(legs, source)`.

The carton-level constraints (world-level standoff, continuity guard, contact release,
serialization, full base lock, the 2.0° / 0.15 s box-parallel bound) are untouched by this lane.
