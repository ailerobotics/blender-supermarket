# XRDF arm-mount self-collision — run-82 reproduction, fix, and collect-approach proof

Firstmate msg 053. Integration run-82 (`pose_goal=False`, seeded KDL→cuMotion joint path) got the
seeded-IK retry working but cuMotion rejected **every** seeded c-space target for the collect
approach with `Invalid c-space position: self collision detected`. run-81 (`pose_goal=True`) failed
the same approach with `trajopt: INVERSE_KINEMATICS_FAILURE`. Both are one cause: cuMotion's XRDF
self-collision model flags the arm-mount region for this pruned model, so every IK candidate / seed
is rejected.

## Reproduced on the stack probe (host lane)

Pinned the exact pairs with a CPU FK + XRDF-sphere overlap check
(`scripts/restock_runtime/xrdf_selfcollision_probe.py`) at the rejected seed config
`[1.165, -0.988, 2.484, -4.633, -1.165, 3.14]` (KDL IK of the approach pose (0,0.778,0.181) quat
(0,-0.71,-0.71,0)). Non-ignored sphere pairs in contact (signed gap, m):

| pair | gap |
|---|---|
| ur_forearm_link ↔ ur_wrist_2_link | −0.034 |
| ur_base_link_inertia ↔ ur_upper_arm_link | −0.021 |
| ur_wrist_1_link ↔ ur_wrist_3_link | −0.002 |
| arm_mount ↔ ur_upper_arm_link | −0.0015 |

These are **not** real geometry collisions: the UR10 upper_arm physically cannot reach the cart base
/ arm mount past the shoulder offset, and the skip-one wrist/forearm pairs never deep-collide — the
auto-generated CONVEX SPHERE covers over-approximate into phantom contact at folded configs.
`prove_separation()` cannot clear them (it reasons on the same conservative spheres), and the SRDF
sampling did not disable them, so cuMotion flags them and rejects every approach IK candidate. (On
the stack probe the GPU task-space IK likewise failed `INVERSE_KINEMATICS_FAILURE` for *every* pose,
including a trivial identity-orientation one — because every candidate hit these same phantom pairs.)

## Fix (host lane, `scripts/twin/restock_host/generate_assets.py`)

Added a small `MANUAL_EXCLUSIONS` set merged into the asset `excluded` map (so it lands in BOTH the
XRDF `self_collision.ignore` and the SRDF `disable_collisions`), with reason
`NonAdjacentReachableClearance`:

```
arm_mount           ↔ ur_upper_arm_link
base_link           ↔ ur_upper_arm_link
ur_base_link_inertia↔ ur_upper_arm_link
ur_forearm_link     ↔ ur_wrist_2_link
ur_wrist_1_link     ↔ ur_wrist_3_link
```

Regenerate the asset set (same spheres, these pairs now ignored; the asset-report sha updates so the
host preflight passes):

```
python3 -m twin.restock_host.generate_assets \
  --urdf   <mobile_ur10.urdf> --mobile-manifest <mobile_manifest.json> \
  --shelfbot-root <shelfbot_newgripper> --output <host-cad dir>
```

Integration must regenerate `host-cad/mobile.xrdf` (+ srdf + asset-report) with this generator so the
mission's XRDF carries the exclusions.

## Collect-approach proof on the stack probe (before → after)

Same host stack, the exact run-81 approach pose, cuMotion:

| | before fix | after fix |
|---|---|---|
| self-collision rejections of the approach seed | yes (`self collision detected`) | **0** |
| exact approach pose (0,0.778,0.181) quat (0,-0.71,-0.71,0), task-space IK | `INVERSE_KINEMATICS_FAILURE` | **plans — 244-point trajectory, 2.5 s** |
| CPU overlap check at the rejected config | 4 pairs in contact | **none** |

Evidence: `/home/manishsaini/datasets/sim-restock-host-v2/{csp-repro-1,ik-diag-2,xrdffix-verify-2}/`.
This also retires the earlier "cuMotion task-space IK is systemically broken" reading
(`cumotion-ik-verdict.md`): the IK was not broken — every candidate was blocked by these phantom
arm-mount self-collisions. With them excluded, both the task-space (`pose_goal=True`) and the
seeded joint path reach the approach.

## Hygiene follow-ups (separate)

- cuMotion seed quality: KDL still returns wrapped branches (q≈±5 rad) far from the start; the
  joint-space plan to a raw wrapped seed times out. The task-space path sidesteps it (plans the
  approach directly), so `pose_goal=True` is now the better path. Seed-count/retry is integration's.
- MoveIt collision-aware IK is degraded (empty arm collision geometry from the URDF bare-path mesh
  URIs; give them a `file://` scheme) — tracked separately (piece-1 place-stage report).
