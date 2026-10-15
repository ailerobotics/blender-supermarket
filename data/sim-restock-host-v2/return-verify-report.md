# HOST side of VERIFY / RETURN (release · retreat · return-to-transport) — stack-probe

Captain's ASAP stage-prep, piece (2), extending `place_probe.py`. Proves the host side of what
ros_host and the planner need after the place: release/detach, and the return-to-transport (stow)
motion. Same probe/runner as piece (1) (`scripts/run_place_probe_gpu.py`, GPU lock taken with
wait/retry so the mission keeps priority, probe < 8 min). Evidence:
`/home/manishsaini/datasets/sim-restock-host-v2/return-probe-3/`.

## Proven

| host capability | result |
|---|---|
| **RELEASE / detach**: MoveIt `ApplyPlanningScene` REMOVE of the attached carton + cuMotion `AttachObject` action `attach_object=false` | ✅ scene success + AttachObject **SUCCEEDED** (status 4), acknowledged |
| **collision-planned joint-goal motion** (`RosMotionPort.begin_joint_goal`) against the shelf ESDF — the mechanism the `stow` phase uses to return to the transport joint pose | ✅ 65-point trajectories (baseline no-carton 2.6 s; and WITH the carton attached 2.3 s) |

`begin_joint_goal` to the 6 `ur_*` transport positions (`mobile_manifest.transport_positions`
`[0,-2.5,1.7,-0.7,-1.5708,0]`) is the identical host call; a collision-planned joint-goal plan is
proven to return a valid trajectory.

## Fix applied (this PR) — RELEASE now clears the attached object before the stow plan

The dedicated return-to-transport plan (start = the measured off-transport config, goal =
transport) was **refused by MoveIt's `CheckStartStateCollision`** with 8 contacts between
`restock_carried_unit` (the carton) and the gripper tines (`surface_plate_1..6`, `suction_cups`,
`cube_without_surface_plates`). So although the detach request was acknowledged by both services,
the attached object **persisted in move_group's planning scene** for the next plan. Both host-side
causes are now fixed in `scripts/twin/restock_manipulation/ros_payload.py` (this PR):

1. **Detach now clears the world copy.** Detaching an `AttachedCollisionObject` re-adds it to the
   planning-scene WORLD, so `restock_carried_unit` lingered as a world obstacle after release and
   tripped the stow's `CheckStartStateCollision`. The detach path now also sends a world
   `CollisionObject` REMOVE for `restock_carried_unit`, so release fully clears it from move_group's
   monitored scene.
2. **Attach `touch_links` now include the gripper tines.** `touch_links` was `['suction_cups']`
   only; the gripper **surface-plate tines** (`surface_plate_1..6`) rest against the held carton,
   so a correctly-attached carton tripped start-state self-collision. `touch_links` now carries
   `suction_cups` + `surface_plate_1..6`.

cuMotion planned WITH the carton attached from the transport start (piece 1, 65 pts), so the
planner handles the carried body; the gap is specifically the MoveIt start-state check for the
post-place stow once the touch-link / detach-propagation issues above are addressed.

## Scope

The RETREAT (Cartesian back-out of the slot) uses the same seeded-IK → cuMotion pose path as the
insertion (piece 1); its exact pose is `wall_policy`-derived at mission time. This proves the host
release + joint-goal return mechanism and surfaces the detach/touch-link gap; the end-to-end
release→retreat→stow with the real grasped geometry is the mission's to run. Contract:
`data/sim-restock-host-v2/place-contract.md`.
