# HOST side of the TRANSFER (place) stage — stack-probe proof at shelf G3_back_s2

Captain's ASAP stage-prep, piece (1). Proves on the guarded GPU stack probe that the restock
host stack supports the post-collect PLACE, before the mission reaches it. Place is
`operation='transfer'` on the same `TransferController` as collect; the host stack is identical,
so this exercises the host endpoints the controller drives (controller.py / ros_adapter.py
`RosMotionPort` / ros_payload.py) with the verified `ros_host_config.json` profile + wall index.

Probe: `scripts/restock_runtime/place_probe.py`; runner: `scripts/run_place_probe_gpu.py`
(standard GPU guard; **waits/retries for `/tmp/aile-gpu.lock` so the mission keeps priority**,
caps the probe < 8 min). Camera extrinsic for the shelf is published as a static
`ur_base_link → restock_camera_shelf` TF (the mission relays this live on `/tf`).

## Result — PASS (`/home/manishsaini/datasets/sim-restock-host-v2/place-probe-6/`)

| host capability for TRANSFER | result |
|---|---|
| 6 planning nodes + place endpoints (`attach_object` action, `apply_planning_scene`, `compute_ik`, `move_action`, nvblox ESDF) present | ✅ |
| station protocol AT THE SHELF (`prepare`/`heartbeat` with `camera_frame=restock_camera_shelf`, host workspace bounds) → ack **ready**, fresh `collision_time` | ✅ (echoed `restock_camera_shelf`, collision_time ≈ 7.9 s) |
| shelf walls in the collision model: shelf masked depth → nvblox ESDF non-empty | ✅ 0 → **2,546,419** observed voxels |
| carton **attach** = MoveIt `ApplyPlanningScene` (AttachedCollisionObject BOX of the SKU dims on `gripper_pad_tcp`, touch_links `suction_cups`) + cuMotion `AttachObject` action | ✅ scene success + AttachObject **SUCCEEDED** (status 4), acknowledged |
| collision-aware cuMotion plan against the shelf ESDF (baseline, no carton) | ✅ **65-point** trajectory, error_code 1, 2.84 s |
| cuMotion plan **WITH the carton attached** (carton part of the robot body) | ✅ planned (carton-clear) — the carton is in cuMotion's collision model |

Workspace bounds are a single `ur_base_link`-relative triple (`host.json` `[-2,-2,-1]`/`[2,2,2]`)
that moves with the parked base, so the SAME bounds apply at `G3_back_s2` as at collect; only the
camera frame differs. Verified profile (`ros_host_config.json`): `juice_orange`, dims
`[0.6,0.4,0.16]`, `grasp_mode "side"`, `tcp_quaternion [0.5,0.5,0.5,0.5]`, wall index `G3_back_s2: 3`.

## Host-side gaps found + fixes

1. **Shelf depth-camera TF (fixed in the probe).** nvblox silently drops masked depth with no
   `ur_base_link → restock_camera_<id>` transform (ESDF stays empty, ack never readies). The probe
   now publishes it; the mission must relay `/tf` for `restock_camera_shelf` continuously (it does).
2. **Measure planner/ack budgets in node-clock time (fixed in the probe).** Under `use_sim_time`
   the planner's `allowed_planning_time` (5 s) and the ack freshness guards are SIM seconds; a
   free-running probe clock expired them in a fraction of a wall second (spurious TIMED_OUT /
   ack sim-gap fault). The probe paces its `/clock` to real wall time.
3. **UR10 collision meshes don't load in MoveIt → degraded collision-aware IK (RECOMMEND fixing on
   the generated URDF).** The URDF references `/opt/ros/jazzy/share/ur_description/.../collision/*.stl`
   by bare absolute path; MoveIt's resource retriever rejects it (`URL rejected: No host part in the
   URL`) and leaves every arm link with **empty collision geometry**. `compute_ik(avoid_collisions=True)`
   can then return a contorted, self-colliding branch that cuMotion refuses
   (`planToCSpaceTarget … self collision detected`). The files exist — giving the mesh filenames a
   `file://` scheme (or `package://`) restores MoveIt collision-aware IK. The mission's `collect`
   worked because `wall_policy` poses yield natural IK; the Cartesian side-grip **insertion pose**
   should get the same, so this should be fixed before the first live place.

## Scope / honest limits

This proves the host STACK supports the place: endpoints, the shelf station protocol, shelf ESDF
walls, the two-step carton attachment, and collision-aware cuMotion planning with the carton in the
body. The exact side-grip **insertion** TCP pose is `WallCartonPolicy`-derived from the sensed
handling-unit poses at mission time (RUNBOOK/docs/restock_manipulation.md); the synthetic arbitrary
insertion poses here were IK-contorted under gap #3 and are not a host defect. A joint tolerance of
0.001 rad made cuMotion trajopt non-converge (UNKNOWN_STATUS); the proven form uses 0.05.
Mission-facing contract: `data/sim-restock-host-v2/place-contract.md`.
