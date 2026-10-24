# sim-robust-release-detach-ack-v1 — collect release-detach acks across live_pose + arm-feed

**Branch:** `fm/sim-robust-release-detach-ack-v1` → PR into `fm/twin-robot-in-store-boot-v1`.
**Deliverable:** the collect release-detach handshake acknowledges reliably whether or not the
`live_pose` transfer mode and the arm masked-depth feed are configured. CPU-only fix + tests; one bounded
GPU proof config handed to the integration lane. No GPU run taken here.

## Symptom

The `coldlive` capstone (cold config + `RESTOCK_TRANSFER_MODE=live_pose` + `RESTOCK_TRANSFER_CARRY=wrist_guard,weld`
+ the arm masked-depth feed on, mast off, visuals on; `~/datasets/twin-robot-in-store-boot-v1/mission/coldlive`)
**failed at the COLLECT release-detach**, before ever reaching the live_pose transfer:

```
action=collect status=succeeded transport_ready=True
SIM_ATTACH_FALLBACK carry collision ENABLED on 2 collider(s)   (re-enable carton collision = detach)
SIM_ATTACH_FALLBACK release
MANIP_HOST_FAILED op=collect phase='failed' reason='payload collision detachment not acknowledged'
arm command heartbeat lost
```

`coldhonest` (same collect config, arm feed off, scripted/still transfer) passed this exact step and
completed all seven stages.

## Root cause — a move_group SIGABRT (double-free) crash+respawn mid-collect

The collect RELEASE detach is a **two-step host handshake** (`scripts/twin/restock_manipulation/ros_payload.py`
`RosPayloadSync.__call__`):

1. a **move_group** `ApplyPlanningScene` diff that REMOVEs the MoveIt `attached_collision_object`
   `restock_carried_unit` (and its world copy), then
2. the **authoritative** object_attachment `AttachObject` detach — clears the cuMotion collision spheres the
   planner actually uses, served by the `object_attachment` node in a **separate** container
   (`component_container_mt`).

move_group is launched `respawn=True` (`scripts/twin/restock_host/launch.py`) because it has an intermittent
**SIGABRT double-free in cuMotion result handling** (documented run-266/268; the reason `respawn=True` exists).

### Exact evidence (`coldlive/host-stack.log`, wall-epoch timestamps)

- Grasp attach landed on the ORIGINAL move_group instance:
  `1792719247.579  [restock_sim.object_attachment]: Object attached successfully with 34 collision spheres to link 'attached_object'`
- During the collect PLACE (a joint-space goal), move_group crashed:
  ```
  1792719550.329 [restock_sim.cumotion_planner]: Using joint space goal
  [ERROR] [move_group-1]: process has died [pid 3542764, exit code -6, cmd '.../moveit_ros_move_group/move_group ...'].
  ```
  (`exit code -6` = SIGABRT.)
- move_group respawned ~3.8 s later (launch `respawn_delay=3.0`):
  `1792719554.106 [restock_sim.move_group.moveit.ros.planning_pipeline]: Successfully loaded planner '... NVIDIA Isaac ROS cuMotion'`
- The host re-sent the in-flight plan goal to the respawned server and the place completed:
  `[python3-4] MOVE_GROUP_CRASH_RETRY plan-result wall-deadline (likely move_group crash+respawn between polls); resending plan goal (retry 1/3)`
- At the DETACH, the REMOVE failed on the respawned (clean) scene:
  `[python3-4] PAYLOAD_SYNC_DIAG gate=scene_apply_unsuccessful held=False s_contact=False age=0.05 max_age=1.5 now=128.9333 scene_ready=True attach_ready=True`
  → `MANIP_HOST_FAILED op=collect phase='failed' reason='payload collision detachment not acknowledged'`.

`coldhonest/host-stack.log` has **zero** `process has died` lines — its move_group never crashed, so its
scene REMOVE succeeded first try and reached `object_attachment: Object detached successfully`.

### Why the REMOVE returns success=False after a respawn

The respawned move_group starts with a **fresh planning scene** that never held `restock_carried_unit`
(the attachment lived only in the crashed instance's scene). `ApplyPlanningScene` REMOVE of an object that
is not in the scene returns `success=False`. The old single-shot handshake `return`ed at
`scene_apply_unsuccessful` **before ever reaching step 2** (the authoritative detach), so the mission
aborted — even though the authoritative object_attachment node was alive (separate container, survived the
crash) and would have detached cleanly.

This is **deterministic once the respawn has happened** (the fresh scene never regains the object), so a
controller-level retry of the same REMOVE would keep failing — the fix must not be a retry. The crash
itself is **stochastic** (not every run; coldhonest did not crash); the arm masked-depth feed (2-camera
nvblox + a 2nd `object_segmenter` loading the host container) raised its likelihood, which is why the
combination surfaced the latent gap and the feed-off run did not.

## Fix

`ros_payload.detach_scene_gate(held, scene_ok, tolerant)` (default on;
`RESTOCK_DETACH_SCENE_TOLERANT=0` reproduces the legacy fail-fast):

- **DETACH** (`held=False`) + unsuccessful REMOVE + tolerant ⇒ **proceed** to the authoritative
  AttachObject detach, which decides the ack (logs `gate=scene_apply_unsuccessful_detach_tolerated`). An
  unsuccessful REMOVE means there is nothing to remove — the scene is already in the post-detach state.
- **ATTACH** (`held=True`) + unsuccessful apply ⇒ **abort** at the scene apply (`scene_apply_unsuccessful`):
  the object must be ADDED to the scene before the lift is planned.
- A **successful** REMOVE (coldhonest) never enters the branch ⇒ **byte-identical**.
- A **genuinely** failing authoritative detach (object_attachment status ≠ SUCCEEDED) still fails fast by
  name (`attach_result_status=…` → the controller's `payload collision detachment not acknowledged`).

The final ack is `attach_acked(status) == (status == 4 / STATUS_SUCCEEDED)` — unchanged, now a named pure
helper shared with the test.

Backward compat: new env-only switch `RESTOCK_DETACH_SCENE_TOLERANT` (default on), parsed by the pure
`resolve_detach_scene_tolerant`. No change to the host config schema, the mission runtime configs, or
coldhonest behaviour.

## CPU proof (no GPU, no ROS graph)

- `scripts/twin/tests/test_restock_detach_ack.py` — pure `detach_scene_gate` / `attach_acked` /
  `resolve_detach_scene_tolerant`, plus a faithful re-enactment of `__call__`'s scene→attach→ack sequence:
  **acknowledged under both configurations** (REMOVE ok; REMOVE unsuccessful but authoritative detach
  SUCCEEDED) and a **genuinely missing detach still fails by name** (`attach_result_status=2`), and
  tolerance-off reproduces the pre-fix abort.
- `scripts/twin/tests/test_restock_manipulation.py::test_release_detach_acked_transitions_to_retreat` /
  `::test_release_detach_not_acknowledged_fails_by_name` — the mission-visible controller contract
  (acked detach → `retreat`; unacked → fail by the named reason).
- Full re-run after rebasing onto PR 222: `251 passed, 22 skipped` across the detach-ack, manipulation,
  ros_split, host, transfer_carry, live_pose, collect_ground_truth and cart_cage_place suites.

## Bounded GPU proof for the integration lane

`docs/examples/collect-release-detach-ack-fragment.json` — collect-only (`--stop-after collect`) with
`masked_depth.enabled: true`, `mast_depth.enabled: false`, env `RESTOCK_TRANSFER_MODE=live_pose` /
`RESTOCK_TRANSFER_CARRY=wrist_guard,weld`. Expected: collect ends cleanly, no
`detachment not acknowledged` / `arm command heartbeat lost`; if move_group crashes this run, a
`gate=scene_apply_unsuccessful_detach_tolerated` line precedes `Object detached successfully`. No code
change needed to activate the fix (default on). Integration lane runs it under
flock `/tmp/aile-gpu.lock`, never forced, never touching whispr.

## Handoff — route the planner crash itself

The move_group SIGABRT (exit -6, double-free in cuMotion result handling) is the ROOT trigger here and a
recurring one (run-266/268; the reason `respawn=True` + `MOVE_GROUP_CRASH_RETRY` exist). This task makes the
detach **survive** a crash+respawn; it does not stop the crash. Firstmate should route the planner-crash fix
as its own lane. A related gap this one did NOT touch: the `MOVE_GROUP_CRASH_RETRY` re-sends the plan goal
but does **not** re-apply the lost MoveIt attached_collision_object to the respawned move_group — harmless
for the detach (scene already post-detach) but worth auditing for any pre-detach move planned on a
respawned instance.
