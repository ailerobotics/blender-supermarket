# sim-robust-movegroup-reattach-after-respawn-v1

Make the carried carton **re-attach succeed on a respawned `move_group`** during collect, closing the
one recovery step that has never worked. PR #247 made the launch respawn bounded and observable; PR #234
and PR #242 fail the stage fast by name when recovery does not complete. What was missing is the recovery
itself: after a `move_group` SIGABRT + respawn, the client's re-attach of the held carton was rejected for
the whole 30 s budget and the op fast-failed (correctly, never carton-less) at lift.

- **Run that motivated it:** `~/datasets/twin-robot-in-store-boot-v1/mission/respawn2` (head `ccc479e`,
  PR #247; collect-only, fresh commission `cc-respawn2`, HOST `movegroup_respawn={max_retries:2, delay:3.0}`,
  the v3 double-death injector killing `move_group` with SIGABRT after the carton attach). `mission.json`
  `state=failed reason='arm command heartbeat lost'`, `stages_driven=['collect']`, peak VRAM 10548 MiB.
- **Evidence:** `~/firstmate/data/twin-robot-in-store-boot-v1/respawn2-evidence.md` and that run's
  `host-stack.log`. `coldwin4` (`coldwin4-evidence.md`) had the same gap-1 root; `dblproof-v2`
  (`dblproof-v2-evidence.md`) validated the named fast-fail, **not** a completed recovery.
- **Base branch / PR target:** `fm/twin-robot-in-store-boot-v1`. The `move_group` SIGABRT itself
  (exit -6, heap corruption) is irreducible upstream MoveIt/cuMotion instability — **out of scope**, as are
  the PR #247 respawn policy and the PR #234/#242 fast-fail logic (beyond surfacing the reason).

## Step 1 — the finding (established on CPU from the respawn2 log, before any code change)

### What the respawned `move_group` actually answered the attach goal

`host-stack.log` names it exactly. The FIRST (successful) attach, right after the grasp:

```
3976 [component_container_mt-2] [restock_sim.object_attachment]: Received attach request
3977 … Executing attachment action
3983 … Generated 34 collision spheres for cuboid object
4009 … Object attached successfully with 34 collision spheres to link 'attached_object'
4010 … Object successfully attached
```

Then `move_group` (pid 1511193) is killed with SIGABRT and the launch respawns it by name (PR #247):

```
[move_group-1]: process has died [pid 1511193, exit code -6, …]
MOVE_GROUP_RESPAWN move_group died pid=1511193 exit=-6; relaunching move_group (attempt 1/2, respawn_delay=3.0s)
```

The client then re-attaches. **Every re-attach retry for the full 30 s sim budget** got the SAME answer
(host-stack.log from 7271, repeated ~dozens of times to 60.2 s sim and beyond):

```
7271 [component_container_mt-2] [restock_sim.object_attachment]: Received attach request
7272 [component_container_mt-2] [WARN] [restock_sim.object_attachment]: Object already attached. Detach first before attaching a new object.
     [python3-4] PAYLOAD_SYNC_DIAG gate=attach_goal_rejected held=True s_contact=True age=0.050 max_age=1.5 now=90.1667 scene_ready=True attach_ready=True
```

i.e. the `AttachObject` goal was **rejected** (`send_goal_async` → `accepted=False`, `ros_payload.py`
`__call__` returns `_diag('attach_goal_rejected')`). The rejection carries no wire message; the *reason* is
only in the server's own WARN log: **"Object already attached. Detach first before attaching a new object."**
The recovery ended with PR #242's named fast-fail at lift — correct, but never recovering:

```
MOVE_GROUP_CRASH_RETRY re-attach of restock_carried_unit NEVER SUCCEEDED within 30s sim …
HOST_FAULT_ESTOP phase=lift reason='lift aborted: the carried carton could not be re-attached …'
```

### Root cause — a stale attachment surviving in a container that did not die

The earlier `coldwin4` hypothesis (the respawned planner "had not finished re-initialising", so the reject
was transient) is **disproved** by respawn2: `held=True s_contact=True scene_ready=True attach_ready=True`,
`age=0.05 s`, and the reject was *persistent* for 30 s, not a startup transient.

The true cause is architectural. `object_attachment` runs in a **separate container**
(`component_container_mt-2`) that does **not** die when the `move_group-1` process is SIGABRT-killed. So
after the respawn it **still holds `restock_carried_unit` attached** (the authoritative cuMotion attach,
with the 34 collision spheres, survives). The client's post-respawn re-attach (`_resync_scene_after_respawn`
→ `payload_sync(True, profile)`) assumes the crash cleared the scene and sends a **bare ADD**, which
`object_attachment` refuses as a duplicate ("Object already attached. Detach first …"). So `payload_sync`
returns not-ok forever and PR #242 fast-fails.

### Diff: post-respawn re-send vs. first-start send

`payload_sync.__call__(held, profile)` (`ros_payload.py`) does the SAME two things at first start and on the
post-respawn re-attach, with identical fields:

| field | first start | post-respawn re-attach |
|---|---|---|
| `AttachedCollisionObject.link_name` | `gripper_pad_tcp` | `gripper_pad_tcp` |
| `.touch_links` | `suction_cups` + `surface_plate_1..6` | same |
| `.object.id` | `restock_carried_unit` | same |
| `.object.header.frame_id` | `ur_base_link` | same |
| `.object.operation` | `ADD` | `ADD` |
| `AttachObject.Goal.attach_object` | `True` | `True` |
| move_group `ApplyPlanningScene` (attached body) | applied (success) | **applied (success)** |
| `object_attachment AttachObject` ADD | **accepted** (nothing attached yet) | **rejected** (already attached) |

So the ONLY difference is the object_attachment server's surviving state: the field content and the
move_group `ApplyPlanningScene` are identical and both succeed; the duplicate `AttachObject` ADD is the only
thing that fails. The respawn2 config publishes no base/world obstacles (`base_collision_object: false`,
`world_obstacles: []`, per `HOST_EFFECTIVE_CONFIG`), so for respawn2 the only scene state the respawned
move_group lost that matters is the attached body — which `ApplyPlanningScene` already re-establishes.

### Fix direction (implemented in Step 2)

On the post-respawn re-attach, **reconcile the surviving attachment before re-adding**: issue a
best-effort `AttachObject` DETACH first (the physical grasp — `contact=True` — is untouched; this only
resets the planner's spheres), then the normal ADD, which the now-cleared server accepts. This handles both
the proven "server still holds it" case and the "server lost it" case (a tolerated no-op detach, then a
clean ADD). The move_group `ApplyPlanningScene` attached body (and any startup base/world obstacles the port
published) are re-established in the same pass. The 30 s budget and the PR #242 gap-1 fast-fail stay as the
backstop. When no respawn happens, nothing calls the reconcile path → byte-identical.

## Step 2 — the fix

All of it is **always-on + self-disabling**: it engages only on a `move_group` crash-retry resend, so a
live-server attach and the whole no-crash path are byte-identical. **No new mission runtime-config key.**

### Reconcile the surviving attach (`ros_payload.py`)
- `RosPayloadSync.__call__` gains `reconcile=False`. When `reconcile and held` (the post-respawn re-attach),
  it first issues `_reconcile_detach()` — the authoritative `object_attachment` `AttachObject` detach
  (`attach_object=False`) — to clear the attachment that **survived the crash**, then falls through to the
  normal ADD, which the now-cleared server accepts. `_reconcile_detach` is **best-effort and tolerant**: the
  physical grasp (`contact=True`) is untouched (it only resets the planner's collision spheres), and a
  detach the server declines (it *did* restart → nothing attached) or that times out is logged and
  swallowed — the caller still ADDs (which then succeeds on the clean server). This handles both the proven
  "server still holds it" case and the "server lost it" case.
- `reattach_reject_gate(reconcile, detach_ok)` (pure) **surfaces the reason by name** in
  `PAYLOAD_SYNC_DIAG`: non-reconcile stays the byte-identical `attach_goal_rejected`; a reconcile reject
  names whether the detach-first cleared the attach yet the ADD was still refused, or the known
  `object_attachment_already_holds_restock_carried_unit` cause (the server logs its "Object already
  attached" only in its own container — a goal rejection carries no wire message).

### Re-establish the rest of the lost scene (`ros_payload.py`)
- `add_base_obstacle` / `add_world_obstacle` now **record** what they publish; `resync_world_after_respawn()`
  re-publishes those static collision objects to the respawned `move_group` (whose planning-scene monitor
  restarts empty). Best-effort + logged; empty on respawn2 (`base_collision_object:false`,
  `world_obstacles:[]`) → no calls → byte-identical. `scene_bootstrap_ops(...)` (pure) is the ordered plan
  the port re-establishes, identical at first start and at resync — the CPU "same scene" proof.

### Client recovery path (`ros_adapter.py`)
- `_resync_scene_after_respawn` now (1) calls `resync_world_after_respawn()` via a guarded `getattr` (a test
  double lambda has no such method → skipped), then (2) re-attaches via `_reattach_payload(profile)`, which
  passes `reconcile=True` **only when the callable's signature accepts it** (`_payload_accepts_reconcile`,
  via `inspect.signature().bind_partial` — a genuine `TypeError` from inside `payload_sync` is never
  masked; a bare `lambda(held, profile)` fixture is detected and called two-arg). The 30 s
  `REATTACH_SIM_BUDGET_S` retry loop and the PR #242 gap-1 fast-fail stay as the backstop.
- The stale `# the respawned planner had not finished re-initialising` comment (the disproven coldwin4
  hypothesis) is corrected in place to name the surviving-container cause.

### CPU tests
- `scripts/twin/tests/test_restock_detach_ack.py`: `reattach_reject_gate` names the surviving-attach cause;
  `scene_bootstrap_ops` first-start == post-respawn; `_reconcile_detach` clears a surviving attach and
  tolerates both "nothing attached" and a timeout; `resync_world_after_respawn` re-publishes in order,
  no-ops when nothing was published, and tolerates a publish raise.
- `scripts/twin/tests/test_ros_adapter_branch.py` (`MoveGroupReattachReconcileAfterRespawn`): a holding
  server blocks the legacy bare ADD; the reconcile re-attach detaches-first then resends and succeeds;
  `reconcile=True` is actually passed; the world-obstacle resync runs (and a raise there does not abort the
  re-attach); `update_payload` never passes reconcile (byte-identical); a two-arg lambda double still works.
- All existing `test_ros_adapter_branch` / `test_movegroup_respawn` / `test_restock_*` suites stay green
  (the signature-detect keeps every pre-existing lambda mock on the unchanged two-arg path).

## Step 3 — bounded GPU proof config (handed to the integration lane)

`docs/examples/movegroup-reattach-after-respawn-mission-fragment.json`. There is **no key to merge** (the
fix is always-on); the hand-off names the two runs on the unchanged respawn2 setup + validator injector:

- **Run A (double death):** respawn2 config unchanged, `movegroup_respawn={max_retries:2}`, v3 injector kills
  twice. **PASS** = each respawn recovers (`MOVE_GROUP_RESPAWN … attempt 1/2` then `2/2`), each re-attach
  logs `PAYLOAD_SYNC_DIAG gate=reconcile_detach_ok …` then re-attaches, and collect completes with the carton
  on the rollwagen (no repeated `attach_goal_rejected`, no gap-1 fast-fail, `will plan WITHOUT it` grep=0).
- **Run B (triple death):** same, injector kills three times. **PASS** = the first two recover as Run A and
  the third exceeds `max_retries` so the respawn supervisor gives up **by name** (PR #247) and the stage
  fails fast (PR #234/#242 backstop) — confirming the fix did not weaken either.

The integration lane holds the GPU; this worker does not run it.
