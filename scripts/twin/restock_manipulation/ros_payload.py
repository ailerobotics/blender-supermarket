"""Acknowledged MoveIt + cuMotion collision attachment in host ROS process.

This never authors physics joints. The native CAD seal is already measured in
Sensors before attach; loss of seal prevents a collision attachment request.
"""
import os
import threading
import time
import numpy as np

from ..restock.pace_watchdog import is_dead as pace_is_dead
from .ros_adapter import frozen_sim_wall_backstop

# fm366: payload grasp/detach ack WALL timeout. ros_host passes config['payload_ack_wall_s'] (sim 15);
# the sim-only env WINS over it when set (default None => the passed value, byte-identical real-rig).
PAYLOAD_ACK_WALL_S = os.environ.get('RESTOCK_PAYLOAD_ACK_WALL_S')
# sim-robust-host-timers-audit-fix-v1: the attach/detach ack budget is now SIM-primary. The
# object_attachment / ApplyPlanningScene handshake runs in a separate container and is compute-bound, but
# under the live_pose masked_depth arm feed the host is busy rendering between spins, so a pure WALL
# timeout can mis-read a live-but-render-slow ack as dead. The SIM budget (an advancing /clock proves the
# ack process is hung, not slow) is the real bound; a generous frozen-sim WALL backstop still hard-stops a
# wholly-stalled sim. Default = the wall ack budget so real rig (sim == wall) fails at the same time ->
# byte-identical. Env RESTOCK_PAYLOAD_ACK_SIM_S (None => mirror the wall budget). See frozen_sim_wall_backstop.
PAYLOAD_ACK_SIM_S = os.environ.get('RESTOCK_PAYLOAD_ACK_SIM_S')

# Sim-clock lead tolerance (run-221): the gripper_pad_tcp /tf (and the Sensors feedback) are stamped
# with Isaac's world.current_time, which can lead the payload node's relayed /clock by up to ~1
# physics step (measured 16.7 ms). A strict `0 <= now-stamp` freshness lower bound then intermittently
# rejects a perfectly fresh transform (payload_sync tf_fresh gate False -> "payload collision
# attachment not acknowledged"). Tolerate the same ~1-step lead the capture (isaac_sensors) and the
# /sensors integrity check (ros_manipulation) already allow, so the gate no longer flickers.
RENDER_LEAD_S = 0.02

# ROS action GoalStatus.STATUS_SUCCEEDED. The object_attachment AttachObject action is the AUTHORITATIVE
# planner-collision attach/detach (it updates the robot description / cuMotion collision spheres the
# planner actually uses); the handshake is acknowledged iff it reports this status.
ATTACH_ACTION_SUCCEEDED = 4

# fm (sim-robust-release-detach-ack-v1): the collect RELEASE detach is a two-step host handshake -- a
# move_group ApplyPlanningScene diff (remove the MoveIt attached_collision_object + its world copy),
# THEN the AUTHORITATIVE object_attachment AttachObject detach (clears the cuMotion collision spheres
# the planner actually uses, in a SEPARATE container). move_group has an intermittent SIGABRT double-free
# in cuMotion result handling (run-266/268) and is launched respawn=True. If it crashes+respawns BETWEEN
# the grasp attach and the release detach -- coldlive did (live_pose + the arm masked-depth feed loaded
# the host container enough to trip it; coldhonest, feed off, never crashed) -- the respawned move_group
# starts with a FRESH scene that never held restock_carried_unit, so the detach's REMOVE returns
# success=False and the single-shot handshake aborted the whole collect (gate=scene_apply_unsuccessful,
# coldlive). But an unsuccessful REMOVE means there is NOTHING to remove: the scene is ALREADY in the
# post-detach state, and the object_attachment node survives the crash, so the detach must PROCEED to
# that authoritative action rather than abort. A successful REMOVE (coldhonest) never enters this branch
# => byte-identical. Default on; RESTOCK_DETACH_SCENE_TOLERANT=0 restores the legacy fail-fast.
DETACH_SCENE_TOLERANT_DEFAULT = True


def detach_scene_gate(held, scene_ok, tolerant):
    """Decide whether the attach/detach handshake may PROCEED past the move_group ApplyPlanningScene step
    to the authoritative AttachObject action, and name the gate for diagnostics. Pure for CPU testing.

    Returns (proceed, note):
      * scene_ok True                           -> (True,  None)   normal path, no note
      * DETACH (held False) + tolerant + ~ok    -> (True,  'scene_apply_unsuccessful_detach_tolerated')
      * ATTACH (held True) ~ok, or tolerance off -> (False, 'scene_apply_unsuccessful')

    An ATTACH must NEVER tolerate an unsuccessful scene apply: the object has to be ADDED to the planning
    scene before the lift is planned. A DETACH may, because the authoritative detach is the AttachObject
    action and an unsuccessful REMOVE means the scene is already in the post-detach state."""
    if scene_ok:
        return True, None
    if not held and tolerant:
        return True, 'scene_apply_unsuccessful_detach_tolerated'
    return False, 'scene_apply_unsuccessful'


def attach_acked(status):
    """The object_attachment AttachObject action is the authoritative attach/detach ack: acknowledged iff
    it reports SUCCEEDED. Pure, shared by RosPayloadSync.__call__ and the CPU test."""
    return status == ATTACH_ACTION_SUCCEEDED


def reattach_reject_gate(reconcile, detach_ok):
    """sim-robust-movegroup-reattach-after-respawn-v1: name the AttachObject-rejection diagnostic gate so
    PAYLOAD_SYNC_DIAG carries the KNOWN cause, not the bare 'attach_goal_rejected'. Pure for CPU testing.

    A goal REJECTION (send_goal_async -> accepted=False) carries no wire message; object_attachment logs the
    reason only in its own container ("Object already attached. Detach first before attaching a new object.").
    So on a post-respawn reconcile re-attach we surface that known reason by name:
      * not reconcile              -> 'attach_goal_rejected' (byte-identical to the legacy first-start gate)
      * reconcile, detach acked    -> the reconcile detach cleared the surviving attach yet the re-ADD was
                                      still refused (an unexpected planner state), named for the next run
      * reconcile, detach not acked-> could not clear object_attachment's surviving attach
                                      ('Object already attached'); the duplicate ADD is refused as a result"""
    if not reconcile:
        return 'attach_goal_rejected'
    if detach_ok:
        return 'attach_goal_rejected_after_reconcile_detach'
    return ('attach_goal_rejected object_attachment_already_holds_restock_carried_unit'
            '(reconcile_detach_not_acked)')


def scene_bootstrap_ops(published_base_obstacle, published_world_obstacles, attached_profile):
    """sim-robust-movegroup-reattach-after-respawn-v1: the ORDERED planning-scene bootstrap ops the payload
    node is responsible for re-establishing on a respawned move_group -- computed identically at first start
    and at the post-respawn resync, which is exactly what makes the two scenes equal. Base obstacle (if ever
    published), then each world obstacle in publish order, then the carton attach (if a carton is held). Pure
    for CPU testing: given the same recorded state, first-start and resync return the same list."""
    ops = []
    if published_base_obstacle:
        ops.append(('base_obstacle',))
    for obj_id, _boxes in published_world_obstacles:
        ops.append(('world_obstacle', str(obj_id)))
    if attached_profile is not None:
        ops.append(('attach', getattr(attached_profile, 'sku', attached_profile)))
    return ops


def resolve_detach_scene_tolerant(env_value):
    """RESTOCK_DETACH_SCENE_TOLERANT -> bool. Unset => the default-on module policy; any set value except
    '0' => on; '0' => off (reproduce the legacy single-shot fail-fast). Pure for CPU testing."""
    if env_value is None:
        return DETACH_SCENE_TOLERANT_DEFAULT
    return env_value != '0'


def relative_pose(base_object, base_tcp):
    """Convert measured object centre from arm base to current measured TCP."""
    p, t = np.asarray(base_object,float), np.asarray(base_tcp,float)
    if p.shape != (7,) or t.shape != (7,) or not np.isfinite([p,t]).all():
        raise ValueError('finite xyzw transforms required')
    def multiply(a,b):
        av,bv = a[:3],b[:3]
        return np.r_[a[3]*bv+b[3]*av+np.cross(av,bv),a[3]*b[3]-av@bv]
    if not all(np.isclose(np.linalg.norm(v[3:]),1) for v in (p,t)):
        raise ValueError('unit quaternions required')
    inverse = np.r_[-t[3:6],t[6]]
    xyz = multiply(multiply(inverse,np.r_[p[:3]-t[:3],0.]),t[3:])[:3]
    return tuple(np.r_[xyz,multiply(inverse,p[3:])])


class RosPayloadSync:
    def __init__(self, ack_wall_s=1.0):
        import rclpy
        from rclpy.parameter import Parameter
        from rclpy.executors import SingleThreadedExecutor
        from tf2_ros import Buffer, TransformListener
        from moveit_msgs.srv import ApplyPlanningScene
        self.ack_wall_s = (max(1.0, min(120.0, float(PAYLOAD_ACK_WALL_S)))
                           if PAYLOAD_ACK_WALL_S is not None else float(ack_wall_s))
        # sim-robust-host-timers-audit-fix-v1: the SIM-primary ack budget (default = the wall budget, so
        # real rig is byte-identical). The WALL leg becomes a generous frozen-sim backstop in _wait().
        self.ack_sim_s = (max(1.0, min(600.0, float(PAYLOAD_ACK_SIM_S)))
                          if PAYLOAD_ACK_SIM_S is not None else float(self.ack_wall_s))
        # Tolerate an unsuccessful move_group scene REMOVE on DETACH (see detach_scene_gate). Default on;
        # RESTOCK_DETACH_SCENE_TOLERANT=0 reproduces the legacy single-shot fail-fast of a pre-fix run.
        self.detach_scene_tolerant = resolve_detach_scene_tolerant(
            os.environ.get('RESTOCK_DETACH_SCENE_TOLERANT'))
        self.node = rclpy.create_node('restock_payload',namespace='/restock_sim',
                       parameter_overrides=[Parameter('use_sim_time',value=True)])
        self.tf = Buffer()
        self.listener = TransformListener(self.tf,self.node)
        self.scene = self.node.create_client(ApplyPlanningScene,'/restock_sim/apply_planning_scene')
        self.attach = None
        self.controller = self.sensors = None
        # sim-robust-movegroup-reattach-after-respawn-v1: remember the STATIC world collision objects
        # published at startup so resync_world_after_respawn() can re-establish them in a respawned
        # move_group (whose planning-scene monitor restarts empty). Empty on respawn2 (no obstacles) ->
        # the resync makes no calls -> byte-identical there.
        self._published_base_obstacle = False
        self._published_world_obstacles = []   # list of (obj_id, boxes) in add_world_obstacle() form
        self.executor = SingleThreadedExecutor()
        self.executor.add_node(self.node)
        self.thread = threading.Thread(target=self.executor.spin,daemon=True)
        self.thread.start()

    def _sim_now(self):
        """The node/sim clock (use_sim_time) in seconds. Advances with Isaac's /clock; under render
        pacing it runs far behind wall."""
        return self.node.get_clock().now().nanoseconds * 1e-9

    def _wait(self, future, timeout=None, sim_budget=None):
        # sim-robust-host-timers-audit-fix-v1: SIM-primary ack budget + frozen-sim WALL backstop on a host
        # ROS service/action future, replacing the pure-WALL event.wait() that false-tripped a live-but-
        # render-slow ack under the live_pose masked_depth arm feed. Poll the future in small wall slices
        # and fail only when the SIM budget is exhausted (an advancing /clock proves the ack process is
        # hung, not slow) OR the generous frozen-sim WALL backstop is exceeded. Real rig (sim == wall)
        # fails at the same ``timeout`` -> byte-identical. ``timeout`` is the wall value historically used;
        # ``sim_budget`` defaults to it (or to self.ack_sim_s for the default ack path).
        if timeout is None:
            timeout, sim_budget = self.ack_wall_s, (self.ack_sim_s if sim_budget is None else sim_budget)
        if sim_budget is None:
            sim_budget = float(timeout)
        wall_backstop = frozen_sim_wall_backstop(sim_budget, timeout)
        event = threading.Event()
        future.add_done_callback(lambda _: event.set())
        sim0, wall0 = self._sim_now(), time.monotonic()
        # Slice the wait so both clocks are sampled; the slice is bounded by the wall value so the real-rig
        # path still returns promptly. A frozen sim leaves sim_elapsed at ~0 -> only the wall backstop fires.
        slice_s = max(0.05, min(0.5, float(timeout)))
        while not event.wait(slice_s):
            sim_elapsed = self._sim_now() - sim0
            wall_elapsed = time.monotonic() - wall0
            if pace_is_dead(sim_elapsed=sim_elapsed, wall_elapsed=wall_elapsed,
                            sim_budget=sim_budget, wall_backstop=wall_backstop, mode='either'):
                raise TimeoutError(
                    'payload acknowledgement deadline exceeded (sim=%.1fs/%.1fs, wall=%.0fs/%.0fs)'
                    % (sim_elapsed, sim_budget, wall_elapsed, wall_backstop))
        return future.result()

    def __call__(self, held, profile, reconcile=False):
        # reconcile (sim-robust-movegroup-reattach-after-respawn-v1): a post-respawn carton RE-ATTACH.
        # object_attachment runs in a container that SURVIVES the move_group SIGABRT, so it still holds
        # restock_carried_unit and refuses a bare ADD ("Object already attached. Detach first ..."). When
        # True (held attach only), clear that surviving attach FIRST, then re-ADD. Default False on every
        # other call site (update_payload, first-start attach, release detach) => byte-identical.
        from rclpy.action import ActionClient
        from rclpy.time import Time
        from moveit_msgs.srv import ApplyPlanningScene
        from moveit_msgs.msg import AttachedCollisionObject, CollisionObject
        from shape_msgs.msg import SolidPrimitive
        from geometry_msgs.msg import Pose
        # NVIDIA messages belong in host/container ROS overlay, not Isaac Python.
        from isaac_ros_cumotion_interfaces.action import AttachObject
        s, c = self.sensors, self.controller
        now = self.node.get_clock().now().nanoseconds*1e-9
        # First in-mission grasp reached only after the attach-fallback propagation fix (run-214):
        # this path returned False at an early gate with no scene/attach call. Name the failing gate
        # so the next GPU run pinpoints it (sim-clock freshness vs service/server readiness vs contact
        # mismatch). PAYLOAD_SYNC_DIAG=0 to silence; one line per call, cheap.
        def _diag(gate):
            import os
            if os.environ.get('PAYLOAD_SYNC_DIAG', '1') != '0':
                age = None if s is None else (now - s.feedback_time - float(getattr(s, 'clock_offset', 0.0)))
                print('PAYLOAD_SYNC_DIAG gate=%s held=%s s_contact=%s age=%s max_age=%s now=%.4f '
                      'scene_ready=%s attach_ready=%s'
                      % (gate, held, (None if s is None else s.contact), age,
                         (None if c is None else c.max_age), now,
                         self.scene.service_is_ready(),
                         (self.attach.server_is_ready() if self.attach is not None else None)),
                      flush=True)
            return False
        # Staged-start clock sync (twin.restock.staged_start): feedback_time is on the packet clock, which
        # a start-phase shortcut leaves a constant clock_offset behind this node's /clock (1.55 s on
        # start_at=transfer -- past the 1.5 s max_age); remove it so the gate measures true age. 0.0 on
        # every cold run => byte-identical.
        offset = 0.0 if s is None else float(getattr(s, 'clock_offset', 0.0))
        if (s is None or c is None or s.contact != held
                or not -RENDER_LEAD_S <= now-s.feedback_time-offset <= c.max_age):
            return _diag('precondition')
        if not self.scene.service_is_ready():
            return _diag('scene_service')
        if self.attach is None:
            self.attach = ActionClient(self.node,AttachObject,'/restock_sim/attach_object')
        if not self.attach.server_is_ready():
            return _diag('attach_server')
        # sim-robust-movegroup-reattach-after-respawn-v1: on a post-respawn re-attach, the surviving
        # object_attachment attach must be cleared BEFORE the re-ADD or the server refuses it as a
        # duplicate. Best-effort + logged; the physical grasp (contact=True) is untouched -- this only
        # resets the planner's collision spheres. A detach the server declines (it DID restart, nothing
        # attached) is tolerated and we still ADD. reconcile_detached names which happened for the gate.
        reconcile_detached = None
        if reconcile and held:
            reconcile_detached = self._reconcile_detach(AttachObject)
        attached = AttachedCollisionObject()
        attached.link_name = 'gripper_pad_tcp'
        # The held carton is seated on the suction panel AND rests against the gripper surface-plate
        # tines, so all of them legitimately touch it. Without the tines in touch_links a correctly
        # attached carton trips the planner's start-state self-collision (restock_carried_unit vs
        # surface_plate_1..6), blocking the lift/place and the post-release stow.
        attached.touch_links = ['suction_cups'] + ['surface_plate_%d' % i for i in range(1, 7)]
        attached.object.id = 'restock_carried_unit'
        attached.object.header.frame_id = 'ur_base_link'
        attached.object.operation = CollisionObject.ADD if held else CollisionObject.REMOVE
        goal = AttachObject.Goal()
        goal.attach_object = held
        if held:
            try:
                transform = self.tf.lookup_transform('ur_base_link','gripper_pad_tcp',Time())
            except Exception as exc:
                import os
                if os.environ.get('PAYLOAD_SYNC_DIAG', '1') != '0':
                    print('PAYLOAD_SYNC_DIAG gate=tf_lookup_raise exc=%r now=%.4f' % (exc, now), flush=True)
                return False
            stamp = transform.header.stamp.sec+transform.header.stamp.nanosec*1e-9
            if not -RENDER_LEAD_S <= now-stamp <= c.max_age:
                return _diag('tf_fresh now_minus_stamp=%.4f stamp=%.4f' % (now-stamp, stamp))
            xyz, quat = transform.transform.translation, transform.transform.rotation
            centre = (*c.source_pose[:2],c.source_pose[2]-profile.dimensions[2]/2,*c.source_pose[3:])
            relative = relative_pose(centre,(xyz.x,xyz.y,xyz.z,quat.x,quat.y,quat.z,quat.w))
            shape = SolidPrimitive(type=SolidPrimitive.BOX,dimensions=list(profile.dimensions))
            pose = Pose()
            pose.position.x,pose.position.y,pose.position.z = centre[:3]
            pose.orientation.x,pose.orientation.y,pose.orientation.z,pose.orientation.w = centre[3:]
            attached.object.primitives = [shape]
            attached.object.primitive_poses = [pose]
            marker = goal.object_config
            marker.header.frame_id = 'gripper_pad_tcp'
            marker.type = marker.CUBE
            marker.scale.x,marker.scale.y,marker.scale.z = profile.dimensions
            marker.pose.position.x,marker.pose.position.y,marker.pose.position.z = relative[:3]
            marker.pose.orientation.x,marker.pose.orientation.y,marker.pose.orientation.z,marker.pose.orientation.w = relative[3:]
        request = ApplyPlanningScene.Request()
        request.scene.is_diff = True
        request.scene.robot_state.is_diff = True
        request.scene.robot_state.attached_collision_objects = [attached]
        if not held:
            # Detaching an AttachedCollisionObject re-adds it to the planning-scene WORLD, so the
            # released carton would linger as a world obstacle and block the post-release stow
            # (CheckStartStateCollision on restock_carried_unit). Remove it from the world too so
            # release fully clears it from move_group's monitored scene.
            world_remove = CollisionObject()
            world_remove.id = 'restock_carried_unit'
            world_remove.header.frame_id = 'ur_base_link'
            world_remove.operation = CollisionObject.REMOVE
            request.scene.world.collision_objects = [world_remove]
        scene_ok = bool(self._wait(self.scene.call_async(request)).success)
        proceed, note = detach_scene_gate(held, scene_ok, self.detach_scene_tolerant)
        if not proceed:
            return _diag(note)
        if note is not None:
            # Tolerated DETACH: the move_group scene REMOVE was a no-op (nothing to remove -- a
            # respawned/clean scene), so LOG the gate and continue to the authoritative object_attachment
            # detach below, which decides the ack. _diag only prints (returns False, ignored here).
            _diag(note)
        pending = self.attach.send_goal_async(goal)
        try:
            accepted = self._wait(pending)
        except TimeoutError:
            def cancel_late(future):
                handle = future.result()
                if handle is not None and handle.accepted:
                    handle.cancel_goal_async()
            pending.add_done_callback(cancel_late)
            raise
        if not accepted.accepted:
            return _diag(reattach_reject_gate(reconcile and held, reconcile_detached))
        try:
            result = self._wait(accepted.get_result_async())
        except TimeoutError:
            accepted.cancel_goal_async()
            raise
        if not attach_acked(result.status):
            return _diag('attach_result_status=%s' % result.status)
        return True

    def _reconcile_detach(self, AttachObject):
        """sim-robust-movegroup-reattach-after-respawn-v1: clear a carton attachment that SURVIVED a
        move_group crash in the still-live object_attachment container, so the following re-ADD is not
        refused as a duplicate ("Object already attached. Detach first ..."). Sends ONLY the authoritative
        object_attachment AttachObject detach (attach_object=False) -- NOT a release: the physical grasp is
        unaffected, this only resets the planner's collision spheres. Best-effort and tolerant: a detach the
        server declines (it DID restart -> nothing attached) or that times out is LOGGED and swallowed, and
        the caller proceeds to the ADD regardless (which then succeeds on the now-clean server). Returns True
        iff the detach was acked SUCCEEDED (the surviving attach was actually cleared)."""
        def _log(gate):
            if os.environ.get('PAYLOAD_SYNC_DIAG', '1') != '0':
                print('PAYLOAD_SYNC_DIAG gate=%s' % gate, flush=True)
        goal = AttachObject.Goal()
        goal.attach_object = False
        try:
            pending = self.attach.send_goal_async(goal)
            accepted = self._wait(pending)
            if not accepted.accepted:
                _log('reconcile_detach_rejected (nothing to clear; proceeding to re-ADD)')
                return False
            result = self._wait(accepted.get_result_async())
        except TimeoutError:
            _log('reconcile_detach_timeout (proceeding to re-ADD)')
            return False
        ok = attach_acked(result.status)
        _log('reconcile_detach_%s (surviving object_attachment attach %s before re-ADD)'
             % (('ok', 'cleared') if ok else ('result_status=%s' % result.status, 'NOT cleared')))
        return ok

    def resync_world_after_respawn(self):
        """sim-robust-movegroup-reattach-after-respawn-v1: re-publish the STATIC world collision objects this
        node established at startup (the base obstacle + each world obstacle) to a RESPAWNED move_group, whose
        planning-scene monitor restarts empty. The carton attached-body is re-established separately by the
        reconcile re-attach; this covers the remaining collision objects. Best-effort + logged; returns the
        ordered list of ops re-sent (scene_bootstrap_ops form, minus the attach). No obstacles recorded
        (respawn2) -> empty, no service calls -> byte-identical there."""
        ops = []
        try:
            if self._published_base_obstacle:
                self.add_base_obstacle()
                ops.append(('base_obstacle',))
            for obj_id, boxes in list(self._published_world_obstacles):
                self.add_world_obstacle(obj_id, boxes)
                ops.append(('world_obstacle', str(obj_id)))
        except Exception as exc:
            print('WORLD_RESYNC after respawn raised (%r); continuing to the carton re-attach' % (exc,),
                  flush=True)
        if ops:
            print('WORLD_RESYNC re-published %d static collision-object group(s) after respawn' % len(ops),
                  flush=True)
        return ops

    def add_base_obstacle(self, allow_links=None, wait_s=30.):
        """Publish the FIXED mobile base (chassis + arm-mount + hitch + wheel boxes) as a STATIC world
        CollisionObject to move_group so cuMotion plans the attached CARTON clear of it (firstmate 295).
        cuMotion drops the base links' self-collision spheres -- they are the fixed base below the arm
        root ur_base_link, not arm links -- and the base is masked out of the nvblox ESDF, so the carton
        otherwise folds/swings straight through the chassis/arm-mount (run-236/240/241).

        The arm itself is MOUNTED ON the base and legitimately folds near it (the stow config
        [0,-2.5,1.7,...] overlaps the coarse box -- run-242 INVALID_INITIAL_CSPACE_POSITION when only the
        two base links were allowed), so the object is allowed-collision vs EVERY robot link present at
        startup (``allow_links=None`` => all current ACM entries). Only the carton payload
        (restock_carried_unit, attached later at grasp and NOT in the startup ACM) is then checked
        against it -- exactly the intent: keep the carton clear while the arm moves freely near its base.
        Boxes in the ur_base_link frame (rigidly fixed to base_link); published once at startup, the base
        is parked throughout manipulation. Gated by the caller on a config flag; real-rig never calls it."""
        from moveit_msgs.srv import ApplyPlanningScene, GetPlanningScene
        from moveit_msgs.msg import CollisionObject, PlanningSceneComponents, AllowedCollisionEntry
        from shape_msgs.msg import SolidPrimitive
        from geometry_msgs.msg import Pose
        OBJ = 'restock_base_obstacle'
        # (size x,y,z), (center x,y,z) in ur_base_link frame = base_link frame shifted down 0.26 m.
        boxes = [((1.0, 0.6, 0.2), (0.0, 0.0, -0.26)),     # chassis deck
                 ((0.24, 0.24, 0.16), (0.0, 0.0, -0.08)),  # arm_mount pedestal under ur_base_link
                 ((0.12, 0.06, 0.04), (-0.56, 0.0, -0.22)),# hitch
                 ((0.32, 0.08, 0.32), (0.0, 0.35, -0.34)), # left wheel envelope
                 ((0.32, 0.08, 0.32), (0.0, -0.35, -0.34))]# right wheel envelope
        co = CollisionObject()
        co.id = OBJ
        co.header.frame_id = 'ur_base_link'
        co.operation = CollisionObject.ADD
        for size, c in boxes:
            prim = SolidPrimitive(type=SolidPrimitive.BOX, dimensions=list(size))
            pose = Pose()
            pose.position.x, pose.position.y, pose.position.z = c
            pose.orientation.w = 1.0
            co.primitives.append(prim)
            co.primitive_poses.append(pose)
        gc = self.node.create_client(GetPlanningScene, '/restock_sim/get_planning_scene')
        if not gc.wait_for_service(timeout_sec=wait_s):
            raise RuntimeError('get_planning_scene service unavailable for base obstacle')
        req = GetPlanningScene.Request()
        req.components.components = PlanningSceneComponents.ALLOWED_COLLISION_MATRIX
        acm = self._wait(gc.call_async(req)).scene.allowed_collision_matrix
        names = list(acm.entry_names)
        rows = [list(e.enabled) for e in acm.entry_values]
        # Allow the base obstacle vs every robot link currently in the scene (None => all): the arm is
        # mounted on the base and folds near it, so only the later-attached carton must avoid the box.
        allow = set(names) if allow_links is None else set(allow_links)
        for i, n in enumerate(names):            # append the object's column to every existing row
            rows[i].append(n in allow)
        names.append(OBJ)                        # object's own row: allowed vs the arm-base links + self
        rows.append([(names[j] in allow) for j in range(len(names) - 1)] + [True])
        from moveit_msgs.msg import AllowedCollisionMatrix
        acm2 = AllowedCollisionMatrix()
        acm2.entry_names = names
        acm2.entry_values = [AllowedCollisionEntry(enabled=r) for r in rows]
        acm2.default_entry_names = list(acm.default_entry_names)
        acm2.default_entry_values = list(acm.default_entry_values)
        req2 = ApplyPlanningScene.Request()
        req2.scene.is_diff = True
        req2.scene.world.collision_objects = [co]
        req2.scene.allowed_collision_matrix = acm2
        if not self._wait(self.scene.call_async(req2)).success:
            raise RuntimeError('base obstacle planning-scene apply unsuccessful')
        self._published_base_obstacle = True   # remember for resync_world_after_respawn()
        print('BASE_OBSTACLE published (%d boxes) allow=%d links (carton checked)'
              % (len(boxes), len(allow)), flush=True)
        return OBJ

    def add_world_obstacle(self, obj_id, boxes, wait_s=30.):
        """Publish a STATIC fixture (the cart cage, an overhead collider, ...) as a world
        CollisionObject so cuMotion plans the arm AND the attached carton clear of it.

        Unlike add_base_obstacle, the obstacle is NOT added to the allowed-collision matrix: the arm and
        the carton must both avoid it (curobo ignores the MoveIt ACM for world obstacles anyway, so the
        caller MUST have proven on CPU -- twin.restock_host.world_obstacles -- that the boxes clear the
        arm at every plan-START configuration, or cuMotion rejects the plan with INVALID_INITIAL_CSPACE,
        the base-box lesson, firstmate 295).  ``boxes`` is a list of (size (x,y,z), center (x,y,z),
        rpy (r,p,y)) in the ur_base_link frame; rpy is optional (default axis-aligned).  Published once
        at startup via the payload node's spinning executor while the base is parked at the station."""
        import math
        from moveit_msgs.srv import ApplyPlanningScene
        from moveit_msgs.msg import CollisionObject
        from shape_msgs.msg import SolidPrimitive
        from geometry_msgs.msg import Pose
        co = CollisionObject()
        co.id = str(obj_id)
        co.header.frame_id = 'ur_base_link'
        co.operation = CollisionObject.ADD
        for box in boxes:
            size, center = box[0], box[1]
            rpy = box[2] if len(box) > 2 else (0.0, 0.0, 0.0)
            if len(size) != 3 or len(center) != 3 or len(rpy) != 3 or any(v <= 0 for v in size):
                raise ValueError('world obstacle box needs positive size(3), center(3), optional rpy(3)')
            prim = SolidPrimitive(type=SolidPrimitive.BOX, dimensions=[float(v) for v in size])
            pose = Pose()
            pose.position.x, pose.position.y, pose.position.z = (float(v) for v in center)
            r, p, y = (float(v) for v in rpy)
            cr, sr, cp, sp, cy, sy = (math.cos(r / 2), math.sin(r / 2), math.cos(p / 2),
                                      math.sin(p / 2), math.cos(y / 2), math.sin(y / 2))
            pose.orientation.w = cr * cp * cy + sr * sp * sy
            pose.orientation.x = sr * cp * cy - cr * sp * sy
            pose.orientation.y = cr * sp * cy + sr * cp * sy
            pose.orientation.z = cr * cp * sy - sr * sp * cy
            co.primitives.append(prim)
            co.primitive_poses.append(pose)
        if not self.scene.wait_for_service(timeout_sec=wait_s):
            raise RuntimeError('apply_planning_scene service unavailable for world obstacle ' + str(obj_id))
        req = ApplyPlanningScene.Request()
        req.scene.is_diff = True
        req.scene.world.collision_objects = [co]
        if not self._wait(self.scene.call_async(req), timeout=wait_s).success:
            raise RuntimeError('world obstacle planning-scene apply unsuccessful: ' + str(obj_id))
        # Remember the spec (in this method's own arg form) for resync_world_after_respawn(); de-dup by id
        # so a resync re-publish is recorded at most once.
        self._published_world_obstacles = [e for e in self._published_world_obstacles if e[0] != str(obj_id)]
        self._published_world_obstacles.append((str(obj_id), list(boxes)))
        print('WORLD_OBSTACLE published id=%s (%d boxes)' % (obj_id, len(co.primitives)), flush=True)
        return co.id

    def close(self):
        self.executor.shutdown(timeout_sec=2.)
        self.thread.join(timeout=2.)
        self.node.destroy_node()
