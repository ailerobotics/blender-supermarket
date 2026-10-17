"""Acknowledged MoveIt + cuMotion collision attachment in host ROS process.

This never authors physics joints. The native CAD seal is already measured in
Sensors before attach; loss of seal prevents a collision attachment request.
"""
import threading
import numpy as np

# Sim-clock lead tolerance (run-221): the gripper_pad_tcp /tf (and the Sensors feedback) are stamped
# with Isaac's world.current_time, which can lead the payload node's relayed /clock by up to ~1
# physics step (measured 16.7 ms). A strict `0 <= now-stamp` freshness lower bound then intermittently
# rejects a perfectly fresh transform (payload_sync tf_fresh gate False -> "payload collision
# attachment not acknowledged"). Tolerate the same ~1-step lead the capture (isaac_sensors) and the
# /sensors integrity check (ros_manipulation) already allow, so the gate no longer flickers.
RENDER_LEAD_S = 0.02


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
        self.ack_wall_s = float(ack_wall_s)
        self.node = rclpy.create_node('restock_payload',namespace='/restock_sim',
                       parameter_overrides=[Parameter('use_sim_time',value=True)])
        self.tf = Buffer()
        self.listener = TransformListener(self.tf,self.node)
        self.scene = self.node.create_client(ApplyPlanningScene,'/restock_sim/apply_planning_scene')
        self.attach = None
        self.controller = self.sensors = None
        self.executor = SingleThreadedExecutor()
        self.executor.add_node(self.node)
        self.thread = threading.Thread(target=self.executor.spin,daemon=True)
        self.thread.start()

    def _wait(self, future, timeout=None):
        # WALL timeout on a host ROS service/action future. Real-rig default 1 s (fast move_group);
        # under the sim's render-paced stepping the host is busy rendering between spins, so the sim
        # profile widens it (ack_wall_s, >= 3x worst case) to avoid a false grasp-time timeout.
        if timeout is None:
            timeout = self.ack_wall_s
        event = threading.Event()
        future.add_done_callback(lambda _:event.set())
        if not event.wait(timeout):
            raise TimeoutError('payload acknowledgement deadline exceeded')
        return future.result()

    def __call__(self, held, profile):
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
                age = None if s is None else (now - s.feedback_time)
                print('PAYLOAD_SYNC_DIAG gate=%s held=%s s_contact=%s age=%s max_age=%s now=%.4f '
                      'scene_ready=%s attach_ready=%s'
                      % (gate, held, (None if s is None else s.contact), age,
                         (None if c is None else c.max_age), now,
                         self.scene.service_is_ready(),
                         (self.attach.server_is_ready() if self.attach is not None else None)),
                      flush=True)
            return False
        if (s is None or c is None or s.contact != held
                or not -RENDER_LEAD_S <= now-s.feedback_time <= c.max_age):
            return _diag('precondition')
        if not self.scene.service_is_ready():
            return _diag('scene_service')
        if self.attach is None:
            self.attach = ActionClient(self.node,AttachObject,'/restock_sim/attach_object')
        if not self.attach.server_is_ready():
            return _diag('attach_server')
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
        if not self._wait(self.scene.call_async(request)).success:
            return _diag('scene_apply_unsuccessful')
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
            return _diag('attach_goal_rejected')
        try:
            result = self._wait(accepted.get_result_async())
        except TimeoutError:
            accepted.cancel_goal_async()
            raise
        if result.status != 4:
            return _diag('attach_result_status=%s' % result.status)
        return True

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
        print('WORLD_OBSTACLE published id=%s (%d boxes)' % (obj_id, len(co.primitives)), flush=True)
        return co.id

    def close(self):
        self.executor.shutdown(timeout_sec=2.)
        self.thread.join(timeout=2.)
        self.node.destroy_node()
