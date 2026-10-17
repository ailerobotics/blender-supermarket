"""Acknowledged MoveIt + cuMotion collision attachment in host ROS process.

This never authors physics joints. The native CAD seal is already measured in
Sensors before attach; loss of seal prevents a collision attachment request.
"""
import threading
import numpy as np


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
    def __init__(self):
        import rclpy
        from rclpy.parameter import Parameter
        from rclpy.executors import SingleThreadedExecutor
        from tf2_ros import Buffer, TransformListener
        from moveit_msgs.srv import ApplyPlanningScene
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

    @staticmethod
    def _wait(future, timeout=1.):
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
        if s is None or c is None or s.contact != held or not 0 <= now-s.feedback_time <= c.max_age:
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
            if not 0 <= now-stamp <= c.max_age:
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

    def close(self):
        self.executor.shutdown(timeout_sec=2.)
        self.thread.join(timeout=2.)
        self.node.destroy_node()
