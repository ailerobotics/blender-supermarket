"""Concrete CAD cuRobo -> Isaac articulation drives, using measured completion.

Must be constructed inside integration's already booted Isaac process. Does not
start SimulationApp, a ROS rig, or any external service. Joint targets are drive
commands, never set_joint_positions teleports. The external shelfbot planner is
imported read-only and configured for the actual imported arm/tool.
"""
import copy
import importlib
from pathlib import Path
import sys

import numpy as np

from .cad_tool import ARM_JOINTS, TINE_JOINTS


def build_cad_planner(source_root, urdf_path, *, voxel_dims=(2.4, 2.4, 2.4), voxel_size=.02):
    import yaml
    source_root = Path(source_root).resolve()
    with (source_root / 'description' / 'shelfbot_gripper_curobo.yml').open() as stream:
        config = copy.deepcopy(yaml.safe_load(stream))
    kin = config['robot_cfg']['kinematics']
    kin['urdf_path'] = str(Path(urdf_path).resolve())
    kin['base_link'] = 'ur_base_link'
    kin['tool_frames'] = ['gripper_pad_tcp']
    kin['lock_joints'] = dict.fromkeys(TINE_JOINTS, 0.)
    # Keep all CAD collision spheres, including the body, panel and six tines.
    # Existing attachment reservation stays on gripper_tcp, with FK conversion
    # handled by attach_object_world (no assumed zero tool offset).
    sys.path.insert(0, str(source_root))
    old = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        module = importlib.import_module('shelfbot.planning.curobo_planner')
    finally:
        sys.dont_write_bytecode = old
        sys.path.remove(str(source_root))
    if not Path(module.__file__).resolve().is_relative_to(source_root):
        raise ValueError('a different shelfbot copy is already imported')
    planner = module.CuroboPlanner(robot_config=config, attach_parent_link='gripper_tcp',
                                   voxel_collision_enabled=True, voxel_dims_m=voxel_dims,
                                   voxel_size_m=voxel_size, attach_surface_radius_m=.002)
    planner.build()
    planner.warmup()
    if tuple(planner.joint_names) != ARM_JOINTS:
        raise ValueError('cuRobo joint order does not match imported CAD UR10')
    return planner


class IsaacCadMotion:
    def __init__(self, articulation, planner, tool, contact, *,
                 max_speed=.1, tracking_tolerance=.03):
        if not planner.is_ready or tuple(planner.joint_names) != tool.arm_joints:
            raise ValueError('ready CAD planner required')
        if not 0 < max_speed <= .1 or tracking_tolerance <= 0:
            raise ValueError('invalid drive speed/tolerance')
        self.robot, self.planner, self.tool, self.contact = articulation, planner, tool, contact
        self.arm_indices = np.array([articulation.get_dof_index(n) for n in ARM_JOINTS])
        self.tine_indices = np.array([articulation.get_dof_index(n) for n in TINE_JOINTS])
        self.max_speed, self.tolerance = max_speed, tracking_tolerance
        self.state, self.held = 'idle', False
        self.path, self.source = None, None
        self.collision_time = -1.
        self.attachment = None
        self.last_grid = None
        self.goal_pose = None

    def joint_positions(self):
        q = np.asarray(self.robot.get_joint_positions(joint_indices=self.arm_indices), float)
        if q.shape != (6,) or not np.isfinite(q).all():
            raise ValueError('invalid measured articulation joints')
        return tuple(q)

    def parked(self):
        linear = np.asarray(self.robot.get_linear_velocity(), float)
        angular = np.asarray(self.robot.get_angular_velocity(), float)
        return bool(np.isfinite(linear).all() and np.isfinite(angular).all()
                    and np.linalg.norm(linear) < .005 and np.linalg.norm(angular) < .005)

    def observe_source(self, observation):
        if observation.source != 'sensor' or observation.frame != 'ur_base_link':
            raise ValueError('payload pose must be measured in the current arm frame')
        self.source = observation

    def update_esdf(self, distances, origin, simulation_time):
        """Actual signed nvblox distances, arm-frame grid; unknown stays occupied.

        Integration must transform/crop its sensor ESDF to this planning frame.
        Unlike upstream's free-space sentinel conversion, -1000/NaN are occupied.
        This method acknowledges time only after the GPU world update succeeds.
        """
        grid = np.asarray(distances, np.float32)
        if grid.ndim != 3 or not grid.size or not np.isfinite(origin).all():
            raise ValueError('invalid sensor ESDF')
        grid = np.where(np.isfinite(grid) & (grid > -999), grid, -.1)
        changed = self.last_grid is not None and (self.last_grid.shape != grid.shape
                  or not np.array_equal(self.last_grid, grid))
        self.planner.update_world_from_esdf(grid, origin)
        if self.held:
            # Upstream warns a world rebuild can drop the reserved collision
            # attachment. Reapply only if its kinematics query says it vanished.
            try:
                self.planner.attached_clear_sphere(self.joint_positions())
            except RuntimeError:
                self.planner.attach_object_world(*self.attachment)
        self.last_grid = grid.copy()
        self.collision_time = float(simulation_time)
        if changed and self.state == 'running':
            # A fresh map is not proof that the old trajectory remains safe.
            # Hold and replan from encoders against the newly acknowledged world.
            goal = self.goal_pose
            self._drive(self.joint_positions())
            self.state = 'idle'
            try:
                self.begin(goal, self.joint_positions())
            except Exception:
                self.stop()
                raise

    def begin(self, pose, joints, straight_line=False, joint_space=False):
        # straight_line is the ros_adapter host-executor contact-descent path; this CAD planner always
        # collision-plans, so the flag is accepted for protocol parity and otherwise ignored here.
        if self.state == 'running':
            raise RuntimeError('arm movement already active')
        if self.collision_time < 0:
            raise RuntimeError('no acknowledged sensor collision world')
        if not np.allclose(joints, self.joint_positions(), atol=.01):
            raise ValueError('start joints changed before planning')
        tines = np.asarray(self.robot.get_joint_positions(joint_indices=self.tine_indices))
        if not np.isfinite(tines).all() or np.max(np.abs(tines)) > .005:
            raise ValueError('suction-only slice requires all six CAD tines retracted')
        result = self.planner.plan_to_pose(joints, pose[:3], pose[3:],
                                           hold_orientation_axes=[0,0,0,1,1,0] if self.held else None)
        if not result.success or tuple(result.joint_names) != ARM_JOINTS:
            raise RuntimeError('CAD collision planning failed: ' + str(result.reason))
        path = np.asarray(result.positions, float)
        limits = np.asarray(self.tool.arm_limits)
        if (path.ndim != 2 or path.shape[1] != 6 or len(path) < 2
                or not np.isfinite(path).all() or np.any(path < limits[:,0])
                or np.any(path > limits[:,1]) or not np.allclose(path[0], joints, atol=.03)):
            raise ValueError('invalid CAD joint trajectory')
        self.goal_pose = tuple(pose)
        self.path, self.index, self.state = path, 1, 'running'
        self.commanded = np.array(joints, float)

    def tick(self, dt):
        """Call once per physics tick, before stepping PhysX."""
        if not np.isfinite(dt) or not 0 < dt <= .1:
            raise ValueError('invalid physics step')
        if self.state != 'running':
            return
        q = np.asarray(self.joint_positions())
        if np.max(np.abs(q-self.commanded)) > .15:
            self.stop()
            return
        goal = self.path[self.index]
        delta = goal-self.commanded
        fraction = min(1., self.max_speed*dt / max(1e-9, np.max(np.abs(delta))))
        self.commanded = self.commanded + fraction*delta
        self._drive(self.commanded)
        if np.max(np.abs(q-goal)) < self.tolerance and np.max(np.abs(self.commanded-goal)) < .001:
            if self.index + 1 < len(self.path):
                self.index += 1
            else:
                velocity = np.asarray(self.robot.get_joint_velocities(joint_indices=self.arm_indices))
                if np.isfinite(velocity).all() and np.max(np.abs(velocity)) < .02:
                    self.state = 'succeeded'

    def _drive(self, positions):
        from isaacsim.core.utils.types import ArticulationAction
        self.robot.apply_action(ArticulationAction(
            joint_positions=np.r_[positions, np.zeros(6)],
            joint_indices=np.r_[self.arm_indices, self.tine_indices]))

    def poll(self):
        return self.state if self.state in ('running', 'succeeded', 'failed') else 'failed'

    def stop(self):
        self.state = 'failed'
        self._drive(self.joint_positions())

    def update_payload(self, held, profile):
        self.tool.validate_profile(profile)
        joints = self.joint_positions()
        if held:
            if not self.contact.read() or self.source is None or self.source.sku != profile.sku:
                return False
            centre = np.array(self.source.pose[:3]) - (0,0,profile.dimensions[2]/2)
            self.attachment = (joints, profile.dimensions, centre, self.source.pose[3:])
            self.planner.attach_object_world(*self.attachment)
            self.held = True
        else:
            if self.contact.read():
                return False
            self.planner.detach_object()
            self.held = False
            self.attachment = None
        # Physical grasp comes solely from NativePanelContact. This only changes
        # planner collision spheres, never a PhysX joint or product transform.
        return True
