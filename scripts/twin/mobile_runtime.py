"""USD drive/coupling adapter callable in an existing Isaac process.

Call on the simulation thread each control tick. Wheel commands use USD angular
DriveAPI (degrees/sec); an Isaac articulation adapter may instead consume the
rad/sec values returned by twin.mobile.wheel_targets directly. Sensor acquisition,
arm position holding and physics stepping belong to the integrating application.
"""
import math
from .mobile import Coupler, MobileSpec, wheel_targets, wrap
from .mobile_assets import author_hitch


class MobileController:
    def __init__(self, stage, robot_body, cart_body, wheel_joint_paths,
                 spec=MobileSpec(), hitch_path='/MobileHitch'):
        from pxr import UsdPhysics
        if len(wheel_joint_paths) != 2 or len(set(wheel_joint_paths)) != 2:
            raise ValueError('left and right wheel joint paths required, in that order')
        for path in wheel_joint_paths:
            if not stage.GetPrimAtPath(path).IsA(UsdPhysics.RevoluteJoint):
                raise ValueError(f'not a revolute wheel joint: {path}')
        for path in (robot_body, cart_body):
            if not stage.GetPrimAtPath(path).HasAPI(UsdPhysics.RigidBodyAPI):
                raise ValueError(f'not a rigid body: {path}')
        if stage.GetPrimAtPath(hitch_path):
            raise ValueError('fresh controller requires an uncoupled stage')
        self.stage, self.robot_body, self.cart_body = stage, robot_body, cart_body
        self.spec, self.hitch_path = spec, hitch_path
        self.coupler = Coupler(spec)
        self.drives = []
        for path in wheel_joint_paths:
            drive = UsdPhysics.DriveAPI.Apply(stage.GetPrimAtPath(path), 'angular')
            drive.CreateTypeAttr('force')
            drive.CreateStiffnessAttr(0.0)
            drive.CreateDampingAttr(100.0)
            drive.CreateMaxForceAttr(80.0)
            drive.CreateTargetVelocityAttr(0.0)
            self.drives.append(drive)

    def park(self):
        for drive in self.drives:
            drive.GetTargetVelocityAttr().Set(0.0)
        return (0.0, 0.0)

    def command(self, v, omega, observation, *, transport_ready=False):
        # Stale/missing kinematics must never leave a previous nonzero drive latched.
        if observation is None or not math.isfinite(observation.age) or not 0 <= observation.age <= 0.1:
            return self.park()
        if self.coupler.attached and not self.coupling_state(observation)['healthy']:
            return self.park()
        try:
            targets = wheel_targets(v, omega, self.spec, transport_ready=transport_ready,
                                    coupled=self.coupler.attached,
                                    articulation=wrap(observation.robot.yaw-observation.cart.yaw))
        except ValueError:
            self.park()
            raise
        for drive, target in zip(self.drives, targets):
            drive.GetTargetVelocityAttr().Set(math.degrees(target))
        return targets

    def attach(self, observation):
        self.park()
        self.coupler.attach(observation, lambda: author_hitch(
            self.stage, self.robot_body, self.cart_body, self.hitch_path, self.spec))
        return self.coupling_state(observation)

    def detach(self, observation):
        self.park()
        def remove():
            if not self.stage.RemovePrim(self.hitch_path) or self.stage.GetPrimAtPath(self.hitch_path):
                raise RuntimeError('could not remove physical hitch')
        self.coupler.detach(observation, remove)
        return self.coupling_state(observation)

    def coupling_state(self, observation):
        from pxr import UsdPhysics
        joint = UsdPhysics.RevoluteJoint.Get(self.stage, self.hitch_path)
        exists = bool(joint) and joint.GetJointEnabledAttr().Get()
        a = observation.robot.point(self.spec.hitch_x)
        b = observation.cart.point(self.spec.cart_hitch_x)
        residual = math.hypot(math.dist(a, b), observation.robot_hitch_z-observation.cart_hitch_z)
        healthy = (self.coupler.attached and exists and math.isfinite(residual) and residual <= 0.04
                   and math.isfinite(observation.age) and 0 <= observation.age <= 0.1
                   and abs(wrap(observation.robot.yaw-observation.cart.yaw)) <= self.spec.max_articulation)
        return {'requested_attached': self.coupler.attached, 'joint_present': bool(exists),
                'healthy': bool(healthy), 'hitch_error_m': residual,
                'evidence_source': 'caller_supplied_simulator_measurements'}
