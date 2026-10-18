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
                 spec=MobileSpec(), hitch_path='/MobileHitch', wheel_angle_reader=None):
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
            drive.CreateDampingAttr(self.DRIVE_DAMPING)
            drive.CreateMaxForceAttr(self.DRIVE_MAX_FORCE)
            drive.CreateTargetVelocityAttr(0.0)
            self.drives.append(drive)
        # SIM-only stationary base hold (twin.restock task base-hold-under-stiff-arm): a real parked
        # base is braked/chocked, but the imported free base is only held by the wheels' VELOCITY
        # drive -- which leaks a creep velocity F/damping under a steady load -- so the stiff arm's
        # reaction torque through the arm mount creeps the base past the measured stationarity gate
        # during a manipulation move (run-161). hold() engages a parking brake: it switches each
        # wheel drive to a stiff POSITION hold at the wheel's current angle, so the wheels cannot
        # roll and the base cannot creep (a position drive's steady-state error is bounded, F/Kp,
        # not a continuous velocity leak). It is gated to the stationary/manipulation phase and
        # released the instant the mission drives again (command()). The real-robot path is
        # untouched -- this only reconfigures the SIM wheel drive while parked; no teleport or
        # kinematic pin. (Verified on GPU: of the available levers only the wheel position-hold
        # helps; base-link rigid-body damping is a PhysX mid-sim no-op and a higher-maxForce
        # velocity brake is counterproductive -- data/surface-gripper-latch-v1 / Follow-up 4.)
        self.wheel_angle_reader = wheel_angle_reader
        self._held = False
        self._braked = False
        self._cart_braked = False
        # Couple-phase cart hold (firstmate couple-free-cart-shove): when set, the cart stays
        # pinned to the world through the WHOLE couple approach + fine-dock so the base cannot
        # shove the free cart; only attach() force-releases it. All other release_cart_brake()
        # callers (command(), the isaac_port per-tick action logic) no-op while held.
        self._cart_hold = False
        self._base_full_braked = False

    DRIVE_DAMPING = 100.0
    DRIVE_MAX_FORCE = 80.0
    # Base-hold stiffening, evidence-based (firstmate msg 251, sim-only, gate unchanged):
    #   1e5: run-164 base_dxy=5.3mm (0.3mm over 5mm gate), dyaw 9.82mrad (within).
    #   3e5: run-165 base_dxy=3.5mm (within gate) but dyaw 10.91mrad (0.91 over the 10mrad gate) --
    #        the roll-lock cut translation; yaw is now the binding axis. The locked wheels resist yaw
    #        via the wheel-separation lever, so a further moderate 2x bump should shave it. Per-step
    #        moderate, cumulative 6x (NOT a 10x single jump -- PhysX destabilises at absurd values).
    HOLD_STIFFNESS = 6.0e5
    HOLD_DAMPING = 5.0e3
    HOLD_MAX_FORCE = 6.0e4

    def hold(self):
        """Engage the stationary parking brake: lock each wheel at its current angle with a stiff
        POSITION drive. Gated to the stationary/manipulation phase; released by command() when the
        mission drives. Needs the current wheel angles (wheel_angle_reader, radians); without a
        reader it degrades to park() (a plain zero-velocity stop)."""
        if self._held:
            return (0.0, 0.0)
        if self.wheel_angle_reader is None:
            return self.park()
        angles = self.wheel_angle_reader()
        for drive, angle in zip(self.drives, angles):
            drive.GetStiffnessAttr().Set(self.HOLD_STIFFNESS)
            drive.GetDampingAttr().Set(self.HOLD_DAMPING)
            drive.GetMaxForceAttr().Set(self.HOLD_MAX_FORCE)
            drive.GetTargetPositionAttr().Set(math.degrees(float(angle))) if drive.GetTargetPositionAttr() \
                else drive.CreateTargetPositionAttr().Set(math.degrees(float(angle)))
            drive.GetTargetVelocityAttr().Set(0.0)
        self._held = True
        return (0.0, 0.0)

    def release(self):
        """Restore the free (zero-stiffness velocity) driving drive so the base can move again."""
        if self._held:
            for drive in self.drives:
                drive.GetStiffnessAttr().Set(0.0)
                drive.GetDampingAttr().Set(self.DRIVE_DAMPING)
                drive.GetMaxForceAttr().Set(self.DRIVE_MAX_FORCE)
                drive.GetTargetVelocityAttr().Set(0.0)
            self._held = False

    BRAKE_PATH = '/MobileParkingBrake'

    def engage_brake(self):
        """SIM PARKING BRAKE (firstmate run-210/212): lock the chassis YAW to the world with a D6 joint at
        its CURRENT pose while the arm works -- the sim equivalent of the brakes a real AMR engages before
        manipulating. The wheel position-hold (hold()) is stiffness-saturated on yaw (runs 164-166), so a
        0.45 kg carton welded to the arm pivots the base past the 25 mrad gate (run-208/209).

        A FULL FixedJoint brake held the yaw (base_dyaw 0.00 mrad, run-210/211) but RIGIDLY over-
        constrained the loop world->base->arm->(weld)->carton->support->world once the carton welds to the
        arm while still on its support, oscillating the arm (GOAL_MISS vel ~3.2 rad/s). So lock ONLY the
        yaw DOF (the 47.6 mrad problem) and leave translation + roll/pitch FREE, so the base can shift a
        little to relieve that loop (base_dxy stayed 8.8 mm, inside the 15 mm gate). Gated to the
        stationary arm phase, released before any base drive. SIM-only; real rig brakes physically."""
        from pxr import UsdPhysics, UsdGeom, Gf, Sdf
        if self._braked:
            return
        world = UsdGeom.XformCache().GetLocalToWorldTransform(self.stage.GetPrimAtPath(self.robot_body))
        t, q = world.ExtractTranslation(), world.ExtractRotationQuat()
        joint = UsdPhysics.Joint.Define(self.stage, self.BRAKE_PATH)     # generic D6
        joint.CreateBody1Rel().SetTargets([self.robot_body])            # body0 empty == world anchor
        joint.CreateLocalPos0Attr(Gf.Vec3f(float(t[0]), float(t[1]), float(t[2])))
        joint.CreateLocalRot0Attr(Gf.Quatf(float(q.GetReal()), Gf.Vec3f(*[float(v) for v in q.GetImaginary()])))
        joint.CreateLocalPos1Attr(Gf.Vec3f(0.0, 0.0, 0.0))
        joint.CreateLocalRot1Attr(Gf.Quatf(1.0))
        # Lock ONLY rotZ (yaw): low>high == locked. transX/Y/Z and rotX/Y get no limit == free.
        lock = UsdPhysics.LimitAPI.Apply(joint.GetPrim(), 'rotZ')
        lock.CreateLowAttr(1.0)
        lock.CreateHighAttr(-1.0)
        self._braked = True
        print('MOBILE_BRAKE engaged (yaw-lock D6) at chassis pose', flush=True)

    def release_brake(self):
        """Release the sim parking brake so the base can drive (couple/tow/park)."""
        if self._braked:
            self.stage.RemovePrim(self.BRAKE_PATH)
            self._braked = False
            print('MOBILE_BRAKE released', flush=True)

    CART_BRAKE_PATH = '/CartParkingBrake'

    def engage_cart_brake(self):
        """SIM CART PARKING BRAKE (firstmate 287): while the robot places a carton ONTO the uncoupled
        cart during collect, pin the cart to the world so the placement contact cannot roll it. A real
        rollwagen at rest has caster brakes + bearing friction; the sim cart's casters are only
        bearing-DAMPED (configure_cart_bearings), which resists creep but not a landing-carton impulse,
        so a 0.45 kg carton rolled the 35 kg cart 15.6 mm past the stationary gate (run-228). A full
        FixedJoint is safe here (unlike the ROBOT yaw-lock rationale): the cart is NOT in the arm/weld
        loop during collect, it is just the adjacent load target. Released before coupling so the cart
        is free to be towed. SIM-only; the real cart brakes physically."""
        from pxr import UsdPhysics, UsdGeom, Gf
        if getattr(self, '_cart_braked', False):
            return
        # Anchor at the cart's CURRENT world pose so the joint is satisfied at creation (zero
        # correction impulse); default coinciding origins would yank the cart to the world origin.
        world = UsdGeom.XformCache().GetLocalToWorldTransform(self.stage.GetPrimAtPath(self.cart_body))
        t, q = world.ExtractTranslation(), world.ExtractRotationQuat()
        joint = UsdPhysics.FixedJoint.Define(self.stage, self.CART_BRAKE_PATH)
        joint.CreateBody1Rel().SetTargets([self.cart_body])   # body0 empty == world anchor
        joint.CreateLocalPos0Attr(Gf.Vec3f(float(t[0]), float(t[1]), float(t[2])))
        joint.CreateLocalRot0Attr(Gf.Quatf(float(q.GetReal()), Gf.Vec3f(*[float(v) for v in q.GetImaginary()])))
        joint.CreateLocalPos1Attr(Gf.Vec3f(0.0, 0.0, 0.0))
        joint.CreateLocalRot1Attr(Gf.Quatf(1.0))
        self._cart_braked = True
        print('CART_BRAKE engaged (fixed-to-world) at cart pose', flush=True)

    def hold_cart_pin(self):
        """Pin the cart to the world and HOLD it through couple (approach + fine-dock).

        Engages the sim cart brake at the cart's current pose and latches the hold, so every
        other release_cart_brake() caller (command(), the isaac_port per-tick action logic)
        no-ops until attach() force-releases it. Phase-local: the caller invokes this once at
        couple start. SIM-only; the real cart brakes physically, so real-rig semantics are
        untouched (this path is reached only when the couple runtime opts in)."""
        self.engage_cart_brake()
        self._cart_hold = True

    def release_cart_brake(self, *, force=False):
        """Release the cart parking brake so the cart is free to be coupled and towed.

        No-op while the couple cart hold is latched (unless force=True), so the cart stays
        pinned through the whole couple approach; attach() force-releases it once coupled."""
        if self._cart_hold and not force:
            return
        self._cart_hold = False
        if getattr(self, '_cart_braked', False):
            self.stage.RemovePrim(self.CART_BRAKE_PATH)
            self._cart_braked = False

    BASE_FULL_BRAKE_PATH = '/MobileGraspBrake'

    def engage_base_full_brake(self):
        """SIM FULL BASE BRAKE for the PRE-WELD grasp press (firstmate 289/run-232): pin the whole
        chassis (translation + yaw) to the world with a FixedJoint while the arm presses into the
        carton to seal the grip, BEFORE any carton welds to the arm. The yaw-lock-only brake leaves
        translation free (to relieve the world->base->arm->weld->carton->support loop during the
        welded lift), but with translation free the side-grip contact-press slides the base ~15.5 mm
        laterally (run-232 > the gate) AND keeps the pad ~0.5 mm off the carton so the grip never
        seals / the flush-contact weld never fires. A FULL FixedJoint is proven safe HERE: with no
        carton welded there is no over-constrained loop yet (the exact full brake held base_dyaw
        0.00 mrad in run-210/211 and only broke down once the carton welded). Switched to the
        yaw-lock-only brake the instant the carton welds (isaac_port), released before any base drive.
        SIM-only; a real AMR holds station on its drive brakes."""
        from pxr import UsdPhysics, UsdGeom, Gf
        if getattr(self, '_base_full_braked', False):
            return
        self.release_brake()   # mutually exclusive with the yaw-lock D6 brake
        # Anchor at the chassis CURRENT world pose so the joint is satisfied at creation (zero
        # correction impulse); coinciding default origins would yank the base to the world origin.
        world = UsdGeom.XformCache().GetLocalToWorldTransform(self.stage.GetPrimAtPath(self.robot_body))
        t, q = world.ExtractTranslation(), world.ExtractRotationQuat()
        joint = UsdPhysics.FixedJoint.Define(self.stage, self.BASE_FULL_BRAKE_PATH)
        joint.CreateBody1Rel().SetTargets([self.robot_body])   # body0 empty == world anchor
        joint.CreateLocalPos0Attr(Gf.Vec3f(float(t[0]), float(t[1]), float(t[2])))
        joint.CreateLocalRot0Attr(Gf.Quatf(float(q.GetReal()), Gf.Vec3f(*[float(v) for v in q.GetImaginary()])))
        joint.CreateLocalPos1Attr(Gf.Vec3f(0.0, 0.0, 0.0))
        joint.CreateLocalRot1Attr(Gf.Quatf(1.0))
        self._base_full_braked = True
        print('BASE_FULL_BRAKE engaged (fixed-to-world) for pre-weld grasp press', flush=True)

    def release_base_full_brake(self):
        """Release the pre-weld full base brake (switch to yaw-lock at the weld, or drive the base)."""
        if getattr(self, '_base_full_braked', False):
            self.stage.RemovePrim(self.BASE_FULL_BRAKE_PATH)
            self._base_full_braked = False
            print('BASE_FULL_BRAKE released', flush=True)
            print('CART_BRAKE released', flush=True)

    def park(self):
        for drive in self.drives:
            drive.GetTargetVelocityAttr().Set(0.0)
        return (0.0, 0.0)

    def command(self, v, omega, observation, *, transport_ready=False):
        self.release_brake()  # resuming driving: drop the sim parking brake before any base motion
        self.release_base_full_brake()  # and the pre-weld full base pin
        self.release_cart_brake()  # cart pin -> free to couple/tow (NO-OP while couple cart-hold latched)
        self.release()  # resuming driving: drop the stationary wheel-hold before commanding wheels
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
        # Author the robot↔cart hitch FIRST (raises if not seated within tolerance, leaving the
        # cart still world-pinned for another fine-dock pass), THEN force-release the world pin so
        # the now-coupled train is free to tow. Order: both constraints hold briefly, never neither.
        self.coupler.attach(observation, lambda: author_hitch(
            self.stage, self.robot_body, self.cart_body, self.hitch_path, self.spec))
        self.release_cart_brake(force=True)
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
