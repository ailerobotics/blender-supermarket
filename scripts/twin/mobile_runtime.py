"""USD drive/coupling adapter callable in an existing Isaac process.

Call on the simulation thread each control tick. Wheel commands use USD angular
DriveAPI (degrees/sec); an Isaac articulation adapter may instead consume the
rad/sec values returned by twin.mobile.wheel_targets directly. Sensor acquisition,
arm position holding and physics stepping belong to the integrating application.
"""
import math
from .mobile import Coupler, MobileSpec, wheel_targets, wrap
from .mobile_assets import author_hitch


def _quat_to_yaw(q):
    """Yaw (rad) from a pxr Gf quaternion (w + xyz), matching isaac_port.body_roll_pitch_deg's ZYX
    convention: yaw = atan2(2(wz+xy), 1-2(yy+zz))."""
    w = float(q.GetReal()); x, y, z = (float(v) for v in q.GetImaginary())
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def _quat_from_rpy(roll, pitch, yaw):
    """(w, x, y, z) for a ZYX (yaw*pitch*roll) euler attitude -- the inverse of the roll/pitch/yaw
    extraction body_roll_pitch_deg / _quat_to_yaw use. Pure; CPU-testable."""
    cy, sy = math.cos(yaw / 2), math.sin(yaw / 2)
    cp, sp = math.cos(pitch / 2), math.sin(pitch / 2)
    cr, sr = math.cos(roll / 2), math.sin(roll / 2)
    return (cr * cp * cy + sr * sp * sy,
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy)


class MobileController:
    def __init__(self, stage, robot_body, cart_body, wheel_joint_paths,
                 spec=MobileSpec(), hitch_path='/MobileHitch', wheel_angle_reader=None,
                 drive_max_force=None, drive_damping=None, station_pitch_hold=False,
                 station_hold_roll_deg=0.0, station_hold_pitch_deg=-1.17,
                 station_place_full_brake=False, station_place_full_brake_effort_limit=None,
                 station_full_brake_engage_phase=None):
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
        # fm431 (A2): the SIM wheel force-velocity drive (torque = damping * (target - measured), capped at
        # maxForce) is the base's drive fidelity. run-325 measured the coupled base UNDER-tracking the
        # commanded (v, omega) on curves -- under-translating (measured ~0.033 m/s vs commanded ~0.0625)
        # and over-rotating (measured yaw-rate up to ~0.19 vs commanded ~0.042) -- so the authored tow
        # lane's feasible turns trip the gate's measured-arc check. These let the mission tune the SIM
        # drive toward the real base's documented limits (0.35 m/s, 0.35 rad/s towing ~215 kg; see
        # docs/mobile_robot_rollwagen.md) WITHOUT bending the lanes around a sim artefact. None => the
        # prior class-constant values (80 N and 100), byte-identical. The parking HOLD (hold()) keeps its
        # own stiff HOLD_* constants. Efficacy is a GPU-measured sweep (run-326), not offline-provable.
        self.drive_max_force = float(self.DRIVE_MAX_FORCE if drive_max_force is None else drive_max_force)
        self.drive_damping = float(self.DRIVE_DAMPING if drive_damping is None else drive_damping)
        if not (self.drive_max_force > 0 and self.drive_damping > 0):
            raise ValueError('wheel drive max_force and damping must be positive')
        self.drives = []
        for path in wheel_joint_paths:
            drive = UsdPhysics.DriveAPI.Apply(stage.GetPrimAtPath(path), 'angular')
            drive.CreateTypeAttr('force')
            drive.CreateStiffnessAttr(0.0)
            drive.CreateDampingAttr(self.drive_damping)
            drive.CreateMaxForceAttr(self.drive_max_force)
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
        # sim-chassis-pitch-stability-v1 FALLBACK (sim-only, opt-in, default OFF => byte-identical): when
        # set, engage_brake ALSO locks the chassis pitch/roll (rotX/rotY), not just yaw, so a free-pitch
        # see-saw chassis cannot rock the welded carton off level at a manipulation station (coldwin33-37).
        # Released with the brake exactly like the yaw lock. Labelled like CAGE_COLLISION; the compliant
        # caster suspension is the preferred source fix, this is the belt-and-suspenders station hold.
        self.station_pitch_hold = bool(station_pitch_hold)
        # coldwin39 fix: the station hold locks pitch/roll to this CONFIGURABLE TARGET attitude (degrees),
        # NOT the chassis's instantaneous attitude. coldwin39 latched the instantaneous -4.25 deg end stop
        # the base had drifted to during grasp, so the carton set down 2.02-2.04 deg (over the 2.0 bound).
        # Default is the rig REST pitch (~-1.17 deg, the attitude the arm's set-down was planned around);
        # roll 0. Set both 0 for world-level.
        self.station_hold_roll_deg = float(station_hold_roll_deg)
        self.station_hold_pitch_deg = float(station_hold_pitch_deg)
        # sim-station-full-brake-place-v1 (opt-in, default OFF => byte-identical): when set, the LOADED
        # collect place descent onto the cart support FULL-locks the base (translation AND rotation) over
        # the place window instead of the rotation-only station hold, so the welded from-above descent
        # reaction cannot shove the translation-free base (coldwin49: 0.08-0.18 m south recoil -> the
        # stationary guard aborted before the contact release fired). Anchored at the pitch/roll TARGET
        # attitude (engage_station_full_lock), returning to the rotation-only hold at stow/withdraw.
        # Requires station_pitch_hold (the lock reuses its target attitude); the port only engages the
        # full lock inside the station_pitch_hold brake branch.
        self.station_place_full_brake = bool(station_place_full_brake)
        # sim-station-lock-from-preplace-v1 (opt-in, default 'place' => byte-identical): WHEN the full D6
        # lock engages relative to the carried pre-place ('place' = today, at the descent; 'preplace' = at
        # the start of leg 1; 'hold_firm' = the moment a carried joint is firmed). The port
        # (isaac_port._apply_base_brake) owns the phase->engage decision (station_full_brake.full_lock_window);
        # this is carried here only so the port can read it off the controller like station_place_full_brake.
        # Resolved/validated (raises on an unknown value) so a bad key fails loudly at construction.
        from .restock import station_full_brake as _sfb
        self.station_full_brake_engage_phase = _sfb.resolve_engage_phase(station_full_brake_engage_phase)
        lim = station_place_full_brake_effort_limit
        if lim is not None:
            lim = float(lim)
            if not (lim > 0 and math.isfinite(lim)):
                raise ValueError('station_place_full_brake_effort_limit must be a positive finite N·m')
        self.station_place_full_brake_effort_limit = lim
        self._station_full_locked = False
        # coldwin39 fix: latch the station hold so it is NOT released+re-engaged every tick/phase (the
        # ~2000 brake cycles coldwin39 logged). While latched, command() keeps the hold on a parked tick.
        self._station_hold_latched = False
        self._held = False
        self._braked = False
        self._cart_braked = False
        # Couple-phase cart hold (firstmate couple-free-cart-shove): when set, the cart stays
        # pinned to the world through the WHOLE couple approach + fine-dock so the base cannot
        # shove the free cart; only attach() force-releases it. All other release_cart_brake()
        # callers (command(), the isaac_port per-tick action logic) no-op while held.
        self._cart_hold = False
        self._base_full_braked = False
        # Controlled welded-stop latch (sim-host-fault-controlled-stop-v1): while set, command()
        # refuses to release the base brake or spin the wheels. A host fault left a carton WELDED to
        # the extended arm (possibly interpenetrating the cart deck), so cutting the drive / releasing
        # the brake would let PhysX resolve the penetration by flinging the base (coldwin33). Latched by
        # the controlled stop (isaac_port.enter_welded_controlled_stop); cleared only once the carton is
        # supported or clear of the deck. SIM-only; no-op on every normal drive (latch stays False).
        self._welded_hold = False

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
                drive.GetDampingAttr().Set(self.drive_damping)      # fm431 (A2): restore the configured drive
                drive.GetMaxForceAttr().Set(self.drive_max_force)
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
        anchor_quat = q
        # sim-chassis-pitch-stability-v1 station pitch hold (opt-in): also lock rotX/rotY so a free-pitch
        # see-saw chassis cannot rock the welded carton off level during manipulation. SIM-only.
        # coldwin39 FIX: anchor the lock at the CONFIGURABLE TARGET attitude (rest pitch/roll + CURRENT
        # yaw), NOT the instantaneous attitude -- so the hold does not latch whatever end stop the base
        # drifted to (coldwin39 latched -4.25 deg -> carton 2.04 deg). The joint then holds the chassis
        # AT the target pitch/roll (a bounded correction from the current attitude).
        if self.station_pitch_hold:
            cur_yaw = _quat_to_yaw(q)
            w2, x2, y2, z2 = _quat_from_rpy(math.radians(self.station_hold_roll_deg),
                                            math.radians(self.station_hold_pitch_deg), cur_yaw)
            anchor_quat = Gf.Quatd(w2, Gf.Vec3d(x2, y2, z2))
        joint = UsdPhysics.Joint.Define(self.stage, self.BRAKE_PATH)     # generic D6
        joint.CreateBody1Rel().SetTargets([self.robot_body])            # body0 empty == world anchor
        joint.CreateLocalPos0Attr(Gf.Vec3f(float(t[0]), float(t[1]), float(t[2])))
        joint.CreateLocalRot0Attr(Gf.Quatf(float(anchor_quat.GetReal()),
                                           Gf.Vec3f(*[float(v) for v in anchor_quat.GetImaginary()])))
        joint.CreateLocalPos1Attr(Gf.Vec3f(0.0, 0.0, 0.0))
        joint.CreateLocalRot1Attr(Gf.Quatf(1.0))
        # Lock ONLY rotZ (yaw): low>high == locked. transX/Y/Z and rotX/Y get no limit == free.
        axes = ['rotZ']
        if self.station_pitch_hold:
            axes += ['rotX', 'rotY']       # also lock roll/pitch (held at the target attitude above)
        for axis in axes:
            lock = UsdPhysics.LimitAPI.Apply(joint.GetPrim(), axis)
            lock.CreateLowAttr(1.0)
            lock.CreateHighAttr(-1.0)
        self._braked = True
        print('MOBILE_BRAKE engaged (%s D6) at chassis pose%s'
              % ('full-rotation-lock@target(roll=%.2f,pitch=%.2f)deg'
                 % (self.station_hold_roll_deg, self.station_hold_pitch_deg)
                 if self.station_pitch_hold else 'yaw-lock',
                 ' [CHASSIS_STATION_PITCH_HOLD sim-only workaround]' if self.station_pitch_hold else ''),
              flush=True)

    def engage_station_hold(self):
        """coldwin39 fix: engage the pitch/roll-at-target station hold and LATCH it, so the brake is held
        across the whole arm phase instead of cycling engage/release every tick (~2000x in coldwin39).
        engage_brake is idempotent, so repeated calls while latched are a no-op. Translation stays free
        (weld-loop relief preserved); only roll/pitch/yaw are locked, pitch/roll at the target attitude."""
        self.engage_brake()
        self._station_hold_latched = True

    def release_station_hold(self):
        """Clear the station-hold latch and release the brake (the base is leaving the arm phase to drive)."""
        self._station_hold_latched = False
        self.release_brake()

    def release_brake(self):
        """Release the sim parking brake so the base can drive (couple/tow/park)."""
        self._station_hold_latched = False   # the brake is gone; keep the latch flag consistent
        if self._braked:
            self.stage.RemovePrim(self.BRAKE_PATH)
            self._braked = False
            print('MOBILE_BRAKE released', flush=True)

    STATION_FULL_BRAKE_PATH = '/MobileStationFullBrake'

    def engage_station_full_lock(self, from_phase=None):
        """sim-station-full-brake-place-v1: FULL D6 lock (translation + rotation) of the chassis for the
        LOADED collect place descent onto the cart support.

        ``from_phase`` (sim-station-lock-from-preplace-v1) names the controller phase the lock engaged at,
        for the one-line log; the port passes it so an engage from pre-place is visible in the console.

        Locks transX/Y/Z AND rotX/Y/Z. Rotation is anchored at the pitch/roll TARGET attitude
        (``station_hold_pitch_deg`` / ``station_hold_roll_deg`` + the CURRENT yaw) -- the same target the
        rotation-only station hold (engage_brake) uses -- so the full lock sits ON TOP of that hold rather
        than latching whatever attitude the chassis drifted to. Translation is locked at the current
        chassis pose (the standoff pose the base holds just before the descent begins), closing off the
        x/y recoil that the yaw-lock-only hold leaves free (coldwin49). Mutually exclusive with the
        rotation-only brake (removes it first; one base D6). Idempotent while locked. Released back to the
        rotation-only hold by :meth:`release_station_full_lock` once the carton is detached and the arm
        has withdrawn. SIM-only; a real AMR holds station on its wheel brakes."""
        from pxr import UsdPhysics, UsdGeom, Gf
        if getattr(self, '_station_full_locked', False):
            return
        self.release_brake()   # drop the rotation-only D6 (a single base D6 is authored at a time)
        world = UsdGeom.XformCache().GetLocalToWorldTransform(self.stage.GetPrimAtPath(self.robot_body))
        t, q = world.ExtractTranslation(), world.ExtractRotationQuat()
        cur_yaw = _quat_to_yaw(q)
        w2, x2, y2, z2 = _quat_from_rpy(math.radians(self.station_hold_roll_deg),
                                        math.radians(self.station_hold_pitch_deg), cur_yaw)
        anchor_quat = Gf.Quatd(w2, Gf.Vec3d(x2, y2, z2))
        joint = UsdPhysics.Joint.Define(self.stage, self.STATION_FULL_BRAKE_PATH)   # generic D6
        joint.CreateBody1Rel().SetTargets([self.robot_body])            # body0 empty == world anchor
        joint.CreateLocalPos0Attr(Gf.Vec3f(float(t[0]), float(t[1]), float(t[2])))
        joint.CreateLocalRot0Attr(Gf.Quatf(float(anchor_quat.GetReal()),
                                           Gf.Vec3f(*[float(v) for v in anchor_quat.GetImaginary()])))
        joint.CreateLocalPos1Attr(Gf.Vec3f(0.0, 0.0, 0.0))
        joint.CreateLocalRot1Attr(Gf.Quatf(1.0))
        # Lock ALL six axes: low>high == locked (same convention as engage_brake's yaw lock).
        for axis in ('transX', 'transY', 'transZ', 'rotX', 'rotY', 'rotZ'):
            lock = UsdPhysics.LimitAPI.Apply(joint.GetPrim(), axis)
            lock.CreateLowAttr(1.0)
            lock.CreateHighAttr(-1.0)
        self._station_full_locked = True
        print('MOBILE_BRAKE engaged (full-D6 lock transX,transY,transZ,rotX,rotY,rotZ @target'
              '(roll=%.2f,pitch=%.2f)deg%s) [STATION_PLACE_FULL_BRAKE loaded-place descent]'
              % (self.station_hold_roll_deg, self.station_hold_pitch_deg,
                 '' if from_phase is None else ' from phase=%s' % from_phase), flush=True)

    def release_station_full_lock(self):
        """Release the loaded-place full D6 lock (the base returns to the rotation-only station hold)."""
        if getattr(self, '_station_full_locked', False):
            self.stage.RemovePrim(self.STATION_FULL_BRAKE_PATH)
            self._station_full_locked = False
            print('MOBILE_BRAKE released (full-D6 lock)', flush=True)

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

    def latch_welded_hold(self):
        """Engage the controlled welded-stop latch (sim-host-fault-controlled-stop-v1): command()
        refuses to release the base brake or drive while set, so a re-begin cannot unpin a welded,
        interpenetrating arm. Idempotent."""
        self._welded_hold = True

    def release_welded_hold(self):
        """Clear the controlled welded-stop latch once the carton is supported or clear of the deck."""
        self._welded_hold = False

    def command(self, v, omega, observation, *, transport_ready=False):
        if getattr(self, '_welded_hold', False):
            # Controlled welded-stop: keep the base PLANTED. Return a zero-velocity park WITHOUT
            # releasing any brake, so a drive command issued by a re-begin cannot unpin a welded,
            # interpenetrating arm (the coldwin33 fling). Cleared by release_welded_hold().
            return self.park()
        if ((getattr(self, '_station_hold_latched', False)
             or getattr(self, '_station_full_locked', False)) and not (v or omega)):
            # coldwin39 fix: a parked (v=omega=0) tick while the station hold is latched must NOT release
            # the brake -- releasing + re-engaging every parked control tick is the ~2000x cycle coldwin39
            # logged. The sim-station-full-brake-place-v1 loaded-place full lock gets the same protection
            # (a parked tick keeps it). A genuine drive command (v or omega != 0) falls through to release.
            return self.park()
        self.release_brake()  # resuming driving: drop the sim parking brake before any base motion
        self.release_station_full_lock()  # and the loaded-place full D6 lock
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

    def uncouple(self, observation):
        """Leave the cart PARKED and release the tow coupling so the robot can drive home empty.

        The mirror of :meth:`attach` (which authors the hitch, then force-releases the cart's
        world pin so the coupled train is free to tow): here we first pin the parked cart to the
        world (``hold_cart_pin`` -- engage the cart brake and LATCH it) so releasing the tow joint
        cannot let the just-parked cart drift, THEN ``detach`` removes the hitch. The latch keeps
        the cart braked after the robot leaves -- every later ``release_cart_brake`` (e.g. the one
        ``command`` issues when the base drives away) no-ops while ``_cart_hold`` is set, so the
        cart stays parked for the rest of the mission. Returns the post-detach ``coupling_state``
        (``joint_present`` False, ``requested_attached`` False once released).

        SIM-only (the real cart brakes on its caster brakes and a real coupler releases the
        physical latch); reached only when the return runtime opts into leaving the cart parked.
        """
        self.park()
        self.hold_cart_pin()          # pin + latch the parked cart before releasing the tow joint
        state = self.detach(observation)
        print('UNCOUPLE: tow hitch released; cart left parked (world-pinned, braked)', flush=True)
        return state

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
