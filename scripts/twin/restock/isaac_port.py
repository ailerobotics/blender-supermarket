"""Measured physics-to-mission port, constructed only inside a guarded Isaac app.

Base/cart rigid-body measurements model ideal proprioception and latch sensors.
They never replace localized navigation or rendered inventory observations.
No method sets a body pose, joint position, payload parent or collision flag.
"""
import math
import time

import numpy as np

from twin.mobile import DockObservation, Pose2, wrap
from twin.navigation.core import Odometry
from twin.navigation.sensors import observation as nav_observation
from twin.urdf_probe import parse_urdf, link_world_transforms, geometry_world_points, resolve_package_uri
from twin.restock_manipulation.isaac_sensors import sensing_render_required
from .runtime import Packet

# Stages whose work consumes freshly rendered pixels (inventory perception, the
# manipulation camera->nvblox depth, the final inventory verification). Only these
# need world.step(render=True) every tick; the navigation-only stages (couple,
# outbound, park, return) drive the base from the PhysX lidar raycast + encoders and
# never read a rendered frame, so they render only on the recorder's cadence ticks.
SENSING_ACTIONS = frozenset({'collect', 'transfer', 'verify'})

# Physics steps to hold the arm frozen at the measured joints after the sim attach-fallback reads
# contact, so the link velocity decays to ~0 before the weld (firstmate run-209; no base-pivot impulse).
ATTACH_SETTLE_STEPS = 15


def _sensing_frame_due(current_action, upcoming_tick, sensing_stride, suspended):
    """Whether this tick must render a fresh SENSING frame: a decode-grade stage on its stride
    cadence, UNLESS the opt-in execution gate has suspended the decode render for the current
    controller phase. While suspended the capture republishes the held registration instead of a
    fresh frame (isaac_sensors.sensing_render_required / PR 80 hold), so no render is forced. Pure
    for CPU testing of the phase->suspend->render-skip wiring; suspended is always False when the
    gate is off, so this is byte-identical to the pre-gate behaviour then."""
    return (not suspended and current_action in SENSING_ACTIONS
            and upcoming_tick % sensing_stride == 0)


def _masked_depth_due(current_action, upcoming_tick, sensing_stride, has_masked_depth):
    """Whether to render + publish masked depth (arm camera -> host nvblox ESDF) this tick. Fires on
    the sensing cadence REGARDLESS of the execution gate's decode suspension: the parked-station
    collision map must stay fresh through the gated reach (run-139 failed 'stale parked-station' when
    the suspension's render-skip also stopped masked depth). The camera lane disables the decode
    render PRODUCT on suspension, so this world.step render stays cheap (arm depth + recorder only,
    not the full-res decode). Pure for CPU testing."""
    return (has_masked_depth and current_action in SENSING_ACTIONS
            and upcoming_tick % sensing_stride == 0)


def body_measurement(body):
    position, quaternion = body.get_world_pose()  # Isaac quaternion is wxyz
    position, quaternion = np.asarray(position,float),np.asarray(quaternion,float)
    linear = np.asarray(body.get_linear_velocity(),float)
    angular = np.asarray(body.get_angular_velocity(),float)
    if (position.shape != (3,) or quaternion.shape != (4,) or
            linear.shape != (3,) or angular.shape != (3,) or
            not all(np.isfinite(v).all() for v in (position,quaternion,linear,angular)) or
            not math.isclose(float(np.linalg.norm(quaternion)),1.,abs_tol=1e-4)):
        raise ValueError('invalid physical body measurement')
    w,x,y,z = quaternion
    yaw = math.atan2(2*(w*z+x*y),1-2*(y*y+z*z))
    return Pose2(float(position[0]),float(position[1]),yaw),float(position[2]),linear,angular


class TransportEnvelope:
    """Current encoder FK against the same collision geometry used offline."""
    def __init__(self, urdf, package_roots, radius):
        self.robot = parse_urdf(str(urdf))
        self.radius = float(radius)
        self.clouds = {}
        for name,link in self.robot.links.items():
            points=[]
            for geometry in link.collisions:
                cloud=geometry_world_points(geometry,np.eye(4),
                    lambda uri:resolve_package_uri(uri,package_roots)[0])
                if cloud is None or not np.isfinite(cloud).all():
                    raise ValueError('unresolved transport collision geometry: '+name)
                points.extend(cloud)
            if points:
                self.clouds[name]=np.asarray(points)
        self.required={joint.name for joint in self.robot.joints
                       if joint.jtype in ('revolute','continuous','prismatic')}

    def contains(self, positions):
        if not self.required <= positions.keys() or not all(
                math.isfinite(positions[name]) for name in self.required):
            return False
        transforms=link_world_transforms(self.robot,positions)
        for name,points in self.clouds.items():
            matrix=transforms[name]
            cloud=points@matrix[:3,:3].T+matrix[:3,3]
            if np.linalg.norm(cloud[:,:2],axis=1).max() > self.radius:
                return False
        return True


class MissionCamera:
    """Keep real pixels/frame IDs while expressing sensor time at mission origin."""
    def __init__(self, camera, clock_origin):
        self.camera,self.clock_origin=camera,float(clock_origin)

    def get_current_frame(self, *args, **kwargs):
        frame=dict(self.camera.get_current_frame(*args,**kwargs))
        if 'rendering_time' in frame:
            frame['rendering_time']=float(frame['rendering_time'])-self.clock_origin
        return frame

    def __getattr__(self,name):
        return getattr(self.camera,name)


class IsaacPort:
    def __init__(self, *, world, robot, robot_body, cart_body, mobile, range_sensor,
                 transport, contact, hold_arm, verification, recording_camera=None,
                 recorder=None, native_transport=None, after_step=None,
                 on_action=None, realtime=True, record_stride=2, sensing_stride=1,
                 masked_depth=None, set_render_suspended=None,
                 stationary_dxy=0.005, stationary_dyaw=0.01, command_gate=None):
        self.world,self.robot=world,robot
        # Opt-in sensing execution gate seam (camera lane). Wired (non-None) only when
        # config['sensing_execution_gate'] is on; gate() then suspends the decode-camera render
        # during a validated-reach execution phase. None => the gate is off and every path below is
        # byte-identical to the pre-gate behaviour.
        self._set_render_suspended=set_render_suspended
        self._sensing_suspended=False
        self.robot_body,self.cart_body=robot_body,cart_body
        self.mobile,self.range_sensor=mobile,range_sensor
        self.transport,self.contact=transport,contact
        self.hold_arm,self.verification=hold_arm,verification
        # ArmCommandGate for the labelled sim attach-fallback's stationary weld (firstmate run-209):
        # when the fallback reads <=0.5mm flush contact, freeze the arm here, settle, then weld.
        self.command_gate=command_gate
        self._attach_freezing=False
        self._attach_wait=0
        self.recording_camera,self.recorder=recording_camera,recorder
        self.native_transport=native_transport
        self.after_step,self.on_action=after_step,on_action
        self.realtime=realtime
        if type(record_stride) is not int or record_stride<1:
            raise ValueError('record_stride must be a positive int (physics ticks per recorded frame)')
        self.record_stride=record_stride
        # Physics ticks per rendered SENSING frame. A sensing stage (collect/transfer/verify) consumes
        # freshly rendered pixels, but the consumers gate on a SIM-time freshness window (controller
        # max_age = 0.5 s; station/nvblox incorporation stamps are sim-clocked), NOT on every physics
        # tick. Rendering the whole scene every tick made collect/grasp run ~1:70 sim:wall (a ~21 s sim
        # grasp -> ~25 min wall, over the ceiling, run-94). Rendering every ``sensing_stride`` ticks keeps
        # the freshest frame <= sensing_stride*dt old (<< 0.5 s for a modest stride) while cutting the
        # dominant per-tick render cost ~stride-fold, so a render-paced grasp completes within the wall
        # budget. Default 1 preserves the legacy every-tick behaviour.
        if type(sensing_stride) is not int or sensing_stride<1:
            raise ValueError('sensing_stride must be a positive int (physics ticks per rendered sensing frame)')
        self.sensing_stride=sensing_stride
        # Optional masked-depth publisher (Isaac-side rig self-pixel invalidation for the
        # host nvblox consumer). Published only on sensing-stage ticks; None keeps the
        # legacy raw-depth OmniGraph path untouched. See masked_depth.MaskedDepthPublisher.
        self.masked_depth=masked_depth
        # Current mission action (set by begin_action); gates per-tick rendering + sensing.
        self.current_action=None
        # Carton-welded latch: True once the sim attach-fallback welds the carton to the arm, so the
        # base brake switches from the pre-weld FULL lock to the yaw-lock-only loop relief. Reset when
        # the mission leaves the arm phase (a fresh grasp starts under the full lock). See advance().
        self._carton_welded=False
        # Last controller phase seen by gate() (phase-aware base brake selection).
        self._phase=None
        # Stage-time value corresponding to mission t=0: world.current_time - now, refreshed every
        # advance so mission_time() recovers the dt-accumulated packet clock (tick*dt) from Isaac's
        # stage time. begin_action('verify') begins the InventoryVerifier on THIS clock -- the same
        # one the packets and freshness windows carry. Initialised here so the verify boundary never
        # raises before the first advance (the former `port.origin` attribute never existed).
        self.origin=0.0
        # Stationary-operation gate: max net base/cart displacement over the 0.25 s window to count as
        # parked. REAL-RIG defaults 5 mm / 10 mrad. The SIM profile relaxes these (firstmate msg 252,
        # sim-only fidelity relaxation, same pattern as the arm-speed profile): the measured plateau was
        # a ~11 mrad chassis-pivot IMPULSE at contact (run-164..166: dxy fixed <5 mm by the wheel hold,
        # dyaw 9.8->11 stiffness-saturated, cart 0.0) -- physically plausible and harmless, not a base
        # rolling away (which these wider values still catch). BASE_DISP still logs every trip.
        self.stationary_dxy=float(stationary_dxy)
        self.stationary_dyaw=float(stationary_dyaw)
        from collections import deque
        self._pose_hist=deque()  # (now, robot (x,y,yaw), cart (x,y,yaw)) for displacement-based parked
        self.wall_origin=time.monotonic()
        self.tick=0
        self.names=tuple(robot.dof_names)
        self.wheels=tuple(self.names.index(n) for n in ('left_wheel_joint','right_wheel_joint'))
        self.sensor_models={'docking':'ideal_simulated_rigid_body_proprioception',
                            'navigation':'wheel_encoders_and_physical_range_scan',
                            'inventory':'rendered_rgb_depth_only'}

    # Controller phases where a WELDED carton rests on a support surface, so the base must NOT be
    # rigidly locked: the world->base->arm->(weld)->carton->support->world loop would be over-
    # constrained and oscillate the arm (run-208/209). 'grasp' = carton on the backstock; 'place'/
    # 'release' = carton on the cart; 'contact' = the descent that fires the flush-contact weld while
    # the carton is still on the backstock. Every OTHER arm phase either has no weld yet (approach/
    # contact pre-weld) or the carton airborne (lift/preplace/retreat) -> no loop -> a full lock is safe.
    _BASE_LOOP_PHASES = frozenset({'grasp', 'place', 'release', 'contact'})

    def _apply_base_brake(self, in_arm_phase, welded, phase):
        """Select the base parking brake for this tick (see advance()):
          * outside the arm phase -> release both (the base is free to drive: couple/tow/park).
          * WELDED carton resting on a support (``phase`` in _BASE_LOOP_PHASES) -> YAW-LOCK-ONLY, so
            translation stays free to relieve the rigid weld loop (the kept run-208/209 behaviour).
          * otherwise (pre-weld grasp press, or a welded carton held AIRBORNE in lift/preplace/retreat)
            -> FULL FixedJoint lock, so the contact-press / arm-reach moment cannot slide the free base
            past the gate (run-232 pre-weld press 15.5 mm; run-233 preplace swing 16.4 mm). Safe: no
            over-constrained loop in these phases.
        Idempotent engage/release; the two base brakes are mutually exclusive."""
        if not in_arm_phase:
            self.mobile.release_base_full_brake()
            self.mobile.release_brake()
        elif welded and phase in self._BASE_LOOP_PHASES:
            self.mobile.release_base_full_brake()
            self.mobile.engage_brake()
        else:
            self.mobile.release_brake()
            self.mobile.engage_base_full_brake()

    def gate(self, phase):
        """Drive the opt-in sensing execution gate from the live manipulation controller phase.
        A no-op unless the camera-lane seam was wired (config['sensing_execution_gate'] on). When
        on: the decode-camera render is SUSPENDED (the capture republishes the held registration,
        PR 80 occlusion hold -- no frame fetch) during a validated-reach execution phase
        (approach/contact/grasp/lift/place/retreat/release/stow/couple, AFTER observe validated the
        registration and the arm occludes the fixed camera), and RESUMED on observe/preplace/verify
        and any unknown phase (fail safe). Called each tick BEFORE advance() with the last known
        phase; a one-tick lag is within the occlusion window the hold already tolerates."""
        self._phase=phase            # also drives the phase-aware base brake (see _apply_base_brake)
        if self._set_render_suspended is None:
            return
        suspended=not sensing_render_required(phase)
        self._sensing_suspended=suspended
        self._set_render_suspended(suspended)

    def _frame_plan(self, upcoming_tick):
        """(sensing_tick, record_tick) for `upcoming_tick`: whether a sensing stage needs a
        fresh render and whether the recorder captures this tick. render = either is True.
        Sensing renders on the ``sensing_stride`` cadence (freshness is sim-time bounded, not
        per-tick), so a render-paced sensing stage advances sim far faster than at every tick.
        While the execution gate has suspended the decode render (``_sensing_suspended``) no sensing
        render is forced -- the capture republishes the held registration instead."""
        sensing=_sensing_frame_due(self.current_action, upcoming_tick, self.sensing_stride,
                                   self._sensing_suspended)
        record=(self.recorder is not None and self.recording_camera is not None
                and upcoming_tick%self.record_stride==0)
        return sensing,record

    def advance(self, dt):
        if not math.isclose(float(self.world.get_physics_dt()),dt,abs_tol=1e-8):
            raise ValueError('mission and physics step mismatch')
        if self.realtime:
            # ROS runs on a separate host interpreter; accelerated simulation can
            # otherwise outrun fresh trajectory commands and map acknowledgements.
            remaining=self.wall_origin+(self.tick+1)*dt-time.monotonic()
            if remaining>0:
                time.sleep(remaining)
        # Render only when this tick produces a frame someone reads: a recorder-cadence
        # frame, or any sensing stage (perception / manip depth / verification). The
        # dominant per-tick cost is the whole-scene render; the navigation-only stages
        # read no pixels, so skipping their renders is the headroom under the wall cap.
        self.sensing_tick,self.record_tick=self._frame_plan(self.tick+1)
        # Masked depth (ESDF) renders on the sensing cadence even while the decode capture is gate-
        # suspended, so the host collision map / parked-station ack stay fresh through the reach.
        self.masked_depth_tick=_masked_depth_due(self.current_action, self.tick+1,
                                                 self.sensing_stride, self.masked_depth is not None)
        self.world.step(render=bool(self.sensing_tick or self.record_tick or self.masked_depth_tick))
        self.tick+=1
        # Labelled sim attach-fallback weld (firstmate run-209/210). On a measured <=0.5mm flush contact,
        # weld directly. No stationary-settle freeze: the SIM PARKING BRAKE (below) rigidly holds the base
        # (run-210 base_dyaw 0.00mrad), so the weld's reaction cannot pivot the chassis -- the freeze the
        # brake superseded also desynced the trajectory_host (it held the arm while the host move kept its
        # clock running, which oscillated the arm on release, run-210). No-op unless the fallback is on.
        if self.contact.sim_attach_contact_ready():
            self.contact.sim_attach_weld()
        # Authoritative live grip signal (native grippedObjects OR the sim attach-fallback weld; a
        # stable bool, keyed on grippedObjects not the flickering status -- cad_tool.read()).
        self._carton_welded = bool(self.contact.read())
        # SIM PARKING BRAKE (firstmate run-210/232): while the arm works (a stationary sensing action),
        # lock the already-parked chassis to the world so the contact-press / welded-carton reaction
        # cannot shift it past the gate (the wheel-hold is translation+yaw-saturated). TWO regimes:
        #  - PRE-WELD grasp press (nothing gripped yet): a FULL FixedJoint (translation + yaw). With
        #    translation free the side-grip press slid the base 15.5 mm AND kept the pad 0.5 mm off the
        #    carton so the grip never sealed (run-232); the full lock is safe here because there is no
        #    weld loop yet (the same full brake held base_dyaw 0.00 mrad pre-weld in run-210/211).
        #  - CARTON HELD, lift/carry (welded to the arm): revert to the YAW-LOCK-ONLY brake, which
        #    leaves translation free to relieve the world->base->arm->weld->carton->support loop (a full
        #    lock there oscillated the arm, run-208/209 -- the kept behaviour). read() is stable (no
        #    status flicker), so the brake does not thrash; engage/release are idempotent.
        # command() releases both before any base drive; release here the moment the mission leaves the
        # arm phase (couple/tow/park).
        self._apply_base_brake(self.current_action in SENSING_ACTIONS, self._carton_welded, self._phase)
        # Cart parking brake (firstmate 287): pin the UNCOUPLED cart to the world while the robot
        # loads a carton onto it during collect, so the placement contact cannot roll it (run-228:
        # cart rolled 15.6 mm > the 15 mm gate). Released the moment collect ends (couple needs the
        # cart free to tow). Only 'collect' loads the cart; other sensing actions do not touch it.
        if self.current_action == 'collect':
            self.mobile.engage_cart_brake()
        else:
            self.mobile.release_cart_brake()
        # Mission-relative acquisition clock on the nominal fixed timestep. Deriving it
        # from world.current_time instead drifts: Isaac's physics dt differs from the
        # mission dt by ~1e-9 per step (both pass the isclose guard above), which left
        # the packet a few nanoseconds ahead of the mission's dt-accumulated clock and
        # tripped the freshness lower bound on the very first tick.
        now=self.tick*dt
        # Host-facing stamp source = Isaac's actual simulation time. The masked depth is stamped at
        # world.current_time, and tf2 never extrapolates into the future, so the relayed /tf,
        # /joint_states and /clock MUST carry world.current_time too (not the dt-accumulated mission
        # clock, which lagged by ~0.05 s and silently dropped every nvblox TF lookup -> no ESDF).
        # The mission's own packet/freshness clock stays `now` (tick*dt); only the host relay uses cur.
        try:
            cur=float(self.world.current_time)
        except Exception:
            cur=now
        # Mission-relative clock origin in stage-time coordinates, refreshed each tick so
        # mission_time() exactly recovers `now` (absorbing the ~1e-9/step physics-vs-mission drift).
        self.origin=cur-now
        if self.tick % 20 == 1:                      # CLOCK_DIAG: dt-accumulated vs Isaac sim time
            print(f'CLOCK_DIAG tick={self.tick} stage_time={now:.4f} world_current_time={cur:.4f} '
                  f'wall={time.monotonic()-self.wall_origin:.2f}', flush=True)
        if self.native_transport is not None:
            self.native_transport.tick(cur)
        joints=np.asarray(self.robot.get_joint_positions(),float)
        if joints.shape != (len(self.names),) or not np.isfinite(joints).all():
            raise ValueError('invalid measured articulation encoders')
        positions=dict(zip(self.names,joints))
        r,rz,rv,rw=body_measurement(self.robot_body)
        c,cz,cv,cw=body_measurement(self.cart_body)
        spec=self.mobile.spec
        frames=spec.manifest()
        docking=DockObservation(r,c,rz+frames['robot_hitch_local'][2],
            cz+frames['cart_hitch_local'][2],float(np.linalg.norm(rv)),float(np.linalg.norm(cv)),
            float(rw[2]),float(cw[2]),age=0.)
        coupled=self.mobile.coupling_state(docking)['healthy']
        # Stationarity from measured net POSE displacement over a short window, NOT the TGS
        # instantaneous velocity: PhysX TGS reports a documented ~14 mm/s velocity bias on a
        # provably-frozen body (the cart on its casters reads ~12-14 mm/s at rest), which the
        # 0.005 m/s instantaneous threshold could never clear -> the base never 'parked' and
        # collect/couple never left the navigate parking gate. Displacement is bias-free.
        hist=self._pose_hist
        hist.append((now,(r.x,r.y,r.yaw),(c.x,c.y,c.yaw)))
        while hist and now-hist[0][0]>0.25:
            hist.popleft()
        def _disp(idx):
            ps=[h[idx] for h in hist]
            dxy=max((math.hypot(p[0]-q[0],p[1]-q[1]) for p in ps for q in ps),default=0.)
            dyaw=max((abs(wrap(p[2]-q[2])) for p in ps for q in ps),default=0.)
            return dxy,dyaw
        bdxy,bdyaw=_disp(1); cdxy,cdyaw=_disp(2)
        windowed=now-hist[0][0]>=0.2
        # The ROBOT BASE gate stays strict always. The CART is excluded from the gate during 'collect'
        # (firstmate 287, margin): the cart is intentionally being LOADED there (a placed carton shifts
        # it slightly even with the cart brake), and it is not yet coupled/towed, so its micro-motion is
        # not a manipulation-safety fault. Once coupled (couple/outbound/park) the cart is rig geometry
        # and its stationarity matters again, so the gate includes it then.
        cart_ok=(self.current_action=='collect'
                 or (cdxy<self.stationary_dxy and cdyaw<self.stationary_dyaw))
        parked=bool(windowed and bdxy<self.stationary_dxy and bdyaw<self.stationary_dyaw and cart_ok)
        # Evidence for tuning the base hold (firstmate): when the 0.25 s window shows the base/cart
        # past the stationarity gate, log how far past, throttled to ~0.2 s.
        if windowed and not parked and now-getattr(self,'_last_bdisp_log',-1.)>=0.2:
            self._last_bdisp_log=now
            print(f'BASE_DISP base_dxy={bdxy*1e3:.1f}mm base_dyaw={bdyaw*1e3:.2f}mrad '
                  f'cart_dxy={cdxy*1e3:.1f}mm cart_dyaw={cdyaw*1e3:.2f}mrad now={now:.2f}',flush=True)
        # Per-tick base-yaw trace through the arm phase (firstmate run-210 option-c evidence): shows
        # whether a weld disturbance is a step or a transient, and proves the brake holds. Bounded to the
        # stationary sensing action; throttled lightly so it does not flood.
        if (windowed and self.current_action in SENSING_ACTIONS
                and now-getattr(self,'_last_byaw_log',-1.)>=0.1):
            self._last_byaw_log=now
            print(f'BASE_YAW_TICK base_dyaw={bdyaw*1e3:.2f}mrad base_dxy={bdxy*1e3:.1f}mm '
                  f'braked={getattr(self.mobile,"_braked",None)} now={now:.2f}',flush=True)
        if self.after_step is not None:
            self.after_step(now,parked)
        ready=self.transport.contains(positions) and not self.contact.read()
        # A towed cart is rig self-geometry only while physically coupled: tell the range sensor
        # the measured latch state so its rear beams hitting the cart become self/unknown (not a
        # phantom obstacle behind the robot) while coupled, and a decoupled cart stays a real
        # obstacle. The sensor unions its coupled_self_paths into the self set when this is True.
        self.range_sensor.coupled=coupled
        # The cart is the work target (not a hazard) while the PARKED robot places a carton onto it in
        # collect: mask it as self so the reachable adjacent cart does not trip obstacle_stop (run-217).
        # Only during the stationary collect action; every other action leaves it a real obstacle.
        self.range_sensor.workspace_self=(self.current_action=='collect')
        scan=self.range_sensor.read(now)
        articulation=wrap(r.yaw-c.yaw)
        # Reference navigation consumes a localized Odometry + an obstacle Observation,
        # not raw wheel encoders and a Scan. The measured base pose is the reference
        # localization (default source 'wheel_odometry_localized'); the Scan is transformed
        # into map-frame obstacles about that pose (source 'simulated_lidar').
        odometry=Odometry((r.x,r.y,r.yaw),now,float(np.linalg.norm(rv)),float(rw[2]))
        navigation=nav_observation(scan,odometry,coupled=coupled,arm_transport=ready,
                                   articulation=articulation)
        if self.recorder is not None:
            # Per-tick nav-coverage trace (JSONL): for every tick, evaluate the uncoupled
            # gate's exact coverage probe (coverage_certifies at the real gate radius
            # footprint.radius(False)+position_uncertainty) plus a bracket, and record the
            # robot pose/heading and the forward-blind bearings. The previous sparse sampler
            # (ticks 1/120/300/600) missed the failing window entirely because collect drives
            # for a while before coverage flips, so the fail tick was never captured. One line
            # per tick localizes exactly when/where insufficient_sensor_coverage first appears.
            try:
                import json as _json, math as _math
                from twin.navigation.core import coverage_certifies, sweep_extent
                heading=odometry.pose[2]
                field=navigation.coverage
                # Gate's measured stopping distance at the worst-case commanded speed and at
                # the turn-in-place command (v=0 -> stop from residual measured speed only).
                sp_drive=max(0.25,float(np.linalg.norm(rv)))
                stop_drive=sp_drive*0.3+sp_drive*sp_drive/(2*0.4)
                sp_turn=float(np.linalg.norm(rv))
                stop_turn=sp_turn*0.3+sp_turn*sp_turn/(2*0.4)
                unc=0.05  # Odometry.position_uncertainty default used in advance()
                rec={}
                for tag,R,stop in (('gate_drive',0.70,stop_drive),('gate_turn',0.70,stop_turn),
                                   ('bracket_0p60',0.60,stop_drive),('bracket_0p90',0.90,stop_drive)):
                    rad=R+unc
                    blind=[round(_math.degrees(wrap(a-heading)),1) for a,reach in field.beams
                           if sweep_extent(wrap(a-heading),rad,stop)>rad+1e-9
                           and reach<sweep_extent(wrap(a-heading),rad,stop)-1e-9]
                    rec[tag]=dict(rad=round(rad,3),stop=round(stop,3),
                                  cert=coverage_certifies(navigation,heading,rad,stop),
                                  nblind=len(blind),blind_deg=blind[:12])
                line=_json.dumps(dict(tick=self.tick,t=round(now,3),
                    wall=round(time.monotonic()-self.wall_origin,3),action=self.current_action,
                    rendered=bool(self.sensing_tick or self.record_tick),
                    x=round(r.x,3),y=round(r.y,3),yaw_deg=round(_math.degrees(heading),1),
                    lin=round(float(np.linalg.norm(rv)),4),coupled=bool(coupled),parked=bool(parked),
                    vel=dict(rv=round(float(np.linalg.norm(rv)),4),rw=round(float(np.linalg.norm(rw)),4),
                             cv=round(float(np.linalg.norm(cv)),4),cw=round(float(np.linalg.norm(cw)),4)),
                    unknown=sum(1 for u in scan.unknown if u),probe=rec))
                with (self.recorder.directory.parent/'nav-trace.jsonl').open('a') as _f:
                    _f.write(line+'\n')
            except Exception as _e:
                try:
                    if self.tick==1:
                        (self.recorder.directory.parent/'nav-trace.err').write_text(repr(_e))
                except Exception: pass
        if self.masked_depth is not None and self.masked_depth_tick:
            # Rig-masked depth (+ camera info) for the host nvblox consumer, published only
            # while a sensing stage needs the manipulation collision world. Off (None) by
            # default; the raw-depth OmniGraph path is unchanged until the host opts in.
            self.masked_depth.publish(now)
        evidence=self.verification(now)
        if self.record_tick:
            frame=self.recording_camera.get_current_frame()
            self.recorder.capture(frame)
        return Packet(f'physics:{self.tick}','sensor',now,
            odometry,navigation,docking,None,
            ready,coupled,articulation,parked,evidence)

    def mission_time(self):
        """Mission-relative acquisition clock (tick*dt) at the current stage time -- the clock the
        packets and freshness windows carry. on_action('verify') begins the InventoryVerifier on
        this so its scan-stamp comparisons use the same clock; `origin` is the stage-time value at
        mission t=0, so world.current_time - origin == the last packet's `now` (no stage drift)."""
        try:
            return float(self.world.current_time)-self.origin
        except Exception:
            return -self.origin

    def begin_action(self,action,job):
        self.current_action=action
        if self.on_action is not None:
            self.on_action(action,job)

    def controls(self, controls):
        for control in controls:
            kind=control.get('kind')
            if kind=='gripper':
                self.contact.command(control['enabled'])
            elif kind=='stop':
                self.stop()
            elif kind=='contact':
                pass   # host-side signal: ros_adapter publishes /arm_contact to trajectory_host; Isaac ignores
            elif kind!='move':
                raise ValueError('unknown manipulation control')
            # Planned move commands are owned by the motion server, not replayed.

    def stop(self):
        try:
            self.mobile.park()
        finally:
            self.hold_arm()
