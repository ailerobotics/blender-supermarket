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
from .collect_capture_gate import release_after_lift

# Stages whose work consumes freshly rendered pixels (inventory perception, the
# manipulation camera->nvblox depth, the final inventory verification). Only these
# need world.step(render=True) every tick; the navigation-only stages (couple,
# outbound, park, return) drive the base from the PhysX lidar raycast + encoders and
# never read a rendered frame, so they render only on the recorder's cadence ticks.
SENSING_ACTIONS = frozenset({'collect', 'transfer', 'verify'})

# Physics steps to hold the arm frozen at the measured joints after the sim attach-fallback reads
# contact, so the link velocity decays to ~0 before the weld (firstmate run-209; no base-pivot impulse).
ATTACH_SETTLE_STEPS = 15


def _sensing_frame_due(current_action, upcoming_tick, sensing_stride, suspended, force_stride1=False):
    """Whether this tick must render a fresh SENSING frame: a decode-grade stage on its stride
    cadence, UNLESS the opt-in execution gate has suspended the decode render for the current
    controller phase. While suspended the capture republishes the held registration instead of a
    fresh frame (isaac_sensors.sensing_render_required / PR 80 hold), so no render is forced. Pure
    for CPU testing of the phase->suspend->render-skip wiring; suspended is always False when the
    gate is off, so this is byte-identical to the pre-gate behaviour then.

    fm452 (COLD): ``force_stride1`` (default False => byte-identical) overrides ``sensing_stride`` with
    an effective stride of 1 for this tick, so the source renders EVERY tick during the short
    collect-observe force-render window (see ``_collect_observe_force_due``)."""
    if suspended or current_action not in SENSING_ACTIONS:
        return False
    stride = 1 if force_stride1 else sensing_stride
    return upcoming_tick % stride == 0


def _collect_observe_force_due(current_action, phase, forced_count, force_ticks):
    """Whether the collect-observe FORCE-RENDER window applies this tick (fm452 COLD): for the first
    ``force_ticks`` ticks of the collect 'observe' stage, render the source EVERY tick (stride 1)
    instead of every ``sensing_stride``, so the source camera's first fresh synchronized frame lands
    INSIDE require_collect_source_evidence's bounded wait rather than racing the GPU-load-dependent
    async first RTX render -- which at collect-observe ENTRY is stale (now-stamp >> the 0.5 s freshness
    budget) and completes asynchronously over several ticks -> 'no new synchronized rendered camera
    frame' (isaac_sensors.py:315), flaky run-to-run (diagnosed commit d25f0c2). Scoped to
    current_action=='collect' AND the observe phase ONLY -- the reach phases suspend the render and
    transfer/verify never force. ``force_ticks==0`` (default) => always False => byte-identical stride
    cadence. Does NOT widen the freshness window (that would certify stale geometry -- forbidden); it
    only renders MORE OFTEN at the collect entry so a fresh frame is guaranteed quickly. Pure for CPU
    testing."""
    return (force_ticks > 0 and current_action == 'collect' and phase == 'observe'
            and forced_count < force_ticks)


def _render_suspended(phase, current_action, collect_keep_source_render):
    """Whether gate() suspends the decode render this phase. Normally the inverse of
    ``sensing_render_required(phase)`` (suspend on the validated-reach execution phases). fm446 (D):
    when ``collect_keep_source_render`` is set AND the current stage is 'collect', NEVER suspend — the
    source decode render stays alive through the collect occlusion window so the Option-B-seeded hold()
    gets a fresh synchronized frame (the sensor gate needs a new monotonic frame each capture). Scoped to
    collect: transfer/couple still suspend as before. False (default) => byte-identical. Pure for CPU test."""
    suspended = not sensing_render_required(phase)
    if suspended and collect_keep_source_render and current_action == 'collect':
        return False
    return suspended


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


def body_roll_pitch_deg(body):
    """Roll, pitch (degrees) of a body from its world quaternion -- the chassis attitude that
    body_measurement() discards (it keeps only yaw). Used to log the caster-lift static tilt
    (fm437: confirm the lifted-caster nose pitch live; a level chassis reads ~0)."""
    _, quaternion = body.get_world_pose()
    w, x, y, z = np.asarray(quaternion, float)
    roll = math.atan2(2*(w*x+y*z), 1-2*(x*x+y*y))
    pitch = math.asin(max(-1.0, min(1.0, 2*(w*y-z*x))))
    return math.degrees(roll), math.degrees(pitch)


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
    # sim-video-speedup-v1 lever handles (set in __init__ from config['sim_speed']); class defaults keep
    # the legacy behaviour for partially-constructed ports (CPU tests build them via __new__).
    _host_pacer = None
    _idle_cameras = None
    _profiler = None

    def __init__(self, *, world, robot, robot_body, cart_body, mobile, range_sensor,
                 transport, contact, hold_arm, verification, recording_camera=None,
                 recorder=None, native_transport=None, after_step=None,
                 on_action=None, realtime=True, record_stride=2, sensing_stride=1,
                 masked_depth=None, set_render_suspended=None,
                 stationary_dxy=0.005, stationary_dyaw=0.01, command_gate=None,
                 position_uncertainty=0.05, phase_drive=None, arm_joints=None,
                 stage_scoped_teardown=None, release_source_capture=None,
                 collect_keep_source_render=False, collect_observe_force_render_ticks=0,
                 trace_carton_path=None, sim_speed=None, idle_cameras=None,
                 vram_monitor=None, progress_writer=None, trace_casters=False):
        self.world,self.robot=world,robot
        # sim-robust-vram-headroom-v1: opt-in VRAM headroom monitor (samples used VRAM on the sim clock
        # and sheds optional load below a floor) and the always-on durable stage-progress writer (so a
        # native crash still leaves a reconstructable verdict). Both None/best-effort => byte-identical.
        self._vram_monitor=vram_monitor
        self._progress_writer=progress_writer
        self._sim_now=0.0
        # sim-video-speedup-v1: opt-in speed levers (twin.restock.sim_speed; config['sim_speed']). None
        # (default) => SimSpeed() == the legacy behaviour on every path below (byte-identical).
        from .sim_speed import SimSpeed, HostStagePacer, IdleCameraPauser, TickProfiler
        self._speed=sim_speed if sim_speed is not None else SimSpeed()
        if self._speed.pacing!='legacy':
            realtime=False            # the legacy absolute real-time pacer would cap a fast run at 1x
        self._host_pacer=(HostStagePacer(self._speed.host_stage_max_ratio)
                          if self._speed.pacing=='host_coupled' else None)
        # Cameras no navigation-only stage reads (the station decode cameras): paused for
        # couple/outbound/park/return, re-enabled when a sensing stage begins (minus the decode cameras
        # that stage does not read, pause_unread_decode); a consumer read resumes a paused one on demand.
        # ``idle_cameras`` is a {name: camera} map or a ready IdleCameraPauser (isaac_bundle builds one).
        self._idle_cameras=None
        if self._speed.pause_idle_cameras and idle_cameras:
            pauser=(idle_cameras if hasattr(idle_cameras,'begin_action')
                    else IdleCameraPauser(idle_cameras))
            self._idle_cameras=pauser.install()
        self._profiler=(TickProfiler(self._speed.tick_profile_s)
                        if self._speed.tick_profile_s else None)
        # Demo-cut evidence (RESTOCK_TRACE_POSES=1 via isaac_bundle): the handling-unit prim whose world
        # pose -- with the cart pose and Isaac stage time -- is appended to each nav-trace row for the
        # offline run checker (twin.demo.run_check). None (default) => the trace row is byte-identical.
        self._trace_carton_path=trace_carton_path
        # fm380: optional callable to DESTROY the collect-only sensing cameras' render products at the
        # collect->couple boundary (stage-scoped sensing). None (default) => never called => byte-
        # identical. isaac_bundle passes it ONLY when RESTOCK_STAGE_SCOPED_SENSING is set.
        self._stage_scoped_teardown=stage_scoped_teardown
        self._stage_scoped_done=False
        # fm384: optional callable to switch the collect capture to destination-only + DESTROY the source
        # render product at grasp-confirm (phase-aware capture). None (default) => never called => byte-
        # identical. isaac_bundle passes it ONLY when RESTOCK_PHASE_AWARE_COLLECT_CAPTURE is set.
        self._release_source_capture=release_source_capture
        self._release_source_done=False
        # fm347/348 (1): phase-dependent arm drive. The soft tuck-settle drive (3e5/8000/400) cannot
        # HOLD the carton-loaded place extend (run-274 shoulder_lift sagged 0.622 rad); the stiff
        # drive (1e6/1e4/2000, bench-s348d proved it extends+holds+settles the real cuMotion carry).
        # gate() applies it IN-PROCESS on the phase it already tracks (self._phase) via the robot it
        # already holds -- no cross-process channel needed. config['phase_drive']={'soft':{kp,kd,eff},
        # 'stiff':{...}}; absent => None => the assemble() drive stays throughout (byte-identical).
        self._phase_drive=phase_drive
        # fm446 (D): keep the collect SOURCE decode render ALIVE through the collect occlusion window
        # (the approach/grasp reach phases) so the Option-B-seeded hold() gets a fresh synchronized frame
        # (the sensor gate needs a new monotonic frame each capture; a paused render fails it). Scoped to
        # current_action=='collect' so transfer/couple still suspend. False (default) => byte-identical.
        self._collect_keep_source_render=bool(collect_keep_source_render)
        # fm452 (COLD): force a stride-1 SOURCE render for the first N ticks of the collect 'observe'
        # stage, so the source camera's first fresh synchronized frame lands inside
        # require_collect_source_evidence's bounded wait instead of racing the GPU-load-dependent async
        # first RTX render (stale at collect-observe entry -> 'no new synchronized rendered camera frame',
        # flaky run-to-run; commit d25f0c2). 0 (default) => OFF => byte-identical stride cadence. Does NOT
        # widen the freshness window (forbidden -- would certify stale geometry); it only renders more
        # often at the collect entry. Counter reset at the collect stage boundary (begin_action).
        force_ticks=int(collect_observe_force_render_ticks)
        if force_ticks < 0:
            raise ValueError('collect_observe_force_render_ticks must be a nonnegative int')
        self._collect_observe_force_render_ticks=force_ticks
        self._collect_observe_forced=0
        self._arm_joints=tuple(arm_joints) if arm_joints else None
        self._last_drive_gains=None
        # Localization uncertainty pad added to the nav footprint radius. The real rig keeps the
        # conservative 0.05 m default; this port feeds IDEAL simulated rigid-body proprioception
        # (sensor_models['localization']), so a sim run may set it to the measured ~0 to avoid an
        # inaccurate pad falsely failing the planner's free(start) check when a trailing obstacle
        # (e.g. the staging support behind the parked robot) sits just outside the true footprint.
        if not math.isfinite(position_uncertainty) or position_uncertainty < 0:
            raise ValueError('position_uncertainty must be finite and nonnegative')
        self._odom_position_uncertainty=float(position_uncertainty)
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
        # sim-robust-throat-momentum-launch-v1 diagnostic (RESTOCK_TRACE_CASTERS=1, default off =>
        # byte-identical: no extra nav-trace fields, no stage reads). The throat pin (the base stops dead
        # at x~=-3.935 on a flat floor with NO collider) is hypothesised to turn on the TRAILING-CASTER
        # swivel state: a caster left angled by the pre-throat alignment turn scrubs instead of rolls from
        # the near-stop. These fields let the next GPU runs SEE the swivel at the pin: the two caster
        # swivel-joint angles (the discriminator the momentum launch addresses) and a geometric ground-
        # contact proxy (each wheel/caster link's world-z vs its rolling radius above the floor z=0).
        self._trace_casters=bool(trace_casters)
        self._caster_swivel_dofs={}
        self._wheel_prims={}
        if self._trace_casters:
            for nm in self.names:
                if nm.endswith('caster_swivel'):
                    self._caster_swivel_dofs[nm]=self.names.index(nm)
            try:
                from pxr import UsdGeom as _UsdGeom
                r_wheel=float(self.mobile.spec.wheel_radius)
                wanted={'left_wheel':r_wheel,'right_wheel':r_wheel,
                        'front_caster_wheel':0.065,'rear_caster_wheel':0.065}
                for prim in self.world.stage.Traverse():
                    nm=prim.GetName()
                    if nm in wanted and nm not in self._wheel_prims and _UsdGeom.Xformable(prim):
                        self._wheel_prims[nm]=(prim,wanted[nm])
            except Exception:
                self._wheel_prims={}
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
        # fm347/348 (1): switch the arm drive to STIFF for the loaded extend/rotate+place (pan/
        # preplace/place) and back to SOFT elsewhere, applied once per transition (no-op when the
        # selected gains are unchanged). Opt-in via config['phase_drive']; a stray phase keeps the
        # current gains (phase_drive_gains returns soft for any non-stiff phase, so the drive can only
        # be soft or the configured stiff -- never undefined).
        if self._phase_drive is not None and self._arm_joints and self.robot is not None:
            from .physics_setup import phase_drive_gains, set_arm_drive_gains
            g=phase_drive_gains(phase, self._phase_drive)
            if g is not None and g!=self._last_drive_gains:
                set_arm_drive_gains(self.robot, self._arm_joints, g['kp'], g['kd'], g['eff'])
                self._last_drive_gains=g
                print(f'PHASE_DRIVE phase={phase} -> kp={g["kp"]:g} kd={g["kd"]:g} eff={g["eff"]:g}',flush=True)
        if self._set_render_suspended is None:
            return
        suspended=_render_suspended(phase, self.current_action, self._collect_keep_source_render)
        self._sensing_suspended=suspended
        self._set_render_suspended(suspended)

    def _frame_plan(self, upcoming_tick):
        """(sensing_tick, record_tick) for `upcoming_tick`: whether a sensing stage needs a
        fresh render and whether the recorder captures this tick. render = either is True.
        Sensing renders on the ``sensing_stride`` cadence (freshness is sim-time bounded, not
        per-tick), so a render-paced sensing stage advances sim far faster than at every tick.
        While the execution gate has suspended the decode render (``_sensing_suspended``) no sensing
        render is forced -- the capture republishes the held registration instead.
        fm452 (COLD): for the first N ticks of the collect 'observe' stage the source is rendered EVERY
        tick (stride 1) so a fresh synchronized frame is guaranteed quickly at the collect entry (see
        ``_collect_observe_force_due``); N=0 (default) => byte-identical stride cadence."""
        force=_collect_observe_force_due(self.current_action, self._phase,
                                         self._collect_observe_forced,
                                         self._collect_observe_force_render_ticks)
        if force:
            self._collect_observe_forced+=1
        sensing=_sensing_frame_due(self.current_action, upcoming_tick, self.sensing_stride,
                                   self._sensing_suspended, force)
        record=(self.recorder is not None and self.recording_camera is not None
                and upcoming_tick%self.record_stride==0)
        return sensing,record

    def _mark(self, name):
        if self._profiler is not None:
            self._profiler.mark(name)

    def _trace_probe(self):
        speed = getattr(self, '_speed', None)
        return True if speed is None else speed.trace_probe

    def advance(self, dt):
        if self._profiler is not None:
            self._profiler.begin()
        if not math.isclose(float(self.world.get_physics_dt()),dt,abs_tol=1e-8):
            raise ValueError('mission and physics step mismatch')
        if self.realtime:
            # ROS runs on a separate host interpreter; accelerated simulation can
            # otherwise outrun fresh trajectory commands and map acknowledgements.
            remaining=self.wall_origin+(self.tick+1)*dt-time.monotonic()
            if remaining>0:
                time.sleep(remaining)
        if self._host_pacer is not None:
            # sim_speed pacing='host_coupled': no pacing in the in-process stages; a sliding real-time
            # cap (host_stage_max_ratio) only while the host stack is in the loop (collect/transfer/verify).
            self._host_pacer.wait(self.current_action, dt)
        self._mark('pace')
        # Render only when this tick produces a frame someone reads: a recorder-cadence
        # frame, or any sensing stage (perception / manip depth / verification). The
        # dominant per-tick cost is the whole-scene render; the navigation-only stages
        # read no pixels, so skipping their renders is the headroom under the wall cap.
        self.sensing_tick,self.record_tick=self._frame_plan(self.tick+1)
        # Masked depth (ESDF) renders on the sensing cadence even while the decode capture is gate-
        # suspended, so the host collision map / parked-station ack stay fresh through the reach.
        self.masked_depth_tick=_masked_depth_due(self.current_action, self.tick+1,
                                                 self.sensing_stride, self.masked_depth is not None)
        _render=bool(self.sensing_tick or self.record_tick or self.masked_depth_tick)
        self.world.step(render=_render)
        self._mark('step_render' if _render else 'step')
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
        # fm381/384 render-product teardown, fm394 TIMING: free the collect-only source render product
        # (stage-scoped teardown + phase-aware dest-only switch) to return its VRAM BEFORE the release peak
        # (sim ~223) -- but only AFTER the carton LIFT has COMPLETED (a post-lift phase), never during the
        # lift. Firing it at grasp-confirm tore the sensing/render loop down mid-lift and the lift
        # goal-missed 2/2 (runs 300d, 305); run-300c lifted cleanly without it. release_after_lift() holds
        # the teardown through observe/approach/contact/grasp/lift and fires at the first post-lift tick
        # (tuck for the staged/reposition flow, else preplace) -- still well before the release peak. None
        # handles => off (byte-identical). begin_action (collect->couple) keeps both calls as a fallback.
        if release_after_lift(self._phase, self._carton_welded):
            self._fire_stage_scoped_teardown()
            self._fire_release_source()
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
        self._mark('post_step')
        # Mission-relative acquisition clock on the nominal fixed timestep. Deriving it
        # from world.current_time instead drifts: Isaac's physics dt differs from the
        # mission dt by ~1e-9 per step (both pass the isclose guard above), which left
        # the packet a few nanoseconds ahead of the mission's dt-accumulated clock and
        # tripped the freshness lower bound on the very first tick.
        now=self.tick*dt
        self._sim_now=now
        # sim-robust-vram-headroom-v1: sample VRAM on the sim clock and shed optional load below the
        # floor (sub-second cadence so a transient spike between the 20 s orchestrator samples is seen).
        # Cheap: it reads the sampler's latest in-memory value. None (default) => byte-identical.
        monitor=getattr(self,'_vram_monitor',None)
        if monitor is not None:
            try:
                monitor.tick(now)
            except Exception as _exc:  # noqa: BLE001 -- monitoring must never break the tick
                print('VRAM_MONITOR_TICK_ERROR %r' % (_exc,), flush=True)
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
        self._mark('host_bridge')
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
        # fm437: log the base chassis attitude (roll/pitch) ~1 Hz in every action, so the caster-lift
        # static tilt is measured live (a level chassis reads ~0; the 5 mm lift predicts ~0.75 deg pitch).
        if now-getattr(self,'_last_batt_log',-1.)>=1.0:
            self._last_batt_log=now
            broll,bpitch=body_roll_pitch_deg(self.robot_body)
            print(f'BASE_ATT roll_deg={broll:.3f} pitch_deg={bpitch:.3f} z_m={rz:.4f} '
                  f'action={self.current_action} now={now:.2f}',flush=True)
        self._mark('measure')
        if self.after_step is not None:
            self.after_step(now,parked)
            self._mark('after_step')
        # fm390: during the mid-collect reposition carry (the controller holds the carry fold in TRANSIT
        # while the BASE drives to place_approach), the gripper is DELIBERATELY holding the carton, so the
        # empty-gripper requirement must not block the drive (run-302 failed 'arm_not_in_transport_timeout'
        # here -- the base refused to move because contact was held). The arm is still gated on the transport
        # ENVELOPE (contains), which the compact carry fold satisfies (CPU-verified 0.62 m <= 0.65 m radius);
        # only the held-payload check is relaxed, and ONLY in TRANSIT -- a phase that exists solely in
        # reposition mode, so an absent reposition leaves this byte-identical.
        carrying_reposition = (self._phase == 'transit')
        ready=self.transport.contains(positions) and (carrying_reposition or not self.contact.read())
        self._mark('envelope')
        # A towed cart is rig self-geometry only while physically coupled: tell the range sensor
        # the measured latch state so its rear beams hitting the cart become self/unknown (not a
        # phantom obstacle behind the robot) while coupled, and a decoupled cart stays a real
        # obstacle. The sensor unions its coupled_self_paths into the self set when this is True.
        self.range_sensor.coupled=coupled
        # The cart is the work target (not a hazard), not an obstacle, in the two actions whose whole
        # purpose is to reach it:
        #   * collect -- the PARKED robot places a carton onto the adjacent cart (run-217: the reachable
        #     cart inside the base footprint disk would otherwise trip obstacle_stop), and
        #   * couple  -- the robot drives UP TO the (still uncoupled) cart to seat its hitch; an unmasked
        #     cart reads as a hard obstacle sitting on the dock goal, so the planner finds no route to
        #     cart_dock and couple fails 'blocked_no_route' before it can ever attach (ds-001).
        # workspace_self masks the cart WITHOUT asserting `coupled`, so the footprint stays the uncoupled
        # base radius (no swept-train envelope) and no coupling is implied. Every OTHER action (outbound/
        # park/transfer/verify/return) leaves the cart a real obstacle (when coupled it is coupled-self).
        self.range_sensor.workspace_self=(self.current_action in ('collect','couple'))
        # fm393: mask the carry-fold arm/carton near-field ring from the lidar ONLY during the mid-collect
        # reposition carry (the controller holds TRANSIT while the base drives to place_approach). Precise
        # signal: self._phase=='transit'. Inert unless carry_self_radius_m is configured => byte-identical.
        self.range_sensor.carry_active=(self._phase=='transit')
        scan=self.range_sensor.read(now)
        self._mark('lidar')
        articulation=wrap(r.yaw-c.yaw)
        # Reference navigation consumes a localized Odometry + an obstacle Observation,
        # not raw wheel encoders and a Scan. The measured base pose is the reference
        # localization (default source 'wheel_odometry_localized'); the Scan is transformed
        # into map-frame obstacles about that pose (source 'simulated_lidar').
        odometry=Odometry((r.x,r.y,r.yaw),now,float(np.linalg.norm(rv)),float(rw[2]),
                          position_uncertainty=self._odom_position_uncertainty)
        navigation=nav_observation(scan,odometry,coupled=coupled,arm_transport=ready,
                                   articulation=articulation)
        self._mark('observation')
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
                # sim_speed trace_probe=False: keep the row, drop the 4-radius coverage probe (a pure
                # diagnostic no consumer reads, evaluated per beam in Python every tick).
                for tag,R,stop in ((('gate_drive',0.70,stop_drive),('gate_turn',0.70,stop_turn),
                                    ('bracket_0p60',0.60,stop_drive),('bracket_0p90',0.90,stop_drive))
                                   if self._trace_probe() else ()):
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
                    unknown=sum(1 for u in scan.unknown if u),
                    **({'probe':rec} if self._trace_probe() else {}),
                    **({'casters':self._caster_trace(positions)} if self._trace_casters else {})))
                if self._trace_carton_path is not None:
                    line=line[:-1]+', '+_json.dumps(self._trace_poses(cur,c,cz))[1:]
                with (self.recorder.directory.parent/'nav-trace.jsonl').open('a') as _f:
                    _f.write(line+'\n')
            except Exception as _e:
                try:
                    if self.tick==1:
                        (self.recorder.directory.parent/'nav-trace.err').write_text(repr(_e))
                except Exception: pass
        self._mark('trace')
        if self.masked_depth is not None and self.masked_depth_tick:
            # Rig-masked depth (+ camera info) for the host nvblox consumer, published only
            # while a sensing stage needs the manipulation collision world. Off (None) by
            # default; the raw-depth OmniGraph path is unchanged until the host opts in.
            self.masked_depth.publish(now)
            self._mark('mast_depth')
        evidence=self.verification(now)
        self._mark('verification')
        if self.record_tick:
            frame=self.recording_camera.get_current_frame()
            self.recorder.capture(frame)
            self._mark('record')
        if self._profiler is not None:
            self._profiler.end(now, _render, self.current_action)
        return Packet(f'physics:{self.tick}','sensor',now,
            odometry,navigation,docking,None,
            ready,coupled,articulation,parked,evidence)

    def _caster_trace(self, positions):
        """Caster-swivel + wheel-contact diagnostic for the nav-trace row (RESTOCK_TRACE_CASTERS=1).

        sim-robust-throat-momentum-launch-v1: the throat pin (base stops dead at x~=-3.935 on a flat floor
        with NO collider) is hypothesised to turn on the TRAILING-CASTER swivel state. Emits:
          * ``swivel_deg``: each passive caster swivel-joint angle (deg) from the measured encoder dict --
            the discriminator the momentum launch addresses (a caster angled by the pre-throat alignment
            turn scrubs instead of rolls from the near-stop);
          * ``wheel_z`` / ``contact``: each drive + caster wheel link's world-z (m) and a GEOMETRIC contact
            proxy (z <= rolling_radius + 10 mm above the floor z=0). A proxy, NOT a PhysX contact report --
            labelled as such; it still shows which wheel has lifted off when the base pitches.
        Never raises (a read failure drops the offending field); costs one encoder-dict read + up to four
        cached-prim world transforms per tick when enabled, and is absent entirely when off."""
        out={}
        try:
            sw={nm.replace('_caster_swivel',''):round(math.degrees(float(positions[nm])),1)
                for nm in self._caster_swivel_dofs if nm in positions}
            if sw:
                out['swivel_deg']=sw
        except Exception:
            pass
        if self._wheel_prims:
            try:
                from pxr import UsdGeom
                cache=UsdGeom.XformCache()
                wz={}; contact={}
                for nm,(prim,rad) in self._wheel_prims.items():
                    if not prim.IsValid():
                        continue
                    z=float(cache.GetLocalToWorldTransform(prim).ExtractTranslation()[2])
                    key=nm.replace('_wheel','').replace('_caster','')
                    wz[key]=round(z,4)
                    contact[key]=bool(z<=rad+0.01)
                if wz:
                    out['wheel_z']=wz
                    out['contact']=contact
            except Exception:
                pass
        return out

    def _trace_poses(self, cur, cart, cart_z):
        """Carton + cart world poses for the nav-trace row (RESTOCK_TRACE_POSES=1). The carton is read
        from the live handling-unit prim (the same stage read isaac_factory's place gate uses); a
        failed read records carton=None rather than a stale value. Never raises."""
        out={'world_t':round(float(cur),4),
             'cart':[round(cart.x,4),round(cart.y,4),round(math.degrees(cart.yaw),2),round(float(cart_z),4)],
             'carton':None}
        try:
            from pxr import UsdGeom
            prim=self.world.stage.GetPrimAtPath(self._trace_carton_path)
            if prim and prim.IsValid():
                m=UsdGeom.XformCache().GetLocalToWorldTransform(prim)
                p=m.ExtractTranslation()
                out['carton']=[round(float(p[0]),4),round(float(p[1]),4),round(float(p[2]),4),
                               round(math.degrees(math.atan2(float(m[0][1]),float(m[0][0]))),2)]
        except Exception:
            pass
        return out

    def mission_time(self):
        """Mission-relative acquisition clock (tick*dt) at the current stage time -- the clock the
        packets and freshness windows carry. on_action('verify') begins the InventoryVerifier on
        this so its scan-stamp comparisons use the same clock; `origin` is the stage-time value at
        mission t=0, so world.current_time - origin == the last packet's `now` (no stage drift)."""
        try:
            return float(self.world.current_time)-self.origin
        except Exception:
            return -self.origin

    def _fire_stage_scoped_teardown(self):
        """fm380/fm381: free the collect-only source(+shelf) station camera render products ONCE.
        Guarded + fail-safe; no-op when there is no handle or it already fired => byte-identical when
        the RESTOCK_STAGE_SCOPED_SENSING knob is off (isaac_bundle passes None then). Called at the
        earliest safe point (grasp confirmed, before lift) and, as a fallback, at the collect->couple
        boundary in begin_action -- whichever fires first wins (the flag makes it idempotent)."""
        if self._stage_scoped_teardown is None or self._stage_scoped_done:
            return
        self._stage_scoped_done = True
        try:
            self._stage_scoped_teardown()
        except Exception as exc:                          # noqa: BLE001
            print('STAGE_SCOPED_SENSING teardown skipped: %r' % (exc,), flush=True)

    def _fire_release_source(self):
        """fm384: switch the collect capture to destination-only + DESTROY the source render product
        ONCE at grasp-confirm (phase-aware capture). Guarded + fail-safe; no-op with no handle or once
        done => byte-identical when RESTOCK_PHASE_AWARE_COLLECT_CAPTURE is off (isaac_bundle passes None).
        Honest: the capture keeps a REAL fresh destination(cart) render after this; nothing fabricated."""
        if self._release_source_capture is None or self._release_source_done:
            return
        self._release_source_done = True
        try:
            self._release_source_capture()
        except Exception as exc:                          # noqa: BLE001
            print('PHASE_AWARE_COLLECT release skipped: %r' % (exc,), flush=True)

    def begin_action(self,action,job):
        # fm380/fm381 stage-scoped sensing FALLBACK: the primary trigger is grasp-confirmed (before
        # lift, in the step path). If a run somehow never welds, the FIRST transition into a non-sensing
        # action (collect is a SENSING_ACTION; couple/outbound/park are not) still frees the cameras.
        if action not in SENSING_ACTIONS:
            self._fire_stage_scoped_teardown()
        # fm452 (COLD): reset the collect-observe force-render budget when the collect stage begins, so
        # the force-stride-1 window opens afresh at this collect's observe entry. No-op when the knob is
        # off (budget stays 0) => byte-identical.
        if action == 'collect':
            self._collect_observe_forced=0
        profiler=getattr(self,'_profiler',None)
        if profiler is not None:
            profiler.flush()              # sim_speed tick_profile_s: one TICK_PROFILE line per stage window
        self.current_action=action
        # sim-robust-vram-headroom-v1: tell the headroom monitor which stage is running (for the
        # stage-safety gate) and durably record the stage transition (crash-safe recording). Both are
        # best-effort and None by default => byte-identical.
        monitor=getattr(self,'_vram_monitor',None)
        if monitor is not None:
            try:
                monitor.begin_stage(action)
            except Exception:  # noqa: BLE001 -- monitoring must never break a stage transition
                pass
        writer=getattr(self,'_progress_writer',None)
        if writer is not None:
            writer.begin(action, getattr(self,'_sim_now',0.0))
        idle=getattr(self,'_idle_cameras',None)
        if idle is not None:
            # sim_speed pause_idle_cameras: couple/outbound/park/return read no station camera, so stop
            # the decode render products there; a sensing stage re-enables the ones it reads.
            idle.begin_action(action)
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
