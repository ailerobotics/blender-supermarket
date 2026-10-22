from pathlib import Path
import sys
from types import SimpleNamespace as NS

import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from twin.restock.isaac_port import (body_measurement, MissionCamera, IsaacPort, SENSING_ACTIONS,
                                      _sensing_frame_due, _collect_observe_force_due)
from twin.restock.native_transport import ArmCommandGate
from twin.restock_manipulation.cad_tool import ARM_JOINTS


def test_cart_solver_tuning_preserves_physical_bodies_and_free_joints():
    Usd=pytest.importorskip('pxr.Usd')
    from pxr import UsdPhysics
    from twin.mobile_assets import author_cart
    from twin.restock.physics_setup import configure_cart_solver
    stage=Usd.Stage.CreateInMemory()
    cart=author_cart(stage)
    before={(str(p.GetPath()),a.GetName()):a.Get() for p in stage.Traverse()
            for a in p.GetAttributes() if a.HasAuthoredValueOpinion()}
    joints={str(p.GetPath()):list(UsdPhysics.Joint(p).GetBody1Rel().GetTargets())
            for p in stage.Traverse() if p.IsA(UsdPhysics.Joint)}
    result=configure_cart_solver(stage,cart['root'])
    assert len(result)==9
    for (path,name),value in before.items():
        assert stage.GetPrimAtPath(path).GetAttribute(name).Get()==value
    for path,targets in joints.items():
        joint=UsdPhysics.Joint(stage.GetPrimAtPath(path))
        assert list(joint.GetBody1Rel().GetTargets())==targets and joint.GetJointEnabledAttr().Get()
    assert not any(p.HasAPI(UsdPhysics.RigidBodyAPI) and UsdPhysics.RigidBodyAPI(p).GetKinematicEnabledAttr().Get()
                   for p in stage.Traverse())


def test_cart_bearings_add_damping_without_touching_physical_model():
    Usd=pytest.importorskip('pxr.Usd')
    from twin.mobile_assets import author_cart
    from twin.restock.physics_setup import configure_cart_bearings, LINEAR_DAMPING, ANGULAR_DAMPING
    stage=Usd.Stage.CreateInMemory()
    cart=author_cart(stage)
    # Snapshot everything the fidelity correction must NOT change: mass, inertia,
    # collision geometry, joint bodies/axes and the free (undriven) revolute model.
    protected={(str(p.GetPath()),a.GetName()):a.Get() for p in stage.Traverse()
               for a in p.GetAttributes() if a.HasAuthoredValueOpinion()}
    result=configure_cart_bearings(stage,cart['root'])
    assert len(result['joints'])==8 and len(result['bodies'])==9
    for (path,name),value in protected.items():
        assert stage.GetPrimAtPath(path).GetAttribute(name).Get()==value
    # Casters stay undriven maximal joints (jointFriction is inert there, so unset).
    for path in result['joints']:
        j=stage.GetPrimAtPath(path)
        assert not j.GetAttribute('physxJoint:jointFriction').HasAuthoredValueOpinion()
        assert j.GetAttribute('physics:excludeFromArticulation').Get() is True
    # Bodies carry bearing/rolling resistance as plain damping (no drive).
    for path in result['bodies']:
        b=stage.GetPrimAtPath(path)
        assert b.GetAttribute('physxRigidBody:linearDamping').Get()==pytest.approx(LINEAR_DAMPING)
        assert b.GetAttribute('physxRigidBody:angularDamping').Get()==pytest.approx(ANGULAR_DAMPING)


def test_wheel_traction_binds_drive_wheel_colliders_despite_instance_proxy():
    """Reproduce the Isaac URDF-import layout that bound 0 colliders: the drive wheels are
    siblings of the articulation-root base_link, and each wheel's collider is referenced under
    <wheel>/collisions with instanceable=True (an instance proxy a naive PrimRange skips and
    cannot bind). The fixed lookup must bind exactly the two drive-wheel colliders."""
    Usd=pytest.importorskip('pxr.Usd')
    from pxr import UsdGeom, UsdPhysics, UsdShade
    from twin.restock.physics_setup import configure_wheel_traction
    stage=Usd.Stage.CreateInMemory()
    # Collider source geometry the importer references per link.
    src=UsdGeom.Xform.Define(stage,'/_colliders/wheel')
    cyl=UsdGeom.Cylinder.Define(stage,'/_colliders/wheel/mesh_0/cylinder')
    UsdPhysics.CollisionAPI.Apply(cyl.GetPrim())
    robot=UsdGeom.Xform.Define(stage,'/MobileRobot').GetPrim()
    base=UsdGeom.Xform.Define(stage,'/MobileRobot/base_link').GetPrim()
    UsdPhysics.ArticulationRootAPI.Apply(base)
    # Drive wheels and a non-drive caster, all siblings of base_link under the robot root.
    for name in ('left_wheel','right_wheel','rear_caster_wheel'):
        wheel=UsdGeom.Xform.Define(stage,f'/MobileRobot/{name}').GetPrim()
        UsdPhysics.RigidBodyAPI.Apply(wheel)
        coll=UsdGeom.Xform.Define(stage,f'/MobileRobot/{name}/collisions').GetPrim()
        coll.GetReferences().AddInternalReference('/_colliders/wheel')
        coll.SetInstanceable(True)
        assert coll.IsInstanceable()
    bound=configure_wheel_traction(stage,'/MobileRobot/base_link',1.0)
    # Exactly the two drive-wheel colliders, not the caster.
    assert sorted(bound)==[
        '/MobileRobot/left_wheel/collisions/mesh_0/cylinder',
        '/MobileRobot/right_wheel/collisions/mesh_0/cylinder',
    ]
    mat=stage.GetPrimAtPath('/RestockWheelMaterial')
    api=UsdPhysics.MaterialAPI(mat)
    assert api.GetStaticFrictionAttr().Get()==pytest.approx(1.0)
    assert api.GetDynamicFrictionAttr().Get()==pytest.approx(0.9)
    for path in bound:
        rel=UsdShade.MaterialBindingAPI(stage.GetPrimAtPath(path)).GetDirectBindingRel(materialPurpose='physics')
        assert [str(t) for t in rel.GetTargets()]==['/RestockWheelMaterial']


def test_scene_physics_enables_stabilization_only():
    Usd=pytest.importorskip('pxr.Usd')
    from pxr import UsdPhysics, Gf
    from twin.restock.physics_setup import configure_scene_physics
    stage=Usd.Stage.CreateInMemory()
    scene=UsdPhysics.Scene.Define(stage,'/World/physicsScene')
    scene.CreateGravityDirectionAttr(Gf.Vec3f(0,0,-1))
    scene.CreateGravityMagnitudeAttr(9.81)
    result=configure_scene_physics(stage,'/World/physicsScene')
    assert result['enable_stabilization'] is True
    prim=stage.GetPrimAtPath('/World/physicsScene')
    assert prim.GetAttribute('physxScene:enableStabilization').Get() is True
    # Gravity (the physical model) is untouched.
    assert list(prim.GetAttribute('physics:gravityDirection').Get())==[0,0,-1]
    assert prim.GetAttribute('physics:gravityMagnitude').Get()==pytest.approx(9.81)


def test_body_measurement_uses_isaac_wxyz_and_rejects_nonfinite_feedback():
    body=NS(get_world_pose=lambda:(np.array([2,3,.24]),np.array([2**-.5,0,0,2**-.5])),
            get_linear_velocity=lambda:np.zeros(3),get_angular_velocity=lambda:np.zeros(3))
    pose,z,_,_=body_measurement(body)
    assert pose.yaw==pytest.approx(np.pi/2) and z==.24
    body.get_linear_velocity=lambda:np.array([np.nan,0,0])
    with pytest.raises(ValueError,match='measurement'):
        body_measurement(body)


def test_mission_camera_preserves_pixels_id_and_original_frame_clock():
    frame=dict(rendering_time=8.5,rendering_frame=23,rgb=np.zeros((2,2,3)))
    camera=MissionCamera(NS(get_current_frame=lambda:frame),8.)
    measured=camera.get_current_frame()
    assert measured['rendering_time']==.5 and measured['rendering_frame']==23
    assert measured['rgb'] is frame['rgb'] and frame['rendering_time']==8.5


def _plan_port(*, action, stride=2, sensing_stride=1, recorder=True, camera=True,
               phase=None, force_render_ticks=0):
    port = IsaacPort.__new__(IsaacPort)   # skip Isaac wiring; only the render-gating state
    port.current_action = action
    port.record_stride = stride
    port.sensing_stride = sensing_stride
    port.recorder = object() if recorder else None
    port.recording_camera = object() if camera else None
    port._sensing_suspended = False       # execution gate off (default); render-gating byte-identical
    port._phase = phase                   # last controller phase (fm452 collect-observe force window)
    port._collect_observe_force_render_ticks = force_render_ticks
    port._collect_observe_forced = 0
    return port


def test_navigation_stage_renders_only_on_recorder_cadence():
    port = _plan_port(action='couple', stride=3)   # couple is navigation-only
    plans = [port._frame_plan(t) for t in range(1, 7)]
    sensing = [s for s, _ in plans]
    record = [r for _, r in plans]
    assert sensing == [False] * 6                   # no sensing render in a nav stage
    assert record == [False, False, True, False, False, True]   # only every 3rd tick


def test_sensing_stage_renders_every_tick():
    for action in sorted(SENSING_ACTIONS):
        port = _plan_port(action=action, stride=100)   # default sensing_stride=1
        assert all(port._frame_plan(t)[0] for t in range(1, 5))   # sensing True every tick


def test_sensing_stage_renders_on_sensing_stride():
    # With a sensing stride, a sensing stage renders only every Nth tick (freshness is sim-time
    # bounded, not per-tick): the render-paced grasp advances sim far faster than at every tick.
    for action in sorted(SENSING_ACTIONS):
        port = _plan_port(action=action, stride=100, sensing_stride=4)
        sensing = [port._frame_plan(t)[0] for t in range(1, 9)]
        assert sensing == [False, False, False, True, False, False, False, True]


def test_collect_observe_force_render_off_is_byte_identical_stride():
    # fm452 (COLD): force_render_ticks=0 (default) => the collect 'observe' stage keeps the plain
    # stride-6 cadence, even in the observe phase. Byte-identical to the pre-fm452 behaviour.
    port = _plan_port(action='collect', stride=100, sensing_stride=6, phase='observe',
                      force_render_ticks=0)
    sensing = [port._frame_plan(t)[0] for t in range(1, 13)]
    assert sensing == [t % 6 == 0 for t in range(1, 13)]   # stride-6: True only on 6, 12
    assert port._collect_observe_forced == 0               # never forced


def test_collect_observe_force_render_forces_stride1_for_first_n_observe_ticks():
    # fm452 (COLD): with N>0 the source renders EVERY tick for the first N ticks of collect observe,
    # then falls back to stride-6. Simulate consecutive observe ticks so the force budget decrements.
    N = 4
    port = _plan_port(action='collect', stride=100, sensing_stride=6, phase='observe',
                      force_render_ticks=N)
    sensing = [port._frame_plan(t)[0] for t in range(1, 13)]
    # First N ticks forced (stride-1 => every tick True); then back to stride-6 (True at 6, 12).
    assert sensing[:N] == [True] * N
    assert sensing[N:] == [t % 6 == 0 for t in range(N + 1, 13)]
    assert port._collect_observe_forced == N               # exactly N ticks consumed the budget


def test_collect_observe_force_render_not_in_reach_or_other_stages():
    # fm452 (COLD): the force window is scoped to current_action=='collect' AND the observe phase.
    # Pure predicate: a reach phase, transfer/verify, or an off-knob all decline to force.
    assert _collect_observe_force_due('collect', 'observe', 0, 4) is True
    assert _collect_observe_force_due('collect', 'grasp', 0, 4) is False     # reach phase
    assert _collect_observe_force_due('collect', 'approach', 0, 4) is False  # reach phase
    assert _collect_observe_force_due('transfer', 'observe', 0, 4) is False  # other sensing stage
    assert _collect_observe_force_due('verify', 'observe', 0, 4) is False
    assert _collect_observe_force_due('collect', 'observe', 4, 4) is False   # budget exhausted
    assert _collect_observe_force_due('collect', 'observe', 0, 0) is False   # knob off
    # And through _frame_plan: a reach phase in collect keeps the plain stride-6 cadence (reach
    # suspends in the real gate anyway, but even unsuspended it is never force-rendered here).
    reach = _plan_port(action='collect', stride=100, sensing_stride=6, phase='grasp',
                       force_render_ticks=4)
    assert [reach._frame_plan(t)[0] for t in range(1, 13)] == [t % 6 == 0 for t in range(1, 13)]
    assert reach._collect_observe_forced == 0
    # A transfer observe never forces either.
    xfer = _plan_port(action='transfer', stride=100, sensing_stride=6, phase='observe',
                      force_render_ticks=4)
    assert [xfer._frame_plan(t)[0] for t in range(1, 13)] == [t % 6 == 0 for t in range(1, 13)]
    assert xfer._collect_observe_forced == 0


def test_sensing_frame_due_force_stride1_is_byte_identical_when_off():
    # The extra force_stride1 arg defaults False and is byte-identical to the 4-arg call.
    for action in ('collect', 'transfer', 'couple'):
        for tick in range(1, 13):
            for stride in (1, 6):
                for susp in (False, True):
                    assert (_sensing_frame_due(action, tick, stride, susp)
                            == _sensing_frame_due(action, tick, stride, susp, False))
    # force_stride1=True renders every tick for a non-suspended sensing stage, nothing otherwise.
    assert _sensing_frame_due('collect', 3, 6, False, True) is True
    assert _sensing_frame_due('collect', 3, 6, True, True) is False   # suspended wins
    assert _sensing_frame_due('couple', 3, 6, False, True) is False   # non-sensing stage


def test_invalid_sensing_stride_rejected():
    import pytest as _pt
    with _pt.raises(ValueError, match='sensing_stride'):
        IsaacPort(world=None, robot=NS(dof_names=('left_wheel_joint', 'right_wheel_joint')),
                  robot_body=None, cart_body=None, mobile=None, range_sensor=None,
                  transport=None, contact=None, hold_arm=None, verification=None,
                  sensing_stride=0)


def test_no_recorder_means_no_record_render():
    port = _plan_port(action='return', stride=1, recorder=False)
    assert all(not r for _, r in (port._frame_plan(t) for t in range(1, 5)))


def test_base_brake_phase_aware_full_lock_vs_yaw_lock():
    # firstmate run-232/233: FULL base lock during the pre-weld grasp press (the free-translation base
    # slid 15.5 mm and the grip never sealed) AND while a welded carton is airborne (preplace swing
    # levered the base 16.4 mm); YAW-LOCK-ONLY only while a WELDED carton rests on a support
    # (grasp/place/release/contact), where a full lock would over-constrain the weld loop (run-208/209).
    class FakeMobile:
        def __init__(self):
            self.calls = []
        def __getattr__(self, name):
            return lambda *a, **k: self.calls.append(name)

    def calls(in_arm_phase, welded, phase):
        port = IsaacPort.__new__(IsaacPort)
        port.mobile = FakeMobile()
        port._apply_base_brake(in_arm_phase, welded, phase)
        return port.mobile.calls

    FULL = ['release_brake', 'engage_base_full_brake']
    YAW = ['release_base_full_brake', 'engage_brake']
    OFF = ['release_base_full_brake', 'release_brake']

    assert calls(True, False, 'approach') == FULL       # pre-weld approach
    assert calls(True, False, 'contact') == FULL        # pre-weld press (not yet welded) -> hold to seal
    assert calls(True, True, 'contact') == YAW          # weld just fired on the backstock support
    assert calls(True, True, 'grasp') == YAW            # welded, on the backstock support
    assert calls(True, True, 'lift') == FULL            # welded but airborne -> hold the base
    assert calls(True, True, 'preplace') == FULL        # welded, airborne swing to the cart -> hold
    assert calls(True, True, 'place') == YAW            # welded, pressing onto the cart (loop)
    assert calls(True, True, 'release') == YAW          # welded->releasing on the cart (loop)
    assert calls(True, True, 'retreat') == FULL         # carton gone, arm retreats -> hold
    assert calls(False, False, 'couple') == OFF         # left the arm phase: base free to drive


def test_begin_action_tracks_current_action_for_gating():
    port = IsaacPort.__new__(IsaacPort)
    port.on_action = None
    port.current_action = None
    port.begin_action('collect', job={})
    assert port.current_action == 'collect'


def test_begin_action_resets_collect_observe_force_budget():
    # fm452 (COLD): entering the collect stage reopens the force-render budget so the stride-1 window
    # fires afresh at this collect's observe entry. Non-collect actions leave it untouched.
    port = IsaacPort.__new__(IsaacPort)
    port.on_action = None
    port.current_action = None
    port._stage_scoped_teardown = None                    # non-collect path fires the teardown fallback
    port._stage_scoped_done = False
    port._collect_observe_forced = 7                      # spent by a previous collect
    port.begin_action('couple', job={})                   # non-collect: budget untouched
    assert port._collect_observe_forced == 7
    port.begin_action('collect', job={})                  # collect: budget reset
    assert port._collect_observe_forced == 0


def test_invalid_collect_observe_force_render_ticks_rejected():
    import pytest as _pt
    with _pt.raises(ValueError, match='collect_observe_force_render_ticks'):
        IsaacPort(world=None, robot=NS(dof_names=('left_wheel_joint', 'right_wheel_joint')),
                  robot_body=None, cart_body=None, mobile=None, range_sensor=None,
                  transport=None, contact=None, hold_arm=None, verification=None,
                  collect_observe_force_render_ticks=-1)


def test_invalid_record_stride_rejected():
    import pytest as _pt
    with _pt.raises(ValueError, match='record_stride'):
        IsaacPort(world=None, robot=NS(dof_names=('left_wheel_joint', 'right_wheel_joint')),
                  robot_body=None, cart_body=None, mobile=None, range_sensor=None,
                  transport=None, contact=None, hold_arm=None, verification=None,
                  record_stride=0)


def test_port_initializes_mission_clock_origin():
    # `origin` must exist so the verify boundary (on_action -> port.mission_time) never raises
    # AttributeError -- the former `port.origin` read assumed an attribute that was never set.
    port = IsaacPort(world=None, robot=NS(dof_names=('left_wheel_joint', 'right_wheel_joint')),
                     robot_body=None, cart_body=None, mobile=None, range_sensor=None,
                     transport=None, contact=None, hold_arm=None, verification=lambda now: None)
    assert port.origin == 0.0


def test_mission_time_recovers_packet_clock_from_stage_time_origin():
    # mission_time() converts Isaac stage time back to the mission clock (tick*dt) the packets carry
    # via `origin` (stage-time value at mission t=0). It is what on_action('verify') begins the
    # InventoryVerifier on, so its scan-stamp comparisons share the packets' clock.
    port = IsaacPort.__new__(IsaacPort)
    port.world = NS(current_time=12.3456)
    port.origin = 12.3456 - 5.0
    assert port.mission_time() == pytest.approx(5.0)
    # A stage clock that cannot be read falls back to -origin (no crash at the boundary).
    port.world = NS()                     # no current_time attribute -> float(...) raises
    assert port.mission_time() == pytest.approx(-(12.3456 - 5.0))


def gate_fixture():
    applied=[]
    robot=NS(get_dof_index=lambda name:ARM_JOINTS.index(name),
             get_joint_positions=lambda **kwargs:np.zeros(6))
    gate=ArmCommandGate(robot,[(-3,3)]*6,lambda q,v,i:applied.append((q.copy(),v.copy())))
    return gate,applied


def test_arm_drive_targets_do_not_accept_old_sequence():
    gate,applied=gate_fixture()
    command=dict(names=list(ARM_JOINTS),positions=[.01]*6,velocities=[.01]*6,
                 efforts=[],stamp=1.,now=1.)
    gate.command(**command)
    gate.command(**command)
    assert len(applied)==1 and not gate.faulted


def test_arm_heartbeat_sim_primary_with_frozen_sim_wall_backstop(monkeypatch):
    # sim-robust-watchdogs: the heartbeat's PRIMARY is the SIM clock -- the host executor is alive iff its
    # command stamps keep advancing, so a dead host (stamps frozen while Isaac keeps stepping) faults once
    # sim now - _host_now exceeds ARM_HEARTBEAT_SIM_BACKSTOP_S, regardless of render pace. The WALL backstop
    # now measures wall-since-SIM-NOW-ADVANCED, catching only a frozen sim (now not advancing while the
    # watchdog keeps ticking), NOT a slow-but-advancing render. (Replaces the old wall-backstop-only
    # contract; the full false-trip/true-death replays live in test_sim_robust_watchdogs.py.)
    import twin.restock.native_transport as nt
    clock = [1000.0]
    monkeypatch.setattr(nt.time, 'monotonic', lambda: clock[0])
    gate, _ = gate_fixture()
    gate.command(names=list(ARM_JOINTS), positions=[.01]*6, velocities=[.01]*6,
                 efforts=[], stamp=5.0, now=5.0)
    # A render-paced tick: huge WALL gap but sim now advanced close to the host stamp -> alive.
    clock[0] = 1000.0 + nt.ARM_HEARTBEAT_WALL_BACKSTOP_S + 100.0
    gate.watchdog(now=5.3)                 # sim gap 0.3 s; now advanced so the wall leg is disarmed
    assert not gate.faulted
    # True host death: stamps frozen (no new command) while sim now advances past the sim budget.
    with pytest.raises(RuntimeError, match='heartbeat lost'):
        gate.watchdog(now=5.0 + nt.ARM_HEARTBEAT_SIM_BACKSTOP_S + 0.1)


def test_arm_heartbeat_wall_backstop_fires_only_on_a_frozen_sim(monkeypatch):
    # The WALL backstop catches a sim whose clock has STOPPED advancing while the watchdog keeps ticking
    # (the sim leg cannot grow then). A frozen now + wall past the backstop faults.
    import twin.restock.native_transport as nt
    clock = [1000.0]
    monkeypatch.setattr(nt.time, 'monotonic', lambda: clock[0])
    gate, _ = gate_fixture()
    gate.command(names=list(ARM_JOINTS), positions=[.01]*6, velocities=[.01]*6,
                 efforts=[], stamp=5.0, now=5.0)
    gate.watchdog(now=5.2)                 # arms the wall timer at now=5.2
    clock[0] = 1000.0 + nt.ARM_HEARTBEAT_WALL_BACKSTOP_S - 1.0
    gate.watchdog(now=5.2)                 # now frozen; wall gap under the backstop -> alive
    assert not gate.faulted
    clock[0] = 1000.0 + nt.ARM_HEARTBEAT_WALL_BACKSTOP_S + 1.0
    with pytest.raises(RuntimeError, match='heartbeat lost'):
        gate.watchdog(now=5.2)             # now still frozen; wall gap past the backstop -> dead


def test_arm_freshness_is_host_domain_relay_lag_cancels():
    # firstmate 291: the freshness gate compares the stamp against the GREATEST host stamp seen
    # (self._host_now), not ISAAC now, so the host<->ISAAC /clock-relay lag cancels (run-244 tripped a
    # 2.5 s lag on a fresh command). The newest command is always fresh however far ISAAC now leads; a
    # genuinely OLD resend (stamp behind the latest by > freshness) is still rejected.
    applied = []
    robot = NS(get_dof_index=lambda name: ARM_JOINTS.index(name),
               get_joint_positions=lambda **k: np.zeros(6))
    gate = ArmCommandGate(robot, [(-3, 3)]*6, lambda q, v, i: applied.append(1))   # default 0.5 s
    # ISAAC now leads the host stamp by 2.5 s (pure relay lag) -> still ACCEPTED (host_age 0).
    gate.command(names=list(ARM_JOINTS), positions=[.01]*6, velocities=[.01]*6,
                 efforts=[], stamp=5.0, now=7.5)
    assert len(applied) == 1 and not gate.faulted
    gate.command(names=list(ARM_JOINTS), positions=[.02]*6, velocities=[.01]*6,
                 efforts=[], stamp=5.4, now=9.0)           # newest -> fresh
    assert len(applied) == 2
    # A command stamped 0.8 s behind the latest host stamp (5.4) is a stale resend -> rejected at 0.5 s.
    with pytest.raises(ValueError, match='invalid or stale'):
        gate.command(names=list(ARM_JOINTS), positions=[.03]*6, velocities=[.01]*6,
                     efforts=[], stamp=4.6, now=9.1)
    # The sim profile widens the bound: the SAME 0.8 s-behind resend is within a 1.0 s window.
    relaxed = ArmCommandGate(robot, [(-3, 3)]*6, lambda q, v, i: applied.append(1), freshness_s=1.0)
    relaxed.command(names=list(ARM_JOINTS), positions=[.01]*6, velocities=[.01]*6,
                    efforts=[], stamp=5.4, now=9.0)
    relaxed.command(names=list(ARM_JOINTS), positions=[.02]*6, velocities=[.01]*6,
                    efforts=[], stamp=4.6, now=9.1)         # 0.8 s behind, < 1.0 s -> accepted
    assert not relaxed.faulted


def test_arm_command_ignores_the_effort_field():
    # Isaac's ROS2SubscribeJointState fills effort to the joint count even though trajectory_host
    # publishes none (run-89: eff=6 all-zero; run-90: eff=6 non-zero measured torques). Effort is
    # never commanded (apply drives only q,v), so it must be IGNORED -- a valid position+velocity
    # command is applied regardless of the effort field (gating on it stalled the executed grasp).
    gate,applied=gate_fixture()
    gate.command(names=list(ARM_JOINTS),positions=[.01]*6,velocities=[.01]*6,
                 efforts=[0.]*6,stamp=1.,now=1.)
    gate.command(names=list(ARM_JOINTS),positions=[.02]*6,velocities=[.01]*6,
                 efforts=[1.3,-2.,0.,4.,0.,5.],stamp=2.,now=2.)
    assert len(applied)==2 and not gate.faulted


def test_arm_tracking_jump_latches_measured_hold_and_rejects_later_move():
    gate,applied=gate_fixture()
    with pytest.raises(ValueError,match='tracking'):
        gate.command(names=list(ARM_JOINTS),positions=[1.]*6,velocities=[.01]*6,
                     efforts=[],stamp=1.,now=1.)
    assert gate.faulted and np.array_equal(applied[-1][0],np.zeros(6))
    gate.command(names=list(ARM_JOINTS),positions=[.01]*6,velocities=[.01]*6,
                 efforts=[],stamp=2.,now=2.)
    assert np.array_equal(applied[-1][0],np.zeros(6))


def test_masked_depth_renders_on_sensing_cadence_through_suspension():
    # ESDF/masked-depth must keep publishing on the sensing cadence even while the decode capture is
    # gate-suspended (run-139 'stale parked-station' when the suspension's render-skip stopped it).
    # It is suspension-independent: fires for decode-grade stages on their stride, never for nav stages
    # or when no masked-depth publisher is wired.
    from twin.restock.isaac_port import _masked_depth_due, _sensing_frame_due
    assert _masked_depth_due('collect', 6, 6, has_masked_depth=True)
    assert _masked_depth_due('collect', 12, 6, has_masked_depth=True)
    assert not _masked_depth_due('collect', 7, 6, has_masked_depth=True)       # off-cadence
    assert not _masked_depth_due('couple', 6, 6, has_masked_depth=True)        # nav stage
    assert not _masked_depth_due('collect', 6, 6, has_masked_depth=False)      # no publisher
    # Decoupled from the decode capture: while suspended, sensing (decode) is skipped but masked depth
    # still renders -> the ESDF stays fresh with the decode render off (once the camera PR lands).
    assert not _sensing_frame_due('collect', 6, 6, suspended=True)
    assert _masked_depth_due('collect', 6, 6, has_masked_depth=True)


def test_gate_phase_drive_switches_stiff_for_place_soft_elsewhere(monkeypatch):
    # fm347/348(1): gate() must switch the arm drive to STIFF for the loaded extend/rotate+place
    # (pan/preplace/place) and back to SOFT elsewhere, once per transition, via the robot it holds.
    # (run-274 shoulder_lift sagged 0.622 under the soft drive; bench-s348d proved stiff holds it.)
    import twin.restock.physics_setup as ps
    calls = []
    monkeypatch.setattr(ps, 'set_arm_drive_gains',
                        lambda robot, aj, kp, kd, eff: calls.append((round(kp), round(kd), round(eff))))
    port = IsaacPort.__new__(IsaacPort)                 # skip Isaac wiring; only the phase-drive state
    port._phase = None
    port._set_render_suspended = None                   # render gate off -> gate() returns after the switch
    port._phase_drive = {'soft': {'kp': 3e5, 'kd': 8e3, 'eff': 400.},
                         'stiff': {'kp': 1e6, 'kd': 1e4, 'eff': 2000.}}
    port._arm_joints = ARM_JOINTS
    port._last_drive_gains = None
    port.robot = object()                               # set_arm_drive_gains is patched; robot unused here
    for ph in ('approach', 'tuck', 'pan', 'preplace', 'place', 'release'):
        port.gate(ph)
    soft, stiff = (300000, 8000, 400), (1000000, 10000, 2000)
    # soft applied entering approach, stiff entering pan (held through preplace/place, no redundant set),
    # soft restored at release -> exactly [soft, stiff, soft].
    assert calls == [soft, stiff, soft], calls


def test_gate_phase_drive_absent_is_noop(monkeypatch):
    # Absent config['phase_drive'] -> no gain switch ever (byte-identical to the assemble() drive).
    import twin.restock.physics_setup as ps
    calls = []
    monkeypatch.setattr(ps, 'set_arm_drive_gains', lambda *a, **k: calls.append(a))
    port = IsaacPort.__new__(IsaacPort)
    port._phase = None
    port._set_render_suspended = None
    port._phase_drive = None
    port._arm_joints = ARM_JOINTS
    port._last_drive_gains = None
    port.robot = object()
    for ph in ('tuck', 'pan', 'place', 'release'):
        port.gate(ph)
    assert calls == []
