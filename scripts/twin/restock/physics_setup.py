"""Native Isaac stage and actuator assembly inside the caller's guarded app.

This is the physical assembly half of the mission factory. It does not claim
camera/mapper readiness. All reset-state authoring happens before any mission
step; after construction callers receive drive targets and measured state only.
"""
from dataclasses import dataclass, fields
import hashlib
import json
from pathlib import Path

import numpy as np

from twin.mobile import MobileSpec
from twin.mobile_runtime import MobileController
from twin.restock_manipulation.cad_tool import inspect_tool, author_native_gripper, NativePanelContact
from twin.restock_manipulation.ros_adapter import require_sim_domain
from .isaac_bootstrap import import_mobile, discover,resolve_meshes
from .native_transport import ArmCommandGate, NativeTransport
from .scene import build, label_stock


def inputs(config, scenario):
    """CPU-only exact generated asset validation, with no Isaac import."""
    manifest_path=Path(config['mobile_manifest']).resolve()
    mobile=json.loads(manifest_path.read_text())
    urdf=Path(mobile['mobile_urdf']).resolve()
    store=Path(config['store_usd']).resolve()
    if not urdf.is_file() or not store.is_file():
        raise ValueError('generated mobile URDF and exported store required')
    spec=MobileSpec(**{f.name:mobile['spec'][f.name] for f in fields(MobileSpec)})
    tool=inspect_tool(urdf)
    positions=mobile['transport_positions']
    if any(name not in positions or not lo<=positions[name]<=hi
           for name,(lo,hi) in zip(tool.arm_joints,tool.arm_limits)):
        raise ValueError('transport joints differ from actual CAD limits')
    if not scenario['initial_staging']['initial_conditions_only']:
        raise ValueError('scenario must declare initial-condition placements')
    hashes={name:hashlib.sha256(path.read_bytes()).hexdigest()
            for name,path in [('mobile_manifest',manifest_path),('mobile_urdf',urdf),('store_usd',store)]}
    hashes['scenario']=hashlib.sha256(json.dumps(scenario,sort_keys=True).encode()).hexdigest()
    return mobile,urdf,store,spec,tool,hashes


def configure_cart_solver(stage, root):
    """Increase constraint convergence for the 35 kg cage/0.25 kg caster forks.

    Preserve all mass, inertia, friction, collision and free-wheel joint models.
    Native measurements must still prove parking; this setting is not a brake.
    """
    from pxr import Usd,UsdPhysics,Sdf
    prim=stage.GetPrimAtPath(root)
    if not prim:
        raise ValueError('cart root absent')
    bodies=[]
    for body in Usd.PrimRange(prim):
        if body.HasAPI(UsdPhysics.RigidBodyAPI):
            body.AddAppliedSchema('PhysxRigidBodyAPI')
            before={}
            for name,value in (('solverPositionIterationCount',64),('solverVelocityIterationCount',16)):
                attr=body.CreateAttribute('physxRigidBody:'+name,Sdf.ValueTypeNames.Int,False)
                before[name]=attr.Get()
                attr.Set(value)
            bodies.append(dict(path=str(body.GetPath()),before=before,position_iterations=64,velocity_iterations=16))
    if len(bodies)!=9:
        raise ValueError('expected cage, four forks and four wheels')
    return bodies


def configure_wheel_traction(stage, articulation_root, coeff):
    """Bind a high-friction PhysX material to the drive-wheel colliders so the base can tow the
    coupled cart (default friction ~0.5 slips the 215 kg robot+cart). Finds the wheel bodies by name
    ('left_wheel'/'right_wheel') in the articulation subtree and binds the material to every collider
    under them. static=coeff, dynamic=0.9*coeff, restitution 0; 'max' combine so the pair friction is
    governed by the wheel. Returns the bound collider paths (empty if the wheels are not found)."""
    from pxr import Usd, UsdShade, UsdPhysics, Sdf
    mat = UsdShade.Material.Define(stage, '/RestockWheelMaterial')
    api = UsdPhysics.MaterialAPI.Apply(mat.GetPrim())
    api.CreateStaticFrictionAttr().Set(float(coeff))
    api.CreateDynamicFrictionAttr().Set(float(coeff) * 0.9)
    api.CreateRestitutionAttr().Set(0.0)
    mat.GetPrim().CreateAttribute('physxMaterial:frictionCombineMode', Sdf.ValueTypeNames.Token, False).Set('max')
    root = stage.GetPrimAtPath(articulation_root)
    bound = []
    if root:
        for p in Usd.PrimRange(root):
            if p.GetName() in ('left_wheel', 'right_wheel'):
                for c in Usd.PrimRange(p):
                    if c.HasAPI(UsdPhysics.CollisionAPI):
                        UsdShade.MaterialBindingAPI(c).Bind(mat, materialPurpose='physics')
                        bound.append(str(c.GetPath()))
    return bound


def configure_scene_physics(stage, scene_path):
    """Enable PhysX resting-contact stabilization on the composed scene.

    Isaac persists ``enableStabilization=False`` by default. On a balanced body
    resting on four contacts that injects depenetration jitter every step - the
    signature observed on the stationary cart was a constant ~55-70 mm/s reported
    +z velocity with ~0 z displacement. Stabilization is a solver-accuracy setting,
    not an applied force or brake: a balanced cart under zero horizontal load must
    not drift. TGS solver, gravity and time step are unchanged.
    """
    from pxr import Sdf
    prim=stage.GetPrimAtPath(scene_path)
    if not prim:
        raise ValueError('composed physics scene absent')
    prim.AddAppliedSchema('PhysxSceneAPI')
    attr=prim.CreateAttribute('physxScene:enableStabilization',Sdf.ValueTypeNames.Bool,False)
    before=attr.Get()
    attr.Set(True)
    return dict(path=str(prim.GetPath()),enable_stabilization=True,before=before)


LINEAR_DAMPING=0.5
ANGULAR_DAMPING=3.0


def configure_cart_bearings(stage, root):
    """Author rolling / bearing resistance as body damping on the cart.

    The cart's four swivel/roll casters were authored with no resistance, so once
    resting-contact jitter is removed the balanced cart still creeps: any residual
    impulse rolls the frictionless wheels forever. The physically faithful remedy is
    caster bearing + rolling resistance. PhysX ``jointFriction`` would be the natural
    Coulomb model but it is inert here - the schema restricts it to articulation
    joints and these casters are maximal-coordinate (excludeFromArticulation). So the
    resistance is authored as rigid-body damping instead: strong ANGULAR damping on
    the bodies (dominated by the four low-inertia wheels, where it opposes spin -
    hence rolling - almost for free: wheel inertia ~1.7e-3 kg m^2, so even a large
    coefficient is a negligible torque against a commanded tow) and mild LINEAR
    damping on the 35 kg cage. This models bearing/rolling/air losses a real cart has;
    it is not a positional brake (no drive, stiffness or target) and leaves mass,
    inertia, collision geometry and the free revolute model untouched. Preserve every
    other authored attribute.
    """
    from pxr import Usd, UsdPhysics, Sdf
    prim=stage.GetPrimAtPath(root)
    if not prim:
        raise ValueError('cart root absent')
    joints,bodies=[],[]
    for p in Usd.PrimRange(prim):
        if p.IsA(UsdPhysics.RevoluteJoint):
            joints.append(str(p.GetPath()))  # recorded for the 8-caster invariant only
        if p.HasAPI(UsdPhysics.RigidBodyAPI):
            p.AddAppliedSchema('PhysxRigidBodyAPI')
            for name,value in (('linearDamping',LINEAR_DAMPING),('angularDamping',ANGULAR_DAMPING)):
                p.CreateAttribute('physxRigidBody:'+name,Sdf.ValueTypeNames.Float,False).Set(value)
            bodies.append(str(p.GetPath()))
    if len(joints)!=8 or len(bodies)!=9:
        raise ValueError('expected eight caster joints and nine cart bodies')
    return dict(linear_damping=LINEAR_DAMPING,angular_damping=ANGULAR_DAMPING,
                joint_friction='inert on maximal-coordinate casters; using body damping',
                joints=joints,bodies=bodies)


@dataclass
class PhysicsAssembly:
    world: object
    stage: object
    robot: object
    robot_body: object
    cart_body: object
    mobile: object
    contact: object
    command_gate: object
    transport: object
    paths: dict
    evidence: dict

    def hold(self):
        self.mobile.park()
        self.command_gate.hold()


def assemble(*, app, config, scenario, output, diagnostic_prepare=None):
    """Import actual mobile CAD, compose scratch USD and initialize drive control.

    Does not create SimulationApp, acquire GPU permission or start a host service.
    The calling launcher must hold the existing guard for this entire lifetime.
    No physics stepping, body relocation or reset is exposed as a task action.
    """
    require_sim_domain()
    mobile,urdf,store,spec,tool,hashes=inputs(config,scenario)
    output=Path(output).resolve()
    output.mkdir(parents=True,exist_ok=True)
    from isaacsim.core.utils.extensions import enable_extension
    from isaacsim.core.utils.stage import open_stage,is_stage_loading
    enable_extension('isaacsim.robot.surface_gripper')
    resolved_urdf=resolve_meshes(urdf,output/'mobile-import.urdf',config['package_roots'])
    robot_usd,_=import_mobile(resolved_urdf,output/'mobile-import.usd')
    staging=scenario['initial_staging']
    composition=build(store,robot_usd,output/'mission.usda',
        robot_pose=staging['robot_base_pose'],cart_pose=staging['cart_body_pose'],
        stock_path=None,sku=scenario['job']['sku'],
        stock_mass_kg=scenario['handling_unit']['mass_kg'],spec=spec,scenario=scenario)
    if not open_stage(composition['stage']):
        raise RuntimeError('could not open task composed stage')
    for _ in range(300):
        app.update()
        if not is_stage_loading():
            break
    else:
        raise RuntimeError('stage loading exceeded bounded update budget')
    import omni.usd
    from pxr import UsdPhysics
    from isaacsim.core.api import World
    from isaacsim.core.prims import SingleArticulation,SingleRigidPrim
    from isaacsim.core.utils.types import ArticulationAction
    stage=omni.usd.get_context().get_stage()
    paths=discover(stage)
    label_stock(stage,composition['stock'],scenario['job']['sku'],output/'handling-unit-qr.png')
    author_native_gripper(stage,tool=tool,tool0_path=paths['ur_tool0'],
        body_path=paths['suction_cups'],gripper_path='/RestockPanelGripper')
    cart_path=composition['assets']['cart']['body']
    cart_root=composition['assets']['cart']['root']
    solver=configure_cart_solver(stage,cart_root)
    bearings=configure_cart_bearings(stage,cart_root)
    # Optional wheel-ground traction tune (data/mobile-restock-v2 traction investigation): the
    # default PhysX friction (~0.5) slips a 215 kg robot+cart, stalling the coupled tow. Guarded so a
    # material-authoring failure degrades to default friction (never breaks the mission/grasp).
    wheel_friction=config.get('wheel_friction')
    if wheel_friction:
        try:
            bound=configure_wheel_traction(stage,paths['articulation'],float(wheel_friction))
            print(f'WHEEL_TRACTION static={wheel_friction} bound={len(bound)} colliders',flush=True)
        except Exception as exc:
            print(f'WHEEL_TRACTION skipped: {type(exc).__name__}: {exc}',flush=True)
    if diagnostic_prepare is not None:
        diagnostic_prepare(stage)
    controller=MobileController(stage,paths['base_link'],cart_path,
        [paths['left_wheel_joint'],paths['right_wheel_joint']],spec=spec)
    scenes=[str(p.GetPath()) for p in stage.Traverse() if p.IsA(UsdPhysics.Scene)]
    if len(scenes)!=1:
        raise ValueError('one composed physics scene required')
    scene_physics=configure_scene_physics(stage,scenes[0])
    world=World(physics_dt=1/60,rendering_dt=1/60,stage_units_in_meters=1.,physics_prim_path=scenes[0])
    # World's PhysicsContext re-applies Isaac defaults at construction
    # (enable_stablization(False)), silently clobbering the USD attribute authored
    # above. Re-assert it through the context BEFORE the first reset so PhysX parses
    # stabilization=True when it builds the live scene; the method only writes USD.
    context=world.get_physics_context()
    context.enable_stablization(True)
    scene_physics['set_via']='isaacsim PhysicsContext.enable_stablization before world.reset'
    robot=world.scene.add(SingleArticulation(prim_path=paths['articulation'],name='restock_robot'))
    robot_body=world.scene.add(SingleRigidPrim(prim_path=paths['base_link'],name='restock_base'))
    cart_body=world.scene.add(SingleRigidPrim(prim_path=cart_path,name='restock_cart'))
    world.reset()
    # Establish documented spawn state before any mission or recorded step.
    # This second reset applies joint defaults; it is never used during a mission.
    positions=np.asarray(robot.get_joint_positions(),float).copy()
    for name,value in mobile['transport_positions'].items():
        positions[robot.get_dof_index(name)]=value
    robot.set_joints_default_state(positions=positions,velocities=np.zeros(len(positions)))
    world.reset()
    actual=np.asarray(robot.get_joint_positions(),float)
    if not np.isfinite(actual).all() or not np.allclose(actual,positions,atol=1e-5):
        raise RuntimeError('initial articulation state did not apply')
    scene_physics['effective_after_reset']=bool(context.is_stablization_enabled())
    if not scene_physics['effective_after_reset']:
        raise RuntimeError('contact stabilization did not take effect after reset')
    def apply(q,v,indices):
        robot.apply_action(ArticulationAction(joint_positions=q,joint_velocities=v,joint_indices=indices))
    gate=ArmCommandGate(robot,tool.arm_limits,apply)
    contact=NativePanelContact(stage,'/RestockPanelGripper')
    # Register native ROS/image graph nodes only after the final scene has
    # finished opening/resetting. A bridge loaded during import/reopen caused
    # a native image-graph crash in the installed Isaac5.1 stack.
    enable_extension('isaacsim.ros2.bridge')
    app.update()
    # Relay /clock + /tf + /joint_states to the host over the host-DDS subprocess (OmniGraph can't
    # reach the host), using the same host-overlay runtime_env the masked-depth bridge uses.
    _md=config.get('masked_depth',{})
    transport=NativeTransport(stage=stage,articulation_path=paths['articulation'],
        arm_base_path=paths['ur_base_link'],tf_paths=[paths['gripper_pad_tcp'],paths['gripper_tcp']],
        command_gate=gate,robot=robot,base_path=paths.get('base_link'),
        runtime_env=_md.get('runtime_env'),python=_md.get('python','python3'))
    stage.GetRootLayer().Save()
    evidence=dict(input_sha256=hashes,composition=composition,paths=paths,
        cart_solver=solver,cart_bearings=bearings,scene_physics=scene_physics,
        initialization='declared initial joint state before mission; never runtime teleport',
        physics_verified=False,camera_mapper_ready=False)
    (output/'assembly.json').write_text(json.dumps(evidence,indent=2)+'\n')
    return PhysicsAssembly(world,stage,robot,robot_body,cart_body,controller,contact,gate,transport,paths,evidence)
